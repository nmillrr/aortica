"""Power consumption validation for edge deployment (US-061b).

Validates that an ONNX edge model draws less than a sustained power budget
when deployed with duty-cycled, on-demand inference on ARM hardware.

The key insight for LMIC / rural deployment is that ECGs arrive infrequently
(a handful per hour), so the device spends the vast majority of its time idle.
The *sustained* power attributable to inference is therefore::

    sustained_power = TDP × duty_cycle

where ``duty_cycle = active_inference_time / inference_interval``.  With
on-demand model loading (see :class:`aortica.edge.deploy_profiles.DutyCycledModelLoader`)
the ONNX session is not kept resident, so idle draw stays near the device's
baseline and the incremental inference power is small.

The default budget is **200 mW sustained**, which an RPi4 (4 W TDP) meets for
any inference interval longer than ~20× the per-inference latency.

Example::

    from aortica.edge import validate_power_consumption

    report = validate_power_consumption("model.onnx", "rpi4", n_inferences=50)
    assert report.passed
    print(report.summary())
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from aortica.edge.profiling import (
    DEFAULT_HARDWARE,
    HARDWARE_TDP,
    profile_inference,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default sustained-power budget for ARM edge deployment (milliwatts).
DEFAULT_POWER_BUDGET_MW: float = 200.0

#: Default assumed interval between ECG acquisitions at a pilot site (seconds).
#: One ECG every five minutes is a conservative clinic cadence.
DEFAULT_INFERENCE_INTERVAL_S: float = 300.0

#: Default synthetic input shape when the model input is fully dynamic.
_DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (1, 12, 5000)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class PowerValidationReport:
    """Result of a power-consumption validation run.

    Attributes:
        hardware_profile: Hardware identifier used for TDP (e.g. ``rpi4``).
        tdp_watts: Thermal design power of the target hardware.
        n_inferences: Number of timed inferences performed.
        mean_latency_ms: Mean per-inference latency in milliseconds.
        energy_per_inference_mj: Energy per inference in millijoules.
        inference_interval_seconds: Assumed interval between ECGs.
        duty_cycle: Fraction of time the device is actively inferring.
        sustained_power_mw: Sustained power attributable to inference (mW).
        threshold_mw: Power budget the run is validated against (mW).
        passed: Whether ``sustained_power_mw`` is below ``threshold_mw``.
        model_path: Path to the validated model.
    """

    hardware_profile: str = DEFAULT_HARDWARE
    tdp_watts: float = 4.0
    n_inferences: int = 0
    mean_latency_ms: float = 0.0
    energy_per_inference_mj: float = 0.0
    inference_interval_seconds: float = DEFAULT_INFERENCE_INTERVAL_S
    duty_cycle: float = 0.0
    sustained_power_mw: float = 0.0
    threshold_mw: float = DEFAULT_POWER_BUDGET_MW
    passed: bool = False
    model_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "hardware_profile": self.hardware_profile,
            "tdp_watts": self.tdp_watts,
            "n_inferences": self.n_inferences,
            "mean_latency_ms": self.mean_latency_ms,
            "energy_per_inference_mj": self.energy_per_inference_mj,
            "inference_interval_seconds": self.inference_interval_seconds,
            "duty_cycle": self.duty_cycle,
            "sustained_power_mw": self.sustained_power_mw,
            "threshold_mw": self.threshold_mw,
            "passed": self.passed,
            "model_path": self.model_path,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the validation."""
        status = "PASS" if self.passed else "FAIL"
        return "\n".join(
            [
                "Edge Power Consumption Validation",
                "=" * 40,
                f"  Hardware:          {self.hardware_profile} ({self.tdp_watts:.1f} W TDP)",
                f"  Inferences:        {self.n_inferences}",
                f"  Mean latency:      {self.mean_latency_ms:.2f} ms",
                f"  Energy/inference:  {self.energy_per_inference_mj:.3f} mJ",
                f"  ECG interval:      {self.inference_interval_seconds:.0f} s",
                f"  Duty cycle:        {self.duty_cycle * 100:.4f} %",
                f"  Sustained power:   {self.sustained_power_mw:.2f} mW",
                f"  Budget:            {self.threshold_mw:.0f} mW",
                f"  Result:            {status}",
            ]
        )


# ---------------------------------------------------------------------------
# Pure power math (independently testable, no onnxruntime required)
# ---------------------------------------------------------------------------


def compute_sustained_power(
    mean_latency_ms: float,
    tdp_watts: float,
    inference_interval_seconds: float,
) -> tuple[float, float]:
    """Compute duty cycle and sustained power from latency and cadence.

    Args:
        mean_latency_ms: Mean active inference latency in milliseconds.
        tdp_watts: Thermal design power of the target hardware in watts.
        inference_interval_seconds: Interval between ECG acquisitions in
            seconds.

    Returns:
        Tuple ``(duty_cycle, sustained_power_mw)`` where ``duty_cycle`` is a
        fraction in ``[0, 1]`` and ``sustained_power_mw`` is in milliwatts.

    Raises:
        ValueError: If ``inference_interval_seconds`` or ``tdp_watts`` is not
            positive, or ``mean_latency_ms`` is negative.
    """
    if inference_interval_seconds <= 0:
        raise ValueError(
            f"inference_interval_seconds must be positive, "
            f"got {inference_interval_seconds}"
        )
    if tdp_watts <= 0:
        raise ValueError(f"tdp_watts must be positive, got {tdp_watts}")
    if mean_latency_ms < 0:
        raise ValueError(f"mean_latency_ms must be non-negative, got {mean_latency_ms}")

    latency_s = mean_latency_ms / 1000.0
    duty_cycle = min(latency_s / inference_interval_seconds, 1.0)
    sustained_power_mw = tdp_watts * duty_cycle * 1000.0
    return duty_cycle, sustained_power_mw


# ---------------------------------------------------------------------------
# Model input helpers
# ---------------------------------------------------------------------------


def _resolve_input_shape(raw_shape: list[Any]) -> tuple[int, ...]:
    """Resolve an ONNX input shape, substituting defaults for dynamic dims.

    Dynamic dimensions (``None`` or symbolic strings) are replaced by sensible
    edge-model defaults: batch = 1, and, for a 3-D ``[batch, leads, samples]``
    shape, leads = 12 and samples = 5000.
    """
    resolved: list[int] = []
    for i, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif i == 0:
            resolved.append(1)  # batch
        elif len(raw_shape) == 3 and i == 1:
            resolved.append(_DEFAULT_INPUT_SHAPE[1])  # leads
        elif len(raw_shape) == 3 and i == 2:
            resolved.append(_DEFAULT_INPUT_SHAPE[2])  # samples
        else:
            resolved.append(1)
    return tuple(resolved)


def _synthetic_input_for_model(model_path: str) -> npt.NDArray[np.float32]:
    """Build a random float32 input tensor matching the model's input shape."""
    import onnxruntime as ort  # local import; guarded by profile_inference too

    session = ort.InferenceSession(model_path)
    raw_shape = session.get_inputs()[0].shape
    shape = _resolve_input_shape(list(raw_shape))
    rng = np.random.default_rng(42)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_power_consumption(
    model_path: Union[str, Path],
    hardware_profile: str = DEFAULT_HARDWARE,
    n_inferences: int = 50,
    inference_interval_seconds: float = DEFAULT_INFERENCE_INTERVAL_S,
    threshold_mw: float = DEFAULT_POWER_BUDGET_MW,
    input_data: Optional[npt.NDArray[np.floating[Any]]] = None,
) -> PowerValidationReport:
    """Validate that an edge model stays within a sustained-power budget.

    Measures per-inference latency via :func:`profile_inference`, then computes
    the sustained power for the assumed inference cadence and checks it against
    ``threshold_mw`` (default 200 mW).

    Args:
        model_path: Path to the ONNX edge model.
        hardware_profile: Hardware profile for TDP lookup (``rpi4``, ``rpi5``,
            ``jetson_nano``, ``jetson_orin_nano``). Default ``rpi4``.
        n_inferences: Number of timed inferences. Default ``50``.
        inference_interval_seconds: Assumed interval between ECGs, used for the
            duty-cycle calculation. Default ``300`` (one every 5 minutes).
        threshold_mw: Sustained-power budget in milliwatts. Default ``200``.
        input_data: Optional input tensor. If ``None``, a random tensor is
            generated to match the model's input shape.

    Returns:
        A :class:`PowerValidationReport`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If the model file does not exist.
        ValueError: If ``n_inferences`` is not positive.
    """
    model_path = str(Path(model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if n_inferences <= 0:
        raise ValueError(f"n_inferences must be positive, got {n_inferences}")

    if input_data is None:
        input_data = _synthetic_input_for_model(model_path)

    profile = profile_inference(
        model_path,
        input_data,
        n_runs=n_inferences,
        hardware_profile=hardware_profile,
    )

    tdp = HARDWARE_TDP.get(hardware_profile, HARDWARE_TDP[DEFAULT_HARDWARE])
    duty_cycle, sustained_power_mw = compute_sustained_power(
        profile.mean_latency_ms, tdp, inference_interval_seconds,
    )

    return PowerValidationReport(
        hardware_profile=hardware_profile,
        tdp_watts=tdp,
        n_inferences=n_inferences,
        mean_latency_ms=profile.mean_latency_ms,
        energy_per_inference_mj=profile.energy_per_inference_mj,
        inference_interval_seconds=inference_interval_seconds,
        duty_cycle=duty_cycle,
        sustained_power_mw=sustained_power_mw,
        threshold_mw=threshold_mw,
        passed=sustained_power_mw < threshold_mw,
        model_path=model_path,
    )
