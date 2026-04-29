"""Inference profiling and power estimation for edge models.

Measures latency statistics (mean, p50, p95), peak memory usage, and model
size on disk for ONNX edge models.  Power estimation uses latency × TDP for
known hardware profiles (RPi4, RPi5, Jetson Nano, Jetson Orin Nano).

The primary entry point is :func:`profile_inference`, which returns an
:class:`InferenceProfile`.

Example::

    from aortica.edge import profile_inference

    profile = profile_inference(
        model_path="aortica_edge_int8.onnx",
        input_data=sample_ecg_array,
        n_runs=100,
    )
    print(profile.summary_table())

"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def _check_onnxruntime() -> None:
    if not HAS_ONNXRUNTIME:
        raise ImportError(
            "onnxruntime is required for inference profiling. "
            "Install with: pip install aortica[edge]"
        )


# ---------------------------------------------------------------------------
# Hardware TDP estimates (watts) — mirrors deploy_profiles.HARDWARE_TDP
# ---------------------------------------------------------------------------

HARDWARE_TDP: dict[str, float] = {
    "rpi4": 4.0,
    "rpi5": 6.0,
    "jetson_nano": 5.0,
    "jetson_orin_nano": 7.0,
}

# Default hardware profile for power estimation
DEFAULT_HARDWARE = "rpi4"

# Default number of profiling runs
DEFAULT_N_RUNS = 100

# Warmup runs to stabilise ONNX Runtime before timing
WARMUP_RUNS = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    """Inference profiling results.

    Attributes:
        mean_latency_ms: Mean inference latency in milliseconds.
        p50_latency_ms: Median (50th percentile) latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        min_latency_ms: Minimum observed latency in milliseconds.
        max_latency_ms: Maximum observed latency in milliseconds.
        std_latency_ms: Standard deviation of latency in milliseconds.
        peak_memory_bytes: Estimated peak memory usage in bytes.
        model_size_bytes: Model file size on disk in bytes.
        n_runs: Number of inference runs performed.
        hardware_profile: Hardware profile used for power estimation.
        tdp_watts: Thermal design power of the target hardware.
        energy_per_inference_mj: Estimated energy per inference in millijoules.
        power_draw_watts: Estimated average power draw during inference.
        latencies_ms: Raw latency measurements in milliseconds.
        input_shape: Shape of the input tensor.
        model_path: Path to the profiled model.
    """

    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    peak_memory_bytes: int = 0
    model_size_bytes: int = 0
    n_runs: int = 0
    hardware_profile: str = DEFAULT_HARDWARE
    tdp_watts: float = 4.0
    energy_per_inference_mj: float = 0.0
    power_draw_watts: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    input_shape: tuple[int, ...] = ()
    model_path: str = ""

    @property
    def model_size_mb(self) -> float:
        """Model file size in megabytes."""
        return self.model_size_bytes / (1024 * 1024)

    @property
    def peak_memory_mb(self) -> float:
        """Estimated peak memory in megabytes."""
        return self.peak_memory_bytes / (1024 * 1024)

    def summary_table(self) -> str:
        """Return a human-readable profiling summary."""
        lines: list[str] = []
        lines.append("Inference Profile")
        lines.append("=" * 50)
        lines.append(f"  Model:              {self.model_path}")
        lines.append(f"  Model size:         {self.model_size_mb:.2f} MB")
        lines.append(f"  Input shape:        {self.input_shape}")
        lines.append(f"  Runs:               {self.n_runs}")
        lines.append("")
        lines.append("Latency")
        lines.append("-" * 50)
        lines.append(f"  Mean:               {self.mean_latency_ms:.2f} ms")
        lines.append(f"  Median (p50):       {self.p50_latency_ms:.2f} ms")
        lines.append(f"  p95:                {self.p95_latency_ms:.2f} ms")
        lines.append(f"  Min:                {self.min_latency_ms:.2f} ms")
        lines.append(f"  Max:                {self.max_latency_ms:.2f} ms")
        lines.append(f"  Std dev:            {self.std_latency_ms:.2f} ms")
        lines.append("")
        lines.append("Memory")
        lines.append("-" * 50)
        lines.append(f"  Peak memory:        {self.peak_memory_mb:.2f} MB")
        lines.append("")
        lines.append("Power Estimation")
        lines.append("-" * 50)
        lines.append(f"  Hardware:           {self.hardware_profile}")
        lines.append(f"  TDP:                {self.tdp_watts:.1f} W")
        lines.append(
            f"  Energy/inference:   {self.energy_per_inference_mj:.3f} mJ"
        )
        lines.append(f"  Power draw:         {self.power_draw_watts:.3f} W")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise profile to a plain dictionary."""
        return {
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "model_size_bytes": self.model_size_bytes,
            "model_size_mb": self.model_size_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "n_runs": self.n_runs,
            "hardware_profile": self.hardware_profile,
            "tdp_watts": self.tdp_watts,
            "energy_per_inference_mj": self.energy_per_inference_mj,
            "power_draw_watts": self.power_draw_watts,
            "input_shape": list(self.input_shape),
            "model_path": self.model_path,
        }


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def _estimate_peak_memory(
    model_path: str,
    input_data: npt.NDArray[np.floating[Any]],
) -> int:
    """Estimate peak memory usage for model inference.

    Uses model file size + input tensor size + estimated activation memory
    (heuristic: 2× model size for intermediate buffers).

    Returns:
        Estimated peak memory in bytes.
    """
    model_size = os.path.getsize(model_path)
    input_size = input_data.nbytes
    # Heuristic: model weights + input + ~2x model size for activations/buffers
    estimated = model_size + input_size + 2 * model_size
    return estimated


# ---------------------------------------------------------------------------
# Power estimation
# ---------------------------------------------------------------------------


def _estimate_power(
    mean_latency_ms: float,
    hardware_profile: str,
) -> tuple[float, float, float]:
    """Estimate power consumption based on latency and hardware TDP.

    Assumes that during active inference, the device draws its full TDP,
    and is idle otherwise.  This provides a conservative upper bound.

    Args:
        mean_latency_ms: Mean inference latency in milliseconds.
        hardware_profile: Hardware identifier (e.g. 'rpi4', 'jetson_nano').

    Returns:
        Tuple of (tdp_watts, energy_per_inference_mj, power_draw_watts).
    """
    tdp = HARDWARE_TDP.get(hardware_profile, HARDWARE_TDP[DEFAULT_HARDWARE])

    # Energy per inference = TDP × latency
    # TDP (W) × latency (s) = energy (J)
    latency_s = mean_latency_ms / 1000.0
    energy_j = tdp * latency_s
    energy_mj = energy_j * 1000.0

    # Power draw at full TDP during inference
    power_draw = tdp * min(latency_s, 1.0)  # capped at 1s duty cycle

    return tdp, energy_mj, power_draw


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------


def profile_inference(
    model_path: Union[str, Path],
    input_data: npt.NDArray[np.floating[Any]],
    n_runs: int = DEFAULT_N_RUNS,
    hardware_profile: str = DEFAULT_HARDWARE,
    warmup_runs: int = WARMUP_RUNS,
    session_options: Optional[dict[str, Any]] = None,
) -> InferenceProfile:
    """Profile inference latency, memory, and power for an ONNX edge model.

    Runs the model ``n_runs`` times on the provided input data and collects
    timing statistics.  Memory is estimated heuristically from model size and
    input shape.  Power is estimated from latency × TDP.

    Args:
        model_path: Path to the ONNX model file.
        input_data: Input numpy array of shape ``[batch, leads, samples]``
            or ``[leads, samples]`` (will be expanded to batch dim).
        n_runs: Number of timed inference runs.  Default ``100``.
        hardware_profile: Hardware profile for power estimation.
            One of ``'rpi4'``, ``'rpi5'``, ``'jetson_nano'``,
            ``'jetson_orin_nano'``.  Default ``'rpi4'``.
        warmup_runs: Number of untimed warmup runs before profiling.
            Default ``5``.
        session_options: Optional ONNX Runtime session options dict.

    Returns:
        :class:`InferenceProfile` with latency statistics, memory
        estimation, and power estimation.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If the model file does not exist.
        ValueError: If ``n_runs`` is not positive.
    """
    _check_onnxruntime()

    model_path = str(Path(model_path))

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")

    # Ensure input has batch dimension
    if input_data.ndim == 2:
        input_data = np.expand_dims(input_data, axis=0)

    input_data = np.asarray(input_data, dtype=np.float32)
    input_shape = tuple(input_data.shape)

    # --- Create ONNX Runtime session ---
    opts = ort.SessionOptions()
    if session_options:
        if "intra_op_num_threads" in session_options:
            opts.intra_op_num_threads = session_options["intra_op_num_threads"]
        if "inter_op_num_threads" in session_options:
            opts.inter_op_num_threads = session_options["inter_op_num_threads"]

    session = ort.InferenceSession(model_path, sess_options=opts)
    input_name = session.get_inputs()[0].name

    # --- Warmup ---
    for _ in range(warmup_runs):
        session.run(None, {input_name: input_data})

    # --- Timed runs ---
    latencies_ms: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        elapsed = time.perf_counter() - start
        latencies_ms.append(elapsed * 1000.0)

    # --- Compute statistics ---
    latencies_arr = np.array(latencies_ms, dtype=np.float64)
    mean_latency = float(np.mean(latencies_arr))
    p50_latency = float(np.percentile(latencies_arr, 50))
    p95_latency = float(np.percentile(latencies_arr, 95))
    min_latency = float(np.min(latencies_arr))
    max_latency = float(np.max(latencies_arr))
    std_latency = float(np.std(latencies_arr))

    # --- Model size ---
    model_size_bytes = os.path.getsize(model_path)

    # --- Memory estimation ---
    peak_memory = _estimate_peak_memory(model_path, input_data)

    # --- Power estimation ---
    tdp, energy_mj, power_draw = _estimate_power(
        mean_latency, hardware_profile,
    )

    return InferenceProfile(
        mean_latency_ms=mean_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        std_latency_ms=std_latency,
        peak_memory_bytes=peak_memory,
        model_size_bytes=model_size_bytes,
        n_runs=n_runs,
        hardware_profile=hardware_profile,
        tdp_watts=tdp,
        energy_per_inference_mj=energy_mj,
        power_draw_watts=power_draw,
        latencies_ms=latencies_ms,
        input_shape=input_shape,
        model_path=model_path,
    )
