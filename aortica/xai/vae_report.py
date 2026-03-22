"""VAE Reporter and Synthetic ECG Rendering.

Provides :func:`vae_report`, which generates a :class:`VAEReport`
describing which latent factors are most activated for a given ECG
input and producing synthetic waveforms that show the effect of
varying each top factor ±2σ.

Usage::

    from aortica.xai import vae_report
    report = vae_report(model, vae, ecg_record, n_top=3)
    # report.top_factors  -> list of LatentActivation
    # report.synthetic_waves  -> {dim: {offset: ndarray}}
    # report.baseline_wave  -> ndarray
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn = types.ModuleType("nn")  # type: ignore[assignment]
    nn.Module = _DummyModule  # type: ignore[attr-defined]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for VAE reporter. "
            "Install with: pip install aortica[torch]"
        )


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class LatentActivation:
    """Activation information for a single latent dimension.

    Attributes:
        dimension: Latent dimension index.
        activation: Absolute value of the latent mean for this dimension
            (measure of how strongly activated this factor is).
        mu_value: Raw latent mean value (signed).
        label: Optional human-readable label (from latent dimension
            labelling, e.g. ``"QRS_duration"``).
    """

    dimension: int
    activation: float
    mu_value: float
    label: Optional[str] = None


@dataclass
class VAEReport:
    """Result of :func:`vae_report` for one ECG record.

    Attributes:
        top_factors: Top-N most activated latent factors, sorted by
            activation (descending).
        synthetic_waves: Mapping from latent dimension index to a dict
            of ``{sigma_offset: waveform_array}``.  Each waveform has
            shape ``[channels, beat_length]``.
        baseline_wave: The baseline reconstructed waveform (decoded
            from the unperturbed latent mean), shape
            ``[channels, beat_length]``.
        mu: Latent mean vector, shape ``[latent_dim]``.
        log_var: Latent log-variance vector, shape ``[latent_dim]``.
    """

    top_factors: list[LatentActivation] = field(default_factory=list)
    synthetic_waves: dict[int, dict[float, NDArray[np.float64]]] = field(
        default_factory=dict
    )
    baseline_wave: Optional[NDArray[np.float64]] = None
    mu: Optional[NDArray[np.float64]] = None
    log_var: Optional[NDArray[np.float64]] = None


# ── Synthetic Waveform Generation ─────────────────────────────────


def _generate_synthetic_waves(
    vae: nn.Module,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    dimension: int,
    offsets: list[float],
) -> dict[float, NDArray[np.float64]]:
    """Generate synthetic waveforms by perturbing one latent dimension.

    For each sigma offset, the latent vector is modified as::

        z[dimension] = mu[dimension] + offset * exp(0.5 * log_var[dimension])

    and the decoder is used to generate the corresponding waveform.

    Args:
        vae: A trained :class:`MedianBeatVAE` (or compatible module
            with a ``decode`` method).
        mu: Latent mean tensor, shape ``[1, latent_dim]``.
        log_var: Latent log-variance tensor, shape ``[1, latent_dim]``.
        dimension: Which latent dimension to perturb.
        offsets: List of sigma offsets (e.g. ``[-2.0, -1.0, 0.0, 1.0, 2.0]``).

    Returns:
        Dict mapping each offset to a waveform array of shape
        ``[channels, beat_length]`` with dtype ``float64``.
    """
    _check_torch()

    std = torch.exp(0.5 * log_var)  # [1, latent_dim]
    result: dict[float, NDArray[np.float64]] = {}

    for offset in offsets:
        z = mu.clone()
        z[0, dimension] = mu[0, dimension] + offset * std[0, dimension]

        with torch.no_grad():
            wave = vae.decode(z)  # [1, channels, beat_length]

        wave_np: NDArray[np.float64] = (
            wave.squeeze(0).cpu().numpy().astype(np.float64)
        )
        result[offset] = wave_np

    return result


# ── Public API ────────────────────────────────────────────────────

# Default sigma offsets for synthetic waveform generation.
DEFAULT_SIGMA_OFFSETS: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]


def vae_report(
    model: nn.Module,
    vae: nn.Module,
    ecg_record: ECGRecord,
    n_top: int = 3,
    sigma_offsets: Optional[list[float]] = None,
    labels: Optional[list[object]] = None,
) -> VAEReport:
    """Generate a VAE latent factor report for an ECG record.

    Encodes the median beat extracted from *ecg_record* into the latent
    space, identifies the most activated latent factors, and generates
    synthetic ECG waveforms showing the effect of varying each top
    factor by ±Nσ.

    Parameters
    ----------
    model:
        The main :class:`~aortica.models.AorticaModel` (currently
        unused but part of the API for future integration).
    vae:
        A trained :class:`~aortica.xai.MedianBeatVAE`.
    ecg_record:
        The ECG recording to analyse.
    n_top:
        Number of top latent factors to report.  Clamped to the
        VAE's latent dimensionality.
    sigma_offsets:
        Sigma offsets for synthetic waveform generation.  Defaults to
        ``[-2.0, -1.0, 0.0, 1.0, 2.0]``.
    labels:
        Optional list of :class:`~aortica.xai.LatentLabel` objects
        (one per latent dimension) for annotating top factors with
        human-readable measurement names.

    Returns
    -------
    VAEReport
        Report containing top factors, synthetic waveforms, baseline
        waveform, and latent statistics.
    """
    _check_torch()

    if sigma_offsets is None:
        sigma_offsets = list(DEFAULT_SIGMA_OFFSETS)

    # ── Extract median beat ──────────────────────────────────────
    from aortica.xai.median_beat_vae import extract_median_beat

    median_beat = extract_median_beat(
        ecg_record.signals,
        ecg_record.sample_rate,
        beat_length=vae.beat_length,  # type: ignore[attr-defined]
    )  # [channels, beat_length]

    # ── Encode to latent space ────────────────────────────────────
    vae.eval()  # type: ignore[attr-defined]
    beat_tensor = torch.tensor(
        median_beat, dtype=torch.float32
    ).unsqueeze(0)  # [1, channels, beat_length]

    with torch.no_grad():
        mu, log_var = vae.encode(beat_tensor)  # type: ignore[attr-defined]
        # mu, log_var: [1, latent_dim]

    mu_np: NDArray[np.float64] = mu.squeeze(0).cpu().numpy().astype(np.float64)
    log_var_np: NDArray[np.float64] = (
        log_var.squeeze(0).cpu().numpy().astype(np.float64)
    )

    latent_dim = len(mu_np)
    n_top = min(n_top, latent_dim)

    # ── Identify top activated factors ────────────────────────────
    activations = np.abs(mu_np)
    # Indices sorted by descending absolute activation
    sorted_dims = np.argsort(activations)[::-1][:n_top]

    # Build label lookup if labels are provided
    label_map: dict[int, str] = {}
    if labels is not None:
        for lbl in labels:
            label_map[lbl.dimension] = lbl.measurement_name  # type: ignore[attr-defined]

    top_factors: list[LatentActivation] = []
    for dim_idx in sorted_dims:
        dim_int = int(dim_idx)
        top_factors.append(
            LatentActivation(
                dimension=dim_int,
                activation=float(activations[dim_int]),
                mu_value=float(mu_np[dim_int]),
                label=label_map.get(dim_int),
            )
        )

    # ── Baseline waveform ────────────────────────────────────────
    with torch.no_grad():
        baseline_tensor = vae.decode(mu)  # type: ignore[attr-defined]
    baseline_wave: NDArray[np.float64] = (
        baseline_tensor.squeeze(0).cpu().numpy().astype(np.float64)
    )

    # ── Synthetic waveforms for each top factor ──────────────────
    synthetic_waves: dict[int, dict[float, NDArray[np.float64]]] = {}
    for factor in top_factors:
        synthetic_waves[factor.dimension] = _generate_synthetic_waves(
            vae, mu, log_var, dimension=factor.dimension, offsets=sigma_offsets
        )

    return VAEReport(
        top_factors=top_factors,
        synthetic_waves=synthetic_waves,
        baseline_wave=baseline_wave,
        mu=mu_np,
        log_var=log_var_np,
    )
