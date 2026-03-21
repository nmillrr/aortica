"""Variational Autoencoder for median ECG beats.

Provides :class:`MedianBeatVAE`, a 1D CNN-based variational autoencoder
with a 24-dimensional latent space, designed to encode and decode
median beats extracted from 12-lead ECGs.

Also provides :func:`train_vae` for training the VAE on PTB-XL
median beats, :func:`label_latent_dimensions` for labelling each
latent dimension by Pearson correlation with standard ECG measurements,
and :func:`extract_median_beat` for beat extraction.

Usage::

    from aortica.xai import MedianBeatVAE, train_vae
    vae = MedianBeatVAE(in_channels=12, latent_dim=24)
    losses = train_vae(vae, median_beats, epochs=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

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
            "PyTorch is required for MedianBeatVAE. "
            "Install with: pip install aortica[torch]"
        )


# ── Constants ─────────────────────────────────────────────────────

DEFAULT_LATENT_DIM: int = 24
DEFAULT_BEAT_LENGTH: int = 250  # samples (0.5 s @ 500 Hz)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class VAEOutput:
    """Output from a VAE forward pass.

    Attributes:
        reconstruction: Reconstructed signal, shape ``[batch, channels, samples]``.
        mu: Latent mean, shape ``[batch, latent_dim]``.
        log_var: Latent log-variance, shape ``[batch, latent_dim]``.
        z: Sampled latent vector, shape ``[batch, latent_dim]``.
    """

    reconstruction: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    z: torch.Tensor


@dataclass
class LatentLabel:
    """Label for a single latent dimension.

    Attributes:
        dimension: Latent dimension index.
        measurement_name: ECG measurement name with highest correlation.
        pearson_r: Pearson correlation coefficient.
    """

    dimension: int
    measurement_name: str
    pearson_r: float


# ── Encoder ───────────────────────────────────────────────────────


class _Encoder(nn.Module):
    """1D CNN encoder for median beat VAE.

    Architecture: 3 convolutional layers with BatchNorm + ReLU,
    followed by adaptive pooling and linear projection to
    mean and log-variance vectors.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ) -> None:
        _check_torch()
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(4)  # → [batch, 128, 4]
        self.flatten_dim = 128 * 4  # 512

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent mean and log-variance.

        Args:
            x: Input tensor, shape ``[batch, channels, samples]``.

        Returns:
            Tuple of ``(mu, log_var)``, each shape ``[batch, latent_dim]``.
        """
        h = self.conv_layers(x)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        mu: torch.Tensor = self.fc_mu(h)
        log_var: torch.Tensor = self.fc_log_var(h)
        return mu, log_var


# ── Decoder ───────────────────────────────────────────────────────


class _Decoder(nn.Module):
    """1D transposed CNN decoder for median beat VAE.

    Architecture: linear projection from latent to spatial representation,
    followed by 3 transposed convolutional layers, with final adaptive
    interpolation to the target beat length.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        beat_length: int,
    ) -> None:
        _check_torch()
        super().__init__()

        self.beat_length = beat_length
        self.fc = nn.Linear(latent_dim, 128 * 4)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                32, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to signal reconstruction.

        Args:
            z: Latent tensor, shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed signal, shape ``[batch, channels, beat_length]``.
        """
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4)
        h = self.deconv_layers(h)
        # Adaptive interpolation to exact beat_length
        h = torch.nn.functional.interpolate(
            h, size=self.beat_length, mode="linear", align_corners=False
        )
        return h


# ── VAE Model ────────────────────────────────────────────────────


class MedianBeatVAE(nn.Module):
    """Variational autoencoder for 12-lead median ECG beats.

    Architecture:

    * **Encoder**: 3-layer 1D CNN (32→64→128 filters) with adaptive
      pooling, projecting to *latent_dim*-dimensional mean and
      log-variance vectors.
    * **Decoder**: linear → reshape → 3-layer transposed 1D CNN
      (128→64→32→out_channels) with final interpolation to the
      target beat length.
    * **Latent**: 24-dimensional (default) with standard Gaussian prior.

    Args:
        in_channels: Number of input channels (ECG leads). Default ``12``.
        latent_dim: Latent space dimensionality. Default ``24``.
        beat_length: Expected beat length in samples. Default ``250``
            (~0.5 s at 500 Hz).

    Example::

        vae = MedianBeatVAE(in_channels=12, latent_dim=24, beat_length=250)
        x = torch.randn(8, 12, 250)
        output = vae(x)
        # output.reconstruction.shape == (8, 12, 250)
        # output.mu.shape == (8, 24)
    """

    def __init__(
        self,
        in_channels: int = 12,
        latent_dim: int = DEFAULT_LATENT_DIM,
        beat_length: int = DEFAULT_BEAT_LENGTH,
    ) -> None:
        _check_torch()
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beat_length = beat_length

        self.encoder = _Encoder(in_channels, latent_dim)
        self.decoder = _Decoder(latent_dim, in_channels, beat_length)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample from latent distribution via reparameterization trick.

        Args:
            mu: Latent mean, shape ``[batch, latent_dim]``.
            log_var: Latent log-variance, shape ``[batch, latent_dim]``.

        Returns:
            Sampled latent vector, shape ``[batch, latent_dim]``.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z: torch.Tensor = mu + eps * std
        return z

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent mean and log-variance.

        Args:
            x: Input tensor, shape ``[batch, channels, samples]``.

        Returns:
            Tuple of ``(mu, log_var)``, each shape ``[batch, latent_dim]``.
        """
        mu, log_var = self.encoder(x)
        mu_t: torch.Tensor = mu
        log_var_t: torch.Tensor = log_var
        return mu_t, log_var_t

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to signal reconstruction.

        Args:
            z: Latent tensor, shape ``[batch, latent_dim]``.

        Returns:
            Reconstructed signal, shape ``[batch, channels, beat_length]``.
        """
        out: torch.Tensor = self.decoder(z)
        return out

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Full VAE forward pass: encode → reparameterize → decode.

        Args:
            x: Input tensor, shape ``[batch, channels, samples]``.

        Returns:
            :class:`VAEOutput` containing reconstruction, mu, log_var, z.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return VAEOutput(
            reconstruction=reconstruction,
            mu=mu,
            log_var=log_var,
            z=z,
        )


# ── Loss Function ────────────────────────────────────────────────


def vae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: reconstruction + KL divergence.

    Args:
        reconstruction: Reconstructed signal from decoder.
        original: Original input signal.
        mu: Latent mean from encoder.
        log_var: Latent log-variance from encoder.
        kl_weight: Weight for KL divergence term (beta-VAE).

    Returns:
        Tuple of ``(total_loss, recon_loss, kl_loss)``.
    """
    _check_torch()

    recon_loss = torch.nn.functional.mse_loss(
        reconstruction, original, reduction="mean"
    )
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ── Median Beat Extraction ───────────────────────────────────────


def extract_median_beat(
    signals: NDArray[np.float64],
    sample_rate: float,
    beat_length: int = DEFAULT_BEAT_LENGTH,
) -> NDArray[np.float64]:
    """Extract the median beat from a multi-lead ECG signal.

    Detects R-peaks on Lead II (or the first available lead), segments
    individual beats centered on each R-peak, and returns the
    element-wise median across all beats.

    Args:
        signals: ECG signal array, shape ``[leads, samples]``.
        sample_rate: Sampling rate in Hz.
        beat_length: Desired beat length in samples.

    Returns:
        Median beat array, shape ``[leads, beat_length]``.
        Returns zeros if fewer than 2 beats are detected.
    """
    from aortica.signal.qrs_detection import _detect_pan_tompkins

    n_leads = signals.shape[0]

    # Use lead index 1 (Lead II) if available, else lead 0
    detect_lead = 1 if n_leads > 1 else 0
    r_peaks = _detect_pan_tompkins(
        signals[detect_lead].astype(np.float64), sample_rate
    )

    if len(r_peaks) < 2:
        return np.zeros((n_leads, beat_length), dtype=np.float64)

    half = beat_length // 2
    beats: list[NDArray[np.float64]] = []

    for r in r_peaks:
        start = int(r) - half
        end = start + beat_length
        if start < 0 or end > signals.shape[1]:
            continue
        beat = signals[:, start:end].astype(np.float64)
        beats.append(beat)

    if len(beats) < 2:
        return np.zeros((n_leads, beat_length), dtype=np.float64)

    stacked = np.stack(beats, axis=0)  # [n_beats, leads, beat_length]
    median_beat: NDArray[np.float64] = np.median(stacked, axis=0)
    return median_beat


# ── Training ─────────────────────────────────────────────────────


@dataclass
class TrainResult:
    """Result of VAE training.

    Attributes:
        total_losses: Per-epoch total loss values.
        recon_losses: Per-epoch reconstruction loss values.
        kl_losses: Per-epoch KL divergence loss values.
    """

    total_losses: list[float]
    recon_losses: list[float]
    kl_losses: list[float]


def train_vae(
    vae: MedianBeatVAE,
    median_beats: NDArray[np.float64],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    seed: Optional[int] = None,
) -> TrainResult:
    """Train the MedianBeatVAE on a set of median beats.

    Args:
        vae: The :class:`MedianBeatVAE` instance to train.
        median_beats: Array of median beats, shape
            ``[n_samples, channels, beat_length]``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        kl_weight: Weight for KL divergence term.
        seed: Random seed for reproducibility.

    Returns:
        :class:`TrainResult` with per-epoch loss histories.
    """
    _check_torch()

    if seed is not None:
        torch.manual_seed(seed)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(median_beats, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    total_losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []

    vae.train()
    for _epoch in range(epochs):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            output = vae(batch)
            total, recon, kl = vae_loss(
                output.reconstruction, batch,
                output.mu, output.log_var,
                kl_weight=kl_weight,
            )
            total.backward()
            optimizer.step()

            epoch_total += total.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        total_losses.append(epoch_total / max(n_batches, 1))
        recon_losses.append(epoch_recon / max(n_batches, 1))
        kl_losses.append(epoch_kl / max(n_batches, 1))

    return TrainResult(
        total_losses=total_losses,
        recon_losses=recon_losses,
        kl_losses=kl_losses,
    )


# ── Latent Dimension Labelling ───────────────────────────────────


def label_latent_dimensions(
    vae: MedianBeatVAE,
    median_beats: NDArray[np.float64],
    measurements: dict[str, NDArray[np.float64]],
) -> list[LatentLabel]:
    """Label each latent dimension by Pearson correlation with ECG measurements.

    Encodes all median beats to obtain latent representations, then
    computes the Pearson correlation between each latent dimension
    and each ECG measurement.  Each dimension is assigned the
    measurement with the highest absolute correlation.

    Args:
        vae: A trained :class:`MedianBeatVAE`.
        median_beats: Array of median beats, shape
            ``[n_samples, channels, beat_length]``.
        measurements: Dict mapping measurement name (e.g. ``"QRS_duration"``,
            ``"heart_rate"``) to a 1-D array of values, one per sample.

    Returns:
        List of :class:`LatentLabel`, one per latent dimension, sorted
        by dimension index.
    """
    _check_torch()

    # Encode all beats to latent space
    vae.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(median_beats, dtype=torch.float32)
        mu, _log_var = vae.encode(input_tensor)
        latent_np: NDArray[np.float64] = mu.cpu().numpy().astype(np.float64)

    n_dims = latent_np.shape[1]
    labels: list[LatentLabel] = []

    for dim_idx in range(n_dims):
        dim_values = latent_np[:, dim_idx]
        best_name = "unknown"
        best_r = 0.0

        for meas_name, meas_values in measurements.items():
            if len(meas_values) != len(dim_values):
                continue
            # Pearson correlation
            std_dim = np.std(dim_values)
            std_meas = np.std(meas_values)
            if std_dim < 1e-12 or std_meas < 1e-12:
                continue
            r_val = float(
                np.corrcoef(dim_values, meas_values)[0, 1]
            )
            if np.isnan(r_val):
                continue
            if abs(r_val) > abs(best_r):
                best_r = r_val
                best_name = meas_name

        labels.append(
            LatentLabel(
                dimension=dim_idx,
                measurement_name=best_name,
                pearson_r=best_r,
            )
        )

    return labels
