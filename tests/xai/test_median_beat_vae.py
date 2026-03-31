"""Tests for MedianBeatVAE and related utilities.

Covers:
- Model construction and architecture
- Encoder/decoder shapes
- Forward pass and VAEOutput
- Reparameterization trick
- VAE loss computation (reconstruction + KL)
- Median beat extraction
- Training loop with loss convergence
- Latent dimension labelling via Pearson correlation
"""

from __future__ import annotations

import numpy as np
import pytest

# ── torch is optional ────────────────────────────────────────────
torch = pytest.importorskip("torch")

from aortica.xai.median_beat_vae import (  # noqa: E402
    DEFAULT_BEAT_LENGTH,
    DEFAULT_LATENT_DIM,
    LatentLabel,
    MedianBeatVAE,
    TrainResult,
    VAEOutput,
    extract_median_beat,
    label_latent_dimensions,
    train_vae,
    vae_loss,
)

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def vae() -> MedianBeatVAE:
    """Default MedianBeatVAE instance."""
    return MedianBeatVAE(in_channels=12, latent_dim=24, beat_length=250)


@pytest.fixture
def small_vae() -> MedianBeatVAE:
    """Small VAE for faster tests."""
    return MedianBeatVAE(in_channels=2, latent_dim=4, beat_length=64)


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Random input tensor for 12-lead, 250-sample beats."""
    torch.manual_seed(42)
    return torch.randn(4, 12, 250)


@pytest.fixture
def small_input() -> torch.Tensor:
    """Random input for small VAE."""
    torch.manual_seed(42)
    return torch.randn(8, 2, 64)


# ── Constants ────────────────────────────────────────────────────


class TestConstants:
    def test_default_latent_dim(self) -> None:
        assert DEFAULT_LATENT_DIM == 24

    def test_default_beat_length(self) -> None:
        assert DEFAULT_BEAT_LENGTH == 250


# ── Construction ─────────────────────────────────────────────────


class TestConstruction:
    def test_default_params(self, vae: MedianBeatVAE) -> None:
        assert vae.in_channels == 12
        assert vae.latent_dim == 24
        assert vae.beat_length == 250

    def test_custom_params(self) -> None:
        m = MedianBeatVAE(in_channels=3, latent_dim=8, beat_length=100)
        assert m.in_channels == 3
        assert m.latent_dim == 8
        assert m.beat_length == 100

    def test_has_encoder_decoder(self, vae: MedianBeatVAE) -> None:
        assert hasattr(vae, "encoder")
        assert hasattr(vae, "decoder")

    def test_param_count_positive(self, vae: MedianBeatVAE) -> None:
        n_params = sum(p.numel() for p in vae.parameters())
        assert n_params > 0


# ── Encoder ──────────────────────────────────────────────────────


class TestEncoder:
    def test_encode_shapes(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        mu, log_var = vae.encode(sample_input)
        assert mu.shape == (4, 24)
        assert log_var.shape == (4, 24)

    def test_encode_shapes_small(self, small_vae: MedianBeatVAE, small_input: torch.Tensor) -> None:
        mu, log_var = small_vae.encode(small_input)
        assert mu.shape == (8, 4)
        assert log_var.shape == (8, 4)


# ── Decoder ──────────────────────────────────────────────────────


class TestDecoder:
    def test_decode_shape(self, vae: MedianBeatVAE) -> None:
        torch.manual_seed(42)
        z = torch.randn(4, 24)
        out = vae.decode(z)
        assert out.shape == (4, 12, 250)

    def test_decode_shape_small(self, small_vae: MedianBeatVAE) -> None:
        torch.manual_seed(42)
        z = torch.randn(8, 4)
        out = small_vae.decode(z)
        assert out.shape == (8, 2, 64)


# ── Forward Pass ─────────────────────────────────────────────────


class TestForward:
    def test_output_type(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        output = vae(sample_input)
        assert isinstance(output, VAEOutput)

    def test_reconstruction_shape(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        output = vae(sample_input)
        assert output.reconstruction.shape == sample_input.shape

    def test_mu_shape(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        output = vae(sample_input)
        assert output.mu.shape == (4, 24)

    def test_log_var_shape(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        output = vae(sample_input)
        assert output.log_var.shape == (4, 24)

    def test_z_shape(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        output = vae(sample_input)
        assert output.z.shape == (4, 24)


# ── Reparameterization ──────────────────────────────────────────


class TestReparameterize:
    def test_shape(self, vae: MedianBeatVAE) -> None:
        mu = torch.zeros(4, 24)
        log_var = torch.zeros(4, 24)
        z = vae.reparameterize(mu, log_var)
        assert z.shape == (4, 24)

    def test_zero_variance_equals_mu(self, vae: MedianBeatVAE) -> None:
        """With log_var → -inf (var → 0), z should approximately equal mu."""
        mu = torch.ones(4, 24) * 5.0
        log_var = torch.ones(4, 24) * (-100.0)  # exp(-100) ≈ 0
        z = vae.reparameterize(mu, log_var)
        assert torch.allclose(z, mu, atol=1e-6)

    def test_stochasticity(self, vae: MedianBeatVAE) -> None:
        """Two calls with same mu/log_var should differ (stochastic)."""
        mu = torch.zeros(4, 24)
        log_var = torch.zeros(4, 24)
        z1 = vae.reparameterize(mu, log_var)
        z2 = vae.reparameterize(mu, log_var)
        assert not torch.allclose(z1, z2)


# ── VAE Loss ────────────────────────────────────────────────────


class TestVAELoss:
    def test_loss_returns_three(self) -> None:
        recon = torch.randn(4, 12, 250)
        original = torch.randn(4, 12, 250)
        mu = torch.randn(4, 24)
        log_var = torch.randn(4, 24)
        total, r_loss, kl = vae_loss(recon, original, mu, log_var)
        assert total.shape == ()
        assert r_loss.shape == ()
        assert kl.shape == ()

    def test_perfect_reconstruction_zero_recon_loss(self) -> None:
        x = torch.randn(4, 12, 250)
        mu = torch.zeros(4, 24)
        log_var = torch.zeros(4, 24)
        _total, r_loss, _kl = vae_loss(x, x, mu, log_var)
        assert r_loss.item() < 1e-8

    def test_kl_zero_for_prior(self) -> None:
        """KL=0 when q(z|x) equals the prior N(0,1)."""
        x = torch.randn(4, 12, 250)
        mu = torch.zeros(4, 24)
        log_var = torch.zeros(4, 24)
        _total, _r, kl = vae_loss(x, x, mu, log_var)
        assert abs(kl.item()) < 1e-6

    def test_kl_positive_for_nonzero_mu(self) -> None:
        x = torch.randn(4, 12, 250)
        mu = torch.ones(4, 24) * 3.0
        log_var = torch.zeros(4, 24)
        _total, _r, kl = vae_loss(x, x, mu, log_var)
        assert kl.item() > 0

    def test_kl_weight(self) -> None:
        x = torch.randn(4, 12, 250)
        mu = torch.ones(4, 24) * 2.0
        log_var = torch.zeros(4, 24)
        total_1, r, kl = vae_loss(x, x, mu, log_var, kl_weight=1.0)
        total_2, _, _ = vae_loss(x, x, mu, log_var, kl_weight=2.0)
        assert abs(total_2.item() - (r.item() + 2.0 * kl.item())) < 1e-4

    def test_loss_gradient_flows(self) -> None:
        x = torch.randn(4, 12, 250, requires_grad=True)
        mu = torch.randn(4, 24, requires_grad=True)
        log_var = torch.randn(4, 24, requires_grad=True)
        total, _, _ = vae_loss(x, torch.randn(4, 12, 250), mu, log_var)
        total.backward()
        assert x.grad is not None
        assert mu.grad is not None
        assert log_var.grad is not None


# ── Gradient Flow ────────────────────────────────────────────────


class TestGradientFlow:
    def test_gradient_through_vae(
        self, small_vae: MedianBeatVAE, small_input: torch.Tensor,
    ) -> None:
        output = small_vae(small_input)
        total, _, _ = vae_loss(output.reconstruction, small_input, output.mu, output.log_var)
        total.backward()
        for name, param in small_vae.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


# ── Eval Mode ────────────────────────────────────────────────────


class TestEvalMode:
    def test_eval_produces_output(self, vae: MedianBeatVAE, sample_input: torch.Tensor) -> None:
        vae.eval()
        output = vae(sample_input)
        assert output.reconstruction.shape == sample_input.shape


# ── Median Beat Extraction ───────────────────────────────────────


class TestExtractMedianBeat:
    def test_returns_correct_shape(self) -> None:
        """Synthetic ECG with clear R-peaks."""
        np.random.seed(42)
        n_leads, n_samples = 12, 5000
        sr = 500.0
        signals = np.random.randn(n_leads, n_samples) * 0.1

        # Inject R-peaks every 500 samples (60 bpm at 500 Hz)
        for r in range(250, n_samples - 250, 500):
            for lead in range(n_leads):
                signals[lead, r - 5 : r + 5] += 3.0  # QRS-like peak

        median_beat = extract_median_beat(signals, sr, beat_length=200)
        assert median_beat.shape == (12, 200)

    def test_returns_zeros_for_short_signal(self) -> None:
        """Fewer than 2 beats → zeros."""
        signals = np.random.randn(12, 100) * 0.01
        result = extract_median_beat(signals, 500.0, beat_length=50)
        assert result.shape == (12, 50)
        assert np.allclose(result, 0.0)

    def test_single_lead(self) -> None:
        """Works with single-lead signals."""
        np.random.seed(42)
        signals = np.random.randn(1, 5000) * 0.1
        for r in range(250, 4750, 500):
            signals[0, r - 5 : r + 5] += 3.0
        result = extract_median_beat(signals, 500.0, beat_length=200)
        assert result.shape == (1, 200)


# ── Training ─────────────────────────────────────────────────────


class TestTrainVAE:
    def test_train_returns_result(self, small_vae: MedianBeatVAE) -> None:
        np.random.seed(42)
        beats = np.random.randn(32, 2, 64).astype(np.float64)
        result = train_vae(small_vae, beats, epochs=3, lr=1e-3, seed=42)
        assert isinstance(result, TrainResult)
        assert len(result.total_losses) == 3
        assert len(result.recon_losses) == 3
        assert len(result.kl_losses) == 3

    def test_loss_decreases(self, small_vae: MedianBeatVAE) -> None:
        """Loss should generally decrease over training."""
        np.random.seed(42)
        beats = np.random.randn(64, 2, 64).astype(np.float64)
        result = train_vae(small_vae, beats, epochs=20, lr=1e-3, seed=42)
        # First epoch loss > last epoch loss (convergence)
        assert result.total_losses[-1] < result.total_losses[0]

    def test_recon_loss_convergence(self, small_vae: MedianBeatVAE) -> None:
        """Reconstruction loss should converge."""
        np.random.seed(42)
        beats = np.random.randn(64, 2, 64).astype(np.float64)
        result = train_vae(small_vae, beats, epochs=30, lr=1e-3, seed=42)
        assert result.recon_losses[-1] < result.recon_losses[0]

    def test_reproducibility(self) -> None:
        """Same seed produces identical loss trajectories."""
        np.random.seed(0)
        beats = np.random.randn(32, 2, 64).astype(np.float64)

        torch.manual_seed(123)
        vae1 = MedianBeatVAE(in_channels=2, latent_dim=4, beat_length=64)
        r1 = train_vae(vae1, beats, epochs=5, seed=123)

        torch.manual_seed(123)
        vae2 = MedianBeatVAE(in_channels=2, latent_dim=4, beat_length=64)
        r2 = train_vae(vae2, beats, epochs=5, seed=123)

        np.testing.assert_allclose(r1.total_losses, r2.total_losses, atol=1e-5)


# ── Latent Labelling ─────────────────────────────────────────────


class TestLabelLatentDimensions:
    def test_returns_list_of_labels(self, small_vae: MedianBeatVAE) -> None:
        beats = np.random.randn(50, 2, 64).astype(np.float64)
        measurements = {
            "heart_rate": np.random.randn(50),
            "qrs_duration": np.random.randn(50),
        }
        labels = label_latent_dimensions(small_vae, beats, measurements)
        assert len(labels) == 4  # latent_dim=4
        assert all(isinstance(lb, LatentLabel) for lb in labels)

    def test_dimensions_indexed(self, small_vae: MedianBeatVAE) -> None:
        beats = np.random.randn(50, 2, 64).astype(np.float64)
        measurements = {"hr": np.random.randn(50)}
        labels = label_latent_dimensions(small_vae, beats, measurements)
        dims = [lb.dimension for lb in labels]
        assert dims == [0, 1, 2, 3]

    def test_correlated_measurement_detected(self) -> None:
        """If a latent dimension correlates perfectly with a measurement,
        it should be labelled with that measurement."""
        vae = MedianBeatVAE(in_channels=2, latent_dim=4, beat_length=64)
        torch.manual_seed(42)
        # Create data that produces varied latent values
        n = 100
        beats = np.random.randn(n, 2, 64).astype(np.float64)

        # Encode to get actual latent values
        vae.eval()
        with torch.no_grad():
            mu, _ = vae.encode(torch.tensor(beats, dtype=torch.float32))
            latent = mu.numpy()

        # Create measurement perfectly correlated with dimension 0
        perfect_measurement = latent[:, 0].copy()
        measurements = {
            "matched": perfect_measurement,
            "noise": np.random.randn(n),
        }

        labels = label_latent_dimensions(vae, beats, measurements)
        dim0_label = labels[0]
        assert dim0_label.measurement_name == "matched"
        assert abs(dim0_label.pearson_r) > 0.99

    def test_empty_measurements(self, small_vae: MedianBeatVAE) -> None:
        beats = np.random.randn(50, 2, 64).astype(np.float64)
        measurements: dict[str, np.ndarray] = {}
        labels = label_latent_dimensions(small_vae, beats, measurements)
        assert len(labels) == 4
        assert all(lb.measurement_name == "unknown" for lb in labels)
        assert all(lb.pearson_r == 0.0 for lb in labels)
