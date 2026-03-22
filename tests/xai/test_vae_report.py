"""Tests for aortica.xai.vae_report — VAE Reporter and Synthetic ECG Rendering."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aortica.io.ecg_record import ECGRecord  # noqa: E402
from aortica.xai.median_beat_vae import (  # noqa: E402
    LatentLabel,
    MedianBeatVAE,
)
from aortica.xai.vae_report import (  # noqa: E402
    LatentActivation,
    VAEReport,
    _generate_synthetic_waves,
    vae_report,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def vae() -> MedianBeatVAE:
    """A small VAE for testing."""
    torch.manual_seed(42)
    return MedianBeatVAE(in_channels=2, latent_dim=8, beat_length=100)


@pytest.fixture()
def ecg_record() -> ECGRecord:
    """Synthetic 2-lead ECG with clear R-peaks for median beat extraction."""
    rng = np.random.default_rng(42)
    sample_rate = 500.0
    duration = 3.0  # 3 seconds → 1500 samples
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Create a signal with clear R-peaks at ~75 bpm (every 0.8 s)
    signals = np.zeros((2, n_samples), dtype=np.float64)
    rr_interval = 0.8  # seconds
    for lead in range(2):
        for beat_time in np.arange(0.4, duration - 0.4, rr_interval):
            # Gaussian QRS complex
            qrs_center = int(beat_time * sample_rate)
            qrs_width = int(0.02 * sample_rate)  # ~20 ms width
            for i in range(
                max(qrs_center - 3 * qrs_width, 0),
                min(qrs_center + 3 * qrs_width, n_samples),
            ):
                signals[lead, i] += 1.5 * np.exp(
                    -0.5 * ((i - qrs_center) / qrs_width) ** 2
                )
        # Add small baseline noise
        signals[lead] += rng.normal(0, 0.02, n_samples)

    return ECGRecord(
        signals=signals,
        sample_rate=sample_rate,
        lead_names=["II", "V1"],
        duration_seconds=duration,
        source_format="synthetic",
        units="µV",
    )


@pytest.fixture()
def dummy_model() -> torch.nn.Module:
    """A dummy model (not used by vae_report logic but required by API)."""

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    return DummyModel()


# ── Data Class Tests ──────────────────────────────────────────────


class TestLatentActivation:
    """Tests for :class:`LatentActivation` dataclass."""

    def test_construction(self) -> None:
        la = LatentActivation(dimension=3, activation=1.5, mu_value=-1.5)
        assert la.dimension == 3
        assert la.activation == 1.5
        assert la.mu_value == -1.5
        assert la.label is None

    def test_with_label(self) -> None:
        la = LatentActivation(
            dimension=0, activation=2.0, mu_value=2.0, label="QRS_duration"
        )
        assert la.label == "QRS_duration"


class TestVAEReport:
    """Tests for :class:`VAEReport` dataclass."""

    def test_default_construction(self) -> None:
        report = VAEReport()
        assert report.top_factors == []
        assert report.synthetic_waves == {}
        assert report.baseline_wave is None
        assert report.mu is None
        assert report.log_var is None

    def test_full_construction(self) -> None:
        wave = np.zeros((2, 100), dtype=np.float64)
        mu = np.zeros(8, dtype=np.float64)
        report = VAEReport(
            top_factors=[
                LatentActivation(dimension=0, activation=1.0, mu_value=1.0)
            ],
            synthetic_waves={0: {0.0: wave}},
            baseline_wave=wave,
            mu=mu,
            log_var=mu,
        )
        assert len(report.top_factors) == 1
        assert 0 in report.synthetic_waves
        assert report.baseline_wave is not None


# ── _generate_synthetic_waves Tests ──────────────────────────────


class TestGenerateSyntheticWaves:
    """Tests for :func:`_generate_synthetic_waves`."""

    def test_returns_dict_with_correct_offsets(self, vae: MedianBeatVAE) -> None:
        vae.eval()
        mu = torch.randn(1, 8)
        log_var = torch.zeros(1, 8)
        offsets = [-2.0, 0.0, 2.0]

        result = _generate_synthetic_waves(vae, mu, log_var, dimension=0, offsets=offsets)

        assert set(result.keys()) == {-2.0, 0.0, 2.0}

    def test_waveform_shapes(self, vae: MedianBeatVAE) -> None:
        vae.eval()
        mu = torch.randn(1, 8)
        log_var = torch.zeros(1, 8)
        offsets = [-1.0, 0.0, 1.0]

        result = _generate_synthetic_waves(vae, mu, log_var, dimension=0, offsets=offsets)

        for offset, wave in result.items():
            assert wave.shape == (2, 100), f"offset {offset}: {wave.shape}"
            assert wave.dtype == np.float64

    def test_zero_offset_matches_baseline(self, vae: MedianBeatVAE) -> None:
        vae.eval()
        mu = torch.randn(1, 8)
        log_var = torch.zeros(1, 8)

        result = _generate_synthetic_waves(
            vae, mu, log_var, dimension=0, offsets=[0.0]
        )

        # Zero offset means z = mu (no perturbation)
        with torch.no_grad():
            baseline = vae.decode(mu).squeeze(0).cpu().numpy().astype(np.float64)

        np.testing.assert_allclose(result[0.0], baseline, atol=1e-5)

    def test_different_offsets_produce_different_waves(
        self, vae: MedianBeatVAE
    ) -> None:
        vae.eval()
        torch.manual_seed(99)
        mu = torch.randn(1, 8)
        log_var = torch.ones(1, 8) * 0.5  # non-zero variance

        result = _generate_synthetic_waves(
            vae, mu, log_var, dimension=0, offsets=[-2.0, 0.0, 2.0]
        )

        # Perturbed waves should differ from baseline
        assert not np.allclose(result[-2.0], result[0.0], atol=1e-5)
        assert not np.allclose(result[2.0], result[0.0], atol=1e-5)

    def test_different_dimensions_produce_different_waves(
        self, vae: MedianBeatVAE
    ) -> None:
        vae.eval()
        torch.manual_seed(99)
        mu = torch.randn(1, 8)
        log_var = torch.ones(1, 8) * 0.5

        result_dim0 = _generate_synthetic_waves(
            vae, mu, log_var, dimension=0, offsets=[2.0]
        )
        result_dim1 = _generate_synthetic_waves(
            vae, mu, log_var, dimension=1, offsets=[2.0]
        )

        # Different dimensions should produce different perturbations
        assert not np.allclose(result_dim0[2.0], result_dim1[2.0], atol=1e-5)


# ── vae_report Tests ──────────────────────────────────────────────


class TestVaeReport:
    """Tests for :func:`vae_report`."""

    def test_returns_vae_report(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=3)
        assert isinstance(report, VAEReport)

    def test_top_factors_count(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=3)
        assert len(report.top_factors) == 3

    def test_top_factors_sorted_descending(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=5)
        activations = [f.activation for f in report.top_factors]
        assert activations == sorted(activations, reverse=True)

    def test_top_factors_are_latent_activations(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=3)
        for factor in report.top_factors:
            assert isinstance(factor, LatentActivation)
            assert 0 <= factor.dimension < 8
            assert factor.activation >= 0
            assert factor.activation == pytest.approx(abs(factor.mu_value))

    def test_n_top_clamped_to_latent_dim(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        # Request more factors than latent dimensions
        report = vae_report(dummy_model, vae, ecg_record, n_top=100)
        assert len(report.top_factors) == 8  # latent_dim = 8

    def test_synthetic_waves_keys(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=3)
        # One entry per top factor
        assert len(report.synthetic_waves) == 3
        for factor in report.top_factors:
            assert factor.dimension in report.synthetic_waves

    def test_synthetic_waves_default_offsets(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=2)
        for dim, waves in report.synthetic_waves.items():
            assert set(waves.keys()) == {-2.0, -1.0, 0.0, 1.0, 2.0}

    def test_synthetic_waves_custom_offsets(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(
            dummy_model, vae, ecg_record, n_top=2, sigma_offsets=[-1.0, 1.0]
        )
        for dim, waves in report.synthetic_waves.items():
            assert set(waves.keys()) == {-1.0, 1.0}

    def test_synthetic_wave_shapes(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=2)
        for dim, waves in report.synthetic_waves.items():
            for offset, wave in waves.items():
                assert wave.shape == (2, 100), f"dim {dim}, offset {offset}"
                assert wave.dtype == np.float64

    def test_baseline_wave_shape(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=2)
        assert report.baseline_wave is not None
        assert report.baseline_wave.shape == (2, 100)
        assert report.baseline_wave.dtype == np.float64

    def test_mu_and_log_var_shapes(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=2)
        assert report.mu is not None
        assert report.log_var is not None
        assert report.mu.shape == (8,)
        assert report.log_var.shape == (8,)

    def test_baseline_matches_zero_offset_synthetic(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(
            dummy_model, vae, ecg_record, n_top=1, sigma_offsets=[0.0]
        )
        dim = report.top_factors[0].dimension
        np.testing.assert_allclose(
            report.baseline_wave,
            report.synthetic_waves[dim][0.0],
            atol=1e-5,
        )

    def test_with_labels(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        labels = [
            LatentLabel(dimension=i, measurement_name=f"meas_{i}", pearson_r=0.5)
            for i in range(8)
        ]
        report = vae_report(
            dummy_model, vae, ecg_record, n_top=3, labels=labels
        )
        for factor in report.top_factors:
            assert factor.label is not None
            assert factor.label == f"meas_{factor.dimension}"

    def test_without_labels(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=3)
        for factor in report.top_factors:
            assert factor.label is None

    def test_unique_dimensions_in_top_factors(
        self, dummy_model: torch.nn.Module, vae: MedianBeatVAE, ecg_record: ECGRecord
    ) -> None:
        report = vae_report(dummy_model, vae, ecg_record, n_top=5)
        dims = [f.dimension for f in report.top_factors]
        assert len(dims) == len(set(dims)), "Top factors should have unique dimensions"


# ── End-to-End Pipeline Test ─────────────────────────────────────


class TestEndToEnd:
    """Full pipeline test: train VAE → generate report."""

    def test_train_then_report(
        self, dummy_model: torch.nn.Module, ecg_record: ECGRecord
    ) -> None:
        from aortica.xai.median_beat_vae import (
            extract_median_beat,
            train_vae,
        )

        # Create a small VAE
        torch.manual_seed(42)
        vae = MedianBeatVAE(in_channels=2, latent_dim=4, beat_length=100)

        # Extract median beats (use the same signal repeated)
        beat = extract_median_beat(
            ecg_record.signals, ecg_record.sample_rate, beat_length=100
        )
        # Stack into a small training set
        beats = np.stack([beat] * 16, axis=0)  # [16, 2, 100]

        # Train briefly
        result = train_vae(vae, beats, epochs=3, batch_size=8, seed=42)
        assert len(result.total_losses) == 3

        # Generate report
        report = vae_report(dummy_model, vae, ecg_record, n_top=2)
        assert isinstance(report, VAEReport)
        assert len(report.top_factors) == 2
        assert len(report.synthetic_waves) == 2
        assert report.baseline_wave is not None
        assert report.mu is not None
        assert report.mu.shape == (4,)


# ── Import from Package Test ─────────────────────────────────────


class TestImports:
    """Verify public symbols are importable from the xai package."""

    def test_import_vae_report(self) -> None:
        from aortica.xai import vae_report as vr_func

        assert callable(vr_func)

    def test_import_vae_report_class(self) -> None:
        from aortica.xai import VAEReport as VR

        assert VR is VAEReport

    def test_import_latent_activation(self) -> None:
        from aortica.xai import LatentActivation as LA

        assert LA is LatentActivation
