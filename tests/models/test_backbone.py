"""Tests for the AorticaBackbone shared encoder (PyTorch)."""

from __future__ import annotations

import pytest

# Gracefully skip all tests if torch is not installed
torch = pytest.importorskip("torch")
nn = torch.nn

from aortica.models.backbone import AorticaBackbone, ResidualBlock1D  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backbone() -> AorticaBackbone:
    """Standard 12-lead backbone with default feature_dim=256."""
    return AorticaBackbone(in_channels=12)


@pytest.fixture
def backbone_small() -> AorticaBackbone:
    """2-lead backbone for fast tests."""
    return AorticaBackbone(in_channels=2, feature_dim=128)


@pytest.fixture
def batch_12lead() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 12, 5000)  # 10s @ 500 Hz


@pytest.fixture
def batch_2lead() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 2, 1000)


# ---------------------------------------------------------------------------
# ResidualBlock1D tests
# ---------------------------------------------------------------------------

class TestResidualBlock1DBackbone:
    def test_identity_block(self) -> None:
        block = ResidualBlock1D(64, 64, kernel_size=7)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 64, 100)

    def test_downsample_block(self) -> None:
        downsample = nn.Sequential(
            nn.Conv1d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm1d(128),
        )
        block = ResidualBlock1D(64, 128, kernel_size=7, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 100)
        out = block(x)
        assert out.shape == (2, 128, 50)


# ---------------------------------------------------------------------------
# AorticaBackbone tests
# ---------------------------------------------------------------------------

class TestAorticaBackbone:
    def test_output_shape_default(
        self, backbone: AorticaBackbone, batch_12lead: torch.Tensor,
    ) -> None:
        out = backbone(batch_12lead)
        assert out.shape == (4, 256)

    def test_output_shape_custom_feature_dim(
        self, backbone_small: AorticaBackbone, batch_2lead: torch.Tensor,
    ) -> None:
        out = backbone_small(batch_2lead)
        assert out.shape == (4, 128)

    def test_varying_sequence_lengths(self, backbone: AorticaBackbone) -> None:
        """Adaptive pooling should handle different signal lengths."""
        for length in [250, 625, 1250, 2500, 5000, 10000]:
            x = torch.randn(2, 12, length)
            out = backbone(x)
            assert out.shape == (2, 256), f"Failed for length={length}"

    def test_varying_sample_rates(self, backbone: AorticaBackbone) -> None:
        """Test with durations/rates that produce different sample counts."""
        configs = [
            (250, 2.5),   # 625 samples
            (500, 5.0),   # 2500 samples
            (500, 10.0),  # 5000 samples
            (1000, 2.5),  # 2500 samples
            (1000, 10.0), # 10000 samples
        ]
        for rate, duration in configs:
            n_samples = int(rate * duration)
            x = torch.randn(2, 12, n_samples)
            out = backbone(x)
            assert out.shape == (2, 256), f"Failed for rate={rate}, dur={duration}"

    def test_single_sample(self, backbone: AorticaBackbone) -> None:
        x = torch.randn(1, 12, 5000)
        out = backbone(x)
        assert out.shape == (1, 256)

    def test_gradient_flow(self, backbone: AorticaBackbone) -> None:
        x = torch.randn(2, 12, 5000, requires_grad=True)
        out = backbone(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_feature_dim_attribute(self, backbone: AorticaBackbone) -> None:
        assert backbone.feature_dim == 256

    def test_custom_feature_dim_attribute(self) -> None:
        bb = AorticaBackbone(feature_dim=512)
        assert bb.feature_dim == 512
        x = torch.randn(1, 12, 2500)
        out = bb(x)
        assert out.shape == (1, 512)

    def test_no_fc_when_default_dim(self, backbone: AorticaBackbone) -> None:
        """When feature_dim=256 matches last stage, no projection layer."""
        assert backbone.fc is None

    def test_has_fc_when_custom_dim(self) -> None:
        bb = AorticaBackbone(feature_dim=128)
        assert bb.fc is not None

    def test_parameter_count(self, backbone: AorticaBackbone) -> None:
        """Backbone should have a reasonable number of parameters."""
        num_params = sum(p.numel() for p in backbone.parameters())
        # 3 stages (64, 128, 256) should result in ~300k-600k params
        assert num_params > 100_000
        assert num_params < 5_000_000

    def test_reproducibility(self) -> None:
        torch.manual_seed(42)
        m1 = AorticaBackbone(in_channels=12)
        torch.manual_seed(42)
        m2 = AorticaBackbone(in_channels=12)

        x = torch.randn(1, 12, 5000)
        assert torch.allclose(m1(x), m2(x))

    def test_eval_mode(self, backbone: AorticaBackbone) -> None:
        """BatchNorm should work in eval mode."""
        backbone.eval()
        x = torch.randn(1, 12, 5000)
        out = backbone(x)
        assert out.shape == (1, 256)

    def test_default_feature_dim_constant(self) -> None:
        assert AorticaBackbone.DEFAULT_FEATURE_DIM == 256


# ---------------------------------------------------------------------------
# TensorFlow backbone tests (skipped if TF not installed)
# ---------------------------------------------------------------------------

class TestAorticaBackboneTF:
    @pytest.fixture(autouse=True)
    def _skip_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    def test_output_shape_default(self) -> None:
        from aortica.models.backbone_tf import build_aortica_backbone_tf

        model = build_aortica_backbone_tf(in_channels=12, input_length=5000)
        import numpy as np
        x = np.random.randn(4, 5000, 12).astype(np.float32)
        out = model(x, training=False)
        assert out.shape == (4, 256)

    def test_output_shape_custom_dim(self) -> None:
        from aortica.models.backbone_tf import build_aortica_backbone_tf

        model = build_aortica_backbone_tf(
            in_channels=2, feature_dim=128, input_length=1000,
        )
        import numpy as np
        x = np.random.randn(4, 1000, 2).astype(np.float32)
        out = model(x, training=False)
        assert out.shape == (4, 128)

    def test_variable_length_input(self) -> None:
        from aortica.models.backbone_tf import build_aortica_backbone_tf

        model = build_aortica_backbone_tf(in_channels=12, input_length=None)
        import numpy as np
        for length in [625, 2500, 5000]:
            x = np.random.randn(2, length, 12).astype(np.float32)
            out = model(x, training=False)
            assert out.shape == (2, 256), f"Failed for length={length}"

    def test_model_summary(self) -> None:
        from aortica.models.backbone_tf import build_aortica_backbone_tf

        model = build_aortica_backbone_tf(in_channels=12, input_length=5000)
        # Should not raise
        model.summary()
