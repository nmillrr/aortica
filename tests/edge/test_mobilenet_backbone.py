"""Tests for :mod:`aortica.edge.mobilenet_backbone` — MobileNetBackbone1D."""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from aortica.edge.mobilenet_backbone import (  # noqa: E402
    DepthwiseSeparableConv1D,
    InvertedResidual1D,
    MobileNetBackbone1D,
)

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

BATCH = 4
LEADS = 12
SAMPLES = 5000  # 10 s @ 500 Hz
DEFAULT_FEATURE_DIM = 256


@pytest.fixture
def backbone() -> MobileNetBackbone1D:
    """Default MobileNetBackbone1D instance."""
    return MobileNetBackbone1D(in_channels=LEADS)


# ---------------------------------------------------------------------------
# DepthwiseSeparableConv1D
# ---------------------------------------------------------------------------


class TestDepthwiseSeparableConv1D:
    """Tests for the depthwise-separable 1D conv building block."""

    def test_output_shape_no_stride(self) -> None:
        block = DepthwiseSeparableConv1D(16, 32, kernel_size=7, stride=1)
        x = torch.randn(2, 16, 100)
        y = block(x)
        assert y.shape == (2, 32, 100)

    def test_output_shape_with_stride(self) -> None:
        block = DepthwiseSeparableConv1D(16, 32, kernel_size=7, stride=2)
        x = torch.randn(2, 16, 100)
        y = block(x)
        assert y.shape == (2, 32, 50)

    def test_gradient_flows(self) -> None:
        block = DepthwiseSeparableConv1D(8, 16)
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# InvertedResidual1D
# ---------------------------------------------------------------------------


class TestInvertedResidual1D:
    """Tests for the inverted residual block."""

    def test_output_shape_no_stride(self) -> None:
        block = InvertedResidual1D(32, 32, kernel_size=7, stride=1)
        x = torch.randn(2, 32, 100)
        y = block(x)
        assert y.shape == (2, 32, 100)

    def test_residual_connection_active(self) -> None:
        """When in_ch == out_ch and stride == 1, residual should be active."""
        block = InvertedResidual1D(32, 32, stride=1)
        assert block.use_residual is True

    def test_residual_connection_inactive_on_stride(self) -> None:
        block = InvertedResidual1D(32, 32, stride=2)
        assert block.use_residual is False

    def test_residual_connection_inactive_on_channel_change(self) -> None:
        block = InvertedResidual1D(32, 64, stride=1)
        assert block.use_residual is False

    def test_output_shape_with_stride(self) -> None:
        block = InvertedResidual1D(32, 64, kernel_size=7, stride=2)
        x = torch.randn(2, 32, 100)
        y = block(x)
        assert y.shape == (2, 64, 50)

    def test_gradient_flows(self) -> None:
        block = InvertedResidual1D(16, 32, stride=2)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — construction
# ---------------------------------------------------------------------------


class TestMobileNetConstruction:
    """Construction and configuration tests."""

    def test_default_construction(self, backbone: MobileNetBackbone1D) -> None:
        assert backbone.feature_dim == DEFAULT_FEATURE_DIM

    def test_stage_channels(self) -> None:
        assert MobileNetBackbone1D.STAGE_CHANNELS == (32, 64, 128)

    def test_custom_feature_dim(self) -> None:
        bb = MobileNetBackbone1D(feature_dim=128)
        assert bb.feature_dim == 128
        # fc should be present (128 != last stage ch 128: actually equal!)
        # 128 == 128 so fc should be None
        assert bb.fc is None

    def test_custom_feature_dim_needs_projection(self) -> None:
        bb = MobileNetBackbone1D(feature_dim=256)
        # 256 != 128 (last stage), so fc should exist
        assert bb.fc is not None

    def test_feature_dim_matches_last_stage_no_fc(self) -> None:
        """When feature_dim equals last stage width (128), no fc needed."""
        bb = MobileNetBackbone1D(feature_dim=128)
        assert bb.fc is None


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — forward pass / shapes
# ---------------------------------------------------------------------------


class TestMobileNetForward:
    """Forward pass shape and correctness tests."""

    def test_output_shape_default(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(BATCH, LEADS, SAMPLES)
        y = backbone(x)
        assert y.shape == (BATCH, DEFAULT_FEATURE_DIM)

    def test_output_shape_short_signal(self, backbone: MobileNetBackbone1D) -> None:
        """2.5 s @ 500 Hz = 1250 samples."""
        x = torch.randn(BATCH, LEADS, 1250)
        y = backbone(x)
        assert y.shape == (BATCH, DEFAULT_FEATURE_DIM)

    def test_output_shape_250hz(self, backbone: MobileNetBackbone1D) -> None:
        """10 s @ 250 Hz = 2500 samples."""
        x = torch.randn(BATCH, LEADS, 2500)
        y = backbone(x)
        assert y.shape == (BATCH, DEFAULT_FEATURE_DIM)

    def test_output_shape_1000hz(self, backbone: MobileNetBackbone1D) -> None:
        """10 s @ 1000 Hz = 10000 samples."""
        x = torch.randn(2, LEADS, 10000)
        y = backbone(x)
        assert y.shape == (2, DEFAULT_FEATURE_DIM)

    def test_output_shape_custom_feature_dim(self) -> None:
        bb = MobileNetBackbone1D(feature_dim=64)
        x = torch.randn(2, LEADS, SAMPLES)
        y = bb(x)
        assert y.shape == (2, 64)

    def test_output_shape_single_lead(self) -> None:
        bb = MobileNetBackbone1D(in_channels=1)
        x = torch.randn(2, 1, SAMPLES)
        y = bb(x)
        assert y.shape == (2, DEFAULT_FEATURE_DIM)

    def test_batch_size_one(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(1, LEADS, SAMPLES)
        y = backbone(x)
        assert y.shape == (1, DEFAULT_FEATURE_DIM)

    def test_output_is_float(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(2, LEADS, SAMPLES)
        y = backbone(x)
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — parameter count
# ---------------------------------------------------------------------------


class TestMobileNetParamCount:
    """Parameter count must stay within the 2.5M budget."""

    def test_param_count_under_budget(self, backbone: MobileNetBackbone1D) -> None:
        total = sum(p.numel() for p in backbone.parameters())
        assert total <= 2_500_000, (
            f"Parameter count {total:,} exceeds 2.5M budget"
        )

    def test_param_count_reasonable_minimum(self, backbone: MobileNetBackbone1D) -> None:
        """Sanity check — should have a reasonable number of params."""
        total = sum(p.numel() for p in backbone.parameters())
        assert total > 10_000, f"Suspiciously low param count: {total:,}"

    def test_fewer_params_than_full_backbone(self) -> None:
        from aortica.models.backbone import AorticaBackbone

        full = AorticaBackbone(in_channels=LEADS)
        edge = MobileNetBackbone1D(in_channels=LEADS)

        full_params = sum(p.numel() for p in full.parameters())
        edge_params = sum(p.numel() for p in edge.parameters())

        assert edge_params < full_params, (
            f"Edge ({edge_params:,}) should have fewer params than "
            f"full ({full_params:,})"
        )


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — gradient flow
# ---------------------------------------------------------------------------


class TestMobileNetGradientFlow:
    """Verify gradients propagate through the entire model."""

    def test_gradient_flow_all_params(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(2, LEADS, SAMPLES, requires_grad=True)
        y = backbone(x)
        loss = y.sum()
        loss.backward()

        # All parameters should have gradients.
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_reaches_input(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(2, LEADS, SAMPLES, requires_grad=True)
        y = backbone(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_nonzero(self, backbone: MobileNetBackbone1D) -> None:
        x = torch.randn(2, LEADS, SAMPLES, requires_grad=True)
        y = backbone(x)
        loss = y.sum()
        loss.backward()
        # At least some gradients should be non-zero.
        has_nonzero = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in backbone.parameters()
        )
        assert has_nonzero


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — eval / train modes
# ---------------------------------------------------------------------------


class TestMobileNetModes:
    """Eval and train mode behaviour."""

    def test_eval_mode_deterministic(self, backbone: MobileNetBackbone1D) -> None:
        backbone.eval()
        x = torch.randn(2, LEADS, SAMPLES)
        y1 = backbone(x)
        y2 = backbone(x)
        torch.testing.assert_close(y1, y2)

    def test_train_mode(self, backbone: MobileNetBackbone1D) -> None:
        backbone.train()
        x = torch.randn(2, LEADS, SAMPLES)
        y = backbone(x)
        assert y.shape == (2, DEFAULT_FEATURE_DIM)


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — reproducibility
# ---------------------------------------------------------------------------


class TestMobileNetReproducibility:
    """Reproducibility with fixed seeds."""

    def test_same_weights_same_output(self) -> None:
        torch.manual_seed(42)
        bb1 = MobileNetBackbone1D(in_channels=LEADS)
        bb1.eval()

        torch.manual_seed(42)
        bb2 = MobileNetBackbone1D(in_channels=LEADS)
        bb2.eval()

        x = torch.randn(2, LEADS, SAMPLES)
        y1 = bb1(x)
        y2 = bb2(x)
        torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# MobileNetBackbone1D — compatibility with task heads
# ---------------------------------------------------------------------------


class TestMobileNetCompatibility:
    """Verify the edge backbone can be used with existing task heads."""

    def test_output_compatible_with_attention(self) -> None:
        """Feature dim should be usable by CrossLeadAttention (divisible by leads)."""
        # feature_dim=252 (divisible by 12 for attention)
        bb = MobileNetBackbone1D(in_channels=LEADS, feature_dim=252)
        x = torch.randn(2, LEADS, SAMPLES)
        y = bb(x)
        assert y.shape == (2, 252)
        assert 252 % LEADS == 0  # compatible with attention module

    def test_default_feature_dim_value(self) -> None:
        """Default feature_dim should match the class constant."""
        assert MobileNetBackbone1D.DEFAULT_FEATURE_DIM == 256


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestImports:
    """Import and export checks."""

    def test_importable_from_edge_package(self) -> None:
        mod = importlib.import_module("aortica.edge")
        assert hasattr(mod, "MobileNetBackbone1D")

    def test_importable_from_mobilenet_module(self) -> None:
        mod = importlib.import_module("aortica.edge.mobilenet_backbone")
        assert hasattr(mod, "MobileNetBackbone1D")
        assert hasattr(mod, "DepthwiseSeparableConv1D")
        assert hasattr(mod, "InvertedResidual1D")

    def test_all_exports(self) -> None:
        from aortica.edge import __all__

        assert "MobileNetBackbone1D" in __all__
