"""Tests for the CrossLeadAttention module (PyTorch and TF/Keras)."""

from __future__ import annotations

import pytest

# Gracefully skip all torch tests if torch is not installed
torch = pytest.importorskip("torch")
nn = torch.nn

from aortica.models.attention import CrossLeadAttention  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def attn() -> CrossLeadAttention:
    """Default 12-lead, 4-head, 64-dim attention module."""
    return CrossLeadAttention(
        feature_dim=240, num_leads=12, num_heads=4, head_dim=64,
    )


@pytest.fixture
def attn_small() -> CrossLeadAttention:
    """Smaller attention module for fast tests."""
    return CrossLeadAttention(
        feature_dim=24, num_leads=6, num_heads=2, head_dim=16,
    )


@pytest.fixture
def features_12lead() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 240)


@pytest.fixture
def features_small() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 24)


# ---------------------------------------------------------------------------
# CrossLeadAttention PyTorch tests
# ---------------------------------------------------------------------------


class TestCrossLeadAttention:
    def test_output_shape(
        self, attn: CrossLeadAttention, features_12lead: torch.Tensor,
    ) -> None:
        out = attn(features_12lead)
        assert out.shape == (4, 240)

    def test_output_shape_small(
        self, attn_small: CrossLeadAttention, features_small: torch.Tensor,
    ) -> None:
        out = attn_small(features_small)
        assert out.shape == (4, 24)

    def test_single_sample(self, attn: CrossLeadAttention) -> None:
        x = torch.randn(1, 240)
        out = attn(x)
        assert out.shape == (1, 240)

    def test_attention_weights_shape(
        self, attn: CrossLeadAttention, features_12lead: torch.Tensor,
    ) -> None:
        attn(features_12lead)
        w = attn.attention_weights
        assert w is not None
        # [batch, num_heads, num_leads, num_leads]
        assert w.shape == (4, 4, 12, 12)

    def test_attention_weights_sum_to_one(
        self, attn: CrossLeadAttention, features_12lead: torch.Tensor,
    ) -> None:
        attn(features_12lead)
        w = attn.attention_weights
        assert w is not None
        # Each row of attention should sum to 1 (softmax)
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_nonnegative(
        self, attn: CrossLeadAttention, features_12lead: torch.Tensor,
    ) -> None:
        attn(features_12lead)
        w = attn.attention_weights
        assert w is not None
        assert (w >= 0).all()

    def test_attention_weights_none_before_forward(self) -> None:
        attn = CrossLeadAttention(feature_dim=240, num_leads=12)
        assert attn.attention_weights is None

    def test_attention_weights_detached(
        self, attn: CrossLeadAttention, features_12lead: torch.Tensor,
    ) -> None:
        """Stored attention weights should be detached from the graph."""
        attn(features_12lead)
        w = attn.attention_weights
        assert w is not None
        assert not w.requires_grad

    def test_gradient_flow(
        self, attn: CrossLeadAttention,
    ) -> None:
        x = torch.randn(2, 240, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_feature_dim_attribute(self, attn: CrossLeadAttention) -> None:
        assert attn.feature_dim == 240

    def test_num_heads_attribute(self, attn: CrossLeadAttention) -> None:
        assert attn.num_heads == 4

    def test_invalid_feature_dim_not_divisible(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            CrossLeadAttention(feature_dim=100, num_leads=12)

    def test_reproducibility(self) -> None:
        torch.manual_seed(42)
        m1 = CrossLeadAttention(feature_dim=240, num_leads=12)
        torch.manual_seed(42)
        m2 = CrossLeadAttention(feature_dim=240, num_leads=12)

        x = torch.randn(2, 240)
        assert torch.allclose(m1(x), m2(x))

    def test_eval_mode(self, attn: CrossLeadAttention) -> None:
        """Attention should work in eval mode."""
        attn.eval()
        x = torch.randn(1, 240)
        out = attn(x)
        assert out.shape == (1, 240)

    def test_with_dropout(self) -> None:
        """Module with dropout should still produce correct shapes."""
        attn = CrossLeadAttention(
            feature_dim=240, num_leads=12, num_heads=4, head_dim=64,
            dropout=0.1,
        )
        x = torch.randn(4, 240)
        out = attn(x)
        assert out.shape == (4, 240)

    def test_parameter_count(self, attn: CrossLeadAttention) -> None:
        """Should have a reasonable number of parameters."""
        num_params = sum(p.numel() for p in attn.parameters())
        # Q, K, V projections + out_proj + LayerNorm
        assert num_params > 1_000
        assert num_params < 1_000_000

    def test_different_head_dims(self) -> None:
        """Different head_dim values should work."""
        for hd in [16, 32, 64, 128]:
            attn = CrossLeadAttention(
                feature_dim=240, num_leads=12, num_heads=4, head_dim=hd,
            )
            x = torch.randn(2, 240)
            out = attn(x)
            assert out.shape == (2, 240)

    def test_backbone_integration(self) -> None:
        """Test that attention works with actual backbone output."""
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=240)
        attn = CrossLeadAttention(feature_dim=240, num_leads=12)

        x = torch.randn(2, 12, 5000)
        features = backbone(x)
        enriched = attn(features)
        assert enriched.shape == (2, 240)


# ---------------------------------------------------------------------------
# TensorFlow attention tests (skipped if TF not installed)
# ---------------------------------------------------------------------------


class TestCrossLeadAttentionTF:
    @pytest.fixture(autouse=True)
    def _skip_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    def test_output_shapes(self) -> None:
        import numpy as np

        from aortica.models.attention_tf import build_cross_lead_attention_tf

        model = build_cross_lead_attention_tf(
            feature_dim=256, num_leads=12, num_heads=4, head_dim=64,
        )
        x = np.random.randn(4, 256).astype(np.float32)
        enriched, attn_weights = model(x, training=False)
        assert enriched.shape == (4, 256)
        assert attn_weights.shape == (4, 4, 12, 12)

    def test_attention_weights_sum_to_one(self) -> None:
        import numpy as np

        from aortica.models.attention_tf import build_cross_lead_attention_tf

        model = build_cross_lead_attention_tf(feature_dim=256, num_leads=12)
        x = np.random.randn(2, 256).astype(np.float32)
        _, attn_weights = model(x, training=False)
        row_sums = np.sum(attn_weights.numpy(), axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_small_config(self) -> None:
        import numpy as np

        from aortica.models.attention_tf import build_cross_lead_attention_tf

        model = build_cross_lead_attention_tf(
            feature_dim=24, num_leads=6, num_heads=2, head_dim=16,
        )
        x = np.random.randn(4, 24).astype(np.float32)
        enriched, attn_weights = model(x, training=False)
        assert enriched.shape == (4, 24)
        assert attn_weights.shape == (4, 2, 6, 6)

    def test_invalid_feature_dim(self) -> None:
        from aortica.models.attention_tf import build_cross_lead_attention_tf

        with pytest.raises(ValueError, match="divisible"):
            build_cross_lead_attention_tf(feature_dim=100, num_leads=12)

    def test_model_summary(self) -> None:
        from aortica.models.attention_tf import build_cross_lead_attention_tf

        model = build_cross_lead_attention_tf(
            feature_dim=256, num_leads=12, num_heads=4, head_dim=64,
        )
        # Should not raise
        model.summary()
