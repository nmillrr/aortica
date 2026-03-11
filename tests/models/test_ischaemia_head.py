"""Tests for Ischaemia & Metabolic Task Head (US-019).

Covers both PyTorch and TensorFlow/Keras implementations:
- Constants and class list
- Output shapes and value ranges
- Loss computation (standard BCE with class weighting)
- Gradient flow
- Backbone + attention integration
- TF/Keras equivalent tests
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aortica.models.ischaemia_head import (  # noqa: E402
    ISCHAEMIA_CLASSES,
    NUM_ISCHAEMIA_CLASSES,
    IschaemiaHead,
    compute_ischaemia_loss,
)

# ---------------------------------------------------------------------------
# Constants and class list
# ---------------------------------------------------------------------------


class TestIschaemiaConstants:
    """Verify canonical class list and count."""

    def test_num_classes(self) -> None:
        assert NUM_ISCHAEMIA_CLASSES == 10

    def test_class_list_length(self) -> None:
        assert len(ISCHAEMIA_CLASSES) == 10

    def test_class_list_unique(self) -> None:
        assert len(set(ISCHAEMIA_CLASSES)) == 10

    def test_expected_classes_present(self) -> None:
        expected = {
            "STEMI", "posterior_MI", "occlusive_NSTEMI", "old_MI",
            "hyperkalaemia", "hypokalaemia", "QTc_prolongation",
        }
        assert expected.issubset(set(ISCHAEMIA_CLASSES))


# ---------------------------------------------------------------------------
# IschaemiaHead — output shapes and ranges
# ---------------------------------------------------------------------------


class TestIschaemiaHeadShape:
    """Output shape and value range tests."""

    @pytest.fixture()
    def head(self) -> IschaemiaHead:
        return IschaemiaHead(feature_dim=256, hidden_dim=128, dropout=0.0)

    def test_output_shape(self, head: IschaemiaHead) -> None:
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 10)

    def test_output_range(self, head: IschaemiaHead) -> None:
        x = torch.randn(8, 256)
        out = head(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_logits_shape(self, head: IschaemiaHead) -> None:
        x = torch.randn(4, 256)
        logits = head.forward_logits(x)
        assert logits.shape == (4, 10)

    def test_logits_unbounded(self, head: IschaemiaHead) -> None:
        """Logits can be negative or greater than 1."""
        torch.manual_seed(42)
        x = torch.randn(64, 256) * 5
        logits = head.forward_logits(x)
        assert logits.min() < 0.0 or logits.max() > 1.0

    def test_single_sample(self, head: IschaemiaHead) -> None:
        x = torch.randn(1, 256)
        out = head(x)
        assert out.shape == (1, 10)

    def test_custom_feature_dim(self) -> None:
        head = IschaemiaHead(feature_dim=512, hidden_dim=64)
        x = torch.randn(2, 512)
        assert head(x).shape == (2, 10)


# ---------------------------------------------------------------------------
# Loss computation — standard BCE
# ---------------------------------------------------------------------------


class TestIschaemiaLoss:
    """Loss function tests (standard BCE)."""

    def test_loss_scalar(self) -> None:
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, 0] = 1.0  # STEMI positive
        loss = compute_ischaemia_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_loss_with_class_weights(self) -> None:
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        weights = torch.ones(10) * 2.0
        loss_weighted = compute_ischaemia_loss(logits, targets, class_weights=weights)
        loss_unweighted = compute_ischaemia_loss(logits, targets)
        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_weighted, loss_unweighted)

    def test_perfect_prediction_low_loss(self) -> None:
        """Loss should be very low for confident correct predictions."""
        targets = torch.zeros(4, 10)
        logits = torch.full((4, 10), -10.0)
        loss = compute_ischaemia_loss(logits, targets)
        assert loss.item() < 0.01

    def test_loss_gradient_flows(self) -> None:
        head = IschaemiaHead(feature_dim=256, dropout=0.0)
        x = torch.randn(4, 256, requires_grad=True)
        logits = head.forward_logits(x)
        targets = torch.zeros(4, 10)
        loss = compute_ischaemia_loss(logits, targets)
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Integration with backbone + attention
# ---------------------------------------------------------------------------


class TestIschaemiaHeadIntegration:
    """Integration with upstream modules."""

    def test_backbone_to_head(self) -> None:
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        head = IschaemiaHead(feature_dim=256)

        x = torch.randn(2, 12, 2500)  # 5s @ 500 Hz
        features = backbone(x)
        probs = head(features)
        assert probs.shape == (2, 10)

    def test_backbone_attention_to_head(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        attention = CrossLeadAttention(feature_dim=256, num_leads=12)
        head = IschaemiaHead(feature_dim=256)

        x = torch.randn(2, 12, 2500)
        features = backbone(x)
        enriched = attention(features)
        probs = head(enriched)
        assert probs.shape == (2, 10)

    def test_end_to_end_gradient(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        attention = CrossLeadAttention(feature_dim=256, num_leads=12)
        head = IschaemiaHead(feature_dim=256, dropout=0.0)

        x = torch.randn(2, 12, 2500, requires_grad=True)
        features = backbone(x)
        enriched = attention(features)
        logits = head.forward_logits(enriched)

        targets = torch.zeros(2, 10)
        loss = compute_ischaemia_loss(logits, targets)
        loss.backward()

        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# TensorFlow / Keras tests
# ---------------------------------------------------------------------------


class TestIschaemiaHeadTF:
    """TF/Keras ischaemia head tests."""

    @pytest.fixture()
    def _skip_if_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_shape(self) -> None:
        from aortica.models.ischaemia_head_tf import build_ischaemia_head_tf

        model = build_ischaemia_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(4, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (4, 10)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_range(self) -> None:
        from aortica.models.ischaemia_head_tf import build_ischaemia_head_tf

        model = build_ischaemia_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(8, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_custom_dims(self) -> None:
        from aortica.models.ischaemia_head_tf import build_ischaemia_head_tf

        model = build_ischaemia_head_tf(feature_dim=512, hidden_dim=64)
        import numpy as np

        x = np.random.randn(2, 512).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (2, 10)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_model_summary(self) -> None:
        from aortica.models.ischaemia_head_tf import build_ischaemia_head_tf

        model = build_ischaemia_head_tf(feature_dim=256)
        # Should not raise
        model.summary()
