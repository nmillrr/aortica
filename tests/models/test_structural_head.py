"""Tests for Structural & Functional Task Head (US-018).

Covers both PyTorch and TensorFlow/Keras implementations:
- Constants and class list
- Output shapes and value ranges
- Loss computation (standard BCE and focal loss)
- Gradient flow
- Backbone + attention integration
- TF/Keras equivalent tests
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aortica.models.structural_head import (  # noqa: E402
    NUM_STRUCTURAL_CLASSES,
    STRUCTURAL_CLASSES,
    StructuralHead,
    compute_structural_loss,
)

# ---------------------------------------------------------------------------
# Constants and class list
# ---------------------------------------------------------------------------


class TestStructuralConstants:
    """Verify canonical class list and count."""

    def test_num_classes(self) -> None:
        assert NUM_STRUCTURAL_CLASSES == 15

    def test_class_list_length(self) -> None:
        assert len(STRUCTURAL_CLASSES) == 15

    def test_class_list_unique(self) -> None:
        assert len(set(STRUCTURAL_CLASSES)) == 15

    def test_expected_classes_present(self) -> None:
        expected = {"LVH", "RVH", "LVSD", "HCM", "DCM", "ARVC",
                    "amyloidosis", "pericarditis", "myocarditis"}
        assert expected.issubset(set(STRUCTURAL_CLASSES))


# ---------------------------------------------------------------------------
# StructuralHead — output shapes and ranges
# ---------------------------------------------------------------------------


class TestStructuralHeadShape:
    """Output shape and value range tests."""

    @pytest.fixture()
    def head(self) -> StructuralHead:
        return StructuralHead(feature_dim=256, hidden_dim=128, dropout=0.0)

    def test_output_shape(self, head: StructuralHead) -> None:
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 15)

    def test_output_range(self, head: StructuralHead) -> None:
        x = torch.randn(8, 256)
        out = head(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_logits_shape(self, head: StructuralHead) -> None:
        x = torch.randn(4, 256)
        logits = head.forward_logits(x)
        assert logits.shape == (4, 15)

    def test_logits_unbounded(self, head: StructuralHead) -> None:
        """Logits can be negative or greater than 1."""
        torch.manual_seed(42)
        x = torch.randn(64, 256) * 5
        logits = head.forward_logits(x)
        assert logits.min() < 0.0 or logits.max() > 1.0

    def test_single_sample(self, head: StructuralHead) -> None:
        x = torch.randn(1, 256)
        out = head(x)
        assert out.shape == (1, 15)

    def test_custom_feature_dim(self) -> None:
        head = StructuralHead(feature_dim=512, hidden_dim=64)
        x = torch.randn(2, 512)
        assert head(x).shape == (2, 15)


# ---------------------------------------------------------------------------
# Loss computation — standard BCE
# ---------------------------------------------------------------------------


class TestStructuralLoss:
    """Loss function tests (standard BCE)."""

    def test_loss_scalar(self) -> None:
        logits = torch.randn(4, 15)
        targets = torch.zeros(4, 15)
        targets[:, 0] = 1.0  # LVH positive
        loss = compute_structural_loss(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_loss_with_class_weights(self) -> None:
        logits = torch.randn(4, 15)
        targets = torch.zeros(4, 15)
        weights = torch.ones(15) * 2.0
        loss_weighted = compute_structural_loss(logits, targets, class_weights=weights)
        loss_unweighted = compute_structural_loss(logits, targets)
        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_weighted, loss_unweighted)

    def test_perfect_prediction_low_loss(self) -> None:
        """Loss should be very low for confident correct predictions."""
        targets = torch.zeros(4, 15)
        logits = torch.full((4, 15), -10.0)
        loss = compute_structural_loss(logits, targets)
        assert loss.item() < 0.01

    def test_loss_gradient_flows(self) -> None:
        head = StructuralHead(feature_dim=256, dropout=0.0)
        x = torch.randn(4, 256, requires_grad=True)
        logits = head.forward_logits(x)
        targets = torch.zeros(4, 15)
        loss = compute_structural_loss(logits, targets)
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Loss computation — focal loss
# ---------------------------------------------------------------------------


class TestStructuralFocalLoss:
    """Focal loss tests for rare class handling."""

    def test_focal_loss_scalar(self) -> None:
        logits = torch.randn(4, 15)
        targets = torch.zeros(4, 15)
        targets[:, 0] = 1.0
        loss = compute_structural_loss(logits, targets, focal=True)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_focal_loss_less_than_bce_for_easy(self) -> None:
        """Focal loss should be smaller than BCE for easy (confident) examples."""
        targets = torch.zeros(4, 15)
        # Strongly correct predictions
        logits = torch.full((4, 15), -5.0)
        bce = compute_structural_loss(logits, targets, focal=False)
        focal = compute_structural_loss(logits, targets, focal=True, focal_gamma=2.0)
        # Focal should down-weight these easy examples
        assert focal.item() < bce.item()

    def test_focal_loss_with_alpha(self) -> None:
        logits = torch.randn(4, 15)
        targets = torch.zeros(4, 15)
        targets[:, 0] = 1.0
        alpha = torch.full((15,), 0.25)
        loss = compute_structural_loss(
            logits, targets, focal=True, focal_alpha=alpha,
        )
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_focal_loss_gamma_zero_equals_bce(self) -> None:
        """With gamma=0, focal loss should equal BCE."""
        torch.manual_seed(99)
        logits = torch.randn(4, 15)
        targets = torch.zeros(4, 15)
        targets[:, 0] = 1.0
        bce = compute_structural_loss(logits, targets, focal=False)
        focal_g0 = compute_structural_loss(
            logits, targets, focal=True, focal_gamma=0.0,
        )
        assert torch.allclose(bce, focal_g0, atol=1e-5)

    def test_focal_gradient_flows(self) -> None:
        head = StructuralHead(feature_dim=256, dropout=0.0)
        x = torch.randn(4, 256, requires_grad=True)
        logits = head.forward_logits(x)
        targets = torch.zeros(4, 15)
        loss = compute_structural_loss(logits, targets, focal=True)
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Integration with backbone + attention
# ---------------------------------------------------------------------------


class TestStructuralHeadIntegration:
    """Integration with upstream modules."""

    def test_backbone_to_head(self) -> None:
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        head = StructuralHead(feature_dim=256)

        x = torch.randn(2, 12, 2500)  # 5s @ 500 Hz
        features = backbone(x)
        probs = head(features)
        assert probs.shape == (2, 15)

    def test_backbone_attention_to_head(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=240)
        attention = CrossLeadAttention(feature_dim=240, num_leads=12)
        head = StructuralHead(feature_dim=240)

        x = torch.randn(2, 12, 2500)
        features = backbone(x)
        enriched = attention(features)
        probs = head(enriched)
        assert probs.shape == (2, 15)

    def test_end_to_end_gradient(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=240)
        attention = CrossLeadAttention(feature_dim=240, num_leads=12)
        head = StructuralHead(feature_dim=240, dropout=0.0)

        x = torch.randn(2, 12, 2500, requires_grad=True)
        features = backbone(x)
        enriched = attention(features)
        logits = head.forward_logits(enriched)

        targets = torch.zeros(2, 15)
        loss = compute_structural_loss(logits, targets)
        loss.backward()

        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# TensorFlow / Keras tests
# ---------------------------------------------------------------------------


class TestStructuralHeadTF:
    """TF/Keras structural head tests."""

    @pytest.fixture()
    def _skip_if_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_shape(self) -> None:
        from aortica.models.structural_head_tf import build_structural_head_tf

        model = build_structural_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(4, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (4, 15)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_range(self) -> None:
        from aortica.models.structural_head_tf import build_structural_head_tf

        model = build_structural_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(8, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_custom_dims(self) -> None:
        from aortica.models.structural_head_tf import build_structural_head_tf

        model = build_structural_head_tf(feature_dim=512, hidden_dim=64)
        import numpy as np

        x = np.random.randn(2, 512).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (2, 15)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_model_summary(self) -> None:
        from aortica.models.structural_head_tf import build_structural_head_tf

        model = build_structural_head_tf(feature_dim=256)
        # Should not raise
        model.summary()
