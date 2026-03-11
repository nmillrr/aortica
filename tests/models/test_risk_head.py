"""Tests for Risk Prediction Task Head (US-020).

Covers both PyTorch and TensorFlow/Keras implementations:
- Constants and output list
- Output shapes and value ranges
- Loss computation (MSE + ranking)
- Gradient flow
- Backbone + attention integration
- TF/Keras equivalent tests
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aortica.models.risk_head import (  # noqa: E402
    NUM_RISK_OUTPUTS,
    RISK_OUTPUTS,
    RiskHead,
    _pairwise_ranking_loss,
    compute_risk_loss,
)

# ---------------------------------------------------------------------------
# Constants and output list
# ---------------------------------------------------------------------------


class TestRiskConstants:
    """Verify canonical output list and count."""

    def test_num_outputs(self) -> None:
        assert NUM_RISK_OUTPUTS == 3

    def test_output_list_length(self) -> None:
        assert len(RISK_OUTPUTS) == 3

    def test_output_list_unique(self) -> None:
        assert len(set(RISK_OUTPUTS)) == 3

    def test_expected_outputs_present(self) -> None:
        expected = {"mortality_1y", "hf_hosp_12m", "af_onset_12m"}
        assert expected == set(RISK_OUTPUTS)


# ---------------------------------------------------------------------------
# RiskHead — output shapes and ranges
# ---------------------------------------------------------------------------


class TestRiskHeadShape:
    """Output shape and value range tests."""

    @pytest.fixture()
    def head(self) -> RiskHead:
        return RiskHead(feature_dim=256, hidden_dim=128, dropout=0.0)

    def test_output_shape(self, head: RiskHead) -> None:
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 3)

    def test_output_range(self, head: RiskHead) -> None:
        x = torch.randn(8, 256)
        out = head(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_logits_shape(self, head: RiskHead) -> None:
        x = torch.randn(4, 256)
        logits = head.forward_logits(x)
        assert logits.shape == (4, 3)

    def test_logits_unbounded(self, head: RiskHead) -> None:
        """Logits can be negative or greater than 1."""
        torch.manual_seed(42)
        x = torch.randn(64, 256) * 5
        logits = head.forward_logits(x)
        assert logits.min() < 0.0 or logits.max() > 1.0

    def test_single_sample(self, head: RiskHead) -> None:
        x = torch.randn(1, 256)
        out = head(x)
        assert out.shape == (1, 3)

    def test_custom_feature_dim(self) -> None:
        head = RiskHead(feature_dim=512, hidden_dim=64)
        x = torch.randn(2, 512)
        assert head(x).shape == (2, 3)


# ---------------------------------------------------------------------------
# Loss computation — MSE + ranking
# ---------------------------------------------------------------------------


class TestRiskLoss:
    """Loss function tests (combined MSE + ranking)."""

    def test_loss_scalar(self) -> None:
        preds = torch.sigmoid(torch.randn(4, 3))
        targets = torch.rand(4, 3)
        loss = compute_risk_loss(preds, targets)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_perfect_prediction_low_loss(self) -> None:
        """Loss should be very low for perfect predictions."""
        targets = torch.rand(4, 3)
        loss = compute_risk_loss(targets, targets)
        assert loss.item() < 0.01

    def test_loss_with_task_weights(self) -> None:
        preds = torch.sigmoid(torch.randn(8, 3))
        targets = torch.rand(8, 3)
        weights = torch.tensor([2.0, 1.0, 0.5])
        loss_weighted = compute_risk_loss(preds, targets, task_weights=weights)
        loss_unweighted = compute_risk_loss(preds, targets)
        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_weighted, loss_unweighted)

    def test_ranking_weight_effect(self) -> None:
        """Higher ranking weight should change loss value."""
        preds = torch.sigmoid(torch.randn(8, 3))
        targets = torch.rand(8, 3)
        loss_low = compute_risk_loss(preds, targets, ranking_weight=0.0)
        loss_high = compute_risk_loss(preds, targets, ranking_weight=1.0)
        # With ranking_weight=0 it's pure MSE; with 1.0 there's an additive term
        assert not torch.allclose(loss_low, loss_high)

    def test_loss_gradient_flows(self) -> None:
        head = RiskHead(feature_dim=256, dropout=0.0)
        x = torch.randn(4, 256, requires_grad=True)
        scores = head(x)
        targets = torch.rand(4, 3)
        loss = compute_risk_loss(scores, targets)
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# Pairwise ranking loss
# ---------------------------------------------------------------------------


class TestPairwiseRankingLoss:
    """Tests for the concordance-index proxy ranking loss."""

    def test_single_sample_returns_zero(self) -> None:
        preds = torch.tensor([0.5])
        targets = torch.tensor([0.8])
        loss = _pairwise_ranking_loss(preds, targets)
        assert loss.item() == 0.0

    def test_perfectly_ordered(self) -> None:
        """Loss should be low when ordering matches."""
        preds = torch.tensor([0.1, 0.5, 0.9])
        targets = torch.tensor([0.1, 0.5, 0.9])
        loss = _pairwise_ranking_loss(preds, targets)
        assert loss.item() < 0.5

    def test_reversed_order_high_loss(self) -> None:
        """Loss should be higher when ordering is reversed."""
        preds_correct = torch.tensor([0.1, 0.5, 0.9])
        preds_wrong = torch.tensor([0.9, 0.5, 0.1])
        targets = torch.tensor([0.1, 0.5, 0.9])
        loss_correct = _pairwise_ranking_loss(preds_correct, targets)
        loss_wrong = _pairwise_ranking_loss(preds_wrong, targets)
        assert loss_wrong > loss_correct

    def test_equal_targets_returns_zero(self) -> None:
        """When all targets are equal, no valid pairs exist."""
        preds = torch.tensor([0.3, 0.7, 0.5])
        targets = torch.tensor([0.5, 0.5, 0.5])
        loss = _pairwise_ranking_loss(preds, targets)
        assert loss.item() == 0.0

    def test_gradient_flows(self) -> None:
        preds = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
        targets = torch.tensor([0.1, 0.5, 0.9])
        loss = _pairwise_ranking_loss(preds, targets)
        loss.backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# Integration with backbone + attention
# ---------------------------------------------------------------------------


class TestRiskHeadIntegration:
    """Integration with upstream modules."""

    def test_backbone_to_head(self) -> None:
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        head = RiskHead(feature_dim=256)

        x = torch.randn(2, 12, 2500)  # 5s @ 500 Hz
        features = backbone(x)
        scores = head(features)
        assert scores.shape == (2, 3)

    def test_backbone_attention_to_head(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        attention = CrossLeadAttention(feature_dim=256, num_leads=12)
        head = RiskHead(feature_dim=256)

        x = torch.randn(2, 12, 2500)
        features = backbone(x)
        enriched = attention(features)
        scores = head(enriched)
        assert scores.shape == (2, 3)

    def test_end_to_end_gradient(self) -> None:
        from aortica.models.attention import CrossLeadAttention
        from aortica.models.backbone import AorticaBackbone

        backbone = AorticaBackbone(in_channels=12, feature_dim=256)
        attention = CrossLeadAttention(feature_dim=256, num_leads=12)
        head = RiskHead(feature_dim=256, dropout=0.0)

        x = torch.randn(2, 12, 2500, requires_grad=True)
        features = backbone(x)
        enriched = attention(features)
        scores = head(enriched)

        targets = torch.rand(2, 3)
        loss = compute_risk_loss(scores, targets)
        loss.backward()

        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------------------------------------------------------------------------
# TensorFlow / Keras tests
# ---------------------------------------------------------------------------


class TestRiskHeadTF:
    """TF/Keras risk head tests."""

    @pytest.fixture()
    def _skip_if_no_tf(self) -> None:
        pytest.importorskip("tensorflow")

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_shape(self) -> None:
        from aortica.models.risk_head_tf import build_risk_head_tf

        model = build_risk_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(4, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (4, 3)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_output_range(self) -> None:
        from aortica.models.risk_head_tf import build_risk_head_tf

        model = build_risk_head_tf(feature_dim=256)
        import numpy as np

        x = np.random.randn(8, 256).astype(np.float32)
        out = model(x, training=False).numpy()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_custom_dims(self) -> None:
        from aortica.models.risk_head_tf import build_risk_head_tf

        model = build_risk_head_tf(feature_dim=512, hidden_dim=64)
        import numpy as np

        x = np.random.randn(2, 512).astype(np.float32)
        out = model(x, training=False).numpy()
        assert out.shape == (2, 3)

    @pytest.mark.usefixtures("_skip_if_no_tf")
    def test_tf_model_summary(self) -> None:
        from aortica.models.risk_head_tf import build_risk_head_tf

        model = build_risk_head_tf(feature_dim=256)
        # Should not raise
        model.summary()
