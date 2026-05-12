"""Tests for US-021: Unified Multi-Task Model Assembly (AorticaModel)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aortica.models.aortica_model import (  # noqa: E402
    TASK_NAMES,
    AorticaModel,
    MultiTaskOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
LEADS = 12
SAMPLES = 2500  # 5s @ 500 Hz
FEATURE_DIM = 252  # divisible by 12 leads


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Random 12-lead ECG input."""
    return torch.randn(BATCH, LEADS, SAMPLES)


@pytest.fixture
def model() -> AorticaModel:
    """Default AorticaModel with all heads enabled."""
    return AorticaModel(
        in_channels=LEADS,
        feature_dim=FEATURE_DIM,
        num_leads=LEADS,
    )


# ---------------------------------------------------------------------------
# MultiTaskOutput
# ---------------------------------------------------------------------------


class TestMultiTaskOutput:
    """Tests for the MultiTaskOutput dataclass."""

    def test_default_none(self) -> None:
        out = MultiTaskOutput()
        assert out.rhythm is None
        assert out.structural is None
        assert out.ischaemia is None
        assert out.risk is None

    def test_as_dict(self) -> None:
        t = torch.zeros(2, 3)
        out = MultiTaskOutput(rhythm=t)
        d = out.as_dict()
        assert set(d.keys()) == {"rhythm", "structural", "ischaemia", "risk"}
        assert d["rhythm"] is t
        assert d["structural"] is None


# ---------------------------------------------------------------------------
# AorticaModel construction
# ---------------------------------------------------------------------------


class TestAorticaModelConstruction:
    """Tests for model construction and configuration."""

    def test_all_tasks_enabled_by_default(self) -> None:
        model = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        assert model.enabled_tasks == TASK_NAMES

    def test_subset_tasks(self) -> None:
        model = AorticaModel(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            enabled_tasks=["rhythm", "risk"],
        )
        assert model.rhythm_head is not None
        assert model.risk_head is not None
        assert model.structural_head is None
        assert model.ischaemia_head is None

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            AorticaModel(
                in_channels=LEADS,
                feature_dim=FEATURE_DIM,
                enabled_tasks=["rhythm", "bogus"],
            )

    def test_single_task(self) -> None:
        model = AorticaModel(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            enabled_tasks=["structural"],
        )
        assert model.structural_head is not None
        assert model.rhythm_head is None
        assert model.ischaemia_head is None
        assert model.risk_head is None


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for the full forward pass."""

    def test_full_forward_shapes(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        output = model(sample_input)
        assert isinstance(output, MultiTaskOutput)
        assert output.rhythm is not None
        assert output.rhythm.shape == (BATCH, 28)
        assert output.structural is not None
        assert output.structural.shape == (BATCH, 19)
        assert output.ischaemia is not None
        assert output.ischaemia.shape == (BATCH, 19)
        assert output.risk is not None
        assert output.risk.shape == (BATCH, 6)

    def test_output_ranges(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        for tensor in [output.rhythm, output.structural, output.ischaemia, output.risk]:
            assert tensor is not None
            assert (tensor >= 0).all()
            assert (tensor <= 1).all()

    def test_forward_subset_tasks(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        output = model(sample_input, tasks=["rhythm", "risk"])
        assert output.rhythm is not None
        assert output.risk is not None
        assert output.structural is None
        assert output.ischaemia is None

    def test_forward_invalid_task_raises(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        # Create a model with only rhythm enabled
        m = AorticaModel(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            enabled_tasks=["rhythm"],
        )
        with pytest.raises(ValueError, match="not enabled"):
            m(sample_input, tasks=["structural"])

    def test_as_dict_output(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        output = model(sample_input)
        d = output.as_dict()
        assert isinstance(d, dict)
        assert "rhythm" in d
        assert d["rhythm"] is not None
        assert d["rhythm"].shape == (BATCH, 28)

    def test_rhythm_only_model(self, sample_input: torch.Tensor) -> None:
        model = AorticaModel(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            enabled_tasks=["rhythm"],
        )
        output = model(sample_input)
        assert output.rhythm is not None
        assert output.structural is None
        assert output.ischaemia is None
        assert output.risk is None


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze
# ---------------------------------------------------------------------------


class TestBackboneFreezeUnfreeze:
    """Tests for backbone freezing and unfreezing."""

    def test_freeze_backbone(self, model: AorticaModel) -> None:
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfreeze_backbone(self, model: AorticaModel) -> None:
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_heads_unaffected_by_freeze(self, model: AorticaModel) -> None:
        model.freeze_backbone()
        # Task head parameters should still be trainable
        assert model.rhythm_head is not None
        for param in model.rhythm_head.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradient_flows_to_all_heads(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        output = model(sample_input)
        # Create a loss from all heads
        loss = torch.tensor(0.0)
        for tensor in [output.rhythm, output.structural, output.ischaemia, output.risk]:
            assert tensor is not None
            loss = loss + tensor.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_frozen_backbone_no_gradient(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        model.freeze_backbone()
        output = model(sample_input)
        assert output.rhythm is not None
        loss = output.rhythm.sum()
        loss.backward()

        # Backbone params should have no gradient
        for param in model.backbone.parameters():
            assert param.grad is None

        # Head params should have gradient
        assert model.rhythm_head is not None
        for param in model.rhythm_head.parameters():
            assert param.grad is not None

    def test_attention_weights_available(
        self, model: AorticaModel, sample_input: torch.Tensor,
    ) -> None:
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)
        weights = model.attention_weights
        assert weights is not None
        # Shape: [batch, num_heads, num_leads, num_leads]
        assert weights.shape[0] == BATCH
        assert weights.shape[2] == LEADS
        assert weights.shape[3] == LEADS


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behaviour."""

    def test_eval_deterministic(self, sample_input: torch.Tensor) -> None:
        torch.manual_seed(42)
        m1 = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        m1.eval()

        torch.manual_seed(42)
        m2 = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        m2.eval()

        with torch.no_grad():
            out1 = m1(sample_input)
            out2 = m2(sample_input)

        assert out1.rhythm is not None and out2.rhythm is not None
        assert torch.allclose(out1.rhythm, out2.rhythm)


# ---------------------------------------------------------------------------
# Variable input lengths
# ---------------------------------------------------------------------------


class TestVariableInput:
    """Tests for different input signal lengths."""

    @pytest.mark.parametrize("samples", [1250, 2500, 5000])
    def test_variable_length(self, samples: int) -> None:
        model = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        x = torch.randn(2, LEADS, samples)
        output = model(x)
        assert output.rhythm is not None
        assert output.rhythm.shape == (2, 28)


# ---------------------------------------------------------------------------
# TF/Keras model
# ---------------------------------------------------------------------------


class TestAorticaModelTF:
    """Tests for the TensorFlow/Keras multi-task model."""

    @pytest.fixture(autouse=True)
    def _require_tf(self) -> None:
        pytest.importorskip("tensorflow")

    def test_full_model_output_shapes(self) -> None:
        import numpy as np

        from aortica.models.aortica_model_tf import build_aortica_model_tf

        model = build_aortica_model_tf(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            input_length=SAMPLES,
        )
        x = np.random.randn(2, SAMPLES, LEADS).astype(np.float32)
        outputs = model.predict(x, verbose=0)
        assert isinstance(outputs, dict)
        assert outputs["rhythm"].shape == (2, 28)
        assert outputs["structural"].shape == (2, 19)
        assert outputs["ischaemia"].shape == (2, 19)
        assert outputs["risk"].shape == (2, 6)

    def test_subset_tasks(self) -> None:
        import numpy as np

        from aortica.models.aortica_model_tf import build_aortica_model_tf

        model = build_aortica_model_tf(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            input_length=SAMPLES,
            enabled_tasks=["rhythm", "risk"],
        )
        x = np.random.randn(2, SAMPLES, LEADS).astype(np.float32)
        outputs = model.predict(x, verbose=0)
        assert "rhythm" in outputs
        assert "risk" in outputs
        assert "structural" not in outputs
        assert "ischaemia" not in outputs

    def test_model_summary(self) -> None:
        from aortica.models.aortica_model_tf import build_aortica_model_tf

        model = build_aortica_model_tf(
            in_channels=LEADS,
            feature_dim=FEATURE_DIM,
            input_length=SAMPLES,
        )
        # Should not raise
        model.summary()


# ---------------------------------------------------------------------------
# Expanded head dimension verification (US-077)
# ---------------------------------------------------------------------------


class TestExpandedHeadDimensions:
    """Verify that AorticaModel produces correct output shapes for all
    expanded task heads (rhythm=28, structural=19, ischaemia=19, risk=6).
    """

    def test_expanded_output_dimensions_match_constants(self) -> None:
        """Output dims auto-detected from head class constants."""
        from aortica.models.ischaemia_head import NUM_ISCHAEMIA_CLASSES
        from aortica.models.rhythm_head import NUM_RHYTHM_CLASSES
        from aortica.models.risk_head import NUM_RISK_OUTPUTS
        from aortica.models.structural_head import NUM_STRUCTURAL_CLASSES

        model = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        x = torch.randn(2, LEADS, SAMPLES)
        output = model(x)

        assert output.rhythm is not None
        assert output.rhythm.shape[1] == NUM_RHYTHM_CLASSES == 28
        assert output.structural is not None
        assert output.structural.shape[1] == NUM_STRUCTURAL_CLASSES == 19
        assert output.ischaemia is not None
        assert output.ischaemia.shape[1] == NUM_ISCHAEMIA_CLASSES == 19
        assert output.risk is not None
        assert output.risk.shape[1] == NUM_RISK_OUTPUTS == 6

    def test_total_output_count(self) -> None:
        """Combined output count across all heads equals 72."""
        model = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        x = torch.randn(1, LEADS, SAMPLES)
        output = model(x)
        total = 0
        for t in [output.rhythm, output.structural, output.ischaemia, output.risk]:
            assert t is not None
            total += t.shape[1]
        assert total == 72, f"Expected 72 total outputs, got {total}"

    def test_gradient_flow_expanded_heads(self) -> None:
        """Gradient flows correctly through all expanded heads."""
        model = AorticaModel(in_channels=LEADS, feature_dim=FEATURE_DIM)
        x = torch.randn(2, LEADS, SAMPLES)
        output = model(x)

        loss = torch.tensor(0.0)
        for t in [output.rhythm, output.structural, output.ischaemia, output.risk]:
            assert t is not None
            loss = loss + t.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_selective_head_disabling_expanded(self) -> None:
        """Disabling individual heads still works with expanded dims."""
        for task in TASK_NAMES:
            model = AorticaModel(
                in_channels=LEADS,
                feature_dim=FEATURE_DIM,
                enabled_tasks=[task],
            )
            x = torch.randn(1, LEADS, SAMPLES)
            output = model(x)
            active = getattr(output, task)
            assert active is not None
            disabled = [t for t in TASK_NAMES if t != task]
            for d in disabled:
                assert getattr(output, d) is None

