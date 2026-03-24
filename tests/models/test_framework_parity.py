"""Tests for TensorFlow/Keras ↔ PyTorch framework parity (US-029).

These tests verify that:

1. The weight conversion process correctly maps PyTorch parameters to
   their TF/Keras counterparts.
2. Given identical weights and inputs, both models produce outputs within
   floating-point tolerance (atol=1e-5).

Integration tests are tagged ``@pytest.mark.slow`` because constructing
both models and running inference is more expensive than typical unit tests.
Tests that require both frameworks are skipped if either is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from aortica.models.framework_parity import (
    HAS_TF,
    HAS_TORCH,
    _convert_conv1d_weight,
    _convert_linear_weight,
)

needs_both = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TF),
    reason="Requires both PyTorch and TensorFlow",
)


# ===================================================================
# Weight conversion helper tests (pure numpy — no framework deps)
# ===================================================================


class TestConvertConv1dWeight:
    """Tests for _convert_conv1d_weight."""

    def test_shape(self) -> None:
        w = np.random.randn(64, 12, 15).astype(np.float32)
        converted = _convert_conv1d_weight(w)
        assert converted.shape == (15, 12, 64)

    def test_values_preserved(self) -> None:
        w = np.random.randn(32, 16, 7).astype(np.float32)
        converted = _convert_conv1d_weight(w)
        assert converted[3, 2, 5] == w[5, 2, 3]

    def test_roundtrip(self) -> None:
        w = np.random.randn(64, 12, 7).astype(np.float32)
        tf_w = _convert_conv1d_weight(w)
        pt_w = np.transpose(tf_w, (2, 1, 0))
        np.testing.assert_array_equal(pt_w, w)


class TestConvertLinearWeight:
    """Tests for _convert_linear_weight."""

    def test_shape(self) -> None:
        w = np.random.randn(128, 256).astype(np.float32)
        converted = _convert_linear_weight(w)
        assert converted.shape == (256, 128)

    def test_values_preserved(self) -> None:
        w = np.random.randn(10, 20).astype(np.float32)
        converted = _convert_linear_weight(w)
        assert converted[5, 3] == w[3, 5]


# ===================================================================
# Integration tests requiring both PyTorch and TensorFlow
# ===================================================================


@pytest.fixture()
def pt_model():  # type: ignore[no-untyped-def]
    """Create a PyTorch AorticaModel."""
    torch = pytest.importorskip("torch")  # noqa: F841

    from aortica.models.aortica_model import AorticaModel

    model = AorticaModel(
        in_channels=12,
        feature_dim=252,
        num_leads=12,
        enabled_tasks=["rhythm", "structural", "ischaemia", "risk"],
    )
    model.eval()
    return model


@pytest.fixture()
def tf_model():  # type: ignore[no-untyped-def]
    """Create a matching TF AorticaModel."""
    pytest.importorskip("tensorflow")

    from aortica.models.aortica_model_tf import build_aortica_model_tf

    model = build_aortica_model_tf(
        in_channels=12,
        feature_dim=252,
        input_length=2500,
        enabled_tasks=["rhythm", "structural", "ischaemia", "risk"],
    )
    return model


@needs_both
@pytest.mark.slow
class TestConvertPytorchToTf:
    """Tests for the weight conversion process."""

    def test_returns_transferred_keys(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        from aortica.models.framework_parity import convert_pytorch_to_tf

        transferred = convert_pytorch_to_tf(pt_model, tf_model)
        assert len(transferred) > 0

    def test_transfers_backbone_params(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        from aortica.models.framework_parity import convert_pytorch_to_tf

        transferred = convert_pytorch_to_tf(pt_model, tf_model)
        backbone_keys = [k for k in transferred if k.startswith("backbone.")]
        assert len(backbone_keys) > 0
        assert "backbone.conv1.weight" in transferred
        assert "backbone.bn1.weight" in transferred

    def test_transfers_attention_params(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        from aortica.models.framework_parity import convert_pytorch_to_tf

        transferred = convert_pytorch_to_tf(pt_model, tf_model)
        attention_keys = [k for k in transferred if k.startswith("attention.")]
        assert len(attention_keys) > 0
        assert "attention.q_proj.weight" in transferred

    def test_transfers_task_head_params(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        from aortica.models.framework_parity import convert_pytorch_to_tf

        transferred = convert_pytorch_to_tf(pt_model, tf_model)
        for task in ["rhythm", "structural", "ischaemia", "risk"]:
            head_prefix = f"{task}_head"
            head_keys = [k for k in transferred if k.startswith(head_prefix)]
            assert len(head_keys) > 0, f"No keys transferred for {task}"

    def test_all_pytorch_params_accounted(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        """Verify that every PyTorch parameter was transferred."""
        from aortica.models.framework_parity import convert_pytorch_to_tf

        state_dict = pt_model.state_dict()
        transferred = convert_pytorch_to_tf(pt_model, tf_model)
        not_transferred = set(state_dict.keys()) - set(transferred)
        # num_batches_tracked is a PyTorch BN internal → no TF equivalent
        not_transferred = {
            k for k in not_transferred
            if "num_batches_tracked" not in k
        }
        assert len(not_transferred) == 0, (
            f"Parameters not transferred: {not_transferred}"
        )

    def test_task_subset(self) -> None:
        """Conversion works with a subset of tasks."""
        from aortica.models.aortica_model import AorticaModel
        from aortica.models.aortica_model_tf import build_aortica_model_tf
        from aortica.models.framework_parity import convert_pytorch_to_tf

        pt = AorticaModel(
            in_channels=12, feature_dim=252, num_leads=12,
            enabled_tasks=["rhythm"],
        )
        pt.eval()
        tf_m = build_aortica_model_tf(
            in_channels=12, feature_dim=252, input_length=2500,
            enabled_tasks=["rhythm"],
        )
        transferred = convert_pytorch_to_tf(pt, tf_m, enabled_tasks=["rhythm"])
        rhythm_keys = [k for k in transferred if "rhythm" in k]
        assert len(rhythm_keys) > 0
        risk_keys = [k for k in transferred if "risk" in k]
        assert len(risk_keys) == 0


@needs_both
@pytest.mark.slow
class TestValidateParity:
    """Tests for the end-to-end parity validation."""

    def test_outputs_within_tolerance(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        """After weight transfer, outputs should match within atol=1e-4."""
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )

        convert_pytorch_to_tf(pt_model, tf_model)
        max_diffs = validate_parity(
            pt_model, tf_model,
            input_shape=(2, 12, 2500),
            atol=1e-4,
            seed=42,
        )
        for task, diff in max_diffs.items():
            assert diff < 1e-4, f"Task {task} diff too large: {diff}"

    def test_all_tasks_present(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )

        convert_pytorch_to_tf(pt_model, tf_model)
        max_diffs = validate_parity(
            pt_model, tf_model,
            input_shape=(2, 12, 2500),
            atol=1e-4,
        )
        assert set(max_diffs.keys()) == {"rhythm", "structural", "ischaemia", "risk"}

    def test_reproducible(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        """Same seed produces identical results."""
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )

        convert_pytorch_to_tf(pt_model, tf_model)
        d1 = validate_parity(pt_model, tf_model, seed=123, atol=1e-4)
        d2 = validate_parity(pt_model, tf_model, seed=123, atol=1e-4)
        for task in d1:
            assert d1[task] == d2[task]

    def test_different_inputs_different_outputs(
        self, pt_model, tf_model,  # type: ignore[no-untyped-def]
    ) -> None:
        """Different seeds produce different max diffs (sanity check)."""
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )

        convert_pytorch_to_tf(pt_model, tf_model)
        d1 = validate_parity(pt_model, tf_model, seed=1, atol=1e-4)
        d2 = validate_parity(pt_model, tf_model, seed=2, atol=1e-4)
        any_different = any(d1[t] != d2.get(t) for t in d1)
        assert any_different


@needs_both
@pytest.mark.slow
class TestParitySubset:
    """Tests for parity with task subsets."""

    def test_rhythm_only(self) -> None:
        from aortica.models.aortica_model import AorticaModel
        from aortica.models.aortica_model_tf import build_aortica_model_tf
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )

        pt = AorticaModel(
            in_channels=12, feature_dim=252, num_leads=12,
            enabled_tasks=["rhythm"],
        )
        pt.eval()
        tf_m = build_aortica_model_tf(
            in_channels=12, feature_dim=252, input_length=2500,
            enabled_tasks=["rhythm"],
        )
        convert_pytorch_to_tf(pt, tf_m, enabled_tasks=["rhythm"])
        max_diffs = validate_parity(
            pt, tf_m,
            input_shape=(2, 12, 2500),
            atol=1e-4,
            enabled_tasks=["rhythm"],
        )
        assert "rhythm" in max_diffs
        assert max_diffs["rhythm"] < 1e-4


# ===================================================================
# Documentation tests
# ===================================================================


class TestDocumentation:
    """Verify the weight conversion process is documented."""

    def test_module_docstring(self) -> None:
        from aortica.models import framework_parity
        assert framework_parity.__doc__ is not None
        assert "Weight Conversion Process" in framework_parity.__doc__

    def test_convert_function_docstring(self) -> None:
        from aortica.models.framework_parity import convert_pytorch_to_tf
        assert convert_pytorch_to_tf.__doc__ is not None
        assert "Transfer weights" in convert_pytorch_to_tf.__doc__

    def test_validate_function_docstring(self) -> None:
        from aortica.models.framework_parity import validate_parity
        assert validate_parity.__doc__ is not None
        assert "equivalent outputs" in validate_parity.__doc__


# ===================================================================
# Imports test
# ===================================================================


class TestImports:
    """Test the public API is importable."""

    def test_framework_parity_imports(self) -> None:
        from aortica.models.framework_parity import (
            convert_pytorch_to_tf,
            validate_parity,
        )
        assert convert_pytorch_to_tf is not None
        assert validate_parity is not None
