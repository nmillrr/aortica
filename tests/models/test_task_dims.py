"""Single-source-of-truth regression tests for task dimensions (US-129)."""

from __future__ import annotations

import pytest

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES
from aortica.models.task_dims import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    TASK_NUM_OUTPUTS,
    TOTAL_OUTPUTS,
)

# ─────────────────────────────────────────────────────────────────
# The canonical map must equal the actual head class-list lengths.
# Any future head expansion that forgets this fails CI immediately.
# ─────────────────────────────────────────────────────────────────


def test_canonical_map_matches_head_class_lengths() -> None:
    assert TASK_NUM_OUTPUTS == {
        "rhythm": len(RHYTHM_CLASSES),
        "structural": len(STRUCTURAL_CLASSES),
        "ischaemia": len(ISCHAEMIA_CLASSES),
        "risk": len(RISK_OUTPUTS),
    }


def test_total_outputs_is_sum() -> None:
    assert TOTAL_OUTPUTS == sum(TASK_NUM_OUTPUTS.values())


def test_task_ordering_constants() -> None:
    assert ALL_TASKS == ["rhythm", "structural", "ischaemia", "risk"]
    assert CLASSIFICATION_TASKS == ["rhythm", "structural", "ischaemia"]


def test_importable_without_torch() -> None:
    """The canonical map must import without forcing a torch/tf import.

    Runs in a fresh subprocess with ``torch``/``tensorflow`` import blocked so
    the check is not polluted by already-cached modules in this process.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import builtins
        _real = builtins.__import__
        def _blocked(name, *a, **k):
            if name == "torch" or name.startswith("torch.") \
               or name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError(f"blocked: {name}")
            return _real(name, *a, **k)
        builtins.__import__ = _blocked
        from aortica.models.task_dims import TASK_NUM_OUTPUTS
        assert TASK_NUM_OUTPUTS["rhythm"] > 0
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ─────────────────────────────────────────────────────────────────
# Every downstream consumer must share the canonical map (no stale copies).
# ─────────────────────────────────────────────────────────────────


def test_all_consumers_share_canonical_map() -> None:
    import importlib

    consumers = [
        ("aortica.evaluation.benchmark", "TASK_NUM_OUTPUTS"),
        ("aortica.edge.distillation", "_TASK_NUM_OUTPUTS"),
        ("aortica.edge.validation", "TASK_NUM_OUTPUTS"),
        ("aortica.models.train_multitask", "_TASK_NUM_OUTPUTS"),
        ("aortica.models.conformal_prediction", "TASK_NUM_OUTPUTS"),
        ("aortica.federated.data_quality", "_TASK_NUM_OUTPUTS"),
        ("aortica.federated.fl_client", "_TASK_NUM_OUTPUTS"),
    ]
    for module_name, attr in consumers:
        module = importlib.import_module(module_name)
        # Each consumer imports the canonical map, so all values are identical.
        assert getattr(module, attr) == TASK_NUM_OUTPUTS, module_name


# ─────────────────────────────────────────────────────────────────
# End-to-end: exercise label-splitting against a REAL AorticaModel so
# dimension drift is caught (not self-consistent synthetic fixtures).
# ─────────────────────────────────────────────────────────────────


class TestAgainstRealModel:
    def test_model_head_widths_match_canonical(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.models.aortica_model import AorticaModel

        model = AorticaModel()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 12, 5000))
        widths = {k: v.shape[-1] for k, v in out.as_dict().items() if v is not None}
        for task, width in widths.items():
            assert width == TASK_NUM_OUTPUTS[task], (
                f"head {task} emits {width} but canonical map says "
                f"{TASK_NUM_OUTPUTS[task]}"
            )

    def test_split_labels_uses_real_widths(self) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from aortica.models.train_multitask import _split_labels

        total = TOTAL_OUTPUTS
        labels = np.arange(2 * total, dtype=np.float32).reshape(2, total)
        splits = _split_labels(labels, ALL_TASKS)
        for task in ALL_TASKS:
            assert splits[task].shape[1] == TASK_NUM_OUTPUTS[task]
        # Concatenated widths reconstruct the full 72-wide label vector.
        assert sum(splits[t].shape[1] for t in ALL_TASKS) == total

    def test_fl_client_split_matches_model(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import _split_labels as fl_split
        from aortica.models.aortica_model import AorticaModel

        model = AorticaModel()
        with torch.no_grad():
            out = model(torch.randn(1, 12, 5000))
        model_widths = {
            k: v.shape[-1] for k, v in out.as_dict().items() if v is not None
        }
        labels = torch.zeros(1, TOTAL_OUTPUTS)
        splits = fl_split(labels, ALL_TASKS)
        for task, width in model_widths.items():
            assert splits[task].shape[1] == width
