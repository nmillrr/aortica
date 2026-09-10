"""``import aortica.data`` must work on a checkout with no corpus present.

``aortica/data/__init__.py`` imported the PTB-XL loader at module level, and
that loader had been silently excluded from the repository by a .gitignore
rule.  The result was an ``ImportError`` on a fresh clone that took out
``aortica train``, ``aortica benchmark``, ``aortica build-index`` and the
CLI entry point along with it.

The package now imports cleanly whether or not a corpus loader is present;
the failure is deferred to the point of actual use and carries an
explanation.
"""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    """The subpackage imports without a dataset on disk."""
    import aortica.data

    assert aortica.data is not None


def test_public_names_are_exported() -> None:
    import aortica.data as data

    for name in data.__all__:
        assert hasattr(data, name), f"{name} is in __all__ but not defined"


def test_load_ptbxl_missing_corpus_is_explicit() -> None:
    """The loader is restored; a missing *corpus* still explains itself.

    This test used to assert that ``load_ptbxl`` raised
    ``DatasetLoaderUnavailableError`` because the loader itself was gone.
    It now asserts the remaining failure mode: the loader exists, and the
    error names the path and tells the caller where to download PTB-XL.
    """
    from aortica.data import PTBXLDataNotFoundError, load_ptbxl

    with pytest.raises(PTBXLDataNotFoundError) as excinfo:
        load_ptbxl("/nonexistent/ptbxl")

    message = str(excinfo.value)
    assert "/nonexistent/ptbxl" in message
    assert "physionet.org" in message


def test_loader_error_is_a_notimplementederror() -> None:
    """Existing ``except NotImplementedError`` handlers keep working.

    ``DatasetLoaderUnavailableError`` outlived the PTB-XL gap: it remains
    the shared signal for a corpus with no loader behind it.
    """
    from aortica.data import DatasetLoaderUnavailableError

    assert issubclass(DatasetLoaderUnavailableError, NotImplementedError)


class TestPerTaskWeights:
    """The label mask survived the loss of the loaders and must stay exact."""

    def test_shapes_match_the_head_class_lists(self) -> None:
        from aortica.data import per_task_weights
        from aortica.models.task_dims import TASK_NUM_OUTPUTS

        weights = per_task_weights()
        assert set(weights) == set(TASK_NUM_OUTPUTS)
        for task, count in TASK_NUM_OUTPUTS.items():
            assert len(weights[task]) == count

    def test_matches_the_released_sidecar(self) -> None:
        """Reproduces the mask the v0.3.0 training run actually wrote."""
        import json
        from pathlib import Path

        from aortica.data import per_task_weights

        sidecar = (
            Path(__file__).resolve().parents[2]
            / "artifacts_combined"
            / "trained_outputs.json"
        )
        if not sidecar.exists():
            pytest.skip("release artifacts not present")

        expected = json.loads(sidecar.read_text())["per_task_weights"]
        assert per_task_weights() == expected

    def test_risk_head_is_fully_masked(self) -> None:
        """No outcome labels exist in either cohort, so risk gets no gradient."""
        from aortica.data import per_task_weights

        assert not any(per_task_weights()["risk"])

    def test_accepts_a_wider_corpus(self) -> None:
        """A dataset that labels more outputs widens the mask."""
        from aortica.data import per_task_weights

        widened = per_task_weights(labelable={"AF", "mortality_1y"})
        assert any(widened["risk"]), "mortality_1y should now carry weight"
        assert sum(widened["rhythm"]) == 1.0
