"""Tests for default-on suppression of untrained model outputs.

The model card has always said that 44 of the 72 outputs are untrained and
must be gated before use.  Grepping the API, CLI and frontend for the two
functions that do the gating used to return zero call sites, so the default
``aortica predict`` and ``POST /api/v1/predict`` paths could serve an
untrained ``mortality_1y`` as if it were a prediction.  These tests hold the
gate shut.
"""

from __future__ import annotations

import pytest

from aortica.models.output_gating import (
    activate_simplified_output_gating,
    default_class_thresholds,
    default_trained_outputs,
    filter_task_outputs,
    is_trained,
    kept_indices,
)


class TestPackagedAllowlist:
    """The allowlist ships inside the package and describes v0.3.0."""

    def test_allowlist_is_non_empty(self) -> None:
        assert len(default_trained_outputs()) == 28

    def test_risk_head_is_entirely_untrained(self) -> None:
        """Neither public cohort has outcome data, so no risk output trained."""
        for name in (
            "mortality_1y",
            "hf_hosp_12m",
            "af_onset_12m",
            "ecg_predicted_ef",
            "conduction_disease_trajectory",
            "sudden_cardiac_death_risk",
        ):
            assert not is_trained(name), f"{name} must not be reported"

    def test_known_trained_outputs_pass(self) -> None:
        for name in ("AF", "RBBB", "LVH", "STEMI", "normal_sinus_rhythm"):
            assert is_trained(name)

    def test_thresholds_cover_the_allowlist(self) -> None:
        """Every trained output has a measured operating point."""
        thresholds = default_class_thresholds()
        assert set(thresholds) == set(default_trained_outputs())
        assert all(0.0 < v < 1.0 for v in thresholds.values())


class TestFiltering:
    """Kept names and probabilities stay positionally aligned."""

    def test_untrained_outputs_are_removed(self) -> None:
        names, probs, suppressed = filter_task_outputs(
            ["AF", "mortality_1y", "RBBB", "sudden_cardiac_death_risk"],
            [0.9, 0.8, 0.1, 0.7],
        )
        assert names == ["AF", "RBBB"]
        assert probs == [0.9, 0.1]
        assert suppressed == ["mortality_1y", "sudden_cardiac_death_risk"]

    def test_alignment_is_preserved(self) -> None:
        """A consumer zipping the two lists still gets the right pairs."""
        source = {"AF": 0.11, "mortality_1y": 0.99, "LVH": 0.42}
        names, probs, _ = filter_task_outputs(
            list(source), [source[n] for n in source]
        )
        assert dict(zip(names, probs)) == {"AF": 0.11, "LVH": 0.42}

    def test_kept_indices_match_filtering(self) -> None:
        names = ["AF", "mortality_1y", "RBBB"]
        keep, suppressed = kept_indices(names)
        assert keep == [0, 2]
        assert suppressed == ["mortality_1y"]

    def test_explicit_allowlist_overrides_the_default(self) -> None:
        names, probs, suppressed = filter_task_outputs(
            ["AF", "mortality_1y"], [0.5, 0.5], allowlist={"mortality_1y"}
        )
        assert names == ["mortality_1y"]
        assert suppressed == ["AF"]

    def test_empty_input(self) -> None:
        assert filter_task_outputs([], []) == ([], [], [])


class TestSimplifiedOutputActivation:
    """The CHW tier logic gets both the allowlist and the calibration."""

    def test_activation_sets_both_tables(self) -> None:
        simplified = pytest.importorskip("aortica.edge.simplified_output")

        simplified.set_trained_outputs(None)
        simplified.set_class_thresholds(None)
        assert simplified.get_trained_outputs() is None

        try:
            activate_simplified_output_gating()
            assert simplified.get_trained_outputs() == set(default_trained_outputs())
            assert simplified.get_class_thresholds() == default_class_thresholds()
        finally:
            simplified.set_trained_outputs(None)
            simplified.set_class_thresholds(None)
