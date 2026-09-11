"""Multi-corpus assembly with per-sample label masks.

The thing these tests protect is the distinction between "this corpus cannot
say" and "this finding is absent".  Both are a 0.0 in the label matrix, and
only the mask tells them apart.  Lose the mask and the model learns to rule
out, on every PTB-XL record, the two findings only Chapman can label.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from aortica.data.combined import (
    CorpusSpec,
    combined_labelable_outputs,
    corpus_mask,
    load_corpora,
    ptbxl_chapman,
)
from aortica.data.ptbxl import _output_columns

PTBXL_ROOT_ENV = "AORTICA_PTBXL_ROOT"
CHAPMAN_ROOT_ENV = "AORTICA_CHAPMAN_ROOT"


@pytest.fixture(scope="module")
def specs() -> list[CorpusSpec]:
    ptbxl = os.environ.get(PTBXL_ROOT_ENV)
    chapman = os.environ.get(CHAPMAN_ROOT_ENV)
    if not ptbxl or not chapman:
        pytest.skip(
            f"{PTBXL_ROOT_ENV} and {CHAPMAN_ROOT_ENV} must both be set"
        )
    if not (Path(ptbxl) / "ptbxl_database.csv").exists():
        pytest.skip("PTB-XL root is not a distribution")
    if not (Path(chapman) / "WFDBRecords").is_dir():
        pytest.skip("Chapman root is not a distribution")
    return ptbxl_chapman(ptbxl, chapman)


# ----------------------------------------------------------------------
# Masks, without touching a corpus
# ----------------------------------------------------------------------


class TestCorpusMask:
    def test_ptbxl_mask_covers_its_26_outputs(self) -> None:
        mask = corpus_mask("ptbxl_scp")
        assert mask.shape == (72,)
        assert mask.sum() == 26.0

    def test_chapman_mask_covers_its_21_outputs(self) -> None:
        assert corpus_mask("chapman_snomed").sum() == 21.0

    def test_masks_mark_the_right_columns(self) -> None:
        names, index = _output_columns(multitask=True)
        ptbxl = corpus_mask("ptbxl_scp")
        chapman = corpus_mask("chapman_snomed")

        # PTB-XL cannot label these two; Chapman can. This is the exact
        # asymmetry the per-sample mask exists to express.
        for name in ("AVRT", "early_repol_vs_STEMI"):
            assert ptbxl[index[name]] == 0.0
            assert chapman[index[name]] == 1.0

        # And the reverse, for a finding only PTB-XL can see.
        for name in ("STEMI", "digitalis_effect", "pacemaker_rhythm"):
            assert ptbxl[index[name]] == 1.0
            assert chapman[index[name]] == 0.0

    def test_risk_head_is_masked_out_by_both(self) -> None:
        from aortica.models.risk_head import RISK_OUTPUTS

        _, index = _output_columns(multitask=True)
        for corpus in ("ptbxl_scp", "chapman_snomed"):
            mask = corpus_mask(corpus)
            for name in RISK_OUTPUTS:
                assert mask[index[name]] == 0.0

    def test_single_task_mask_is_the_rhythm_width(self) -> None:
        from aortica.models.rhythm_head import NUM_RHYTHM_CLASSES

        mask = corpus_mask("ptbxl_scp", multitask=False)
        assert mask.shape == (NUM_RHYTHM_CLASSES,)


class TestCombinedLabelableOutputs:
    def test_union_reproduces_the_shipped_mask(self) -> None:
        """The strongest available check on both reconstructions at once.

        ``LABELABLE_OUTPUTS`` was recovered last from the released
        ``trained_outputs.json``. The two corpus mappings were recovered
        independently, from published per-class counts and from prevalence
        ratios. Their union landing exactly on the shipped 28 is three
        separate derivations agreeing.
        """
        from aortica.data.ptbxl_labels import LABELABLE_OUTPUTS

        union = combined_labelable_outputs(
            ptbxl_chapman("/unused", "/unused")
        )
        assert union == frozenset(LABELABLE_OUTPUTS)
        assert len(union) == 28

    def test_a_single_corpus_is_narrower_than_the_union(self) -> None:
        specs = ptbxl_chapman("/unused", "/unused")
        assert len(combined_labelable_outputs(specs[:1])) == 26
        assert len(combined_labelable_outputs(specs[1:])) == 21

    def test_empty_corpus_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one corpus"):
            load_corpora([])


# ----------------------------------------------------------------------
# Corpus-backed
# ----------------------------------------------------------------------


class TestAgainstTheRealCorpora:
    def test_records_labels_and_masks_stay_aligned(self, specs) -> None:
        for records, labels, masks in load_corpora(
            specs, sampling_rate=100, limit=20
        ):
            assert len(records) == labels.shape[0] == masks.shape[0]
            assert labels.shape == masks.shape

    def test_each_record_carries_its_own_corpus_mask(self, specs) -> None:
        records, _, masks = load_corpora(
            specs, sampling_rate=100, limit=15
        )[0]
        expected = {"ptbxl": 26.0, "chapman": 21.0}
        for record, mask in zip(records, masks):
            source = (record.patient_metadata or {})["source_dataset"]
            assert mask.sum() == expected[source]

    def test_both_corpora_are_present_in_every_split(self, specs) -> None:
        for records, _, _ in load_corpora(specs, sampling_rate=100, limit=10):
            sources = {
                (r.patient_metadata or {})["source_dataset"] for r in records
            }
            assert sources == {"ptbxl", "chapman"}

    def test_a_label_is_never_set_outside_its_mask(self, specs) -> None:
        """A positive in a column the corpus cannot label is a mapping bug."""
        for _, labels, masks in load_corpora(
            specs, sampling_rate=100, limit=50
        ):
            assert not np.any((labels > 0) & (masks == 0))

    def test_masks_union_to_the_trained_output_count(self, specs) -> None:
        _, _, masks = load_corpora(specs, sampling_rate=100, limit=40)[0]
        assert int((masks.sum(axis=0) > 0).sum()) == 28

    def test_dataset_yields_signal_label_and_mask(self, specs) -> None:
        """ECGDataset must pass the mask through to the training loop."""
        from aortica.data.dataset import ECGDataset

        records, labels, masks = load_corpora(
            specs, sampling_rate=100, limit=8
        )[0]
        dataset = ECGDataset(
            records, labels, target_hz=100.0, label_masks=masks
        )
        item = dataset[0]
        assert len(item) == 3
        assert item[2].shape[-1] == labels.shape[1]
