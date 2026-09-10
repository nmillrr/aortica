"""PTB-XL loader and the recovered SCP-code mapping.

The mapping in ``aortica/data/label_maps/ptbxl_scp.yaml`` was reconstructed
by solving for the code sets that reproduce the per-class positive counts
published in ``artifacts_combined/test_metrics.json``.  That reconstruction
is the fragile part of this loader — the file I/O either works or raises,
but a wrong code set silently trains the wrong thing.

:class:`TestMappingMatchesReleasedMetrics` is therefore the test that
matters most here.  It re-checks the reconstruction against the released
metrics on every run and needs no corpus on disk, so it guards the mapping
in CI where the 2 GB of waveforms will never be.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from aortica.data.label_mapping import LabelMapError, load_label_map
from aortica.data.ptbxl import (
    LABEL_MAP_NAME,
    PTBXLDataNotFoundError,
    build_labels,
    load_ptbxl,
    parse_scp_codes,
    present_codes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASED_METRICS = REPO_ROOT / "artifacts_combined" / "test_metrics.json"

#: Point this at an extracted PTB-XL 1.0.3 distribution to run the tests that
#: actually read waveforms.  They skip when it is unset.
PTBXL_ROOT_ENV = "AORTICA_PTBXL_ROOT"


@pytest.fixture(scope="module")
def label_map():
    return load_label_map(LABEL_MAP_NAME)


@pytest.fixture(scope="module")
def ptbxl_root() -> Path:
    raw = os.environ.get(PTBXL_ROOT_ENV)
    if not raw:
        pytest.skip(f"{PTBXL_ROOT_ENV} is not set; corpus tests skipped")
    root = Path(raw)
    if not (root / "ptbxl_database.csv").exists():
        pytest.skip(f"{PTBXL_ROOT_ENV}={root} has no ptbxl_database.csv")
    return root


# ----------------------------------------------------------------------
# The reconstruction guard
# ----------------------------------------------------------------------


class TestMappingMatchesReleasedMetrics:
    """The recovered mapping must keep agreeing with what v0.3.0 measured."""

    @pytest.fixture(scope="class")
    def released(self) -> dict[str, int]:
        if not RELEASED_METRICS.exists():
            pytest.skip("release artifacts not present")
        metrics = json.loads(RELEASED_METRICS.read_text())
        return {
            key.split("/", 1)[1]: value["test_pos"]
            for key, value in metrics.items()
            if key.startswith("ptbxl/")
        }

    def test_covers_exactly_the_released_ptbxl_outputs(
        self, label_map, released
    ) -> None:
        """No output gained or lost since the weights were trained."""
        assert set(label_map.classes) == set(released)

    def test_every_class_records_the_count_it_reproduces(
        self, label_map, released
    ) -> None:
        """Each entry's verified_test_pos matches the published test_pos.

        This is the audit trail. If someone edits a code list without
        re-solving against fold 10, either they update this number and this
        test catches the disagreement with the release, or they don't and
        the corpus-backed test below catches it.
        """
        for name, expected in released.items():
            assert label_map.classes[name].verified_test_pos == expected, (
                f"{name}: mapping claims "
                f"{label_map.classes[name].verified_test_pos} positives on "
                f"fold 10, release measured {expected}"
            )

    def test_is_a_subset_of_the_combined_label_mask(self, label_map) -> None:
        """PTB-XL labels 26 of the 28 outputs the v0.3.0 corpus covered.

        AVRT and early_repol_vs_STEMI came from Chapman-Shaoxing alone. If
        this ever fails, either the mask or this mapping has drifted.
        """
        from aortica.data.ptbxl_labels import LABELABLE_OUTPUTS

        assert set(label_map.classes) < set(LABELABLE_OUTPUTS)
        assert set(LABELABLE_OUTPUTS) - set(label_map.classes) == {
            "AVRT",
            "early_repol_vs_STEMI",
        }


# ----------------------------------------------------------------------
# Mapping file integrity
# ----------------------------------------------------------------------


class TestLabelMapFile:
    def test_class_names_are_real_model_outputs(self, label_map) -> None:
        from aortica.data.ptbxl import _output_columns

        names, _ = _output_columns(multitask=True)
        for name in label_map.classes:
            assert name in names

    def test_tasks_agree_with_the_owning_head(self, label_map) -> None:
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
        from aortica.models.rhythm_head import RHYTHM_CLASSES
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        owner = {
            "rhythm": set(RHYTHM_CLASSES),
            "structural": set(STRUCTURAL_CLASSES),
            "ischaemia": set(ISCHAEMIA_CLASSES),
        }
        for name, mapping in label_map.classes.items():
            assert name in owner[mapping.task]

    def test_no_source_code_feeds_two_outputs(self, label_map) -> None:
        """A code driving two outputs would double-label silently."""
        for code, outputs in label_map.code_to_classes.items():
            assert len(outputs) == 1, f"{code} maps to {outputs}"

    def test_risk_head_stays_unlabelable(self, label_map) -> None:
        """PTB-XL has no outcome linkage; no entry may claim otherwise."""
        assert label_map.classes_for_task("risk") == ()

    def test_splits_are_the_published_stratification(self, label_map) -> None:
        assert label_map.splits["train"] == (1, 2, 3, 4, 5, 6, 7, 8)
        assert label_map.splits["val"] == (9,)
        assert label_map.splits["test"] == (10,)

    def test_judgement_calls_carry_a_rationale(self, label_map) -> None:
        """Any entry pooling several codes must say why.

        Single-code entries are self-explanatory; multi-code ones encode a
        clinical decision that a reviewer has to be able to challenge.
        """
        for name, mapping in label_map.classes.items():
            if len(mapping.codes) > 1:
                assert mapping.note, f"{name} pools {mapping.codes} without a note"

    def test_unknown_output_name_is_rejected(self, tmp_path) -> None:
        from aortica.data import label_mapping

        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1\n"
            "source: {name: Test, release: '1'}\n"
            "splits: {train: [1], val: [2], test: [3]}\n"
            "classes:\n"
            "  not_a_real_output:\n"
            "    task: rhythm\n"
            "    codes: [X]\n",
            encoding="utf-8",
        )
        original = label_mapping.LABEL_MAP_DIR
        label_mapping.LABEL_MAP_DIR = tmp_path
        try:
            label_mapping.load_label_map.cache_clear()
            with pytest.raises(LabelMapError, match="not an output"):
                label_mapping.load_label_map("bad")
        finally:
            label_mapping.LABEL_MAP_DIR = original
            label_mapping.load_label_map.cache_clear()


# ----------------------------------------------------------------------
# scp_codes parsing
# ----------------------------------------------------------------------


class TestParseScpCodes:
    def test_parses_the_dict_literal(self) -> None:
        assert parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}") == {
            "NORM": 100.0,
            "SR": 0.0,
        }

    @pytest.mark.parametrize("raw", ["", "not a dict", "[1, 2]", "{oops"])
    def test_malformed_cells_yield_nothing(self, raw: str) -> None:
        assert parse_scp_codes(raw) == {}

    def test_zero_likelihood_still_counts_as_present(self, label_map) -> None:
        """The single most consequential detail in the whole loader.

        0.0 means "no likelihood assigned", not "absent". It is the most
        common value in the corpus — treating it as absence drops nearly
        half the labels and stops reproducing the published counts.
        """
        codes = parse_scp_codes("{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}")
        assert present_codes(codes, label_map) == {"NORM", "LVOLT", "SR"}


# ----------------------------------------------------------------------
# Label matrix construction
# ----------------------------------------------------------------------


class TestBuildLabels:
    def test_multitask_is_the_full_concatenation(self, label_map) -> None:
        from aortica.models.task_dims import TOTAL_OUTPUTS

        labels = build_labels([{}], multitask=True, label_map=label_map)
        assert labels.shape == (1, TOTAL_OUTPUTS)
        assert labels.dtype == np.float32

    def test_single_task_is_the_rhythm_head_width(self, label_map) -> None:
        from aortica.models.rhythm_head import NUM_RHYTHM_CLASSES

        labels = build_labels([{}], multitask=False, label_map=label_map)
        assert labels.shape == (1, NUM_RHYTHM_CLASSES)

    def test_sets_the_column_a_code_maps_to(self, label_map) -> None:
        from aortica.data.ptbxl import _output_columns

        names, index = _output_columns(multitask=True)
        labels = build_labels(
            [{"AFIB": 100.0}], multitask=True, label_map=label_map
        )
        assert labels[0, index["AF"]] == 1.0
        assert labels[0].sum() == 1.0

    def test_pooled_codes_set_one_shared_column(self, label_map) -> None:
        """CRBBB and IRBBB both mean RBBB — and only RBBB."""
        from aortica.data.ptbxl import _output_columns

        _, index = _output_columns(multitask=True)
        for code in ("CRBBB", "IRBBB"):
            labels = build_labels(
                [{code: 100.0}], multitask=True, label_map=label_map
            )
            assert labels[0, index["RBBB"]] == 1.0
            assert labels[0].sum() == 1.0

    def test_unmapped_codes_label_nothing(self, label_map) -> None:
        """A deliberately excluded code must not leak into any column."""
        labels = build_labels(
            [{"NDT": 100.0, "ISC_": 100.0, "STD_": 100.0}],
            multitask=True,
            label_map=label_map,
        )
        assert labels.sum() == 0.0

    def test_unlabelable_columns_stay_zero(self, label_map) -> None:
        """Every output PTB-XL cannot label is left for the mask to handle."""
        from aortica.data.ptbxl import _output_columns

        names, _ = _output_columns(multitask=True)
        labels = build_labels(
            [{code: 100.0} for code in label_map.code_to_classes],
            multitask=True,
            label_map=label_map,
        )
        for column, name in enumerate(names):
            if name not in label_map.classes:
                assert labels[:, column].sum() == 0.0, f"{name} was labelled"

    def test_row_count_matches_input(self, label_map) -> None:
        labels = build_labels(
            [{"AFIB": 100.0}, {}, {"SR": 0.0}],
            multitask=False,
            label_map=label_map,
        )
        assert labels.shape[0] == 3


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


class TestFailureModes:
    def test_missing_corpus_names_the_file_and_the_download(
        self, tmp_path
    ) -> None:
        with pytest.raises(PTBXLDataNotFoundError) as excinfo:
            load_ptbxl(tmp_path / "nowhere")
        message = str(excinfo.value)
        assert "physionet.org" in message

    def test_incomplete_corpus_names_what_is_missing(self, tmp_path) -> None:
        (tmp_path / "ptbxl_database.csv").write_text("ecg_id\n", encoding="utf-8")
        with pytest.raises(PTBXLDataNotFoundError, match="records100"):
            load_ptbxl(tmp_path, sampling_rate=100)

    def test_error_is_a_filenotfounderror(self) -> None:
        assert issubclass(PTBXLDataNotFoundError, FileNotFoundError)

    @pytest.mark.parametrize("rate", [250, 1000, 0])
    def test_only_the_shipped_record_trees_are_accepted(
        self, tmp_path, rate: int
    ) -> None:
        with pytest.raises(ValueError, match="100 Hz and 500 Hz"):
            load_ptbxl(tmp_path, sampling_rate=rate)


# ----------------------------------------------------------------------
# Corpus-backed (skipped unless AORTICA_PTBXL_ROOT is set)
# ----------------------------------------------------------------------


class TestAgainstTheRealCorpus:
    def test_reproduces_every_published_fold10_count(self, ptbxl_root) -> None:
        """The end-to-end check that the reconstruction is faithful.

        Loads fold 10 through the real loader and counts positives per
        class. Any drift in parsing, presence rules, or the mapping shows
        up here as a disagreement with the shipped metrics.
        """
        if not RELEASED_METRICS.exists():
            pytest.skip("release artifacts not present")

        from aortica.data.ptbxl import _output_columns

        metrics = json.loads(RELEASED_METRICS.read_text())
        expected = {
            key.split("/", 1)[1]: value["test_pos"]
            for key, value in metrics.items()
            if key.startswith("ptbxl/")
        }

        _, _, (records, labels) = load_ptbxl(
            ptbxl_root,
            sampling_rate=100,
            multitask=True,
            folds={"train": [], "val": [], "test": [10]},
        )
        assert len(records) == 2198

        names, index = _output_columns(multitask=True)
        mismatched = {
            name: (int(labels[:, index[name]].sum()), count)
            for name, count in expected.items()
            if int(labels[:, index[name]].sum()) != count
        }
        assert not mismatched, f"got vs expected: {mismatched}"

    def test_splits_are_disjoint_by_patient(self, ptbxl_root) -> None:
        """Folds must not leak a patient across the evaluation boundary."""
        splits = load_ptbxl(ptbxl_root, sampling_rate=100, limit=400)
        patients = [
            {
                r.patient_metadata.get("ptbxl_patient_id")
                for r in records
                if r.patient_metadata
            }
            for records, _ in splits
        ]
        assert not patients[0] & patients[1]
        assert not patients[0] & patients[2]
        assert not patients[1] & patients[2]

    def test_records_are_twelve_lead(self, ptbxl_root) -> None:
        (records, _), _, _ = load_ptbxl(ptbxl_root, sampling_rate=100, limit=5)
        for record in records:
            assert record.num_leads == 12
            assert record.sample_rate == 100.0
            assert record.units == "µV"
            assert record.source_format == "ptbxl"

    def test_demographics_survive_loading(self, ptbxl_root) -> None:
        """Age and sex are what the equity gate stratifies on."""
        (records, _), _, _ = load_ptbxl(ptbxl_root, sampling_rate=100, limit=50)
        with_sex = [
            r for r in records
            if (r.patient_metadata or {}).get("sex") in {"male", "female"}
        ]
        assert with_sex, "no record carried a decoded sex"
        assert any((r.patient_metadata or {}).get("age") for r in records)

    def test_labels_align_row_for_row_with_records(self, ptbxl_root) -> None:
        for records, labels in load_ptbxl(
            ptbxl_root, sampling_rate=100, limit=40
        ):
            assert len(records) == labels.shape[0]
