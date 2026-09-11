"""Chapman-Shaoxing loader and the recovered SNOMED mapping.

The Chapman mapping rests on weaker evidence than the PTB-XL one and the
tests are shaped accordingly.  PTB-XL publishes folds, so its reconstruction
could be checked against exact per-class counts; Chapman publishes no split,
so its published ``test_pos`` numbers are not reproducible and nothing here
asserts that they are.

What *is* checkable, and what :class:`TestReconstructionEvidence` checks, is
the structural argument: the seven outputs PTB-XL labels but Chapman does not
are exactly the seven whose SNOMED codes fall outside the dataset's shipped
vocabulary or cannot be separated within it.  That argument is what makes the
mapping credible, so it is the thing a regression must not be allowed to
break quietly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from aortica.data.chapman import (
    LABEL_MAP_NAME,
    SPLIT_IS_RECONSTRUCTED,
    ChapmanDataNotFoundError,
    build_labels,
    load_chapman,
    load_condition_names,
    parse_header_metadata,
    split_indices,
)
from aortica.data.label_mapping import load_label_map

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASED_METRICS = REPO_ROOT / "artifacts_combined" / "test_metrics.json"

#: Point this at an extracted Chapman 1.0.0 distribution to run the tests
#: that read the corpus.  They skip when it is unset.
CHAPMAN_ROOT_ENV = "AORTICA_CHAPMAN_ROOT"

#: Outputs PTB-XL labels and Chapman does not, with the reason each is
#: absent.  This is the reconstruction's central evidence, asserted rather
#: than merely written down in the mapping file's comments.
EXPECTED_ABSENCES = {
    "LAFB",              # 445118002 outside the shipped vocabulary
    "LPFB",              # 445211001 outside the shipped vocabulary
    "LA_enlargement",    # 67741000119109 outside the shipped vocabulary
    "pacemaker_rhythm",  # 10370003 outside the shipped vocabulary
    "STEMI",             # no acute-injury concept in the vocabulary
    "posterior_MI",      # all MI territories share one concept id
    "digitalis_effect",  # no digitalis concept in the vocabulary
}


@pytest.fixture(scope="module")
def label_map():
    return load_label_map(LABEL_MAP_NAME)


@pytest.fixture(scope="module")
def ptbxl_map():
    return load_label_map("ptbxl_scp")


@pytest.fixture(scope="module")
def chapman_root() -> Path:
    raw = os.environ.get(CHAPMAN_ROOT_ENV)
    if not raw:
        pytest.skip(f"{CHAPMAN_ROOT_ENV} is not set; corpus tests skipped")
    root = Path(raw)
    if not (root / "WFDBRecords").is_dir():
        pytest.skip(f"{CHAPMAN_ROOT_ENV}={root} has no WFDBRecords/")
    return root


@pytest.fixture(scope="module")
def released() -> dict[str, int]:
    if not RELEASED_METRICS.exists():
        pytest.skip("release artifacts not present")
    metrics = json.loads(RELEASED_METRICS.read_text())
    return {
        key.split("/", 1)[1]: value["test_pos"]
        for key, value in metrics.items()
        if key.startswith("chapman/")
    }


# ----------------------------------------------------------------------
# The reconstruction evidence
# ----------------------------------------------------------------------


class TestReconstructionEvidence:
    def test_covers_exactly_the_released_chapman_outputs(
        self, label_map, released
    ) -> None:
        assert set(label_map.classes) == set(released)

    def test_absences_are_exactly_the_explained_ones(
        self, label_map, ptbxl_map
    ) -> None:
        """The structural argument for the whole reconstruction.

        Every output PTB-XL labels that Chapman does not must be one the
        shipped SNOMED vocabulary genuinely cannot express. A new class
        appearing here with no explanation means the mapping is incomplete;
        one disappearing means the reasoning no longer holds.
        """
        absent = set(ptbxl_map.classes) - set(label_map.classes)
        assert absent == EXPECTED_ABSENCES

    def test_the_two_corpora_together_cover_the_released_mask(
        self, label_map, ptbxl_map
    ) -> None:
        """PTB-XL's 26 plus Chapman's 21 must be exactly the trained 28."""
        from aortica.data.ptbxl_labels import LABELABLE_OUTPUTS

        union = set(ptbxl_map.classes) | set(label_map.classes)
        assert union == set(LABELABLE_OUTPUTS)

    def test_chapman_only_outputs_are_the_two_v030_added(
        self, label_map, ptbxl_map
    ) -> None:
        assert set(label_map.classes) - set(ptbxl_map.classes) == {
            "AVRT",
            "early_repol_vs_STEMI",
        }

    def test_published_counts_are_recorded_faithfully(
        self, label_map, released
    ) -> None:
        """The mapping file must quote the release, not a rounded memory."""
        for name, count in released.items():
            assert label_map.classes[name].raw.get("published_test_pos") == count

    def test_implied_ratios_are_internally_consistent(self, label_map) -> None:
        """published_test_pos / corpus_total must equal the stated ratio."""
        for name, mapping in label_map.classes.items():
            raw = mapping.raw
            expected = raw["published_test_pos"] / raw["corpus_total"]
            assert abs(expected - raw["implied_ratio"]) < 5e-4, name

    def test_ratios_sit_in_a_plausible_band(self, label_map) -> None:
        """A ratio far off ~0.10 would mean the code set is wrong.

        The band is deliberately wide: these are binomial draws, and the
        smallest classes have single-digit counts. It is a smoke alarm for a
        mapping that is wrong by a factor, not a precision instrument.
        """
        for name, mapping in label_map.classes.items():
            ratio = mapping.raw["implied_ratio"]
            assert 0.06 < ratio < 0.16, f"{name} ratio {ratio}"

    def test_the_three_pinning_classes_excluded_the_wider_codes(
        self, label_map
    ) -> None:
        """LVH, RBBB and PVC are what prove the vocabulary-only hypothesis.

        Each has a second, more common SNOMED code in the record headers
        that the shipped vocabulary omits. Including any of them would put
        the ratio an order of magnitude out — and would silently change what
        the corresponding shipped output means.
        """
        assert "55827005" not in label_map.classes["LVH"].codes
        assert "713427006" not in label_map.classes["RBBB"].codes
        assert "427172004" not in label_map.classes["PVC"].codes

    def test_the_split_is_declared_as_reconstructed(self) -> None:
        """Nobody should be able to flip this without a test failing."""
        assert SPLIT_IS_RECONSTRUCTED is True


# ----------------------------------------------------------------------
# Mapping file integrity
# ----------------------------------------------------------------------


class TestLabelMapFile:
    def test_no_published_folds(self, label_map) -> None:
        """Chapman ships no stratification; the policy must say so."""
        assert label_map.splits is None
        assert label_map.split_policy["kind"] == "random"

    def test_no_source_code_feeds_two_outputs(self, label_map) -> None:
        for code, outputs in label_map.code_to_classes.items():
            assert len(outputs) == 1, f"{code} maps to {outputs}"

    def test_risk_head_stays_unlabelable(self, label_map) -> None:
        assert label_map.classes_for_task("risk") == ()

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

    def test_judgement_calls_carry_a_rationale(self, label_map) -> None:
        for name, mapping in label_map.classes.items():
            if len(mapping.codes) > 1:
                assert mapping.note, f"{name} pools {mapping.codes} without a note"

    def test_atrial_flutter_is_mapped_by_concept_not_acronym(
        self, label_map
    ) -> None:
        """The trap this corpus sets for a careless mapping.

        Chapman's acronym for atrial flutter is "AF" — Aortica's name for
        atrial fibrillation. Mapping by acronym swaps two common classes and
        nothing downstream would notice.
        """
        assert label_map.classes["AFL"].codes == ("164890007",)
        assert label_map.classes["AF"].codes == ("164889003",)


# ----------------------------------------------------------------------
# Header parsing
# ----------------------------------------------------------------------


class TestParseHeaderMetadata:
    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "JS00001.hea"
        path.write_text(body, encoding="utf-8")
        return path

    def test_reads_age_sex_and_dx(self, tmp_path: Path) -> None:
        header = self._write(
            tmp_path,
            "JS00001 12 500 5000\n#Age: 77\n#Sex: Male\n"
            "#Dx: 164889003,55827005,428750005\n#Rx: Unknown\n",
        )
        meta = parse_header_metadata(header)
        assert meta["age"] == 77.0
        assert meta["sex"] == "male"
        assert meta["dx"] == ("164889003", "55827005", "428750005")

    def test_missing_dx_yields_no_codes(self, tmp_path: Path) -> None:
        header = self._write(tmp_path, "JS00001 12 500 5000\n#Age: 50\n")
        assert parse_header_metadata(header)["dx"] == ()

    def test_unparseable_age_is_none(self, tmp_path: Path) -> None:
        header = self._write(tmp_path, "JS1 12 500 5000\n#Age: NaN yrs\n")
        assert parse_header_metadata(header)["age"] is None

    def test_unknown_sex_is_normalised(self, tmp_path: Path) -> None:
        header = self._write(tmp_path, "JS1 12 500 5000\n#Sex: Unknown\n")
        assert parse_header_metadata(header)["sex"] == "unknown"


# ----------------------------------------------------------------------
# Label matrix construction
# ----------------------------------------------------------------------


class TestBuildLabels:
    def test_multitask_is_the_full_concatenation(self, label_map) -> None:
        from aortica.models.task_dims import TOTAL_OUTPUTS

        labels = build_labels([()], multitask=True, label_map=label_map)
        assert labels.shape == (1, TOTAL_OUTPUTS)
        assert labels.dtype == np.float32

    def test_sets_the_column_a_code_maps_to(self, label_map) -> None:
        from aortica.data.ptbxl import _output_columns

        _, index = _output_columns(multitask=True)
        labels = build_labels(
            [("164889003",)], multitask=True, label_map=label_map
        )
        assert labels[0, index["AF"]] == 1.0
        assert labels[0].sum() == 1.0

    def test_out_of_vocabulary_codes_label_nothing(self, label_map) -> None:
        """The 43 header codes outside the shipped vocabulary stay inert."""
        labels = build_labels(
            [("55827005", "713427006", "427172004", "10370003")],
            multitask=True,
            label_map=label_map,
        )
        assert labels.sum() == 0.0

    def test_unlabelable_columns_stay_zero(self, label_map) -> None:
        from aortica.data.ptbxl import _output_columns

        names, _ = _output_columns(multitask=True)
        labels = build_labels(
            [(code,) for code in label_map.code_to_classes],
            multitask=True,
            label_map=label_map,
        )
        for column, name in enumerate(names):
            if name not in label_map.classes:
                assert labels[:, column].sum() == 0.0, f"{name} was labelled"


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------


class TestSplitIndices:
    POLICY = {
        "kind": "random",
        "seed": 42,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
    }

    def test_covers_every_record_exactly_once(self) -> None:
        splits = split_indices(1000, self.POLICY)
        combined = np.concatenate(list(splits.values()))
        assert sorted(combined.tolist()) == list(range(1000))

    def test_splits_are_disjoint(self) -> None:
        splits = split_indices(1000, self.POLICY)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not set(splits[a]) & set(splits[b])

    def test_fractions_are_respected(self) -> None:
        splits = split_indices(1000, self.POLICY)
        assert len(splits["train"]) == 800
        assert len(splits["val"]) == 100
        assert len(splits["test"]) == 100

    def test_is_deterministic_for_a_seed(self) -> None:
        a = split_indices(500, self.POLICY)
        b = split_indices(500, self.POLICY)
        for key in a:
            assert np.array_equal(a[key], b[key])

    def test_seed_override_changes_the_split(self) -> None:
        a = split_indices(500, self.POLICY)
        b = split_indices(500, self.POLICY, seed=7)
        assert not np.array_equal(a["test"], b["test"])

    def test_fractions_must_sum_to_one(self) -> None:
        policy = {**self.POLICY, "fractions": {"train": 0.8, "val": 0.1, "test": 0.2}}
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_indices(100, policy)

    def test_unsupported_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported split kind"):
            split_indices(100, {**self.POLICY, "kind": "temporal"})


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


class TestFailureModes:
    def test_missing_corpus_names_the_download(self, tmp_path: Path) -> None:
        with pytest.raises(ChapmanDataNotFoundError) as excinfo:
            load_chapman(tmp_path / "nowhere")
        assert "physionet.org" in str(excinfo.value)

    def test_incomplete_corpus_names_what_is_missing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "WFDBRecords").mkdir()
        with pytest.raises(ChapmanDataNotFoundError, match="ConditionNames"):
            load_chapman(tmp_path)

    def test_error_is_a_filenotfounderror(self) -> None:
        assert issubclass(ChapmanDataNotFoundError, FileNotFoundError)

    def test_sampling_rate_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            load_chapman(tmp_path, sampling_rate=0)


# ----------------------------------------------------------------------
# Corpus-backed (skipped unless AORTICA_CHAPMAN_ROOT is set)
# ----------------------------------------------------------------------


class TestAgainstTheRealCorpus:
    def test_corpus_totals_match_the_mapping_file(
        self, chapman_root, label_map
    ) -> None:
        """Every `corpus_total` must still be true of the corpus on disk.

        This is the closest Chapman gets to PTB-XL's exact-count check. It
        cannot confirm the mapping is the one v0.3.0 used, but it does catch
        a code list edited without re-deriving the evidence beside it.
        """
        from aortica.data.chapman import index_records

        headers = index_records(chapman_root)
        assert len(headers) == 45152

        dx_sets = [parse_header_metadata(h)["dx"] for h in headers]
        for name, mapping in label_map.classes.items():
            codes = set(mapping.codes)
            actual = sum(1 for dx in dx_sets if codes & set(dx))
            assert actual == mapping.raw["corpus_total"], (
                f"{name}: mapping says {mapping.raw['corpus_total']} "
                f"positives corpus-wide, found {actual}"
            )

    def test_vocabulary_is_the_size_the_evidence_assumes(
        self, chapman_root
    ) -> None:
        vocabulary = load_condition_names(
            chapman_root / "ConditionNames_SNOMED-CT.csv"
        )
        assert len(vocabulary) == 55

    def test_records_are_twelve_lead(self, chapman_root) -> None:
        (records, _), _, _ = load_chapman(
            chapman_root, sampling_rate=100, limit=5
        )
        for record in records:
            assert record.num_leads == 12
            assert record.sample_rate == 100.0
            assert record.units == "µV"
            assert record.source_format == "chapman"

    def test_demographics_and_codes_survive_loading(
        self, chapman_root
    ) -> None:
        (records, _), _, _ = load_chapman(
            chapman_root, sampling_rate=100, limit=25
        )
        assert any(
            (r.patient_metadata or {}).get("sex") in {"male", "female"}
            for r in records
        )
        assert any((r.patient_metadata or {}).get("snomed_dx") for r in records)

    def test_labels_align_row_for_row_with_records(self, chapman_root) -> None:
        for records, labels in load_chapman(
            chapman_root, sampling_rate=100, limit=20
        ):
            assert len(records) == labels.shape[0]
