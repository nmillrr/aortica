"""Tests for aortica.reports.csv_export — CSV Batch Analytics Export.

Covers:
* Column count and header structure
* Single and batch row generation
* Dict, dataclass, and partial-data inputs
* Streaming write to file (export_csv)
* In-memory string export (export_csv_string)
* Auxiliary metadata columns (filenames, quality, urgency, OOD)
* Length-mismatch validation errors
* Large batch streaming (10,000 rows)
* API endpoint (POST /api/v1/export/csv)
"""

from __future__ import annotations

import csv
import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES
from aortica.reports.csv_export import (
    _all_columns,
    _build_row,
    _extract_predictions,
    export_csv,
    export_csv_string,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_CLASSES = (
    len(RHYTHM_CLASSES)
    + len(STRUCTURAL_CLASSES)
    + len(ISCHAEMIA_CLASSES)
    + len(RISK_OUTPUTS)
)

# Expected column count: filename, quality_score, all class columns,
# urgency_score, OOD_flag
EXPECTED_COLUMNS = 2 + TOTAL_CLASSES + 2


@dataclass
class FakeMultiTaskOutput:
    """Mimics MultiTaskOutput for testing."""

    rhythm: List[float] = field(default_factory=list)
    structural: List[float] = field(default_factory=list)
    ischaemia: List[float] = field(default_factory=list)
    risk: List[float] = field(default_factory=list)


def _make_result(seed: float = 0.5) -> Dict[str, List[float]]:
    """Create a synthetic result dict with all tasks populated."""
    return {
        "rhythm": [seed + i * 0.001 for i in range(len(RHYTHM_CLASSES))],
        "structural": [seed + i * 0.001 for i in range(len(STRUCTURAL_CLASSES))],
        "ischaemia": [seed + i * 0.001 for i in range(len(ISCHAEMIA_CLASSES))],
        "risk": [seed + i * 0.01 for i in range(len(RISK_OUTPUTS))],
    }


# ---------------------------------------------------------------------------
# Column structure tests
# ---------------------------------------------------------------------------


class TestColumnStructure:
    """Verify the CSV column definitions."""

    def test_total_column_count(self) -> None:
        cols = _all_columns()
        assert len(cols) == EXPECTED_COLUMNS

    def test_starts_with_filename(self) -> None:
        cols = _all_columns()
        assert cols[0] == "filename"

    def test_second_column_quality(self) -> None:
        cols = _all_columns()
        assert cols[1] == "quality_score"

    def test_ends_with_ood_flag(self) -> None:
        cols = _all_columns()
        assert cols[-1] == "OOD_flag"

    def test_urgency_before_ood(self) -> None:
        cols = _all_columns()
        assert cols[-2] == "urgency_score"

    def test_rhythm_columns_present(self) -> None:
        cols = _all_columns()
        for cls in RHYTHM_CLASSES:
            assert f"rhythm_{cls}" in cols

    def test_structural_columns_present(self) -> None:
        cols = _all_columns()
        for cls in STRUCTURAL_CLASSES:
            assert f"structural_{cls}" in cols

    def test_ischaemia_columns_present(self) -> None:
        cols = _all_columns()
        for cls in ISCHAEMIA_CLASSES:
            assert f"ischaemia_{cls}" in cols

    def test_risk_columns_present(self) -> None:
        cols = _all_columns()
        for cls in RISK_OUTPUTS:
            assert f"risk_{cls}" in cols

    def test_no_duplicate_columns(self) -> None:
        cols = _all_columns()
        assert len(cols) == len(set(cols))


# ---------------------------------------------------------------------------
# Prediction extraction tests
# ---------------------------------------------------------------------------


class TestExtractPredictions:
    """Verify _extract_predictions with various input types."""

    def test_dict_input(self) -> None:
        result = _make_result(0.3)
        preds = _extract_predictions(result)
        assert set(preds.keys()) == {"rhythm", "structural", "ischaemia", "risk"}
        assert len(preds["rhythm"]) == len(RHYTHM_CLASSES)

    def test_dataclass_input(self) -> None:
        output = FakeMultiTaskOutput(
            rhythm=[0.1] * len(RHYTHM_CLASSES),
            structural=[0.2] * len(STRUCTURAL_CLASSES),
            ischaemia=[0.3] * len(ISCHAEMIA_CLASSES),
            risk=[0.4] * len(RISK_OUTPUTS),
        )
        preds = _extract_predictions(output)
        assert len(preds["structural"]) == len(STRUCTURAL_CLASSES)
        assert preds["risk"][0] == pytest.approx(0.4)

    def test_partial_input(self) -> None:
        result = {"rhythm": [0.5] * len(RHYTHM_CLASSES)}
        preds = _extract_predictions(result)
        assert "rhythm" in preds
        assert "structural" not in preds

    def test_batch_dimension_unwrap(self) -> None:
        """Batch dimension [1, N] should unwrap to [N]."""
        inner = [0.7] * len(RHYTHM_CLASSES)
        result = {"rhythm": [inner]}
        preds = _extract_predictions(result)
        assert len(preds["rhythm"]) == len(RHYTHM_CLASSES)
        assert preds["rhythm"][0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Row building tests
# ---------------------------------------------------------------------------


class TestBuildRow:
    """Verify _build_row produces correct row data."""

    def test_row_length_matches_columns(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, filename="test.dat")
        assert len(row) == EXPECTED_COLUMNS

    def test_filename_in_first_column(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, filename="sample.wfdb")
        assert row[0] == "sample.wfdb"

    def test_quality_score_formatted(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, quality_score=85.5)
        assert row[1] == "85.5"

    def test_quality_score_none(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, quality_score=None)
        assert row[1] == ""

    def test_urgency_score_formatted(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, urgency_score=72.3)
        assert row[-2] == "72.3"

    def test_ood_flag_true(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, ood_flag=True)
        assert row[-1] == "1"

    def test_ood_flag_false(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, ood_flag=False)
        assert row[-1] == "0"

    def test_ood_flag_none(self) -> None:
        result = _make_result(0.5)
        row = _build_row(result, ood_flag=None)
        assert row[-1] == ""

    def test_partial_predictions_fill_blanks(self) -> None:
        """Missing tasks produce empty cells in their columns."""
        result: Dict[str, Any] = {"rhythm": [0.9] * len(RHYTHM_CLASSES)}
        row = _build_row(result, filename="partial.dat")
        assert len(row) == EXPECTED_COLUMNS
        # First rhythm column should be populated
        assert row[2] != ""
        # First structural column should be empty
        rhythm_end = 2 + len(RHYTHM_CLASSES)
        assert row[rhythm_end] == ""


# ---------------------------------------------------------------------------
# File export tests
# ---------------------------------------------------------------------------


class TestExportCsv:
    """Verify export_csv produces valid CSV files."""

    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(results, tmp_path / "out.csv")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_header_row(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(results, tmp_path / "out.csv")
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == _all_columns()

    def test_row_count(self, tmp_path: Path) -> None:
        results = [_make_result(i * 0.1) for i in range(5)]
        out = export_csv(results, tmp_path / "out.csv")
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 5 data rows
        assert len(rows) == 6

    def test_column_count_per_row(self, tmp_path: Path) -> None:
        results = [_make_result(0.5), _make_result(0.6)]
        out = export_csv(results, tmp_path / "out.csv")
        with open(out) as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == EXPECTED_COLUMNS

    def test_custom_filenames(self, tmp_path: Path) -> None:
        results = [_make_result(0.5), _make_result(0.6)]
        names = ["alpha.dat", "beta.dat"]
        out = export_csv(results, tmp_path / "out.csv", filenames=names)
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row1 = next(reader)
            row2 = next(reader)
        assert row1[0] == "alpha.dat"
        assert row2[0] == "beta.dat"

    def test_default_filenames(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(results, tmp_path / "out.csv")
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[0] == "ecg_00000"

    def test_quality_scores(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(
            results, tmp_path / "out.csv", quality_scores=[92.3]
        )
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[1] == "92.3"

    def test_urgency_and_ood(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(
            results,
            tmp_path / "out.csv",
            urgency_scores=[88.0],
            ood_flags=[True],
        )
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[-2] == "88.0"
        assert row[-1] == "1"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        deep = tmp_path / "a" / "b" / "c" / "out.csv"
        out = export_csv(results, deep)
        assert out.exists()

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        out = export_csv(results, tmp_path / "out.csv")
        assert out.is_absolute()


# ---------------------------------------------------------------------------
# String export tests
# ---------------------------------------------------------------------------


class TestExportCsvString:
    """Verify export_csv_string produces valid CSV content."""

    def test_returns_string(self) -> None:
        results = [_make_result(0.5)]
        content = export_csv_string(results)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_header_present(self) -> None:
        results = [_make_result(0.5)]
        content = export_csv_string(results)
        reader = csv.reader(io.StringIO(content))
        header = next(reader)
        assert header == _all_columns()

    def test_row_count(self) -> None:
        results = [_make_result(i * 0.1) for i in range(3)]
        content = export_csv_string(results)
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 4  # 1 header + 3 data

    def test_column_count_per_row(self) -> None:
        results = [_make_result(0.5)]
        content = export_csv_string(results)
        reader = csv.reader(io.StringIO(content))
        for row in reader:
            assert len(row) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Verify length-mismatch validation raises ValueError."""

    def test_filenames_length_mismatch(self, tmp_path: Path) -> None:
        results = [_make_result(0.5), _make_result(0.6)]
        with pytest.raises(ValueError, match="filenames length"):
            export_csv(results, tmp_path / "out.csv", filenames=["one"])

    def test_quality_scores_length_mismatch(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        with pytest.raises(ValueError, match="quality_scores length"):
            export_csv(results, tmp_path / "out.csv", quality_scores=[1, 2])

    def test_urgency_scores_length_mismatch(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        with pytest.raises(ValueError, match="urgency_scores length"):
            export_csv(results, tmp_path / "out.csv", urgency_scores=[1, 2])

    def test_ood_flags_length_mismatch(self, tmp_path: Path) -> None:
        results = [_make_result(0.5)]
        with pytest.raises(ValueError, match="ood_flags length"):
            export_csv(results, tmp_path / "out.csv", ood_flags=[True, False])

    def test_string_export_validates_too(self) -> None:
        results = [_make_result(0.5)]
        with pytest.raises(ValueError, match="filenames length"):
            export_csv_string(results, filenames=["a", "b"])


# ---------------------------------------------------------------------------
# Large batch streaming test
# ---------------------------------------------------------------------------


class TestLargeBatch:
    """Verify streaming write handles 10,000 rows without issues."""

    def test_10k_rows(self, tmp_path: Path) -> None:
        n = 10_000
        results = [_make_result(i * 0.0001) for i in range(n)]
        out = export_csv(results, tmp_path / "big.csv")
        assert out.exists()

        # Verify row count by streaming read
        count = 0
        with open(out) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for _ in reader:
                count += 1
        assert count == n

    def test_10k_column_count_spot_check(self, tmp_path: Path) -> None:
        n = 10_000
        results = [_make_result(i * 0.0001) for i in range(n)]
        out = export_csv(results, tmp_path / "big.csv")

        with open(out) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            # Spot-check first, middle, and last rows
            rows = list(reader)
            assert len(rows[0]) == EXPECTED_COLUMNS
            assert len(rows[n // 2]) == EXPECTED_COLUMNS
            assert len(rows[-1]) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestCSVExportEndpoint:
    """Verify the POST /api/v1/export/csv API endpoint."""

    @pytest.fixture()
    def client(self) -> Any:
        """Create a test client with auth disabled."""
        from fastapi.testclient import TestClient

        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_export_csv_success(self, client: Any) -> None:
        body = {
            "results": [_make_result(0.5), _make_result(0.6)],
            "filenames": ["test1.dat", "test2.dat"],
        }
        resp = client.post("/api/v1/export/csv", json=body)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in resp.headers.get("content-disposition", "")

        # Parse content
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[0] == _all_columns()

    def test_export_csv_empty_results(self, client: Any) -> None:
        body: Dict[str, Any] = {"results": []}
        resp = client.post("/api/v1/export/csv", json=body)
        assert resp.status_code == 422

    def test_export_csv_no_results_key(self, client: Any) -> None:
        body: Dict[str, Any] = {}
        resp = client.post("/api/v1/export/csv", json=body)
        assert resp.status_code == 422

    def test_export_csv_with_all_metadata(self, client: Any) -> None:
        body = {
            "results": [_make_result(0.5)],
            "filenames": ["meta.dat"],
            "quality_scores": [95.0],
            "urgency_scores": [42.0],
            "ood_flags": [False],
        }
        resp = client.post("/api/v1/export/csv", json=body)
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        next(reader)  # skip header
        row = next(reader)
        assert row[0] == "meta.dat"
        assert row[1] == "95.0"
        assert row[-2] == "42.0"
        assert row[-1] == "0"

    def test_export_csv_mismatched_lengths(self, client: Any) -> None:
        body = {
            "results": [_make_result(0.5)],
            "filenames": ["a.dat", "b.dat"],
        }
        resp = client.post("/api/v1/export/csv", json=body)
        assert resp.status_code == 422
        assert "filenames length" in resp.json()["detail"]
