"""Tests for aortica.validation.prospective_collector — US-099.

Covers:
- ProspectiveCollector initialization
- ECG ingestion with encrypted storage
- Ground-truth outcome linkage
- Record queries (by ID, by hash, list with filters)
- export_study_data() CSV generation
- Context manager usage
- Module imports
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aortica.validation.prospective_collector import (
    ProspectiveCollector,
    StudyRecord,
    export_study_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collector(tmp_path: Path) -> ProspectiveCollector:
    """Create a ProspectiveCollector with a temporary database."""
    return ProspectiveCollector(db_dir=str(tmp_path / "study"))


@pytest.fixture
def populated_collector(
    collector: ProspectiveCollector,
) -> ProspectiveCollector:
    """Collector with several ingested records."""
    # Site A — 3 records
    for i in range(3):
        rid = collector.ingest_ecg(
            ecg_hash=f"ecg_a_{i}",
            site_id="site_A",
            predictions={"rhythm": [0.1 * i, 0.9 - 0.1 * i]},
            quality={"overall": 80 + i},
            metadata={"age": 50 + i, "sex": "M" if i % 2 == 0 else "F"},
            timestamp=1000.0 + i,
        )
        # Link ground truth for the first 2
        if i < 2:
            collector.add_outcome(
                record_id=rid,
                ground_truth={"AF": i % 2, "STEMI": 0},
                clinician_id=f"dr_{i}",
                outcome={"mace_30d": False},
                outcome_timestamp=2000.0 + i,
            )

    # Site B — 2 records (no ground truth)
    for i in range(2):
        collector.ingest_ecg(
            ecg_hash=f"ecg_b_{i}",
            site_id="site_B",
            predictions={"structural": [0.2, 0.8]},
            quality={"overall": 90},
            metadata={"age": 65, "sex": "F"},
            timestamp=3000.0 + i,
        )

    return collector


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test ProspectiveCollector initialization."""

    def test_creates_database(self, tmp_path: Path) -> None:
        collector = ProspectiveCollector(db_dir=str(tmp_path / "init_test"))
        db_path = tmp_path / "init_test" / "prospective.db"
        assert db_path.exists()
        collector.close()

    def test_creates_encryption_key(self, tmp_path: Path) -> None:
        collector = ProspectiveCollector(db_dir=str(tmp_path / "key_test"))
        key_path = tmp_path / "key_test" / "prospective.key"
        assert key_path.exists()
        collector.close()

    def test_reuses_existing_key(self, tmp_path: Path) -> None:
        db_dir = str(tmp_path / "reuse_test")
        c1 = ProspectiveCollector(db_dir=db_dir)
        rid = c1.ingest_ecg("hash1", "site1", {"a": 1}, {"overall": 90})
        c1.close()

        # Reopen — should use same key and decrypt successfully
        c2 = ProspectiveCollector(db_dir=db_dir)
        rec = c2.get_record(rid)
        assert rec is not None
        assert rec.predictions == {"a": 1}
        c2.close()

    def test_custom_encryption_key(self, tmp_path: Path) -> None:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        collector = ProspectiveCollector(
            db_dir=str(tmp_path / "custom_key"),
            encryption_key=key,
        )
        rid = collector.ingest_ecg("h1", "s1", {"x": 42}, {"overall": 85})
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.predictions == {"x": 42}
        collector.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        with ProspectiveCollector(
            db_dir=str(tmp_path / "ctx")
        ) as collector:
            rid = collector.ingest_ecg("h", "s", {}, {})
            assert rid > 0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    """Test ECG ingestion."""

    def test_ingest_returns_id(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg(
            ecg_hash="test_hash",
            site_id="site_X",
            predictions={"rhythm": [0.5]},
            quality={"overall": 75},
        )
        assert isinstance(rid, int)
        assert rid > 0

    def test_predictions_encrypted(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg(
            "hash", "site", {"secret": "data"}, {"q": 1}
        )
        # Read raw from database — predictions_json should be encrypted
        row = collector._conn.execute(
            "SELECT predictions_json FROM prospective_records WHERE id = ?",
            (rid,),
        ).fetchone()
        raw = row[0]
        # Should not be readable as plain JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_metadata_stored(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg(
            "hash", "site", {}, {},
            metadata={"age": 55, "sex": "M"},
        )
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.metadata == {"age": 55, "sex": "M"}

    def test_custom_timestamp(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg(
            "hash", "site", {}, {},
            timestamp=1234567890.0,
        )
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.timestamp == 1234567890.0

    def test_site_id_stored(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg("hash", "my_site", {}, {})
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.site_id == "my_site"


# ---------------------------------------------------------------------------
# Outcome linkage
# ---------------------------------------------------------------------------


class TestOutcomeLinkage:
    """Test ground-truth and outcome linkage."""

    def test_add_outcome(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg("h1", "s1", {"r": 1}, {"q": 1})
        success = collector.add_outcome(
            record_id=rid,
            ground_truth={"AF": 1, "STEMI": 0},
            clinician_id="dr_jones",
        )
        assert success is True

    def test_outcome_retrievable(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg("h2", "s1", {"r": 1}, {"q": 1})
        collector.add_outcome(
            record_id=rid,
            ground_truth={"AF": 1},
            clinician_id="dr_smith",
            outcome={"mace_30d": True},
        )
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.ground_truth == {"AF": 1}
        assert rec.clinician_id == "dr_smith"
        assert rec.outcome == {"mace_30d": True}

    def test_add_outcome_nonexistent_record(
        self, collector: ProspectiveCollector
    ) -> None:
        success = collector.add_outcome(
            record_id=99999,
            ground_truth={"AF": 1},
        )
        assert success is False

    def test_outcome_timestamp(
        self, collector: ProspectiveCollector
    ) -> None:
        rid = collector.ingest_ecg("h3", "s1", {}, {})
        collector.add_outcome(
            record_id=rid,
            ground_truth={"VT": 1},
            outcome_timestamp=9999.0,
        )
        rec = collector.get_record(rid)
        assert rec is not None
        assert rec.outcome_timestamp == 9999.0


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    """Test record queries."""

    def test_get_by_hash(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        rec = populated_collector.get_record_by_hash("ecg_a_0")
        assert rec is not None
        assert rec.ecg_hash == "ecg_a_0"
        assert rec.site_id == "site_A"

    def test_get_nonexistent_hash(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        rec = populated_collector.get_record_by_hash("nonexistent")
        assert rec is None

    def test_list_all(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        records = populated_collector.list_records()
        assert len(records) == 5  # 3 site_A + 2 site_B

    def test_list_by_site(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        records = populated_collector.list_records(site_id="site_A")
        assert len(records) == 3
        assert all(r.site_id == "site_A" for r in records)

    def test_list_linked_only(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        records = populated_collector.list_records(linked_only=True)
        assert len(records) == 2  # Only first 2 of site_A have ground truth
        assert all(r.ground_truth != {} for r in records)

    def test_list_linked_by_site(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        records = populated_collector.list_records(
            site_id="site_B", linked_only=True
        )
        assert len(records) == 0  # Site B has no ground truth

    def test_count_all(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        assert populated_collector.count() == 5

    def test_count_by_site(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        assert populated_collector.count(site_id="site_A") == 3
        assert populated_collector.count(site_id="site_B") == 2

    def test_count_linked(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        assert populated_collector.count(linked_only=True) == 2

    def test_pagination(
        self, populated_collector: ProspectiveCollector
    ) -> None:
        page1 = populated_collector.list_records(limit=2, offset=0)
        page2 = populated_collector.list_records(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should be different
        ids1 = {r.id for r in page1}
        ids2 = {r.id for r in page2}
        assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportStudyData:
    """Test export_study_data()."""

    def test_export_creates_file(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "export.csv")
        count = export_study_data(populated_collector, output)
        assert count > 0
        assert Path(output).exists()

    def test_export_linked_only_default(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "linked.csv")
        count = export_study_data(populated_collector, output)
        # Default linked_only=True → only 2 records with ground truth
        assert count == 2

    def test_export_all_records(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "all.csv")
        count = export_study_data(
            populated_collector, output, linked_only=False
        )
        assert count == 5

    def test_export_by_site(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "site_b.csv")
        count = export_study_data(
            populated_collector, output,
            site_id="site_B", linked_only=False,
        )
        assert count == 2

    def test_csv_has_headers(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "headers.csv")
        export_study_data(populated_collector, output)
        with open(output, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        expected = [
            "record_id", "ecg_hash", "site_id", "age", "sex",
            "quality_overall", "predictions_json", "ground_truth_json",
            "clinician_id", "outcome_json", "timestamp", "outcome_timestamp",
        ]
        assert headers == expected

    def test_csv_deidentified(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        """CSV should include age/sex but not patient names or MRNs."""
        output = str(tmp_path / "deidentified.csv")
        export_study_data(populated_collector, output)
        content = Path(output).read_text()
        # Should have age values from metadata
        assert "50" in content or "51" in content
        # Should NOT have any name/MRN fields (we didn't add any)
        assert "patient_name" not in content
        assert "mrn" not in content

    def test_export_empty(
        self, collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        """Empty collector should export CSV with headers only."""
        output = str(tmp_path / "empty.csv")
        count = export_study_data(collector, output)
        assert count == 0
        with open(output, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1  # Headers only

    def test_export_creates_parent_dirs(
        self, populated_collector: ProspectiveCollector, tmp_path: Path
    ) -> None:
        output = str(tmp_path / "deep" / "nested" / "export.csv")
        count = export_study_data(
            populated_collector, output, linked_only=False
        )
        assert count > 0
        assert Path(output).exists()


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify validation module exports are accessible."""

    def test_import_prospective_collector(self) -> None:
        from aortica.validation import ProspectiveCollector  # noqa: F811

    def test_import_study_record(self) -> None:
        from aortica.validation import StudyRecord  # noqa: F811

    def test_import_export_study_data(self) -> None:
        from aortica.validation import export_study_data  # noqa: F811

    def test_import_from_submodule(self) -> None:
        from aortica.validation.prospective_collector import (  # noqa: F811
            ProspectiveCollector,
            StudyRecord,
            export_study_data,
        )
