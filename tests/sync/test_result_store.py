"""Tests for aortica.sync.result_store — SQLite local storage with AES-256 encryption."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip if cryptography is not installed
# ---------------------------------------------------------------------------
cryptography = pytest.importorskip("cryptography", reason="cryptography required for sync tests")

from aortica.sync.result_store import (  # noqa: E402
    HAS_CRYPTOGRAPHY,
    ResultStore,
    StoredResult,
    _CREATE_TABLE_SQL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for the database."""
    return tmp_path / "aortica_results"


@pytest.fixture
def store(db_dir: Path) -> ResultStore:
    """Return a ResultStore instance using a temporary directory."""
    s = ResultStore(db_dir)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def sample_predictions() -> dict[str, Any]:
    return {
        "rhythm": {"AF": 0.92, "normal_sinus_rhythm": 0.08},
        "structural": {"LVH": 0.45},
        "ischaemia": {"STEMI": 0.01},
        "risk": {"mortality_1y": 0.12, "hf_12m": 0.08, "af_12m": 0.85},
    }


@pytest.fixture
def sample_quality() -> dict[str, Any]:
    return {
        "overall_score": 82,
        "classification": "good",
        "per_lead": {"I": 90, "II": 85, "V1": 70},
    }


# ===================================================================
# 1. Constants and module-level checks
# ===================================================================


class TestModuleConstants:
    """Module-level constants and import guards."""

    def test_has_cryptography_flag(self) -> None:
        assert HAS_CRYPTOGRAPHY is True

    def test_create_table_sql_contains_required_columns(self) -> None:
        for col in [
            "id INTEGER PRIMARY KEY",
            "ecg_hash TEXT",
            "predictions_json BLOB",
            "quality_json TEXT",
            "timestamp REAL",
            "synced INTEGER",
        ]:
            assert col in _CREATE_TABLE_SQL


# ===================================================================
# 2. StoredResult dataclass
# ===================================================================


class TestStoredResult:
    """StoredResult data container."""

    def test_construction(self) -> None:
        r = StoredResult(
            id=1,
            ecg_hash="abc",
            predictions={"rhythm": {}},
            quality={"overall": 80},
            timestamp=1000.0,
            synced=False,
        )
        assert r.id == 1
        assert r.ecg_hash == "abc"
        assert r.synced is False

    def test_default_metadata(self) -> None:
        r = StoredResult(
            id=1, ecg_hash="x", predictions={}, quality={}, timestamp=0, synced=True
        )
        assert r.metadata == {}

    def test_custom_metadata(self) -> None:
        r = StoredResult(
            id=1,
            ecg_hash="x",
            predictions={},
            quality={},
            timestamp=0,
            synced=False,
            metadata={"device": "RPi4"},
        )
        assert r.metadata["device"] == "RPi4"


# ===================================================================
# 3. Initialisation
# ===================================================================


class TestInitialisation:
    """ResultStore construction and database setup."""

    def test_creates_directory(self, db_dir: Path) -> None:
        assert not db_dir.exists()
        s = ResultStore(db_dir)
        s.close()
        assert db_dir.exists()

    def test_creates_database_file(self, db_dir: Path) -> None:
        s = ResultStore(db_dir)
        s.close()
        assert (db_dir / "results.db").exists()

    def test_creates_key_file(self, db_dir: Path) -> None:
        s = ResultStore(db_dir)
        s.close()
        assert (db_dir / "result_store.key").exists()

    def test_key_file_permissions(self, db_dir: Path) -> None:
        s = ResultStore(db_dir)
        s.close()
        mode = os.stat(db_dir / "result_store.key").st_mode & 0o777
        assert mode == 0o600

    def test_custom_db_filename(self, db_dir: Path) -> None:
        s = ResultStore(db_dir, db_filename="custom.db")
        s.close()
        assert (db_dir / "custom.db").exists()

    def test_explicit_key(self, db_dir: Path) -> None:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        s = ResultStore(db_dir, encryption_key=key)
        s.close()
        # Key file should NOT be created when key is provided explicitly
        assert not (db_dir / "result_store.key").exists()

    def test_reuses_existing_key(self, db_dir: Path) -> None:
        """Second open reuses the persisted key (can read data from first)."""
        s1 = ResultStore(db_dir)
        rid = s1.store_result("hash1", {"a": 1}, {"q": 90})
        s1.close()

        s2 = ResultStore(db_dir)
        result = s2.get_result_by_id(rid)
        s2.close()
        assert result is not None
        assert result.predictions == {"a": 1}

    def test_context_manager(self, db_dir: Path) -> None:
        with ResultStore(db_dir) as s:
            s.store_result("cm_hash", {"p": 1}, {"q": 2})
        # connection closed — should not raise

    def test_results_table_exists(self, store: ResultStore) -> None:
        """The results table should exist after construction."""
        conn = sqlite3.connect(str(store._db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
        )
        assert cursor.fetchone() is not None
        conn.close()


# ===================================================================
# 4. store_result
# ===================================================================


class TestStoreResult:
    """Inserting results into the store."""

    def test_returns_row_id(
        self,
        store: ResultStore,
        sample_predictions: dict[str, Any],
        sample_quality: dict[str, Any],
    ) -> None:
        rid = store.store_result("h1", sample_predictions, sample_quality)
        assert isinstance(rid, int)
        assert rid >= 1

    def test_auto_timestamp(self, store: ResultStore) -> None:
        before = time.time()
        store.store_result("h1", {"a": 1}, {"q": 1})
        after = time.time()

        result = store.get_result("h1")
        assert result is not None
        assert before <= result.timestamp <= after

    def test_explicit_timestamp(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {}, timestamp=12345.0)
        result = store.get_result("h1")
        assert result is not None
        assert result.timestamp == 12345.0

    def test_synced_defaults_false(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        result = store.get_result("h1")
        assert result is not None
        assert result.synced is False

    def test_metadata_stored(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {}, metadata={"patient_age": 65})
        result = store.get_result("h1")
        assert result is not None
        assert result.metadata["patient_age"] == 65

    def test_metadata_defaults_empty(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        result = store.get_result("h1")
        assert result is not None
        assert result.metadata == {}

    def test_multiple_results_same_hash(self, store: ResultStore) -> None:
        """Multiple results for the same ECG hash are allowed."""
        store.store_result("h1", {"v": 1}, {}, timestamp=100.0)
        store.store_result("h1", {"v": 2}, {}, timestamp=200.0)
        # get_result returns the newest
        result = store.get_result("h1")
        assert result is not None
        assert result.predictions["v"] == 2

    def test_sequential_ids(self, store: ResultStore) -> None:
        id1 = store.store_result("h1", {}, {})
        id2 = store.store_result("h2", {}, {})
        assert id2 > id1


# ===================================================================
# 5. Encryption verification
# ===================================================================


class TestEncryption:
    """Verify predictions are encrypted at rest, not plaintext."""

    def test_predictions_not_plaintext_in_db(
        self,
        store: ResultStore,
        sample_predictions: dict[str, Any],
        sample_quality: dict[str, Any],
    ) -> None:
        """Raw database column must NOT contain plaintext JSON."""
        store.store_result("enc_test", sample_predictions, sample_quality)

        conn = sqlite3.connect(str(store._db_path))
        row = conn.execute(
            "SELECT predictions_json FROM results WHERE ecg_hash = ?", ("enc_test",)
        ).fetchone()
        conn.close()

        assert row is not None
        raw_blob = row[0]

        # The raw blob should NOT be valid JSON
        if isinstance(raw_blob, bytes):
            try:
                parsed = json.loads(raw_blob)
                # If it parses as JSON, it should NOT match the original predictions
                assert parsed != sample_predictions, "Predictions stored as plaintext!"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Expected — encrypted data is not valid JSON

    def test_quality_is_plaintext(
        self,
        store: ResultStore,
        sample_quality: dict[str, Any],
    ) -> None:
        """Quality column is NOT encrypted (per PRD: only predictions are encrypted)."""
        store.store_result("q_test", {}, sample_quality)

        conn = sqlite3.connect(str(store._db_path))
        row = conn.execute(
            "SELECT quality_json FROM results WHERE ecg_hash = ?", ("q_test",)
        ).fetchone()
        conn.close()

        assert row is not None
        parsed = json.loads(row[0])
        assert parsed == sample_quality

    def test_decryption_round_trip(
        self,
        store: ResultStore,
        sample_predictions: dict[str, Any],
        sample_quality: dict[str, Any],
    ) -> None:
        """Encrypt → store → retrieve → decrypt should yield original data."""
        store.store_result("rt_test", sample_predictions, sample_quality)
        result = store.get_result("rt_test")
        assert result is not None
        assert result.predictions == sample_predictions

    def test_wrong_key_cannot_decrypt(
        self,
        db_dir: Path,
        sample_predictions: dict[str, Any],
    ) -> None:
        """Opening with a different key should fail to decrypt."""
        from cryptography.fernet import Fernet, InvalidToken

        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        s1 = ResultStore(db_dir, encryption_key=key1)
        s1.store_result("sec_test", sample_predictions, {})
        s1.close()

        s2 = ResultStore(db_dir, encryption_key=key2)
        with pytest.raises(InvalidToken):
            s2.get_result("sec_test")
        s2.close()


# ===================================================================
# 6. get_result
# ===================================================================


class TestGetResult:
    """Retrieving results."""

    def test_returns_none_for_missing(self, store: ResultStore) -> None:
        assert store.get_result("nonexistent") is None

    def test_returns_stored_result_type(self, store: ResultStore) -> None:
        store.store_result("h1", {"a": 1}, {"q": 1})
        result = store.get_result("h1")
        assert isinstance(result, StoredResult)

    def test_get_by_id(self, store: ResultStore) -> None:
        rid = store.store_result("h1", {"val": 42}, {"q": 1})
        result = store.get_result_by_id(rid)
        assert result is not None
        assert result.predictions["val"] == 42

    def test_get_by_id_missing(self, store: ResultStore) -> None:
        assert store.get_result_by_id(9999) is None

    def test_returns_newest_for_hash(self, store: ResultStore) -> None:
        store.store_result("h1", {"v": "old"}, {}, timestamp=100.0)
        store.store_result("h1", {"v": "new"}, {}, timestamp=200.0)
        result = store.get_result("h1")
        assert result is not None
        assert result.predictions["v"] == "new"


# ===================================================================
# 7. list_results
# ===================================================================


class TestListResults:
    """Listing and pagination."""

    def test_empty_store(self, store: ResultStore) -> None:
        assert store.list_results() == []

    def test_returns_all(self, store: ResultStore) -> None:
        for i in range(5):
            store.store_result(f"h{i}", {"i": i}, {})
        results = store.list_results()
        assert len(results) == 5

    def test_newest_first(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {}, timestamp=100.0)
        store.store_result("h2", {}, {}, timestamp=200.0)
        store.store_result("h3", {}, {}, timestamp=300.0)
        results = store.list_results()
        assert results[0].ecg_hash == "h3"
        assert results[-1].ecg_hash == "h1"

    def test_limit(self, store: ResultStore) -> None:
        for i in range(10):
            store.store_result(f"h{i}", {}, {})
        results = store.list_results(limit=3)
        assert len(results) == 3

    def test_offset(self, store: ResultStore) -> None:
        for i in range(5):
            store.store_result(f"h{i}", {"i": i}, {}, timestamp=float(i))
        results = store.list_results(limit=2, offset=2)
        assert len(results) == 2
        # Newest first: h4, h3, [h2, h1], h0 — offset=2 gives h2, h1
        assert results[0].predictions["i"] == 2
        assert results[1].predictions["i"] == 1

    def test_filter_synced_true(self, store: ResultStore) -> None:
        id1 = store.store_result("h1", {}, {})
        store.store_result("h2", {}, {})
        store.mark_synced(id1)

        synced = store.list_results(synced=True)
        assert len(synced) == 1
        assert synced[0].ecg_hash == "h1"

    def test_filter_synced_false(self, store: ResultStore) -> None:
        id1 = store.store_result("h1", {}, {})
        store.store_result("h2", {}, {})
        store.mark_synced(id1)

        unsynced = store.list_results(synced=False)
        assert len(unsynced) == 1
        assert unsynced[0].ecg_hash == "h2"


# ===================================================================
# 8. delete_result
# ===================================================================


class TestDeleteResult:
    """Deleting results."""

    def test_delete_by_hash(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        deleted = store.delete_result("h1")
        assert deleted == 1
        assert store.get_result("h1") is None

    def test_delete_nonexistent_hash(self, store: ResultStore) -> None:
        deleted = store.delete_result("nonexistent")
        assert deleted == 0

    def test_delete_multiple_same_hash(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        store.store_result("h1", {}, {})
        deleted = store.delete_result("h1")
        assert deleted == 2

    def test_delete_by_id(self, store: ResultStore) -> None:
        rid = store.store_result("h1", {}, {})
        success = store.delete_result_by_id(rid)
        assert success is True
        assert store.get_result_by_id(rid) is None

    def test_delete_by_id_nonexistent(self, store: ResultStore) -> None:
        success = store.delete_result_by_id(9999)
        assert success is False


# ===================================================================
# 9. mark_synced
# ===================================================================


class TestMarkSynced:
    """Sync status management."""

    def test_mark_synced(self, store: ResultStore) -> None:
        rid = store.store_result("h1", {}, {})
        assert store.mark_synced(rid) is True
        result = store.get_result_by_id(rid)
        assert result is not None
        assert result.synced is True

    def test_mark_synced_nonexistent(self, store: ResultStore) -> None:
        assert store.mark_synced(9999) is False


# ===================================================================
# 10. count
# ===================================================================


class TestCount:
    """Result counting."""

    def test_empty(self, store: ResultStore) -> None:
        assert store.count() == 0

    def test_after_inserts(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        store.store_result("h2", {}, {})
        assert store.count() == 2

    def test_count_synced(self, store: ResultStore) -> None:
        id1 = store.store_result("h1", {}, {})
        store.store_result("h2", {}, {})
        store.mark_synced(id1)
        assert store.count(synced=True) == 1
        assert store.count(synced=False) == 1

    def test_count_after_delete(self, store: ResultStore) -> None:
        store.store_result("h1", {}, {})
        store.store_result("h2", {}, {})
        store.delete_result("h1")
        assert store.count() == 1


# ===================================================================
# 11. Imports
# ===================================================================


class TestImports:
    """Package-level imports."""

    def test_import_from_sync_package(self) -> None:
        from aortica.sync import ResultStore as RS

        assert RS is ResultStore

    def test_import_stored_result(self) -> None:
        from aortica.sync.result_store import StoredResult

        assert StoredResult is not None
