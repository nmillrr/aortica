"""Tests for aortica.sync.config — SyncConfig, connectivity, scheduler, anonymisation."""

from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aortica.sync.config import (
    AutoSyncScheduler,
    ConnectivityStatus,
    SyncConfig,
    _DEFAULT_SENSITIVE_KEYS,
    _coerce_value,
    _simple_yaml_load,
    anonymise_result,
    check_connectivity,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture()
def tmp_yaml(tmp_path: Path) -> Path:
    """Return a path to a temporary YAML file location."""
    return tmp_path / "sync_config.yaml"


@pytest.fixture()
def sample_config() -> SyncConfig:
    """Return a non-default SyncConfig."""
    return SyncConfig(
        sync_interval_minutes=15,
        min_bandwidth_kbps=512,
        max_batch_size=10,
        remote_url="https://central.example.com/api/v1/sync",
        device_id="device-001",
        sensitive_keys=["custom_field"],
    )


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler returning 200 with a fixed body."""

    response_body: bytes = b'{"status":"ok"}'

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # Suppress stderr output


class _ErrorHandler(http.server.BaseHTTPRequestHandler):
    """Handler that returns 500."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(500)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


# -------------------------------------------------------------------------
# SyncConfig — defaults and construction
# -------------------------------------------------------------------------


class TestSyncConfigDefaults:
    """Tests for SyncConfig default values."""

    def test_default_interval(self) -> None:
        cfg = SyncConfig()
        assert cfg.sync_interval_minutes == 30

    def test_default_bandwidth(self) -> None:
        cfg = SyncConfig()
        assert cfg.min_bandwidth_kbps == 256

    def test_default_batch_size(self) -> None:
        cfg = SyncConfig()
        assert cfg.max_batch_size == 20

    def test_default_remote_url(self) -> None:
        cfg = SyncConfig()
        assert cfg.remote_url == ""

    def test_default_device_id(self) -> None:
        cfg = SyncConfig()
        assert cfg.device_id == ""

    def test_default_sensitive_keys(self) -> None:
        cfg = SyncConfig()
        assert cfg.sensitive_keys == []

    def test_custom_values(self, sample_config: SyncConfig) -> None:
        assert sample_config.sync_interval_minutes == 15
        assert sample_config.min_bandwidth_kbps == 512
        assert sample_config.max_batch_size == 10
        assert sample_config.remote_url == "https://central.example.com/api/v1/sync"
        assert sample_config.device_id == "device-001"
        assert sample_config.sensitive_keys == ["custom_field"]


# -------------------------------------------------------------------------
# SyncConfig — serialisation
# -------------------------------------------------------------------------


class TestSyncConfigSerialisation:
    """Tests for to_dict / from_dict round-trip."""

    def test_to_dict_keys(self, sample_config: SyncConfig) -> None:
        d = sample_config.to_dict()
        assert "sync_interval_minutes" in d
        assert "min_bandwidth_kbps" in d
        assert "max_batch_size" in d
        assert "remote_url" in d
        assert "device_id" in d
        assert "sensitive_keys" in d

    def test_from_dict_roundtrip(self, sample_config: SyncConfig) -> None:
        d = sample_config.to_dict()
        restored = SyncConfig.from_dict(d)
        assert restored.sync_interval_minutes == sample_config.sync_interval_minutes
        assert restored.min_bandwidth_kbps == sample_config.min_bandwidth_kbps
        assert restored.remote_url == sample_config.remote_url

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"sync_interval_minutes": 5, "unknown_key": "ignored"}
        cfg = SyncConfig.from_dict(d)
        assert cfg.sync_interval_minutes == 5

    def test_from_dict_partial(self) -> None:
        cfg = SyncConfig.from_dict({"device_id": "dev-42"})
        assert cfg.device_id == "dev-42"
        assert cfg.sync_interval_minutes == 30  # default


# -------------------------------------------------------------------------
# SyncConfig — YAML I/O
# -------------------------------------------------------------------------


class TestSyncConfigYAML:
    """Tests for YAML file loading and saving."""

    def test_to_yaml_creates_file(
        self, sample_config: SyncConfig, tmp_yaml: Path
    ) -> None:
        sample_config.to_yaml(tmp_yaml)
        assert tmp_yaml.exists()

    def test_from_yaml_roundtrip(
        self, sample_config: SyncConfig, tmp_yaml: Path
    ) -> None:
        sample_config.to_yaml(tmp_yaml)
        restored = SyncConfig.from_yaml(tmp_yaml)
        assert restored.sync_interval_minutes == sample_config.sync_interval_minutes
        assert restored.min_bandwidth_kbps == sample_config.min_bandwidth_kbps
        assert restored.remote_url == sample_config.remote_url
        assert restored.device_id == sample_config.device_id
        assert restored.max_batch_size == sample_config.max_batch_size

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SyncConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_minimal_file(self, tmp_path: Path) -> None:
        p = tmp_path / "min.yaml"
        p.write_text("sync_interval_minutes: 10\ndevice_id: abc\n")
        cfg = SyncConfig.from_yaml(p)
        assert cfg.sync_interval_minutes == 10
        assert cfg.device_id == "abc"

    def test_from_yaml_with_list(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text(
            "device_id: x\nsensitive_keys:\n  - field_a\n  - field_b\n"
        )
        cfg = SyncConfig.from_yaml(p)
        assert cfg.sensitive_keys == ["field_a", "field_b"]

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = SyncConfig.from_yaml(p)
        assert cfg.sync_interval_minutes == 30  # all defaults


# -------------------------------------------------------------------------
# Simple YAML fallback parser
# -------------------------------------------------------------------------


class TestSimpleYAMLParser:
    """Tests for the fallback YAML parser."""

    def test_coerce_int(self) -> None:
        assert _coerce_value("42") == 42

    def test_coerce_float(self) -> None:
        assert _coerce_value("3.14") == 3.14

    def test_coerce_bool_true(self) -> None:
        assert _coerce_value("true") is True

    def test_coerce_bool_false(self) -> None:
        assert _coerce_value("false") is False

    def test_coerce_string(self) -> None:
        assert _coerce_value("hello") == "hello"

    def test_coerce_quoted_string(self) -> None:
        assert _coerce_value('"hello world"') == "hello world"

    def test_simple_load_flat(self, tmp_path: Path) -> None:
        p = tmp_path / "flat.yaml"
        p.write_text("key1: 100\nkey2: hello\n")
        data = _simple_yaml_load(p)
        assert data["key1"] == 100
        assert data["key2"] == "hello"

    def test_simple_load_comments(self, tmp_path: Path) -> None:
        p = tmp_path / "comments.yaml"
        p.write_text("# Comment\nkey: value\n")
        data = _simple_yaml_load(p)
        assert data["key"] == "value"

    def test_simple_load_list(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("items:\n  - one\n  - two\n  - three\n")
        data = _simple_yaml_load(p)
        assert data["items"] == ["one", "two", "three"]


# -------------------------------------------------------------------------
# ConnectivityStatus
# -------------------------------------------------------------------------


class TestConnectivityStatus:
    """Tests for the ConnectivityStatus dataclass."""

    def test_defaults(self) -> None:
        s = ConnectivityStatus(available=True)
        assert s.available is True
        assert s.latency_ms == 0.0
        assert s.bandwidth_kbps == 0.0
        assert s.error == ""

    def test_custom(self) -> None:
        s = ConnectivityStatus(
            available=True, latency_ms=50.0, bandwidth_kbps=1024.0
        )
        assert s.latency_ms == 50.0
        assert s.bandwidth_kbps == 1024.0


# -------------------------------------------------------------------------
# check_connectivity
# -------------------------------------------------------------------------


class TestCheckConnectivity:
    """Tests for check_connectivity function."""

    def test_reachable_server(self) -> None:
        server = http.server.HTTPServer(("127.0.0.1", 0), _HealthHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            status = check_connectivity(f"http://127.0.0.1:{port}/health")
            assert status.available is True
            assert status.latency_ms > 0
            assert status.bandwidth_kbps > 0
            assert status.error == ""
        finally:
            server.server_close()

    def test_unreachable_server(self) -> None:
        status = check_connectivity(
            "http://192.0.2.1:1/health", timeout=0.5
        )
        assert status.available is False
        assert status.error != ""

    def test_server_error_still_available(self) -> None:
        server = http.server.HTTPServer(("127.0.0.1", 0), _ErrorHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            status = check_connectivity(f"http://127.0.0.1:{port}/health")
            assert status.available is True
            assert "500" in status.error
        finally:
            server.server_close()


# -------------------------------------------------------------------------
# AutoSyncScheduler
# -------------------------------------------------------------------------


class TestAutoSyncScheduler:
    """Tests for the AutoSyncScheduler."""

    def test_not_running_initially(self) -> None:
        cfg = SyncConfig(remote_url="http://example.com")
        scheduler = AutoSyncScheduler(cfg, lambda url: None)
        assert scheduler.running is False

    def test_start_sets_running(self) -> None:
        cfg = SyncConfig(
            sync_interval_minutes=60,
            remote_url="http://example.com",
        )
        scheduler = AutoSyncScheduler(cfg, lambda url: None)
        scheduler.start()
        try:
            assert scheduler.running is True
        finally:
            scheduler.stop()

    def test_stop_clears_running(self) -> None:
        cfg = SyncConfig(
            sync_interval_minutes=60,
            remote_url="http://example.com",
        )
        scheduler = AutoSyncScheduler(cfg, lambda url: None)
        scheduler.start()
        scheduler.stop()
        assert scheduler.running is False

    def test_start_idempotent(self) -> None:
        cfg = SyncConfig(
            sync_interval_minutes=60,
            remote_url="http://example.com",
        )
        scheduler = AutoSyncScheduler(cfg, lambda url: None)
        scheduler.start()
        scheduler.start()  # should not raise
        try:
            assert scheduler.running is True
        finally:
            scheduler.stop()

    def test_tick_calls_sync_when_available(self) -> None:
        cfg = SyncConfig(
            remote_url="http://example.com",
            min_bandwidth_kbps=0,
        )
        sync_fn = MagicMock()
        scheduler = AutoSyncScheduler(cfg, sync_fn)

        with patch(
            "aortica.sync.config.check_connectivity",
            return_value=ConnectivityStatus(
                available=True, bandwidth_kbps=1000.0
            ),
        ):
            status = scheduler.tick()

        assert status is not None
        assert status.available is True
        sync_fn.assert_called_once_with(cfg.remote_url)
        assert scheduler.last_sync_time is not None

    def test_tick_skips_when_below_bandwidth(self) -> None:
        cfg = SyncConfig(
            remote_url="http://example.com",
            min_bandwidth_kbps=1000,
        )
        sync_fn = MagicMock()
        scheduler = AutoSyncScheduler(cfg, sync_fn)

        with patch(
            "aortica.sync.config.check_connectivity",
            return_value=ConnectivityStatus(
                available=True, bandwidth_kbps=100.0
            ),
        ):
            status = scheduler.tick()

        assert status is not None
        sync_fn.assert_not_called()

    def test_tick_skips_when_unavailable(self) -> None:
        cfg = SyncConfig(
            remote_url="http://example.com",
            min_bandwidth_kbps=0,
        )
        sync_fn = MagicMock()
        scheduler = AutoSyncScheduler(cfg, sync_fn)

        with patch(
            "aortica.sync.config.check_connectivity",
            return_value=ConnectivityStatus(available=False),
        ):
            status = scheduler.tick()

        sync_fn.assert_not_called()

    def test_tick_skips_when_no_remote_url(self) -> None:
        cfg = SyncConfig(remote_url="")
        sync_fn = MagicMock()
        scheduler = AutoSyncScheduler(cfg, sync_fn)

        status = scheduler.tick()
        assert status is None
        sync_fn.assert_not_called()

    def test_last_status_updated(self) -> None:
        cfg = SyncConfig(
            remote_url="http://example.com",
            min_bandwidth_kbps=0,
        )
        expected = ConnectivityStatus(available=True, bandwidth_kbps=999.0)
        scheduler = AutoSyncScheduler(cfg, lambda url: None)

        with patch(
            "aortica.sync.config.check_connectivity",
            return_value=expected,
        ):
            scheduler.tick()

        assert scheduler.last_status is not None
        assert scheduler.last_status.bandwidth_kbps == 999.0

    def test_tick_handles_sync_error(self) -> None:
        cfg = SyncConfig(
            remote_url="http://example.com",
            min_bandwidth_kbps=0,
        )

        def failing_sync(url: str) -> None:
            msg = "Network failure"
            raise ConnectionError(msg)

        scheduler = AutoSyncScheduler(cfg, failing_sync)

        with patch(
            "aortica.sync.config.check_connectivity",
            return_value=ConnectivityStatus(
                available=True, bandwidth_kbps=1000.0
            ),
        ):
            # Should not raise — errors are swallowed
            status = scheduler.tick()
            assert status is not None


# -------------------------------------------------------------------------
# Anonymisation
# -------------------------------------------------------------------------


class TestAnonymisation:
    """Tests for the anonymise_result function."""

    def test_strips_patient_name(self) -> None:
        data = {"metadata": {"patient_name": "John Doe", "device": "ECG-100"}}
        result = anonymise_result(data)
        assert "patient_name" not in result["metadata"]
        assert result["metadata"]["device"] == "ECG-100"

    def test_strips_multiple_sensitive_keys(self) -> None:
        data = {
            "metadata": {
                "patient_id": "P001",
                "dob": "1990-01-01",
                "mrn": "MRN-123",
                "device": "dev-1",
            }
        }
        result = anonymise_result(data)
        assert "patient_id" not in result["metadata"]
        assert "dob" not in result["metadata"]
        assert "mrn" not in result["metadata"]
        assert result["metadata"]["device"] == "dev-1"

    def test_case_insensitive(self) -> None:
        data = {"metadata": {"Patient_Name": "Jane", "DEVICE": "ECG"}}
        result = anonymise_result(data)
        assert "Patient_Name" not in result["metadata"]
        assert result["metadata"]["DEVICE"] == "ECG"

    def test_extra_sensitive_keys(self) -> None:
        data = {"metadata": {"custom_field": "secret", "safe": "ok"}}
        result = anonymise_result(data, extra_sensitive_keys=["custom_field"])
        assert "custom_field" not in result["metadata"]
        assert result["metadata"]["safe"] == "ok"

    def test_patient_metadata_key(self) -> None:
        data = {
            "patient_metadata": {"patient_name": "Test", "age": 65},
            "metadata": {},
        }
        result = anonymise_result(data)
        assert "patient_name" not in result["patient_metadata"]
        assert result["patient_metadata"]["age"] == 65

    def test_no_metadata_key(self) -> None:
        data = {"ecg_hash": "abc123", "predictions": {"rhythm": [0.5]}}
        result = anonymise_result(data)
        assert result == data

    def test_does_not_mutate_original(self) -> None:
        data = {"metadata": {"patient_name": "X", "device": "D"}}
        _ = anonymise_result(data)
        assert data["metadata"]["patient_name"] == "X"

    def test_preserves_non_metadata_fields(self) -> None:
        data = {
            "ecg_hash": "hash1",
            "predictions": {"af": 0.9},
            "metadata": {"patient_name": "X"},
        }
        result = anonymise_result(data)
        assert result["ecg_hash"] == "hash1"
        assert result["predictions"]["af"] == 0.9

    def test_empty_metadata(self) -> None:
        data = {"metadata": {}}
        result = anonymise_result(data)
        assert result["metadata"] == {}

    def test_all_default_sensitive_keys_exist(self) -> None:
        """Verify the default sensitive key set has expected members."""
        assert "patient_name" in _DEFAULT_SENSITIVE_KEYS
        assert "patient_id" in _DEFAULT_SENSITIVE_KEYS
        assert "date_of_birth" in _DEFAULT_SENSITIVE_KEYS
        assert "dob" in _DEFAULT_SENSITIVE_KEYS
        assert "ssn" in _DEFAULT_SENSITIVE_KEYS
        assert "email" in _DEFAULT_SENSITIVE_KEYS
        assert "phone" in _DEFAULT_SENSITIVE_KEYS
        assert "address" in _DEFAULT_SENSITIVE_KEYS


# -------------------------------------------------------------------------
# Module-level constants and imports
# -------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_default_sensitive_keys_is_frozenset(self) -> None:
        assert isinstance(_DEFAULT_SENSITIVE_KEYS, frozenset)

    def test_default_sensitive_keys_non_empty(self) -> None:
        assert len(_DEFAULT_SENSITIVE_KEYS) > 0


class TestImports:
    """Tests that public API is importable."""

    def test_import_sync_config(self) -> None:
        from aortica.sync.config import SyncConfig as _SyncConfig
        assert _SyncConfig is not None

    def test_import_check_connectivity(self) -> None:
        from aortica.sync.config import check_connectivity as _cc
        assert callable(_cc)

    def test_import_anonymise_result(self) -> None:
        from aortica.sync.config import anonymise_result as _ar
        assert callable(_ar)

    def test_import_auto_sync_scheduler(self) -> None:
        from aortica.sync.config import AutoSyncScheduler as _AS
        assert _AS is not None

    def test_import_from_package(self) -> None:
        from aortica.sync import SyncConfig as _SC
        assert _SC is not None
