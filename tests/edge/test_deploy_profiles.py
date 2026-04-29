"""Tests for aortica.edge.deploy_profiles module.

Covers RaspberryPiProfile dataclass construction, validation, serialisation,
systemd service generation, shell script generation, file writers, and
edge package imports.
"""

from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from aortica.edge.deploy_profiles import (
    DEFAULT_EDGE_MODEL_NAME,
    DEFAULT_MODEL_DIR,
    HARDWARE_TDP,
    SUPPORTED_QUANTIZATIONS,
    RaspberryPiProfile,
    generate_pi_image_script,
    generate_systemd_service,
    write_pi_image_script,
    write_systemd_service,
)


# =====================================================================
# Constants
# =====================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_default_model_dir(self) -> None:
        assert DEFAULT_MODEL_DIR == "~/.cache/aortica"

    def test_default_edge_model_name(self) -> None:
        assert DEFAULT_EDGE_MODEL_NAME == "aortica_edge_int8.onnx"

    def test_supported_quantizations_contains_int8(self) -> None:
        assert "INT8" in SUPPORTED_QUANTIZATIONS

    def test_supported_quantizations_contains_fp16(self) -> None:
        assert "FP16" in SUPPORTED_QUANTIZATIONS

    def test_supported_quantizations_contains_fp32(self) -> None:
        assert "FP32" in SUPPORTED_QUANTIZATIONS

    def test_hardware_tdp_rpi4(self) -> None:
        assert HARDWARE_TDP["rpi4"] == 4.0

    def test_hardware_tdp_rpi5(self) -> None:
        assert HARDWARE_TDP["rpi5"] == 6.0


# =====================================================================
# RaspberryPiProfile — Construction & Defaults
# =====================================================================


class TestProfileDefaults:
    """Default construction of RaspberryPiProfile."""

    def test_default_quantization(self) -> None:
        p = RaspberryPiProfile()
        assert p.quantization == "INT8"

    def test_default_max_memory_mb(self) -> None:
        p = RaspberryPiProfile()
        assert p.max_memory_mb == 512

    def test_default_target_latency_ms(self) -> None:
        p = RaspberryPiProfile()
        assert p.target_latency_ms == 350

    def test_default_device_name(self) -> None:
        p = RaspberryPiProfile()
        assert p.device_name == "raspberry_pi_4"

    def test_default_num_threads(self) -> None:
        p = RaspberryPiProfile()
        assert p.num_threads == 4

    def test_default_enable_sync(self) -> None:
        p = RaspberryPiProfile()
        assert p.enable_sync is True

    def test_default_sync_interval(self) -> None:
        p = RaspberryPiProfile()
        assert p.sync_interval_minutes == 30

    def test_default_data_dir(self) -> None:
        p = RaspberryPiProfile()
        assert p.data_dir == "/var/lib/aortica/data"

    def test_default_log_dir(self) -> None:
        p = RaspberryPiProfile()
        assert p.log_dir == "/var/log/aortica"

    def test_default_service_user(self) -> None:
        p = RaspberryPiProfile()
        assert p.service_user == "aortica"

    def test_default_model_path_contains_edge_model(self) -> None:
        p = RaspberryPiProfile()
        assert DEFAULT_EDGE_MODEL_NAME in p.model_path

    def test_default_watchdog_timeout(self) -> None:
        p = RaspberryPiProfile()
        assert p.watchdog_timeout_sec == 60


class TestProfileCustom:
    """Custom construction of RaspberryPiProfile."""

    def test_custom_quantization_fp16(self) -> None:
        p = RaspberryPiProfile(quantization="FP16")
        assert p.quantization == "FP16"

    def test_custom_max_memory(self) -> None:
        p = RaspberryPiProfile(max_memory_mb=1024)
        assert p.max_memory_mb == 1024

    def test_custom_target_latency(self) -> None:
        p = RaspberryPiProfile(target_latency_ms=200)
        assert p.target_latency_ms == 200

    def test_custom_device_name(self) -> None:
        p = RaspberryPiProfile(device_name="rpi5")
        assert p.device_name == "rpi5"

    def test_custom_model_path(self) -> None:
        p = RaspberryPiProfile(model_path="/opt/models/custom.onnx")
        assert p.model_path == "/opt/models/custom.onnx"

    def test_custom_num_threads(self) -> None:
        p = RaspberryPiProfile(num_threads=2)
        assert p.num_threads == 2


# =====================================================================
# Validation
# =====================================================================


class TestProfileValidation:
    """Validation errors on invalid field values."""

    def test_invalid_quantization_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization"):
            RaspberryPiProfile(quantization="INT4")

    def test_zero_max_memory_raises(self) -> None:
        with pytest.raises(ValueError, match="max_memory_mb"):
            RaspberryPiProfile(max_memory_mb=0)

    def test_negative_max_memory_raises(self) -> None:
        with pytest.raises(ValueError, match="max_memory_mb"):
            RaspberryPiProfile(max_memory_mb=-100)

    def test_zero_target_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="target_latency_ms"):
            RaspberryPiProfile(target_latency_ms=0)

    def test_negative_target_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="target_latency_ms"):
            RaspberryPiProfile(target_latency_ms=-50)

    def test_zero_num_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="num_threads"):
            RaspberryPiProfile(num_threads=0)

    def test_zero_sync_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="sync_interval_minutes"):
            RaspberryPiProfile(sync_interval_minutes=0)

    def test_zero_watchdog_raises(self) -> None:
        with pytest.raises(ValueError, match="watchdog_timeout_sec"):
            RaspberryPiProfile(watchdog_timeout_sec=0)


# =====================================================================
# Serialisation
# =====================================================================


class TestProfileSerialisation:
    """to_dict / from_dict / to_json / from_json round-trips."""

    def test_to_dict_returns_dict(self) -> None:
        p = RaspberryPiProfile()
        d = p.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_includes_all_fields(self) -> None:
        p = RaspberryPiProfile()
        d = p.to_dict()
        for field_name in RaspberryPiProfile.__dataclass_fields__:
            assert field_name in d

    def test_from_dict_roundtrip(self) -> None:
        p = RaspberryPiProfile(max_memory_mb=256, target_latency_ms=200)
        d = p.to_dict()
        p2 = RaspberryPiProfile.from_dict(d)
        assert p == p2

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = RaspberryPiProfile().to_dict()
        d["unknown_extra_key"] = "should be ignored"
        p = RaspberryPiProfile.from_dict(d)
        assert p.quantization == "INT8"

    def test_to_json_roundtrip(self, tmp_path: Path) -> None:
        p = RaspberryPiProfile(quantization="FP16", num_threads=2)
        json_path = tmp_path / "profile.json"
        p.to_json(json_path)

        p2 = RaspberryPiProfile.from_json(json_path)
        assert p == p2

    def test_json_file_is_valid_json(self, tmp_path: Path) -> None:
        p = RaspberryPiProfile()
        json_path = tmp_path / "profile.json"
        p.to_json(json_path)

        with open(json_path) as fh:
            data = json.load(fh)
        assert data["quantization"] == "INT8"

    def test_json_file_is_indented(self, tmp_path: Path) -> None:
        p = RaspberryPiProfile()
        json_path = tmp_path / "profile.json"
        p.to_json(json_path)

        content = json_path.read_text()
        assert "\n  " in content  # indented with 2 spaces


# =====================================================================
# Helpers
# =====================================================================


class TestProfileHelpers:
    """Helper methods on RaspberryPiProfile."""

    def test_resolved_model_path_expands_tilde(self) -> None:
        p = RaspberryPiProfile()
        resolved = p.resolved_model_path()
        assert "~" not in resolved
        assert os.path.expanduser("~") in resolved

    def test_resolved_model_path_absolute_unchanged(self) -> None:
        p = RaspberryPiProfile(model_path="/opt/models/custom.onnx")
        assert p.resolved_model_path() == "/opt/models/custom.onnx"

    def test_tdp_watts_rpi4(self) -> None:
        p = RaspberryPiProfile(device_name="rpi4")
        assert p.tdp_watts() == 4.0

    def test_tdp_watts_rpi5(self) -> None:
        p = RaspberryPiProfile(device_name="rpi5")
        assert p.tdp_watts() == 6.0

    def test_tdp_watts_unknown_device_fallback(self) -> None:
        p = RaspberryPiProfile(device_name="unknown_board")
        assert p.tdp_watts() == 4.0

    def test_onnxruntime_session_options_type(self) -> None:
        p = RaspberryPiProfile()
        opts = p.onnxruntime_session_options()
        assert isinstance(opts, dict)

    def test_onnxruntime_session_options_threads(self) -> None:
        p = RaspberryPiProfile(num_threads=2)
        opts = p.onnxruntime_session_options()
        assert opts["intra_op_num_threads"] == 2
        assert opts["inter_op_num_threads"] == 1

    def test_onnxruntime_session_options_mem_arena(self) -> None:
        p = RaspberryPiProfile(max_memory_mb=512)
        assert p.onnxruntime_session_options()["enable_cpu_mem_arena"] is True

        p2 = RaspberryPiProfile(max_memory_mb=128)
        assert p2.onnxruntime_session_options()["enable_cpu_mem_arena"] is False

    def test_summary_returns_string(self) -> None:
        p = RaspberryPiProfile()
        s = p.summary()
        assert isinstance(s, str)

    def test_summary_contains_device_name(self) -> None:
        p = RaspberryPiProfile()
        assert "raspberry_pi_4" in p.summary()

    def test_summary_contains_quantization(self) -> None:
        p = RaspberryPiProfile()
        assert "INT8" in p.summary()

    def test_summary_contains_memory(self) -> None:
        p = RaspberryPiProfile()
        assert "512" in p.summary()


# =====================================================================
# Systemd service generation
# =====================================================================


class TestSystemdService:
    """Tests for generate_systemd_service()."""

    def test_returns_string(self) -> None:
        content = generate_systemd_service()
        assert isinstance(content, str)

    def test_contains_unit_section(self) -> None:
        content = generate_systemd_service()
        assert "[Unit]" in content

    def test_contains_service_section(self) -> None:
        content = generate_systemd_service()
        assert "[Service]" in content

    def test_contains_install_section(self) -> None:
        content = generate_systemd_service()
        assert "[Install]" in content

    def test_contains_exec_start(self) -> None:
        content = generate_systemd_service()
        assert "ExecStart=" in content

    def test_contains_user(self) -> None:
        content = generate_systemd_service()
        assert "User=aortica" in content

    def test_contains_memory_limit(self) -> None:
        content = generate_systemd_service()
        assert "MemoryMax=512M" in content

    def test_custom_profile_memory(self) -> None:
        p = RaspberryPiProfile(max_memory_mb=1024)
        content = generate_systemd_service(p)
        assert "MemoryMax=1024M" in content

    def test_custom_profile_user(self) -> None:
        p = RaspberryPiProfile(service_user="ecg_svc")
        content = generate_systemd_service(p)
        assert "User=ecg_svc" in content

    def test_contains_security_hardening(self) -> None:
        content = generate_systemd_service()
        assert "ProtectSystem=strict" in content
        assert "NoNewPrivileges=yes" in content
        assert "PrivateTmp=yes" in content

    def test_contains_watchdog(self) -> None:
        content = generate_systemd_service()
        assert "WatchdogSec=60" in content

    def test_custom_watchdog(self) -> None:
        p = RaspberryPiProfile(watchdog_timeout_sec=120)
        content = generate_systemd_service(p)
        assert "WatchdogSec=120" in content

    def test_contains_restart_policy(self) -> None:
        content = generate_systemd_service()
        assert "Restart=on-failure" in content

    def test_contains_num_threads_env(self) -> None:
        p = RaspberryPiProfile(num_threads=2)
        content = generate_systemd_service(p)
        assert "ORT_NUM_THREADS=2" in content


# =====================================================================
# Pi image script generation
# =====================================================================


class TestPiImageScript:
    """Tests for generate_pi_image_script()."""

    def test_returns_string(self) -> None:
        content = generate_pi_image_script()
        assert isinstance(content, str)

    def test_starts_with_shebang(self) -> None:
        content = generate_pi_image_script()
        assert content.startswith("#!/usr/bin/env bash")

    def test_contains_set_euo_pipefail(self) -> None:
        content = generate_pi_image_script()
        assert "set -euo pipefail" in content

    def test_contains_apt_install(self) -> None:
        content = generate_pi_image_script()
        assert "apt-get install" in content
        assert "python3" in content

    def test_contains_venv_creation(self) -> None:
        content = generate_pi_image_script()
        assert "python3 -m venv" in content

    def test_contains_pip_install_aortica(self) -> None:
        content = generate_pi_image_script()
        assert "aortica[cli,edge]" in content

    def test_contains_model_download(self) -> None:
        content = generate_pi_image_script()
        assert "load_pretrained" in content

    def test_contains_systemd_enable(self) -> None:
        content = generate_pi_image_script()
        assert "systemctl enable" in content

    def test_contains_mkdir_data_dir(self) -> None:
        content = generate_pi_image_script()
        assert "mkdir" in content
        assert "/var/lib/aortica/data" in content

    def test_contains_useradd(self) -> None:
        content = generate_pi_image_script()
        assert "useradd" in content

    def test_custom_profile_device(self) -> None:
        p = RaspberryPiProfile(device_name="rpi5")
        content = generate_pi_image_script(p)
        assert "rpi5" in content


# =====================================================================
# File writers
# =====================================================================


class TestFileWriters:
    """Tests for write_systemd_service() and write_pi_image_script()."""

    def test_write_systemd_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "aortica-edge.service"
        result = write_systemd_service(out)
        assert result.exists()

    def test_write_systemd_content(self, tmp_path: Path) -> None:
        out = tmp_path / "aortica-edge.service"
        write_systemd_service(out)
        content = out.read_text()
        assert "[Unit]" in content

    def test_write_systemd_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "svc.service"
        result = write_systemd_service(out)
        assert result.exists()

    def test_write_pi_script_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "setup.sh"
        result = write_pi_image_script(out)
        assert result.exists()

    def test_write_pi_script_is_executable(self, tmp_path: Path) -> None:
        out = tmp_path / "setup.sh"
        write_pi_image_script(out)
        mode = out.stat().st_mode
        assert mode & stat.S_IXUSR  # owner execute bit

    def test_write_pi_script_content(self, tmp_path: Path) -> None:
        out = tmp_path / "setup.sh"
        write_pi_image_script(out)
        content = out.read_text()
        assert content.startswith("#!/usr/bin/env bash")

    def test_write_pi_script_with_custom_profile(self, tmp_path: Path) -> None:
        p = RaspberryPiProfile(max_memory_mb=1024, device_name="rpi5")
        out = tmp_path / "setup.sh"
        write_pi_image_script(out, profile=p)
        content = out.read_text()
        assert "rpi5" in content


# =====================================================================
# Documentation requirements
# =====================================================================


class TestDocumentation:
    """Verify documentation artefacts exist."""

    def test_rpi_docs_exist(self) -> None:
        docs_path = Path(__file__).parent.parent.parent / "docs" / "deployment" / "raspberry-pi.md"
        assert docs_path.exists(), f"RPi deployment docs missing at {docs_path}"

    def test_rpi_docs_contain_hardware(self) -> None:
        docs_path = Path(__file__).parent.parent.parent / "docs" / "deployment" / "raspberry-pi.md"
        content = docs_path.read_text()
        assert "Hardware" in content

    def test_rpi_docs_contain_systemd(self) -> None:
        docs_path = Path(__file__).parent.parent.parent / "docs" / "deployment" / "raspberry-pi.md"
        content = docs_path.read_text()
        assert "systemd" in content or "systemctl" in content

    def test_rpi_docs_contain_first_run(self) -> None:
        docs_path = Path(__file__).parent.parent.parent / "docs" / "deployment" / "raspberry-pi.md"
        content = docs_path.read_text()
        assert "aortica predict" in content


# =====================================================================
# Edge package imports
# =====================================================================


class TestImports:
    """Verify deploy_profiles is accessible from edge package."""

    def test_import_module(self) -> None:
        import aortica.edge.deploy_profiles  # noqa: F401

    def test_import_profile_class(self) -> None:
        from aortica.edge.deploy_profiles import RaspberryPiProfile  # noqa: F401

    def test_import_from_edge_package(self) -> None:
        from aortica.edge import RaspberryPiProfile  # noqa: F401

    def test_import_generators(self) -> None:
        from aortica.edge import (  # noqa: F401
            generate_pi_image_script,
            generate_systemd_service,
        )

    def test_import_writers(self) -> None:
        from aortica.edge import (  # noqa: F401
            write_pi_image_script,
            write_systemd_service,
        )
