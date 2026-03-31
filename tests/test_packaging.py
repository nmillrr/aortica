"""Tests for US-036: PyPI Entry Points and Package Metadata Update.

Verifies:
- Version is 0.2.0 across both pyproject.toml and aortica.__version__
- Console script entry points are correctly defined
- Optional dependency groups exist (api, grpc, cli, edge)
- run_server function exists and is importable
- CLI main entry point exists and is importable
"""

from __future__ import annotations

import configparser
import pathlib
import re
from typing import Any, Dict, Set

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _read_pyproject_raw() -> str:
    """Read pyproject.toml as raw text."""
    return PYPROJECT.read_text(encoding="utf-8")


def _parse_pyproject_version() -> str:
    """Extract version from pyproject.toml via regex (avoids toml dep)."""
    text = _read_pyproject_raw()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "Could not find version in pyproject.toml"
    return match.group(1)


def _parse_pyproject_scripts() -> Dict[str, str]:
    """Extract [project.scripts] entries from pyproject.toml via regex."""
    text = _read_pyproject_raw()
    scripts: Dict[str, str] = {}
    in_scripts = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts:
            if stripped.startswith("["):
                break
            if "=" in stripped and not stripped.startswith("#"):
                key, val = stripped.split("=", 1)
                scripts[key.strip()] = val.strip().strip('"')
    return scripts


def _parse_optional_dep_groups() -> Set[str]:
    """Extract optional dependency group names from pyproject.toml."""
    text = _read_pyproject_raw()
    groups: Set[str] = set()
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project.optional-dependencies]":
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("[") and not stripped.startswith("[["):
                break
            # Lines like `api = [` define a group
            if "=" in stripped and "[" in stripped:
                key = stripped.split("=", 1)[0].strip()
                groups.add(key)
    return groups


# ---------------------------------------------------------------------------
# Version Tests
# ---------------------------------------------------------------------------


class TestVersion:
    """Version bump to 0.2.0."""

    def test_pyproject_version_is_0_2_0(self) -> None:
        assert _parse_pyproject_version() == "0.2.0"

    def test_package_version_matches(self) -> None:
        import aortica

        assert aortica.__version__ == "0.2.0"

    def test_versions_in_sync(self) -> None:
        import aortica

        assert aortica.__version__ == _parse_pyproject_version()


# ---------------------------------------------------------------------------
# Entry Point Tests
# ---------------------------------------------------------------------------


class TestEntryPoints:
    """Console script entry points."""

    def test_aortica_script_defined(self) -> None:
        scripts = _parse_pyproject_scripts()
        assert "aortica" in scripts

    def test_aortica_script_target(self) -> None:
        scripts = _parse_pyproject_scripts()
        assert scripts["aortica"] == "aortica.cli:main"

    def test_aortica_server_script_defined(self) -> None:
        scripts = _parse_pyproject_scripts()
        assert "aortica-server" in scripts

    def test_aortica_server_script_target(self) -> None:
        scripts = _parse_pyproject_scripts()
        assert scripts["aortica-server"] == "aortica.api:run_server"

    def test_cli_main_importable(self) -> None:
        from aortica.cli import main

        assert callable(main)

    def test_run_server_importable(self) -> None:
        from aortica.api import run_server

        assert callable(run_server)

    def test_run_server_signature(self) -> None:
        """run_server accepts host, port, reload keyword arguments."""
        import inspect

        from aortica.api import run_server

        sig = inspect.signature(run_server)
        params = set(sig.parameters.keys())
        assert "host" in params
        assert "port" in params
        assert "reload" in params

    def test_run_server_defaults(self) -> None:
        """run_server has sensible defaults for host, port, reload."""
        import inspect

        from aortica.api import run_server

        sig = inspect.signature(run_server)
        assert sig.parameters["host"].default == "0.0.0.0"
        assert sig.parameters["port"].default == 8000
        assert sig.parameters["reload"].default is False


# ---------------------------------------------------------------------------
# Optional Dependency Groups
# ---------------------------------------------------------------------------


class TestOptionalDependencies:
    """New optional dependency groups for modular installation."""

    def test_api_group_exists(self) -> None:
        assert "api" in _parse_optional_dep_groups()

    def test_grpc_group_exists(self) -> None:
        assert "grpc" in _parse_optional_dep_groups()

    def test_cli_group_exists(self) -> None:
        assert "cli" in _parse_optional_dep_groups()

    def test_edge_group_exists(self) -> None:
        assert "edge" in _parse_optional_dep_groups()

    def test_dev_group_still_exists(self) -> None:
        assert "dev" in _parse_optional_dep_groups()

    def test_torch_group_still_exists(self) -> None:
        assert "torch" in _parse_optional_dep_groups()

    def test_tf_group_still_exists(self) -> None:
        assert "tf" in _parse_optional_dep_groups()

    def test_signal_group_still_exists(self) -> None:
        assert "signal" in _parse_optional_dep_groups()

    def test_api_contains_fastapi(self) -> None:
        text = _read_pyproject_raw()
        assert "fastapi" in text

    def test_api_contains_uvicorn(self) -> None:
        text = _read_pyproject_raw()
        assert "uvicorn" in text

    def test_api_contains_python_multipart(self) -> None:
        text = _read_pyproject_raw()
        assert "python-multipart" in text

    def test_cli_contains_click(self) -> None:
        text = _read_pyproject_raw()
        assert "click" in text

    def test_cli_contains_rich(self) -> None:
        text = _read_pyproject_raw()
        assert "rich" in text

    def test_grpc_contains_grpcio(self) -> None:
        text = _read_pyproject_raw()
        assert "grpcio" in text

    def test_edge_contains_onnx(self) -> None:
        text = _read_pyproject_raw()
        assert "onnx" in text

    def test_edge_contains_onnxruntime(self) -> None:
        text = _read_pyproject_raw()
        assert "onnxruntime" in text


# ---------------------------------------------------------------------------
# Installation Smoke Tests
# ---------------------------------------------------------------------------


class TestInstallation:
    """Verify the package installs correctly."""

    def test_package_importable(self) -> None:
        import aortica

        assert hasattr(aortica, "__version__")

    def test_api_subpackage_importable(self) -> None:
        from aortica.api import create_app

        assert callable(create_app)

    def test_cli_subpackage_importable(self) -> None:
        from aortica.cli import main

        assert callable(main)

    def test_create_app_export(self) -> None:
        import aortica.api

        assert "create_app" in aortica.api.__all__

    def test_run_server_export(self) -> None:
        import aortica.api

        assert "run_server" in aortica.api.__all__

    def test_cli_main_export(self) -> None:
        import aortica.cli

        assert "main" in aortica.cli.__all__
