"""Tests for ``aortica federated test-connection`` CLI command.

Covers the connection test helper function, URL parsing, and the Click
CLI command using CliRunner with mocked sockets to avoid real network calls.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

click = pytest.importorskip("click")
rich = pytest.importorskip("rich")

from click.testing import CliRunner  # noqa: E402

from aortica.cli import _build_cli  # noqa: E402
from aortica.cli.federated_cmd import (  # noqa: E402
    _parse_server_url,
    check_fl_connection,
    federated_group,
    test_connection_cmd,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Click CliRunner for isolated CLI testing."""
    return CliRunner(mix_stderr=False)


# ---------------------------------------------------------------------------
# Tests: _parse_server_url
# ---------------------------------------------------------------------------


class TestParseServerUrl:
    """URL parsing for host:port strings."""

    def test_valid_host_port(self) -> None:
        host, port = _parse_server_url("fl.example.com:8080")
        assert host == "fl.example.com"
        assert port == 8080

    def test_localhost(self) -> None:
        host, port = _parse_server_url("localhost:8080")
        assert host == "localhost"
        assert port == 8080

    def test_ip_address(self) -> None:
        host, port = _parse_server_url("192.168.1.100:9090")
        assert host == "192.168.1.100"
        assert port == 9090

    def test_ipv6_address(self) -> None:
        host, port = _parse_server_url("[::1]:8080")
        assert host == "::1"
        assert port == 8080

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid server address"):
            _parse_server_url("")

    def test_no_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid server address"):
            _parse_server_url("localhost")

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="Host cannot be empty"):
            _parse_server_url(":8080")

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid port"):
            _parse_server_url("localhost:notaport")

    def test_port_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid port 0"):
            _parse_server_url("localhost:0")

    def test_port_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid port 70000"):
            _parse_server_url("localhost:70000")

    def test_standard_port(self) -> None:
        host, port = _parse_server_url("server.local:443")
        assert host == "server.local"
        assert port == 443

    def test_port_1(self) -> None:
        host, port = _parse_server_url("host:1")
        assert host == "host"
        assert port == 1

    def test_port_65535(self) -> None:
        host, port = _parse_server_url("host:65535")
        assert host == "host"
        assert port == 65535

    def test_ipv6_missing_bracket_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid IPv6"):
            _parse_server_url("[::1:8080")

    def test_ipv6_missing_port_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid server address"):
            _parse_server_url("[::1]8080")


# ---------------------------------------------------------------------------
# Tests: check_fl_connection (mocked socket)
# ---------------------------------------------------------------------------


class TestFlConnection:
    """FL connection test with mocked sockets."""

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_successful_connection(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.return_value = None

        result = check_fl_connection("localhost:8080", timeout=5.0)

        assert result["success"] is True
        assert result["host"] == "localhost"
        assert result["port"] == 8080
        assert result["latency_ms"] is not None
        assert isinstance(result["latency_ms"], float)
        assert result["error"] is None
        mock_sock.settimeout.assert_called_once_with(5.0)
        mock_sock.connect.assert_called_once_with(("localhost", 8080))
        mock_sock.close.assert_called_once()

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_connection_refused(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = OSError("Connection refused")

        result = check_fl_connection("localhost:8080")

        assert result["success"] is False
        assert result["error"] == "Connection refused"
        assert result["latency_ms"] is None
        mock_sock.close.assert_called_once()

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_connection_timeout(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = socket.timeout("timed out")

        result = check_fl_connection("localhost:8080", timeout=3.0)

        assert result["success"] is False
        assert "timed out" in str(result["error"])
        assert result["latency_ms"] is None
        mock_sock.close.assert_called_once()

    def test_invalid_url_returns_error(self) -> None:
        result = check_fl_connection("badurl")

        assert result["success"] is False
        assert "Invalid server address" in str(result["error"])
        assert result["latency_ms"] is None

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_socket_closed_on_success(
        self, mock_socket_class: MagicMock
    ) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.return_value = None

        check_fl_connection("localhost:8080")
        mock_sock.close.assert_called_once()

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_socket_closed_on_error(
        self, mock_socket_class: MagicMock
    ) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = OSError("Network unreachable")

        check_fl_connection("localhost:8080")
        mock_sock.close.assert_called_once()

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_custom_timeout(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.return_value = None

        check_fl_connection("localhost:8080", timeout=30.0)
        mock_sock.settimeout.assert_called_once_with(30.0)

    @patch("aortica.cli.federated_cmd.socket.socket")
    def test_result_keys(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.return_value = None

        result = check_fl_connection("localhost:8080")

        assert set(result.keys()) == {
            "success",
            "host",
            "port",
            "latency_ms",
            "error",
        }

    def test_empty_url(self) -> None:
        result = check_fl_connection("")
        assert result["success"] is False
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Tests: CLI command — test-connection
# ---------------------------------------------------------------------------


class TestTestConnectionCommand:
    """test-connection CLI command."""

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(test_connection_cmd, ["--help"])
        assert result.exit_code == 0
        assert "Test connectivity" in result.output
        assert "SERVER_URL" in result.output

    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(test_connection_cmd, ["--help"])
        assert "--timeout" in result.output
        assert "--format" in result.output

    def test_missing_server_url(self, runner: CliRunner) -> None:
        result = runner.invoke(test_connection_cmd, [])
        assert result.exit_code != 0

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_successful_text_output(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": True,
            "host": "localhost",
            "port": 8080,
            "latency_ms": 12.3,
            "error": None,
        }

        result = runner.invoke(
            test_connection_cmd, ["localhost:8080"]
        )
        assert result.exit_code == 0
        assert "succeeded" in result.output or "✓" in result.output

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_failed_text_output(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": False,
            "host": "localhost",
            "port": 8080,
            "latency_ms": None,
            "error": "Connection refused",
        }

        result = runner.invoke(
            test_connection_cmd, ["localhost:8080"]
        )
        assert result.exit_code == 1

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_json_output_success(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": True,
            "host": "localhost",
            "port": 8080,
            "latency_ms": 5.0,
            "error": None,
        }

        result = runner.invoke(
            test_connection_cmd,
            ["localhost:8080", "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["success"] is True
        assert parsed["latency_ms"] == 5.0

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_json_output_failure(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": False,
            "host": "localhost",
            "port": 8080,
            "latency_ms": None,
            "error": "Connection refused",
        }

        result = runner.invoke(
            test_connection_cmd,
            ["localhost:8080", "--format", "json"],
        )
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["success"] is False
        assert parsed["error"] == "Connection refused"

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_custom_timeout_passed(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": True,
            "host": "localhost",
            "port": 8080,
            "latency_ms": 1.0,
            "error": None,
        }

        runner.invoke(
            test_connection_cmd,
            ["localhost:8080", "--timeout", "30"],
        )
        mock_conn.assert_called_once_with("localhost:8080", timeout=30.0)


# ---------------------------------------------------------------------------
# Tests: CLI group registration
# ---------------------------------------------------------------------------


class TestFederatedGroupRegistration:
    """Federated command group is registered in the CLI."""

    def test_federated_group_is_click_group(self) -> None:
        assert isinstance(federated_group, click.MultiCommand)

    def test_test_connection_is_registered(self) -> None:
        commands = federated_group.commands
        assert "test-connection" in commands

    def test_federated_registered_in_main_cli(self) -> None:
        cli = _build_cli()
        assert "federated" in cli.commands

    def test_federated_help(self, runner: CliRunner) -> None:
        cli = _build_cli()
        result = runner.invoke(cli, ["federated", "--help"])
        assert result.exit_code == 0
        assert "test-connection" in result.output

    @patch("aortica.cli.federated_cmd.check_fl_connection")
    def test_via_cli_group(
        self, mock_conn: MagicMock, runner: CliRunner
    ) -> None:
        mock_conn.return_value = {
            "success": True,
            "host": "localhost",
            "port": 8080,
            "latency_ms": 10.0,
            "error": None,
        }

        cli = _build_cli()
        result = runner.invoke(
            cli,
            ["federated", "test-connection", "localhost:8080"],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Tests: docs existence
# ---------------------------------------------------------------------------


class TestDocsExist:
    """Verify federated learning docs are present."""

    def test_dua_template_exists(self) -> None:
        from pathlib import Path

        dua = Path(__file__).resolve().parents[2] / "docs" / "federated" / "DUA_TEMPLATE.md"
        assert dua.exists(), f"DUA_TEMPLATE.md not found at {dua}"

    def test_onboarding_guide_exists(self) -> None:
        from pathlib import Path

        guide = Path(__file__).resolve().parents[2] / "docs" / "federated" / "ONBOARDING.md"
        assert guide.exists(), f"ONBOARDING.md not found at {guide}"

    def test_dua_template_has_content(self) -> None:
        from pathlib import Path

        dua = Path(__file__).resolve().parents[2] / "docs" / "federated" / "DUA_TEMPLATE.md"
        content = dua.read_text(encoding="utf-8")
        assert "Data Use Agreement" in content
        assert "data retention" in content.lower()
        assert "model update" in content.lower()
        assert "publication" in content.lower()
        assert "withdrawal" in content.lower()

    def test_onboarding_has_content(self) -> None:
        from pathlib import Path

        guide = Path(__file__).resolve().parents[2] / "docs" / "federated" / "ONBOARDING.md"
        content = guide.read_text(encoding="utf-8")
        assert "Onboarding" in content
        assert "prerequisite" in content.lower()
        assert "test-connection" in content
        assert "first federated round" in content.lower() or "first round" in content.lower()
