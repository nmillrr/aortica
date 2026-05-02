"""Tests for ``aortica federated server`` and ``aortica federated client`` CLI commands.

Covers:
- Config validation helpers (_validate_server_config, _validate_client_config)
- Server command (--help, --dry-run text/json, missing config, invalid config)
- Client command (--help, --dry-run text/json, missing config, invalid config)
- Group registration (server and client registered in federated group and main CLI)
- FLClientConfig YAML round-trip serialisation
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

click = pytest.importorskip("click")
rich = pytest.importorskip("rich")

from click.testing import CliRunner  # noqa: E402

from aortica.cli import _build_cli  # noqa: E402
from aortica.cli.federated_cmd import (  # noqa: E402
    _print_config_summary,
    _validate_client_config,
    _validate_server_config,
    client_cmd,
    federated_group,
    server_cmd,
)
from aortica.federated.fl_client import FLClientConfig  # noqa: E402
from aortica.federated.fl_server import FLServerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Click CliRunner for isolated CLI testing."""
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def server_config_file(tmp_path: Path) -> Path:
    """Create a valid FL server YAML config file."""
    config = FLServerConfig(
        num_rounds=3,
        min_fit_clients=2,
        min_evaluate_clients=2,
        server_address="0.0.0.0:8080",
        strategy="fedavg",
    )
    fpath = tmp_path / "server_config.yaml"
    config.to_yaml(fpath)
    return fpath


@pytest.fixture()
def client_config_file(tmp_path: Path) -> Path:
    """Create a valid FL client YAML config file."""
    config = FLClientConfig(
        data_path="/data/ecg",
        server_address="localhost:8080",
        local_epochs=2,
        batch_size=16,
        lr=0.001,
    )
    config.to_yaml(tmp_path / "client_config.yaml")
    return tmp_path / "client_config.yaml"


@pytest.fixture()
def invalid_config_file(tmp_path: Path) -> Path:
    """Create a YAML file with invalid config values."""
    fpath = tmp_path / "bad_config.yaml"
    fpath.write_text("num_rounds: -5\n")
    return fpath


@pytest.fixture()
def invalid_client_config_file(tmp_path: Path) -> Path:
    """Create a YAML file with invalid client config values."""
    fpath = tmp_path / "bad_client_config.yaml"
    fpath.write_text("local_epochs: 0\nbatch_size: -1\n")
    return fpath


# ---------------------------------------------------------------------------
# Tests: _validate_server_config
# ---------------------------------------------------------------------------


class TestValidateServerConfig:
    """Config validation helper for server."""

    def test_valid_config(self, server_config_file: Path) -> None:
        config = _validate_server_config(str(server_config_file))
        assert isinstance(config, FLServerConfig)
        assert config.num_rounds == 3
        assert config.strategy == "fedavg"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="Config file not found"):
            _validate_server_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_values_raises(self, invalid_config_file: Path) -> None:
        with pytest.raises(click.ClickException, match="Invalid server config"):
            _validate_server_config(str(invalid_config_file))


# ---------------------------------------------------------------------------
# Tests: _validate_client_config
# ---------------------------------------------------------------------------


class TestValidateClientConfig:
    """Config validation helper for client."""

    def test_valid_config(self, client_config_file: Path) -> None:
        config = _validate_client_config(str(client_config_file))
        assert isinstance(config, FLClientConfig)
        assert config.data_path == "/data/ecg"
        assert config.local_epochs == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="Config file not found"):
            _validate_client_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_values_raises(
        self, invalid_client_config_file: Path
    ) -> None:
        with pytest.raises(click.ClickException, match="Invalid client config"):
            _validate_client_config(str(invalid_client_config_file))


# ---------------------------------------------------------------------------
# Tests: _print_config_summary
# ---------------------------------------------------------------------------


class TestPrintConfigSummary:
    """Config summary output helpers."""

    def test_json_output(self, runner: CliRunner) -> None:
        data = {"num_rounds": 5, "strategy": "fedavg"}
        # Capture stdout
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            pass
        # Just verify it doesn't raise
        _print_config_summary("Test", data, "json")

    def test_text_output(self) -> None:
        data = {"num_rounds": 5, "strategy": "fedavg"}
        # Should not raise
        _print_config_summary("Test", data, "text")


# ---------------------------------------------------------------------------
# Tests: server command
# ---------------------------------------------------------------------------


class TestServerCommand:
    """``aortica federated server`` CLI command."""

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(server_cmd, ["--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output
        assert "--dry-run" in result.output
        assert "--format" in result.output

    def test_help_shows_yaml_fields(self, runner: CliRunner) -> None:
        result = runner.invoke(server_cmd, ["--help"])
        assert "num_rounds" in result.output
        assert "strategy" in result.output
        assert "server_address" in result.output

    def test_dry_run_text(
        self, runner: CliRunner, server_config_file: Path
    ) -> None:
        result = runner.invoke(
            server_cmd, [str(server_config_file), "--dry-run"]
        )
        assert result.exit_code == 0
        # Should show the config is valid (text or Rich)
        assert "valid" in result.output.lower() or "✓" in result.output

    def test_dry_run_json(
        self, runner: CliRunner, server_config_file: Path
    ) -> None:
        result = runner.invoke(
            server_cmd,
            [str(server_config_file), "--dry-run", "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["num_rounds"] == 3
        assert parsed["strategy"] == "fedavg"

    def test_missing_config(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            server_cmd,
            [str(tmp_path / "nonexistent.yaml"), "--dry-run"],
        )
        assert result.exit_code != 0
        all_output = (result.output + (result.stderr or "")).lower()
        assert "not found" in all_output or "error" in all_output

    def test_invalid_config(
        self, runner: CliRunner, invalid_config_file: Path
    ) -> None:
        result = runner.invoke(
            server_cmd, [str(invalid_config_file), "--dry-run"]
        )
        assert result.exit_code != 0
        all_output = (result.output + (result.stderr or "")).lower()
        assert "invalid" in all_output or "error" in all_output

    def test_missing_argument(self, runner: CliRunner) -> None:
        result = runner.invoke(server_cmd, [])
        assert result.exit_code != 0

    @patch("aortica.cli.federated_cmd._validate_server_config")
    @patch("aortica.federated.fl_server.FLServer")
    def test_actual_start(
        self,
        mock_server_class: MagicMock,
        mock_validate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Verify that without --dry-run the server is actually started."""
        mock_config = FLServerConfig(num_rounds=1)
        mock_validate.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.round_metrics = []
        mock_server_class.return_value = mock_instance

        # Create dummy config file (validation is mocked)
        cfg = tmp_path / "dummy.yaml"
        cfg.write_text("num_rounds: 1\n")

        result = runner.invoke(server_cmd, [str(cfg)])
        assert result.exit_code == 0
        mock_instance.start.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: client command
# ---------------------------------------------------------------------------


class TestClientCommand:
    """``aortica federated client`` CLI command."""

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(client_cmd, ["--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output
        assert "--dry-run" in result.output
        assert "--format" in result.output

    def test_help_shows_yaml_fields(self, runner: CliRunner) -> None:
        result = runner.invoke(client_cmd, ["--help"])
        assert "data_path" in result.output
        assert "server_address" in result.output
        assert "local_epochs" in result.output
        assert "batch_size" in result.output

    def test_dry_run_text(
        self, runner: CliRunner, client_config_file: Path
    ) -> None:
        result = runner.invoke(
            client_cmd, [str(client_config_file), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "✓" in result.output

    def test_dry_run_json(
        self, runner: CliRunner, client_config_file: Path
    ) -> None:
        result = runner.invoke(
            client_cmd,
            [str(client_config_file), "--dry-run", "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["data_path"] == "/data/ecg"
        assert parsed["local_epochs"] == 2
        assert parsed["batch_size"] == 16

    def test_missing_config(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            client_cmd,
            [str(tmp_path / "nonexistent.yaml"), "--dry-run"],
        )
        assert result.exit_code != 0
        all_output = (result.output + (result.stderr or "")).lower()
        assert "not found" in all_output or "error" in all_output

    def test_invalid_config(
        self, runner: CliRunner, invalid_client_config_file: Path
    ) -> None:
        result = runner.invoke(
            client_cmd, [str(invalid_client_config_file), "--dry-run"]
        )
        assert result.exit_code != 0

    def test_missing_argument(self, runner: CliRunner) -> None:
        result = runner.invoke(client_cmd, [])
        assert result.exit_code != 0

    @patch("aortica.cli.federated_cmd._validate_client_config")
    @patch("aortica.federated.fl_client.AorticaFlowerClient")
    def test_actual_start(
        self,
        mock_client_class: MagicMock,
        mock_validate: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Verify that without --dry-run the client is actually started."""
        mock_config = FLClientConfig(data_path="/data/ecg")
        mock_validate.return_value = mock_config

        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        cfg = tmp_path / "dummy.yaml"
        cfg.write_text('data_path: "/data/ecg"\n')

        result = runner.invoke(client_cmd, [str(cfg)])
        assert result.exit_code == 0
        mock_instance.start.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: group registration
# ---------------------------------------------------------------------------


class TestGroupRegistration:
    """server and client commands registered in the federated group."""

    def test_server_registered_in_federated_group(self) -> None:
        assert "server" in federated_group.commands

    def test_client_registered_in_federated_group(self) -> None:
        assert "client" in federated_group.commands

    def test_test_connection_still_registered(self) -> None:
        assert "test-connection" in federated_group.commands

    def test_all_three_commands_present(self) -> None:
        commands = set(federated_group.commands.keys())
        assert {"server", "client", "test-connection"}.issubset(commands)

    def test_federated_help_shows_all_commands(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(federated_group, ["--help"])
        assert result.exit_code == 0
        assert "server" in result.output
        assert "client" in result.output
        assert "test-connection" in result.output

    def test_via_main_cli_server_help(self, runner: CliRunner) -> None:
        cli = _build_cli()
        result = runner.invoke(cli, ["federated", "server", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output

    def test_via_main_cli_client_help(self, runner: CliRunner) -> None:
        cli = _build_cli()
        result = runner.invoke(cli, ["federated", "client", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_FILE" in result.output

    def test_via_main_cli_dry_run_server(
        self, runner: CliRunner, server_config_file: Path
    ) -> None:
        cli = _build_cli()
        result = runner.invoke(
            cli,
            ["federated", "server", str(server_config_file), "--dry-run"],
        )
        assert result.exit_code == 0

    def test_via_main_cli_dry_run_client(
        self, runner: CliRunner, client_config_file: Path
    ) -> None:
        cli = _build_cli()
        result = runner.invoke(
            cli,
            ["federated", "client", str(client_config_file), "--dry-run"],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Tests: FLClientConfig YAML round-trip
# ---------------------------------------------------------------------------


class TestFLClientConfigYaml:
    """FLClientConfig YAML serialisation and deserialisation."""

    def test_round_trip(self, tmp_path: Path) -> None:
        original = FLClientConfig(
            data_path="/ecg/data",
            server_address="fl.example.com:9090",
            local_epochs=3,
            batch_size=64,
            lr=5e-4,
            seed=123,
        )
        fpath = tmp_path / "config.yaml"
        original.to_yaml(fpath)

        loaded = FLClientConfig.from_yaml(fpath)
        assert loaded.data_path == original.data_path
        assert loaded.server_address == original.server_address
        assert loaded.local_epochs == original.local_epochs
        assert loaded.batch_size == original.batch_size
        assert loaded.seed == original.seed

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FLClientConfig.from_yaml(tmp_path / "nope.yaml")

    def test_from_yaml_invalid_content(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.yaml"
        fpath.write_text("local_epochs: -1\n")
        with pytest.raises(ValueError):
            FLClientConfig.from_yaml(fpath)

    def test_to_yaml_creates_parent_dirs(self, tmp_path: Path) -> None:
        config = FLClientConfig()
        fpath = tmp_path / "sub" / "dir" / "config.yaml"
        config.to_yaml(fpath)
        assert fpath.exists()

    def test_defaults_round_trip(self, tmp_path: Path) -> None:
        """Default config survives YAML round-trip."""
        original = FLClientConfig()
        fpath = tmp_path / "defaults.yaml"
        original.to_yaml(fpath)

        loaded = FLClientConfig.from_yaml(fpath)
        assert loaded.data_path == ""
        assert loaded.server_address == "localhost:8080"
        assert loaded.local_epochs == 1
        assert loaded.batch_size == 32

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        fpath = tmp_path / "extra.yaml"
        fpath.write_text('data_path: "/data"\nextra_field: 42\n')
        config = FLClientConfig.from_yaml(fpath)
        assert config.data_path == "/data"
        assert not hasattr(config, "extra_field")

    def test_from_dict(self) -> None:
        data = {"data_path": "/ecg", "local_epochs": 5, "unknown": "x"}
        config = FLClientConfig.from_dict(data)
        assert config.data_path == "/ecg"
        assert config.local_epochs == 5

    def test_to_dict(self) -> None:
        config = FLClientConfig(data_path="/test", seed=99)
        d = config.to_dict()
        assert d["data_path"] == "/test"
        assert d["seed"] == 99
        assert "enabled_tasks" in d


# ---------------------------------------------------------------------------
# Tests: FLServerConfig YAML schema documented fields
# ---------------------------------------------------------------------------


class TestServerConfigSchemaFields:
    """All documented YAML fields are accepted by FLServerConfig."""

    def test_all_fields_accepted(self, tmp_path: Path) -> None:
        fpath = tmp_path / "full_server.yaml"
        fpath.write_text(
            textwrap.dedent("""\
            num_rounds: 10
            min_fit_clients: 3
            min_evaluate_clients: 3
            min_available_clients: 3
            server_address: "0.0.0.0:9090"
            strategy: "fedprox"
            fraction_fit: 0.8
            fraction_evaluate: 0.5
            log_dir: "/tmp/fl_logs"
            """)
        )
        config = FLServerConfig.from_yaml(fpath)
        assert config.num_rounds == 10
        assert config.strategy == "fedprox"
        assert config.fraction_fit == 0.8
        assert config.log_dir == "/tmp/fl_logs"


class TestClientConfigSchemaFields:
    """All documented YAML fields are accepted by FLClientConfig."""

    def test_all_fields_accepted(self, tmp_path: Path) -> None:
        fpath = tmp_path / "full_client.yaml"
        fpath.write_text(
            textwrap.dedent("""\
            data_path: "/data/ecg"
            server_address: "fl.example.com:8080"
            local_epochs: 5
            batch_size: 64
            lr: 0.0005
            sampling_rate: 500
            window_seconds: 10.0
            weight_decay: 0.0001
            max_grad_norm: 1.0
            feature_dim: 256
            head_hidden_dim: 128
            head_dropout: 0.3
            seed: 42
            """)
        )
        config = FLClientConfig.from_yaml(fpath)
        assert config.data_path == "/data/ecg"
        assert config.local_epochs == 5
        assert config.batch_size == 64
        assert config.lr == 0.0005
        assert config.feature_dim == 256
