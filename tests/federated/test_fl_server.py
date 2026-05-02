"""Tests for aortica.federated.fl_server — Flower FL server scaffold.

Tests cover:
- Constants and supported strategies
- RoundMetrics dataclass (construction, defaults, to_dict)
- FLServerConfig dataclass (defaults, custom, validation, serialisation, YAML I/O)
- Simple YAML parser fallback
- FLServer construction and properties
- Strategy building (with mock Flower)
- Metric extraction from mock History
- Metric file logging
- Start method (with mock flwr.server.start_server)
- Imports and exports
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_supported_strategies_contains_fedavg(self) -> None:
        from aortica.federated.fl_server import _SUPPORTED_STRATEGIES

        assert "fedavg" in _SUPPORTED_STRATEGIES

    def test_supported_strategies_is_frozenset(self) -> None:
        from aortica.federated.fl_server import _SUPPORTED_STRATEGIES

        assert isinstance(_SUPPORTED_STRATEGIES, frozenset)

    def test_default_server_config_exists(self) -> None:
        from aortica.federated.fl_server import DEFAULT_SERVER_CONFIG

        assert DEFAULT_SERVER_CONFIG is not None

    def test_has_flower_is_bool(self) -> None:
        from aortica.federated.fl_server import HAS_FLOWER

        assert isinstance(HAS_FLOWER, bool)


# ---------------------------------------------------------------------------
# RoundMetrics
# ---------------------------------------------------------------------------


class TestRoundMetrics:
    """Tests for the RoundMetrics dataclass."""

    def test_construction_defaults(self) -> None:
        from aortica.federated.fl_server import RoundMetrics

        rm = RoundMetrics(round_number=1)
        assert rm.round_number == 1
        assert rm.loss is None
        assert rm.metrics == {}
        assert rm.num_clients == 0

    def test_construction_custom(self) -> None:
        from aortica.federated.fl_server import RoundMetrics

        rm = RoundMetrics(
            round_number=3,
            loss=0.45,
            metrics={"rhythm_f1": 0.87, "structural_f1": 0.72},
            num_clients=5,
        )
        assert rm.round_number == 3
        assert rm.loss == 0.45
        assert rm.metrics["rhythm_f1"] == 0.87
        assert rm.num_clients == 5

    def test_to_dict(self) -> None:
        from aortica.federated.fl_server import RoundMetrics

        rm = RoundMetrics(round_number=2, loss=0.3, metrics={"f1": 0.9})
        d = rm.to_dict()
        assert isinstance(d, dict)
        assert d["round_number"] == 2
        assert d["loss"] == 0.3
        assert d["metrics"]["f1"] == 0.9

    def test_to_dict_is_serialisable(self) -> None:
        from aortica.federated.fl_server import RoundMetrics

        rm = RoundMetrics(round_number=1, loss=0.5, metrics={"a": 1.0})
        text = json.dumps(rm.to_dict())
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# FLServerConfig — defaults and construction
# ---------------------------------------------------------------------------


class TestFLServerConfigDefaults:
    """Tests for FLServerConfig default values and basic construction."""

    def test_default_num_rounds(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.num_rounds == 5

    def test_default_min_fit_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.min_fit_clients == 2

    def test_default_min_evaluate_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.min_evaluate_clients == 2

    def test_default_min_available_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.min_available_clients == 2

    def test_default_server_address(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.server_address == "0.0.0.0:8080"

    def test_default_strategy(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.strategy == "fedavg"

    def test_default_fraction_fit(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.fraction_fit == 1.0

    def test_default_fraction_evaluate(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.fraction_evaluate == 1.0

    def test_default_log_dir(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        assert cfg.log_dir is None

    def test_custom_values(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig(
            num_rounds=20,
            min_fit_clients=5,
            min_evaluate_clients=3,
            min_available_clients=5,
            server_address="localhost:9090",
            strategy="fedavg",
            fraction_fit=0.5,
            fraction_evaluate=0.8,
            log_dir="/tmp/fl_logs",
        )
        assert cfg.num_rounds == 20
        assert cfg.min_fit_clients == 5
        assert cfg.server_address == "localhost:9090"
        assert cfg.fraction_fit == 0.5
        assert cfg.log_dir == "/tmp/fl_logs"


# ---------------------------------------------------------------------------
# FLServerConfig — validation
# ---------------------------------------------------------------------------


class TestFLServerConfigValidation:
    """Tests for FLServerConfig post-init validation."""

    def test_invalid_num_rounds_zero(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="num_rounds"):
            FLServerConfig(num_rounds=0)

    def test_invalid_num_rounds_negative(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="num_rounds"):
            FLServerConfig(num_rounds=-1)

    def test_invalid_min_fit_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="min_fit_clients"):
            FLServerConfig(min_fit_clients=0)

    def test_invalid_min_evaluate_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="min_evaluate_clients"):
            FLServerConfig(min_evaluate_clients=0)

    def test_invalid_min_available_clients(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="min_available_clients"):
            FLServerConfig(min_available_clients=0)

    def test_empty_server_address(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="server_address"):
            FLServerConfig(server_address="")

    def test_unsupported_strategy(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="Unsupported strategy"):
            FLServerConfig(strategy="invalid_strategy")

    def test_invalid_fraction_fit_zero(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="fraction_fit"):
            FLServerConfig(fraction_fit=0.0)

    def test_invalid_fraction_fit_over(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="fraction_fit"):
            FLServerConfig(fraction_fit=1.5)

    def test_invalid_fraction_evaluate_zero(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(ValueError, match="fraction_evaluate"):
            FLServerConfig(fraction_evaluate=0.0)


# ---------------------------------------------------------------------------
# FLServerConfig — serialisation
# ---------------------------------------------------------------------------


class TestFLServerConfigSerialisation:
    """Tests for FLServerConfig dict/YAML serialisation."""

    def test_to_dict(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["num_rounds"] == 5
        assert d["strategy"] == "fedavg"

    def test_from_dict(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        d = {"num_rounds": 10, "strategy": "fedavg", "server_address": "host:1234"}
        cfg = FLServerConfig.from_dict(d)
        assert cfg.num_rounds == 10
        assert cfg.server_address == "host:1234"

    def test_from_dict_ignores_unknown_keys(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        d = {"num_rounds": 3, "unknown_key": "ignored"}
        cfg = FLServerConfig.from_dict(d)
        assert cfg.num_rounds == 3

    def test_roundtrip_dict(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg1 = FLServerConfig(num_rounds=7, min_fit_clients=3)
        cfg2 = FLServerConfig.from_dict(cfg1.to_dict())
        assert cfg1.to_dict() == cfg2.to_dict()

    def test_to_yaml_and_from_yaml(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig(num_rounds=12, server_address="host:5555")
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = FLServerConfig.from_yaml(yaml_path)
        assert loaded.num_rounds == 12
        assert loaded.server_address == "host:5555"

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServerConfig

        with pytest.raises(FileNotFoundError):
            FLServerConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_to_dict_is_json_serialisable(self) -> None:
        from aortica.federated.fl_server import FLServerConfig

        cfg = FLServerConfig()
        text = json.dumps(cfg.to_dict())
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Simple YAML parser
# ---------------------------------------------------------------------------


class TestSimpleYamlParser:
    """Tests for the _simple_yaml_load fallback parser."""

    def test_parses_int(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("num_rounds: 10")
        assert result["num_rounds"] == 10

    def test_parses_float(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("fraction_fit: 0.5")
        assert result["fraction_fit"] == 0.5

    def test_parses_string(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load('strategy: "fedavg"')
        assert result["strategy"] == "fedavg"

    def test_parses_null(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("log_dir: null")
        assert result["log_dir"] is None

    def test_parses_bool(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("enabled: true\ndisabled: false")
        assert result["enabled"] is True
        assert result["disabled"] is False

    def test_skips_comments(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("# comment\nnum_rounds: 5")
        assert "comment" not in result
        assert result["num_rounds"] == 5

    def test_skips_empty_lines(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("\n\nnum_rounds: 5\n\n")
        assert result["num_rounds"] == 5

    def test_unquoted_string(self) -> None:
        from aortica.federated.fl_server import _simple_yaml_load

        result = _simple_yaml_load("server_address: 0.0.0.0:8080")
        # Contains ":", so partition splits at first ":"
        assert "server_address" in result


# ---------------------------------------------------------------------------
# FLServer — construction and properties
# ---------------------------------------------------------------------------


class TestFLServerConstruction:
    """Tests for FLServer construction and property access."""

    def test_default_construction(self) -> None:
        from aortica.federated.fl_server import FLServer

        server = FLServer()
        assert server.config is not None
        assert server.config.num_rounds == 5

    def test_custom_config(self) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig

        cfg = FLServerConfig(num_rounds=10)
        server = FLServer(cfg)
        assert server.config.num_rounds == 10

    def test_initial_round_metrics_empty(self) -> None:
        from aortica.federated.fl_server import FLServer

        server = FLServer()
        assert server.round_metrics == []

    def test_initial_history_none(self) -> None:
        from aortica.federated.fl_server import FLServer

        server = FLServer()
        assert server.history is None

    def test_round_metrics_returns_copy(self) -> None:
        from aortica.federated.fl_server import FLServer

        server = FLServer()
        metrics = server.round_metrics
        metrics.append(MagicMock())
        assert len(server.round_metrics) == 0

    def test_repr(self) -> None:
        from aortica.federated.fl_server import FLServer

        server = FLServer()
        r = repr(server)
        assert "FLServer" in r
        assert "fedavg" in r
        assert "8080" in r


# ---------------------------------------------------------------------------
# FLServer — strategy building (mock Flower)
# ---------------------------------------------------------------------------


class TestFLServerStrategy:
    """Tests for FLServer._build_strategy with mocked Flower."""

    def test_build_fedavg_strategy(self) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig

        mock_fedavg = MagicMock()
        mock_fl = MagicMock()
        mock_fl.server.strategy.FedAvg = mock_fedavg

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer(FLServerConfig(strategy="fedavg"))
                strategy = server._build_strategy()

        mock_fedavg.assert_called_once()
        call_kwargs = mock_fedavg.call_args[1]
        assert call_kwargs["fraction_fit"] == 1.0
        assert call_kwargs["min_fit_clients"] == 2

    def test_build_strategy_raises_without_flower(self) -> None:
        from aortica.federated.fl_server import FLServer

        with patch("aortica.federated.fl_server.HAS_FLOWER", False):
            server = FLServer()
            with pytest.raises(ImportError, match="Flower"):
                server._build_strategy()


# ---------------------------------------------------------------------------
# FLServer — metric extraction from mock History
# ---------------------------------------------------------------------------


class TestFLServerMetricExtraction:
    """Tests for FLServer._extract_round_metrics."""

    def _make_history(
        self,
        losses_centralized: list[tuple[int, float]] | None = None,
        losses_distributed: list[tuple[int, float]] | None = None,
        metrics_distributed_fit: dict[str, list[tuple[int, float]]] | None = None,
        metrics_distributed: dict[str, list[tuple[int, float]]] | None = None,
        metrics_centralized: dict[str, list[tuple[int, float]]] | None = None,
    ) -> MagicMock:
        """Create a mock Flower History object."""
        h = MagicMock()
        h.losses_centralized = losses_centralized or []
        h.losses_distributed = losses_distributed or []
        h.metrics_distributed_fit = metrics_distributed_fit or {}
        h.metrics_distributed = metrics_distributed or {}
        h.metrics_centralized = metrics_centralized or {}
        return h

    def test_empty_history(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history()
        metrics = FLServer._extract_round_metrics(history)
        assert metrics == []

    def test_centralized_losses(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            losses_centralized=[(1, 0.5), (2, 0.3), (3, 0.2)]
        )
        metrics = FLServer._extract_round_metrics(history)
        assert len(metrics) == 3
        assert metrics[0].round_number == 1
        assert metrics[0].loss == 0.5
        assert metrics[2].loss == 0.2

    def test_distributed_losses(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            losses_distributed=[(1, 0.7), (2, 0.4)]
        )
        metrics = FLServer._extract_round_metrics(history)
        assert len(metrics) == 2
        assert metrics[0].loss == 0.7

    def test_centralized_takes_precedence(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            losses_centralized=[(1, 0.3)],
            losses_distributed=[(1, 0.5)],
        )
        metrics = FLServer._extract_round_metrics(history)
        assert metrics[0].loss == 0.3  # centralized processed first

    def test_distributed_fit_metrics(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            metrics_distributed_fit={"rhythm_f1": [(1, 0.85), (2, 0.90)]}
        )
        metrics = FLServer._extract_round_metrics(history)
        assert len(metrics) == 2
        assert metrics[0].metrics["rhythm_f1"] == 0.85
        assert metrics[1].metrics["rhythm_f1"] == 0.90

    def test_centralized_metrics(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            metrics_centralized={"accuracy": [(1, 0.92)]}
        )
        metrics = FLServer._extract_round_metrics(history)
        assert metrics[0].metrics["accuracy"] == 0.92

    def test_metrics_sorted_by_round(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            losses_centralized=[(3, 0.1), (1, 0.5), (2, 0.3)]
        )
        metrics = FLServer._extract_round_metrics(history)
        rounds = [m.round_number for m in metrics]
        assert rounds == [1, 2, 3]

    def test_mixed_metrics_and_losses(self) -> None:
        from aortica.federated.fl_server import FLServer

        history = self._make_history(
            losses_centralized=[(1, 0.5), (2, 0.3)],
            metrics_distributed_fit={"f1": [(1, 0.8), (2, 0.9)]},
        )
        metrics = FLServer._extract_round_metrics(history)
        assert len(metrics) == 2
        assert metrics[0].loss == 0.5
        assert metrics[0].metrics["f1"] == 0.8


# ---------------------------------------------------------------------------
# FLServer — metric file logging
# ---------------------------------------------------------------------------


class TestFLServerLogging:
    """Tests for FLServer metric file logging."""

    def test_write_metrics_log(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig, RoundMetrics

        cfg = FLServerConfig(log_dir=str(tmp_path / "logs"))
        server = FLServer(cfg)
        server._round_metrics = [
            RoundMetrics(round_number=1, loss=0.5, metrics={"f1": 0.8}),
            RoundMetrics(round_number=2, loss=0.3, metrics={"f1": 0.9}),
        ]
        server._write_metrics_log(cfg.log_dir)  # type: ignore[arg-type]

        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        assert (log_dir / "round_0001.json").exists()
        assert (log_dir / "round_0002.json").exists()
        assert (log_dir / "summary.json").exists()

    def test_round_json_content(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig, RoundMetrics

        cfg = FLServerConfig(log_dir=str(tmp_path / "logs"))
        server = FLServer(cfg)
        server._round_metrics = [
            RoundMetrics(round_number=1, loss=0.5),
        ]
        server._write_metrics_log(cfg.log_dir)  # type: ignore[arg-type]

        data = json.loads((tmp_path / "logs" / "round_0001.json").read_text())
        assert data["round_number"] == 1
        assert data["loss"] == 0.5

    def test_summary_json_content(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig, RoundMetrics

        cfg = FLServerConfig(log_dir=str(tmp_path / "logs"), num_rounds=3)
        server = FLServer(cfg)
        server._round_metrics = [
            RoundMetrics(round_number=1, loss=0.5),
        ]
        server._write_metrics_log(cfg.log_dir)  # type: ignore[arg-type]

        data = json.loads((tmp_path / "logs" / "summary.json").read_text())
        assert data["num_rounds_completed"] == 1
        assert data["config"]["num_rounds"] == 3
        assert len(data["rounds"]) == 1


# ---------------------------------------------------------------------------
# FLServer — start method (mock Flower)
# ---------------------------------------------------------------------------


class TestFLServerStart:
    """Tests for FLServer.start() with fully mocked Flower."""

    def _setup_mock_flower(self) -> tuple[MagicMock, MagicMock]:
        """Return (mock_fl_module, mock_history)."""
        mock_history = MagicMock()
        mock_history.losses_centralized = [(1, 0.5), (2, 0.3)]
        mock_history.losses_distributed = []
        mock_history.metrics_distributed_fit = {"f1": [(1, 0.8), (2, 0.9)]}
        mock_history.metrics_distributed = {}
        mock_history.metrics_centralized = {}

        mock_fl = MagicMock()
        mock_fl.server.start_server.return_value = mock_history
        mock_fl.server.ServerConfig.return_value = MagicMock()
        mock_fl.server.strategy.FedAvg.return_value = MagicMock()

        return mock_fl, mock_history

    def test_start_returns_history(self) -> None:
        from aortica.federated.fl_server import FLServer

        mock_fl, mock_history = self._setup_mock_flower()

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer()
                result = server.start()

        assert result is mock_history

    def test_start_populates_round_metrics(self) -> None:
        from aortica.federated.fl_server import FLServer

        mock_fl, _ = self._setup_mock_flower()

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer()
                server.start()

        assert len(server.round_metrics) == 2
        assert server.round_metrics[0].loss == 0.5
        assert server.round_metrics[1].metrics["f1"] == 0.9

    def test_start_stores_history(self) -> None:
        from aortica.federated.fl_server import FLServer

        mock_fl, mock_history = self._setup_mock_flower()

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer()
                server.start()

        assert server.history is mock_history

    def test_start_calls_start_server(self) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig

        mock_fl, _ = self._setup_mock_flower()
        cfg = FLServerConfig(server_address="localhost:9999", num_rounds=3)

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer(cfg)
                server.start()

        mock_fl.server.start_server.assert_called_once()
        call_kwargs = mock_fl.server.start_server.call_args[1]
        assert call_kwargs["server_address"] == "localhost:9999"

    def test_start_with_log_dir(self, tmp_path: Path) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig

        mock_fl, _ = self._setup_mock_flower()
        cfg = FLServerConfig(log_dir=str(tmp_path / "fl_logs"))

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer(cfg)
                server.start()

        assert (tmp_path / "fl_logs" / "summary.json").exists()

    def test_start_raises_without_flower(self) -> None:
        from aortica.federated.fl_server import FLServer

        with patch("aortica.federated.fl_server.HAS_FLOWER", False):
            server = FLServer()
            with pytest.raises(ImportError, match="Flower"):
                server.start()

    def test_start_passes_config_to_server_config(self) -> None:
        from aortica.federated.fl_server import FLServer, FLServerConfig

        mock_fl, _ = self._setup_mock_flower()
        cfg = FLServerConfig(num_rounds=7)

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_server.HAS_FLOWER", True):
                server = FLServer(cfg)
                server.start()

        mock_fl.server.ServerConfig.assert_called_once_with(num_rounds=7)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Tests for module and package imports."""

    def test_import_fl_server_module(self) -> None:
        import aortica.federated.fl_server  # noqa: F401

    def test_import_federated_package(self) -> None:
        import aortica.federated  # noqa: F401

    def test_package_exports_fl_server(self) -> None:
        from aortica.federated import FLServer  # noqa: F401

    def test_package_exports_config(self) -> None:
        from aortica.federated import FLServerConfig  # noqa: F401

    def test_package_exports_round_metrics(self) -> None:
        from aortica.federated import RoundMetrics  # noqa: F401
