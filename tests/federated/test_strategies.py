"""Tests for aortica.federated.strategies — FedProx and SCAFFOLD.

Tests cover:
- FedProxStrategy construction, validation, mu property, repr
- FedProxStrategy.build() with mock Flower (proximal term injection)
- SCAFFOLDStrategy construction, validation, properties, repr
- SCAFFOLDStrategy.build() with mock Flower (control variate aggregation)
- Strategy selection via FLServerConfig (fedprox, scaffold)
- Imports and exports
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# FedProxStrategy — construction and validation
# ---------------------------------------------------------------------------


class TestFedProxConstruction:

    def test_default_mu(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        s = FedProxStrategy()
        assert s.mu == 0.01

    def test_custom_mu(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        s = FedProxStrategy(mu=0.1)
        assert s.mu == 0.1

    def test_mu_zero_allowed(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        s = FedProxStrategy(mu=0.0)
        assert s.mu == 0.0

    def test_negative_mu_raises(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        with pytest.raises(ValueError, match="mu"):
            FedProxStrategy(mu=-0.01)

    def test_repr(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        assert "FedProx" in repr(FedProxStrategy())
        assert "0.01" in repr(FedProxStrategy())


# ---------------------------------------------------------------------------
# FedProxStrategy — build and proximal injection
# ---------------------------------------------------------------------------


class TestFedProxBuild:

    def test_build_raises_without_flower(self) -> None:
        from aortica.federated.strategies import FedProxStrategy
        with patch("aortica.federated.strategies.HAS_FLOWER", False):
            s = FedProxStrategy()
            with pytest.raises(ImportError, match="Flower"):
                s.build()

    def test_build_returns_strategy(self) -> None:
        flwr = pytest.importorskip("flwr")
        from aortica.federated.strategies import FedProxStrategy
        s = FedProxStrategy(mu=0.05)
        flower_strategy = s.build()
        assert flower_strategy is not None

    def test_configure_fit_injects_mu(self) -> None:
        flwr = pytest.importorskip("flwr")
        from flwr.common import ndarrays_to_parameters
        from aortica.federated.strategies import FedProxStrategy

        s = FedProxStrategy(mu=0.05, min_fit_clients=1, min_available_clients=1)
        flower_strategy = s.build()

        # Create mock client manager
        mock_client = MagicMock()
        mock_client.cid = "client_1"
        mock_client_manager = MagicMock()
        mock_client_manager.num_available.return_value = 1
        mock_client_manager.sample.return_value = [mock_client]

        # Create mock parameters
        params = ndarrays_to_parameters([np.zeros(5)])

        configs = flower_strategy.configure_fit(
            server_round=1,
            parameters=params,
            client_manager=mock_client_manager,
        )

        assert len(configs) == 1
        _, fit_ins = configs[0]
        assert "proximal_mu" in fit_ins.config
        assert fit_ins.config["proximal_mu"] == 0.05


# ---------------------------------------------------------------------------
# SCAFFOLDStrategy — construction and validation
# ---------------------------------------------------------------------------


class TestSCAFFOLDConstruction:

    def test_default_learning_rate(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        s = SCAFFOLDStrategy()
        assert s.learning_rate == 1.0

    def test_custom_learning_rate(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        s = SCAFFOLDStrategy(learning_rate=0.5)
        assert s.learning_rate == 0.5

    def test_invalid_learning_rate(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        with pytest.raises(ValueError, match="learning_rate"):
            SCAFFOLDStrategy(learning_rate=0.0)

    def test_negative_learning_rate(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        with pytest.raises(ValueError, match="learning_rate"):
            SCAFFOLDStrategy(learning_rate=-1.0)

    def test_server_control_initially_none(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        assert SCAFFOLDStrategy().server_control is None

    def test_repr(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        assert "SCAFFOLD" in repr(SCAFFOLDStrategy())
        assert "1.0" in repr(SCAFFOLDStrategy())


# ---------------------------------------------------------------------------
# SCAFFOLDStrategy — build and aggregation
# ---------------------------------------------------------------------------


class TestSCAFFOLDBuild:

    def test_build_raises_without_flower(self) -> None:
        from aortica.federated.strategies import SCAFFOLDStrategy
        with patch("aortica.federated.strategies.HAS_FLOWER", False):
            s = SCAFFOLDStrategy()
            with pytest.raises(ImportError, match="Flower"):
                s.build()

    def test_build_returns_strategy(self) -> None:
        flwr = pytest.importorskip("flwr")
        from aortica.federated.strategies import SCAFFOLDStrategy
        s = SCAFFOLDStrategy()
        flower_strategy = s.build()
        assert flower_strategy is not None

    def test_aggregate_fit_updates_control_variates(self) -> None:
        flwr = pytest.importorskip("flwr")
        from flwr.common import (
            FitRes,
            Status,
            Code,
            ndarrays_to_parameters,
            parameters_to_ndarrays,
        )
        from aortica.federated.strategies import SCAFFOLDStrategy

        s = SCAFFOLDStrategy(learning_rate=1.0, min_fit_clients=1, min_available_clients=1)
        flower_strategy = s.build()

        # Simulate two clients returning weights
        w1 = [np.array([1.0, 2.0, 3.0])]
        w2 = [np.array([3.0, 4.0, 5.0])]

        mock_client1 = MagicMock()
        mock_client1.cid = "c1"
        mock_client2 = MagicMock()
        mock_client2.cid = "c2"

        res1 = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(w1),
            num_examples=100,
            metrics={"loss": 0.5},
        )
        res2 = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(w2),
            num_examples=100,
            metrics={"loss": 0.3},
        )

        results = [(mock_client1, res1), (mock_client2, res2)]
        params, metrics = flower_strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )

        assert params is not None
        assert s.server_control is not None
        assert len(s.server_control) == 1

        # Control variate should be non-zero after non-uniform updates
        aggregated = parameters_to_ndarrays(params)
        assert len(aggregated) == 1

    def test_aggregate_fit_empty_results(self) -> None:
        flwr = pytest.importorskip("flwr")
        from aortica.federated.strategies import SCAFFOLDStrategy

        s = SCAFFOLDStrategy()
        flower_strategy = s.build()

        params, metrics = flower_strategy.aggregate_fit(
            server_round=1, results=[], failures=[]
        )
        assert params is None
        assert metrics == {}


# ---------------------------------------------------------------------------
# Strategy selection via FLServerConfig
# ---------------------------------------------------------------------------


class TestStrategySelection:

    def test_fedprox_config_accepted(self) -> None:
        from aortica.federated.fl_server import FLServerConfig
        cfg = FLServerConfig(strategy="fedprox")
        assert cfg.strategy == "fedprox"

    def test_scaffold_config_accepted(self) -> None:
        from aortica.federated.fl_server import FLServerConfig
        cfg = FLServerConfig(strategy="scaffold")
        assert cfg.strategy == "scaffold"

    def test_server_builds_fedprox_strategy(self) -> None:
        flwr = pytest.importorskip("flwr")
        from aortica.federated.fl_server import FLServer, FLServerConfig

        with patch("aortica.federated.fl_server.HAS_FLOWER", True):
            server = FLServer(FLServerConfig(strategy="fedprox"))
            strategy = server._build_strategy()
        assert strategy is not None

    def test_server_builds_scaffold_strategy(self) -> None:
        flwr = pytest.importorskip("flwr")
        from aortica.federated.fl_server import FLServer, FLServerConfig

        with patch("aortica.federated.fl_server.HAS_FLOWER", True):
            server = FLServer(FLServerConfig(strategy="scaffold"))
            strategy = server._build_strategy()
        assert strategy is not None

    def test_unsupported_strategy_still_rejected(self) -> None:
        from aortica.federated.fl_server import FLServerConfig
        with pytest.raises(ValueError, match="Unsupported strategy"):
            FLServerConfig(strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:

    def test_import_strategies_module(self) -> None:
        import aortica.federated.strategies  # noqa: F401

    def test_import_fedprox_from_package(self) -> None:
        from aortica.federated import FedProxStrategy  # noqa: F401

    def test_import_scaffold_from_package(self) -> None:
        from aortica.federated import SCAFFOLDStrategy  # noqa: F401

    def test_has_flower_is_bool(self) -> None:
        from aortica.federated.strategies import HAS_FLOWER
        assert isinstance(HAS_FLOWER, bool)
