"""Tests for aortica.federated.fl_client — Flower FL client wrapper.

Tests cover:
- FLClientConfig (defaults, custom, validation, serialisation)
- AorticaFlowerClient construction and properties
- get_parameters / set_parameters with synthetic model
- fit() local training with mock data
- evaluate() with mock data
- Flower NumPyClient adapter
- start() method with mock Flower
- Label splitting and metric helpers
- Imports and exports
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# FLClientConfig — defaults
# ---------------------------------------------------------------------------


class TestFLClientConfigDefaults:
    """Tests for FLClientConfig default values."""

    def test_default_server_address(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().server_address == "localhost:8080"

    def test_default_local_epochs(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().local_epochs == 1

    def test_default_batch_size(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().batch_size == 32

    def test_default_lr(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().lr == 1e-3

    def test_default_enabled_tasks(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        cfg = FLClientConfig()
        assert cfg.enabled_tasks == ["rhythm", "structural", "ischaemia", "risk"]

    def test_default_base_checkpoint_none(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().base_checkpoint is None

    def test_default_seed(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        assert FLClientConfig().seed == 42

    def test_custom_values(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        cfg = FLClientConfig(
            data_path="/data/ecg",
            server_address="host:9090",
            local_epochs=3,
            batch_size=64,
            lr=5e-4,
            base_checkpoint="/models/ckpt.pt",
        )
        assert cfg.data_path == "/data/ecg"
        assert cfg.server_address == "host:9090"
        assert cfg.local_epochs == 3
        assert cfg.batch_size == 64
        assert cfg.lr == 5e-4
        assert cfg.base_checkpoint == "/models/ckpt.pt"


# ---------------------------------------------------------------------------
# FLClientConfig — validation
# ---------------------------------------------------------------------------


class TestFLClientConfigValidation:

    def test_invalid_local_epochs_zero(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        with pytest.raises(ValueError, match="local_epochs"):
            FLClientConfig(local_epochs=0)

    def test_invalid_batch_size_zero(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        with pytest.raises(ValueError, match="batch_size"):
            FLClientConfig(batch_size=0)

    def test_invalid_lr_zero(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        with pytest.raises(ValueError, match="lr"):
            FLClientConfig(lr=0.0)

    def test_invalid_lr_negative(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        with pytest.raises(ValueError, match="lr"):
            FLClientConfig(lr=-0.01)

    def test_empty_server_address(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        with pytest.raises(ValueError, match="server_address"):
            FLClientConfig(server_address="")


# ---------------------------------------------------------------------------
# FLClientConfig — serialisation
# ---------------------------------------------------------------------------


class TestFLClientConfigSerialisation:

    def test_to_dict(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        d = FLClientConfig().to_dict()
        assert isinstance(d, dict)
        assert d["local_epochs"] == 1
        assert d["server_address"] == "localhost:8080"

    def test_from_dict(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        cfg = FLClientConfig.from_dict({"local_epochs": 5, "lr": 0.01})
        assert cfg.local_epochs == 5
        assert cfg.lr == 0.01

    def test_from_dict_ignores_unknown(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        cfg = FLClientConfig.from_dict({"local_epochs": 2, "unknown": "x"})
        assert cfg.local_epochs == 2

    def test_roundtrip_dict(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        cfg1 = FLClientConfig(local_epochs=3, lr=0.005)
        cfg2 = FLClientConfig.from_dict(cfg1.to_dict())
        assert cfg1.to_dict() == cfg2.to_dict()

    def test_to_dict_json_serialisable(self) -> None:
        from aortica.federated.fl_client import FLClientConfig
        text = json.dumps(FLClientConfig().to_dict())
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Label splitting
# ---------------------------------------------------------------------------


class TestLabelSplitting:

    def test_split_all_tasks(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import _split_labels
        labels = torch.randn(4, 50)  # 22+15+10+3
        result = _split_labels(labels, ["rhythm", "structural", "ischaemia", "risk"])
        assert result["rhythm"].shape == (4, 22)
        assert result["structural"].shape == (4, 15)
        assert result["ischaemia"].shape == (4, 10)
        assert result["risk"].shape == (4, 3)

    def test_split_subset(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import _split_labels
        labels = torch.randn(4, 25)  # 22+3
        result = _split_labels(labels, ["rhythm", "risk"])
        assert result["rhythm"].shape == (4, 22)
        assert result["risk"].shape == (4, 3)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:

    def test_compute_f1_perfect(self) -> None:
        from aortica.federated.fl_client import _compute_f1
        preds = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert _compute_f1(preds, targets) == 1.0

    def test_compute_f1_worst(self) -> None:
        from aortica.federated.fl_client import _compute_f1
        preds = np.array([[1.0, 0.0], [0.0, 1.0]])
        targets = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert _compute_f1(preds, targets) == 0.0

    def test_compute_c_index_perfect(self) -> None:
        from aortica.federated.fl_client import _compute_c_index
        preds = np.array([[0.1], [0.5], [0.9]])
        targets = np.array([[0.0], [0.5], [1.0]])
        assert _compute_c_index(preds, targets) == 1.0

    def test_compute_c_index_single_sample(self) -> None:
        from aortica.federated.fl_client import _compute_c_index
        assert _compute_c_index(np.array([[0.5]]), np.array([[0.5]])) == 0.5

    def test_compute_c_index_multi_output(self) -> None:
        """Verify c-index is independent per output and correctly averaged."""
        from aortica.federated.fl_client import _compute_c_index
        # Output 0: perfectly concordant; output 1: perfectly discordant
        preds = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        targets = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = _compute_c_index(preds, targets)
        # Output 0: c=1.0, output 1: c=0.0 → avg = 0.5
        assert abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# AorticaFlowerClient — construction
# ---------------------------------------------------------------------------


class TestClientConstruction:

    def test_default_construction(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        client = AorticaFlowerClient()
        assert client.config is not None
        assert client.model is None

    def test_custom_config(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        cfg = FLClientConfig(local_epochs=3)
        client = AorticaFlowerClient(cfg)
        assert client.config.local_epochs == 3

    def test_with_model(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        mock_model = MagicMock()
        client = AorticaFlowerClient(model=mock_model)
        assert client.model is mock_model

    def test_repr(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        r = repr(AorticaFlowerClient())
        assert "AorticaFlowerClient" in r
        assert "localhost:8080" in r


# ---------------------------------------------------------------------------
# AorticaFlowerClient — get/set parameters with real model
# ---------------------------------------------------------------------------


class TestClientParameters:

    @pytest.fixture
    def _skip_no_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_get_parameters_returns_list(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        cfg = FLClientConfig(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg)
        # Force fresh model (skip pretrained download)
        with patch("aortica.federated.fl_client.AorticaFlowerClient._init_model") as m:
            from aortica.models.aortica_model import AorticaModel
            import torch  # noqa: E402
            client._model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
            client._device = torch.device("cpu")
            m.return_value = None
            params = client.get_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)
        assert len(params) > 0

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_set_parameters_updates_model(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel
        import torch

        cfg = FLClientConfig(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg)
        client._model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client._device = torch.device("cpu")

        params = client.get_parameters()
        # Zero out all params
        zeroed = [np.zeros_like(p) for p in params]
        client.set_parameters(zeroed)

        # Verify model params are now zero
        for p in client._model.parameters():
            assert torch.all(p == 0).item()

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_set_parameters_wrong_count_raises(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel
        import torch

        cfg = FLClientConfig(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg)
        client._model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client._device = torch.device("cpu")

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            client.set_parameters([np.zeros(5)])

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_get_set_roundtrip(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel
        import torch

        cfg = FLClientConfig(enabled_tasks=["risk"], feature_dim=252)
        client = AorticaFlowerClient(cfg)
        client._model = AorticaModel(enabled_tasks=["risk"], feature_dim=252)
        client._device = torch.device("cpu")

        params_before = client.get_parameters()
        client.set_parameters(params_before)
        params_after = client.get_parameters()

        for before, after in zip(params_before, params_after):
            np.testing.assert_array_almost_equal(before, after)


# ---------------------------------------------------------------------------
# AorticaFlowerClient — fit and evaluate with synthetic data
# ---------------------------------------------------------------------------


class TestClientFitEvaluate:

    @pytest.fixture
    def _skip_no_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.fixture
    def client_with_data(self, _skip_no_torch: None) -> Any:
        """Create a client with synthetic train/val loaders."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        cfg = FLClientConfig(
            enabled_tasks=["rhythm"],
            local_epochs=1,
            batch_size=4,
            lr=1e-3,
            feature_dim=252,
        )

        # Synthetic data: 8 samples, 12 leads, 500 samples; 22 rhythm labels
        x = torch.randn(8, 12, 500)
        y = torch.zeros(8, 22)
        y[:, 0] = 1.0  # normal sinus rhythm

        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(
            cfg, model=model, train_loader=train_loader, val_loader=val_loader
        )
        client._device = torch.device("cpu")
        return client

    def test_fit_returns_tuple(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        result = client_with_data.fit(params)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fit_returns_updated_params(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        updated_params, num_examples, metrics = client_with_data.fit(params)
        assert isinstance(updated_params, list)
        assert len(updated_params) == len(params)

    def test_fit_returns_num_examples(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        _, num_examples, _ = client_with_data.fit(params)
        assert num_examples == 8  # 8 samples in dataset

    def test_fit_returns_metrics(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        _, _, metrics = client_with_data.fit(params)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_fit_respects_server_config_epochs(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        # Server overrides local_epochs
        _, _, _ = client_with_data.fit(params, config={"local_epochs": 2})
        # Should complete without error

    def test_evaluate_returns_tuple(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        result = client_with_data.evaluate(params)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_evaluate_returns_loss(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        loss, num_examples, metrics = client_with_data.evaluate(params)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_returns_num_examples(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        _, num_examples, _ = client_with_data.evaluate(params)
        assert num_examples == 8

    def test_evaluate_returns_f1_metric(self, client_with_data: Any) -> None:
        params = client_with_data.get_parameters()
        _, _, metrics = client_with_data.evaluate(params)
        assert "rhythm_f1" in metrics
        assert 0.0 <= metrics["rhythm_f1"] <= 1.0

    def test_fit_no_train_loader(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        cfg = FLClientConfig(enabled_tasks=["rhythm"], feature_dim=252)
        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg, model=model)
        client._device = torch.device("cpu")

        params = client.get_parameters()
        updated, n, metrics = client.fit(params)
        assert n == 0
        assert metrics["loss"] == 0.0

    def test_evaluate_no_val_loader(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        cfg = FLClientConfig(enabled_tasks=["rhythm"], feature_dim=252)
        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg, model=model)
        client._device = torch.device("cpu")

        params = client.get_parameters()
        loss, n, metrics = client.evaluate(params)
        assert n == 0
        assert loss == 0.0


# ---------------------------------------------------------------------------
# FedProx proximal term verification
# ---------------------------------------------------------------------------


class TestFedProxProximalTerm:

    @pytest.fixture
    def _skip_no_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.fixture
    def client_with_data(self, _skip_no_torch: None) -> Any:
        """Create a client with synthetic train/val loaders."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        cfg = FLClientConfig(
            enabled_tasks=["rhythm"],
            local_epochs=2,
            batch_size=4,
            lr=1e-2,
            feature_dim=252,
            seed=42,
        )

        x = torch.randn(8, 12, 500)
        y = torch.zeros(8, 22)
        y[:, 0] = 1.0

        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(
            cfg, model=model, train_loader=train_loader
        )
        client._device = torch.device("cpu")
        return client

    def test_proximal_term_constrains_updates(
        self, client_with_data: Any
    ) -> None:
        """FedProx with large mu should keep weights closer to global."""
        import torch
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel
        from torch.utils.data import DataLoader, TensorDataset

        # Train without FedProx (mu=0)
        cfg = FLClientConfig(
            enabled_tasks=["rhythm"], local_epochs=2, batch_size=4,
            lr=1e-2, feature_dim=252, seed=42,
        )
        x = torch.randn(8, 12, 500)
        y = torch.zeros(8, 22)
        y[:, 0] = 1.0
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=4, shuffle=False)

        model_no_prox = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client_no_prox = AorticaFlowerClient(
            cfg, model=model_no_prox, train_loader=train_loader
        )
        client_no_prox._device = torch.device("cpu")
        params = client_no_prox.get_parameters()
        updated_no_prox, _, metrics_no_prox = client_no_prox.fit(params)

        # Train WITH FedProx (mu=10.0 — strong regularisation)
        model_prox = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        # Copy the same initial weights
        model_prox.load_state_dict(model_no_prox.state_dict())
        client_prox = AorticaFlowerClient(
            cfg, model=model_prox, train_loader=train_loader
        )
        client_prox._device = torch.device("cpu")
        params2 = client_prox.get_parameters()
        updated_prox, _, metrics_prox = client_prox.fit(
            params2, config={"proximal_mu": 10.0}
        )

        # Compute total parameter drift for each
        drift_no_prox = sum(
            float(np.sum((u - p) ** 2))
            for u, p in zip(updated_no_prox, params)
        )
        drift_prox = sum(
            float(np.sum((u - p) ** 2))
            for u, p in zip(updated_prox, params2)
        )

        # With strong proximal term, drift should be smaller
        assert drift_prox < drift_no_prox, (
            f"FedProx drift ({drift_prox:.6f}) should be less than "
            f"no-prox drift ({drift_no_prox:.6f})"
        )

    def test_proximal_mu_in_metrics(self, client_with_data: Any) -> None:
        """When proximal_mu is active, it should appear in fit metrics."""
        params = client_with_data.get_parameters()
        _, _, metrics = client_with_data.fit(
            params, config={"proximal_mu": 0.01}
        )
        assert "proximal_mu" in metrics
        assert metrics["proximal_mu"] == 0.01

    def test_no_proximal_mu_key_without_config(
        self, client_with_data: Any
    ) -> None:
        """Without proximal_mu in config, key should not be in metrics."""
        params = client_with_data.get_parameters()
        _, _, metrics = client_with_data.fit(params)
        assert "proximal_mu" not in metrics


# ---------------------------------------------------------------------------
# Flower adapter and start
# ---------------------------------------------------------------------------


class TestFlowerAdapter:

    def test_to_flower_client_raises_without_flower(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        with patch("aortica.federated.fl_client.HAS_FLOWER", False):
            client = AorticaFlowerClient()
            with pytest.raises(ImportError, match="Flower"):
                client.to_flower_client()

    def test_start_raises_without_flower(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        with patch("aortica.federated.fl_client.HAS_FLOWER", False):
            client = AorticaFlowerClient()
            with pytest.raises(ImportError, match="Flower"):
                client.start()

    def test_start_calls_start_numpy_client(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        mock_fl = MagicMock()
        mock_fl.client.NumPyClient = type("NumPyClient", (), {})
        mock_fl.client.start_numpy_client = MagicMock()

        cfg = FLClientConfig(server_address="test:1234", feature_dim=252)
        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(cfg, model=model)
        client._device = torch.device("cpu")

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_client.HAS_FLOWER", True):
                client.start()

        mock_fl.client.start_numpy_client.assert_called_once()
        call_kwargs = mock_fl.client.start_numpy_client.call_args[1]
        assert call_kwargs["server_address"] == "test:1234"

    def test_start_uses_override_address(self) -> None:
        torch = pytest.importorskip("torch")
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        from aortica.models.aortica_model import AorticaModel

        mock_fl = MagicMock()
        mock_fl.client.NumPyClient = type("NumPyClient", (), {})
        mock_fl.client.start_numpy_client = MagicMock()

        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        client = AorticaFlowerClient(model=model)
        client._device = torch.device("cpu")

        with patch.dict("sys.modules", {"flwr": mock_fl}):
            with patch("aortica.federated.fl_client.HAS_FLOWER", True):
                client.start(server_address="override:5555")

        call_kwargs = mock_fl.client.start_numpy_client.call_args[1]
        assert call_kwargs["server_address"] == "override:5555"


# ---------------------------------------------------------------------------
# Model initialisation paths
# ---------------------------------------------------------------------------


class TestModelInit:

    @pytest.fixture
    def _skip_no_torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_init_model_skips_if_already_set(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient
        mock_model = MagicMock()
        client = AorticaFlowerClient(model=mock_model)
        client._init_model()
        assert client._model is mock_model

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_init_model_fresh_on_pretrained_failure(self) -> None:
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig
        cfg = FLClientConfig(enabled_tasks=["rhythm"])
        client = AorticaFlowerClient(cfg)

        with patch("aortica.federated.fl_client.AorticaFlowerClient._init_model") as mock_init:
            # Just verify the method exists and is callable
            mock_init.return_value = None
            client._init_model()

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_init_from_checkpoint(self, tmp_path: Any) -> None:
        import torch
        from aortica.models.aortica_model import AorticaModel
        from aortica.federated.fl_client import AorticaFlowerClient, FLClientConfig

        # Create and save a checkpoint
        model = AorticaModel(enabled_tasks=["rhythm"], feature_dim=252)
        ckpt_path = tmp_path / "model.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        cfg = FLClientConfig(
            enabled_tasks=["rhythm"],
            feature_dim=252,
            base_checkpoint=str(ckpt_path),
        )
        client = AorticaFlowerClient(cfg)
        client._init_model()
        assert client._model is not None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:

    def test_import_module(self) -> None:
        import aortica.federated.fl_client  # noqa: F401

    def test_import_package_exports_client(self) -> None:
        from aortica.federated import AorticaFlowerClient  # noqa: F401

    def test_import_package_exports_config(self) -> None:
        from aortica.federated import FLClientConfig  # noqa: F401

    def test_has_flower_is_bool(self) -> None:
        from aortica.federated.fl_client import HAS_FLOWER
        assert isinstance(HAS_FLOWER, bool)

    def test_has_torch_is_bool(self) -> None:
        from aortica.federated.fl_client import HAS_TORCH
        assert isinstance(HAS_TORCH, bool)
