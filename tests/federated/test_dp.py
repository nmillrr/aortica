"""Tests for aortica.federated.dp — Differential Privacy wrapper.

Tests cover:
- DPConfig (defaults, validation, serialisation)
- Privacy accounting functions
- PrivacyBudgetTracker
- DPWrapper gradient clipping, noise injection, fit()
- Imports
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# DPConfig
# ---------------------------------------------------------------------------


class TestDPConfig:

    def test_defaults(self) -> None:
        from aortica.federated.dp import DPConfig
        cfg = DPConfig()
        assert cfg.epsilon == 1.0
        assert cfg.delta == 1e-5
        assert cfg.max_grad_norm == 1.0
        assert cfg.noise_multiplier is None
        assert cfg.expected_rounds == 10

    def test_invalid_epsilon(self) -> None:
        from aortica.federated.dp import DPConfig
        with pytest.raises(ValueError, match="epsilon"):
            DPConfig(epsilon=0)

    def test_invalid_delta_low(self) -> None:
        from aortica.federated.dp import DPConfig
        with pytest.raises(ValueError, match="delta"):
            DPConfig(delta=0)

    def test_invalid_delta_high(self) -> None:
        from aortica.federated.dp import DPConfig
        with pytest.raises(ValueError, match="delta"):
            DPConfig(delta=1.0)

    def test_invalid_max_grad_norm(self) -> None:
        from aortica.federated.dp import DPConfig
        with pytest.raises(ValueError, match="max_grad_norm"):
            DPConfig(max_grad_norm=0)

    def test_invalid_noise_multiplier(self) -> None:
        from aortica.federated.dp import DPConfig
        with pytest.raises(ValueError, match="noise_multiplier"):
            DPConfig(noise_multiplier=-0.1)

    def test_to_dict(self) -> None:
        from aortica.federated.dp import DPConfig
        d = DPConfig().to_dict()
        assert d["epsilon"] == 1.0
        assert d["delta"] == 1e-5


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------


class TestPrivacyAccounting:

    def test_rdp_epsilon_finite(self) -> None:
        from aortica.federated.dp import _compute_rdp_epsilon
        eps = _compute_rdp_epsilon(noise_multiplier=1.0, num_rounds=10, delta=1e-5)
        assert eps > 0
        assert eps < float("inf")

    def test_rdp_epsilon_zero_noise(self) -> None:
        from aortica.federated.dp import _compute_rdp_epsilon
        eps = _compute_rdp_epsilon(noise_multiplier=0, num_rounds=10, delta=1e-5)
        assert eps == float("inf")

    def test_rdp_epsilon_increases_with_rounds(self) -> None:
        from aortica.federated.dp import _compute_rdp_epsilon
        eps5 = _compute_rdp_epsilon(1.0, 5, 1e-5)
        eps10 = _compute_rdp_epsilon(1.0, 10, 1e-5)
        assert eps10 > eps5

    def test_rdp_epsilon_decreases_with_noise(self) -> None:
        from aortica.federated.dp import _compute_rdp_epsilon
        eps_low = _compute_rdp_epsilon(0.5, 10, 1e-5)
        eps_high = _compute_rdp_epsilon(5.0, 10, 1e-5)
        assert eps_high < eps_low

    def test_calibrate_noise(self) -> None:
        from aortica.federated.dp import _calibrate_noise_multiplier, _compute_rdp_epsilon
        sigma = _calibrate_noise_multiplier(epsilon=1.0, delta=1e-5, num_rounds=10)
        assert sigma > 0
        achieved = _compute_rdp_epsilon(sigma, 10, 1e-5)
        assert achieved <= 1.0 + 0.01  # within tolerance


# ---------------------------------------------------------------------------
# PrivacyBudgetTracker
# ---------------------------------------------------------------------------


class TestPrivacyBudgetTracker:

    def test_initial_state(self) -> None:
        from aortica.federated.dp import PrivacyBudgetTracker
        t = PrivacyBudgetTracker(total_epsilon=1.0, delta=1e-5, noise_multiplier=1.0)
        assert t.rounds_consumed == 0
        assert t.epsilon_spent == 0.0
        assert not t.is_exhausted

    def test_consume_round(self) -> None:
        from aortica.federated.dp import PrivacyBudgetTracker
        t = PrivacyBudgetTracker(total_epsilon=100.0, delta=1e-5, noise_multiplier=1.0)
        eps = t.consume_round()
        assert eps > 0
        assert t.rounds_consumed == 1

    def test_budget_decreases(self) -> None:
        from aortica.federated.dp import PrivacyBudgetTracker
        t = PrivacyBudgetTracker(total_epsilon=100.0, delta=1e-5, noise_multiplier=1.0)
        t.consume_round()
        assert t.epsilon_remaining < t.total_epsilon

    def test_exhaustion_raises(self) -> None:
        from aortica.federated.dp import PrivacyBudgetTracker
        # Very small budget with low noise = quick exhaustion
        t = PrivacyBudgetTracker(total_epsilon=0.001, delta=1e-5, noise_multiplier=0.1)
        with pytest.raises(RuntimeError, match="exhausted"):
            for _ in range(1000):
                t.consume_round()

    def test_summary(self) -> None:
        from aortica.federated.dp import PrivacyBudgetTracker
        t = PrivacyBudgetTracker(total_epsilon=1.0, delta=1e-5, noise_multiplier=1.0)
        s = t.summary()
        assert "epsilon_total" in s
        assert "epsilon_spent" in s
        assert "rounds_consumed" in s


# ---------------------------------------------------------------------------
# DPWrapper
# ---------------------------------------------------------------------------


class TestDPWrapper:

    def _make_mock_client(self) -> MagicMock:
        """Create a mock client that returns slightly modified params."""
        client = MagicMock()
        def mock_fit(params, config=None):
            # Simulate local training: shift params by 0.1
            updated = [p + 0.1 for p in params]
            return updated, 100, {"loss": 0.5}
        client.fit = mock_fit
        client.get_parameters = MagicMock(return_value=[np.zeros(5)])
        client.evaluate = MagicMock(return_value=(0.5, 100, {"f1": 0.8}))
        return client

    def test_construction(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(epsilon=1.0))
        assert dp.config.epsilon == 1.0
        assert dp.noise_multiplier > 0

    def test_explicit_noise_multiplier(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(noise_multiplier=2.0))
        assert dp.noise_multiplier == 2.0

    def test_clip_parameters(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(max_grad_norm=0.5))
        ref = [np.zeros(5)]
        updated = [np.ones(5) * 10.0]  # large update
        clipped = dp.clip_parameters(updated, ref)
        # L2 norm of delta should be <= max_grad_norm
        delta = clipped[0] - ref[0]
        norm = float(np.linalg.norm(delta))
        assert norm <= 0.5 + 1e-6

    def test_clip_no_clip_needed(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(max_grad_norm=100.0))
        ref = [np.zeros(5)]
        updated = [np.ones(5) * 0.01]  # tiny update
        clipped = dp.clip_parameters(updated, ref)
        np.testing.assert_array_almost_equal(clipped[0], updated[0])

    def test_add_noise_changes_params(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(noise_multiplier=1.0))
        params = [np.zeros(100)]
        noisy = dp.add_noise(params)
        # Noise should make them non-zero
        assert not np.allclose(noisy[0], 0.0)

    def test_fit_returns_noisy_params(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(noise_multiplier=1.0))
        params = [np.zeros(50)]
        result_params, n, metrics = dp.fit(params)
        assert n == 100
        assert "dp_epsilon_spent" in metrics
        assert "dp_epsilon_remaining" in metrics
        assert metrics["dp_epsilon_spent"] > 0

    def test_fit_updates_budget(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig(epsilon=10.0, noise_multiplier=1.0))
        dp.fit([np.zeros(5)])
        assert dp.budget_tracker.rounds_consumed == 1
        assert dp.budget_tracker.epsilon_spent > 0

    def test_evaluate_delegates(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig())
        loss, n, metrics = dp.evaluate([np.zeros(5)])
        assert loss == 0.5

    def test_get_parameters_delegates(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig())
        params = dp.get_parameters()
        assert isinstance(params, list)

    def test_repr(self) -> None:
        from aortica.federated.dp import DPWrapper, DPConfig
        client = self._make_mock_client()
        dp = DPWrapper(client, DPConfig())
        assert "DPWrapper" in repr(dp)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:

    def test_import_dp_module(self) -> None:
        import aortica.federated.dp  # noqa: F401

    def test_import_from_package(self) -> None:
        from aortica.federated import DPWrapper, DPConfig  # noqa: F401

    def test_import_tracker(self) -> None:
        from aortica.federated import PrivacyBudgetTracker  # noqa: F401
