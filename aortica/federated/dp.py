"""Differential privacy wrapper for federated learning clients.

Provides :class:`DPWrapper` which wraps an :class:`AorticaFlowerClient`
with per-round gradient clipping and Gaussian noise injection for
(ε, δ)-differential privacy.

Uses Rényi Differential Privacy (RDP) composition for tight privacy
accounting across multiple training rounds.

Example::

    from aortica.federated import AorticaFlowerClient, FLClientConfig
    from aortica.federated.dp import DPWrapper, DPConfig

    client = AorticaFlowerClient(FLClientConfig(data_path="/data"))
    dp = DPWrapper(client, DPConfig(epsilon=1.0, delta=1e-5))
    dp_params, n, metrics = dp.fit(server_params)
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Configuration for differential privacy.

    Attributes:
        epsilon: Total privacy budget ε.  Default ``1.0``.
        delta: Privacy parameter δ.  Default ``1e-5``.
        max_grad_norm: Maximum L2 norm for per-sample gradient clipping.
            Default ``1.0``.
        noise_multiplier: Ratio of noise std to clipping norm.  If ``None``,
            it is automatically calibrated from ``epsilon``, ``delta``,
            and expected number of rounds.
        expected_rounds: Expected total number of FL rounds (used for
            automatic noise calibration).  Default ``10``.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    expected_rounds: int = 10

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if self.noise_multiplier is not None and self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be >= 0")
        if self.expected_rounds < 1:
            raise ValueError("expected_rounds must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Privacy accounting (simplified RDP composition)
# ---------------------------------------------------------------------------


def _compute_rdp_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    delta: float,
) -> float:
    """Compute (ε, δ)-DP guarantee via simplified RDP composition.

    Uses the moments accountant approach: for Gaussian mechanism with
    noise σ, each round contributes RDP at order α of α/(2σ²).
    The total RDP after T rounds is T * α/(2σ²).
    Converting RDP to (ε,δ)-DP: ε = rdp - log(δ)/(α-1).

    Args:
        noise_multiplier: Noise standard deviation relative to sensitivity.
        num_rounds: Number of composition rounds.
        delta: Target δ.

    Returns:
        The ε value of the (ε, δ)-DP guarantee.
    """
    if noise_multiplier <= 0:
        return float("inf")

    best_eps = float("inf")
    # Search over RDP orders
    for alpha in [1.5, 2, 3, 4, 5, 8, 16, 32, 64]:
        rdp = alpha / (2 * noise_multiplier**2)
        total_rdp = num_rounds * rdp
        eps = total_rdp + math.log(1 / delta) / (alpha - 1)
        best_eps = min(best_eps, eps)

    return best_eps


def _calibrate_noise_multiplier(
    epsilon: float,
    delta: float,
    num_rounds: int,
) -> float:
    """Binary search for noise_multiplier achieving target (ε, δ)-DP.

    Args:
        epsilon: Target ε.
        delta: Target δ.
        num_rounds: Expected number of rounds.

    Returns:
        The noise multiplier σ/sensitivity.
    """
    lo, hi = 0.01, 100.0

    for _ in range(64):  # binary search iterations
        mid = (lo + hi) / 2.0
        achieved_eps = _compute_rdp_epsilon(mid, num_rounds, delta)
        if achieved_eps <= epsilon:
            hi = mid
        else:
            lo = mid

    return hi


# ---------------------------------------------------------------------------
# Privacy budget tracker
# ---------------------------------------------------------------------------


class PrivacyBudgetTracker:
    """Tracks cumulative privacy expenditure across FL rounds.

    Attributes:
        total_epsilon: The total privacy budget.
        delta: The δ parameter.
        noise_multiplier: Calibrated noise multiplier.
    """

    def __init__(
        self,
        total_epsilon: float,
        delta: float,
        noise_multiplier: float,
    ) -> None:
        self._total_epsilon = total_epsilon
        self._delta = delta
        self._noise_multiplier = noise_multiplier
        self._rounds_consumed: int = 0

    @property
    def total_epsilon(self) -> float:
        """Return the total privacy budget."""
        return self._total_epsilon

    @property
    def rounds_consumed(self) -> int:
        """Return the number of rounds consumed."""
        return self._rounds_consumed

    @property
    def epsilon_spent(self) -> float:
        """Return the ε spent so far."""
        if self._rounds_consumed == 0:
            return 0.0
        return _compute_rdp_epsilon(
            self._noise_multiplier, self._rounds_consumed, self._delta
        )

    @property
    def epsilon_remaining(self) -> float:
        """Return the remaining ε budget."""
        return max(0.0, self._total_epsilon - self.epsilon_spent)

    @property
    def is_exhausted(self) -> bool:
        """Return True if the privacy budget is exhausted."""
        return self.epsilon_spent >= self._total_epsilon

    def consume_round(self) -> float:
        """Record one round of privacy expenditure.

        Returns:
            The cumulative ε spent after this round.

        Raises:
            RuntimeError: If the privacy budget is already exhausted.
        """
        if self.is_exhausted:
            raise RuntimeError(
                f"Privacy budget exhausted: ε_spent={self.epsilon_spent:.4f} "
                f">= ε_total={self._total_epsilon:.4f}"
            )

        self._rounds_consumed += 1
        spent = self.epsilon_spent

        # Warn at 80% expenditure
        if spent > 0.8 * self._total_epsilon:
            logger.warning(
                "Privacy budget %.1f%% consumed: ε_spent=%.4f / ε_total=%.4f",
                100 * spent / self._total_epsilon,
                spent,
                self._total_epsilon,
            )

        return spent

    def summary(self) -> Dict[str, float]:
        """Return a summary of privacy expenditure."""
        return {
            "epsilon_total": self._total_epsilon,
            "epsilon_spent": self.epsilon_spent,
            "epsilon_remaining": self.epsilon_remaining,
            "delta": self._delta,
            "noise_multiplier": self._noise_multiplier,
            "rounds_consumed": float(self._rounds_consumed),
        }


# ---------------------------------------------------------------------------
# DPWrapper
# ---------------------------------------------------------------------------


class DPWrapper:
    """Differential privacy wrapper for an AorticaFlowerClient.

    Wraps the client's ``fit()`` method to apply:

    1. **Gradient clipping**: Each parameter update is clipped to
       ``max_grad_norm`` L2 norm.
    2. **Gaussian noise injection**: IID Gaussian noise with calibrated
       standard deviation is added to clipped updates.

    Privacy accounting uses simplified RDP composition.

    Args:
        client: The underlying :class:`AorticaFlowerClient`.
        config: Differential privacy configuration.

    Example::

        dp = DPWrapper(client, DPConfig(epsilon=1.0))
        params, n, metrics = dp.fit(server_params)
        print(dp.budget_tracker.epsilon_spent)
    """

    def __init__(
        self,
        client: Any,
        config: Optional[DPConfig] = None,
    ) -> None:
        self._client = client
        self._config = config or DPConfig()

        # Calibrate noise if not explicitly set
        if self._config.noise_multiplier is not None:
            noise_mult = self._config.noise_multiplier
        else:
            noise_mult = _calibrate_noise_multiplier(
                self._config.epsilon,
                self._config.delta,
                self._config.expected_rounds,
            )

        self._noise_multiplier = noise_mult
        self._budget_tracker = PrivacyBudgetTracker(
            total_epsilon=self._config.epsilon,
            delta=self._config.delta,
            noise_multiplier=noise_mult,
        )

    @property
    def config(self) -> DPConfig:
        """Return the DP configuration."""
        return self._config

    @property
    def noise_multiplier(self) -> float:
        """Return the calibrated noise multiplier."""
        return self._noise_multiplier

    @property
    def budget_tracker(self) -> PrivacyBudgetTracker:
        """Return the privacy budget tracker."""
        return self._budget_tracker

    def clip_parameters(
        self, parameters: List[np.ndarray], reference: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Clip parameter deltas to max_grad_norm L2 norm.

        Args:
            parameters: Updated model parameters.
            reference: Reference (global) parameters before local training.

        Returns:
            Clipped parameters.
        """
        # Compute delta
        deltas = [p - r for p, r in zip(parameters, reference)]

        # Compute global L2 norm of all deltas
        total_norm = math.sqrt(
            sum(float(np.sum(d**2)) for d in deltas)
        )

        # Clip if necessary
        clip_factor = min(1.0, self._config.max_grad_norm / max(total_norm, 1e-8))

        clipped = [r + d * clip_factor for r, d in zip(reference, deltas)]
        return clipped

    def add_noise(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Add calibrated Gaussian noise to parameters.

        Args:
            parameters: Model parameters (already clipped).

        Returns:
            Noisy parameters.
        """
        noise_std = self._noise_multiplier * self._config.max_grad_norm
        noisy = []
        for p in parameters:
            noise = np.random.normal(0, noise_std, size=p.shape).astype(p.dtype)
            noisy.append(p + noise)
        return noisy

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Run local training with DP guarantees.

        1. Delegates to the wrapped client's ``fit()``
        2. Clips parameter updates
        3. Adds calibrated noise
        4. Updates the privacy budget tracker

        Args:
            parameters: Global model parameters from the server.
            config: Optional per-round config from the server.

        Returns:
            Tuple of (dp_parameters, num_examples, metrics_with_dp_info).
        """
        # Record the reference (global) parameters
        reference = [p.copy() for p in parameters]

        # Run local training
        updated_params, num_examples, metrics = self._client.fit(
            parameters, config
        )

        # Apply DP: clip + noise
        clipped = self.clip_parameters(updated_params, reference)
        noisy = self.add_noise(clipped)

        # Update budget
        eps_spent = self._budget_tracker.consume_round()
        metrics["dp_epsilon_spent"] = eps_spent
        metrics["dp_epsilon_remaining"] = self._budget_tracker.epsilon_remaining
        metrics["dp_noise_multiplier"] = self._noise_multiplier

        return noisy, num_examples, metrics

    def get_parameters(
        self, config: Optional[Dict[str, Any]] = None
    ) -> List[np.ndarray]:
        """Delegate to the wrapped client."""
        return self._client.get_parameters(config)

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        """Delegate to the wrapped client (no DP needed for evaluation)."""
        return self._client.evaluate(parameters, config)

    def __repr__(self) -> str:
        return (
            f"DPWrapper(ε={self._config.epsilon}, "
            f"δ={self._config.delta}, "
            f"σ={self._noise_multiplier:.4f})"
        )
