"""Pluggable aggregation strategies for federated learning.

Provides :class:`FedProxStrategy` and :class:`SCAFFOLDStrategy` as
drop-in alternatives to Flower's built-in ``FedAvg``.

- **FedProx** adds a proximal term ``(μ/2) * ||w - w_global||²`` to the
  client loss, improving convergence on heterogeneous (non-IID) data.
- **SCAFFOLD** uses server/client control variates to correct for client
  drift, providing variance reduction across heterogeneous sites.

Both strategies are selectable via the server config YAML::

    strategy: fedprox   # or: scaffold

Example::

    from aortica.federated.strategies import FedProxStrategy

    strategy = FedProxStrategy(mu=0.01, min_fit_clients=2)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import flwr  # noqa: F401
    from flwr.common import (
        FitIns,
        FitRes,
        MetricsAggregationFn,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg

    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False


def _check_flower() -> None:
    """Raise ``ImportError`` if Flower is not installed."""
    if not HAS_FLOWER:
        raise ImportError(
            "Flower is required for aggregation strategies. "
            "Install with: pip install 'aortica[federated]'"
        )


# ---------------------------------------------------------------------------
# FedProx Strategy
# ---------------------------------------------------------------------------


class FedProxStrategy:
    """FedProx aggregation strategy with configurable proximal term.

    Extends FedAvg by injecting the proximal penalty coefficient ``μ``
    into each client's fit configuration so that the client-side training
    loop can apply the regularisation term.

    The proximal term is ``(μ/2) * ||w - w_global||²`` where ``w_global``
    are the aggregated parameters from the previous round.

    Args:
        mu: Proximal term coefficient.  Default ``0.01``.
        fraction_fit: Fraction of clients sampled per fit round.
        fraction_evaluate: Fraction of clients sampled per evaluate round.
        min_fit_clients: Minimum clients required for fit.
        min_evaluate_clients: Minimum clients required for evaluate.
        min_available_clients: Minimum available clients before starting.
        fit_metrics_aggregation_fn: Optional custom metric aggregation.
        evaluate_metrics_aggregation_fn: Optional custom metric aggregation.

    Example::

        strategy = FedProxStrategy(mu=0.05, min_fit_clients=3)
        server = FLServer(config)
        # Pass strategy to Flower server
    """

    def __init__(
        self,
        mu: float = 0.01,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fit_metrics_aggregation_fn: Optional[Any] = None,
        evaluate_metrics_aggregation_fn: Optional[Any] = None,
    ) -> None:
        if mu < 0:
            raise ValueError("mu must be >= 0")
        self._mu = mu
        self._fraction_fit = fraction_fit
        self._fraction_evaluate = fraction_evaluate
        self._min_fit_clients = min_fit_clients
        self._min_evaluate_clients = min_evaluate_clients
        self._min_available_clients = min_available_clients
        self._fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self._evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self._flower_strategy: Any = None

    @property
    def mu(self) -> float:
        """Return the proximal term coefficient."""
        return self._mu

    def build(self) -> Any:
        """Build and return the underlying Flower strategy.

        Returns:
            A Flower ``Strategy`` instance (``FedAvg`` subclass with
            proximal config injection).

        Raises:
            ImportError: If Flower is not installed.
        """
        _check_flower()

        outer = self

        class _FedProxFlower(FedAvg):  # type: ignore[misc]
            """FedAvg subclass injecting the proximal μ into fit configs."""

            def configure_fit(
                self,
                server_round: int,
                parameters: Parameters,
                client_manager: Any,
            ) -> List[Tuple[ClientProxy, FitIns]]:
                """Inject ``proximal_mu`` into each client's fit config."""
                configs = super().configure_fit(
                    server_round, parameters, client_manager
                )
                augmented: List[Tuple[ClientProxy, FitIns]] = []
                for client_proxy, fit_ins in configs:
                    config = dict(fit_ins.config)
                    config["proximal_mu"] = outer._mu
                    augmented.append(
                        (client_proxy, FitIns(fit_ins.parameters, config))
                    )
                return augmented

        strategy = _FedProxFlower(
            fraction_fit=self._fraction_fit,
            fraction_evaluate=self._fraction_evaluate,
            min_fit_clients=self._min_fit_clients,
            min_evaluate_clients=self._min_evaluate_clients,
            min_available_clients=self._min_available_clients,
        )
        self._flower_strategy = strategy
        return strategy

    def __repr__(self) -> str:
        return f"FedProxStrategy(mu={self._mu})"


# ---------------------------------------------------------------------------
# SCAFFOLD Strategy
# ---------------------------------------------------------------------------


class SCAFFOLDStrategy:
    """SCAFFOLD aggregation strategy with control variates.

    Implements the SCAFFOLD algorithm (Karimireddy et al., 2020) which
    uses server and client control variates to correct for client drift
    in heterogeneous federated settings.

    The server maintains a global control variate ``c`` and tracks
    previous global weights to compute per-client control variate
    deltas ``Δc_i`` using the SCAFFOLD update rule::

        Δc_i = (w_global_prev - w_i_new) / (K * η) - c
        c_new = c + (1/N) * Σ(Δc_i)

    The server also injects ``scaffold_round`` into each client's fit
    config so that clients can apply the control variate correction
    during local training.

    Args:
        fraction_fit: Fraction of clients sampled per fit round.
        fraction_evaluate: Fraction of clients sampled per evaluate round.
        min_fit_clients: Minimum clients required for fit.
        min_evaluate_clients: Minimum clients required for evaluate.
        min_available_clients: Minimum available clients before starting.
        learning_rate: Server learning rate for control variate updates.

    Example::

        strategy = SCAFFOLDStrategy(min_fit_clients=3)
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        learning_rate: float = 1.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self._fraction_fit = fraction_fit
        self._fraction_evaluate = fraction_evaluate
        self._min_fit_clients = min_fit_clients
        self._min_evaluate_clients = min_evaluate_clients
        self._min_available_clients = min_available_clients
        self._learning_rate = learning_rate
        self._server_control: Optional[List[np.ndarray]] = None
        self._previous_global: Optional[List[np.ndarray]] = None
        self._flower_strategy: Any = None

    @property
    def learning_rate(self) -> float:
        """Return the server learning rate."""
        return self._learning_rate

    @property
    def server_control(self) -> Optional[List[np.ndarray]]:
        """Return the current server control variate (None before first round)."""
        return self._server_control

    def build(self) -> Any:
        """Build and return the underlying Flower strategy.

        Returns:
            A Flower ``Strategy`` instance implementing SCAFFOLD.

        Raises:
            ImportError: If Flower is not installed.
        """
        _check_flower()

        outer = self

        class _SCAFFOLDFlower(FedAvg):  # type: ignore[misc]
            """FedAvg subclass implementing SCAFFOLD aggregation."""

            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Aggregate fit results with SCAFFOLD control variate correction."""
                if not results:
                    return None, {}

                # Extract weights from results
                weights_results: List[Tuple[List[np.ndarray], int]] = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]

                # Initialise server control variate on first round
                if outer._server_control is None:
                    first_weights = weights_results[0][0]
                    outer._server_control = [
                        np.zeros_like(w) for w in first_weights
                    ]

                # Weighted average of client updates (standard FedAvg)
                total_examples = sum(n for _, n in weights_results)
                if total_examples == 0:
                    return None, {}

                averaged: List[np.ndarray] = [
                    np.zeros_like(w) for w in weights_results[0][0]
                ]
                for client_weights, num_ex in weights_results:
                    weight = num_ex / total_examples
                    for i, w in enumerate(client_weights):
                        averaged[i] += w * weight

                # SCAFFOLD control variate update:
                # Δc_i = (w_global_prev - w_i_new) / (K * η) - c
                # c_new = c + (1/N) * Σ Δc_i
                # We use learning_rate as the (K * η) proxy.
                n_clients = len(results)
                if outer._previous_global is not None:
                    for i in range(len(outer._server_control)):
                        delta_c_sum = np.zeros_like(outer._server_control[i])
                        for client_weights, _ in weights_results:
                            delta_c_i = (
                                (outer._previous_global[i] - client_weights[i])
                                / outer._learning_rate
                            ) - outer._server_control[i]
                            delta_c_sum += delta_c_i
                        outer._server_control[i] += (
                            delta_c_sum / max(n_clients, 1)
                        )

                # Store current global weights for next round's Δc_i
                outer._previous_global = [w.copy() for w in averaged]

                # Apply control variate correction to averaged weights
                corrected: List[np.ndarray] = []
                for w, c in zip(averaged, outer._server_control):
                    corrected.append(w - outer._learning_rate * c)

                # Aggregate metrics
                metrics_aggregated: Dict[str, Scalar] = {}
                if results:
                    for _, fit_res in results:
                        for k, v in fit_res.metrics.items():
                            if isinstance(v, (int, float)):
                                metrics_aggregated[k] = (
                                    metrics_aggregated.get(k, 0.0) + float(v)
                                )
                    n = len(results)
                    metrics_aggregated = {
                        k: v / n  # type: ignore[operator]
                        for k, v in metrics_aggregated.items()
                    }

                return ndarrays_to_parameters(corrected), metrics_aggregated

            def configure_fit(
                self,
                server_round: int,
                parameters: Parameters,
                client_manager: Any,
            ) -> List[Tuple[ClientProxy, FitIns]]:
                """Inject SCAFFOLD metadata into each client's fit config."""
                configs = super().configure_fit(
                    server_round, parameters, client_manager
                )
                augmented: List[Tuple[ClientProxy, FitIns]] = []
                for client_proxy, fit_ins in configs:
                    config = dict(fit_ins.config)
                    config["scaffold_round"] = server_round
                    augmented.append(
                        (client_proxy, FitIns(fit_ins.parameters, config))
                    )
                return augmented

        strategy = _SCAFFOLDFlower(
            fraction_fit=self._fraction_fit,
            fraction_evaluate=self._fraction_evaluate,
            min_fit_clients=self._min_fit_clients,
            min_evaluate_clients=self._min_evaluate_clients,
            min_available_clients=self._min_available_clients,
        )
        self._flower_strategy = strategy
        return strategy

    def __repr__(self) -> str:
        return f"SCAFFOLDStrategy(lr={self._learning_rate})"
