"""Flower-based federated learning server for collaborative model training.

Provides ``FLServer``, a thin wrapper around Flower's ``start_server()`` that
configures FedAvg aggregation, YAML-loadable server config, and per-round
metric logging.

Example usage::

    from aortica.federated import FLServer, FLServerConfig

    config = FLServerConfig.from_yaml("fl_config.yaml")
    server = FLServer(config)
    history = server.start()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import flwr  # noqa: F401

    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False


def _check_flower() -> None:
    """Raise ``ImportError`` if Flower is not installed."""
    if not HAS_FLOWER:
        raise ImportError(
            "Flower is required for federated learning. "
            "Install with: pip install 'aortica[federated]'"
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Metrics collected for a single federated training round.

    Attributes:
        round_number: 1-indexed round number.
        loss: Aggregated loss for the round (may be ``None`` if not reported).
        metrics: Arbitrary per-task metrics dict (e.g. ``{"rhythm_f1": 0.87}``).
        num_clients: Number of clients that participated in this round.
    """

    round_number: int
    loss: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)
    num_clients: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)


@dataclass
class FLServerConfig:
    """Configuration for the Flower federated learning server.

    Attributes:
        num_rounds: Number of federated training rounds.
        min_fit_clients: Minimum clients required for a fit round.
        min_evaluate_clients: Minimum clients required for an evaluate round.
        min_available_clients: Minimum clients that must be connected before
            the server starts a round.
        server_address: ``host:port`` address the Flower server listens on.
        strategy: Aggregation strategy name (``"fedavg"`` supported out of
            the box; future: ``"fedprox"``, ``"scaffold"``).
        fraction_fit: Fraction of available clients sampled for fit.
        fraction_evaluate: Fraction of available clients sampled for evaluate.
        log_dir: Directory for per-round metric JSON logs. ``None`` disables
            file logging.
    """

    num_rounds: int = 5
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    server_address: str = "0.0.0.0:8080"
    strategy: str = "fedavg"
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    log_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if self.min_fit_clients < 1:
            raise ValueError("min_fit_clients must be >= 1")
        if self.min_evaluate_clients < 1:
            raise ValueError("min_evaluate_clients must be >= 1")
        if self.min_available_clients < 1:
            raise ValueError("min_available_clients must be >= 1")
        if not self.server_address:
            raise ValueError("server_address must not be empty")
        if self.strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{self.strategy}'. "
                f"Supported: {sorted(_SUPPORTED_STRATEGIES)}"
            )
        if not 0.0 < self.fraction_fit <= 1.0:
            raise ValueError("fraction_fit must be in (0, 1]")
        if not 0.0 < self.fraction_evaluate <= 1.0:
            raise ValueError("fraction_evaluate must be in (0, 1]")

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (JSON-serialisable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FLServerConfig:
        """Create from a plain dict."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file.

        Uses ``yaml.safe_dump`` when available, otherwise writes a simple
        key: value format.
        """
        data = self.to_dict()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml  # type: ignore[import-untyped]

            with open(path_obj, "w") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        except ImportError:
            lines: list[str] = []
            for key, val in data.items():
                if val is None:
                    lines.append(f"{key}: null")
                elif isinstance(val, bool):
                    lines.append(f"{key}: {'true' if val else 'false'}")
                elif isinstance(val, str):
                    lines.append(f"{key}: \"{val}\"")
                else:
                    lines.append(f"{key}: {val}")
            path_obj.write_text("\n".join(lines) + "\n")

    @classmethod
    def from_yaml(cls, path: str | Path) -> FLServerConfig:
        """Load config from a YAML file.

        Uses ``yaml.safe_load`` when available, otherwise parses simple
        key: value lines (sufficient for flat config).
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")

        text = path_obj.read_text()

        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(text)
        except ImportError:
            data = _simple_yaml_load(text)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_STRATEGIES = frozenset({"fedavg", "fedprox", "scaffold"})

DEFAULT_SERVER_CONFIG = FLServerConfig()


# ---------------------------------------------------------------------------
# Simple YAML fallback parser
# ---------------------------------------------------------------------------


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse a flat YAML file without requiring PyYAML."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, raw_val = stripped.partition(":")
        key = key.strip()
        val_str = raw_val.strip()

        # Remove surrounding quotes
        if len(val_str) >= 2 and val_str[0] == val_str[-1] and val_str[0] in ('"', "'"):
            result[key] = val_str[1:-1]
        elif val_str.lower() == "null" or val_str == "~" or val_str == "":
            result[key] = None
        elif val_str.lower() == "true":
            result[key] = True
        elif val_str.lower() == "false":
            result[key] = False
        else:
            # Try int, then float, then keep as string
            try:
                result[key] = int(val_str)
            except ValueError:
                try:
                    result[key] = float(val_str)
                except ValueError:
                    result[key] = val_str
    return result


# ---------------------------------------------------------------------------
# FLServer
# ---------------------------------------------------------------------------


class FLServer:
    """Flower-based federated learning server.

    Wraps :func:`flwr.server.start_server` with:
    - Configurable number of rounds, min clients, and aggregation strategy.
    - FedAvg strategy out of the box.
    - Per-round metric logging to structured ``RoundMetrics`` objects.
    - Optional JSON metric file logging.
    - Optional SQLite persistence via :class:`FLMetricsStore` for
      dashboard consumption (US-113).

    Args:
        config: Server configuration. Defaults to ``FLServerConfig()`` with
            sensible defaults (5 rounds, 2 min clients, FedAvg, port 8080).
        metrics_store: Optional :class:`FLMetricsStore` for persisting
            per-round metrics to SQLite for the monitoring dashboard.

    Example::

        server = FLServer(FLServerConfig(num_rounds=10))
        history = server.start()
        for rm in server.round_metrics:
            print(rm.round_number, rm.loss)
    """

    def __init__(
        self,
        config: Optional[FLServerConfig] = None,
        metrics_store: Any = None,
    ) -> None:
        self._config = config or FLServerConfig()
        self._round_metrics: list[RoundMetrics] = []
        self._history: Any = None
        self._metrics_store = metrics_store

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> FLServerConfig:
        """Return the server configuration."""
        return self._config

    @property
    def round_metrics(self) -> list[RoundMetrics]:
        """Return collected per-round metrics."""
        return list(self._round_metrics)

    @property
    def history(self) -> Any:
        """Return the raw Flower ``History`` object from the last run."""
        return self._history

    # -- Strategy builder ----------------------------------------------------

    def _build_strategy(self) -> Any:
        """Build the Flower aggregation strategy from config."""
        _check_flower()
        import flwr as fl

        if self._config.strategy == "fedavg":
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=self._config.fraction_fit,
                fraction_evaluate=self._config.fraction_evaluate,
                min_fit_clients=self._config.min_fit_clients,
                min_evaluate_clients=self._config.min_evaluate_clients,
                min_available_clients=self._config.min_available_clients,
            )
        elif self._config.strategy == "fedprox":
            from aortica.federated.strategies import FedProxStrategy

            proxy = FedProxStrategy(
                fraction_fit=self._config.fraction_fit,
                fraction_evaluate=self._config.fraction_evaluate,
                min_fit_clients=self._config.min_fit_clients,
                min_evaluate_clients=self._config.min_evaluate_clients,
                min_available_clients=self._config.min_available_clients,
            )
            strategy = proxy.build()
        elif self._config.strategy == "scaffold":
            from aortica.federated.strategies import SCAFFOLDStrategy

            scaffold = SCAFFOLDStrategy(
                fraction_fit=self._config.fraction_fit,
                fraction_evaluate=self._config.fraction_evaluate,
                min_fit_clients=self._config.min_fit_clients,
                min_evaluate_clients=self._config.min_evaluate_clients,
                min_available_clients=self._config.min_available_clients,
            )
            strategy = scaffold.build()
        else:
            raise ValueError(f"Unsupported strategy: {self._config.strategy}")

        return strategy

    # -- Start ---------------------------------------------------------------

    def start(self) -> Any:
        """Start the Flower server and block until all rounds complete.

        Returns:
            The Flower ``History`` object containing per-round metrics.

        Raises:
            ImportError: If Flower is not installed.
            ValueError: If the strategy is unsupported.
        """
        _check_flower()
        import flwr as fl

        strategy = self._build_strategy()

        logger.info(
            "Starting FL server on %s — %d rounds, strategy=%s, "
            "min_fit=%d, min_eval=%d",
            self._config.server_address,
            self._config.num_rounds,
            self._config.strategy,
            self._config.min_fit_clients,
            self._config.min_evaluate_clients,
        )

        server_config = fl.server.ServerConfig(
            num_rounds=self._config.num_rounds,
        )

        # Start campaign in metrics store if available
        if self._metrics_store is not None:
            self._metrics_store.start_campaign(
                name=f"FL-{self._config.strategy}",
                total_rounds=self._config.num_rounds,
                strategy=self._config.strategy,
            )

        history = fl.server.start_server(
            server_address=self._config.server_address,
            config=server_config,
            strategy=strategy,
        )

        self._history = history
        self._round_metrics = self._extract_round_metrics(history)

        # Persist rounds to metrics store for dashboard consumption
        if self._metrics_store is not None:
            for rm in self._round_metrics:
                self._metrics_store.record_round(
                    round_number=rm.round_number,
                    loss=rm.loss,
                    metrics=rm.metrics,
                    num_clients=rm.num_clients,
                )
            self._metrics_store.complete_campaign()

        # Log metrics to files if configured
        if self._config.log_dir:
            self._write_metrics_log(self._config.log_dir)

        return history

    # -- Metric extraction ---------------------------------------------------

    @staticmethod
    def _extract_round_metrics(history: Any) -> list[RoundMetrics]:
        """Extract per-round metrics from a Flower ``History`` object.

        Flower stores metrics as ``{metric_name: [(round, value), ...]}``.
        This method pivots that into per-round ``RoundMetrics`` objects.
        """
        rounds_map: dict[int, RoundMetrics] = {}

        # Centralised (aggregated) losses
        losses_centralized: list[tuple[int, float]] = getattr(
            history, "losses_centralized", []
        )
        for rnd, loss_val in losses_centralized:
            rm = rounds_map.setdefault(rnd, RoundMetrics(round_number=rnd))
            rm.loss = loss_val

        # Distributed losses
        losses_distributed: list[tuple[int, float]] = getattr(
            history, "losses_distributed", []
        )
        for rnd, loss_val in losses_distributed:
            if rnd not in rounds_map:
                rounds_map[rnd] = RoundMetrics(round_number=rnd)
            rm = rounds_map[rnd]
            if rm.loss is None:
                rm.loss = loss_val

        # Distributed fit metrics
        metrics_distributed_fit: dict[str, list[tuple[int, Any]]] = getattr(
            history, "metrics_distributed_fit", {}
        )
        for metric_name, values in metrics_distributed_fit.items():
            for rnd, val in values:
                rm = rounds_map.setdefault(rnd, RoundMetrics(round_number=rnd))
                if isinstance(val, (int, float)):
                    rm.metrics[metric_name] = float(val)

        # Distributed evaluate metrics
        metrics_distributed: dict[str, list[tuple[int, Any]]] = getattr(
            history, "metrics_distributed", {}
        )
        for metric_name, values in metrics_distributed.items():
            for rnd, val in values:
                rm = rounds_map.setdefault(rnd, RoundMetrics(round_number=rnd))
                if isinstance(val, (int, float)):
                    rm.metrics[metric_name] = float(val)

        # Centralised metrics
        metrics_centralized: dict[str, list[tuple[int, Any]]] = getattr(
            history, "metrics_centralized", {}
        )
        for metric_name, values in metrics_centralized.items():
            for rnd, val in values:
                rm = rounds_map.setdefault(rnd, RoundMetrics(round_number=rnd))
                if isinstance(val, (int, float)):
                    rm.metrics[metric_name] = float(val)

        return sorted(rounds_map.values(), key=lambda rm: rm.round_number)

    # -- Logging -------------------------------------------------------------

    def _write_metrics_log(self, log_dir: str) -> None:
        """Write per-round metrics to JSON files in ``log_dir``."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Write individual round files
        for rm in self._round_metrics:
            fpath = log_path / f"round_{rm.round_number:04d}.json"
            fpath.write_text(json.dumps(rm.to_dict(), indent=2) + "\n")

        # Write summary
        summary_path = log_path / "summary.json"
        summary: dict[str, Any] = {
            "config": self._config.to_dict(),
            "num_rounds_completed": len(self._round_metrics),
            "rounds": [rm.to_dict() for rm in self._round_metrics],
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        logger.info("Metrics written to %s (%d round files)", log_path, len(self._round_metrics))

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FLServer(strategy={self._config.strategy!r}, "
            f"num_rounds={self._config.num_rounds}, "
            f"address={self._config.server_address!r})"
        )
