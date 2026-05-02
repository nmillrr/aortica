"""Flower federated learning client for site-local training.

Provides :class:`AorticaFlowerClient`, a Flower ``NumPyClient`` wrapper
that runs local training and evaluation on site-local ECG data using the
Aortica multi-task training pipeline.

Example usage::

    from aortica.federated import AorticaFlowerClient, FLClientConfig

    config = FLClientConfig(data_path="/path/to/local/data")
    client = AorticaFlowerClient(config)
    client.start(server_address="localhost:8080")
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import flwr  # noqa: F401

    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    import types

    torch = types.ModuleType("torch")  # type: ignore[assignment]
    nn = types.ModuleType("nn")  # type: ignore[assignment]

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn.Module = _DummyModule  # type: ignore[attr-defined]


def _check_flower() -> None:
    """Raise ``ImportError`` if Flower is not installed."""
    if not HAS_FLOWER:
        raise ImportError(
            "Flower is required for federated learning. "
            "Install with: pip install 'aortica[federated]'"
        )


def _check_torch() -> None:
    """Raise ``ImportError`` if PyTorch is not installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for federated learning client. "
            "Install with: pip install 'aortica[torch]'"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FLClientConfig:
    """Configuration for the Flower federated learning client.

    Attributes:
        data_path: Path to site-local ECG dataset directory.
        server_address: ``host:port`` of the Flower FL server.
        local_epochs: Number of local training epochs per round.
        batch_size: Training batch size.
        lr: Learning rate for local training.
        sampling_rate: ECG sampling rate (100 or 500 Hz).
        window_seconds: Signal window length in seconds.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping max norm.
        enabled_tasks: Which task heads to train.
        feature_dim: Backbone feature dimension.
        head_hidden_dim: Task head hidden dimension.
        head_dropout: Task head dropout rate.
        seed: Random seed for reproducibility.
        base_checkpoint: Path to a local checkpoint to initialise from.
            If ``None``, uses ``load_pretrained('latest')``.
    """

    data_path: str = ""
    server_address: str = "localhost:8080"
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    sampling_rate: int = 500
    window_seconds: float = 10.0
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    enabled_tasks: List[str] = field(
        default_factory=lambda: ["rhythm", "structural", "ischaemia", "risk"]
    )
    feature_dim: int = 256
    head_hidden_dim: int = 128
    head_dropout: float = 0.3
    seed: int = 42
    base_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not self.server_address:
            raise ValueError("server_address must not be empty")

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (JSON-serialisable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FLClientConfig:
        """Create from a plain dict, ignoring unknown keys."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Task output dimensions
# ---------------------------------------------------------------------------

_TASK_NUM_OUTPUTS: Dict[str, int] = {
    "rhythm": 22,
    "structural": 15,
    "ischaemia": 10,
    "risk": 3,
}


# ---------------------------------------------------------------------------
# Helper: split concatenated labels
# ---------------------------------------------------------------------------


def _split_labels(
    labels: "torch.Tensor",
    enabled_tasks: List[str],
) -> Dict[str, "torch.Tensor"]:
    """Split a concatenated label tensor into per-task tensors."""
    result: Dict[str, Any] = {}
    offset = 0
    for task in enabled_tasks:
        width = _TASK_NUM_OUTPUTS[task]
        result[task] = labels[:, offset : offset + width]
        offset += width
    return result


# ---------------------------------------------------------------------------
# Helper: compute F1 and C-index (inlined for module independence)
# ---------------------------------------------------------------------------


def _compute_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute macro-F1 from sigmoid outputs."""
    pred_bin = (predictions >= threshold).astype(np.float32)
    num_classes = targets.shape[1]
    f1_scores: List[float] = []

    for c in range(num_classes):
        tp = float(np.sum((pred_bin[:, c] == 1) & (targets[:, c] == 1)))
        fp = float(np.sum((pred_bin[:, c] == 1) & (targets[:, c] == 0)))
        fn = float(np.sum((pred_bin[:, c] == 0) & (targets[:, c] == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _compute_c_index(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute concordance index averaged across risk outputs."""
    n, k = predictions.shape
    if n < 2:
        return 0.5

    c_indices: List[float] = []
    for t in range(k):
        concordant: float = 0.0
        discordant: float = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if targets[i, t] == targets[j, t]:
                    continue
                if targets[i, t] > targets[j, t]:
                    hi, lo = i, j
                else:
                    hi, lo = j, i
                if predictions[hi, t] > predictions[lo, t]:
                    concordant += 1.0
                elif predictions[hi, t] < predictions[lo, t]:
                    discordant += 1.0
                else:
                    concordant += 0.5
                    discordant += 0.5
            total = concordant + discordant
        c_indices.append(concordant / total if total > 0 else 0.5)

    return float(np.mean(c_indices))


# ---------------------------------------------------------------------------
# AorticaFlowerClient
# ---------------------------------------------------------------------------


class AorticaFlowerClient:
    """Flower federated learning client wrapping the Aortica training pipeline.

    Implements the Flower ``NumPyClient`` interface for participating in
    federated training rounds.  On each ``fit()`` call the client runs
    local training on its site-local data and returns updated model weights.

    The client initialises its model via :func:`load_pretrained` by default
    (so FL rounds start from the canonical public checkpoint) unless a
    custom ``base_checkpoint`` is provided in the config.

    Args:
        config: Client configuration.  Defaults to ``FLClientConfig()``.
        model: Optional pre-built ``AorticaModel`` instance.  If ``None``,
            the client will create one from the config or load a pretrained
            checkpoint.
        train_loader: Optional PyTorch DataLoader for training data.  If
            ``None``, the client will attempt to create one from
            ``config.data_path``.
        val_loader: Optional PyTorch DataLoader for evaluation data.

    Example::

        client = AorticaFlowerClient(
            FLClientConfig(data_path="/data/ecg", server_address="fl.example:8080")
        )
        client.start()
    """

    def __init__(
        self,
        config: Optional[FLClientConfig] = None,
        model: Optional[Any] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
    ) -> None:
        self._config = config or FLClientConfig()
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device: Any = None

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> FLClientConfig:
        """Return the client configuration."""
        return self._config

    @property
    def model(self) -> Any:
        """Return the current model (may be ``None`` before initialisation)."""
        return self._model

    # -- Model initialisation ------------------------------------------------

    def _init_model(self) -> None:
        """Initialise the AorticaModel, loading from checkpoint if needed."""
        _check_torch()

        if self._model is not None:
            return

        from aortica.models.aortica_model import AorticaModel

        if self._config.base_checkpoint:
            # Load from explicit local checkpoint
            checkpoint = torch.load(
                self._config.base_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model = AorticaModel(
                    enabled_tasks=self._config.enabled_tasks,
                    feature_dim=self._config.feature_dim,
                    head_hidden_dim=self._config.head_hidden_dim,
                    head_dropout=self._config.head_dropout,
                )
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model = AorticaModel(
                    enabled_tasks=self._config.enabled_tasks,
                    feature_dim=self._config.feature_dim,
                    head_hidden_dim=self._config.head_hidden_dim,
                    head_dropout=self._config.head_dropout,
                )
                self._model.load_state_dict(checkpoint)
        else:
            # Try load_pretrained, fall back to fresh model
            try:
                from aortica.models.registry import load_pretrained

                self._model = load_pretrained("latest")
                logger.info("Loaded pretrained model for FL initialisation")
            except Exception:
                logger.info(
                    "Could not load pretrained model, creating fresh AorticaModel"
                )
                self._model = AorticaModel(
                    enabled_tasks=self._config.enabled_tasks,
                    feature_dim=self._config.feature_dim,
                    head_hidden_dim=self._config.head_hidden_dim,
                    head_dropout=self._config.head_dropout,
                )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)

    # -- NumPyClient interface -----------------------------------------------

    def get_parameters(
        self, config: Optional[Dict[str, Any]] = None
    ) -> List[np.ndarray]:
        """Return model weights as a list of numpy arrays.

        This method is called by the Flower server to retrieve the current
        model parameters before the first round and after aggregation.

        Args:
            config: Optional configuration dict from the server (unused).

        Returns:
            List of numpy arrays representing model state dict values.
        """
        _check_torch()
        self._init_model()

        params: List[np.ndarray] = []
        for param in self._model.state_dict().values():
            arr: np.ndarray = param.cpu().numpy()
            params.append(arr)
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model weights from a list of numpy arrays.

        Called by the Flower framework after aggregation to update the
        local model with the globally aggregated weights.

        Args:
            parameters: List of numpy arrays (one per state dict entry).
        """
        _check_torch()
        self._init_model()

        state_dict = self._model.state_dict()
        keys = list(state_dict.keys())

        if len(parameters) != len(keys):
            raise ValueError(
                f"Parameter count mismatch: received {len(parameters)}, "
                f"expected {len(keys)}"
            )

        new_state: Dict[str, Any] = {}
        for key, param_array in zip(keys, parameters):
            new_state[key] = torch.tensor(param_array, dtype=state_dict[key].dtype)

        self._model.load_state_dict(new_state)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Run local training and return updated weights.

        Called by the Flower framework for each fit round.  Sets the model
        weights from the server-provided parameters, runs local training
        for ``local_epochs`` on site-local data, and returns the updated
        weights along with the number of training examples and per-task
        metrics.

        Args:
            parameters: Aggregated model weights from the server.
            config: Optional per-round configuration from the server.

        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict).
        """
        _check_torch()
        self._init_model()
        self.set_parameters(parameters)

        # Override local_epochs from server config if provided
        local_epochs = self._config.local_epochs
        if config and "local_epochs" in config:
            local_epochs = int(config["local_epochs"])

        # Train
        num_examples = 0
        total_loss = 0.0
        per_task_losses: Dict[str, float] = {}

        if self._train_loader is not None:
            optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self._config.lr,
                weight_decay=self._config.weight_decay,
            )

            for _epoch in range(local_epochs):
                epoch_loss, epoch_per_task = self._train_one_epoch(optimizer)
                total_loss += epoch_loss
                for k, v in epoch_per_task.items():
                    per_task_losses[k] = per_task_losses.get(k, 0.0) + v

            # Count total training examples
            num_examples = len(self._train_loader.dataset)

            # Average losses over epochs
            if local_epochs > 0:
                total_loss /= local_epochs
                per_task_losses = {
                    k: v / local_epochs for k, v in per_task_losses.items()
                }

        metrics: Dict[str, float] = {"loss": total_loss}
        metrics.update(per_task_losses)

        updated_params = self.get_parameters()
        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        """Run local evaluation and return loss + metrics.

        Called by the Flower framework for each evaluate round.  Sets the
        model weights from the server-provided parameters and evaluates on
        site-local validation data.

        Args:
            parameters: Aggregated model weights from the server.
            config: Optional per-round configuration from the server.

        Returns:
            Tuple of (loss, num_examples, metrics_dict).
        """
        _check_torch()
        self._init_model()
        self.set_parameters(parameters)

        num_examples = 0
        total_loss = 0.0
        metrics: Dict[str, float] = {}

        if self._val_loader is not None:
            total_loss, metrics = self._evaluate_model()
            num_examples = len(self._val_loader.dataset)

        return total_loss, num_examples, metrics

    # -- Training helpers ----------------------------------------------------

    def _train_one_epoch(
        self,
        optimizer: Any,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch and return (avg_loss, per_task_losses)."""
        from aortica.models.ischaemia_head import compute_ischaemia_loss
        from aortica.models.rhythm_head import compute_rhythm_loss
        from aortica.models.risk_head import compute_risk_loss
        from aortica.models.structural_head import compute_structural_loss

        self._model.train()
        running_loss = 0.0
        running_per_task: Dict[str, float] = {}
        num_batches = 0

        for batch_x, batch_y in self._train_loader:
            batch_x = batch_x.to(self._device)
            batch_y = batch_y.to(self._device).float()

            optimizer.zero_grad()

            features = self._model.backbone(batch_x)
            features = self._model.attention(features)

            task_labels = _split_labels(batch_y, self._config.enabled_tasks)

            total_loss = torch.tensor(
                0.0, device=self._device, dtype=features.dtype
            )
            per_task: Dict[str, float] = {}

            if "rhythm" in task_labels and self._model.rhythm_head is not None:
                logits = self._model.rhythm_head.forward_logits(features)
                loss = compute_rhythm_loss(logits, task_labels["rhythm"])
                total_loss = total_loss + loss
                per_task["rhythm"] = loss.item()

            if (
                "structural" in task_labels
                and self._model.structural_head is not None
            ):
                logits = self._model.structural_head.forward_logits(features)
                loss = compute_structural_loss(logits, task_labels["structural"])
                total_loss = total_loss + loss
                per_task["structural"] = loss.item()

            if (
                "ischaemia" in task_labels
                and self._model.ischaemia_head is not None
            ):
                logits = self._model.ischaemia_head.forward_logits(features)
                loss = compute_ischaemia_loss(logits, task_labels["ischaemia"])
                total_loss = total_loss + loss
                per_task["ischaemia"] = loss.item()

            if "risk" in task_labels and self._model.risk_head is not None:
                preds = self._model.risk_head(features)
                loss = compute_risk_loss(preds, task_labels["risk"])
                total_loss = total_loss + loss
                per_task["risk"] = loss.item()

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self._model.parameters(), self._config.max_grad_norm
            )
            optimizer.step()

            running_loss += total_loss.item()
            for k, v in per_task.items():
                running_per_task[k] = running_per_task.get(k, 0.0) + v
            num_batches += 1

        n = max(num_batches, 1)
        avg_loss = running_loss / n
        avg_per_task = {k: v / n for k, v in running_per_task.items()}
        return avg_loss, avg_per_task

    def _evaluate_model(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on validation data and return (loss, metrics)."""
        from aortica.models.ischaemia_head import compute_ischaemia_loss
        from aortica.models.rhythm_head import compute_rhythm_loss
        from aortica.models.risk_head import compute_risk_loss
        from aortica.models.structural_head import compute_structural_loss

        self._model.eval()
        running_loss = 0.0
        num_batches = 0

        collectors: Dict[str, Dict[str, List[np.ndarray]]] = {
            t: {"preds": [], "targets": []}
            for t in self._config.enabled_tasks
        }

        with torch.no_grad():
            for batch_x, batch_y in self._val_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device).float()

                features = self._model.backbone(batch_x)
                features = self._model.attention(features)

                task_labels = _split_labels(
                    batch_y, self._config.enabled_tasks
                )

                batch_loss = torch.tensor(
                    0.0, device=self._device, dtype=features.dtype
                )

                if (
                    "rhythm" in task_labels
                    and self._model.rhythm_head is not None
                ):
                    logits = self._model.rhythm_head.forward_logits(features)
                    loss = compute_rhythm_loss(logits, task_labels["rhythm"])
                    batch_loss = batch_loss + loss

                if (
                    "structural" in task_labels
                    and self._model.structural_head is not None
                ):
                    logits = self._model.structural_head.forward_logits(
                        features
                    )
                    loss = compute_structural_loss(
                        logits, task_labels["structural"]
                    )
                    batch_loss = batch_loss + loss

                if (
                    "ischaemia" in task_labels
                    and self._model.ischaemia_head is not None
                ):
                    logits = self._model.ischaemia_head.forward_logits(features)
                    loss = compute_ischaemia_loss(
                        logits, task_labels["ischaemia"]
                    )
                    batch_loss = batch_loss + loss

                if (
                    "risk" in task_labels
                    and self._model.risk_head is not None
                ):
                    preds = self._model.risk_head(features)
                    loss = compute_risk_loss(preds, task_labels["risk"])
                    batch_loss = batch_loss + loss

                running_loss += batch_loss.item()
                num_batches += 1

                # Collect predictions for metrics
                for task in self._config.enabled_tasks:
                    head = getattr(self._model, f"{task}_head", None)
                    if head is not None:
                        pred_out = head(features)
                        collectors[task]["preds"].append(
                            pred_out.cpu().numpy()
                        )
                        collectors[task]["targets"].append(
                            task_labels[task].cpu().numpy()
                        )

        n = max(num_batches, 1)
        avg_loss = running_loss / n

        # Compute per-task metrics
        metrics: Dict[str, float] = {}
        for task in self._config.enabled_tasks:
            if not collectors[task]["preds"]:
                continue
            preds_np = np.concatenate(collectors[task]["preds"], axis=0)
            tgts_np = np.concatenate(collectors[task]["targets"], axis=0)

            if task in ("rhythm", "structural", "ischaemia"):
                f1_val = _compute_f1(preds_np, tgts_np)
                metrics[f"{task}_f1"] = f1_val
            elif task == "risk":
                c_idx = _compute_c_index(preds_np, tgts_np)
                metrics["risk_c_index"] = c_idx

        return avg_loss, metrics

    # -- Flower NumPyClient adapter ------------------------------------------

    def to_flower_client(self) -> Any:
        """Create a Flower ``NumPyClient`` adapter for this client.

        Returns a Flower-compatible client object that delegates all
        methods to this :class:`AorticaFlowerClient` instance.

        Returns:
            A ``flwr.client.NumPyClient`` subclass instance.

        Raises:
            ImportError: If Flower is not installed.
        """
        _check_flower()
        import flwr as fl

        outer = self

        class _FlowerAdapter(fl.client.NumPyClient):  # type: ignore[misc]
            """Flower NumPyClient delegating to AorticaFlowerClient."""

            def get_parameters(
                self, config: Dict[str, Any]
            ) -> List[np.ndarray]:
                return outer.get_parameters(config)

            def fit(
                self,
                parameters: List[np.ndarray],
                config: Dict[str, Any],
            ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
                return outer.fit(parameters, config)

            def evaluate(
                self,
                parameters: List[np.ndarray],
                config: Dict[str, Any],
            ) -> Tuple[float, int, Dict[str, float]]:
                return outer.evaluate(parameters, config)

        return _FlowerAdapter()

    # -- Start method --------------------------------------------------------

    def start(
        self,
        server_address: Optional[str] = None,
    ) -> None:
        """Connect to the FL server and start federated training.

        Args:
            server_address: Override ``config.server_address``.  If ``None``,
                uses the value from the client configuration.

        Raises:
            ImportError: If Flower or PyTorch is not installed.
        """
        _check_flower()
        _check_torch()
        self._init_model()

        import flwr as fl

        address = server_address or self._config.server_address

        logger.info(
            "Starting FL client, connecting to %s", address
        )

        flower_client = self.to_flower_client()
        fl.client.start_numpy_client(
            server_address=address,
            client=flower_client,
        )

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AorticaFlowerClient("
            f"server={self._config.server_address!r}, "
            f"local_epochs={self._config.local_epochs}, "
            f"tasks={self._config.enabled_tasks})"
        )
