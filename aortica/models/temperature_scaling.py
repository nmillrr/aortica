"""Temperature Scaling Calibration for post-hoc model calibration.

Provides :class:`TemperatureScaling`, a lightweight module that learns a
single temperature parameter per task head on the validation set to improve
probability calibration.

Also provides :class:`CalibratedModel`, a wrapper around a trained
:class:`AorticaModel` that applies temperature scaling at inference time,
and :func:`calibrate`, which optimises the temperature parameters on a
given validation set.

Finally, :func:`expected_calibration_error` computes ECE (the primary
calibration metric) and :func:`reliability_diagram_data` returns the
binned data needed to plot a reliability diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn = types.ModuleType("nn")  # type: ignore[assignment]
    nn.Module = _DummyModule  # type: ignore[attr-defined]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for TemperatureScaling. "
            "Install with: pip install aortica[torch]"
        )


# Task names matching AorticaModel
CLASSIFICATION_TASKS: list[str] = ["rhythm", "structural", "ischaemia"]
ALL_TASKS: list[str] = ["rhythm", "structural", "ischaemia", "risk"]


@dataclass
class ReliabilityDiagramData:
    """Data for plotting a reliability diagram.

    Attributes:
        bin_confidences: Mean predicted confidence per bin.
        bin_accuracies: Fraction of positives per bin.
        bin_counts: Number of samples per bin.
        num_bins: Number of bins used.
        ece: Expected calibration error.
    """

    bin_confidences: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)
    num_bins: int = 10
    ece: float = 0.0


class TemperatureScaling(nn.Module):
    """Learn a single temperature parameter per task head.

    Temperature scaling divides logits by a learned scalar T > 0 before
    applying sigmoid.  A higher T → softer probabilities (less confident),
    lower T → sharper probabilities (more confident).

    The optimal T is found by minimising the negative log-likelihood (NLL)
    on the validation set via :func:`calibrate`.

    Args:
        tasks: List of task names for which to learn temperatures.
            Default: all classification tasks (rhythm, structural, ischaemia).

    Example::

        ts = TemperatureScaling(tasks=["rhythm", "structural"])
        # After calibration:
        calibrated_logits = ts(logits_dict)
    """

    def __init__(
        self,
        tasks: Optional[list[str]] = None,
    ) -> None:
        _check_torch()
        super().__init__()

        if tasks is None:
            tasks = list(CLASSIFICATION_TASKS)

        self.tasks = tasks

        # One learnable log-temperature per task (initialised to log(1) = 0
        # so that T=1 initially, i.e. no change).
        self.log_temperatures = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(1)) for task in tasks}
        )

    def get_temperature(self, task: str) -> torch.Tensor:
        """Return the temperature for a task (always > 0)."""
        log_t: torch.Tensor = self.log_temperatures[task]
        return torch.exp(log_t)

    def forward(
        self,
        logits: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply temperature scaling to a dict of per-task logits.

        Args:
            logits: Mapping from task name to logit tensor ``[batch, C]``.

        Returns:
            Dict of scaled logit tensors (same keys/shapes).
        """
        scaled: dict[str, torch.Tensor] = {}
        for task, task_logits in logits.items():
            if task in self.log_temperatures:
                temperature = self.get_temperature(task)
                scaled[task] = task_logits / temperature
            else:
                scaled[task] = task_logits
        return scaled


class CalibratedModel(nn.Module):
    """Wrapper that applies temperature scaling at inference time.

    Wraps a trained :class:`AorticaModel` and a fitted
    :class:`TemperatureScaling` module.  The forward pass extracts raw
    logits from the model, scales them, and returns calibrated sigmoid
    probabilities.

    Args:
        model: A trained ``AorticaModel``.
        temperature_scaling: A fitted ``TemperatureScaling`` module.

    Example::

        calibrated = CalibratedModel(model, ts)
        output = calibrated(ecg_tensor)
        output["rhythm"]  # calibrated probabilities [batch, 22]
    """

    def __init__(
        self,
        model: nn.Module,
        temperature_scaling: TemperatureScaling,
    ) -> None:
        _check_torch()
        super().__init__()
        self.model = model
        self.temperature_scaling = temperature_scaling

    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Run calibrated inference.

        Args:
            x: Input ECG tensor ``[batch, leads, samples]``.
            tasks: Optional subset of tasks to run.

        Returns:
            Dict mapping task name → calibrated probability tensor.
        """
        with torch.no_grad():
            self.model.eval()

            # Forward through backbone + attention to get features
            features = self.model.backbone(x)
            features = self.model.attention(features)

            active_tasks = tasks if tasks is not None else self.model.enabled_tasks

            # Collect raw logits from each head
            logits: dict[str, torch.Tensor] = {}
            if "rhythm" in active_tasks and self.model.rhythm_head is not None:
                logits["rhythm"] = self.model.rhythm_head.forward_logits(features)
            if "structural" in active_tasks and self.model.structural_head is not None:
                logits["structural"] = self.model.structural_head.forward_logits(
                    features
                )
            if "ischaemia" in active_tasks and self.model.ischaemia_head is not None:
                logits["ischaemia"] = self.model.ischaemia_head.forward_logits(
                    features
                )
            if "risk" in active_tasks and self.model.risk_head is not None:
                logits["risk"] = self.model.risk_head.forward_logits(features)

            # Apply temperature scaling
            scaled_logits = self.temperature_scaling(logits)

            # Apply sigmoid to get calibrated probabilities
            calibrated: dict[str, torch.Tensor] = {}
            for task, task_logits in scaled_logits.items():
                calibrated[task] = torch.sigmoid(task_logits)

            return calibrated


def calibrate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,  # type: ignore[name-defined]
    tasks: Optional[list[str]] = None,
    lr: float = 0.01,
    max_iter: int = 100,
    device: Optional[torch.device] = None,
) -> TemperatureScaling:
    """Optimise temperature parameters on the validation set.

    Runs the model in eval mode to collect logits, then optimises a
    per-task temperature to minimise NLL (binary cross-entropy with logits).

    Args:
        model: A trained ``AorticaModel`` (or compatible module with
            ``backbone``, ``attention``, and task head attributes).
        val_loader: Validation data loader yielding ``(x, labels)`` tuples
            where ``labels`` has columns for each enabled task.
        tasks: Which task heads to calibrate.  Default: all classification
            tasks present in ``model.enabled_tasks``.
        lr: Learning rate for LBFGS optimiser.  Default ``0.01``.
        max_iter: Maximum LBFGS iterations.  Default ``100``.
        device: Device to run on.  Default: inferred from model parameters.

    Returns:
        Fitted :class:`TemperatureScaling` module.
    """
    _check_torch()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Determine which tasks to calibrate
    if tasks is None:
        tasks = [t for t in CLASSIFICATION_TASKS if t in model.enabled_tasks]

    ts_module = TemperatureScaling(tasks=tasks)
    ts_module.to(device)
    ts: TemperatureScaling = ts_module

    # Collect all logits and labels from the validation set
    model.eval()
    all_logits: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}
    all_labels: dict[str, list[torch.Tensor]] = {t: [] for t in tasks}

    # Task output sizes for splitting concatenated labels. Must stay in sync
    # with the head class constants (rhythm=28, structural=19, ischaemia=19,
    # risk=6) and benchmark.TASK_NUM_OUTPUTS.
    task_sizes: dict[str, int] = {
        "rhythm": 28,
        "structural": 19,
        "ischaemia": 19,
        "risk": 6,
    }

    with torch.no_grad():
        for batch_x, batch_labels in val_loader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)

            # Get features
            features = model.backbone(batch_x)
            features = model.attention(features)

            # Split labels by task (assumes concatenated in enabled_tasks order)
            label_offset = 0
            for task_name in model.enabled_tasks:
                size = task_sizes[task_name]
                if task_name in tasks:
                    task_labels = batch_labels[:, label_offset : label_offset + size]
                    all_labels[task_name].append(task_labels)

                    # Get logits from the appropriate head
                    head = getattr(model, f"{task_name}_head")
                    task_logits: torch.Tensor = head.forward_logits(features)
                    all_logits[task_name].append(task_logits)

                label_offset += size

    # Concatenate all collected logits and labels
    cat_logits: dict[str, torch.Tensor] = {
        t: torch.cat(all_logits[t], dim=0) for t in tasks
    }
    cat_labels: dict[str, torch.Tensor] = {
        t: torch.cat(all_labels[t], dim=0) for t in tasks
    }

    # Optimise temperatures using LBFGS
    optimizer = torch.optim.LBFGS(
        ts.parameters(), lr=lr, max_iter=max_iter
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        total_nll = torch.tensor(0.0, device=device)
        scaled = ts(cat_logits)
        for task in tasks:
            nll = nn.functional.binary_cross_entropy_with_logits(
                scaled[task], cat_labels[task]
            )
            total_nll = total_nll + nll
        total_nll.backward()
        return total_nll

    optimizer.step(closure)

    return ts


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the weighted average gap between predicted confidence
    and observed accuracy across probability bins.

    Args:
        probabilities: Predicted probabilities ``[N]`` or ``[N, C]``.
        labels: Binary ground-truth labels of the same shape.
        num_bins: Number of equal-width bins.  Default ``10``.

    Returns:
        ECE as a float in ``[0, 1]``.
    """
    _check_torch()

    probs = probabilities.detach().flatten()
    labs = labels.detach().flatten().float()

    n_total = probs.numel()
    if n_total == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)

    ece = 0.0
    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = mask.sum().item()
        if n_bin == 0:
            continue

        avg_confidence = probs[mask].mean().item()
        avg_accuracy = labs[mask].mean().item()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece


def reliability_diagram_data(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> ReliabilityDiagramData:
    """Compute binned data for a reliability diagram.

    Args:
        probabilities: Predicted probabilities ``[N]`` or ``[N, C]``.
        labels: Binary ground-truth labels of the same shape.
        num_bins: Number of equal-width bins.  Default ``10``.

    Returns:
        :class:`ReliabilityDiagramData` with per-bin statistics and ECE.
    """
    _check_torch()

    probs = probabilities.detach().flatten()
    labs = labels.detach().flatten().float()

    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)

    confidences: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = int(mask.sum().item())
        counts.append(n_bin)

        if n_bin == 0:
            confidences.append(float((lo + hi) / 2))
            accuracies.append(0.0)
        else:
            confidences.append(float(probs[mask].mean().item()))
            accuracies.append(float(labs[mask].mean().item()))

    ece = expected_calibration_error(probabilities, labels, num_bins)

    return ReliabilityDiagramData(
        bin_confidences=confidences,
        bin_accuracies=accuracies,
        bin_counts=counts,
        num_bins=num_bins,
        ece=ece,
    )
