"""Risk Prediction Task Head for multi-task ECG analysis.

Provides :class:`RiskHead`, a regression head producing 6 continuous outputs
(sigmoid-scaled to 0–1) corresponding to clinical risk predictions.

The 6 outputs are:
    mortality_1y           — 1-year all-cause mortality score
    hf_hosp_12m            — 12-month heart failure hospitalisation probability
    af_onset_12m           — 12-month atrial fibrillation onset risk
    ecg_predicted_ef       — ECG-predicted ejection fraction (continuous 0–1 scaled)
    conduction_disease_trajectory — Progressive conduction disease trajectory score
    sudden_cardiac_death_risk     — Sudden cardiac death risk score

The head connects to the backbone + attention output and uses a combined
MSE + ranking (concordance index proxy) loss.
"""

from __future__ import annotations

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
            "PyTorch is required for RiskHead. "
            "Install with: pip install aortica[torch]"
        )


# Canonical output ordering (6 risk prediction targets).
RISK_OUTPUTS: list[str] = [
    "mortality_1y",
    "hf_hosp_12m",
    "af_onset_12m",
    # Phase 3 risk refinement (US-076)
    "ecg_predicted_ef",
    "conduction_disease_trajectory",
    "sudden_cardiac_death_risk",
]

NUM_RISK_OUTPUTS: int = len(RISK_OUTPUTS)


class RiskHead(nn.Module):
    """Regression head producing 6 sigmoid-scaled risk scores.

    Architecture:

    * Linear(feature_dim, hidden_dim) + ReLU + Dropout
    * Linear(hidden_dim, 6)
    * Sigmoid activation (applied in :meth:`forward`)

    The head outputs raw logits via :meth:`forward_logits` or
    sigmoid-scaled probabilities (0–1) via :meth:`forward`.

    Args:
        feature_dim: Dimension of the input feature vector from the
            backbone + attention module.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Example::

        head = RiskHead(feature_dim=256)
        features = torch.randn(4, 256)
        scores = head(features)             # [4, 6], values in (0, 1)
        logits = head.forward_logits(features)  # [4, 6], raw logits
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        _check_torch()
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_outputs = NUM_RISK_OUTPUTS

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_RISK_OUTPUTS),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (pre-sigmoid values).

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Logit tensor of shape ``[batch, 6]``.
        """
        logits: torch.Tensor = self.regressor(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-scaled risk scores.

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Score tensor of shape ``[batch, 6]``, values in ``(0, 1)``.
        """
        logits = self.forward_logits(x)
        scores: torch.Tensor = torch.sigmoid(logits)
        return scores


def _pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute a differentiable pairwise ranking loss (concordance proxy).

    For each pair (i, j) where target_i > target_j, we want prediction_i >
    prediction_j.  This is implemented as a soft margin ranking loss:

        loss = mean( log(1 + exp(-(pred_i - pred_j))) )  for target_i > target_j

    Args:
        predictions: Predicted scores of shape ``[N]``.
        targets: Ground-truth values of shape ``[N]``.

    Returns:
        Scalar ranking loss.  Returns ``0.0`` if fewer than 2 samples.
    """
    n = predictions.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

    # Pairwise differences: diff_ij = pred_i - pred_j
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [N, N]
    tgt_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [N, N]

    # Only consider pairs where target_i > target_j (upper triangle of ordering)
    valid_mask = tgt_diff > 0  # [N, N]

    if not valid_mask.any():
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

    # Soft margin ranking: log(1 + exp(-pred_diff)) for valid pairs
    # Clamp for numerical stability
    losses = torch.log1p(torch.exp(-pred_diff.clamp(-50.0, 50.0)))
    ranking_loss: torch.Tensor = losses[valid_mask].mean()
    return ranking_loss


def compute_risk_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ranking_weight: float = 0.1,
    task_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute combined MSE + ranking loss for risk prediction.

    The total loss is::

        loss = MSE(predictions, targets) + ranking_weight * mean(ranking_loss_per_task)

    The ranking component is a differentiable concordance-index proxy that
    encourages correct ordering of predictions with respect to targets.

    Args:
        predictions: Sigmoid-scaled predicted scores of shape ``[batch, 6]``.
        targets: Ground-truth risk values of shape ``[batch, 6]``, in ``[0, 1]``.
        ranking_weight: Weight for the ranking loss component.  Default ``0.1``.
        task_weights: Optional per-task weight tensor of shape ``[6]``.
            Applied to the MSE component.  If ``None``, uniform weighting.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()

    # MSE component
    if task_weights is not None:
        # Per-task weighted MSE: weight each output column
        mse = ((predictions - targets) ** 2 * task_weights.unsqueeze(0)).mean()
    else:
        mse = nn.functional.mse_loss(predictions, targets)

    # Ranking component — average ranking loss across all risk outputs
    rank_losses = []
    num_tasks = predictions.shape[1]
    for t in range(num_tasks):
        rank_losses.append(
            _pairwise_ranking_loss(predictions[:, t], targets[:, t])
        )
    ranking = torch.stack(rank_losses).mean()

    loss: torch.Tensor = mse + ranking_weight * ranking
    return loss
