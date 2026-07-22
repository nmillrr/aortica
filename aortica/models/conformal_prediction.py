"""Conformal Prediction and Uncertainty Estimation.

Provides :class:`ConformalPredictor`, a wrapper around a trained model that
generates prediction sets at a user-specified coverage level, detects
out-of-distribution (OOD) inputs via Mahalanobis distance on backbone
features, and returns an :class:`UncertaintyReport` alongside predictions.

Usage::

    # Fit the conformal predictor on calibration data
    cp = ConformalPredictor(model, coverage=0.90)
    cp.fit(cal_loader)

    # At inference time
    preds, report = cp.predict(x)
    report.ood_flag       # True if input is OOD
    report.entropy_score  # Prediction entropy per sample
    report.prediction_sets  # Per-task prediction sets (label indices)
    report.confidence_intervals  # Per-task confidence intervals for risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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
            "PyTorch is required for ConformalPredictor. "
            "Install with: pip install aortica[torch]"
        )


# Single source of truth (US-129), derived from the head class-list constants.
from aortica.models.task_dims import (  # noqa: E402
    CLASSIFICATION_TASKS,
    TASK_NUM_OUTPUTS,
)


@dataclass
class UncertaintyReport:
    """Uncertainty information returned alongside predictions.

    Attributes:
        prediction_sets: Per-task dict mapping task name to a list of
            prediction sets (one per sample).  Each prediction set is a
            list of class indices whose conformal score exceeds the
            threshold, guaranteeing ``coverage`` marginal coverage.
            Only populated for classification tasks.
        confidence_intervals: Per-task dict mapping task name to
            ``(lower, upper)`` tensors of shape ``[batch, C]`` for risk
            outputs.  Only populated for the ``risk`` task.
        ood_flags: Boolean tensor ``[batch]`` indicating whether each
            input is flagged as out-of-distribution.
        entropy_scores: Tensor ``[batch]`` of mean prediction entropies
            across all task outputs.
        mahalanobis_distances: Tensor ``[batch]`` of Mahalanobis
            distances from the training feature distribution.
    """

    prediction_sets: dict[str, list[list[int]]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    ood_flags: Optional[torch.Tensor] = None
    entropy_scores: Optional[torch.Tensor] = None
    mahalanobis_distances: Optional[torch.Tensor] = None


class ConformalPredictor(nn.Module):
    """Conformal prediction wrapper with OOD detection.

    Wraps a trained :class:`AorticaModel` to produce:

    1. **Prediction sets** for classification tasks at a specified
       coverage level (default 90%) using split conformal prediction.
    2. **OOD detection** via Mahalanobis distance on backbone features.
    3. **Uncertainty reports** with entropy scores.

    The wrapper must be *fitted* on a calibration set (held-out from
    training and validation) before it can produce prediction sets or
    OOD flags.

    Args:
        model: A trained ``AorticaModel``.
        coverage: Target marginal coverage for prediction sets.
            Default ``0.90``.
        ood_percentile: Percentile threshold on calibration Mahalanobis
            distances above which an input is flagged OOD.
            Default ``95.0``.

    Example::

        cp = ConformalPredictor(model, coverage=0.90)
        cp.fit(cal_loader)
        preds, report = cp.predict(x)
    """

    def __init__(
        self,
        model: nn.Module,
        coverage: float = 0.90,
        ood_percentile: float = 95.0,
    ) -> None:
        _check_torch()
        super().__init__()
        self.model = model
        self.coverage = coverage
        self.ood_percentile = ood_percentile

        # Fitted state — populated by fit()
        self._quantiles: dict[str, float] = {}  # per-task conformal quantile
        self._risk_residual_quantile: Optional[float] = None

        # OOD detection state
        self._feature_mean: Optional[torch.Tensor] = None
        self._feature_cov_inv: Optional[torch.Tensor] = None
        self._ood_threshold: Optional[float] = None

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Fitting (calibration)
    # ------------------------------------------------------------------

    def fit(
        self,
        cal_loader: torch.utils.data.DataLoader,  # type: ignore[name-defined]
        device: Optional[torch.device] = None,
    ) -> None:
        """Fit conformal thresholds and OOD detector on calibration data.

        Args:
            cal_loader: Calibration data loader yielding ``(x, labels)``
                tuples where ``labels`` columns are concatenated in
                ``model.enabled_tasks`` order.
            device: Device to run on.  Default: inferred from model.
        """
        _check_torch()

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.model.eval()
        enabled_tasks: list[str] = self.model.enabled_tasks

        # Collect backbone features, per-task logits, and labels
        all_features: list[torch.Tensor] = []
        all_logits: dict[str, list[torch.Tensor]] = {t: [] for t in enabled_tasks}
        all_labels: dict[str, list[torch.Tensor]] = {t: [] for t in enabled_tasks}

        with torch.no_grad():
            for batch_x, batch_labels in cal_loader:
                batch_x = batch_x.to(device)
                batch_labels = batch_labels.to(device)

                # Extract backbone features (before attention) for OOD
                features = self.model.backbone(batch_x)
                all_features.append(features)

                # Get attention-enriched features for task heads
                enriched = self.model.attention(features)

                # Split labels and collect logits
                label_offset = 0
                for task_name in enabled_tasks:
                    size = TASK_NUM_OUTPUTS[task_name]
                    task_labels = batch_labels[
                        :, label_offset : label_offset + size
                    ]
                    all_labels[task_name].append(task_labels)

                    head = getattr(self.model, f"{task_name}_head")
                    task_logits: torch.Tensor = head.forward_logits(enriched)
                    all_logits[task_name].append(task_logits)

                    label_offset += size

        # Concatenate
        cat_features = torch.cat(all_features, dim=0)
        cat_logits = {t: torch.cat(all_logits[t], dim=0) for t in enabled_tasks}
        cat_labels = {t: torch.cat(all_labels[t], dim=0) for t in enabled_tasks}

        # --- 1. Conformal quantiles for classification tasks ---
        for task in enabled_tasks:
            if task in CLASSIFICATION_TASKS:
                self._quantiles[task] = self._compute_conformal_quantile(
                    cat_logits[task], cat_labels[task]
                )

        # --- 2. Conformal quantile for risk task (residual-based) ---
        if "risk" in enabled_tasks:
            preds = torch.sigmoid(cat_logits["risk"])
            residuals = (preds - cat_labels["risk"]).abs()
            # Use the max residual across the 3 outputs per sample
            max_residuals = residuals.max(dim=1).values
            n = max_residuals.shape[0]
            q_level = min(
                np.ceil((n + 1) * self.coverage) / n, 1.0
            )
            sorted_r = torch.sort(max_residuals).values
            idx = min(int(np.ceil(q_level * n)) - 1, n - 1)
            self._risk_residual_quantile = float(sorted_r[idx].item())

        # --- 3. OOD detection via Mahalanobis distance ---
        self._fit_ood_detector(cat_features)

        self._is_fitted = True

    def _compute_conformal_quantile(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute the conformal quantile for a classification task.

        Uses the *non-conformity score* = 1 − sigmoid(logit) for the
        true class.  The quantile is chosen so that the prediction set
        achieves at least ``coverage`` marginal coverage.

        For multi-label tasks, we compute a score per (sample, class)
        pair and take the maximum across classes per sample.
        """
        probs = torch.sigmoid(logits)
        # Non-conformity: 1 − p for positive labels, p for negative labels
        # This is equivalent to 1 − p_true where p_true is the conformity
        scores = torch.where(labels == 1, 1 - probs, probs)
        # Per-sample score is the max across classes
        per_sample_scores = scores.max(dim=1).values

        n = per_sample_scores.shape[0]
        q_level = min(
            np.ceil((n + 1) * self.coverage) / n, 1.0
        )
        sorted_scores = torch.sort(per_sample_scores).values
        idx = min(int(np.ceil(q_level * n)) - 1, n - 1)
        return float(sorted_scores[idx].item())

    def _fit_ood_detector(self, features: torch.Tensor) -> None:
        """Fit Mahalanobis distance OOD detector on backbone features."""
        # Compute mean and covariance of calibration features
        self._feature_mean = features.mean(dim=0)  # [feature_dim]
        centered = features - self._feature_mean.unsqueeze(0)
        cov = (centered.T @ centered) / max(features.shape[0] - 1, 1)

        # Regularise covariance for numerical stability
        reg = 1e-5 * torch.eye(cov.shape[0], device=cov.device)
        cov = cov + reg

        self._feature_cov_inv = torch.linalg.inv(cov)

        # Compute distances on calibration set to set the threshold
        cal_distances = self._mahalanobis_distance(features)
        threshold_val = float(
            np.percentile(
                cal_distances.cpu().numpy(), self.ood_percentile
            )
        )
        self._ood_threshold = threshold_val

    def _mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from the calibration distribution.

        Args:
            features: Feature tensor ``[batch, feature_dim]``.

        Returns:
            Distance tensor ``[batch]``.
        """
        assert self._feature_mean is not None
        assert self._feature_cov_inv is not None

        centered = features - self._feature_mean.unsqueeze(0)
        # Mahalanobis: sqrt( (x-μ)^T Σ^{-1} (x-μ) )
        # For numerical stability, compute the full form:
        left = centered @ self._feature_cov_inv  # [batch, feature_dim]
        dist_sq = (left * centered).sum(dim=1)  # [batch]
        return torch.sqrt(torch.clamp(dist_sq, min=0.0))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        tasks: Optional[list[str]] = None,
    ) -> tuple[dict[str, torch.Tensor], UncertaintyReport]:
        """Run inference with conformal prediction and OOD detection.

        Args:
            x: Input ECG tensor ``[batch, leads, samples]``.
            tasks: Optional subset of tasks.

        Returns:
            Tuple of ``(predictions, report)`` where ``predictions``
            maps task name to probability tensor and ``report`` is an
            :class:`UncertaintyReport`.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        _check_torch()
        if not self._is_fitted:
            raise RuntimeError(
                "ConformalPredictor must be fitted before prediction. "
                "Call .fit(cal_loader) first."
            )

        self.model.eval()
        active_tasks = tasks if tasks is not None else self.model.enabled_tasks

        with torch.no_grad():
            # Feature extraction
            features = self.model.backbone(x)
            enriched = self.model.attention(features)

            # Per-task predictions
            predictions: dict[str, torch.Tensor] = {}
            report = UncertaintyReport()

            all_entropies: list[torch.Tensor] = []

            for task in active_tasks:
                head = getattr(self.model, f"{task}_head")
                if head is None:
                    continue

                logits: torch.Tensor = head.forward_logits(enriched)
                probs = torch.sigmoid(logits)
                predictions[task] = probs

                if task in CLASSIFICATION_TASKS and task in self._quantiles:
                    # Build prediction sets
                    q = self._quantiles[task]
                    pred_sets: list[list[int]] = []
                    for i in range(probs.shape[0]):
                        sample_set: list[int] = []
                        for j in range(probs.shape[1]):
                            # Include class j if its non-conformity score
                            # (1 − prob for positive, prob for negative)
                            # is ≤ quantile.  Equivalently, include if
                            # prob ≥ 1 − q OR (1 − prob) ≤ q.
                            # Both conditions reduce to: prob ≥ 1 − q
                            # for the "include as positive" direction,
                            # but we also include classes where prob ≤ q
                            # would NOT be included.
                            #
                            # Standard approach: include class j if
                            # 1 − prob_j ≤ quantile (positive direction)
                            p = float(probs[i, j].item())
                            if 1 - p <= q:
                                sample_set.append(j)
                        pred_sets.append(sample_set)
                    report.prediction_sets[task] = pred_sets

                # Confidence intervals for risk task
                if task == "risk" and self._risk_residual_quantile is not None:
                    q_r = self._risk_residual_quantile
                    lower = torch.clamp(probs - q_r, min=0.0)
                    upper = torch.clamp(probs + q_r, max=1.0)
                    report.confidence_intervals[task] = (lower, upper)

                # Entropy: −p log(p) − (1−p) log(1−p) for each output
                eps = 1e-7
                p_clamped = torch.clamp(probs, eps, 1 - eps)
                entropy = -(
                    p_clamped * torch.log(p_clamped)
                    + (1 - p_clamped) * torch.log(1 - p_clamped)
                )
                # Mean entropy across outputs for this task
                all_entropies.append(entropy.mean(dim=1))

            # Aggregate entropy across tasks
            if all_entropies:
                stacked = torch.stack(all_entropies, dim=1)
                report.entropy_scores = stacked.mean(dim=1)

            # OOD detection
            distances = self._mahalanobis_distance(features)
            report.mahalanobis_distances = distances

            assert self._ood_threshold is not None
            report.ood_flags = distances > self._ood_threshold

        return predictions, report

    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[list[str]] = None,
    ) -> tuple[dict[str, torch.Tensor], UncertaintyReport]:
        """Alias for :meth:`predict` to support ``nn.Module`` interface."""
        return self.predict(x, tasks=tasks)
