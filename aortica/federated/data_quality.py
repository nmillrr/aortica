"""Pre-training data quality gating for federated learning sites.

Before a site joins a federated learning campaign (US-063 client wrapper),
it should validate that its site-local data meets minimum quality standards.
A site with noisy signals, too few samples, or missing labels can degrade
the aggregated model.  :class:`DataQualityGate` runs a set of configurable
checks and produces a :class:`DataQualityReport` describing which checks
passed, detailed statistics, and actionable recommendations.

The gate is intentionally free of any heavy ML dependency (no ``torch`` /
``flwr``); it operates on :class:`~aortica.io.ecg_record.ECGRecord` objects
and a label matrix so it can run standalone on a site before training
infrastructure is set up.

Example::

    from aortica.federated import DataQualityGate

    gate = DataQualityGate()
    report = gate.validate((records, labels))
    if report.blocking:
        print("Site cannot join:", report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord
from aortica.signal.quality_scoring import score_quality

# ---------------------------------------------------------------------------
# Task-head output dimensions (label-matrix column layout)
#
# The site-local label matrix concatenates per-task columns in this fixed
# order.  Must stay in sync with the task-head class constants and with
# ``aortica.federated.fl_client._TASK_NUM_OUTPUTS`` /
# ``aortica.models.train_multitask._TASK_NUM_OUTPUTS``.
# ---------------------------------------------------------------------------

_TASK_SUPERCLASSES: List[str] = ["rhythm", "structural", "ischaemia", "risk"]
_TASK_NUM_OUTPUTS: Dict[str, int] = {
    "rhythm": 28,
    "structural": 19,
    "ischaemia": 19,
    "risk": 6,
}


def _task_slices(
    task_dims: Dict[str, int],
) -> Dict[str, Tuple[int, int]]:
    """Return ``{task: (start, stop)}`` column slices for the label matrix."""
    slices: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for task in _TASK_SUPERCLASSES:
        width = task_dims[task]
        slices[task] = (offset, offset + width)
        offset += width
    return slices


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class QualityCheck:
    """Result of a single data-quality check.

    Attributes:
        name: Machine-readable check identifier.
        passed: ``True`` if the check passed.
        blocking: ``True`` if a failure blocks the site from joining.
        message: Human-readable explanation of the result.
        stats: Detailed statistics gathered for the check.
    """

    name: str
    passed: bool
    blocking: bool
    message: str
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain, JSON-serialisable dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "blocking": self.blocking,
            "message": self.message,
            "stats": self.stats,
        }


@dataclass
class DataQualityReport:
    """Aggregate data-quality report for a site-local dataset.

    Attributes:
        passed: ``True`` if every check passed.
        blocking: ``True`` if at least one *blocking* check failed; a
            blocking failure means the site must not join the campaign.
        checks: Per-check results.
        statistics: Dataset-level summary statistics.
        recommendations: Actionable recommendations for the site.
    """

    passed: bool
    blocking: bool
    checks: List[QualityCheck]
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def check(self, name: str) -> Optional[QualityCheck]:
        """Return the check with *name*, or ``None`` if absent."""
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain, JSON-serialisable dict."""
        return {
            "passed": self.passed,
            "blocking": self.blocking,
            "checks": [c.to_dict() for c in self.checks],
            "statistics": self.statistics,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        """Return a concise multi-line text summary."""
        verdict = "PASS" if self.passed else ("BLOCKED" if self.blocking else "WARN")
        lines = [f"Data quality gate: {verdict}"]
        for c in self.checks:
            mark = "✓" if c.passed else ("✗" if c.blocking else "!")
            lines.append(f"  [{mark}] {c.name}: {c.message}")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server-side admission policy
# ---------------------------------------------------------------------------

#: Valid server-side data-quality policies.
DQ_POLICIES: Tuple[str, ...] = ("strict", "warn", "permissive")


def site_admitted(report: DataQualityReport, policy: str = "strict") -> bool:
    """Decide whether a site may join a campaign given its report.

    Args:
        report: The site's :class:`DataQualityReport`.
        policy: One of ``"strict"`` (exclude any blocking failure),
            ``"warn"`` (exclude only when a blocking check fails, but
            treat non-blocking warnings as admissible), or ``"permissive"``
            (admit all sites regardless of result).

    Returns:
        ``True`` if the site is admitted under *policy*.

    Note:
        ``strict`` and ``warn`` both exclude sites with a *blocking*
        failure; they differ in intent for downstream logging/alerting.
        A non-blocking (warning-only) failure never excludes a site.
    """
    if policy not in DQ_POLICIES:
        raise ValueError(
            f"Unknown policy '{policy}'. Valid: {DQ_POLICIES}"
        )
    if policy == "permissive":
        return True
    # strict / warn: a blocking failure excludes the site.
    return not report.blocking


# ---------------------------------------------------------------------------
# Data-quality gate
# ---------------------------------------------------------------------------


class DataQualityGate:
    """Runs pre-training data-quality validation on site-local ECG data.

    Args:
        min_sample_size: Target minimum number of ECGs (default 500).
        warn_sample_size: Emit a warning below this count (default 200).
        block_sample_size: Block the site below this count (default 100).
        quality_score_threshold: Minimum per-ECG quality score counted as
            "marginal or better" (default 40).
        quality_fraction_required: Fraction of ECGs that must meet the
            quality threshold (default 0.70).
        label_completeness_required: Fraction of ECGs that must carry at
            least one diagnostic label (default 0.80).
        diversity_min_superclasses: Minimum number of task superclasses
            that must be adequately represented (default 3 of 4).
        diversity_positive_threshold: Minimum positive examples for a
            superclass to count as represented (default 10).
        task_dims: Per-task output dimensions used to slice the label
            matrix.  Defaults to the canonical task-head dimensions.
    """

    def __init__(
        self,
        *,
        min_sample_size: int = 500,
        warn_sample_size: int = 200,
        block_sample_size: int = 100,
        quality_score_threshold: float = 40.0,
        quality_fraction_required: float = 0.70,
        label_completeness_required: float = 0.80,
        diversity_min_superclasses: int = 3,
        diversity_positive_threshold: int = 10,
        task_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        if not (block_sample_size <= warn_sample_size <= min_sample_size):
            raise ValueError(
                "sample-size thresholds must satisfy "
                "block <= warn <= min"
            )
        self.min_sample_size = min_sample_size
        self.warn_sample_size = warn_sample_size
        self.block_sample_size = block_sample_size
        self.quality_score_threshold = quality_score_threshold
        self.quality_fraction_required = quality_fraction_required
        self.label_completeness_required = label_completeness_required
        self.diversity_min_superclasses = diversity_min_superclasses
        self.diversity_positive_threshold = diversity_positive_threshold
        self.task_dims = dict(task_dims) if task_dims else dict(_TASK_NUM_OUTPUTS)

    # -- Public API ---------------------------------------------------------

    def validate(self, dataset: Any) -> DataQualityReport:
        """Validate *dataset* and return a :class:`DataQualityReport`.

        Args:
            dataset: One of:

                * an object exposing ``.records`` (list of ``ECGRecord``)
                  and ``.labels`` (array-like), e.g. an ``ECGDataset``;
                * a ``(records, labels)`` tuple.

        Returns:
            A :class:`DataQualityReport`.  ``report.blocking`` is ``True``
            when a blocking check fails and the site must not join.
        """
        records, labels = self._normalise(dataset)
        n = len(records)

        checks: List[QualityCheck] = []
        recommendations: List[str] = []

        # 1. Sample size --------------------------------------------------
        size_check = self._check_sample_size(n, recommendations)
        checks.append(size_check)

        # 2. Format consistency + per-record quality scoring --------------
        # score_quality exercises the signal-reading path, so we compute
        # quality scores and format consistency in a single pass.
        quality_scores, format_check = self._check_format_and_quality(
            records, recommendations
        )
        checks.append(format_check)

        # 3. Signal quality distribution ----------------------------------
        checks.append(self._check_quality(quality_scores, n, recommendations))

        # 4. Label completeness -------------------------------------------
        checks.append(self._check_label_completeness(labels, n, recommendations))

        # 5. Label diversity ----------------------------------------------
        diversity_check, per_superclass = self._check_label_diversity(
            labels, recommendations
        )
        checks.append(diversity_check)

        passed = all(c.passed for c in checks)
        blocking = any((not c.passed) and c.blocking for c in checks)

        valid_scores = [s for s in quality_scores if s is not None]
        statistics: Dict[str, Any] = {
            "num_ecgs": n,
            "mean_quality_score": (
                float(np.mean(valid_scores)) if valid_scores else 0.0
            ),
            "num_format_errors": sum(1 for s in quality_scores if s is None),
            "positive_per_superclass": per_superclass,
        }

        return DataQualityReport(
            passed=passed,
            blocking=blocking,
            checks=checks,
            statistics=statistics,
            recommendations=recommendations,
        )

    # -- Individual checks --------------------------------------------------

    def _check_sample_size(
        self, n: int, recommendations: List[str]
    ) -> QualityCheck:
        stats = {
            "count": n,
            "min_required": self.min_sample_size,
            "warn_below": self.warn_sample_size,
            "block_below": self.block_sample_size,
        }
        if n < self.block_sample_size:
            recommendations.append(
                f"Collect more data: {n} ECGs is below the hard minimum of "
                f"{self.block_sample_size}."
            )
            return QualityCheck(
                name="sample_size",
                passed=False,
                blocking=True,
                message=(
                    f"{n} ECGs is below the blocking threshold "
                    f"({self.block_sample_size})."
                ),
                stats=stats,
            )
        if n < self.warn_sample_size:
            recommendations.append(
                f"Sample size ({n}) is low; aim for at least "
                f"{self.min_sample_size} ECGs for a full contribution."
            )
            return QualityCheck(
                name="sample_size",
                passed=False,
                blocking=False,
                message=(
                    f"{n} ECGs is below the warning threshold "
                    f"({self.warn_sample_size})."
                ),
                stats=stats,
            )
        if n < self.min_sample_size:
            return QualityCheck(
                name="sample_size",
                passed=False,
                blocking=False,
                message=(
                    f"{n} ECGs is below the recommended minimum "
                    f"({self.min_sample_size}) but above the warning "
                    f"threshold."
                ),
                stats=stats,
            )
        return QualityCheck(
            name="sample_size",
            passed=True,
            blocking=False,
            message=f"{n} ECGs meets the minimum of {self.min_sample_size}.",
            stats=stats,
        )

    def _check_format_and_quality(
        self,
        records: Sequence[ECGRecord],
        recommendations: List[str],
    ) -> Tuple[List[Optional[float]], QualityCheck]:
        """Run each record through quality scoring.

        Returns a list of per-record overall scores (``None`` where the
        record failed to process) plus the format-consistency check.
        """
        scores: List[Optional[float]] = []
        errors: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            try:
                report = score_quality(record)
                scores.append(float(report.overall_score))
            except Exception as exc:  # noqa: BLE001 - report, don't raise
                scores.append(None)
                errors.append({"index": idx, "error": str(exc)})

        num_errors = len(errors)
        stats = {
            "num_errors": num_errors,
            "num_records": len(records),
            # Cap stored errors to keep the report compact.
            "errors": errors[:10],
        }
        if num_errors > 0:
            recommendations.append(
                f"{num_errors} ECG(s) failed preprocessing; inspect and "
                f"remove or repair malformed records."
            )
            return scores, QualityCheck(
                name="format_consistency",
                passed=False,
                blocking=True,
                message=(
                    f"{num_errors} of {len(records)} ECGs failed to pass "
                    f"through the preprocessing pipeline."
                ),
                stats=stats,
            )
        return scores, QualityCheck(
            name="format_consistency",
            passed=True,
            blocking=True,
            message=f"All {len(records)} ECGs passed preprocessing.",
            stats=stats,
        )

    def _check_quality(
        self,
        quality_scores: List[Optional[float]],
        n: int,
        recommendations: List[str],
    ) -> QualityCheck:
        valid = [s for s in quality_scores if s is not None]
        num_marginal_or_better = sum(
            1 for s in valid if s >= self.quality_score_threshold
        )
        # Fraction is computed over the full dataset: records that fail
        # preprocessing count against the quality distribution.
        fraction = (num_marginal_or_better / n) if n > 0 else 0.0
        stats = {
            "fraction_marginal_or_better": fraction,
            "required_fraction": self.quality_fraction_required,
            "quality_threshold": self.quality_score_threshold,
            "num_marginal_or_better": num_marginal_or_better,
            "num_ecgs": n,
        }
        passed = fraction >= self.quality_fraction_required
        if not passed:
            recommendations.append(
                f"Signal quality is low: only {fraction:.0%} of ECGs score "
                f">= {self.quality_score_threshold:.0f}. Improve acquisition "
                f"or filter low-quality records."
            )
        return QualityCheck(
            name="signal_quality",
            passed=passed,
            blocking=True,
            message=(
                f"{fraction:.0%} of ECGs score >= "
                f"{self.quality_score_threshold:.0f} "
                f"(required {self.quality_fraction_required:.0%})."
            ),
            stats=stats,
        )

    def _check_label_completeness(
        self,
        labels: NDArray[np.float64],
        n: int,
        recommendations: List[str],
    ) -> QualityCheck:
        if n == 0:
            num_labeled = 0
        else:
            # A record is "labeled" if it carries at least one positive
            # (non-zero) value anywhere in its label vector.
            num_labeled = int(np.sum(np.any(labels != 0, axis=1)))
        fraction = (num_labeled / n) if n > 0 else 0.0
        stats = {
            "fraction_labeled": fraction,
            "required_fraction": self.label_completeness_required,
            "num_labeled": num_labeled,
            "num_ecgs": n,
        }
        passed = fraction >= self.label_completeness_required
        if not passed:
            recommendations.append(
                f"Label completeness is low: only {fraction:.0%} of ECGs "
                f"carry a diagnostic label. Ensure annotations are exported."
            )
        return QualityCheck(
            name="label_completeness",
            passed=passed,
            blocking=True,
            message=(
                f"{fraction:.0%} of ECGs carry a diagnostic label "
                f"(required {self.label_completeness_required:.0%})."
            ),
            stats=stats,
        )

    def _check_label_diversity(
        self,
        labels: NDArray[np.float64],
        recommendations: List[str],
    ) -> Tuple[QualityCheck, Dict[str, int]]:
        slices = _task_slices(self.task_dims)
        total_width = sum(self.task_dims.values())
        per_superclass: Dict[str, int] = {}

        # Guard against a label matrix that does not match the expected
        # column layout; count zero positives for missing columns.
        cols = labels.shape[1] if labels.ndim == 2 else 0
        for task, (start, stop) in slices.items():
            if cols >= stop and labels.shape[0] > 0:
                task_block = labels[:, start:stop]
                # A record is a positive example for this superclass if any
                # column in its block is non-zero.
                positives = int(np.sum(np.any(task_block != 0, axis=1)))
            else:
                positives = 0
            per_superclass[task] = positives

        represented = [
            task
            for task, count in per_superclass.items()
            if count >= self.diversity_positive_threshold
        ]
        num_represented = len(represented)
        stats = {
            "positives_per_superclass": per_superclass,
            "represented_superclasses": represented,
            "num_represented": num_represented,
            "required": self.diversity_min_superclasses,
            "positive_threshold": self.diversity_positive_threshold,
            "expected_label_width": total_width,
            "observed_label_width": cols,
        }
        passed = num_represented >= self.diversity_min_superclasses
        if not passed:
            recommendations.append(
                f"Label diversity is low: only {num_represented} of "
                f"{len(self.task_dims)} task superclasses have "
                f">= {self.diversity_positive_threshold} positive examples. "
                f"A balanced case mix improves the aggregated model."
            )
        return (
            QualityCheck(
                name="label_diversity",
                passed=passed,
                blocking=True,
                message=(
                    f"{num_represented} of {len(self.task_dims)} superclasses "
                    f"have >= {self.diversity_positive_threshold} positives "
                    f"(required {self.diversity_min_superclasses})."
                ),
                stats=stats,
            ),
            per_superclass,
        )

    # -- Helpers ------------------------------------------------------------

    def _normalise(
        self, dataset: Any
    ) -> Tuple[List[ECGRecord], NDArray[np.float64]]:
        """Extract ``(records, labels)`` from the accepted input forms."""
        if isinstance(dataset, tuple) and len(dataset) == 2:
            records, labels = dataset
        elif hasattr(dataset, "records") and hasattr(dataset, "labels"):
            records, labels = dataset.records, dataset.labels
        else:
            raise TypeError(
                "dataset must be a (records, labels) tuple or expose "
                "`.records` and `.labels` attributes"
            )

        records = list(records)
        label_arr = np.asarray(labels, dtype=np.float64)
        if label_arr.ndim == 1:
            label_arr = label_arr.reshape(len(records), -1)
        if label_arr.size and label_arr.shape[0] != len(records):
            raise ValueError(
                f"Number of labels ({label_arr.shape[0]}) must match "
                f"number of records ({len(records)})"
            )
        return records, label_arr
