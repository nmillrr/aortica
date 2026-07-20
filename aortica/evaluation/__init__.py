"""Benchmarking, metrics, and demographic subgroup reporting."""

from aortica.evaluation.benchmark import (
    BenchmarkReport,
    ClassMetrics,
    SubgroupReport,
    TaskReport,
    benchmark,
)
from aortica.evaluation.equity_gate import (
    ComparisonResult,
    EquityGateResult,
    GroupMetrics,
    equity_gate,
)
from aortica.evaluation.model_comparison import (
    ClassDelta,
    ModelComparisonReport,
    SubgroupDelta,
    TaskDelta,
    compare_models,
    compare_predictions,
)
from aortica.evaluation.performance_card import (
    PerformanceCardResult,
    generate_performance_card,
)
from aortica.evaluation.regulatory_gate import (
    ClassGateResult,
    RegulatoryGateResult,
    regulatory_gate,
)
from aortica.evaluation.site_validation import (
    ReleaseReadiness,
    SiteValidation,
    SiteValidationRegistry,
    classify_region,
)

__all__ = [
    "BenchmarkReport",
    "ClassDelta",
    "ClassGateResult",
    "ClassMetrics",
    "ComparisonResult",
    "ModelComparisonReport",
    "SubgroupDelta",
    "TaskDelta",
    "compare_models",
    "compare_predictions",
    "EquityGateResult",
    "GroupMetrics",
    "PerformanceCardResult",
    "RegulatoryGateResult",
    "ReleaseReadiness",
    "SiteValidation",
    "SiteValidationRegistry",
    "SubgroupReport",
    "TaskReport",
    "benchmark",
    "classify_region",
    "equity_gate",
    "generate_performance_card",
    "regulatory_gate",
]

