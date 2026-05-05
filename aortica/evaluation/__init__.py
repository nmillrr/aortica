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

__all__ = [
    "BenchmarkReport",
    "ClassMetrics",
    "ComparisonResult",
    "EquityGateResult",
    "GroupMetrics",
    "SubgroupReport",
    "TaskReport",
    "benchmark",
    "equity_gate",
]
