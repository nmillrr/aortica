"""Cross-hardware benchmark suite with platform-specific pass/fail targets.

Provides systematic benchmarking against per-platform latency, memory, and
throughput targets defined in ``platform_targets.yaml``.  Integrates with
the existing :mod:`aortica.edge.profiling` module for actual measurements
and adds pass/fail gating for CI release pipelines.

The primary entry points are:

- :func:`hardware_benchmark` — run a benchmark for a single platform profile
- :func:`benchmark_all_platforms` — run benchmarks for all locally-available
  platforms and produce a consolidated comparison table

Example::

    from aortica.edge.hardware_benchmark import hardware_benchmark

    report = hardware_benchmark(
        model_path="aortica_edge_int8.onnx",
        platform_profile="server_cpu",
    )
    print(report.summary_table())
    assert report.overall_pass

"""

from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


# ---------------------------------------------------------------------------
# Platform profile
# ---------------------------------------------------------------------------

_TARGETS_YAML = Path(__file__).parent / "platform_targets.yaml"


@dataclass
class PlatformProfile:
    """A hardware platform's benchmark targets.

    Attributes:
        name: Platform identifier (e.g. ``rpi4``, ``server_cpu``).
        arch: CPU architecture (``arm64``, ``amd64``).
        accelerator: GPU/accelerator type (``none``, ``cuda``, ``nnapi``).
        latency_p95_ms: Maximum allowable p95 latency in milliseconds.
        peak_memory_mb: Maximum peak memory (RSS) in megabytes.
        min_throughput_ips: Minimum throughput (inferences/second). 0 = not checked.
        input_mode: Expected lead configuration (``12-lead``, ``6-lead``, ``single-lead``).
        description: Human-readable platform description.
    """

    name: str
    arch: str = "amd64"
    accelerator: str = "none"
    latency_p95_ms: float = 100.0
    peak_memory_mb: float = 2048.0
    min_throughput_ips: float = 0.0
    input_mode: str = "12-lead"
    description: str = ""


def load_platform_profiles(
    yaml_path: Optional[str] = None,
) -> Dict[str, PlatformProfile]:
    """Load platform profiles from the YAML targets file.

    Parameters
    ----------
    yaml_path:
        Path to a custom ``platform_targets.yaml``.  Defaults to the
        bundled file shipped with Aortica.

    Returns
    -------
    dict[str, PlatformProfile]
        Mapping from platform name to :class:`PlatformProfile`.
    """
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to load platform targets. "
            "Install with: pip install pyyaml"
        )

    path = Path(yaml_path) if yaml_path else _TARGETS_YAML
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    profiles: Dict[str, PlatformProfile] = {}
    for name, cfg in data.items():
        profiles[name] = PlatformProfile(
            name=name,
            arch=cfg.get("arch", "amd64"),
            accelerator=cfg.get("accelerator", "none"),
            latency_p95_ms=float(cfg.get("latency_p95_ms", 100)),
            peak_memory_mb=float(cfg.get("peak_memory_mb", 2048)),
            min_throughput_ips=float(cfg.get("min_throughput_ips", 0)),
            input_mode=cfg.get("input_mode", "12-lead"),
            description=cfg.get("description", ""),
        )

    return profiles


# ---------------------------------------------------------------------------
# Metric result
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Pass/fail result for a single benchmark metric."""

    metric_name: str
    measured: float
    target: float
    passed: bool
    unit: str = ""
    comparison: str = "<="  # "<=" means measured must be <= target

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{self.metric_name}: {self.measured:.2f}{self.unit} "
            f"(target {self.comparison} {self.target:.2f}{self.unit}) "
            f"[{status}]"
        )


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------


@dataclass
class HardwareBenchmarkReport:
    """Results of a hardware benchmark run against a specific platform profile.

    Attributes:
        platform_name: Platform identifier.
        model_variant: Model variant string (e.g. ``full``, ``edge_int8``).
        mean_latency_ms: Mean inference latency in milliseconds.
        p50_latency_ms: Median latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        peak_memory_mb: Peak memory usage (RSS) in megabytes.
        throughput_ips: Throughput in inferences per second.
        model_size_bytes: Model file size on disk in bytes.
        metric_results: List of individual metric pass/fail results.
        overall_pass: Whether all metric targets were met.
        n_runs: Number of inference runs performed.
        latencies_ms: Raw latency measurements.
    """

    platform_name: str = ""
    model_variant: str = ""
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_ips: float = 0.0
    model_size_bytes: int = 0
    metric_results: List[MetricResult] = field(default_factory=list)
    overall_pass: bool = False
    n_runs: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def model_size_mb(self) -> float:
        """Model file size in megabytes."""
        return self.model_size_bytes / (1024 * 1024)

    def summary_table(self) -> str:
        """Return a human-readable benchmark summary with pass/fail status."""
        lines: List[str] = []
        lines.append(f"Hardware Benchmark Report — {self.platform_name}")
        lines.append("=" * 60)
        lines.append(f"  Model variant:      {self.model_variant}")
        lines.append(f"  Model size:         {self.model_size_mb:.2f} MB")
        lines.append(f"  Runs:               {self.n_runs}")
        lines.append("")
        lines.append("Latency")
        lines.append("-" * 60)
        lines.append(f"  Mean:               {self.mean_latency_ms:.2f} ms")
        lines.append(f"  Median (p50):       {self.p50_latency_ms:.2f} ms")
        lines.append(f"  p95:                {self.p95_latency_ms:.2f} ms")
        lines.append(f"  p99:                {self.p99_latency_ms:.2f} ms")
        lines.append("")
        lines.append("Resources")
        lines.append("-" * 60)
        lines.append(f"  Peak memory:        {self.peak_memory_mb:.2f} MB")
        lines.append(f"  Throughput:         {self.throughput_ips:.2f} inferences/s")
        lines.append("")
        lines.append("Metric Results")
        lines.append("-" * 60)
        for mr in self.metric_results:
            lines.append(f"  {mr}")
        lines.append("")
        status = "✅ ALL TARGETS MET" if self.overall_pass else "❌ TARGETS MISSED"
        lines.append(f"Overall: {status}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise report to a plain dictionary."""
        return {
            "platform_name": self.platform_name,
            "model_variant": self.model_variant,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "throughput_ips": self.throughput_ips,
            "model_size_bytes": self.model_size_bytes,
            "model_size_mb": self.model_size_mb,
            "n_runs": self.n_runs,
            "overall_pass": self.overall_pass,
            "metric_results": [
                {
                    "metric_name": mr.metric_name,
                    "measured": mr.measured,
                    "target": mr.target,
                    "passed": mr.passed,
                    "unit": mr.unit,
                }
                for mr in self.metric_results
            ],
        }


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def _input_shape_for_mode(input_mode: str) -> tuple[int, int, int]:
    """Return (batch, leads, samples) for a given input mode."""
    if input_mode == "single-lead":
        return (1, 1, 5000)
    elif input_mode == "6-lead":
        return (1, 6, 5000)
    else:  # 12-lead default
        return (1, 12, 5000)


def _measure_peak_memory() -> float:
    """Measure current process peak RSS in MB."""
    try:
        import resource

        # maxrss on macOS is in bytes, on Linux in KB
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        else:
            return usage / 1024
    except (ImportError, AttributeError):
        return 0.0


def hardware_benchmark(
    model_path: str,
    platform_profile: str,
    dataset_sample: Optional[npt.NDArray[np.float32]] = None,
    n_runs: int = 50,
    targets_yaml: Optional[str] = None,
    *,
    profiles: Optional[Dict[str, PlatformProfile]] = None,
) -> HardwareBenchmarkReport:
    """Run a full inference benchmark against a specific platform profile.

    Parameters
    ----------
    model_path:
        Path to an ONNX model file.
    platform_profile:
        Name of the platform profile (e.g. ``server_cpu``, ``rpi4``).
    dataset_sample:
        Optional input data array.  If *None*, synthetic data matching
        the platform's input mode is generated.
    n_runs:
        Number of inference runs (default 50).
    targets_yaml:
        Path to a custom platform targets YAML.  Defaults to bundled.
    profiles:
        Pre-loaded platform profiles (for testing; avoids YAML dependency).

    Returns
    -------
    HardwareBenchmarkReport
        Benchmark results with per-metric pass/fail status.
    """
    # Load profiles
    if profiles is None:
        all_profiles = load_platform_profiles(targets_yaml)
    else:
        all_profiles = profiles

    if platform_profile not in all_profiles:
        available = ", ".join(sorted(all_profiles.keys()))
        raise ValueError(
            f"Unknown platform profile '{platform_profile}'. "
            f"Available: {available}"
        )

    profile = all_profiles[platform_profile]

    # Resolve model path and size
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_size_bytes = model_file.stat().st_size

    # Determine model variant from filename
    model_variant = "unknown"
    name_lower = model_file.name.lower()
    if "int8" in name_lower:
        model_variant = "edge_int8"
    elif "edge" in name_lower:
        model_variant = "edge"
    elif "full" in name_lower:
        model_variant = "full"
    else:
        model_variant = model_file.stem

    # Build input data
    input_shape = _input_shape_for_mode(profile.input_mode)
    if dataset_sample is not None:
        input_data = dataset_sample
    else:
        input_data = np.random.randn(*input_shape).astype(np.float32)

    # Run benchmark
    if not HAS_ONNXRUNTIME:
        raise ImportError(
            "onnxruntime is required for hardware benchmarking. "
            "Install with: pip install aortica[edge]"
        )

    session = ort.InferenceSession(str(model_file))  # type: ignore[union-attr]
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(min(5, n_runs)):
        session.run(None, {input_name: input_data})

    # Timed runs
    latencies: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_arr = np.array(latencies)
    mean_lat = float(np.mean(latencies_arr))
    p50_lat = float(np.percentile(latencies_arr, 50))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    peak_mem = _measure_peak_memory()

    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Evaluate against targets
    metric_results: List[MetricResult] = []

    # Latency p95
    lat_pass = p95_lat <= profile.latency_p95_ms
    metric_results.append(
        MetricResult(
            metric_name="latency_p95",
            measured=p95_lat,
            target=profile.latency_p95_ms,
            passed=lat_pass,
            unit=" ms",
            comparison="<=",
        )
    )

    # Peak memory
    mem_pass = peak_mem <= profile.peak_memory_mb
    metric_results.append(
        MetricResult(
            metric_name="peak_memory",
            measured=peak_mem,
            target=profile.peak_memory_mb,
            passed=mem_pass,
            unit=" MB",
            comparison="<=",
        )
    )

    # Throughput (if target specified)
    if profile.min_throughput_ips > 0:
        tp_pass = throughput >= profile.min_throughput_ips
        metric_results.append(
            MetricResult(
                metric_name="throughput",
                measured=throughput,
                target=profile.min_throughput_ips,
                passed=tp_pass,
                unit=" ips",
                comparison=">=",
            )
        )

    overall_pass = all(mr.passed for mr in metric_results)

    return HardwareBenchmarkReport(
        platform_name=platform_profile,
        model_variant=model_variant,
        mean_latency_ms=mean_lat,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        p99_latency_ms=p99_lat,
        peak_memory_mb=peak_mem,
        throughput_ips=throughput,
        model_size_bytes=model_size_bytes,
        metric_results=metric_results,
        overall_pass=overall_pass,
        n_runs=n_runs,
        latencies_ms=latencies,
    )


# ---------------------------------------------------------------------------
# Benchmark all platforms
# ---------------------------------------------------------------------------


def benchmark_all_platforms(
    model_path: str,
    n_runs: int = 50,
    targets_yaml: Optional[str] = None,
    *,
    profiles: Optional[Dict[str, PlatformProfile]] = None,
) -> List[HardwareBenchmarkReport]:
    """Run benchmarks for all locally-available platform profiles.

    Currently, benchmarks run on the local CPU simulating each platform's
    input mode.  In a real CI environment, platform-specific runners
    would execute the appropriate profile.

    Parameters
    ----------
    model_path:
        Path to an ONNX model file.
    n_runs:
        Number of inference runs per platform.
    targets_yaml:
        Path to a custom platform targets YAML.
    profiles:
        Pre-loaded platform profiles (for testing).

    Returns
    -------
    list[HardwareBenchmarkReport]
        One report per platform, sorted by platform name.
    """
    if profiles is None:
        all_profiles = load_platform_profiles(targets_yaml)
    else:
        all_profiles = profiles

    reports: List[HardwareBenchmarkReport] = []
    for name in sorted(all_profiles.keys()):
        report = hardware_benchmark(
            model_path=model_path,
            platform_profile=name,
            n_runs=n_runs,
            profiles=all_profiles,
        )
        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def consolidated_markdown_table(reports: List[HardwareBenchmarkReport]) -> str:
    """Generate a consolidated markdown comparison table from multiple reports."""
    lines: List[str] = []
    lines.append("# Cross-Hardware Benchmark Results")
    lines.append("")
    lines.append(
        "| Platform | Model | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Memory (MB) | Throughput (ips) | Pass |"
    )
    lines.append(
        "|----------|-------|----------|----------|----------|"
        "-------------|------------------|------|"
    )

    for r in reports:
        status = "✅" if r.overall_pass else "❌"
        lines.append(
            f"| {r.platform_name} | {r.model_variant} "
            f"| {r.p50_latency_ms:.1f} | {r.p95_latency_ms:.1f} "
            f"| {r.p99_latency_ms:.1f} | {r.peak_memory_mb:.1f} "
            f"| {r.throughput_ips:.1f} | {status} |"
        )

    lines.append("")
    all_pass = all(r.overall_pass for r in reports)
    if all_pass:
        lines.append("**All platform targets met.** ✅")
    else:
        failed = [r.platform_name for r in reports if not r.overall_pass]
        lines.append(f"**Targets missed for:** {', '.join(failed)} ❌")

    return "\n".join(lines)


def consolidated_csv(reports: List[HardwareBenchmarkReport]) -> str:
    """Generate a CSV string from multiple benchmark reports."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "platform",
            "model_variant",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "peak_memory_mb",
            "throughput_ips",
            "model_size_mb",
            "overall_pass",
        ]
    )
    for r in reports:
        writer.writerow(
            [
                r.platform_name,
                r.model_variant,
                f"{r.mean_latency_ms:.2f}",
                f"{r.p50_latency_ms:.2f}",
                f"{r.p95_latency_ms:.2f}",
                f"{r.p99_latency_ms:.2f}",
                f"{r.peak_memory_mb:.2f}",
                f"{r.throughput_ips:.2f}",
                f"{r.model_size_mb:.2f}",
                str(r.overall_pass),
            ]
        )
    return output.getvalue()
