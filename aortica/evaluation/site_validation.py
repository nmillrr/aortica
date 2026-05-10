"""Non-Western site validation tracker (US-071).

Tracks multi-site validation results and enforces a release readiness
gate requiring at least 2 non-Western site validations before a model
can be promoted to ``v-stable``.

Region classification:
    - **Western:** North America, Western Europe, Australia/NZ
    - **Non-Western:** All other regions

Usage::

    from aortica.evaluation.site_validation import SiteValidationRegistry

    registry = SiteValidationRegistry(path="validations.json")
    registry.register_validation("site_mumbai", "South Asia", benchmark_report)
    registry.register_validation("site_nairobi", "East Africa", benchmark_report)

    readiness = registry.check_release_readiness()
    assert readiness.ready is True

Persistence is to a JSON file so that validation state survives across
process restarts and CI runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

#: Regions considered "Western" for the purposes of the equity release gate.
WESTERN_REGIONS: frozenset[str] = frozenset({
    # North America
    "north america",
    "united states",
    "usa",
    "canada",
    # Western Europe
    "western europe",
    "united kingdom",
    "uk",
    "france",
    "germany",
    "italy",
    "spain",
    "netherlands",
    "belgium",
    "switzerland",
    "austria",
    "ireland",
    "portugal",
    "sweden",
    "norway",
    "denmark",
    "finland",
    "luxembourg",
    "iceland",
    # Australia / New Zealand
    "australia",
    "new zealand",
    "australia/nz",
})


def classify_region(region: str) -> str:
    """Classify a free-text region string as ``'western'`` or ``'non-western'``.

    The comparison is case-insensitive and strips leading/trailing whitespace.

    Args:
        region: Free-text region identifier (e.g. ``"South Asia"``,
            ``"Western Europe"``).

    Returns:
        ``"western"`` or ``"non-western"``.
    """
    normalised = region.strip().lower()
    if normalised in WESTERN_REGIONS:
        return "western"
    return "non-western"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SiteValidation:
    """A single site validation record.

    Attributes:
        site_id: Unique identifier for the validation site.
        region: Free-text region string (e.g. ``"South Asia"``).
        region_class: Computed classification (``"western"`` or
            ``"non-western"``).
        dataset_size: Number of samples in the validation dataset.
        benchmark_summary: Flat dict summary of the benchmark report
            (serialisable to JSON).
        timestamp: ISO-8601 timestamp of when the validation was registered.
    """

    site_id: str = ""
    region: str = ""
    region_class: str = ""
    dataset_size: int = 0
    benchmark_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class ReleaseReadiness:
    """Result of the release readiness check.

    Attributes:
        ready: Overall pass/fail — ``True`` iff ≥ ``min_non_western``
            non-Western site validations are registered.
        total_validations: Total number of registered site validations.
        western_count: Number of Western site validations.
        non_western_count: Number of non-Western site validations.
        min_non_western: Required minimum non-Western validations.
        non_western_sites: List of non-Western site IDs that have
            registered validations.
        western_sites: List of Western site IDs that have registered
            validations.
    """

    ready: bool = False
    total_validations: int = 0
    western_count: int = 0
    non_western_count: int = 0
    min_non_western: int = 2
    non_western_sites: list[str] = field(default_factory=list)
    western_sites: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the readiness check."""
        status = "READY ✓" if self.ready else "NOT READY ✗"
        lines: list[str] = [
            f"Release Readiness: {status}",
            f"  Total validations: {self.total_validations}",
            f"  Western sites ({self.western_count}): "
            f"{', '.join(self.western_sites) or '(none)'}",
            f"  Non-Western sites ({self.non_western_count}): "
            f"{', '.join(self.non_western_sites) or '(none)'}",
            f"  Required non-Western: ≥{self.min_non_western}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SiteValidationRegistry:
    """Persistent registry of multi-site validation results.

    Stores validation records in a JSON file and provides a release
    readiness check requiring a configurable minimum number of
    non-Western site validations.

    Args:
        path: Path to the JSON persistence file. Created automatically
            on first write if it does not exist.
        min_non_western: Minimum non-Western validations required for
            release readiness. Default ``2``.
    """

    def __init__(
        self,
        path: str | Path = "site_validations.json",
        min_non_western: int = 2,
    ) -> None:
        self._path = Path(path)
        self._min_non_western = min_non_western
        self._validations: list[SiteValidation] = []
        self._load()

    # -- public API ---------------------------------------------------------

    def register_validation(
        self,
        site_id: str,
        region: str,
        benchmark_report: Any,
        dataset_size: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> SiteValidation:
        """Register a new site validation.

        If a validation for the same ``site_id`` already exists it is
        **replaced** (latest result wins).

        Args:
            site_id: Unique site identifier.
            region: Free-text region string.
            benchmark_report: A :class:`BenchmarkReport` instance **or**
                a plain dict (e.g. from ``BenchmarkReport.as_dict()``).
            dataset_size: Number of samples. If ``None``, attempts to
                read ``benchmark_report.n_samples`` or the ``"n_samples"``
                key from a dict.
            timestamp: ISO-8601 timestamp string. If ``None``, the
                current UTC time is used.

        Returns:
            The created :class:`SiteValidation` record.
        """
        # Resolve benchmark summary
        if hasattr(benchmark_report, "as_dict"):
            summary = benchmark_report.as_dict()
        elif isinstance(benchmark_report, dict):
            summary = benchmark_report
        else:
            summary = {"raw": str(benchmark_report)}

        # Resolve dataset size
        if dataset_size is None:
            if hasattr(benchmark_report, "n_samples"):
                dataset_size = int(benchmark_report.n_samples)
            elif isinstance(benchmark_report, dict):
                dataset_size = int(benchmark_report.get("n_samples", 0))
            else:
                dataset_size = 0

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        region_class = classify_region(region)

        validation = SiteValidation(
            site_id=site_id,
            region=region,
            region_class=region_class,
            dataset_size=dataset_size,
            benchmark_summary=summary,
            timestamp=timestamp,
        )

        # Replace existing validation for same site_id
        self._validations = [
            v for v in self._validations if v.site_id != site_id
        ]
        self._validations.append(validation)
        self._save()

        return validation

    def check_release_readiness(
        self,
        min_non_western: Optional[int] = None,
    ) -> ReleaseReadiness:
        """Check whether the release readiness gate is satisfied.

        Args:
            min_non_western: Override for the minimum non-Western count.
                If ``None``, uses the value set at construction time.

        Returns:
            :class:`ReleaseReadiness` with pass/fail and details.
        """
        required = (
            min_non_western
            if min_non_western is not None
            else self._min_non_western
        )

        western_sites: list[str] = []
        non_western_sites: list[str] = []

        for v in self._validations:
            if v.region_class == "western":
                western_sites.append(v.site_id)
            else:
                non_western_sites.append(v.site_id)

        return ReleaseReadiness(
            ready=len(non_western_sites) >= required,
            total_validations=len(self._validations),
            western_count=len(western_sites),
            non_western_count=len(non_western_sites),
            min_non_western=required,
            non_western_sites=sorted(non_western_sites),
            western_sites=sorted(western_sites),
        )

    def get_validations(self) -> list[SiteValidation]:
        """Return all registered validations (defensive copy)."""
        return list(self._validations)

    def get_validation(self, site_id: str) -> Optional[SiteValidation]:
        """Return validation for a specific site, or ``None``."""
        for v in self._validations:
            if v.site_id == site_id:
                return v
        return None

    def remove_validation(self, site_id: str) -> bool:
        """Remove a validation by site ID.

        Returns:
            ``True`` if a validation was removed, ``False`` if not found.
        """
        before = len(self._validations)
        self._validations = [
            v for v in self._validations if v.site_id != site_id
        ]
        removed = len(self._validations) < before
        if removed:
            self._save()
        return removed

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        """Load validations from the JSON file, if it exists."""
        if not self._path.exists():
            self._validations = []
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._validations = []
            return

        if not isinstance(data, list):
            self._validations = []
            return

        loaded: list[SiteValidation] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            loaded.append(
                SiteValidation(
                    site_id=entry.get("site_id", ""),
                    region=entry.get("region", ""),
                    region_class=entry.get("region_class", ""),
                    dataset_size=entry.get("dataset_size", 0),
                    benchmark_summary=entry.get("benchmark_summary", {}),
                    timestamp=entry.get("timestamp", ""),
                )
            )
        self._validations = loaded

    def _save(self) -> None:
        """Persist validations to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(v) for v in self._validations]
        self._path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
