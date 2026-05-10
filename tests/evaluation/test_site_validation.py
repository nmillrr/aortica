"""Tests for site_validation module (US-071)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aortica.evaluation.site_validation import (
    ReleaseReadiness,
    SiteValidation,
    SiteValidationRegistry,
    WESTERN_REGIONS,
    classify_region,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark_dict(n_samples: int = 500) -> dict[str, Any]:
    """Create a minimal benchmark report dict for testing."""
    return {
        "n_samples": n_samples,
        "tasks_evaluated": ["rhythm"],
        "overall": {
            "rhythm": {
                "task_name": "rhythm",
                "macro_f1": 0.88,
                "ece": 0.05,
                "per_class": [],
            }
        },
        "subgroups": [],
    }


class _FakeBenchmarkReport:
    """Minimal object mimicking BenchmarkReport for testing."""

    def __init__(self, n_samples: int = 500) -> None:
        self.n_samples = n_samples

    def as_dict(self) -> dict[str, Any]:
        return _make_benchmark_dict(self.n_samples)


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

class TestClassifyRegion:
    def test_western_north_america(self) -> None:
        assert classify_region("North America") == "western"
        assert classify_region("United States") == "western"
        assert classify_region("USA") == "western"
        assert classify_region("Canada") == "western"

    def test_western_europe(self) -> None:
        assert classify_region("Western Europe") == "western"
        assert classify_region("United Kingdom") == "western"
        assert classify_region("France") == "western"
        assert classify_region("Germany") == "western"
        assert classify_region("Italy") == "western"
        assert classify_region("Spain") == "western"

    def test_western_oceania(self) -> None:
        assert classify_region("Australia") == "western"
        assert classify_region("New Zealand") == "western"
        assert classify_region("Australia/NZ") == "western"

    def test_non_western(self) -> None:
        assert classify_region("South Asia") == "non-western"
        assert classify_region("East Africa") == "non-western"
        assert classify_region("Southeast Asia") == "non-western"
        assert classify_region("Latin America") == "non-western"
        assert classify_region("Middle East") == "non-western"
        assert classify_region("East Asia") == "non-western"
        assert classify_region("Central Africa") == "non-western"

    def test_case_insensitive(self) -> None:
        assert classify_region("FRANCE") == "western"
        assert classify_region("france") == "western"
        assert classify_region("  France  ") == "western"

    def test_unknown_is_non_western(self) -> None:
        assert classify_region("Unknown Region") == "non-western"
        assert classify_region("") == "non-western"

    def test_western_regions_frozen(self) -> None:
        assert isinstance(WESTERN_REGIONS, frozenset)
        assert len(WESTERN_REGIONS) > 0


# ---------------------------------------------------------------------------
# SiteValidation dataclass
# ---------------------------------------------------------------------------

class TestSiteValidation:
    def test_defaults(self) -> None:
        sv = SiteValidation()
        assert sv.site_id == ""
        assert sv.region == ""
        assert sv.region_class == ""
        assert sv.dataset_size == 0
        assert sv.benchmark_summary == {}
        assert sv.timestamp == ""

    def test_custom_values(self) -> None:
        sv = SiteValidation(
            site_id="site_mumbai",
            region="South Asia",
            region_class="non-western",
            dataset_size=1000,
            benchmark_summary={"macro_f1": 0.89},
            timestamp="2026-01-15T10:00:00+00:00",
        )
        assert sv.site_id == "site_mumbai"
        assert sv.region_class == "non-western"
        assert sv.dataset_size == 1000


# ---------------------------------------------------------------------------
# ReleaseReadiness dataclass
# ---------------------------------------------------------------------------

class TestReleaseReadiness:
    def test_defaults(self) -> None:
        rr = ReleaseReadiness()
        assert rr.ready is False
        assert rr.total_validations == 0
        assert rr.min_non_western == 2

    def test_ready(self) -> None:
        rr = ReleaseReadiness(
            ready=True,
            total_validations=3,
            western_count=1,
            non_western_count=2,
            non_western_sites=["site_a", "site_b"],
            western_sites=["site_c"],
        )
        assert rr.ready is True

    def test_summary_ready(self) -> None:
        rr = ReleaseReadiness(
            ready=True,
            total_validations=2,
            non_western_count=2,
            non_western_sites=["site_a", "site_b"],
        )
        s = rr.summary()
        assert "READY" in s
        assert "site_a" in s
        assert "site_b" in s

    def test_summary_not_ready(self) -> None:
        rr = ReleaseReadiness(ready=False, total_validations=0)
        s = rr.summary()
        assert "NOT READY" in s


# ---------------------------------------------------------------------------
# Registry — registration
# ---------------------------------------------------------------------------

class TestRegistryRegistration:
    def test_register_with_dict(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        report = _make_benchmark_dict(300)
        sv = registry.register_validation("site_nairobi", "East Africa", report)
        assert sv.site_id == "site_nairobi"
        assert sv.region == "East Africa"
        assert sv.region_class == "non-western"
        assert sv.dataset_size == 300
        assert sv.timestamp != ""

    def test_register_with_object(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        report = _FakeBenchmarkReport(n_samples=750)
        sv = registry.register_validation("site_delhi", "South Asia", report)
        assert sv.dataset_size == 750
        assert sv.benchmark_summary["n_samples"] == 750

    def test_register_replaces_existing(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("site_a", "East Africa", {"n_samples": 100})
        registry.register_validation("site_a", "East Africa", {"n_samples": 200})
        assert len(registry.get_validations()) == 1
        assert registry.get_validation("site_a") is not None
        assert registry.get_validation("site_a").dataset_size == 200  # type: ignore[union-attr]

    def test_register_custom_timestamp(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        ts = "2026-03-01T12:00:00+00:00"
        sv = registry.register_validation(
            "site_b", "South Asia", {}, timestamp=ts,
        )
        assert sv.timestamp == ts

    def test_register_custom_dataset_size(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        sv = registry.register_validation(
            "site_c", "East Africa", {}, dataset_size=999,
        )
        assert sv.dataset_size == 999


# ---------------------------------------------------------------------------
# Registry — retrieval
# ---------------------------------------------------------------------------

class TestRegistryRetrieval:
    def test_get_validations(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        registry.register_validation("s2", "South Asia", {})
        assert len(registry.get_validations()) == 2

    def test_get_validation_by_id(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        v = registry.get_validation("s1")
        assert v is not None
        assert v.site_id == "s1"

    def test_get_validation_missing(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        assert registry.get_validation("nonexistent") is None

    def test_remove_validation(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        assert registry.remove_validation("s1") is True
        assert len(registry.get_validations()) == 0

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        assert registry.remove_validation("none") is False


# ---------------------------------------------------------------------------
# Registry — release readiness
# ---------------------------------------------------------------------------

class TestReleaseReadinessCheck:
    def test_empty_registry_not_ready(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        result = registry.check_release_readiness()
        assert result.ready is False
        assert result.total_validations == 0

    def test_one_non_western_not_ready(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        result = registry.check_release_readiness()
        assert result.ready is False
        assert result.non_western_count == 1

    def test_two_non_western_ready(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        registry.register_validation("s2", "South Asia", {})
        result = registry.check_release_readiness()
        assert result.ready is True
        assert result.non_western_count == 2

    def test_western_only_not_ready(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "United States", {})
        registry.register_validation("s2", "France", {})
        registry.register_validation("s3", "Australia", {})
        result = registry.check_release_readiness()
        assert result.ready is False
        assert result.western_count == 3
        assert result.non_western_count == 0

    def test_mixed_sites_ready(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "USA", {})
        registry.register_validation("s2", "East Africa", {})
        registry.register_validation("s3", "South Asia", {})
        result = registry.check_release_readiness()
        assert result.ready is True
        assert result.western_count == 1
        assert result.non_western_count == 2

    def test_custom_min_non_western(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(
            path=tmp_path / "reg.json", min_non_western=3,
        )
        registry.register_validation("s1", "East Africa", {})
        registry.register_validation("s2", "South Asia", {})
        result = registry.check_release_readiness()
        assert result.ready is False
        assert result.min_non_western == 3

    def test_override_min_non_western(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("s1", "East Africa", {})
        result = registry.check_release_readiness(min_non_western=1)
        assert result.ready is True

    def test_sites_are_sorted(self, tmp_path: Path) -> None:
        registry = SiteValidationRegistry(path=tmp_path / "reg.json")
        registry.register_validation("z_site", "East Africa", {})
        registry.register_validation("a_site", "South Asia", {})
        result = registry.check_release_readiness()
        assert result.non_western_sites == ["a_site", "z_site"]


# ---------------------------------------------------------------------------
# Registry — JSON persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        reg1 = SiteValidationRegistry(path=path)
        reg1.register_validation("s1", "East Africa", {"n_samples": 100})
        reg1.register_validation("s2", "France", {"n_samples": 200})

        # Load in a new instance
        reg2 = SiteValidationRegistry(path=path)
        assert len(reg2.get_validations()) == 2
        v1 = reg2.get_validation("s1")
        assert v1 is not None
        assert v1.region == "East Africa"
        assert v1.region_class == "non-western"
        assert v1.dataset_size == 100

    def test_persistence_file_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        registry = SiteValidationRegistry(path=path)
        registry.register_validation("s1", "East Africa", {})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["site_id"] == "s1"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        registry = SiteValidationRegistry(path=path)
        assert len(registry.get_validations()) == 0

    def test_load_corrupted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{", encoding="utf-8")
        registry = SiteValidationRegistry(path=path)
        assert len(registry.get_validations()) == 0

    def test_load_wrong_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        registry = SiteValidationRegistry(path=path)
        assert len(registry.get_validations()) == 0

    def test_remove_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "reg.json"
        reg1 = SiteValidationRegistry(path=path)
        reg1.register_validation("s1", "East Africa", {})
        reg1.remove_validation("s1")

        reg2 = SiteValidationRegistry(path=path)
        assert len(reg2.get_validations()) == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "reg.json"
        registry = SiteValidationRegistry(path=path)
        registry.register_validation("s1", "East Africa", {})
        assert path.exists()


# ---------------------------------------------------------------------------
# Import / module tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_top_level_import(self) -> None:
        from aortica.evaluation.site_validation import (  # noqa: F811
            SiteValidationRegistry,
        )
        assert SiteValidationRegistry is not None

    def test_dataclasses_importable(self) -> None:
        from aortica.evaluation.site_validation import (  # noqa: F811
            ReleaseReadiness,
            SiteValidation,
        )
        assert SiteValidation is not None
        assert ReleaseReadiness is not None

    def test_classify_region_importable(self) -> None:
        from aortica.evaluation.site_validation import (  # noqa: F811
            classify_region,
        )
        assert callable(classify_region)


# ---------------------------------------------------------------------------
# Typecheck placeholder
# ---------------------------------------------------------------------------

class TestTypecheck:
    def test_module_attributes(self) -> None:
        import importlib
        mod = importlib.import_module("aortica.evaluation.site_validation")
        assert hasattr(mod, "SiteValidationRegistry")
        assert hasattr(mod, "SiteValidation")
        assert hasattr(mod, "ReleaseReadiness")
        assert hasattr(mod, "classify_region")
        assert hasattr(mod, "WESTERN_REGIONS")
