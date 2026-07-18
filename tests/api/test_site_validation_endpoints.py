"""Tests for site validation endpoints — US-114.

Tests the three new API endpoints:
- POST /api/v1/validation/sites
- GET /api/v1/validation/sites
- GET /api/v1/validation/readiness
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from aortica.api.validation_endpoints import (  # noqa: E402
    ReleaseReadinessResponse,
    SiteRegistrationResponse,
    SiteValidationListResponse,
    create_validation_router,
)
from aortica.evaluation.site_validation import SiteValidationRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(registry: SiteValidationRegistry | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(create_validation_router(site_registry=registry))
    return TestClient(app)


def _tmp_registry() -> SiteValidationRegistry:
    path = Path(tempfile.mkdtemp()) / "test_validations.json"
    return SiteValidationRegistry(path=str(path))


# ---------------------------------------------------------------------------
# POST /api/v1/validation/sites
# ---------------------------------------------------------------------------


class TestPostSites:
    """Tests for registering a new site validation."""

    def test_register_non_western(self) -> None:
        registry = _tmp_registry()
        client = _client(registry)
        resp = client.post(
            "/api/v1/validation/sites",
            json={
                "site_id": "site_mumbai",
                "region": "South Asia",
                "benchmark_report": {"overall_pass": True, "auc": 0.92},
                "dataset_size": 1500,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["site_id"] == "site_mumbai"
        assert body["region_class"] == "non-western"
        assert body["status"] == "registered"
        SiteRegistrationResponse(**body)

    def test_register_western(self) -> None:
        registry = _tmp_registry()
        client = _client(registry)
        resp = client.post(
            "/api/v1/validation/sites",
            json={
                "site_id": "site_london",
                "region": "United Kingdom",
                "benchmark_report": {"overall_pass": True},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["region_class"] == "western"

    def test_register_no_registry(self) -> None:
        """Without a registry, endpoint returns no_registry_configured."""
        client = _client(None)
        resp = client.post(
            "/api/v1/validation/sites",
            json={"site_id": "x", "region": "y"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_registry_configured"

    def test_replace_existing(self) -> None:
        """Re-registering the same site_id replaces the old entry."""
        registry = _tmp_registry()
        client = _client(registry)
        client.post(
            "/api/v1/validation/sites",
            json={"site_id": "s1", "region": "East Africa", "dataset_size": 100},
        )
        client.post(
            "/api/v1/validation/sites",
            json={"site_id": "s1", "region": "East Africa", "dataset_size": 200},
        )
        sites = client.get("/api/v1/validation/sites").json()
        assert sites["total"] == 1
        assert sites["sites"][0]["dataset_size"] == 200


# ---------------------------------------------------------------------------
# GET /api/v1/validation/sites
# ---------------------------------------------------------------------------


class TestGetSites:
    """Tests for listing all site validations."""

    def test_empty_list(self) -> None:
        registry = _tmp_registry()
        client = _client(registry)
        resp = client.get("/api/v1/validation/sites")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sites"] == []
        assert body["total"] == 0
        SiteValidationListResponse(**body)

    def test_populated_list(self) -> None:
        registry = _tmp_registry()
        registry.register_validation("s1", "South Asia", {"overall_pass": True, "auc": 0.9}, 500)
        registry.register_validation("s2", "Western Europe", {"overall_pass": False, "auc": 0.7}, 300)

        client = _client(registry)
        body = client.get("/api/v1/validation/sites").json()
        assert body["total"] == 2
        by_id = {s["site_id"]: s for s in body["sites"]}
        assert by_id["s1"]["region_class"] == "non-western"
        assert by_id["s1"]["overall_pass"] is True
        assert by_id["s2"]["region_class"] == "western"
        assert by_id["s2"]["overall_pass"] is False

    def test_no_registry(self) -> None:
        client = _client(None)
        body = client.get("/api/v1/validation/sites").json()
        assert body["total"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/validation/readiness
# ---------------------------------------------------------------------------


class TestGetReadiness:
    """Tests for release readiness check."""

    def test_not_ready_empty(self) -> None:
        registry = _tmp_registry()
        client = _client(registry)
        resp = client.get("/api/v1/validation/readiness")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is False
        assert body["non_western_count"] == 0
        ReleaseReadinessResponse(**body)

    def test_not_ready_insufficient(self) -> None:
        registry = _tmp_registry()
        registry.register_validation("s1", "South Asia", {})
        client = _client(registry)
        body = client.get("/api/v1/validation/readiness").json()
        assert body["ready"] is False
        assert body["non_western_count"] == 1

    def test_ready_with_two_non_western(self) -> None:
        registry = _tmp_registry()
        registry.register_validation("s1", "South Asia", {})
        registry.register_validation("s2", "East Africa", {})
        client = _client(registry)
        body = client.get("/api/v1/validation/readiness").json()
        assert body["ready"] is True
        assert body["non_western_count"] == 2
        assert "s1" in body["non_western_sites"]
        assert "s2" in body["non_western_sites"]

    def test_western_only_not_ready(self) -> None:
        registry = _tmp_registry()
        registry.register_validation("s1", "Western Europe", {})
        registry.register_validation("s2", "North America", {})
        client = _client(registry)
        body = client.get("/api/v1/validation/readiness").json()
        assert body["ready"] is False
        assert body["western_count"] == 2
        assert body["non_western_count"] == 0

    def test_no_registry(self) -> None:
        client = _client(None)
        body = client.get("/api/v1/validation/readiness").json()
        assert body["ready"] is False


# ---------------------------------------------------------------------------
# Full app integration
# ---------------------------------------------------------------------------


class TestSiteValidationInFullApp:
    """Verify endpoints are wired into the main app factory."""

    def test_routes_registered(self) -> None:
        from aortica.api.app import create_app

        client = TestClient(create_app(model_loaded=False, enable_auth=False))

        resp_sites = client.get("/api/v1/validation/sites")
        assert resp_sites.status_code == 200

        resp_readiness = client.get("/api/v1/validation/readiness")
        assert resp_readiness.status_code == 200

    def test_post_and_get(self) -> None:
        from aortica.api.app import create_app

        client = TestClient(create_app(model_loaded=False, enable_auth=False))

        # Register a site
        resp = client.post(
            "/api/v1/validation/sites",
            json={"site_id": "integ_test", "region": "East Asia"},
        )
        assert resp.status_code == 200
        assert resp.json()["region_class"] == "non-western"
