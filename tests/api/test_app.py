"""Tests for aortica.api.app — FastAPI application scaffold."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import (  # noqa: E402
    DEFAULT_TASK_HEADS,
    SUPPORTED_FORMATS,
    HealthResponse,
    InfoResponse,
    create_app,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> fastapi.FastAPI:
    """Return a default-configured FastAPI application."""
    return create_app()


@pytest.fixture()
def client(app: fastapi.FastAPI) -> TestClient:
    """Synchronous test client for the default app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_construction(self) -> None:
        resp = HealthResponse(status="ok")
        assert resp.status == "ok"

    def test_json_roundtrip(self) -> None:
        resp = HealthResponse(status="ok")
        data = resp.model_dump()
        assert data == {"status": "ok"}


class TestInfoResponse:
    """Tests for InfoResponse model."""

    def test_construction(self) -> None:
        resp = InfoResponse(
            version="0.1.0",
            supported_formats=["wfdb"],
            enabled_task_heads=["rhythm"],
            model_loaded=False,
        )
        assert resp.version == "0.1.0"
        assert resp.supported_formats == ["wfdb"]
        assert resp.enabled_task_heads == ["rhythm"]
        assert resp.model_loaded is False

    def test_default_name(self) -> None:
        resp = InfoResponse(
            version="0.1.0",
            supported_formats=[],
            enabled_task_heads=[],
            model_loaded=False,
        )
        assert resp.name == "aortica"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_returns_fastapi_instance(self) -> None:
        app = create_app()
        assert isinstance(app, fastapi.FastAPI)

    def test_default_cors_origins(self) -> None:
        app = create_app()
        # The CORS middleware is added; verify via middleware stack
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_custom_cors_origins(self) -> None:
        app = create_app(cors_origins=["http://localhost:3000"])
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_default_enabled_tasks(self) -> None:
        app = create_app()
        assert app.state.enabled_tasks == list(DEFAULT_TASK_HEADS)

    def test_custom_enabled_tasks(self) -> None:
        app = create_app(enabled_tasks=["rhythm", "risk"])
        assert app.state.enabled_tasks == ["rhythm", "risk"]

    def test_model_loaded_default_false(self) -> None:
        app = create_app()
        assert app.state.model_loaded is False

    def test_model_loaded_true(self) -> None:
        app = create_app(model_loaded=True)
        assert app.state.model_loaded is True


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_response_matches_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        parsed = HealthResponse(**body)
        assert parsed.status == "ok"

    def test_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------


class TestInfoEndpoint:
    """Tests for the GET /info endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200

    def test_has_version(self, client: TestClient) -> None:
        body = client.get("/info").json()
        assert "version" in body
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0

    def test_has_supported_formats(self, client: TestClient) -> None:
        body = client.get("/info").json()
        assert body["supported_formats"] == SUPPORTED_FORMATS

    def test_has_enabled_task_heads(self, client: TestClient) -> None:
        body = client.get("/info").json()
        assert body["enabled_task_heads"] == list(DEFAULT_TASK_HEADS)

    def test_has_model_loaded(self, client: TestClient) -> None:
        body = client.get("/info").json()
        assert body["model_loaded"] is False

    def test_response_matches_model(self, client: TestClient) -> None:
        body = client.get("/info").json()
        parsed = InfoResponse(**body)
        assert parsed.name == "aortica"
        assert parsed.supported_formats == list(SUPPORTED_FORMATS)

    def test_custom_tasks_reflected(self) -> None:
        app = create_app(enabled_tasks=["rhythm"])
        c = TestClient(app)
        body = c.get("/info").json()
        assert body["enabled_task_heads"] == ["rhythm"]

    def test_model_loaded_true_reflected(self) -> None:
        app = create_app(model_loaded=True)
        c = TestClient(app)
        body = c.get("/info").json()
        assert body["model_loaded"] is True

    def test_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_supported_formats_list(self) -> None:
        assert isinstance(SUPPORTED_FORMATS, list)
        assert len(SUPPORTED_FORMATS) >= 6
        assert "wfdb" in SUPPORTED_FORMATS
        assert "dicom" in SUPPORTED_FORMATS

    def test_default_task_heads(self) -> None:
        assert DEFAULT_TASK_HEADS == ["rhythm", "structural", "ischaemia", "risk"]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API imports work."""

    def test_import_create_app(self) -> None:
        from aortica.api import create_app as _create_app

        assert callable(_create_app)

    def test_import_models(self) -> None:
        from aortica.api import app as api_app

        assert hasattr(api_app, "HealthResponse")
        assert hasattr(api_app, "InfoResponse")
