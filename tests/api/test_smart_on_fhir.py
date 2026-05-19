"""Tests for US-085: SMART on FHIR Launch Context Support."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.smart_on_fhir import (  # noqa: E402
    SMARTConfig,
    SMARTLaunchContext,
    SMARTMetadata,
    SMARTSessionStore,
    SMARTStatusResponse,
    _parse_capability_statement,
    _parse_token_response,
    create_smart_router,
)


# ---------------------------------------------------------------------------
# SMARTConfig
# ---------------------------------------------------------------------------


class TestSMARTConfig:
    def test_default_not_configured(self) -> None:
        cfg = SMARTConfig()
        assert not cfg.is_configured

    def test_configured_with_minimum(self) -> None:
        cfg = SMARTConfig(client_id="app1", redirect_uri="http://localhost/cb")
        assert cfg.is_configured

    def test_from_env(self) -> None:
        env = {
            "SMART_CLIENT_ID": "cid",
            "SMART_CLIENT_SECRET": "sec",
            "SMART_REDIRECT_URI": "http://localhost/cb",
            "SMART_FHIR_SERVER_URL": "http://fhir.example.com",
            "SMART_SCOPES": "launch patient/*.read",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = SMARTConfig.from_env()
        assert cfg.client_id == "cid"
        assert cfg.client_secret == "sec"
        assert cfg.redirect_uri == "http://localhost/cb"
        assert cfg.fhir_server_url == "http://fhir.example.com"
        assert cfg.scopes == "launch patient/*.read"

    def test_from_env_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = SMARTConfig.from_env()
        assert cfg.client_id == ""
        assert not cfg.is_configured


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestSMARTLaunchContext:
    def test_minimal(self) -> None:
        ctx = SMARTLaunchContext(access_token="tok123")
        assert ctx.access_token == "tok123"
        assert ctx.patient is None
        assert ctx.token_type == "Bearer"

    def test_full(self) -> None:
        ctx = SMARTLaunchContext(
            access_token="tok",
            patient="Patient/123",
            encounter="Encounter/456",
            fhir_server="http://fhir.example.com",
            expires_in=3600,
            scope="launch patient/*.read",
        )
        assert ctx.patient == "Patient/123"
        assert ctx.encounter == "Encounter/456"

    def test_roundtrip(self) -> None:
        ctx = SMARTLaunchContext(access_token="t", patient="P/1")
        d = ctx.model_dump()
        ctx2 = SMARTLaunchContext(**d)
        assert ctx2.patient == ctx.patient


class TestSMARTMetadata:
    def test_construction(self) -> None:
        m = SMARTMetadata(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            capabilities=["launch-ehr"],
        )
        assert "launch-ehr" in m.capabilities


class TestSMARTStatusResponse:
    def test_not_configured(self) -> None:
        r = SMARTStatusResponse(configured=False)
        assert not r.configured
        assert not r.active_session

    def test_active(self) -> None:
        r = SMARTStatusResponse(
            configured=True,
            active_session=True,
            patient="Patient/1",
        )
        assert r.patient == "Patient/1"


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


class TestSMARTSessionStore:
    def test_create_and_get_pending(self) -> None:
        store = SMARTSessionStore()
        state = store.create_launch("http://fhir", "http://auth", "http://tok")
        assert store.pending_count == 1
        pending = store.get_pending(state)
        assert pending is not None
        assert pending.fhir_server_url == "http://fhir"
        assert store.pending_count == 0  # consumed

    def test_get_pending_invalid(self) -> None:
        store = SMARTSessionStore()
        assert store.get_pending("nonexistent") is None

    def test_active_context(self) -> None:
        store = SMARTSessionStore()
        assert not store.has_active_session
        ctx = SMARTLaunchContext(access_token="tok", patient="P/1")
        store.set_active_context(ctx)
        assert store.has_active_session
        assert store.get_active_context() is not None
        assert store.get_active_context().patient == "P/1"  # type: ignore[union-attr]

    def test_clear(self) -> None:
        store = SMARTSessionStore()
        store.create_launch("a", "b", "c")
        store.set_active_context(SMARTLaunchContext(access_token="t"))
        store.clear()
        assert store.pending_count == 0
        assert not store.has_active_session


# ---------------------------------------------------------------------------
# Token response parsing
# ---------------------------------------------------------------------------


class TestParseTokenResponse:
    def test_minimal(self) -> None:
        ctx = _parse_token_response({"access_token": "tok"})
        assert ctx.access_token == "tok"

    def test_with_context(self) -> None:
        ctx = _parse_token_response({
            "access_token": "tok",
            "patient": "Patient/123",
            "encounter": "Encounter/456",
            "fhirServer": "http://fhir.example.com",
            "scope": "launch patient/*.read",
            "expires_in": 3600,
        })
        assert ctx.patient == "Patient/123"
        assert ctx.encounter == "Encounter/456"
        assert ctx.fhir_server == "http://fhir.example.com"

    def test_fhir_server_snake_case(self) -> None:
        ctx = _parse_token_response({
            "access_token": "tok",
            "fhir_server": "http://alt.example.com",
        })
        assert ctx.fhir_server == "http://alt.example.com"


# ---------------------------------------------------------------------------
# Capability statement parsing
# ---------------------------------------------------------------------------


class TestParseCapabilityStatement:
    def test_valid(self) -> None:
        data = {
            "rest": [{
                "security": {
                    "extension": [{
                        "url": (
                            "http://fhir-registry.smarthealthit.org/"
                            "StructureDefinition/oauth-uris"
                        ),
                        "extension": [
                            {"url": "authorize", "valueUri": "https://auth/authorize"},
                            {"url": "token", "valueUri": "https://auth/token"},
                        ],
                    }]
                }
            }]
        }
        meta = _parse_capability_statement(data)
        assert meta.authorization_endpoint == "https://auth/authorize"
        assert meta.token_endpoint == "https://auth/token"

    def test_missing_endpoints(self) -> None:
        with pytest.raises(ValueError, match="does not contain"):
            _parse_capability_statement({"rest": []})


# ---------------------------------------------------------------------------
# Router endpoints (via TestClient)
# ---------------------------------------------------------------------------


def _make_app(
    config: SMARTConfig | None = None,
    session_store: SMARTSessionStore | None = None,
) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    cfg = config or SMARTConfig(
        client_id="test-app",
        redirect_uri="http://localhost:5173/smart/callback",
        fhir_server_url="http://fhir.example.com",
    )
    router = create_smart_router(cfg, session_store)
    app.include_router(router)
    return app


class TestSmartLaunchEndpoint:
    def test_not_configured_returns_501(self) -> None:
        app = _make_app(config=SMARTConfig())
        client = TestClient(app)
        resp = client.get("/api/v1/smart/launch", follow_redirects=False)
        assert resp.status_code == 501

    def test_no_fhir_server_returns_400(self) -> None:
        cfg = SMARTConfig(
            client_id="app", redirect_uri="http://cb", fhir_server_url=""
        )
        app = _make_app(config=cfg)
        client = TestClient(app)
        resp = client.get("/api/v1/smart/launch", follow_redirects=False)
        assert resp.status_code == 400

    @patch("aortica.api.smart_on_fhir.discover_smart_metadata")
    def test_successful_launch_redirect(self, mock_discover: Any) -> None:
        mock_discover.return_value = SMARTMetadata(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
        )
        app = _make_app()
        client = TestClient(app)
        resp = client.get(
            "/api/v1/smart/launch",
            params={"launch": "xyz", "iss": "http://fhir.example.com"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert "response_type=code" in location
        assert "client_id=test-app" in location
        assert "launch=xyz" in location

    @patch("aortica.api.smart_on_fhir.discover_smart_metadata")
    def test_launch_uses_config_fhir_server(self, mock_discover: Any) -> None:
        mock_discover.return_value = SMARTMetadata(
            authorization_endpoint="https://auth/a",
            token_endpoint="https://auth/t",
        )
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/smart/launch", follow_redirects=False)
        assert resp.status_code == 302
        mock_discover.assert_called_with("http://fhir.example.com")

    @patch("aortica.api.smart_on_fhir.discover_smart_metadata")
    def test_launch_discovery_failure(self, mock_discover: Any) -> None:
        mock_discover.side_effect = ValueError("discovery failed")
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/smart/launch", follow_redirects=False)
        assert resp.status_code == 502


class TestSmartCallbackEndpoint:
    def test_invalid_state_returns_400(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get(
            "/api/v1/smart/callback",
            params={"code": "abc", "state": "bad"},
        )
        assert resp.status_code == 400

    @patch("aortica.api.smart_on_fhir.exchange_code_for_token")
    def test_successful_callback(self, mock_exchange: Any) -> None:
        store = SMARTSessionStore()
        state = store.create_launch(
            "http://fhir.example.com",
            "https://auth/authorize",
            "https://auth/token",
        )
        mock_exchange.return_value = SMARTLaunchContext(
            access_token="tok",
            patient="Patient/123",
            encounter="Encounter/456",
            fhir_server="http://fhir.example.com",
        )
        app = _make_app(session_store=store)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/smart/callback",
            params={"code": "authcode", "state": state},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["patient"] == "Patient/123"
        assert data["access_token"] == "tok"
        assert store.has_active_session

    @patch("aortica.api.smart_on_fhir.exchange_code_for_token")
    def test_callback_populates_fhir_server(self, mock_exchange: Any) -> None:
        store = SMARTSessionStore()
        state = store.create_launch(
            "http://fhir.example.com", "https://auth/a", "https://auth/t"
        )
        mock_exchange.return_value = SMARTLaunchContext(
            access_token="tok", fhir_server=None
        )
        app = _make_app(session_store=store)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/smart/callback",
            params={"code": "c", "state": state},
        )
        assert resp.status_code == 200
        assert resp.json()["fhir_server"] == "http://fhir.example.com"

    @patch("aortica.api.smart_on_fhir.exchange_code_for_token")
    def test_callback_exchange_failure(self, mock_exchange: Any) -> None:
        store = SMARTSessionStore()
        state = store.create_launch("a", "b", "c")
        mock_exchange.side_effect = ValueError("exchange failed")
        app = _make_app(session_store=store)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/smart/callback",
            params={"code": "c", "state": state},
        )
        assert resp.status_code == 502


class TestSmartStatusEndpoint:
    def test_no_session(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/smart/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is True
        assert data["active_session"] is False
        assert data["patient"] is None

    def test_with_active_session(self) -> None:
        store = SMARTSessionStore()
        store.set_active_context(
            SMARTLaunchContext(
                access_token="t",
                patient="Patient/99",
                encounter="Encounter/55",
                fhir_server="http://fhir.local",
            )
        )
        app = _make_app(session_store=store)
        client = TestClient(app)
        resp = client.get("/api/v1/smart/status")
        data = resp.json()
        assert data["active_session"] is True
        assert data["patient"] == "Patient/99"
        assert data["encounter"] == "Encounter/55"


class TestSmartContextEndpoint:
    def test_no_context_returns_404(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/smart/context")
        assert resp.status_code == 404

    def test_with_context(self) -> None:
        store = SMARTSessionStore()
        store.set_active_context(
            SMARTLaunchContext(access_token="t", patient="P/1")
        )
        app = _make_app(session_store=store)
        client = TestClient(app)
        resp = client.get("/api/v1/smart/context")
        assert resp.status_code == 200
        assert resp.json()["patient"] == "P/1"


# ---------------------------------------------------------------------------
# Integration with main app
# ---------------------------------------------------------------------------


class TestAppIntegration:
    def test_smart_routes_registered(self) -> None:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        paths = [r.path for r in app.routes]
        assert "/api/v1/smart/launch" in paths
        assert "/api/v1/smart/callback" in paths
        assert "/api/v1/smart/status" in paths
        assert "/api/v1/smart/context" in paths


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_module_importable(self) -> None:
        import aortica.api.smart_on_fhir as mod

        assert hasattr(mod, "SMARTConfig")
        assert hasattr(mod, "SMARTLaunchContext")
        assert hasattr(mod, "SMARTSessionStore")
        assert hasattr(mod, "create_smart_router")
        assert hasattr(mod, "discover_smart_metadata")
        assert hasattr(mod, "exchange_code_for_token")
