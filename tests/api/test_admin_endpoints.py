"""Tests for aortica.api.admin_endpoints — Admin dashboard API."""

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.admin_endpoints import (  # noqa: E402
    VALID_ROLES,
    VALID_STATUSES,
    APIKeyListResponse,
    APIKeyRecord,
    APIKeyRevokeResponse,
    ActivityLog,
    ActivityLogEntry,
    ActivityLogResponse,
    SystemHealthResponse,
    UserListResponse,
    UserRecord,
    UserStore,
    UserStoreRecord,
    UserUpdateRequest,
    UserUpdateResponse,
    create_admin_router,
)
from aortica.api.app import create_app  # noqa: E402
from aortica.api.auth import APIKeyStore, UserInfo, create_access_token  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin_token() -> str:
    """Create a JWT for an admin user."""
    user = UserInfo(sub="admin-user", email="admin@aortica.io", name="Admin", provider="local")
    return create_access_token(user, expires_in=3600)


def _clinician_token() -> str:
    """Create a JWT for a non-admin user."""
    user = UserInfo(sub="clin-user", email="clin@aortica.io", name="Clinician", provider="local")
    return create_access_token(user, expires_in=3600)


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_admin_token()}"}


def _clinician_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_clinician_token()}"}


def _make_app_with_admin(
    *,
    enable_auth: bool = True,
    seed_users: bool = True,
) -> tuple[fastapi.FastAPI, UserStore, ActivityLog]:
    """Create an app and return it with its user_store and activity_log."""
    app = create_app(enable_auth=enable_auth)
    user_store: UserStore = app.state.user_store  # type: ignore[attr-defined]
    activity_log: ActivityLog = app.state.activity_log  # type: ignore[attr-defined]

    if seed_users:
        # Seed the admin user
        user_store.add(UserStoreRecord(
            id="admin-user",
            email="admin@aortica.io",
            name="Admin",
            role="admin",
            last_login=time.time(),
            status="active",
            provider="local",
        ))
        # Seed a clinician
        user_store.add(UserStoreRecord(
            id="clin-user",
            email="clin@aortica.io",
            name="Clinician",
            role="clinician",
            last_login=time.time(),
            status="active",
            provider="local",
        ))
        # Seed a researcher
        user_store.add(UserStoreRecord(
            id="res-user",
            email="res@aortica.io",
            name="Researcher",
            role="researcher",
            last_login=time.time() - 86400,
            status="active",
            provider="github",
        ))

    return app, user_store, activity_log


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Tests for admin Pydantic response/request models."""

    def test_user_record_construction(self) -> None:
        r = UserRecord(id="u1", role="admin", status="active", provider="local")
        assert r.id == "u1"
        assert r.role == "admin"
        assert r.email is None

    def test_user_list_response(self) -> None:
        resp = UserListResponse(users=[], total=0)
        assert resp.total == 0

    def test_user_update_request(self) -> None:
        req = UserUpdateRequest(role="researcher")
        assert req.role == "researcher"
        assert req.status is None

    def test_user_update_response(self) -> None:
        resp = UserUpdateResponse(id="u1", role="admin", status="active")
        assert resp.updated is True

    def test_api_key_record(self) -> None:
        r = APIKeyRecord(key_id="abc123", name="test", user_sub="u1", created_at=1000.0)
        assert r.last_used is None

    def test_api_key_list_response(self) -> None:
        resp = APIKeyListResponse(keys=[], total=0)
        assert resp.total == 0

    def test_api_key_revoke_response(self) -> None:
        resp = APIKeyRevokeResponse(key_id="abc", revoked=True)
        assert resp.revoked is True

    def test_system_health_response(self) -> None:
        resp = SystemHealthResponse()
        assert resp.status == "ok"
        assert resp.model_loaded is False

    def test_activity_log_entry(self) -> None:
        entry = ActivityLogEntry(
            timestamp=1000.0, endpoint="/health", method="GET", status_code=200
        )
        assert entry.user is None
        assert entry.duration_ms is None

    def test_activity_log_response(self) -> None:
        resp = ActivityLogResponse(entries=[], total=0, page=1, page_size=50)
        assert resp.page == 1


# ---------------------------------------------------------------------------
# UserStore tests
# ---------------------------------------------------------------------------


class TestUserStore:
    """Tests for UserStore in-memory store."""

    def test_add_and_get(self) -> None:
        store = UserStore()
        store.add(UserStoreRecord(id="u1", email="u1@test.com", role="admin"))
        user = store.get("u1")
        assert user is not None
        assert user.email == "u1@test.com"

    def test_get_missing(self) -> None:
        store = UserStore()
        assert store.get("missing") is None

    def test_list_all(self) -> None:
        store = UserStore()
        store.add(UserStoreRecord(id="u1"))
        store.add(UserStoreRecord(id="u2"))
        assert len(store.list_all()) == 2

    def test_update_role(self) -> None:
        store = UserStore()
        store.add(UserStoreRecord(id="u1", role="clinician"))
        updated = store.update("u1", role="admin")
        assert updated is not None
        assert updated.role == "admin"

    def test_update_status(self) -> None:
        store = UserStore()
        store.add(UserStoreRecord(id="u1", status="active"))
        updated = store.update("u1", status="disabled")
        assert updated is not None
        assert updated.status == "disabled"

    def test_update_missing_user(self) -> None:
        store = UserStore()
        assert store.update("missing", role="admin") is None

    def test_upsert_from_auth_new(self) -> None:
        store = UserStore()
        rec = store.upsert_from_auth("u1", email="u1@test.com", provider="google")
        assert rec.id == "u1"
        assert rec.role == "clinician"  # default
        assert rec.last_login is not None
        assert store.count == 1

    def test_upsert_from_auth_existing(self) -> None:
        store = UserStore()
        store.add(UserStoreRecord(id="u1", role="admin", last_login=1000.0))
        rec = store.upsert_from_auth("u1", email="new@test.com")
        assert rec.role == "admin"  # unchanged
        assert rec.email == "new@test.com"
        assert rec.last_login is not None and rec.last_login > 1000.0

    def test_count_property(self) -> None:
        store = UserStore()
        assert store.count == 0
        store.add(UserStoreRecord(id="u1"))
        assert store.count == 1


# ---------------------------------------------------------------------------
# ActivityLog tests
# ---------------------------------------------------------------------------


class TestActivityLog:
    """Tests for ActivityLog in-memory log."""

    def test_record_and_list(self) -> None:
        log = ActivityLog()
        log.record(endpoint="/health", method="GET", status_code=200)
        entries, total = log.list_entries()
        assert total == 1
        assert entries[0].endpoint == "/health"

    def test_list_newest_first(self) -> None:
        log = ActivityLog()
        log.record(endpoint="/a", method="GET", status_code=200)
        log.record(endpoint="/b", method="POST", status_code=201)
        entries, _ = log.list_entries()
        assert entries[0].endpoint == "/b"  # newest first

    def test_pagination(self) -> None:
        log = ActivityLog()
        for i in range(10):
            log.record(endpoint=f"/ep{i}", method="GET", status_code=200)
        entries, total = log.list_entries(page=1, page_size=3)
        assert total == 10
        assert len(entries) == 3
        entries_p2, _ = log.list_entries(page=2, page_size=3)
        assert len(entries_p2) == 3

    def test_max_entries_eviction(self) -> None:
        log = ActivityLog(max_entries=5)
        for i in range(10):
            log.record(endpoint=f"/ep{i}", method="GET", status_code=200)
        assert log.count == 5

    def test_count_property(self) -> None:
        log = ActivityLog()
        assert log.count == 0
        log.record(endpoint="/x", method="GET", status_code=200)
        assert log.count == 1

    def test_user_and_duration(self) -> None:
        log = ActivityLog()
        log.record(
            endpoint="/predict",
            method="POST",
            status_code=200,
            user="admin",
            duration_ms=42.5,
        )
        entries, _ = log.list_entries()
        assert entries[0].user == "admin"
        assert entries[0].duration_ms == 42.5


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_valid_roles(self) -> None:
        assert "admin" in VALID_ROLES
        assert "clinician" in VALID_ROLES
        assert "researcher" in VALID_ROLES

    def test_valid_statuses(self) -> None:
        assert "active" in VALID_STATUSES
        assert "disabled" in VALID_STATUSES


# ---------------------------------------------------------------------------
# Router factory tests
# ---------------------------------------------------------------------------


class TestCreateAdminRouter:
    """Tests for create_admin_router factory."""

    def test_returns_tuple(self) -> None:
        result = create_admin_router()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_router_has_routes(self) -> None:
        router, _, _ = create_admin_router()
        paths = [r.path for r in router.routes]  # type: ignore[union-attr]
        assert "/api/v1/admin/users" in paths
        assert "/api/v1/admin/system-health" in paths
        assert "/api/v1/admin/activity-log" in paths


# ---------------------------------------------------------------------------
# Endpoint integration tests — RBAC
# ---------------------------------------------------------------------------


class TestAdminRBAC:
    """Tests for role-based access control on admin endpoints."""

    def test_admin_user_can_access(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/users", headers=_admin_headers())
        assert resp.status_code == 200

    def test_non_admin_gets_403(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/users", headers=_clinician_headers())
        assert resp.status_code == 403

    def test_unauthenticated_gets_401(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/users")
        assert resp.status_code == 401

    def test_all_admin_endpoints_require_auth(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        endpoints = [
            ("GET", "/api/v1/admin/users"),
            ("GET", "/api/v1/admin/system-health"),
            ("GET", "/api/v1/admin/activity-log"),
            ("GET", "/api/v1/admin/api-keys"),
        ]
        for method, path in endpoints:
            resp = client.request(method, path)
            assert resp.status_code == 401, f"{method} {path} should require auth"

    def test_all_admin_endpoints_require_admin_role(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        endpoints = [
            ("GET", "/api/v1/admin/users"),
            ("GET", "/api/v1/admin/system-health"),
            ("GET", "/api/v1/admin/activity-log"),
            ("GET", "/api/v1/admin/api-keys"),
        ]
        for method, path in endpoints:
            resp = client.request(method, path, headers=_clinician_headers())
            assert resp.status_code == 403, f"{method} {path} should require admin"

    def test_auth_disabled_allows_access(self) -> None:
        app, _, _ = _make_app_with_admin(enable_auth=False)
        client = TestClient(app)
        resp = client.get("/api/v1/admin/users")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v1/admin/users
# ---------------------------------------------------------------------------


class TestListUsers:
    """Tests for GET /api/v1/admin/users."""

    def test_returns_seeded_users(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/users", headers=_admin_headers())
        body = resp.json()
        assert resp.status_code == 200
        assert body["total"] == 3
        ids = [u["id"] for u in body["users"]]
        assert "admin-user" in ids
        assert "clin-user" in ids
        assert "res-user" in ids

    def test_user_fields_present(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/users", headers=_admin_headers()).json()
        user = next(u for u in body["users"] if u["id"] == "admin-user")
        assert user["email"] == "admin@aortica.io"
        assert user["role"] == "admin"
        assert user["status"] == "active"
        assert user["provider"] == "local"

    def test_empty_store(self) -> None:
        app, _, _ = _make_app_with_admin(seed_users=False)
        client = TestClient(app)
        # Auth disabled to skip RBAC for this test
        app.state.enable_auth = False  # type: ignore[attr-defined]
        body = client.get("/api/v1/admin/users").json()
        assert body["total"] == 0
        assert body["users"] == []

    def test_response_matches_model(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/users", headers=_admin_headers()).json()
        parsed = UserListResponse(**body)
        assert parsed.total == 3


# ---------------------------------------------------------------------------
# PATCH /api/v1/admin/users/:id
# ---------------------------------------------------------------------------


class TestUpdateUser:
    """Tests for PATCH /api/v1/admin/users/:id."""

    def test_update_role(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"role": "researcher"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "researcher"
        assert body["updated"] is True

    def test_update_status_disable(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"status": "disabled"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    def test_update_status_enable(self) -> None:
        app, user_store, _ = _make_app_with_admin()
        user_store.update("clin-user", status="disabled")
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"status": "active"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_invalid_role_returns_422(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"role": "superadmin"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 422

    def test_invalid_status_returns_422(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"status": "banned"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 422

    def test_missing_user_returns_404(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/nonexistent",
            json={"role": "admin"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 404

    def test_non_admin_cannot_update(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/admin/users/clin-user",
            json={"role": "admin"},
            headers=_clinician_headers(),
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# API Key management
# ---------------------------------------------------------------------------


class TestAPIKeyManagement:
    """Tests for API key admin endpoints."""

    def test_list_keys_empty(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/api-keys", headers=_admin_headers())
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_keys_with_data(self) -> None:
        app, _, _ = _make_app_with_admin()
        # Generate some API keys
        key_store: APIKeyStore = app.state.api_key_store  # type: ignore[attr-defined]
        key_store.generate(name="test-key-1", user_sub="admin-user")
        key_store.generate(name="test-key-2", user_sub="clin-user")
        client = TestClient(app)
        resp = client.get("/api/v1/admin/api-keys", headers=_admin_headers())
        body = resp.json()
        assert body["total"] == 2
        names = [k["name"] for k in body["keys"]]
        assert "test-key-1" in names
        assert "test-key-2" in names

    def test_revoke_key(self) -> None:
        app, _, _ = _make_app_with_admin()
        key_store: APIKeyStore = app.state.api_key_store  # type: ignore[attr-defined]
        _, record = key_store.generate(name="revoke-me", user_sub="admin-user")
        client = TestClient(app)
        resp = client.delete(
            f"/api/v1/admin/api-keys/{record.key_hash}",
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True
        assert key_store.count == 0

    def test_revoke_nonexistent_key(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.delete(
            "/api/v1/admin/api-keys/nonexistent-hash",
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is False

    def test_non_admin_cannot_list_keys(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/api-keys", headers=_clinician_headers())
        assert resp.status_code == 403

    def test_non_admin_cannot_revoke_key(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.delete(
            "/api/v1/admin/api-keys/some-hash",
            headers=_clinician_headers(),
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/v1/admin/system-health
# ---------------------------------------------------------------------------


class TestSystemHealth:
    """Tests for GET /api/v1/admin/system-health."""

    def test_returns_health_metrics(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/system-health", headers=_admin_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "uptime_seconds" in body
        assert "database_size_bytes" in body
        assert "total_ecgs_processed" in body

    def test_response_matches_model(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/system-health", headers=_admin_headers()).json()
        parsed = SystemHealthResponse(**body)
        assert parsed.status == "ok"

    def test_uptime_positive(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/system-health", headers=_admin_headers()).json()
        assert body["uptime_seconds"] >= 0

    def test_model_loaded_false_by_default(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/system-health", headers=_admin_headers()).json()
        assert body["model_loaded"] is False

    def test_onnx_runtime_field_present(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/system-health", headers=_admin_headers()).json()
        assert "onnx_runtime_available" in body

    def test_sync_engine_status_default(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        body = client.get("/api/v1/admin/system-health", headers=_admin_headers()).json()
        assert body["sync_engine_status"] == "inactive"


# ---------------------------------------------------------------------------
# GET /api/v1/admin/activity-log
# ---------------------------------------------------------------------------


class TestActivityLogEndpoint:
    """Tests for GET /api/v1/admin/activity-log."""

    def test_empty_log(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get("/api/v1/admin/activity-log", headers=_admin_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["entries"] == []

    def test_log_with_entries(self) -> None:
        app, _, activity_log = _make_app_with_admin()
        activity_log.record(
            endpoint="/api/v1/predict",
            method="POST",
            status_code=200,
            user="admin-user",
            duration_ms=125.5,
        )
        activity_log.record(
            endpoint="/health",
            method="GET",
            status_code=200,
        )
        client = TestClient(app)
        resp = client.get("/api/v1/admin/activity-log", headers=_admin_headers())
        body = resp.json()
        assert body["total"] == 2
        assert len(body["entries"]) == 2
        # Newest first
        assert body["entries"][0]["endpoint"] == "/health"

    def test_pagination_params(self) -> None:
        app, _, activity_log = _make_app_with_admin()
        for i in range(10):
            activity_log.record(
                endpoint=f"/ep{i}",
                method="GET",
                status_code=200,
            )
        client = TestClient(app)
        resp = client.get(
            "/api/v1/admin/activity-log?page=2&page_size=3",
            headers=_admin_headers(),
        )
        body = resp.json()
        assert body["page"] == 2
        assert body["page_size"] == 3
        assert body["total"] == 10
        assert len(body["entries"]) == 3

    def test_page_size_capped(self) -> None:
        app, _, _ = _make_app_with_admin()
        client = TestClient(app)
        resp = client.get(
            "/api/v1/admin/activity-log?page_size=999",
            headers=_admin_headers(),
        )
        body = resp.json()
        assert body["page_size"] == 100  # capped at 100

    def test_response_matches_model(self) -> None:
        app, _, activity_log = _make_app_with_admin()
        activity_log.record(
            endpoint="/predict", method="POST", status_code=200
        )
        client = TestClient(app)
        body = client.get("/api/v1/admin/activity-log", headers=_admin_headers()).json()
        parsed = ActivityLogResponse(**body)
        assert parsed.total == 1


# ---------------------------------------------------------------------------
# Integration: app.py registration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    """Tests verifying admin router is properly registered in the app."""

    def test_admin_routes_registered(self) -> None:
        app = create_app()
        paths = [r.path for r in app.routes]  # type: ignore[union-attr]
        assert "/api/v1/admin/users" in paths
        assert "/api/v1/admin/system-health" in paths
        assert "/api/v1/admin/activity-log" in paths

    def test_user_store_on_state(self) -> None:
        app = create_app()
        assert hasattr(app.state, "user_store")
        assert isinstance(app.state.user_store, UserStore)

    def test_activity_log_on_state(self) -> None:
        app = create_app()
        assert hasattr(app.state, "activity_log")
        assert isinstance(app.state.activity_log, ActivityLog)

    def test_existing_endpoints_unaffected(self) -> None:
        """Regression: admin router must not break existing endpoints."""
        app = create_app()
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/info").status_code == 200
