"""Tests for US-048: User Authentication System.

Covers:
  - Pydantic models (TokenResponse, APIKeyResponse, UserInfo)
  - API key store (generate, validate, list, revoke, hash security)
  - JWT helpers (create, decode, expiry, refresh tokens)
  - Auth dependency (require_auth via Bearer JWT and X-API-Key)
  - Auth router endpoints (/token, /refresh, OAuth stubs)
  - App factory auth integration (enable_auth flag, protected endpoints)
  - Imports
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
jwt_mod = pytest.importorskip("jwt")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.auth import (  # noqa: E402
    API_KEY_LENGTH,
    API_KEY_PREFIX,
    JWT_ALGORITHM,
    JWT_EXPIRY_SECONDS,
    JWT_REFRESH_EXPIRY_SECONDS,
    APIKeyResponse,
    APIKeyStore,
    StoredAPIKey,
    TokenResponse,
    UserInfo,
    create_access_token,
    create_refresh_token,
    decode_token,
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestTokenResponse:
    def test_construction(self) -> None:
        resp = TokenResponse(access_token="abc", expires_in=3600)
        assert resp.access_token == "abc"
        assert resp.token_type == "bearer"
        assert resp.expires_in == 3600
        assert resp.refresh_token is None

    def test_with_refresh(self) -> None:
        resp = TokenResponse(
            access_token="abc", expires_in=3600, refresh_token="xyz"
        )
        assert resp.refresh_token == "xyz"

    def test_roundtrip(self) -> None:
        resp = TokenResponse(access_token="t", expires_in=60)
        d = resp.model_dump()
        resp2 = TokenResponse(**d)
        assert resp2.access_token == resp.access_token


class TestAPIKeyResponse:
    def test_construction(self) -> None:
        resp = APIKeyResponse(api_key="ak_test", name="dev", created_at=1.0)
        assert resp.api_key == "ak_test"
        assert resp.name == "dev"

    def test_roundtrip(self) -> None:
        resp = APIKeyResponse(api_key="k", name="n", created_at=0.0)
        d = resp.model_dump()
        resp2 = APIKeyResponse(**d)
        assert resp2.api_key == resp.api_key


class TestUserInfo:
    def test_construction(self) -> None:
        u = UserInfo(sub="local:user@test.com", email="user@test.com")
        assert u.sub == "local:user@test.com"
        assert u.provider == "local"

    def test_provider(self) -> None:
        u = UserInfo(sub="google:123", provider="google")
        assert u.provider == "google"


# ---------------------------------------------------------------------------
# API key store
# ---------------------------------------------------------------------------


class TestAPIKeyStore:
    def test_generate_returns_key_and_record(self) -> None:
        store = APIKeyStore()
        raw_key, record = store.generate("test", "user:1")
        assert raw_key.startswith(API_KEY_PREFIX)
        assert record.name == "test"
        assert record.user_sub == "user:1"
        assert store.count == 1

    def test_validate_valid_key(self) -> None:
        store = APIKeyStore()
        raw_key, _record = store.generate("k", "u")
        result = store.validate(raw_key)
        assert result is not None
        assert result.user_sub == "u"

    def test_validate_invalid_key(self) -> None:
        store = APIKeyStore()
        assert store.validate("ak_invalid") is None

    def test_validate_wrong_key(self) -> None:
        store = APIKeyStore()
        store.generate("k", "u")
        assert store.validate("ak_wrong") is None

    def test_key_not_stored_in_plaintext(self) -> None:
        store = APIKeyStore()
        raw_key, _record = store.generate("k", "u")
        # Raw key should not appear anywhere in the store's internal dict
        for h in store._keys:
            assert raw_key not in h

    def test_list_keys(self) -> None:
        store = APIKeyStore()
        store.generate("a", "user1")
        store.generate("b", "user1")
        store.generate("c", "user2")
        assert len(store.list_keys("user1")) == 2
        assert len(store.list_keys("user2")) == 1

    def test_revoke(self) -> None:
        store = APIKeyStore()
        raw_key, record = store.generate("k", "u")
        assert store.count == 1
        revoked = store.revoke(record.key_hash)
        assert revoked is True
        assert store.count == 0
        assert store.validate(raw_key) is None

    def test_revoke_nonexistent(self) -> None:
        store = APIKeyStore()
        assert store.revoke("nonexistent") is False

    def test_multiple_keys_independent(self) -> None:
        store = APIKeyStore()
        k1, _ = store.generate("a", "u")
        k2, _ = store.generate("b", "u")
        assert store.validate(k1) is not None
        assert store.validate(k2) is not None
        assert k1 != k2


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

TEST_SECRET = "test-secret-key-for-auth-tests"


class TestJWT:
    def test_create_access_token(self) -> None:
        user = UserInfo(sub="u1", email="a@b.com", provider="local")
        token = create_access_token(user, secret=TEST_SECRET)
        assert isinstance(token, str)
        assert len(token) > 10

    def test_decode_valid_token(self) -> None:
        user = UserInfo(sub="u1", email="a@b.com", name="Test", provider="local")
        token = create_access_token(user, secret=TEST_SECRET)
        payload = decode_token(token, secret=TEST_SECRET)
        assert payload["sub"] == "u1"
        assert payload["email"] == "a@b.com"
        assert payload["name"] == "Test"
        assert payload["provider"] == "local"

    def test_decode_expired_token(self) -> None:
        user = UserInfo(sub="u1")
        token = create_access_token(user, secret=TEST_SECRET, expires_in=-1)
        with pytest.raises(Exception):
            decode_token(token, secret=TEST_SECRET)

    def test_decode_bad_secret(self) -> None:
        user = UserInfo(sub="u1")
        token = create_access_token(user, secret=TEST_SECRET)
        with pytest.raises(Exception):
            decode_token(token, secret="wrong-secret")

    def test_decode_garbage_token(self) -> None:
        with pytest.raises(Exception):
            decode_token("not.a.token", secret=TEST_SECRET)

    def test_refresh_token(self) -> None:
        user = UserInfo(sub="u1")
        token = create_refresh_token(user, secret=TEST_SECRET)
        payload = decode_token(token, secret=TEST_SECRET)
        assert payload["type"] == "refresh"
        assert payload["sub"] == "u1"

    def test_access_token_no_type_field(self) -> None:
        user = UserInfo(sub="u1")
        token = create_access_token(user, secret=TEST_SECRET)
        payload = decode_token(token, secret=TEST_SECRET)
        assert "type" not in payload

    def test_token_expiry_times(self) -> None:
        user = UserInfo(sub="u1")
        access = create_access_token(user, secret=TEST_SECRET, expires_in=100)
        refresh = create_refresh_token(user, secret=TEST_SECRET, expires_in=1000)
        p_access = decode_token(access, secret=TEST_SECRET)
        p_refresh = decode_token(refresh, secret=TEST_SECRET)
        # Refresh should expire later
        assert p_refresh["exp"] > p_access["exp"]


# ---------------------------------------------------------------------------
# Auth dependency (via test client)
# ---------------------------------------------------------------------------


class TestRequireAuth:
    """Test the require_auth dependency through FastAPI endpoint calls."""

    @pytest.fixture()
    def app_with_auth(self) -> Any:
        from aortica.api.app import create_app

        return create_app(enable_auth=True)

    @pytest.fixture()
    def client(self, app_with_auth: Any) -> TestClient:
        return TestClient(app_with_auth)

    def test_predict_401_without_auth(self, client: TestClient) -> None:
        """Protected endpoints return 401 when no auth is provided."""
        resp = client.post("/api/v1/predict", files={"file": ("test.csv", b"data")})
        assert resp.status_code == 401

    def test_predict_401_bad_token(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.csv", b"data")},
            headers={"Authorization": "Bearer bad-token"},
        )
        assert resp.status_code == 401

    def test_predict_with_valid_jwt(self, client: TestClient, app_with_auth: Any) -> None:
        """Protected endpoint accessible with valid JWT."""
        user = UserInfo(sub="test_user", email="t@t.com")
        token = create_access_token(user)
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.csv", b"data")},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should get past auth (may fail on file parsing, that's fine —
        # we're testing auth, not inference)
        assert resp.status_code != 401

    def test_predict_with_api_key(self, client: TestClient, app_with_auth: Any) -> None:
        """Protected endpoint accessible with valid API key."""
        store: APIKeyStore = app_with_auth.state.api_key_store
        raw_key, _ = store.generate("test", "user:1")
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.csv", b"data")},
            headers={"X-API-Key": raw_key},
        )
        assert resp.status_code != 401

    def test_predict_bad_api_key(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.csv", b"data")},
            headers={"X-API-Key": "ak_invalid"},
        )
        assert resp.status_code == 401

    def test_batch_401_without_auth(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            files=[("files", ("a.csv", b"a"))],
        )
        assert resp.status_code == 401

    def test_health_no_auth_needed(self, client: TestClient) -> None:
        """Health endpoint should not require auth."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_info_no_auth_needed(self, client: TestClient) -> None:
        """Info endpoint should not require auth."""
        resp = client.get("/info")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth disabled mode
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    @pytest.fixture()
    def client(self) -> TestClient:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        return TestClient(app)

    def test_predict_without_auth_when_disabled(self, client: TestClient) -> None:
        """When auth is disabled, predict should not require credentials."""
        resp = client.post("/api/v1/predict", files={"file": ("test.csv", b"data")})
        # Should not be 401 — may be 422 (bad file) but not auth failure
        assert resp.status_code != 401

    def test_batch_without_auth_when_disabled(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/predict/batch",
            files=[("files", ("a.csv", b"a"))],
        )
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Auth router endpoints
# ---------------------------------------------------------------------------


class TestAuthRouter:
    @pytest.fixture()
    def app_with_auth(self) -> Any:
        from aortica.api.app import create_app

        return create_app(enable_auth=True)

    @pytest.fixture()
    def client(self, app_with_auth: Any) -> TestClient:
        return TestClient(app_with_auth)

    def test_token_endpoint_requires_auth(self, client: TestClient) -> None:
        """POST /api/v1/auth/token requires a valid JWT to generate API key."""
        resp = client.post("/api/v1/auth/token")
        assert resp.status_code in (401, 422)

    def test_token_endpoint_with_jwt(self, client: TestClient) -> None:
        """Authenticated user can generate an API key."""
        user = UserInfo(sub="test_user")
        token = create_access_token(user)
        resp = client.post(
            "/api/v1/auth/token",
            params={"name": "my-key"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "api_key" in data
        assert data["api_key"].startswith(API_KEY_PREFIX)
        assert data["name"] == "my-key"

    def test_refresh_endpoint_with_valid_refresh_token(self, client: TestClient) -> None:
        """Refresh endpoint returns new access token."""
        user = UserInfo(sub="test_user")
        refresh = create_refresh_token(user)
        resp = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {refresh}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["expires_in"] == JWT_EXPIRY_SECONDS

    def test_refresh_endpoint_with_access_token_fails(self, client: TestClient) -> None:
        """Refresh endpoint rejects regular access tokens."""
        user = UserInfo(sub="test_user")
        access = create_access_token(user)
        resp = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {access}"},
        )
        assert resp.status_code == 400

    def test_refresh_endpoint_no_token(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/refresh")
        assert resp.status_code == 401

    def test_google_login_not_configured(self, client: TestClient) -> None:
        """Google login returns 501 when not configured."""
        resp = client.get("/api/v1/auth/login/google", follow_redirects=False)
        assert resp.status_code == 501

    def test_github_login_not_configured(self, client: TestClient) -> None:
        """GitHub login returns 501 when not configured."""
        resp = client.get("/api/v1/auth/login/github", follow_redirects=False)
        assert resp.status_code == 501

    def test_generated_api_key_works(self, client: TestClient, app_with_auth: Any) -> None:
        """API key generated via /token endpoint can authenticate requests."""
        user = UserInfo(sub="test_user")
        token = create_access_token(user)
        # Generate key
        resp = client.post(
            "/api/v1/auth/token",
            params={"name": "test-key"},
            headers={"Authorization": f"Bearer {token}"},
        )
        api_key = resp.json()["api_key"]
        # Use key to access protected endpoint
        resp2 = client.post(
            "/api/v1/predict",
            files={"file": ("test.csv", b"data")},
            headers={"X-API-Key": api_key},
        )
        assert resp2.status_code != 401


# ---------------------------------------------------------------------------
# App factory integration
# ---------------------------------------------------------------------------


class TestAppFactoryAuth:
    def test_enable_auth_default_true(self) -> None:
        from aortica.api.app import create_app

        app = create_app()
        assert app.state.enable_auth is True

    def test_enable_auth_false(self) -> None:
        from aortica.api.app import create_app

        app = create_app(enable_auth=False)
        assert app.state.enable_auth is False

    def test_api_key_store_created(self) -> None:
        from aortica.api.app import create_app

        app = create_app()
        assert hasattr(app.state, "api_key_store")
        assert isinstance(app.state.api_key_store, APIKeyStore)

    def test_auth_routes_registered(self) -> None:
        from aortica.api.app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/api/v1/auth/token" in paths
        assert "/api/v1/auth/refresh" in paths
        assert "/api/v1/auth/login/google" in paths
        assert "/api/v1/auth/login/github" in paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_jwt_algorithm(self) -> None:
        assert JWT_ALGORITHM == "HS256"

    def test_jwt_expiry(self) -> None:
        assert JWT_EXPIRY_SECONDS == 86400

    def test_refresh_expiry(self) -> None:
        assert JWT_REFRESH_EXPIRY_SECONDS == 604800

    def test_api_key_prefix(self) -> None:
        assert API_KEY_PREFIX == "ak_"

    def test_api_key_length(self) -> None:
        assert API_KEY_LENGTH == 40


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_auth_module_importable(self) -> None:
        import aortica.api.auth

        assert hasattr(aortica.api.auth, "require_auth")
        assert hasattr(aortica.api.auth, "create_auth_router")
        assert hasattr(aortica.api.auth, "APIKeyStore")
        assert hasattr(aortica.api.auth, "TokenResponse")
        assert hasattr(aortica.api.auth, "UserInfo")

    def test_create_oauth_importable(self) -> None:
        from aortica.api.auth import create_oauth

        assert callable(create_oauth)
