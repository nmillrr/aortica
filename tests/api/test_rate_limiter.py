"""Tests for aortica.api.rate_limiter — Rate limiting middleware."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from aortica.api.app import create_app  # noqa: E402
from aortica.api.rate_limiter import (  # noqa: E402
    EndpointTier,
    InMemoryBackend,
    RateLimitConfig,
    RateLimiterMiddleware,
    TierConfig,
    TokenBucket,
    _extract_client_key,
    add_rate_limiting,
    classify_endpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> RateLimitConfig:
    """Rate limit config with very low limits for fast testing."""
    return RateLimitConfig(
        enabled=True,
        global_requests_per_minute=200,
        default_requests_per_minute=60,
        tiers={
            "predict": TierConfig(requests_per_minute=5, burst_size=2),
            "report": TierConfig(requests_per_minute=10, burst_size=3),
            "admin": TierConfig(requests_per_minute=20, burst_size=5),
            "system": TierConfig(requests_per_minute=50, burst_size=10),
        },
    )


@pytest.fixture()
def app(config: RateLimitConfig) -> fastapi.FastAPI:
    """FastAPI app with rate limiting enabled and auth disabled."""
    return create_app(rate_limit_config=config, enable_auth=False)


@pytest.fixture()
def client(app: fastapi.FastAPI) -> TestClient:
    """Test client for rate-limited app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    """Tests for the TokenBucket data structure."""

    def test_initial_tokens_equal_capacity(self) -> None:
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.remaining == 10

    def test_consume_reduces_tokens(self) -> None:
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.consume() is True
        assert bucket.remaining == 9

    def test_consume_multiple(self) -> None:
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.remaining == 5

    def test_consume_rejects_when_empty(self) -> None:
        bucket = TokenBucket(capacity=2.0, refill_rate=0.001)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(capacity=10.0, refill_rate=100.0)
        bucket.consume(10)
        assert bucket.remaining == 0
        # With refill_rate=100/s, after ~0.1s we'd have ~10 tokens
        time.sleep(0.15)
        assert bucket.remaining >= 5

    def test_capacity_is_max(self) -> None:
        bucket = TokenBucket(capacity=5.0, refill_rate=100.0)
        time.sleep(0.1)
        assert bucket.remaining <= 5

    def test_reset_seconds_when_available(self) -> None:
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.reset_seconds == 0.0

    def test_reset_seconds_when_empty(self) -> None:
        bucket = TokenBucket(capacity=2.0, refill_rate=1.0)
        bucket.consume(2)
        assert bucket.reset_seconds > 0


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class TestInMemoryBackend:
    """Tests for the in-memory rate limiter backend."""

    def test_creates_bucket(self) -> None:
        backend = InMemoryBackend()
        bucket = backend.get_or_create_bucket("user:1", 10.0, 1.0)
        assert isinstance(bucket, TokenBucket)
        assert bucket.capacity == 10.0

    def test_returns_same_bucket(self) -> None:
        backend = InMemoryBackend()
        b1 = backend.get_or_create_bucket("user:1", 10.0, 1.0)
        b2 = backend.get_or_create_bucket("user:1", 10.0, 1.0)
        assert b1 is b2

    def test_separate_buckets_for_different_keys(self) -> None:
        backend = InMemoryBackend()
        b1 = backend.get_or_create_bucket("user:1", 10.0, 1.0)
        b2 = backend.get_or_create_bucket("user:2", 10.0, 1.0)
        assert b1 is not b2

    def test_global_bucket(self) -> None:
        backend = InMemoryBackend()
        gb = backend.get_or_create_global_bucket(100.0, 5.0)
        assert isinstance(gb, TokenBucket)
        assert gb.capacity == 100.0

    def test_global_bucket_singleton(self) -> None:
        backend = InMemoryBackend()
        gb1 = backend.get_or_create_global_bucket(100.0, 5.0)
        gb2 = backend.get_or_create_global_bucket(100.0, 5.0)
        assert gb1 is gb2

    def test_reset_clears_all(self) -> None:
        backend = InMemoryBackend()
        backend.get_or_create_bucket("user:1", 10.0, 1.0)
        backend.get_or_create_global_bucket(100.0, 5.0)
        backend.reset()
        # After reset, new buckets are created
        b = backend.get_or_create_bucket("user:1", 20.0, 2.0)
        assert b.capacity == 20.0


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    """Tests for endpoint tier classification."""

    def test_predict_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/predict") == EndpointTier.PREDICT

    def test_predict_batch_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/predict/batch") == EndpointTier.PREDICT

    def test_report_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/report/pdf") == EndpointTier.REPORT

    def test_export_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/export/csv") == EndpointTier.REPORT

    def test_admin_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/admin/users") == EndpointTier.ADMIN

    def test_auth_endpoint(self) -> None:
        assert classify_endpoint("/api/v1/auth/login") == EndpointTier.ADMIN

    def test_health_endpoint(self) -> None:
        assert classify_endpoint("/health") == EndpointTier.SYSTEM

    def test_info_endpoint(self) -> None:
        assert classify_endpoint("/info") == EndpointTier.SYSTEM

    def test_unknown_api_defaults_to_report(self) -> None:
        assert classify_endpoint("/api/v1/something") == EndpointTier.REPORT

    def test_unknown_non_api_defaults_to_system(self) -> None:
        assert classify_endpoint("/docs") == EndpointTier.SYSTEM


# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    """Tests for rate limit configuration loading."""

    def test_default_config(self) -> None:
        cfg = RateLimitConfig()
        assert cfg.enabled is True
        assert cfg.global_requests_per_minute == 200
        assert cfg.default_requests_per_minute == 60
        assert "predict" in cfg.tiers
        assert "report" in cfg.tiers
        assert "admin" in cfg.tiers

    def test_default_tiers(self) -> None:
        cfg = RateLimitConfig()
        assert cfg.tiers["predict"].requests_per_minute == 30
        assert cfg.tiers["report"].requests_per_minute == 60
        assert cfg.tiers["admin"].requests_per_minute == 120

    def test_custom_config(self) -> None:
        cfg = RateLimitConfig(
            global_requests_per_minute=500,
            default_requests_per_minute=100,
            exempt_keys=["test-key"],
        )
        assert cfg.global_requests_per_minute == 500
        assert cfg.exempt_keys == ["test-key"]

    def test_from_env(self) -> None:
        env = {
            "AORTICA_RATE_LIMIT_ENABLED": "true",
            "AORTICA_RATE_LIMIT_GLOBAL_RPM": "300",
            "AORTICA_RATE_LIMIT_DEFAULT_RPM": "100",
            "AORTICA_RATE_LIMIT_EXEMPT_KEYS": "key1,key2",
            "AORTICA_RATE_LIMIT_EXEMPT_IPS": "127.0.0.1,10.0.0.1",
        }
        with patch.dict("os.environ", env):
            cfg = RateLimitConfig.from_env()
            assert cfg.enabled is True
            assert cfg.global_requests_per_minute == 300
            assert cfg.default_requests_per_minute == 100
            assert cfg.exempt_keys == ["key1", "key2"]
            assert cfg.exempt_ips == ["127.0.0.1", "10.0.0.1"]

    def test_from_env_disabled(self) -> None:
        with patch.dict("os.environ", {"AORTICA_RATE_LIMIT_ENABLED": "false"}):
            cfg = RateLimitConfig.from_env()
            assert cfg.enabled is False

    def test_from_env_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            cfg = RateLimitConfig.from_env()
            assert cfg.enabled is True
            assert cfg.global_requests_per_minute == 200

    def test_from_yaml(self, tmp_path: Any) -> None:
        yaml_file = tmp_path / "rate_limits.yaml"
        yaml_file.write_text(
            "enabled: true\n"
            "global_requests_per_minute: 400\n"
            "default_requests_per_minute: 80\n"
            "exempt_keys:\n"
            "  - key-abc\n"
        )
        cfg = RateLimitConfig.from_yaml(str(yaml_file))
        assert cfg.global_requests_per_minute == 400
        assert cfg.exempt_keys == ["key-abc"]


# ---------------------------------------------------------------------------
# Rate limit headers
# ---------------------------------------------------------------------------


class TestRateLimitHeaders:
    """Tests for RFC 6585 compliant rate limit headers."""

    def test_health_returns_rate_limit_headers(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_info_returns_rate_limit_headers(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers

    def test_rate_limit_remaining_decreases(self, client: TestClient) -> None:
        r1 = client.get("/health")
        rem1 = int(r1.headers["X-RateLimit-Remaining"])
        r2 = client.get("/health")
        rem2 = int(r2.headers["X-RateLimit-Remaining"])
        assert rem2 <= rem1

    def test_rate_limit_reset_is_unix_timestamp(self, client: TestClient) -> None:
        resp = client.get("/health")
        reset_ts = int(resp.headers["X-RateLimit-Reset"])
        # Should be a reasonable Unix timestamp (after 2024)
        assert reset_ts >= 1700000000


# ---------------------------------------------------------------------------
# Rate limit enforcement (429)
# ---------------------------------------------------------------------------


class TestRateLimitEnforcement:
    """Tests for 429 Too Many Requests responses."""

    def test_429_when_limit_exceeded(self) -> None:
        """Exhaust the predict tier (5 RPM + 2 burst = 7 tokens)."""
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            tiers={
                "predict": TierConfig(requests_per_minute=3, burst_size=1),
                "report": TierConfig(requests_per_minute=100, burst_size=20),
                "admin": TierConfig(requests_per_minute=100, burst_size=20),
                "system": TierConfig(requests_per_minute=100, burst_size=20),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Exhaust the predict bucket (capacity = 3 + 1 = 4)
        for _ in range(4):
            r = c.post(
                "/api/v1/predict",
                files={"file": ("test.dat", b"\x00" * 10)},
            )
            # Will be 200 or 422 (no real model), but NOT 429
            assert r.status_code != 429

        # Next request should be 429
        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429

    def test_429_has_retry_after(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            tiers={
                "predict": TierConfig(requests_per_minute=2, burst_size=1),
                "report": TierConfig(requests_per_minute=100, burst_size=20),
                "admin": TierConfig(requests_per_minute=100, burst_size=20),
                "system": TierConfig(requests_per_minute=100, burst_size=20),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Exhaust (capacity = 2 + 1 = 3)
        for _ in range(3):
            c.post(
                "/api/v1/predict",
                files={"file": ("test.dat", b"\x00" * 10)},
            )

        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        retry_after = int(r.headers["Retry-After"])
        assert retry_after >= 1

    def test_429_body_has_detail(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            tiers={
                "predict": TierConfig(requests_per_minute=1, burst_size=1),
                "report": TierConfig(requests_per_minute=100, burst_size=20),
                "admin": TierConfig(requests_per_minute=100, burst_size=20),
                "system": TierConfig(requests_per_minute=100, burst_size=20),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Exhaust (capacity = 1 + 1 = 2)
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )

        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429
        body = r.json()
        assert "detail" in body
        assert body["detail"] == "Too Many Requests"
        assert "retry_after" in body


# ---------------------------------------------------------------------------
# Tier differentiation
# ---------------------------------------------------------------------------


class TestTierDifferentiation:
    """Tests verifying different tiers have different limits."""

    def test_predict_has_lower_limit_than_system(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            tiers={
                "predict": TierConfig(requests_per_minute=2, burst_size=1),
                "report": TierConfig(requests_per_minute=10, burst_size=5),
                "admin": TierConfig(requests_per_minute=20, burst_size=5),
                "system": TierConfig(requests_per_minute=50, burst_size=10),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Predict tier: limit = 2 RPM, 1 burst → 3 tokens
        for _ in range(3):
            r = c.post(
                "/api/v1/predict",
                files={"file": ("test.dat", b"\x00" * 10)},
            )
            assert r.status_code != 429

        # Predict should be exhausted
        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429

        # System tier should still work
        r = c.get("/health")
        assert r.status_code == 200

    def test_tiers_are_independent(self) -> None:
        """Exhausting one tier does not affect another."""
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            tiers={
                "predict": TierConfig(requests_per_minute=1, burst_size=1),
                "report": TierConfig(requests_per_minute=100, burst_size=20),
                "admin": TierConfig(requests_per_minute=100, burst_size=20),
                "system": TierConfig(requests_per_minute=100, burst_size=20),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Exhaust predict (capacity = 1 + 1 = 2)
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429

        # Health (system tier) is fine
        r = c.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Exempt keys and IPs
# ---------------------------------------------------------------------------


class TestExemptions:
    """Tests for exempt API keys and IPs."""

    def test_exempt_api_key_bypasses_limit(self) -> None:
        cfg = RateLimitConfig(
            enabled=True,
            global_requests_per_minute=200,
            exempt_keys=["trusted-key"],
            tiers={
                "predict": TierConfig(requests_per_minute=1, burst_size=1),
                "report": TierConfig(requests_per_minute=100, burst_size=20),
                "admin": TierConfig(requests_per_minute=100, burst_size=20),
                "system": TierConfig(requests_per_minute=100, burst_size=20),
            },
        )
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Non-exempt: first two succeed (capacity=2), third blocked
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        r = c.post(
            "/api/v1/predict",
            files={"file": ("test.dat", b"\x00" * 10)},
        )
        assert r.status_code == 429

        # Exempt key: always passes
        for _ in range(5):
            r = c.post(
                "/api/v1/predict",
                files={"file": ("test.dat", b"\x00" * 10)},
                headers={"X-Api-Key": "trusted-key"},
            )
            assert r.status_code != 429


# ---------------------------------------------------------------------------
# Disabled rate limiting
# ---------------------------------------------------------------------------


class TestDisabledRateLimiting:
    """Tests for when rate limiting is disabled."""

    def test_disabled_allows_all(self) -> None:
        cfg = RateLimitConfig(enabled=False)
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        # Should never get 429
        for _ in range(20):
            r = c.get("/health")
            assert r.status_code == 200

    def test_disabled_no_rate_limit_headers(self) -> None:
        cfg = RateLimitConfig(enabled=False)
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        c = TestClient(app)

        r = c.get("/health")
        assert r.status_code == 200
        # When disabled, no rate limit headers should be present
        assert "X-RateLimit-Limit" not in r.headers


# ---------------------------------------------------------------------------
# IP-based rate limiting
# ---------------------------------------------------------------------------


class TestIPBasedRateLimiting:
    """Tests for IP-based rate limiting on unauthenticated endpoints."""

    def test_ip_based_key_extraction(self) -> None:
        """When no API key is provided, client IP is used."""
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        key = _extract_client_key(mock_request)
        assert key == "ip:192.168.1.1"

    def test_api_key_based_key_extraction(self) -> None:
        """When API key is provided, it's used as the identifier."""
        mock_request = MagicMock()
        mock_request.headers = {"x-api-key": "my-key-123"}

        key = _extract_client_key(mock_request)
        assert key == "key:my-key-123"


# ---------------------------------------------------------------------------
# Redis backend (mock tests — no real Redis needed)
# ---------------------------------------------------------------------------


class TestRedisBackend:
    """Tests for Redis backend configuration (mocked)."""

    def test_redis_url_config(self) -> None:
        cfg = RateLimitConfig(redis_url="redis://localhost:6379")
        assert cfg.redis_url == "redis://localhost:6379"

    def test_from_env_redis_url(self) -> None:
        with patch.dict(
            "os.environ",
            {"AORTICA_RATE_LIMIT_REDIS_URL": "redis://myhost:6379/0"},
        ):
            cfg = RateLimitConfig.from_env()
            assert cfg.redis_url == "redis://myhost:6379/0"


# ---------------------------------------------------------------------------
# Integration: rate limiter on create_app
# ---------------------------------------------------------------------------


class TestCreateAppIntegration:
    """Tests for rate limiting integration with create_app."""

    def test_app_has_rate_limit_config_state(self) -> None:
        cfg = RateLimitConfig()
        app = create_app(rate_limit_config=cfg, enable_auth=False)
        assert hasattr(app.state, "rate_limit_config")

    def test_app_default_rate_limiting(self) -> None:
        """Default app has rate limiting enabled."""
        app = create_app(enable_auth=False)
        assert hasattr(app.state, "rate_limit_config")
        assert app.state.rate_limit_config.enabled is True

    def test_app_with_yaml_config(self, tmp_path: Any) -> None:
        yaml_file = tmp_path / "rl.yaml"
        yaml_file.write_text(
            "enabled: true\n"
            "global_requests_per_minute: 999\n"
        )
        app = create_app(
            rate_limit_config_path=str(yaml_file),
            enable_auth=False,
        )
        assert app.state.rate_limit_config.global_requests_per_minute == 999


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API imports work."""

    def test_import_rate_limiter(self) -> None:
        from aortica.api import rate_limiter

        assert hasattr(rate_limiter, "RateLimiterMiddleware")
        assert hasattr(rate_limiter, "RateLimitConfig")
        assert hasattr(rate_limiter, "add_rate_limiting")

    def test_import_endpoint_tier(self) -> None:
        from aortica.api.rate_limiter import EndpointTier

        assert EndpointTier.PREDICT.value == "predict"
        assert EndpointTier.REPORT.value == "report"
        assert EndpointTier.ADMIN.value == "admin"
        assert EndpointTier.SYSTEM.value == "system"

    def test_import_token_bucket(self) -> None:
        from aortica.api.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.capacity == 10.0
