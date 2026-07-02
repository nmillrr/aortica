"""Rate limiting middleware for the Aortica API.

Provides token-bucket rate limiting with:
- Per-user (API key / IP) and global request throttling
- Tiered endpoint limits (predict, report, admin)
- Redis-backed or in-memory backends
- RFC 6585 compliant rate-limit headers
- Configurable via YAML or environment variables
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class EndpointTier(str, Enum):
    """Rate limit tiers for different endpoint categories."""

    PREDICT = "predict"
    REPORT = "report"
    ADMIN = "admin"
    SYSTEM = "system"


class TierConfig(BaseModel):
    """Rate limit configuration for a single endpoint tier."""

    requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute for this tier",
        gt=0,
    )
    burst_size: int = Field(
        default=10,
        description="Maximum burst size (token bucket capacity above rate)",
        gt=0,
    )


class RateLimitConfig(BaseModel):
    """Full rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable/disable rate limiting")
    global_requests_per_minute: int = Field(
        default=200,
        description="Global rate limit across all users",
        gt=0,
    )
    default_requests_per_minute: int = Field(
        default=60,
        description="Default per-user rate limit",
        gt=0,
    )
    tiers: Dict[str, TierConfig] = Field(
        default_factory=lambda: {
            EndpointTier.PREDICT.value: TierConfig(
                requests_per_minute=30, burst_size=5
            ),
            EndpointTier.REPORT.value: TierConfig(
                requests_per_minute=60, burst_size=10
            ),
            EndpointTier.ADMIN.value: TierConfig(
                requests_per_minute=120, burst_size=20
            ),
            EndpointTier.SYSTEM.value: TierConfig(
                requests_per_minute=300, burst_size=50
            ),
        },
        description="Per-tier rate limit overrides",
    )
    exempt_keys: List[str] = Field(
        default_factory=list,
        description="API keys exempt from rate limiting",
    )
    exempt_ips: List[str] = Field(
        default_factory=list,
        description="IP addresses exempt from rate limiting",
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for distributed rate limiting (None = in-memory)",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "RateLimitConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load rate limit config from YAML. "
                "Install it with: pip install pyyaml"
            ) from None

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Load configuration from environment variables."""
        kwargs: Dict[str, Any] = {}

        if os.environ.get("AORTICA_RATE_LIMIT_ENABLED") is not None:
            kwargs["enabled"] = os.environ["AORTICA_RATE_LIMIT_ENABLED"].lower() in (
                "1",
                "true",
                "yes",
            )
        if os.environ.get("AORTICA_RATE_LIMIT_GLOBAL_RPM"):
            kwargs["global_requests_per_minute"] = int(
                os.environ["AORTICA_RATE_LIMIT_GLOBAL_RPM"]
            )
        if os.environ.get("AORTICA_RATE_LIMIT_DEFAULT_RPM"):
            kwargs["default_requests_per_minute"] = int(
                os.environ["AORTICA_RATE_LIMIT_DEFAULT_RPM"]
            )
        if os.environ.get("AORTICA_RATE_LIMIT_REDIS_URL"):
            kwargs["redis_url"] = os.environ["AORTICA_RATE_LIMIT_REDIS_URL"]
        if os.environ.get("AORTICA_RATE_LIMIT_EXEMPT_KEYS"):
            kwargs["exempt_keys"] = [
                k.strip()
                for k in os.environ["AORTICA_RATE_LIMIT_EXEMPT_KEYS"].split(",")
                if k.strip()
            ]
        if os.environ.get("AORTICA_RATE_LIMIT_EXEMPT_IPS"):
            kwargs["exempt_ips"] = [
                ip.strip()
                for ip in os.environ["AORTICA_RATE_LIMIT_EXEMPT_IPS"].split(",")
                if ip.strip()
            ]

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Token bucket implementation
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """In-memory token bucket for rate limiting.

    Tokens are replenished at a fixed rate.  Each request consumes one
    token.  When the bucket is empty, requests are rejected until tokens
    are replenished.
    """

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def consume(self, count: int = 1) -> bool:
        """Try to consume *count* tokens.  Returns ``True`` on success."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    @property
    def remaining(self) -> int:
        """Number of tokens currently available (floor)."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        current = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return max(0, int(current))

    @property
    def reset_seconds(self) -> float:
        """Seconds until a token becomes available (0 if available now)."""
        if self.remaining > 0:
            return 0.0
        deficit = 1.0 - self.tokens
        if self.refill_rate <= 0:
            return 60.0
        return max(0.0, deficit / self.refill_rate)


# ---------------------------------------------------------------------------
# Rate limiter backends
# ---------------------------------------------------------------------------


class InMemoryBackend:
    """In-memory rate limiter backend using token buckets.

    Suitable for single-process / edge deployments.
    """

    def __init__(self) -> None:
        self._buckets: Dict[str, TokenBucket] = {}
        self._global_bucket: Optional[TokenBucket] = None

    def get_or_create_bucket(
        self, key: str, capacity: float, refill_rate: float
    ) -> TokenBucket:
        """Get an existing bucket or create a new one for *key*."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=capacity, refill_rate=refill_rate
            )
        return self._buckets[key]

    def get_or_create_global_bucket(
        self, capacity: float, refill_rate: float
    ) -> TokenBucket:
        """Get or create the global rate limit bucket."""
        if self._global_bucket is None:
            self._global_bucket = TokenBucket(
                capacity=capacity, refill_rate=refill_rate
            )
        return self._global_bucket

    def reset(self) -> None:
        """Clear all buckets (for testing)."""
        self._buckets.clear()
        self._global_bucket = None


class RedisBackend:
    """Redis-backed rate limiter backend for multi-process deployments.

    Uses a Lua script for atomic token bucket operations.
    """

    # Lua script for atomic token bucket consume
    _LUA_CONSUME = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local count = tonumber(ARGV[4])
    local ttl = tonumber(ARGV[5])

    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1])
    local last_refill = tonumber(bucket[2])

    if tokens == nil then
        tokens = capacity
        last_refill = now
    end

    local elapsed = now - last_refill
    tokens = math.min(capacity, tokens + elapsed * refill_rate)
    last_refill = now

    local allowed = 0
    if tokens >= count then
        tokens = tokens - count
        allowed = 1
    end

    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, ttl)

    local remaining = math.max(0, math.floor(tokens))
    local reset_secs = 0
    if tokens < 1 then
        reset_secs = math.ceil((1 - tokens) / refill_rate)
    end

    return {allowed, remaining, reset_secs}
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "redis package is required for Redis-backed rate limiting. "
                "Install it with: pip install redis"
            ) from None

        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        self._script = self._redis.register_script(self._LUA_CONSUME)

    def consume(
        self,
        key: str,
        capacity: float,
        refill_rate: float,
        count: int = 1,
    ) -> tuple[bool, int, float]:
        """Atomically consume tokens.

        Returns (allowed, remaining, reset_seconds).
        """
        now = time.time()
        ttl = int(capacity / max(refill_rate, 0.001)) + 60  # bucket TTL

        result = self._script(
            keys=[f"aortica:rl:{key}"],
            args=[capacity, refill_rate, now, count, ttl],
        )

        allowed = bool(int(result[0]))
        remaining = int(result[1])
        reset_secs = float(result[2])

        return allowed, remaining, reset_secs


# ---------------------------------------------------------------------------
# Tier routing
# ---------------------------------------------------------------------------

# Default path prefix → tier mappings
_DEFAULT_TIER_ROUTES: Dict[str, EndpointTier] = {
    "/api/v1/predict": EndpointTier.PREDICT,
    "/api/v1/export": EndpointTier.REPORT,
    "/api/v1/report": EndpointTier.REPORT,
    "/api/v1/retrieve": EndpointTier.REPORT,
    "/api/v1/compare": EndpointTier.REPORT,
    "/api/v1/worklist": EndpointTier.REPORT,
    "/api/v1/admin": EndpointTier.ADMIN,
    "/api/v1/auth": EndpointTier.ADMIN,
    "/api/v1/api-keys": EndpointTier.ADMIN,
    "/health": EndpointTier.SYSTEM,
    "/info": EndpointTier.SYSTEM,
}


def classify_endpoint(path: str) -> EndpointTier:
    """Classify a request path into a rate limit tier."""
    for prefix, tier in _DEFAULT_TIER_ROUTES.items():
        if path.startswith(prefix):
            return tier
    # Default to REPORT tier for unknown API paths
    if path.startswith("/api/"):
        return EndpointTier.REPORT
    return EndpointTier.SYSTEM


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _extract_client_key(request: Any) -> str:
    """Extract a client identifier from the request.

    Uses the API key header if present, otherwise falls back to IP address.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    # Fallback to IP
    client_host = getattr(request.client, "host", None) if request.client else None
    return f"ip:{client_host or 'unknown'}"


def _extract_client_ip(request: Any) -> Optional[str]:
    """Extract the client IP from the request."""
    if request.client:
        return getattr(request.client, "host", None)
    return None


def _extract_api_key(request: Any) -> Optional[str]:
    """Extract the API key from the request headers."""
    value: Optional[str] = request.headers.get("x-api-key")
    return value


class RateLimiterMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """FastAPI middleware implementing token-bucket rate limiting.

    Supports per-user and global rate limits with tiered endpoint
    configuration.  Uses in-memory backend by default; switch to
    Redis for multi-process deployments.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RateLimitConfig] = None,
    ) -> None:
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self._backend: Optional[InMemoryBackend] = None
        self._redis_backend: Optional[RedisBackend] = None

        if self.config.redis_url:
            try:
                self._redis_backend = RedisBackend(self.config.redis_url)
            except ImportError:
                # Fall back to in-memory if redis not installed
                self._backend = InMemoryBackend()
        else:
            self._backend = InMemoryBackend()

    async def dispatch(self, request: Request, call_next: Any) -> Any:  # type: ignore[override]
        """Process request through rate limiting."""
        if not self.config.enabled:
            response = await call_next(request)
            return response

        path = request.url.path
        client_key = _extract_client_key(request)
        client_ip = _extract_client_ip(request)
        api_key = _extract_api_key(request)

        # Check exempt list
        if api_key and api_key in self.config.exempt_keys:
            response = await call_next(request)
            return response
        if client_ip and client_ip in self.config.exempt_ips:
            response = await call_next(request)
            return response

        # Determine tier
        tier = classify_endpoint(path)
        tier_config = self.config.tiers.get(
            tier.value,
            TierConfig(requests_per_minute=self.config.default_requests_per_minute),
        )

        per_user_rpm = tier_config.requests_per_minute
        per_user_capacity = per_user_rpm + tier_config.burst_size
        per_user_refill = per_user_rpm / 60.0

        global_rpm = self.config.global_requests_per_minute
        global_capacity = global_rpm + 50  # burst headroom
        global_refill = global_rpm / 60.0

        # --- Rate limit check ---
        if self._redis_backend is not None:
            # Redis backend
            user_allowed, user_remaining, user_reset = self._redis_backend.consume(
                f"user:{client_key}:{tier.value}",
                per_user_capacity,
                per_user_refill,
            )
            global_allowed, global_remaining, global_reset = (
                self._redis_backend.consume(
                    "global",
                    global_capacity,
                    global_refill,
                )
            )
            allowed = user_allowed and global_allowed
            remaining = min(user_remaining, global_remaining)
            reset_seconds = max(user_reset, global_reset)
        else:
            # In-memory backend
            assert self._backend is not None
            user_bucket = self._backend.get_or_create_bucket(
                f"{client_key}:{tier.value}",
                per_user_capacity,
                per_user_refill,
            )
            global_bucket = self._backend.get_or_create_global_bucket(
                global_capacity, global_refill
            )

            user_allowed = user_bucket.consume()
            global_allowed = global_bucket.consume()

            allowed = user_allowed and global_allowed
            remaining = min(user_bucket.remaining, global_bucket.remaining)
            reset_seconds = max(user_bucket.reset_seconds, global_bucket.reset_seconds)

        # Build rate limit headers (RFC 6585 compliant)
        headers = {
            "X-RateLimit-Limit": str(per_user_rpm),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(int(time.time() + reset_seconds)),
        }

        if not allowed:
            retry_after = max(1, int(reset_seconds))
            headers["Retry-After"] = str(retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too Many Requests",
                    "retry_after": retry_after,
                },
                headers=headers,
            )

        # Allowed — proceed and attach headers
        response = await call_next(request)
        for hdr_name, hdr_value in headers.items():
            response.headers[hdr_name] = hdr_value
        return response


# ---------------------------------------------------------------------------
# Convenience: attach middleware to app
# ---------------------------------------------------------------------------


def add_rate_limiting(
    app: FastAPI,
    config: Optional[RateLimitConfig] = None,
    config_path: Optional[str] = None,
) -> None:
    """Add rate limiting middleware to a FastAPI application.

    Parameters
    ----------
    app:
        The FastAPI application instance.
    config:
        A ``RateLimitConfig`` instance.  If *None*, attempts to load
        from *config_path*, then from environment variables, then uses
        defaults.
    config_path:
        Path to a ``rate_limits.yaml`` configuration file.
    """
    if config is None:
        if config_path and os.path.exists(config_path):
            config = RateLimitConfig.from_yaml(config_path)
        else:
            config = RateLimitConfig.from_env()

    app.add_middleware(RateLimiterMiddleware, config=config)
    app.state.rate_limit_config = config  # type: ignore[attr-defined]
