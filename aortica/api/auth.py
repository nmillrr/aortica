"""Authentication module for the Aortica API.

Provides OAuth 2.0 (Google/GitHub) via ``authlib``, local API key
authentication (``X-API-Key`` header), JWT token issuance, and FastAPI
security dependencies.

All sensitive values (OAuth client IDs/secrets, JWT secret) are read from
environment variables at startup.  In a self-hosted deployment the admin
sets these in ``.env`` or Docker/Compose environment.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from fastapi import Depends, Header, HTTPException, Request, status
    from fastapi.security import HTTPBearer

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

try:
    import jwt as pyjwt  # PyJWT

    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    pyjwt = None  # type: ignore[assignment]

try:
    from authlib.integrations.starlette_client import OAuth as AuthlibOAuth

    HAS_AUTHLIB = True
except ImportError:
    HAS_AUTHLIB = False
    AuthlibOAuth = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 86400  # 24 hours
JWT_REFRESH_EXPIRY_SECONDS = 604800  # 7 days
API_KEY_PREFIX = "ak_"
API_KEY_LENGTH = 40  # total length including prefix

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    """Response model for token endpoints."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Seconds until token expires")
    refresh_token: Optional[str] = Field(
        default=None, description="Refresh token for obtaining new access tokens"
    )


class APIKeyResponse(BaseModel):
    """Response model for API key generation."""

    api_key: str = Field(..., description="Generated API key")
    name: str = Field(..., description="Key display name")
    created_at: float = Field(..., description="Unix timestamp of creation")


class UserInfo(BaseModel):
    """Authenticated user information."""

    sub: str = Field(..., description="User subject identifier")
    email: Optional[str] = Field(default=None, description="User email")
    name: Optional[str] = Field(default=None, description="User display name")
    provider: str = Field(
        default="local", description="Auth provider (google, github, api_key)"
    )


# ---------------------------------------------------------------------------
# API key store (in-memory for single-instance self-hosted deployment)
# ---------------------------------------------------------------------------


@dataclass
class StoredAPIKey:
    """An API key record in the store."""

    key_hash: str
    name: str
    user_sub: str
    created_at: float = field(default_factory=time.time)


class APIKeyStore:
    """Simple in-memory API key store.

    Keys are stored as SHA-256 hashes.  The raw key is returned
    only at creation time and never persisted.
    """

    def __init__(self) -> None:
        self._keys: Dict[str, StoredAPIKey] = {}  # hash -> record

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def generate(self, name: str, user_sub: str) -> tuple[str, StoredAPIKey]:
        """Generate a new API key, returning ``(raw_key, record)``."""
        raw_key = API_KEY_PREFIX + secrets.token_urlsafe(API_KEY_LENGTH - len(API_KEY_PREFIX))
        h = self._hash_key(raw_key)
        record = StoredAPIKey(key_hash=h, name=name, user_sub=user_sub, created_at=time.time())
        self._keys[h] = record
        return raw_key, record

    def validate(self, raw_key: str) -> Optional[StoredAPIKey]:
        """Return the record if *raw_key* is valid, else ``None``."""
        h = self._hash_key(raw_key)
        return self._keys.get(h)

    def list_keys(self, user_sub: str) -> List[StoredAPIKey]:
        """List all keys for a given user."""
        return [k for k in self._keys.values() if k.user_sub == user_sub]

    def revoke(self, key_hash: str) -> bool:
        """Revoke a key by its hash.  Returns ``True`` if found."""
        return self._keys.pop(key_hash, None) is not None

    @property
    def count(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _get_jwt_secret() -> str:
    """Return the JWT signing secret from env, or a default for dev."""
    return os.environ.get("AORTICA_JWT_SECRET", "aortica-dev-secret-change-me")


def create_access_token(
    user: UserInfo,
    *,
    secret: Optional[str] = None,
    expires_in: int = JWT_EXPIRY_SECONDS,
) -> str:
    """Create a signed JWT access token for *user*."""
    if not HAS_JWT:
        raise ImportError(
            "PyJWT is required for authentication. "
            "Install with: pip install aortica[auth]"
        )
    sec = secret or _get_jwt_secret()
    now = time.time()
    payload = {
        "sub": user.sub,
        "email": user.email,
        "name": user.name,
        "provider": user.provider,
        "iat": int(now),
        "exp": int(now + expires_in),
    }
    token: str = pyjwt.encode(payload, sec, algorithm=JWT_ALGORITHM)  # type: ignore[union-attr]
    return token


def create_refresh_token(
    user: UserInfo,
    *,
    secret: Optional[str] = None,
    expires_in: int = JWT_REFRESH_EXPIRY_SECONDS,
) -> str:
    """Create a long-lived refresh token for *user*."""
    if not HAS_JWT:
        raise ImportError(
            "PyJWT is required for authentication. "
            "Install with: pip install aortica[auth]"
        )
    sec = secret or _get_jwt_secret()
    now = time.time()
    payload = {
        "sub": user.sub,
        "type": "refresh",
        "iat": int(now),
        "exp": int(now + expires_in),
    }
    token: str = pyjwt.encode(payload, sec, algorithm=JWT_ALGORITHM)  # type: ignore[union-attr]
    return token


def decode_token(
    token: str,
    *,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode and verify a JWT token.  Raises on expiry / bad signature."""
    if not HAS_JWT:
        raise ImportError(
            "PyJWT is required for authentication. "
            "Install with: pip install aortica[auth]"
        )
    sec = secret or _get_jwt_secret()
    payload: Dict[str, Any] = pyjwt.decode(  # type: ignore[union-attr]
        token, sec, algorithms=[JWT_ALGORITHM]
    )
    return payload


# ---------------------------------------------------------------------------
# OAuth setup helper
# ---------------------------------------------------------------------------


def create_oauth(
    *,
    google_client_id: Optional[str] = None,
    google_client_secret: Optional[str] = None,
    github_client_id: Optional[str] = None,
    github_client_secret: Optional[str] = None,
) -> Any:
    """Create an ``authlib`` OAuth instance configured with Google and GitHub.

    Reads credentials from parameters or environment variables
    ``GOOGLE_CLIENT_ID``, ``GOOGLE_CLIENT_SECRET``, etc.
    """
    if not HAS_AUTHLIB:
        raise ImportError(
            "authlib is required for OAuth. "
            "Install with: pip install aortica[auth]"
        )

    oauth = AuthlibOAuth()

    g_id = google_client_id or os.environ.get("GOOGLE_CLIENT_ID", "")
    g_secret = google_client_secret or os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if g_id and g_secret:
        oauth.register(  # type: ignore[union-attr]
            name="google",
            client_id=g_id,
            client_secret=g_secret,
            server_metadata_url=(
                "https://accounts.google.com/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email profile"},
        )

    gh_id = github_client_id or os.environ.get("GITHUB_CLIENT_ID", "")
    gh_secret = github_client_secret or os.environ.get("GITHUB_CLIENT_SECRET", "")
    if gh_id and gh_secret:
        oauth.register(  # type: ignore[union-attr]
            name="github",
            client_id=gh_id,
            client_secret=gh_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )

    return oauth


# ---------------------------------------------------------------------------
# FastAPI security dependencies
# ---------------------------------------------------------------------------

# Reusable bearer scheme (auto_error=False so we can also check X-API-Key)
_bearer_scheme = HTTPBearer(auto_error=False) if HAS_FASTAPI else None


def _extract_user_from_token(token: str) -> UserInfo:
    """Decode JWT and return *UserInfo*, or raise 401."""
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserInfo(
        sub=payload["sub"],
        email=payload.get("email"),
        name=payload.get("name"),
        provider=payload.get("provider", "local"),
    )


async def require_auth(
    request: Request,  # type: ignore[arg-type]
    x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
) -> UserInfo:
    """FastAPI dependency that authenticates via JWT **or** API key.

    Check order:
      1. ``Authorization: Bearer <jwt>`` header
      2. ``X-API-Key: <api_key>`` header

    Returns the authenticated :class:`UserInfo`.
    Raises ``401 Unauthorized`` if neither is valid.
    """
    # 1. Bearer JWT
    auth_header: Optional[str] = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        return _extract_user_from_token(token)

    # 2. API key
    if x_api_key:
        store: Optional[APIKeyStore] = getattr(request.app.state, "api_key_store", None)
        if store is not None:
            record = store.validate(x_api_key)
            if record is not None:
                return UserInfo(
                    sub=record.user_sub,
                    provider="api_key",
                )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Auth router factory
# ---------------------------------------------------------------------------


def create_auth_router(api_key_store: APIKeyStore) -> Any:
    """Build a FastAPI APIRouter with authentication endpoints.

    Endpoints:
      - ``POST /api/v1/auth/token``  — generate API key (requires JWT)
      - ``POST /api/v1/auth/refresh`` — refresh JWT with refresh token
      - ``GET  /api/v1/auth/login/google`` — redirect to Google OAuth
      - ``GET  /api/v1/auth/callback/google`` — Google OAuth callback
      - ``GET  /api/v1/auth/login/github`` — redirect to GitHub OAuth
      - ``GET  /api/v1/auth/callback/github`` — GitHub OAuth callback
    """
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

    # ── API key generation (protected) ───────────────────────────────

    @router.post(
        "/token",
        response_model=APIKeyResponse,
        summary="Generate API key",
    )
    async def generate_api_key(
        request: Request,  # type: ignore[arg-type]
        name: str = "default",
        user: Any = Depends(require_auth),
    ) -> APIKeyResponse:
        """Generate a new API key for the authenticated user."""
        raw_key, record = api_key_store.generate(name=name, user_sub=user.sub)
        return APIKeyResponse(
            api_key=raw_key,
            name=record.name,
            created_at=record.created_at,
        )

    # ── JWT refresh ──────────────────────────────────────────────────

    @router.post(
        "/refresh",
        response_model=TokenResponse,
        summary="Refresh access token",
    )
    async def refresh_token(
        request: Request,  # type: ignore[arg-type]
    ) -> TokenResponse:
        """Exchange a valid refresh token for a new access token."""
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token required",
            )
        raw_token = auth_header[7:]
        try:
            payload = decode_token(raw_token)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not a refresh token",
            )
        user = UserInfo(
            sub=payload["sub"],
            provider=payload.get("provider", "local"),
        )
        access = create_access_token(user)
        return TokenResponse(
            access_token=access,
            expires_in=JWT_EXPIRY_SECONDS,
        )

    # ── OAuth login / callback stubs ─────────────────────────────────
    # These redirect to the external provider.  The callback endpoint
    # exchanges the code for user info and returns a JWT.

    @router.get("/login/google", summary="Redirect to Google OAuth")
    async def login_google(request: Request) -> Any:  # type: ignore[arg-type]
        """Initiate Google OAuth flow."""
        oauth = getattr(request.app.state, "oauth", None)
        if oauth is None or not hasattr(oauth, "google"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Google OAuth not configured",
            )
        redirect_uri = str(request.url_for("callback_google"))
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @router.get(
        "/callback/google",
        response_model=TokenResponse,
        summary="Google OAuth callback",
    )
    async def callback_google(request: Request) -> TokenResponse:  # type: ignore[arg-type]
        """Handle Google OAuth callback and return JWT."""
        oauth = getattr(request.app.state, "oauth", None)
        if oauth is None or not hasattr(oauth, "google"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Google OAuth not configured",
            )
        token_data = await oauth.google.authorize_access_token(request)
        id_info = token_data.get("userinfo", {})
        user = UserInfo(
            sub=f"google:{id_info.get('sub', '')}",
            email=id_info.get("email"),
            name=id_info.get("name"),
            provider="google",
        )
        access = create_access_token(user)
        refresh = create_refresh_token(user)
        return TokenResponse(
            access_token=access,
            token_type="bearer",
            expires_in=JWT_EXPIRY_SECONDS,
            refresh_token=refresh,
        )

    @router.get("/login/github", summary="Redirect to GitHub OAuth")
    async def login_github(request: Request) -> Any:  # type: ignore[arg-type]
        """Initiate GitHub OAuth flow."""
        oauth = getattr(request.app.state, "oauth", None)
        if oauth is None or not hasattr(oauth, "github"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="GitHub OAuth not configured",
            )
        redirect_uri = str(request.url_for("callback_github"))
        return await oauth.github.authorize_redirect(request, redirect_uri)

    @router.get(
        "/callback/github",
        response_model=TokenResponse,
        summary="GitHub OAuth callback",
    )
    async def callback_github(request: Request) -> TokenResponse:  # type: ignore[arg-type]
        """Handle GitHub OAuth callback and return JWT."""
        oauth = getattr(request.app.state, "oauth", None)
        if oauth is None or not hasattr(oauth, "github"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="GitHub OAuth not configured",
            )
        token_data = await oauth.github.authorize_access_token(request)
        # Fetch user info from GitHub API
        resp = await oauth.github.get("user", token=token_data)
        gh_user = resp.json()
        user = UserInfo(
            sub=f"github:{gh_user.get('id', '')}",
            email=gh_user.get("email"),
            name=gh_user.get("name") or gh_user.get("login"),
            provider="github",
        )
        access = create_access_token(user)
        refresh = create_refresh_token(user)
        return TokenResponse(
            access_token=access,
            token_type="bearer",
            expires_in=JWT_EXPIRY_SECONDS,
            refresh_token=refresh,
        )

    return router
