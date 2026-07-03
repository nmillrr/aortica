"""FastAPI router for admin dashboard endpoints.

Provides admin-only endpoints for managing users, API keys, system
health monitoring, and activity logging.

Endpoints:
  - ``GET    /api/v1/admin/users``            — list all users
  - ``PATCH  /api/v1/admin/users/:id``        — update user role/status
  - ``DELETE /api/v1/admin/api-keys/:key_id`` — revoke an API key
  - ``GET    /api/v1/admin/system-health``     — system health metrics
  - ``GET    /api/v1/admin/activity-log``      — paginated activity log

All endpoints require an authenticated user with ``role == 'admin'``.
Non-admin users receive ``403 Forbidden``.

Usage::

    from aortica.api.admin_endpoints import create_admin_router

    router = create_admin_router(user_store, api_key_store, activity_log)
    app.include_router(router)
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class UserRecord(BaseModel):
    """A registered user record."""

    id: str = Field(..., description="User unique identifier")
    email: Optional[str] = Field(default=None, description="User email address")
    name: Optional[str] = Field(default=None, description="User display name")
    role: str = Field(default="clinician", description="User role: admin, clinician, researcher")
    last_login: Optional[float] = Field(default=None, description="Unix timestamp of last login")
    status: str = Field(default="active", description="Account status: active or disabled")
    provider: str = Field(default="local", description="Auth provider")


class UserListResponse(BaseModel):
    """Response for GET /api/v1/admin/users."""

    users: List[UserRecord] = Field(default_factory=list, description="List of registered users")
    total: int = Field(default=0, description="Total number of users")


class UserUpdateRequest(BaseModel):
    """Request body for PATCH /api/v1/admin/users/:id."""

    role: Optional[str] = Field(default=None, description="New role: admin, clinician, researcher")
    status: Optional[str] = Field(default=None, description="New status: active, disabled")


class UserUpdateResponse(BaseModel):
    """Response for PATCH /api/v1/admin/users/:id."""

    id: str = Field(..., description="Updated user ID")
    role: str = Field(..., description="Current role after update")
    status: str = Field(..., description="Current status after update")
    updated: bool = Field(default=True, description="Whether the update was applied")


class APIKeyRecord(BaseModel):
    """An API key record for admin listing."""

    key_id: str = Field(..., description="Hash-based key identifier")
    name: str = Field(..., description="Key display name")
    user_sub: str = Field(..., description="User who owns this key")
    created_at: float = Field(..., description="Unix timestamp of creation")
    last_used: Optional[float] = Field(default=None, description="Unix timestamp of last use")


class APIKeyListResponse(BaseModel):
    """Response for listing API keys."""

    keys: List[APIKeyRecord] = Field(default_factory=list, description="List of API keys")
    total: int = Field(default=0, description="Total number of keys")


class APIKeyRevokeResponse(BaseModel):
    """Response for DELETE /api/v1/admin/api-keys/:key_id."""

    key_id: str = Field(..., description="Revoked key identifier")
    revoked: bool = Field(..., description="Whether the key was found and revoked")


class SystemHealthResponse(BaseModel):
    """Response for GET /api/v1/admin/system-health."""

    status: str = Field(default="ok", description="Overall system status")
    model_version: Optional[str] = Field(default=None, description="Currently loaded model version")
    model_loaded: bool = Field(default=False, description="Whether a model is loaded")
    database_size_bytes: int = Field(default=0, description="SQLite database size in bytes")
    total_ecgs_processed: int = Field(default=0, description="Total ECGs processed since startup")
    uptime_seconds: float = Field(default=0.0, description="Server uptime in seconds")
    onnx_runtime_available: bool = Field(default=False, description="Whether ONNX Runtime is available")
    sync_engine_status: str = Field(default="inactive", description="Sync engine status")


class ActivityLogEntry(BaseModel):
    """A single activity log entry."""

    timestamp: float = Field(..., description="Unix timestamp of the request")
    user: Optional[str] = Field(default=None, description="User who made the request")
    endpoint: str = Field(..., description="Request endpoint path")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., description="Response status code")
    duration_ms: Optional[float] = Field(default=None, description="Request duration in ms")


class ActivityLogResponse(BaseModel):
    """Response for GET /api/v1/admin/activity-log."""

    entries: List[ActivityLogEntry] = Field(default_factory=list, description="Activity log entries")
    total: int = Field(default=0, description="Total number of entries")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Entries per page")


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

VALID_ROLES = {"admin", "clinician", "researcher"}
VALID_STATUSES = {"active", "disabled"}


@dataclass
class UserStoreRecord:
    """Internal user record in the store."""

    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    role: str = "clinician"
    last_login: Optional[float] = None
    status: str = "active"
    provider: str = "local"


class UserStore:
    """Simple in-memory user store for self-hosted deployments.

    In production, this would be backed by a database.  For the
    self-hosted model, an in-memory store is sufficient.
    """

    def __init__(self) -> None:
        self._users: Dict[str, UserStoreRecord] = {}

    def add(self, record: UserStoreRecord) -> None:
        """Add or update a user record."""
        self._users[record.id] = record

    def get(self, user_id: str) -> Optional[UserStoreRecord]:
        """Return user by ID, or None."""
        return self._users.get(user_id)

    def list_all(self) -> List[UserStoreRecord]:
        """Return all user records."""
        return list(self._users.values())

    def update(
        self,
        user_id: str,
        *,
        role: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[UserStoreRecord]:
        """Update a user's role or status.  Returns updated record or None."""
        record = self._users.get(user_id)
        if record is None:
            return None
        if role is not None:
            record.role = role
        if status is not None:
            record.status = status
        return record

    def upsert_from_auth(
        self,
        sub: str,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        provider: str = "local",
    ) -> UserStoreRecord:
        """Upsert a user from authentication info, updating last_login."""
        existing = self._users.get(sub)
        if existing is not None:
            existing.last_login = time.time()
            if email:
                existing.email = email
            if name:
                existing.name = name
            return existing
        record = UserStoreRecord(
            id=sub,
            email=email,
            name=name,
            role="clinician",
            last_login=time.time(),
            status="active",
            provider=provider,
        )
        self._users[sub] = record
        return record

    @property
    def count(self) -> int:
        return len(self._users)


class ActivityLog:
    """In-memory activity log with bounded capacity.

    Stores the most recent *max_entries* (default 1000) API activity
    records.  Older entries are discarded FIFO.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: Deque[ActivityLogEntry] = deque(maxlen=max_entries)
        self._max_entries = max_entries

    def record(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        user: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record an API activity entry."""
        entry = ActivityLogEntry(
            timestamp=time.time(),
            user=user,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        self._entries.append(entry)

    def list_entries(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[List[ActivityLogEntry], int]:
        """Return a paginated slice of log entries (newest first).

        Returns ``(entries, total_count)``.
        """
        all_entries = list(reversed(self._entries))
        total = len(all_entries)
        start = (page - 1) * page_size
        end = start + page_size
        return all_entries[start:end], total

    @property
    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Role-based access control dependency
# ---------------------------------------------------------------------------


def _require_admin(user_store: UserStore) -> Any:
    """Build a FastAPI dependency that enforces admin role."""

    from aortica.api.auth import UserInfo, require_auth

    async def _check_admin(
        request: Request,  # type: ignore[arg-type]
        x_api_key: Optional[str] = Header(default=None),  # type: ignore[assignment]
    ) -> UserInfo:
        # Check if auth is disabled
        enable_auth: bool = getattr(request.app.state, "enable_auth", True)
        if not enable_auth:
            # In no-auth mode, grant admin to all requests
            return UserInfo(sub="anonymous", provider="local")

        user = await require_auth(request, x_api_key=x_api_key)

        # Check role in user store
        stored = user_store.get(user.sub)
        if stored is None:
            # Auto-register on first admin access attempt
            stored = user_store.upsert_from_auth(
                user.sub,
                email=user.email,
                name=user.name,
                provider=user.provider,
            )

        if stored.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required",
            )

        if stored.status == "disabled":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled",
            )

        return user

    return _check_admin


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

# Module-level startup timestamp for uptime calculation
_startup_time = time.time()

# Module-level ECG processing counter
_ecgs_processed = 0


def get_ecgs_processed() -> int:
    """Return the current count of ECGs processed."""
    return _ecgs_processed


def increment_ecgs_processed() -> None:
    """Increment the ECG processing counter."""
    global _ecgs_processed
    _ecgs_processed += 1


def create_admin_router(
    user_store: Optional[UserStore] = None,
    api_key_store: Any = None,
    activity_log: Optional[ActivityLog] = None,
) -> Any:
    """Build a FastAPI APIRouter with admin dashboard endpoints.

    Parameters
    ----------
    user_store:
        User store instance.  Created internally if not provided.
    api_key_store:
        APIKeyStore instance from auth module.  If None, key management
        endpoints return empty results.
    activity_log:
        ActivityLog instance.  Created internally if not provided.

    Returns
    -------
    APIRouter
        Configured router to include in the FastAPI app.
    """
    if user_store is None:
        user_store = UserStore()
    if activity_log is None:
        activity_log = ActivityLog()

    router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

    admin_dep = _require_admin(user_store)

    # ── GET /api/v1/admin/users ──────────────────────────────────

    @router.get(
        "/users",
        response_model=UserListResponse,
        summary="List all users (admin-only)",
    )
    async def list_users(
        _admin: Any = Depends(admin_dep),
    ) -> UserListResponse:
        """List all registered users with their roles and status."""
        records = user_store.list_all()
        users = [
            UserRecord(
                id=r.id,
                email=r.email,
                name=r.name,
                role=r.role,
                last_login=r.last_login,
                status=r.status,
                provider=r.provider,
            )
            for r in records
        ]
        return UserListResponse(users=users, total=len(users))

    # ── PATCH /api/v1/admin/users/:id ────────────────────────────

    @router.patch(
        "/users/{user_id}",
        response_model=UserUpdateResponse,
        summary="Update user role or status (admin-only)",
    )
    async def update_user(
        user_id: str,
        body: UserUpdateRequest,
        _admin: Any = Depends(admin_dep),
    ) -> UserUpdateResponse:
        """Update a user's role or account status."""
        if body.role is not None and body.role not in VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid role '{body.role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}",
            )
        if body.status is not None and body.status not in VALID_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid status '{body.status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
            )

        updated = user_store.update(
            user_id,
            role=body.role,
            status=body.status,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not found",
            )

        return UserUpdateResponse(
            id=updated.id,
            role=updated.role,
            status=updated.status,
            updated=True,
        )

    # ── GET /api/v1/admin/api-keys ───────────────────────────────

    @router.get(
        "/api-keys",
        response_model=APIKeyListResponse,
        summary="List all API keys (admin-only)",
    )
    async def list_api_keys(
        _admin: Any = Depends(admin_dep),
    ) -> APIKeyListResponse:
        """List all issued API keys across all users."""
        if api_key_store is None:
            return APIKeyListResponse(keys=[], total=0)

        # Access internal storage (APIKeyStore._keys)
        all_keys: Dict[str, Any] = getattr(api_key_store, "_keys", {})
        key_records = [
            APIKeyRecord(
                key_id=key_hash,
                name=record.name,
                user_sub=record.user_sub,
                created_at=record.created_at,
                last_used=getattr(record, "last_used", None),
            )
            for key_hash, record in all_keys.items()
        ]
        return APIKeyListResponse(keys=key_records, total=len(key_records))

    # ── DELETE /api/v1/admin/api-keys/:key_id ────────────────────

    @router.delete(
        "/api-keys/{key_id}",
        response_model=APIKeyRevokeResponse,
        summary="Revoke an API key (admin-only)",
    )
    async def revoke_api_key(
        key_id: str,
        _admin: Any = Depends(admin_dep),
    ) -> APIKeyRevokeResponse:
        """Revoke an API key by its hash identifier."""
        if api_key_store is None:
            return APIKeyRevokeResponse(key_id=key_id, revoked=False)

        revoked = api_key_store.revoke(key_id)
        return APIKeyRevokeResponse(key_id=key_id, revoked=revoked)

    # ── GET /api/v1/admin/system-health ──────────────────────────

    @router.get(
        "/system-health",
        response_model=SystemHealthResponse,
        summary="System health metrics (admin-only)",
    )
    async def system_health(
        request: Request,  # type: ignore[arg-type]
        _admin: Any = Depends(admin_dep),
    ) -> SystemHealthResponse:
        """Return current system health metrics."""
        import aortica

        # Model info
        model_loaded: bool = getattr(request.app.state, "model_loaded", False)
        model_version: Optional[str] = None
        if model_loaded:
            model_obj = getattr(request.app.state, "model", None)
            if model_obj is not None:
                model_version = getattr(model_obj, "version", aortica.__version__)
            else:
                model_version = aortica.__version__

        # Database size
        db_size = 0
        for db_name in ["results.db", "feedback.db", "validation.db"]:
            db_path = os.path.join(os.getcwd(), db_name)
            if os.path.exists(db_path):
                db_size += os.path.getsize(db_path)

        # ONNX Runtime check
        onnx_available = False
        try:
            import onnxruntime  # noqa: F401

            onnx_available = True
        except ImportError:
            pass

        # Sync engine status
        sync_status = "inactive"
        sync_engine = getattr(request.app.state, "sync_engine", None)
        if sync_engine is not None:
            sync_status = getattr(sync_engine, "status", "active")

        return SystemHealthResponse(
            status="ok",
            model_version=model_version,
            model_loaded=model_loaded,
            database_size_bytes=db_size,
            total_ecgs_processed=get_ecgs_processed(),
            uptime_seconds=time.time() - _startup_time,
            onnx_runtime_available=onnx_available,
            sync_engine_status=sync_status,
        )

    # ── GET /api/v1/admin/activity-log ───────────────────────────

    @router.get(
        "/activity-log",
        response_model=ActivityLogResponse,
        summary="Activity log (admin-only)",
    )
    async def get_activity_log(
        page: int = 1,
        page_size: int = 50,
        _admin: Any = Depends(admin_dep),
    ) -> ActivityLogResponse:
        """Return paginated activity log entries (newest first)."""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 1
        if page_size > 100:
            page_size = 100

        entries, total = activity_log.list_entries(
            page=page,
            page_size=page_size,
        )
        return ActivityLogResponse(
            entries=entries,
            total=total,
            page=page,
            page_size=page_size,
        )

    return router, user_store, activity_log
