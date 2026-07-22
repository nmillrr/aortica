"""FastAPI router and middleware for the audit trail (US-121).

* ``GET /api/v1/audit/events`` — filtered audit events (admin-only)
* ``GET /api/v1/audit/verify`` — HMAC-chain integrity check (admin-only)

Plus :func:`AuditMiddleware`, which auto-logs ``prediction_generated``,
``report_generated``, and ``ehr_submitted`` events based on the request
path/method, without per-endpoint instrumentation.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request
    from starlette.middleware.base import BaseHTTPMiddleware

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    BaseHTTPMiddleware = object  # type: ignore[assignment,misc]


# Path/method → auto-logged event type for the middleware.
_AUTO_LOG_RULES = [
    ("POST", "/api/v1/predict", "prediction_generated"),
    ("POST", "/api/v1/report/", "report_generated"),
    ("POST", "/api/v1/ehr/", "ehr_submitted"),
]


class AuditEventModel(BaseModel):
    """A single audit event."""

    id: int
    timestamp: str
    event_type: str
    user_id: Optional[str] = None
    ecg_reference_id: Optional[str] = None
    model_version: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    event_details: dict = Field(default_factory=dict)
    hmac: str


class IntegrityModel(BaseModel):
    """Integrity-check response."""

    valid: bool
    total_rows: int
    broken_links: List[int]
    detail: str


def _match_auto_event(method: str, path: str) -> Optional[str]:
    for m, prefix, event in _AUTO_LOG_RULES:
        if method == m and path.startswith(prefix):
            return event
    return None


class AuditMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Auto-logs audit events for predict/report/EHR endpoints."""

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        try:
            logger = getattr(request.app.state, "audit_logger", None)
            if logger is not None and 200 <= response.status_code < 300:
                event = _match_auto_event(request.method, request.url.path)
                if event is not None:
                    user = getattr(request.state, "user_id", None)
                    client = request.client.host if request.client else None
                    logger.log(
                        event,
                        user_id=user,
                        ip_address=client,
                        event_details={
                            "path": request.url.path,
                            "status_code": response.status_code,
                        },
                    )
        except Exception:  # noqa: BLE001 - auditing must not break requests
            pass
        return response


def create_audit_router(logger: Optional[Any] = None) -> Any:
    """Create the audit API router backed by *logger*."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the audit router. "
            "Install with: pip install aortica[api]"
        )

    from aortica.audit import AuditLogger

    if logger is None:
        import os
        import tempfile

        logger = AuditLogger(os.path.join(tempfile.mkdtemp(), "audit.db"))

    router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

    async def _require_admin(request: Request) -> None:  # type: ignore[valid-type]
        """Enforce admin access when auth is enabled."""
        auth_dep = getattr(request.app.state, "auth_dependency", None)
        if auth_dep is None:
            return
        user = await auth_dep(request)
        # When auth is enabled, require an admin role if the user carries one.
        role = getattr(user, "role", None)
        if user is not None and role is not None and role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

    @router.get("/events", response_model=List[AuditEventModel])
    async def get_events(
        request: Request,  # type: ignore[valid-type]
        event_type: Optional[str] = Query(default=None),
        user: Optional[str] = Query(default=None),
        ecg_reference_id: Optional[str] = Query(default=None),
        date_from: Optional[str] = Query(default=None),
        date_to: Optional[str] = Query(default=None),
        limit: int = Query(default=1000, ge=1, le=10000),
        offset: int = Query(default=0, ge=0),
        _admin: None = Depends(_require_admin),
    ) -> List[AuditEventModel]:
        """Return filtered audit events (admin-only)."""
        events = logger.query(
            event_type=event_type,
            user_id=user,
            ecg_reference_id=ecg_reference_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return [AuditEventModel(**e.to_dict()) for e in events]

    @router.get("/verify", response_model=IntegrityModel)
    async def verify(
        request: Request,  # type: ignore[valid-type]
        _admin: None = Depends(_require_admin),
    ) -> IntegrityModel:
        """Run HMAC-chain integrity verification (admin-only)."""
        report = logger.verify()
        return IntegrityModel(**report.to_dict())

    return router
