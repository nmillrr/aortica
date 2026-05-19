"""SMART on FHIR launch context support for Aortica (US-085).

Implements the `SMART App Launch Framework`_ (OAuth 2.0 with EHR launch
sequence) so that the Aortica web UI can be embedded inside an EHR and
receive patient/encounter context automatically.

Configuration is via environment variables:
  - ``SMART_CLIENT_ID``
  - ``SMART_REDIRECT_URI``
  - ``SMART_FHIR_SERVER_URL``

Or via :class:`SMARTConfig` passed to :func:`create_smart_router`.

.. _SMART App Launch Framework:
   https://hl7.org/fhir/smart-app-launch/

"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request, status
    from fastapi.responses import RedirectResponse

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SMARTConfig:
    """SMART on FHIR application configuration.

    All values can be overridden via environment variables when the
    dataclass is constructed with :func:`SMARTConfig.from_env`.
    """

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    fhir_server_url: str = ""
    scopes: str = "launch openid fhirUser patient/*.read"

    @classmethod
    def from_env(cls) -> "SMARTConfig":
        """Build config from environment variables."""
        return cls(
            client_id=os.environ.get("SMART_CLIENT_ID", ""),
            client_secret=os.environ.get("SMART_CLIENT_SECRET", ""),
            redirect_uri=os.environ.get("SMART_REDIRECT_URI", ""),
            fhir_server_url=os.environ.get("SMART_FHIR_SERVER_URL", ""),
            scopes=os.environ.get(
                "SMART_SCOPES", "launch openid fhirUser patient/*.read"
            ),
        )

    @property
    def is_configured(self) -> bool:
        """Return ``True`` if minimum required fields are set."""
        return bool(self.client_id and self.redirect_uri)


# ---------------------------------------------------------------------------
# Pydantic response / data models
# ---------------------------------------------------------------------------


class SMARTLaunchContext(BaseModel):
    """Parsed SMART launch context returned after OAuth callback."""

    patient: Optional[str] = Field(
        default=None,
        description="FHIR Patient resource ID from launch context",
    )
    encounter: Optional[str] = Field(
        default=None,
        description="FHIR Encounter resource ID from launch context",
    )
    fhir_server: Optional[str] = Field(
        default=None,
        description="FHIR server base URL from launch context",
    )
    access_token: str = Field(
        ...,
        description="Bearer token for FHIR server access",
    )
    token_type: str = Field(
        default="Bearer",
        description="Token type (always Bearer)",
    )
    expires_in: Optional[int] = Field(
        default=None,
        description="Token lifetime in seconds",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Granted scopes",
    )
    id_token: Optional[str] = Field(
        default=None,
        description="OpenID Connect id_token if openid scope granted",
    )
    refresh_token: Optional[str] = Field(
        default=None,
        description="Refresh token if offline_access scope granted",
    )


class SMARTMetadata(BaseModel):
    """Subset of FHIR server SMART configuration metadata."""

    authorization_endpoint: str = Field(
        ..., description="OAuth2 authorization endpoint"
    )
    token_endpoint: str = Field(
        ..., description="OAuth2 token endpoint"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="SMART capabilities advertised by the server",
    )


class SMARTStatusResponse(BaseModel):
    """Status of the current SMART on FHIR session."""

    configured: bool = Field(
        ..., description="Whether SMART on FHIR is configured"
    )
    active_session: bool = Field(
        default=False, description="Whether an active launch context exists"
    )
    patient: Optional[str] = Field(
        default=None, description="Current patient reference"
    )
    encounter: Optional[str] = Field(
        default=None, description="Current encounter reference"
    )
    fhir_server: Optional[str] = Field(
        default=None, description="FHIR server URL"
    )


# ---------------------------------------------------------------------------
# SMART metadata discovery
# ---------------------------------------------------------------------------


def discover_smart_metadata(fhir_server_url: str) -> SMARTMetadata:
    """Fetch SMART configuration from a FHIR server's well-known endpoint.

    Tries ``/.well-known/smart-configuration`` first, then falls back
    to the ``/metadata`` CapabilityStatement ``rest.security`` extension.

    Raises :class:`ValueError` if discovery fails.
    """
    if not HAS_HTTPX:
        raise ImportError(
            "httpx is required for SMART on FHIR support. "
            "Install with: pip install httpx"
        )

    base = fhir_server_url.rstrip("/")

    # 1. Try .well-known/smart-configuration
    try:
        resp = httpx.get(
            f"{base}/.well-known/smart-configuration",
            headers={"Accept": "application/json"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return SMARTMetadata(
                authorization_endpoint=data["authorization_endpoint"],
                token_endpoint=data["token_endpoint"],
                capabilities=data.get("capabilities", []),
            )
    except Exception:
        pass

    # 2. Fallback — CapabilityStatement /metadata
    try:
        resp = httpx.get(
            f"{base}/metadata",
            headers={"Accept": "application/fhir+json"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return _parse_capability_statement(data)
    except Exception:
        pass

    raise ValueError(
        f"Unable to discover SMART metadata from {fhir_server_url}"
    )


def _parse_capability_statement(data: Dict[str, Any]) -> SMARTMetadata:
    """Extract OAuth endpoints from a FHIR CapabilityStatement."""
    auth_url = ""
    token_url = ""

    for rest in data.get("rest", []):
        security = rest.get("security", {})
        for ext in security.get("extension", []):
            if ext.get("url") == (
                "http://fhir-registry.smarthealthit.org/"
                "StructureDefinition/oauth-uris"
            ):
                for sub_ext in ext.get("extension", []):
                    if sub_ext.get("url") == "authorize":
                        auth_url = sub_ext.get("valueUri", "")
                    elif sub_ext.get("url") == "token":
                        token_url = sub_ext.get("valueUri", "")

    if not auth_url or not token_url:
        raise ValueError(
            "CapabilityStatement does not contain SMART OAuth endpoints"
        )

    return SMARTMetadata(
        authorization_endpoint=auth_url,
        token_endpoint=token_url,
    )


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------


def exchange_code_for_token(
    code: str,
    token_endpoint: str,
    client_id: str,
    redirect_uri: str,
    client_secret: str = "",
) -> SMARTLaunchContext:
    """Exchange an OAuth2 authorization code for a SMART launch context.

    Performs the ``authorization_code`` grant token exchange per the SMART
    App Launch specification.

    Returns a :class:`SMARTLaunchContext` populated from the token
    response, including ``patient``, ``encounter``, and ``fhirServer``
    context parameters when present.
    """
    if not HAS_HTTPX:
        raise ImportError(
            "httpx is required for SMART on FHIR support. "
            "Install with: pip install httpx"
        )

    payload: Dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
    }
    if client_secret:
        payload["client_secret"] = client_secret

    resp = httpx.post(
        token_endpoint,
        data=payload,
        headers={"Accept": "application/json"},
        timeout=15.0,
    )

    if resp.status_code != 200:
        raise ValueError(
            f"Token exchange failed with status {resp.status_code}: "
            f"{resp.text}"
        )

    data = resp.json()
    return _parse_token_response(data)


def _parse_token_response(data: Dict[str, Any]) -> SMARTLaunchContext:
    """Build a :class:`SMARTLaunchContext` from a token response dict."""
    return SMARTLaunchContext(
        access_token=data["access_token"],
        token_type=data.get("token_type", "Bearer"),
        expires_in=data.get("expires_in"),
        scope=data.get("scope"),
        patient=data.get("patient"),
        encounter=data.get("encounter"),
        fhir_server=data.get("fhirServer") or data.get("fhir_server"),
        id_token=data.get("id_token"),
        refresh_token=data.get("refresh_token"),
    )


# ---------------------------------------------------------------------------
# Launch context store (in-memory, per-session)
# ---------------------------------------------------------------------------


@dataclass
class _LaunchState:
    """Internal state for an in-progress SMART launch."""

    state_param: str
    fhir_server_url: str
    authorization_endpoint: str
    token_endpoint: str


class SMARTSessionStore:
    """Thread-safe in-memory store for SMART launch sessions.

    Tracks pending launches (keyed by ``state`` param) and the active
    launch context after successful callback.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, _LaunchState] = {}
        self._active_context: Optional[SMARTLaunchContext] = None

    def create_launch(
        self,
        fhir_server_url: str,
        authorization_endpoint: str,
        token_endpoint: str,
    ) -> str:
        """Register a pending launch and return the ``state`` parameter."""
        state = secrets.token_urlsafe(32)
        self._pending[state] = _LaunchState(
            state_param=state,
            fhir_server_url=fhir_server_url,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
        )
        return state

    def get_pending(self, state: str) -> Optional[_LaunchState]:
        """Retrieve and remove a pending launch state."""
        return self._pending.pop(state, None)

    def set_active_context(self, ctx: SMARTLaunchContext) -> None:
        """Store the active SMART launch context."""
        self._active_context = ctx

    def get_active_context(self) -> Optional[SMARTLaunchContext]:
        """Return the current active context, or ``None``."""
        return self._active_context

    def clear(self) -> None:
        """Reset all state."""
        self._pending.clear()
        self._active_context = None

    @property
    def has_active_session(self) -> bool:
        return self._active_context is not None

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# FastAPI router factory
# ---------------------------------------------------------------------------


def create_smart_router(
    config: Optional[SMARTConfig] = None,
    session_store: Optional[SMARTSessionStore] = None,
) -> Any:
    """Build a FastAPI :class:`APIRouter` with SMART on FHIR endpoints.

    Endpoints
    ---------
    - ``GET /api/v1/smart/launch`` — initiate EHR launch (redirect)
    - ``GET /api/v1/smart/callback`` — OAuth callback (token exchange)
    - ``GET /api/v1/smart/status`` — current session status
    - ``GET /api/v1/smart/context`` — full active launch context

    Parameters
    ----------
    config:
        SMART configuration.  Defaults to :meth:`SMARTConfig.from_env`.
    session_store:
        Session store.  A new one is created if not provided.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the Aortica API. "
            "Install it with: pip install aortica[api]"
        )

    if config is None:
        config = SMARTConfig.from_env()
    if session_store is None:
        session_store = SMARTSessionStore()

    router = APIRouter(prefix="/api/v1/smart", tags=["smart-on-fhir"])

    # ── GET /api/v1/smart/launch ─────────────────────────────────────

    @router.get(
        "/launch",
        summary="Initiate SMART on FHIR EHR launch",
    )
    async def smart_launch(
        launch: Optional[str] = Query(
            default=None,
            description="EHR-provided launch context token",
        ),
        iss: Optional[str] = Query(
            default=None,
            description="FHIR server base URL (issuer)",
        ),
    ) -> Any:
        """Handle the SMART EHR launch redirect.

        The EHR sends ``?launch=<token>&iss=<fhir_server>`` as query
        parameters.  This endpoint discovers the server's OAuth
        endpoints and redirects the user to the authorization endpoint.
        """
        if not config.is_configured:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SMART on FHIR is not configured",
            )

        fhir_server = iss or config.fhir_server_url
        if not fhir_server:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No FHIR server URL provided. "
                    "Pass ?iss= or set SMART_FHIR_SERVER_URL"
                ),
            )

        # Discover OAuth endpoints
        try:
            metadata = discover_smart_metadata(fhir_server)
        except (ValueError, ImportError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"SMART metadata discovery failed: {exc}",
            )

        # Create pending launch state
        state = session_store.create_launch(
            fhir_server_url=fhir_server,
            authorization_endpoint=metadata.authorization_endpoint,
            token_endpoint=metadata.token_endpoint,
        )

        # Build authorization URL
        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": config.scopes,
            "state": state,
            "aud": fhir_server,
        }
        if launch:
            params["launch"] = launch

        query = "&".join(f"{k}={v}" for k, v in params.items())
        auth_url = f"{metadata.authorization_endpoint}?{query}"

        return RedirectResponse(url=auth_url, status_code=302)

    # ── GET /api/v1/smart/callback ───────────────────────────────────

    @router.get(
        "/callback",
        response_model=SMARTLaunchContext,
        summary="SMART on FHIR OAuth callback",
    )
    async def smart_callback(
        code: str = Query(
            ..., description="Authorization code from EHR"
        ),
        state: str = Query(
            ..., description="State parameter for CSRF verification"
        ),
    ) -> SMARTLaunchContext:
        """Handle the OAuth callback from the EHR authorization server.

        Exchanges the authorization code for an access token and extracts
        the SMART launch context (patient, encounter, fhirServer).
        """
        # Validate state
        launch_state = session_store.get_pending(state)
        if launch_state is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state parameter",
            )

        # Exchange code for token
        try:
            ctx = exchange_code_for_token(
                code=code,
                token_endpoint=launch_state.token_endpoint,
                client_id=config.client_id,
                redirect_uri=config.redirect_uri,
                client_secret=config.client_secret,
            )
        except (ValueError, ImportError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token exchange failed: {exc}",
            )

        # Populate fhir_server if not returned in token response
        if not ctx.fhir_server:
            ctx.fhir_server = launch_state.fhir_server_url

        # Store active context
        session_store.set_active_context(ctx)

        return ctx

    # ── GET /api/v1/smart/status ─────────────────────────────────────

    @router.get(
        "/status",
        response_model=SMARTStatusResponse,
        summary="SMART on FHIR session status",
    )
    async def smart_status() -> SMARTStatusResponse:
        """Return the current SMART on FHIR session status."""
        ctx = session_store.get_active_context()
        if ctx is not None:
            return SMARTStatusResponse(
                configured=config.is_configured,
                active_session=True,
                patient=ctx.patient,
                encounter=ctx.encounter,
                fhir_server=ctx.fhir_server,
            )
        return SMARTStatusResponse(
            configured=config.is_configured,
            active_session=False,
        )

    # ── GET /api/v1/smart/context ────────────────────────────────────

    @router.get(
        "/context",
        response_model=SMARTLaunchContext,
        summary="Active SMART launch context",
    )
    async def smart_context() -> SMARTLaunchContext:
        """Return the full active SMART launch context.

        Returns ``404`` if no active launch context exists.
        """
        ctx = session_store.get_active_context()
        if ctx is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active SMART launch context",
            )
        return ctx

    return router
