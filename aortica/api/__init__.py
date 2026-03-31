"""Aortica REST API — FastAPI application for ECG inference."""

from aortica.api.app import create_app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start the Aortica API server via Uvicorn.

    This is the entry point for the ``aortica-server`` console script.

    Parameters
    ----------
    host:
        Bind address.  Defaults to ``0.0.0.0``.
    port:
        Bind port.  Defaults to ``8000``.
    reload:
        Enable auto-reload for development.
    """
    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        raise ImportError(
            "Uvicorn is required to run the Aortica API server. "
            "Install it with: pip install aortica[api]"
        ) from None

    app = create_app(model_loaded=False)
    uvicorn.run(app, host=host, port=port, reload=reload)


__all__ = ["create_app", "run_server"]
