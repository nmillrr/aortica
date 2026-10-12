"""Shared test fixtures for Aortica."""

import numpy as np
import pytest

from aortica.io import ecg_record


@pytest.fixture
def sample_ecg_record() -> ecg_record.ECGRecord:
    """Create a sample 12-lead ECG record for testing."""
    lead_names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    signals = np.random.randn(12, 5000).astype(np.float64)
    return ecg_record.ECGRecord(
        signals=signals,
        sample_rate=500.0,
        lead_names=lead_names,
    )


def route_paths(app) -> set:  # type: ignore[no-untyped-def]
    """Every path an app serves, independent of how routers were included.

    FastAPI >= 0.13x wraps included routers in objects with no ``.path``, so
    ``[r.path for r in app.routes]`` raises.  The OpenAPI schema lists every
    included route; directly mounted routes (and any hidden from the schema)
    are picked up from ``app.routes``.
    """
    paths = {getattr(r, "path") for r in app.routes if hasattr(r, "path")}
    paths |= set(app.openapi().get("paths", {}))
    return paths
