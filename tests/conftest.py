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
