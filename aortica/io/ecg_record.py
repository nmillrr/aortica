"""Canonical ECG data representation.

Placeholder module — full implementation in US-003.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ECGRecord:
    """Standardized in-memory ECG representation.

    Attributes:
        signals: ECG signal data, shape [leads, samples].
        sample_rate: Sampling rate in Hz.
        lead_names: List of lead names (e.g. ["I", "II", ...]).
        duration_seconds: Duration of the recording in seconds.
        patient_metadata: Optional patient metadata dictionary.
        source_format: File format the record was loaded from.
        units: Signal units, default "µV".
    """

    signals: NDArray[np.float64]
    sample_rate: float
    lead_names: list[str]
    duration_seconds: float = 0.0
    patient_metadata: Optional[dict[str, object]] = field(default=None)
    source_format: str = "unknown"
    units: str = "µV"

    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.signals.ndim != 2:
            raise ValueError(
                f"signals must be 2D [leads, samples], got shape {self.signals.shape}"
            )
        if len(self.lead_names) != self.signals.shape[0]:
            raise ValueError(
                f"lead_names length ({len(self.lead_names)}) must match "
                f"signals lead dimension ({self.signals.shape[0]})"
            )
        if self.duration_seconds == 0.0:
            self.duration_seconds = self.signals.shape[1] / self.sample_rate

    @property
    def num_leads(self) -> int:
        """Number of leads in the recording."""
        return self.signals.shape[0]

    @property
    def num_samples(self) -> int:
        """Number of samples per lead."""
        return self.signals.shape[1]
