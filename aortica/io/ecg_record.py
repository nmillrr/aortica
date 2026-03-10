"""Canonical ECG data representation.

Provides the standardized ``ECGRecord`` dataclass used throughout Aortica.
All format readers produce an ``ECGRecord`` so that downstream processing
sees a single, consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal


@dataclass
class ECGRecord:
    """Standardized in-memory ECG representation.

    Attributes:
        signals: ECG signal data, shape ``[leads, samples]``.
        sample_rate: Sampling rate in Hz (must be positive).
        lead_names: List of lead names (e.g. ``["I", "II", ...]``).
        duration_seconds: Duration of the recording in seconds.
            Computed automatically from ``signals`` and ``sample_rate``
            if not provided (or left at the default ``0.0``).
        patient_metadata: Optional patient metadata dictionary.
        source_format: File format the record was loaded from.
        units: Signal amplitude units, default ``"µV"``.
    """

    signals: NDArray[np.float64]
    sample_rate: float
    lead_names: list[str]
    duration_seconds: float = 0.0
    patient_metadata: Optional[dict[str, object]] = field(default=None)
    source_format: str = "unknown"
    units: str = "µV"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate fields and compute derived values."""
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_leads(self) -> int:
        """Number of leads in the recording."""
        return int(self.signals.shape[0])

    @property
    def num_samples(self) -> int:
        """Number of samples per lead."""
        return int(self.signals.shape[1])

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def resample(self, target_hz: float) -> ECGRecord:
        """Return a new ``ECGRecord`` resampled to *target_hz*.

        Uses :func:`scipy.signal.resample` (Fourier-method) to
        change the sampling rate while preserving signal content.

        Args:
            target_hz: Desired sampling rate in Hz (must be positive).

        Returns:
            A new ``ECGRecord`` with resampled signals.

        Raises:
            ValueError: If *target_hz* is not positive.
        """
        if target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")

        if target_hz == self.sample_rate:
            return ECGRecord(
                signals=self.signals.copy(),
                sample_rate=self.sample_rate,
                lead_names=list(self.lead_names),
                patient_metadata=dict(self.patient_metadata)
                if self.patient_metadata
                else None,
                source_format=self.source_format,
                units=self.units,
            )

        num_target_samples = int(round(self.num_samples * target_hz / self.sample_rate))
        resampled = scipy_signal.resample(self.signals, num_target_samples, axis=1)

        return ECGRecord(
            signals=resampled.astype(np.float64),
            sample_rate=target_hz,
            lead_names=list(self.lead_names),
            patient_metadata=dict(self.patient_metadata)
            if self.patient_metadata
            else None,
            source_format=self.source_format,
            units=self.units,
        )

    def select_leads(self, lead_list: list[str]) -> ECGRecord:
        """Return a new ``ECGRecord`` containing only the requested leads.

        Args:
            lead_list: Ordered list of lead names to keep.

        Returns:
            A new ``ECGRecord`` with the selected leads.

        Raises:
            ValueError: If any requested lead is not present.
        """
        indices: list[int] = []
        for name in lead_list:
            if name not in self.lead_names:
                raise ValueError(
                    f"Lead '{name}' not found. Available leads: {self.lead_names}"
                )
            indices.append(self.lead_names.index(name))

        return ECGRecord(
            signals=self.signals[indices, :].copy(),
            sample_rate=self.sample_rate,
            lead_names=list(lead_list),
            duration_seconds=self.duration_seconds,
            patient_metadata=dict(self.patient_metadata)
            if self.patient_metadata
            else None,
            source_format=self.source_format,
            units=self.units,
        )

    def to_millivolts(self) -> ECGRecord:
        """Return a new ``ECGRecord`` with signals converted to millivolts.

        Supported source units:

        * ``"µV"`` / ``"uV"`` — divided by 1000
        * ``"mV"`` — returned as-is (copy)
        * ``"V"`` — multiplied by 1000

        Returns:
            A new ``ECGRecord`` with ``units="mV"``.

        Raises:
            ValueError: If the current ``units`` value is not recognised.
        """
        unit = self.units.strip()
        if unit in ("mV",):
            scale = 1.0
        elif unit in ("µV", "uV"):
            scale = 1e-3
        elif unit in ("V",):
            scale = 1e3
        else:
            raise ValueError(f"Cannot convert units '{self.units}' to mV")

        return ECGRecord(
            signals=(self.signals * scale).astype(np.float64),
            sample_rate=self.sample_rate,
            lead_names=list(self.lead_names),
            duration_seconds=self.duration_seconds,
            patient_metadata=dict(self.patient_metadata)
            if self.patient_metadata
            else None,
            source_format=self.source_format,
            units="mV",
        )
    