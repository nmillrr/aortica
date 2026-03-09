"""CSV format reader.

Loads ECG recordings stored as CSV files and returns
:class:`~aortica.io.ecg_record.ECGRecord` instances.

No additional dependencies beyond ``numpy`` and the standard library are
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from aortica.io.ecg_record import ECGRecord


@dataclass
class CSVConfig:
    """Configuration for reading ECG data from a CSV file.

    Attributes:
        sample_rate: Sampling rate in Hz (required).
        lead_columns: Ordered list of column names containing lead data.
            If *None*, all numeric columns are used in file order.
        time_column: Name of the column containing time/sample indices.
            If provided this column is excluded from the signal data.
        units: Amplitude units of the signal data (default ``"µV"``).
        orientation: ``"column_per_lead"`` (default) means each column is
            a lead; ``"row_per_lead"`` means each row is a lead.
        delimiter: Column delimiter (default ``","``).
        skip_rows: Number of header rows to skip before the data starts
            (the first non-skipped row is treated as the column header).
    """

    sample_rate: float
    lead_columns: Optional[list[str]] = None
    time_column: Optional[str] = None
    units: str = "µV"
    orientation: str = "column_per_lead"  # or "row_per_lead"
    delimiter: str = ","
    skip_rows: int = 0
    patient_metadata: Optional[dict[str, object]] = field(default=None)


def read_csv(path: str | Path, config: CSVConfig) -> ECGRecord:
    """Read an ECG recording from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.
    config:
        A :class:`CSVConfig` describing the file layout.

    Returns
    -------
    ECGRecord
        A standardised ECG record.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the configuration is inconsistent with the file contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read CSV using numpy for simplicity — we only need numeric data.
    # First, read the header line to get column names.
    with open(path, newline="") as fh:
        # Skip extra header rows.
        for _ in range(config.skip_rows):
            fh.readline()
        header_line = fh.readline().strip()

    col_names = [c.strip() for c in header_line.split(config.delimiter)]

    # Load the numeric data (skip the header rows + column-name row).
    raw = np.genfromtxt(
        str(path),
        delimiter=config.delimiter,
        skip_header=config.skip_rows + 1,
        dtype=np.float64,
    )

    if raw.ndim == 1:
        # Single column/row — reshape.
        raw = raw.reshape(-1, 1)

    if config.orientation == "row_per_lead":
        # Each row is a lead.  Determine lead names.
        lead_names = config.lead_columns if config.lead_columns else col_names
        # The first column may be non-numeric lead labels if orientation is
        # row_per_lead and there is a time/index column — but in practice
        # the matrix was parsed numerically, so we just use the rows as-is.
        signals = raw.copy()

        # If there is a time_column, it corresponds to the first *column*
        # (which in row-per-lead means each row has an index in its first cell).
        if config.time_column and config.time_column in col_names:
            col_idx = col_names.index(config.time_column)
            signals = np.delete(signals, col_idx, axis=1)

        # lead_names should match the number of rows.
        if config.lead_columns and len(config.lead_columns) != signals.shape[0]:
            raise ValueError(
                f"lead_columns length ({len(config.lead_columns)}) does not match "
                f"number of data rows ({signals.shape[0]})"
            )
        if not config.lead_columns:
            lead_names = [f"Lead_{i}" for i in range(signals.shape[0])]
    else:
        # Default: column_per_lead — each column is a lead.
        # Exclude the time column if specified.
        if config.time_column:
            if config.time_column not in col_names:
                raise ValueError(
                    f"time_column '{config.time_column}' not found in CSV columns: "
                    f"{col_names}"
                )
            time_idx = col_names.index(config.time_column)
            data_col_names = [c for i, c in enumerate(col_names) if i != time_idx]
            data = np.delete(raw, time_idx, axis=1)
        else:
            data_col_names = list(col_names)
            data = raw

        if config.lead_columns:
            # Select only the requested columns in the requested order.
            indices: list[int] = []
            for lc in config.lead_columns:
                if lc not in data_col_names:
                    raise ValueError(
                        f"lead_column '{lc}' not found in CSV columns: {data_col_names}"
                    )
                indices.append(data_col_names.index(lc))
            data = data[:, indices]
            lead_names = list(config.lead_columns)
        else:
            lead_names = data_col_names

        # Transpose to [leads, samples].
        signals = data.T

    return ECGRecord(
        signals=signals.astype(np.float64),
        sample_rate=config.sample_rate,
        lead_names=lead_names,
        source_format="csv",
        units=config.units,
        patient_metadata=config.patient_metadata,
    )
