"""Georgia 12-lead ECG Challenge Database loader (external validation only).

The Georgia cohort from the PhysioNet/CinC Challenge 2021 is the external
test set for the v0.4.0 whitepaper: 10,344 ECGs from Emory University
Hospital, Atlanta, recorded at 500 Hz, 10 seconds, labelled in SNOMED CT.
It shares no patients, devices or annotators with PTB-XL (Germany) or
Chapman (China).

It is never trained or tuned on, so this loader returns a single
evaluation set rather than train/val/test splits.  Records are stored as
WFDB headers pointing at MATLAB v4 ``.mat`` signal files, which
:func:`aortica.io.read_ecg` reads directly; diagnoses live in ``# Dx:``
header comments in the same format as Chapman's, so the Chapman header
parser and label builder are reused with Georgia's own mapping file.

Download (open access, CC BY 4.0)::

    https://physionet.org/content/challenge-2021/1.0.3/training/georgia/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aortica.data.chapman import EXPECTED_LEADS, build_labels, parse_header_metadata
from aortica.data.label_mapping import LabelMap, load_label_map
from aortica.io.ecg_record import ECGRecord

logger = logging.getLogger(__name__)

#: Mapping file backing this loader.
LABEL_MAP_NAME = "georgia_snomed"


class GeorgiaDataNotFoundError(FileNotFoundError):
    """The Georgia distribution is missing at the given path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"No Georgia records (g*/*.hea) under '{path}'.\n"
            "  https://physionet.org/content/challenge-2021/1.0.3/training/georgia/"
        )


def index_records(path: Path) -> list[Path]:
    """Every header in the cohort, sorted for a machine-independent order."""
    return sorted(Path(path).glob("g*/*.hea"))


def load_georgia(
    path: str | Path,
    *,
    sampling_rate: int = 100,
    multitask: bool = True,
    limit: int | None = None,
    label_map: LabelMap | None = None,
    **kwargs: Any,
) -> tuple[list[ECGRecord], NDArray[np.float32], dict[str, list[str]]]:
    """Load the Georgia cohort as one evaluation set.

    Args:
        path: Directory holding the ``g1/`` … ``g11/`` subfolders.
        sampling_rate: Target rate in Hz (records are 500 Hz on disk).
        multitask: Emit full 72-column labels rather than rhythm only.
        limit: Read at most this many records.
        label_map: Override the shipped mapping (mainly for tests).

    Returns:
        ``(records, labels, dropped)`` where *dropped* maps a reason to the
        record names excluded for it.  Nothing is dropped silently.
    """
    root = Path(path)
    headers = index_records(root)
    if not headers:
        raise GeorgiaDataNotFoundError(root)
    if limit is not None:
        headers = headers[:limit]

    from aortica.io.dispatcher import read_ecg

    mapping = label_map or load_label_map(LABEL_MAP_NAME)
    records: list[ECGRecord] = []
    dx_sets: list[tuple[str, ...]] = []
    dropped: dict[str, list[str]] = {"unreadable": [], "non_finite": [], "wrong_leads": []}

    for header in headers:
        meta = parse_header_metadata(header)
        try:
            record = read_ecg(header.with_suffix(""), target_rate=float(sampling_rate))
        except Exception as exc:  # third-party parser on someone else's files
            dropped["unreadable"].append(header.stem)
            logger.debug("georgia: skipping %s — %s: %s", header.stem, type(exc).__name__, exc)
            continue
        if not np.isfinite(record.signals).all():
            dropped["non_finite"].append(header.stem)
            continue
        if record.num_leads != EXPECTED_LEADS:
            dropped["wrong_leads"].append(header.stem)
            continue

        metadata = dict(record.patient_metadata or {})
        metadata.update({
            "source_dataset": "georgia",
            "georgia_record": header.stem,
            "age": meta["age"],
            "sex": meta["sex"],
            "snomed_dx": list(meta["dx"]),
        })
        records.append(ECGRecord(
            signals=record.signals,
            sample_rate=record.sample_rate,
            lead_names=record.lead_names,
            duration_seconds=record.duration_seconds,
            patient_metadata=metadata,
            source_format="georgia",
            units=record.units,
        ))
        dx_sets.append(meta["dx"])

    n_dropped = sum(len(v) for v in dropped.values())
    if n_dropped:
        logger.warning(
            "georgia: dropped %d record(s) — %s. %d kept.",
            n_dropped, {k: len(v) for k, v in dropped.items()}, len(records),
        )

    labels = build_labels(dx_sets, multitask=multitask, label_map=mapping)
    return records, labels, dropped
