"""Chapman-Shaoxing/Ningbo corpus loader.

Reads the PhysioNet "large scale 12-lead electrocardiogram database for
arrhythmia study" (1.0.0) into :class:`~aortica.io.ecg_record.ECGRecord`
objects with multi-label targets aligned to the model's task heads.

.. rubric:: Reconstruction note

Like the PTB-XL loader, the original was never committed and was lost to an
unanchored ``data/`` rule in ``.gitignore``.  Unlike PTB-XL, it cannot be
recovered exactly, and the difference is worth understanding before trusting
anything computed here.

PTB-XL publishes stratification folds, so its test split is a known set of
records and each published ``test_pos`` is an exact equation to solve.
Chapman publishes no split at all.  Its ``test_pos`` counts therefore depend
on a seed and a strategy recorded nowhere, and thousands of plausible random
splits land within sampling noise of the published numbers.

What *was* recovered is the mapping, from prevalence ratios and — decisively
— from what the published output set omits.  Seven outputs PTB-XL labels are
absent from Chapman in v0.3.0, and all seven are explained by the lost
loader having mapped through the dataset's shipped
``ConditionNames_SNOMED-CT.csv`` vocabulary alone, ignoring the 43 further
SNOMED codes that appear in the record headers.  The evidence is laid out in
``label_maps/chapman_snomed.yaml``.

The consequence for callers: metrics computed on this loader's test split
are comparable to the published ``chapman/*`` numbers *in distribution*, and
never record-for-record.  :data:`SPLIT_IS_RECONSTRUCTED` says so in code, and
:func:`load_chapman` will not silently pretend otherwise.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from aortica.data.label_mapping import LabelMap, load_label_map
from aortica.io.ecg_record import ECGRecord

#: Mapping file backing this loader.
LABEL_MAP_NAME = "chapman_snomed"

#: The split this loader produces is **not** the one v0.3.0 used; that split
#: is unrecoverable.  Anything comparing against published ``chapman/*``
#: metrics must account for it.
SPLIT_IS_RECONSTRUCTED = True

_DOWNLOAD_HINT = (
    "Chapman-Shaoxing/Ningbo is a free, open-access dataset — no "
    "credentialing required.\n"
    "  https://physionet.org/content/ecg-arrhythmia/1.0.0/\n"
    "Point --data-path at the directory holding ConditionNames_SNOMED-CT.csv "
    "and the WFDBRecords/ tree."
)


class ChapmanDataNotFoundError(FileNotFoundError):
    """The Chapman distribution is missing or incomplete at the given path."""

    def __init__(self, path: Path, missing: str) -> None:
        self.path = path
        self.missing = missing
        super().__init__(
            f"Chapman-Shaoxing is not present at '{path}' — missing "
            f"{missing}.\n{_DOWNLOAD_HINT}"
        )


# ----------------------------------------------------------------------
# Header parsing
# ----------------------------------------------------------------------


def parse_header_metadata(header_path: Path) -> dict[str, Any]:
    """Read the ``#Age:`` / ``#Sex:`` / ``#Dx:`` comment lines of a header.

    Chapman carries its diagnostic labels in WFDB header comments rather
    than in a database table, so the label for a record lives next to its
    waveform instead of in a central CSV.

    Args:
        header_path: Path to the ``.hea`` file.

    Returns:
        ``{"age": float|None, "sex": str, "dx": tuple[str, ...]}``.  A record
        whose header carries no ``#Dx:`` line yields an empty ``dx``.
    """
    age: float | None = None
    sex = "unknown"
    dx: tuple[str, ...] = ()

    with open(header_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            key, _, value = line[1:].partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "age":
                try:
                    age = float(value)
                except ValueError:
                    age = None
            elif key == "sex":
                sex = {"male": "male", "female": "female"}.get(
                    value.strip().lower(), "unknown"
                )
            elif key == "dx":
                dx = tuple(c.strip() for c in value.split(",") if c.strip())

    return {"age": age, "sex": sex, "dx": dx}


def load_condition_names(path: Path) -> dict[str, list[str]]:
    """Read ``ConditionNames_SNOMED-CT.csv`` into ``{code: [acronyms]}``.

    This file is the dataset's own vocabulary, and the evidence says it is
    exactly the set of codes the v0.3.0 mapping drew on — the record headers
    carry 43 further SNOMED codes that it does not cover.

    Args:
        path: Path to the CSV.

    Returns:
        Mapping of SNOMED concept id to the dataset's short names for it.
    """
    vocabulary: dict[str, list[str]] = {}
    # The shipped file has a UTF-8 BOM; utf-8-sig strips it so the first
    # column name does not come back as "﻿Acronym Name".
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("Snomed_CT") or "").strip()
            acronym = (row.get("Acronym Name") or "").strip()
            if code:
                vocabulary.setdefault(code, []).append(acronym)
    return vocabulary


# ----------------------------------------------------------------------
# Label construction
# ----------------------------------------------------------------------


def build_labels(
    dx_sets: Sequence[Iterable[str]],
    *,
    multitask: bool = False,
    label_map: LabelMap | None = None,
) -> NDArray[np.float32]:
    """Build the multi-label target matrix for a sequence of records.

    Outputs Chapman cannot label are left at zero and must be masked out of
    the loss by the caller — see
    :func:`aortica.data.ptbxl_labels.per_task_weights`.

    Args:
        dx_sets: One iterable of SNOMED codes per record.
        multitask: Emit the full 72-column matrix rather than rhythm only.
        label_map: Override the shipped mapping (mainly for tests).

    Returns:
        Float32 array of shape ``[n_records, n_outputs]``.
    """
    from aortica.data.ptbxl import _output_columns

    mapping = label_map or load_label_map(LABEL_MAP_NAME)
    names, index = _output_columns(multitask)
    labels = np.zeros((len(dx_sets), len(names)), dtype=np.float32)

    code_index = mapping.code_to_classes
    for row, codes in enumerate(dx_sets):
        for code in codes:
            for output in code_index.get(code, ()):
                column = index.get(output)
                if column is not None:
                    labels[row, column] = 1.0
    return labels


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------


def split_indices(
    n_records: int,
    policy: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, NDArray[np.int64]]:
    """Derive train/val/test indices under a mapping file's split policy.

    Chapman ships no stratification, so this is a reconstruction rather than
    a reproduction.  Each record is a distinct patient, so a record-level
    split does not leak a patient across the evaluation boundary the way it
    would in PTB-XL.

    Args:
        n_records: Number of records to split.
        policy: The mapping file's ``split_policy`` block.
        seed: Override the policy's seed.

    Returns:
        ``{"train": idx, "val": idx, "test": idx}``.

    Raises:
        ValueError: If the policy names an unsupported ``kind``, or its
            fractions do not sum to 1.
    """
    kind = policy.get("kind", "random")
    if kind != "random":
        raise ValueError(
            f"Unsupported split kind '{kind}'; only 'random' is implemented."
        )

    fractions = policy.get("fractions") or {}
    train_f = float(fractions.get("train", 0.8))
    val_f = float(fractions.get("val", 0.1))
    test_f = float(fractions.get("test", 0.1))
    total = train_f + val_f + test_f
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split fractions must sum to 1.0, got {total:.6f}"
        )

    rng = np.random.RandomState(
        int(policy.get("seed", 42)) if seed is None else seed
    )
    order = rng.permutation(n_records)
    first = int(n_records * train_f)
    second = int(n_records * (train_f + val_f))
    return {
        "train": order[:first],
        "val": order[first:second],
        "test": order[second:],
    }


# ----------------------------------------------------------------------
# Corpus loading
# ----------------------------------------------------------------------


def _validate_chapman_path(path: Path) -> None:
    if not path.exists():
        raise ChapmanDataNotFoundError(path, "the directory itself")
    if not (path / "WFDBRecords").is_dir():
        raise ChapmanDataNotFoundError(path, "WFDBRecords/")
    if not (path / "ConditionNames_SNOMED-CT.csv").exists():
        raise ChapmanDataNotFoundError(path, "ConditionNames_SNOMED-CT.csv")


def index_records(path: Path) -> list[Path]:
    """List every WFDB header in the corpus, in a stable order.

    Sorted so that a split derived from a seed is reproducible across
    machines — filesystem iteration order is not.
    """
    return sorted((path / "WFDBRecords").glob("*/*/*.hea"))


def load_chapman(
    path: str | Path,
    *,
    sampling_rate: int = 500,
    multitask: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    label_map: LabelMap | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[list[ECGRecord], NDArray[np.float32]],
    tuple[list[ECGRecord], NDArray[np.float32]],
    tuple[list[ECGRecord], NDArray[np.float32]],
]:
    """Load the Chapman corpus, split into train/val/test.

    .. warning::
       The split is **reconstructed**, not the one v0.3.0 used — see the
       module docstring and :data:`SPLIT_IS_RECONSTRUCTED`.  Metrics from
       this loader's test split are comparable to the published
       ``chapman/*`` numbers in distribution only.

    Columns the corpus cannot label are zero-filled here and must be masked
    by the caller via :func:`aortica.data.ptbxl_labels.per_task_weights`.

    Args:
        path: Root of the extracted distribution — the directory holding
            ``ConditionNames_SNOMED-CT.csv`` and ``WFDBRecords/``.
        sampling_rate: Target rate in Hz.  The corpus is recorded at 500 Hz;
            anything lower resamples on read (v0.3.0 trained at 100 Hz).
        multitask: Emit full 72-column labels rather than rhythm only.
        seed: Override the mapping file's split seed.
        limit: Read at most this many records per split.  For smoke tests —
            a full load is 45,152 records.
        label_map: Override the shipped mapping (mainly for tests).

    Returns:
        ``((train_records, train_labels), (val_records, val_labels),
        (test_records, test_labels))``.  Labels are float32 of shape
        ``[n_records, 28]`` with ``multitask=False`` and ``[n_records, 72]``
        with ``multitask=True``.

    Raises:
        ChapmanDataNotFoundError: If the distribution is missing or
            incomplete.
        ValueError: If *sampling_rate* is not positive.
    """
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

    root = Path(path)
    _validate_chapman_path(root)

    mapping = label_map or load_label_map(LABEL_MAP_NAME)
    headers = index_records(root)
    if not headers:
        raise ChapmanDataNotFoundError(root, "any .hea records under WFDBRecords/")

    splits = split_indices(len(headers), mapping.split_policy, seed=seed)

    def _build(split: str) -> tuple[list[ECGRecord], NDArray[np.float32]]:
        from aortica.io.dispatcher import read_ecg

        chosen = splits[split]
        if limit is not None:
            chosen = chosen[:limit]

        records: list[ECGRecord] = []
        dx_sets: list[tuple[str, ...]] = []

        for position in chosen:
            header = headers[int(position)]
            meta = parse_header_metadata(header)
            try:
                record = read_ecg(
                    header.with_suffix(""), target_rate=float(sampling_rate)
                )
            except (FileNotFoundError, ValueError):
                # A partially-downloaded corpus should not take the run down;
                # skipping here keeps records and labels aligned.
                continue

            metadata = dict(record.patient_metadata or {})
            metadata.update(
                {
                    "source_dataset": "chapman",
                    "chapman_record": header.stem,
                    "age": meta["age"],
                    "sex": meta["sex"],
                    "snomed_dx": list(meta["dx"]),
                }
            )
            records.append(
                ECGRecord(
                    signals=record.signals,
                    sample_rate=record.sample_rate,
                    lead_names=record.lead_names,
                    duration_seconds=record.duration_seconds,
                    patient_metadata=metadata,
                    source_format="chapman",
                    units=record.units,
                )
            )
            dx_sets.append(meta["dx"])

        labels = build_labels(dx_sets, multitask=multitask, label_map=mapping)
        return records, labels

    return _build("train"), _build("val"), _build("test")
