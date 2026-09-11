"""PTB-XL corpus loader.

Reads the PTB-XL 1.0.3 distribution into :class:`~aortica.io.ecg_record.
ECGRecord` objects with multi-label targets aligned to the model's task
heads, split on the corpus's own patient-disjoint stratification folds.

.. rubric:: Reconstruction note

The original implementation of this module was never committed — an
unanchored ``data/`` rule in ``.gitignore`` matched ``aortica/data/`` at
depth and excluded it — so the v0.3.0 weights in ``artifacts_combined/``
were produced by code that no longer existed anywhere in the repository.

This is a rewrite, and the part that mattered was not the file I/O but the
SCP-code to output-name mapping, which encodes clinical judgements that
cannot be guessed.  It was not guessed: ``artifacts_combined/
test_metrics.json`` records a positive count on fold 10 for each of the 26
outputs PTB-XL labels, and fold 10 is a fixed set of 2,198 records, so each
class has a known integer that a correct mapping must reproduce.  Candidate
code sets were counted against fold 10 and solved back to the mapping.  All
26 reproduce their published count exactly.

The mapping is data, not code — see ``label_maps/ptbxl_scp.yaml``, which
records the evidence for each entry and, more usefully, the places where the
recovered mapping is clinically questionable.  Read the ``STEMI`` note there
before trusting that output.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aortica.data.label_mapping import LabelMap, load_label_map
from aortica.io.ecg_record import ECGRecord

#: Mapping file backing this loader.
LABEL_MAP_NAME = "ptbxl_scp"

_DOWNLOAD_HINT = (
    "PTB-XL is a free, open-access dataset — no credentialing required.\n"
    "  https://physionet.org/content/ptb-xl/1.0.3/\n"
    "Point --data-path at the directory holding ptbxl_database.csv, "
    "scp_statements.csv, records100/ and records500/."
)


class DatasetLoaderUnavailableError(NotImplementedError):
    """Raised when a dataset loader is referenced but absent from the tree.

    Retained as the shared "this corpus has no loader" signal — the PTB-XL
    loader below no longer raises it, but the error type is part of
    ``aortica.data``'s public surface and existing ``except
    NotImplementedError`` handlers depend on it.
    """


class PTBXLDataNotFoundError(FileNotFoundError):
    """The PTB-XL distribution is missing or incomplete at the given path."""

    def __init__(self, path: Path, missing: str) -> None:
        self.path = path
        self.missing = missing
        super().__init__(
            f"PTB-XL is not present at '{path}' — missing {missing}.\n"
            f"{_DOWNLOAD_HINT}"
        )


# ----------------------------------------------------------------------
# Label construction
# ----------------------------------------------------------------------


def parse_scp_codes(raw: str) -> dict[str, float]:
    """Parse PTB-XL's ``scp_codes`` cell into ``{code: likelihood}``.

    The column holds a Python dict literal, e.g.
    ``"{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}"``.

    Args:
        raw: Cell contents.

    Returns:
        Code-to-likelihood mapping; empty if the cell is blank or malformed.
    """
    if not raw or not isinstance(raw, str):
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def present_codes(
    scp_codes: dict[str, float], label_map: LabelMap
) -> set[str]:
    """Codes counting as asserted for a record, per the map's presence rule.

    A likelihood of ``0.0`` in PTB-XL means *no likelihood was assigned*,
    not *absent* — it is the most common value in the corpus.  Treating it
    as absence discards nearly half the labels and fails to reproduce the
    published class counts, so the shipped rule keeps zero-likelihood
    statements.
    """
    if label_map.include_zero_likelihood:
        return set(scp_codes)
    return {
        code
        for code, likelihood in scp_codes.items()
        if likelihood >= label_map.min_likelihood and likelihood > 0.0
    }


def _output_columns(multitask: bool) -> tuple[list[str], dict[str, int]]:
    """Ordered output names for the label matrix, and their column indices.

    With ``multitask=True`` this is the full 72-column concatenation in
    :data:`~aortica.models.task_dims.ALL_TASKS` order; otherwise it is the
    rhythm head's class list alone.
    """
    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.risk_head import RISK_OUTPUTS
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    if multitask:
        names = [
            *RHYTHM_CLASSES,
            *STRUCTURAL_CLASSES,
            *ISCHAEMIA_CLASSES,
            *RISK_OUTPUTS,
        ]
    else:
        names = list(RHYTHM_CLASSES)
    return names, {name: i for i, name in enumerate(names)}


def build_labels(
    scp_code_sets: Sequence[dict[str, float]],
    *,
    multitask: bool = False,
    label_map: LabelMap | None = None,
) -> NDArray[np.float32]:
    """Build the multi-label target matrix for a sequence of records.

    Outputs PTB-XL cannot label are left at zero.  They are *not* negatives
    and must be masked out of the loss by the caller via
    :func:`aortica.data.ptbxl_labels.per_task_weights` — training them
    against implicit zeros teaches the heads to rule those findings out.

    Args:
        scp_code_sets: One ``{code: likelihood}`` mapping per record.
        multitask: Emit the full 72-column matrix rather than rhythm only.
        label_map: Override the shipped mapping (mainly for tests).

    Returns:
        Float32 array of shape ``[n_records, n_outputs]``.
    """
    mapping = label_map or load_label_map(LABEL_MAP_NAME)
    names, index = _output_columns(multitask)
    labels = np.zeros((len(scp_code_sets), len(names)), dtype=np.float32)

    code_index = mapping.code_to_classes
    for row, codes in enumerate(scp_code_sets):
        for code in present_codes(codes, mapping):
            for output in code_index.get(code, ()):
                column = index.get(output)
                if column is not None:
                    labels[row, column] = 1.0
    return labels


# ----------------------------------------------------------------------
# Corpus loading
# ----------------------------------------------------------------------


def _validate_ptbxl_path(path: Path, sampling_rate: int) -> None:
    if not path.exists():
        raise PTBXLDataNotFoundError(path, "the directory itself")
    if not (path / "ptbxl_database.csv").exists():
        raise PTBXLDataNotFoundError(path, "ptbxl_database.csv")
    record_dir = "records100" if sampling_rate == 100 else "records500"
    if not (path / record_dir).exists():
        raise PTBXLDataNotFoundError(path, f"{record_dir}/")


def _read_record(base_path: Path, target_rate: float) -> ECGRecord:
    """Read one WFDB record through the shared dispatcher."""
    from aortica.io.dispatcher import read_ecg

    return read_ecg(base_path, target_rate=target_rate)


def load_ptbxl(
    path: str | Path,
    *,
    sampling_rate: int = 100,
    multitask: bool = False,
    folds: dict[str, Iterable[int]] | None = None,
    limit: int | None = None,
    label_map: LabelMap | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[list[ECGRecord], NDArray[np.float32]],
    tuple[list[ECGRecord], NDArray[np.float32]],
    tuple[list[ECGRecord], NDArray[np.float32]],
]:
    """Load the PTB-XL corpus, split into train/val/test.

    Splits come from PTB-XL's own ``strat_fold`` column — folds 1-8 train,
    9 validation, 10 test.  Patients recur across records but never cross
    folds, so any other split leaks a patient across the evaluation
    boundary and inflates every number downstream.

    Columns the corpus cannot label are zero-filled here and must be masked
    by the caller via :func:`aortica.data.ptbxl_labels.per_task_weights`.

    Args:
        path: Root of the extracted PTB-XL distribution — the directory
            holding ``ptbxl_database.csv``, ``scp_statements.csv``, and the
            ``records100/`` and ``records500/`` WFDB trees.
        sampling_rate: 100 or 500 Hz, selecting which record tree to read.
        multitask: Emit full 72-column labels rather than rhythm only.
        folds: Override the fold assignment, e.g.
            ``{"train": [1, 2], "val": [3], "test": [4]}``.  Defaults to the
            corpus's published split.  Overriding this invalidates any
            comparison against the released v0.3.0 metrics.
        limit: Read at most this many records per split.  For smoke tests
            and CI — a full 100 Hz load is ~21,800 records and ~2 GB.
        label_map: Override the shipped SCP mapping (mainly for tests).

    Returns:
        ``((train_records, train_labels), (val_records, val_labels),
        (test_records, test_labels))``.  Labels are float32 of shape
        ``[n_records, 28]`` with ``multitask=False`` and ``[n_records, 72]``
        with ``multitask=True``.

    Raises:
        PTBXLDataNotFoundError: If the distribution is missing or incomplete.
        ValueError: If *sampling_rate* is not 100 or 500.
    """
    if sampling_rate not in (100, 500):
        raise ValueError(
            f"PTB-XL ships 100 Hz and 500 Hz record trees; got {sampling_rate}."
        )

    root = Path(path)
    _validate_ptbxl_path(root, sampling_rate)

    mapping = label_map or load_label_map(LABEL_MAP_NAME)
    if folds is not None:
        split_folds = {k: tuple(int(f) for f in v) for k, v in folds.items()}
    elif mapping.splits is not None:
        split_folds = mapping.splits
    else:
        # Only reachable if someone swaps the PTB-XL map for one describing a
        # corpus with no published folds — which this loader cannot split,
        # because it splits on PTB-XL's own strat_fold column.
        raise ValueError(
            f"Label map '{mapping.source_name}' publishes no folds, so it "
            "cannot drive the PTB-XL loader. Pass folds= explicitly."
        )

    database = pd.read_csv(root / "ptbxl_database.csv")
    for column in ("strat_fold", "scp_codes", "filename_lr", "filename_hr"):
        if column not in database.columns:
            raise PTBXLDataNotFoundError(
                root, f"column '{column}' in ptbxl_database.csv"
            )

    filename_column = "filename_lr" if sampling_rate == 100 else "filename_hr"

    def _build(split: str) -> tuple[list[ECGRecord], NDArray[np.float32]]:
        wanted = split_folds.get(split, ())
        rows = database[database["strat_fold"].isin(wanted)]
        if limit is not None:
            rows = rows.head(limit)

        records: list[ECGRecord] = []
        code_sets: list[dict[str, float]] = []

        for row in rows.itertuples(index=False):
            base = root / str(getattr(row, filename_column))
            try:
                record = _read_record(base, float(sampling_rate))
            except (FileNotFoundError, ValueError):
                # A truncated or partially-downloaded distribution should not
                # take the whole run down; the record is skipped and its
                # labels with it, keeping records and labels aligned.
                continue

            metadata = dict(record.patient_metadata or {})
            metadata.update(_record_metadata(row))
            records.append(
                ECGRecord(
                    signals=record.signals,
                    sample_rate=record.sample_rate,
                    lead_names=record.lead_names,
                    duration_seconds=record.duration_seconds,
                    patient_metadata=metadata,
                    source_format="ptbxl",
                    units=record.units,
                )
            )
            code_sets.append(parse_scp_codes(str(getattr(row, "scp_codes"))))

        labels = build_labels(
            code_sets, multitask=multitask, label_map=mapping
        )
        return records, labels

    return _build("train"), _build("val"), _build("test")


def _record_metadata(row: Any) -> dict[str, object]:
    """Per-record metadata carried through for subgroup and equity analysis.

    Age and sex are the fields the equity gate stratifies on, and the ones
    a demographics-conditioned model would consume — the model does not use
    them today, but discarding them at load time is what makes that change
    expensive later.
    """
    metadata: dict[str, object] = {"source_dataset": "ptbxl"}

    for source, key in (
        ("ecg_id", "ptbxl_ecg_id"),
        ("patient_id", "ptbxl_patient_id"),
        ("strat_fold", "ptbxl_strat_fold"),
        ("age", "age"),
        ("sex", "sex"),
        ("height", "height"),
        ("weight", "weight"),
        ("device", "device"),
        ("recording_date", "recording_date"),
        ("report", "report"),
    ):
        value = getattr(row, source, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        metadata[key] = value

    # PTB-XL encodes sex as 0 = male, 1 = female.
    raw_sex = metadata.get("sex")
    if raw_sex is not None:
        try:
            coded = int(float(str(raw_sex)))
        except (TypeError, ValueError):
            metadata["sex"] = "unknown"
        else:
            metadata["sex"] = {0: "male", 1: "female"}.get(coded, "unknown")

    return metadata
