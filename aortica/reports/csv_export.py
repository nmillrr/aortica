"""CSV Batch Analytics Export.

Generates CSV files from batches of multi-task ECG predictions for
research and QA workflows.

Features:

* One row per ECG with columns for every class confidence, risk score,
  quality score, urgency score, and OOD flag
* Human-readable header row
* Streaming write to handle batches up to 10,000 without excessive memory
* Consistent column ordering across exports

Usage::

    from aortica.reports import export_csv
    export_csv(results, "batch_results.csv")
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Fixed metadata columns at the start of each row
_META_COLUMNS: List[str] = [
    "filename",
    "quality_score",
]

# Fixed tail columns
_TAIL_COLUMNS: List[str] = [
    "urgency_score",
    "OOD_flag",
]

# Build human-readable class column names
_RHYTHM_COLUMNS = [f"rhythm_{c}" for c in RHYTHM_CLASSES]
_STRUCTURAL_COLUMNS = [f"structural_{c}" for c in STRUCTURAL_CLASSES]
_ISCHAEMIA_COLUMNS = [f"ischaemia_{c}" for c in ISCHAEMIA_CLASSES]
_RISK_COLUMNS = [f"risk_{c}" for c in RISK_OUTPUTS]


def _all_columns() -> List[str]:
    """Return the full ordered list of CSV column names."""
    return (
        _META_COLUMNS
        + _RHYTHM_COLUMNS
        + _STRUCTURAL_COLUMNS
        + _ISCHAEMIA_COLUMNS
        + _RISK_COLUMNS
        + _TAIL_COLUMNS
    )


# ---------------------------------------------------------------------------
# Prediction extraction helper (reused pattern from pdf_report / jsonld)
# ---------------------------------------------------------------------------


def _extract_predictions(
    multi_task_output: Any,
) -> Dict[str, List[float]]:
    """Extract per-task prediction lists from various output formats.

    Handles MultiTaskOutput dataclass, dict, and tensor/array types.
    """
    import numpy as np

    tasks = {
        "rhythm": RHYTHM_CLASSES,
        "structural": STRUCTURAL_CLASSES,
        "ischaemia": ISCHAEMIA_CLASSES,
        "risk": RISK_OUTPUTS,
    }

    result: Dict[str, List[float]] = {}

    for task_name, class_names in tasks.items():
        # Get raw values
        if hasattr(multi_task_output, task_name):
            values = getattr(multi_task_output, task_name)
        elif isinstance(multi_task_output, dict):
            values = multi_task_output.get(task_name)
        else:
            values = None

        if values is None:
            continue

        # Convert tensor/array to list
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "tolist"):
            values = values.tolist()  # type: ignore[union-attr]
        if isinstance(values, np.ndarray):
            values = values.tolist()

        # Handle batch dimension
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], list):
                values = values[0]  # Take first sample from batch

        # Ensure length matches class count
        n = min(len(values), len(class_names))
        result[task_name] = [float(v) for v in values[:n]]

    return result


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def _build_row(
    multi_task_output: Any,
    filename: str = "",
    quality_score: Optional[float] = None,
    urgency_score: Optional[float] = None,
    ood_flag: Optional[bool] = None,
) -> List[str]:
    """Build a single CSV row from a multi-task output.

    Parameters
    ----------
    multi_task_output:
        A single prediction result (dict, dataclass, etc.).
    filename:
        ECG filename for this result.
    quality_score:
        Overall signal quality score (0–100), or ``None``.
    urgency_score:
        Worklist urgency score (0–100), or ``None``.
    ood_flag:
        Out-of-distribution flag, or ``None``.

    Returns
    -------
    list[str]
        A list of string values for one CSV row.
    """
    predictions = _extract_predictions(multi_task_output)

    row: List[str] = []

    # Metadata columns
    row.append(filename)
    row.append(f"{quality_score:.1f}" if quality_score is not None else "")

    # Classification task columns
    for task_name, class_names in [
        ("rhythm", RHYTHM_CLASSES),
        ("structural", STRUCTURAL_CLASSES),
        ("ischaemia", ISCHAEMIA_CLASSES),
    ]:
        preds = predictions.get(task_name, [])
        for i in range(len(class_names)):
            if i < len(preds):
                row.append(f"{preds[i]:.6f}")
            else:
                row.append("")

    # Risk columns
    risk_preds = predictions.get("risk", [])
    for i in range(len(RISK_OUTPUTS)):
        if i < len(risk_preds):
            row.append(f"{risk_preds[i]:.6f}")
        else:
            row.append("")

    # Tail columns
    row.append(f"{urgency_score:.1f}" if urgency_score is not None else "")
    row.append(str(int(ood_flag)) if ood_flag is not None else "")

    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_csv(
    results: Sequence[Any],
    output_path: Union[str, Path],
    *,
    filenames: Optional[Sequence[str]] = None,
    quality_scores: Optional[Sequence[Optional[float]]] = None,
    urgency_scores: Optional[Sequence[Optional[float]]] = None,
    ood_flags: Optional[Sequence[Optional[bool]]] = None,
) -> Path:
    """Export batch multi-task predictions to a CSV file.

    Parameters
    ----------
    results:
        Sequence of multi-task prediction outputs (``MultiTaskOutput``,
        dicts, or similar).  Each element represents one ECG.
    output_path:
        File path for the output CSV.
    filenames:
        Optional per-result filenames.  Length must match ``results``.
    quality_scores:
        Optional per-result quality scores (0–100).
    urgency_scores:
        Optional per-result urgency scores (0–100).
    ood_flags:
        Optional per-result OOD flags.

    Returns
    -------
    Path
        Absolute path to the generated CSV file.

    Raises
    ------
    ValueError
        If auxiliary sequence lengths don't match ``results``.
    """
    output_path = Path(output_path)
    n = len(results)

    # Validate auxiliary sequences
    if filenames is not None and len(filenames) != n:
        raise ValueError(
            f"filenames length ({len(filenames)}) must match "
            f"results length ({n})"
        )
    if quality_scores is not None and len(quality_scores) != n:
        raise ValueError(
            f"quality_scores length ({len(quality_scores)}) must match "
            f"results length ({n})"
        )
    if urgency_scores is not None and len(urgency_scores) != n:
        raise ValueError(
            f"urgency_scores length ({len(urgency_scores)}) must match "
            f"results length ({n})"
        )
    if ood_flags is not None and len(ood_flags) != n:
        raise ValueError(
            f"ood_flags length ({len(ood_flags)}) must match "
            f"results length ({n})"
        )

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _all_columns()

    # Streaming write — one row at a time to limit memory
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        # Header
        writer.writerow(columns)

        # Data rows
        for i, result in enumerate(results):
            fname = filenames[i] if filenames is not None else f"ecg_{i:05d}"
            qscore = quality_scores[i] if quality_scores is not None else None
            uscore = urgency_scores[i] if urgency_scores is not None else None
            ood = ood_flags[i] if ood_flags is not None else None

            row = _build_row(
                result,
                filename=fname,
                quality_score=qscore,
                urgency_score=uscore,
                ood_flag=ood,
            )
            writer.writerow(row)

    return output_path.resolve()


def export_csv_string(
    results: Sequence[Any],
    *,
    filenames: Optional[Sequence[str]] = None,
    quality_scores: Optional[Sequence[Optional[float]]] = None,
    urgency_scores: Optional[Sequence[Optional[float]]] = None,
    ood_flags: Optional[Sequence[Optional[bool]]] = None,
) -> str:
    """Export batch multi-task predictions to a CSV string.

    Same interface as :func:`export_csv` but returns the CSV content
    as a string instead of writing to a file.  Useful for the API
    endpoint response.

    Returns
    -------
    str
        The CSV content as a string.
    """
    n = len(results)

    # Validate auxiliary sequences
    if filenames is not None and len(filenames) != n:
        raise ValueError(
            f"filenames length ({len(filenames)}) must match "
            f"results length ({n})"
        )
    if quality_scores is not None and len(quality_scores) != n:
        raise ValueError(
            f"quality_scores length ({len(quality_scores)}) must match "
            f"results length ({n})"
        )
    if urgency_scores is not None and len(urgency_scores) != n:
        raise ValueError(
            f"urgency_scores length ({len(urgency_scores)}) must match "
            f"results length ({n})"
        )
    if ood_flags is not None and len(ood_flags) != n:
        raise ValueError(
            f"ood_flags length ({len(ood_flags)}) must match "
            f"results length ({n})"
        )

    columns = _all_columns()

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    writer.writerow(columns)

    # Data rows
    for i, result in enumerate(results):
        fname = filenames[i] if filenames is not None else f"ecg_{i:05d}"
        qscore = quality_scores[i] if quality_scores is not None else None
        uscore = urgency_scores[i] if urgency_scores is not None else None
        ood = ood_flags[i] if ood_flags is not None else None

        row = _build_row(
            result,
            filename=fname,
            quality_score=qscore,
            urgency_score=uscore,
            ood_flag=ood,
        )
        writer.writerow(row)

    return buf.getvalue()
