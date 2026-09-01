"""PTB-XL corpus loader — **not present in this repository**.

The module that implemented :func:`load_ptbxl` was never committed: an
unanchored ``data/`` rule in ``.gitignore`` matched ``aortica/data/`` at
depth and silently excluded it, and no copy survives in any branch, tag, or
reflog of this repository.  The v0.3.0 weights in ``artifacts_combined/``
were therefore produced by code that exists only on the machine that ran the
training, and **v0.3.0 is not reproducible from this checkout**.

This module exists so that the failure is loud, immediate, and localised:
``import aortica.data`` succeeds, every unrelated command keeps working, and
any code path that actually needs the corpus raises
:class:`DatasetLoaderUnavailableError` with an explanation instead of an
opaque ``ModuleNotFoundError``.

Restoring it means writing a loader that satisfies the contract documented
on :func:`load_ptbxl` below.  The label side of the pipeline survived — see
:mod:`aortica.data.ptbxl_labels`, whose weights were recovered exactly from
the ``trained_outputs.json`` sidecar — so what is missing is the WFDB record
reader, the fold-based split, and the SCP-code to class-name mapping.
"""

from __future__ import annotations

from typing import Any

_RESTORE_HINT = (
    "The PTB-XL loader (aortica/data/ptbxl.py) was lost before it was ever "
    "committed — see the module docstring for the full story and the "
    "load_ptbxl() docstring for the contract a replacement must satisfy. "
    "Training, benchmarking, and index building cannot run until it is "
    "restored; inference from a released checkpoint is unaffected."
)


class DatasetLoaderUnavailableError(NotImplementedError):
    """Raised when a dataset loader is referenced but absent from the tree."""


def load_ptbxl(
    path: str,
    *,
    sampling_rate: int = 100,
    multitask: bool = False,
    **kwargs: Any,
) -> Any:
    """Load the PTB-XL corpus, split into train/val/test.

    .. warning::
       Not implemented in this repository — always raises
       :class:`DatasetLoaderUnavailableError`.

    A replacement must return a 3-tuple of ``(records, labels)`` pairs in
    train, validation, test order, where ``records`` is a list of
    :class:`aortica.io.ecg_record.ECGRecord` and ``labels`` is a float32
    array of shape ``[n_records, n_outputs]``.  With ``multitask=False`` the
    label width is the rhythm head's; with ``multitask=True`` it is the full
    72-column concatenation in :data:`aortica.models.task_dims.ALL_TASKS`
    order.  PTB-XL's own ``strat_fold`` column defines the splits (folds 1-8
    train, 9 validation, 10 test) — patients do not cross folds, so any
    other split leaks across the evaluation boundary.

    Columns the corpus cannot label must be masked by the caller via
    :func:`aortica.data.ptbxl_labels.per_task_weights`, not zero-filled here.

    Args:
        path: Root of the extracted PTB-XL distribution — the directory
            holding ``ptbxl_database.csv``, ``scp_statements.csv``, and the
            ``records100/`` and ``records500/`` WFDB trees.
        sampling_rate: 100 or 500 Hz, selecting which record tree to read.
        multitask: Emit full 72-column labels rather than rhythm only.

    Raises:
        DatasetLoaderUnavailableError: Always.
    """
    raise DatasetLoaderUnavailableError(_RESTORE_HINT)
