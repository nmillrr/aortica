"""Assemble several corpora into one training set with per-sample masks.

Two corpora almost never label the same outputs.  PTB-XL supplies ground
truth for 26 of the model's 72 outputs and Chapman for 21; they overlap on
19, and between them cover 28.  Concatenating their label matrices is
therefore not enough — a zero in a PTB-XL row's ``early_repol_vs_STEMI``
column means *this corpus cannot say*, while a zero in a Chapman row's same
column means *no early repolarisation here*.  Training on both as if they
were the same thing teaches the head to rule the finding out on every
PTB-XL record it ever sees.

So each record carries a **mask** alongside its labels: 1.0 for a column its
source corpus can actually speak to, 0.0 otherwise.  :class:`~aortica.data.
dataset.ECGDataset` accepts these directly and yields ``(signal, label,
mask)``, and :func:`~aortica.models.train_multitask.train_multitask` takes
the resulting loaders.

This is the mechanism the v0.3.0 model card calls "per-sample label
masking".  The corpus-level mask in
:func:`aortica.data.ptbxl_labels.per_task_weights` is the degenerate case of
it — correct only while every record comes from the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from aortica.data.label_mapping import load_label_map
from aortica.io.ecg_record import ECGRecord

Split = tuple[list[ECGRecord], NDArray[np.float32], NDArray[np.float32]]


@dataclass(frozen=True)
class CorpusSpec:
    """One corpus to fold into a combined training set.

    Attributes:
        name: Short identifier, recorded on each record as
            ``patient_metadata["source_dataset"]``.
        path: Root of the extracted distribution.
        loader: Callable with the ``load_*(path, sampling_rate=, multitask=,
            limit=)`` signature the corpus loaders share.
        label_map: Name of the mapping file describing what it can label.
    """

    name: str
    path: str | Path
    loader: Callable[..., tuple]
    label_map: str


def ptbxl_chapman(
    ptbxl_path: str | Path, chapman_path: str | Path
) -> list[CorpusSpec]:
    """The two corpora v0.3.0 was trained on.

    Args:
        ptbxl_path: Root of the PTB-XL 1.0.3 distribution.
        chapman_path: Root of the Chapman 1.0.0 distribution.

    Returns:
        Specs suitable for :func:`load_corpora`.
    """
    from aortica.data.chapman import load_chapman
    from aortica.data.ptbxl import load_ptbxl

    return [
        CorpusSpec("ptbxl", ptbxl_path, load_ptbxl, "ptbxl_scp"),
        CorpusSpec("chapman", chapman_path, load_chapman, "chapman_snomed"),
    ]


def corpus_mask(label_map_name: str, *, multitask: bool = True) -> NDArray[np.float32]:
    """Build the column mask for one corpus.

    Args:
        label_map_name: Mapping file name, e.g. ``"ptbxl_scp"``.
        multitask: Mask the full 72-column layout rather than rhythm only.

    Returns:
        Float32 vector, 1.0 where the corpus can supply ground truth.
    """
    from aortica.data.ptbxl import _output_columns

    mapping = load_label_map(label_map_name)
    names, _ = _output_columns(multitask)
    labelable = mapping.labelable_outputs
    return np.array(
        [1.0 if name in labelable else 0.0 for name in names],
        dtype=np.float32,
    )


def load_corpora(
    corpora: Sequence[CorpusSpec],
    *,
    sampling_rate: int = 100,
    multitask: bool = True,
    limit: int | None = None,
) -> tuple[Split, Split, Split]:
    """Load several corpora and concatenate them with per-sample masks.

    Each corpus is split by its own policy — PTB-XL by its published folds,
    Chapman by the seeded split in its mapping file — and the splits are
    then concatenated split-wise, so no record crosses the train/test
    boundary of its own corpus.

    Args:
        corpora: Corpora to combine, e.g. from :func:`ptbxl_chapman`.
        sampling_rate: Target rate in Hz, applied to every corpus.
        multitask: Emit full 72-column labels and masks.
        limit: Per-corpus, per-split record cap.  For smoke tests.

    Returns:
        ``(train, val, test)``, each a ``(records, labels, masks)`` triple.
        ``masks`` has the same shape as ``labels``.

    Raises:
        ValueError: If *corpora* is empty, or two corpora disagree on label
            width (which would mean one was loaded with a different
            ``multitask`` setting).
    """
    if not corpora:
        raise ValueError("load_corpora needs at least one corpus")

    per_split: list[list[tuple[list[ECGRecord], NDArray, NDArray]]] = [
        [],
        [],
        [],
    ]

    for spec in corpora:
        mask = corpus_mask(spec.label_map, multitask=multitask)
        splits = spec.loader(
            spec.path,
            sampling_rate=sampling_rate,
            multitask=multitask,
            limit=limit,
        )
        for position, (records, labels) in enumerate(splits):
            if labels.shape[1] != mask.shape[0]:
                raise ValueError(
                    f"{spec.name} produced {labels.shape[1]} label columns "
                    f"but its mask has {mask.shape[0]} — check multitask="
                )
            masks = np.tile(mask, (labels.shape[0], 1))
            per_split[position].append((records, labels, masks))

    def _concat(parts: list[tuple[list[ECGRecord], NDArray, NDArray]]) -> Split:
        records: list[ECGRecord] = []
        for chunk, _, _ in parts:
            records.extend(chunk)
        width = parts[0][1].shape[1]
        labels = (
            np.concatenate([p[1] for p in parts], axis=0)
            if records
            else np.empty((0, width), dtype=np.float32)
        )
        masks = (
            np.concatenate([p[2] for p in parts], axis=0)
            if records
            else np.empty((0, width), dtype=np.float32)
        )
        return records, labels.astype(np.float32), masks.astype(np.float32)

    return _concat(per_split[0]), _concat(per_split[1]), _concat(per_split[2])


def combined_labelable_outputs(corpora: Sequence[CorpusSpec]) -> frozenset[str]:
    """Outputs at least one of *corpora* can label.

    This is the union the combined training run actually covers — 28 for
    PTB-XL plus Chapman, which is what v0.3.0 shipped.  It is what belongs in
    ``trained_outputs.json``, and it is deliberately *not* what any single
    record's mask looks like.
    """
    names: set[str] = set()
    for spec in corpora:
        names |= set(load_label_map(spec.label_map).labelable_outputs)
    return frozenset(names)
