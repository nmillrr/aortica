#!/usr/bin/env python3
"""Score the Ribeiro et al. (2020) ECG model on Aortica's evaluation cohorts.

Baseline for the v0.4.0 whitepaper: the published 12-lead ResNet from
*Automatic diagnosis of the 12-lead ECG using a deep neural network*
(Nat Commun 11, 1760), trained on 2.3 M exams from the CODE study (Brazil),
predicting 1dAVb, RBBB, LBBB, SB, AF and ST.  Weights: Zenodo record
3765717 (``model/model.hdf5``).

Runs in a TensorFlow environment (it needs ``tf_keras`` to load the Keras 2
HDF5 file), separate from Aortica's PyTorch one::

    TF_USE_LEGACY_KERAS=1 .venv-tf/bin/python scripts/baseline_ribeiro.py \\
        --model data/ribeiro/model/model.hdf5 \\
        --ptbxl data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
        --georgia data/georgia --out releases/v0.4.0/_work/ribeiro

Preprocessing choice
--------------------
The published instructions conflict.  The Zenodo README orders the limb
leads ``AVL, AVF, AVR`` while the GitHub README says ``AVR, AVL, AVF``, and
both say signals are "at the scale 1e-4 V" yet also "if the signal is in V
multiply by 1000" (i.e. mV).  Zero-padding position is not stated.

Rather than guess, every combination is scored on **PTB-XL fold 9 (the
validation fold)** and the best one is used for the test cohorts.  No test
data informs the choice.  The selection table is written alongside the
predictions so the choice is auditable.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Ribeiro output order.
RIBEIRO_CLASSES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]

#: PTB-XL SCP statements matching each Ribeiro class — complete blocks only,
#: as Ribeiro defines them.  Used for preprocessing selection on fold 9.
PTBXL_CODES = {
    "1dAVb": {"1AVB"}, "RBBB": {"CRBBB"}, "LBBB": {"CLBBB"},
    "SB": {"SBRAD"}, "AF": {"AFIB"}, "ST": {"STACH"},
}

STANDARD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ORDERS = {
    "github_aVR_aVL_aVF": STANDARD,
    "zenodo_aVL_aVF_aVR": ["I", "II", "III", "aVL", "aVF", "aVR",
                           "V1", "V2", "V3", "V4", "V5", "V6"],
}
#: Multiplier from µV (Aortica's unit) to the model's input unit.
SCALES = {"mV": 1e-3, "1e-4V": 1e-2}
PADDINGS = ("center", "right")

TARGET_RATE = 400.0
TARGET_LEN = 4096


def _auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    from scipy.stats import rankdata

    t = targets > 0.5
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def to_ribeiro(signals_uv: np.ndarray, fs: float, order: Sequence[str], scale: float,
               padding: str) -> np.ndarray:
    """One record ``[12, n]`` in µV, standard lead order -> ``[4096, 12]``."""
    idx = [STANDARD.index(name) for name in order]
    x = signals_uv[idx].astype(np.float64)
    n_out = int(round(x.shape[1] * TARGET_RATE / fs))
    x = sp_signal.resample(x, n_out, axis=1) if n_out != x.shape[1] else x
    x = x[:, :TARGET_LEN] * scale
    pad = TARGET_LEN - x.shape[1]
    if pad > 0:
        left = pad // 2 if padding == "center" else 0
        x = np.pad(x, ((0, 0), (left, pad - left)))
    return x.T.astype(np.float32)


def predict(model, records: Sequence, order, scale, padding, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(records), batch):
        xb = np.stack([to_ribeiro(r.signals, r.sample_rate, order, scale, padding)
                       for r in records[i:i + batch]])
        out.append(model.predict(xb, verbose=0))
    return np.concatenate(out)


def ptbxl_fold(root: Path, fold: int) -> Tuple[list, np.ndarray, List[str]]:
    from aortica.data.ptbxl import load_ptbxl

    _, _, (records, _) = load_ptbxl(
        root, sampling_rate=500, folds={"train": [], "val": [], "test": [fold]}
    )
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    ids = [int(r.patient_metadata["ptbxl_ecg_id"]) for r in records]
    codes = [set(ast.literal_eval(db.loc[i, "scp_codes"])) for i in ids]
    y = np.array([[bool(c & PTBXL_CODES[k]) for k in RIBEIRO_CLASSES] for c in codes],
                 dtype=np.float32)
    return records, y, [str(i) for i in ids]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ptbxl", type=Path, required=True)
    ap.add_argument("--georgia", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Records per cohort (smoke test)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    import tensorflow as tf
    import tf_keras

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf_keras.models.load_model(str(args.model), compile=False)
    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. Preprocessing selection on the validation fold ────────────────
    val_rec, val_y, _ = ptbxl_fold(args.ptbxl, 9)
    if args.limit:
        val_rec, val_y = val_rec[: args.limit], val_y[: args.limit]
    grid: List[Dict] = []
    for (oname, order), (sname, scale), padding in itertools.product(
        LEAD_ORDERS.items(), SCALES.items(), PADDINGS
    ):
        p = predict(model, val_rec, order, scale, padding)
        aucs = {k: _auroc(p[:, j], val_y[:, j]) for j, k in enumerate(RIBEIRO_CLASSES)}
        mean = float(np.nanmean(list(aucs.values())))
        grid.append({"lead_order": oname, "scale": sname, "padding": padding,
                     "mean_auroc": mean, "per_class_auroc": aucs})
        print(f"fold 9  {oname:20s} {sname:6s} {padding:6s}  mean AUROC {mean:.4f}", flush=True)
    best = max(grid, key=lambda g: g["mean_auroc"])
    order, scale, padding = (LEAD_ORDERS[best["lead_order"]], SCALES[best["scale"]],
                             best["padding"])
    print(f"selected: {best['lead_order']} / {best['scale']} / {best['padding']}", flush=True)

    # ── 2. Test cohorts with the selected preprocessing ──────────────────
    from aortica.data.georgia import load_georgia

    te_rec, _, te_ids = ptbxl_fold(args.ptbxl, 10)
    ga_rec, _, _ = load_georgia(args.georgia, sampling_rate=500, multitask=True,
                                limit=args.limit)
    if args.limit:
        te_rec, te_ids = te_rec[: args.limit], te_ids[: args.limit]
    ga_ids = [r.patient_metadata["georgia_record"] for r in ga_rec]

    np.savez_compressed(
        args.out / "ribeiro_predictions.npz",
        classes=np.array(RIBEIRO_CLASSES),
        ptbxl_ids=np.array(te_ids), ptbxl_probs=predict(model, te_rec, order, scale, padding),
        georgia_ids=np.array(ga_ids), georgia_probs=predict(model, ga_rec, order, scale, padding),
    )
    (args.out / "ribeiro_preprocessing_selection.json").write_text(json.dumps({
        "selected": {k: best[k] for k in ("lead_order", "scale", "padding")},
        "selection_set": "PTB-XL 1.0.3 strat_fold 9 (validation)",
        "selection_metric": "mean AUROC over the six Ribeiro classes",
        "grid": grid,
        "model_file": str(args.model.name),
        "tensorflow": tf.__version__,
    }, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
