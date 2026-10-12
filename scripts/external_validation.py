#!/usr/bin/env python3
"""External validation on the Georgia cohort (PhysioNet/CinC Challenge 2021).

Scores the v0.4.0 reference model and, for continuity, the shipped v0.3.0
edge models (FP32 and INT8 ONNX) on all 10,344 Georgia ECGs — a US cohort
sharing no patients, devices or annotators with PTB-XL or Chapman.

Nothing is tuned on Georgia.  Each model is evaluated at its own operating
points as released (tuned on its validation split); AUROC and AUPRC are
threshold-free.  Reported:

* per class: n, positives, AUROC (95% bootstrap CI when >= 10 positives),
  AUPRC, and sensitivity / specificity / PPV / F1 at the frozen threshold;
* macro-AUROC over classes with >= 1 and >= 10 positives, with CIs;
* the same macro-AUROC by sex and age band;
* a sensitivity analysis counting incomplete RBBB/LBBB as positive, which
  is how PTB-XL's training labels define those classes.

Usage::

    python scripts/external_validation.py --georgia data/georgia \\
        --reference releases/v0.4.0/aortica_v0.4.0.pt \\
        --out releases/v0.4.0/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("external_validation")

TASKS = ("rhythm", "structural", "ischaemia", "risk")
INCOMPLETE_BBB = {"RBBB": "713426002", "LBBB": "251120003"}
AGE_BANDS = (("<40", 0, 40), ("40-64", 40, 65), (">=65", 65, 200))


def _torch_predictor(path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], Dict[str, float]]:
    import torch

    from aortica.models.checkpoint import load_checkpoint

    model = load_checkpoint(path)

    def run(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(x)).as_dict()
        return np.concatenate([out[t].numpy() for t in TASKS], axis=1)

    return run, dict(model.class_thresholds)


def _onnx_predictor(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path))

    def run(x: np.ndarray) -> np.ndarray:
        return np.concatenate(session.run(None, {"ecg_input": x}), axis=1)

    return run


def _predict_all(run: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                 batch: int = 256) -> np.ndarray:
    return np.concatenate([run(x[i:i + batch]) for i in range(0, len(x), batch)])


def _summarise(probs: np.ndarray, labels: np.ndarray, cols: Sequence[int],
               names: Sequence[str], thresholds: Dict[str, float],
               seed: int) -> Dict[str, Any]:
    from aortica.evaluation.classification import (
        bootstrap_macro_auroc_ci,
        evaluate_classes,
        macro_auroc,
    )

    p, y = probs[:, cols], labels[:, cols]
    res = evaluate_classes(p, y, names, thresholds=thresholds, seed=seed)
    out: Dict[str, Any] = {"per_class": {k: v.to_dict() for k, v in res.items()}}
    for min_pos in (1, 10):
        m, k = macro_auroc(res, min_pos=min_pos)
        keep = [i for i, n in enumerate(names) if res[n].n_pos >= min_pos]
        out[f"macro_auroc_min{min_pos}pos"] = {
            "value": m, "n_classes": k,
            "ci95": bootstrap_macro_auroc_ci(p, y, keep, min_pos=1, seed=seed),
        }
    return out


def _subgroups(probs: np.ndarray, labels: np.ndarray, cols: Sequence[int],
               names: Sequence[str], meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    from aortica.evaluation.classification import evaluate_classes, macro_auroc

    groups: Dict[str, np.ndarray] = {}
    sex = np.array([m.get("sex", "unknown") for m in meta])
    for s in ("male", "female"):
        groups[f"sex={s}"] = sex == s
    age = np.array([m.get("age") if m.get("age") is not None else np.nan for m in meta],
                   dtype=float)
    for label, lo, hi in AGE_BANDS:
        groups[f"age {label}"] = (age >= lo) & (age < hi)
    out: Dict[str, Any] = {}
    for g, sel in groups.items():
        if sel.sum() == 0:
            continue
        res = evaluate_classes(probs[sel][:, cols], labels[sel][:, cols], names, n_boot=0,
                               min_pos_for_ci=10**9)
        m, k = macro_auroc(res, min_pos=10)
        out[g] = {"n_records": int(sel.sum()), "macro_auroc_min10pos": m, "n_classes": k}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--georgia", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True,
                    help="v0.4.0 release checkpoint (.pt)")
    ap.add_argument("--edge-dir", type=Path, default=REPO_ROOT / "releases" / "v0.3.0",
                    help="Directory with the v0.3.0 ONNX models and thresholds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None,
                    help="Where to save raw predictions (default: <out>/../_work)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from aortica.data.georgia import LABEL_MAP_NAME, load_georgia
    from aortica.data.label_mapping import load_label_map
    from aortica.data.ptbxl import _output_columns
    from aortica.signal.preprocess import model_input

    t0 = time.time()
    records, labels, dropped = load_georgia(args.georgia, sampling_rate=100, limit=args.limit)
    x = np.stack([model_input(r) for r in records])
    meta = [dict(r.patient_metadata or {}) for r in records]
    ids = [m["georgia_record"] for m in meta]
    logger.info("georgia: %d records loaded in %.0fs", len(records), time.time() - t0)

    names, index = _output_columns(True)
    georgia_map = load_label_map(LABEL_MAP_NAME)
    scored = [n for n in names if n in georgia_map.labelable_outputs]

    # Sensitivity analysis: incomplete blocks counted as positive.
    alt_labels = labels.copy()
    for cls, code in INCOMPLETE_BBB.items():
        hit = np.array([code in m.get("snomed_dx", []) for m in meta])
        alt_labels[hit, index[cls]] = 1.0

    models: Dict[str, tuple[Callable, Dict[str, float]]] = {}
    models["v0.4.0 reference (PyTorch)"] = _torch_predictor(args.reference)
    edge_thr = json.loads((args.edge_dir / "class_thresholds.json").read_text())
    for kind in ("fp32", "int8"):
        path = args.edge_dir / f"aortica_edge_{kind}_v0.3.0.onnx"
        models[f"v0.3.0 edge {kind.upper()} (ONNX)"] = (_onnx_predictor(path), edge_thr)

    report: Dict[str, Any] = {
        "cohort": {
            "name": "Georgia 12-lead ECG Challenge Database (PhysioNet/CinC 2021, v1.0.3)",
            "n_records": len(records),
            "dropped": {k: len(v) for k, v in dropped.items()},
            "label_map": f"aortica/data/label_maps/{LABEL_MAP_NAME}.yaml",
            "positives": {n: int(labels[:, index[n]].sum()) for n in scored},
            "sex": {s: int(sum(m.get("sex") == s for m in meta))
                    for s in ("male", "female", "unknown")},
        },
        "protocol": (
            "No tuning on Georgia. Operating points are each model's released "
            "validation-tuned thresholds. CIs are percentile bootstrap over records "
            "(1000 resamples per class, 500 for macro-AUROC)."
        ),
        "models": {},
    }
    work = args.work or (args.out.parent / "_work")
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    cols = [index[n] for n in scored]
    for name, (run, thresholds) in models.items():
        t = time.time()
        probs = _predict_all(run, x)
        logger.info("%s: inference %.0fs", name, time.time() - t)
        entry = _summarise(probs, labels, cols, scored, thresholds, args.seed)
        entry["subgroups"] = _subgroups(probs, labels, cols, scored, meta)
        sens = _summarise(probs, alt_labels, [index[c] for c in INCOMPLETE_BBB],
                          list(INCOMPLETE_BBB), thresholds, args.seed)
        entry["sensitivity_incomplete_bbb_positive"] = sens["per_class"]
        report["models"][name] = entry
        m1, m10 = entry["macro_auroc_min1pos"], entry["macro_auroc_min10pos"]
        logger.info("%s: macro-AUROC %.4f (>=1 pos, %d cls) / %.4f (>=10 pos, %d cls)", name,
                    m1["value"], m1["n_classes"], m10["value"], m10["n_classes"])
        if name.startswith("v0.4.0"):
            np.savez_compressed(work / "georgia_reference_predictions.npz",
                                ids=np.array(ids), probs=probs, labels=labels,
                                snomed=np.array([",".join(m.get("snomed_dx", [])) for m in meta]))

    (args.out / "external_georgia.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s in %.0f min", args.out / "external_georgia.json", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
