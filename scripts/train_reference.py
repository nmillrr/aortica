#!/usr/bin/env python3
"""Seeded, reproducible reference training run (PTB-XL + Chapman).

Produces a release directory containing everything needed to audit or
repeat the run:

=============================  ===============================================
``config.yaml``                Exact hyperparameters, seed and data roots used
``splits.csv``                 Every record considered, its split, and whether
                               it was kept or why it was excluded
``run_manifest.json``          Git commit, library versions, thread count,
                               data checksums, checkpoint SHA-256, timings
``training_history.json``      Per-epoch losses and macro-F1
``class_thresholds.json``      Per-class operating points (tuned on validation)
``trained_outputs.json``       Allowlist + per-task loss mask
``test_metrics.json``          Held-out metrics per corpus, with 95% CIs
=============================  ===============================================

The weights (``aortica_<version>.pt``, ~10 MB) are written next to them but
are not committed; publish them with the release and check them against the
SHA-256 in the manifest.

Usage::

    python scripts/train_reference.py --config releases/v0.4.0/config.yaml

Re-running with the same config on the same hardware and thread count
reproduces the same weights.  Different CPUs/thread counts change
floating-point reduction order, so expect metric agreement to ~1e-3 rather
than bit-identical weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("train_reference")

#: Records whose peak absolute amplitude exceeds this are excluded.  PTB-XL's
#: maximum across 21,799 records is ~20 mV; the Chapman outliers above it are
#: acquisition faults, and left in they drive the loss to NaN.
MAX_ABS_UV = 25_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_dict_sha256(state: Dict[str, Any]) -> str:
    """Hash of tensor contents alone — independent of how the file was written."""
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _record_id(record: Any) -> str:
    meta = record.patient_metadata or {}
    if meta.get("source_dataset") == "ptbxl":
        return str(int(meta["ptbxl_ecg_id"]))
    return str(meta.get("chapman_record"))


def _task_masks(trained: Sequence[str]) -> Dict[str, List[float]]:
    """Per-task 0/1 loss mask for the union of labelable outputs."""
    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.risk_head import RISK_OUTPUTS
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    keep = set(trained)
    return {
        task: [1.0 if n in keep else 0.0 for n in names]
        for task, names in (
            ("rhythm", RHYTHM_CLASSES),
            ("structural", STRUCTURAL_CLASSES),
            ("ischaemia", ISCHAEMIA_CLASSES),
            ("risk", RISK_OUTPUTS),
        )
    }


def _chapman_expected(root: Path, seed: int | None) -> Dict[str, List[str]]:
    """Record stems the Chapman loader is asked for, per split."""
    from aortica.data.chapman import LABEL_MAP_NAME, index_records, split_indices
    from aortica.data.label_mapping import load_label_map

    headers = index_records(root)
    policy = load_label_map(LABEL_MAP_NAME).split_policy
    idx = split_indices(len(headers), policy, seed=seed)
    return {k: [headers[int(i)].stem for i in v] for k, v in idx.items()}


def _predict(model: Any, records: Sequence[Any], batch_size: int = 256) -> np.ndarray:
    """Concatenated sigmoid outputs ``[N, 72]`` in ALL_TASKS order."""
    import torch

    from aortica.signal.preprocess import MODEL_NUM_SAMPLES, window_signals

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            x = np.stack(
                [window_signals(r.signals, MODEL_NUM_SAMPLES) for r in records[i : i + batch_size]]
            ).astype(np.float32)
            o = model(torch.from_numpy(x)).as_dict()
            outs.append(
                np.concatenate(
                    [o[t].numpy() for t in ("rhythm", "structural", "ischaemia", "risk")], axis=1
                )
            )
    return np.concatenate(outs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None, help="Release dir (default: config's directory)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Per-split cap (smoke test)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke test)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from torch.utils.data import DataLoader

    from aortica.data.combined import combined_labelable_outputs, load_corpora, ptbxl_chapman
    from aortica.data.dataset import ECGDataset
    from aortica.data.ptbxl import _output_columns
    from aortica.evaluation.classification import (
        bootstrap_macro_auroc_ci,
        evaluate_classes,
        macro_auroc,
        tune_threshold,
    )
    from aortica.models.aortica_model import AorticaModel
    from aortica.models.train_multitask import MultiTaskTrainConfig, train_multitask
    from aortica.signal.preprocess import MODEL_SAMPLE_RATE_HZ, MODEL_WINDOW_SECONDS

    raw = yaml.safe_load(args.config.read_text())
    out_dir = (args.out or args.config.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"  # full checkpoints with optimizer state; gitignored
    work_dir.mkdir(exist_ok=True)

    version: str = raw["version"]
    seed: int = int(raw["seed"])
    threads: int = int(raw.get("torch_threads", 8))
    ptbxl_root = (REPO_ROOT / raw["data"]["ptbxl"]).resolve()
    chapman_root = (REPO_ROOT / raw["data"]["chapman"]).resolve()
    train_cfg: Dict[str, Any] = dict(raw["train"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
        train_cfg["warmup_epochs"] = min(train_cfg.get("warmup_epochs", 5), args.epochs)

    if float(train_cfg.get("sampling_rate", MODEL_SAMPLE_RATE_HZ)) != MODEL_SAMPLE_RATE_HZ:
        raise SystemExit("config sampling_rate disagrees with aortica.signal.preprocess")

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    # Captured before training: the code that ran is the code at start.
    git_commit = _git("rev-parse", "HEAD")
    git_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))

    # ── Data ────────────────────────────────────────────────────────────
    specs = ptbxl_chapman(ptbxl_root, chapman_root)
    (tr_rec, tr_y, tr_m), (va_rec, va_y, va_m), (te_rec, te_y, te_m) = load_corpora(
        specs, sampling_rate=int(MODEL_SAMPLE_RATE_HZ), multitask=True, limit=args.limit
    )
    t_loaded = time.time()
    logger.info("loaded %d/%d/%d records in %.0fs", len(tr_rec), len(va_rec), len(te_rec),
                t_loaded - t0)

    rows: List[Dict[str, str]] = []

    def _qc(split: str, recs: list, y: np.ndarray, m: np.ndarray) -> tuple:
        keep = []
        for i, r in enumerate(recs):
            peak = float(np.abs(r.signals).max())
            status = "kept" if peak <= MAX_ABS_UV else "excluded_amplitude_gt_25mV"
            meta = r.patient_metadata or {}
            rows.append({
                "corpus": str(meta.get("source_dataset")),
                "record_id": _record_id(r),
                "patient_id": str(int(meta["ptbxl_patient_id"]))
                if "ptbxl_patient_id" in meta else "",
                "split": split,
                "status": status,
            })
            if status == "kept":
                keep.append(i)
        idx = np.array(keep, dtype=np.int64)
        return [recs[i] for i in keep], y[idx], m[idx]

    tr_rec, tr_y, tr_m = _qc("train", tr_rec, tr_y, tr_m)
    va_rec, va_y, va_m = _qc("val", va_rec, va_y, va_m)
    te_rec, te_y, te_m = _qc("test", te_rec, te_y, te_m)

    # Chapman records the loader itself dropped (unreadable, dead lead,
    # wrong lead count) never reach the QC pass; list them too, so the
    # split manifest accounts for the whole corpus.
    if args.limit is None:
        kept_chapman = {(r["split"], r["record_id"]) for r in rows if r["corpus"] == "chapman"}
        for split, stems in _chapman_expected(chapman_root, None).items():
            for stem in stems:
                if (split, stem) not in kept_chapman:
                    rows.append({"corpus": "chapman", "record_id": stem, "patient_id": "",
                                 "split": split, "status": "dropped_by_loader"})

    rows.sort(key=lambda r: (r["corpus"], r["split"], r["record_id"]))
    with open(out_dir / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "record_id", "patient_id", "split", "status"])
        w.writeheader()
        w.writerows(rows)
    counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r['corpus']}/{r['split']}/{r['status']}"
        counts[key] = counts.get(key, 0) + 1
    logger.info("split manifest: %s", counts)

    # ── Train ───────────────────────────────────────────────────────────
    trained = sorted(combined_labelable_outputs(specs))
    task_masks = _task_masks(trained)
    config = MultiTaskTrainConfig(
        data_path="",
        seed=seed,
        output_dir=str(work_dir),
        per_class_weights=task_masks,
        loss_weights={"rhythm": 1.0, "structural": 1.0, "ischaemia": 1.0, "risk": 0.0},
        **train_cfg,
    )

    train_ds = ECGDataset(tr_rec, tr_y, label_masks=tr_m, window_seconds=MODEL_WINDOW_SECONDS,
                          augment=True, random_seed=seed)
    val_ds = ECGDataset(va_rec, va_y, label_masks=va_m, window_seconds=MODEL_WINDOW_SECONDS,
                        augment=False)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              drop_last=True, generator=gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    history = train_multitask(config, train_loader=train_loader, val_loader=val_loader)
    t_trained = time.time()
    (out_dir / "training_history.json").write_text(
        json.dumps([asdict(h) for h in history], indent=2) + "\n"
    )

    # ── Best checkpoint → evaluation ────────────────────────────────────
    ckpt = torch.load(work_dir / "best_multitask_model.pt", map_location="cpu")
    model = AorticaModel(in_channels=12, feature_dim=config.feature_dim,
                         head_hidden_dim=config.head_hidden_dim,
                         head_dropout=config.head_dropout,
                         enabled_tasks=config.enabled_tasks)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = int(ckpt["epoch"])

    names, _ = _output_columns(True)
    cols = [names.index(n) for n in trained]

    va_p = _predict(model, va_rec)
    thresholds = {
        names[c]: tune_threshold(va_p[va_m[:, c] > 0, c], va_y[va_m[:, c] > 0, c]) for c in cols
    }
    (out_dir / "class_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    te_p = _predict(model, te_rec)
    np.savez_compressed(work_dir / "test_predictions.npz", probs=te_p, labels=te_y, masks=te_m,
                        ids=np.array([_record_id(r) for r in te_rec]),
                        corpus=np.array([r.patient_metadata["source_dataset"] for r in te_rec]))
    corpus_of = np.array([r.patient_metadata["source_dataset"] for r in te_rec])
    test_metrics: Dict[str, Any] = {"summary": {}}
    for corpus in ("ptbxl", "chapman"):
        sel = corpus_of == corpus
        res = evaluate_classes(te_p[sel][:, cols], te_y[sel][:, cols], trained,
                               mask=te_m[sel][:, cols], thresholds=thresholds, seed=seed)
        for name, r in res.items():
            test_metrics[f"{corpus}/{name}"] = r.to_dict()
        labelled = [i for i, n in enumerate(trained) if n in res]
        m1, k1 = macro_auroc(res, min_pos=1)
        m10, k10 = macro_auroc(res, min_pos=10)
        test_metrics["summary"][corpus] = {
            "n_records": int(sel.sum()),
            "macro_auroc": m1, "n_classes": k1,
            "macro_auroc_ci95": bootstrap_macro_auroc_ci(
                te_p[sel][:, cols], te_y[sel][:, cols], labelled,
                mask=te_m[sel][:, cols], seed=seed),
            "macro_auroc_min10pos": m10, "n_classes_min10pos": k10,
            "macro_auroc_min10pos_ci95": bootstrap_macro_auroc_ci(
                te_p[sel][:, cols], te_y[sel][:, cols],
                [i for i, n in enumerate(trained) if n in res and res[n].n_pos >= 10],
                mask=te_m[sel][:, cols], seed=seed),
        }
        logger.info("test %s: macro-AUROC %.4f over %d classes (%.4f over %d with >=10 pos)",
                    corpus, m1, k1, m10, k10)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")

    # ── Release weights + sidecars ──────────────────────────────────────
    weights_path = out_dir / f"aortica_{version}.pt"
    # Machine-specific paths would make two identical runs hash differently.
    saved_config = {k: v for k, v in asdict(config).items()
                    if k not in ("output_dir", "data_path")}
    torch.save({
        "model_state_dict": model.state_dict(),
        "version": version,
        "best_epoch": best_epoch,
        "config": saved_config,
        "sampling_rate": MODEL_SAMPLE_RATE_HZ,
        "window_seconds": MODEL_WINDOW_SECONDS,
        "units": "µV",
        "trained_outputs": trained,
        "class_thresholds": thresholds,
    }, weights_path)

    (out_dir / "trained_outputs.json").write_text(json.dumps({
        "trained_outputs": trained,
        "per_task_weights": task_masks,
        "sampling_rate": int(MODEL_SAMPLE_RATE_HZ),
        "datasets": ["PTB-XL 1.0.3", "Chapman-Shaoxing 1.0.0"],
        "n_trained": len(trained),
    }, indent=2) + "\n")

    import scipy
    manifest = {
        "version": version,
        "seed": seed,
        "smoke_test": args.limit is not None or args.epochs is not None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "data": {
            "ptbxl_database_csv_sha256": _sha256(ptbxl_root / "ptbxl_database.csv"),
            "chapman_sha256sums_sha256": _sha256(chapman_root / "SHA256SUMS.txt"),
        },
        "records": {
            "train": len(tr_rec), "val": len(va_rec), "test": len(te_rec),
            "by_status": counts,
        },
        "best_epoch": best_epoch,
        "weights_file": weights_path.name,
        "weights_sha256": _sha256(weights_path),
        "state_dict_sha256": _state_dict_sha256(model.state_dict()),
        "splits_sha256": _sha256(out_dir / "splits.csv"),
        "seconds": {
            "load": round(t_loaded - t0),
            "train": round(t_trained - t_loaded),
            "total": round(time.time() - t0),
        },
        "env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
                if k in os.environ},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("done in %.0f min — %s", (time.time() - t0) / 60, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
