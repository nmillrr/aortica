#!/usr/bin/env python3
"""Paired comparison: Aortica v0.4.0 vs the Ribeiro et al. (2020) model.

Both models are scored on the **same records** for the six findings they
share.  Ground truth is defined from each cohort's source codes, using the
narrower (Ribeiro) definition of a bundle branch block — complete blocks
only — so that neither model's training label map defines the target.

=========  =====================  ==========================  ===============
Ribeiro    Aortica output         PTB-XL SCP                  Georgia SNOMED
=========  =====================  ==========================  ===============
1dAVb      av_block_1st           1AVB                        270492004
RBBB       RBBB                   CRBBB                       59118001, 713427006
LBBB       LBBB                   CLBBB                       164909002, 733534002
SB         sinus_brady            SBRAD                       426177001
AF         AF                     AFIB                        164889003
ST         sinus_tachy            STACH                       427084000
=========  =====================  ==========================  ===============

Cohorts:

* **Georgia** (primary) — external to both models.
* **PTB-XL fold 10** — Aortica's held-out test fold, but drawn from the same
  distribution it trained on; external to Ribeiro.  This comparison favours
  Aortica and is reported for completeness only.

For each class: AUROC and AUPRC for both models, and the paired difference
(Aortica - Ribeiro) with a 95% percentile bootstrap CI over records (2000
resamples, both models resampled together).

Inputs are the prediction files written by ``external_validation.py``,
``train_reference.py`` and ``baseline_ribeiro.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PAIRS = [
    # (Ribeiro, Aortica, PTB-XL SCP codes, Georgia SNOMED codes)
    ("1dAVb", "av_block_1st", {"1AVB"}, {"270492004"}),
    ("RBBB", "RBBB", {"CRBBB"}, {"59118001", "713427006"}),
    ("LBBB", "LBBB", {"CLBBB"}, {"164909002", "733534002"}),
    ("SB", "sinus_brady", {"SBRAD"}, {"426177001"}),
    ("AF", "AF", {"AFIB"}, {"164889003"}),
    ("ST", "sinus_tachy", {"STACH"}, {"427084000"}),
]


def _paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int,
            seed: int) -> Dict[str, Any]:
    from aortica.evaluation.classification import auroc, average_precision

    out: Dict[str, Any] = {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "aortica_auroc": auroc(a, y), "ribeiro_auroc": auroc(b, y),
        "aortica_auprc": average_precision(a, y), "ribeiro_auprc": average_precision(b, y),
    }
    out["delta_auroc"] = out["aortica_auroc"] - out["ribeiro_auroc"]
    out["delta_auprc"] = out["aortica_auprc"] - out["ribeiro_auprc"]
    if y.sum() < 10:
        out["delta_auroc_ci95"] = out["delta_auprc_ci95"] = None
        return out
    rng = np.random.default_rng(seed)
    d_roc, d_prc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        d_roc.append(auroc(a[i], y[i]) - auroc(b[i], y[i]))
        d_prc.append(average_precision(a[i], y[i]) - average_precision(b[i], y[i]))
    out["delta_auroc_ci95"] = [float(v) for v in np.percentile(d_roc, [2.5, 97.5])]
    out["delta_auprc_ci95"] = [float(v) for v in np.percentile(d_prc, [2.5, 97.5])]
    return out


def _cohort(aortica: np.ndarray, ribeiro: np.ndarray, truth: np.ndarray, n_boot: int,
            seed: int) -> Dict[str, Any]:
    per = {}
    for j, (rname, aname, _, _) in enumerate(PAIRS):
        per[rname] = {"aortica_output": aname,
                      **_paired(aortica[:, j], ribeiro[:, j], truth[:, j], n_boot, seed + j)}
    mean = lambda k: float(np.mean([v[k] for v in per.values()]))  # noqa: E731
    return {"n_records": int(len(truth)), "per_class": per,
            "mean_auroc": {"aortica": mean("aortica_auroc"), "ribeiro": mean("ribeiro_auroc")},
            "mean_auprc": {"aortica": mean("aortica_auprc"), "ribeiro": mean("ribeiro_auprc")}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "releases" / "v0.4.0" / "_work")
    ap.add_argument("--ptbxl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from aortica.data.ptbxl import _output_columns

    names, index = _output_columns(True)
    a_cols = [index[a] for _, a, _, _ in PAIRS]
    rib = np.load(args.work / "ribeiro" / "ribeiro_predictions.npz")
    assert list(rib["classes"]) == [r for r, _, _, _ in PAIRS]
    report: Dict[str, Any] = {
        "baseline": ("Ribeiro AH et al. Nat Commun 11, 1760 (2020); "
                     "weights Zenodo 3765717 model.hdf5"),
        "selection": json.loads(
            (args.work / "ribeiro" / "ribeiro_preprocessing_selection.json").read_text()
        )["selected"],
        "cohorts": {},
    }

    # ── Georgia ──────────────────────────────────────────────────────────
    ga = np.load(args.work / "georgia_reference_predictions.npz")
    r_pos = {str(k): i for i, k in enumerate(rib["georgia_ids"])}
    shared = [i for i, k in enumerate(ga["ids"]) if str(k) in r_pos]
    a = ga["probs"][shared][:, a_cols]
    b = rib["georgia_probs"][[r_pos[str(ga["ids"][i])] for i in shared]]
    dx = [set(str(ga["snomed"][i]).split(",")) for i in shared]
    y = np.array([[bool(d & codes) for _, _, _, codes in PAIRS] for d in dx], dtype=float)
    report["cohorts"]["georgia"] = {
        "role": "primary — external to both models",
        **_cohort(a, b, y, args.n_boot, args.seed),
    }

    # ── PTB-XL fold 10 ───────────────────────────────────────────────────
    te = np.load(args.work / "test_predictions.npz")
    sel = te["corpus"] == "ptbxl"
    ids = [str(i) for i in te["ids"][sel]]
    r_pos = {str(k): i for i, k in enumerate(rib["ptbxl_ids"])}
    keep = [j for j, k in enumerate(ids) if k in r_pos]
    a = te["probs"][sel][keep][:, a_cols]
    b = rib["ptbxl_probs"][[r_pos[ids[j]] for j in keep]]
    db = pd.read_csv(args.ptbxl / "ptbxl_database.csv", index_col="ecg_id")
    codes = [set(ast.literal_eval(db.loc[int(ids[j]), "scp_codes"])) for j in keep]
    y = np.array([[bool(c & scp) for _, _, scp, _ in PAIRS] for c in codes], dtype=float)
    report["cohorts"]["ptbxl_fold10"] = {
        "role": "secondary — Aortica's in-distribution test fold; external to Ribeiro",
        **_cohort(a, b, y, args.n_boot, args.seed),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_ribeiro.json").write_text(json.dumps(report, indent=2) + "\n")
    for cohort, body in report["cohorts"].items():
        print(f"\n{cohort} (n={body['n_records']})")
        for r, v in body["per_class"].items():
            ci = v["delta_auroc_ci95"]
            ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "(n<10)"
            print(f"  {r:6s} n+={v['n_pos']:5d}  Aortica {v['aortica_auroc']:.3f}  "
                  f"Ribeiro {v['ribeiro_auroc']:.3f}  delta {v['delta_auroc']:+.3f} {ci_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
