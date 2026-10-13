#!/usr/bin/env python3
"""Build the committed INT8 calibration set from PTB-XL's validation fold.

Static INT8 quantization needs real ECGs to set activation ranges; on noise
the edge model scores near chance (see aortica/edge/quantization.py). This
writes a small, fixed set so quantization — including the federated release
workflow — can run without a local copy of PTB-XL:

* 100 records drawn with a fixed seed from PTB-XL 1.0.3 ``strat_fold`` 9
  (validation; never the test fold 10),
* prepared with :func:`aortica.signal.preprocess.model_input` (12 leads,
  100 Hz, 10 s, µV),
* stored as float16 ``[100, 12, 1000]`` to halve the size; the loader casts
  back to float32,
* with the chosen ``ecg_id`` values written alongside.

Usage::

    python scripts/build_calibration_set.py \\
        --ptbxl data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \\
        --out releases/calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

N_SAMPLES = 100
SEED = 0
FOLD = 9
NAME = "ptbxl_fold9_calibration_100"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ptbxl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "releases" / "calibration")
    args = ap.parse_args(argv)

    from aortica.data.ptbxl import load_ptbxl
    from aortica.signal.preprocess import model_input

    _, (records, _), _ = load_ptbxl(
        args.ptbxl, sampling_rate=100, folds={"train": [], "val": [FOLD], "test": []}
    )
    order = np.random.default_rng(SEED).permutation(len(records))[:N_SAMPLES]
    chosen = [records[i] for i in sorted(order)]
    arr = np.stack([model_input(r) for r in chosen]).astype(np.float16)
    if not np.isfinite(arr).all():
        raise SystemExit("non-finite values after float16 conversion")

    args.out.mkdir(parents=True, exist_ok=True)
    npy = args.out / f"{NAME}.npy"
    np.save(npy, arr)
    meta = {
        "source": "PTB-XL 1.0.3 (Wagner et al., PhysioNet, CC BY 4.0)",
        "split": f"strat_fold {FOLD} (validation)",
        "seed": SEED,
        "n": len(chosen),
        "shape": list(arr.shape),
        "dtype": "float16 (cast to float32 on load)",
        "contract": "aortica.signal.preprocess.model_input: 12 leads, 100 Hz, 10 s, uV",
        "ecg_ids": [int(r.patient_metadata["ptbxl_ecg_id"]) for r in chosen],
        "sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
    }
    (args.out / f"{NAME}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {npy} ({npy.stat().st_size / 1e6:.2f} MB), sha256 {meta['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
