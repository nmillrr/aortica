# INT8 calibration set

`ptbxl_fold9_calibration_100.npy` — 100 real 12-lead ECGs used to calibrate
static INT8 quantization of the edge models, so quantization (including the
federated release workflow) runs without a local copy of PTB-XL.

| | |
|---|---|
| Source | PTB-XL 1.0.3, `strat_fold` 9 — the **validation** fold; never the test fold |
| Selection | 100 records, `numpy.random.default_rng(0).permutation`; ids in `ptbxl_fold9_calibration_100.json` |
| Contents | `[100, 12, 1000]` float16, prepared with `aortica.signal.preprocess.model_input` (12 leads, 100 Hz, 10 s, µV); cast to float32 on load |
| Integrity | SHA-256 in the JSON; `tests/edge/test_quantization.py` checks it |
| Rebuild | `python scripts/build_calibration_set.py --ptbxl <PTB-XL root>` (byte-identical) |

Why real ECGs: calibrated on Gaussian noise, the v0.3.0 INT8 student scores
macro-AUROC 0.527 on PTB-XL fold 10. Calibrated on this set it scores 0.927
(FP32: 0.943).

Use it with `aortica federated release --calibration-data
releases/calibration/ptbxl_fold9_calibration_100.npy`, or
`aortica.edge.quantization.load_calibration_data(...)`.

PTB-XL: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W.,
Schaeffter, T. (2022). *PTB-XL, a large publicly available electrocardiography
dataset* (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45 —
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The signals here
are resampled and windowed from the original records.
