# Releases

One directory per model release. Weights for PyTorch models are not committed
(each manifest records their SHA-256); ONNX edge models are small enough to
live here.

| Release | What it is | Status |
|---|---|---|
| [`v0.4.0/`](v0.4.0/) | **Reference model** (ResNet-1D teacher, PTB-XL + Chapman, 28 outputs). Seeded, with saved splits, run manifest, internal and external (Georgia) evaluation and a baseline comparison. | The model the whitepaper reports. No edge (distilled/INT8) variant yet. |
| [`v0.3.0/`](v0.3.0/) | Edge release: distilled MobileNet-1D student, FP32 and INT8 ONNX, trained on PTB-XL + Chapman. | **Shipped edge model** — PWA offline fallback, Android app, Raspberry Pi. Its teacher weights were not preserved and its Chapman split is not recoverable. |
| [`v0.2.0/`](v0.2.0/) | First edge release, PTB-XL only (26 outputs). | Superseded. |

[`calibration/`](calibration/) holds 100 PTB-XL validation-fold ECGs for INT8
calibration (quantizing on noise yields a model that scores near chance).

`scripts/sync_release_artifacts.py` copies the shipped edge model and its
sidecars from `v0.3.0/` to the frontend, the Android assets and
`aortica/models/artifacts/`; CI fails if they drift.

Every model here is **research use only** and not a medical device.
