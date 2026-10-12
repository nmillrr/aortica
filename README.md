# Aortica

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-orange.svg)](releases/v0.4.0)

**Open-source 12-lead ECG analysis** — a multi-task deep learning model, the
tooling to train and evaluate it reproducibly, and the plumbing to run it on
your own hardware: format readers, signal quality, explanations, a REST API,
CLI, web UI, an offline edge model, and EHR integrations.

> [!CAUTION]
> **Research use only. Not a medical device.** Aortica is not FDA-cleared or
> CE-marked and must not be used to diagnose, treat or triage patients.

---

## Status at a glance

| | |
|---|---|
| **Trained outputs** | **28 of 72.** Rhythm & conduction (18), structural (4), ischaemia & repolarisation (6). The other 44 — including the entire risk head — have no label source in any public dataset used and are **suppressed by default** on every inference path. |
| **Training data** | PTB-XL 1.0.3 (Germany) + Chapman-Shaoxing 1.0.0 (China), 53,405 training ECGs. |
| **Internal test** | Macro-AUROC 0.941 (PTB-XL fold 10) and 0.975 (Chapman test split), classes with ≥10 positives. |
| **External test** | Macro-AUROC 0.894 on 10,338 ECGs from the Georgia cohort (Emory, US) — never trained or tuned on. |
| **Baseline** | On the six findings it shares with Ribeiro et al. (2020) — a model trained on about 2.3 million ECGs — both score a mean AUROC of 0.967 on Georgia: Aortica is ahead on sinus bradycardia and slightly behind (by 0.001–0.014) on first-degree AV block, RBBB, LBBB and sinus tachycardia, with AF tied. |
| **Reproducibility** | Seeded run with saved splits and a run manifest in [`releases/v0.4.0/`](releases/v0.4.0). An independent rerun of the full 30-epoch config on the same machine reproduced the weights bit for bit (identical `state_dict_sha256`). |
| **Distribution** | Not on PyPI, Docker Hub or a model hub. Install from source. |

Full per-class results, confidence intervals and subgroups:
[`releases/v0.4.0/RESULTS.md`](releases/v0.4.0/RESULTS.md).

---

## What the model detects

| Head | Trained outputs |
|---|---|
| **Rhythm & conduction** | AF, AFL, SVT, AVRT, sinus bradycardia, sinus tachycardia, PAC, PVC, 1st/2nd/3rd-degree AV block, LBBB, RBBB, LAFB, LPFB, WPW, paced rhythm, normal sinus rhythm |
| **Structural** | LVH, RVH, left atrial enlargement, right atrial enlargement |
| **Ischaemia & repolarisation** | STEMI, posterior MI, old MI, QTc prolongation, digitalis effect, early repolarisation |

**Not trained** (present in the architecture, withheld from every response):
VT/VF, Brugada, Wellens and de Winter patterns, cardiomyopathies (HCM, DCM,
ARVC, amyloid), electrolyte disorders, and all risk predictions (1-year
mortality, heart-failure hospitalisation, AF onset, ECG-predicted EF).
Responses name them in `suppressed_classes` so "not predicted" is never
confused with "predicted negative". See
[`aortica/models/output_gating.py`](aortica/models/output_gating.py).

Several trained classes have too few positives in the test sets to be
validated (for example AVRT, 2nd/3rd-degree AV block, RA enlargement). Check
`n_pos` in the metrics before relying on any single class.

### Decision thresholds

Each class has its own operating point, tuned on validation data and shipped
with the checkpoint. Raw probabilities are not comparable across classes —
rare findings peak far below common ones — so every "is this present?"
decision (clinical suggestions, CLI severity, the CHW tiers) uses the
per-class threshold, and API responses return it alongside each probability.

### Input contract

Every checkpoint is trained on **12 leads, 100 Hz, 10 s (1,000 samples),
micro-volts, unfiltered**. All inference paths build model input through
[`aortica.signal.preprocess.model_input`](aortica/signal/preprocess.py), which
resamples, windows and unit-converts from whatever was read. Denoising is
applied for signal-quality scoring only. (Earlier versions fed 500 Hz,
denoised input to this 100 Hz model and scored near chance; a regression test
on real PTB-XL records now guards against it.)

---

## 🚀 Quick start

### Install (from source)

```bash
git clone https://github.com/nmillrr/aortica.git
cd aortica
pip install -e ".[cli,api,torch]"   # add edge, dev, ... as needed
```

### Run on an ECG

```bash
aortica predict recording.hea --model releases/v0.4.0/aortica_v0.4.0.pt
aortica predict recording.hea --model ... --format json --tasks rhythm,ischaemia
```

The v0.4.0 weights are produced by the reference run below (they are not
hosted yet). Without `--model`, `aortica predict` reports signal quality only.

### REST API

```bash
aortica-server                       # http://localhost:8000
curl -X POST http://localhost:8000/api/v1/predict -F "file=@recording.xml"
```

### Python

```python
import torch
from aortica.io import read_ecg
from aortica.models.checkpoint import load_checkpoint
from aortica.models.output_gating import filter_task_outputs, is_positive
from aortica.models.task_dims import TASK_CLASS_NAMES
from aortica.signal.preprocess import model_input

model = load_checkpoint("releases/v0.4.0/aortica_v0.4.0.pt")
ecg = read_ecg("recording.hea")                 # any supported format
x = torch.from_numpy(model_input(ecg))[None]    # [1, 12, 1000]: 100 Hz, 10 s, µV

with torch.no_grad():
    out = model(x).as_dict()

for task in ("rhythm", "structural", "ischaemia"):
    kept, probs, _ = filter_task_outputs(TASK_CLASS_NAMES[task], out[task][0].tolist())
    for name, p in zip(kept, probs):
        if is_positive(name, p, model.class_thresholds):
            print(task, name, round(p, 3))
```

For the full pipeline (reading, quality scoring, gating, explanations,
suggestions) use `aortica.api.predict.run_inference_pipeline`.

---

## 🔁 Reproducing the results

All datasets are open access on PhysioNet.

| Dataset | Use | Location expected |
|---|---|---|
| [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | train / val / test (folds 1–8 / 9 / 10) | `data/ptb-xl-…-1.0.3/` |
| [Chapman-Shaoxing 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/) | train / val / test (seeded 80/10/10) | `data/a-large-scale-…-1.0.0/` |
| [Georgia, Challenge 2021](https://physionet.org/content/challenge-2021/1.0.3/training/georgia/) | external test only | `data/georgia/` |
| [Ribeiro et al. 2020 weights](https://zenodo.org/records/3765717) | baseline | `data/ribeiro/model/model.hdf5` |

```bash
# 1. Train (≈4 h on an 8-core CPU). Writes splits.csv, run_manifest.json,
#    class_thresholds.json, test_metrics.json and the weights.
python scripts/train_reference.py --config releases/v0.4.0/config.yaml

# 2. External validation on Georgia.
python scripts/external_validation.py --georgia data/georgia \
    --reference releases/v0.4.0/aortica_v0.4.0.pt --out releases/v0.4.0/evaluation

# 3. Baseline (needs TensorFlow 2.16 + tf_keras in a separate env).
TF_USE_LEGACY_KERAS=1 python scripts/baseline_ribeiro.py \
    --model data/ribeiro/model/model.hdf5 --ptbxl data/ptb-xl-…-1.0.3 \
    --georgia data/georgia --out releases/v0.4.0/_work/ribeiro
python scripts/compare_baseline.py --ptbxl data/ptb-xl-…-1.0.3 \
    --out releases/v0.4.0/evaluation

# 4. Tables.
python scripts/render_results.py releases/v0.4.0
```

`splits.csv` lists every record considered, its split, and whether it was
kept or why it was excluded (154 Chapman records: unreadable, a dead lead, or
amplitudes above 25 mV). `run_manifest.json` records the commit, library
versions, thread count, data checksums and the SHA-256 of the weights.

---

## 🧩 Platform

- **Format readers** — WFDB, DICOM (Supp 30/130), SCP-ECG, HL7 aECG XML, CSV,
  MAT, and scanned PDFs/images; `aortica.io.read_ecg` auto-detects.
- **Signal processing** — QRS detection (NeuroKit2, Pan-Tompkins), denoising,
  per-lead quality scoring.
- **Explainability** — integrated-gradient attributions mapped to named ECG
  segments, returned at the model's sample rate.
- **Interfaces** — FastAPI REST + gRPC, Click CLI, React PWA (12-lead viewer,
  findings panel with clinical suggestions, second-reader mode, batch mode).
- **Edge** — distilled MobileNet-1D student, INT8 ONNX (0.42 MB, v0.3.0) for
  ARM devices and an in-browser fallback. On-device latency has not been
  measured on hardware.
- **Integration** — FHIR R4, HL7 v2 ORU, DICOM SR, DIMSE, serial capture,
  SMART on FHIR; PDF/JSON-LD/CSV reports.
- **Federated learning** — Flower client/server with FedAvg, FedProx and
  SCAFFOLD; raw ECGs stay on site. Model updates are sent **without
  differential privacy or secure aggregation** (the `dp` and `secure_agg`
  modules are experimental and not wired in).
- **Evaluation & governance** — per-class metrics with bootstrap CIs,
  demographic subgroups, equity gate, performance cards, and templates for
  IEC 80601-2-86, FDA pre-submission, CE-MDR and TRIPOD/STARD/CONSORT-AI.

```
aortica/
├── io/          format readers          ├── api/          REST + gRPC
├── signal/      quality, denoising,     ├── cli/          `aortica` commands
│                model input contract    ├── edge/         ONNX, distillation, INT8
├── models/      backbone, heads,        ├── federated/    Flower FL
│                gating, checkpoints     ├── integration/  FHIR, HL7, DICOM
├── data/        PTB-XL, Chapman,        ├── reports/      PDF, JSON-LD, CSV
│                Georgia loaders         ├── retrieval/    similar-case search
├── evaluation/  metrics, gates          └── validation/   prospective tooling
└── xai/         attributions
releases/        v0.2.0, v0.3.0 (edge), v0.4.0 (reference) — metrics, splits, manifests
frontend/        React + Vite PWA        mobile/android/   Kotlin app
docs/            MkDocs site             scripts/          training & evaluation
```

### Docker

No images are published; build locally:

```bash
make dev                                    # API + frontend + docs (+ edge)
docker build -f Dockerfile.server -t aortica-server .
docker build -f Dockerfile.edge   -t aortica-edge .
```

---

## 🧪 Development

```bash
pip install -e ".[dev,cli,api,torch,edge]"
pytest -q -m "not slow"
ruff check aortica tests scripts
cd frontend && npm ci && npm run lint && npm run build
```

---

## Limitations

- 44 of 72 outputs are untrained; no risk prediction is available.
- Two training cohorts and one external cohort. Generalisation to other
  populations, devices and especially low-resource settings is untested.
- Label definitions differ between corpora (PTB-XL counts incomplete bundle
  branch blocks as RBBB/LBBB; Chapman and Georgia do not), which affects both
  training and evaluation; see the sensitivity analysis in the results.
- Operating points were tuned on in-distribution validation data; on an
  external cohort sensitivity/specificity at those points can shift even when
  AUROC holds.
- Several trained classes have too few test positives to be validated.
- 100 Hz input; ST-segment detail may benefit from higher rates.

---

## License & acknowledgements

Apache 2.0 — see [LICENSE](LICENSE). Planning documents from the build-out
are in [`docs/archive/`](docs/archive/); they do not describe current
capabilities.

- **PTB-XL** — Wagner P. et al. *Sci Data* 7, 154 (2020). CC BY 4.0.
- **Chapman-Shaoxing** — Zheng J. et al. *Sci Data* 7, 48 (2020).
- **Georgia cohort** — PhysioNet/Computing in Cardiology Challenge 2021,
  Reyna M.A. et al. *Comput Cardiol* (2021). CC BY 4.0.
- **Baseline** — Ribeiro A.H. et al. *Nat Commun* 11, 1760 (2020).
- **PhysioNet** — Goldberger A. et al. *Circulation* 101(23) (2000).
