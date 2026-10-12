---
license: cc-by-4.0
tags:
  - ecg
  - electrocardiography
  - medical
  - time-series
  - onnx
  - edge
library_name: pytorch
---

# Aortica — 12-lead ECG multi-task model (v0.3.0)

Multi-task 12-lead ECG analysis for rhythm, structural, and ischaemia findings,
with an INT8 edge variant sized for Raspberry Pi / Android deployment.

**v0.3.0 adds Chapman-Shaoxing** alongside PTB-XL: 28 trained outputs (up from
26), 3× the training data, and real statistical support for several rhythm
classes that v0.2.0 could not validate.

> [!CAUTION]
> **Research use only. Not a medical device.** Not FDA-cleared, CE-marked, or
> validated for clinical decision-making. Do not use to diagnose, treat, or
> triage patients.

## The most important thing to know

The model architecture emits **72 outputs**, but only **28 of them were
trained**. Neither dataset can supply labels for the other 44 — they describe
findings requiring echocardiography, serum chemistry, angiography, or
longitudinal follow-up, none of which exist in either source.

**The 44 untrained outputs emit meaningless values.** They were held out of the
loss entirely rather than trained against fabricated negatives, so they sit at
initialization and drift with the backbone. They must be suppressed before any
downstream use:

```python
from aortica.edge.simplified_output import (
    load_trained_outputs, load_class_thresholds, simplify_output,
)

load_trained_outputs("trained_outputs.json")       # 28-name allowlist
load_class_thresholds("class_thresholds.json")     # per-class calibration
report = simplify_output(model_output)             # untrained outputs ignored
```

> [!NOTE]
> Pass **`class_thresholds.json`**, not `edge_test_metrics.json`. The metrics
> file is keyed `ptbxl/<class>` and `chapman/<class>` to keep the two
> evaluations distinct; those prefixed keys match no class name, so loading it
> would silently calibrate nothing. `class_thresholds.json` is the flat
> `{class: threshold}` mapping the runtime expects — for each class it takes
> the threshold from whichever split had more test positives
> (`class_thresholds_provenance.json` records which, and why).

Without this gate, an untrained head can escalate a triage decision on noise.
The entire `risk` head (`mortality_1y`, `hf_hosp_12m`, `af_onset_12m`,
`ecg_predicted_ef`, `conduction_disease_trajectory`,
`sudden_cardiac_death_risk`) is untrained — neither dataset has outcome data.

## Trained outputs (28)

**rhythm (19):** AF, AFL, SVT, AVRT, sinus_brady, sinus_tachy, PAC, PVC,
av_block_1st, av_block_2nd, av_block_3rd, LBBB, RBBB, LAFB, LPFB, WPW,
pacemaker_rhythm, normal_sinus_rhythm
**structural (4):** LVH, RVH, LA_enlargement, RA_enlargement
**ischaemia (6):** STEMI, posterior_MI, old_MI, digitalis_effect,
QTc_prolongation, early_repol_vs_STEMI

`AVRT` and `early_repol_vs_STEMI` are new in v0.3.0 — PTB-XL cannot label
either.

## Performance

Macro-AUROC, evaluated on each dataset's held-out test split:

| model | params | size | PTB-XL cols | Chapman cols |
|---|---|---|---|---|
| teacher (ResNet-1D) | 2.39 M | 28 MB | 0.943 | 0.969 |
| student INT8 (ONNX) | 0.32 M | **0.42 MB** | 0.929 | 0.948 |

For reference, v0.2.0 (PTB-XL only) scored 0.946 on the PTB-XL columns. **Adding
Chapman did not degrade PTB-XL performance** (0.943 vs 0.946 is within run
variance) while adding two classes and materially improving several others.

### Where v0.3.0 is clearly better

Chapman supplies far more positives for several rhythm classes than PTB-XL,
turning single-sample artifacts into trustworthy measurements (INT8, Chapman
test split):

| class | PTB-XL n+ | Chapman n+ | AUROC | F1 |
|---|---|---|---|---|
| sinus_brady | 64 | **1650** | 0.999 | 0.984 |
| sinus_tachy | 82 | **743** | 0.998 | 0.970 |
| AFL | 7 | **838** | 0.974 | 0.871 |
| SVT | 5 | **105** | 0.990 | 0.692 |
| PAC | 40 | **143** | 0.961 | 0.718 |

### Best-supported classes on PTB-XL fold 10 (INT8)

| class | n+ | AUROC | F1 |
|---|---|---|---|
| normal_sinus_rhythm | 1674 | 0.909 | 0.930 |
| AF | 152 | 0.985 | 0.913 |
| LBBB | 62 | 0.992 | 0.885 |
| sinus_tachy | 82 | 0.992 | 0.854 |
| pacemaker_rhythm | 28 | 0.915 | 0.792 |
| PVC | 122 | 0.979 | 0.778 |
| LAFB | 162 | 0.981 | 0.760 |
| RBBB | 166 | 0.984 | 0.733 |
| old_MI | 526 | 0.909 | 0.700 |
| LVH | 234 | 0.932 | 0.650 |

> [!WARNING]
> **Several trained classes remain unvalidated.** `av_block_2nd` (n=1 in PTB-XL
> fold 10), `av_block_3rd` (n=2), `AVRT` (n=3 in Chapman), `SVT` (n=5 in
> PTB-XL), `AFL` (n=7 in PTB-XL), `RA_enlargement` (n=4 in Chapman) — where a
> class has single-digit positives, its AUROC is an artifact and its F1 is
> frequently 0.000. Check `test_pos` in the metrics JSON before trusting any
> class. Note that some of these (SVT, AFL, av_block_2nd/3rd) *are* well
> supported on the Chapman split even though they are not on PTB-XL's.

Per-class metrics: `test_metrics.json` (teacher), `edge_test_metrics.json`
(INT8). Both are keyed `ptbxl/<class>` and `chapman/<class>`.

## Thresholds and calibration

Decision thresholds were tuned per class on each dataset's validation split and
are stored in the metrics JSON. **A fixed 0.5 threshold performs materially
worse** on rare classes, several of which have operating points at 0.05–0.20.

The CHW tier thresholds in `simplified_output` express *clinical* confidence
("escalate at 60% sure"), which assumes a calibrated model. This model is not
calibrated that way. `load_class_thresholds` rescales each class
piecewise-linearly so its operating point maps to 0.5; the mapping is
monotonic, so ranking and AUROC are unchanged.

## Training

- **Data:** PTB-XL 1.0.3 (CC BY 4.0; Wagner et al. 2020) + Chapman-Shaoxing
  1.0.0 (Zheng et al. 2020), both PhysioNet. 100 Hz, 10-second windows.
  53,426 train / 6,674 validation records.
- **Label masking:** each record carries a per-sample validity mask, so a
  column its source dataset cannot label contributes no gradient rather than
  counting as a true negative. This is what makes mixing the two label
  vocabularies safe.
- **Data quality:** 154 Chapman records (0.34%) were excluded — non-finite
  signal values after resampling, or amplitudes above 25 mV (physiologically
  implausible; PTB-XL's max across 21,799 records is ~20 mV). Two further
  headers (JS01052, JS23074) are malformed and unreadable. Left in, these
  produce NaN loss.
- **Teacher:** 30 epochs, AdamW, lr 1e-3, cosine decay with 5 warm-up epochs,
  batch 64, best epoch 22 by masked validation loss.
- **Student:** knowledge distillation (T=4.0, alpha=0.7), 30 epochs, best
  epoch 28.
- **Quantization:** static INT8 QDQ, 100 real ECGs as calibration.

## Limitations

- 44 of 72 outputs untrained; the whole risk head is untrained.
- Several trained classes lack the test positives needed to validate them.
- Two source cohorts only (German and Chinese) — no external validation, and
  no evidence of generalization to other populations, devices, or geographies.
  This matters especially for deployment contexts unlike either source cohort.
- Class imbalance is uncorrected: the loss supports per-class weights but not
  `pos_weight`, so rare classes remain under-fit.
- Cross-dataset disagreement exists: `AF` scores 0.985 AUROC on PTB-XL but
  0.920 on Chapman, and `WPW` 0.916 vs 0.762 — the same label means somewhat
  different things across the two annotation schemes.
- 100 Hz only; ST-segment detail may benefit from 500 Hz.

## Citation

PTB-XL: Wagner, P., Strodthoff, N., Bousseljot, RD. et al. *PTB-XL, a large
publicly available electrocardiography dataset.* Sci Data 7, 154 (2020).

Chapman-Shaoxing: Zheng, J., Zhang, J., Danioko, S. et al. *A 12-lead
electrocardiogram database for arrhythmia research covering more than 10,000
patients.* Sci Data 7, 48 (2020).
