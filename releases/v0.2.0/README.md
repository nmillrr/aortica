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

# Aortica — 12-lead ECG multi-task model

Multi-task 12-lead ECG analysis for rhythm, structural, and ischaemia findings,
with an INT8 edge variant sized for Raspberry Pi / Android deployment.

> [!CAUTION]
> **Research use only. Not a medical device.** Not FDA-cleared, CE-marked, or
> validated for clinical decision-making. Do not use to diagnose, treat, or
> triage patients.

## The most important thing to know

The model architecture emits **72 outputs**, but only **26 of them were
trained**. PTB-XL cannot supply labels for the other 46 — they describe
findings requiring echocardiography, serum chemistry, angiography, or
longitudinal follow-up, none of which exist in the dataset.

**The 46 untrained outputs emit meaningless values.** They were held out of the
loss entirely (zero per-class weight) rather than trained against fabricated
negatives, so they sit at initialization and drift with the backbone. They must
be suppressed before any downstream use:

```python
from aortica.edge.simplified_output import (
    load_trained_outputs, load_class_thresholds, simplify_output,
)

load_trained_outputs("trained_outputs.json")       # 26-name allowlist
load_class_thresholds("edge_test_metrics.json")    # per-class calibration
report = simplify_output(model_output)             # untrained outputs ignored
```

Without this gate, an untrained head can escalate a triage decision on noise.
The entire `risk` head (`mortality_1y`, `hf_hosp_12m`, `af_onset_12m`,
`ecg_predicted_ef`, `conduction_disease_trajectory`,
`sudden_cardiac_death_risk`) is untrained — PTB-XL has no outcome data at all.

## Trained outputs

**rhythm (17):** AF, AFL, SVT, sinus_brady, sinus_tachy, PAC, PVC,
av_block_1st, av_block_2nd, av_block_3rd, LBBB, RBBB, LAFB, LPFB, WPW,
pacemaker_rhythm, normal_sinus_rhythm
**structural (4):** LVH, RVH, LA_enlargement, RA_enlargement
**ischaemia (5):** STEMI, posterior_MI, old_MI, digitalis_effect,
QTc_prolongation

## Performance — PTB-XL fold 10 (held out)

Macro-AUROC over the 26 trained classes:

| model | params | size | macro-AUROC |
|---|---|---|---|
| teacher (ResNet-1D) | 2.39 M | 28 MB | 0.946 |
| student (MobileNet-1D) | 0.32 M | 3.97 MB | 0.947 |
| student INT8 (ONNX) | 0.32 M | **0.42 MB** | 0.938 |

Classes with enough test positives to be trustworthy (INT8, tuned thresholds):

| class | n+ | AUROC | F1 |
|---|---|---|---|
| normal_sinus_rhythm | 1674 | 0.912 | 0.932 |
| AF | 152 | 0.985 | 0.884 |
| sinus_tachy | 82 | 0.988 | 0.871 |
| LBBB | 62 | 0.990 | 0.852 |
| PVC | 122 | 0.993 | 0.819 |
| RBBB | 166 | 0.987 | 0.810 |
| pacemaker_rhythm | 28 | 0.943 | 0.769 |
| LAFB | 162 | 0.982 | 0.761 |
| old_MI | 526 | 0.914 | 0.726 |
| LVH | 234 | 0.945 | 0.642 |

> [!WARNING]
> **The remaining 16 trained classes are not validated.** Several have almost no
> positives in the test fold — `av_block_2nd` (n=1), `av_block_3rd` (n=2),
> `SVT` (n=5), `AFL` (n=7), `WPW` (n=8) — so their AUROC values are
> single-sample artifacts and carry no statistical weight. Treat only the ten
> classes above as having measured performance.

Per-class metrics: `test_metrics.json` (teacher), `edge_test_metrics.json` (INT8).

## Thresholds and calibration

Decision thresholds were tuned per class on validation fold 9 and are stored in
the metrics JSON. **A fixed 0.5 threshold performs materially worse** on rare
classes — PAC scores F1 0.043 at 0.5 versus 0.444 at its tuned threshold.

This matters beyond raw F1. The CHW tier thresholds in `simplified_output`
express *clinical* confidence ("escalate at 60% sure"), which assumes a
calibrated model. This model is not calibrated that way — rare classes peak
well below 0.5, so raw probabilities compared against clinical thresholds
silently under-fire. Concretely, a STEMI at raw probability 0.38 — above its
0.35 operating point — reports **"low risk"** uncalibrated and **"urgent"**
once `load_class_thresholds` is active.

`load_class_thresholds` rescales each class piecewise-linearly so its operating
point maps to 0.5. The mapping is monotonic, so ranking and AUROC are
unchanged; it only restores the meaning the clinical thresholds already assume.

## Training

- **Data:** PTB-XL 1.0.3 (PhysioNet, CC BY 4.0; Wagner et al. 2020), 21,799
  records, 100 Hz, 10-second windows. Standard splits: folds 1–8 train, 9
  validation, 10 test.
- **Teacher:** 30 epochs, AdamW, lr 1e-3, cosine decay with 5 warm-up epochs,
  batch 64, best epoch 19 by masked validation loss.
- **Student:** knowledge distillation (T=4.0, alpha=0.7), 30 epochs, best epoch 26.
- **Quantization:** static INT8 QDQ, 100 real ECGs from fold 9 as calibration.
- No proprietary data. Chapman-Shaoxing was **not** used in this release.

## Limitations

- 46 of 72 outputs untrained; the whole risk head is untrained.
- 16 of the 26 trained classes lack the test positives needed to validate them.
- Trained on a single-source German cohort (PTB-XL) — no external validation,
  and no evidence of generalization across populations, devices, or geographies.
  This matters especially for deployment contexts unlike the source cohort.
- Class imbalance is uncorrected: the loss supports per-class weights but not
  `pos_weight`, so rare classes are under-fit.
- 100 Hz only; ST-segment detail may benefit from 500 Hz.

## Citation

PTB-XL: Wagner, P., Strodthoff, N., Bousseljot, RD. et al. *PTB-XL, a large
publicly available electrocardiography dataset.* Sci Data 7, 154 (2020).
