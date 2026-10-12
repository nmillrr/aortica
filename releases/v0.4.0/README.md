---
license: cc-by-4.0
tags: [ecg, electrocardiography, medical, time-series, pytorch]
library_name: pytorch
---

# Aortica v0.4.0 — reference 12-lead ECG model

The model the Aortica whitepaper reports: a ResNet-1D backbone with
cross-lead attention and multi-label task heads, trained on PTB-XL and
Chapman-Shaoxing with a fixed seed and a saved split, evaluated on both
corpora's held-out test sets and externally on the Georgia cohort, and
compared against the published Ribeiro et al. (2020) model.

> [!CAUTION]
> **Research use only. Not a medical device.** Not FDA-cleared or CE-marked;
> not validated for diagnosis, treatment or triage.

**Results: [`RESULTS.md`](RESULTS.md)** (generated from the JSON files here).

## Key findings

Numbers from [`RESULTS.md`](RESULTS.md); macro-AUROC over classes with ≥10 positives.

- **Internal:** 0.941 [0.929, 0.951] on PTB-XL fold 10, 0.975 [0.970, 0.979]
  on the Chapman test split — in line with v0.3.0's published 0.943 / 0.969.
- **External (Georgia, 10,338 ECGs):** 0.894 [0.885, 0.902]. The drop is
  consistent across sex and age bands (0.89–0.90).
- **Baseline:** on the six findings shared with Ribeiro et al. (2020), mean
  Georgia AUROC is 0.967 for both. Aortica leads on sinus bradycardia
  (+0.043); Ribeiro leads by 0.001–0.014 on 1st-degree AV block, RBBB, LBBB
  and sinus tachycardia; AF is tied.
- **Label semantics matter.** Aortica's RBBB is far less precise than
  Ribeiro's on Georgia (AUPRC 0.53 vs 0.81) because PTB-XL's training label
  includes incomplete RBBB; counting incomplete blocks as positive raises
  Georgia RBBB AUROC from 0.949 to 0.955.
- **Operating points do not transfer for every class.** Validation-tuned
  thresholds keep high specificity on Georgia but sensitivity falls sharply
  for QTc prolongation (0.11), LA enlargement (0.08) and early
  repolarisation (0.06), although AUROC stays 0.69–0.85. Site-level
  recalibration would be needed before any threshold-based use.
- **Pooled thresholds favour one corpus.** Thresholds are tuned on the
  pooled validation set, so some classes score well on one corpus and badly
  on the other at the same threshold (Chapman AF F1 0.09 at AUROC 0.93;
  Chapman RBBB F1 0.13 at AUROC 0.99).
- **Quantization costs more out of distribution.** The v0.3.0 INT8 edge model
  scores 0.860 on Georgia vs 0.895 for its FP32 counterpart.

## Files

| File | Contents |
|---|---|
| `config.yaml` | Every hyperparameter, the seed, thread count and data roots |
| `splits.csv` | Every record considered: corpus, id, PTB-XL patient id, split, and `kept` / `excluded_amplitude_gt_25mV` / `dropped_by_loader` |
| `run_manifest.json` | Git commit, library versions, platform, data checksums, record counts, timings, SHA-256 of the weights and of `splits.csv` |
| `training_history.json` | Per-epoch losses and macro-F1 |
| `class_thresholds.json` | Per-class operating points (validation-tuned) |
| `trained_outputs.json` | The 28 trained outputs and per-task loss mask |
| `test_metrics.json` | Held-out metrics per corpus and class, with 95% CIs |
| `evaluation/external_georgia.json` | External validation (this model and the v0.3.0 edge models) |
| `evaluation/baseline_ribeiro.json` | Paired comparison with Ribeiro et al. (2020) |
| `aortica_v0.4.0.pt` | Weights — **not committed**; regenerate with the command below and check against the manifest's SHA-256 |

## Reproduce

```bash
python scripts/train_reference.py --config releases/v0.4.0/config.yaml
```

About four hours on an 8-thread x86 CPU. Determinism: fixed seeds for
Python/NumPy/PyTorch and the DataLoader, `torch.use_deterministic_algorithms`,
a fixed thread count and single-process data loading. A second full run on the same
machine reproduced the weights bit for bit (same `state_dict_sha256` and
`splits.csv` hash); on other CPUs,
floating-point reduction order can differ, so expect metric agreement rather
than identical bytes.

## Data

| Corpus | Train | Val | Test | Split |
|---|---:|---:|---:|---|
| PTB-XL 1.0.3 | 17,418 | 2,183 | 2,198 | Published patient-disjoint folds 1–8 / 9 / 10 |
| Chapman-Shaoxing 1.0.0 | 35,987 | 4,504 | 4,507 | Seeded (42) record-level 80/10/10; each record is a distinct patient |

154 Chapman records are excluded and listed in `splits.csv`: 99 the loader
cannot use (unreadable header, a lead of non-finite samples, wrong lead
count) and 55 with peak amplitude above 25 mV, which are acquisition faults
(PTB-XL's maximum is ~20 mV) and drive the loss to NaN.

Inputs are 12 leads at **100 Hz**, a **10 s** window, in **µV**, unfiltered —
the contract in `aortica/signal/preprocess.py` that every inference path uses.

Labels come from `aortica/data/label_maps/ptbxl_scp.yaml` and
`chapman_snomed.yaml`. Each record carries a per-output mask: a column its
corpus cannot label contributes no loss, rather than counting as a negative.

## Training

AdamW (lr 1e-3, weight decay 1e-4), batch 64, 30 epochs, linear warm-up for
5 epochs then cosine decay, gradient clipping at 1.0, masked BCE per head
(risk head weight 0). Augmentation: lead dropout, time shift, amplitude
scaling, Gaussian noise. Checkpoint selected by masked validation loss.
Operating points: per class, the F1-maximising threshold on a 0.05 grid over
the pooled validation set (rows its corpus can label).

## Outputs

**Trained (28):** AF, AFL, SVT, AVRT, sinus_brady, sinus_tachy, PAC, PVC,
av_block_1st, av_block_2nd, av_block_3rd, LBBB, RBBB, LAFB, LPFB, WPW,
pacemaker_rhythm, normal_sinus_rhythm, LVH, RVH, LA_enlargement,
RA_enlargement, STEMI, posterior_MI, old_MI, QTc_prolongation,
digitalis_effect, early_repol_vs_STEMI.

**Untrained (44), suppressed by default:** everything else, including the
whole risk head. They sit near initialisation and must never be shown.

## Limitations

- Several trained classes have too few test positives to be validated; read
  `n_pos` before trusting a class.
- Label semantics differ across corpora: PTB-XL's RBBB/LBBB include
  incomplete blocks, Chapman's and Georgia's do not; PTB-XL's STEMI is
  derived from injury-pattern statements. See the label-map notes.
- Operating points are tuned in-distribution; on external data, AUROC is the
  robust comparison and sensitivity/specificity at these thresholds can
  shift.
- No edge (distilled/INT8) variant of this model exists; the shipped edge
  model is v0.3.0.

## Citation

PTB-XL — Wagner P. et al. *Sci Data* 7, 154 (2020).
Chapman-Shaoxing — Zheng J. et al. *Sci Data* 7, 48 (2020).
Georgia — Reyna M.A. et al., PhysioNet/CinC Challenge 2021.
Baseline — Ribeiro A.H. et al. *Nat Commun* 11, 1760 (2020).
