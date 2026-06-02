# TRIPOD-AI Reporting Checklist — Aortica AI ECG Analysis

## Document Control

| Field | Value |
|-------|-------|
| **Model Version** | [FILL: aortica model version] |
| **Date** | 2026-06-02 |
| **Author** | [FILL: lead author / team] |
| **Study Title** | [FILL: title of the manuscript or report] |

> **Reference**: Collins GS, et al. "Protocol for development of a reporting guideline (TRIPOD-AI) and risk of bias tool (PROBAST-AI) for diagnostic and prognostic prediction model studies based on artificial intelligence." BMJ Open. 2021.
>
> TRIPOD-AI extends the TRIPOD statement for studies developing, validating, or updating prediction models that use AI/ML methods.

---

## Checklist Items

### Title and Abstract

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 1a | Identify the study as developing and/or validating a multivariable prediction model, the target population, and the outcome to be predicted | Title | [FILL: e.g. "Development and validation of a multi-task deep learning model for 12-lead ECG analysis"] | [ ] |
| 1b | Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions | Abstract | [FILL: structured abstract] | [ ] |

### Introduction

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 2 | Provide the medical context and rationale for developing or validating the prediction model, including references to existing models | Introduction | Aortica addresses gaps in clinical ECG: poor device generalisation, narrow single-task models, black-box outputs, and exclusion of LMIC settings. See PRD Introduction for full context. | [ ] |
| 3a | Specify the objectives, including whether the study describes the development or validation (or both) of the prediction model | Introduction | [FILL: e.g. "To develop and internally validate a multi-task deep learning model for simultaneous rhythm classification (28 classes), structural screening (19 classes), ischaemia detection (19 classes), and risk prediction (6 outputs) from 12-lead ECGs"] | [ ] |
| 3b | Specify the AI-specific objectives: model architecture selection rationale, comparison with non-AI baselines if applicable | Introduction | 1D ResNet backbone with cross-lead attention selected for temporal pattern capture and inter-lead reasoning. Architecture described in [IEC 80601-2-86 ATD § 1](IEC_80601_2_86_ATD.md#1-algorithm-description). | [ ] |

### Methods — Source of Data

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 4a | Describe the study design (e.g. retrospective cohort, registry) | Methods | Retrospective cohort study using PTB-XL dataset (Wagner et al. 2020, PhysioNet). | [ ] |
| 4b | Specify the key study dates (start, end, follow-up) | Methods | [FILL: study period dates] | [ ] |
| 5a | Describe the study setting (e.g. primary care, hospital) and relevant data sources | Methods | [FILL: auto-populated from benchmark report — PTB-XL: single German centre, 21,799 10-second 12-lead ECGs, CC BY 4.0] | [ ] |
| 5b | Specify the computing environment and hardware used for model training and validation | Methods | [FILL: e.g. "Training: NVIDIA A100 GPU, PyTorch 2.x. Validation: CPU-only (reproducibility). Edge: Raspberry Pi 4, ONNX Runtime ARM64."] | [ ] |

### Methods — Participants

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 6a | Specify the eligibility criteria (inclusion and exclusion) | Methods | [FILL: e.g. "All 12-lead ECG recordings in PTB-XL with valid SCP statement labels. No exclusion criteria applied."] | [ ] |
| 6b | Describe any data cleaning, labelling, or annotation procedures, including inter-annotator agreement | Methods | PTB-XL labels assigned by up to 2 cardiologists per record. SCP statement codes mapped to Aortica's four-domain taxonomy. | [ ] |

### Methods — Outcome

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 7a | Clearly define the outcome, how it was measured, and the time horizon for risk prediction | Methods | Classification: multi-label per task head (rhythm/structural/ischaemia). Risk: 1-year mortality, 12-month HF hospitalisation, 12-month AF onset, EF estimation, conduction disease trajectory, SCD risk. | [ ] |
| 7b | Report whether outcome assessors were blinded to predictor information | Methods | [FILL: blinding status of PTB-XL annotators] | [ ] |

### Methods — Predictors / Features

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 8a | Describe all predictors (features) used in the model, including how and when they were measured | Methods | Input: raw 12-lead ECG signal (shape [12, samples], 500 Hz). Preprocessing: denoising (baseline wander, powerline, HF), quality scoring, resampling. No hand-crafted features — end-to-end deep learning. | [ ] |
| 8b | Describe the AI-specific feature engineering or representation learning approach | Methods | 1D ResNet backbone extracts hierarchical features at 64/128/256 filter widths. Cross-lead attention (4 heads, 64-dim) captures inter-lead relationships. No manual feature selection. | [ ] |

### Methods — Sample Size

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 9 | Explain how the study size was determined, including any power calculations | Methods | [FILL: auto-populated — PTB-XL full dataset: ~21,799 records. Split: folds 1-8 train (~17,440), fold 9 validation (~2,180), fold 10 test (~2,180). Rare class N reported per class.] | [ ] |

### Methods — Missing Data

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 10 | Describe how missing data were handled and report the amount of missing data for predictors and outcome | Methods | [FILL: missing data handling — e.g. "No missing signal data (ECG signals are complete). Missing demographic metadata: age missing in X%, sex missing in Y%. Missing demographics handled by exclusion from subgroup analysis."] | [ ] |

### Methods — Model Development

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 11a | Describe the full model specification, including all hyperparameters and their selection | Methods | Architecture: 1D ResNet (64/128/256 blocks) + 4-head cross-lead attention + 4 task heads. Hyperparameters: cosine annealing with warmup, gradient clipping max_norm=1.0, per-task loss weights (configurable). Full spec in training config YAML. | [ ] |
| 11b | Describe the AI model architecture, including depth, width, activation functions, and any architectural innovations | Methods | ResNet residual blocks with ReLU, adaptive average pooling, multi-head attention with extractable weights. Task heads: sigmoid (classification), sigmoid-scaled regression (risk). | [ ] |
| 11c | Report the model training procedure: optimiser, learning rate schedule, epochs, batch size, convergence criteria | Methods | [FILL: auto-populated — optimizer, LR schedule, epochs trained, batch size, early stopping criteria, best checkpoint selection metric] | [ ] |
| 11d | Specify data augmentation techniques used | Methods | Random lead dropout, Gaussian noise injection, time-shift, amplitude scaling. Configurable per training run. | [ ] |

### Methods — Model Performance

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 12 | Specify the measures used to evaluate model performance and how they were calculated | Methods | Classification: per-class AUC, sensitivity, specificity, F1, macro-F1, ECE. Risk: C-index, Brier score. All computed by `aortica.evaluation.benchmark()`. See [IEC 80601-2-86 ATD § 6.2](IEC_80601_2_86_ATD.md#62-metric-definitions). | [ ] |
| 12b | Report calibration assessment methodology | Methods | Temperature scaling per task head, ECE with 10 bins, reliability diagrams. Conformal prediction at 90% coverage. | [ ] |

### Methods — Statistical Analysis

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 13a | Describe the validation approach (e.g. internal, external, temporal) | Methods | Internal validation: PTB-XL fold 10 held-out test set (recommended split, never used for training or validation). [FILL: external validation datasets if applicable]. | [ ] |
| 13b | Report any fairness or equity analyses performed | Methods | Equity gating via `aortica.evaluation.equity_gate()`: sex and age decile comparison with Bonferroni correction. No release if significant demographic gap (p<0.05 corrected) for classes with N>100. | [ ] |

### Results — Participants

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 14a | Report the number of participants and events in each analysis | Results | [FILL: auto-populated — N total, N per split, N per demographic subgroup, N per class] | [ ] |
| 14b | Describe participant flow, including any exclusions | Results | [FILL: participant flow diagram or description] | [ ] |

### Results — Model Performance

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 15a | Report overall model performance with confidence intervals | Results | [FILL: auto-populated — macro-F1, mean AUC with 95% CI per task head] | [ ] |
| 15b | Report per-class performance metrics | Results | [FILL: auto-populated — per-class AUC, sensitivity, specificity, F1 for all 72 outputs. See IEC 80601-2-86 ATD § 4.2–4.5] | [ ] |
| 15c | Report calibration results (ECE, reliability diagrams) | Results | [FILL: auto-populated — per-task ECE. See IEC 80601-2-86 ATD § 4.6] | [ ] |
| 15d | Report demographic subgroup performance | Results | [FILL: auto-populated — per-task AUC by sex and age decile from equity gate report] | [ ] |

### Discussion

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 16 | Discuss limitations of the study (e.g. non-representative study population, limited sample size, missing data) | Discussion | Known limitations: European-biased training data (single German centre), limited rare arrhythmia examples, risk labels from proxy derivation, PDF-scanned ECGs capped at marginal quality. See [IEC 80601-2-86 ATD § 5](IEC_80601_2_86_ATD.md#5-known-limitations). | [ ] |
| 17 | Discuss implications for clinical practice and future research | Discussion | [FILL: clinical implications, deployment considerations, federated learning for multi-site improvement, prospective validation plans] | [ ] |

### Other Information

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 18 | Provide information about the availability of data, code, and the AI model | Other | Open source: Apache 2.0 license. Code: GitHub. Pre-trained models: HuggingFace Hub (`nmillrr/aortica`). Training data: PTB-XL (PhysioNet, CC BY 4.0). No proprietary data used. | [ ] |
| 19 | Provide the source of funding and the role of funders | Other | [FILL: funding sources and role] | [ ] |

---

*This checklist follows the TRIPOD-AI reporting guideline for AI/ML prediction model studies. Items marked `[FILL: auto-populated ...]` can be partially filled using `aortica.regulatory.generate_reporting_checklist(template='tripod_ai', benchmark_report=report)`. Cross-references to [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md) are provided where applicable.*
