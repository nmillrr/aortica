# STARD-AI Reporting Checklist — Aortica AI ECG Diagnostic Accuracy Study

## Document Control

| Field | Value |
|-------|-------|
| **Model Version** | [FILL: aortica model version] |
| **Date** | [FILL: document date] |
| **Author** | [FILL: lead author / team] |
| **Study Title** | [FILL: title of the diagnostic accuracy study] |

> **Reference**: Stable-Ford M, et al. "STARD-AI: Standards for Reporting of Diagnostic Accuracy Studies — Artificial Intelligence." BMJ. 2024.
>
> STARD-AI extends the STARD statement for diagnostic accuracy studies evaluating AI/ML-based diagnostic tools.

---

## Checklist Items

### Title and Abstract

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 1 | Identify the article as a study of diagnostic accuracy using AI | Title | [FILL: e.g. "Diagnostic accuracy of Aortica multi-task deep learning model for 12-lead ECG interpretation: a retrospective diagnostic accuracy study"] | [ ] |
| 2 | Provide a structured abstract including: objective, design, setting, participants, index test, reference standard, results, conclusions | Abstract | [FILL: structured abstract] | [ ] |

### Introduction

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 3 | Describe the clinical problem and rationale for the study | Introduction | ECG interpretation is critical for arrhythmia detection, ischaemia diagnosis, and risk stratification. Automated AI analysis can improve diagnostic consistency and reduce time-to-diagnosis, particularly in resource-limited settings. | [ ] |
| 4 | State the study objectives and hypotheses | Introduction | [FILL: e.g. "To evaluate the diagnostic accuracy of Aortica for detection of [specific conditions] compared to expert cardiologist interpretation as reference standard"] | [ ] |

### Methods — Study Design

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 5 | Describe the study design (prospective/retrospective, cross-sectional/cohort) | Methods | [FILL: study design, e.g. "Retrospective cross-sectional diagnostic accuracy study"] | [ ] |
| 6 | Describe participant recruitment (consecutive, random, convenience) | Methods | [FILL: recruitment method and period] | [ ] |

### Methods — Participants

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 7 | Describe the eligibility criteria | Methods | [FILL: inclusion/exclusion criteria for participants and ECG recordings] | [ ] |
| 8 | Describe where and when participants were identified | Methods | [FILL: auto-populated from benchmark report — e.g. "PTB-XL dataset: Schiller AG ECG recordings from Physikalisch-Technische Bundesanstalt, 1989–1996"] | [ ] |
| 9 | Report the number of eligible and enrolled participants, with reasons for exclusion (STARD flow diagram) | Methods | [FILL: auto-populated — total records, excluded records with reasons, final analysis set size] | [ ] |

### Methods — Index Test (Aortica AI)

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 10a | Describe the index test in sufficient detail to allow replication | Methods | Aortica v[FILL: version]: 1D ResNet backbone + cross-lead attention + 4 multi-task heads (rhythm/structural/ischaemia/risk). Full preprocessing pipeline: denoising → quality scoring → resampling to 500 Hz. Architecture: [IEC 80601-2-86 ATD § 1](IEC_80601_2_86_ATD.md#1-algorithm-description). | [ ] |
| 10b | Report AI-specific details: architecture, training data, hyperparameters, software version | Methods | Architecture: 1D ResNet (64/128/256) + 4-head attention (64-dim). Training: PTB-XL folds 1–8, cosine annealing + warmup, gradient clipping max_norm=1.0. Software: Python, PyTorch. Pre-trained checkpoint: HuggingFace Hub `nmillrr/aortica`. | [ ] |
| 10c | Describe the threshold(s) used to define a positive result | Methods | Default classification threshold: 0.5 (sigmoid output). Configurable per class. Sensitivity analysis at alternative thresholds recommended. | [ ] |
| 10d | Report whether the AI was applied with or without human oversight | Methods | [FILL: e.g. "Aortica operated autonomously without human input for the purpose of this accuracy study. In clinical deployment, all outputs are decision support requiring clinician review."] | [ ] |

### Methods — Reference Standard

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 11a | Describe the reference standard and its rationale | Methods | [FILL: reference standard, e.g. "Expert cardiologist interpretation (up to 2 per ECG) using SCP statement codes, as provided by PTB-XL dataset annotations"] | [ ] |
| 11b | Report the qualifications and experience of reference standard assessors | Methods | [FILL: assessor qualifications, e.g. "Board-certified cardiologists with >5 years ECG interpretation experience"] | [ ] |
| 12 | Report whether assessors of the reference standard were blinded to the index test results | Methods | [FILL: blinding status] | [ ] |

### Methods — Analysis

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 13a | Describe the statistical methods used to calculate diagnostic accuracy | Methods | Per-class: AUC (one-vs-rest), sensitivity, specificity, F1 at threshold 0.5. Macro-F1 (unweighted mean). Calibration: ECE (10 bins). Risk: C-index, Brier score. All computed by `aortica.evaluation.benchmark()`. | [ ] |
| 13b | Describe methods for estimating uncertainty of diagnostic accuracy measures | Methods | [FILL: auto-populated — e.g. "95% confidence intervals via bootstrap (1000 iterations) or DeLong method for AUC comparisons"] | [ ] |
| 13c | Describe any subgroup or sensitivity analyses planned | Methods | Demographic subgroup analysis: per-task AUC stratified by sex and age decile. Equity gating via `aortica.evaluation.equity_gate()` with Bonferroni correction. | [ ] |
| 14 | Report how missing data were handled | Methods | [FILL: missing data handling for both index test and reference standard] | [ ] |

### Results — Participants

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 15 | Report participant demographics and clinical characteristics | Results | [FILL: auto-populated — age distribution (mean, SD, range), sex distribution, prevalence of each condition in the study population] | [ ] |
| 16 | Report the STARD participant flow diagram | Results | [FILL: flow diagram showing identification, eligibility, enrolment, index test execution, reference standard execution, and final analysis] | [ ] |

### Results — Diagnostic Accuracy

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 17a | Report overall diagnostic accuracy with confidence intervals | Results | [FILL: auto-populated — macro-F1, mean AUC with 95% CI per task head from benchmark report] | [ ] |
| 17b | Report cross-tabulation (2×2 table) of index test results by reference standard for key conditions | Results | [FILL: 2×2 tables for primary conditions (AF, STEMI, LVSD, VT)] | [ ] |
| 18 | Report per-class diagnostic accuracy (sensitivity, specificity, PPV, NPV, AUC) | Results | [FILL: auto-populated — per-class metrics for all 72 outputs. See IEC 80601-2-86 ATD § 4.2–4.5] | [ ] |
| 19 | Report any subgroup analyses | Results | [FILL: auto-populated — per-task AUC by sex and age decile from equity gate report] | [ ] |
| 20 | Report any adverse events from the index test or reference standard | Results | [FILL: adverse events if applicable, or "Not applicable — non-invasive software analysis of existing ECG recordings"] | [ ] |

### Discussion

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 21 | Discuss the clinical applicability of the findings | Discussion | [FILL: clinical applicability, limitations, comparison to existing methods/tools] | [ ] |
| 22 | Discuss limitations including spectrum bias, verification bias, and generalisability | Discussion | Spectrum bias: PTB-XL represents a single German centre. Generalisability: European-biased cohort, limited rare arrhythmia examples. See [IEC 80601-2-86 ATD § 5](IEC_80601_2_86_ATD.md#5-known-limitations). | [ ] |
| 22b | Discuss AI-specific limitations: training data representativeness, concept drift risk, failure modes | Discussion | Training data limited to PTB-XL (European). Concept drift mitigated by performance monitoring. Failure modes: OOD inputs flagged by Mahalanobis distance. PDF-scanned ECGs capped at marginal quality. | [ ] |

### Other Information

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 23 | Report registration and protocol information | Other | [FILL: study registration (e.g. ClinicalTrials.gov, PROSPERO) and protocol availability] | [ ] |
| 24 | Report data and code availability | Other | Open source: Apache 2.0. Code: GitHub. Models: HuggingFace Hub. Data: PTB-XL (PhysioNet, CC BY 4.0). Benchmark reproducible via `aortica benchmark`. | [ ] |
| 25 | Report funding sources | Other | [FILL: funding sources and conflicts of interest] | [ ] |

---

*This checklist follows the STARD-AI reporting guideline for diagnostic accuracy studies of AI/ML tools. Items marked `[FILL: auto-populated ...]` can be partially filled using `aortica.regulatory.generate_reporting_checklist(template='stard_ai', benchmark_report=report)`. Cross-references to [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md) are provided where applicable.*
