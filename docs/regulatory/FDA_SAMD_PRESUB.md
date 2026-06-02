# FDA Software as a Medical Device (SaMD) Pre-Submission Document

## Document Control

| Field | Value |
|-------|-------|
| **Document Version** | [FILL: document version, e.g. 1.0] |
| **Model Version** | [FILL: aortica model version] |
| **Date** | [FILL: document date] |
| **Author** | [FILL: responsible person / team] |
| **Status** | [FILL: draft / in-review / approved] |
| **Submission Type** | Pre-Submission (Q-Sub) |
| **Product Code** | [FILL: applicable FDA product code, e.g. QMT for electrocardiograph analysis software] |
| **Regulation Number** | [FILL: 21 CFR part, e.g. 870.2340] |

---

## 1. Device Description

### 1.1 Product Overview

Aortica is an open-source, multi-task deep learning software system for 12-lead electrocardiogram (ECG) analysis. It performs simultaneous classification and risk prediction across four clinical task domains:

- **Rhythm & Conduction** (28 classes): detection of arrhythmias, conduction abnormalities, and rare rhythm disorders including AF, VT, VF, Brugada, WPW, LBBB/RBBB, and AV blocks
- **Structural & Functional** (19 classes): screening for structural heart disease including LVH, RVH, LVSD, cardiomyopathies, strain patterns, and pericarditis/myocarditis
- **Ischaemia & Metabolic** (19 classes): detection of ischaemic patterns (STEMI, NSTEMI, STEMI mimics), metabolic emergencies (hyperkalaemia), and drug effects (digoxin, TCA toxicity)
- **Risk Prediction** (6 continuous outputs): prognostic scoring for mortality, heart failure hospitalization, AF onset, ejection fraction estimation, conduction disease trajectory, and sudden cardiac death risk

### 1.2 Software Architecture

The system employs a shared-backbone deep learning architecture:

1. **Backbone Encoder** (`AorticaBackbone`): 1D ResNet with residual blocks at 64, 128, 256 filter widths; adaptive pooling for variable input lengths (250–1000 Hz, 2.5–10s windows)
2. **Cross-Lead Temporal Attention** (`CrossLeadAttention`): 4-head, 64-dim multi-head attention capturing inter-lead relationships for axis, ischaemia territory, and conduction pattern reasoning
3. **Task Heads**: Four independent classification/regression heads connected to the shared representation
4. **Post-Hoc Calibration**: Temperature scaling per task head for calibrated probability outputs
5. **Uncertainty Estimation**: Conformal prediction sets at configurable coverage level with Mahalanobis OOD detection

### 1.3 Software Level of Concern

[FILL: Level of Concern determination — typically "Major" for ECG diagnostic decision support. Justify based on FDA SaMD guidance "Software as a Medical Device: Possible Framework for Risk Categorization and Corresponding Considerations" and IMDRF categorization.]

### 1.4 Distribution Model

Aortica is a **self-hosted, open-source toolkit** distributed via `aortica.io`. Clinicians and institutions download and run it locally (Docker or pip install). No patient data is transmitted to external servers. All inference is performed on-site. This architecture:

- Preserves patient privacy by design
- Eliminates recurring infrastructure costs
- Enables deployment in data-sovereignty-constrained settings (HIPAA, GDPR)
- Supports intermittent-connectivity and offline environments (rural clinics, LMIC sites)

### 1.5 Deployment Variants

| Variant | Platform | Model | Latency Target |
|---------|----------|-------|----------------|
| **Server** | amd64 workstation/server | Full PyTorch | [FILL: measured latency] |
| **Edge** | Raspberry Pi 4+ (arm64) | INT8 ONNX | ≤ 350 ms |
| **Browser (PWA)** | Any modern browser | INT8 ONNX via WebAssembly | [FILL: measured latency] |

---

## 2. Intended Use Statement

### 2.1 Indications for Use

[FILL: indications for use statement, e.g. "Aortica is intended to assist qualified healthcare professionals in the interpretation of standard 12-lead resting electrocardiograms (ECGs) by providing AI-generated multi-task decision support findings across rhythm analysis, structural screening, ischaemia detection, and risk prediction. Aortica is intended to be used as a decision support tool and does not replace clinical judgment. The device is intended for use by qualified healthcare professionals trained in ECG interpretation."]

### 2.2 Intended Use Population

[FILL: intended patient population, e.g. "Adults (≥18 years) undergoing standard 12-lead resting ECG recording in clinical settings including hospitals, clinics, and field deployment sites."]

### 2.3 Intended Use Environment

[FILL: use environment, e.g. "Hospital, outpatient clinic, community health centre, or field deployment site equipped with standard 12-lead ECG hardware and a workstation, laptop, or Raspberry Pi running Docker or pip-installed Aortica."]

### 2.4 Intended User Profile

[FILL: user profiles, e.g. "Qualified healthcare professionals including cardiologists, emergency physicians, internists, primary care physicians, and trained clinical staff competent in ECG interpretation. For LMIC deployment: community health workers (CHWs) using simplified 3-tier output mode under physician supervision."]

### 2.5 SaMD Categorisation (IMDRF Framework)

| Dimension | Value |
|-----------|-------|
| **Significance of Information** | [FILL: "Treat or diagnose" / "Drive clinical management" / "Inform clinical management"] |
| **Seriousness of Health Condition** | [FILL: "Critical" / "Serious" / "Non-serious"] |
| **SaMD Category** | [FILL: I / II / III / IV per IMDRF N12] |

---

## 3. Technological Characteristics

### 3.1 Deep Learning Architecture

The core model is a 1D convolutional neural network with residual connections (ResNet-style) and multi-head cross-lead attention. Full architectural details are documented in [IEC 80601-2-86 ATD § 1](IEC_80601_2_86_ATD.md#1-algorithm-description).

| Parameter | Value |
|-----------|-------|
| **Input** | 12-lead ECG signal, shape `[batch, 12, samples]` |
| **Sample Rates** | 250–1000 Hz (adaptive pooling) |
| **Window Length** | 2.5–10 seconds |
| **Total Parameters** | [FILL: parameter count for full model] |
| **Edge Parameters** | ≤ 2.5M (MobileNet-1D backbone) |
| **Output** | 72 values (28 rhythm + 19 structural + 19 ischaemia + 6 risk) |
| **Frameworks** | PyTorch (primary), TensorFlow/Keras (parity-validated) |

### 3.2 Signal Preprocessing Pipeline

1. **Format Ingestion**: Universal reader supporting WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, PDF/image scan
2. **Signal Denoising**: Baseline wander removal (0.5 Hz highpass), powerline notch (50/60 Hz auto-detected), high-frequency lowpass (40 Hz)
3. **Quality Assessment**: Per-lead quality scoring (0–100) with accept/review/reject recommendation. PDF/image-scanned ECGs are automatically capped at "marginal" quality (≤69/100)
4. **Resampling**: To configurable target rate (default 500 Hz)
5. **Lead Normalisation**: Standard 12-lead ordering (I, II, III, aVR, aVL, aVF, V1–V6)

### 3.3 Calibration and Uncertainty

- **Temperature Scaling**: Post-hoc single-parameter calibration per task head, optimised on validation set NLL
- **Conformal Prediction**: Prediction sets at user-specified coverage level (default 90%)
- **OOD Detection**: Mahalanobis distance on backbone features; flags inputs beyond configurable percentile threshold
- **Reliability Diagrams**: Expected Calibration Error (ECE) computed per task head; see [IEC 80601-2-86 ATD § 4.6](IEC_80601_2_86_ATD.md#46-calibration)

### 3.4 Explainability (XAI)

- **Integrated Gradients**: Per-lead gradient attributions mapped to named ECG segments (P wave, PR interval, QRS complex, ST segment, T wave, QT/QTc)
- **Feature Attribution**: Top-3 contributing features per active diagnosis with delta-contribution scores
- **VAE Latent Factor Model**: 24-dimensional variational autoencoder on median beats with clinically labelled latent dimensions
- **Case-Based Retrieval**: Top-3 phenotypically similar historical ECGs from de-identified PhysioNet corpus

### 3.5 Edge Model Optimisation

- **Knowledge Distillation**: MobileNet-1D student trained from full ResNet teacher (KL divergence + hard label loss)
- **INT8 Quantisation**: ONNX Runtime static quantisation with calibration; quantised model validated to perform within 3% AUC of full model
- **Offline PWA**: Service worker caches ONNX model for in-browser inference via ONNX Runtime Web (WebAssembly)

---

## 4. Performance Benchmark Plan

### 4.1 Primary Dataset

| Parameter | Value |
|-----------|-------|
| **Dataset** | PTB-XL |
| **Version** | 1.0.3 |
| **License** | CC BY 4.0 (Wagner et al. 2020, PhysioNet) |
| **Total Records** | ~21,799 10-second 12-lead ECGs |
| **Split** | Folds 1–8 train, fold 9 validation, fold 10 test (recommended split) |
| **Data Provenance** | No proprietary data used. All training on public PhysioNet datasets. |

### 4.2 Evaluation Harness

Aortica provides a standardised, deterministic benchmark harness: `aortica.evaluation.benchmark()`. See [IEC 80601-2-86 ATD § 6](IEC_80601_2_86_ATD.md#6-test-methodology) for full methodology.

**Key properties:**
- Fixed random seed for reproducibility (identical model + dataset + seed → identical results)
- Held-out test set (PTB-XL fold 10, never used for training or validation)
- Full pipeline evaluation (raw ECG → preprocessing → inference → metrics)

### 4.3 Performance Metrics

| Metric | Definition | Task Type |
|--------|-----------|-----------|
| **AUC** | Area under ROC curve (one-vs-rest) | Classification |
| **Sensitivity** | True positive rate at threshold 0.5 | Classification |
| **Specificity** | True negative rate at threshold 0.5 | Classification |
| **Macro-F1** | Unweighted mean of per-class F1 scores | Classification |
| **ECE** | Expected Calibration Error (10 bins) | Classification |
| **C-index** | Concordance index for ranking accuracy | Risk prediction |
| **Brier Score** | Mean squared error of predicted probabilities | Risk prediction |

### 4.4 Performance Targets

| Condition / Task | Metric | Target |
|------------------|--------|--------|
| Overall Rhythm | Macro-F1 | ≥ 0.90 |
| AF Detection | AUC | ≥ 0.95 |
| STEMI Detection | Sensitivity | ≥ 0.90 |
| LVSD Screening | AUC | ≥ 0.88 |
| Edge Model (all tasks) | AUC degradation | ≤ 3% vs. full model |
| [FILL: additional site-specific performance targets] |

### 4.5 Equity Gating

Aortica enforces demographic equity at release time via `aortica.evaluation.equity_gate()`:

- **Sex comparison**: Per-task AUC compared across male/female subgroups using permutation test with Bonferroni correction
- **Age comparison**: Per-task AUC compared across age deciles (30–80)
- **Pass criteria**: No statistically significant performance gap (Bonferroni-corrected p<0.05) for any class with N>100 test examples
- **CI enforcement**: GitHub Actions blocks release on equity gate failure

See [IEC 80601-2-86 ATD § 5.2](IEC_80601_2_86_ATD.md#52-demographic-limitations) for demographic limitation details.

### 4.6 Benchmark Report References

Performance evidence is generated automatically by Aortica's CI/CD pipeline:

| Evidence | Source | Description |
|----------|--------|-------------|
| **Full Benchmark Report** | `aortica.evaluation.benchmark()` | Per-task, per-class metrics with demographic stratification |
| **Equity Gate Report** | `aortica.evaluation.equity_gate()` | Demographic fairness analysis with statistical testing |
| **Performance Card** | `aortica.evaluation.generate_performance_card()` | Public-facing markdown + CSV summary |
| **Edge Validation Report** | `aortica.edge.validate_edge()` | Full vs. edge model comparison per task |
| **Calibration Analysis** | Temperature scaling ECE | Per-task expected calibration error |

---

## 5. Predicate Comparison

### 5.1 Predicate Device(s)

[FILL: identify predicate device(s) for 510(k) pathway, or state "De Novo classification" if no suitable predicate. Examples of potential predicates:
- Eko ATRIA (DEN200062) — AI ECG analysis for AF detection
- AliveCor KardiaMobile (K143734) — AI-based ECG rhythm analysis
- Tempus Pro (K192004) — ECG monitoring with algorithm
- Caption Health (DEN200043) — AI-guided diagnostic ultrasound]

### 5.2 Comparison Table

| Feature | Aortica | [FILL: Predicate 1] | [FILL: Predicate 2] |
|---------|---------|---------------------|---------------------|
| **Intended Use** | Multi-task ECG decision support | [FILL: predicate intended use] | [FILL: predicate intended use] |
| **ECG Lead Count** | 12-lead | [FILL: lead count] | [FILL: lead count] |
| **Number of Conditions** | 72 (28 rhythm + 19 structural + 19 ischaemia + 6 risk) | [FILL: condition count] | [FILL: condition count] |
| **AI Architecture** | 1D ResNet + cross-lead attention | [FILL: architecture] | [FILL: architecture] |
| **Calibration** | Temperature scaling + conformal prediction | [FILL: calibration method] | [FILL: calibration method] |
| **Explainability** | Integrated gradients + named ECG features | [FILL: XAI approach] | [FILL: XAI approach] |
| **Edge Deployment** | INT8 ONNX (ARM64, WebAssembly) | [FILL: edge capability] | [FILL: edge capability] |
| **Open Source** | Yes (Apache 2.0) | [FILL: licensing] | [FILL: licensing] |
| [FILL: additional comparison dimensions] |

### 5.3 Substantial Equivalence Justification

[FILL: narrative justification for substantial equivalence (510(k)) or explanation of why De Novo is appropriate. Address: same intended use, same technological characteristics or different characteristics that do not raise new safety/effectiveness questions.]

---

## 6. Software Development Lifecycle Summary

### 6.1 Standards Compliance

| Standard | Applicability |
|----------|--------------|
| **IEC 62304** | Software lifecycle processes |
| **IEC 62366** | Usability engineering |
| **ISO 14971** | Risk management |
| **IEC 80601-2-86** | Algorithm testing documentation |
| **FDA AI/ML SaMD Guidance** | Good Machine Learning Practice (GMLP) |

See [IEC 80601-2-86 ATD § 8](IEC_80601_2_86_ATD.md#8-software-development-lifecycle) for SDLC details.

### 6.2 Development Process

- **Version Control**: Git (GitHub) with branching model and pull request reviews
- **CI/CD**: GitHub Actions — automated lint (`ruff`), type check (`mypy`), and test (`pytest`) on every push/PR
- **Code Quality**: Minimum test coverage ≥ 80% for core modules
- **Release Pipeline**: Full benchmark → equity gate → performance card → ONNX export → INT8 quantisation → edge validation → artifact upload to HuggingFace Hub

### 6.3 Change Management

- All changes via pull request with mandatory code review
- Breaking changes require version bump and migration documentation
- Model architecture changes trigger full re-benchmark and equity gate
- [FILL: additional change management procedures per IEC 62304 requirements]

### 6.4 Algorithm Change Protocol (PCCP)

Per FDA's "Predetermined Change Control Plan" framework:

- **Modifications Scope**: [FILL: describe modifications covered by PCCP — e.g. retraining with additional data, class weight adjustments, threshold tuning]
- **Performance Guardrails**: Regulatory gate enforces minimum per-class performance targets (see § 4.4). Any retraining that falls below targets is automatically blocked.
- **Verification Protocol**: [FILL: describe verification steps for each modification type]
- **Reporting**: Quarterly public performance reports generated automatically by `aortica.validation.generate_quarterly_report()`

---

## 7. Pre-Submission Questions

[FILL: specific questions for FDA pre-submission meeting. Typical questions include:

1. Does the FDA agree with the proposed classification and regulatory pathway (510(k) vs. De Novo) for a multi-task ECG AI decision support tool?
2. Does the FDA agree with the proposed performance benchmark plan using PTB-XL as the primary evaluation dataset? Are additional datasets required?
3. Does the FDA agree with the proposed equity gating methodology (Bonferroni-corrected permutation tests across sex and age subgroups) for demonstrating demographic fairness?
4. What clinical evidence (if any) is expected beyond analytical performance testing — e.g. prospective multi-site clinical study, or retrospective reader study?
5. Does the FDA agree that the Predetermined Change Control Plan scope (retraining with additional public data, threshold adjustments) is appropriate for future modifications?
6. For the edge-deployed variant (INT8 ONNX on ARM64), does the FDA consider this a separate device variant requiring independent validation, or is the edge validation report (≤3% AUC degradation) sufficient?
7. Does the FDA agree with the proposed open-source distribution model (self-hosted, no cloud dependency) and its implications for post-market surveillance?]

---

## 8. Cybersecurity Considerations

### 8.1 Architecture Security

| Aspect | Implementation |
|--------|---------------|
| **Data at Rest** | AES-256 encryption (Fernet) for local result storage |
| **Data in Transit** | HTTPS for API, TLS for gRPC, encrypted sync engine |
| **Authentication** | OAuth 2.0 (Google, GitHub) + API key authentication |
| **Authorisation** | JWT-based role access, protected API endpoints |
| **Federated Learning** | CKKS homomorphic encryption for gradient exchange, differential privacy (ε=1.0 default) |
| **Model Integrity** | SHA-256 checksum verification on pre-trained model download |
| **Network Isolation** | Self-hosted deployment — no mandatory external data flows |

### 8.2 Threat Model

[FILL: threat model per FDA Premarket Cybersecurity Guidance (2023). Address: SBOM (Software Bill of Materials), vulnerability management, update mechanism, incident response.]

---

## 9. Labelling

### 9.1 Software Labelling

All Aortica outputs include the following labelling:

- **"AI Decision Support — Requires Clinical Review"** watermark on all PDF reports
- **"Decision support only — requires clinician judgment"** disclaimer on all copilot findings
- Model version, timestamp, and inference mode (server/edge) on all outputs
- Signal quality warning for PDF/image-scanned ECGs

### 9.2 Instructions for Use

[FILL: reference to instructions for use (IFU) document. IFU should cover: installation, initial setup, intended use, contraindications, warnings, output interpretation, troubleshooting.]

---

## Appendices

### Appendix A: Performance Evidence References

| Document | Location |
|----------|----------|
| IEC 80601-2-86 ATD | [IEC_80601_2_86_ATD.md](IEC_80601_2_86_ATD.md) |
| CE-MDR Technical File | [CE_MDR_TECHFILE.md](CE_MDR_TECHFILE.md) |
| Full Benchmark Report | [FILL: path to benchmark report output] |
| Equity Gate Report | [FILL: path to equity gate report output] |
| Performance Card | [FILL: path to public performance card] |
| Edge Validation Report | [FILL: path to edge validation report] |

### Appendix B: Class List (72 Outputs)

See [IEC 80601-2-86 ATD Appendix A](IEC_80601_2_86_ATD.md) for the complete list of all 72 output classes with clinical definitions.

### Appendix C: Software Bill of Materials (SBOM)

[FILL: generate SBOM from pyproject.toml and package-lock.json covering all runtime dependencies]

---

*This template follows FDA Pre-Submission (Q-Sub) guidance for Software as a Medical Device. Sections marked with `[FILL: ...]` require site-specific information. Performance evidence is generated automatically by Aortica's benchmark harness and equity gating CI pipeline. Cross-references to [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md) and [CE-MDR Technical File](CE_MDR_TECHFILE.md) are provided where applicable.*
