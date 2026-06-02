# CE-MDR Technical File — Aortica AI ECG Analysis Software

## Document Control

| Field | Value |
|-------|-------|
| **Document Version** | [FILL: document version, e.g. 1.0] |
| **Model Version** | [FILL: aortica model version] |
| **Date** | [FILL: document date] |
| **Author** | [FILL: responsible person / team] |
| **Status** | [FILL: draft / in-review / approved] |
| **UDI-DI** | [FILL: Unique Device Identifier — Device Identifier] |
| **EUDAMED SRN** | [FILL: Single Registration Number once registered] |
| **Risk Class** | [FILL: Class I / IIa / IIb / III per MDR Annex VIII Rule 11] |

---

## 1. Device Description and Specification

### 1.1 General Description

Aortica is an open-source, standalone AI software system for automated analysis of standard 12-lead resting electrocardiograms (ECGs). The software accepts digitised ECG data from compatible ECG devices and produces multi-task clinical decision support outputs.

The system analyses each ECG across four clinical domains simultaneously:

- **Rhythm & Conduction** (28 classes): arrhythmias, conduction abnormalities, rare rhythm disorders
- **Structural & Functional** (19 classes): structural heart disease screening, strain patterns
- **Ischaemia & Metabolic** (19 classes): ischaemic patterns, STEMI mimics, metabolic/drug effects
- **Risk Prediction** (6 continuous outputs): mortality, heart failure, AF onset, ejection fraction, conduction disease, sudden cardiac death

### 1.2 Software of Unknown Provenance (SOUP)

| Component | Version | License | Purpose |
|-----------|---------|---------|---------|
| PyTorch | [FILL: version] | BSD-3 | Primary deep learning framework |
| TensorFlow | [FILL: version] | Apache 2.0 | Parity deep learning framework |
| ONNX Runtime | [FILL: version] | MIT | Edge inference engine |
| FastAPI | [FILL: version] | MIT | REST API server |
| NeuroKit2 | [FILL: version] | MIT | Signal processing (QRS detection) |
| SciPy | [FILL: version] | BSD-3 | Signal filtering |
| NumPy | [FILL: version] | BSD-3 | Numerical computation |
| Flower (flwr) | [FILL: version] | Apache 2.0 | Federated learning |
| OpenDP | [FILL: version] | MIT | Differential privacy |
| WeasyPrint | [FILL: version] | BSD-3 | PDF report generation |
| [FILL: additional SOUP components] |

### 1.3 Variants and Configurations

| Variant | Description | Key Differences |
|---------|-------------|-----------------|
| **Server** | Full model on amd64 workstation/server | PyTorch backbone, full precision |
| **Edge** | INT8 quantised model on ARM64 | MobileNet-1D backbone, ≤ 2.5M parameters, ONNX Runtime |
| **PWA Browser** | In-browser inference via WebAssembly | Same INT8 ONNX model, ONNX Runtime Web |

---

## 2. Essential Requirements Checklist (MDR Annex I)

### Chapter I — General Requirements

| Req. | Description | Compliance | Evidence |
|------|-------------|------------|----------|
| **1** | Devices shall achieve performance intended by manufacturer and shall not compromise health/safety | [FILL: compliance statement] | Benchmark report (§ 6), equity gate report, performance card |
| **2** | Risk management according to EN ISO 14971 | [FILL: compliance statement] | Risk management file (§ 4) |
| **3** | Devices shall be designed and manufactured taking into account the state of the art | [FILL: compliance statement] | Architecture description ([IEC 80601-2-86 ATD § 1](IEC_80601_2_86_ATD.md#1-algorithm-description)), literature review |
| **4** | General safety and performance requirements not covered elsewhere | [FILL: compliance statement] | Technical file (this document) |

### Chapter II — Requirements Regarding Design and Manufacture

| Req. | Description | Compliance | Evidence |
|------|-------------|------------|----------|
| **5** | Chemical, physical, biological properties | N/A (standalone software) | — |
| **10** | Devices with a measuring function | Partially applicable | ECG signal measurement accuracy verified in preprocessing pipeline |
| **14** | Software devices and devices incorporating software | Applicable | IEC 62304 lifecycle (§ 5), [IEC 80601-2-86 ATD § 8](IEC_80601_2_86_ATD.md#8-software-development-lifecycle) |
| **14.1** | Repeatability, reliability, performance | [FILL: compliance statement] | Fixed-seed reproducibility, CI pipeline, benchmark harness |
| **14.2** | State of the art development practices | [FILL: compliance statement] | IEC 62304, automated testing, type checking, linting |
| **14.5** | Interaction with IT environment | [FILL: compliance statement] | Deployment guide, Docker images, API documentation |
| **17** | Electronic programmable systems | Applicable | IEC 62304 SDLC documentation |

### Chapter III — Requirements Regarding Information Supplied with the Device

| Req. | Description | Compliance | Evidence |
|------|-------------|------------|----------|
| **23** | Label and instructions for use | [FILL: compliance statement] | IFU document, output labelling (§ 9.1) |
| **23.1** | Device identification and manufacturer | [FILL: compliance statement] | UDI, EUDAMED registration |
| **23.4** | Information on residual risks | [FILL: compliance statement] | Risk management file, known limitations ([IEC 80601-2-86 ATD § 5](IEC_80601_2_86_ATD.md#5-known-limitations)) |

---

## 3. Clinical Evaluation Plan (MEDDEV 2.7/1 Rev 4)

### 3.1 Scope and Objectives

This clinical evaluation demonstrates the safety and performance of Aortica AI ECG Analysis Software through:

1. **Literature review** of published evidence for deep learning ECG analysis systems
2. **Analytical performance testing** using the standardised benchmark harness
3. **Equivalence analysis** with state-of-the-art published ECG AI models
4. [FILL: prospective clinical investigation if required by Notified Body]

### 3.2 Device Under Evaluation

See § 1 (Device Description) and [IEC 80601-2-86 ATD § 1](IEC_80601_2_86_ATD.md#1-algorithm-description) for full algorithm description.

### 3.3 Literature Search Strategy

| Parameter | Value |
|-----------|-------|
| **Databases** | [FILL: PubMed, Embase, Cochrane, IEEE Xplore, etc.] |
| **Search Terms** | [FILL: search strategy, e.g. "deep learning" AND "electrocardiogram" AND ("diagnosis" OR "classification" OR "screening")] |
| **Date Range** | [FILL: search date range] |
| **Inclusion Criteria** | [FILL: study inclusion criteria] |
| **Exclusion Criteria** | [FILL: study exclusion criteria] |

### 3.4 Analytical Performance Evidence

Analytical performance is evaluated using Aortica's built-in benchmark harness and equity gating:

| Evidence Source | Description | Cross-Reference |
|-----------------|-------------|-----------------|
| `aortica.evaluation.benchmark()` | Per-task, per-class AUC, sensitivity, specificity, F1, ECE, C-index, Brier score | [IEC 80601-2-86 ATD § 4](IEC_80601_2_86_ATD.md#4-performance-metrics) |
| `aortica.evaluation.equity_gate()` | Demographic fairness: sex and age subgroup comparison with Bonferroni correction | [IEC 80601-2-86 ATD § 5.2](IEC_80601_2_86_ATD.md#52-demographic-limitations) |
| `aortica.evaluation.generate_performance_card()` | Public-facing markdown + CSV with demographic stratification | — |
| `aortica.edge.validate_edge()` | Edge vs. full model comparison (≤3% AUC degradation threshold) | — |

### 3.5 Performance Targets

| Condition / Task | Metric | Target | Rationale |
|------------------|--------|--------|-----------|
| Overall Rhythm | Macro-F1 | ≥ 0.90 | Phase 1 target; literature benchmark level |
| AF Detection | AUC | ≥ 0.95 | Clinically required sensitivity for screening |
| STEMI Detection | Sensitivity | ≥ 0.90 | Time-critical condition — high sensitivity paramount |
| LVSD Screening | AUC | ≥ 0.88 | Published benchmark for ECG-based LVSD detection |
| Edge Model | AUC degradation | ≤ 3% | Ensures clinically equivalent performance on edge devices |
| [FILL: additional performance targets agreed with Notified Body] |

### 3.6 Clinical Investigation

[FILL: if prospective clinical investigation is planned, describe: study design, primary/secondary endpoints, sample size calculation, multi-site plan. Reference the prospective study protocol template at `docs/validation/PROSPECTIVE_PROTOCOL.md` (US-098).]

### 3.7 Post-Market Clinical Follow-Up (PMCF) Plan

[FILL: PMCF plan per MEDDEV 2.12/2. Should reference:
- Automated performance monitoring (`aortica.validation.PerformanceMonitor`)
- Quarterly public performance reports (`aortica.validation.generate_quarterly_report()`)
- Voluntary adverse event reporting (`POST /api/v1/validation/adverse-event`)
- Non-Western site validation tracking (`aortica.evaluation.SiteValidationRegistry`)
- Clinician feedback collection (`POST /api/v1/feedback`)]

---

## 4. Risk Management (ISO 14971)

### 4.1 Scope

Risk management covers the entire lifecycle of Aortica including design, development, validation, deployment, and post-market surveillance.

### 4.2 Risk Acceptability Criteria

[FILL: define risk acceptability matrix per organisational policy. State whether using ALARP (As Low As Reasonably Practicable) or threshold-based criteria.]

### 4.3 Hazard Identification

| ID | Hazard | Harm | Severity | Probability | Risk Level | Mitigation | Residual Risk |
|----|--------|------|----------|-------------|------------|------------|---------------|
| H-001 | False negative — missed critical finding (STEMI, VT, VF) | Delayed treatment, patient deterioration | Critical | [FILL: probability estimate] | [FILL: risk level] | Uncertainty quantification, OOD detection, "decision support only" labelling, clinical workflow requiring physician review | [FILL: residual risk] |
| H-002 | False positive — incorrect critical finding | Unnecessary invasive procedures, patient anxiety | Serious | [FILL: probability estimate] | [FILL: risk level] | Calibrated probabilities, conformal prediction sets, clinical suggestion prompts requiring clinician judgment | [FILL: residual risk] |
| H-003 | Model drift — performance degradation | Degraded decision support accuracy | Serious | [FILL: probability estimate] | [FILL: risk level] | Performance monitoring, quarterly reports, automated drift detection, equity gating | [FILL: residual risk] |
| H-004 | Data privacy breach | Patient data exposure | Serious | [FILL: probability estimate] | [FILL: risk level] | Local-only deployment, AES-256 encryption at rest, differential privacy in federated learning, no mandatory external data flows | [FILL: residual risk] |
| H-005 | Edge model reduced accuracy | Lower quality decision support | Moderate | [FILL: probability estimate] | [FILL: risk level] | Automated edge validation (≤3% AUC degradation), connection status banner indicating inference mode | [FILL: residual risk] |
| H-006 | Poor quality ECG input | Unreliable analysis results | Moderate | [FILL: probability estimate] | [FILL: risk level] | Signal quality scoring, quality classification (good/marginal/poor), scan-origin quality cap, QualityReport in output | [FILL: residual risk] |
| H-007 | Demographic bias — underperformance on specific populations | Inequitable care, missed diagnoses in underrepresented groups | Serious | [FILL: probability estimate] | [FILL: risk level] | Equity gating (Bonferroni-corrected demographic comparison), non-Western site validation requirement, performance cards with subgroup breakdowns | [FILL: residual risk] |
| [FILL: additional hazards identified through systematic hazard analysis] |

See [IEC 80601-2-86 ATD § 9](IEC_80601_2_86_ATD.md#9-risk-management-reference) for risk management reference.

### 4.4 Risk/Benefit Analysis

[FILL: overall risk/benefit analysis demonstrating that residual risks are acceptable in light of the intended clinical benefits]

---

## 5. Software Lifecycle (IEC 62304)

### 5.1 Safety Classification

| Software Item | Safety Class | Justification |
|---------------|-------------|---------------|
| Signal Preprocessing | [FILL: A / B / C] | [FILL: justification based on contribution to hazardous situation] |
| AI Inference Engine | [FILL: A / B / C] | [FILL: justification] |
| Calibration Layer | [FILL: A / B / C] | [FILL: justification] |
| API / Web UI | [FILL: A / B / C] | [FILL: justification] |
| Result Storage | [FILL: A / B / C] | [FILL: justification] |
| Edge Model | [FILL: A / B / C] | [FILL: justification] |

### 5.2 Software Development Process

| Activity | IEC 62304 Ref. | Evidence |
|----------|---------------|----------|
| Software development planning | § 5.1 | Development plan, PRD, project plan |
| Requirements analysis | § 5.2 | PRD (100+ user stories with acceptance criteria) |
| Architectural design | § 5.3 | Architecture documentation, module structure |
| Detailed design | § 5.4 | Code documentation, docstrings, type annotations |
| Implementation | § 5.5 | Source code (Python, TypeScript), version controlled |
| Verification | § 5.6 | Unit tests (pytest), integration tests, CI pipeline |
| Software release | § 5.7 | Release checklist: benchmark + equity gate + performance card + edge validation |
| Configuration management | § 5.8 | Git, semantic versioning, HuggingFace Hub model registry |
| Problem resolution | § 5.9 | GitHub Issues, adverse event reporting |

See [IEC 80601-2-86 ATD § 8](IEC_80601_2_86_ATD.md#8-software-development-lifecycle) for SDLC details.

### 5.3 Software Maintenance

- CI/CD automated on every push/PR: `ruff` lint, `mypy` type check, `pytest` test suite
- Minimum test coverage: ≥ 80% for core modules
- Equity gate and benchmark re-run on every model change
- Federated learning rounds start from canonical pre-trained checkpoint

---

## 6. Usability Engineering (IEC 62366)

### 6.1 Use Specification

| Parameter | Value |
|-----------|-------|
| **Primary Users** | [FILL: cardiologists, emergency physicians, internists, clinical staff] |
| **Secondary Users** | [FILL: community health workers (simplified 3-tier output mode), researchers] |
| **Use Environment** | [FILL: hospital, clinic, rural field site, home office] |
| **Training Requirements** | [FILL: training required before use] |

### 6.2 User Interface Design

The Aortica web UI provides:

1. **Dashboard**: Overview of recent analyses with urgency-sorted worklist
2. **Upload**: Drag-and-drop ECG file upload with format auto-detection
3. **Results**: Interactive ECG waveform display with AI findings panels (rhythm, structural, ischaemia, risk)
4. **Copilot Panel**: Ranked AI findings with confidence levels and clinical suggestion prompts
5. **XAI Overlay**: Integrated gradient attribution mapped to named ECG features
6. **Second Reader Mode**: Clinician interpretation input with AI comparison diff
7. **Edge Case Spotlight**: Dedicated panel for rare but dangerous findings
8. **Connection Status**: Real-time indicator of server vs. offline inference mode

### 6.3 Hazard-Related Use Scenarios

| Scenario | Hazard | Mitigation |
|----------|--------|------------|
| User over-relies on AI findings without clinical review | H-001, H-002 | Prominent "decision support only" disclaimers, clinical suggestion prompts labelled as requiring clinician judgment |
| User misinterprets confidence levels | H-001, H-002 | Color-coded severity indicators, uncertainty confidence intervals, conformal prediction sets |
| User ignores edge-case spotlight | H-001 | Pulsing visual indicator, distinct panel styling, critical findings highlighted with red accent |
| CHW misinterprets simplified output tiers | H-001 | Three clear tiers with plain-language descriptions, localisation support, training materials |
| [FILL: additional hazard-related use scenarios from formative/summative evaluation] |

### 6.4 Usability Testing

[FILL: describe formative and summative usability testing plans. Include: test objectives, participant profiles, task scenarios, success criteria, and reference to usability test report.]

---

## 7. Post-Market Surveillance Plan

### 7.1 Proactive Surveillance

| Activity | Frequency | Tool |
|----------|-----------|------|
| Automated performance monitoring | Continuous (30-day rolling window) | `aortica.validation.PerformanceMonitor` |
| Drift detection alerts | On threshold breach | Webhook + log alerts |
| Quarterly public performance report | Quarterly | `aortica.validation.generate_quarterly_report()` |
| Clinician feedback analysis | Monthly | `GET /api/v1/feedback/stats` |
| Non-Western site validation tracking | Per site registration | `aortica.evaluation.SiteValidationRegistry` |

### 7.2 Reactive Surveillance

| Activity | Trigger | Tool |
|----------|---------|------|
| Adverse event investigation | Voluntary adverse event report | `POST /api/v1/validation/adverse-event` |
| Field safety corrective action (FSCA) | Serious adverse event or safety signal | [FILL: FSCA process] |
| Trend analysis | Periodic (quarterly) | Adverse event summary statistics |

### 7.3 PMCF Reporting

[FILL: PMCF evaluation report schedule and Notified Body reporting requirements]

---

## 8. Labelling (MDR Annex I, Chapter III)

### 8.1 Device Label

| Element | Content |
|---------|---------|
| **Device Name** | Aortica AI ECG Analysis Software |
| **Manufacturer** | [FILL: legal manufacturer name and address] |
| **UDI** | [FILL: UDI barcode and human-readable format] |
| **Version** | [FILL: software version] |
| **Intended Purpose** | AI-assisted 12-lead ECG analysis for clinical decision support |
| **Warnings** | "AI Decision Support — Requires Clinical Review", "Not validated for paediatric ECGs" |

### 8.2 Instructions for Use

[FILL: reference to full IFU document covering: installation, intended use, contraindications, warnings and precautions, operating instructions, output interpretation guide, troubleshooting, maintenance, and disposal]

---

## 9. Declaration of Conformity

### 9.1 EU Declaration of Conformity (MDR)

[FILL: formal EU Declaration of Conformity per MDR Article 19 and Annex IV. Must include:
- Manufacturer name and address
- Product identification (name, model, version)
- Statement of conformity with MDR (EU) 2017/745
- Reference to harmonised standards applied (IEC 62304, ISO 14971, IEC 62366, IEC 80601-2-86)
- Notified Body identification (if applicable for Class IIa/IIb)
- Signature of authorised representative]

---

## Appendices

### Appendix A: Performance Evidence References

| Document | Location |
|----------|----------|
| IEC 80601-2-86 ATD | [IEC_80601_2_86_ATD.md](IEC_80601_2_86_ATD.md) |
| FDA SaMD Pre-Submission | [FDA_SAMD_PRESUB.md](FDA_SAMD_PRESUB.md) |
| Full Benchmark Report | [FILL: path to benchmark report output] |
| Equity Gate Report | [FILL: path to equity gate report output] |
| Performance Card | [FILL: path to public performance card] |
| Edge Validation Report | [FILL: path to edge validation report] |

### Appendix B: Harmonised Standards Applied

| Standard | Title | Sections Applied |
|----------|-------|-----------------|
| **EN ISO 14971:2019** | Medical devices — Application of risk management | Full |
| **EN IEC 62304:2006+A1:2015** | Medical device software — Software life cycle processes | Full |
| **EN IEC 62366-1:2015+A1:2020** | Medical devices — Usability engineering | Full |
| **IEC 80601-2-86:2023** | Particular requirements for basic safety and essential performance of ECG-based heart characterisation software | ATD (Annex GG) |
| **EN ISO 13485:2016** | Medical devices — Quality management systems | [FILL: applicable sections] |

### Appendix C: Class List (72 Outputs)

See [IEC 80601-2-86 ATD Appendix A](IEC_80601_2_86_ATD.md) for the complete list of all 72 output classes with clinical definitions.

### Appendix D: Notified Body Correspondence

[FILL: record of correspondence with Notified Body regarding classification, conformity assessment route, and any additional requirements]

---

*This template follows EU MDR (2017/745) technical documentation requirements (Annexes II and III). Sections marked with `[FILL: ...]` require site-specific information. Performance evidence is generated automatically by Aortica's benchmark harness and equity gating CI pipeline. Cross-references to [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md) and [FDA SaMD Pre-Submission](FDA_SAMD_PRESUB.md) are provided where applicable.*
