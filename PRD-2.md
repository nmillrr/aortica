# PRD-2: Aortica — Full Platform, Edge Deployment & Federated Learning (Phase 2–4)

> **Prerequisite:** Complete all user stories in [PRD.md](PRD.md) (Phase 0 + Phase 1) before beginning this PRD.

## Introduction

This PRD covers the remaining phases of the Aortica roadmap — transforming the core ML pipeline from PRD.md into a fully deployable ECG copilot for clinicians with API/CLI packaging, web UI, edge/mobile deployment, federated learning, EHR integration, and regulatory scaffolding.

## Clinical Mission: AI ECG Copilot for Cardiologists

Aortica's primary clinical identity is an **AI ECG copilot** — a decision-support tool that sits alongside cardiologists and catches the subtle, edge-case findings that are most dangerous when missed.

Cardiologists are highly skilled at routine ECG interpretation, but certain patterns are reliably difficult for even experienced readers:

- **Edge-case arrhythmias:** Intermittent pre-excitation (WPW), Brugada pattern (type 2/3), subtle atrial flutter with variable block, junctional rhythms masquerading as sinus, fascicular VT, short-coupled PVCs with malignant potential
- **Strain patterns:** Early/subclinical LV strain from hypertension or aortic stenosis, RV strain in pulmonary embolism (S1Q3T3 variants), apical ballooning (Takotsubo) vs. ACS, and diffuse strain from infiltrative cardiomyopathies (amyloid, sarcoid)
- **Signals hiding in plain sight:** Early repolarization vs. STEMI mimics, de Winter T-waves, Wellens syndrome (biphasic T-wave warning of LAD lesion), aVR ST-elevation suggesting left main/3-vessel disease, subtle QTc prolongation, and Sgarbossa criteria in paced/LBBB rhythms
- **Risk signals:** ECG-derived markers of future AF onset, subclinical LV dysfunction (normal ECG with reduced EF), and progressive conduction disease before symptomatic block

Every feature in this PRD—from edge deployment to federated learning—serves this mission: making a copilot that is **accurate on the hard cases**, **trustworthy in its confidence estimates**, **explainable in clinical language**, and **accessible everywhere cardiologists practice**.

**Phase 2 (Months 9–14):** Edge & Rural Deployment — INT8 quantization, ONNX export, Android app, Raspberry Pi deployment, offline sync infrastructure, and initial LMIC pilot sites.

**Phase 3 (Months 15–20):** Federated Learning & Equity — Flower-based FL SDK, differential privacy, equity gating CI checks, and federated model release.

**Phase 4 (Months 21–30):** Regulatory & Scale — Regulatory document library, prospective validation, FHIR ECG management system plugin, worklist prioritization, and national programme support.

**Tech stack additions:** FastAPI + React web UI (Option C from planning), ONNX Runtime Mobile, Flower (flwr) federated learning, OpenDP differential privacy.

## Scope Overview

| Area | PRD (Done) | PRD-2 (This Document) |
|------|-----------|----------------------|
| Core ML pipeline | ✅ | Maintained |
| REST API + gRPC | — | **New** |
| CLI tool (`aortica` command) | — | **New** |
| Web UI (FastAPI + React) | — | **New** |
| Docker images (amd64 + arm64) | — | **New** |
| Documentation site | — | **New** |
| PyPI package (`pip install aortica`) | — | **New** |
| ONNX export + INT8 quantization | — | **New** |
| Android ONNX app | — | **New** |
| Raspberry Pi deployment profile | — | **New** |
| Offline sync infrastructure | — | **New** |
| Federated learning SDK | — | **New** |
| Differential privacy (OpenDP) | — | **New** |
| Equity gating CI checks | — | **New** |
| FHIR R4 / HL7 output | — | **New** |
| DICOM SR write-back | — | **New** |
| PDF / JSON-LD report generation | — | **New** |
| Case-based retrieval system | — | **New** |
| Regulatory document templates | — | **New** |
| Prospective validation tooling | — | **New** |

## Story Areas (To Be Fully Specified After PRD Completion)

The following story areas will be broken into individual user stories when this PRD is activated. Each story will follow the same format as PRD.md (US-030+) and be sized for ~30 min implementation by a small team.

---

### Phase 2 — Edge & Rural Deployment

#### Platform Packaging
- REST API service (FastAPI) wrapping the multi-task inference pipeline
- gRPC service for high-throughput integrations
- CLI tool: `aortica predict <file>`, `aortica benchmark <dataset>`, `aortica train <config>`
- PyPI package with entry points
- Docker images for amd64 (server/GPU) and arm64 (edge)
- Documentation site (MkDocs or similar) with API reference, deployment guides, clinical background

#### Web UI (Option C)
- React frontend with ECG waveform visualization (interactive, zoomable)
- Upload ECG → view multi-task results + XAI annotations
- Batch processing dashboard
- User authentication (OAuth 2.0 / local API key)

#### AI Copilot & Second Reader Mode
- **Copilot overlay:** When viewing an ECG, the AI surfaces a ranked list of findings with confidence levels, highlighting regions of clinical concern directly on the waveform
- **Second reader workflow:** Cardiologist enters their initial interpretation; Aortica compares it against model output and flags discrepancies (e.g., cardiologist reads "normal sinus rhythm" but model detects subtle pre-excitation or early strain)
- **Edge-case spotlight:** Dedicated panel showing low-prevalence findings the model detected with moderate-to-high confidence — specifically designed to catch what routine reads miss
- **Explanation cards:** For each flagged finding, the copilot provides: the named ECG feature(s) driving the detection (from US-025 XAI), a confidence interval (from US-024 conformal prediction), and optionally similar historical cases (from Phase 4 case-based retrieval)
- **Feedback loop:** Cardiologist can accept, reject, or modify AI findings — logged for future model improvement and calibration monitoring

#### Edge Optimization
- ONNX export pipeline for the full and edge model variants
- Knowledge distillation: train MobileNet-1D edge model from full AorticaModel
- INT8 quantization (ONNX Runtime quantization tools)
- Performance validation: edge model AUC within 3% of full model
- Hardware benchmarks: Raspberry Pi 4, Jetson Nano, Android (Snapdragon 660+)

#### Mobile Deployment
- Android app using ONNX Runtime Mobile
- Single-lead and 6-lead input support
- Plain-language output tiers: 'Low risk', 'Refer for assessment', 'Urgent referral recommended'
- Fully offline operation; anonymized audit log sync when online

#### Offline Infrastructure
- SQLite local result storage with AES-256 encryption
- Offline-first sync with vector clock conflict resolution
- Configurable sync thresholds (bandwidth, frequency)
- Raspberry Pi SD card image with pre-installed Aortica edge model

#### LMIC Pilot Deployment
- Deployment guide for 2 target pilot sites
- CHW-facing simplified interface
- Power consumption optimization (< 200 mW on ARM hardware)

---

### Phase 3 — Federated Learning & Equity

#### Federated Learning SDK
- Flower (flwr) integration with pluggable aggregation (FedAvg, FedProx, SCAFFOLD)
- Client wrapper for Aortica training pipeline
- Differential privacy via OpenDP (configurable ε, default 1.0)
- Secure aggregation via CKKS homomorphic encryption for gradient exchange
- DUA (data-use agreement) template in repository
- Federated training documentation and quickstart

#### Equity Infrastructure
- Equity gating CI checks: automated demographic subgroup performance comparison
- Bonferroni-corrected statistical tests for performance parity across sex, age deciles
- Public performance card generator (markdown + CSV) for every model release
- Minimum 2 non-Western site validations before v-stable tagging

#### Additional Task Capabilities — Edge-Case & Subtle Pattern Expansion
- **Rare arrhythmia subtypes:** Add detection classes for Brugada pattern (types 1–3), short QT syndrome, catecholaminergic polymorphic VT (CPVT), fascicular VT, atypical atrial flutter, and inappropriate sinus tachycardia
- **STEMI mimics & subtle ischaemia:** Expand ischaemia head with early repolarization vs. STEMI classifier, de Winter T-wave pattern, Wellens syndrome subtypes (type A/B), aVR ST-elevation pattern, and Sgarbossa criteria in LBBB/paced rhythms
- **Strain pattern refinement:** Train dedicated sub-classifiers for LV strain grade (mild/moderate/severe), RV strain in PE (S1Q3T3 + right axis + T-wave inversions V1-V4), Takotsubo vs. ACS differentiation, and infiltrative cardiomyopathy strain signatures
- **Metabolic & drug effects:** Add detection for hyperkalaemia severity grading, hypothermia (Osborn waves), tricyclic antidepressant toxicity (wide QRS + right axis), and digoxin effect vs. toxicity
- **Risk prediction refinement:** Improve models for subclinical LVSD detection (ECG-predicted EF), progressive conduction disease trajectory, and sudden cardiac death risk stratification using federated multi-site data

---

### Phase 4 — Regulatory & Scale

#### EHR & ECG System Integration
- FHIR R4 DiagnosticReport and Observation resource output
- HL7 v2.x ORU^R01 message generation
- DICOM SR structured report write-back
- DICOM DIMSE C-STORE/C-FIND for GE MUSE-style ECG management systems
- SCP-ECG serial port capture for legacy carts
- SMART on FHIR launch context support
- Worklist prioritization module (AI-sorted by urgency)

#### Report Generation
- PDF clinical report with ECG waveform, multi-task results, and XAI annotations
- JSON-LD machine-readable report
- CSV batch analytics export

#### Case-Based Retrieval
- Latent space index over 50,000 de-identified PhysioNet ECGs
- Top-3 similar historical ECG retrieval with verified diagnoses and outcomes
- Integration with XAI report

#### Regulatory Document Library
- IEC 80601-2-86 Algorithm Testing Documentation template
- FDA SaMD pre-submission package template
- CE-MDR technical file template
- TRIPOD-AI / STARD-AI / CONSORT-AI reporting templates
- CI pipeline enforcing minimum performance targets per device class

#### Prospective Validation Tooling
- Multi-site prospective study protocol template
- Data collection pipeline for prospective ECGs with outcome linkage
- Automated performance monitoring against labeled subsets
- Quarterly public performance report generator
- Voluntary adverse event reporting form

---

## Non-Goals (PRD-2)

- Aortica will not seek FDA/CE clearance during this PRD — it will achieve *regulatory readiness*
- No proprietary data partnerships — all training uses public or federated data
- No commercial cloud hosting — Aortica Cloud is a future sustainability initiative
- No automated treatment recommendations — outputs are decision support only

## Success Metrics

| Metric | Target |
|--------|--------|
| Edge model AUC vs. full model | Within 3% on PTB-XL |
| Android app inference latency | < 200 ms (single-lead, Snapdragon 660+) |
| Raspberry Pi inference latency | < 350 ms (12-lead, edge model) |
| Federated learning partners | 5+ sites across ≥ 3 continents |
| Equity gates | Passed for all v-stable releases |
| STEMI sensitivity | ≥ 90% on held-out external test |
| Risk prediction C-statistic | ≥ 0.72 |
| GitHub stars | 2,500+ within 12 months of v1.0 |
| Pilot deployments | 2+ rural/LMIC sites processing real patient ECGs |
| Peer-reviewed publications | 3+ citing or using Aortica |
| FDA pre-submission meeting | Completed by Month 22 |
