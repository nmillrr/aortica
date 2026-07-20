# PRD: Aortica — AI ECG Copilot Platform (Phase 0–4)

## Introduction

Aortica is an open-source AI ECG analysis platform designed to close the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

This PRD covers **Phase 0 (Foundation)**, **Phase 1 (Core Engine)**, **Phase 2 (Edge & Rural Deployment)**, **Phase 3 (Federated Learning & Equity)**, and **Phase 4 (Regulatory & Scale)**. Phase 0–1 delivered the core ML pipeline. Phase 2 extends this with REST API, CLI, React web UI with AI copilot, edge-optimized models (ONNX + INT8), offline sync, Docker, docs, Android ONNX app, and LMIC pilot deployment. Phase 3 adds federated learning via Flower, differential privacy, equity gating CI checks, public performance cards, and expanded task heads for rare arrhythmias, STEMI mimics, strain patterns, and metabolic/drug effects. Phase 4 completes the platform with FHIR R4 / HL7 EHR integration, DICOM SR write-back, worklist prioritization, PDF/JSON-LD report generation, case-based ECG retrieval, regulatory document library (IEC 80601-2-86, FDA SaMD, CE-MDR templates), and prospective validation tooling.

**Distribution model:** Aortica is a self-hosted, open-source toolkit distributed via `aortica.io`. Clinicians and institutions download and run it locally (Docker or pip install) — no data ever leaves the deployment site. This preserves patient privacy, eliminates recurring infrastructure costs, and enables deployment in data-sovereignty-constrained settings. A landing page at `aortica.io` provides download links, documentation, and demo assets.

**Deployment target:** The primary deployment scenario is a rural or resource-limited clinic with a laptop or workstation, intermittent internet, and USB-attached 10/12-lead ECG hardware. The FastAPI backend runs locally (Docker or bare-metal), and the React frontend is served as a **Progressive Web App (PWA)** that caches itself and the ONNX edge model for fully offline use after first load. When the local server is reachable, the full model is used; when offline, inference falls back to ONNX Runtime Web (WebAssembly) running the INT8 edge model directly in the browser. This hybrid architecture requires no internet dependency after initial setup.

**Tech stack:** Python with PyTorch (primary) and TensorFlow/Keras (parallel) for ML. FastAPI for REST API, gRPC for high-throughput service. React + Vite + TypeScript for web UI with PWA service worker. ONNX Runtime (server-side), ONNX Runtime Web/WASM (in-browser offline inference), and ONNX Runtime Mobile (Android) for edge deployment. Click + Rich for CLI. Docker for packaging. Flower (flwr) for federated learning. OpenDP for differential privacy. FHIR R4 via `fhir.resources` for EHR integration. HL7 v2.x via `hl7apy`. DICOM SR via `pydicom`. WeasyPrint for PDF report generation. JSON-LD via `pyld`. Annoy/FAISS for latent space nearest-neighbor retrieval. OpenCV + pdfplumber for PDF/image ECG scan digitization.

**Team:** Small team; stories sized at ~30 min of focused implementation each.

## Clinical Mission: AI ECG Copilot for Cardiologists

Aortica's primary clinical identity is an **AI ECG copilot** — a decision-support tool that sits alongside cardiologists and catches the subtle, edge-case findings that are most dangerous when missed.

Cardiologists are highly skilled at routine ECG interpretation, but certain patterns are reliably difficult for even experienced readers:

- **Edge-case arrhythmias:** Intermittent pre-excitation (WPW), Brugada pattern (type 2/3), subtle atrial flutter with variable block, junctional rhythms masquerading as sinus, fascicular VT, short-coupled PVCs with malignant potential
- **Strain patterns:** Early/subclinical LV strain from hypertension or aortic stenosis, RV strain in pulmonary embolism (S1Q3T3 variants), apical ballooning (Takotsubo) vs. ACS, and diffuse strain from infiltrative cardiomyopathies (amyloid, sarcoid)
- **Signals hiding in plain sight:** Early repolarization vs. STEMI mimics, de Winter T-waves, Wellens syndrome (biphasic T-wave warning of LAD lesion), aVR ST-elevation suggesting left main/3-vessel disease, subtle QTc prolongation, and Sgarbossa criteria in paced/LBBB rhythms
- **Risk signals:** ECG-derived markers of future AF onset, subclinical LV dysfunction (normal ECG with reduced EF), and progressive conduction disease before symptomatic block

Every feature in this PRD — from edge deployment to federated learning — serves this mission: making a copilot that is **accurate on the hard cases**, **trustworthy in its confidence estimates**, **explainable in clinical language**, and **accessible everywhere cardiologists practice**.

## Goals

- Build a universal ECG format reader supporting WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, and XML
- Implement AI-based signal quality assessment: QRS detection, denoising, and per-segment quality scoring
- Develop a multi-task deep learning engine with four task heads (rhythm, structural, ischaemia, risk) sharing a single backbone
- Achieve PTB-XL rhythm macro-F1 ≥ 0.88 (Phase 0 baseline) progressing to ≥ 0.90 (Phase 1)
- Deliver calibrated uncertainty estimation and conformal prediction wrappers
- Implement ECG-native explainability: integrated gradient attribution mapped to named ECG features, plus a VAE latent factor model
- Provide a reproducible benchmark/evaluation harness with demographic subgroup reporting
- Maintain dual PyTorch + TensorFlow/Keras model implementations
- Expose the full inference pipeline as a REST API (FastAPI) and gRPC service
- Ship a CLI tool: `aortica predict`, `aortica benchmark`, `aortica train`
- Build a React web UI with interactive ECG visualization, AI copilot findings panel, second reader comparison mode, and edge-case spotlight
- Export edge-optimized models via ONNX with INT8 quantization for ARM deployment
- Achieve edge model AUC within 3% of full model on PTB-XL across all tasks
- Implement offline-first result storage with AES-256 encryption and vector clock sync
- Provide Docker images for server (amd64) and edge (arm64) deployment
- Deliver a MkDocs documentation site with API reference, deployment guides, and clinical background
- Support Raspberry Pi deployment and LMIC pilot sites with CHW-facing simplified output tiers
- Deploy LMIC pilot sites with deployment guides, power consumption optimization (<200 mW on ARM), and community health worker training materials
- Ship an Android app using ONNX Runtime Mobile with single-lead and 6-lead input, plain-language output tiers, fully offline operation, and anonymized audit log sync
- Implement federated learning SDK with Flower (FedAvg, FedProx, SCAFFOLD) and differential privacy via OpenDP (ε=1.0 default)
- Enforce equity gating: no statistically significant demographic performance gap (Bonferroni-corrected p<0.05) for any class with N>100 test examples
- Generate public performance cards (markdown + CSV) with demographic subgroup breakdowns for every model release
- Expand rhythm head with 6 rare arrhythmia subtypes (Brugada, short QT, CPVT, fascicular VT, atypical flutter, inappropriate sinus tachycardia)
- Expand ischaemia head with 5 STEMI mimic / subtle ischaemia patterns (early repol vs STEMI, de Winter, Wellens A/B, aVR ST-elevation, Sgarbossa)
- Add strain pattern sub-classifiers (LV strain grade, RV strain in PE, Takotsubo vs ACS, infiltrative cardiomyopathy)
- Add metabolic/drug effect detectors (hyperkalaemia grading, Osborn waves, TCA toxicity, digoxin effect vs toxicity)
- Refine risk prediction for subclinical LVSD, progressive conduction disease, and sudden cardiac death
- Generate FHIR R4 DiagnosticReport and Observation resources from multi-task predictions
- Generate HL7 v2.x ORU^R01 messages for legacy EHR integration
- Write-back DICOM SR structured reports compatible with ECG management systems
- Support DICOM DIMSE C-STORE/C-FIND for GE MUSE-style ECG management systems
- Implement SCP-ECG serial port capture for legacy ECG carts
- Support SMART on FHIR launch context for embedded EHR launching
- Provide AI-driven worklist prioritization sorted by clinical urgency
- Generate PDF clinical reports with ECG waveform, multi-task results, and XAI annotations
- Generate JSON-LD machine-readable reports for automated consumption
- Export CSV batch analytics for research and QA workflows
- Build latent space index over de-identified PhysioNet ECGs for case-based retrieval
- Retrieve top-3 similar historical ECGs with verified diagnoses and outcomes for each prediction
- Ship regulatory document templates: IEC 80601-2-86, FDA SaMD pre-submission, CE-MDR technical file, TRIPOD-AI/STARD-AI/CONSORT-AI
- Enforce minimum performance targets per device class via CI pipeline
- Provide multi-site prospective study protocol template and data collection pipeline
- Automate quarterly public performance report generation
- Include voluntary adverse event reporting form
- Provide MIMIC-IV-ECG dataset loader alongside PTB-XL for cross-dataset training and evaluation
- Validate per-platform latency targets (RPi < 350ms, Android < 200ms, Jetson Nano < 150ms) via cross-hardware benchmark suite with CI gates
- Implement API rate limiting and abuse protection with per-tier request throttling
- Ship a Docker Compose full-stack environment for single-command developer onboarding
- Build aortica.io landing page with downloads, docs, and live demo links
- Deliver Android app build pipeline with CI/CD, APK signing, and OTA model updates
- Build React frontend pages for: result history browser, admin dashboard, worklist dashboard, report download, performance monitoring, federated learning monitoring, site validation registry, prospective data collection, adverse event reporting, model comparison, and site analytics
- Provide multi-language i18n support for the web UI (English, French, Spanish, Swahili)
- Implement federated model release pipeline: FL aggregation → validation → equity gate → versioned release
- Add federated data quality gating to validate site-local datasets before FL round participation
- Provide model version comparison tool for quantifying deltas between model releases with regression detection
- Implement centralized, HMAC-chained audit trail for regulatory compliance across all clinical decision workflows
- Build FHIR Subscription and webhook notification service for push-based EHR alerting
- Provide extensible ECG management system plugin architecture with MUSE, FHIR, and file watcher reference plugins
- Deliver end-to-end EHR integration orchestrator connecting DIMSE → inference → DICOM SR → FHIR → worklist → notification
- Implement worklist-to-EHR notification pipeline pushing urgent findings to clinicians via FHIR, HL7, webhook, and email channels
- Build copilot-to-report-to-EHR clinical workflow with clinician attestation and single-click submission
- Connect edge sync to central analytics pipeline for unified cross-site performance monitoring and anomaly detection


## User Stories

---

### Phase 0 — Foundation

---

### US-001: Project Scaffolding and Repository Structure
**Description:** As a developer, I want a well-organized monorepo with standard Python packaging so that the team can collaborate effectively from day one.

**Acceptance Criteria:**
- [x] Repository initialized with `pyproject.toml` (or `setup.cfg`) supporting `pip install -e .`
- [x] Package structure: `aortica/` with subpackages `io/`, `signal/`, `models/`, `xai/`, `evaluation/`, `data/`, `utils/`
- [x] `.gitignore`, `LICENSE` (Apache 2.0), `README.md` with project overview
- [x] `tests/` directory mirroring source structure with `conftest.py`
- [x] `requirements.txt` and `requirements-dev.txt` (or equivalent in pyproject.toml)
- [x] Typecheck passes

---

### US-002: CI/CD Pipeline Setup
**Description:** As a developer, I want automated linting, type checking, and testing on every push so that code quality is enforced continuously.

**Acceptance Criteria:**
- [x] GitHub Actions workflow running on push and PR to `main`
- [x] Steps: `ruff` lint, `mypy` type check, `pytest` test suite
- [x] Workflow fails on any lint error, type error, or test failure
- [x] Badge in README showing CI status
- [x] Typecheck passes

---

### US-003: Canonical ECG Data Representation
**Description:** As a developer, I want a standardized in-memory ECG representation so that all format readers produce a consistent object for downstream processing.

**Acceptance Criteria:**
- [x] `ECGRecord` dataclass with fields: `signals` (numpy array, shape [leads, samples]), `sample_rate` (Hz), `lead_names` (list), `duration_seconds`, `patient_metadata` (optional dict), `source_format` (str), `units` (str, default µV)
- [x] Utility methods: `resample(target_hz)`, `select_leads(lead_list)`, `to_millivolts()`, `num_leads`, `num_samples`
- [x] Validation on construction: checks lead count matches signal shape, sample_rate > 0
- [x] Unit tests covering construction, validation errors, resampling, and lead selection
- [x] Typecheck passes

---

### US-004: WFDB Format Reader
**Description:** As a researcher, I want to load ECG recordings in WFDB format (.hea/.dat) so that I can use PhysioNet datasets directly.

**Acceptance Criteria:**
- [x] `aortica.io.read_wfdb(path)` returns an `ECGRecord`
- [x] Handles single-segment and multi-segment records
- [x] Correctly reads lead names, sample rate, and physical units from header
- [x] Tested with at least 2 real PhysioNet WFDB files (MIT-BIH, PTB-XL samples)
- [x] Typecheck passes

---

### US-005: CSV and MAT Format Readers
**Description:** As a researcher, I want to load ECG data from CSV and MATLAB .mat files so that I can work with common research data exports.

**Acceptance Criteria:**
- [x] `aortica.io.read_csv(path, config)` returns an `ECGRecord`; config specifies column mapping, sample rate, and units
- [x] `aortica.io.read_mat(path, config)` returns an `ECGRecord`; config specifies variable names for signals and metadata
- [x] Handles both row-per-sample and column-per-lead orientations for CSV
- [x] Unit tests with synthetic CSV and MAT files
- [x] Typecheck passes

---

### US-006: DICOM and SCP-ECG Format Reader
**Description:** As a developer, I want to load ECG files in DICOM (Supplement 30/130) and SCP-ECG formats so that Aortica can ingest clinical ECG exports.

**Acceptance Criteria:**
- [x] `aortica.io.read_dicom(path)` returns an `ECGRecord` from a DICOM ECG waveform object
- [x] `aortica.io.read_scp(path)` returns an `ECGRecord` from SCP-ECG files
- [x] Extracts lead data, sample rate, patient demographics (if present), and acquisition metadata
- [x] Unit tests with sample DICOM and SCP-ECG files (synthetic or from public sources)
- [x] Typecheck passes

---

### US-007: HL7 aECG XML Format Reader
**Description:** As a developer, I want to load HL7 aECG (FDA XML) format files so that Aortica can support regulatory submission datasets.

**Acceptance Criteria:**
- [x] `aortica.io.read_hl7_aecg(path)` returns an `ECGRecord`
- [x] Parses lead waveforms, timing info, and demographic annotations from the XML structure
- [x] Unit tests with a synthetic HL7 aECG XML file
- [x] Typecheck passes

---

### US-008: Universal Format Dispatcher
**Description:** As a user, I want a single `aortica.io.read_ecg(path)` function that auto-detects the file format and dispatches to the correct reader.

**Acceptance Criteria:**
- [x] `read_ecg(path)` detects format by extension and/or file magic bytes
- [x] Falls back to explicit `format=` parameter if auto-detection fails
- [x] Raises clear `UnsupportedFormatError` for unknown formats
- [x] Returns a normalized `ECGRecord` with consistent lead ordering (I, II, III, aVR, aVL, aVF, V1-V6 for 12-lead)
- [x] Resamples to a configurable target rate (default 500 Hz)
- [x] Unit tests covering each format dispatch and the fallback path
- [x] Typecheck passes

---

### US-008b: PDF and Image ECG Scan Digitization Reader
**Description:** As a clinician in a setting where only paper or PDF ECG printouts are available, I want to upload a scanned ECG image or PDF and have Aortica digitize it into a signal so that AI analysis is possible even without direct device connectivity.

**Acceptance Criteria:**
- [x] `aortica.io.read_pdf_ecg(path, config=None)` accepts a PDF or image file (PNG, JPG, TIFF) and returns an `ECGRecord`
- [x] PDF extraction: uses `pdfplumber` or `pymupdf` to rasterize the ECG page to a high-resolution image (≥300 DPI)
- [x] Grid detection: uses OpenCV to detect the standard ECG grid (horizontal/vertical lines), compute mm/pixel scale, and calibrate amplitude (mV/pixel) and time (s/pixel) axes
- [x] Waveform trace extraction: isolates the red/black ECG trace from the background grid via color thresholding and contour detection; reconstructs a time-series signal per detected lead region
- [x] Lead region segmentation: auto-detects lead arrangement (standard 12-lead 3×4 layout or rhythm strip) from the image layout; falls back to user-specified grid via `PDFECGConfig(rows, cols, lead_order)`
- [x] Returns an `ECGRecord` with `source_format='pdf_scan'` and a prominently set `scan_quality_warning=True` flag in `patient_metadata`
- [x] `score_quality()` automatically assigns a 'marginal' floor to scan-derived records (scan quality cannot exceed 69/100) with a `scan_origin` flag in the `QualityReport`
- [x] `read_ecg()` universal dispatcher auto-detects `.pdf`, `.png`, `.jpg`, `.tiff` extensions and routes to `read_pdf_ecg()`
- [x] Digitization accuracy target: ≥85% waveform shape correlation (Pearson r) against ground-truth signal on a held-out set of synthetic ECG-to-image-to-signal round-trips
- [x] Unit tests with synthetic ECG images (rendered from known signals) verifying: grid detection, signal extraction shape, amplitude calibration within 15%, round-trip Pearson r ≥ 0.85
- [x] Typecheck passes

---

### US-009: QRS Detection Module
**Description:** As an ML engineer, I want reliable R-peak detection so that downstream modules can segment beats and compute intervals.

**Acceptance Criteria:**
- [x] `aortica.signal.detect_qrs(ecg_record, method='neurokit')` returns an array of R-peak sample indices
- [x] Supports at least two detection backends: NeuroKit2 and Pan-Tompkins
- [x] Works on single-lead and multi-lead inputs (uses Lead II by default, configurable)
- [x] Sensitivity ≥ 99% on a subset of MIT-BIH Arrhythmia Database annotations
- [x] Unit tests with synthetic and real ECG segments
- [x] Typecheck passes

---

### US-010: Signal Denoising Module
**Description:** As an ML engineer, I want to remove baseline wander, powerline interference, and high-frequency noise so that model inputs are clean.

**Acceptance Criteria:**
- [x] `aortica.signal.denoise(ecg_record, methods=['baseline', 'powerline', 'highfreq'])` returns a cleaned `ECGRecord`
- [x] Baseline wander removal via wavelet or highpass filter (configurable cutoff, default 0.5 Hz)
- [x] Powerline noise removal via notch filter (50/60 Hz, auto-detected or configurable)
- [x] High-frequency noise via lowpass filter (default 40 Hz cutoff)
- [x] Each filter can be applied independently or in combination
- [x] Unit tests verifying SNR improvement on synthetically corrupted signals
- [x] Typecheck passes

---

### US-011: Signal Quality Scoring Module
**Description:** As a clinician, I want each ECG segment scored for signal quality so that unreliable segments can be flagged or excluded before AI analysis.

**Acceptance Criteria:**
- [x] `aortica.signal.score_quality(ecg_record)` returns a per-lead quality score (0–100) and an overall score
- [x] Detects and flags: lead-off / flatline, excessive baseline wander, motion artifact, saturation / clipping
- [x] Quality classification: 'good' (≥70), 'marginal' (40–69), 'poor' (<40) with configurable thresholds
- [x] Returns a `QualityReport` object with per-lead scores, flags, and an overall accept/review/reject recommendation
- [x] Unit tests with synthetic clean, noisy, and lead-off signals
- [x] Typecheck passes

---

### US-012: PTB-XL Dataset Loader
**Description:** As a researcher, I want a convenient loader for the PTB-XL dataset so that I can train and benchmark models without manual data wrangling.

**Acceptance Criteria:**
- [x] `aortica.data.load_ptbxl(path, sampling_rate=500)` returns train/val/test splits as lists of `ECGRecord` objects with labels
- [x] Uses the official PTB-XL recommended folds (1-8 train, 9 val, 10 test)
- [x] Parses SCP statement codes into Aortica's label taxonomy (rhythm, structural, ischaemia superclasses)
- [x] Supports loading at 100 Hz or 500 Hz
- [x] Returns label vectors compatible with PyTorch Dataset and TF tf.data pipelines
- [x] Unit tests verifying correct split sizes, label distributions, and data shapes
- [x] Typecheck passes

---

### US-103: MIMIC-IV-ECG Dataset Loader
**Description:** As a researcher, I want a convenient loader for the MIMIC-IV-ECG dataset so that I can train, benchmark, and build case-based retrieval indexes using the second-largest public clinical ECG archive alongside PTB-XL.

**Background and rationale:** PTB-XL (US-012) is the primary training dataset, but the case-based retrieval index (US-091) targets 50,000 ECGs from both PTB-XL and MIMIC-IV-ECG. Without a dedicated MIMIC-IV-ECG loader, building the retrieval index and validating cross-dataset generalization requires ad hoc data wrangling. MIMIC-IV-ECG uses WFDB format but has its own metadata schema (linked to MIMIC-IV clinical tables), requiring a purpose-built loader.

**Acceptance Criteria:**
- [x] `aortica.data.load_mimic_iv_ecg(path, sampling_rate=500)` returns train/val/test splits as lists of `ECGRecord` objects with labels
- [x] Parses MIMIC-IV-ECG WFDB records and links to `machine_measurements.csv` and `record_list.csv` for metadata
- [x] Extracts diagnostic labels from MIMIC-IV clinical tables (`diagnoses_icd`) when available, mapped to Aortica's label taxonomy (rhythm, structural, ischaemia superclasses)
- [x] Supports configurable split strategy: random (default), patient-level (no patient in both train and test), or temporal (by admission date)
- [x] Handles the PhysioNet credentialed access model: clear error message if data files are not present, with download instructions referencing PhysioNet credentialing requirements
- [x] Returns label vectors compatible with PyTorch Dataset and TF tf.data pipelines (same interface as `load_ptbxl`)
- [x] `aortica.data.load_combined(ptbxl_path, mimic_path, sampling_rate=500)` merges both datasets with source tagging for cross-dataset evaluation
- [x] Unit tests verifying correct split sizes, label distributions, data shapes, and combined loader merge logic
- [x] Typecheck passes

---

### US-013: ECG Dataset and DataLoader Utilities
**Description:** As an ML engineer, I want PyTorch Dataset/DataLoader and TF tf.data wrappers so that I can efficiently feed ECG data into training loops.

**Acceptance Criteria:**
- [x] `aortica.data.ECGDataset` (PyTorch `Dataset` subclass) wrapping a list of `ECGRecord` + labels with configurable augmentations
- [x] `aortica.data.create_tf_dataset()` producing a `tf.data.Dataset` from the same data
- [x] Augmentations: random lead dropout, Gaussian noise injection, time-shift, amplitude scaling
- [x] Configurable window length (2.5s, 5s, 10s) with padding/truncation
- [x] Both produce tensors of shape `[batch, leads, samples]` with corresponding label tensors
- [x] Unit tests verifying shapes, augmentation effects, and batch iteration
- [x] Typecheck passes

---

### US-014: Baseline Rhythm Classification Model (PTB-XL)
**Description:** As a researcher, I want a baseline CNN rhythm classifier trained on PTB-XL so that we have a reproducible performance reference.

**Acceptance Criteria:**
- [x] 1D ResNet-18 adapted for 12-lead ECG input implemented in PyTorch
- [x] Training script: `aortica/models/train_baseline.py` with configurable hyperparameters (lr, epochs, batch_size)
- [x] Achieves rhythm superclass macro-F1 ≥ 0.88 on PTB-XL test fold
- [x] Saves model checkpoint and training metrics (loss, F1 per epoch) to disk
- [x] Reproducible: fixed random seeds produce identical results
- [x] Typecheck passes

---

### Phase 1 — Core Multi-Task Engine

---

### US-015: Shared ResNet Backbone Encoder
**Description:** As an ML engineer, I want a modular shared-backbone encoder so that multiple task heads can leverage the same learned ECG representations.

**Acceptance Criteria:**
- [x] `aortica.models.AorticaBackbone` class: 1D ResNet with residual blocks at 64, 128, 256 filter widths
- [x] Accepts input shape `[batch, leads, samples]` with adaptive pooling to handle 250–1000 Hz sampling rates and 2.5–10s windows
- [x] Returns a feature tensor suitable for downstream task heads
- [x] Implemented in both PyTorch and TensorFlow/Keras (separate files, identical architecture)
- [x] Unit tests verifying output shapes for various input configurations
- [x] Typecheck passes

---

### US-016: Cross-Lead Temporal Attention Module
**Description:** As an ML engineer, I want a multi-head attention module that captures inter-lead relationships so that the model can reason about axis, ischaemia territory, and conduction patterns.

**Acceptance Criteria:**
- [x] `aortica.models.CrossLeadAttention` module: 4-head attention, 64-dim per head
- [x] Takes backbone feature output, applies cross-lead attention, returns enriched representation
- [x] Attention weights are extractable for XAI purposes
- [x] Implemented in both PyTorch and TensorFlow/Keras
- [x] Unit tests verifying output shapes and attention weight dimensions
- [x] Typecheck passes

---

### US-017: Rhythm & Conduction Task Head (22 Classes)
**Description:** As a clinician, I want AI detection of 22 rhythm and conduction abnormalities from a single ECG so that a comprehensive rhythm assessment is produced.

**Acceptance Criteria:**
- [x] `aortica.models.RhythmHead`: multi-label classification head producing 22 sigmoid outputs
- [x] Classes: AF, AFL, SVT, AVNRT, AVRT, VT, VF, idioventricular, sinus brady/tachy, PAC, PVC, 1st/2nd/3rd AV block, LBBB, RBBB, LAFB, LPFB, WPW, pacemaker rhythm, normal sinus rhythm
- [x] Connects to backbone + attention output
- [x] Loss: binary cross-entropy with class-weight balancing
- [x] Implemented in both PyTorch and TensorFlow/Keras
- [x] Unit tests verifying output shape and loss computation
- [x] Typecheck passes

---

### US-018: Structural & Functional Task Head (15 Classes)
**Description:** As a clinician, I want AI screening for 15 structural and functional cardiac abnormalities so that subclinical disease can be flagged for further workup.

**Acceptance Criteria:**
- [x] `aortica.models.StructuralHead`: multi-label classification head producing 15 sigmoid outputs
- [x] Classes: LVH, RVH, LVSD, HFpEF risk, DCM, HCM, ARVC, amyloidosis, aortic stenosis, mitral regurgitation, pulmonary HTN, LA enlargement, RA enlargement, pericarditis pattern, myocarditis pattern
- [x] Connects to backbone + attention output
- [x] Loss: binary cross-entropy with focal loss option for rare classes
- [x] Implemented in both PyTorch and TensorFlow/Keras
- [x] Unit tests verifying output shape and loss computation
- [x] Typecheck passes

---

### US-019: Ischaemia & Metabolic Task Head (10 Classes)
**Description:** As a clinician, I want AI detection of ischaemic and metabolic ECG patterns so that acute MI and electrolyte disorders are caught early.

**Acceptance Criteria:**
- [x] `aortica.models.IschaemiaHead`: multi-label classification head producing 10 sigmoid outputs
- [x] Classes: STEMI (per territory), posterior MI, occlusive NSTEMI, old MI, hyperkalaemia, hypokalaemia, hypercalcaemia, hypothyroidism pattern, digitalis effect, QTc prolongation
- [x] Connects to backbone + attention output
- [x] Loss: binary cross-entropy with class weighting
- [x] Implemented in both PyTorch and TensorFlow/Keras
- [x] Unit tests verifying output shape and loss computation
- [x] Typecheck passes

---

### US-020: Risk Prediction Task Head (3 Continuous Outputs)
**Description:** As a clinician, I want continuous risk scores for mortality, heart failure hospitalization, and AF onset so that high-risk patients can be identified proactively.

**Acceptance Criteria:**
- [x] `aortica.models.RiskHead`: regression head producing 3 continuous outputs (sigmoid-scaled 0–1)
- [x] Outputs: 1-year all-cause mortality score, 12-month HF hospitalization probability, 12-month AF onset risk
- [x] Connects to backbone + attention output
- [x] Loss: combined MSE + ranking loss (concordance index proxy)
- [x] Implemented in both PyTorch and TensorFlow/Keras
- [x] Unit tests verifying output shape and loss computation
- [x] Typecheck passes

---

### US-021: Unified Multi-Task Model Assembly
**Description:** As an ML engineer, I want a single `AorticaModel` class that combines the backbone, attention, and all four task heads into one forward pass.

**Acceptance Criteria:**
- [x] `aortica.models.AorticaModel` composes backbone + attention + 4 task heads
- [x] Single forward pass returns a `MultiTaskOutput` dict/dataclass with keys: `rhythm`, `structural`, `ischaemia`, `risk`
- [x] Configurable: any task head can be disabled (e.g., train rhythm-only)
- [x] Supports freezing/unfreezing backbone independently of heads
- [x] Implemented in both PyTorch (`nn.Module`) and TensorFlow/Keras (`tf.keras.Model`)
- [x] Unit tests verifying full forward pass, selective head disabling, and gradient flow
- [x] Typecheck passes

---

### US-022: Multi-Task Training Pipeline
**Description:** As an ML engineer, I want a training pipeline that jointly optimizes all task heads with configurable loss weighting so that the multi-task model converges effectively.

**Acceptance Criteria:**
- [x] Training script accepting config (YAML) with per-task loss weights, learning rate schedule, epochs, batch size
- [x] Weighted sum of task losses with configurable coefficients (default: equal weighting)
- [x] Learning rate: cosine annealing with warmup
- [x] Gradient clipping (configurable, default max_norm=1.0)
- [x] Logs per-task loss, overall loss, and per-task metrics (F1 for classification, C-index for risk) each epoch
- [x] Saves best checkpoint by validation metric (configurable which metric)
- [x] Works with both PyTorch and TF/Keras backends
- [x] Typecheck passes

---

### US-023: Temperature Scaling Calibration Layer
**Description:** As an ML engineer, I want post-hoc calibration so that model probability outputs are well-calibrated and clinically trustworthy.

**Acceptance Criteria:**
- [x] `aortica.models.TemperatureScaling` module: learns a single temperature parameter per task head on the validation set
- [x] `calibrate(model, val_loader)` optimizes temperature on NLL loss
- [x] `CalibratedModel` wrapper applies temperature scaling at inference time
- [x] Produces reliability diagrams (expected calibration error) as part of evaluation
- [x] Unit tests verifying ECE improvement on synthetic miscalibrated logits
- [x] Typecheck passes

---

### US-024: Conformal Prediction and Uncertainty Estimation
**Description:** As a clinician, I want per-prediction confidence intervals and out-of-distribution flagging so that I know when to trust the AI and when to override.

**Acceptance Criteria:**
- [x] `aortica.models.ConformalPredictor` wrapper: generates prediction sets at a user-specified coverage level (default 90%)
- [x] OOD detection via Mahalanobis distance on backbone features; flags inputs beyond a configurable percentile threshold
- [x] `UncertaintyReport` object returned alongside predictions with: confidence interval, OOD flag, entropy score
- [x] Unit tests with in-distribution and synthetic OOD inputs
- [x] Typecheck passes

---

### US-025: Integrated Gradient XAI with Named ECG Features
**Description:** As a clinician, I want AI explanations mapped to named ECG features (QRS width, ST slope, T-wave morphology) rather than generic heatmaps so that I can reconcile AI findings with my visual interpretation.

**Acceptance Criteria:**
- [x] `aortica.xai.explain(model, ecg_record, task='rhythm')` returns a `FeatureAttribution` object
- [x] Computes integrated gradients per lead
- [x] Maps gradient attributions onto named ECG segments: P wave, PR interval, QRS complex, ST segment, T wave, QT/QTc
- [x] Segment boundaries determined by a rule-based delineation algorithm (using R-peak + interval heuristics)
- [x] Returns top-3 contributing features per active diagnosis with delta-contribution scores
- [x] Unit tests verifying attribution shape and feature mapping on synthetic ECG
- [x] Typecheck passes

---

### US-026: VAE Latent Factor Model
**Description:** As a researcher, I want a variational autoencoder that encodes median beats into interpretable latent factors so that the model's internal representations can be visualized and understood.

**Acceptance Criteria:**
- [x] `aortica.xai.MedianBeatVAE`: VAE with 24-dimensional latent space, trained on median beats extracted from 12-lead ECGs
- [x] Encoder: 1D CNN; Decoder: transposed 1D CNN; Loss: reconstruction + KL divergence
- [x] Training script for the VAE on PTB-XL median beats
- [x] Each latent dimension labeled by Pearson correlation with standard ECG measurements (from PTB-XL metadata)
- [x] Unit tests verifying encode/decode shapes and reconstruction loss convergence
- [x] Typecheck passes

---

### US-027: VAE Reporter and Synthetic ECG Rendering
**Description:** As a clinician, I want to see how changing a single latent factor affects the ECG waveform so that I can understand what the model has learned.

**Acceptance Criteria:**
- [x] `aortica.xai.vae_report(model, vae, ecg_record)` returns a `VAEReport` object
- [x] Reports which latent factors are most activated for the given prediction
- [x] Generates synthetic ECG waveforms showing the effect of varying each top factor ±2σ
- [x] Synthetic waveforms returned as numpy arrays (rendering to image is deferred to PRD-2)
- [x] Unit tests verifying report generation and synthetic waveform shapes
- [x] Typecheck passes

---

### US-028: Multi-Task Evaluation Harness
**Description:** As a researcher, I want a benchmark harness that evaluates all task heads with proper metrics and demographic subgroup breakdowns so that model performance is transparent and reproducible.

**Acceptance Criteria:**
- [x] `aortica.evaluation.benchmark(model, dataset, tasks='all')` returns a `BenchmarkReport`
- [x] Metrics per classification task: macro-F1, per-class AUC, per-class sensitivity/specificity, ECE
- [x] Metrics per risk task: C-index, Brier score
- [x] Subgroup stratification by age decile and sex (when demographic metadata is available)
- [x] Outputs results as structured dict, printable summary table, and CSV export
- [x] Reproducible: same model + dataset + seed produces identical results
- [x] Unit tests verifying metric computation on synthetic predictions
- [x] Typecheck passes

---

### US-029: TensorFlow/Keras Parity Validation
**Description:** As a developer, I want automated tests confirming that the TF/Keras model implementation produces equivalent outputs to the PyTorch implementation so that framework parity is maintained.

**Acceptance Criteria:**
- [x] Script that loads identical weights into both PyTorch and TF/Keras models
- [x] Feeds the same input tensor through both and asserts outputs are within floating-point tolerance (atol=1e-5)
- [x] Included as a CI test (may be slow; tagged as `@pytest.mark.slow`)
- [x] Documents the weight conversion process between frameworks
- [x] Typecheck passes

---

### Phase 2 — Edge & Rural Deployment

---

#### Platform Packaging

---

### US-030: FastAPI Application Scaffold
**Description:** As a developer, I want a FastAPI application skeleton with standard middleware so that the inference pipeline can be exposed as a REST API.

**Acceptance Criteria:**
- [x] `aortica/api/` subpackage with `app.py`, `__init__.py`
- [x] FastAPI app with CORS middleware (configurable origins)
- [x] `GET /health` returns `{"status": "ok"}`
- [x] `GET /info` returns model version, supported formats, and enabled task heads
- [x] Pydantic response models for all endpoints
- [x] Unit tests for health and info endpoints
- [x] Typecheck passes

---

### US-031: Single ECG Inference API Endpoint
**Description:** As a clinician, I want to upload a single ECG file via REST API and receive multi-task AI predictions so that I can integrate Aortica into my workflow.

**Acceptance Criteria:**
- [x] `POST /api/v1/predict` accepts file upload (multipart/form-data)
- [x] Runs full pipeline: `read_ecg` → `denoise` → `score_quality` → model inference
- [x] Returns JSON with: quality report, per-task predictions (rhythm, structural, ischaemia, risk), uncertainty report
- [x] Supports `format` query parameter for explicit format override
- [x] Returns `422` with clear error for unsupported formats
- [x] Unit tests with synthetic ECG file upload
- [x] Typecheck passes

---

### US-032: Batch ECG Inference API Endpoint
**Description:** As a researcher, I want to submit multiple ECG files for batch processing so that I can analyze large datasets efficiently.

**Acceptance Criteria:**
- [x] `POST /api/v1/predict/batch` accepts multiple file uploads
- [x] Returns list of per-file results (same schema as single predict)
- [x] Includes per-file status (success/error) with error messages for failed files
- [x] Configurable max batch size (default 50)
- [x] Unit tests with multiple synthetic ECG uploads
- [x] Typecheck passes

---

### US-033: gRPC Service Definition and Server
**Description:** As a device integrator, I want a gRPC service for high-throughput ECG inference so that I can integrate Aortica with low-latency clinical systems.

**Acceptance Criteria:**
- [x] `aortica/api/ecg_service.proto` defining `ECGPredictionService` with `Predict` and `PredictBatch` RPCs
- [x] Generated Python stubs via `grpcio-tools`
- [x] `aortica/api/grpc_server.py` implementing the service, wrapping the same inference pipeline as REST
- [x] Server startup function with configurable port
- [x] Unit tests using gRPC test channel
- [x] Typecheck passes

---

### US-034: CLI Tool — `aortica predict` Command
**Description:** As a clinician, I want to run `aortica predict <file>` from the command line so that I can get AI predictions without writing code.

**Acceptance Criteria:**
- [x] `aortica/cli/` subpackage with Click-based CLI
- [x] `aortica predict <file>` runs full inference pipeline and prints results
- [x] `--format` flag for output format: `table` (default), `json`
- [x] `--tasks` flag to select which task heads to run (default: all)
- [x] `--model` flag to specify model checkpoint path
- [x] Colored terminal output with severity indicators (via `rich`)
- [x] Unit tests for CLI invocation (Click CliRunner)
- [x] Typecheck passes

---

### US-035: CLI Tool — `aortica benchmark` and `aortica train` Commands
**Description:** As a researcher, I want CLI commands for benchmarking and training so that I can reproduce results and train models without custom scripts.

**Acceptance Criteria:**
- [x] `aortica benchmark <dataset_path>` wraps `evaluation.benchmark()` with CLI args for tasks, output format, CSV export path
- [x] `aortica train <config.yaml>` wraps `train_multitask.train()` with YAML config file
- [x] Both commands share the same Click group as `aortica predict`
- [x] `--help` shows all available options with descriptions
- [x] Unit tests for CLI invocation
- [x] Typecheck passes

---

### US-036: PyPI Entry Points and Package Metadata Update
**Description:** As a user, I want to install Aortica via `pip install aortica` and have the CLI and API available immediately.

**Acceptance Criteria:**
- [x] `[project.scripts]` entry point in `pyproject.toml`: `aortica = "aortica.cli:main"`
- [x] New optional dependency groups: `api` (fastapi, uvicorn, python-multipart), `grpc` (grpcio, grpcio-tools), `cli` (click, rich), `edge` (onnx, onnxruntime)
- [x] `aortica-server` entry point for API server: `aortica.api:run_server`
- [x] Version bump to 0.2.0
- [x] Installation test: `pip install -e .[cli,api]` succeeds
- [x] Typecheck passes

---

### US-036b: Pre-Trained Model Distribution via HuggingFace Hub
**Description:** As a clinician or field-deployment engineer, I want to download a ready-to-use, pre-trained Aortica checkpoint without running a training job so that I can get AI predictions immediately after `pip install aortica`, even in settings with no GPU or dataset access.

**Background and rationale:** Aortica's primary deployment scenario is a rural clinic where re-training from raw PTB-XL data is impractical (hardware, data storage, expertise). The current PRD requires users to download ~21 GB of PTB-XL and run a multi-epoch training run before any inference is possible. Distributing canonical, versioned checkpoints trained on publicly licensed data (PTB-XL — CC BY 4.0, MIMIC-IV-ECG — PhysioNet Credentialed) eliminates this barrier entirely. The `aortica train` CLI remains the research path for fine-tuning, federated rounds, and custom datasets. The pre-trained checkpoint becomes the universal starting point for all downstream workflows — including federated learning (Phase 3), which fine-tunes from the public checkpoint rather than training from scratch.

**Acceptance Criteria:**
- [x] `aortica/models/registry.py` module implementing `load_pretrained(version='latest', cache_dir=None, force_download=False) -> AorticaModel`
- [x] Checkpoint hosted on HuggingFace Hub at `nmillrr/aortica` (or equivalent org namespace) with semantic versioning tags matching the package version (e.g., `aortica-v0.2.0.pt`)
- [x] Downloads and caches the checkpoint to `~/.cache/aortica/` on first call; subsequent calls load from cache without network access
- [x] `load_pretrained()` verifies the downloaded file against a published SHA-256 hash before loading, raising `ChecksumError` on mismatch (tamper protection for clinical deployment)
- [x] CLI shortcut: `aortica predict <file>` and `aortica benchmark <path>` automatically call `load_pretrained('latest')` when no `--model` flag is provided
- [x] Distributed bundle includes **both** the full PyTorch checkpoint (`aortica_full_v{version}.pt`) and the INT8 ONNX edge model (`aortica_edge_int8_v{version}.onnx`); `load_pretrained(variant='edge')` fetches the ONNX artifact instead
- [x] Each release on HuggingFace Hub includes a **model card** (`README.md`) documenting: training data (PTB-XL dataset, version, fold split), all task head class lists and output dimensions, per-task performance metrics (macro-F1, AUC, C-index from US-028 benchmark), demographic subgroup performance (age decile, sex) from equity gate (US-069), known limitations (European-heavy training cohort, PDF-scan origin ECGs capped to marginal quality), and attribution/license notices
- [x] The model card includes a prominent data provenance section: "Trained on PTB-XL (CC BY 4.0, Wagner et al. 2020, PhysioNet). No proprietary data used. No patient data leaves this deployment." This directly satisfies the PRD non-goal: *"No proprietary data partnerships — all training uses public or federated data."*
- [x] CI/CD: GitHub Actions `release.yml` workflow step that — on push of a version tag — runs the full benchmark (US-028), runs equity gate (US-069), generates and uploads the performance card (US-070), exports the ONNX edge model (US-037), quantizes to INT8 (US-040), and pushes all artifacts to HuggingFace Hub with the version tag
- [x] `aortica.models.registry.list_available_versions()` queries the Hub API and returns a list of available version strings with release dates and performance summary
- [x] `aortica info` CLI command updated to show: currently loaded model version, checkpoint source (hub vs. local path), SHA-256 hash, and training data attribution
- [x] Federated learning client (US-063) `AorticaFlowerClient` initialises its model via `load_pretrained()` by default, so FL rounds always start from the canonical public checkpoint unless a custom `--base-checkpoint` is specified
- [x] Unit tests: mock HuggingFace Hub responses to test `load_pretrained()` cache behaviour, checksum validation (good and tampered), version listing, and CLI `--model` flag override
- [x] Typecheck passes

---

### US-111: API Rate Limiting and Abuse Protection
**Description:** As a deployment administrator, I want rate limiting and abuse protection on the Aortica API so that a single client cannot overwhelm the server and degrade service for other users.

**Acceptance Criteria:**
- [x] `aortica/api/rate_limiter.py` with `RateLimiter` middleware for FastAPI
- [x] Token bucket rate limiting: configurable per-user and global rate limits (default: 60 requests/minute per API key, 200 requests/minute global)
- [x] Rate limit headers in responses: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (RFC 6585 compliant)
- [x] Returns `429 Too Many Requests` with `Retry-After` header when limit exceeded
- [x] Separate rate limit tiers: `predict` endpoints (lower limit, compute-intensive), `report` endpoints (medium), `admin`/`auth` endpoints (higher limit)
- [x] Backend: Redis-backed for multi-process deployments, with in-memory fallback for single-process/edge deployments
- [x] Configurable via `rate_limits.yaml` or environment variables
- [x] IP-based rate limiting for unauthenticated endpoints (health, info)
- [x] Exempt list for trusted internal services (configurable)
- [x] Unit tests: rate limit enforcement, header correctness, tier differentiation, Redis and in-memory backends
- [x] Typecheck passes

---

#### Edge Optimization

---

### US-037: ONNX Export Pipeline
**Description:** As an ML engineer, I want to export AorticaModel to ONNX format so that it can be deployed on edge devices and non-Python runtimes.

**Acceptance Criteria:**
- [x] `aortica/edge/` subpackage with `__init__.py`
- [x] `aortica.edge.export_onnx(model, output_path, opset_version=17)` exports full AorticaModel to ONNX
- [x] Dynamic axes for batch size and signal length dimensions
- [x] Validates exported model: feeds same input through PyTorch and ONNX Runtime, asserts outputs within atol=1e-4
- [x] Supports exporting with subset of task heads (matches model's enabled_tasks)
- [x] Unit tests with synthetic model and input
- [x] Typecheck passes

---

### US-038: MobileNet-1D Edge Backbone
**Description:** As an ML engineer, I want a lightweight 1D MobileNet-style backbone so that Aortica can run on resource-constrained edge devices.

**Acceptance Criteria:**
- [x] `aortica.edge.MobileNetBackbone1D` class: depthwise-separable 1D convolutions at 32, 64, 128 filter widths
- [x] Parameter count ≤ 2.5M (vs AorticaBackbone's larger size)
- [x] Accepts same input shape `[batch, leads, samples]` as AorticaBackbone
- [x] Returns same feature_dim output (default 256) for compatibility with existing task heads
- [x] Unit tests verifying output shapes, parameter count, and gradient flow
- [x] Typecheck passes

---

### US-039: Knowledge Distillation Training Script
**Description:** As an ML engineer, I want to train the edge model via knowledge distillation from the full model so that the edge model retains most of the full model's accuracy.

**Acceptance Criteria:**
- [x] `aortica.edge.train_distillation(teacher, student, train_loader, val_loader, config)` function
- [x] Distillation loss: KL divergence on temperature-scaled soft targets + hard label cross-entropy (configurable alpha weighting)
- [x] Temperature parameter configurable (default T=4.0)
- [x] Saves best student checkpoint by validation metric
- [x] Logs per-epoch distillation loss, hard loss, and per-task metrics
- [x] Unit tests with synthetic teacher/student models and data
- [x] Typecheck passes

---

### US-040: INT8 Quantization Pipeline
**Description:** As an ML engineer, I want to quantize the ONNX edge model to INT8 so that inference is faster and uses less memory on ARM hardware.

**Acceptance Criteria:**
- [x] `aortica.edge.quantize_int8(onnx_model_path, calibration_data, output_path)` function
- [x] Uses ONNX Runtime quantization tools (static quantization with calibration)
- [x] Calibration data: representative ECG samples (configurable count, default 100)
- [x] Validates quantized model runs successfully via ONNX Runtime
- [x] Compares quantized vs. original model output (logs max absolute difference)
- [x] Unit tests with synthetic ONNX model and calibration data
- [x] Typecheck passes

---

### US-041: Edge Model Validation Harness
**Description:** As a researcher, I want to validate that the edge model performs within 3% of the full model so that edge deployment quality is guaranteed.

**Acceptance Criteria:**
- [x] `aortica.edge.validate_edge(full_model, edge_model_path, dataset, tasks='all')` function
- [x] Compares per-task AUC, F1, and C-index between full and edge models
- [x] Returns `EdgeValidationReport` with pass/fail status per task (threshold configurable, default 3%)
- [x] Measures inference latency per sample for the edge model
- [x] Prints summary table of full vs. edge metrics
- [x] Unit tests with synthetic models and predictions
- [x] Typecheck passes

---

### US-104: Cross-Hardware Benchmark Suite with Platform Targets
**Description:** As a release manager, I want a systematic cross-hardware benchmark suite so that every release is validated against specific latency, memory, and throughput targets for each deployment platform (Raspberry Pi 4, Jetson Nano, Android Snapdragon 660+).

**Background and rationale:** US-041 validates edge model AUC within 3% of the full model, and US-061 profiles inference on individual hardware. But the PRD-2 Success Metrics require specific per-platform latency targets (RPi < 350ms, Android < 200ms) that need systematic validation with pass/fail gates integrated into CI. This story creates that validation harness.

**Acceptance Criteria:**
- [x] `aortica.edge.hardware_benchmark(model_path, platform_profile, dataset_sample, n_runs=50)` function running full inference benchmark on a specified platform profile
- [x] `HardwareBenchmarkReport` dataclass with: platform name, model variant, mean/p50/p95/p99 latency, peak memory (RSS), throughput (inferences/second), model size on disk, pass/fail per metric
- [x] Platform profiles defined in `aortica/edge/platform_targets.yaml` with per-platform pass/fail thresholds:
  - Raspberry Pi 4 (arm64): latency p95 < 350ms, peak memory < 512MB
  - Jetson Nano (arm64 + CUDA): latency p95 < 150ms, peak memory < 1GB
  - Android Snapdragon 660+ (arm64): latency p95 < 200ms (single-lead), < 300ms (6-lead)
  - Server amd64 (CPU): latency p95 < 100ms, throughput > 10 inferences/second
  - Server amd64 (GPU): latency p95 < 30ms, throughput > 50 inferences/second
- [x] CLI command `aortica benchmark-hardware --platform <name> --model <path>` runs the benchmark and prints the report
- [x] `aortica.edge.benchmark_all_platforms(model_path)` runs benchmarks for all locally-available platforms and produces a consolidated comparison table
- [x] GitHub Actions CI step that runs available platform benchmarks (at minimum CPU server) and blocks release if any target is missed
- [x] Consolidated report output as markdown table and CSV for inclusion in performance cards (US-070)
- [x] Unit tests with mock platform profiles and synthetic timing data
- [x] Typecheck passes

---

#### Web UI

---

### US-042: React + Vite Frontend Scaffold
**Description:** As a developer, I want a React frontend application with routing and a premium design system so that the web UI has a solid foundation.

**Acceptance Criteria:**
- [x] `frontend/` directory with Vite + React + TypeScript project
- [x] React Router with routes: `/` (dashboard), `/upload`, `/results/:id`, `/batch`, `/login`
- [x] Layout component: sidebar navigation, header with logo, main content area
- [x] Dark theme design system with CSS variables (colors, typography, spacing)
- [x] Google Fonts (Inter) loaded
- [x] Responsive layout (mobile-friendly breakpoints)
- [x] `npm run dev` starts the development server
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-042b: PWA Offline-Capable Inference Infrastructure
**Description:** As a clinician in a setting with intermittent connectivity, I want the Aortica web app to work fully offline after the first load so that ECG analysis is never blocked by network availability.

**Acceptance Criteria:**
- [x] Vite PWA plugin (`vite-plugin-pwa`) configured with a service worker that caches: app shell (JS/CSS/HTML), the INT8 ONNX edge model (~5–8 MB), and ONNX Runtime Web WASM binaries
- [x] `frontend/public/manifest.json` with app name, icons (192×192, 512×512), theme color, and `display: standalone` for installability on Android Chrome and desktop
- [x] `InferenceClient` TypeScript module with `predict(ecgData)` method implementing the hybrid fallback strategy:
  1. Attempt `POST http://localhost:8000/api/v1/predict` (local FastAPI server) with a 3-second timeout
  2. On timeout or network error, fall back to in-browser ONNX Runtime Web inference using the cached edge model
  3. Annotate the result with `inference_mode: 'server' | 'edge_wasm'` so the UI can display which path was used
- [x] ONNX Runtime Web (`onnxruntime-web`) loaded as a frontend dependency; edge model fetched and cached via the service worker on first app load
- [x] `ConnectionStatusBanner` React component: green badge ("Server — Full Model") when local server responds, amber badge ("Offline Mode — Edge Model") when using WASM fallback
- [x] First-load behavior: service worker pre-caches the entire app shell + model bundle in the background; subsequent loads work fully offline
- [x] Edge model served from `frontend/public/models/aortica_edge_int8.onnx`; build script copies the latest quantized model from `aortica/edge/` artifacts
- [x] Verified offline: Chrome DevTools Network → Offline mode shows full inference working with WASM fallback
- [x] Typecheck passes

---

### US-043: ECG Waveform Visualization Component
**Description:** As a clinician, I want an interactive ECG waveform display so that I can visually inspect the 12-lead ECG trace with standard formatting.

**Acceptance Criteria:**
- [x] `ECGWaveformChart` React component rendering 12-lead ECG in standard clinical layout
- [x] Standard ECG grid background (25mm/s paper speed, 10mm/mV gain)
- [x] Lead labels (I, II, III, aVR, aVL, aVF, V1-V6)
- [x] Pan and zoom interactions (mouse wheel zoom, click-drag pan)
- [x] Accepts waveform data as JSON props (`{leads: string[], signals: number[][], sample_rate: number}`)
- [x] Caliper tool for measuring intervals
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-044: ECG Upload and Single Inference Page
**Description:** As a clinician, I want to upload an ECG file and see AI predictions so that I can get a second opinion on my interpretation.

**Acceptance Criteria:**
- [x] Upload page with drag-and-drop file dropzone
- [x] Supported format indicator (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG)
- [x] Calls `POST /api/v1/predict` on upload
- [x] Loading state with progress animation
- [x] Redirects to results page on success showing waveform + results summary
- [x] Error state with clear message for unsupported formats or server errors
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-045: Multi-Task Results Display Panels
**Description:** As a clinician, I want to see all AI findings organized by task head so that I can quickly review rhythm, structural, ischaemia, and risk predictions.

**Acceptance Criteria:**
- [x] Results page showing ECG waveform + four collapsible task panels
- [x] Each panel shows per-class predictions with confidence bars (0–100%)
- [x] Color-coded severity: red (≥80% confidence positive), yellow (50–79%), green (<50%)
- [x] Risk scores displayed as gauges (0–1 scale) with clinical labels
- [x] Signal quality badge (good/marginal/poor) with per-lead breakdown on hover
- [x] Uncertainty indicators (conformal prediction set size) per prediction
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-046: XAI Annotations Overlay on Waveform
**Description:** As a clinician, I want to see which ECG features drive each AI finding so that I can reconcile AI output with my visual interpretation.

**Acceptance Criteria:**
- [x] Integrated gradient heatmap overlay on the ECG waveform (color intensity = attribution strength)
- [x] Named segment callouts: P wave, PR interval, QRS complex, ST segment, T wave markers
- [x] Top-3 contributing features displayed per active finding with delta-contribution scores
- [x] Toggle overlay on/off per finding
- [x] Per-lead attribution visibility controls
- [x] API endpoint accepts `include_xai=true` parameter returning attribution data
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-047: Batch Processing Dashboard
**Description:** As a researcher, I want to upload and process multiple ECGs at once so that I can analyze datasets efficiently through the web UI.

**Acceptance Criteria:**
- [x] Batch upload page accepting multiple files (drag-and-drop or file picker)
- [x] Progress bar showing processing status per file
- [x] Results table with columns: filename, quality score, top rhythm finding, top structural finding, risk scores
- [x] Sortable and filterable columns
- [x] CSV export button downloading all results
- [x] Click row to navigate to individual results page
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-048: User Authentication System
**Description:** As an admin, I want user authentication so that the Aortica web UI and API are access-controlled.

**Acceptance Criteria:**
- [x] FastAPI OAuth 2.0 integration (Google and GitHub providers via `authlib`)
- [x] Local API key authentication for programmatic access (`X-API-Key` header)
- [x] `POST /api/v1/auth/token` for API key generation
- [x] FastAPI `Depends()` security dependency protecting all `/api/v1/` endpoints
- [x] React login page with OAuth buttons and API key input
- [x] JWT token storage and refresh in React
- [x] Protected React routes redirect to login when unauthenticated
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-105: Patient ECG History and Result Browser
**Description:** As a clinician, I want to browse and search historical ECG results stored locally so that I can review past analyses, track patient trends over time, and retrieve previous findings without re-uploading files.

**Acceptance Criteria:**
- [x] `ResultBrowser` React page at route `/history` showing all stored ECG results from the local SQLite database (US-054)
- [x] Table view with columns: timestamp, patient identifier (if available), quality score, top finding, urgency tier, sync status
- [x] Search and filter controls: date range picker, finding filter (dropdown of all detected conditions), quality filter (good/marginal/poor), urgency filter, free-text search on patient metadata
- [x] Sortable columns (click column header to sort ascending/descending)
- [x] Pagination with configurable page size (default 25, options 10/25/50/100)
- [x] Click row to navigate to full results page (`/results/:id`) with waveform, predictions, and XAI
- [x] `GET /api/v1/results` API endpoint with query parameters: `page`, `per_page`, `date_from`, `date_to`, `finding`, `quality`, `urgency`, `search`
- [x] `GET /api/v1/results/:id` returns full stored result (predictions, quality, XAI data) for a specific result ID
- [x] Bulk actions: select multiple results for CSV export or batch report generation
- [x] Responsive layout for tablet and desktop use
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-106: Admin Dashboard — User and API Key Management
**Description:** As a deployment administrator, I want an admin dashboard so that I can manage users, API keys, and system health from the web UI without direct database access.

**Acceptance Criteria:**
- [x] `AdminDashboard` React page at route `/admin` (protected: requires admin role)
- [x] **User management panel:** list all registered users, their roles (admin/clinician/researcher), last login timestamp, and account status (active/disabled); ability to disable/enable accounts and change roles
- [x] **API key management panel:** list all issued API keys with creation date, last-used timestamp, and associated user; ability to revoke keys and generate new keys with configurable expiry
- [x] **System health panel:** current server status, model version loaded, database size, total ECGs processed, uptime, ONNX Runtime status, sync engine status
- [x] **Activity log:** recent API requests with timestamp, user, endpoint, and response status (last 100, paginated)
- [x] `GET /api/v1/admin/users` returns user list (admin-only)
- [x] `PATCH /api/v1/admin/users/:id` updates user role or status (admin-only)
- [x] `DELETE /api/v1/admin/api-keys/:key_id` revokes an API key (admin-only)
- [x] `GET /api/v1/admin/system-health` returns system status metrics
- [x] `GET /api/v1/admin/activity-log` returns paginated activity log
- [x] Role-based access control: admin endpoints return `403` for non-admin users
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-107: Multi-Language Internationalization for Web UI
**Description:** As a clinician practicing in a non-English-speaking region, I want the Aortica web UI available in my language so that I can use the platform without language barriers.

**Background and rationale:** US-060 provides localization for simplified CHW output, and US-061b/US-061c provide localization for deployment guides and the Android app. But the main React web UI — which clinicians interact with daily — has no i18n support. For LMIC deployment success, the web UI must support at minimum French, Spanish, and Swahili alongside English.

**Acceptance Criteria:**
- [x] `react-i18next` library integrated into the React frontend
- [x] All user-facing strings extracted to JSON locale files in `frontend/src/locales/{lang}/translation.json`
- [x] Locale files provided: English (en, complete), French (fr, complete), Spanish (es, complete), Swahili (sw, template stub with high-priority strings translated)
- [x] Language selector in the header/settings allowing runtime language switching without page reload
- [x] Selected language persisted in `localStorage` and applied on next visit
- [x] Clinical terminology follows locale-appropriate medical conventions (e.g., "ischaemia" vs. "ischemia" for en-GB vs. en-US)
- [x] RTL layout support stubbed for future Arabic/Hebrew localization (CSS logical properties used throughout)
- [x] Date and number formatting follows locale conventions via `Intl` API
- [x] All condition names in the copilot panel, results panels, and explanation cards use localized display names from the locale files
- [x] Verify changes work in browser with at least English and French
- [x] Typecheck passes

---

#### AI Copilot & Second Reader Mode

---

### US-049: Copilot Findings Panel
**Description:** As a clinician, I want a ranked AI findings panel so that the most important detections are surfaced first with clear confidence levels and relevant clinical next-step prompts.

**Acceptance Criteria:**
- [x] `CopilotPanel` React component showing all positive findings ranked by confidence
- [x] Each finding shows: condition name, confidence percentage, severity badge (critical/warning/info)
- [x] Each finding includes a **clinical suggestion prompt** — a short, non-prescriptive plain-language cue (e.g. "Consider cardiology referral", "Electrolytes warrant checking", "Urgent 12-lead repeat recommended") sourced from the condition's suggestion map (see US-049b)
- [x] Suggestion prompts are clearly labelled as AI-generated clinical prompts, not treatment orders, with a visible disclaimer: "Decision support only — requires clinician judgment"
- [x] Critical findings (≥90% confidence on high-severity conditions) highlighted with red accent
- [x] Clicking a finding scrolls the waveform to the relevant region and activates XAI overlay
- [x] Empty state message when no significant findings detected
- [x] Integrates with results page layout
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-049b: Clinical Suggestion Prompt Data Layer
**Description:** As a developer, I want a structured, maintainable mapping of AI-detected conditions to plain-language clinical suggestion prompts so that the copilot panel can surface non-prescriptive next-step cues without hard-coding them in the frontend.

**Acceptance Criteria:**
- [x] `aortica/api/clinical_suggestions.py` with a `CONDITION_SUGGESTIONS` dict mapping each class name (from RHYTHM_CLASSES, STRUCTURAL_CLASSES, ISCHAEMIA_CLASSES) to a `ClinicalSuggestion` dataclass containing: `prompt` (str, ≤100 chars), `urgency` ("routine" | "prompt" | "urgent" | "emergent"), `rationale` (str, 1–2 sentence clinical justification)
- [x] Populated for all high-severity conditions at minimum: STEMI territories, VT, VF, Wellens, de Winter, Brugada, WPW, severe hyperkalaemia, complete AV block, LVSD
- [x] `GET /api/v1/suggestions/{condition_name}` endpoint returning the `ClinicalSuggestion` for a given class
- [x] Inference response (`POST /api/v1/predict`) optionally includes suggestions for active findings when `include_suggestions=true` query param is set
- [x] Suggestions are loaded from an editable JSON file (`data/clinical_suggestions.json`) so clinicians can customize prompts without code changes
- [x] Unit tests verifying: all high-severity conditions have entries, JSON round-trip, API endpoint 200/404 behavior, inference response inclusion
- [x] Typecheck passes

---

### US-050: Second Reader Comparison Mode
**Description:** As a cardiologist, I want to enter my interpretation and compare it against the AI so that I can identify discrepancies I might have missed.

**Acceptance Criteria:**
- [x] `SecondReaderMode` React component with structured interpretation input (checkboxes for common findings + free-text)
- [x] `POST /api/v1/compare` endpoint accepting cardiologist interpretation + ECG reference
- [x] Backend comparison logic: maps cardiologist findings to model output classes, identifies agreements and discrepancies
- [x] Visual diff: green (agreement), red (AI found but cardiologist missed), yellow (cardiologist found but AI didn't)
- [x] Discrepancy severity ranking based on clinical importance
- [x] Unit tests for comparison logic
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-051: Edge-Case Spotlight Panel
**Description:** As a cardiologist, I want a dedicated panel highlighting rare but dangerous findings so that edge-case conditions are never buried in routine results.

**Acceptance Criteria:**
- [x] `EdgeCaseSpotlight` React component filtering for low-prevalence, moderate-to-high confidence findings
- [x] Configurable edge-case condition list (WPW, Brugada pattern, subtle flutter, fascicular VT, de Winter T-waves, Wellens syndrome, aVR ST-elevation)
- [x] Each flagged item shows: condition, confidence, why it's flagged as edge-case (prevalence note)
- [x] Visual emphasis: distinct panel styling with pulsing indicator for new edge-case detections
- [x] Links to explanation cards for each finding
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-052: Explanation Cards
**Description:** As a clinician, I want detailed explanation cards per finding so that I understand exactly why the AI flagged each condition and what clinical actions to consider.

**Acceptance Criteria:**
- [x] `ExplanationCard` React component showing per-finding detail view
- [x] Sections: named ECG features driving detection (from XAI), confidence interval (from conformal prediction), clinical reference text
- [x] **Clinical Suggestions section**: displays the `ClinicalSuggestion.prompt` and `rationale` from US-049b with urgency color-coding (routine=grey, prompt=yellow, urgent=orange, emergent=red); labelled "Suggested Next Steps" with a subtitle clarifying these are AI-generated prompts requiring clinician judgment
- [x] Feature attributions displayed as a ranked bar chart (top-3 features with delta scores)
- [x] Confidence interval displayed as a range bar
- [x] Placeholder section for "Similar historical cases" (to be implemented in Phase 4)
- [x] Expandable/collapsible within the findings panel
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-053: Clinician Feedback Collection API
**Description:** As a developer, I want to collect clinician feedback on AI findings so that future model calibration and improvement can be data-driven.

**Acceptance Criteria:**
- [x] `POST /api/v1/feedback` endpoint accepting: ECG reference ID, finding ID, action (accept/reject/modify), optional comment, clinician ID
- [x] `aortica/api/feedback.py` with Pydantic models and SQLite-backed storage
- [x] `GET /api/v1/feedback/stats` returning aggregate feedback statistics (agreement rate, most-rejected findings)
- [x] React feedback buttons (accept, reject, modify) on each finding in the copilot panel
- [x] Unit tests for feedback CRUD and stats aggregation
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Offline Infrastructure

---

### US-054: SQLite Local Result Storage with Encryption
**Description:** As a deployment admin, I want ECG results stored locally with encryption so that patient data is protected at rest on edge devices.

**Acceptance Criteria:**
- [x] `aortica/sync/` subpackage with `result_store.py`
- [x] `ResultStore` class wrapping SQLite with schema: `results` table (id, ecg_hash, predictions_json, quality_json, timestamp, synced)
- [x] AES-256 encryption for the predictions_json column (using `cryptography` library Fernet)
- [x] CRUD methods: `store_result()`, `get_result()`, `list_results()`, `delete_result()`
- [x] Database file created automatically in configurable directory
- [x] Unit tests for CRUD operations and encrypted vs. plaintext verification
- [x] Typecheck passes

---

### US-055: Offline-First Sync Engine with Vector Clocks
**Description:** As a deployment admin, I want results to sync automatically when connectivity is available so that offline devices don't lose data.

**Acceptance Criteria:**
- [x] `aortica/sync/sync_engine.py` with `SyncEngine` class
- [x] Vector clock per device for conflict resolution
- [x] `queue_for_sync(result_id)` marks results as pending upload
- [x] `sync_to_remote(remote_url)` uploads pending results via HTTPS POST
- [x] `pull_from_remote(remote_url)` downloads new results from central server
- [x] Conflict resolution: last-writer-wins with vector clock comparison, no data loss
- [x] Unit tests with mock HTTP server
- [x] Typecheck passes

---

### US-056: Sync Configuration and Bandwidth Management
**Description:** As a deployment admin, I want configurable sync thresholds so that sync doesn't consume excessive bandwidth on limited connections.

**Acceptance Criteria:**
- [x] `aortica/sync/config.py` with `SyncConfig` dataclass
- [x] Configurable fields: sync_interval_minutes (default 30), min_bandwidth_kbps (default 256), max_batch_size (default 20), remote_url, device_id
- [x] `check_connectivity(url)` function testing network availability and estimating bandwidth
- [x] Auto-sync scheduler that respects bandwidth thresholds
- [x] Anonymization function stripping patient metadata before sync
- [x] YAML config file loading
- [x] Unit tests for config loading, connectivity check, and anonymization
- [x] Typecheck passes

---

#### Docker & Documentation

---

### US-057: Docker Images for Multi-Architecture Deployment
**Description:** As a DevOps engineer, I want Docker images for server (amd64) and edge (arm64) so that Aortica can be deployed consistently across environments.

**Acceptance Criteria:**
- [x] `Dockerfile.server` for amd64: Python 3.12, installs `aortica[api,cli]`, exposes port 8000, runs uvicorn
- [x] `Dockerfile.edge` for arm64: Python 3.12 slim, installs `aortica[cli,edge]`, optimized for small image size
- [x] `docker-compose.yml` for local development (API server + frontend dev server)
- [x] `.dockerignore` excluding tests, docs, and development files
- [x] Build instructions in README
- [x] Dockerfiles pass `hadolint` linting (no critical warnings)
- [x] Typecheck passes

---

### US-108: Docker Compose Full-Stack Development Environment
**Description:** As a developer or deployment engineer, I want a single `docker-compose up` command that spins up the full Aortica stack so that onboarding takes minutes rather than hours.

**Background and rationale:** US-057 defines individual Dockerfiles for server (amd64) and edge (arm64), and a basic `docker-compose.yml` for local development. But the full Aortica stack involves: FastAPI backend, React frontend (dev or production), ONNX edge model serving, documentation site, and SQLite result storage. A comprehensive Docker Compose configuration ensures reproducible development environments and simplifies production-like local testing.

**Acceptance Criteria:**
- [x] `docker-compose.full.yml` defining services: `api` (FastAPI backend with ONNX Runtime), `frontend` (React dev server or nginx-served production build), `docs` (MkDocs serve), `edge` (edge model ONNX Runtime server on arm64 emulation or native)
- [x] Shared volumes: `./data/` for SQLite databases, `./models/` for model checkpoints, `./logs/` for application logs
- [x] Environment file template (`.env.example`) with all configurable variables: `AORTICA_MODEL_PATH`, `AORTICA_SECRET_KEY`, `AORTICA_OAUTH_CLIENT_ID`, `AORTICA_SYNC_URL`, `AORTICA_LOG_LEVEL`
- [x] Health check configuration for all services (Docker `HEALTHCHECK` instructions)
- [x] `make dev` Makefile target that runs `docker-compose -f docker-compose.full.yml up --build` with sensible defaults
- [x] `make prod` target that builds production images and runs with nginx reverse proxy and TLS termination (self-signed cert for local testing)
- [x] Service dependency ordering: API starts before frontend, model download completes before API accepts requests
- [x] Documentation in `docs/deployment/DOCKER_QUICKSTART.md` covering: prerequisites, first-run setup, service architecture diagram, common troubleshooting
- [x] Unit tests: Dockerfile lint via hadolint, compose config validation via `docker-compose config`
- [x] Typecheck passes

---

### US-058: MkDocs Documentation Site
**Description:** As a developer, I want a documentation site so that users can find API reference, deployment guides, and clinical background in one place.

**Acceptance Criteria:**
- [x] `docs/` directory with MkDocs project (`mkdocs.yml` using Material theme)
- [x] Sections: Getting Started, API Reference, CLI Reference, Deployment Guide, Clinical Background, Contributing
- [x] API reference auto-generated from docstrings (using `mkdocstrings`)
- [x] `mkdocs serve` runs local preview
- [x] `mkdocs build` produces static site in `site/`
- [x] GitHub Actions step to deploy docs to GitHub Pages on push to main
- [x] Typecheck passes

---

### US-109: aortica.io Landing Page and Distribution Portal
**Description:** As a potential user or institutional evaluator, I want a professional landing page at aortica.io so that I can understand what Aortica does, download it, access documentation, and try a live demo — all from a single entry point.

**Background and rationale:** The PRD introduction states Aortica is "distributed via aortica.io" with "download links, documentation, and demo assets." This story delivers that landing page as a static site (deployable to GitHub Pages, Netlify, or Vercel) that serves as the public face of the project.

**Acceptance Criteria:**
- [x] `landing/` directory with static site (HTML/CSS/JS or lightweight framework like Astro/11ty)
- [x] Hero section: project name, tagline ("AI ECG Copilot — Open Source"), animated ECG waveform background, prominent CTA buttons: "Get Started" → docs, "Download" → releases, "Live Demo" → hosted demo
- [x] Features section: 6 feature cards covering multi-task AI, explainability, edge deployment, federated learning, EHR integration, and regulatory readiness — each linking to relevant docs
- [x] Clinical mission section: adapted from PRD Clinical Mission content, emphasizing the copilot identity and hard-case detection capability
- [x] Download section: links to PyPI (`pip install aortica`), Docker Hub images, GitHub releases (APK, ONNX models), and HuggingFace Hub (pre-trained checkpoints)
- [x] Demo section: embedded or linked interactive demo with a sample ECG showing multi-task predictions, XAI overlay, and copilot panel (can use static screenshot + link to hosted instance initially)
- [x] Documentation link: routes to MkDocs site (US-058)
- [x] Footer: GitHub repo link, license (Apache 2.0), citation info, community links
- [x] SEO: proper meta tags, Open Graph tags for social sharing, structured data (Schema.org SoftwareApplication)
- [x] Mobile-responsive design
- [x] GitHub Actions deployment to GitHub Pages on push to `main`
- [x] Typecheck passes (if using TypeScript framework)

---

#### LMIC & Raspberry Pi Deployment

---

### US-059: Raspberry Pi Deployment Profile
**Description:** As a field deployment engineer, I want a Raspberry Pi deployment profile so that Aortica can run on low-cost ARM hardware in rural clinics.

**Acceptance Criteria:**
- [x] `aortica/edge/deploy_profiles.py` with `RaspberryPiProfile` dataclass (model_path, quantization=INT8, max_memory_mb=512, target_latency_ms=350)
- [x] `create_pi_image_script.sh` shell script that assembles: Python environment, edge model, CLI tool, systemd service file
- [x] Systemd service file (`aortica-edge.service`) for auto-start on boot
- [x] `aortica predict` works with edge model on ARM (ONNX Runtime ARM64)
- [x] Documentation: hardware requirements, SD card preparation, first-run instructions
- [x] Unit tests for profile configuration
- [x] Typecheck passes

---

### US-060: CHW-Facing Simplified Output Interface
**Description:** As a community health worker, I want plain-language risk tiers instead of detailed medical findings so that I can act on ECG results without cardiology training.

**Acceptance Criteria:**
- [x] `aortica.edge.simplify_output(multi_task_output, thresholds=None)` function
- [x] Maps multi-task predictions to three tiers: 'Low risk — no immediate action', 'Refer for assessment — schedule follow-up', 'Urgent referral recommended — seek immediate care'
- [x] Tier assignment based on configurable confidence thresholds per condition category
- [x] Returns `SimplifiedReport` dataclass with tier, key finding summary (1-2 sentences), and recommended actions
- [x] Localization support: output strings loadable from JSON locale files (English default)
- [x] Unit tests with synthetic predictions mapping to each tier
- [x] Typecheck passes

---

### US-061: Inference Profiling and Power Optimization
**Description:** As a deployment engineer, I want inference profiling so that I can verify the edge model meets latency and power targets on ARM hardware.

**Acceptance Criteria:**
- [x] `aortica.edge.profile_inference(model_path, input_data, n_runs=100)` function
- [x] Measures: mean/p50/p95 latency, peak memory usage, model size on disk
- [x] Returns `InferenceProfile` dataclass with all measurements
- [x] Power estimation based on latency × TDP for known hardware profiles (RPi4: 4W, Jetson Nano: 5W)
- [x] CLI integration: `aortica profile <model_path>` command
- [x] Unit tests with synthetic ONNX model
- [x] Typecheck passes

---

### US-061b: LMIC Pilot Deployment Package
**Description:** As a field deployment engineer, I want a complete LMIC pilot deployment package so that Aortica can be deployed at 2 target rural/LMIC pilot sites with minimal on-site expertise and maximal reliability under resource constraints.

**Background and rationale:** The edge model and Raspberry Pi profile (US-038–US-041, US-059) provide the inference engine, and the CHW-facing simplified output (US-060) provides the user-facing output layer. This story packages those components into a deployable, field-tested pilot kit — including site-specific deployment guides, power consumption validation, training materials for community health workers, and monitoring infrastructure. Without this story, individual edge components exist but are not assembled into a deployable whole.

**Acceptance Criteria:**
- [x] `docs/deployment/LMIC_PILOT_GUIDE.md` covering: site prerequisites (power, connectivity, hardware), SD card image preparation (reuses US-059 `create_pi_image_script.sh`), first-boot walkthrough with screenshots, daily operational procedures, troubleshooting guide for common failure modes (power loss, SD card corruption, serial capture timeout)
- [x] `docs/deployment/CHW_TRAINING.md` with plain-language, image-heavy training guide for community health workers covering: device power-on, ECG acquisition, result interpretation (three-tier output from US-060), when to refer, and data sync verification
- [x] Localization support: deployment guide and CHW training materials translatable via JSON locale files (English and French provided; Spanish and Swahili as template stubs)
- [x] `aortica.edge.validate_power_consumption(model_path, hardware_profile, n_inferences=50)` function measuring per-inference energy consumption; asserts < 200 mW sustained on ARM hardware profiles (RPi4 at 4W TDP × duty cycle)
- [x] Power optimization: inference duty-cycling configuration in `deploy_profiles.py` — model loads on-demand per ECG rather than keeping ONNX session resident, reducing idle power draw
- [x] `aortica.edge.SiteMonitor` class providing: daily inference count, error rate, sync status, storage utilization, and last-sync timestamp; exposes `GET /edge/status` endpoint on the local edge server
- [x] `aortica edge site-report` CLI command generating a daily site activity summary (inferences, errors, sync status) for remote monitoring
- [x] `docs/deployment/PILOT_CHECKLIST.md` — pre-deployment checklist covering: hardware verification, network connectivity test, edge model validation (reuses US-041), CHW competency sign-off, ethics/IRB documentation
- [x] Unit tests for power consumption validation, site monitor, and site report generation
- [x] Typecheck passes

---

### US-061c: Android ONNX Mobile Application
**Description:** As a community health worker or clinician in the field, I want an Android app that runs Aortica's edge model locally on my phone so that I can get AI ECG analysis anywhere — even without internet access or a Raspberry Pi.

**Background and rationale:** While the PWA (US-042b) provides browser-based offline inference via WASM, a native Android app using ONNX Runtime Mobile delivers lower latency, better battery efficiency, and a more reliable offline experience on low-end Android devices (Snapdragon 660+) common in LMIC settings. The app complements the Raspberry Pi deployment path — providing a zero-infrastructure alternative when a dedicated edge device is unavailable. Single-lead input support is critical for low-cost, handheld ECG devices (e.g., AliveCor KardiaMobile) prevalent in rural settings.

**Acceptance Criteria:**
- [x] `mobile/android/` directory with Android Studio project (Kotlin, min SDK 26 / Android 8.0)
- [x] ONNX Runtime Mobile (`onnxruntime-android`) integrated for local inference using the INT8 quantized edge model (from US-040)
- [x] ECG input support: single-lead (Lead I or Lead II) and 6-lead (limb leads) via file import (CSV, WFDB) or direct Bluetooth LE capture from compatible devices (AliveCor protocol stub, extensible interface)
- [x] Inference pipeline: signal preprocessing (resampling to 500 Hz, zero-padding missing leads) → ONNX edge model → multi-task output → simplified output mapping (reuses US-060 tier logic)
- [x] Plain-language output tiers displayed prominently: **'Low risk'** (green), **'Refer for assessment'** (amber), **'Urgent referral recommended'** (red) with 1–2 sentence finding summary and recommended actions
- [x] Inference latency < 200 ms on Snapdragon 660+ (single-lead input, edge model)
- [x] **Fully offline operation:** app functions with no network connectivity after initial install; model bundled in APK (or downloaded on first launch with progress indicator)
- [x] Anonymized audit log: stores each inference (timestamp, input hash, tier result, inference latency) in local SQLite; auto-syncs to configured remote endpoint when connectivity detected (reuses sync protocol from US-055/US-056)
- [x] Audit log sync strips all patient-identifiable metadata before upload (reuses anonymization function from US-056)
- [x] Settings screen: configurable remote sync URL, device ID, sync frequency, language selection
- [x] Localization: UI strings externalized to `res/values-*/strings.xml`; English default, locale stubs for French, Spanish, Swahili matching LMIC deployment guide (US-061b)
- [x] Accessibility: high-contrast mode for outdoor/bright-light use, large touch targets (≥48dp), screen reader support via Android content descriptions
- [x] `aortica.edge.export_mobile_model(model_path, output_path)` Python utility that packages the INT8 ONNX model with Android-specific metadata (input/output names, shape expectations, version tag)
- [x] Unit tests (Android instrumentation tests via Espresso/JUnit4): model loading, inference pipeline with synthetic ECG, tier mapping, audit log CRUD, sync queue behavior
- [x] Typecheck passes (Kotlin)

---

### US-110: Android App Build and Distribution Pipeline
**Description:** As a release manager, I want an automated build and distribution pipeline for the Aortica Android app so that new releases are built, signed, tested, and distributed consistently alongside backend releases.

**Background and rationale:** US-061c defines the Android app's features and acceptance criteria. This story covers the CI/CD infrastructure needed to build, test, sign, and distribute the APK — which is a separate workflow from the Python package and Docker image pipelines.

**Acceptance Criteria:**
- [x] `mobile/android/` Gradle build configuration with release signing config (keystore path, alias, passwords via environment variables)
- [x] GitHub Actions workflow `android_build.yml`: triggers on push to `main` or version tag; runs lint (ktlint), unit tests, instrumentation tests (on Firebase Test Lab or local emulator), builds signed release APK and AAB
- [x] APK includes the latest INT8 ONNX edge model bundled in `assets/` (copied from `aortica/edge/` artifacts during build)
- [x] Version management: app version code and name derived from git tag (e.g., tag `v0.3.0` → versionCode 300, versionName "0.3.0")
- [x] Distribution channels:
  - GitHub Releases: signed APK attached to each version tag release
  - Sideload distribution page on aortica.io (US-109) with QR code linking to latest APK
  - Google Play Store: AAB uploaded via Gradle Play Publisher plugin (manual promotion from internal → production track)
- [x] OTA model update mechanism: app checks configured endpoint for newer model version on startup (when online); downloads and caches new ONNX model without requiring app update; falls back to bundled model if download fails
- [x] `POST /api/v1/mobile/model-manifest` API endpoint returning: latest model version, download URL, SHA-256 hash, minimum app version required
- [x] App size budget: APK < 25 MB (excluding model), total installed size < 40 MB (including bundled model)
- [x] Unit tests for version derivation, model manifest parsing, and OTA update logic
- [x] Typecheck passes (Kotlin)

---

### Phase 3 — Federated Learning & Equity

---

#### Federated Learning SDK

---

### US-062: Flower Federated Learning Server Scaffold
**Description:** As an ML engineer, I want a Flower-based federated learning server so that multiple institutions can collaboratively train Aortica models without sharing raw data.

**Acceptance Criteria:**
- [x] `aortica/federated/` subpackage with `__init__.py`
- [x] `aortica.federated.FLServer` class wrapping `flwr.server.start_server()` with configurable rounds, min clients, and strategy
- [x] Supports FedAvg aggregation strategy out of the box
- [x] Server configuration via YAML (num_rounds, min_fit_clients, min_evaluate_clients, server_address)
- [x] Logs per-round aggregated metrics (loss, per-task F1)
- [x] Unit tests with mock Flower server
- [x] Typecheck passes

---

### US-063: Flower Federated Learning Client Wrapper
**Description:** As a site administrator, I want a Flower client that wraps the Aortica training pipeline so that my site can participate in federated training with minimal setup.

**Acceptance Criteria:**
- [x] `aortica.federated.AorticaFlowerClient` class implementing `flwr.client.NumPyClient`
- [x] `get_parameters()` returns model weights as numpy arrays
- [x] `fit()` runs local training (configurable epochs, batch size) on site-local data and returns updated weights + num_examples
- [x] `evaluate()` runs local evaluation and returns loss + per-task metrics
- [x] Client startup function accepting data path + server address
- [x] Unit tests with synthetic data and mock server connection
- [x] Typecheck passes

---

### US-064: Pluggable Aggregation Strategies (FedProx, SCAFFOLD)
**Description:** As an ML engineer, I want pluggable aggregation strategies beyond FedAvg so that federated training is robust to non-IID data distributions across sites.

**Acceptance Criteria:**
- [x] `aortica.federated.FedProxStrategy` implementing Flower Strategy with proximal term (μ configurable, default 0.01)
- [x] `aortica.federated.SCAFFOLDStrategy` implementing SCAFFOLD with server/client control variates
- [x] Strategy selectable via server config YAML (`strategy: fedavg | fedprox | scaffold`)
- [x] Unit tests verifying FedProx proximal penalty is applied, SCAFFOLD control variates update correctly
- [x] Typecheck passes

---

### US-065: Differential Privacy Integration (OpenDP)
**Description:** As a privacy officer, I want differential privacy applied during federated training so that individual patient data cannot be reconstructed from model updates.

**Acceptance Criteria:**
- [x] `aortica.federated.DPWrapper` class wrapping the Flower client with per-round gradient clipping and Gaussian noise injection
- [x] Uses OpenDP library for privacy accounting (Rényi DP composition)
- [x] Configurable privacy budget ε (default 1.0), δ (default 1e-5), and max_grad_norm (default 1.0)
- [x] Privacy budget tracker that logs cumulative ε spent and warns when budget approaches exhaustion
- [x] Unit tests verifying noise is applied, budget decrements per round, and gradients are clipped
- [x] Typecheck passes

---

### US-066: Secure Aggregation via CKKS Homomorphic Encryption
**Description:** As a security engineer, I want homomorphic encryption for gradient exchange so that model updates are encrypted in transit and at rest on the aggregation server.

**Acceptance Criteria:**
- [x] `aortica.federated.SecureAggregator` class using TenSEAL (CKKS scheme) for encrypting/decrypting model weight updates
- [x] Client encrypts weight deltas before sending to server; server aggregates in encrypted space; result decrypted by clients
- [x] Key generation and distribution protocol documented
- [x] Configurable polynomial modulus degree and coefficient modulus for security/performance trade-off
- [x] Unit tests verifying encrypted aggregation produces same result as plaintext aggregation (within floating-point tolerance)
- [x] Typecheck passes

---

### US-067: Data Use Agreement Template and Onboarding Guide
**Description:** As a site administrator, I want a DUA template and onboarding guide so that new federated learning partners can join with minimal legal and technical friction.

**Acceptance Criteria:**
- [x] `docs/federated/DUA_TEMPLATE.md` with standard data use agreement covering: data retention, model update usage, publication rights, withdrawal process
- [x] `docs/federated/ONBOARDING.md` with step-by-step guide: prerequisites, client installation, data preparation, connection test, first federated round
- [x] CLI command `aortica federated test-connection <server_url>` that verifies client can reach server and authenticate
- [x] Unit tests for connection test command
- [x] Typecheck passes

---

### US-068: Federated Training CLI and Configuration
**Description:** As an ML engineer, I want CLI commands for federated training so that server and client operations are scriptable and reproducible.

**Acceptance Criteria:**
- [x] `aortica federated server <config.yaml>` starts the FL server with configured strategy and rounds
- [x] `aortica federated client <config.yaml>` starts the FL client connecting to specified server
- [x] YAML config schema documented with all configurable fields
- [x] Both commands share the Click group with existing CLI commands
- [x] `--dry-run` flag that validates config without starting training
- [x] Unit tests for CLI invocation
- [x] Typecheck passes

---

### US-112: Federated Model Release Pipeline
**Description:** As a release manager, I want an automated pipeline for releasing federated-trained models so that aggregated models from FL rounds are validated, versioned, and distributed alongside centrally-trained checkpoints.

**Background and rationale:** US-036b covers centrally-trained model distribution via HuggingFace Hub. But federated models emerge from a different workflow: FL rounds produce aggregated weights that must be validated against the equity gate (US-069), benchmarked (US-028/US-079), and published as a distinct model variant. Without this story, there's no defined process for going from FL aggregation output to a released, trustworthy model.

**Acceptance Criteria:**
- [x] `aortica.federated.release_pipeline(aggregated_weights_path, base_version, config)` function orchestrating the full release workflow
- [x] Pipeline steps: (1) load aggregated weights into AorticaModel, (2) run full benchmark suite (US-028), (3) run equity gate (US-069), (4) run regulatory gate (US-097), (5) export ONNX + INT8 edge model (US-037/US-040), (6) generate performance card (US-070), (7) push to HuggingFace Hub with `federated-` version prefix
- [x] Federated models versioned as `aortica-federated-v{version}-r{round}.pt` (e.g., `aortica-federated-v0.3.0-r50.pt`) to distinguish from centrally-trained models
- [x] Model card for federated releases includes: participating site count (anonymized), total training samples contributed, aggregation strategy used, differential privacy parameters (ε spent), equity gate results per site region
- [x] `load_pretrained(variant='federated')` fetches the latest federated model; `load_pretrained(variant='federated', version='v0.3.0-r50')` fetches a specific federated release
- [x] CLI command `aortica federated release --weights <path> --version <v>` runs the full release pipeline
- [x] Abort-on-failure: pipeline halts and reports if any gate (equity, regulatory, benchmark threshold) fails
- [x] GitHub Actions workflow `federated_release.yml` triggered manually or on completion of FL server run
- [x] Unit tests with synthetic aggregated weights verifying pipeline orchestration, gating logic, and version naming
- [x] Typecheck passes

---

### US-113: Federated Learning Monitoring Dashboard
**Description:** As an FL campaign coordinator, I want a web dashboard showing federated training progress so that I can monitor round completion, per-site contributions, convergence, and privacy budget consumption in real time.

**Acceptance Criteria:**
- [x] `FLDashboard` React page at route `/federated` (protected: requires admin role)
- [x] **Campaign overview panel:** active FL campaign name, current round number, total rounds configured, aggregation strategy, start timestamp, elapsed time
- [x] **Per-round metrics chart:** line chart showing aggregated loss and per-task F1 across rounds (updates after each round completion)
- [x] **Site participation panel:** table of connected sites (anonymized site IDs) with: connection status (online/offline), samples contributed, last communication timestamp, local training time
- [x] **Privacy budget panel:** cumulative ε spent per site, projected ε at campaign end, visual warning when any site approaches budget exhaustion (> 80% of configured ε)
- [x] **Convergence indicators:** gradient norm trend, loss plateau detection, early stopping recommendation
- [x] `GET /api/v1/federated/status` API endpoint returning current FL campaign state
- [x] `GET /api/v1/federated/rounds` API endpoint returning per-round aggregated metrics
- [x] `GET /api/v1/federated/sites` API endpoint returning anonymized site participation stats
- [x] FL server (US-062) updated to persist per-round metrics to SQLite for dashboard consumption
- [x] WebSocket or polling-based live update (configurable interval, default 30s)
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Equity Infrastructure

---

### US-069: Equity Gating CI Check
**Description:** As a release manager, I want automated equity checks in CI so that no model release has statistically significant demographic performance disparities.

**Acceptance Criteria:**
- [x] `aortica.evaluation.equity_gate(benchmark_report, alpha=0.05, correction='bonferroni')` function
- [x] Compares per-task AUC across sex groups (male/female) using permutation test or bootstrap CI
- [x] Compares per-task AUC across age deciles (30–80) using same statistical test
- [x] Returns `EquityGateResult` with pass/fail, per-group metrics, p-values, and failing comparisons
- [x] Fails if any comparison shows p < alpha after Bonferroni correction for classes with N>100 test examples
- [x] GitHub Actions step that runs equity gate after benchmark and blocks release on failure
- [x] Unit tests with synthetic predictions showing passing and failing scenarios
- [x] Typecheck passes

---

### US-070: Public Performance Card Generator
**Description:** As a release manager, I want auto-generated performance cards so that every model release has transparent, demographic-stratified performance documentation.

**Acceptance Criteria:**
- [x] `aortica.evaluation.generate_performance_card(benchmark_report, model_version, output_dir)` function
- [x] Generates `PERFORMANCE_CARD.md` with: model version, training data summary, per-task metrics (AUC, F1, sensitivity, specificity), demographic subgroup breakdowns (age decile, sex), equity gate results
- [x] Generates `performance_card.csv` with the same data in tabular format
- [x] Card includes timestamp, dataset split info, and reproducibility hash (model weights SHA-256)
- [x] CLI command `aortica performance-card <benchmark_report.json> --version <v>` generates the card
- [x] Unit tests with synthetic benchmark report producing valid markdown and CSV
- [x] Typecheck passes

---

### US-071: Non-Western Site Validation Tracker
**Description:** As a release manager, I want to track non-Western site validations so that v-stable releases require at least 2 non-Western site validations.

**Acceptance Criteria:**
- [x] `aortica.evaluation.SiteValidationRegistry` class tracking validation results per site (site_id, region, dataset_size, benchmark_report, timestamp)
- [x] `register_validation(site_id, region, benchmark_report)` stores validation result
- [x] `check_release_readiness()` returns pass/fail requiring ≥2 non-Western region validations
- [x] Region classification: Western (North America, Western Europe, Australia/NZ) vs. non-Western (all others)
- [x] Persists registry to JSON file
- [x] Unit tests for registration, region classification, and readiness check
- [x] Typecheck passes

---

### US-114: Site Validation Registry UI
**Description:** As a release manager, I want a web interface for the non-Western site validation registry so that I can register validation results, view per-site benchmark reports, and check release readiness without using the Python API directly.

**Acceptance Criteria:**
- [x] `SiteValidationPage` React page at route `/validation/sites` (protected: requires admin role)
- [x] **Registry table:** list of all registered validation sites with: site ID, region, dataset size, validation date, overall pass/fail, link to full benchmark report
- [x] **Add validation form:** site ID, region (dropdown with Western/non-Western auto-classification), upload benchmark report JSON, dataset size
- [x] **Release readiness indicator:** prominent badge showing whether v-stable requirements are met (≥2 non-Western validations), with breakdown of which sites satisfy the requirement
- [x] **Region map visualization:** world map (lightweight SVG) with markers showing validated site locations, color-coded by region classification
- [x] `POST /api/v1/validation/sites` API endpoint for registering a new site validation
- [x] `GET /api/v1/validation/sites` API endpoint returning all registered validations
- [x] `GET /api/v1/validation/readiness` API endpoint returning release readiness status
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Additional Task Capabilities — Edge-Case & Subtle Pattern Expansion

---

### US-072: Rare Arrhythmia Subtypes — Expanded Rhythm Head
**Description:** As a clinician, I want AI detection of rare but dangerous arrhythmia subtypes so that conditions like Brugada and CPVT are not missed.

**Acceptance Criteria:**
- [x] Extend `RHYTHM_CLASSES` list with 6 new classes: Brugada pattern (types 1–3 combined), short QT syndrome, catecholaminergic polymorphic VT (CPVT), fascicular VT, atypical atrial flutter, inappropriate sinus tachycardia
- [x] Update `RhythmHead` output dimension from 22 to 28 (both PyTorch and TF/Keras)
- [x] Update `compute_rhythm_loss` to handle 28-class output with class weights for new rare classes
- [x] Update all downstream code referencing RHYTHM_CLASSES count (training pipeline, evaluation, API responses)
- [x] Unit tests verifying new output shape, loss computation, and backward compatibility
- [x] Typecheck passes

---

### US-073: STEMI Mimics & Subtle Ischaemia — Expanded Ischaemia Head
**Description:** As an emergency physician, I want AI detection of STEMI mimics and subtle ischaemia patterns so that dangerous conditions masquerading as benign findings are caught.

**Acceptance Criteria:**
- [x] Extend `ISCHAEMIA_CLASSES` list with 5 new classes: early repolarization vs STEMI, de Winter T-wave pattern, Wellens syndrome (type A/B combined), aVR ST-elevation pattern, Sgarbossa criteria (LBBB/paced)
- [x] Update `IschaemiaHead` output dimension from 10 to 15 (both PyTorch and TF/Keras)
- [x] Update `compute_ischaemia_loss` to handle 15-class output with class weights
- [x] Update all downstream code referencing ISCHAEMIA_CLASSES count
- [x] Unit tests verifying new output shape, loss computation, and backward compatibility
- [x] Typecheck passes

---

### US-074: Strain Pattern Sub-Classifiers
**Description:** As a cardiologist, I want AI grading of strain patterns so that LV strain severity, RV strain in PE, and Takotsubo can be differentiated from generic structural findings.

**Acceptance Criteria:**
- [x] Extend `STRUCTURAL_CLASSES` list with 4 new classes: LV strain grade (mild/moderate/severe as ordinal), RV strain in PE (S1Q3T3 pattern), Takotsubo pattern, infiltrative cardiomyopathy strain
- [x] Update `StructuralHead` output dimension from 15 to 19 (both PyTorch and TF/Keras)
- [x] LV strain grade uses ordinal encoding (single sigmoid output mapped to severity thresholds)
- [x] Update `compute_structural_loss` to handle 19-class output
- [x] Update all downstream code referencing STRUCTURAL_CLASSES count
- [x] Unit tests verifying new output shape, loss computation, and ordinal encoding
- [x] Typecheck passes

---

### US-075: Metabolic & Drug Effect Detectors
**Description:** As a clinician, I want AI detection of metabolic emergencies and drug effects so that hyperkalaemia severity, hypothermia, and drug toxicity are flagged.

**Acceptance Criteria:**
- [x] Extend `ISCHAEMIA_CLASSES` list with 4 additional classes: hyperkalaemia severity grading (mild/moderate/severe as ordinal), hypothermia (Osborn waves), tricyclic antidepressant toxicity, digoxin effect vs toxicity
- [x] Update `IschaemiaHead` output dimension from 15 to 19 (both PyTorch and TF/Keras)
- [x] Hyperkalaemia grading uses ordinal encoding similar to strain grade
- [x] Update `compute_ischaemia_loss` to handle 19-class output
- [x] Update all downstream code referencing ISCHAEMIA_CLASSES count
- [x] Unit tests verifying new output shape and ordinal encoding
- [x] Typecheck passes

---

### US-076: Risk Prediction Refinement — Subclinical LVSD and SCD Risk
**Description:** As a clinician, I want refined risk scores for subclinical LV dysfunction and sudden cardiac death so that high-risk patients are identified earlier.

**Acceptance Criteria:**
- [x] Extend `RISK_OUTPUTS` list with 3 new outputs: ECG-predicted ejection fraction (continuous 0–1 scaled), progressive conduction disease trajectory score, sudden cardiac death risk score
- [x] Update `RiskHead` output dimension from 3 to 6 (both PyTorch and TF/Keras)
- [x] Update `compute_risk_loss` to handle 6-output regression with per-output ranking loss
- [x] Update all downstream code referencing RISK_OUTPUTS count
- [x] Unit tests verifying new output shape, loss computation, and ranking loss
- [x] Typecheck passes

---

### US-077: Multi-Task Model Assembly Update for Expanded Heads
**Description:** As an ML engineer, I want AorticaModel to seamlessly support the expanded task heads so that the unified model handles all new classes.

**Acceptance Criteria:**
- [x] `AorticaModel` auto-detects updated head dimensions from the head class constants
- [x] `MultiTaskOutput` dataclass updated to reflect new output dimensions
- [x] Forward pass works correctly with expanded heads (rhythm=28, structural=19, ischaemia=19, risk=6)
- [x] TF/Keras model builder (`build_aortica_model_tf`) updated for new dimensions
- [x] Framework parity validation updated for new output sizes
- [x] Unit tests verifying full forward pass with expanded heads, selective head disabling, and gradient flow
- [x] Typecheck passes

---

### US-078: Training Pipeline Update for Expanded Tasks
**Description:** As an ML engineer, I want the multi-task training pipeline to support the expanded head dimensions so that models can be trained with the new classes.

**Acceptance Criteria:**
- [x] `_TASK_NUM_OUTPUTS` mapping updated to reflect new dimensions (rhythm=28, structural=19, ischaemia=19, risk=6)
- [x] `_split_labels` correctly splits label tensors for new dimensions
- [x] Training config YAML supports per-class weights for new rare classes (higher weights for edge-case conditions)
- [x] Evaluation metrics computed for all new classes
- [x] ONNX export and edge model pipeline compatible with expanded heads
- [x] Unit tests for label splitting, loss computation, and metric evaluation with expanded dimensions
- [x] Typecheck passes

---

### US-079: Evaluation Harness Update for Expanded Tasks and Equity
**Description:** As a researcher, I want the benchmark harness to evaluate all expanded classes and produce equity-gated reports so that model releases are comprehensive and fair.

**Acceptance Criteria:**
- [x] `benchmark()` handles expanded output dimensions (28+19+19+6 = 72 total outputs)
- [x] Per-class metrics computed for all new classes including rare arrhythmia subtypes
- [x] Benchmark report integrates equity gate results (pass/fail per subgroup comparison)
- [x] Performance card generation triggered automatically after benchmark
- [x] Summary table includes new classes with clear labeling
- [x] Unit tests verifying metric computation for expanded class counts
- [x] Typecheck passes

---

### US-115: Federated Data Quality Gating
**Description:** As an FL campaign coordinator, I want automated data quality checks at each federated learning site so that sites with insufficient data quality, sample size, or label completeness are flagged before they participate in training rounds.

**Background and rationale:** US-063 defines the FL client wrapper that trains on site-local data, but there's no validation that the local data meets minimum quality standards. A site with noisy data, too few samples, or missing labels could degrade the aggregated model. This story adds pre-training quality gates that run at the client before joining an FL round.

**Acceptance Criteria:**
- [x] `aortica.federated.DataQualityGate` class performing pre-training validation on site-local data
- [x] Quality checks:
  - Minimum sample size: configurable threshold (default 500 ECGs), warns below 200, blocks below 100
  - Signal quality distribution: ≥70% of ECGs must have quality score ≥ 40 ("marginal" or better)
  - Label completeness: ≥80% of ECGs must have at least one diagnostic label
  - Label diversity: at least 3 of 4 task superclasses (rhythm, structural, ischaemia, risk) must have ≥10 positive examples
  - Format consistency: all ECGs must successfully pass through the preprocessing pipeline without error
- [x] `gate.validate(dataset)` returns `DataQualityReport` with pass/fail per check, detailed statistics, and recommendations
- [x] FL client (US-063) runs data quality gate before first FL round; reports results to server; server can be configured to exclude failing sites
- [x] `aortica federated validate-data <data_path>` CLI command for sites to pre-check their data before joining a campaign
- [x] Server-side configurable policy: `strict` (exclude failing sites), `warn` (include with warning), `permissive` (include all)
- [x] Unit tests with synthetic datasets hitting each quality threshold
- [x] Typecheck passes

---

### US-116: Model Version Comparison and A/B Analysis
**Description:** As an ML engineer, I want to compare two model versions side-by-side so that I can quantify the impact of federated training, expanded task heads, or any model update before releasing it.

**Acceptance Criteria:**
- [x] `aortica.evaluation.compare_models(model_a_path, model_b_path, dataset, tasks='all')` returns a `ModelComparisonReport`
- [x] Per-task delta metrics: ΔAUC, ΔF1, Δsensitivity, Δspecificity, ΔC-index for risk tasks
- [x] Statistical significance testing: paired bootstrap test per class, reporting p-values and 95% confidence intervals for each delta
- [x] Demographic subgroup comparison: per-group delta metrics (age decile, sex) to detect equity regressions
- [x] Regression detection: flags any class where model B performs statistically worse than model A (p < 0.05)
- [x] Generates `MODEL_COMPARISON.md` report with: version IDs, summary table of deltas, per-task breakdown, regression warnings, recommendation (upgrade/hold/investigate)
- [x] CLI command `aortica compare --model-a <path> --model-b <path> --dataset <path>` generates the comparison report
- [x] React page at `/compare` allowing upload of two benchmark reports (JSON) and displaying side-by-side metrics with delta highlighting (green for improvement, red for regression)
- [x] Unit tests with synthetic predictions showing improvement, regression, and mixed scenarios
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### Phase 4 — Regulatory & Scale

---

#### EHR & ECG System Integration

---

### US-080: FHIR R4 DiagnosticReport Output
**Description:** As a hospital integrator, I want Aortica to output FHIR R4 DiagnosticReport and Observation resources so that AI results can be ingested by any FHIR-compliant EHR.

**Acceptance Criteria:**
- [x] `aortica/integration/` subpackage with `__init__.py`
- [x] `aortica.integration.fhir.to_diagnostic_report(multi_task_output, patient_ref, ecg_metadata)` returns a FHIR R4 `DiagnosticReport` resource (JSON)
- [x] Generates child `Observation` resources for each positive finding with LOINC/SNOMED codes where available
- [x] Maps confidence scores to Observation `interpretation` (high/low/normal)
- [x] Risk scores encoded as FHIR `RiskAssessment` resources with probability values
- [x] Uses `fhir.resources` library for model validation
- [x] Output validates against FHIR R4 resource schemas
- [x] Unit tests with synthetic multi-task output producing valid FHIR JSON
- [x] Typecheck passes

---

### US-081: HL7 v2.x ORU^R01 Message Generation
**Description:** As a hospital integrator, I want Aortica to generate HL7 v2.x ORU^R01 messages so that AI results can be sent to legacy EHR systems via MLLP.

**Acceptance Criteria:**
- [x] `aortica.integration.hl7v2.to_oru_r01(multi_task_output, patient_id, order_id)` returns a valid HL7 v2.x ORU^R01 message string
- [x] OBX segments encode each active finding with local code, description, and confidence
- [x] Maps risk scores to numeric OBX segments with units
- [x] Handles special characters and segment delimiters per HL7 v2.x spec
- [x] Uses `hl7apy` library for message construction and validation
- [x] Unit tests verifying valid HL7 v2.x message structure and parsability
- [x] Typecheck passes

---

### US-082: DICOM SR Structured Report Write-Back
**Description:** As a PACS administrator, I want Aortica to write DICOM Structured Reports so that AI findings are stored alongside the original ECG in the imaging archive.

**Acceptance Criteria:**
- [x] `aortica.integration.dicom_sr.to_structured_report(multi_task_output, original_dicom_ref)` returns a `pydicom.Dataset` containing a DICOM SR object
- [x] SR content tree uses TID 2000 (Basic Diagnostic Imaging Report) structure
- [x] Each finding encoded as a CONTENT ITEM with concept name (coded entry), value (coded/text/numeric), and confidence
- [x] References original DICOM ECG instance via Referenced SOP Instance UID
- [x] SR validates against DICOM conformance (correct IOD modules present)
- [x] Unit tests with synthetic input producing valid DICOM SR dataset
- [x] Typecheck passes

---

### US-083: DICOM DIMSE C-STORE/C-FIND Client
**Description:** As a device integrator, I want DICOM DIMSE support so that Aortica can send and query ECG management systems like GE MUSE.

**Acceptance Criteria:**
- [x] `aortica.integration.dimse.DicomClient` class wrapping `pynetdicom` for DIMSE operations
- [x] `c_store(dataset, remote_ae, remote_host, remote_port)` sends a DICOM dataset (ECG waveform or SR) to a remote AE
- [x] `c_find(query, remote_ae, remote_host, remote_port)` queries for ECG studies matching patient/date/modality criteria
- [x] Handles association negotiation, presentation context selection, and timeouts
- [x] Configurable local AE title and port
- [x] Unit tests with mock DICOM SCP verifying association and data exchange
- [x] Typecheck passes

---

### US-084: SCP-ECG Serial Port Capture
**Description:** As a field deployment engineer, I want SCP-ECG serial port capture so that Aortica can ingest ECGs directly from legacy carts that output via serial/USB.

**Acceptance Criteria:**
- [x] `aortica.integration.serial_capture.SCPSerialCapture` class wrapping `pyserial`
- [x] `listen(port, baud_rate=115200, timeout=30)` blocks until a complete SCP-ECG frame is received
- [x] Parses received bytes into an `ECGRecord` using the existing `read_scp()` reader
- [x] Handles framing, CRC validation, and incomplete transmission retries
- [x] Configurable serial port, baud rate, and timeout
- [x] Returns captured `ECGRecord` or raises `CaptureTimeoutError`
- [x] Unit tests with mock serial port and synthetic SCP-ECG data
- [x] Typecheck passes

---

### US-085: SMART on FHIR Launch Context Support
**Description:** As a hospital integrator, I want SMART on FHIR launch support so that Aortica's web UI can be launched from within an EHR with patient context pre-populated.

**Acceptance Criteria:**
- [x] `aortica/api/smart_on_fhir.py` with SMART on FHIR launch handler
- [x] Implements SMART App Launch Framework (OAuth 2.0 with EHR launch sequence)
- [x] Extracts `patient`, `encounter`, and `fhirServer` from launch context
- [x] Pre-populates patient reference in API calls when launched from EHR
- [x] FastAPI endpoint `GET /api/v1/smart/launch` handles the SMART redirect
- [x] Configuration via environment variables (client_id, redirect_uri, fhir_server_url)
- [x] Unit tests for launch sequence parsing and OAuth token exchange
- [x] Typecheck passes

---

### US-086: Worklist Prioritization Module
**Description:** As a cardiologist, I want AI-sorted worklists so that the most urgent ECGs are reviewed first, reducing time-to-interpretation for critical findings.

**Acceptance Criteria:**
- [x] `aortica.integration.worklist.WorklistPrioritizer` class
- [x] `prioritize(results: list[MultiTaskOutput])` returns sorted list with urgency score (0–100) per ECG
- [x] Urgency scoring based on configurable rules: critical findings (STEMI, VT, VF, severe hyperkalaemia) = highest priority; moderate findings (new AF, Brugada, Wellens) = medium; routine = low
- [x] Returns `PrioritizedWorklist` with items sorted by urgency, each showing: ECG ID, urgency score, top finding, recommended action
- [x] `POST /api/v1/worklist/prioritize` API endpoint accepting batch results and returning prioritized list
- [x] Configurable urgency rules via YAML file
- [x] Unit tests with synthetic results spanning all priority tiers
- [x] Typecheck passes

---

### US-117: FHIR Subscription and Webhook Notification Service
**Description:** As a hospital integrator, I want FHIR Subscription resources and webhook notifications so that the EHR is automatically alerted when Aortica detects critical findings, without polling the API.

**Acceptance Criteria:**
- [x] `aortica.integration.fhir_subscription.SubscriptionManager` class managing FHIR R4 Subscription resources
- [x] Supports `rest-hook` channel type: sends HTTP POST to a configured EHR webhook URL when a matching event occurs
- [x] Subscription criteria: configurable filters by finding severity (critical/warning), specific conditions (STEMI, VT, VF, Brugada), urgency score threshold
- [x] `POST /api/v1/subscriptions` creates a new subscription with criteria and webhook URL
- [x] `GET /api/v1/subscriptions` lists active subscriptions
- [x] `DELETE /api/v1/subscriptions/:id` removes a subscription
- [x] When a `POST /api/v1/predict` result matches a subscription's criteria, the system sends a FHIR Bundle (type: `subscription-notification`) to the webhook URL within 5 seconds
- [x] Webhook payload includes: FHIR DiagnosticReport reference, matched finding(s), urgency score, and timestamp
- [x] Retry logic: 3 retries with exponential backoff on webhook delivery failure; dead-letter queue for persistently failed notifications
- [x] `GET /api/v1/subscriptions/:id/notifications` returns delivery history for a subscription (sent, failed, pending)
- [x] Unit tests: subscription CRUD, matching logic, webhook delivery (mock server), retry behavior
- [x] Typecheck passes

---

### US-118: ECG Management System Plugin Architecture
**Description:** As a hospital IT administrator, I want an extensible plugin architecture so that Aortica can embed into existing ECG management platforms (GE MUSE, Philips TraceMasterVue, Mortara/Welch Allyn) as a decision-support module.

**Background and rationale:** PRD-2 Phase 4 mentions "FHIR ECG management system plugin" and "national programme support." Individual integration protocols exist (DICOM DIMSE in US-083, SCP-ECG in US-084, FHIR in US-080), but there's no pluggable architecture that ties them together into a coherent plugin interface that ECG management system vendors or hospital IT teams can adapt.

**Acceptance Criteria:**
- [x] `aortica/integration/plugins/` subpackage with `base.py` defining `ECGSystemPlugin` abstract base class
- [x] Plugin interface methods: `connect(config)`, `poll_for_ecgs()`, `submit_result(ecg_id, result)`, `get_worklist()`, `health_check()`
- [x] Reference plugin implementations:
  - `MusePlugin`: DIMSE C-FIND to poll for new ECGs, C-STORE to write back DICOM SR results (wraps US-083)
  - `FHIRPlugin`: FHIR R4 search to poll for ECGs, DiagnosticReport POST to submit results (wraps US-080)
  - `FileWatcherPlugin`: watches a configurable directory for new ECG files (SCP-ECG, WFDB, CSV), processes them automatically, writes results to output directory
- [x] Plugin registry: `aortica.integration.plugins.register_plugin(name, cls)` and `get_plugin(name)` for discovery
- [x] Plugin configuration via YAML (`plugins.yaml`) with per-plugin connection parameters
- [x] Daemon mode: `aortica plugin run --config plugins.yaml` starts a long-running service that polls the configured ECG management system, processes new ECGs, and submits results back
- [x] Event hooks: `on_ecg_received`, `on_result_generated`, `on_critical_finding` for custom post-processing
- [x] Unit tests for plugin registry, base class contract, and FileWatcherPlugin with synthetic ECG files
- [x] Typecheck passes

---

### US-119: Worklist Dashboard Page
**Description:** As a cardiologist, I want a web dashboard showing the AI-prioritized ECG worklist so that I can review the most urgent ECGs first and work through my queue efficiently.

**Acceptance Criteria:**
- [x] `WorklistDashboard` React page at route `/worklist`
- [x] Table view with columns: urgency score (0–100) with color-coded badge, ECG ID, acquisition timestamp, patient identifier (if available), top finding, recommended action, review status (pending/in-progress/completed)
- [x] Default sort by urgency score (highest first); sortable by all columns
- [x] Filter controls: urgency tier (critical/moderate/routine), finding type, date range, review status
- [x] Critical findings row styling: red left border + animated urgency indicator for scores ≥80
- [x] Click row to open full results page with ECG waveform, predictions, and copilot panel
- [x] Inline actions: "Mark as Reviewed", "Assign to" (dropdown of registered clinicians), "Generate Report"
- [x] `PATCH /api/v1/worklist/:ecg_id` API endpoint for updating review status and assignee
- [x] `GET /api/v1/worklist` API endpoint returning prioritized worklist with status filters
- [x] Real-time update: new ECGs appear in the worklist automatically (polling or WebSocket, configurable interval default 30s)
- [x] Worklist summary bar: total pending, critical count, average time-to-review
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Report Generation

---

### US-087: PDF Clinical Report Generator
**Description:** As a clinician, I want a PDF clinical report with the ECG waveform, AI findings, and XAI annotations so that I have a printable, shareable document for the patient record.

**Acceptance Criteria:**
- [x] `aortica/reports/` subpackage with `__init__.py`
- [x] `aortica.reports.generate_pdf(multi_task_output, ecg_record, xai_report, output_path)` generates a PDF report
- [x] Report sections: patient demographics (if available), ECG waveform plot (12-lead standard layout), signal quality summary, per-task findings with confidence bars, XAI feature attributions (top-3 per finding), risk scores with clinical labels, uncertainty/OOD flags
- [x] Uses WeasyPrint (HTML-to-PDF) with a professional clinical report template
- [x] ECG waveform rendered as SVG via matplotlib with standard grid
- [x] Header includes model version, timestamp, and "AI Decision Support — Requires Clinical Review" watermark
- [x] Unit tests with synthetic data producing valid PDF (non-zero file size, parseable)
- [x] Typecheck passes

---

### US-088: JSON-LD Machine-Readable Report
**Description:** As a data engineer, I want JSON-LD reports so that AI findings are machine-readable and linked to standard medical ontologies for downstream analytics.

**Acceptance Criteria:**
- [x] `aortica.reports.generate_jsonld(multi_task_output, ecg_metadata)` returns a JSON-LD document
- [x] Uses Schema.org `MedicalTest` and `MedicalObservation` types where applicable
- [x] Findings linked to SNOMED CT and LOINC codes via `@context` with standard ontology IRIs
- [x] Includes provenance metadata: model version, inference timestamp, input file hash, confidence intervals
- [x] JSON-LD validates against JSON-LD 1.1 spec (compaction/expansion round-trips)
- [x] Unit tests with synthetic output verifying valid JSON-LD structure and code mappings
- [x] Typecheck passes

---

### US-089: CSV Batch Analytics Export
**Description:** As a researcher, I want CSV export of batch results so that I can perform statistical analysis on AI findings across large cohorts.

**Acceptance Criteria:**
- [x] `aortica.reports.export_csv(results: list[MultiTaskOutput], output_path)` generates a CSV file
- [x] One row per ECG with columns: filename, quality_score, each rhythm class confidence, each structural class confidence, each ischaemia class confidence, each risk score, urgency_score, OOD_flag
- [x] Header row with human-readable column names
- [x] Handles batch sizes up to 10,000 without excessive memory usage (streaming write)
- [x] `POST /api/v1/export/csv` API endpoint for batch export from stored results
- [x] Unit tests with synthetic batch results producing valid CSV with correct column count
- [x] Typecheck passes

---

### US-090: Report API Endpoints
**Description:** As a developer, I want API endpoints for report generation so that reports can be triggered programmatically from the web UI or external systems.

**Acceptance Criteria:**
- [x] `POST /api/v1/report/pdf/{result_id}` generates and returns a PDF report for a stored result
- [x] `POST /api/v1/report/jsonld/{result_id}` returns a JSON-LD report
- [x] `POST /api/v1/report/fhir/{result_id}` returns a FHIR R4 DiagnosticReport bundle
- [x] `POST /api/v1/report/hl7/{result_id}` returns an HL7 v2.x ORU^R01 message
- [x] All endpoints require authentication (existing auth system)
- [x] Returns `404` for unknown result IDs, `422` for invalid parameters
- [x] Unit tests for each endpoint with stored synthetic results
- [x] Typecheck passes

---

### US-120: Report Generation and Download Page
**Description:** As a clinician, I want a web page for generating and downloading clinical reports so that I can produce PDF, FHIR, HL7, and JSON-LD reports from the UI without using the API directly.

**Acceptance Criteria:**
- [x] `ReportPage` React page at route `/reports/:result_id` accessible from results page and worklist
- [x] **Format selector:** radio buttons or tabs for: PDF Clinical Report, FHIR R4 Bundle, HL7 v2.x Message, JSON-LD, CSV (for batch)
- [x] **PDF preview:** inline preview of the PDF report using a PDF viewer component (e.g., `react-pdf`), with download button
- [x] **FHIR/HL7/JSON-LD preview:** syntax-highlighted JSON/text view of the report content with copy-to-clipboard button
- [x] **Download button:** triggers file download with appropriate filename and MIME type for each format
- [x] **Batch report generation:** from the result browser (US-105) or batch dashboard (US-047), select multiple results and generate a combined CSV analytics export or individual PDF reports as a ZIP archive
- [x] **Report history:** list of previously generated reports for a given result, with timestamp and download link
- [x] `GET /api/v1/reports/:result_id` API endpoint listing available/generated reports for a result
- [x] Loading state with progress indicator for PDF generation (which may take 2–5 seconds)
- [x] Error handling: clear message if report generation fails (e.g., missing XAI data, incomplete result)
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Case-Based Retrieval

---

### US-091: Latent Space Index Construction
**Description:** As an ML engineer, I want a latent space index over de-identified PhysioNet ECGs so that similar historical cases can be retrieved for any new prediction.

**Acceptance Criteria:**
- [x] `aortica/retrieval/` subpackage with `__init__.py`
- [x] `aortica.retrieval.build_index(model, dataset, output_path)` encodes all ECGs through the backbone, extracts feature vectors, and builds an approximate nearest neighbor index
- [x] Uses Annoy or FAISS library for ANN index (configurable, default Annoy with 100 trees)
- [x] Stores metadata sidecar (JSON) mapping index IDs to: anonymized record ID, verified diagnoses (from dataset labels), demographic info (age/sex if available)
- [x] Index building targets 50,000 ECGs from PTB-XL + MIMIC-IV-ECG (public PhysioNet sources)
- [x] CLI command `aortica build-index --dataset <path> --model <path> --output <path>`
- [x] Unit tests with synthetic feature vectors verifying index build and query
- [x] Typecheck passes

---

### US-092: Top-K Similar ECG Retrieval
**Description:** As a clinician, I want to see the 3 most similar historical ECGs for any AI finding so that I can compare my patient's tracing against phenotypically similar cases with known outcomes.

**Acceptance Criteria:**
- [x] `aortica.retrieval.retrieve_similar(model, ecg_record, index_path, k=3)` returns top-K similar ECGs
- [x] Returns `SimilarCaseResult` list with: similarity score (cosine distance), record ID, verified diagnoses, patient demographics (anonymized), outcome summary
- [x] Filters results to only return cases with verified diagnoses (no unlabeled matches)
- [x] API endpoint `POST /api/v1/retrieve/similar` accepts ECG reference and returns similar cases
- [x] Query latency < 50ms for K=3 on an index of 50,000 vectors
- [x] Unit tests with synthetic index and query vectors
- [x] Typecheck passes

---

### US-093: Case-Based Retrieval Integration with XAI Report
**Description:** As a clinician, I want similar case references embedded in the XAI explanation cards so that historical comparisons appear alongside feature attributions.

**Acceptance Criteria:**
- [x] `ExplanationCard` React component updated with "Similar Historical Cases" section (replacing Phase 2 placeholder)
- [x] Each similar case displayed with: similarity percentage, diagnosis summary, key demographic info, outcome if available
- [x] Clicking a similar case opens a side-by-side waveform comparison (current ECG vs. historical)
- [x] API response for `include_xai=true` now includes `similar_cases` field
- [x] Graceful fallback if no index is loaded (shows "Case retrieval unavailable" message)
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Regulatory Document Library

---

### US-094: IEC 80601-2-86 Algorithm Testing Documentation Template
**Description:** As a regulatory affairs specialist, I want an IEC 80601-2-86 ATD template so that algorithm testing documentation is structured correctly for ECG AI submissions.

**Acceptance Criteria:**
- [x] `docs/regulatory/` directory with organized template structure
- [x] `IEC_80601_2_86_ATD.md` template covering: algorithm description, intended use, training data description, performance metrics, known limitations, test methodology, device compatibility matrix
- [x] Placeholders marked with `[FILL: description]` for site-specific information
- [x] Auto-population script `aortica.regulatory.populate_atd(benchmark_report, model_version)` fills performance metrics and dataset sections from benchmark output
- [x] CI pipeline step that validates ATD template is complete (no empty `[FILL:]` markers) before v-stable tag
- [x] Unit tests for auto-population script with synthetic benchmark data
- [x] Typecheck passes

---

### US-095: FDA SaMD Pre-Submission and CE-MDR Templates
**Description:** As a regulatory affairs specialist, I want FDA and CE-MDR submission templates so that the regulatory pathway is documented and actionable.

**Acceptance Criteria:**
- [x] `docs/regulatory/FDA_SAMD_PRESUB.md` template covering: device description, intended use statement, technological characteristics, performance benchmark plan, predicate comparison, software development lifecycle summary
- [x] `docs/regulatory/CE_MDR_TECHFILE.md` template covering: essential requirements checklist (Annex I), clinical evaluation plan, risk management (ISO 14971 reference), software lifecycle (IEC 62304), usability engineering (IEC 62366)
- [x] Both templates reference Aortica's benchmark harness and equity gating outputs as evidence sources
- [x] Cross-references to IEC 80601-2-86 ATD template
- [x] Placeholders marked consistently with `[FILL: description]`
- [x] Typecheck passes

---

### US-096: TRIPOD-AI / STARD-AI / CONSORT-AI Reporting Templates
**Description:** As a researcher, I want AI reporting guideline templates so that publications and validation studies follow accepted reporting standards.

**Acceptance Criteria:**
- [x] `docs/regulatory/TRIPOD_AI.md` checklist template for diagnostic/prognostic AI model reporting
- [x] `docs/regulatory/STARD_AI.md` checklist template for diagnostic accuracy study reporting
- [x] `docs/regulatory/CONSORT_AI.md` checklist template for randomized controlled trial reporting with AI interventions
- [x] Each template includes: checklist items with section references, auto-fillable fields from benchmark reports, guidance notes for Aortica-specific context
- [x] `aortica.regulatory.generate_reporting_checklist(template='tripod_ai', benchmark_report=None)` generates a partially pre-filled checklist
- [x] Unit tests for checklist generation with synthetic data
- [x] Typecheck passes

---

### US-097: CI Pipeline Regulatory Performance Enforcement
**Description:** As a release manager, I want CI to enforce minimum performance targets per regulatory device class so that no release falls below the documented thresholds.

**Acceptance Criteria:**
- [x] `aortica.evaluation.regulatory_gate(benchmark_report, targets_yaml)` function
- [x] Reads per-class minimum performance targets from `regulatory_targets.yaml` (AUC, sensitivity, specificity thresholds per condition)
- [x] Returns `RegulatoryGateResult` with pass/fail per class, actual vs. target metrics, and overall pass/fail
- [x] Default targets: STEMI sensitivity ≥ 0.90, AF AUC ≥ 0.95, LVSD AUC ≥ 0.88, overall rhythm F1 ≥ 0.90
- [x] GitHub Actions step that runs regulatory gate and blocks release on failure
- [x] Unit tests with synthetic benchmarks showing passing and failing scenarios
- [x] Typecheck passes

---

### US-121: Audit Trail and Compliance Logging
**Description:** As a regulatory affairs specialist, I want a centralized, immutable audit trail so that every clinical decision-support interaction is logged for regulatory compliance, post-market surveillance, and forensic investigation.

**Background and rationale:** Individual audit logging exists in the Android app (US-061c) and clinician feedback (US-053), but there's no centralized, tamper-evident audit trail covering the full lifecycle: ECG ingestion → AI prediction → clinician review → report generation → EHR submission. Regulatory frameworks (IEC 62304, FDA SaMD guidance) require traceability of AI-assisted clinical decisions.

**Acceptance Criteria:**
- [x] `aortica/audit/` subpackage with `__init__.py`
- [x] `aortica.audit.AuditLogger` class recording immutable audit events to an append-only SQLite table with HMAC integrity verification per row
- [x] Audit event types: `ecg_ingested`, `prediction_generated`, `xai_computed`, `finding_accepted`, `finding_rejected`, `finding_modified`, `report_generated`, `report_exported`, `ehr_submitted`, `adverse_event_reported`, `model_loaded`, `model_updated`
- [x] Each event includes: timestamp (UTC), event_type, user_id (if authenticated), ecg_reference_id, model_version, session_id, IP address, event_details (JSON), HMAC signature
- [x] HMAC chain: each row's HMAC includes the previous row's hash, creating a tamper-evident chain (any modification to a past row invalidates all subsequent HMACs)
- [x] `aortica.audit.verify_integrity(audit_db_path)` validates the HMAC chain and reports any broken links
- [x] FastAPI middleware that automatically logs `prediction_generated`, `report_generated`, and `ehr_submitted` events without requiring manual instrumentation in each endpoint
- [x] `GET /api/v1/audit/events` API endpoint with filters: date range, event type, user, ECG reference (admin-only)
- [x] `GET /api/v1/audit/verify` API endpoint that runs integrity verification and returns pass/fail with details
- [x] CLI command `aortica audit export --from <date> --to <date> --format csv|json` exports audit log for regulatory submission
- [x] Audit log rotation: configurable max database size with archival to compressed files
- [x] Unit tests: event logging, HMAC chain verification (valid and tampered), integrity check, export
- [x] Typecheck passes

---

#### Prospective Validation Tooling

---

### US-098: Multi-Site Prospective Study Protocol Template
**Description:** As a clinical researcher, I want a study protocol template so that prospective validation studies at multiple sites follow a standardized methodology.

**Acceptance Criteria:**
- [x] `docs/validation/PROSPECTIVE_PROTOCOL.md` template covering: study objectives, primary/secondary endpoints, sample size calculation, inclusion/exclusion criteria, site requirements, data collection procedures, statistical analysis plan, ethical considerations, timeline
- [x] Pre-filled with Aortica-specific endpoints: STEMI sensitivity, AF detection AUC, LVSD PPV/NPV
- [x] Configurable per study (number of sites, target N, study duration)
- [x] IRB-ready language and consent form template
- [x] Typecheck passes

---

### US-099: Prospective Data Collection Pipeline
**Description:** As a site coordinator, I want a data collection pipeline so that prospective ECGs with outcome linkage are captured systematically for validation studies.

**Acceptance Criteria:**
- [x] `aortica/validation/` subpackage with `__init__.py`
- [x] `aortica.validation.ProspectiveCollector` class managing: ECG ingestion with timestamp and site ID, AI prediction storage, ground-truth outcome entry (clinician-verified diagnosis at follow-up), outcome linkage (prediction ↔ ground truth pairing)
- [x] SQLite backend with encrypted storage (reuses `ResultStore` encryption from US-054)
- [x] `POST /api/v1/validation/submit` endpoint for sites to submit ECG + outcome pairs
- [x] Data export: `aortica.validation.export_study_data(collector, output_path)` generates de-identified CSV for statistical analysis
- [x] Unit tests for data ingestion, outcome linkage, and export
- [x] Typecheck passes

---

### US-100: Automated Performance Monitoring
**Description:** As a deployment admin, I want automated performance monitoring so that model accuracy is tracked against labeled subsets in production.

**Acceptance Criteria:**
- [x] `aortica.validation.PerformanceMonitor` class tracking: rolling AUC, F1, and calibration metrics against labeled production data
- [x] Configurable monitoring window (default: 30 days rolling)
- [x] Drift detection: flags when any per-task metric drops below configurable threshold or deviates >5% from baseline
- [x] Generates alert (log + optional webhook) when drift detected
- [x] `GET /api/v1/validation/monitor/status` endpoint returning current monitoring metrics and drift flags
- [x] Unit tests with synthetic labeled production data showing stable and drifting scenarios
- [x] Typecheck passes

---

### US-101: Quarterly Public Performance Report Generator
**Description:** As a release manager, I want automated quarterly performance reports so that Aortica's production accuracy is publicly transparent.

**Acceptance Criteria:**
- [x] `aortica.validation.generate_quarterly_report(monitor, output_dir, quarter, year)` function
- [x] Report includes: period summary, per-task metrics (AUC, F1, sensitivity, specificity), demographic subgroup breakdowns, drift alerts (if any), comparison to previous quarter, total ECGs processed
- [x] Outputs `QUARTERLY_REPORT_YYYY_QN.md` (markdown) and `quarterly_report_YYYY_QN.csv` (tabular)
- [x] CLI command `aortica validation quarterly-report --quarter <Q> --year <Y>`
- [x] Unit tests with synthetic monitor data producing valid report
- [x] Typecheck passes

---

### US-102: Voluntary Adverse Event Reporting Form
**Description:** As a clinician, I want to report adverse events related to AI findings so that safety signals are captured for post-market surveillance.

**Acceptance Criteria:**
- [x] `POST /api/v1/validation/adverse-event` endpoint accepting: reporter ID, ECG reference, event description, severity (minor/moderate/serious/critical), AI finding that contributed, patient outcome
- [x] `aortica/validation/adverse_events.py` with Pydantic models and SQLite-backed storage
- [x] `GET /api/v1/validation/adverse-events` returns list of reported events (admin-only, requires authentication)
- [x] `GET /api/v1/validation/adverse-events/summary` returns aggregate statistics (count by severity, most-reported findings)
- [x] React form component in web UI for submitting adverse event reports
- [x] Events stored with timestamp, immutable audit trail (append-only)
- [x] Unit tests for event CRUD and summary aggregation
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-122: Prospective Study Data Collection UI
**Description:** As a site coordinator running a prospective validation study, I want a web interface for submitting ECGs with ground-truth outcomes so that data collection is systematic and doesn't require API scripting.

**Acceptance Criteria:**
- [x] `ProspectiveDataPage` React page at route `/validation/prospective` (protected: requires authenticated site coordinator role)
- [x] **ECG submission form:** upload ECG file, auto-runs prediction pipeline, displays results for clinician review
- [x] **Ground-truth entry form:** structured input for clinician-verified diagnosis at follow-up — checkbox list of conditions (matching Aortica's class taxonomy), free-text notes, follow-up date, outcome category (confirmed/ruled-out/indeterminate)
- [x] **Outcome linkage:** pairs the AI prediction with the ground-truth entry for the same ECG, auto-computing per-class concordance (TP/FP/TN/FN)
- [x] **Collection progress dashboard:** total ECGs submitted per site, target vs. actual enrollment, per-class label distribution, concordance summary
- [x] **Data quality indicators:** flags incomplete submissions (missing ground-truth, missing demographics), highlights outlier predictions for review
- [x] Export button: generates de-identified CSV (reuses US-099 `export_study_data`) downloadable from the UI
- [x] `GET /api/v1/validation/prospective/progress` API endpoint returning collection progress stats
- [x] Multi-site support: each authenticated user is associated with a site ID; data is tagged and filterable by site
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-123: Performance Monitoring Dashboard
**Description:** As a deployment administrator, I want a web dashboard showing live performance monitoring metrics so that I can detect model drift, track accuracy trends, and respond to degradation without running CLI commands.

**Acceptance Criteria:**
- [x] `PerformanceMonitorPage` React page at route `/validation/monitor` (protected: requires admin role)
- [x] **Metrics overview panel:** current rolling AUC, F1, and calibration (ECE) per task head, with trend arrows (↑↓↔) vs. previous period
- [x] **Time-series charts:** line charts of per-task AUC and F1 over the monitoring window (default 30 days), with configurable window selector
- [x] **Drift alert panel:** list of active drift alerts with: alert timestamp, affected task/class, metric value, threshold, deviation percentage, severity (warning/critical)
- [x] **Demographic breakdown:** per-subgroup (age decile, sex) metric cards showing current performance vs. baseline, with equity gate status
- [x] **Volume metrics:** total ECGs processed (daily/weekly/monthly), inference latency trends, error rate
- [x] **Baseline comparison:** overlay baseline metrics (from original benchmark) on the time-series chart for visual drift assessment
- [x] `GET /api/v1/validation/monitor/metrics` API endpoint returning current monitoring metrics with time-series data
- [x] `GET /api/v1/validation/monitor/alerts` API endpoint returning active drift alerts
- [x] Auto-refresh with configurable interval (default 5 minutes)
- [x] Verify changes work in browser
- [x] Typecheck passes

---

### US-124: Adverse Event Reporting Form (Dedicated Frontend)
**Description:** As a clinician, I want a polished, standalone adverse event reporting form so that safety events are captured with minimal friction and maximum completeness.

**Background and rationale:** US-102 defines the backend API and mentions "React form component" as an acceptance criterion. However, adverse event reporting is a safety-critical workflow that deserves a dedicated, carefully designed frontend story ensuring the form guides clinicians through a complete report with required fields, severity classification, and confirmation flow — not just a checkbox item on a backend story.

**Acceptance Criteria:**
- [x] `AdverseEventForm` React page at route `/report-event` accessible from the copilot panel, results page, and sidebar navigation
- [x] **Guided form flow:** multi-step wizard with: (1) Event identification — ECG reference (auto-populated if navigated from results page), event date, reporter information; (2) Event details — description (free-text, required, min 50 characters), severity classification (minor/moderate/serious/critical with tooltip definitions), AI finding that contributed (auto-populated from predictions if available); (3) Patient outcome — outcome description, follow-up status, was clinical harm prevented; (4) Review and submit — summary of all fields with edit links, confirmation checkbox ("I confirm this report is accurate")
- [x] **Severity guidance:** tooltip/popover for each severity level with clinical definitions and examples (e.g., "Serious: resulted in hospitalization or significant intervention")
- [x] **Auto-population:** when accessed from a results page, pre-fills ECG reference, AI findings, and confidence levels
- [x] **Draft saving:** form state persisted to `localStorage` to prevent data loss on accidental navigation; draft indicator with resume option
- [x] **Submission confirmation:** success modal with event reference number and option to download a copy of the report
- [x] **Event history page:** at `/report-event/history`, list of previously submitted reports by the current user with status indicators
- [x] Accessibility: all form fields properly labeled, keyboard-navigable, screen reader tested
- [x] Verify changes work in browser
- [x] Typecheck passes

---

#### Integration & Orchestration Workflows

---

### US-125: End-to-End EHR Integration Workflow
**Description:** As a hospital IT administrator, I want a complete end-to-end EHR integration workflow so that ECGs arriving from the ECG management system are automatically processed by Aortica and results are written back to the EHR and PACS without manual intervention.

**Background and rationale:** Individual integration components exist in isolation: DICOM DIMSE (US-083), FHIR output (US-080), HL7 output (US-081), DICOM SR (US-082), worklist (US-086), and SMART on FHIR (US-085). But no story connects them into a complete automated workflow: ECG arrives → Aortica processes → results written to PACS + EHR + worklist + notifications. This orchestration story provides that glue layer.

**Acceptance Criteria:**
- [ ] `aortica.integration.orchestrator.IntegrationOrchestrator` class managing the full EHR integration loop
- [ ] Workflow steps: (1) ECG ingestion — via DIMSE C-STORE listener, FHIR Subscription, or file watcher (configurable); (2) AI processing — full inference pipeline + XAI; (3) Result storage — persist to local SQLite (US-054); (4) PACS write-back — DICOM SR via C-STORE (US-082/083); (5) EHR submission — FHIR DiagnosticReport or HL7 ORU^R01 (US-080/081); (6) Worklist update — add to prioritized worklist (US-086); (7) Notification — trigger FHIR Subscription notifications for matching criteria (US-117)
- [ ] Configurable via `integration.yaml`: which output channels are enabled, EHR connection parameters, PACS connection parameters, notification rules
- [ ] Error handling: per-step failure isolation (e.g., PACS write-back failure doesn't block EHR submission), retry queue for failed steps, dead-letter log for persistently failed integrations
- [ ] `aortica integration run --config integration.yaml` CLI command starting the orchestrator as a long-running daemon
- [ ] `GET /api/v1/integration/status` API endpoint showing: orchestrator status, queue depth, per-channel success/failure counts, last error per channel
- [ ] Integration health monitoring: alerts if any channel error rate exceeds configurable threshold (default 5% over 1 hour)
- [ ] Unit tests with mock EHR/PACS endpoints verifying full workflow execution, per-step error isolation, and retry logic
- [ ] Typecheck passes

---

### US-126: Worklist to EHR Notification Pipeline
**Description:** As a cardiologist, I want urgent AI findings to automatically trigger notifications in my EHR so that critical ECGs are flagged for immediate review without me checking the Aortica worklist manually.

**Acceptance Criteria:**
- [ ] `aortica.integration.notifications.UrgentFindingNotifier` class monitoring worklist for critical findings and pushing alerts to configured EHR channels
- [ ] Notification channels:
  - FHIR CommunicationRequest resource sent to the FHIR server (for EHRs supporting FHIR R4 communication)
  - HL7 v2.x ADT^A08 (patient update) or ORU^R01 (unsolicited result) with ORC segment flagging urgency
  - Webhook POST to configurable URL (generic, for EHRs with custom alert endpoints)
  - Email notification via SMTP (configurable, for fallback alerting)
- [ ] Notification trigger rules (configurable in `notification_rules.yaml`): condition list, minimum confidence threshold, urgency score threshold, de-duplication window (don't re-notify for same patient + finding within configurable hours)
- [ ] Notification payload: patient identifier (if available), finding name, confidence, urgency score, Aortica result URL, recommended action
- [ ] Delivery tracking: per-notification status (sent/delivered/failed/acknowledged), stored in SQLite
- [ ] `GET /api/v1/notifications` API endpoint returning notification history with delivery status
- [ ] Unit tests with mock EHR endpoints verifying: trigger logic, de-duplication, multi-channel delivery, failure handling
- [ ] Typecheck passes

---

### US-127: Copilot to Report to EHR Clinical Workflow
**Description:** As a cardiologist, I want a single-click workflow that takes my reviewed AI findings, generates a clinical report, and submits it to the EHR so that the entire AI-assisted interpretation cycle is completed without leaving the Aortica interface.

**Acceptance Criteria:**
- [ ] **"Finalize & Submit" button** on the copilot panel (US-049) visible after the clinician has accepted/rejected/modified all AI findings
- [ ] Workflow on click: (1) Collect clinician-reviewed findings (accepted findings + modifications from US-053 feedback); (2) Generate PDF clinical report incorporating clinician decisions (US-087); (3) Generate FHIR DiagnosticReport with clinician attestation (US-080); (4) Submit to configured EHR channel (orchestrator from US-125); (5) Update worklist status to "completed" (US-119)
- [ ] **Attestation step:** before submission, display summary modal with: report preview, findings included, excluded findings, clinician name + timestamp; require explicit "Attest and Submit" confirmation
- [ ] **Configurable output channels:** checkboxes for which outputs to generate (PDF, FHIR, HL7, DICOM SR) — pre-configured per deployment
- [ ] **Post-submission state:** results page shows "Submitted to EHR" badge with timestamp, EHR reference ID, and link to download the finalized report
- [ ] `POST /api/v1/workflow/finalize` API endpoint accepting: result_id, reviewed_findings, attestation, output_channels; orchestrates the full workflow
- [ ] Audit trail: `finalize_and_submit` event logged with clinician ID, all reviewed findings, generated report references, EHR submission status (US-121)
- [ ] Verify changes work in browser
- [ ] Typecheck passes

---

### US-128: Edge Sync to Central Analytics Pipeline
**Description:** As a deployment coordinator managing multiple edge sites, I want synced edge results to flow automatically into the central performance monitor and analytics pipeline so that I have a unified view of AI performance across all deployment sites.

**Background and rationale:** Edge sites store results locally (US-054) and sync them to a central server (US-055/US-056). The performance monitor (US-100) tracks accuracy against labeled data. But there's no story connecting synced edge data into the central monitoring pipeline. Without this, edge deployments are invisible to the central performance tracking system.

**Acceptance Criteria:**
- [ ] `aortica.sync.CentralAggregator` class that receives synced results from edge devices and ingests them into the central analytics pipeline
- [ ] Sync receiver: `POST /api/v1/sync/receive` endpoint (authenticated per-device) accepting batch result uploads from edge sync engines (US-055)
- [ ] Per-site tagging: each synced result tagged with source device_id and site_id for site-level analytics
- [ ] Performance monitor integration: synced results automatically fed into `PerformanceMonitor` (US-100) when ground-truth labels are available (from prospective collection US-099)
- [ ] Site-level analytics: `GET /api/v1/analytics/sites` API endpoint returning per-site metrics: total ECGs processed, finding distribution, quality score distribution, sync status, last sync timestamp
- [ ] Cross-site dashboard: `SiteAnalyticsPage` React page at route `/analytics/sites` showing all edge sites with per-site metrics, geographic distribution (if location configured), and aggregate performance
- [ ] Anomaly detection: flag sites with significantly different finding distributions or quality scores vs. fleet average (z-score > 2.0)
- [ ] Data reconciliation: detect and report sync gaps (expected vs. received result count per device based on inference timestamps)
- [ ] Unit tests with synthetic multi-site sync data verifying aggregation, site-level metrics, and anomaly detection
- [ ] Verify changes work in browser
- [ ] Typecheck passes

---

#### Technical Debt & Correctness Hardening

---

### US-129: Single Source of Truth for Task-Head Output Dimensions
**Description:** As an ML engineer, I want the per-task output dimensions (rhythm/structural/ischaemia/risk) defined in exactly one place so that expanding a task head can never again leave downstream modules silently mis-sized.

**Background and rationale:** The task-head expansion (US-072–US-079) raised the head sizes to rhythm=28, structural=19, ischaemia=19, risk=6 (72 total). However, the per-task dimension map (`TASK_NUM_OUTPUTS` / `_TASK_NUM_OUTPUTS`) is **duplicated across at least seven modules** — `models/train_multitask.py`, `models/conformal_prediction.py`, `models/temperature_scaling.py`, `evaluation/benchmark.py`, `edge/distillation.py`, `edge/validation.py`, and `federated/fl_client.py`. Several copies were not updated during the expansion and remained at the old 22/15/10/3 values. This caused real, silent correctness bugs: the federated client trained against mis-sized labels, and the conformal predictor, temperature-scaling calibrator, INT8 distillation, and edge-validation harness all split concatenated labels at the wrong offsets when run against the real (expanded) model. The acceptance criterion "Update all downstream code referencing RHYTHM_CLASSES count" was checked off without these copies being caught, because each module's unit tests built synthetic data sized to its own stale copy and therefore stayed self-consistently green.

**Acceptance Criteria:**
- [x] A single canonical definition of per-task output dimensions, derived from the head class constants (`len(RHYTHM_CLASSES)`, `len(STRUCTURAL_CLASSES)`, `len(ISCHAEMIA_CLASSES)`, `len(RISK_OUTPUTS)`), exposed as e.g. `aortica.models.task_dims.TASK_NUM_OUTPUTS`
- [x] All seven (or more) existing copies replaced by an import of the canonical map; no module hardcodes the integers
- [x] The canonical map is importable without forcing a heavyweight (torch/tf) import at module load, preserving the existing lazy-import behaviour of `federated/fl_client.py`
- [x] A regression test asserts the canonical map equals the actual head class-list lengths, so any future head expansion that forgets a downstream update fails CI immediately
- [x] A test that exercises conformal prediction, temperature scaling, distillation label-splitting, and the FL client against a real `AorticaModel` (not synthetic fixed-width data) so dimension drift is caught end-to-end
- [x] Typecheck passes

---

### US-130: Demographic-Stratified Production Monitoring
**Description:** As a release manager, I want the production performance monitor and quarterly report to actually break metrics down by demographic subgroup so that the equity claims in US-100/US-101 are backed by real data rather than absent.

**Background and rationale:** US-100 and US-101 are marked complete, and US-101's acceptance criteria promise "demographic subgroup breakdowns" in the quarterly report. In practice `PerformanceMonitor` records only `(ecg_id, task, class_name, prediction, ground_truth, timestamp)` — it stores **no demographic attributes**, so neither the monitor nor `generate_quarterly_report` can produce any subgroup stratification. The report's "demographic subgroup breakdowns" criterion is therefore unsatisfiable as built. This story closes the gap between the stated criterion and the implementation.

**Acceptance Criteria:**
- [ ] `PerformanceMonitor.record_prediction(...)` accepts optional demographic attributes (at minimum `age` and `sex`), persisted alongside each prediction
- [ ] `get_status()` / a new `get_subgroup_status()` computes rolling AUC/F1/ECE stratified by sex and age decile for subgroups with sufficient sample size (configurable minimum, default N≥30)
- [ ] `generate_quarterly_report` renders a real demographic-subgroup section (markdown table + CSV rows) instead of omitting it; when demographics are absent it states so explicitly rather than implying a breakdown exists
- [ ] Drift detection optionally runs per subgroup so an equity regression in one group is flagged even when the aggregate metric looks stable
- [ ] Unit tests with synthetic demographic-tagged production data verifying subgroup stratification, the minimum-sample guard, and the "no demographics available" path
- [ ] Typecheck passes

---

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

---

## Non-Goals

- **No native iOS app** — iOS support deferred to future work; Android app covers the primary LMIC mobile deployment scenario
- **No FDA 510(k) or CE-MDR submission** — Phase 4 achieves *regulatory readiness* (templates, validation tooling, pre-submission meeting) but does not complete clearance
- **No commercial cloud hosting** — Aortica Cloud is a future sustainability initiative
- **No automated treatment recommendations** — all outputs are decision support only, requiring clinician review
- **No proprietary data partnerships** — all training uses public or federated data
- **No real-time streaming ECG analysis** — batch and single-ECG inference only; continuous telemetry monitoring deferred

## Technical Considerations

- **PTB-XL** is the primary dataset for training and evaluation; users must download it separately (we provide a download helper script)
- **PyTorch** is the primary training framework; TensorFlow/Keras is maintained in parallel for portability
- Signal processing uses **NeuroKit2** and **scipy.signal** as dependencies
- WFDB reading uses the **wfdb** Python package; DICOM uses **pydicom**; MAT uses **scipy.io**
- **FastAPI** serves the REST API; **gRPC** (grpcio) for high-throughput integrations
- **React + Vite + TypeScript** for web frontend in `frontend/` directory
- **ONNX Runtime** for edge model inference; ONNX Runtime quantization tools for INT8; **ONNX Runtime Mobile** (`onnxruntime-android`) for native Android inference
- **Click + Rich** for CLI tooling
- **MkDocs Material** for documentation site
- Docker images target **amd64** (server/GPU) and **arm64** (edge/RPi)
- **cryptography** (Fernet) for AES-256 encryption of local result storage
- **Flower (flwr)** for federated learning; **OpenDP** for differential privacy accounting
- **TenSEAL** (CKKS scheme) for homomorphic encryption in secure aggregation
- Expanded task head dimensions: rhythm=28, structural=19, ischaemia=19, risk=6 (72 total outputs)
- All models should be designed for ONNX export (avoid non-exportable operations where possible)
- **PyYAML** is a core (non-optional) runtime dependency: training, sync, federated, integration, and the regulatory gate all load YAML config, and `aortica.evaluation` imports the regulatory gate at package import time. It must be declared in the base `dependencies`, not behind an extra
- **fhir.resources** for FHIR R4 resource validation and generation
- **hl7apy** for HL7 v2.x message construction and validation
- **pynetdicom** for DICOM DIMSE network operations (C-STORE, C-FIND)
- **pyserial** for SCP-ECG serial port capture from legacy ECG carts
- **WeasyPrint** for HTML-to-PDF clinical report generation
- **pyld** for JSON-LD report generation and validation
- **Annoy** or **FAISS** for approximate nearest neighbor retrieval in case-based ECG comparison
- Regulatory templates stored in `docs/regulatory/`; validation templates in `docs/validation/`
- Prospective validation data stored in encrypted SQLite (reuses sync module encryption)
- Minimum Python version: 3.10
- Target test coverage: ≥ 80% for `io/`, `signal/`, `evaluation/`, `api/`, `cli/`, `edge/`, `sync/`, `federated/`, `integration/`, `reports/`, `retrieval/`, and `validation/` modules
