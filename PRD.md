# PRD: Aortica — Core AI ECG Engine (Phase 0 + Phase 1)

## Introduction

Aortica is an open-source AI ECG analysis platform designed to close the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

This PRD covers **Phase 0 (Foundation)**, **Phase 1 (Core Engine)**, **Phase 2 (Edge & Rural Deployment)**, **Phase 3 (Federated Learning & Equity)**, and **Phase 4 (Regulatory & Scale)**. Phase 0–1 delivered the core ML pipeline. Phase 2 extends this with REST API, CLI, React web UI with AI copilot, edge-optimized models (ONNX + INT8), offline sync, Docker, docs, and LMIC deployment. Phase 3 adds federated learning via Flower, differential privacy, equity gating CI checks, public performance cards, and expanded task heads for rare arrhythmias, STEMI mimics, strain patterns, and metabolic/drug effects. Phase 4 completes the platform with FHIR R4 / HL7 EHR integration, DICOM SR write-back, worklist prioritization, PDF/JSON-LD report generation, case-based ECG retrieval, regulatory document library (IEC 80601-2-86, FDA SaMD, CE-MDR templates), and prospective validation tooling. Native mobile apps are deferred to future work.

**Distribution model:** Aortica is a self-hosted, open-source toolkit distributed via `aortica.io`. Clinicians and institutions download and run it locally (Docker or pip install) — no data ever leaves the deployment site. This preserves patient privacy, eliminates recurring infrastructure costs, and enables deployment in data-sovereignty-constrained settings. A landing page at `aortica.io` provides download links, documentation, and demo assets.

**Deployment target:** The primary deployment scenario is a rural or resource-limited clinic with a laptop or workstation, intermittent internet, and USB-attached 10/12-lead ECG hardware. The FastAPI backend runs locally (Docker or bare-metal), and the React frontend is served as a **Progressive Web App (PWA)** that caches itself and the ONNX edge model for fully offline use after first load. When the local server is reachable, the full model is used; when offline, inference falls back to ONNX Runtime Web (WebAssembly) running the INT8 edge model directly in the browser. This hybrid architecture requires no internet dependency after initial setup.

**Tech stack:** Python with PyTorch (primary) and TensorFlow/Keras (parallel) for ML. FastAPI for REST API, gRPC for high-throughput service. React + Vite + TypeScript for web UI with PWA service worker. ONNX Runtime (server-side) and ONNX Runtime Web/WASM (in-browser offline inference) for edge deployment. Click + Rich for CLI. Docker for packaging. Flower (flwr) for federated learning. OpenDP for differential privacy. FHIR R4 via `fhir.resources` for EHR integration. HL7 v2.x via `hl7apy`. DICOM SR via `pydicom`. WeasyPrint for PDF report generation. JSON-LD via `pyld`. Annoy/FAISS for latent space nearest-neighbor retrieval. OpenCV + pdfplumber for PDF/image ECG scan digitization.

**Team:** Small team; stories sized at ~30 min of focused implementation each.

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
- [ ] `aortica.integration.worklist.WorklistPrioritizer` class
- [ ] `prioritize(results: list[MultiTaskOutput])` returns sorted list with urgency score (0–100) per ECG
- [ ] Urgency scoring based on configurable rules: critical findings (STEMI, VT, VF, severe hyperkalaemia) = highest priority; moderate findings (new AF, Brugada, Wellens) = medium; routine = low
- [ ] Returns `PrioritizedWorklist` with items sorted by urgency, each showing: ECG ID, urgency score, top finding, recommended action
- [ ] `POST /api/v1/worklist/prioritize` API endpoint accepting batch results and returning prioritized list
- [ ] Configurable urgency rules via YAML file
- [ ] Unit tests with synthetic results spanning all priority tiers
- [ ] Typecheck passes

---

#### Report Generation

---

### US-087: PDF Clinical Report Generator
**Description:** As a clinician, I want a PDF clinical report with the ECG waveform, AI findings, and XAI annotations so that I have a printable, shareable document for the patient record.

**Acceptance Criteria:**
- [ ] `aortica/reports/` subpackage with `__init__.py`
- [ ] `aortica.reports.generate_pdf(multi_task_output, ecg_record, xai_report, output_path)` generates a PDF report
- [ ] Report sections: patient demographics (if available), ECG waveform plot (12-lead standard layout), signal quality summary, per-task findings with confidence bars, XAI feature attributions (top-3 per finding), risk scores with clinical labels, uncertainty/OOD flags
- [ ] Uses WeasyPrint (HTML-to-PDF) with a professional clinical report template
- [ ] ECG waveform rendered as SVG via matplotlib with standard grid
- [ ] Header includes model version, timestamp, and "AI Decision Support — Requires Clinical Review" watermark
- [ ] Unit tests with synthetic data producing valid PDF (non-zero file size, parseable)
- [ ] Typecheck passes

---

### US-088: JSON-LD Machine-Readable Report
**Description:** As a data engineer, I want JSON-LD reports so that AI findings are machine-readable and linked to standard medical ontologies for downstream analytics.

**Acceptance Criteria:**
- [ ] `aortica.reports.generate_jsonld(multi_task_output, ecg_metadata)` returns a JSON-LD document
- [ ] Uses Schema.org `MedicalTest` and `MedicalObservation` types where applicable
- [ ] Findings linked to SNOMED CT and LOINC codes via `@context` with standard ontology IRIs
- [ ] Includes provenance metadata: model version, inference timestamp, input file hash, confidence intervals
- [ ] JSON-LD validates against JSON-LD 1.1 spec (compaction/expansion round-trips)
- [ ] Unit tests with synthetic output verifying valid JSON-LD structure and code mappings
- [ ] Typecheck passes

---

### US-089: CSV Batch Analytics Export
**Description:** As a researcher, I want CSV export of batch results so that I can perform statistical analysis on AI findings across large cohorts.

**Acceptance Criteria:**
- [ ] `aortica.reports.export_csv(results: list[MultiTaskOutput], output_path)` generates a CSV file
- [ ] One row per ECG with columns: filename, quality_score, each rhythm class confidence, each structural class confidence, each ischaemia class confidence, each risk score, urgency_score, OOD_flag
- [ ] Header row with human-readable column names
- [ ] Handles batch sizes up to 10,000 without excessive memory usage (streaming write)
- [ ] `POST /api/v1/export/csv` API endpoint for batch export from stored results
- [ ] Unit tests with synthetic batch results producing valid CSV with correct column count
- [ ] Typecheck passes

---

### US-090: Report API Endpoints
**Description:** As a developer, I want API endpoints for report generation so that reports can be triggered programmatically from the web UI or external systems.

**Acceptance Criteria:**
- [ ] `POST /api/v1/report/pdf/{result_id}` generates and returns a PDF report for a stored result
- [ ] `POST /api/v1/report/jsonld/{result_id}` returns a JSON-LD report
- [ ] `POST /api/v1/report/fhir/{result_id}` returns a FHIR R4 DiagnosticReport bundle
- [ ] `POST /api/v1/report/hl7/{result_id}` returns an HL7 v2.x ORU^R01 message
- [ ] All endpoints require authentication (existing auth system)
- [ ] Returns `404` for unknown result IDs, `422` for invalid parameters
- [ ] Unit tests for each endpoint with stored synthetic results
- [ ] Typecheck passes

---

#### Case-Based Retrieval

---

### US-091: Latent Space Index Construction
**Description:** As an ML engineer, I want a latent space index over de-identified PhysioNet ECGs so that similar historical cases can be retrieved for any new prediction.

**Acceptance Criteria:**
- [ ] `aortica/retrieval/` subpackage with `__init__.py`
- [ ] `aortica.retrieval.build_index(model, dataset, output_path)` encodes all ECGs through the backbone, extracts feature vectors, and builds an approximate nearest neighbor index
- [ ] Uses Annoy or FAISS library for ANN index (configurable, default Annoy with 100 trees)
- [ ] Stores metadata sidecar (JSON) mapping index IDs to: anonymized record ID, verified diagnoses (from dataset labels), demographic info (age/sex if available)
- [ ] Index building targets 50,000 ECGs from PTB-XL + MIMIC-IV-ECG (public PhysioNet sources)
- [ ] CLI command `aortica build-index --dataset <path> --model <path> --output <path>`
- [ ] Unit tests with synthetic feature vectors verifying index build and query
- [ ] Typecheck passes

---

### US-092: Top-K Similar ECG Retrieval
**Description:** As a clinician, I want to see the 3 most similar historical ECGs for any AI finding so that I can compare my patient's tracing against phenotypically similar cases with known outcomes.

**Acceptance Criteria:**
- [ ] `aortica.retrieval.retrieve_similar(model, ecg_record, index_path, k=3)` returns top-K similar ECGs
- [ ] Returns `SimilarCaseResult` list with: similarity score (cosine distance), record ID, verified diagnoses, patient demographics (anonymized), outcome summary
- [ ] Filters results to only return cases with verified diagnoses (no unlabeled matches)
- [ ] API endpoint `POST /api/v1/retrieve/similar` accepts ECG reference and returns similar cases
- [ ] Query latency < 50ms for K=3 on an index of 50,000 vectors
- [ ] Unit tests with synthetic index and query vectors
- [ ] Typecheck passes

---

### US-093: Case-Based Retrieval Integration with XAI Report
**Description:** As a clinician, I want similar case references embedded in the XAI explanation cards so that historical comparisons appear alongside feature attributions.

**Acceptance Criteria:**
- [ ] `ExplanationCard` React component updated with "Similar Historical Cases" section (replacing Phase 2 placeholder)
- [ ] Each similar case displayed with: similarity percentage, diagnosis summary, key demographic info, outcome if available
- [ ] Clicking a similar case opens a side-by-side waveform comparison (current ECG vs. historical)
- [ ] API response for `include_xai=true` now includes `similar_cases` field
- [ ] Graceful fallback if no index is loaded (shows "Case retrieval unavailable" message)
- [ ] Verify changes work in browser
- [ ] Typecheck passes

---

#### Regulatory Document Library

---

### US-094: IEC 80601-2-86 Algorithm Testing Documentation Template
**Description:** As a regulatory affairs specialist, I want an IEC 80601-2-86 ATD template so that algorithm testing documentation is structured correctly for ECG AI submissions.

**Acceptance Criteria:**
- [ ] `docs/regulatory/` directory with organized template structure
- [ ] `IEC_80601_2_86_ATD.md` template covering: algorithm description, intended use, training data description, performance metrics, known limitations, test methodology, device compatibility matrix
- [ ] Placeholders marked with `[FILL: description]` for site-specific information
- [ ] Auto-population script `aortica.regulatory.populate_atd(benchmark_report, model_version)` fills performance metrics and dataset sections from benchmark output
- [ ] CI pipeline step that validates ATD template is complete (no empty `[FILL:]` markers) before v-stable tag
- [ ] Unit tests for auto-population script with synthetic benchmark data
- [ ] Typecheck passes

---

### US-095: FDA SaMD Pre-Submission and CE-MDR Templates
**Description:** As a regulatory affairs specialist, I want FDA and CE-MDR submission templates so that the regulatory pathway is documented and actionable.

**Acceptance Criteria:**
- [ ] `docs/regulatory/FDA_SAMD_PRESUB.md` template covering: device description, intended use statement, technological characteristics, performance benchmark plan, predicate comparison, software development lifecycle summary
- [ ] `docs/regulatory/CE_MDR_TECHFILE.md` template covering: essential requirements checklist (Annex I), clinical evaluation plan, risk management (ISO 14971 reference), software lifecycle (IEC 62304), usability engineering (IEC 62366)
- [ ] Both templates reference Aortica's benchmark harness and equity gating outputs as evidence sources
- [ ] Cross-references to IEC 80601-2-86 ATD template
- [ ] Placeholders marked consistently with `[FILL: description]`
- [ ] Typecheck passes

---

### US-096: TRIPOD-AI / STARD-AI / CONSORT-AI Reporting Templates
**Description:** As a researcher, I want AI reporting guideline templates so that publications and validation studies follow accepted reporting standards.

**Acceptance Criteria:**
- [ ] `docs/regulatory/TRIPOD_AI.md` checklist template for diagnostic/prognostic AI model reporting
- [ ] `docs/regulatory/STARD_AI.md` checklist template for diagnostic accuracy study reporting
- [ ] `docs/regulatory/CONSORT_AI.md` checklist template for randomized controlled trial reporting with AI interventions
- [ ] Each template includes: checklist items with section references, auto-fillable fields from benchmark reports, guidance notes for Aortica-specific context
- [ ] `aortica.regulatory.generate_reporting_checklist(template='tripod_ai', benchmark_report=None)` generates a partially pre-filled checklist
- [ ] Unit tests for checklist generation with synthetic data
- [ ] Typecheck passes

---

### US-097: CI Pipeline Regulatory Performance Enforcement
**Description:** As a release manager, I want CI to enforce minimum performance targets per regulatory device class so that no release falls below the documented thresholds.

**Acceptance Criteria:**
- [ ] `aortica.evaluation.regulatory_gate(benchmark_report, targets_yaml)` function
- [ ] Reads per-class minimum performance targets from `regulatory_targets.yaml` (AUC, sensitivity, specificity thresholds per condition)
- [ ] Returns `RegulatoryGateResult` with pass/fail per class, actual vs. target metrics, and overall pass/fail
- [ ] Default targets: STEMI sensitivity ≥ 0.90, AF AUC ≥ 0.95, LVSD AUC ≥ 0.88, overall rhythm F1 ≥ 0.90
- [ ] GitHub Actions step that runs regulatory gate and blocks release on failure
- [ ] Unit tests with synthetic benchmarks showing passing and failing scenarios
- [ ] Typecheck passes

---

#### Prospective Validation Tooling

---

### US-098: Multi-Site Prospective Study Protocol Template
**Description:** As a clinical researcher, I want a study protocol template so that prospective validation studies at multiple sites follow a standardized methodology.

**Acceptance Criteria:**
- [ ] `docs/validation/PROSPECTIVE_PROTOCOL.md` template covering: study objectives, primary/secondary endpoints, sample size calculation, inclusion/exclusion criteria, site requirements, data collection procedures, statistical analysis plan, ethical considerations, timeline
- [ ] Pre-filled with Aortica-specific endpoints: STEMI sensitivity, AF detection AUC, LVSD PPV/NPV
- [ ] Configurable per study (number of sites, target N, study duration)
- [ ] IRB-ready language and consent form template
- [ ] Typecheck passes

---

### US-099: Prospective Data Collection Pipeline
**Description:** As a site coordinator, I want a data collection pipeline so that prospective ECGs with outcome linkage are captured systematically for validation studies.

**Acceptance Criteria:**
- [ ] `aortica/validation/` subpackage with `__init__.py`
- [ ] `aortica.validation.ProspectiveCollector` class managing: ECG ingestion with timestamp and site ID, AI prediction storage, ground-truth outcome entry (clinician-verified diagnosis at follow-up), outcome linkage (prediction ↔ ground truth pairing)
- [ ] SQLite backend with encrypted storage (reuses `ResultStore` encryption from US-054)
- [ ] `POST /api/v1/validation/submit` endpoint for sites to submit ECG + outcome pairs
- [ ] Data export: `aortica.validation.export_study_data(collector, output_path)` generates de-identified CSV for statistical analysis
- [ ] Unit tests for data ingestion, outcome linkage, and export
- [ ] Typecheck passes

---

### US-100: Automated Performance Monitoring
**Description:** As a deployment admin, I want automated performance monitoring so that model accuracy is tracked against labeled subsets in production.

**Acceptance Criteria:**
- [ ] `aortica.validation.PerformanceMonitor` class tracking: rolling AUC, F1, and calibration metrics against labeled production data
- [ ] Configurable monitoring window (default: 30 days rolling)
- [ ] Drift detection: flags when any per-task metric drops below configurable threshold or deviates >5% from baseline
- [ ] Generates alert (log + optional webhook) when drift detected
- [ ] `GET /api/v1/validation/monitor/status` endpoint returning current monitoring metrics and drift flags
- [ ] Unit tests with synthetic labeled production data showing stable and drifting scenarios
- [ ] Typecheck passes

---

### US-101: Quarterly Public Performance Report Generator
**Description:** As a release manager, I want automated quarterly performance reports so that Aortica's production accuracy is publicly transparent.

**Acceptance Criteria:**
- [ ] `aortica.validation.generate_quarterly_report(monitor, output_dir, quarter, year)` function
- [ ] Report includes: period summary, per-task metrics (AUC, F1, sensitivity, specificity), demographic subgroup breakdowns, drift alerts (if any), comparison to previous quarter, total ECGs processed
- [ ] Outputs `QUARTERLY_REPORT_YYYY_QN.md` (markdown) and `quarterly_report_YYYY_QN.csv` (tabular)
- [ ] CLI command `aortica validation quarterly-report --quarter <Q> --year <Y>`
- [ ] Unit tests with synthetic monitor data producing valid report
- [ ] Typecheck passes

---

### US-102: Voluntary Adverse Event Reporting Form
**Description:** As a clinician, I want to report adverse events related to AI findings so that safety signals are captured for post-market surveillance.

**Acceptance Criteria:**
- [ ] `POST /api/v1/validation/adverse-event` endpoint accepting: reporter ID, ECG reference, event description, severity (minor/moderate/serious/critical), AI finding that contributed, patient outcome
- [ ] `aortica/validation/adverse_events.py` with Pydantic models and SQLite-backed storage
- [ ] `GET /api/v1/validation/adverse-events` returns list of reported events (admin-only, requires authentication)
- [ ] `GET /api/v1/validation/adverse-events/summary` returns aggregate statistics (count by severity, most-reported findings)
- [ ] React form component in web UI for submitting adverse event reports
- [ ] Events stored with timestamp, immutable audit trail (append-only)
- [ ] Unit tests for event CRUD and summary aggregation
- [ ] Verify changes work in browser
- [ ] Typecheck passes

---

## Non-Goals (Phase 4)

- **No native Android or iOS apps** — mobile access via responsive web UI and ONNX-based edge inference; native app store distribution deferred to future work
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
- **ONNX Runtime** for edge model inference; ONNX Runtime quantization tools for INT8
- **Click + Rich** for CLI tooling
- **MkDocs Material** for documentation site
- Docker images target **amd64** (server/GPU) and **arm64** (edge/RPi)
- **cryptography** (Fernet) for AES-256 encryption of local result storage
- **Flower (flwr)** for federated learning; **OpenDP** for differential privacy accounting
- **TenSEAL** (CKKS scheme) for homomorphic encryption in secure aggregation
- Expanded task head dimensions: rhythm=28, structural=19, ischaemia=19, risk=6 (72 total outputs)
- All models should be designed for ONNX export (avoid non-exportable operations where possible)
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
