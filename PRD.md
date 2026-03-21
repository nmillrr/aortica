# PRD: Aortica — Core AI ECG Engine (Phase 0 + Phase 1)

## Introduction

Aortica is an open-source AI ECG analysis platform designed to close the most critical gaps in clinical ECG: poor device generalization, narrow single-task models, black-box outputs, inaccessible tooling, and exclusion of rural/low-resource settings.

This PRD covers **Phase 0 (Foundation)** and **Phase 1 (Core Engine)** — the core ML pipeline from raw ECG signal ingestion through multi-task model training, inference, evaluation, and explainability. The scope is the ML pipeline only; API, CLI packaging, web UI, Docker images, documentation site, edge deployment, and federated learning are deferred to [PRD-2.md](PRD-2.md in repo).

**Tech stack:** Python with both PyTorch (primary training) and TensorFlow/Keras (parallel implementation for export/portability). No web UI in this phase.

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
- [ ] `aortica.models.ConformalPredictor` wrapper: generates prediction sets at a user-specified coverage level (default 90%)
- [ ] OOD detection via Mahalanobis distance on backbone features; flags inputs beyond a configurable percentile threshold
- [ ] `UncertaintyReport` object returned alongside predictions with: confidence interval, OOD flag, entropy score
- [ ] Unit tests with in-distribution and synthetic OOD inputs
- [ ] Typecheck passes

---

### US-025: Integrated Gradient XAI with Named ECG Features
**Description:** As a clinician, I want AI explanations mapped to named ECG features (QRS width, ST slope, T-wave morphology) rather than generic heatmaps so that I can reconcile AI findings with my visual interpretation.

**Acceptance Criteria:**
- [ ] `aortica.xai.explain(model, ecg_record, task='rhythm')` returns a `FeatureAttribution` object
- [ ] Computes integrated gradients per lead
- [ ] Maps gradient attributions onto named ECG segments: P wave, PR interval, QRS complex, ST segment, T wave, QT/QTc
- [ ] Segment boundaries determined by a rule-based delineation algorithm (using R-peak + interval heuristics)
- [ ] Returns top-3 contributing features per active diagnosis with delta-contribution scores
- [ ] Unit tests verifying attribution shape and feature mapping on synthetic ECG
- [ ] Typecheck passes

---

### US-026: VAE Latent Factor Model
**Description:** As a researcher, I want a variational autoencoder that encodes median beats into interpretable latent factors so that the model's internal representations can be visualized and understood.

**Acceptance Criteria:**
- [ ] `aortica.xai.MedianBeatVAE`: VAE with 24-dimensional latent space, trained on median beats extracted from 12-lead ECGs
- [ ] Encoder: 1D CNN; Decoder: transposed 1D CNN; Loss: reconstruction + KL divergence
- [ ] Training script for the VAE on PTB-XL median beats
- [ ] Each latent dimension labeled by Pearson correlation with standard ECG measurements (from PTB-XL metadata)
- [ ] Unit tests verifying encode/decode shapes and reconstruction loss convergence
- [ ] Typecheck passes

---

### US-027: VAE Reporter and Synthetic ECG Rendering
**Description:** As a clinician, I want to see how changing a single latent factor affects the ECG waveform so that I can understand what the model has learned.

**Acceptance Criteria:**
- [ ] `aortica.xai.vae_report(model, vae, ecg_record)` returns a `VAEReport` object
- [ ] Reports which latent factors are most activated for the given prediction
- [ ] Generates synthetic ECG waveforms showing the effect of varying each top factor ±2σ
- [ ] Synthetic waveforms returned as numpy arrays (rendering to image is deferred to PRD-2)
- [ ] Unit tests verifying report generation and synthetic waveform shapes
- [ ] Typecheck passes

---

### US-028: Multi-Task Evaluation Harness
**Description:** As a researcher, I want a benchmark harness that evaluates all task heads with proper metrics and demographic subgroup breakdowns so that model performance is transparent and reproducible.

**Acceptance Criteria:**
- [ ] `aortica.evaluation.benchmark(model, dataset, tasks='all')` returns a `BenchmarkReport`
- [ ] Metrics per classification task: macro-F1, per-class AUC, per-class sensitivity/specificity, ECE
- [ ] Metrics per risk task: C-index, Brier score
- [ ] Subgroup stratification by age decile and sex (when demographic metadata is available)
- [ ] Outputs results as structured dict, printable summary table, and CSV export
- [ ] Reproducible: same model + dataset + seed produces identical results
- [ ] Unit tests verifying metric computation on synthetic predictions
- [ ] Typecheck passes

---

### US-029: TensorFlow/Keras Parity Validation
**Description:** As a developer, I want automated tests confirming that the TF/Keras model implementation produces equivalent outputs to the PyTorch implementation so that framework parity is maintained.

**Acceptance Criteria:**
- [ ] Script that loads identical weights into both PyTorch and TF/Keras models
- [ ] Feeds the same input tensor through both and asserts outputs are within floating-point tolerance (atol=1e-5)
- [ ] Included as a CI test (may be slow; tagged as `@pytest.mark.slow`)
- [ ] Documents the weight conversion process between frameworks
- [ ] Typecheck passes

---

## Non-Goals

- **No web UI** — deferred to PRD-2
- **No REST API or gRPC service** — deferred to PRD-2
- **No CLI tool packaging** — deferred to PRD-2 (scripts are acceptable)
- **No Docker images** — deferred to PRD-2
- **No documentation site** — README and docstrings only in this phase
- **No edge/mobile deployment** — no ONNX export, INT8 quantization, or ARM optimization (PRD-2)
- **No federated learning** — deferred to PRD-2
- **No data privacy/anonymization infrastructure** — deferred to PRD-2
- **No FHIR/HL7 output or EHR integration** — deferred to PRD-2
- **No report generation** (PDF, JSON-LD) — deferred to PRD-2
- **No case-based retrieval system** — deferred to PRD-2
- **No regulatory document templates** — deferred to PRD-2
- **No prospective clinical validation** — deferred to PRD-2

## Technical Considerations

- **PTB-XL** is the primary dataset for training and evaluation; users must download it separately (we provide a download helper script)
- **PyTorch** is the primary training framework; TensorFlow/Keras is maintained in parallel for portability
- Signal processing uses **NeuroKit2** and **scipy.signal** as dependencies
- WFDB reading uses the **wfdb** Python package
- DICOM reading uses **pydicom**; MAT reading uses **scipy.io**
- All models should be designed with future ONNX export in mind (avoid non-exportable operations where possible)
- Minimum Python version: 3.10
- Target test coverage: ≥ 80% for `io/`, `signal/`, and `evaluation/` modules
