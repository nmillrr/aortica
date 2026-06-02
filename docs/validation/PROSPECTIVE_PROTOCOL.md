# Multi-Site Prospective Validation Study Protocol — Aortica AI ECG Analysis

## Document Control

| Field | Value |
|-------|-------|
| **Protocol Version** | 1.0 |
| **Model Version** | [FILL: aortica model version, e.g. 0.3.0] |
| **Date** | [FILL: protocol date] |
| **Principal Investigator** | [FILL: PI name and affiliation] |
| **Study Title** | [FILL: full study title] |
| **Study Registration** | [FILL: ClinicalTrials.gov NCT number or equivalent] |
| **Ethics Approval** | [FILL: IRB/ethics committee and approval number] |
| **Sponsor** | [FILL: study sponsor or "Investigator-initiated"] |

> **Scope**: This template provides a standardised protocol for multi-site prospective validation of the Aortica AI ECG analysis platform. It is designed to be IRB-ready and configurable per study. All `[FILL:]` markers require study-specific information.

---

## 1. Study Objectives

### 1.1 Primary Objective

To evaluate the prospective diagnostic accuracy of the Aortica multi-task deep learning model for 12-lead ECG interpretation across multiple clinical sites.

### 1.2 Primary Endpoints

| Endpoint | Metric | Target |
|----------|--------|--------|
| STEMI detection sensitivity | Sensitivity (per-territory) | ≥ 0.90 |
| Atrial fibrillation detection | AUC (one-vs-rest) | ≥ 0.95 |
| Left ventricular systolic dysfunction screening | PPV / NPV | PPV ≥ 0.70, NPV ≥ 0.95 |

### 1.3 Secondary Objectives

- Evaluate per-task macro-F1 (rhythm ≥ 0.90, structural ≥ 0.85, ischaemia ≥ 0.85) in a prospective cohort
- Assess calibration quality (ECE < 0.05 per task head)
- Measure edge model concordance with full model (AUC within 3%)
- Evaluate demographic equity: no statistically significant performance gap by sex or age decile (Bonferroni-corrected p<0.05)
- Assess clinician-AI concordance: inter-rater agreement (Cohen's κ) between Aortica and cardiologist interpretation
- Measure time-to-interpretation impact when AI decision support is available vs. standard care

### 1.4 Exploratory Endpoints

- Risk prediction calibration: 1-year mortality C-index, HF hospitalisation C-index
- Edge-case detection sensitivity for rare conditions (Brugada, Wellens, de Winter, CPVT)
- Clinician acceptance rate and override rate for AI findings
- Signal quality distribution across sites

---

## 2. Study Design

### 2.1 Study Type

[FILL: e.g. "Prospective, multi-site, observational diagnostic accuracy study"]

### 2.2 Study Population

**Setting:** [FILL: number of sites] clinical sites across [FILL: geographic regions]

**Inclusion criteria:**
- Adults (≥ 18 years) presenting with a clinical indication for 12-lead ECG
- ECG recording of diagnostic quality (Aortica quality score ≥ 40, i.e. not "poor")
- Written informed consent obtained (or waiver of consent approved by ethics committee for minimal-risk, de-identified analysis)

**Exclusion criteria:**
- ECG recordings with < 10 seconds of interpretable signal
- Paced rhythms (unless specifically included in a pacemaker sub-study)
- Patients who withdraw consent
- [FILL: any additional study-specific exclusion criteria]

### 2.3 Study Duration

| Phase | Duration |
|-------|----------|
| Site setup and training | [FILL: e.g. 4 weeks] |
| Enrolment period | [FILL: e.g. 12 months] |
| Follow-up (for risk endpoints) | [FILL: e.g. 12 months post-ECG] |
| Data cleaning and analysis | [FILL: e.g. 8 weeks] |
| **Total** | [FILL: e.g. 28 months] |

---

## 3. Sample Size Calculation

### 3.1 Primary Power Calculation

Target: Estimate STEMI detection sensitivity with a 95% confidence interval half-width of ≤ 5%.

Assumptions:
- Expected STEMI sensitivity: 0.92 (from PTB-XL internal validation)
- Expected STEMI prevalence in study population: [FILL: e.g. 3%]
- Alpha: 0.05 (two-sided)

Using the binomial exact method:
- Required STEMI-positive ECGs: [FILL: calculated N, e.g. 200]
- Required total ECGs (given prevalence): [FILL: calculated N, e.g. 6,667]
- Per-site minimum: [FILL: e.g. 500 ECGs]
- Accounting for 10% data attrition: **Target N = [FILL: e.g. 7,400 total, ~740 per site across 10 sites]**

### 3.2 Per-Site Minimum

Each site must contribute a minimum of [FILL: e.g. 500] ECGs to ensure adequate statistical power for site-level analysis and to permit non-Western site validation tracking (see `aortica.evaluation.SiteValidationRegistry`).

### 3.3 Subgroup Power

Demographic subgroup analyses require a minimum of 100 samples per group per class (per the equity gate specification in US-069).

---

## 4. Site Requirements

### 4.1 Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| 12-lead ECG device | Any device producing standard digital format (WFDB, DICOM, SCP-ECG, HL7 aECG, PDF scan) |
| Aortica deployment | Docker (recommended) or bare-metal Python installation |
| Hardware | Minimum: 4 GB RAM, 2-core CPU. Recommended: 8 GB RAM, 4-core, NVIDIA GPU |
| Network | Intermittent acceptable (offline-first architecture); required for initial setup and periodic sync |
| Storage | ≥ 50 GB for local result storage |

### 4.2 Personnel Requirements

- Study coordinator (ECG upload, data quality monitoring)
- Reference standard reader(s): board-certified cardiologist(s) with ≥ 3 years ECG interpretation experience
- IT support for Aortica installation and maintenance

### 4.3 Site Selection Criteria

- Minimum [FILL: e.g. 2] sites in non-Western regions (per SiteValidationRegistry release readiness requirement)
- Sites should represent a range of: geographic regions, patient demographics, ECG device manufacturers, clinical settings (tertiary hospital, community clinic, rural clinic)

---

## 5. Data Collection Procedures

### 5.1 ECG Acquisition and Upload

1. Standard 12-lead ECG recorded per clinical indication (no study-specific ECGs)
2. ECG exported from device in digital format (WFDB, DICOM, SCP-ECG, or PDF scan)
3. Uploaded to local Aortica instance via web UI, CLI (`aortica predict`), or API (`POST /api/v1/predict`)
4. Aortica automatically runs the full pipeline: denoising → quality scoring → multi-task inference → XAI attribution
5. Result stored locally in encrypted SQLite (AES-256, see US-054)

### 5.2 Reference Standard Collection

The reference standard is expert cardiologist interpretation performed **independently** of Aortica output:

1. Each enrolled ECG reviewed by [FILL: e.g. 2] board-certified cardiologist(s)
2. Readers blinded to Aortica AI output during reference standard generation
3. Structured interpretation entered via `POST /api/v1/validation/submit` (see US-099)
4. Discrepancies between readers resolved by [FILL: e.g. "consensus with a third independent reader" or "majority vote"]
5. Follow-up clinical data (echo, angiography, outcomes) collected at [FILL: e.g. 30 days and 12 months]

### 5.3 Data Fields Collected

| Category | Fields |
|----------|--------|
| ECG metadata | Date, time, site ID, device manufacturer, recording duration, sample rate |
| Patient demographics | Age, sex, ethnicity (self-reported, optional), height, weight |
| Clinical context | Indication for ECG, presenting symptoms, relevant medical history |
| AI prediction | Full multi-task output, quality report, XAI attributions, uncertainty report |
| Reference standard | Cardiologist interpretation (structured + free-text), reader ID, blinding confirmation |
| Follow-up outcomes | Confirmed diagnosis, 30-day MACE, 12-month mortality, echocardiography results (EF) |

### 5.4 Data Quality Monitoring

- Aortica signal quality scoring automatically flags `poor` quality ECGs (score < 40) for exclusion
- Weekly data quality reports generated per site (completeness, quality distribution, enrolment rate)
- Quarterly interim analysis to detect early safety signals

---

## 6. Statistical Analysis Plan

### 6.1 Primary Analysis

| Endpoint | Method |
|----------|--------|
| STEMI sensitivity | Exact binomial 95% CI |
| AF AUC | DeLong method with 95% CI |
| LVSD PPV/NPV | Exact binomial 95% CI |

Analysis performed using `aortica.evaluation.benchmark()` against the reference standard labels.

### 6.2 Secondary Analyses

- **Macro-F1 per task**: computed by `aortica.evaluation.benchmark()` with 95% bootstrap CI (1000 iterations)
- **Calibration**: Expected Calibration Error (ECE) with 10 bins, reliability diagrams
- **Edge model concordance**: paired comparison of full vs. edge model AUC per task
- **Equity gate**: `aortica.evaluation.equity_gate()` with Bonferroni correction across sex and age decile subgroups

### 6.3 Subgroup Analyses

Pre-specified subgroup analyses stratified by:
- Sex (male, female)
- Age decile (18–29, 30–39, ..., 80–89, 90+)
- Site (per-site performance)
- Device manufacturer
- Clinical setting (tertiary vs. community vs. rural)
- Signal quality tier (good ≥ 70, marginal 40–69)

### 6.4 Missing Data

- Missing demographic fields: excluded from the relevant subgroup analysis only
- Missing reference standard: ECG excluded from primary analysis (per-protocol population)
- Missing follow-up: analysed using available data; sensitivity analysis comparing complete vs. incomplete cases

### 6.5 Interim Analysis

Planned interim analysis at 50% enrolment:
- Futility assessment: if STEMI sensitivity point estimate < 0.80, consider stopping for futility
- Safety assessment: if any condition has significantly higher false-negative rate than expected (p < 0.01), convene DSMB

---

## 7. Ethical Considerations

### 7.1 Informed Consent

[FILL: select one or describe consent approach]

**Option A — Full consent:** Written informed consent obtained from all participants prior to ECG analysis. A consent form template is provided in Appendix A.

**Option B — Waiver of consent:** IRB/ethics committee waiver of consent for minimal-risk, retrospective analysis of clinically-indicated ECGs. Justified by: (1) minimal risk to participants (non-invasive software analysis of existing recordings), (2) research cannot practicably be carried out without the waiver, (3) rights and welfare of subjects adequately protected.

### 7.2 Data Protection

- All ECG data processed locally — **no patient data leaves the deployment site**
- Local storage encrypted at rest (AES-256 via `aortica.sync.ResultStore`)
- Sync to central analysis server (if applicable) uses:
  - HTTPS encryption in transit
  - Automated de-identification prior to sync (`aortica.sync.config.anonymize()`)
- Compliant with [FILL: e.g. "GDPR Article 9(2)(j)", "HIPAA Safe Harbor", "local data protection regulations"]

### 7.3 Data Retention

- Site-local data retained for [FILL: e.g. 5 years] after study completion
- Central de-identified analysis dataset retained for [FILL: e.g. 10 years]
- Adverse event reports retained indefinitely

### 7.4 Adverse Event Reporting

- Clinically significant AI errors (false negatives for critical conditions: STEMI, VT, VF, severe hyperkalaemia) reported via `POST /api/v1/validation/adverse-event` (see US-102)
- Serious adverse events reported to IRB within [FILL: e.g. 48 hours]
- All AI-related adverse events included in the quarterly performance report

---

## 8. Study Organisation

### 8.1 Steering Committee

[FILL: composition, meeting frequency, responsibilities]

### 8.2 Data Safety Monitoring Board (DSMB)

[FILL: DSMB composition — recommend independent statistician, cardiologist, and clinical trialist]

### 8.3 Site Coordination

- Central coordination: [FILL: coordinating centre]
- Site principal investigators: [FILL: listed per site]
- Regular site calls: [FILL: e.g. monthly]

---

## 9. Timeline

| Milestone | Target Date |
|-----------|-------------|
| Protocol finalisation | [FILL] |
| Ethics approval (all sites) | [FILL] |
| Aortica deployment at all sites | [FILL] |
| Enrolment start | [FILL] |
| 50% enrolment (interim analysis) | [FILL] |
| Enrolment completion | [FILL] |
| 12-month follow-up completion | [FILL] |
| Data lock | [FILL] |
| Primary analysis | [FILL] |
| Manuscript submission | [FILL] |

---

## 10. Publications and Reporting

- Primary results manuscript submitted to a peer-reviewed journal
- Reported per STARD-AI and TRIPOD-AI checklists (see `docs/regulatory/STARD_AI.md` and `docs/regulatory/TRIPOD_AI.md`)
- All sites acknowledged as contributing authors or collaborators per ICMJE criteria
- Study data archived and benchmark results published as a public performance card via `aortica.evaluation.generate_performance_card()`

---

## Appendix A: Consent Form Template

### INFORMED CONSENT FORM

**Study Title:** [FILL: full study title]

**Principal Investigator:** [FILL: name, institution, contact]

#### Purpose

You are being invited to participate in a research study evaluating an artificial intelligence (AI) computer programme for analysing heart tracings (electrocardiograms or ECGs). The study aims to assess how accurately the AI programme can detect heart conditions from ECG recordings.

#### What Will Happen

- Your ECG, which was recorded as part of your routine clinical care, will be analysed by the AI programme (Aortica)
- The AI analysis is performed **in addition to** standard clinical interpretation by your doctor — it does not replace your doctor's judgement
- Your ECG data will be stored securely on a local computer at this hospital. No data will be sent over the internet unless you give explicit permission
- We will also collect basic information about you (age, sex) and your clinical follow-up to compare the AI's findings with your actual diagnosis

#### Risks

- **Minimal risk**: This study analyses ECGs that have already been recorded for clinical purposes. No additional procedures are performed
- Your clinical care will not be affected by participation in this study — your doctor will make all clinical decisions independently of the AI

#### Benefits

- There is no direct benefit to you from participating
- The study may help improve AI tools for heart diagnosis, which could benefit future patients

#### Confidentiality

- Your data will be stored with a study code number, not your name
- All data is encrypted and stored securely
- Only authorised research staff will have access to your identifiable information
- Results will be published in aggregate — no individual patients will be identifiable

#### Voluntary Participation

- Participation is entirely voluntary
- You may withdraw at any time without giving a reason
- Withdrawal will not affect your clinical care in any way

#### Contact

For questions about this study, contact: [FILL: PI contact information]
For concerns about your rights as a research participant, contact: [FILL: IRB/ethics committee contact]

**Signature:** _______________________________________ **Date:** _______________

**Printed Name:** _____________________________________

**Witness (if required):** _____________________________ **Date:** _______________

---

*This protocol template is provided by the Aortica platform for prospective validation studies. All `[FILL:]` markers require study-specific information. The template follows ICH-GCP E6(R2) principles and is designed to be IRB-ready. Cross-references to Aortica tooling (`aortica.evaluation`, `aortica.validation`, `aortica.sync`) are provided where applicable.*
