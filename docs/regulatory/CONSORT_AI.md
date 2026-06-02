# CONSORT-AI Reporting Checklist — Aortica AI ECG Randomised Controlled Trial

## Document Control

| Field | Value |
|-------|-------|
| **Model Version** | [FILL: aortica model version] |
| **Date** | [FILL: document date] |
| **Author** | [FILL: lead author / team] |
| **Study Title** | [FILL: title of the RCT] |
| **Trial Registration** | [FILL: registration number, e.g. ClinicalTrials.gov NCTxxxxxxxx] |

> **Reference**: Liu X, et al. "Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension." Nature Medicine. 2020;26:1364–1374.
>
> CONSORT-AI extends the CONSORT 2010 statement for randomised controlled trials evaluating AI interventions.

---

## Checklist Items

### Title and Abstract

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 1a | Identify the study as a randomised trial involving an AI intervention | Title | [FILL: e.g. "Effect of AI-assisted ECG interpretation (Aortica) on diagnostic accuracy and clinical decision-making: a randomised controlled trial"] | [ ] |
| 1b | Provide a structured abstract | Abstract | [FILL: structured abstract including objective, design, setting, participants, intervention, main outcome measures, results, conclusions] | [ ] |

### Introduction

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 2a | Describe the scientific background and rationale | Introduction | AI-assisted ECG analysis has potential to improve diagnostic consistency, reduce interpretation time, and catch rare conditions. Aortica provides multi-task decision support across 72 ECG findings. | [ ] |
| 2b | Describe the AI intervention, including its intended role (autonomous, assistive, or augmentative) | Introduction | Aortica serves as an **assistive** decision support tool. All AI findings require clinician review. The system provides ranked findings with confidence levels, XAI explanations, and clinical suggestion prompts. | [ ] |
| 3 | State specific objectives or hypotheses | Introduction | [FILL: e.g. "To evaluate whether AI-assisted ECG interpretation with Aortica improves diagnostic accuracy and reduces time-to-interpretation compared to unassisted interpretation by clinicians"] | [ ] |

### Methods — Trial Design

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 4a | Describe the trial design (e.g. parallel, crossover, factorial) | Methods | [FILL: trial design] | [ ] |
| 4b | Describe any changes to the trial design after commencement, with reasons | Methods | [FILL: protocol deviations if any] | [ ] |

### Methods — Participants

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 5a | Describe the eligibility criteria for participants | Methods | [FILL: patient eligibility criteria, e.g. "Adults (≥18 years) presenting with clinical indication for 12-lead ECG at participating sites"] | [ ] |
| 5b | Describe the eligibility criteria for sites and clinician participants | Methods | [FILL: site and clinician inclusion criteria, e.g. "Board-certified physicians with ≥1 year experience in ECG interpretation"] | [ ] |
| 6a | Describe the settings and locations where data were collected | Methods | [FILL: participating sites, geographic distribution] | [ ] |

### Methods — Interventions

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 7a | **Intervention arm**: Describe the AI intervention with sufficient detail for replication | Methods | Clinician interprets 12-lead ECG with access to Aortica AI decision support. Aortica displays: ranked findings with confidence levels, XAI attributions mapped to named ECG features, clinical suggestion prompts, edge-case spotlight for rare conditions, and signal quality assessment. | [ ] |
| 7b | **Control arm**: Describe the comparison intervention | Methods | [FILL: control condition, e.g. "Clinician interprets 12-lead ECG without AI assistance, using standard clinical tools and references only"] | [ ] |
| 7c | Describe the AI system version, training data, and deployment configuration | Methods | Aortica v[FILL: version]. Architecture: 1D ResNet + cross-lead attention + 4 task heads (72 outputs). Pre-trained on PTB-XL (CC BY 4.0). Deployed via [FILL: Docker/pip install] on [FILL: hardware]. Full spec: [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md). | [ ] |
| 7d | Describe the human-AI interaction design | Methods | Aortica presents findings via web UI copilot panel. Clinician reviews AI outputs alongside ECG waveform with XAI overlay. Clinician retains full autonomy to accept, reject, or modify each AI finding. Feedback collected via `POST /api/v1/feedback`. | [ ] |
| 7e | Describe any training provided to participants on using the AI intervention | Methods | [FILL: training protocol for clinicians in intervention arm, e.g. "30-minute orientation covering: system overview, output interpretation, XAI overlay usage, confidence level interpretation, and limitations"] | [ ] |

### Methods — Outcomes

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 8a | Describe the primary and secondary outcome measures | Methods | [FILL: e.g. "Primary: diagnostic accuracy (sensitivity, specificity) for [critical conditions]. Secondary: time-to-interpretation, clinician confidence, appropriate referral rate, inter-rater agreement"] | [ ] |
| 8b | Describe the reference standard used to adjudicate outcomes | Methods | [FILL: reference standard, e.g. "Expert panel consensus (3 cardiologists, majority vote) blinded to intervention arm"] | [ ] |
| 8c | Describe any AI-specific outcome measures | Methods | [FILL: e.g. "AI acceptance rate, AI override rate, concordance between AI and clinician, time spent reviewing AI findings, clinician trust/satisfaction (Likert scale)"] | [ ] |

### Methods — Sample Size

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 9 | Report how sample size was determined, including assumptions used | Methods | [FILL: sample size calculation — e.g. "Assuming baseline sensitivity 85%, expected improvement to 92%, alpha=0.05, power=80%, calculated N=XXX per arm. Allowing for 10% dropout: target N=XXX per arm."] | [ ] |

### Methods — Randomisation

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 10 | Describe the randomisation method (sequence generation) | Methods | [FILL: randomisation method, e.g. "Computer-generated block randomisation (block size 4), stratified by site and clinician experience level"] | [ ] |
| 11a | Describe allocation concealment mechanism | Methods | [FILL: allocation concealment] | [ ] |
| 12a | Describe who was blinded after assignment (participants, care providers, outcome assessors) | Methods | [FILL: blinding — note that blinding to AI availability is typically infeasible; describe blinding of outcome assessors to allocation] | [ ] |

### Methods — Statistical Analysis

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 13a | Describe the primary statistical analysis methods | Methods | [FILL: statistical methods, e.g. "McNemar test for paired proportions (sensitivity, specificity), Mann-Whitney U for time-to-interpretation, mixed-effects model accounting for clinician and site clustering"] | [ ] |
| 13b | Describe any subgroup or sensitivity analyses | Methods | Prespecified subgroup analyses: by clinician experience level, by ECG condition prevalence, by signal quality tier (good/marginal/poor), by demographic subgroup (sex, age decile). | [ ] |

### Results — Participant Flow

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 14a | Report the number of participants screened, enrolled, allocated, and analysed (CONSORT flow diagram) | Results | [FILL: CONSORT flow diagram] | [ ] |
| 14b | Report the number of AI system failures, errors, or unavailability events | Results | [FILL: AI system uptime, error rate, fallback to edge model events, quality score rejections] | [ ] |

### Results — Baseline Data

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 15 | Report baseline demographic and clinical characteristics for each group | Results | [FILL: baseline characteristics table — age, sex, indication for ECG, comorbidities, ECG condition distribution] | [ ] |

### Results — Outcomes

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 16a | Report the primary outcome for each group with effect estimate and confidence interval | Results | [FILL: primary outcome results — e.g. "Sensitivity for critical findings: intervention 93.2% (95% CI 90.1–95.6) vs control 86.4% (82.5–89.7), p=0.002"] | [ ] |
| 16b | Report secondary outcomes | Results | [FILL: secondary outcome results — time-to-interpretation, referral rates, etc.] | [ ] |
| 16c | Report AI-specific outcomes | Results | [FILL: AI acceptance rate, override rate, concordance, trust metrics] | [ ] |
| 17 | Report adverse events and AI-related safety events | Results | [FILL: adverse events — e.g. false negative rate for critical conditions in each arm, adverse event reports via `POST /api/v1/validation/adverse-event`] | [ ] |

### Discussion

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 18 | Discuss limitations, including sources of bias and imprecision | Discussion | [FILL: limitations — e.g. "Single-centre, Hawthorne effect, inability to blind clinicians to AI availability, potential automation bias in intervention arm"] | [ ] |
| 19 | Discuss generalisability of findings | Discussion | Aortica trained on PTB-XL (European-biased). Performance may vary in populations underrepresented in training data. Equity gating enforces no significant demographic gaps. See [IEC 80601-2-86 ATD § 5](IEC_80601_2_86_ATD.md#5-known-limitations). | [ ] |
| 19b | Discuss AI-specific implications: automation bias, human factors, integration into clinical workflow | Discussion | [FILL: discussion of automation bias risk, clinician de-skilling concerns, appropriate trust calibration, workflow integration] | [ ] |

### Other Information

| # | Item | Section | Aortica Context | Status |
|---|------|---------|-----------------|--------|
| 20 | Report trial registration number and registry name | Other | [FILL: registration details] | [ ] |
| 21 | Report where the full trial protocol can be accessed | Other | [FILL: protocol URL or reference. See also `docs/validation/PROSPECTIVE_PROTOCOL.md` for Aortica-specific protocol template] | [ ] |
| 22 | Report funding sources and role of funders | Other | [FILL: funding and conflicts] | [ ] |
| 23 | Report AI system and data availability | Other | Open source: Apache 2.0. Code: GitHub. Models: HuggingFace Hub. Benchmark reproducible via `aortica benchmark`. Training data: PTB-XL (PhysioNet, CC BY 4.0). | [ ] |

---

*This checklist follows the CONSORT-AI reporting guideline for RCTs evaluating AI interventions. Items marked `[FILL: ...]` require study-specific information. Some items can be partially pre-filled using `aortica.regulatory.generate_reporting_checklist(template='consort_ai', benchmark_report=report)`. Cross-references to [IEC 80601-2-86 ATD](IEC_80601_2_86_ATD.md) and the [prospective protocol template](../validation/PROSPECTIVE_PROTOCOL.md) are provided where applicable.*
