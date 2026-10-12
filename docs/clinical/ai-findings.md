# AI Findings Guide

Understanding Aortica's multi-task AI output and clinical context.

!!! warning "Decision Support Only"
    Aortica is an AI decision-support tool — **not a diagnostic device**. All findings require clinician review and clinical correlation. AI predictions should never be the sole basis for treatment decisions.

!!! danger "Most outputs are not trained"
    The architecture has 72 outputs; the released model trains **28**:
    AF, AFL, SVT, AVRT, sinus bradycardia/tachycardia, PAC, PVC,
    1st/2nd/3rd-degree AV block, LBBB, RBBB, LAFB, LPFB, WPW, paced rhythm,
    normal sinus rhythm, LVH, RVH, left/right atrial enlargement, STEMI,
    posterior MI, old MI, QTc prolongation, digitalis effect and early
    repolarisation. Every other output below — including VT/VF, Brugada,
    cardiomyopathies, electrolyte disorders and all risk scores — is
    untrained and is withheld from every response. Results for the trained
    outputs: `releases/v0.4.0/RESULTS.md`.

## Task Heads

### Rhythm & Conduction (28 outputs, 18 trained)

Detects rhythm abnormalities and conduction disorders:

| Finding | Clinical Significance |
|---------|----------------------|
| Atrial Fibrillation (AF) | Irregular rhythm, stroke risk |
| Atrial Flutter (AFL) | Regular reentrant tachycardia |
| Supraventricular Tachycardia (SVT) | Rapid narrow-complex rhythm |
| Ventricular Tachycardia (VT) | **Life-threatening** — wide-complex tachycardia |
| Ventricular Fibrillation (VF) | **Cardiac arrest** — requires immediate defibrillation |
| 1st / 2nd / 3rd degree AV Block | Conduction delay or complete block |
| LBBB / RBBB | Bundle branch blocks |
| WPW | Accessory pathway — risk of rapid conduction in AF |
| Pacemaker Rhythm | Ventricular pacing spikes |
| Normal Sinus Rhythm | No abnormality detected |

### Structural & Functional (19 outputs, 4 trained)

Screens for structural heart disease:

| Finding | Clinical Significance |
|---------|----------------------|
| LVH / RVH | Ventricular hypertrophy (pressure/volume overload) |
| LVSD | Left ventricular systolic dysfunction |
| HCM / DCM | Hypertrophic / Dilated cardiomyopathy |
| ARVC | Arrhythmogenic right ventricular cardiomyopathy |
| Pericarditis / Myocarditis | Inflammatory patterns |
| LA / RA Enlargement | Atrial dilation |

### Ischaemia & Metabolic (19 outputs, 6 trained)

Detects acute and chronic ischaemic patterns:

| Finding | Clinical Significance |
|---------|----------------------|
| STEMI (per territory) | **Acute MI** — requires emergent cath lab activation |
| Posterior MI | Often missed — reciprocal changes in V1–V3 |
| Hyperkalaemia | **Metabolic emergency** — peaked T waves, wide QRS |
| QTc Prolongation | Risk of Torsades de Pointes |
| Digitalis Effect | Drug-related ST changes |

### Risk Prediction (6 outputs, none trained)

Continuous 0–1 risk scores:

| Score | Prediction Window |
|-------|-------------------|
| All-cause mortality | 1-year |
| HF hospitalization | 12-month |
| AF onset | 12-month |

## Confidence & Uncertainty

### Confidence Levels

- 🔴 **≥80%** — High confidence positive finding
- 🟡 **50–79%** — Moderate confidence — clinical review recommended
- 🟢 **<50%** — Low confidence — likely negative

### Uncertainty Indicators

- **Conformal prediction set size** — smaller sets = higher confidence
- **OOD flag** — input may be outside the model's training distribution
- **Entropy score** — higher entropy = more uncertain prediction

## XAI Explanations

For each finding, Aortica provides:

1. **Feature attribution** — which ECG segments (P wave, QRS, ST, T wave) drove the detection
2. **Top-3 contributing features** — ranked by importance with delta-contribution scores
3. **Confidence intervals** — conformal prediction bounds

## Clinical Suggestions

Non-prescriptive next-step prompts are provided for each finding with urgency levels:

- 🔴 **Emergent** — immediate action (e.g., "Activate cath lab for STEMI")
- 🟠 **Urgent** — prompt evaluation (e.g., "Urgent 12-lead repeat recommended")
- 🟡 **Prompt** — scheduled follow-up (e.g., "Consider cardiology referral")
- ⚪ **Routine** — standard monitoring (e.g., "Routine follow-up appropriate")
