import { useState, useMemo } from 'react';
import './ExplanationCard.css';

/* ── Types ──────────────────────────────────────────────── */

export interface ECGFeatureAttribution {
  feature_name: string;
  lead: string;
  delta_score: number;
}

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  coverage: number;       // e.g. 0.90 for 90%
  predictionSetSize?: number;
}

export interface ClinicalSuggestionInfo {
  prompt: string;
  urgency: 'routine' | 'prompt' | 'urgent' | 'emergent';
  rationale?: string;
}

export interface ExplanationCardProps {
  /** Finding name (condition) */
  findingName: string;
  /** Task head this finding belongs to */
  task: string;
  /** Confidence probability (0–1) */
  confidence: number;
  /** Top-3 ECG feature attributions from XAI (integrated gradients) */
  featureAttributions?: ECGFeatureAttribution[];
  /** Confidence interval from conformal prediction */
  confidenceInterval?: ConfidenceInterval;
  /** Clinical reference text for this condition */
  clinicalReference?: string;
  /** Clinical suggestion with urgency */
  suggestion?: ClinicalSuggestionInfo;
  /** Whether this card starts expanded */
  defaultExpanded?: boolean;
  /** Unique id for testing */
  id?: string;
}

/* ── Clinical reference data ───────────────────────────── */

/**
 * Comprehensive clinical reference texts for each detectable condition.
 * These are used when no external reference is provided via props.
 */
const CLINICAL_REFERENCES: Record<string, string> = {
  /* Rhythm & Conduction */
  'AF':                  'Atrial fibrillation is characterised by disorganised atrial electrical activity with irregularly irregular ventricular response. It increases stroke risk 5-fold and may cause heart failure if rate is uncontrolled.',
  'Atrial Fibrillation': 'Atrial fibrillation is characterised by disorganised atrial electrical activity with irregularly irregular ventricular response. It increases stroke risk 5-fold and may cause heart failure if rate is uncontrolled.',
  'AFL':                 'Atrial flutter involves a re-entrant circuit in the right atrium, producing a sawtooth pattern at ~300 bpm with regular ventricular response (typically 150 bpm with 2:1 block).',
  'SVT':                 'Supraventricular tachycardia encompasses regular narrow-complex tachycardias arising above the ventricles. Most are paroxysmal and hemodynamically stable.',
  'AVNRT':               'AV nodal re-entrant tachycardia is the most common regular SVT. Dual AV nodal pathways create a re-entry circuit with retrograde P waves often hidden in QRS.',
  'AVRT':                'AV re-entrant tachycardia uses an accessory pathway for re-entry. Orthodromic AVRT produces narrow QRS; antidromic AVRT produces wide QRS.',
  'VT':                  'Ventricular tachycardia is a life-threatening arrhythmia originating below the bundle of His. Sustained VT (>30s) can deteriorate to ventricular fibrillation.',
  'VF':                  'Ventricular fibrillation produces chaotic ventricular activity with no effective cardiac output. This is a cardiac arrest rhythm requiring immediate defibrillation.',
  'Idioventricular':     'Accelerated idioventricular rhythm (60–100 bpm) is a wide-complex rhythm often seen after reperfusion therapy. Usually self-limiting and rarely requires treatment.',
  'Sinus Brady':         'Sinus bradycardia (<60 bpm) may be normal in athletes or during sleep. Pathological causes include hypothyroidism, increased ICP, and drug effects.',
  'Sinus Tachy':         'Sinus tachycardia (>100 bpm) is a physiological response to increased sympathetic tone. Always evaluate for underlying cause rather than treating the rate alone.',
  'Sinus Tachycardia':   'Sinus tachycardia (>100 bpm) is a physiological response to increased sympathetic tone. Always evaluate for underlying cause rather than treating the rate alone.',
  'PAC':                 'Premature atrial complexes are ectopic atrial beats occurring before the expected sinus beat. Isolated PACs are benign; frequent PACs may predict AF development.',
  'PVC':                 'Premature ventricular complexes are early wide-complex beats. Burden >10% of total beats warrants investigation for cardiomyopathy development.',
  '1st AVB':             'First-degree AV block (PR >200ms) is usually benign, reflecting delayed conduction through the AV node. May be normal in athletes and vagotonic individuals.',
  '2nd AVB':             'Second-degree AV block has two forms: Mobitz I (Wenckebach) with progressive PR prolongation (usually benign) and Mobitz II with sudden dropped QRS (risk of complete block).',
  '3rd AVB':             'Complete heart block shows AV dissociation with independent atrial and ventricular rhythms. Narrow escape rhythm suggests junctional; wide escape suggests ventricular origin.',
  'LBBB':                'Left bundle branch block produces QRS ≥120ms with broad R waves in I and V5-V6. New LBBB with chest pain meets STEMI-equivalent criteria per Sgarbossa.',
  'RBBB':                'Right bundle branch block shows rsR\' in V1 with wide S in I and V6. Isolated RBBB is often benign; new RBBB may indicate right heart strain or PE.',
  'LAFB':                'Left anterior fascicular block produces left axis deviation (-45° to -90°) with qR in I/aVL and rS in II/III/aVF. Usually benign in isolation.',
  'LPFB':                'Left posterior fascicular block shows right axis deviation (>90°) with rS in I/aVL and qR in II/III/aVF. Rare; consider structural heart disease.',
  'WPW':                 'Wolff-Parkinson-White syndrome involves an accessory pathway causing pre-excitation (short PR, delta wave). Risk of sudden death during atrial fibrillation with rapid ventricular response.',
  'Pacemaker':           'Pacemaker rhythm shows pacing spikes before atrial and/or ventricular complexes. Evaluate for appropriate capture, sensing, and pacing mode.',
  'Normal Sinus':        'Normal sinus rhythm with regular rate 60–100 bpm, upright P waves in II, and consistent PR interval. Represents normal cardiac electrical activity.',
  'Normal Sinus Rhythm': 'Normal sinus rhythm with regular rate 60–100 bpm, upright P waves in II, and consistent PR interval. Represents normal cardiac electrical activity.',

  /* Structural & Functional */
  'LVH':                       'Left ventricular hypertrophy increases QRS voltage and may produce repolarisation abnormalities (strain pattern). Common causes include hypertension and aortic stenosis.',
  'Left Ventricular Hypertrophy': 'Left ventricular hypertrophy increases QRS voltage and may produce repolarisation abnormalities (strain pattern). Common causes include hypertension and aortic stenosis.',
  'RVH':                       'Right ventricular hypertrophy shows right axis deviation, dominant R in V1, and RV strain pattern. Suggests pulmonary hypertension or congenital heart disease.',
  'LVSD':                      'Left ventricular systolic dysfunction may manifest as poor R-wave progression, Q waves, or conduction abnormalities on ECG. Echocardiographic confirmation is essential.',
  'HFpEF Risk':                'Heart failure with preserved ejection fraction risk features include LA enlargement, AF, and LVH patterns. NT-proBNP and diastolic function assessment needed.',
  'DCM':                       'Dilated cardiomyopathy shows poor R-wave progression, conduction delays, and sometimes pseudo-infarct patterns. Comprehensive imaging and genetic evaluation indicated.',
  'Dilated Cardiomyopathy':    'Dilated cardiomyopathy shows poor R-wave progression, conduction delays, and sometimes pseudo-infarct patterns. Comprehensive imaging and genetic evaluation indicated.',
  'HCM':                       'Hypertrophic cardiomyopathy is the most common inherited cardiac condition (1:500). ECG shows LVH, deep narrow Q waves, T-wave inversions. Sudden death risk stratification critical.',
  'ARVC':                      'Arrhythmogenic right ventricular cardiomyopathy causes fibro-fatty replacement of RV myocardium. Leading cause of SCD in young athletes. Epsilon waves and T-wave inversions V1-V3 are hallmarks.',
  'Amyloidosis':               'Cardiac amyloidosis causes restrictive cardiomyopathy with classic voltage-mass mismatch. Low voltage on ECG despite increased wall thickness on echo. ATTR and AL types require different treatment.',
  'Aortic Stenosis':           'Aortic stenosis produces LVH with strain pattern on ECG. Severity correlates poorly with ECG findings — echocardiographic assessment is mandatory.',
  'Mitral Regurgitation':      'Mitral regurgitation may show LA enlargement, AF, and LVH on ECG. Severity assessment requires echocardiography with quantitative measures.',
  'Pulmonary HTN':             'Pulmonary hypertension shows right heart strain pattern: right axis, P pulmonale, RVH, and RBBB. Right heart catheterisation is definitive.',
  'LA Enlargement':            'Left atrial enlargement manifests as broad, notched P waves in II and deep negative P wave in V1. Associated with AF risk, mitral valve disease, and LV dysfunction.',
  'RA Enlargement':            'Right atrial enlargement shows tall, peaked P waves (>2.5mm) in II. Associated with COPD, pulmonary hypertension, and right-sided valvular disease.',
  'Pericarditis':              'Acute pericarditis shows diffuse ST elevation (concave up), PR depression, and absence of reciprocal changes. Differentiate from STEMI by distribution and morphology.',
  'Pericarditis Pattern':      'Acute pericarditis shows diffuse ST elevation (concave up), PR depression, and absence of reciprocal changes. Differentiate from STEMI by distribution and morphology.',
  'Myocarditis':               'Myocarditis may mimic MI with ST changes and troponin elevation. Often viral aetiology. Cardiac MRI with late gadolinium enhancement confirms diagnosis.',
  'Myocarditis Pattern':       'Myocarditis may mimic MI with ST changes and troponin elevation. Often viral aetiology. Cardiac MRI with late gadolinium enhancement confirms diagnosis.',

  /* Ischaemia & Metabolic */
  'STEMI':                      'ST-elevation myocardial infarction indicates acute transmural ischaemia from coronary artery occlusion. Time-critical: door-to-balloon <90 minutes. Identify territory by lead distribution.',
  'ST Elevation (Lateral)':     'Lateral ST elevation (I, aVL, V5-V6) suggests LCx territory involvement. Activate emergent reperfusion pathway per local protocol.',
  'Posterior MI':               'Posterior MI is often missed on standard 12-lead, showing reciprocal changes (ST depression, tall R waves) in V1-V3. Posterior leads V7-V9 confirm the diagnosis.',
  'Occlusive NSTEMI':           'Occlusive NSTEMI has acute coronary occlusion without classic ST elevation. Includes de Winter, Wellens, and aVR patterns. Requires urgent coronary intervention despite \"NSTEMI\" classification.',
  'Old MI':                     'Pathological Q waves with normal ST segments indicate prior myocardial infarction. Compare with prior ECGs for chronology. Optimise secondary prevention.',
  'Hyperkalaemia':              'ECG changes correlate with serum K+: peaked T waves (>5.5), PR prolongation (>6.0), P-wave loss and QRS widening (>6.5), sine wave (>7.0). Each stage has life-threatening potential.',
  'Hypokalaemia':               'Low potassium flattens T waves, produces U waves, prolongs QT interval, and causes ST depression. Increases digitalis toxicity risk. Replace K+ and Mg²⁺ concurrently.',
  'Hypercalcaemia':             'Elevated calcium shortens QT interval (specifically the ST segment). Severe hypercalcaemia can cause Osborn waves and ventricular arrhythmias.',
  'Hypothyroidism':             'Hypothyroidism causes sinus bradycardia, low voltage, and prolonged QT interval. Pericardial effusion may contribute to low voltage. Treat underlying thyroid disorder.',
  'Hypothyroidism Pattern':     'Hypothyroidism causes sinus bradycardia, low voltage, and prolonged QT interval. Pericardial effusion may contribute to low voltage. Treat underlying thyroid disorder.',
  'Digitalis':                  'Digitalis effect shows reverse-tick ST depression (Salvador Dalí moustache), shortened QT, and PR prolongation. Toxicity produces any arrhythmia — check drug level.',
  'Digitalis Effect':           'Digitalis effect shows reverse-tick ST depression (Salvador Dalí moustache), shortened QT, and PR prolongation. Toxicity produces any arrhythmia — check drug level.',
  'QTc Prolongation':           'Prolonged QTc (>470ms male, >480ms female) increases risk of Torsades de Pointes. Common causes: drugs (Class III antiarrhythmics, antibiotics, antipsychotics), electrolyte imbalances, congenital LQTS.',
};

/**
 * Urgency-to-display config for the Clinical Suggestions section.
 */
const URGENCY_CONFIG: Record<string, { label: string; color: string }> = {
  routine:  { label: 'Routine',  color: 'var(--explanation-urgency-routine)' },
  prompt:   { label: 'Prompt',   color: 'var(--explanation-urgency-prompt)' },
  urgent:   { label: 'Urgent',   color: 'var(--explanation-urgency-urgent)' },
  emergent: { label: 'Emergent', color: 'var(--explanation-urgency-emergent)' },
};

/* ── Sub-components ─────────────────────────────────────── */

function FeatureAttributionChart({ features }: { features: ECGFeatureAttribution[] }) {
  if (features.length === 0) return null;

  const maxScore = Math.max(...features.map(f => Math.abs(f.delta_score)), 1e-10);

  return (
    <div className="explanation-section" id="explanation-feature-chart">
      <h4 className="explanation-section-heading">
        <span className="explanation-section-icon">📊</span>
        Feature Attributions
      </h4>
      <p className="explanation-section-subtitle">
        Top ECG features driving this detection, ranked by contribution strength
      </p>
      <div className="explanation-feature-bars">
        {features.map((f, i) => {
          const pct = (Math.abs(f.delta_score) / maxScore) * 100;
          const isPositive = f.delta_score >= 0;

          return (
            <div key={`${f.feature_name}-${f.lead}-${i}`} className="explanation-feature-row">
              <div className="explanation-feature-meta">
                <span className="explanation-feature-rank">#{i + 1}</span>
                <span className="explanation-feature-name">{f.feature_name}</span>
                <span className="explanation-feature-lead">{f.lead}</span>
              </div>
              <div className="explanation-feature-bar-track">
                <div
                  className={`explanation-feature-bar-fill ${isPositive ? 'explanation-feature-bar-fill--positive' : 'explanation-feature-bar-fill--negative'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`explanation-feature-delta ${isPositive ? 'explanation-feature-delta--positive' : 'explanation-feature-delta--negative'}`}>
                {isPositive ? '+' : ''}{f.delta_score.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ConfidenceIntervalBar({ ci, confidence }: { ci: ConfidenceInterval; confidence: number }) {
  // Clamp values to [0, 1]
  const lower = Math.max(0, Math.min(1, ci.lower));
  const upper = Math.max(0, Math.min(1, ci.upper));
  const point = Math.max(0, Math.min(1, confidence));

  return (
    <div className="explanation-section" id="explanation-confidence-interval">
      <h4 className="explanation-section-heading">
        <span className="explanation-section-icon">📏</span>
        Confidence Interval
      </h4>
      <p className="explanation-section-subtitle">
        {(ci.coverage * 100).toFixed(0)}% coverage conformal prediction interval
      </p>
      <div className="explanation-ci-container">
        <div className="explanation-ci-track">
          {/* Range bar */}
          <div
            className="explanation-ci-range"
            style={{
              left: `${lower * 100}%`,
              width: `${(upper - lower) * 100}%`,
            }}
          />
          {/* Point estimate marker */}
          <div
            className="explanation-ci-point"
            style={{ left: `${point * 100}%` }}
            title={`Point estimate: ${(point * 100).toFixed(1)}%`}
          />
          {/* Lower bound marker */}
          <div
            className="explanation-ci-bound explanation-ci-bound--lower"
            style={{ left: `${lower * 100}%` }}
          />
          {/* Upper bound marker */}
          <div
            className="explanation-ci-bound explanation-ci-bound--upper"
            style={{ left: `${upper * 100}%` }}
          />
        </div>
        <div className="explanation-ci-labels">
          <span className="explanation-ci-label">{(lower * 100).toFixed(1)}%</span>
          <span className="explanation-ci-label explanation-ci-label--point">
            {(point * 100).toFixed(1)}%
          </span>
          <span className="explanation-ci-label">{(upper * 100).toFixed(1)}%</span>
        </div>
        {ci.predictionSetSize !== undefined && (
          <div className="explanation-ci-set-size">
            Prediction set size: <strong>{ci.predictionSetSize}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Main ExplanationCard component ─────────────────────── */

export function ExplanationCard({
  findingName,
  task,
  confidence,
  featureAttributions = [],
  confidenceInterval,
  clinicalReference,
  suggestion,
  defaultExpanded = false,
  id,
}: ExplanationCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  // Resolve clinical reference text
  const referenceText = useMemo(() => {
    if (clinicalReference) return clinicalReference;
    return CLINICAL_REFERENCES[findingName] || null;
  }, [clinicalReference, findingName]);

  // Task display label
  const taskLabel = useMemo(() => {
    switch (task) {
      case 'rhythm':     return 'Rhythm & Conduction';
      case 'structural': return 'Structural & Functional';
      case 'ischaemia':  return 'Ischaemia & Metabolic';
      default:           return task;
    }
  }, [task]);

  const cardId = id || `explanation-card-${findingName.replace(/\s+/g, '-').toLowerCase()}`;

  return (
    <div className={`explanation-card ${expanded ? 'explanation-card--expanded' : ''}`} id={cardId}>
      {/* Toggle button */}
      <button
        className={`explanation-card-toggle ${expanded ? 'explanation-card-toggle--open' : ''}`}
        onClick={() => setExpanded(prev => !prev)}
        aria-expanded={expanded}
        aria-controls={`${cardId}-content`}
        id={`${cardId}-toggle`}
        title={expanded ? 'Collapse explanation' : 'Show detailed explanation'}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="explanation-card-toggle-icon">
          <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.2" />
          <path d="M7 4.5v5M4.5 7h5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        <span>{expanded ? 'Hide Details' : 'Explain'}</span>
      </button>

      {/* Expandable content */}
      {expanded && (
        <div className="explanation-card-content" id={`${cardId}-content`}>
          {/* Header with finding summary */}
          <div className="explanation-card-header">
            <div className="explanation-card-header-left">
              <span className="explanation-card-finding-name">{findingName}</span>
              <span className="explanation-card-task-badge">{taskLabel}</span>
            </div>
            <span className="explanation-card-confidence">
              {(confidence * 100).toFixed(0)}% confidence
            </span>
          </div>

          {/* Clinical Reference */}
          {referenceText && (
            <div className="explanation-section">
              <h4 className="explanation-section-heading">
                <span className="explanation-section-icon">📖</span>
                Clinical Reference
              </h4>
              <p className="explanation-section-text">{referenceText}</p>
            </div>
          )}

          {/* Clinical Suggestions */}
          {suggestion && (
            <div className="explanation-section explanation-section--suggestion">
              <h4 className="explanation-section-heading">
                <span className="explanation-section-icon">💡</span>
                Suggested Next Steps
              </h4>
              <p className="explanation-section-subtitle explanation-suggestion-disclaimer">
                AI-generated clinical prompts — requires clinician judgment
              </p>
              <div className="explanation-suggestion-content">
                <span
                  className={`explanation-urgency-tag explanation-urgency-tag--${suggestion.urgency}`}
                  style={{ '--urgency-color': URGENCY_CONFIG[suggestion.urgency]?.color } as React.CSSProperties}
                >
                  {URGENCY_CONFIG[suggestion.urgency]?.label || suggestion.urgency}
                </span>
                <span className="explanation-suggestion-prompt">{suggestion.prompt}</span>
              </div>
              {suggestion.rationale && (
                <p className="explanation-suggestion-rationale">{suggestion.rationale}</p>
              )}
            </div>
          )}

          {/* Feature Attributions (XAI) */}
          {featureAttributions.length > 0 && (
            <FeatureAttributionChart features={featureAttributions} />
          )}

          {/* Confidence Interval */}
          {confidenceInterval && (
            <ConfidenceIntervalBar ci={confidenceInterval} confidence={confidence} />
          )}

          {/* Similar Historical Cases — Placeholder */}
          <div className="explanation-section explanation-section--placeholder">
            <h4 className="explanation-section-heading">
              <span className="explanation-section-icon">🔍</span>
              Similar Historical Cases
            </h4>
            <div className="explanation-placeholder-content">
              <span className="explanation-placeholder-badge">Phase 4</span>
              <p className="explanation-placeholder-text">
                Case-based retrieval from de-identified ECG databases will be available in a future release.
                Top-3 similar historical ECGs with verified diagnoses and outcomes will be displayed here.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
