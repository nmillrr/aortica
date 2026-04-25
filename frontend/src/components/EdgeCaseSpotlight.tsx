import { useMemo, useState } from 'react';
import './EdgeCaseSpotlight.css';

/* ── Edge-case condition definitions ──────────────────── */

export interface EdgeCaseCondition {
  /** Canonical class name (must match finding.name from predictions) */
  className: string;
  /** Human-friendly display name */
  displayName: string;
  /** Short note on why this is an edge case */
  prevalenceNote: string;
  /** Minimum confidence to flag (0–1) */
  minConfidence: number;
  /** Clinical urgency level for visual emphasis */
  urgency: 'routine' | 'prompt' | 'urgent' | 'emergent';
}

/**
 * Default edge-case condition list — low-prevalence, high-consequence patterns
 * that should NEVER be buried in routine results.
 */
export const DEFAULT_EDGE_CASES: EdgeCaseCondition[] = [
  {
    className: 'WPW',
    displayName: 'Wolff-Parkinson-White',
    prevalenceNote: 'Prevalence ~0.1–0.3% — risk of sudden cardiac death if undiagnosed',
    minConfidence: 0.25,
    urgency: 'urgent',
  },
  {
    className: 'VT',
    displayName: 'Ventricular Tachycardia',
    prevalenceNote: 'Life-threatening arrhythmia — requires immediate assessment',
    minConfidence: 0.20,
    urgency: 'emergent',
  },
  {
    className: 'VF',
    displayName: 'Ventricular Fibrillation',
    prevalenceNote: 'Cardiac arrest rhythm — immediate defibrillation required',
    minConfidence: 0.15,
    urgency: 'emergent',
  },
  {
    className: 'STEMI',
    displayName: 'ST-Elevation MI',
    prevalenceNote: 'Acute coronary occlusion — door-to-balloon time critical',
    minConfidence: 0.20,
    urgency: 'emergent',
  },
  {
    className: '3rd AVB',
    displayName: 'Complete Heart Block',
    prevalenceNote: 'Prevalence <0.05% — may require emergency pacing',
    minConfidence: 0.20,
    urgency: 'emergent',
  },
  {
    className: 'ARVC',
    displayName: 'Arrhythmogenic RV Cardiomyopathy',
    prevalenceNote: 'Prevalence ~1:2000–5000 — leading cause of SCD in young athletes',
    minConfidence: 0.15,
    urgency: 'urgent',
  },
  {
    className: 'HCM',
    displayName: 'Hypertrophic Cardiomyopathy',
    prevalenceNote: 'Prevalence ~1:500 — most common inherited cardiac condition, SCD risk',
    minConfidence: 0.20,
    urgency: 'urgent',
  },
  {
    className: 'Amyloidosis',
    displayName: 'Cardiac Amyloidosis',
    prevalenceNote: 'Often missed — early detection dramatically changes prognosis',
    minConfidence: 0.15,
    urgency: 'urgent',
  },
  {
    className: 'Hyperkalaemia',
    displayName: 'Severe Hyperkalaemia',
    prevalenceNote: 'ECG changes may precede fatal arrhythmia — urgent correction needed',
    minConfidence: 0.30,
    urgency: 'emergent',
  },
  {
    className: 'QTc Prolongation',
    displayName: 'QTc Prolongation',
    prevalenceNote: 'Risk of Torsades de Pointes — review QT-prolonging medications',
    minConfidence: 0.25,
    urgency: 'urgent',
  },
  {
    className: 'Posterior MI',
    displayName: 'Posterior MI',
    prevalenceNote: 'Frequently missed on standard 12-lead — obtain V7-V9',
    minConfidence: 0.20,
    urgency: 'emergent',
  },
  {
    className: 'Occlusive NSTEMI',
    displayName: 'Occlusive NSTEMI',
    prevalenceNote: 'Subtle presentation — serial troponins and urgent cardiology review',
    minConfidence: 0.25,
    urgency: 'urgent',
  },
];

/* ── Flagged edge case type ──────────────────────────── */

export interface FlaggedEdgeCase {
  condition: EdgeCaseCondition;
  confidence: number;
  task: string;
  isNew: boolean;
}

/* ── Explanation card data ───────────────────────────── */

interface ExplanationCardData {
  clinicalBackground: string;
  keyECGFeatures: string[];
  recommendedAction: string;
}

const EXPLANATION_CARDS: Record<string, ExplanationCardData> = {
  'WPW': {
    clinicalBackground: 'Pre-excitation via an accessory pathway (Bundle of Kent) bypassing the AV node. Risk of rapid ventricular response during AF leading to VF.',
    keyECGFeatures: ['Short PR interval (<120ms)', 'Delta wave (slurred QRS upstroke)', 'Wide QRS complex (>120ms)'],
    recommendedAction: 'Electrophysiology referral for risk stratification and potential ablation. Avoid AV nodal blocking agents if AF develops.',
  },
  'VT': {
    clinicalBackground: 'Sustained VT is a life-threatening arrhythmia originating from ventricular myocardium. Can degenerate to VF.',
    keyECGFeatures: ['Wide QRS (>120ms)', 'AV dissociation', 'Fusion/capture beats', 'Concordance across precordial leads'],
    recommendedAction: 'Assess hemodynamic stability. Unstable → immediate cardioversion. Stable → IV amiodarone and cardiology review.',
  },
  'VF': {
    clinicalBackground: 'Chaotic ventricular electrical activity resulting in cardiac arrest. No effective cardiac output.',
    keyECGFeatures: ['Irregular undulating waveform', 'No discernible QRS complexes', 'Variable amplitude and frequency'],
    recommendedAction: 'Immediate defibrillation. Follow ACLS protocol. Identify and treat reversible causes (Hs and Ts).',
  },
  'STEMI': {
    clinicalBackground: 'Complete coronary artery occlusion causing transmural myocardial ischaemia. Time-critical presentation.',
    keyECGFeatures: ['ST elevation ≥1mm in limb leads or ≥2mm in precordial leads', 'Reciprocal ST depression', 'Hyperacute T waves'],
    recommendedAction: 'Activate cath lab immediately. Dual antiplatelet therapy. Target door-to-balloon <90 minutes.',
  },
  '3rd AVB': {
    clinicalBackground: 'Complete dissociation between atrial and ventricular activity. Escape rhythm determines hemodynamic stability.',
    keyECGFeatures: ['Regular P-P intervals', 'Regular R-R intervals', 'No consistent PR relationship', 'Ventricular rate typically 25-45 bpm'],
    recommendedAction: 'Temporary transcutaneous/transvenous pacing if symptomatic. Evaluate for reversible causes. Permanent pacemaker likely needed.',
  },
  'ARVC': {
    clinicalBackground: 'Progressive fibro-fatty replacement of RV myocardium. Major cause of SCD in young athletes.',
    keyECGFeatures: ['Epsilon waves (V1-V3)', 'T-wave inversions V1-V3', 'Prolonged terminal activation duration', 'VT with LBBB morphology'],
    recommendedAction: 'Cardiac MRI for definitive evaluation. Genetic testing. Exercise restriction pending risk assessment.',
  },
  'HCM': {
    clinicalBackground: 'Asymmetric septal hypertrophy with LVOT obstruction risk. Most common inherited cardiac condition.',
    keyECGFeatures: ['LVH voltage criteria', 'Deep narrow Q waves (septal hypertrophy)', 'T-wave inversions (apical variant)', 'LA enlargement'],
    recommendedAction: 'Echocardiography for wall thickness and LVOT gradient. Family screening. SCD risk stratification.',
  },
  'Amyloidosis': {
    clinicalBackground: 'Infiltrative cardiomyopathy from abnormal protein deposition. ATTR and AL types require different treatment.',
    keyECGFeatures: ['Low voltage despite thick walls (voltage-mass mismatch)', 'Pseudo-infarct pattern (Q waves)', 'Conduction abnormalities'],
    recommendedAction: 'Tc-PYP scintigraphy for ATTR. Serum/urine immunofixation for AL. Haematology referral if AL suspected.',
  },
  'Hyperkalaemia': {
    clinicalBackground: 'Progressive ECG changes with rising K+: peaked T → P-wave loss → widened QRS → sine wave → VF/asystole.',
    keyECGFeatures: ['Tall peaked T waves', 'Shortened QT interval', 'Widened QRS (severe)', 'Loss of P waves (severe)'],
    recommendedAction: 'Urgent serum potassium level. IV calcium gluconate for cardiac protection if K+ >6.5. Insulin/dextrose for K+ shifting.',
  },
  'QTc Prolongation': {
    clinicalBackground: 'Delayed ventricular repolarisation increasing risk of Torsades de Pointes polymorphic VT.',
    keyECGFeatures: ['QTc >470ms (males) or >480ms (females)', 'U waves', 'T-wave morphology changes', 'Notched T waves'],
    recommendedAction: 'Review and discontinue QT-prolonging medications. Check Mg²⁺/K⁺/Ca²⁺. If QTc >500ms, consider telemetry.',
  },
  'Posterior MI': {
    clinicalBackground: 'Isolated posterior wall infarction often missed on standard 12-lead. Usually from LCx or dominant RCA occlusion.',
    keyECGFeatures: ['ST depression V1-V3 (reciprocal)', 'Tall R waves V1-V2', 'Upright T waves V1-V2', 'ST elevation V7-V9 (posterior leads)'],
    recommendedAction: 'Obtain posterior leads (V7-V9). If confirmed, activate emergent cath lab as per STEMI protocol.',
  },
  'Occlusive NSTEMI': {
    clinicalBackground: 'Acute coronary occlusion without classic ST elevation. High-risk subset of NSTEMI requiring urgent intervention.',
    keyECGFeatures: ['Hyperacute T waves', 'De Winter pattern', 'Wellens syndrome (biphasic/deep T V2-V3)', 'aVR ST elevation with diffuse depression'],
    recommendedAction: 'Serial troponins at 0/1/3h. Early cardiology consultation. Consider urgent angiography if high-risk features.',
  },
};

/* ── Component props ─────────────────────────────────── */

interface EdgeCaseSpotlightProps {
  /** All findings from classification task heads */
  findings: { name: string; prob: number; task: string }[];
  /** Custom edge-case list (defaults to DEFAULT_EDGE_CASES) */
  edgeCases?: EdgeCaseCondition[];
  /** Callback when an edge case is clicked */
  onEdgeCaseClick?: (findingName: string, task: string) => void;
}

/* ── Main component ──────────────────────────────────── */

export function EdgeCaseSpotlight({
  findings,
  edgeCases = DEFAULT_EDGE_CASES,
  onEdgeCaseClick,
}: EdgeCaseSpotlightProps) {
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>({});

  // Match findings against the edge-case list
  const flagged = useMemo((): FlaggedEdgeCase[] => {
    const result: FlaggedEdgeCase[] = [];

    for (const ec of edgeCases) {
      const finding = findings.find(f => f.name === ec.className);
      if (finding && finding.prob >= ec.minConfidence) {
        result.push({
          condition: ec,
          confidence: finding.prob,
          task: finding.task,
          isNew: true, // In future, compare against previous session to determine novelty
        });
      }
    }

    // Sort by confidence descending, then by urgency
    const urgencyOrder: Record<string, number> = { emergent: 0, urgent: 1, prompt: 2, routine: 3 };
    result.sort((a, b) => {
      const urgDiff = (urgencyOrder[a.condition.urgency] ?? 3) - (urgencyOrder[b.condition.urgency] ?? 3);
      if (urgDiff !== 0) return urgDiff;
      return b.confidence - a.confidence;
    });

    return result;
  }, [findings, edgeCases]);

  const toggleCard = (className: string) => {
    setExpandedCards(prev => ({
      ...prev,
      [className]: !prev[className],
    }));
  };

  // Don't render if no edge cases flagged
  if (flagged.length === 0) return null;

  const emergentCount = flagged.filter(f => f.condition.urgency === 'emergent').length;

  return (
    <div className="edge-spotlight card" id="edge-case-spotlight">
      {/* Header with pulsing indicator */}
      <div className="edge-spotlight-header">
        <div className="edge-spotlight-header-left">
          <span className={`edge-spotlight-pulse ${emergentCount > 0 ? 'edge-spotlight-pulse--emergent' : 'edge-spotlight-pulse--active'}`} />
          <h3 className="edge-spotlight-title">
            <span className="edge-spotlight-icon">⚡</span>
            Edge-Case Spotlight
          </h3>
          <span className="edge-spotlight-count">
            {flagged.length} flagged
          </span>
        </div>
        {emergentCount > 0 && (
          <span className="edge-spotlight-emergent-badge" id="edge-spotlight-emergent-badge">
            🚨 {emergentCount} Emergent
          </span>
        )}
      </div>

      {/* Description */}
      <p className="edge-spotlight-description">
        Rare but clinically dangerous findings detected with moderate-to-high confidence.
        These conditions have low prevalence and are easy to miss — review carefully.
      </p>

      {/* Flagged conditions list */}
      <div className="edge-spotlight-list" id="edge-spotlight-list">
        {flagged.map((item, idx) => {
          const explanation = EXPLANATION_CARDS[item.condition.className];
          const isExpanded = expandedCards[item.condition.className] ?? false;

          return (
            <div
              key={item.condition.className}
              className={`edge-spotlight-item edge-spotlight-item--${item.condition.urgency} ${item.isNew ? 'edge-spotlight-item--new' : ''}`}
              id={`edge-case-${idx}`}
            >
              {/* Main content — clickable */}
              <button
                className="edge-spotlight-item-main"
                onClick={() => onEdgeCaseClick?.(item.condition.className, item.task)}
                title={`Click to view ${item.condition.displayName} on waveform`}
              >
                {/* Urgency indicator */}
                <div className={`edge-spotlight-urgency-indicator edge-spotlight-urgency-indicator--${item.condition.urgency}`}>
                  {item.condition.urgency === 'emergent' ? '🔴' :
                   item.condition.urgency === 'urgent' ? '🟠' :
                   item.condition.urgency === 'prompt' ? '🟡' : '🟢'}
                </div>

                {/* Condition details */}
                <div className="edge-spotlight-item-body">
                  <div className="edge-spotlight-item-top">
                    <span className="edge-spotlight-item-name">{item.condition.displayName}</span>
                    <span className={`edge-spotlight-urgency-tag edge-spotlight-urgency-tag--${item.condition.urgency}`}>
                      {item.condition.urgency.charAt(0).toUpperCase() + item.condition.urgency.slice(1)}
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div className="edge-spotlight-confidence-row">
                    <div className="edge-spotlight-confidence-bg">
                      <div
                        className={`edge-spotlight-confidence-fill edge-spotlight-confidence-fill--${item.condition.urgency}`}
                        style={{ width: `${item.confidence * 100}%` }}
                      />
                    </div>
                    <span className="edge-spotlight-confidence-pct">
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Prevalence note */}
                  <p className="edge-spotlight-prevalence">
                    {item.condition.prevalenceNote}
                  </p>
                </div>

                {/* New badge */}
                {item.isNew && (
                  <span className="edge-spotlight-new-badge">NEW</span>
                )}
              </button>

              {/* Explanation card toggle */}
              {explanation && (
                <>
                  <button
                    className={`edge-spotlight-expand-btn ${isExpanded ? 'edge-spotlight-expand-btn--open' : ''}`}
                    onClick={() => toggleCard(item.condition.className)}
                    aria-expanded={isExpanded}
                    aria-controls={`explanation-card-${idx}`}
                    id={`edge-case-expand-${idx}`}
                    title={isExpanded ? 'Collapse explanation' : 'Show explanation card'}
                  >
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M7 3v8M3 7h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    <span>{isExpanded ? 'Hide Details' : 'Explanation Card'}</span>
                  </button>

                  {isExpanded && (
                    <div
                      className="edge-spotlight-explanation"
                      id={`explanation-card-${idx}`}
                    >
                      <div className="explanation-section">
                        <h4 className="explanation-heading">Clinical Background</h4>
                        <p className="explanation-text">{explanation.clinicalBackground}</p>
                      </div>
                      <div className="explanation-section">
                        <h4 className="explanation-heading">Key ECG Features</h4>
                        <ul className="explanation-features">
                          {explanation.keyECGFeatures.map((feat, fi) => (
                            <li key={fi} className="explanation-feature">
                              <span className="explanation-feature-dot" />
                              {feat}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="explanation-section">
                        <h4 className="explanation-heading">Recommended Action</h4>
                        <p className="explanation-text explanation-text--action">
                          {explanation.recommendedAction}
                        </p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
