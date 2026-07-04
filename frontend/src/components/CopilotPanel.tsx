import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import './CopilotPanel.css';

/* ── Types ──────────────────────────────────────────── */

export interface CopilotFinding {
  name: string;
  confidence: number;          // 0–1
  task: string;                // 'rhythm' | 'structural' | 'ischaemia'
  severity: 'critical' | 'warning' | 'info';
  predictionSetSize?: number;
}

export interface ClinicalSuggestion {
  prompt: string;
  urgency: 'routine' | 'prompt' | 'urgent' | 'emergent';
}

/* ── Clinical suggestion map ─────────────────────────── */

/**
 * Non-prescriptive clinical cues sourced from standard cardiology practice guides.
 * These are surfaced as decision-support prompts only — they are NOT treatment orders.
 */
const CONDITION_SUGGESTIONS: Record<string, ClinicalSuggestion> = {
  /* Rhythm & Conduction — Critical */
  'AF':                  { prompt: 'Evaluate stroke risk (CHA₂DS₂-VASc) and consider anticoagulation', urgency: 'prompt' },
  'Atrial Fibrillation': { prompt: 'Evaluate stroke risk (CHA₂DS₂-VASc) and consider anticoagulation', urgency: 'prompt' },
  'AFL':                 { prompt: 'Cardiology referral recommended for rate/rhythm control', urgency: 'prompt' },
  'SVT':                 { prompt: 'If symptomatic, consider vagal manoeuvres or adenosine', urgency: 'prompt' },
  'AVNRT':               { prompt: 'Electrophysiology referral for ablation consideration', urgency: 'prompt' },
  'AVRT':                { prompt: 'Electrophysiology referral — assess for accessory pathway', urgency: 'prompt' },
  'VT':                  { prompt: 'Urgent cardiology review — assess hemodynamic stability', urgency: 'emergent' },
  'VF':                  { prompt: 'Immediate defibrillation and ACLS protocol', urgency: 'emergent' },
  'Idioventricular':     { prompt: 'Monitor — usually self-limiting post-reperfusion', urgency: 'routine' },
  'Sinus Brady':         { prompt: 'Assess for symptoms; consider reversible causes', urgency: 'routine' },
  'Sinus Tachy':         { prompt: 'Evaluate underlying cause (fever, pain, volume status)', urgency: 'routine' },
  'Sinus Tachycardia':   { prompt: 'Evaluate underlying cause (fever, pain, volume status)', urgency: 'routine' },
  'PAC':                 { prompt: 'Usually benign — reassurance unless highly symptomatic', urgency: 'routine' },
  'PVC':                 { prompt: 'If frequent (>10%), consider echocardiography', urgency: 'routine' },
  '1st AVB':             { prompt: 'Monitor — usually benign unless progressive', urgency: 'routine' },
  '2nd AVB':             { prompt: 'Differentiate Mobitz I vs II — cardiology input if Mobitz II', urgency: 'prompt' },
  '3rd AVB':             { prompt: 'Urgent cardiology — temporary pacing may be required', urgency: 'emergent' },
  'LBBB':                { prompt: 'New LBBB warrants acute coronary syndrome workup', urgency: 'urgent' },
  'RBBB':                { prompt: 'Evaluate for right heart strain if new onset', urgency: 'routine' },
  'LAFB':                { prompt: 'Usually benign — note for trend monitoring', urgency: 'routine' },
  'LPFB':                { prompt: 'Rare — consider structural heart disease evaluation', urgency: 'prompt' },
  'WPW':                 { prompt: 'Electrophysiology referral — risk stratification for SCD', urgency: 'urgent' },
  'Pacemaker':           { prompt: 'Verify appropriate pacing capture and sensing', urgency: 'routine' },
  'Normal Sinus':        { prompt: 'Normal sinus rhythm — no specific action needed', urgency: 'routine' },
  'Normal Sinus Rhythm': { prompt: 'Normal sinus rhythm — no specific action needed', urgency: 'routine' },

  /* Structural & Functional */
  'LVH':                       { prompt: 'Blood pressure optimisation and echo assessment', urgency: 'prompt' },
  'Left Ventricular Hypertrophy': { prompt: 'Blood pressure optimisation and echo assessment', urgency: 'prompt' },
  'RVH':                       { prompt: 'Evaluate for pulmonary hypertension or right heart strain', urgency: 'prompt' },
  'LVSD':                      { prompt: 'Echocardiography and HF workup recommended', urgency: 'urgent' },
  'HFpEF Risk':                { prompt: 'NT-proBNP and exercise testing may clarify diagnosis', urgency: 'prompt' },
  'DCM':                       { prompt: 'Cardiology referral for comprehensive imaging', urgency: 'prompt' },
  'Dilated Cardiomyopathy':    { prompt: 'Cardiology referral for comprehensive imaging', urgency: 'prompt' },
  'HCM':                       { prompt: 'Family screening and SCD risk assessment', urgency: 'urgent' },
  'ARVC':                      { prompt: 'Cardiac MRI and genetic testing recommended', urgency: 'urgent' },
  'Amyloidosis':               { prompt: 'Consider Tc-PYP scan and haematology referral', urgency: 'urgent' },
  'Aortic Stenosis':           { prompt: 'Echocardiography for valve assessment', urgency: 'prompt' },
  'Mitral Regurgitation':      { prompt: 'Echo assessment for severity grading', urgency: 'prompt' },
  'Pulmonary HTN':             { prompt: 'Right heart catheterisation may be needed', urgency: 'prompt' },
  'LA Enlargement':            { prompt: 'Evaluate atrial fibrillation risk and valvular disease', urgency: 'routine' },
  'RA Enlargement':            { prompt: 'Assess for right-sided volume/pressure overload', urgency: 'routine' },
  'Pericarditis':              { prompt: 'NSAIDs ± colchicine; rule out effusion', urgency: 'prompt' },
  'Pericarditis Pattern':      { prompt: 'NSAIDs ± colchicine; rule out effusion', urgency: 'prompt' },
  'Myocarditis':               { prompt: 'Cardiac MRI and troponin trend recommended', urgency: 'urgent' },
  'Myocarditis Pattern':       { prompt: 'Cardiac MRI and troponin trend recommended', urgency: 'urgent' },

  /* Ischaemia & Metabolic */
  'STEMI':                      { prompt: 'Activate cath lab — door-to-balloon time critical', urgency: 'emergent' },
  'ST Elevation (Lateral)':     { prompt: 'Activate cath lab — door-to-balloon time critical', urgency: 'emergent' },
  'Posterior MI':               { prompt: 'Obtain posterior leads (V7-V9); emergent cath consideration', urgency: 'emergent' },
  'Occlusive NSTEMI':           { prompt: 'Serial troponins and urgent cardiology consultation', urgency: 'urgent' },
  'Old MI':                     { prompt: 'Verify with prior ECGs; optimise secondary prevention', urgency: 'routine' },
  'Hyperkalaemia':              { prompt: 'Urgent electrolytes — calcium gluconate if severe', urgency: 'emergent' },
  'Hypokalaemia':               { prompt: 'Electrolytes warrant checking and potassium replacement', urgency: 'prompt' },
  'Hypercalcaemia':             { prompt: 'Check corrected calcium and parathyroid hormone', urgency: 'prompt' },
  'Hypothyroidism':             { prompt: 'Thyroid function tests recommended', urgency: 'routine' },
  'Hypothyroidism Pattern':     { prompt: 'Thyroid function tests recommended', urgency: 'routine' },
  'Digitalis':                  { prompt: 'Check digoxin level — assess for toxicity signs', urgency: 'prompt' },
  'Digitalis Effect':           { prompt: 'Check digoxin level — assess for toxicity signs', urgency: 'prompt' },
  'QTc Prolongation':           { prompt: 'Review QT-prolonging medications; electrolytes check', urgency: 'urgent' },
};

/**
 * Look up a clinical suggestion for a condition name.
 * Falls back to a generic suggestion if no match is found.
 */
function getSuggestion(conditionName: string): ClinicalSuggestion | null {
  // Direct lookup
  if (CONDITION_SUGGESTIONS[conditionName]) {
    return CONDITION_SUGGESTIONS[conditionName];
  }
  // Partial match (for names like "ST Elevation (Lateral)" matching "STEMI")
  for (const [key, val] of Object.entries(CONDITION_SUGGESTIONS)) {
    if (conditionName.toLowerCase().includes(key.toLowerCase()) ||
        key.toLowerCase().includes(conditionName.toLowerCase())) {
      return val;
    }
  }
  return null;
}

/* ── Severity classification ──────────────────────── */

function classifyCopilotSeverity(
  confidence: number,
  urgency: string | undefined,
): 'critical' | 'warning' | 'info' {
  // ≥90% on emergent/urgent conditions = critical
  if (confidence >= 0.90 && (urgency === 'emergent' || urgency === 'urgent')) return 'critical';
  // ≥80% on any condition = critical
  if (confidence >= 0.90) return 'critical';
  // ≥50% = warning
  if (confidence >= 0.50) return 'warning';
  return 'info';
}

/* ── Component props ─────────────────────────────────── */

interface CopilotPanelProps {
  /** All findings from classification task heads (rhythm, structural, ischaemia) */
  findings: { name: string; prob: number; task: string; predictionSetSize?: number }[];
  /** Threshold for showing findings — default 0.30 (30%) */
  confidenceThreshold?: number;
  /** Callback when a finding is clicked — used to scroll waveform + activate XAI */
  onFindingClick?: (findingName: string, task: string) => void;
  /** Currently active/selected finding */
  activeFinding?: string | null;
}

/* ── Main component ───────────────────────────────────── */

export function CopilotPanel({
  findings,
  confidenceThreshold = 0.30,
  onFindingClick,
  activeFinding,
}: CopilotPanelProps) {
  const { t } = useTranslation();
  // Localized condition display name, falling back to the raw name from the model.
  const conditionName = (name: string) => t(`conditions.${name}`, { defaultValue: name });
  // Localized clinical suggestion text, falling back to the guideline default.
  const suggestionText = (name: string, fallback: string) =>
    t(`suggestions.${name}`, { defaultValue: fallback });

  // Filter positive findings above threshold and rank by confidence
  const posFindings = useMemo(() => {
    return findings
      .filter(f => f.prob >= confidenceThreshold)
      .sort((a, b) => b.prob - a.prob)
      .map(f => {
        const suggestion = getSuggestion(f.name);
        const severity = classifyCopilotSeverity(f.prob, suggestion?.urgency);
        return {
          ...f,
          confidence: f.prob,
          severity,
          suggestion,
        };
      });
  }, [findings, confidenceThreshold]);

  const criticalFindings = posFindings.filter(f => f.severity === 'critical');
  const hasCritical = criticalFindings.length > 0;

  if (posFindings.length === 0) {
    return (
      <div className="copilot-panel card" id="copilot-panel">
        <div className="copilot-header">
          <div className="copilot-header-title">
            <span className="copilot-icon">🤖</span>
            <h3>{t('copilot.title')}</h3>
          </div>
        </div>
        <div className="copilot-empty" id="copilot-empty-state">
          <div className="copilot-empty-icon">✓</div>
          <p className="copilot-empty-text">{t('copilot.noFindings')}</p>
          <p className="copilot-empty-sub">
            {t('copilot.noFindingsSub')}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`copilot-panel card ${hasCritical ? 'copilot-panel--critical' : ''}`} id="copilot-panel">
      {/* Header */}
      <div className="copilot-header">
        <div className="copilot-header-title">
          <span className="copilot-icon">🤖</span>
          <h3>{t('copilot.title')}</h3>
          <span className="copilot-count">{t('copilot.findingsCount', { count: posFindings.length })}</span>
        </div>
        {hasCritical && (
          <span className="copilot-critical-badge" id="copilot-critical-badge">
            {t('copilot.criticalCount', { count: criticalFindings.length })}
          </span>
        )}
      </div>

      {/* Disclaimer */}
      <div className="copilot-disclaimer" id="copilot-disclaimer">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.2" />
          <path d="M7 4v3.5M7 9.5v.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
        {t('copilot.disclaimer')}
      </div>

      {/* Findings list */}
      <div className="copilot-findings" id="copilot-findings-list">
        {posFindings.map((f, idx) => {
          const isActive = activeFinding === f.name;
          const isCritical = f.severity === 'critical';

          return (
            <button
              key={`${f.name}-${f.task}`}
              className={`copilot-finding ${isActive ? 'copilot-finding--active' : ''} ${isCritical ? 'copilot-finding--critical' : ''}`}
              onClick={() => onFindingClick?.(f.name, f.task)}
              id={`copilot-finding-${idx}`}
              title={t('copilot.viewOnWaveform', { name: conditionName(f.name) })}
            >
              {/* Rank + severity indicator */}
              <div className="copilot-finding-rank">
                <span className={`copilot-severity-dot copilot-severity-dot--${f.severity}`} />
                <span className="copilot-rank-num">#{idx + 1}</span>
              </div>

              {/* Condition info */}
              <div className="copilot-finding-body">
                <div className="copilot-finding-top">
                  <span className="copilot-finding-name">{conditionName(f.name)}</span>
                  <span className={`copilot-severity-badge copilot-severity-badge--${f.severity}`}>
                    {t(`copilot.${f.severity}`)}
                  </span>
                </div>

                {/* Confidence bar */}
                <div className="copilot-confidence-row">
                  <div className="copilot-confidence-bar-bg">
                    <div
                      className={`copilot-confidence-bar copilot-confidence-bar--${f.severity}`}
                      style={{ width: `${f.confidence * 100}%` }}
                    />
                  </div>
                  <span className="copilot-confidence-pct">
                    {(f.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Clinical suggestion */}
                {f.suggestion && (
                  <div className="copilot-suggestion">
                    <span className={`copilot-urgency-tag copilot-urgency-tag--${f.suggestion.urgency}`}>
                      {t(`copilot.urgency.${f.suggestion.urgency}`, { defaultValue: f.suggestion.urgency })}
                    </span>
                    <span className="copilot-suggestion-text">
                      {suggestionText(f.name, f.suggestion.prompt)}
                    </span>
                  </div>
                )}
              </div>

              {/* Action caret */}
              <div className="copilot-finding-action" aria-hidden="true">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
