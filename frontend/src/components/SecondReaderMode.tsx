import { useState, useMemo, useCallback } from 'react';
import './SecondReaderMode.css';

/* ---------- Types -------------------------------------------------------- */

interface DiscrepancyItem {
  class_name: string;
  task: string;
  status: 'agreement' | 'ai_only' | 'clinician_only';
  ai_probability: number | null;
  clinician_selected: boolean;
  clinical_importance: number;
}

interface CompareResponse {
  agreements: DiscrepancyItem[];
  ai_only: DiscrepancyItem[];
  clinician_only: DiscrepancyItem[];
  summary: {
    total_agreements: number;
    total_ai_only: number;
    total_clinician_only: number;
  };
  unmatched_clinician_inputs: string[];
}

interface SecondReaderModeProps {
  /** AI predictions per task, as returned by the predict API */
  aiPredictions: Record<string, number[]>;
  /** Whether the component is visible */
  visible: boolean;
  /** Toggle visibility */
  onToggle: () => void;
}

/* ---------- Constants ---------------------------------------------------- */

const FINDING_CATEGORIES = {
  rhythm: {
    label: 'Rhythm & Conduction',
    icon: '♥',
    items: [
      { name: 'AF', label: 'Atrial Fibrillation' },
      { name: 'AFL', label: 'Atrial Flutter' },
      { name: 'SVT', label: 'Supraventricular Tachycardia' },
      { name: 'VT', label: 'Ventricular Tachycardia' },
      { name: 'VF', label: 'Ventricular Fibrillation' },
      { name: 'sinus_brady', label: 'Sinus Bradycardia' },
      { name: 'sinus_tachy', label: 'Sinus Tachycardia' },
      { name: 'PAC', label: 'Premature Atrial Complex' },
      { name: 'PVC', label: 'Premature Ventricular Complex' },
      { name: 'av_block_1st', label: '1st Degree AV Block' },
      { name: 'av_block_2nd', label: '2nd Degree AV Block' },
      { name: 'av_block_3rd', label: '3rd Degree AV Block' },
      { name: 'LBBB', label: 'Left Bundle Branch Block' },
      { name: 'RBBB', label: 'Right Bundle Branch Block' },
      { name: 'WPW', label: 'Wolff-Parkinson-White' },
      { name: 'AVNRT', label: 'AV Nodal Reentrant Tachycardia' },
      { name: 'AVRT', label: 'AV Reentrant Tachycardia' },
      { name: 'LAFB', label: 'Left Anterior Fascicular Block' },
      { name: 'LPFB', label: 'Left Posterior Fascicular Block' },
      { name: 'idioventricular', label: 'Idioventricular Rhythm' },
      { name: 'pacemaker_rhythm', label: 'Pacemaker Rhythm' },
      { name: 'normal_sinus_rhythm', label: 'Normal Sinus Rhythm' },
    ],
  },
  structural: {
    label: 'Structural & Functional',
    icon: '◇',
    items: [
      { name: 'LVH', label: 'Left Ventricular Hypertrophy' },
      { name: 'RVH', label: 'Right Ventricular Hypertrophy' },
      { name: 'LVSD', label: 'LV Systolic Dysfunction' },
      { name: 'HFpEF_risk', label: 'HFpEF Risk' },
      { name: 'DCM', label: 'Dilated Cardiomyopathy' },
      { name: 'HCM', label: 'Hypertrophic Cardiomyopathy' },
      { name: 'ARVC', label: 'ARVC' },
      { name: 'amyloidosis', label: 'Cardiac Amyloidosis' },
      { name: 'aortic_stenosis', label: 'Aortic Stenosis' },
      { name: 'mitral_regurgitation', label: 'Mitral Regurgitation' },
      { name: 'pulmonary_HTN', label: 'Pulmonary Hypertension' },
      { name: 'LA_enlargement', label: 'Left Atrial Enlargement' },
      { name: 'RA_enlargement', label: 'Right Atrial Enlargement' },
      { name: 'pericarditis', label: 'Pericarditis' },
      { name: 'myocarditis', label: 'Myocarditis' },
    ],
  },
  ischaemia: {
    label: 'Ischaemia & Metabolic',
    icon: '△',
    items: [
      { name: 'STEMI', label: 'STEMI' },
      { name: 'posterior_MI', label: 'Posterior MI' },
      { name: 'occlusive_NSTEMI', label: 'Occlusive NSTEMI' },
      { name: 'old_MI', label: 'Old MI' },
      { name: 'hyperkalaemia', label: 'Hyperkalaemia' },
      { name: 'hypokalaemia', label: 'Hypokalaemia' },
      { name: 'hypercalcaemia', label: 'Hypercalcaemia' },
      { name: 'hypothyroidism_pattern', label: 'Hypothyroidism Pattern' },
      { name: 'digitalis_effect', label: 'Digitalis Effect' },
      { name: 'QTc_prolongation', label: 'QTc Prolongation' },
    ],
  },
};

/* ---------- Local comparison engine (offline-capable) -------------------- */

/**
 * Perform comparison locally (for offline/WASM mode or when server
 * isn't needed). Mirrors the backend logic.
 */
function compareLocally(
  selectedFindings: Set<string>,
  aiPredictions: Record<string, number[]>,
  threshold: number,
): CompareResponse {
  const allClasses: Record<string, string[]> = {
    rhythm: FINDING_CATEGORIES.rhythm.items.map(i => i.name),
    structural: FINDING_CATEGORIES.structural.items.map(i => i.name),
    ischaemia: FINDING_CATEGORIES.ischaemia.items.map(i => i.name),
  };

  // Build AI positive set
  const aiPositive = new Map<string, { prob: number; task: string }>();
  const aiAll = new Map<string, { prob: number; task: string }>();

  for (const [task, classes] of Object.entries(allClasses)) {
    const probs = aiPredictions[task] || [];
    classes.forEach((className, i) => {
      const prob = probs[i] ?? 0;
      aiAll.set(className, { prob, task });
      if (prob >= threshold) {
        aiPositive.set(className, { prob, task });
      }
    });
  }

  const agreements: DiscrepancyItem[] = [];
  const aiOnly: DiscrepancyItem[] = [];
  const clinicianOnly: DiscrepancyItem[] = [];

  // Agreements
  for (const className of selectedFindings) {
    if (aiPositive.has(className)) {
      const info = aiPositive.get(className)!;
      agreements.push({
        class_name: className,
        task: info.task,
        status: 'agreement',
        ai_probability: info.prob,
        clinician_selected: true,
        clinical_importance: getClinicalImportance(className),
      });
    }
  }

  // AI only
  for (const [className, info] of aiPositive) {
    if (!selectedFindings.has(className)) {
      aiOnly.push({
        class_name: className,
        task: info.task,
        status: 'ai_only',
        ai_probability: info.prob,
        clinician_selected: false,
        clinical_importance: getClinicalImportance(className),
      });
    }
  }

  // Clinician only
  for (const className of selectedFindings) {
    if (!aiPositive.has(className)) {
      const info = aiAll.get(className);
      clinicianOnly.push({
        class_name: className,
        task: info?.task || 'unknown',
        status: 'clinician_only',
        ai_probability: info?.prob ?? null,
        clinician_selected: true,
        clinical_importance: getClinicalImportance(className),
      });
    }
  }

  // Sort by importance
  const byImportance = (a: DiscrepancyItem, b: DiscrepancyItem) =>
    b.clinical_importance - a.clinical_importance;

  agreements.sort(byImportance);
  aiOnly.sort(byImportance);
  clinicianOnly.sort(byImportance);

  return {
    agreements,
    ai_only: aiOnly,
    clinician_only: clinicianOnly,
    summary: {
      total_agreements: agreements.length,
      total_ai_only: aiOnly.length,
      total_clinician_only: clinicianOnly.length,
    },
    unmatched_clinician_inputs: [],
  };
}

const IMPORTANCE_MAP: Record<string, number> = {
  VF: 10, STEMI: 10, VT: 9, av_block_3rd: 9, posterior_MI: 9,
  occlusive_NSTEMI: 9, WPW: 8, LVSD: 8, hyperkalaemia: 8,
  av_block_2nd: 7, AVRT: 7, AVNRT: 7, HCM: 7, ARVC: 7, DCM: 7,
  amyloidosis: 7, myocarditis: 7, QTc_prolongation: 7,
  AF: 6, AFL: 6, idioventricular: 6, aortic_stenosis: 6,
  pulmonary_HTN: 6, hypokalaemia: 6, hypercalcaemia: 6,
  SVT: 5, LBBB: 5, LVH: 5, RVH: 5, mitral_regurgitation: 5,
  HFpEF_risk: 5, pericarditis: 5, old_MI: 5,
  RBBB: 4, av_block_1st: 4, LA_enlargement: 4, RA_enlargement: 4,
  digitalis_effect: 4,
  LAFB: 3, LPFB: 3, sinus_tachy: 3, sinus_brady: 3,
  hypothyroidism_pattern: 3,
  PAC: 2, PVC: 2, pacemaker_rhythm: 2,
  normal_sinus_rhythm: 1,
};

function getClinicalImportance(className: string): number {
  return IMPORTANCE_MAP[className] ?? 3;
}

function getHumanLabel(className: string): string {
  for (const cat of Object.values(FINDING_CATEGORIES)) {
    for (const item of cat.items) {
      if (item.name === className) return item.label;
    }
  }
  return className;
}

/* ---------- Sub-components ----------------------------------------------- */

function ImportanceBadge({ importance }: { importance: number }) {
  let level: string;
  if (importance >= 8) level = 'critical';
  else if (importance >= 5) level = 'high';
  else if (importance >= 3) level = 'moderate';
  else level = 'low';

  return (
    <span className={`sr-importance sr-importance--${level}`} title={`Clinical importance: ${importance}/10`}>
      {importance}/10
    </span>
  );
}

function DiscrepancyRow({ item }: { item: DiscrepancyItem }) {
  return (
    <div className={`sr-discrepancy-row sr-discrepancy-row--${item.status}`} id={`sr-item-${item.class_name}`}>
      <div className="sr-discrepancy-indicator">
        {item.status === 'agreement' && <span className="sr-status-dot sr-status-dot--green" title="Agreement" />}
        {item.status === 'ai_only' && <span className="sr-status-dot sr-status-dot--red" title="AI found, clinician missed" />}
        {item.status === 'clinician_only' && <span className="sr-status-dot sr-status-dot--yellow" title="Clinician found, AI didn't" />}
      </div>
      <div className="sr-discrepancy-info">
        <span className="sr-discrepancy-name">{getHumanLabel(item.class_name)}</span>
        <span className="sr-discrepancy-task">{item.task}</span>
      </div>
      {item.ai_probability !== null && (
        <div className="sr-discrepancy-prob">
          <div className="sr-prob-bar-bg">
            <div
              className={`sr-prob-bar sr-prob-bar--${item.status}`}
              style={{ width: `${(item.ai_probability ?? 0) * 100}%` }}
            />
          </div>
          <span className="sr-prob-value">{((item.ai_probability ?? 0) * 100).toFixed(0)}%</span>
        </div>
      )}
      <ImportanceBadge importance={item.clinical_importance} />
    </div>
  );
}

/* ---------- Main component ----------------------------------------------- */

export function SecondReaderMode({ aiPredictions, visible, onToggle }: SecondReaderModeProps) {
  const [selectedFindings, setSelectedFindings] = useState<Set<string>>(new Set());
  const [freeText, setFreeText] = useState('');
  const [comparisonResult, setComparisonResult] = useState<CompareResponse | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['rhythm']));

  const toggleFinding = useCallback((name: string) => {
    setSelectedFindings(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
    // Clear previous comparison when input changes
    setComparisonResult(null);
  }, []);

  const toggleCategory = useCallback((cat: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }, []);

  const handleCompare = useCallback(() => {
    const result = compareLocally(selectedFindings, aiPredictions, 0.50);
    setComparisonResult(result);
  }, [selectedFindings, aiPredictions]);

  const handleClear = useCallback(() => {
    setSelectedFindings(new Set());
    setFreeText('');
    setComparisonResult(null);
  }, []);

  const totalSelected = selectedFindings.size;

  const totalDiscrepancies = useMemo(() => {
    if (!comparisonResult) return 0;
    return comparisonResult.summary.total_ai_only + comparisonResult.summary.total_clinician_only;
  }, [comparisonResult]);

  if (!visible) return null;

  return (
    <div className="second-reader card" id="second-reader-panel">
      {/* Header */}
      <div className="sr-header">
        <div className="sr-header-left">
          <span className="sr-icon">🔍</span>
          <h3 className="sr-title">Second Reader Mode</h3>
          <span className="sr-subtitle">Compare your interpretation against AI</span>
        </div>
        <button className="sr-close" onClick={onToggle} title="Close Second Reader">✕</button>
      </div>

      <div className="sr-body">
        {/* Input section */}
        <div className="sr-input-section" id="sr-input-section">
          <div className="sr-input-header">
            <h4>Your Interpretation</h4>
            <span className="sr-selection-count">{totalSelected} selected</span>
          </div>

          {/* Finding checkboxes by category */}
          {Object.entries(FINDING_CATEGORIES).map(([key, cat]) => (
            <div key={key} className="sr-category" id={`sr-cat-${key}`}>
              <button
                className="sr-category-header"
                onClick={() => toggleCategory(key)}
                aria-expanded={expandedCategories.has(key)}
              >
                <span className="sr-category-icon">{cat.icon}</span>
                <span className="sr-category-label">{cat.label}</span>
                <span className={`sr-category-chevron ${expandedCategories.has(key) ? 'sr-category-chevron--open' : ''}`}>
                  ▾
                </span>
              </button>
              {expandedCategories.has(key) && (
                <div className="sr-category-items">
                  {cat.items.map(item => (
                    <label
                      key={item.name}
                      className={`sr-checkbox-item ${selectedFindings.has(item.name) ? 'sr-checkbox-item--selected' : ''}`}
                      id={`sr-check-${item.name}`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedFindings.has(item.name)}
                        onChange={() => toggleFinding(item.name)}
                        className="sr-checkbox-input"
                      />
                      <span className="sr-checkbox-custom" />
                      <span className="sr-checkbox-label">{item.label}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Free-text */}
          <div className="sr-freetext">
            <label className="sr-freetext-label" htmlFor="sr-freetext-input">Additional Notes</label>
            <textarea
              id="sr-freetext-input"
              className="sr-freetext-input"
              value={freeText}
              onChange={e => setFreeText(e.target.value)}
              placeholder="Enter any additional observations..."
              rows={3}
            />
          </div>

          {/* Action buttons */}
          <div className="sr-actions">
            <button
              className="sr-compare-btn"
              onClick={handleCompare}
              disabled={totalSelected === 0}
              id="sr-compare-btn"
            >
              Compare Interpretations
            </button>
            <button className="sr-clear-btn" onClick={handleClear} id="sr-clear-btn">
              Clear
            </button>
          </div>
        </div>

        {/* Results section */}
        {comparisonResult && (
          <div className="sr-results-section" id="sr-results-section">
            <div className="sr-results-header">
              <h4>Comparison Results</h4>
              {totalDiscrepancies > 0 && (
                <span className="sr-discrepancy-count">{totalDiscrepancies} discrepancies found</span>
              )}
            </div>

            {/* Summary badges */}
            <div className="sr-summary">
              <div className="sr-summary-badge sr-summary-badge--green" id="sr-summary-agreements">
                <span className="sr-summary-count">{comparisonResult.summary.total_agreements}</span>
                <span className="sr-summary-label">Agreements</span>
              </div>
              <div className="sr-summary-badge sr-summary-badge--red" id="sr-summary-ai-only">
                <span className="sr-summary-count">{comparisonResult.summary.total_ai_only}</span>
                <span className="sr-summary-label">AI Found Only</span>
              </div>
              <div className="sr-summary-badge sr-summary-badge--yellow" id="sr-summary-clinician-only">
                <span className="sr-summary-count">{comparisonResult.summary.total_clinician_only}</span>
                <span className="sr-summary-label">You Found Only</span>
              </div>
            </div>

            {/* Legend */}
            <div className="sr-legend">
              <span className="sr-legend-item"><span className="sr-legend-dot sr-legend-dot--green" /> Agreement</span>
              <span className="sr-legend-item"><span className="sr-legend-dot sr-legend-dot--red" /> AI found, you missed</span>
              <span className="sr-legend-item"><span className="sr-legend-dot sr-legend-dot--yellow" /> You found, AI didn't</span>
            </div>

            {/* Agreements */}
            {comparisonResult.agreements.length > 0 && (
              <div className="sr-result-group" id="sr-agreements">
                <h5 className="sr-group-title sr-group-title--green">✓ Agreements</h5>
                {comparisonResult.agreements.map(item => (
                  <DiscrepancyRow key={item.class_name} item={item} />
                ))}
              </div>
            )}

            {/* AI Only — highlighted prominently */}
            {comparisonResult.ai_only.length > 0 && (
              <div className="sr-result-group sr-result-group--alert" id="sr-ai-only">
                <h5 className="sr-group-title sr-group-title--red">⚠ AI Detected — Review These</h5>
                <p className="sr-group-subtitle">The AI found these but you did not select them. Consider reviewing.</p>
                {comparisonResult.ai_only.map(item => (
                  <DiscrepancyRow key={item.class_name} item={item} />
                ))}
              </div>
            )}

            {/* Clinician Only */}
            {comparisonResult.clinician_only.length > 0 && (
              <div className="sr-result-group" id="sr-clinician-only">
                <h5 className="sr-group-title sr-group-title--yellow">◉ Your Additional Findings</h5>
                <p className="sr-group-subtitle">You selected these but the AI did not flag them.</p>
                {comparisonResult.clinician_only.map(item => (
                  <DiscrepancyRow key={item.class_name} item={item} />
                ))}
              </div>
            )}

            {/* No discrepancies */}
            {totalDiscrepancies === 0 && comparisonResult.agreements.length > 0 && (
              <div className="sr-no-discrepancies" id="sr-no-discrepancies">
                <span className="sr-no-disc-icon">✓</span>
                <span>Perfect agreement — your interpretation matches the AI.</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
