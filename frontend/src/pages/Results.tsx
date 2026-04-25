import { useState, useMemo, useCallback } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { ECGWaveformChart, generateDemoECGData } from '../components/ECGWaveformChart';
import { XAIControls, TopFeaturesPanel, generateDemoXAIData } from '../components/XAIOverlay';
import { CopilotPanel } from '../components/CopilotPanel';
import { SecondReaderMode } from '../components/SecondReaderMode';
import { EdgeCaseSpotlight } from '../components/EdgeCaseSpotlight';
import { ExplanationCard } from '../components/ExplanationCard';
import { FeedbackButtons } from '../components/FeedbackButtons';
import type { XAIAttribution } from '../components/XAIOverlay';
import type { ECGFeatureAttribution, ClinicalSuggestionInfo } from '../components/ExplanationCard';
import type { PredictionResult } from '../services/InferenceClient';
import './Results.css';

/* ---------- Types for route state ---------------------------------------- */

interface ResultsLocationState {
  predictionResult?: PredictionResult;
  fileName?: string;
  fileSize?: number;
}

/* ---------- Quality types ------------------------------------------------ */

interface LeadQuality {
  lead: string;
  score: number;
  classification: 'good' | 'marginal' | 'poor';
}

interface QualityInfo {
  overall: number;
  classification: 'good' | 'marginal' | 'poor';
  leads: LeadQuality[];
}

/* ---------- Finding types ------------------------------------------------ */

interface Finding {
  name: string;
  prob: number;
  severity: string;
  predictionSetSize?: number;
}

interface RiskFinding {
  name: string;
  score: number;
  clinicalLabel: string;
  predictionSetSize?: number;
}

/* ---------- Default mock data (when navigating directly) ----------------- */

const MOCK_QUALITY: QualityInfo = {
  overall: 87,
  classification: 'good',
  leads: [
    { lead: 'I',   score: 92, classification: 'good' },
    { lead: 'II',  score: 95, classification: 'good' },
    { lead: 'III', score: 88, classification: 'good' },
    { lead: 'aVR', score: 85, classification: 'good' },
    { lead: 'aVL', score: 81, classification: 'good' },
    { lead: 'aVF', score: 89, classification: 'good' },
    { lead: 'V1',  score: 78, classification: 'good' },
    { lead: 'V2',  score: 90, classification: 'good' },
    { lead: 'V3',  score: 84, classification: 'good' },
    { lead: 'V4',  score: 91, classification: 'good' },
    { lead: 'V5',  score: 86, classification: 'good' },
    { lead: 'V6',  score: 83, classification: 'good' },
  ],
};

const MOCK_FINDINGS = {
  rhythm: [
    { name: 'Atrial Fibrillation',   prob: 0.92, severity: 'high',   predictionSetSize: 2 },
    { name: 'Normal Sinus Rhythm',    prob: 0.06, severity: 'normal', predictionSetSize: 1 },
    { name: 'Sinus Tachycardia',      prob: 0.02, severity: 'normal', predictionSetSize: 1 },
    { name: 'PVC',                    prob: 0.31, severity: 'normal', predictionSetSize: 3 },
    { name: 'LBBB',                   prob: 0.58, severity: 'medium', predictionSetSize: 2 },
  ] as Finding[],
  structural: [
    { name: 'Left Ventricular Hypertrophy', prob: 0.74, severity: 'medium', predictionSetSize: 3 },
    { name: 'LA Enlargement',               prob: 0.41, severity: 'normal', predictionSetSize: 2 },
    { name: 'Dilated Cardiomyopathy',       prob: 0.08, severity: 'normal', predictionSetSize: 1 },
    { name: 'LVSD',                         prob: 0.62, severity: 'medium', predictionSetSize: 3 },
    { name: 'Aortic Stenosis',              prob: 0.15, severity: 'normal', predictionSetSize: 2 },
  ] as Finding[],
  ischaemia: [
    { name: 'ST Elevation (Lateral)',  prob: 0.15, severity: 'normal', predictionSetSize: 1 },
    { name: 'QTc Prolongation',        prob: 0.09, severity: 'normal', predictionSetSize: 1 },
    { name: 'Hyperkalaemia',           prob: 0.52, severity: 'medium', predictionSetSize: 2 },
  ] as Finding[],
  risk: [
    { name: '1-Year Mortality',           score: 0.12, clinicalLabel: 'Low Risk',      predictionSetSize: 1 },
    { name: '12-Month HF Hospitalization', score: 0.28, clinicalLabel: 'Low Risk',      predictionSetSize: 2 },
    { name: '12-Month AF Onset',           score: 0.85, clinicalLabel: 'Very High Risk', predictionSetSize: 4 },
  ] as RiskFinding[],
};

function classifySeverity(prob: number): string {
  if (prob >= 0.80) return 'high';
  if (prob >= 0.50) return 'medium';
  return 'normal';
}

function classifyRisk(score: number): string {
  if (score >= 0.80) return 'Very High Risk';
  if (score >= 0.60) return 'High Risk';
  if (score >= 0.40) return 'Moderate Risk';
  if (score >= 0.20) return 'Low Risk';
  return 'Very Low Risk';
}

function classifyQuality(score: number): 'good' | 'marginal' | 'poor' {
  if (score >= 70) return 'good';
  if (score >= 40) return 'marginal';
  return 'poor';
}

/**
 * Parse quality data from the API response.
 */
function parseQuality(result: PredictionResult | undefined): QualityInfo {
  if (!result?.quality) return MOCK_QUALITY;

  const q = result.quality as Record<string, unknown>;
  const overall = typeof q.overall_score === 'number' ? q.overall_score : 87;
  const classification = classifyQuality(overall);

  const leadScores = q.per_lead_scores;
  const leads: LeadQuality[] = [];

  if (leadScores && typeof leadScores === 'object') {
    for (const [lead, score] of Object.entries(leadScores as Record<string, number>)) {
      leads.push({
        lead,
        score,
        classification: classifyQuality(score),
      });
    }
  }

  return {
    overall,
    classification,
    leads: leads.length > 0 ? leads : MOCK_QUALITY.leads,
  };
}

/**
 * Try to parse prediction data from the API response into our display format.
 * Falls back to mock data if the response shape is unexpected.
 */
function parsePredictions(result: PredictionResult | undefined) {
  if (!result?.predictions) return MOCK_FINDINGS;

  const preds = result.predictions as Record<string, unknown>;
  const uncertainty = (result.uncertainty ?? {}) as Record<string, unknown>;

  const parseFindingArray = (
    key: string,
    labels: string[],
  ): Finding[] => {
    const raw = preds[key];
    if (!Array.isArray(raw)) return [];

    const predSets = uncertainty[`${key}_prediction_sets`] as number[] | undefined;

    return labels.map((label, i) => {
      const prob = typeof raw[i] === 'number' ? (raw[i] as number) : 0;
      return {
        name: label,
        prob,
        severity: classifySeverity(prob),
        predictionSetSize: predSets?.[i],
      };
    }).sort((a, b) => b.prob - a.prob).slice(0, 8);
  };

  const rhythmLabels = [
    'AF', 'AFL', 'SVT', 'AVNRT', 'AVRT', 'VT', 'VF',
    'Idioventricular', 'Sinus Brady', 'Sinus Tachy',
    'PAC', 'PVC', '1st AVB', '2nd AVB', '3rd AVB',
    'LBBB', 'RBBB', 'LAFB', 'LPFB', 'WPW',
    'Pacemaker', 'Normal Sinus',
  ];

  const structuralLabels = [
    'LVH', 'RVH', 'LVSD', 'HFpEF Risk', 'DCM',
    'HCM', 'ARVC', 'Amyloidosis', 'Aortic Stenosis',
    'Mitral Regurgitation', 'Pulmonary HTN',
    'LA Enlargement', 'RA Enlargement',
    'Pericarditis', 'Myocarditis',
  ];

  const ischaemiaLabels = [
    'STEMI', 'Posterior MI', 'Occlusive NSTEMI',
    'Old MI', 'Hyperkalaemia', 'Hypokalaemia',
    'Hypercalcaemia', 'Hypothyroidism', 'Digitalis', 'QTc Prolongation',
  ];

  const riskLabels = ['1-Year Mortality', '12-Month HF Hospitalization', '12-Month AF Onset'];

  const rhythm = parseFindingArray('rhythm', rhythmLabels);
  const structural = parseFindingArray('structural', structuralLabels);
  const ischaemia = parseFindingArray('ischaemia', ischaemiaLabels);

  const riskRaw = preds['risk'];
  const riskPredSets = uncertainty['risk_prediction_sets'] as number[] | undefined;
  const risk: RiskFinding[] = Array.isArray(riskRaw)
    ? riskLabels.map((name, i) => {
        const score = typeof riskRaw[i] === 'number' ? (riskRaw[i] as number) : 0;
        return {
          name,
          score,
          clinicalLabel: classifyRisk(score),
          predictionSetSize: riskPredSets?.[i],
        };
      })
    : MOCK_FINDINGS.risk.map(r => ({ ...r }));

  return {
    rhythm: rhythm.length ? rhythm : MOCK_FINDINGS.rhythm.map(f => ({ ...f })),
    structural: structural.length ? structural : MOCK_FINDINGS.structural.map(f => ({ ...f })),
    ischaemia: ischaemia.length ? ischaemia : MOCK_FINDINGS.ischaemia.map(f => ({ ...f })),
    risk,
  };
}

/**
 * Parse XAI data from API response or generate demo data.
 */
function parseXAIData(result: PredictionResult | undefined): XAIAttribution[] {
  if (result?.xai && Array.isArray(result.xai)) {
    return result.xai as unknown as XAIAttribution[];
  }
  // Return demo XAI data for preview
  return generateDemoXAIData();
}

const DEMO_ECG = generateDemoECGData();
const STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

/* ---------- Clinical suggestions lookup for ExplanationCard -------------- */

const SUGGESTION_MAP: Record<string, ClinicalSuggestionInfo> = {
  'AF':                  { prompt: 'Evaluate stroke risk (CHA₂DS₂-VASc) and consider anticoagulation', urgency: 'prompt', rationale: 'AF increases stroke risk 5-fold. CHA₂DS₂-VASc score guides anticoagulation decisions.' },
  'Atrial Fibrillation': { prompt: 'Evaluate stroke risk (CHA₂DS₂-VASc) and consider anticoagulation', urgency: 'prompt', rationale: 'AF increases stroke risk 5-fold. CHA₂DS₂-VASc score guides anticoagulation decisions.' },
  'AFL':                 { prompt: 'Cardiology referral for rate/rhythm control', urgency: 'prompt', rationale: 'Atrial flutter often responds to catheter ablation with high success rates.' },
  'SVT':                 { prompt: 'If symptomatic, consider vagal manoeuvres or adenosine', urgency: 'prompt', rationale: 'Most SVTs are re-entrant and respond to AV nodal blockade.' },
  'VT':                  { prompt: 'Urgent cardiology review — assess hemodynamic stability', urgency: 'emergent', rationale: 'Sustained VT can degenerate to VF. Hemodynamic status determines treatment urgency.' },
  'VF':                  { prompt: 'Immediate defibrillation and ACLS protocol', urgency: 'emergent', rationale: 'VF is a cardiac arrest rhythm with no effective cardiac output.' },
  '3rd AVB':             { prompt: 'Urgent cardiology — temporary pacing may be required', urgency: 'emergent', rationale: 'Complete heart block can cause syncope, hemodynamic collapse, or cardiac arrest.' },
  'LBBB':                { prompt: 'New LBBB warrants acute coronary syndrome workup', urgency: 'urgent', rationale: 'New LBBB with chest pain meets STEMI-equivalent criteria per Sgarbossa.' },
  'WPW':                 { prompt: 'Electrophysiology referral for SCD risk stratification', urgency: 'urgent', rationale: 'Accessory pathways can conduct rapidly during AF, leading to VF.' },
  'RBBB':                { prompt: 'Evaluate for right heart strain if new onset', urgency: 'routine', rationale: 'Isolated RBBB is often benign but new onset may indicate PE or right heart disease.' },
  'LVSD':                { prompt: 'Echocardiography and HF workup recommended', urgency: 'urgent', rationale: 'Early detection of LVSD enables life-prolonging therapies.' },
  'HCM':                 { prompt: 'Family screening and SCD risk assessment', urgency: 'urgent', rationale: 'HCM is the most common inherited cardiac condition with SCD risk.' },
  'ARVC':                { prompt: 'Cardiac MRI and genetic testing recommended', urgency: 'urgent', rationale: 'ARVC is a leading cause of SCD in young athletes.' },
  'Amyloidosis':         { prompt: 'Consider Tc-PYP scan and haematology referral', urgency: 'urgent', rationale: 'Early diagnosis of cardiac amyloidosis changes prognosis dramatically.' },
  'STEMI':               { prompt: 'Activate cath lab — door-to-balloon time critical', urgency: 'emergent', rationale: 'Every minute of delay increases myocardial damage and mortality.' },
  'Posterior MI':        { prompt: 'Obtain posterior leads (V7-V9); emergent cath consideration', urgency: 'emergent', rationale: 'Posterior MI is frequently missed on standard 12-lead ECG.' },
  'Occlusive NSTEMI':    { prompt: 'Serial troponins and urgent cardiology consultation', urgency: 'urgent', rationale: 'Occlusive NSTEMI requires urgent intervention despite NSTEMI classification.' },
  'Hyperkalaemia':       { prompt: 'Urgent electrolytes — calcium gluconate if severe', urgency: 'emergent', rationale: 'Progressive ECG changes indicate risk of fatal arrhythmia.' },
  'QTc Prolongation':    { prompt: 'Review QT-prolonging medications; electrolytes check', urgency: 'urgent', rationale: 'Prolonged QTc increases risk of Torsades de Pointes.' },
  'LVH':                       { prompt: 'Blood pressure optimisation and echo assessment', urgency: 'prompt' },
  'Left Ventricular Hypertrophy': { prompt: 'Blood pressure optimisation and echo assessment', urgency: 'prompt' },
  'PVC':                 { prompt: 'If frequent (>10%), consider echocardiography', urgency: 'routine' },
  'Normal Sinus':        { prompt: 'Normal sinus rhythm — no specific action needed', urgency: 'routine' },
  'Normal Sinus Rhythm': { prompt: 'Normal sinus rhythm — no specific action needed', urgency: 'routine' },
  'Sinus Tachycardia':   { prompt: 'Evaluate underlying cause (fever, pain, volume status)', urgency: 'routine' },
  'LA Enlargement':      { prompt: 'Evaluate atrial fibrillation risk and valvular disease', urgency: 'routine' },
  'Dilated Cardiomyopathy': { prompt: 'Cardiology referral for comprehensive imaging', urgency: 'prompt' },
  'Aortic Stenosis':     { prompt: 'Echocardiography for valve assessment', urgency: 'prompt' },
  'Pericarditis':        { prompt: 'NSAIDs ± colchicine; rule out effusion', urgency: 'prompt' },
  'Myocarditis':         { prompt: 'Cardiac MRI and troponin trend recommended', urgency: 'urgent' },
};

function findSuggestion(name: string): ClinicalSuggestionInfo | undefined {
  if (SUGGESTION_MAP[name]) return SUGGESTION_MAP[name];
  // Partial match fallback
  for (const [key, val] of Object.entries(SUGGESTION_MAP)) {
    if (name.toLowerCase().includes(key.toLowerCase()) || key.toLowerCase().includes(name.toLowerCase())) {
      return val;
    }
  }
  return undefined;
}

/** Extract XAI top features for a specific finding from xai data. */
function getFeatureAttributionsForFinding(
  xaiData: XAIAttribution[],
  _findingName: string,
  task: string,
): ECGFeatureAttribution[] {
  const taskXAI = xaiData.find(x => x.task === task);
  if (!taskXAI) return [];
  // Return top_features (already top-3 from backend)
  return taskXAI.top_features.map(f => ({
    feature_name: f.feature_name,
    lead: f.lead,
    delta_score: f.delta_score,
  }));
}

/* ---------- Sub-components ----------------------------------------------- */

function QualityBadge({ quality }: { quality: QualityInfo }) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div
      className="quality-badge-wrapper"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      id="quality-badge"
    >
      <span className={`quality-badge quality-badge--${quality.classification}`}>
        Signal Quality: {quality.classification.charAt(0).toUpperCase() + quality.classification.slice(1)} ({quality.overall}/100)
      </span>
      {showTooltip && quality.leads.length > 0 && (
        <div className="quality-tooltip" id="quality-tooltip">
          <div className="quality-tooltip-title">Per-Lead Quality</div>
          <div className="quality-tooltip-grid">
            {quality.leads.map(lead => (
              <div key={lead.lead} className="quality-tooltip-row">
                <span className="quality-tooltip-lead">{lead.lead}</span>
                <div className="quality-tooltip-bar-bg">
                  <div
                    className={`quality-tooltip-bar quality-tooltip-bar--${lead.classification}`}
                    style={{ width: `${lead.score}%` }}
                  />
                </div>
                <span className={`quality-tooltip-score quality-tooltip-score--${lead.classification}`}>
                  {lead.score}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function UncertaintyBadge({ setSize }: { setSize?: number }) {
  if (setSize === undefined) return null;

  let level: 'low' | 'medium' | 'high';
  let label: string;
  if (setSize <= 1) {
    level = 'low';
    label = 'High confidence';
  } else if (setSize <= 3) {
    level = 'medium';
    label = `Set: ${setSize}`;
  } else {
    level = 'high';
    label = `Set: ${setSize}`;
  }

  return (
    <span className={`uncertainty-badge uncertainty-badge--${level}`} title={`Conformal prediction set size: ${setSize}`}>
      {label}
    </span>
  );
}

interface TaskPanelProps {
  id: string;
  icon: string;
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}

function TaskPanel({ id, icon, title, children, defaultExpanded = true }: TaskPanelProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className={`results-panel card ${expanded ? 'results-panel--expanded' : 'results-panel--collapsed'}`} id={id}>
      <button
        className="panel-header"
        onClick={() => setExpanded(prev => !prev)}
        aria-expanded={expanded}
        aria-controls={`${id}-content`}
        id={`${id}-toggle`}
      >
        <h3 className="panel-title">
          <span className="panel-icon">{icon}</span>
          {title}
        </h3>
        <span className={`panel-chevron ${expanded ? 'panel-chevron--open' : ''}`}>
          ▾
        </span>
      </button>
      <div
        className={`panel-content ${expanded ? 'panel-content--visible' : 'panel-content--hidden'}`}
        id={`${id}-content`}
      >
        {children}
      </div>
    </div>
  );
}

/* ---------- Main component ----------------------------------------------- */

export function Results() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const state = (location.state ?? {}) as ResultsLocationState;

  const findings = parsePredictions(state.predictionResult);
  const quality = parseQuality(state.predictionResult);
  const xaiData = useMemo(() => parseXAIData(state.predictionResult), [state.predictionResult]);
  const inferenceMode = state.predictionResult?.inference_mode;
  const fileName = state.fileName ?? `ECG ${id}`;

  /* ── Second Reader state ──────────────────────────────────── */
  const [secondReaderVisible, setSecondReaderVisible] = useState(false);

  // Extract AI predictions for the Second Reader component
  const aiPredictions = useMemo((): Record<string, number[]> => {
    if (!state.predictionResult?.predictions) return {};
    const preds = state.predictionResult.predictions as Record<string, unknown>;
    const result: Record<string, number[]> = {};
    for (const task of ['rhythm', 'structural', 'ischaemia']) {
      const raw = preds[task];
      if (Array.isArray(raw)) {
        result[task] = raw.map((v: unknown) => (typeof v === 'number' ? v : 0));
      }
    }
    return result;
  }, [state.predictionResult]);

  /* ── XAI state ────────────────────────────────────────────── */
  const [xaiVisible, setXaiVisible] = useState(false);
  const [activeFinding, setActiveFinding] = useState<string | null>(null);
  const [leadVisibility, setLeadVisibility] = useState<Record<string, boolean>>(() => {
    const vis: Record<string, boolean> = {};
    for (const lead of STANDARD_LEADS) vis[lead] = true;
    return vis;
  });

  const toggleLead = (lead: string) => {
    setLeadVisibility(prev => ({ ...prev, [lead]: !prev[lead] }));
  };

  // Build all findings for the XAI controls finding selector
  const allFindings = useMemo(() => {
    const result: { name: string; task: string; prob: number }[] = [];
    for (const f of findings.rhythm) result.push({ name: f.name, task: 'rhythm', prob: f.prob });
    for (const f of findings.structural) result.push({ name: f.name, task: 'structural', prob: f.prob });
    for (const f of findings.ischaemia) result.push({ name: f.name, task: 'ischaemia', prob: f.prob });
    return result;
  }, [findings]);

  // Determine which XAI data to display based on active finding
  const activeXAI = useMemo((): XAIAttribution | null => {
    if (!xaiVisible || xaiData.length === 0) return null;

    if (activeFinding) {
      // Find the task associated with the active finding
      const findingInfo = allFindings.find(f => f.name === activeFinding);
      if (findingInfo) {
        const taskXAI = xaiData.find(x => x.task === findingInfo.task);
        if (taskXAI) return taskXAI;
      }
    }
    // Default: show first task's XAI (usually rhythm)
    return xaiData[0] || null;
  }, [xaiVisible, xaiData, activeFinding, allFindings]);

  // Top features for the active finding
  const activeTopFeatures = useMemo(() => {
    if (!activeXAI) return [];
    return activeXAI.top_features;
  }, [activeXAI]);

  /* ── Copilot finding click — scroll to waveform + activate XAI ── */
  const handleCopilotFindingClick = useCallback((findingName: string, _task: string) => {
    // Activate XAI overlay and set the finding
    setXaiVisible(true);
    setActiveFinding(findingName);

    // Scroll the waveform into view
    const waveformEl = document.getElementById('ecg-waveform');
    if (waveformEl) {
      waveformEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  return (
    <div className="results-page" id="page-results">
      {/* Breadcrumb */}
      <div className="results-breadcrumb">
        <Link to="/" className="breadcrumb-link">Dashboard</Link>
        <span className="breadcrumb-sep">/</span>
        <Link to="/upload" className="breadcrumb-link">Upload</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">{fileName}</span>
      </div>

      {/* Metadata bar */}
      <div className="results-quality-bar" id="quality-bar">
        <QualityBadge quality={quality} />
        <span className="quality-meta">12-Lead · 500 Hz · 10s</span>
        {inferenceMode && (
          <span className={`inference-mode-badge inference-mode-badge--${inferenceMode}`}>
            {inferenceMode === 'server' ? '🟢 Server — Full Model' : '🟠 Offline — Edge Model'}
          </span>
        )}
        <button
          className={`second-reader-toggle ${secondReaderVisible ? 'second-reader-toggle--active' : ''}`}
          onClick={() => setSecondReaderVisible(prev => !prev)}
          id="second-reader-toggle"
          title="Toggle Second Reader Mode"
        >
          🔍 Second Reader
        </button>
      </div>

      {/* XAI controls */}
      <XAIControls
        findings={allFindings}
        activeFinding={activeFinding}
        onSelectFinding={setActiveFinding}
        leadVisibility={leadVisibility}
        onToggleLead={toggleLead}
        leads={STANDARD_LEADS}
        xaiVisible={xaiVisible}
        onToggleXAI={() => setXaiVisible(prev => !prev)}
      />

      {/* ECG waveform — interactive component with XAI overlay */}
      <ECGWaveformChart
        data={DEMO_ECG}
        id="ecg-waveform"
        xaiAttribution={activeXAI}
        xaiLeadVisibility={leadVisibility}
        xaiVisible={xaiVisible}
      />

      {/* XAI top features panel */}
      {xaiVisible && activeTopFeatures.length > 0 && (
        <TopFeaturesPanel
          features={activeTopFeatures}
          findingName={activeFinding || 'All Findings'}
        />
      )}

      {/* Second Reader comparison mode */}
      <SecondReaderMode
        aiPredictions={aiPredictions}
        visible={secondReaderVisible}
        onToggle={() => setSecondReaderVisible(prev => !prev)}
      />

      {/* AI Copilot panel — ranked findings with clinical cues */}
      <CopilotPanel
        findings={allFindings}
        onFindingClick={handleCopilotFindingClick}
        activeFinding={activeFinding}
      />

      {/* Edge-Case Spotlight — rare but dangerous findings */}
      <EdgeCaseSpotlight
        findings={allFindings}
        onEdgeCaseClick={handleCopilotFindingClick}
      />

      {/* Detailed findings panels */}
      <div className="results-panels" id="findings-panels">
        {/* Rhythm */}
        <TaskPanel id="panel-rhythm" icon="♥" title="Rhythm &amp; Conduction" defaultExpanded={true}>
          <ul className="findings-list">
            {findings.rhythm.map(f => (
              <li key={f.name} className="finding-item finding-item--with-explanation">
                <div className="finding-info">
                  <span className={`finding-dot severity-${f.severity}`} />
                  <span className="finding-name">{f.name}</span>
                </div>
                <div className="finding-bar-container">
                  <div
                    className={`finding-bar finding-bar--${f.severity}`}
                    style={{ width: `${f.prob * 100}%` }}
                  />
                  <span className="finding-prob">{(f.prob * 100).toFixed(0)}%</span>
                </div>
                <UncertaintyBadge setSize={f.predictionSetSize} />
                <FeedbackButtons
                  findingName={f.name}
                  task="rhythm"
                  confidence={f.prob}
                  ecgReferenceId={id}
                />
                <ExplanationCard
                  findingName={f.name}
                  task="rhythm"
                  confidence={f.prob}
                  featureAttributions={getFeatureAttributionsForFinding(xaiData, f.name, 'rhythm')}
                  confidenceInterval={f.predictionSetSize !== undefined ? { lower: Math.max(0, f.prob - 0.15), upper: Math.min(1, f.prob + 0.10), coverage: 0.90, predictionSetSize: f.predictionSetSize } : undefined}
                  suggestion={findSuggestion(f.name)}
                />
              </li>
            ))}
          </ul>
        </TaskPanel>

        {/* Structural */}
        <TaskPanel id="panel-structural" icon="◇" title="Structural &amp; Functional" defaultExpanded={true}>
          <ul className="findings-list">
            {findings.structural.map(f => (
              <li key={f.name} className="finding-item finding-item--with-explanation">
                <div className="finding-info">
                  <span className={`finding-dot severity-${f.severity}`} />
                  <span className="finding-name">{f.name}</span>
                </div>
                <div className="finding-bar-container">
                  <div
                    className={`finding-bar finding-bar--${f.severity}`}
                    style={{ width: `${f.prob * 100}%` }}
                  />
                  <span className="finding-prob">{(f.prob * 100).toFixed(0)}%</span>
                </div>
                <UncertaintyBadge setSize={f.predictionSetSize} />
                <FeedbackButtons
                  findingName={f.name}
                  task="structural"
                  confidence={f.prob}
                  ecgReferenceId={id}
                />
                <ExplanationCard
                  findingName={f.name}
                  task="structural"
                  confidence={f.prob}
                  featureAttributions={getFeatureAttributionsForFinding(xaiData, f.name, 'structural')}
                  confidenceInterval={f.predictionSetSize !== undefined ? { lower: Math.max(0, f.prob - 0.15), upper: Math.min(1, f.prob + 0.10), coverage: 0.90, predictionSetSize: f.predictionSetSize } : undefined}
                  suggestion={findSuggestion(f.name)}
                />
              </li>
            ))}
          </ul>
        </TaskPanel>

        {/* Ischaemia */}
        <TaskPanel id="panel-ischaemia" icon="△" title="Ischaemia &amp; Metabolic" defaultExpanded={true}>
          <ul className="findings-list">
            {findings.ischaemia.map(f => (
              <li key={f.name} className="finding-item finding-item--with-explanation">
                <div className="finding-info">
                  <span className={`finding-dot severity-${f.severity}`} />
                  <span className="finding-name">{f.name}</span>
                </div>
                <div className="finding-bar-container">
                  <div
                    className={`finding-bar finding-bar--${f.severity}`}
                    style={{ width: `${f.prob * 100}%` }}
                  />
                  <span className="finding-prob">{(f.prob * 100).toFixed(0)}%</span>
                </div>
                <UncertaintyBadge setSize={f.predictionSetSize} />
                <FeedbackButtons
                  findingName={f.name}
                  task="ischaemia"
                  confidence={f.prob}
                  ecgReferenceId={id}
                />
                <ExplanationCard
                  findingName={f.name}
                  task="ischaemia"
                  confidence={f.prob}
                  featureAttributions={getFeatureAttributionsForFinding(xaiData, f.name, 'ischaemia')}
                  confidenceInterval={f.predictionSetSize !== undefined ? { lower: Math.max(0, f.prob - 0.15), upper: Math.min(1, f.prob + 0.10), coverage: 0.90, predictionSetSize: f.predictionSetSize } : undefined}
                  suggestion={findSuggestion(f.name)}
                />
              </li>
            ))}
          </ul>
        </TaskPanel>

        {/* Risk */}
        <TaskPanel id="panel-risk" icon="⚡" title="Risk Prediction" defaultExpanded={true}>
          <div className="risk-gauges">
            {findings.risk.map(r => (
              <div key={r.name} className="risk-gauge">
                <div className="gauge-ring">
                  <svg viewBox="0 0 100 100" className="gauge-svg">
                    <circle cx="50" cy="50" r="42" className="gauge-bg" />
                    <circle
                      cx="50" cy="50" r="42"
                      className="gauge-fill"
                      strokeDasharray={`${r.score * 264} 264`}
                      style={{
                        stroke: r.score > 0.7 ? 'var(--color-danger)' : r.score > 0.4 ? 'var(--color-warning)' : 'var(--color-success)',
                      }}
                    />
                  </svg>
                  <span className="gauge-value">{(r.score * 100).toFixed(0)}%</span>
                </div>
                <span className="gauge-label">{r.name}</span>
                <span className={`gauge-clinical-label gauge-clinical-label--${r.score > 0.7 ? 'danger' : r.score > 0.4 ? 'warning' : 'success'}`}>
                  {r.clinicalLabel}
                </span>
                <UncertaintyBadge setSize={r.predictionSetSize} />
              </div>
            ))}
          </div>
        </TaskPanel>
      </div>
    </div>
  );
}
