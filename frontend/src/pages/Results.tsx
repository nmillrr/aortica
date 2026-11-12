import { useState, useMemo, useCallback } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';
import { ECGWaveformChart, generateDemoECGData } from '../components/ECGWaveformChart';
import { XAIControls, TopFeaturesPanel } from '../components/XAIOverlay';
import { CopilotPanel } from '../components/CopilotPanel';
import { FinalizeSubmit } from '../components/FinalizeSubmit';
import { SecondReaderMode } from '../components/SecondReaderMode';
import { EdgeCaseSpotlight } from '../components/EdgeCaseSpotlight';
import { ExplanationCard } from '../components/ExplanationCard';
import { FeedbackButtons } from '../components/FeedbackButtons';
import type { XAIAttribution } from '../components/XAIOverlay';
import type { ECGFeatureAttribution, ClinicalSuggestionInfo } from '../components/ExplanationCard';
import type { PredictionResult } from '../services/InferenceClient';
import {
  MODEL_SAMPLE_RATE_HZ,
  labelFor,
  suppressedOutputs,
  toFindings,
  type Finding,
  type QualityReport,
} from '../services/predictions';
import './Results.css';

/* ---------- Types for route state ---------------------------------------- */

interface ResultsLocationState {
  predictionResult?: PredictionResult;
  fileName?: string;
  fileSize?: number;
}

/**
 * Order findings for display: positives first, then by how close each comes
 * to its own threshold. Raw probabilities are not comparable across classes,
 * so they are never the primary sort key.
 */
function byClinicalRelevance(a: Finding, b: Finding): number {
  if (a.positive !== b.positive) return a.positive ? -1 : 1;
  const ratio = (f: Finding) => (f.threshold ? f.prob / f.threshold : f.prob);
  return ratio(b) - ratio(a);
}

function parseXAIData(result: PredictionResult | undefined): XAIAttribution[] {
  return Array.isArray(result?.xai) ? (result.xai as unknown as XAIAttribution[]) : [];
}

// The API does not return the signal yet, so the strip is illustrative only.
const DEMO_ECG = generateDemoECGData();
const STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

/* ---------- Clinical suggestions ----------------------------------------- */

/**
 * Offline fallback only — the server returns its own suggestions for
 * positive findings. Keyed by the model's class names, and limited to
 * outputs the checkpoint trained: a cue for a class the model cannot
 * detect would never fire, and must not be matched by substring.
 */
const OFFLINE_SUGGESTIONS: Record<string, ClinicalSuggestionInfo> = {
  AF:                  { prompt: 'Evaluate stroke risk (CHA₂DS₂-VASc) and consider anticoagulation', urgency: 'prompt', rationale: 'AF increases stroke risk 5-fold. CHA₂DS₂-VASc score guides anticoagulation decisions.' },
  AFL:                 { prompt: 'Cardiology referral for rate/rhythm control', urgency: 'prompt', rationale: 'Atrial flutter often responds to catheter ablation with high success rates.' },
  SVT:                 { prompt: 'If symptomatic, consider vagal manoeuvres or adenosine', urgency: 'prompt', rationale: 'Most SVTs are re-entrant and respond to AV nodal blockade.' },
  av_block_3rd:        { prompt: 'Urgent cardiology — temporary pacing may be required', urgency: 'emergent', rationale: 'Complete heart block can cause syncope, hemodynamic collapse, or cardiac arrest.' },
  LBBB:                { prompt: 'New LBBB warrants acute coronary syndrome workup', urgency: 'urgent', rationale: 'New LBBB with chest pain meets STEMI-equivalent criteria per Sgarbossa.' },
  WPW:                 { prompt: 'Electrophysiology referral for SCD risk stratification', urgency: 'urgent', rationale: 'Accessory pathways can conduct rapidly during AF, leading to VF.' },
  RBBB:                { prompt: 'Evaluate for right heart strain if new onset', urgency: 'routine', rationale: 'Isolated RBBB is often benign but new onset may indicate PE or right heart disease.' },
  STEMI:               { prompt: 'Activate cath lab — door-to-balloon time critical', urgency: 'emergent', rationale: 'Every minute of delay increases myocardial damage and mortality.' },
  posterior_MI:        { prompt: 'Obtain posterior leads (V7-V9); emergent cath consideration', urgency: 'emergent', rationale: 'Posterior MI is frequently missed on standard 12-lead ECG.' },
  QTc_prolongation:    { prompt: 'Review QT-prolonging medications; electrolytes check', urgency: 'urgent', rationale: 'Prolonged QTc increases risk of Torsades de Pointes.' },
  LVH:                 { prompt: 'Blood pressure optimisation and echo assessment', urgency: 'prompt' },
  PVC:                 { prompt: 'If frequent (>10%), consider echocardiography', urgency: 'routine' },
  normal_sinus_rhythm: { prompt: 'Normal sinus rhythm — no specific action needed', urgency: 'routine' },
  sinus_tachy:         { prompt: 'Evaluate underlying cause (fever, pain, volume status)', urgency: 'routine' },
  LA_enlargement:      { prompt: 'Evaluate atrial fibrillation risk and valvular disease', urgency: 'routine' },
};

const URGENCIES = new Set(['routine', 'prompt', 'urgent', 'emergent']);

function suggestionsFor(result: PredictionResult): Map<string, ClinicalSuggestionInfo> {
  const map = new Map<string, ClinicalSuggestionInfo>();
  for (const s of result.suggestions) {
    map.set(s.condition, {
      prompt: s.prompt,
      urgency: (URGENCIES.has(s.urgency) ? s.urgency : 'routine') as ClinicalSuggestionInfo['urgency'],
      rationale: s.rationale || undefined,
    });
  }
  return map;
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

function QualityBadge({ quality }: { quality: QualityReport }) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div
      className="quality-badge-wrapper"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      id="quality-badge"
    >
      <span className={`quality-badge quality-badge--${quality.classification}`}>
        Signal Quality: {quality.classification.charAt(0).toUpperCase() + quality.classification.slice(1)} ({Math.round(quality.overall)}/100)
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
                  {Math.round(lead.score)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
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

/** One finding row: probability bar with the class's own threshold marked. */
function FindingRow({
  finding: f,
  resultId,
  xaiData,
  suggestion,
}: {
  finding: Finding;
  resultId?: string;
  xaiData: XAIAttribution[];
  suggestion?: ClinicalSuggestionInfo;
}) {
  const severity = f.positive ? 'high' : 'normal';
  return (
    <li className="finding-item finding-item--with-explanation">
      <div className="finding-info">
        <span className={`finding-dot severity-${severity}`} />
        <span className="finding-name">{f.label}</span>
        {f.positive && <span className="finding-positive-tag">Positive</span>}
      </div>
      <div className="finding-bar-container">
        <div className="finding-bar-track">
          <div
            className={`finding-bar finding-bar--${severity}`}
            style={{ width: `${f.prob * 100}%` }}
          />
          {f.threshold !== null && (
            <span
              className="finding-threshold"
              style={{ left: `${f.threshold * 100}%` }}
              title={`Operating point ${(f.threshold * 100).toFixed(0)}%`}
            />
          )}
        </div>
        <span className="finding-prob">
          {(f.prob * 100).toFixed(0)}%
          {f.threshold !== null && (
            <span className="finding-threshold-label"> / {(f.threshold * 100).toFixed(0)}%</span>
          )}
        </span>
      </div>
      <FeedbackButtons
        findingName={f.name}
        task={f.task}
        confidence={f.prob}
        ecgReferenceId={resultId}
      />
      <ExplanationCard
        findingName={f.label}
        task={f.task}
        confidence={f.prob}
        featureAttributions={getFeatureAttributionsForFinding(xaiData, f.name, f.task)}
        suggestion={suggestion}
      />
    </li>
  );
}

const TASK_PANELS = [
  { task: 'rhythm', icon: '♥', title: 'Rhythm & Conduction' },
  { task: 'structural', icon: '◇', title: 'Structural & Functional' },
  { task: 'ischaemia', icon: '△', title: 'Ischaemia & Metabolic' },
] as const;

export function Results() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const state = (location.state ?? {}) as ResultsLocationState;
  const result = state.predictionResult;
  const fileName = state.fileName ?? `ECG ${id}`;

  if (!result || result.predictions.length === 0) {
    return (
      <div className="results-page" id="page-results">
        <div className="results-breadcrumb">
          <Link to="/" className="breadcrumb-link">Dashboard</Link>
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-current">{fileName}</span>
        </div>
        <div className="results-empty card" id="results-empty">
          <h2>No results to show</h2>
          <p>
            {result
              ? 'The analysis returned no readable predictions.'
              : 'Results are only available right after an analysis.'}
          </p>
          <Link to="/upload" className="btn btn-primary">Upload an ECG</Link>
        </div>
      </div>
    );
  }

  return <ResultsView result={result} resultId={id} fileName={fileName} />;
}

function ResultsView({
  result,
  resultId,
  fileName,
}: {
  result: PredictionResult;
  resultId?: string;
  fileName: string;
}) {
  const findings = useMemo(() => toFindings(result.predictions), [result]);
  const positives = useMemo(() => findings.filter(f => f.positive), [findings]);
  const suppressed = useMemo(() => suppressedOutputs(result.predictions), [result]);
  const serverSuggestions = useMemo(() => suggestionsFor(result), [result]);
  const xaiData = useMemo(() => parseXAIData(result), [result]);
  const quality = result.quality;
  const inferenceMode = result.inference_mode;

  const suggestionFor = useCallback((f: Finding): ClinicalSuggestionInfo | undefined => {
    if (!f.positive) return undefined;
    return inferenceMode === 'server' ? serverSuggestions.get(f.name) : OFFLINE_SUGGESTIONS[f.name];
  }, [inferenceMode, serverSuggestions]);

  /* ── Second Reader state ──────────────────────────────────── */
  const [secondReaderVisible, setSecondReaderVisible] = useState(false);

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

  // Determine which XAI data to display based on active finding
  const activeXAI = useMemo((): XAIAttribution | null => {
    if (!xaiVisible || xaiData.length === 0) return null;

    if (activeFinding) {
      const findingInfo = findings.find(f => f.name === activeFinding);
      if (findingInfo) {
        const taskXAI = xaiData.find(x => x.task === findingInfo.task);
        if (taskXAI) return taskXAI;
      }
    }
    return xaiData[0] || null;
  }, [xaiVisible, xaiData, activeFinding, findings]);

  const activeTopFeatures = activeXAI ? activeXAI.top_features : [];

  /* ── Copilot finding click — scroll to waveform + activate XAI ── */
  const handleCopilotFindingClick = useCallback((findingName: string) => {
    setXaiVisible(true);
    setActiveFinding(findingName);
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
        {quality && <QualityBadge quality={quality} />}
        <span className="quality-meta" title="Every recording is resampled and windowed to this before scoring">
          Model input: 12-lead · {MODEL_SAMPLE_RATE_HZ} Hz · 10 s
        </span>
        <span className={`inference-mode-badge inference-mode-badge--${inferenceMode}`}>
          {inferenceMode === 'server' ? '🟢 Server — Full Model' : '🟠 Offline — Edge Model'}
        </span>
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
        findings={findings.map(f => ({ name: f.name, task: f.task, prob: f.prob }))}
        activeFinding={activeFinding}
        onSelectFinding={setActiveFinding}
        leadVisibility={leadVisibility}
        onToggleLead={toggleLead}
        leads={STANDARD_LEADS}
        xaiVisible={xaiVisible}
        onToggleXAI={() => setXaiVisible(prev => !prev)}
      />

      {/* ECG waveform — illustrative until the API returns the signal */}
      <ECGWaveformChart
        data={DEMO_ECG}
        id="ecg-waveform"
        xaiAttribution={activeXAI}
        xaiLeadVisibility={leadVisibility}
        xaiVisible={xaiVisible}
      />
      <p className="results-waveform-note" id="waveform-note">
        Illustrative waveform — the uploaded recording is not displayed yet.
      </p>

      {xaiVisible && activeTopFeatures.length > 0 && (
        <TopFeaturesPanel
          features={activeTopFeatures}
          findingName={activeFinding ? labelFor(activeFinding) : 'All Findings'}
        />
      )}

      <SecondReaderMode
        aiFindings={findings}
        visible={secondReaderVisible}
        onToggle={() => setSecondReaderVisible(prev => !prev)}
      />

      {/* AI Copilot panel — positive findings only, ranked */}
      <CopilotPanel
        findings={positives}
        confidenceThreshold={0}
        onFindingClick={handleCopilotFindingClick}
        activeFinding={activeFinding}
      />

      <FinalizeSubmit
        resultId={resultId ?? ''}
        ecgId={fileName}
        findings={positives}
      />

      <EdgeCaseSpotlight
        findings={positives}
        onEdgeCaseClick={handleCopilotFindingClick}
      />

      {/* Detailed findings panels */}
      <div className="results-panels" id="findings-panels">
        {TASK_PANELS.map(({ task, icon, title }) => {
          const taskFindings = findings.filter(f => f.task === task).sort(byClinicalRelevance);
          if (taskFindings.length === 0) return null;
          return (
            <TaskPanel key={task} id={`panel-${task}`} icon={icon} title={title} defaultExpanded={true}>
              <ul className="findings-list">
                {taskFindings.map(f => (
                  <FindingRow
                    key={f.name}
                    finding={f}
                    resultId={resultId}
                    xaiData={xaiData}
                    suggestion={suggestionFor(f)}
                  />
                ))}
              </ul>
            </TaskPanel>
          );
        })}
      </div>

      {suppressed.length > 0 && (
        <details className="results-not-assessed card" id="not-assessed">
          <summary>Not assessed by this model ({suppressed.length})</summary>
          <p>
            These outputs were never trained, so the model cannot detect them.
            Their absence here is not a negative result.
          </p>
          <p className="results-not-assessed-list">{suppressed.map(labelFor).join(' · ')}</p>
        </details>
      )}
    </div>
  );
}
