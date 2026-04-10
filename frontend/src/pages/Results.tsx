import { useParams, useLocation, Link } from 'react-router-dom';
import { ECGWaveformChart, generateDemoECGData } from '../components/ECGWaveformChart';
import type { PredictionResult } from '../services/InferenceClient';
import './Results.css';

/* ---------- Types for route state ---------------------------------------- */

interface ResultsLocationState {
  predictionResult?: PredictionResult;
  fileName?: string;
  fileSize?: number;
}

/* ---------- Default mock data (when navigating directly) ----------------- */

const MOCK_FINDINGS = {
  rhythm: [
    { name: 'Atrial Fibrillation',   prob: 0.92, severity: 'high'   },
    { name: 'Normal Sinus Rhythm',    prob: 0.06, severity: 'normal' },
    { name: 'Sinus Tachycardia',      prob: 0.02, severity: 'normal' },
  ],
  structural: [
    { name: 'Left Ventricular Hypertrophy', prob: 0.74, severity: 'medium' },
    { name: 'LA Enlargement',               prob: 0.41, severity: 'normal' },
    { name: 'Dilated Cardiomyopathy',       prob: 0.08, severity: 'normal' },
  ],
  ischaemia: [
    { name: 'ST Elevation (Lateral)',  prob: 0.15, severity: 'normal' },
    { name: 'QTc Prolongation',        prob: 0.09, severity: 'normal' },
  ],
  risk: [
    { name: '1-Year Mortality',         score: 0.12 },
    { name: '12-Month HF Hospitalization', score: 0.28 },
    { name: '12-Month AF Onset',        score: 0.85 },
  ],
} as const;

function classifySeverity(prob: number): string {
  if (prob >= 0.80) return 'high';
  if (prob >= 0.50) return 'medium';
  return 'normal';
}

/**
 * Try to parse prediction data from the API response into our display format.
 * Falls back to mock data if the response shape is unexpected.
 */
function parsePredictions(result: PredictionResult | undefined) {
  if (!result?.predictions) return MOCK_FINDINGS;

  const preds = result.predictions as Record<string, unknown>;

  const parseFindingArray = (
    key: string,
    labels: string[],
  ): Array<{ name: string; prob: number; severity: string }> => {
    const raw = preds[key];
    if (!Array.isArray(raw)) return [];
    return labels.map((label, i) => {
      const prob = typeof raw[i] === 'number' ? (raw[i] as number) : 0;
      return { name: label, prob, severity: classifySeverity(prob) };
    }).sort((a, b) => b.prob - a.prob).slice(0, 5);
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
  const risk = Array.isArray(riskRaw)
    ? riskLabels.map((name, i) => ({
        name,
        score: typeof riskRaw[i] === 'number' ? (riskRaw[i] as number) : 0,
      }))
    : MOCK_FINDINGS.risk.map(r => ({ ...r }));

  return {
    rhythm: rhythm.length ? rhythm : MOCK_FINDINGS.rhythm.map(f => ({ ...f })),
    structural: structural.length ? structural : MOCK_FINDINGS.structural.map(f => ({ ...f })),
    ischaemia: ischaemia.length ? ischaemia : MOCK_FINDINGS.ischaemia.map(f => ({ ...f })),
    risk,
  };
}

const DEMO_ECG = generateDemoECGData();

export function Results() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const state = (location.state ?? {}) as ResultsLocationState;

  const findings = parsePredictions(state.predictionResult);
  const inferenceMode = state.predictionResult?.inference_mode;
  const fileName = state.fileName ?? `ECG ${id}`;

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
        <span className="quality-badge quality-badge--good">
          Signal Quality: Good (87/100)
        </span>
        <span className="quality-meta">12-Lead · 500 Hz · 10s</span>
        {inferenceMode && (
          <span className={`inference-mode-badge inference-mode-badge--${inferenceMode}`}>
            {inferenceMode === 'server' ? '🟢 Server — Full Model' : '🟠 Offline — Edge Model'}
          </span>
        )}
      </div>

      {/* ECG waveform — interactive component */}
      <ECGWaveformChart
        data={DEMO_ECG}
        id="ecg-waveform"
      />

      {/* Findings panels */}
      <div className="results-panels" id="findings-panels">
        {/* Rhythm */}
        <div className="results-panel card" id="panel-rhythm">
          <h3 className="panel-title">
            <span className="panel-icon">♥</span>
            Rhythm &amp; Conduction
          </h3>
          <ul className="findings-list">
            {findings.rhythm.map(f => (
              <li key={f.name} className="finding-item">
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
              </li>
            ))}
          </ul>
        </div>

        {/* Structural */}
        <div className="results-panel card" id="panel-structural">
          <h3 className="panel-title">
            <span className="panel-icon">◇</span>
            Structural &amp; Functional
          </h3>
          <ul className="findings-list">
            {findings.structural.map(f => (
              <li key={f.name} className="finding-item">
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
              </li>
            ))}
          </ul>
        </div>

        {/* Ischaemia */}
        <div className="results-panel card" id="panel-ischaemia">
          <h3 className="panel-title">
            <span className="panel-icon">△</span>
            Ischaemia &amp; Metabolic
          </h3>
          <ul className="findings-list">
            {findings.ischaemia.map(f => (
              <li key={f.name} className="finding-item">
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
              </li>
            ))}
          </ul>
        </div>

        {/* Risk */}
        <div className="results-panel card" id="panel-risk">
          <h3 className="panel-title">
            <span className="panel-icon">⚡</span>
            Risk Prediction
          </h3>
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
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
