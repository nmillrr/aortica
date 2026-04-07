import { useParams, Link } from 'react-router-dom';
import './Results.css';

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

export function Results() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="results-page" id="page-results">
      {/* Breadcrumb */}
      <div className="results-breadcrumb">
        <Link to="/" className="breadcrumb-link">Dashboard</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">ECG {id}</span>
      </div>

      {/* Quality badge */}
      <div className="results-quality-bar" id="quality-bar">
        <span className="quality-badge quality-badge--good">
          Signal Quality: Good (87/100)
        </span>
        <span className="quality-meta">12-Lead · 500 Hz · 10s</span>
      </div>

      {/* ECG waveform placeholder */}
      <div className="results-waveform card" id="ecg-waveform">
        <div className="waveform-placeholder">
          <div className="waveform-grid" />
          <svg className="waveform-trace" viewBox="0 0 800 200" preserveAspectRatio="none">
            <path
              d="M0,100 L50,100 L60,98 L70,90 L80,100 L85,100 L90,40 L95,160 L100,100 L110,100 L120,95 L140,85 L160,100
                 L200,100 L210,98 L220,90 L230,100 L235,100 L240,35 L245,165 L250,100 L260,100 L270,95 L290,85 L310,100
                 L350,100 L360,98 L370,90 L380,100 L385,100 L390,38 L395,162 L400,100 L410,100 L420,95 L440,85 L460,100
                 L500,100 L510,98 L520,90 L530,100 L535,100 L540,42 L545,158 L550,100 L560,100 L570,95 L590,85 L610,100
                 L650,100 L660,98 L670,90 L680,100 L685,100 L690,36 L695,164 L700,100 L710,100 L720,95 L740,85 L760,100 L800,100"
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth="2"
            />
          </svg>
          <div className="waveform-lead-labels">
            {['I', 'II', 'III', 'aVR', 'aVL', 'aVF'].map(lead => (
              <span key={lead} className="waveform-lead-label">{lead}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Findings panels */}
      <div className="results-panels" id="findings-panels">
        {/* Rhythm */}
        <div className="results-panel card" id="panel-rhythm">
          <h3 className="panel-title">
            <span className="panel-icon">♥</span>
            Rhythm & Conduction
          </h3>
          <ul className="findings-list">
            {MOCK_FINDINGS.rhythm.map(f => (
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
            Structural & Functional
          </h3>
          <ul className="findings-list">
            {MOCK_FINDINGS.structural.map(f => (
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
            Ischaemia & Metabolic
          </h3>
          <ul className="findings-list">
            {MOCK_FINDINGS.ischaemia.map(f => (
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
            {MOCK_FINDINGS.risk.map(r => (
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
