import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './ProspectiveData.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Progress {
  total: number;
  linked: number;
  unlinked: number;
  completion_rate: number;
  site_id: string | null;
}

interface TaskPrediction {
  task: string;
  class_names: string[];
  probabilities: number[];
}

interface PredictResponse {
  predictions: TaskPrediction[];
}

const API_BASE = 'http://localhost:8000';

const OUTCOME_CATEGORIES = ['confirmed', 'ruled-out', 'indeterminate'];

// A compact subset of the class taxonomy for the ground-truth checklist.
const GROUND_TRUTH_CONDITIONS = [
  'AF', 'AFL', 'SVT', 'VT', 'VF', 'sinus_brady', 'sinus_tachy',
  'LBBB', 'RBBB', 'normal_sinus_rhythm', 'LVH', 'STEMI', 'brugada_pattern',
];

const TARGET_ENROLLMENT = 100;

export function ProspectiveDataPage() {
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();

  const [progress, setProgress] = useState<Progress | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [ecgRef, setEcgRef] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ground-truth form state.
  const [checkedConditions, setCheckedConditions] = useState<Set<string>>(new Set());
  const [notes, setNotes] = useState('');
  const [followUpDate, setFollowUpDate] = useState('');
  const [outcomeCategory, setOutcomeCategory] = useState('confirmed');
  const [submitted, setSubmitted] = useState(false);

  const fetchProgress = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/validation/prospective/progress`, {
        headers: { ...getAuthHeader() },
      });
      if (resp.ok) setProgress(await resp.json());
    } catch {
      /* progress is best-effort */
    }
  }, [getAuthHeader]);

  useEffect(() => {
    fetchProgress();
  }, [fetchProgress]);

  const runPrediction = useCallback(
    async (file: File) => {
      setRunning(true);
      setError(null);
      setPrediction(null);
      try {
        const form = new FormData();
        form.append('file', file);
        const resp = await fetch(`${API_BASE}/api/v1/predict`, {
          method: 'POST',
          headers: { ...getAuthHeader() },
          body: form,
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        setPrediction(await resp.json());
        setEcgRef(file.name);
        fetchProgress();
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setRunning(false);
      }
    },
    [getAuthHeader, fetchProgress],
  );

  const toggleCondition = (c: string) => {
    setCheckedConditions((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });
  };

  const submitGroundTruth = useCallback(async () => {
    setError(null);
    try {
      const ground_truth: Record<string, number> = {};
      checkedConditions.forEach((c) => (ground_truth[c] = 1));
      const resp = await fetch(`${API_BASE}/api/v1/validation/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
        body: JSON.stringify({
          ecg_hash: ecgRef || `ecg_${Date.now()}`,
          site_id: 'site_default',
          predictions: predictionToDict(prediction),
          quality: {},
          ground_truth,
          metadata: { notes, follow_up_date: followUpDate, outcome_category: outcomeCategory },
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setSubmitted(true);
      setCheckedConditions(new Set());
      setNotes('');
      fetchProgress();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [checkedConditions, ecgRef, prediction, notes, followUpDate, outcomeCategory, getAuthHeader, fetchProgress]);

  const exportCsv = useCallback(() => {
    window.open(`${API_BASE}/api/v1/validation/export?format=csv`, '_blank');
  }, []);

  const enrollmentPct = useMemo(
    () => Math.min(100, Math.round(((progress?.total ?? 0) / TARGET_ENROLLMENT) * 100)),
    [progress],
  );

  const incompleteWarning = prediction !== null && checkedConditions.size === 0;

  return (
    <div className="prospective-data" id="prospective-page">
      <header className="pd-header">
        <h1>{t('prospective.title', 'Prospective Data Collection')}</h1>
      </header>

      {/* Progress dashboard */}
      {progress && (
        <section className="pd-progress">
          <div className="pd-stat">
            <span className="pd-stat-value">{progress.total}</span>
            <span className="pd-stat-label">{t('prospective.submitted', 'ECGs submitted')}</span>
          </div>
          <div className="pd-stat">
            <span className="pd-stat-value">{progress.linked}</span>
            <span className="pd-stat-label">{t('prospective.linked', 'With ground-truth')}</span>
          </div>
          <div className="pd-stat">
            <span className="pd-stat-value">{Math.round(progress.completion_rate * 100)}%</span>
            <span className="pd-stat-label">{t('prospective.concordance', 'Completion')}</span>
          </div>
          <div className="pd-stat pd-stat-enroll">
            <span className="pd-stat-label">
              {t('prospective.enrollment', 'Enrollment')}: {progress.total}/{TARGET_ENROLLMENT}
            </span>
            <div className="pd-progress-bar">
              <div className="pd-progress-fill" style={{ width: `${enrollmentPct}%` }} />
            </div>
          </div>
        </section>
      )}

      <div className="pd-actions">
        <label className="pd-upload-btn">
          {running ? t('prospective.running', 'Running…') : t('prospective.upload', 'Upload ECG')}
          <input
            type="file"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) runPrediction(f);
            }}
          />
        </label>
        <button className="pd-btn" onClick={exportCsv}>
          {t('prospective.export', 'Export CSV')}
        </button>
      </div>

      {error && <div className="pd-error" role="alert">{error}</div>}
      {submitted && (
        <div className="pd-success" role="status">
          {t('prospective.submittedOk', 'Ground-truth linked successfully.')}
        </div>
      )}

      {/* Prediction review + ground-truth entry */}
      {prediction && (
        <div className="pd-review">
          <section className="pd-panel">
            <h2>{t('prospective.aiResults', 'AI Predictions')} — {ecgRef}</h2>
            {prediction.predictions.map((tp) => (
              <div key={tp.task} className="pd-task">
                <h3>{tp.task}</h3>
                <ul>
                  {tp.class_names
                    .map((name, i) => ({ name, p: tp.probabilities[i] }))
                    .filter((x) => x.p >= 0.3)
                    .sort((a, b) => b.p - a.p)
                    .slice(0, 5)
                    .map((x) => (
                      <li key={x.name}>
                        {x.name}: <strong>{(x.p * 100).toFixed(0)}%</strong>
                      </li>
                    ))}
                </ul>
              </div>
            ))}
          </section>

          <section className="pd-panel">
            <h2>{t('prospective.groundTruth', 'Ground-Truth Entry')}</h2>
            {incompleteWarning && (
              <div className="pd-flag">{t('prospective.incomplete', 'No conditions selected yet')}</div>
            )}
            <div className="pd-conditions">
              {GROUND_TRUTH_CONDITIONS.map((c) => (
                <label key={c} className="pd-check">
                  <input
                    type="checkbox"
                    checked={checkedConditions.has(c)}
                    onChange={() => toggleCondition(c)}
                  />
                  {c}
                </label>
              ))}
            </div>
            <label className="pd-field">
              {t('prospective.outcomeCategory', 'Outcome category')}
              <select value={outcomeCategory} onChange={(e) => setOutcomeCategory(e.target.value)}>
                {OUTCOME_CATEGORIES.map((o) => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
            </label>
            <label className="pd-field">
              {t('prospective.followUp', 'Follow-up date')}
              <input type="date" value={followUpDate} onChange={(e) => setFollowUpDate(e.target.value)} />
            </label>
            <label className="pd-field">
              {t('prospective.notes', 'Notes')}
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} />
            </label>
            <button className="pd-btn pd-btn-primary" onClick={submitGroundTruth}>
              {t('prospective.submit', 'Submit Ground-Truth')}
            </button>
          </section>
        </div>
      )}
    </div>
  );
}

function predictionToDict(pred: PredictResponse | null): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  if (!pred) return out;
  for (const tp of pred.predictions) {
    out[tp.task] = {};
    tp.class_names.forEach((name, i) => (out[tp.task][name] = tp.probabilities[i]));
  }
  return out;
}
