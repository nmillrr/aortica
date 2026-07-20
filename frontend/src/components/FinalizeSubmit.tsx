import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './FinalizeSubmit.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Finding {
  name: string;
  prob: number;
  task: string;
}

interface FinalizeSubmitProps {
  resultId: string;
  ecgId?: string;
  findings: Finding[];
}

interface FinalizeResponse {
  status: string;
  ehr_reference: string;
  submitted_at: number;
  report_references: Record<string, string>;
  channels_generated: string[];
  worklist_updated: boolean;
}

const API_BASE = 'http://localhost:8000';
const OUTPUT_CHANNELS = ['pdf', 'fhir', 'hl7', 'dicom_sr'];

export function FinalizeSubmit({ resultId, ecgId, findings }: FinalizeSubmitProps) {
  const { t } = useTranslation();
  const { getAuthHeader, user } = useAuth();

  const [showModal, setShowModal] = useState(false);
  const [channels, setChannels] = useState<Set<string>>(new Set(['pdf', 'fhir']));
  const [confirmed, setConfirmed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState<FinalizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reviewedFindings = useMemo(() => {
    const out: Record<string, Record<string, number>> = {};
    for (const f of findings) {
      out[f.task] = out[f.task] || {};
      out[f.task][f.name] = f.prob;
    }
    return out;
  }, [findings]);

  const toggleChannel = (c: string) =>
    setChannels((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c);
      else next.add(c);
      return next;
    });

  const submit = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/v1/workflow/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
        body: JSON.stringify({
          result_id: resultId,
          ecg_id: ecgId,
          reviewed_findings: reviewedFindings,
          attestation: { clinician: user?.name || user?.sub || 'clinician', confirmed: true },
          output_channels: [...channels],
        }),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${resp.status}`);
      }
      setSubmitted(await resp.json());
      setShowModal(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [resultId, ecgId, reviewedFindings, channels, user, getAuthHeader]);

  if (submitted) {
    return (
      <div className="fs-badge" id="fs-submitted-badge">
        <span className="fs-badge-check">✓</span>
        <div>
          <div className="fs-badge-title">
            {t('finalize.submitted', 'Submitted to EHR')}
          </div>
          <div className="fs-badge-meta">
            {t('finalize.ref', 'Reference')}: {submitted.ehr_reference} ·{' '}
            {new Date(submitted.submitted_at * 1000).toLocaleString()}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fs-container">
      <button
        className="fs-finalize-btn"
        id="fs-finalize-btn"
        onClick={() => setShowModal(true)}
      >
        {t('finalize.button', 'Finalize & Submit')}
      </button>

      {error && <div className="fs-error" role="alert">{error}</div>}

      {showModal && (
        <div className="fs-modal-backdrop" role="dialog" aria-modal="true">
          <div className="fs-modal">
            <h2>{t('finalize.attestTitle', 'Attest and Submit')}</h2>

            <section className="fs-section">
              <h3>{t('finalize.included', 'Findings included')}</h3>
              {findings.length === 0 ? (
                <p className="fs-muted">{t('finalize.none', 'No positive findings.')}</p>
              ) : (
                <ul className="fs-findings">
                  {findings.map((f) => (
                    <li key={`${f.task}-${f.name}`}>
                      {f.name} <span className="fs-conf">{(f.prob * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="fs-section">
              <h3>{t('finalize.channels', 'Output channels')}</h3>
              <div className="fs-channels">
                {OUTPUT_CHANNELS.map((c) => (
                  <label key={c} className="fs-channel">
                    <input
                      type="checkbox"
                      checked={channels.has(c)}
                      onChange={() => toggleChannel(c)}
                    />
                    {c.toUpperCase()}
                  </label>
                ))}
              </div>
            </section>

            <section className="fs-section">
              <div className="fs-attest-meta">
                {t('finalize.clinician', 'Clinician')}: {user?.name || user?.sub} ·{' '}
                {new Date().toLocaleString()}
              </div>
              <label className="fs-confirm">
                <input
                  type="checkbox"
                  checked={confirmed}
                  onChange={(e) => setConfirmed(e.target.checked)}
                />
                {t('finalize.confirm', 'I attest this report is accurate')}
              </label>
            </section>

            <div className="fs-actions">
              <button className="fs-btn" onClick={() => setShowModal(false)}>
                {t('finalize.cancel', 'Cancel')}
              </button>
              <button
                className="fs-btn fs-btn-primary"
                disabled={!confirmed || channels.size === 0 || submitting}
                onClick={submit}
              >
                {submitting
                  ? t('finalize.submitting', 'Submitting…')
                  : t('finalize.attestSubmit', 'Attest and Submit')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
