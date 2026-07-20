import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './AdverseEventWizard.css';

// ---------------------------------------------------------------------------
// Types & constants
// ---------------------------------------------------------------------------

const API_BASE = 'http://localhost:8000';
const DRAFT_KEY = 'aortica_adverse_event_draft';

type Severity = 'minor' | 'moderate' | 'serious' | 'critical';

const SEVERITY_INFO: { value: Severity; label: string; definition: string }[] = [
  { value: 'minor', label: 'Minor', definition: 'No harm; minimal or no intervention required.' },
  { value: 'moderate', label: 'Moderate', definition: 'Temporary harm requiring intervention.' },
  { value: 'serious', label: 'Serious', definition: 'Resulted in hospitalization or significant intervention.' },
  { value: 'critical', label: 'Critical', definition: 'Life-threatening or resulted in permanent harm/death.' },
];

interface DraftState {
  ecgReference: string;
  eventDate: string;
  reporterId: string;
  description: string;
  severity: Severity | '';
  aiFinding: string;
  patientOutcome: string;
  followUpStatus: string;
  harmPrevented: string;
}

const EMPTY_DRAFT: DraftState = {
  ecgReference: '', eventDate: '', reporterId: '', description: '',
  severity: '', aiFinding: '', patientOutcome: '', followUpStatus: '', harmPrevented: '',
};

const STEPS = ['identification', 'details', 'outcome', 'review'];

export function AdverseEventWizard() {
  const { t } = useTranslation();
  const { getAuthHeader, user } = useAuth();
  const [params] = useSearchParams();

  const [step, setStep] = useState(0);
  const [form, setForm] = useState<DraftState>(EMPTY_DRAFT);
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submittedRef, setSubmittedRef] = useState<string | null>(null);
  const [hasDraft, setHasDraft] = useState(false);

  // Auto-populate from query params (e.g. navigated from a results page).
  useEffect(() => {
    const ecg = params.get('ecg');
    const finding = params.get('finding');
    const stored = localStorage.getItem(DRAFT_KEY);
    if (stored) {
      setHasDraft(true);
    }
    setForm((prev) => ({
      ...prev,
      ecgReference: ecg || prev.ecgReference,
      aiFinding: finding || prev.aiFinding,
      reporterId: prev.reporterId || user?.sub || '',
    }));
  }, [params, user]);

  // Persist draft to localStorage on every change (unless already submitted).
  useEffect(() => {
    if (!submittedRef) {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(form));
    }
  }, [form, submittedRef]);

  const set = (key: keyof DraftState, value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const resumeDraft = () => {
    const stored = localStorage.getItem(DRAFT_KEY);
    if (stored) setForm(JSON.parse(stored));
    setHasDraft(false);
  };

  const discardDraft = () => {
    localStorage.removeItem(DRAFT_KEY);
    setForm(EMPTY_DRAFT);
    setHasDraft(false);
  };

  const stepValid = useMemo(() => {
    switch (step) {
      case 0:
        return form.ecgReference.trim() && form.eventDate && form.reporterId.trim();
      case 1:
        return form.description.trim().length >= 50 && form.severity;
      case 2:
        return true; // outcome fields optional
      case 3:
        return confirmed;
      default:
        return false;
    }
  }, [step, form, confirmed]);

  const submit = useCallback(async () => {
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/v1/validation/adverse-event`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
        body: JSON.stringify({
          reporter_id: form.reporterId,
          ecg_reference: form.ecgReference,
          event_description: form.description,
          severity: form.severity,
          ai_finding: form.aiFinding || 'unspecified',
          patient_outcome: [
            form.patientOutcome,
            form.followUpStatus && `Follow-up: ${form.followUpStatus}`,
            form.harmPrevented && `Harm prevented: ${form.harmPrevented}`,
          ].filter(Boolean).join(' | '),
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setSubmittedRef(data.id);
      localStorage.removeItem(DRAFT_KEY);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [form, getAuthHeader]);

  const downloadCopy = () => {
    const blob = new Blob([JSON.stringify({ reference: submittedRef, ...form }, null, 2)],
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `adverse_event_${submittedRef}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Success modal
  if (submittedRef) {
    return (
      <div className="ae-wizard">
        <div className="ae-success-modal" role="dialog" aria-modal="true">
          <div className="ae-success-icon">✓</div>
          <h2>{t('adverseWizard.submitted', 'Report Submitted')}</h2>
          <p>{t('adverseWizard.reference', 'Event reference')}: <strong>{submittedRef}</strong></p>
          <div className="ae-success-actions">
            <button className="ae-btn" onClick={downloadCopy}>
              {t('adverseWizard.downloadCopy', 'Download a copy')}
            </button>
            <Link className="ae-btn ae-btn-primary" to="/report-event/history">
              {t('adverseWizard.viewHistory', 'View history')}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="ae-wizard" id="adverse-wizard">
      <header className="ae-header">
        <h1>{t('adverseWizard.title', 'Report an Adverse Event')}</h1>
        <Link className="ae-history-link" to="/report-event/history">
          {t('adverseWizard.history', 'History')}
        </Link>
      </header>

      {hasDraft && (
        <div className="ae-draft-banner" role="status">
          {t('adverseWizard.draftFound', 'A saved draft was found.')}
          <button className="ae-link-btn" onClick={resumeDraft}>{t('adverseWizard.resume', 'Resume')}</button>
          <button className="ae-link-btn" onClick={discardDraft}>{t('adverseWizard.discard', 'Discard')}</button>
        </div>
      )}

      {/* Step indicator */}
      <ol className="ae-steps">
        {STEPS.map((s, i) => (
          <li key={s} className={`ae-step ${i === step ? 'current' : ''} ${i < step ? 'done' : ''}`}>
            <span className="ae-step-num">{i + 1}</span>
            {t(`adverseWizard.step_${s}`, s)}
          </li>
        ))}
      </ol>

      {error && <div className="ae-error" role="alert">{error}</div>}

      <div className="ae-panel">
        {step === 0 && (
          <div className="ae-fields">
            <label className="ae-field">
              {t('adverseWizard.ecgRef', 'ECG reference')} *
              <input value={form.ecgReference} onChange={(e) => set('ecgReference', e.target.value)} />
            </label>
            <label className="ae-field">
              {t('adverseWizard.eventDate', 'Event date')} *
              <input type="date" value={form.eventDate} onChange={(e) => set('eventDate', e.target.value)} />
            </label>
            <label className="ae-field">
              {t('adverseWizard.reporter', 'Reporter')} *
              <input value={form.reporterId} onChange={(e) => set('reporterId', e.target.value)} />
            </label>
          </div>
        )}

        {step === 1 && (
          <div className="ae-fields">
            <label className="ae-field">
              {t('adverseWizard.description', 'Event description')} * ({form.description.trim().length}/50 min)
              <textarea
                rows={5}
                value={form.description}
                onChange={(e) => set('description', e.target.value)}
              />
            </label>
            <div className="ae-severity-group">
              <span className="ae-field-label">{t('adverseWizard.severity', 'Severity')} *</span>
              <div className="ae-severity-grid">
                {SEVERITY_INFO.map((s) => (
                  <button
                    key={s.value}
                    type="button"
                    className={`ae-severity severity--${s.value} ${form.severity === s.value ? 'selected' : ''}`}
                    onClick={() => set('severity', s.value)}
                    title={s.definition}
                  >
                    <span className="ae-severity-label">{s.label}</span>
                    <span className="ae-severity-def">{s.definition}</span>
                  </button>
                ))}
              </div>
            </div>
            <label className="ae-field">
              {t('adverseWizard.aiFinding', 'Contributing AI finding')}
              <input value={form.aiFinding} onChange={(e) => set('aiFinding', e.target.value)} />
            </label>
          </div>
        )}

        {step === 2 && (
          <div className="ae-fields">
            <label className="ae-field">
              {t('adverseWizard.outcome', 'Patient outcome')}
              <textarea rows={3} value={form.patientOutcome} onChange={(e) => set('patientOutcome', e.target.value)} />
            </label>
            <label className="ae-field">
              {t('adverseWizard.followUp', 'Follow-up status')}
              <input value={form.followUpStatus} onChange={(e) => set('followUpStatus', e.target.value)} />
            </label>
            <label className="ae-field">
              {t('adverseWizard.harm', 'Was clinical harm prevented?')}
              <select value={form.harmPrevented} onChange={(e) => set('harmPrevented', e.target.value)}>
                <option value="">—</option>
                <option value="yes">{t('adverseWizard.yes', 'Yes')}</option>
                <option value="no">{t('adverseWizard.no', 'No')}</option>
                <option value="unknown">{t('adverseWizard.unknown', 'Unknown')}</option>
              </select>
            </label>
          </div>
        )}

        {step === 3 && (
          <div className="ae-review">
            <dl>
              <dt>{t('adverseWizard.ecgRef', 'ECG reference')}</dt><dd>{form.ecgReference}</dd>
              <dt>{t('adverseWizard.eventDate', 'Event date')}</dt><dd>{form.eventDate}</dd>
              <dt>{t('adverseWizard.severity', 'Severity')}</dt><dd>{form.severity}</dd>
              <dt>{t('adverseWizard.description', 'Description')}</dt><dd>{form.description}</dd>
              <dt>{t('adverseWizard.aiFinding', 'AI finding')}</dt><dd>{form.aiFinding || '—'}</dd>
              <dt>{t('adverseWizard.outcome', 'Outcome')}</dt><dd>{form.patientOutcome || '—'}</dd>
            </dl>
            <label className="ae-confirm">
              <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
              {t('adverseWizard.confirm', 'I confirm this report is accurate')}
            </label>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="ae-nav">
        {step > 0 && (
          <button className="ae-btn" onClick={() => setStep((s) => s - 1)}>
            {t('adverseWizard.back', 'Back')}
          </button>
        )}
        {step < STEPS.length - 1 ? (
          <button className="ae-btn ae-btn-primary" disabled={!stepValid} onClick={() => setStep((s) => s + 1)}>
            {t('adverseWizard.next', 'Next')}
          </button>
        ) : (
          <button className="ae-btn ae-btn-primary" disabled={!stepValid} onClick={submit}>
            {t('adverseWizard.submit', 'Submit Report')}
          </button>
        )}
      </div>
    </div>
  );
}
