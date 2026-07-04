import { useState, useCallback, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import './AdverseEventForm.css';

/* ---------- Types -------------------------------------------------------- */

interface AdverseEventFormProps {
  /** Pre-fill the ECG reference field (e.g. from a results page). */
  ecgReference?: string;
  /** Pre-fill the AI finding field. */
  aiFinding?: string;
  /** Called after successful submission. */
  onSubmitted?: (eventId: string) => void;
  /** Called when the form is cancelled. */
  onCancel?: () => void;
}

type Severity = 'minor' | 'moderate' | 'serious' | 'critical';

const SEVERITY_OPTIONS: { value: Severity; icon: string }[] = [
  { value: 'minor', icon: '●' },
  { value: 'moderate', icon: '▲' },
  { value: 'serious', icon: '◆' },
  { value: 'critical', icon: '⬟' },
];

/* ---------- Component ---------------------------------------------------- */

export function AdverseEventForm({
  ecgReference = '',
  aiFinding = '',
  onSubmitted,
  onCancel,
}: AdverseEventFormProps) {
  const { t } = useTranslation();
  const [reporterId, setReporterId] = useState('');
  const [ecgRef, setEcgRef] = useState(ecgReference);
  const [description, setDescription] = useState('');
  const [severity, setSeverity] = useState<Severity | null>(null);
  const [finding, setFinding] = useState(aiFinding);
  const [patientOutcome, setPatientOutcome] = useState('');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submittedId, setSubmittedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isValid =
    reporterId.trim().length > 0 &&
    ecgRef.trim().length > 0 &&
    description.trim().length > 0 &&
    severity !== null &&
    finding.trim().length > 0;

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!isValid || severity === null) return;

      setIsSubmitting(true);
      setError(null);

      const payload = {
        reporter_id: reporterId.trim(),
        ecg_reference: ecgRef.trim(),
        event_description: description.trim(),
        severity,
        ai_finding: finding.trim(),
        patient_outcome: patientOutcome.trim(),
      };

      try {
        const response = await fetch('/api/v1/validation/adverse-event', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({ detail: 'Server error' }));
          throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        setSubmittedId(data.id || 'unknown');
        onSubmitted?.(data.id);
      } catch (err) {
        // Offline fallback — store locally
        const stored = JSON.parse(
          localStorage.getItem('aortica_pending_adverse_events') || '[]',
        );
        const localId = `local-${Date.now()}`;
        stored.push({ ...payload, id: localId, timestamp: new Date().toISOString() });
        localStorage.setItem('aortica_pending_adverse_events', JSON.stringify(stored));
        setSubmittedId(localId);
        onSubmitted?.(localId);
      } finally {
        setIsSubmitting(false);
      }
    },
    [reporterId, ecgRef, description, severity, finding, patientOutcome, isValid, onSubmitted],
  );

  const handleReset = useCallback(() => {
    setSubmittedId(null);
    setReporterId('');
    setEcgRef(ecgReference);
    setDescription('');
    setSeverity(null);
    setFinding(aiFinding);
    setPatientOutcome('');
    setError(null);
  }, [ecgReference, aiFinding]);

  /* ---- Success state ---- */
  if (submittedId) {
    return (
      <div className="adverse-event-form" id="adverse-event-form">
        <div className="ae-form-card ae-success">
          <div className="ae-success-icon">✓</div>
          <h3>{t('adverseEvent.submitted')}</h3>
          <p>
            {t('adverseEvent.submittedMessage').split('\n').map((line, i) => (
              <span key={i}>
                {i > 0 && <br />}
                {line}
              </span>
            ))}
          </p>
          <div className="ae-success-id">{t('adverseEvent.eventId', { id: submittedId })}</div>
          <div>
            <button className="ae-submit-btn" onClick={handleReset}>
              {t('adverseEvent.submitAnother')}
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ---- Form ---- */
  return (
    <div className="adverse-event-form" id="adverse-event-form">
      <div className="ae-form-header">
        <h2>{t('adverseEvent.title')}</h2>
        <p>{t('adverseEvent.description')}</p>
      </div>

      <form className="ae-form-card" onSubmit={handleSubmit}>
        {error && <div className="ae-error">{error}</div>}

        {/* Reporter ID */}
        <div className="ae-field">
          <label htmlFor="ae-reporter-id">
            {t('adverseEvent.reporterId')}<span className="ae-required">{t('common.required')}</span>
          </label>
          <input
            id="ae-reporter-id"
            type="text"
            value={reporterId}
            onChange={(e) => setReporterId(e.target.value)}
            placeholder={t('adverseEvent.reporterPlaceholder')}
            required
          />
        </div>

        {/* ECG Reference */}
        <div className="ae-field">
          <label htmlFor="ae-ecg-reference">
            {t('adverseEvent.ecgReference')}<span className="ae-required">{t('common.required')}</span>
          </label>
          <input
            id="ae-ecg-reference"
            type="text"
            value={ecgRef}
            onChange={(e) => setEcgRef(e.target.value)}
            placeholder={t('adverseEvent.ecgRefPlaceholder')}
            required
          />
        </div>

        {/* Severity */}
        <div className="ae-field">
          <label>
            {t('adverseEvent.severity')}<span className="ae-required">{t('common.required')}</span>
          </label>
          <div className="ae-severity-grid">
            {SEVERITY_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={`ae-severity-option severity--${opt.value}${
                  severity === opt.value ? ' selected' : ''
                }`}
                onClick={() => setSeverity(opt.value)}
                id={`ae-severity-${opt.value}`}
              >
                <span className="ae-severity-icon">{opt.icon}</span>
                <span className="ae-severity-label">{t(`adverseEvent.severityOptions.${opt.value}`)}</span>
              </button>
            ))}
          </div>
        </div>

        {/* AI Finding */}
        <div className="ae-field">
          <label htmlFor="ae-ai-finding">
            {t('adverseEvent.aiFinding')}<span className="ae-required">{t('common.required')}</span>
          </label>
          <input
            id="ae-ai-finding"
            type="text"
            value={finding}
            onChange={(e) => setFinding(e.target.value)}
            placeholder={t('adverseEvent.aiFindingPlaceholder')}
            required
          />
        </div>

        {/* Event Description */}
        <div className="ae-field">
          <label htmlFor="ae-description">
            {t('adverseEvent.eventDescription')}<span className="ae-required">{t('common.required')}</span>
          </label>
          <textarea
            id="ae-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t('adverseEvent.eventDescPlaceholder')}
            required
          />
        </div>

        {/* Patient Outcome */}
        <div className="ae-field">
          <label htmlFor="ae-patient-outcome">{t('adverseEvent.patientOutcome')}</label>
          <textarea
            id="ae-patient-outcome"
            value={patientOutcome}
            onChange={(e) => setPatientOutcome(e.target.value)}
            placeholder={t('adverseEvent.outcomePlaceholder')}
            rows={3}
          />
        </div>

        {/* Disclaimer */}
        <div className="ae-disclaimer">
          <span className="ae-disclaimer-icon">ℹ</span>
          <span className="ae-disclaimer-text">
            {t('adverseEvent.disclaimer')}
          </span>
        </div>

        {/* Actions */}
        <div className="ae-submit-row">
          {onCancel && (
            <button type="button" className="ae-cancel-btn" onClick={onCancel}>
              {t('common.cancel')}
            </button>
          )}
          <button
            type="submit"
            className="ae-submit-btn"
            disabled={!isValid || isSubmitting}
            id="ae-submit-btn"
          >
            {isSubmitting ? t('adverseEvent.submitting') : t('adverseEvent.submitReport')}
          </button>
        </div>
      </form>
    </div>
  );
}
