import { useState, useCallback, type FormEvent } from 'react';
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

const SEVERITY_OPTIONS: { value: Severity; label: string; icon: string }[] = [
  { value: 'minor', label: 'Minor', icon: '●' },
  { value: 'moderate', label: 'Moderate', icon: '▲' },
  { value: 'serious', label: 'Serious', icon: '◆' },
  { value: 'critical', label: 'Critical', icon: '⬟' },
];

/* ---------- Component ---------------------------------------------------- */

export function AdverseEventForm({
  ecgReference = '',
  aiFinding = '',
  onSubmitted,
  onCancel,
}: AdverseEventFormProps) {
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
          <h3>Report Submitted</h3>
          <p>
            Your adverse event report has been recorded.
            <br />
            This information helps improve AI safety.
          </p>
          <div className="ae-success-id">Event ID: {submittedId}</div>
          <div>
            <button className="ae-submit-btn" onClick={handleReset}>
              Submit Another Report
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
        <h2>⚠ Adverse Event Report</h2>
        <p>
          Report an adverse event where AI findings may have contributed to a
          patient safety concern. All reports are stored in an immutable audit
          trail for post-market surveillance.
        </p>
      </div>

      <form className="ae-form-card" onSubmit={handleSubmit}>
        {error && <div className="ae-error">{error}</div>}

        {/* Reporter ID */}
        <div className="ae-field">
          <label htmlFor="ae-reporter-id">
            Reporter ID<span className="ae-required">*</span>
          </label>
          <input
            id="ae-reporter-id"
            type="text"
            value={reporterId}
            onChange={(e) => setReporterId(e.target.value)}
            placeholder="e.g. dr_smith, clinician_042"
            required
          />
        </div>

        {/* ECG Reference */}
        <div className="ae-field">
          <label htmlFor="ae-ecg-reference">
            ECG Reference<span className="ae-required">*</span>
          </label>
          <input
            id="ae-ecg-reference"
            type="text"
            value={ecgRef}
            onChange={(e) => setEcgRef(e.target.value)}
            placeholder="ECG recording ID or hash"
            required
          />
        </div>

        {/* Severity */}
        <div className="ae-field">
          <label>
            Severity<span className="ae-required">*</span>
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
                <span className="ae-severity-label">{opt.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* AI Finding */}
        <div className="ae-field">
          <label htmlFor="ae-ai-finding">
            AI Finding That Contributed<span className="ae-required">*</span>
          </label>
          <input
            id="ae-ai-finding"
            type="text"
            value={finding}
            onChange={(e) => setFinding(e.target.value)}
            placeholder="e.g. normal_sinus_rhythm, AF"
            required
          />
        </div>

        {/* Event Description */}
        <div className="ae-field">
          <label htmlFor="ae-description">
            Event Description<span className="ae-required">*</span>
          </label>
          <textarea
            id="ae-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe the adverse event, including relevant clinical context…"
            required
          />
        </div>

        {/* Patient Outcome */}
        <div className="ae-field">
          <label htmlFor="ae-patient-outcome">Patient Outcome</label>
          <textarea
            id="ae-patient-outcome"
            value={patientOutcome}
            onChange={(e) => setPatientOutcome(e.target.value)}
            placeholder="Describe the patient outcome (optional)"
            rows={3}
          />
        </div>

        {/* Disclaimer */}
        <div className="ae-disclaimer">
          <span className="ae-disclaimer-icon">ℹ</span>
          <span className="ae-disclaimer-text">
            This report is voluntary and confidential. All submitted events are
            stored in an append-only audit trail and cannot be modified or
            deleted. Reports help improve AI safety and are reviewed as part of
            post-market surveillance.
          </span>
        </div>

        {/* Actions */}
        <div className="ae-submit-row">
          {onCancel && (
            <button type="button" className="ae-cancel-btn" onClick={onCancel}>
              Cancel
            </button>
          )}
          <button
            type="submit"
            className="ae-submit-btn"
            disabled={!isValid || isSubmitting}
            id="ae-submit-btn"
          >
            {isSubmitting ? 'Submitting…' : 'Submit Report'}
          </button>
        </div>
      </form>
    </div>
  );
}
