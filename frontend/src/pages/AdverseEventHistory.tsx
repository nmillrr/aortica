import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './AdverseEventWizard.css';

interface AdverseEventRecord {
  id: string;
  reporter_id: string;
  ecg_reference: string;
  event_description: string;
  severity: string;
  ai_finding: string;
  patient_outcome: string;
  timestamp: string;
}

const API_BASE = 'http://localhost:8000';

export function AdverseEventHistory() {
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();
  const [events, setEvents] = useState<AdverseEventRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchEvents = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/validation/adverse-events`, {
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setEvents(Array.isArray(data) ? data : data.events ?? []);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [getAuthHeader]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  return (
    <div className="ae-wizard" id="adverse-history">
      <header className="ae-header">
        <h1>{t('adverseWizard.historyTitle', 'Adverse Event History')}</h1>
        <Link className="ae-history-link" to="/report-event">
          {t('adverseWizard.newReport', 'New report')}
        </Link>
      </header>

      {error && <div className="ae-error" role="alert">{error}</div>}

      {events.length === 0 ? (
        <p className="ae-empty">{t('adverseWizard.noEvents', 'No reports submitted yet.')}</p>
      ) : (
        <div className="ae-table-scroll">
          <table className="ae-table">
            <thead>
              <tr>
                <th>{t('adverseWizard.id', 'ID')}</th>
                <th>{t('adverseWizard.ecgRef', 'ECG')}</th>
                <th>{t('adverseWizard.severity', 'Severity')}</th>
                <th>{t('adverseWizard.aiFinding', 'AI finding')}</th>
                <th>{t('adverseWizard.timestamp', 'Submitted')}</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id}>
                  <td>{e.id}</td>
                  <td>{e.ecg_reference}</td>
                  <td>
                    <span className={`ae-sev-pill severity--${e.severity}`}>{e.severity}</span>
                  </td>
                  <td>{e.ai_finding}</td>
                  <td>{new Date(e.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
