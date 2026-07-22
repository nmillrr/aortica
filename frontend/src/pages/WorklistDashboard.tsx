import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './WorklistDashboard.css';

// ---------------------------------------------------------------------------
// Types — mirror aortica.integration.worklist_store.WorklistEntry.to_dict()
// ---------------------------------------------------------------------------

interface WorklistEntry {
  ecg_id: string;
  urgency_score: number;
  urgency_tier: string;
  top_finding: string;
  recommended_action: string;
  patient_id: string | null;
  acquired_at: string | null;
  review_status: string;
  assignee: string | null;
  created_at: string | null;
  reviewed_at: string | null;
  active_findings: Array<Record<string, unknown>>;
}

interface WorklistSummary {
  total: number;
  total_pending: number;
  critical_count: number;
  completed_count: number;
  avg_time_to_review_seconds: number | null;
}

interface WorklistResponse {
  items: WorklistEntry[];
  summary: WorklistSummary;
}

const API_BASE = 'http://localhost:8000';
const POLL_INTERVAL_MS = 30_000;

const REVIEW_STATUSES = ['pending', 'in-progress', 'completed'];
const CLINICIANS = ['dr_smith', 'dr_jones', 'dr_patel', 'dr_nguyen'];

type SortKey = 'urgency_score' | 'ecg_id' | 'acquired_at' | 'top_finding' | 'review_status';

function tierBadgeClass(tier: string): string {
  return `wl-badge wl-badge-${tier}`;
}

function fmtDuration(seconds: number | null): string {
  if (seconds == null) return '—';
  const mins = Math.round(seconds / 60);
  if (mins < 60) return `${mins}m`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

export function WorklistDashboard() {
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();
  const navigate = useNavigate();

  const [data, setData] = useState<WorklistResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState('');
  const [tierFilter, setTierFilter] = useState('');
  const [findingFilter, setFindingFilter] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('urgency_score');
  const [sortAsc, setSortAsc] = useState(false);

  const fetchWorklist = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (statusFilter) params.set('status', statusFilter);
      if (tierFilter) params.set('tier', tierFilter);
      if (findingFilter) params.set('finding', findingFilter);
      const resp = await fetch(`${API_BASE}/api/v1/worklist?${params}`, {
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setData(await resp.json());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [statusFilter, tierFilter, findingFilter, getAuthHeader]);

  // Poll on an interval for new ECGs.
  useEffect(() => {
    fetchWorklist();
    const id = setInterval(fetchWorklist, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [fetchWorklist]);

  const patchEntry = useCallback(
    async (ecgId: string, patch: { review_status?: string; assignee?: string }) => {
      try {
        const resp = await fetch(`${API_BASE}/api/v1/worklist/${ecgId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
          body: JSON.stringify(patch),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        await fetchWorklist();
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [fetchWorklist, getAuthHeader],
  );

  const sortedItems = useMemo(() => {
    const items = [...(data?.items ?? [])];
    items.sort((a, b) => {
      const av = a[sortKey] ?? '';
      const bv = b[sortKey] ?? '';
      let cmp = 0;
      if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv;
      else cmp = String(av).localeCompare(String(bv));
      return sortAsc ? cmp : -cmp;
    });
    return items;
  }, [data, sortKey, sortAsc]);

  const findingOptions = useMemo(() => {
    const set = new Set<string>();
    (data?.items ?? []).forEach((i) => set.add(i.top_finding));
    return [...set].sort();
  }, [data]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setSortAsc((s) => !s);
    else {
      setSortKey(key);
      setSortAsc(key === 'ecg_id' || key === 'top_finding');
    }
  };

  const summary = data?.summary;

  return (
    <div className="worklist-dashboard" id="worklist-page">
      <header className="wl-header">
        <h1>{t('worklist.title', 'ECG Worklist')}</h1>
      </header>

      {/* Summary bar */}
      {summary && (
        <section className="wl-summary-bar">
          <div className="wl-summary-stat">
            <span className="wl-summary-value">{summary.total_pending}</span>
            <span className="wl-summary-label">{t('worklist.pending', 'Pending')}</span>
          </div>
          <div className="wl-summary-stat wl-summary-critical">
            <span className="wl-summary-value">{summary.critical_count}</span>
            <span className="wl-summary-label">{t('worklist.critical', 'Critical')}</span>
          </div>
          <div className="wl-summary-stat">
            <span className="wl-summary-value">{summary.completed_count}</span>
            <span className="wl-summary-label">{t('worklist.completed', 'Completed')}</span>
          </div>
          <div className="wl-summary-stat">
            <span className="wl-summary-value">
              {fmtDuration(summary.avg_time_to_review_seconds)}
            </span>
            <span className="wl-summary-label">
              {t('worklist.avgReview', 'Avg. time-to-review')}
            </span>
          </div>
        </section>
      )}

      {/* Filters */}
      <section className="wl-filters">
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="">{t('worklist.allStatuses', 'All statuses')}</option>
          {REVIEW_STATUSES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
          <option value="">{t('worklist.allTiers', 'All tiers')}</option>
          <option value="critical">critical</option>
          <option value="high">high</option>
          <option value="moderate">moderate</option>
          <option value="low">low</option>
        </select>
        <select value={findingFilter} onChange={(e) => setFindingFilter(e.target.value)}>
          <option value="">{t('worklist.allFindings', 'All findings')}</option>
          {findingOptions.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </section>

      {error && <div className="wl-error" role="alert">{error}</div>}

      {/* Table */}
      <div className="wl-table-scroll">
        <table className="wl-table">
          <thead>
            <tr>
              <th onClick={() => toggleSort('urgency_score')} className="wl-sortable">
                {t('worklist.urgency', 'Urgency')}
              </th>
              <th onClick={() => toggleSort('ecg_id')} className="wl-sortable">ECG ID</th>
              <th onClick={() => toggleSort('acquired_at')} className="wl-sortable">
                {t('worklist.acquired', 'Acquired')}
              </th>
              <th>{t('worklist.patient', 'Patient')}</th>
              <th onClick={() => toggleSort('top_finding')} className="wl-sortable">
                {t('worklist.topFinding', 'Top finding')}
              </th>
              <th>{t('worklist.action', 'Recommended action')}</th>
              <th onClick={() => toggleSort('review_status')} className="wl-sortable">
                {t('worklist.status', 'Status')}
              </th>
              <th>{t('worklist.actions', 'Actions')}</th>
            </tr>
          </thead>
          <tbody>
            {sortedItems.map((entry) => (
              <tr
                key={entry.ecg_id}
                className={
                  entry.urgency_score >= 80 ? 'wl-row-critical' : ''
                }
                onClick={() => navigate(`/results/${entry.ecg_id}`)}
              >
                <td>
                  <span className={tierBadgeClass(entry.urgency_tier)}>
                    {entry.urgency_score}
                    {entry.urgency_score >= 80 && (
                      <span className="wl-pulse" aria-hidden="true" />
                    )}
                  </span>
                </td>
                <td>{entry.ecg_id}</td>
                <td>{entry.acquired_at ?? '—'}</td>
                <td>{entry.patient_id ?? '—'}</td>
                <td>{entry.top_finding}</td>
                <td>{entry.recommended_action}</td>
                <td>
                  <span className={`wl-status wl-status-${entry.review_status}`}>
                    {entry.review_status}
                  </span>
                  {entry.assignee && (
                    <span className="wl-assignee"> · {entry.assignee}</span>
                  )}
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  <div className="wl-actions">
                    <button
                      className="wl-action-btn"
                      onClick={() => patchEntry(entry.ecg_id, { review_status: 'completed' })}
                      disabled={entry.review_status === 'completed'}
                    >
                      {t('worklist.markReviewed', 'Mark Reviewed')}
                    </button>
                    <select
                      className="wl-assign-select"
                      value={entry.assignee ?? ''}
                      onChange={(e) => patchEntry(entry.ecg_id, { assignee: e.target.value })}
                    >
                      <option value="">{t('worklist.assignTo', 'Assign to…')}</option>
                      {CLINICIANS.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                    <button
                      className="wl-action-btn"
                      onClick={() => navigate(`/results/${entry.ecg_id}`)}
                    >
                      {t('worklist.generateReport', 'Generate Report')}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {sortedItems.length === 0 && (
              <tr>
                <td colSpan={8} className="wl-empty">
                  {t('worklist.empty', 'No ECGs in the worklist.')}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
