import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './PerformanceMonitor.css';

// ---------------------------------------------------------------------------
// Types — mirror the /monitor/metrics and /monitor/alerts responses
// ---------------------------------------------------------------------------

interface TaskMetric {
  task_name: string;
  auc: number;
  f1: number;
  ece: number;
  n_samples: number;
  baseline: { auc: number | null; f1: number | null; ece: number | null };
  trend: { auc: string; f1: string; ece: string };
}

interface MetricsResponse {
  task_metrics: Record<string, TaskMetric>;
  window_days: number;
  volume: { total_predictions: number; total_labeled: number };
  last_updated: number;
}

interface DriftAlert {
  task_name: string;
  metric_name: string;
  current_value: number;
  baseline_value: number;
  threshold: number;
  alert_type: string;
  timestamp: number;
  message: string;
}

interface AlertsResponse {
  alerts: DriftAlert[];
  has_drift: boolean;
}

const API_BASE = 'http://localhost:8000';
const REFRESH_INTERVALS = [
  { label: '1m', ms: 60_000 },
  { label: '5m', ms: 300_000 },
  { label: '15m', ms: 900_000 },
];

function trendArrow(trend: string): string {
  return trend === 'up' ? '↑' : trend === 'down' ? '↓' : '↔';
}

export function PerformanceMonitorPage() {
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshMs, setRefreshMs] = useState(300_000);

  const fetchAll = useCallback(async () => {
    try {
      const [m, a] = await Promise.all([
        fetch(`${API_BASE}/api/v1/validation/monitor/metrics`, { headers: { ...getAuthHeader() } }),
        fetch(`${API_BASE}/api/v1/validation/monitor/alerts`, { headers: { ...getAuthHeader() } }),
      ]);
      if (m.ok) setMetrics(await m.json());
      if (a.ok) setAlerts(await a.json());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [getAuthHeader]);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, refreshMs);
    return () => clearInterval(id);
  }, [fetchAll, refreshMs]);

  const tasks = useMemo(
    () => Object.values(metrics?.task_metrics ?? {}),
    [metrics],
  );

  return (
    <div className="perf-monitor" id="monitor-page">
      <header className="pm-header">
        <h1>{t('monitor.title', 'Performance Monitoring')}</h1>
        <div className="pm-refresh">
          <span>{t('monitor.refresh', 'Auto-refresh')}:</span>
          {REFRESH_INTERVALS.map((r) => (
            <button
              key={r.label}
              className={`pm-refresh-btn ${refreshMs === r.ms ? 'active' : ''}`}
              onClick={() => setRefreshMs(r.ms)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </header>

      {error && <div className="pm-error" role="alert">{error}</div>}

      {/* Volume metrics */}
      {metrics && (
        <section className="pm-volume">
          <div className="pm-vol-stat">
            <span className="pm-vol-value">{metrics.volume.total_predictions}</span>
            <span className="pm-vol-label">{t('monitor.predictions', 'Total predictions')}</span>
          </div>
          <div className="pm-vol-stat">
            <span className="pm-vol-value">{metrics.volume.total_labeled}</span>
            <span className="pm-vol-label">{t('monitor.labeled', 'Labeled')}</span>
          </div>
          <div className="pm-vol-stat">
            <span className="pm-vol-value">{metrics.window_days}d</span>
            <span className="pm-vol-label">{t('monitor.window', 'Window')}</span>
          </div>
        </section>
      )}

      {/* Metrics overview per task */}
      <section className="pm-metrics">
        <h2>{t('monitor.overview', 'Metrics Overview')}</h2>
        <div className="pm-cards">
          {tasks.length === 0 && (
            <div className="pm-empty">{t('monitor.noData', 'No monitoring data yet.')}</div>
          )}
          {tasks.map((task) => (
            <div key={task.task_name} className="pm-card">
              <h3>{task.task_name}</h3>
              <div className="pm-metric-row">
                <span>AUC</span>
                <span className="pm-metric-val">
                  {task.auc.toFixed(3)}
                  <span className={`pm-trend pm-trend-${task.trend.auc}`}>{trendArrow(task.trend.auc)}</span>
                </span>
              </div>
              <div className="pm-metric-row">
                <span>F1</span>
                <span className="pm-metric-val">
                  {task.f1.toFixed(3)}
                  <span className={`pm-trend pm-trend-${task.trend.f1}`}>{trendArrow(task.trend.f1)}</span>
                </span>
              </div>
              <div className="pm-metric-row">
                <span>ECE</span>
                <span className="pm-metric-val">
                  {task.ece.toFixed(3)}
                  <span className={`pm-trend pm-trend-${task.trend.ece}`}>{trendArrow(task.trend.ece)}</span>
                </span>
              </div>
              {task.baseline.f1 != null && (
                <div className="pm-baseline">
                  {t('monitor.baseline', 'Baseline F1')}: {task.baseline.f1.toFixed(3)}
                </div>
              )}
              <div className="pm-samples">n = {task.n_samples}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Drift alerts */}
      <section className="pm-alerts">
        <h2>
          {t('monitor.alerts', 'Drift Alerts')}
          {alerts?.has_drift && <span className="pm-alert-badge">{alerts.alerts.length}</span>}
        </h2>
        {alerts && alerts.alerts.length > 0 ? (
          <table className="pm-alert-table">
            <thead>
              <tr>
                <th>{t('monitor.task', 'Task')}</th>
                <th>{t('monitor.metric', 'Metric')}</th>
                <th>{t('monitor.current', 'Current')}</th>
                <th>{t('monitor.threshold', 'Threshold')}</th>
                <th>{t('monitor.type', 'Type')}</th>
              </tr>
            </thead>
            <tbody>
              {alerts.alerts.map((a, i) => (
                <tr key={i} className="pm-alert-row">
                  <td>{a.task_name}</td>
                  <td>{a.metric_name}</td>
                  <td>{a.current_value.toFixed(3)}</td>
                  <td>{a.threshold.toFixed(3)}</td>
                  <td>{a.alert_type}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="pm-no-drift">✓ {t('monitor.noDrift', 'No drift detected')}</p>
        )}
      </section>
    </div>
  );
}
