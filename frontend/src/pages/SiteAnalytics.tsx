import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './SiteAnalytics.css';

// ---------------------------------------------------------------------------
// Types — mirror GET /api/v1/analytics/sites
// ---------------------------------------------------------------------------

interface SiteMetrics {
  site_id: string;
  device_ids: string[];
  total_ecgs: number;
  mean_quality: number;
  critical_rate: number;
  finding_distribution: Record<string, number>;
  last_sync: number | null;
}

interface Anomaly {
  site_id: string;
  metric: string;
  value: number;
  fleet_mean: number;
  fleet_std: number;
  z_score: number;
}

interface AnalyticsResponse {
  sites: SiteMetrics[];
  anomalies: Anomaly[];
  fleet: { total_sites: number; total_ecgs: number; total_devices: number };
}

const API_BASE = 'http://localhost:8000';
const POLL_INTERVAL_MS = 60_000;

export function SiteAnalyticsPage() {
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/analytics/sites`, {
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setData(await resp.json());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [getAuthHeader]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [fetchData]);

  const anomalousSites = useMemo(
    () => new Set((data?.anomalies ?? []).map((a) => a.site_id)),
    [data],
  );

  return (
    <div className="site-analytics" id="site-analytics-page">
      <header className="sa-header">
        <h1>{t('siteAnalytics.title', 'Cross-Site Analytics')}</h1>
      </header>

      {error && <div className="sa-error" role="alert">{error}</div>}

      {/* Fleet summary */}
      {data && (
        <section className="sa-fleet">
          <div className="sa-fleet-stat">
            <span className="sa-fleet-value">{data.fleet.total_sites}</span>
            <span className="sa-fleet-label">{t('siteAnalytics.sites', 'Sites')}</span>
          </div>
          <div className="sa-fleet-stat">
            <span className="sa-fleet-value">{data.fleet.total_devices}</span>
            <span className="sa-fleet-label">{t('siteAnalytics.devices', 'Devices')}</span>
          </div>
          <div className="sa-fleet-stat">
            <span className="sa-fleet-value">{data.fleet.total_ecgs}</span>
            <span className="sa-fleet-label">{t('siteAnalytics.ecgs', 'ECGs synced')}</span>
          </div>
          <div className="sa-fleet-stat sa-fleet-anomaly">
            <span className="sa-fleet-value">{data.anomalies.length}</span>
            <span className="sa-fleet-label">{t('siteAnalytics.anomalies', 'Anomalies')}</span>
          </div>
        </section>
      )}

      {/* Anomaly alerts */}
      {data && data.anomalies.length > 0 && (
        <section className="sa-anomalies">
          <h2>{t('siteAnalytics.anomalyAlerts', 'Anomaly Alerts')}</h2>
          <ul>
            {data.anomalies.map((a, i) => (
              <li key={i} className="sa-anomaly-item">
                <span className="sa-anomaly-site">{a.site_id}</span>
                {t('siteAnalytics.anomalyDetail', {
                  defaultValue: '{{metric}} = {{value}} (z = {{z}}, fleet mean {{mean}})',
                  metric: a.metric,
                  value: a.value.toFixed(3),
                  z: a.z_score.toFixed(2),
                  mean: a.fleet_mean.toFixed(3),
                })}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Per-site table */}
      <section className="sa-sites">
        <h2>{t('siteAnalytics.perSite', 'Per-Site Metrics')}</h2>
        <div className="sa-table-scroll">
          <table className="sa-table">
            <thead>
              <tr>
                <th>{t('siteAnalytics.site', 'Site')}</th>
                <th>{t('siteAnalytics.devices', 'Devices')}</th>
                <th>{t('siteAnalytics.ecgs', 'ECGs')}</th>
                <th>{t('siteAnalytics.quality', 'Mean quality')}</th>
                <th>{t('siteAnalytics.critical', 'Critical rate')}</th>
                <th>{t('siteAnalytics.topFindings', 'Top findings')}</th>
                <th>{t('siteAnalytics.lastSync', 'Last sync')}</th>
              </tr>
            </thead>
            <tbody>
              {(data?.sites ?? []).map((s) => (
                <tr
                  key={s.site_id}
                  className={anomalousSites.has(s.site_id) ? 'sa-row-anomaly' : ''}
                >
                  <td>
                    {anomalousSites.has(s.site_id) && <span className="sa-flag">⚠</span>}{' '}
                    {s.site_id}
                  </td>
                  <td>{s.device_ids.length}</td>
                  <td>{s.total_ecgs}</td>
                  <td>{s.mean_quality.toFixed(1)}</td>
                  <td>{(s.critical_rate * 100).toFixed(0)}%</td>
                  <td>
                    {Object.entries(s.finding_distribution)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([name, count]) => `${name} (${count})`)
                      .join(', ') || '—'}
                  </td>
                  <td>{s.last_sync ? new Date(s.last_sync * 1000).toLocaleString() : '—'}</td>
                </tr>
              ))}
              {(data?.sites ?? []).length === 0 && (
                <tr>
                  <td colSpan={7} className="sa-empty">
                    {t('siteAnalytics.empty', 'No sites have synced data yet.')}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
