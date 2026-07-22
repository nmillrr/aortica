import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import './FLDashboard.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CampaignStatus {
  campaign_name: string;
  current_round: number;
  total_rounds: number;
  strategy: string;
  start_timestamp: number;
  elapsed_seconds: number;
  status: string;
  convergence: ConvergenceData | null;
}

interface ConvergenceData {
  gradient_norms: number[];
  loss_trend: (number | null)[];
  plateau_detected: boolean;
  early_stop_recommended: boolean;
  plateau_window: number;
}

interface RoundMetrics {
  round_number: number;
  loss: number | null;
  metrics: Record<string, number>;
  num_clients: number;
  gradient_norm: number | null;
  timestamp: number;
}

interface SiteParticipation {
  site_id: string;
  status: string;
  samples_contributed: number;
  last_communication: number;
  local_training_time_ms: number;
  epsilon_spent: number;
  epsilon_budget_pct: number;
}

interface SitesData {
  sites: SiteParticipation[];
  total: number;
  epsilon_budget: number;
}

type Tab = 'overview' | 'rounds' | 'sites' | 'privacy' | 'convergence';

const API_BASE = 'http://localhost:8000';
const POLL_INTERVAL_MS = 30_000; // 30s default

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTimestamp(ts: number): string {
  if (ts <= 0) return '—';
  return new Date(ts * 1000).toLocaleString();
}

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function strategyLabel(strategy: string): string {
  const labels: Record<string, string> = {
    fedavg: 'FedAvg',
    fedprox: 'FedProx',
    scaffold: 'SCAFFOLD',
  };
  return labels[strategy] || strategy;
}

function statusBadgeClass(status: string): string {
  switch (status) {
    case 'running': return 'badge-running';
    case 'completed': return 'badge-completed';
    case 'failed': return 'badge-failed';
    case 'online': return 'badge-online';
    case 'offline': return 'badge-offline';
    default: return 'badge-idle';
  }
}

// ---------------------------------------------------------------------------
// Simple SVG Line Chart
// ---------------------------------------------------------------------------

function MiniLineChart({
  data,
  width = 600,
  height = 200,
  color = 'var(--color-accent)',
  label = '',
}: {
  data: (number | null)[];
  width?: number;
  height?: number;
  color?: string;
  label?: string;
}) {
  const numericData = data.map((v, i) => ({ x: i, y: v })).filter((d): d is { x: number; y: number } => d.y !== null);
  if (numericData.length < 2) {
    return <div className="fl-chart-empty">Not enough data for chart</div>;
  }

  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const minY = Math.min(...numericData.map(d => d.y));
  const maxY = Math.max(...numericData.map(d => d.y));
  const rangeY = maxY - minY || 1;

  const scaleX = (x: number) => padding.left + (x / (data.length - 1)) * chartW;
  const scaleY = (y: number) => padding.top + chartH - ((y - minY) / rangeY) * chartH;

  const pathData = numericData
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.x)} ${scaleY(d.y)}`)
    .join(' ');

  // Y-axis ticks (5 ticks)
  const yTicks = Array.from({ length: 5 }, (_, i) => minY + (rangeY * i) / 4);

  return (
    <div className="fl-chart-container">
      {label && <div className="fl-chart-label">{label}</div>}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="fl-chart-svg">
        {/* Grid lines */}
        {yTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={scaleY(tick)}
              x2={width - padding.right}
              y2={scaleY(tick)}
              stroke="var(--color-surface-border)"
              strokeDasharray="4 4"
              opacity={0.5}
            />
            <text
              x={padding.left - 8}
              y={scaleY(tick) + 4}
              textAnchor="end"
              fill="var(--color-text-tertiary)"
              fontSize="11"
            >
              {tick.toFixed(3)}
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {numericData.filter((_, i) => i % Math.max(1, Math.floor(numericData.length / 8)) === 0).map(d => (
          <text
            key={d.x}
            x={scaleX(d.x)}
            y={height - 8}
            textAnchor="middle"
            fill="var(--color-text-tertiary)"
            fontSize="11"
          >
            R{d.x + 1}
          </text>
        ))}

        {/* Area fill */}
        <path
          d={`${pathData} L ${scaleX(numericData[numericData.length - 1].x)} ${padding.top + chartH} L ${scaleX(numericData[0].x)} ${padding.top + chartH} Z`}
          fill={color}
          opacity={0.08}
        />

        {/* Line */}
        <path d={pathData} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />

        {/* Dots */}
        {numericData.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.x)}
            cy={scaleY(d.y)}
            r={3}
            fill={color}
            opacity={0.8}
          >
            <title>Round {d.x + 1}: {d.y.toFixed(4)}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FLDashboard() {
  const { getAuthHeader } = useAuth();
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollInterval, setPollInterval] = useState(POLL_INTERVAL_MS);

  // Data state
  const [campaign, setCampaign] = useState<CampaignStatus | null>(null);
  const [rounds, setRounds] = useState<RoundMetrics[]>([]);
  const [sitesData, setSitesData] = useState<SitesData | null>(null);

  // ---- Fetch functions --------------------------------------------------

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const headers = getAuthHeader();
      const [statusRes, roundsRes, sitesRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/federated/status`, { headers }),
        fetch(`${API_BASE}/api/v1/federated/rounds`, { headers }),
        fetch(`${API_BASE}/api/v1/federated/sites`, { headers }),
      ]);

      if (statusRes.ok) setCampaign(await statusRes.json());
      if (roundsRes.ok) {
        const data = await roundsRes.json();
        setRounds(data.rounds || []);
      }
      if (sitesRes.ok) setSitesData(await sitesRes.json());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch FL data');
    } finally {
      setLoading(false);
    }
  }, [getAuthHeader]);

  // Initial fetch
  useEffect(() => { fetchAll(); }, [fetchAll]);

  // Polling
  useEffect(() => {
    if (campaign?.status !== 'running') return;
    const id = setInterval(fetchAll, pollInterval);
    return () => clearInterval(id);
  }, [campaign?.status, pollInterval, fetchAll]);

  // ---- Derived data -----------------------------------------------------

  const metricKeys = rounds.length > 0
    ? [...new Set(rounds.flatMap(r => Object.keys(r.metrics)))]
    : [];

  const lossData = rounds.map(r => r.loss);
  const gradNormData = rounds.map(r => r.gradient_norm);

  const tabs: { key: Tab; label: string; icon: string }[] = [
    { key: 'overview', label: 'Overview', icon: '📊' },
    { key: 'rounds', label: 'Round Metrics', icon: '📈' },
    { key: 'sites', label: 'Sites', icon: '🏥' },
    { key: 'privacy', label: 'Privacy Budget', icon: '🔒' },
    { key: 'convergence', label: 'Convergence', icon: '🎯' },
  ];

  // ---- Render -----------------------------------------------------------

  return (
    <div className="fl-dashboard" id="fl-dashboard">
      {/* Header */}
      <div className="fl-header">
        <h1>
          <span className="page-icon">🌐</span>
          {t('federated.title', 'Federated Learning')}
        </h1>
        <div className="fl-header-actions">
          <select
            className="fl-poll-select"
            value={pollInterval}
            onChange={e => setPollInterval(Number(e.target.value))}
            title="Polling interval"
          >
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
            <option value={60000}>60s</option>
            <option value={300000}>5m</option>
          </select>
          <button
            className={`fl-refresh-btn ${loading ? 'loading' : ''}`}
            onClick={fetchAll}
            disabled={loading}
            id="fl-refresh-btn"
          >
            ↻ {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="fl-error-banner" role="alert">
          <span className="fl-error-icon">⚠</span> {error}
          <button className="fl-error-dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Tabs */}
      <div className="fl-tabs" role="tablist">
        {tabs.map(tab => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            className={`fl-tab ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
            id={`fl-tab-${tab.key}`}
          >
            <span className="fl-tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="fl-tab-content">
        {activeTab === 'overview' && (
          <OverviewPanel campaign={campaign} rounds={rounds} sitesData={sitesData} lossData={lossData} />
        )}
        {activeTab === 'rounds' && (
          <RoundsPanel rounds={rounds} lossData={lossData} metricKeys={metricKeys} />
        )}
        {activeTab === 'sites' && (
          <SitesPanel sitesData={sitesData} />
        )}
        {activeTab === 'privacy' && (
          <PrivacyPanel sitesData={sitesData} />
        )}
        {activeTab === 'convergence' && (
          <ConvergencePanel campaign={campaign} gradNormData={gradNormData} lossData={lossData} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab Panels
// ---------------------------------------------------------------------------

function OverviewPanel({
  campaign,
  sitesData,
  lossData,
}: {
  campaign: CampaignStatus | null;
  rounds: RoundMetrics[];
  sitesData: SitesData | null;
  lossData: (number | null)[];
}) {
  if (!campaign) return <div className="fl-empty">No campaign data available</div>;

  const onlineSites = sitesData?.sites.filter(s => s.status === 'online').length ?? 0;
  const totalSites = sitesData?.total ?? 0;
  const progress = campaign.total_rounds > 0
    ? Math.round((campaign.current_round / campaign.total_rounds) * 100)
    : 0;

  return (
    <div className="fl-overview">
      {/* Campaign status card */}
      <div className="fl-card fl-campaign-card">
        <div className="fl-card-header">
          <h3>Campaign</h3>
          <span className={`fl-status-badge ${statusBadgeClass(campaign.status)}`}>
            {campaign.status}
          </span>
        </div>
        <div className="fl-campaign-name">{campaign.campaign_name}</div>
        <div className="fl-stat-grid">
          <div className="fl-stat">
            <div className="fl-stat-value">{campaign.current_round} / {campaign.total_rounds}</div>
            <div className="fl-stat-label">Rounds</div>
          </div>
          <div className="fl-stat">
            <div className="fl-stat-value">{strategyLabel(campaign.strategy)}</div>
            <div className="fl-stat-label">Strategy</div>
          </div>
          <div className="fl-stat">
            <div className="fl-stat-value">{formatElapsed(campaign.elapsed_seconds)}</div>
            <div className="fl-stat-label">Elapsed</div>
          </div>
          <div className="fl-stat">
            <div className="fl-stat-value">{onlineSites} / {totalSites}</div>
            <div className="fl-stat-label">Sites Online</div>
          </div>
        </div>
        {/* Progress bar */}
        <div className="fl-progress-container">
          <div className="fl-progress-bar">
            <div className="fl-progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="fl-progress-label">{progress}%</span>
        </div>
      </div>

      {/* Loss chart */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Loss Trend</h3>
        </div>
        <MiniLineChart data={lossData} label="Aggregated Loss" />
      </div>

      {/* Quick convergence indicators */}
      {campaign.convergence && (
        <div className="fl-card">
          <div className="fl-card-header">
            <h3>Convergence</h3>
          </div>
          <div className="fl-convergence-badges">
            <span className={`fl-conv-badge ${campaign.convergence.plateau_detected ? 'warning' : 'ok'}`}>
              {campaign.convergence.plateau_detected ? '⚠ Plateau Detected' : '✓ Converging'}
            </span>
            {campaign.convergence.early_stop_recommended && (
              <span className="fl-conv-badge warning">
                ⏹ Early Stop Recommended
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function RoundsPanel({
  rounds,
  lossData,
  metricKeys,
}: {
  rounds: RoundMetrics[];
  lossData: (number | null)[];
  metricKeys: string[];
}) {
  return (
    <div className="fl-rounds">
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Loss per Round</h3>
        </div>
        <MiniLineChart data={lossData} label="Loss" />
      </div>

      {/* Per-metric charts */}
      {metricKeys.map(key => (
        <div className="fl-card" key={key}>
          <div className="fl-card-header">
            <h3>{key.replace(/_/g, ' ')}</h3>
          </div>
          <MiniLineChart
            data={rounds.map(r => r.metrics[key] ?? null)}
            label={key}
            color="var(--color-info)"
          />
        </div>
      ))}

      {/* Rounds table */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Round Details</h3>
        </div>
        <div className="fl-table-wrapper">
          <table className="fl-table" id="fl-rounds-table">
            <thead>
              <tr>
                <th>Round</th>
                <th>Loss</th>
                <th>Clients</th>
                <th>Grad Norm</th>
                {metricKeys.map(k => <th key={k}>{k}</th>)}
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {rounds.map(r => (
                <tr key={r.round_number}>
                  <td className="fl-td-round">{r.round_number}</td>
                  <td>{r.loss?.toFixed(4) ?? '—'}</td>
                  <td>{r.num_clients}</td>
                  <td>{r.gradient_norm?.toFixed(4) ?? '—'}</td>
                  {metricKeys.map(k => (
                    <td key={k}>{r.metrics[k]?.toFixed(4) ?? '—'}</td>
                  ))}
                  <td className="fl-td-ts">{formatTimestamp(r.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SitesPanel({ sitesData }: { sitesData: SitesData | null }) {
  const sites = sitesData?.sites ?? [];

  return (
    <div className="fl-sites">
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Site Participation</h3>
          <span className="fl-card-badge">{sitesData?.total ?? 0} sites</span>
        </div>
        <div className="fl-table-wrapper">
          <table className="fl-table" id="fl-sites-table">
            <thead>
              <tr>
                <th>Site ID</th>
                <th>Status</th>
                <th>Samples</th>
                <th>Last Communication</th>
                <th>Training Time</th>
                <th>ε Spent</th>
                <th>ε Budget %</th>
              </tr>
            </thead>
            <tbody>
              {sites.map(site => (
                <tr key={site.site_id}>
                  <td className="fl-td-site-id">{site.site_id}</td>
                  <td>
                    <span className={`fl-status-dot ${statusBadgeClass(site.status)}`} />
                    {site.status}
                  </td>
                  <td>{site.samples_contributed.toLocaleString()}</td>
                  <td className="fl-td-ts">{formatTimestamp(site.last_communication)}</td>
                  <td>{(site.local_training_time_ms / 1000).toFixed(1)}s</td>
                  <td>{site.epsilon_spent.toFixed(3)}</td>
                  <td>
                    <div className="fl-epsilon-bar-container">
                      <div
                        className={`fl-epsilon-bar ${site.epsilon_budget_pct > 80 ? 'critical' : site.epsilon_budget_pct > 50 ? 'warning' : ''}`}
                        style={{ width: `${Math.min(site.epsilon_budget_pct, 100)}%` }}
                      />
                      <span className="fl-epsilon-label">{site.epsilon_budget_pct.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
              {sites.length === 0 && (
                <tr>
                  <td colSpan={7} className="fl-td-empty">No sites connected</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PrivacyPanel({ sitesData }: { sitesData: SitesData | null }) {
  const sites = sitesData?.sites ?? [];
  const budget = sitesData?.epsilon_budget ?? 1.0;

  // Projected epsilon: extrapolate from current spend rate
  const totalSpent = sites.reduce((sum, s) => sum + s.epsilon_spent, 0);
  const avgSpent = sites.length > 0 ? totalSpent / sites.length : 0;

  return (
    <div className="fl-privacy">
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Privacy Budget Overview</h3>
        </div>
        <div className="fl-stat-grid">
          <div className="fl-stat">
            <div className="fl-stat-value">{budget.toFixed(2)}</div>
            <div className="fl-stat-label">Total ε Budget</div>
          </div>
          <div className="fl-stat">
            <div className="fl-stat-value">{avgSpent.toFixed(3)}</div>
            <div className="fl-stat-label">Avg ε Spent</div>
          </div>
          <div className="fl-stat">
            <div className="fl-stat-value">{sites.filter(s => s.epsilon_budget_pct > 80).length}</div>
            <div className="fl-stat-label">Sites Near Exhaustion</div>
          </div>
        </div>
      </div>

      {/* Per-site privacy bars */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Per-Site ε Consumption</h3>
        </div>
        <div className="fl-privacy-bars">
          {sites.map(site => (
            <div key={site.site_id} className="fl-privacy-row">
              <span className="fl-privacy-site-label">{site.site_id}</span>
              <div className="fl-privacy-bar-track">
                <div
                  className={`fl-privacy-bar-fill ${
                    site.epsilon_budget_pct > 80 ? 'critical' :
                    site.epsilon_budget_pct > 50 ? 'warning' : ''
                  }`}
                  style={{ width: `${Math.min(site.epsilon_budget_pct, 100)}%` }}
                />
                {/* 80% threshold line */}
                <div className="fl-privacy-threshold" style={{ left: '80%' }} />
              </div>
              <span className="fl-privacy-value">
                {site.epsilon_spent.toFixed(3)} / {budget.toFixed(2)}
              </span>
              {site.epsilon_budget_pct > 80 && (
                <span className="fl-privacy-warn" title="Approaching budget exhaustion">⚠</span>
              )}
            </div>
          ))}
          {sites.length === 0 && (
            <div className="fl-empty">No site privacy data available</div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConvergencePanel({
  campaign,
  gradNormData,
  lossData,
}: {
  campaign: CampaignStatus | null;
  gradNormData: (number | null)[];
  lossData: (number | null)[];
}) {
  const convergence = campaign?.convergence;

  return (
    <div className="fl-convergence">
      {/* Indicators */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Convergence Status</h3>
        </div>
        <div className="fl-convergence-indicators">
          <div className={`fl-indicator ${convergence?.plateau_detected ? 'warning' : 'ok'}`}>
            <div className="fl-indicator-icon">
              {convergence?.plateau_detected ? '⚠' : '✓'}
            </div>
            <div className="fl-indicator-text">
              <div className="fl-indicator-title">
                {convergence?.plateau_detected ? 'Loss Plateau Detected' : 'Loss Still Decreasing'}
              </div>
              <div className="fl-indicator-detail">
                Window: {convergence?.plateau_window ?? 5} rounds
              </div>
            </div>
          </div>
          <div className={`fl-indicator ${convergence?.early_stop_recommended ? 'warning' : 'ok'}`}>
            <div className="fl-indicator-icon">
              {convergence?.early_stop_recommended ? '⏹' : '▶'}
            </div>
            <div className="fl-indicator-text">
              <div className="fl-indicator-title">
                {convergence?.early_stop_recommended ? 'Early Stopping Recommended' : 'Continue Training'}
              </div>
              <div className="fl-indicator-detail">
                Based on loss plateau detection
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Gradient norm chart */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Gradient Norm Trend</h3>
        </div>
        <MiniLineChart
          data={gradNormData}
          label="Avg Gradient L2 Norm"
          color="var(--color-warning)"
        />
      </div>

      {/* Loss chart */}
      <div className="fl-card">
        <div className="fl-card-header">
          <h3>Loss Trend</h3>
        </div>
        <MiniLineChart data={lossData} label="Aggregated Loss" />
      </div>
    </div>
  );
}
