import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import './AdminDashboard.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface UserRecord {
  id: string;
  email: string | null;
  name: string | null;
  role: string;
  last_login: number | null;
  status: string;
  provider: string;
}

interface APIKeyRecord {
  key_id: string;
  name: string;
  user_sub: string;
  created_at: number;
  last_used: number | null;
}

interface SystemHealth {
  status: string;
  model_version: string | null;
  model_loaded: boolean;
  database_size_bytes: number;
  total_ecgs_processed: number;
  uptime_seconds: number;
  onnx_runtime_available: boolean;
  sync_engine_status: string;
}

interface ActivityLogEntry {
  timestamp: number;
  user: string | null;
  endpoint: string;
  method: string;
  status_code: number;
  duration_ms: number | null;
}

type Tab = 'users' | 'keys' | 'health' | 'activity';

const API_BASE = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTimestamp(ts: number | null): string {
  if (ts == null) return '—';
  return new Date(ts * 1000).toLocaleString();
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function statusCodeClass(code: number): string {
  if (code >= 200 && code < 300) return 'code-2xx';
  if (code >= 400 && code < 500) return 'code-4xx';
  return 'code-5xx';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AdminDashboard() {
  const { getAuthHeader } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('users');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [apiKeys, setApiKeys] = useState<APIKeyRecord[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [activityLog, setActivityLog] = useState<ActivityLogEntry[]>([]);
  const [activityTotal, setActivityTotal] = useState(0);
  const [activityPage, setActivityPage] = useState(1);
  const activityPageSize = 50;

  // ── Data fetching ─────────────────────────────────────────

  const headers = getAuthHeader();

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/users`, { headers });
      if (!resp.ok) {
        if (resp.status === 403) throw new Error('Admin access required');
        throw new Error(`Failed to load users (${resp.status})`);
      }
      const data = await resp.json();
      setUsers(data.users || []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load users');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchApiKeys = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/api-keys`, { headers });
      if (!resp.ok) throw new Error(`Failed to load API keys (${resp.status})`);
      const data = await resp.json();
      setApiKeys(data.keys || []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load API keys');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/system-health`, { headers });
      if (!resp.ok) throw new Error(`Failed to load system health (${resp.status})`);
      const data = await resp.json();
      setHealth(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load system health');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchActivityLog = useCallback(async (page: number = 1) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(
        `${API_BASE}/api/v1/admin/activity-log?page=${page}&page_size=${activityPageSize}`,
        { headers }
      );
      if (!resp.ok) throw new Error(`Failed to load activity log (${resp.status})`);
      const data = await resp.json();
      setActivityLog(data.entries || []);
      setActivityTotal(data.total || 0);
      setActivityPage(data.page || 1);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load activity log');
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Tab switching triggers data loading ────────────────────

  useEffect(() => {
    switch (activeTab) {
      case 'users':    fetchUsers(); break;
      case 'keys':     fetchApiKeys(); break;
      case 'health':   fetchHealth(); break;
      case 'activity': fetchActivityLog(1); break;
    }
  }, [activeTab]);

  // ── Actions ────────────────────────────────────────────────

  const handleRoleChange = async (userId: string, newRole: string) => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/users/${userId}`, {
        method: 'PATCH',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole }),
      });
      if (!resp.ok) throw new Error('Failed to update role');
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to update role');
    }
  };

  const handleStatusToggle = async (userId: string, currentStatus: string) => {
    const newStatus = currentStatus === 'active' ? 'disabled' : 'active';
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/users/${userId}`, {
        method: 'PATCH',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus }),
      });
      if (!resp.ok) throw new Error('Failed to update status');
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, status: newStatus } : u));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to update status');
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/admin/api-keys/${keyId}`, {
        method: 'DELETE',
        headers,
      });
      if (!resp.ok) throw new Error('Failed to revoke key');
      setApiKeys(prev => prev.filter(k => k.key_id !== keyId));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to revoke key');
    }
  };

  const handleRefresh = () => {
    switch (activeTab) {
      case 'users':    fetchUsers(); break;
      case 'keys':     fetchApiKeys(); break;
      case 'health':   fetchHealth(); break;
      case 'activity': fetchActivityLog(activityPage); break;
    }
  };

  // ── Tab definitions ────────────────────────────────────────

  const tabs: { key: Tab; icon: string; label: string; count?: number }[] = [
    { key: 'users',    icon: '👥', label: 'Users',        count: users.length },
    { key: 'keys',     icon: '🔑', label: 'API Keys',     count: apiKeys.length },
    { key: 'health',   icon: '💚', label: 'System Health' },
    { key: 'activity', icon: '📊', label: 'Activity Log',  count: activityTotal },
  ];

  // ── Render ─────────────────────────────────────────────────

  return (
    <div className="admin-dashboard" id="admin-dashboard">
      {/* Header */}
      <div className="admin-header">
        <h1>
          <span className="page-icon">⚙</span>
          Admin Dashboard
        </h1>
        <button
          className={`admin-refresh-btn ${loading ? 'loading' : ''}`}
          onClick={handleRefresh}
          disabled={loading}
          id="admin-refresh-btn"
        >
          ↻ Refresh
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="admin-error" id="admin-error">
          <span>⚠</span>
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="admin-tabs" id="admin-tabs">
        {tabs.map(tab => (
          <button
            key={tab.key}
            className={`admin-tab ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
            id={`admin-tab-${tab.key}`}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
            {tab.count !== undefined && tab.count > 0 && (
              <span className="tab-badge">{tab.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* Panel content */}
      {activeTab === 'users' && (
        <UsersPanel
          users={users}
          loading={loading}
          onRoleChange={handleRoleChange}
          onStatusToggle={handleStatusToggle}
        />
      )}
      {activeTab === 'keys' && (
        <APIKeysPanel
          keys={apiKeys}
          loading={loading}
          onRevoke={handleRevokeKey}
        />
      )}
      {activeTab === 'health' && (
        <SystemHealthPanel health={health} loading={loading} />
      )}
      {activeTab === 'activity' && (
        <ActivityLogPanel
          entries={activityLog}
          loading={loading}
          total={activityTotal}
          page={activityPage}
          pageSize={activityPageSize}
          onPageChange={(p) => fetchActivityLog(p)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-panels
// ---------------------------------------------------------------------------

function UsersPanel({
  users, loading, onRoleChange, onStatusToggle,
}: {
  users: UserRecord[];
  loading: boolean;
  onRoleChange: (id: string, role: string) => void;
  onStatusToggle: (id: string, status: string) => void;
}) {
  if (loading) return <LoadingState />;

  return (
    <div className="admin-panel" id="admin-users-panel">
      <div className="admin-panel-header">
        <h2 className="admin-panel-title">
          👥 User Management
        </h2>
        <span style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--font-size-sm)' }}>
          {users.length} registered user{users.length !== 1 ? 's' : ''}
        </span>
      </div>

      {users.length === 0 ? (
        <EmptyState icon="👥" text="No users registered yet" />
      ) : (
        <div className="admin-table-container">
          <table className="admin-table" id="admin-users-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Email</th>
                <th>Role</th>
                <th>Provider</th>
                <th>Last Login</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map(user => (
                <tr key={user.id} id={`admin-user-row-${user.id}`}>
                  <td>
                    <strong>{user.name || user.id}</strong>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                      {user.id}
                    </div>
                  </td>
                  <td>{user.email || '—'}</td>
                  <td>
                    <select
                      className="admin-role-select"
                      value={user.role}
                      onChange={(e) => onRoleChange(user.id, e.target.value)}
                      id={`admin-role-select-${user.id}`}
                    >
                      <option value="admin">Admin</option>
                      <option value="clinician">Clinician</option>
                      <option value="researcher">Researcher</option>
                    </select>
                  </td>
                  <td>
                    <span className="admin-badge role-clinician">
                      {user.provider}
                    </span>
                  </td>
                  <td>{formatTimestamp(user.last_login)}</td>
                  <td>
                    <span className={`admin-badge status-${user.status}`}>
                      {user.status}
                    </span>
                  </td>
                  <td>
                    <button
                      className={`admin-action-btn ${user.status === 'active' ? 'danger' : ''}`}
                      onClick={() => onStatusToggle(user.id, user.status)}
                      id={`admin-status-toggle-${user.id}`}
                    >
                      {user.status === 'active' ? 'Disable' : 'Enable'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function APIKeysPanel({
  keys, loading, onRevoke,
}: {
  keys: APIKeyRecord[];
  loading: boolean;
  onRevoke: (keyId: string) => void;
}) {
  if (loading) return <LoadingState />;

  return (
    <div className="admin-panel" id="admin-keys-panel">
      <div className="admin-panel-header">
        <h2 className="admin-panel-title">
          🔑 API Key Management
        </h2>
        <span style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--font-size-sm)' }}>
          {keys.length} active key{keys.length !== 1 ? 's' : ''}
        </span>
      </div>

      {keys.length === 0 ? (
        <EmptyState icon="🔑" text="No API keys issued" />
      ) : (
        <div className="admin-table-container">
          <table className="admin-table" id="admin-keys-table">
            <thead>
              <tr>
                <th>Key ID</th>
                <th>Name</th>
                <th>User</th>
                <th>Created</th>
                <th>Last Used</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {keys.map(key => (
                <tr key={key.key_id} id={`admin-key-row-${key.key_id}`}>
                  <td>
                    <code style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
                      {key.key_id.substring(0, 12)}…
                    </code>
                  </td>
                  <td>{key.name}</td>
                  <td>{key.user_sub}</td>
                  <td>{formatTimestamp(key.created_at)}</td>
                  <td>{formatTimestamp(key.last_used)}</td>
                  <td>
                    <button
                      className="admin-action-btn danger"
                      onClick={() => onRevoke(key.key_id)}
                      id={`admin-revoke-key-${key.key_id}`}
                    >
                      Revoke
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SystemHealthPanel({
  health, loading,
}: {
  health: SystemHealth | null;
  loading: boolean;
}) {
  if (loading || !health) return <LoadingState />;

  const healthCards = [
    {
      label: 'Server Status',
      value: health.status.toUpperCase(),
      dot: health.status === 'ok' ? 'green' : 'red',
    },
    {
      label: 'Model Loaded',
      value: health.model_loaded ? 'Active' : 'Not Loaded',
      dot: health.model_loaded ? 'green' : 'yellow',
      detail: health.model_version ? `v${health.model_version}` : undefined,
    },
    {
      label: 'ONNX Runtime',
      value: health.onnx_runtime_available ? 'Available' : 'Unavailable',
      dot: health.onnx_runtime_available ? 'green' : 'red',
    },
    {
      label: 'Sync Engine',
      value: health.sync_engine_status,
      dot: health.sync_engine_status === 'active' ? 'green' : 'yellow',
    },
    {
      label: 'Database Size',
      value: formatBytes(health.database_size_bytes),
    },
    {
      label: 'ECGs Processed',
      value: health.total_ecgs_processed.toLocaleString(),
    },
    {
      label: 'Server Uptime',
      value: formatUptime(health.uptime_seconds),
    },
  ];

  return (
    <div className="admin-panel" id="admin-health-panel">
      <div className="admin-panel-header">
        <h2 className="admin-panel-title">
          💚 System Health
        </h2>
        <span className={`admin-badge status-${health.status === 'ok' ? 'ok' : 'unavailable'}`}>
          {health.status === 'ok' ? '● Healthy' : '● Degraded'}
        </span>
      </div>

      <div className="health-grid">
        {healthCards.map((card, i) => (
          <div className="health-card" key={i} id={`health-card-${i}`}>
            <span className="health-card-label">{card.label}</span>
            <span className="health-card-value">
              {card.dot && <span className={`health-dot ${card.dot}`} />}
              {card.value}
            </span>
            {card.detail && (
              <span className="health-card-detail">{card.detail}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivityLogPanel({
  entries, loading, total, page, pageSize, onPageChange,
}: {
  entries: ActivityLogEntry[];
  loading: boolean;
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}) {
  if (loading) return <LoadingState />;

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="admin-panel" id="admin-activity-panel">
      <div className="admin-panel-header">
        <h2 className="admin-panel-title">
          📊 Activity Log
        </h2>
        <span style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--font-size-sm)' }}>
          {total} total entr{total !== 1 ? 'ies' : 'y'}
        </span>
      </div>

      {entries.length === 0 ? (
        <EmptyState icon="📊" text="No activity recorded yet" />
      ) : (
        <>
          <div className="admin-table-container">
            <table className="admin-table" id="admin-activity-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>User</th>
                  <th>Method</th>
                  <th>Endpoint</th>
                  <th>Status</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, i) => (
                  <tr key={i} id={`admin-activity-row-${i}`}>
                    <td>{formatTimestamp(entry.timestamp)}</td>
                    <td>{entry.user || '—'}</td>
                    <td>
                      <span className="admin-badge role-clinician">
                        {entry.method}
                      </span>
                    </td>
                    <td>
                      <code style={{ fontSize: 'var(--font-size-xs)' }}>
                        {entry.endpoint}
                      </code>
                    </td>
                    <td>
                      <span className={`admin-badge ${statusCodeClass(entry.status_code)}`}>
                        {entry.status_code}
                      </span>
                    </td>
                    <td>
                      {entry.duration_ms != null ? `${entry.duration_ms.toFixed(1)}ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="admin-pagination">
            <span className="admin-pagination-info">
              Page {page} of {totalPages} ({total} entries)
            </span>
            <div className="admin-pagination-controls">
              <button
                className="admin-page-btn"
                onClick={() => onPageChange(page - 1)}
                disabled={page <= 1}
                id="admin-activity-prev"
              >
                ←
              </button>
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                const pageNum = i + 1;
                return (
                  <button
                    key={pageNum}
                    className={`admin-page-btn ${page === pageNum ? 'active' : ''}`}
                    onClick={() => onPageChange(pageNum)}
                    id={`admin-activity-page-${pageNum}`}
                  >
                    {pageNum}
                  </button>
                );
              })}
              <button
                className="admin-page-btn"
                onClick={() => onPageChange(page + 1)}
                disabled={page >= totalPages}
                id="admin-activity-next"
              >
                →
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared states
// ---------------------------------------------------------------------------

function LoadingState() {
  return (
    <div className="admin-panel">
      <div className="admin-loading">
        <div className="admin-loading-spinner" />
        <div>Loading…</div>
      </div>
    </div>
  );
}

function EmptyState({ icon, text }: { icon: string; text: string }) {
  return (
    <div className="admin-empty">
      <div className="admin-empty-icon">{icon}</div>
      <p className="admin-empty-text">{text}</p>
    </div>
  );
}
