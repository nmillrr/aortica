import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import './SiteValidation.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SiteValidation {
  site_id: string;
  region: string;
  region_class: string;
  dataset_size: number;
  benchmark_summary: Record<string, any>;
  timestamp: string;
  overall_pass: boolean;
}

interface ReleaseReadiness {
  ready: boolean;
  total_validations: number;
  western_count: number;
  non_western_count: number;
  min_non_western: number;
  non_western_sites: string[];
  western_sites: string[];
}

const API_BASE = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Region options for the dropdown
// ---------------------------------------------------------------------------

const REGION_OPTIONS = [
  // Non-Western
  { value: 'South Asia', label: 'South Asia', classification: 'non-western' },
  { value: 'East Africa', label: 'East Africa', classification: 'non-western' },
  { value: 'West Africa', label: 'West Africa', classification: 'non-western' },
  { value: 'Southeast Asia', label: 'Southeast Asia', classification: 'non-western' },
  { value: 'East Asia', label: 'East Asia', classification: 'non-western' },
  { value: 'Latin America', label: 'Latin America', classification: 'non-western' },
  { value: 'Middle East', label: 'Middle East', classification: 'non-western' },
  { value: 'Central Asia', label: 'Central Asia', classification: 'non-western' },
  { value: 'Eastern Europe', label: 'Eastern Europe', classification: 'non-western' },
  { value: 'North Africa', label: 'North Africa', classification: 'non-western' },
  { value: 'Southern Africa', label: 'Southern Africa', classification: 'non-western' },
  { value: 'Central Africa', label: 'Central Africa', classification: 'non-western' },
  { value: 'Pacific Islands', label: 'Pacific Islands', classification: 'non-western' },
  { value: 'Caribbean', label: 'Caribbean', classification: 'non-western' },
  // Western
  { value: 'North America', label: 'North America', classification: 'western' },
  { value: 'Western Europe', label: 'Western Europe', classification: 'western' },
  { value: 'Australia', label: 'Australia/NZ', classification: 'western' },
];

// ---------------------------------------------------------------------------
// SVG World Map — simplified continent outlines with site markers
// ---------------------------------------------------------------------------

// Approximate mercator x,y coordinates for each region (relative to 800x400 viewbox)
const REGION_COORDS: Record<string, { x: number; y: number }> = {
  'North America': { x: 180, y: 130 },
  'Latin America': { x: 230, y: 270 },
  'Caribbean': { x: 250, y: 210 },
  'Western Europe': { x: 430, y: 115 },
  'Eastern Europe': { x: 490, y: 110 },
  'North Africa': { x: 440, y: 175 },
  'West Africa': { x: 400, y: 225 },
  'East Africa': { x: 500, y: 245 },
  'Central Africa': { x: 460, y: 245 },
  'Southern Africa': { x: 475, y: 310 },
  'Middle East': { x: 530, y: 170 },
  'South Asia': { x: 590, y: 195 },
  'Central Asia': { x: 560, y: 130 },
  'East Asia': { x: 650, y: 145 },
  'Southeast Asia': { x: 650, y: 220 },
  'Australia': { x: 700, y: 310 },
  'Pacific Islands': { x: 740, y: 275 },
};

function WorldMap({ sites }: { sites: SiteValidation[] }) {
  // Group sites by region for marker placement
  const sitesByRegion: Record<string, SiteValidation[]> = {};
  for (const site of sites) {
    const key = site.region;
    if (!sitesByRegion[key]) sitesByRegion[key] = [];
    sitesByRegion[key].push(site);
  }

  return (
    <div className="sv-map-container">
      <svg viewBox="0 0 800 400" className="sv-map-svg" aria-label="World map showing validation site locations">
        {/* Ocean background */}
        <rect x="0" y="0" width="800" height="400" fill="var(--color-bg-secondary)" rx="8" />

        {/* Simplified continent outlines */}
        {/* North America */}
        <path d="M100,60 L210,60 L230,90 L240,130 L230,170 L200,180 L170,200 L140,180 L120,140 L100,100 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* South America */}
        <path d="M200,210 L260,200 L280,230 L280,280 L260,330 L240,350 L220,340 L210,300 L200,250 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* Europe */}
        <path d="M390,60 L460,55 L500,70 L520,90 L510,120 L480,130 L440,130 L410,120 L390,100 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* Africa */}
        <path d="M400,150 L470,145 L510,170 L520,220 L500,280 L480,330 L450,340 L420,310 L410,260 L400,200 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* Asia */}
        <path d="M520,60 L700,50 L720,80 L710,130 L680,160 L650,180 L600,190 L560,180 L530,150 L520,110 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* Southeast Asia / Indonesia */}
        <path d="M620,195 L680,190 L720,210 L710,240 L660,240 L630,225 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />
        {/* Australia */}
        <path d="M660,280 L740,275 L760,300 L740,340 L700,345 L670,330 L660,300 Z"
          fill="var(--color-surface)" stroke="var(--color-surface-border)" strokeWidth="1" opacity="0.7" />

        {/* Site markers */}
        {Object.entries(sitesByRegion).map(([region, regionSites]) => {
          const coords = REGION_COORDS[region];
          if (!coords) return null;
          const isWestern = regionSites[0]?.region_class === 'western';
          const color = isWestern ? 'var(--color-info)' : 'var(--color-success)';

          return (
            <g key={region}>
              {/* Pulse ring */}
              <circle cx={coords.x} cy={coords.y} r="12"
                fill="none" stroke={color} strokeWidth="1.5" opacity="0.4">
                <animate attributeName="r" values="8;14;8" dur="3s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.4;0.1;0.4" dur="3s" repeatCount="indefinite" />
              </circle>
              {/* Marker dot */}
              <circle cx={coords.x} cy={coords.y} r="6"
                fill={color} stroke="var(--color-bg-primary)" strokeWidth="2"
                style={{ cursor: 'pointer' }}>
                <title>{`${region}: ${regionSites.map(s => s.site_id).join(', ')}`}</title>
              </circle>
              {/* Count badge if multiple sites */}
              {regionSites.length > 1 && (
                <text x={coords.x + 10} y={coords.y - 8}
                  fill={color} fontSize="10" fontWeight="bold">
                  {regionSites.length}
                </text>
              )}
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(20, 360)">
          <circle cx="0" cy="0" r="5" fill="var(--color-success)" />
          <text x="10" y="4" fill="var(--color-text-secondary)" fontSize="10">Non-Western</text>
          <circle cx="100" cy="0" r="5" fill="var(--color-info)" />
          <text x="110" y="4" fill="var(--color-text-secondary)" fontSize="10">Western</text>
        </g>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SiteValidationPage() {
  const { getAuthHeader } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Data
  const [sites, setSites] = useState<SiteValidation[]>([]);
  const [readiness, setReadiness] = useState<ReleaseReadiness | null>(null);

  // Form state
  const [showForm, setShowForm] = useState(false);
  const [formSiteId, setFormSiteId] = useState('');
  const [formRegion, setFormRegion] = useState('');
  const [formDatasetSize, setFormDatasetSize] = useState('');
  const [formBenchmark, setFormBenchmark] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // ---- Fetch functions --------------------------------------------------

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const headers = getAuthHeader();
      const [sitesRes, readinessRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/validation/sites`, { headers }),
        fetch(`${API_BASE}/api/v1/validation/readiness`, { headers }),
      ]);

      if (sitesRes.ok) {
        const data = await sitesRes.json();
        setSites(data.sites || []);
      }
      if (readinessRes.ok) {
        setReadiness(await readinessRes.json());
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch validation data');
    } finally {
      setLoading(false);
    }
  }, [getAuthHeader]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  // ---- Form submission --------------------------------------------------

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(null);

    try {
      let benchmarkReport: Record<string, any> = {};
      if (formBenchmark.trim()) {
        try {
          benchmarkReport = JSON.parse(formBenchmark);
        } catch {
          setError('Invalid benchmark report JSON');
          setSubmitting(false);
          return;
        }
      }

      const body = {
        site_id: formSiteId,
        region: formRegion,
        benchmark_report: benchmarkReport,
        dataset_size: formDatasetSize ? parseInt(formDatasetSize, 10) : null,
      };

      const resp = await fetch(`${API_BASE}/api/v1/validation/sites`, {
        method: 'POST',
        headers: { ...getAuthHeader(), 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (resp.ok) {
        const data = await resp.json();
        setSuccess(`Site "${data.site_id}" registered as ${data.region_class}`);
        setFormSiteId('');
        setFormRegion('');
        setFormDatasetSize('');
        setFormBenchmark('');
        setShowForm(false);
        fetchAll();
      } else {
        const err = await resp.json();
        setError(err.detail || 'Registration failed');
      }
    } catch (err: any) {
      setError(err.message || 'Registration failed');
    } finally {
      setSubmitting(false);
    }
  };

  // ---- Derived data -----------------------------------------------------

  const selectedRegionInfo = REGION_OPTIONS.find(r => r.value === formRegion);

  // ---- Render -----------------------------------------------------------

  return (
    <div className="sv-dashboard" id="site-validation-page">
      {/* Header */}
      <div className="sv-header">
        <h1>
          <span className="page-icon">🌍</span>
          Site Validation Registry
        </h1>
        <div className="sv-header-actions">
          <button
            className={`sv-refresh-btn ${loading ? 'loading' : ''}`}
            onClick={fetchAll}
            disabled={loading}
            id="sv-refresh-btn"
          >
            ↻ {loading ? 'Refreshing…' : 'Refresh'}
          </button>
          <button
            className="sv-add-btn"
            onClick={() => setShowForm(!showForm)}
            id="sv-add-btn"
          >
            {showForm ? '✕ Cancel' : '+ Add Validation'}
          </button>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="sv-message sv-error" role="alert">
          <span>⚠ {error}</span>
          <button className="sv-msg-dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}
      {success && (
        <div className="sv-message sv-success" role="status">
          <span>✓ {success}</span>
          <button className="sv-msg-dismiss" onClick={() => setSuccess(null)}>×</button>
        </div>
      )}

      {/* Release readiness badge */}
      {readiness && (
        <div className={`sv-readiness-card ${readiness.ready ? 'ready' : 'not-ready'}`} id="sv-readiness-badge">
          <div className="sv-readiness-badge-icon">
            {readiness.ready ? '✓' : '✗'}
          </div>
          <div className="sv-readiness-info">
            <div className="sv-readiness-title">
              {readiness.ready ? 'Release Ready — v-stable' : 'Not Release Ready'}
            </div>
            <div className="sv-readiness-detail">
              {readiness.non_western_count} / {readiness.min_non_western} non-Western validations required
            </div>
            <div className="sv-readiness-breakdown">
              <span className="sv-readiness-tag non-western">
                Non-Western: {readiness.non_western_sites.length > 0 ? readiness.non_western_sites.join(', ') : 'none'}
              </span>
              <span className="sv-readiness-tag western">
                Western: {readiness.western_sites.length > 0 ? readiness.western_sites.join(', ') : 'none'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Add validation form */}
      {showForm && (
        <div className="sv-card sv-form-card">
          <div className="sv-card-header">
            <h3>Register Site Validation</h3>
          </div>
          <form onSubmit={handleSubmit} className="sv-form" id="sv-add-form">
            <div className="sv-form-grid">
              <div className="sv-field">
                <label htmlFor="sv-site-id">Site ID *</label>
                <input
                  id="sv-site-id"
                  type="text"
                  value={formSiteId}
                  onChange={e => setFormSiteId(e.target.value)}
                  placeholder="e.g. site_mumbai"
                  required
                />
              </div>
              <div className="sv-field">
                <label htmlFor="sv-region">Region *</label>
                <select
                  id="sv-region"
                  value={formRegion}
                  onChange={e => setFormRegion(e.target.value)}
                  required
                >
                  <option value="">Select region…</option>
                  <optgroup label="Non-Western">
                    {REGION_OPTIONS.filter(r => r.classification === 'non-western').map(r => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Western">
                    {REGION_OPTIONS.filter(r => r.classification === 'western').map(r => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                  </optgroup>
                </select>
                {selectedRegionInfo && (
                  <span className={`sv-region-class-badge ${selectedRegionInfo.classification}`}>
                    {selectedRegionInfo.classification}
                  </span>
                )}
              </div>
              <div className="sv-field">
                <label htmlFor="sv-dataset-size">Dataset Size</label>
                <input
                  id="sv-dataset-size"
                  type="number"
                  value={formDatasetSize}
                  onChange={e => setFormDatasetSize(e.target.value)}
                  placeholder="Number of samples"
                  min="0"
                />
              </div>
            </div>
            <div className="sv-field sv-field-full">
              <label htmlFor="sv-benchmark">Benchmark Report JSON</label>
              <textarea
                id="sv-benchmark"
                value={formBenchmark}
                onChange={e => setFormBenchmark(e.target.value)}
                placeholder='{"overall_pass": true, "auc": 0.92, ...}'
                rows={4}
              />
            </div>
            <div className="sv-form-actions">
              <button type="submit" className="sv-submit-btn" disabled={submitting}>
                {submitting ? 'Registering…' : 'Register Validation'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Region map */}
      <div className="sv-card">
        <div className="sv-card-header">
          <h3>Validation Site Locations</h3>
          <span className="sv-card-badge">{sites.length} sites</span>
        </div>
        <WorldMap sites={sites} />
      </div>

      {/* Registry table */}
      <div className="sv-card">
        <div className="sv-card-header">
          <h3>Registered Validations</h3>
        </div>
        <div className="sv-table-wrapper">
          <table className="sv-table" id="sv-registry-table">
            <thead>
              <tr>
                <th>Site ID</th>
                <th>Region</th>
                <th>Classification</th>
                <th>Dataset Size</th>
                <th>Validation Date</th>
                <th>Pass/Fail</th>
                <th>Report</th>
              </tr>
            </thead>
            <tbody>
              {sites.map(site => (
                <tr key={site.site_id}>
                  <td className="sv-td-site-id">{site.site_id}</td>
                  <td>{site.region}</td>
                  <td>
                    <span className={`sv-class-badge ${site.region_class}`}>
                      {site.region_class}
                    </span>
                  </td>
                  <td>{site.dataset_size > 0 ? site.dataset_size.toLocaleString() : '—'}</td>
                  <td className="sv-td-ts">
                    {site.timestamp ? new Date(site.timestamp).toLocaleDateString() : '—'}
                  </td>
                  <td>
                    <span className={`sv-pass-badge ${site.overall_pass ? 'pass' : 'fail'}`}>
                      {site.overall_pass ? '✓ Pass' : '✗ Fail'}
                    </span>
                  </td>
                  <td>
                    {Object.keys(site.benchmark_summary).length > 0 ? (
                      <button
                        className="sv-report-btn"
                        onClick={() => {
                          const w = window.open('', '_blank');
                          if (w) {
                            w.document.write(`<pre>${JSON.stringify(site.benchmark_summary, null, 2)}</pre>`);
                          }
                        }}
                        title="View benchmark report"
                      >
                        📄 View
                      </button>
                    ) : '—'}
                  </td>
                </tr>
              ))}
              {sites.length === 0 && (
                <tr>
                  <td colSpan={7} className="sv-td-empty">
                    No validations registered yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
