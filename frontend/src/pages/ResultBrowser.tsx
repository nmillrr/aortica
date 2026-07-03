import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import './ResultBrowser.css';

/* ---------- Types -------------------------------------------------------- */

interface ResultSummary {
  id: number;
  ecg_hash: string;
  timestamp: number;
  synced: boolean;
  patient_id: string | null;
  quality_score: number | null;
  quality_class: string | null;
  top_finding: string | null;
  top_finding_prob: number | null;
  urgency_tier: string | null;
}

interface ResultListResponse {
  results: ResultSummary[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

interface Filters {
  dateFrom: string;
  dateTo: string;
  finding: string;
  quality: string;
  urgency: string;
  search: string;
}

type SortField = 'timestamp' | 'quality_score' | 'urgency_tier' | 'top_finding';
type SortOrder = 'asc' | 'desc';

/* ---------- Constants ---------------------------------------------------- */

const API_BASE = 'http://localhost:8000';
const PAGE_SIZES = [10, 25, 50, 100] as const;

const EMPTY_FILTERS: Filters = {
  dateFrom: '',
  dateTo: '',
  finding: '',
  quality: '',
  urgency: '',
  search: '',
};

/* ---------- Helpers ------------------------------------------------------ */

function formatTimestamp(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffMin < 1440) return `${Math.floor(diffMin / 60)}h ago`;
  if (diffMin < 10080) return `${Math.floor(diffMin / 1440)}d ago`;

  return d.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatFinding(name: string | null): string {
  if (!name) return '—';
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/* ---------- Component ---------------------------------------------------- */

export function ResultBrowser() {
  const navigate = useNavigate();

  // Data state
  const [results, setResults] = useState<ResultSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState<number>(25);

  // Sorting
  const [sortBy, setSortBy] = useState<SortField>('timestamp');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  // Filters
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [activeFilters, setActiveFilters] = useState<Filters>(EMPTY_FILTERS);

  // Selection for bulk actions
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  // Fetch results from API
  const fetchResults = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      params.set('page', String(page));
      params.set('per_page', String(perPage));
      params.set('sort_by', sortBy);
      params.set('sort_order', sortOrder);

      if (activeFilters.dateFrom) {
        params.set('date_from', String(new Date(activeFilters.dateFrom).getTime() / 1000));
      }
      if (activeFilters.dateTo) {
        params.set('date_to', String(new Date(activeFilters.dateTo).getTime() / 1000));
      }
      if (activeFilters.finding) params.set('finding', activeFilters.finding);
      if (activeFilters.quality) params.set('quality', activeFilters.quality);
      if (activeFilters.urgency) params.set('urgency', activeFilters.urgency);
      if (activeFilters.search) params.set('search', activeFilters.search);

      const response = await fetch(`${API_BASE}/api/v1/results?${params.toString()}`);

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data: ResultListResponse = await response.json();
      setResults(data.results);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
      setResults([]);
      setTotal(0);
      setTotalPages(1);
    } finally {
      setLoading(false);
    }
  }, [page, perPage, sortBy, sortOrder, activeFilters]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  // Handlers
  const handleSort = (field: SortField) => {
    if (sortBy === field) {
      setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setPage(1);
  };

  const handleApplyFilters = () => {
    setActiveFilters({ ...filters });
    setPage(1);
  };

  const handleClearFilters = () => {
    setFilters(EMPTY_FILTERS);
    setActiveFilters(EMPTY_FILTERS);
    setPage(1);
  };

  const handleRowClick = (resultId: number) => {
    navigate(`/results/${resultId}`);
  };

  const handleToggleSelect = (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    if (selectedIds.size === results.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(results.map(r => r.id)));
    }
  };

  const handleExportCSV = async () => {
    if (selectedIds.size === 0) return;

    try {
      const response = await fetch(`${API_BASE}/api/v1/results/export/csv`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_ids: Array.from(selectedIds) }),
      });

      if (!response.ok) throw new Error('Export failed');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ecg_results_export.csv';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CSV export failed');
    }
  };

  // Key handler for filters
  const handleFilterKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleApplyFilters();
    }
  };

  // Has active filters?
  const hasActiveFilters = useMemo(
    () => Object.values(activeFilters).some(v => v !== ''),
    [activeFilters],
  );

  // Sort arrow helper
  const sortArrow = (field: SortField) => {
    if (sortBy !== field) return '↕';
    return sortOrder === 'desc' ? '↓' : '↑';
  };

  // Pagination page numbers
  const pageNumbers = useMemo(() => {
    const pages: number[] = [];
    const start = Math.max(1, page - 2);
    const end = Math.min(totalPages, page + 2);
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  }, [page, totalPages]);

  return (
    <div className="result-browser" id="page-result-browser">
      {/* Header */}
      <header className="result-browser-header">
        <h1>
          <span className="page-icon">📋</span>
          ECG History
          {!loading && (
            <span className="result-browser-total">
              {total} result{total !== 1 ? 's' : ''}
            </span>
          )}
        </h1>
        <div className="result-browser-actions">
          <button
            className="filter-btn filter-btn--clear"
            onClick={fetchResults}
            id="btn-refresh-results"
          >
            ↻ Refresh
          </button>
        </div>
      </header>

      {/* Bulk action bar */}
      {selectedIds.size > 0 && (
        <div className="bulk-action-bar" id="bulk-action-bar">
          <span className="bulk-count">
            {selectedIds.size} selected
          </span>
          <button
            className="bulk-action-btn"
            onClick={handleExportCSV}
            id="btn-export-csv"
          >
            ↓ Export CSV
          </button>
          <button
            className="bulk-action-btn"
            onClick={() => setSelectedIds(new Set())}
            id="btn-clear-selection"
          >
            ✕ Clear
          </button>
        </div>
      )}

      {/* Filter bar */}
      <div className="result-browser-filters" id="filter-bar">
        <div className="filter-group">
          <label htmlFor="filter-search">Search</label>
          <input
            id="filter-search"
            className="filter-input"
            type="text"
            placeholder="Patient ID, ECG hash…"
            value={filters.search}
            onChange={e => setFilters(f => ({ ...f, search: e.target.value }))}
            onKeyDown={handleFilterKeyDown}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="filter-date-from">From</label>
          <input
            id="filter-date-from"
            className="filter-input"
            type="date"
            value={filters.dateFrom}
            onChange={e => setFilters(f => ({ ...f, dateFrom: e.target.value }))}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="filter-date-to">To</label>
          <input
            id="filter-date-to"
            className="filter-input"
            type="date"
            value={filters.dateTo}
            onChange={e => setFilters(f => ({ ...f, dateTo: e.target.value }))}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="filter-finding">Finding</label>
          <input
            id="filter-finding"
            className="filter-input"
            type="text"
            placeholder="e.g. AF, LBBB…"
            value={filters.finding}
            onChange={e => setFilters(f => ({ ...f, finding: e.target.value }))}
            onKeyDown={handleFilterKeyDown}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="filter-quality">Quality</label>
          <select
            id="filter-quality"
            className="filter-select"
            value={filters.quality}
            onChange={e => setFilters(f => ({ ...f, quality: e.target.value }))}
          >
            <option value="">All</option>
            <option value="good">Good</option>
            <option value="marginal">Marginal</option>
            <option value="poor">Poor</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="filter-urgency">Urgency</label>
          <select
            id="filter-urgency"
            className="filter-select"
            value={filters.urgency}
            onChange={e => setFilters(f => ({ ...f, urgency: e.target.value }))}
          >
            <option value="">All</option>
            <option value="critical">Critical</option>
            <option value="urgent">Urgent</option>
            <option value="routine">Routine</option>
            <option value="normal">Normal</option>
          </select>
        </div>

        <div className="filter-actions">
          <button
            className="btn btn-primary filter-btn"
            onClick={handleApplyFilters}
            id="btn-apply-filters"
          >
            Filter
          </button>
          {hasActiveFilters && (
            <button
              className="filter-btn filter-btn--clear"
              onClick={handleClearFilters}
              id="btn-clear-filters"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="result-browser-error" id="error-banner">
          ⚠ {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="result-browser-loading" id="loading-spinner">
          <div className="loading-spinner" />
          <p>Loading results…</p>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && results.length === 0 && (
        <div className="result-browser-table-wrapper">
          <div className="result-browser-empty" id="empty-state">
            <div className="empty-icon">🔍</div>
            <h3>No results found</h3>
            <p>
              {hasActiveFilters
                ? 'No ECG results match your current filters. Try adjusting your search criteria.'
                : 'No ECG results have been stored yet. Upload and analyze an ECG to see results here.'}
            </p>
          </div>
        </div>
      )}

      {/* Results table */}
      {!loading && results.length > 0 && (
        <div className="result-browser-table-wrapper" id="results-table-wrapper">
          <table className="result-browser-table" id="results-table">
            <thead>
              <tr>
                <th className="th-checkbox">
                  <input
                    type="checkbox"
                    className="result-checkbox"
                    checked={selectedIds.size === results.length && results.length > 0}
                    onChange={handleSelectAll}
                    id="checkbox-select-all"
                    title="Select all"
                  />
                </th>
                <th
                  className={sortBy === 'timestamp' ? 'th-sorted' : ''}
                  onClick={() => handleSort('timestamp')}
                  id="th-timestamp"
                >
                  Timestamp <span className="sort-arrow">{sortArrow('timestamp')}</span>
                </th>
                <th id="th-patient">Patient</th>
                <th
                  className={sortBy === 'quality_score' ? 'th-sorted' : ''}
                  onClick={() => handleSort('quality_score')}
                  id="th-quality"
                >
                  Quality <span className="sort-arrow">{sortArrow('quality_score')}</span>
                </th>
                <th
                  className={sortBy === 'top_finding' ? 'th-sorted' : ''}
                  onClick={() => handleSort('top_finding')}
                  id="th-finding"
                >
                  Top Finding <span className="sort-arrow">{sortArrow('top_finding')}</span>
                </th>
                <th
                  className={sortBy === 'urgency_tier' ? 'th-sorted' : ''}
                  onClick={() => handleSort('urgency_tier')}
                  id="th-urgency"
                >
                  Urgency <span className="sort-arrow">{sortArrow('urgency_tier')}</span>
                </th>
                <th id="th-sync">Sync</th>
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr
                  key={r.id}
                  onClick={() => handleRowClick(r.id)}
                  id={`result-row-${r.id}`}
                >
                  <td>
                    <input
                      type="checkbox"
                      className="result-checkbox"
                      checked={selectedIds.has(r.id)}
                      onClick={e => handleToggleSelect(r.id, e)}
                      onChange={() => {/* controlled */}}
                      id={`checkbox-${r.id}`}
                    />
                  </td>
                  <td>{formatTimestamp(r.timestamp)}</td>
                  <td>{r.patient_id || r.ecg_hash.slice(0, 8)}</td>
                  <td>
                    {r.quality_class ? (
                      <span className={`quality-badge quality-badge--${r.quality_class}`}>
                        {r.quality_class === 'good' ? '●' : r.quality_class === 'marginal' ? '◐' : '○'}
                        {' '}{r.quality_score != null ? `${Math.round(r.quality_score)}%` : r.quality_class}
                      </span>
                    ) : (
                      <span className="quality-badge quality-badge--good">—</span>
                    )}
                  </td>
                  <td>
                    <div className="finding-text">
                      <span className="finding-name">{formatFinding(r.top_finding)}</span>
                      {r.top_finding_prob != null && (
                        <span className="finding-prob">
                          {(r.top_finding_prob * 100).toFixed(1)}% confidence
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    {r.urgency_tier ? (
                      <span className={`urgency-badge urgency-badge--${r.urgency_tier}`}>
                        {r.urgency_tier}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>
                    <span className="sync-badge">
                      <span className={`sync-dot sync-dot--${r.synced ? 'synced' : 'pending'}`} />
                      {r.synced ? 'Synced' : 'Pending'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div className="result-browser-pagination" id="pagination">
            <div className="pagination-info">
              Showing {(page - 1) * perPage + 1}–{Math.min(page * perPage, total)} of {total}
              {' · '}
              <select
                className="page-size-select"
                value={perPage}
                onChange={e => { setPerPage(Number(e.target.value)); setPage(1); }}
                id="page-size-select"
              >
                {PAGE_SIZES.map(size => (
                  <option key={size} value={size}>{size} / page</option>
                ))}
              </select>
            </div>

            <div className="pagination-controls">
              <button
                className="pagination-btn"
                disabled={page <= 1}
                onClick={() => setPage(1)}
                id="btn-page-first"
              >
                «
              </button>
              <button
                className="pagination-btn"
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
                id="btn-page-prev"
              >
                ‹
              </button>
              {pageNumbers.map(p => (
                <button
                  key={p}
                  className={`pagination-btn ${p === page ? 'pagination-btn--active' : ''}`}
                  onClick={() => setPage(p)}
                  id={`btn-page-${p}`}
                >
                  {p}
                </button>
              ))}
              <button
                className="pagination-btn"
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
                id="btn-page-next"
              >
                ›
              </button>
              <button
                className="pagination-btn"
                disabled={page >= totalPages}
                onClick={() => setPage(totalPages)}
                id="btn-page-last"
              >
                »
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
