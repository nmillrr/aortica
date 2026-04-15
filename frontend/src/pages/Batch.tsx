import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictBatch } from '../services/InferenceClient';
import './Batch.css';

/* ── Types ─────────────────────────────────────────────────── */

type FileStatus = 'queued' | 'processing' | 'done' | 'error';
type SortKey = 'filename' | 'quality' | 'rhythm' | 'structural' | 'riskMort' | 'riskHF' | 'status';
type SortDir = 'asc' | 'desc';

interface BatchRow {
  id: string;
  filename: string;
  fileSize: number;
  status: FileStatus;
  progress: number;          // 0–100
  quality: number | null;
  qualityLabel: string;
  rhythm: string;
  structural: string;
  riskMort: number | null;
  riskHF: number | null;
  errorMsg: string | null;
  /** The raw prediction result stored for navigation */
  predictionResult: unknown;
}

/* ── Constants ──────────────────────────────────────────────── */


const PIPELINE_STEPS = [
  'Reading file',
  'Denoising signal',
  'Quality assessment',
  'AI inference',
  'Generating report',
];

/* ── Helpers ────────────────────────────────────────────────── */


function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function extractTopFinding(predictions: unknown, taskKey: string): string {
  if (!predictions || typeof predictions !== 'object') return '—';
  const preds = predictions as Record<string, unknown>;
  const task = preds[taskKey];
  if (!task || typeof task !== 'object') return '—';
  const entries = Object.entries(task as Record<string, number>);
  if (entries.length === 0) return '—';
  const [name] = entries.sort((a, b) => b[1] - a[1])[0];
  return name.replace(/_/g, ' ');
}

function extractRisk(predictions: unknown, idx: number): number | null {
  if (!predictions || typeof predictions !== 'object') return null;
  const preds = predictions as Record<string, unknown>;
  const risk = preds['risk'];
  if (!Array.isArray(risk) || risk.length <= idx) return null;
  const val = risk[idx] as unknown;
  return typeof val === 'number' ? val : null;
}

function extractQuality(quality: unknown): { score: number; label: string } {
  if (!quality || typeof quality !== 'object') return { score: 0, label: 'unknown' };
  const q = quality as Record<string, unknown>;
  const score = typeof q['overall_score'] === 'number' ? q['overall_score'] : 0;
  const label =
    score >= 70 ? 'good' :
    score >= 40 ? 'marginal' : 'poor';
  return { score: Math.round(score), label };
}

function rowFromResult(result: unknown): Partial<BatchRow> {
  const r = result as Record<string, unknown> | null;
  if (!r) return {};
  const { score, label } = extractQuality(r['quality']);
  return {
    quality: score,
    qualityLabel: label,
    rhythm: extractTopFinding(r['predictions'], 'rhythm'),
    structural: extractTopFinding(r['predictions'], 'structural'),
    riskMort: extractRisk(r['predictions'], 0),
    riskHF: extractRisk(r['predictions'], 1),
  };
}

function exportCSV(rows: BatchRow[]): void {
  const header = ['Filename', 'Size', 'Status', 'Quality', 'Top Rhythm', 'Top Structural', 'Mortality Risk (%)', 'HF Risk (%)'];
  const lines = rows.map(r => [
    r.filename,
    formatBytes(r.fileSize),
    r.status,
    r.quality != null ? String(r.quality) : '',
    r.rhythm,
    r.structural,
    r.riskMort != null ? (r.riskMort * 100).toFixed(1) : '',
    r.riskHF != null ? (r.riskHF * 100).toFixed(1) : '',
  ].map(v => `"${v.replace(/"/g, '""')}"`).join(','));

  const csv = [header.join(','), ...lines].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `aortica_batch_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function sortRows(rows: BatchRow[], key: SortKey, dir: SortDir): BatchRow[] {
  return [...rows].sort((a, b) => {
    let av: string | number = 0;
    let bv: string | number = 0;
    switch (key) {
      case 'filename':   av = a.filename;      bv = b.filename;      break;
      case 'quality':    av = a.quality ?? -1; bv = b.quality ?? -1; break;
      case 'rhythm':     av = a.rhythm;        bv = b.rhythm;        break;
      case 'structural': av = a.structural;    bv = b.structural;    break;
      case 'riskMort':   av = a.riskMort ?? -1; bv = b.riskMort ?? -1; break;
      case 'riskHF':     av = a.riskHF ?? -1;  bv = b.riskHF ?? -1;   break;
      case 'status':     av = a.status;        bv = b.status;        break;
    }
    if (av < bv) return dir === 'asc' ? -1 : 1;
    if (av > bv) return dir === 'asc' ? 1 : -1;
    return 0;
  });
}

/* ── Component ──────────────────────────────────────────────── */

export function Batch() {
  const navigate = useNavigate();

  const [isDragging, setIsDragging] = useState(false);
  const [rows, setRows] = useState<BatchRow[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const [sortKey, setSortKey] = useState<SortKey>('filename');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [filterText, setFilterText] = useState('');
  const [filterStatus, setFilterStatus] = useState<FileStatus | 'all'>('all');

  /* ─ File selection ──────────────────────────────────────── */


  /* ─ Batch processing ────────────────────────────────────── */

  const processAll = useCallback(async () => {
    const queued = rows.filter(r => r.status === 'queued');
    if (queued.length === 0) return;

    setIsProcessing(true);

    // Process each file sequentially with step animation
    for (const row of queued) {
      // Mark as processing
      setRows(prev => prev.map(r =>
        r.id === row.id ? { ...r, status: 'processing', progress: 5 } : r
      ));

      // Animate pipeline steps while real request runs
      let stepIdx = 0;
      const stepInterval = setInterval(() => {
        stepIdx = Math.min(stepIdx + 1, PIPELINE_STEPS.length - 1);
        setCurrentStep(stepIdx);
        const progPct = Math.round(((stepIdx + 1) / PIPELINE_STEPS.length) * 90);
        setRows(prev => prev.map(r =>
          r.id === row.id ? { ...r, progress: progPct } : r
        ));
      }, 800);

      try {
        // Find the actual File object. We stored filename/size on the row but
        // need to re-locate it from a re-render-safe source. Since we can't hold
        // File objects in state (they can't be serialized), we pass the file
        // directly here. We'll keep a separate ref map below.
        const result = await predictBatch([_fileMap.current[row.id]]);
        clearInterval(stepInterval);

        const fileResult = Array.isArray(result) && result.length > 0 ? result[0] : null;
        const extracted = rowFromResult(fileResult);

        setRows(prev => prev.map(r =>
          r.id === row.id
            ? {
                ...r,
                status: 'done',
                progress: 100,
                predictionResult: fileResult,
                quality: extracted.quality ?? null,
                qualityLabel: extracted.qualityLabel ?? 'unknown',
                rhythm: extracted.rhythm ?? '—',
                structural: extracted.structural ?? '—',
                riskMort: extracted.riskMort ?? null,
                riskHF: extracted.riskHF ?? null,
              }
            : r
        ));
      } catch (err) {
        clearInterval(stepInterval);
        const msg = err instanceof Error ? err.message : String(err);
        setRows(prev => prev.map(r =>
          r.id === row.id ? { ...r, status: 'error', progress: 100, errorMsg: msg } : r
        ));
      }
    }

    setIsProcessing(false);
    setCurrentStep(0);
  }, [rows]);

  /* We need to store File references outside of React state */
  const _fileMap = useRef<Record<string, File>>({});

  const addFilesWithMap = useCallback((files: File[]) => {
    const newRows: BatchRow[] = files.map(f => {
      const id = `${f.name}-${f.lastModified}-${Math.random()}`;
      _fileMap.current[id] = f;
      return {
        id,
        filename: f.name,
        fileSize: f.size,
        status: 'queued',
        progress: 0,
        quality: null,
        qualityLabel: 'unknown',
        rhythm: '—',
        structural: '—',
        riskMort: null,
        riskHF: null,
        errorMsg: null,
        predictionResult: null,
      };
    });
    setRows(prev => [...prev, ...newRows]);
  }, []);

  const handleDragOverFinal = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleDropFinal = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) addFilesWithMap(files);
  }, [addFilesWithMap]);

  const handleFileSelectFinal = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) addFilesWithMap(files);
    e.target.value = '';
  }, [addFilesWithMap]);

  const removeRow = useCallback((id: string) => {
    delete _fileMap.current[id];
    setRows(prev => prev.filter(r => r.id !== id));
  }, []);

  const clearDone = useCallback(() => {
    rows.filter(r => r.status === 'done' || r.status === 'error').forEach(r => {
      delete _fileMap.current[r.id];
    });
    setRows(prev => prev.filter(r => r.status !== 'done' && r.status !== 'error'));
  }, [rows]);

  /* ─ Sorting & filtering ─────────────────────────────────── */

  const handleSort = useCallback((key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  }, [sortKey]);

  const filteredRows = sortRows(
    rows.filter(r => {
      const matchText = filterText === '' || r.filename.toLowerCase().includes(filterText.toLowerCase());
      const matchStatus = filterStatus === 'all' || r.status === filterStatus;
      return matchText && matchStatus;
    }),
    sortKey,
    sortDir,
  );

  /* ─ Stats ───────────────────────────────────────────────── */

  const stats = {
    total: rows.length,
    done: rows.filter(r => r.status === 'done').length,
    processing: rows.filter(r => r.status === 'processing').length,
    error: rows.filter(r => r.status === 'error').length,
    queued: rows.filter(r => r.status === 'queued').length,
  };
  const overallProgress = stats.total > 0
    ? Math.round((stats.done + stats.error) / stats.total * 100)
    : 0;

  /* ─ Render ──────────────────────────────────────────────── */

  const SortIcon = ({ col }: { col: SortKey }) => (
    <span className="sort-icon" aria-hidden="true">
      {sortKey === col ? (sortDir === 'asc' ? '▲' : '▼') : '⇅'}
    </span>
  );

  return (
    <div className="batch-page" id="page-batch">

      {/* ── Upload Zone ─────────────────────────────────────── */}
      <div
        className={`batch-dropzone ${isDragging ? 'batch-dropzone--active' : ''}`}
        onDragOver={handleDragOverFinal}
        onDragLeave={handleDragLeave}
        onDrop={handleDropFinal}
        id="batch-dropzone"
      >
        <div className="batch-dropzone-icon">⬆</div>
        <div className="batch-dropzone-text">
          <span className="batch-dropzone-title">Drop ECG files here</span>
          <span className="batch-dropzone-subtitle">or click to browse • Any format • Up to 50 files</span>
        </div>
        <label className="btn btn-secondary batch-browse-btn" id="batch-browse-btn">
          Browse Files
          <input
            type="file"
            multiple
            className="batch-file-input"
            onChange={handleFileSelectFinal}
            id="batch-file-input"
          />
        </label>
      </div>

      {/* ── Toolbar ─────────────────────────────────────────── */}
      {rows.length > 0 && (
        <div className="batch-toolbar" id="batch-toolbar">
          {/* Stats badges */}
          <div className="batch-stats">
            <span className="batch-stat batch-stat--total">{stats.total} files</span>
            {stats.done > 0 && <span className="batch-stat batch-stat--done">{stats.done} done</span>}
            {stats.processing > 0 && <span className="batch-stat batch-stat--processing">{stats.processing} processing</span>}
            {stats.error > 0 && <span className="batch-stat batch-stat--error">{stats.error} errors</span>}
            {stats.queued > 0 && <span className="batch-stat batch-stat--queued">{stats.queued} queued</span>}
          </div>

          {/* Overall progress bar */}
          {(isProcessing || overallProgress > 0) && (
            <div className="batch-overall-progress" id="batch-overall-progress">
              <div className="batch-overall-progress-bar">
                <div
                  className="batch-overall-progress-fill"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
              <span className="batch-overall-progress-label">{overallProgress}%</span>
            </div>
          )}

          {/* Actions */}
          <div className="batch-actions">
            {stats.queued > 0 && !isProcessing && (
              <button
                className="btn btn-primary"
                onClick={processAll}
                id="batch-process-btn"
              >
                ▶ Process {stats.queued === rows.length ? 'All' : `${stats.queued} Queued`}
              </button>
            )}
            {(stats.done > 0 || stats.error > 0) && (
              <button
                className="btn btn-secondary"
                onClick={() => exportCSV(rows.filter(r => r.status === 'done'))}
                id="batch-export-csv-btn"
              >
                ⬇ Export CSV
              </button>
            )}
            {(stats.done > 0 || stats.error > 0) && !isProcessing && (
              <button
                className="btn btn-ghost"
                onClick={clearDone}
                id="batch-clear-done-btn"
              >
                Clear completed
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Filter bar ──────────────────────────────────────── */}
      {rows.length > 0 && (
        <div className="batch-filter-bar" id="batch-filter-bar">
          <div className="batch-search-wrapper">
            <span className="batch-search-icon">🔍</span>
            <input
              type="text"
              className="batch-search-input"
              placeholder="Filter by filename…"
              value={filterText}
              onChange={e => setFilterText(e.target.value)}
              id="batch-search-input"
            />
            {filterText && (
              <button className="batch-search-clear" onClick={() => setFilterText('')} id="batch-search-clear">✕</button>
            )}
          </div>
          <div className="batch-status-filters" role="group" aria-label="Filter by status">
            {(['all', 'queued', 'processing', 'done', 'error'] as const).map(s => (
              <button
                key={s}
                className={`batch-filter-chip ${filterStatus === s ? 'batch-filter-chip--active' : ''}`}
                onClick={() => setFilterStatus(s)}
                id={`batch-filter-${s}`}
              >
                {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Results table ───────────────────────────────────── */}
      {rows.length > 0 && (
        <div className="batch-table-container card" id="batch-results-table">
          {filteredRows.length === 0 ? (
            <div className="batch-empty-filter">No files match your filter.</div>
          ) : (
            <table className="batch-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort('filename')} id="col-filename">
                    Filename <SortIcon col="filename" />
                  </th>
                  <th onClick={() => handleSort('quality')} id="col-quality">
                    Quality <SortIcon col="quality" />
                  </th>
                  <th onClick={() => handleSort('rhythm')} id="col-rhythm">
                    Top Rhythm <SortIcon col="rhythm" />
                  </th>
                  <th onClick={() => handleSort('structural')} id="col-structural">
                    Top Structural <SortIcon col="structural" />
                  </th>
                  <th onClick={() => handleSort('riskMort')} id="col-risk-mort">
                    Mortality Risk <SortIcon col="riskMort" />
                  </th>
                  <th onClick={() => handleSort('riskHF')} id="col-risk-hf">
                    HF Risk <SortIcon col="riskHF" />
                  </th>
                  <th onClick={() => handleSort('status')} id="col-status">
                    Status <SortIcon col="status" />
                  </th>
                  <th style={{ width: 40 }} />
                </tr>
              </thead>
              <tbody>
                {filteredRows.map(row => (
                  <tr
                    key={row.id}
                    className={`batch-row batch-row--${row.status}`}
                    onClick={() => {
                      if (row.status === 'done') {
                        navigate(`/results/${encodeURIComponent(row.filename)}`, {
                          state: { predictionResult: row.predictionResult, fileName: row.filename },
                        });
                      }
                    }}
                    style={{ cursor: row.status === 'done' ? 'pointer' : undefined }}
                  >
                    {/* Filename + progress bar */}
                    <td className="batch-td-filename">
                      <div className="batch-filename-wrap">
                        <span className="batch-filename" id={`batch-row-${row.id}`}>{row.filename}</span>
                        <span className="batch-filesize">{formatBytes(row.fileSize)}</span>
                      </div>
                      {row.status === 'processing' && (
                        <div className="batch-progress-wrap" id={`batch-progress-${row.id}`}>
                          <div className="batch-progress-bar">
                            <div
                              className="batch-progress-fill"
                              style={{ width: `${row.progress}%` }}
                            />
                          </div>
                          <span className="batch-progress-step">
                            {PIPELINE_STEPS[Math.min(currentStep, PIPELINE_STEPS.length - 1)]}…
                          </span>
                        </div>
                      )}
                      {row.status === 'error' && row.errorMsg && (
                        <div className="batch-error-msg">{row.errorMsg}</div>
                      )}
                    </td>

                    {/* Quality */}
                    <td>
                      {row.quality != null ? (
                        <span className={`quality-pill quality-pill--${row.qualityLabel}`}>
                          {row.quality}
                        </span>
                      ) : (
                        <span className="batch-dash">—</span>
                      )}
                    </td>

                    {/* Top rhythm */}
                    <td className="batch-finding">{row.rhythm}</td>

                    {/* Top structural */}
                    <td className="batch-finding">{row.structural}</td>

                    {/* Mortality risk */}
                    <td>
                      {row.riskMort != null ? (
                        <span className={`risk-cell risk-cell--${row.riskMort > 0.4 ? 'high' : row.riskMort > 0.2 ? 'medium' : 'low'}`}>
                          {(row.riskMort * 100).toFixed(1)}%
                        </span>
                      ) : <span className="batch-dash">—</span>}
                    </td>

                    {/* HF risk */}
                    <td>
                      {row.riskHF != null ? (
                        <span className={`risk-cell risk-cell--${row.riskHF > 0.4 ? 'high' : row.riskHF > 0.2 ? 'medium' : 'low'}`}>
                          {(row.riskHF * 100).toFixed(1)}%
                        </span>
                      ) : <span className="batch-dash">—</span>}
                    </td>

                    {/* Status */}
                    <td>
                      <span className={`status-pill status-pill--${row.status}`}>
                        {row.status === 'queued'      && '⏳ Queued'}
                        {row.status === 'processing'  && '⚡ Processing'}
                        {row.status === 'done'        && '✓ Done'}
                        {row.status === 'error'       && '⚠ Error'}
                      </span>
                    </td>

                    {/* Remove button */}
                    <td onClick={e => { e.stopPropagation(); removeRow(row.id); }}>
                      <button
                        className="batch-remove-btn"
                        aria-label={`Remove ${row.filename}`}
                        id={`batch-remove-${row.id}`}
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* ── Empty state ─────────────────────────────────────── */}
      {rows.length === 0 && (
        <div className="batch-empty-state card" id="batch-empty-state">
          <div className="batch-empty-icon">📋</div>
          <h2 className="batch-empty-title">No ECG files loaded</h2>
          <p className="batch-empty-subtitle">
            Upload multiple ECG files above to batch-process them with the AI engine.
          </p>
          <div className="batch-empty-tip">
            Tip: You can drag &amp; drop up to 50 files at once.
          </div>
        </div>
      )}
    </div>
  );
}
