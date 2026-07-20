import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import './ReportPage.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ReportFormat {
  format: string;
  label: string;
  endpoint: string;
  media_type: string;
  extension: string;
}

interface HistoryEntry {
  format: string;
  filename: string;
  generated_at: number;
}

interface ReportListing {
  result_id: number;
  available_formats: ReportFormat[];
  history: HistoryEntry[];
}

const API_BASE = 'http://localhost:8000';

// PDF is binary → preview via blob URL; the rest are text.
const TEXT_FORMATS = new Set(['fhir', 'hl7', 'jsonld']);

function reportUrl(fmt: ReportFormat, resultId: string): string {
  return `${API_BASE}${fmt.endpoint.replace('{result_id}', resultId)}`;
}

export function ReportPage() {
  const { result_id: resultId = '' } = useParams();
  const { t } = useTranslation();
  const { getAuthHeader } = useAuth();

  const [listing, setListing] = useState<ReportListing | null>(null);
  const [selected, setSelected] = useState<string>('pdf');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textContent, setTextContent] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Load the available formats + history.
  const fetchListing = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/v1/reports/${resultId}`, {
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setListing(await resp.json());
    } catch (e) {
      setError((e as Error).message);
    }
  }, [resultId, getAuthHeader]);

  useEffect(() => {
    fetchListing();
  }, [fetchListing]);

  const formats = listing?.available_formats ?? [];
  const currentFormat = useMemo(
    () => formats.find((f) => f.format === selected),
    [formats, selected],
  );

  // Generate the currently-selected report.
  const generate = useCallback(async () => {
    if (!currentFormat) return;
    setLoading(true);
    setError(null);
    setTextContent(null);
    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    setPdfUrl(null);
    try {
      const resp = await fetch(reportUrl(currentFormat, resultId), {
        method: 'POST',
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${resp.status}`);
      }
      if (TEXT_FORMATS.has(currentFormat.format)) {
        const text = await resp.text();
        // Pretty-print JSON payloads.
        setTextContent(
          currentFormat.format === 'hl7'
            ? text
            : JSON.stringify(JSON.parse(text), null, 2),
        );
      } else {
        const blob = await resp.blob();
        setPdfUrl(URL.createObjectURL(blob));
      }
      fetchListing(); // refresh history
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [currentFormat, resultId, getAuthHeader, pdfUrl, fetchListing]);

  const download = useCallback(async () => {
    if (!currentFormat) return;
    try {
      const resp = await fetch(reportUrl(currentFormat, resultId), {
        method: 'POST',
        headers: { ...getAuthHeader() },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${resultId}.${currentFormat.extension}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [currentFormat, resultId, getAuthHeader]);

  const copyText = useCallback(() => {
    if (textContent) {
      navigator.clipboard.writeText(textContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  }, [textContent]);

  return (
    <div className="report-page" id="report-page">
      <header className="rp-header">
        <h1>{t('reports.title', 'Clinical Report')}</h1>
        <span className="rp-subtitle">
          {t('reports.forResult', 'Result')} #{resultId}
        </span>
      </header>

      {/* Format selector */}
      <div className="rp-tabs" role="tablist">
        {formats.map((f) => (
          <button
            key={f.format}
            role="tab"
            className={`rp-tab ${selected === f.format ? 'active' : ''}`}
            onClick={() => setSelected(f.format)}
          >
            {f.label}
          </button>
        ))}
        <button
          className={`rp-tab ${selected === 'csv' ? 'active' : ''}`}
          onClick={() => setSelected('csv')}
        >
          {t('reports.csvBatch', 'CSV (Batch)')}
        </button>
      </div>

      <div className="rp-actions">
        {selected !== 'csv' && (
          <>
            <button className="rp-btn rp-btn-primary" onClick={generate} disabled={loading}>
              {loading ? t('reports.generating', 'Generating…') : t('reports.generate', 'Generate Preview')}
            </button>
            <button className="rp-btn" onClick={download}>
              {t('reports.download', 'Download')}
            </button>
          </>
        )}
        {textContent && (
          <button className="rp-btn" onClick={copyText}>
            {copied ? t('reports.copied', 'Copied!') : t('reports.copy', 'Copy to Clipboard')}
          </button>
        )}
      </div>

      {loading && (
        <div className="rp-loading">
          <div className="rp-spinner" aria-hidden="true" />
          <span>{t('reports.generatingHint', 'Generating report — this can take a few seconds…')}</span>
        </div>
      )}

      {error && <div className="rp-error" role="alert">{error}</div>}

      {/* Preview area */}
      <div className="rp-preview">
        {selected === 'csv' ? (
          <div className="rp-csv-note">
            {t(
              'reports.csvNote',
              'CSV analytics export is generated for batches of results. Select multiple results in the ECG History or Batch Analysis view and export them together.',
            )}
          </div>
        ) : pdfUrl ? (
          <object data={pdfUrl} type="application/pdf" className="rp-pdf" aria-label="PDF preview">
            <p>
              {t('reports.pdfFallback', 'PDF preview is unavailable. Use the Download button.')}
            </p>
          </object>
        ) : textContent ? (
          <pre className="rp-code">{textContent}</pre>
        ) : (
          <div className="rp-placeholder">
            {t('reports.placeholder', 'Select a format and click “Generate Preview”.')}
          </div>
        )}
      </div>

      {/* Report history */}
      <section className="rp-history">
        <h2>{t('reports.history', 'Report History')}</h2>
        {listing && listing.history.length > 0 ? (
          <ul className="rp-history-list">
            {listing.history.map((h, i) => (
              <li key={i}>
                <span className="rp-history-fmt">{h.format.toUpperCase()}</span>
                <span className="rp-history-file">{h.filename}</span>
                <span className="rp-history-time">
                  {new Date(h.generated_at * 1000).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="rp-history-empty">
            {t('reports.noHistory', 'No reports generated yet.')}
          </p>
        )}
      </section>
    </div>
  );
}
