import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { predict, type PredictionResult } from '../services/InferenceClient';
import './Upload.css';

const SUPPORTED_FORMATS = [
  { label: 'WFDB (.hea/.dat)', ext: ['.hea', '.dat'] },
  { label: 'DICOM', ext: ['.dcm', '.dicom'] },
  { label: 'SCP-ECG', ext: ['.scp'] },
  { label: 'CSV', ext: ['.csv'] },
  { label: 'MAT', ext: ['.mat'] },
  { label: 'HL7 aECG (XML)', ext: ['.xml'] },
  { label: 'PDF / Image Scan', ext: ['.pdf', '.png', '.jpg', '.jpeg', '.tiff'] },
] as const;

/** All supported file extensions (flattened) */
const ALL_EXTENSIONS = SUPPORTED_FORMATS.flatMap(f => f.ext);

/** Analysis pipeline steps shown during loading */
const PIPELINE_STEPS = [
  { label: 'Reading ECG file', icon: '📄', duration: 800 },
  { label: 'Signal denoising', icon: '🔊', duration: 1200 },
  { label: 'Quality assessment', icon: '📊', duration: 600 },
  { label: 'AI multi-task inference', icon: '🧠', duration: 2000 },
  { label: 'Generating report', icon: '📋', duration: 400 },
] as const;

function getFileExtension(name: string): string {
  const dotIdx = name.lastIndexOf('.');
  return dotIdx >= 0 ? name.slice(dotIdx).toLowerCase() : '';
}

function isFormatSupported(filename: string): boolean {
  const ext = getFileExtension(filename);
  return ALL_EXTENSIONS.includes(ext as typeof ALL_EXTENSIONS[number]);
}

export function Upload() {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [formatWarning, setFormatWarning] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const selectFile = useCallback((f: File) => {
    setError(null);
    setFormatWarning(false);
    if (!isFormatSupported(f.name)) {
      setFormatWarning(true);
    }
    setFile(f);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) selectFile(dropped);
  }, [selectFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) selectFile(selected);
  }, [selectFile]);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    setCurrentStep(0);

    // Animate through pipeline steps while the real request runs
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      stepIdx += 1;
      if (stepIdx < PIPELINE_STEPS.length) {
        setCurrentStep(stepIdx);
      }
    }, 900);

    let result: PredictionResult;

    try {
      result = await predict(file);
    } catch (err) {
      clearInterval(stepInterval);
      setIsAnalyzing(false);

      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes('422') || msg.toLowerCase().includes('unsupported')) {
        setError(`Unsupported file format. Please upload a file in one of the supported ECG formats.`);
      } else if (msg.includes('413') || msg.toLowerCase().includes('too large')) {
        setError(`File too large. Please upload a smaller ECG file.`);
      } else {
        setError(`Analysis failed: ${msg}. Please check your connection and try again.`);
      }
      return;
    }

    clearInterval(stepInterval);
    // Show final step briefly then navigate
    setCurrentStep(PIPELINE_STEPS.length - 1);

    // Small delay so user sees the completed animation
    setTimeout(() => {
      const resultId = `ecg-${Date.now()}`;
      navigate(`/results/${resultId}`, {
        state: {
          predictionResult: result,
          fileName: file.name,
          fileSize: file.size,
        },
      });
    }, 400);
  }, [file, navigate]);

  const handleRemoveFile = useCallback(() => {
    setFile(null);
    setError(null);
    setFormatWarning(false);
  }, []);

  return (
    <div className="upload-page" id="page-upload">
      <div className="upload-container">
        {/* ── Analysis overlay ──────────────────────────────────── */}
        {isAnalyzing && (
          <div className="upload-analyzing-overlay" id="analyzing-overlay">
            <div className="analyzing-card glass">
              <div className="analyzing-pulse-ring" />
              <div className="analyzing-icon">♥</div>
              <h2 className="analyzing-title">Analyzing ECG</h2>
              <p className="analyzing-file">{file?.name}</p>

              <div className="analyzing-steps">
                {PIPELINE_STEPS.map((step, i) => (
                  <div
                    key={step.label}
                    className={`analyzing-step ${
                      i < currentStep ? 'analyzing-step--done' :
                      i === currentStep ? 'analyzing-step--active' : ''
                    }`}
                  >
                    <span className="step-icon">
                      {i < currentStep ? '✓' : step.icon}
                    </span>
                    <span className="step-label">{step.label}</span>
                    {i === currentStep && (
                      <span className="step-spinner" />
                    )}
                  </div>
                ))}
              </div>

              <div className="analyzing-progress-bar">
                <div
                  className="analyzing-progress-fill"
                  style={{
                    width: `${((currentStep + 1) / PIPELINE_STEPS.length) * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* ── Error banner ──────────────────────────────────────── */}
        {error && (
          <div className="upload-error animate-fade-in" id="upload-error">
            <span className="upload-error-icon">⚠</span>
            <div className="upload-error-content">
              <span className="upload-error-title">Analysis Error</span>
              <span className="upload-error-message">{error}</span>
            </div>
            <button
              className="btn btn-ghost upload-error-dismiss"
              onClick={() => setError(null)}
              id="dismiss-error-btn"
            >
              ✕
            </button>
          </div>
        )}

        {/* ── Dropzone ─────────────────────────────────────────── */}
        <div
          className={`upload-dropzone ${isDragging ? 'upload-dropzone--active' : ''} ${file ? 'upload-dropzone--has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          id="ecg-dropzone"
        >
          {!file ? (
            <>
              <div className="upload-icon">↑</div>
              <h2 className="upload-title">Upload ECG File</h2>
              <p className="upload-subtitle">
                Drag & drop your ECG file here, or click to browse
              </p>
              <label className="btn btn-secondary upload-browse-btn" id="browse-files-btn">
                Browse Files
                <input
                  type="file"
                  className="upload-file-input"
                  onChange={handleFileSelect}
                  id="ecg-file-input"
                />
              </label>
            </>
          ) : (
            <div className="upload-selected animate-fade-in">
              <div className="upload-file-icon">♥</div>
              <div className="upload-file-info">
                <span className="upload-file-name">{file.name}</span>
                <span className="upload-file-size">
                  {(file.size / 1024).toFixed(1)} KB
                </span>
                {formatWarning && (
                  <span className="upload-format-warning">
                    ⚠ Unrecognized format — analysis will attempt auto-detection
                  </span>
                )}
              </div>
              <button
                className="btn btn-ghost upload-remove-btn"
                onClick={handleRemoveFile}
                id="remove-file-btn"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* ── Analyze button ───────────────────────────────────── */}
        {file && !isAnalyzing && (
          <button
            className="btn btn-primary upload-analyze-btn"
            onClick={handleAnalyze}
            id="analyze-btn"
          >
            Analyze ECG
          </button>
        )}

        {/* ── Supported formats ────────────────────────────────── */}
        <div className="upload-formats card" id="supported-formats">
          <h3 className="card-title">Supported Formats</h3>
          <div className="format-chips">
            {SUPPORTED_FORMATS.map(fmt => (
              <span className="format-chip" key={fmt.label}>{fmt.label}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
