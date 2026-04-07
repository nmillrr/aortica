import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './Upload.css';

const SUPPORTED_FORMATS = [
  'WFDB (.hea/.dat)',
  'DICOM',
  'SCP-ECG',
  'CSV',
  'MAT',
  'HL7 aECG (XML)',
  'PDF / Image Scan',
] as const;

export function Upload() {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  }, []);

  const handleAnalyze = useCallback(() => {
    if (!file) return;
    setIsAnalyzing(true);
    // Simulate analysis — in production this calls POST /api/v1/predict
    setTimeout(() => {
      navigate('/results/ecg-new');
    }, 2000);
  }, [file, navigate]);

  return (
    <div className="upload-page" id="page-upload">
      <div className="upload-container">
        {/* Dropzone */}
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
              </div>
              <button
                className="btn btn-ghost upload-remove-btn"
                onClick={() => setFile(null)}
                id="remove-file-btn"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* Analyze button */}
        {file && (
          <button
            className={`btn btn-primary upload-analyze-btn ${isAnalyzing ? 'upload-analyze-btn--loading' : ''}`}
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            id="analyze-btn"
          >
            {isAnalyzing ? (
              <>
                <span className="upload-spinner" />
                Analyzing...
              </>
            ) : (
              'Analyze ECG'
            )}
          </button>
        )}

        {/* Supported formats */}
        <div className="upload-formats card" id="supported-formats">
          <h3 className="card-title">Supported Formats</h3>
          <div className="format-chips">
            {SUPPORTED_FORMATS.map(fmt => (
              <span className="format-chip" key={fmt}>{fmt}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
