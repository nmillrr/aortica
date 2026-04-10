import { useRef, useState, useCallback, useEffect } from 'react';
import './ECGWaveformChart.css';

/* ── Public API ──────────────────────────────────────────────── */

export interface ECGWaveformData {
  leads: string[];
  signals: number[][];   // [leadIndex][sampleIndex]
  sample_rate: number;
}

export interface CaliperMeasurement {
  startX: number;   // px within the canvas
  endX: number;
  deltaMs: number;
  bpm: number;
}

interface Props {
  data: ECGWaveformData;
  /** Pixels per second – default 250 (≈25 mm/s at 10 px/mm) */
  pixelsPerSecond?: number;
  /** Pixels per millivolt – default 100 (≈10 mm/mV at 10 px/mm) */
  pixelsPerMV?: number;
  /** Height per lead row in px */
  rowHeight?: number;
  /** Whether this data is in µV (true, default) or mV (false) */
  isMicrovolts?: boolean;
  id?: string;
}

/* ── Constants ───────────────────────────────────────────────── */

const STANDARD_12_LEADS = [
  'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
  'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
];

// Standard 3×4 clinical layout: each column = 2.5 s, 4 columns side-by-side
// Row 0: I,  aVR, V1, V4
// Row 1: II, aVL, V2, V5
// Row 2: III,aVF, V3, V6
const LAYOUT_3x4: string[][] = [
  ['I',   'aVR', 'V1', 'V4'],
  ['II',  'aVL', 'V2', 'V5'],
  ['III', 'aVF', 'V3', 'V6'],
];

const MINOR_GRID_PX = 10;         // 1 mm at standard scale
const MAJOR_GRID_PX = MINOR_GRID_PX * 5; // 5 mm

/* ── Helpers ─────────────────────────────────────────────────── */

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Generate an SVG path `d` for a single lead's waveform */
function buildLeadPath(
  samples: number[],
  sampleRate: number,
  pxPerSec: number,
  pxPerMV: number,
  isMicro: boolean,
  offsetX: number,
  baselineY: number,
  startSample: number,
  endSample: number,
): string {
  const parts: string[] = [];
  const scale = isMicro ? 1e-3 : 1; // µV → mV if needed
  for (let i = startSample; i < endSample && i < samples.length; i++) {
    const t = (i - startSample) / sampleRate; // seconds from segment start
    const x = offsetX + t * pxPerSec;
    const mv = samples[i] * scale;
    const y = baselineY - mv * pxPerMV; // up is negative Y
    parts.push(i === startSample ? `M${x.toFixed(1)},${y.toFixed(1)}` : `L${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return parts.join(' ');
}

/* ── Component ───────────────────────────────────────────────── */

export function ECGWaveformChart({
  data,
  pixelsPerSecond = 250,
  pixelsPerMV = 100,
  rowHeight = 200,
  isMicrovolts = true,
  id = 'ecg-waveform-chart',
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  /* ── Zoom & Pan state ─────────────────────────────────────── */
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, px: 0, py: 0 });

  /* ── Caliper state ────────────────────────────────────────── */
  const [caliperActive, setCaliperActive] = useState(false);
  const [caliperStart, setCaliperStart] = useState<{ x: number; y: number } | null>(null);
  const [caliperEnd, setCaliperEnd] = useState<{ x: number; y: number } | null>(null);
  const [caliperResult, setCaliperResult] = useState<CaliperMeasurement | null>(null);

  /* ── Layout computation ───────────────────────────────────── */
  const durationSec = data.signals.length > 0
    ? data.signals[0].length / data.sample_rate
    : 10;

  // Build a lookup: lead name → index in data.signals
  const leadIndex = new Map<string, number>();
  data.leads.forEach((name, idx) => leadIndex.set(name.toUpperCase(), idx));

  // Use 3×4 layout if we have ≥12 leads; otherwise single-column
  const use3x4 = data.leads.length >= 12;
  const numRows = use3x4 ? 3 : data.leads.length;
  const numCols = use3x4 ? 4 : 1;
  const colDurationSec = use3x4 ? durationSec / numCols : durationSec;

  const effectivePPS = pixelsPerSecond * zoom;
  const colWidthPx = colDurationSec * effectivePPS;
  const totalWidth = colWidthPx * numCols;
  const totalHeight = rowHeight * numRows;

  /* ── Mouse handlers ───────────────────────────────────────── */
  const toSvgX = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      return (clientX - rect.left - panX) / zoom;
    },
    [panX, zoom],
  );

  const toSvgY = useCallback(
    (clientY: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      return (clientY - rect.top - panY) / zoom;
    },
    [panY, zoom],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((z) => clamp(z * delta, 0.25, 8));
    },
    [],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (caliperActive) {
        // Start caliper measurement
        const svgX = toSvgX(e.clientX);
        const svgY = toSvgY(e.clientY);
        setCaliperStart({ x: svgX, y: svgY });
        setCaliperEnd({ x: svgX, y: svgY });
        setCaliperResult(null);
        return;
      }
      // Start pan
      setIsPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, px: panX, py: panY };
    },
    [caliperActive, panX, panY, toSvgX, toSvgY],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (caliperActive && caliperStart) {
        const svgX = toSvgX(e.clientX);
        const svgY = toSvgY(e.clientY);
        setCaliperEnd({ x: svgX, y: svgY });
        return;
      }
      if (!isPanning) return;
      const dx = e.clientX - panStart.current.x;
      const dy = e.clientY - panStart.current.y;
      setPanX(panStart.current.px + dx);
      setPanY(panStart.current.py + dy);
    },
    [caliperActive, caliperStart, isPanning, toSvgX, toSvgY],
  );

  const handleMouseUp = useCallback(
    () => {
      if (caliperActive && caliperStart && caliperEnd) {
        const dxPx = Math.abs(caliperEnd.x - caliperStart.x);
        const dtSec = dxPx / (pixelsPerSecond * zoom);
        const dtMs = dtSec * 1000;
        const bpm = dtSec > 0 ? 60 / dtSec : 0;
        setCaliperResult({
          startX: caliperStart.x,
          endX: caliperEnd.x,
          deltaMs: Math.round(dtMs),
          bpm: Math.round(bpm),
        });
      }
      setIsPanning(false);
    },
    [caliperActive, caliperStart, caliperEnd, pixelsPerSecond, zoom],
  );

  /* ── Reset on data change ─────────────────────────────────── */
  useEffect(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
    setCaliperStart(null);
    setCaliperEnd(null);
    setCaliperResult(null);
  }, [data]);

  /* ── Render ────────────────────────────────────────────────── */
  const renderGrid = () => {
    const lines: React.ReactElement[] = [];
    // minor vertical
    for (let x = 0; x <= totalWidth; x += MINOR_GRID_PX) {
      const isMajor = x % MAJOR_GRID_PX === 0;
      lines.push(
        <line key={`v${x}`} x1={x} y1={0} x2={x} y2={totalHeight}
          className={isMajor ? 'ecg-grid-major' : 'ecg-grid-minor'} />,
      );
    }
    // minor horizontal
    for (let y = 0; y <= totalHeight; y += MINOR_GRID_PX) {
      const isMajor = y % MAJOR_GRID_PX === 0;
      lines.push(
        <line key={`h${y}`} x1={0} y1={y} x2={totalWidth} y2={y}
          className={isMajor ? 'ecg-grid-major' : 'ecg-grid-minor'} />,
      );
    }
    return <g className="ecg-grid-group">{lines}</g>;
  };

  const renderTraces = () => {
    const paths: React.ReactElement[] = [];
    const labels: React.ReactElement[] = [];

    if (use3x4) {
      // 3×4 clinical layout
      for (let row = 0; row < LAYOUT_3x4.length; row++) {
        for (let col = 0; col < LAYOUT_3x4[row].length; col++) {
          const leadName = LAYOUT_3x4[row][col];
          const idx = leadIndex.get(leadName.toUpperCase());
          if (idx === undefined) continue;
          const samples = data.signals[idx];
          const baselineY = row * rowHeight + rowHeight / 2;
          const offsetX = col * colWidthPx;
          const startSample = Math.floor((col * colDurationSec) * data.sample_rate);
          const endSample = Math.floor(((col + 1) * colDurationSec) * data.sample_rate);

          paths.push(
            <path
              key={`trace-${leadName}`}
              d={buildLeadPath(samples, data.sample_rate, effectivePPS, pixelsPerMV, isMicrovolts, offsetX, baselineY, startSample, endSample)}
              className="ecg-trace"
            />,
          );
          labels.push(
            <text
              key={`label-${leadName}`}
              x={offsetX + 6}
              y={row * rowHeight + 18}
              className="ecg-lead-label"
            >
              {leadName}
            </text>,
          );
        }
      }
    } else {
      // Single-column: each lead gets its own row, full duration
      data.leads.forEach((leadName, idx) => {
        const samples = data.signals[idx];
        const baselineY = idx * rowHeight + rowHeight / 2;
        paths.push(
          <path
            key={`trace-${leadName}`}
            d={buildLeadPath(samples, data.sample_rate, effectivePPS, pixelsPerMV, isMicrovolts, 0, baselineY, 0, samples.length)}
            className="ecg-trace"
          />,
        );
        labels.push(
          <text
            key={`label-${leadName}`}
            x={6}
            y={idx * rowHeight + 18}
            className="ecg-lead-label"
          >
            {leadName}
          </text>,
        );
      });
    }

    // Row separator lines
    const seps: React.ReactElement[] = [];
    for (let r = 1; r < numRows; r++) {
      seps.push(
        <line
          key={`sep-${r}`}
          x1={0} y1={r * rowHeight}
          x2={totalWidth} y2={r * rowHeight}
          className="ecg-row-sep"
        />,
      );
    }

    return (
      <>
        <g className="ecg-traces-group">{paths}</g>
        <g className="ecg-labels-group">{labels}</g>
        <g className="ecg-seps-group">{seps}</g>
      </>
    );
  };

  const renderCaliper = () => {
    if (!caliperStart || !caliperEnd) return null;
    const x1 = caliperStart.x;
    const x2 = caliperEnd.x;
    const midY = (caliperStart.y + caliperEnd.y) / 2;

    return (
      <g className="ecg-caliper-group">
        <line x1={x1} y1={0} x2={x1} y2={totalHeight} className="ecg-caliper-line" />
        <line x1={x2} y1={0} x2={x2} y2={totalHeight} className="ecg-caliper-line" />
        <line x1={x1} y1={midY} x2={x2} y2={midY} className="ecg-caliper-connector" />
        {caliperResult && (
          <foreignObject x={Math.min(x1, x2)} y={midY - 36} width={Math.abs(x2 - x1)} height={34}>
            <div className="ecg-caliper-badge">
              <span>{caliperResult.deltaMs} ms</span>
              <span className="ecg-caliper-bpm">{caliperResult.bpm} bpm</span>
            </div>
          </foreignObject>
        )}
      </g>
    );
  };

  /* ── Calibration marker (1 mV box) ─────────────────────── */
  const renderCalibMarker = () => {
    const x = 4;
    const y = rowHeight / 2;
    const h = pixelsPerMV; // 1 mV
    return (
      <g className="ecg-calib-group">
        <line x1={x} y1={y} x2={x} y2={y - h} className="ecg-calib-line" />
        <line x1={x} y1={y - h} x2={x + 20} y2={y - h} className="ecg-calib-line" />
        <line x1={x} y1={y} x2={x + 20} y2={y} className="ecg-calib-line" />
        <text x={x + 24} y={y - h / 2 + 4} className="ecg-calib-text">1 mV</text>
      </g>
    );
  };

  return (
    <div className="ecg-waveform-container" id={id}>
      {/* Toolbar */}
      <div className="ecg-toolbar" id="ecg-toolbar">
        <div className="ecg-toolbar-group">
          <button
            className={`ecg-tool-btn ${!caliperActive ? 'ecg-tool-btn--active' : ''}`}
            onClick={() => { setCaliperActive(false); setCaliperStart(null); setCaliperEnd(null); setCaliperResult(null); }}
            title="Pan mode"
            id="ecg-btn-pan"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 1v14M1 8h14M3 3l2 2M11 3l2 2M3 13l2-2M11 13l2-2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            Pan
          </button>
          <button
            className={`ecg-tool-btn ${caliperActive ? 'ecg-tool-btn--active' : ''}`}
            onClick={() => setCaliperActive(true)}
            title="Caliper tool — click and drag to measure intervals"
            id="ecg-btn-caliper"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 2v12M13 2v12M3 8h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
            Caliper
          </button>
        </div>
        <div className="ecg-toolbar-group">
          <button className="ecg-tool-btn" onClick={() => setZoom((z) => clamp(z * 1.25, 0.25, 8))} title="Zoom in" id="ecg-btn-zoom-in">+</button>
          <span className="ecg-zoom-label" id="ecg-zoom-level">{Math.round(zoom * 100)}%</span>
          <button className="ecg-tool-btn" onClick={() => setZoom((z) => clamp(z * 0.8, 0.25, 8))} title="Zoom out" id="ecg-btn-zoom-out">−</button>
          <button className="ecg-tool-btn" onClick={() => { setZoom(1); setPanX(0); setPanY(0); }} title="Reset view" id="ecg-btn-reset">⌂</button>
        </div>
        <div className="ecg-toolbar-meta">
          <span className="ecg-meta-chip">{data.sample_rate} Hz</span>
          <span className="ecg-meta-chip">{durationSec.toFixed(1)}s</span>
          <span className="ecg-meta-chip">{data.leads.length}-lead</span>
          <span className="ecg-meta-chip">25 mm/s</span>
          <span className="ecg-meta-chip">10 mm/mV</span>
        </div>
      </div>

      {/* Caliper result banner */}
      {caliperResult && (
        <div className="ecg-caliper-result" id="ecg-caliper-result">
          Interval: <strong>{caliperResult.deltaMs} ms</strong> · Rate: <strong>{caliperResult.bpm} bpm</strong>
          <button className="ecg-caliper-clear" onClick={() => { setCaliperStart(null); setCaliperEnd(null); setCaliperResult(null); }}>✕</button>
        </div>
      )}

      {/* Chart area */}
      <div
        ref={containerRef}
        className={`ecg-chart-viewport ${caliperActive ? 'ecg-chart-viewport--caliper' : ''}`}
        id="ecg-chart-viewport"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          ref={svgRef}
          className="ecg-chart-svg"
          width={totalWidth}
          height={totalHeight}
          viewBox={`0 0 ${totalWidth} ${totalHeight}`}
          style={{
            transform: `translate(${panX}px, ${panY}px)`,
          }}
        >
          {renderGrid()}
          {renderCalibMarker()}
          {renderTraces()}
          {renderCaliper()}
        </svg>
      </div>
    </div>
  );
}

/* ── Demo mock data generator ────────────────────────────────── */

export function generateDemoECGData(): ECGWaveformData {
  const sampleRate = 500;
  const duration = 10; // seconds
  const numSamples = sampleRate * duration;
  const leads = [...STANDARD_12_LEADS];
  const signals: number[][] = [];

  for (let l = 0; l < 12; l++) {
    const sig = new Array<number>(numSamples);
    // Generate realistic-ish ECG waveform per lead
    const hr = 72; // bpm
    const beatInterval = sampleRate * 60 / hr;
    const ampScale = 0.5 + Math.random() * 0.8;
    const invertLead = (l === 3); // aVR is normally inverted

    for (let i = 0; i < numSamples; i++) {
      const phase = (i % beatInterval) / beatInterval;
      let v = 0;

      // P wave: phase 0.06–0.14
      if (phase > 0.06 && phase < 0.14) {
        const t = (phase - 0.06) / 0.08;
        v += 0.1 * Math.sin(t * Math.PI) * ampScale;
      }
      // QRS complex: phase 0.18–0.26
      if (phase > 0.18 && phase < 0.20) {
        const t = (phase - 0.18) / 0.02;
        v -= 0.15 * Math.sin(t * Math.PI) * ampScale; // Q
      }
      if (phase > 0.20 && phase < 0.24) {
        const t = (phase - 0.20) / 0.04;
        v += 1.2 * Math.sin(t * Math.PI) * ampScale; // R
      }
      if (phase > 0.24 && phase < 0.26) {
        const t = (phase - 0.24) / 0.02;
        v -= 0.25 * Math.sin(t * Math.PI) * ampScale; // S
      }
      // T wave: phase 0.36–0.52
      if (phase > 0.36 && phase < 0.52) {
        const t = (phase - 0.36) / 0.16;
        v += 0.3 * Math.sin(t * Math.PI) * ampScale;
      }

      // Baseline noise
      v += (Math.random() - 0.5) * 0.02;

      sig[i] = (invertLead ? -v : v) * 1000; // convert to µV
    }
    signals.push(sig);
  }

  return { leads, signals, sample_rate: sampleRate };
}
