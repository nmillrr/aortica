import { useState, useMemo } from 'react';
import './XAIOverlay.css';

/* ── Public API ──────────────────────────────────────────────── */

export interface XAIFeatureContribution {
  feature_name: string;
  lead: string;
  delta_score: number;
}

export interface XAISegmentAttribution {
  lead: string;
  segments: Record<string, number>;
}

export interface SegmentBoundary {
  p_start: number;
  p_end: number;
  qrs_start: number;
  qrs_end: number;
  t_start: number;
  t_end: number;
}

export interface XAIAttribution {
  task: string;
  per_lead_attributions: Record<string, number[]>;
  segment_attributions: XAISegmentAttribution[];
  top_features: XAIFeatureContribution[];
  segment_boundaries: Record<string, SegmentBoundary[]>;
}

/* ── Segment label formatting ─────────────────────────────── */

const SEGMENT_COLORS: Record<string, string> = {
  'P wave':      'rgba(100, 149, 237, 0.7)',
  'PR interval': 'rgba(147, 112, 219, 0.6)',
  'QRS complex': 'rgba(255, 99, 71, 0.7)',
  'ST segment':  'rgba(255, 215, 0, 0.7)',
  'T wave':      'rgba(50, 205, 50, 0.7)',
  'QT/QTc':      'rgba(169, 169, 169, 0.4)',
};

const SEGMENT_LABELS_SHORT: Record<string, string> = {
  'P wave':      'P',
  'PR interval': 'PR',
  'QRS complex': 'QRS',
  'ST segment':  'ST',
  'T wave':      'T',
};

/* ── Heatmap overlay props ─────────────────────────────────── */

interface HeatmapOverlayProps {
  /** Per-sample attribution values for this lead */
  attributions: number[];
  /** Lead signal sample rate */
  sampleRate: number;
  /** Effective pixels per second (includes zoom) */
  effectivePPS: number;
  /** X offset in SVG for this lead's column */
  offsetX: number;
  /** Center Y (baseline) for this lead row */
  baselineY: number;
  /** Row height */
  rowHeight: number;
  /** Start sample index for this column segment */
  startSample: number;
  /** End sample index for this column segment */
  endSample: number;
  /** Opacity of the heatmap (0-1) */
  opacity?: number;
}

/**
 * Renders a semi-transparent heatmap overlay on top of the ECG trace,
 * where colour intensity corresponds to attribution strength.
 * Uses a vertical gradient stripe per sample-bucket.
 */
export function HeatmapOverlay({
  attributions,
  sampleRate,
  effectivePPS,
  offsetX,
  baselineY,
  rowHeight,
  startSample,
  endSample,
  opacity = 0.45,
}: HeatmapOverlayProps) {
  // Bucket attributions into pixel-width chunks for performance
  const bucketWidth = 3; // px per heatmap stripe
  const segmentSamples = endSample - startSample;
  const segmentWidthPx = (segmentSamples / sampleRate) * effectivePPS;
  const numBuckets = Math.max(1, Math.floor(segmentWidthPx / bucketWidth));

  const rects = useMemo(() => {
    const result: React.ReactElement[] = [];
    const samplesPerBucket = segmentSamples / numBuckets;

    // Find max attribution for normalisation
    let maxAttr = 0;
    for (let i = startSample; i < endSample && i < attributions.length; i++) {
      const v = Math.abs(attributions[i]);
      if (v > maxAttr) maxAttr = v;
    }
    if (maxAttr < 1e-10) return result;

    for (let b = 0; b < numBuckets; b++) {
      const sStart = Math.floor(startSample + b * samplesPerBucket);
      const sEnd = Math.min(
        Math.floor(startSample + (b + 1) * samplesPerBucket),
        endSample,
        attributions.length
      );

      // Average absolute attribution in this bucket
      let sum = 0;
      let count = 0;
      for (let i = sStart; i < sEnd; i++) {
        sum += Math.abs(attributions[i]);
        count++;
      }
      const avg = count > 0 ? sum / count : 0;
      const intensity = avg / maxAttr;

      if (intensity < 0.05) continue; // Skip near-zero

      // Determine colour based on sign (mean signed value)
      let signedSum = 0;
      for (let i = sStart; i < sEnd; i++) {
        signedSum += attributions[i];
      }
      const isPositive = signedSum >= 0;

      // Positive = warm red/orange, Negative = cool blue
      const r = isPositive ? 255 : 60;
      const g = isPositive ? Math.round(80 + 100 * (1 - intensity)) : 120;
      const bCol = isPositive ? 60 : 255;

      const x = offsetX + (b * bucketWidth);
      const y = baselineY - rowHeight / 2;
      const h = rowHeight;

      result.push(
        <rect
          key={`hm-${b}`}
          x={x}
          y={y}
          width={bucketWidth}
          height={h}
          fill={`rgba(${r}, ${g}, ${bCol}, ${intensity * opacity})`}
          className="xai-heatmap-rect"
        />
      );
    }
    return result;
  }, [attributions, sampleRate, effectivePPS, offsetX, baselineY, rowHeight, startSample, endSample, numBuckets, segmentSamples, opacity]);

  return <g className="xai-heatmap-group">{rects}</g>;
}

/* ── Segment Callout Markers ──────────────────────────────── */

interface SegmentCalloutProps {
  boundaries: SegmentBoundary[];
  sampleRate: number;
  effectivePPS: number;
  offsetX: number;
  baselineY: number;
  rowHeight: number;
  startSample: number;
  endSample: number;
}

/**
 * Renders named segment markers (P, QRS, ST, T) as semi-transparent
 * coloured regions with labels above each segment.
 */
export function SegmentCallouts({
  boundaries,
  sampleRate,
  effectivePPS,
  offsetX,
  baselineY,
  rowHeight,
  startSample,
  endSample,
}: SegmentCalloutProps) {
  const elements = useMemo(() => {
    const result: React.ReactElement[] = [];

    const sampleToX = (s: number) =>
      offsetX + ((s - startSample) / sampleRate) * effectivePPS;

    for (let bi = 0; bi < boundaries.length; bi++) {
      const b = boundaries[bi];

      // Segments to render
      const segments: { name: string; start: number; end: number }[] = [
        { name: 'P wave',      start: b.p_start,   end: b.p_end },
        { name: 'QRS complex', start: b.qrs_start, end: b.qrs_end },
        { name: 'ST segment',  start: b.qrs_end,   end: b.t_start },
        { name: 'T wave',      start: b.t_start,   end: b.t_end },
      ];

      for (const seg of segments) {
        if (seg.start < 0 || seg.end <= seg.start) continue;
        if (seg.end < startSample || seg.start >= endSample) continue;

        const x1 = Math.max(sampleToX(seg.start), offsetX);
        const x2 = Math.min(sampleToX(seg.end), offsetX + ((endSample - startSample) / sampleRate) * effectivePPS);
        const w = x2 - x1;
        if (w < 2) continue;

        const color = SEGMENT_COLORS[seg.name] || 'rgba(200, 200, 200, 0.3)';
        const label = SEGMENT_LABELS_SHORT[seg.name] || seg.name;
        const y = baselineY - rowHeight / 2;

        result.push(
          <g key={`seg-${bi}-${seg.name}`}>
            <rect
              x={x1} y={y}
              width={w} height={rowHeight}
              fill={color}
              opacity={0.12}
              className="xai-segment-region"
            />
            <line
              x1={x1} y1={y}
              x2={x1} y2={y + rowHeight}
              stroke={color}
              strokeWidth={1}
              strokeDasharray="3 2"
              opacity={0.5}
            />
            <text
              x={x1 + w / 2}
              y={y + 12}
              textAnchor="middle"
              className="xai-segment-label"
              fill={color}
            >
              {label}
            </text>
          </g>
        );
      }
    }
    return result;
  }, [boundaries, sampleRate, effectivePPS, offsetX, baselineY, rowHeight, startSample, endSample]);

  return <g className="xai-segment-callouts">{elements}</g>;
}

/* ── Top Features Panel ───────────────────────────────────── */

interface TopFeaturesPanelProps {
  features: XAIFeatureContribution[];
  findingName: string;
}

export function TopFeaturesPanel({ features, findingName }: TopFeaturesPanelProps) {
  if (features.length === 0) return null;

  const maxScore = Math.max(...features.map(f => Math.abs(f.delta_score)), 1e-10);

  return (
    <div className="xai-top-features" id="xai-top-features">
      <div className="xai-top-features-title">
        Top Contributing Features — {findingName}
      </div>
      <div className="xai-top-features-list">
        {features.map((f, i) => {
          const pct = (Math.abs(f.delta_score) / maxScore) * 100;
          return (
            <div key={`${f.feature_name}-${f.lead}-${i}`} className="xai-feature-item">
              <span className="xai-feature-rank">#{i + 1}</span>
              <span className="xai-feature-name">{f.feature_name}</span>
              <span className="xai-feature-lead">{f.lead}</span>
              <div className="xai-feature-bar-bg">
                <div
                  className="xai-feature-bar"
                  style={{
                    width: `${pct}%`,
                    background: SEGMENT_COLORS[f.feature_name] || 'var(--color-accent)',
                  }}
                />
              </div>
              <span className="xai-feature-score">{f.delta_score.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── XAI Controls Toolbar ─────────────────────────────────── */

interface XAIControlsProps {
  /** Available finding names with their associated task */
  findings: { name: string; task: string; prob: number }[];
  /** Currently selected finding for overlay */
  activeFinding: string | null;
  /** Set the active finding */
  onSelectFinding: (name: string | null) => void;
  /** Per-lead visibility */
  leadVisibility: Record<string, boolean>;
  /** Toggle lead visibility */
  onToggleLead: (lead: string) => void;
  /** All leads */
  leads: string[];
  /** Whether XAI overlay is globally visible */
  xaiVisible: boolean;
  /** Toggle global XAI visibility */
  onToggleXAI: () => void;
}

export function XAIControls({
  findings,
  activeFinding,
  onSelectFinding,
  leadVisibility,
  onToggleLead,
  leads,
  xaiVisible,
  onToggleXAI,
}: XAIControlsProps) {
  const [showLeadControls, setShowLeadControls] = useState(false);

  // Only show findings with prob > 30% as meaningful to explain
  const meaningfulFindings = findings
    .filter(f => f.prob >= 0.30)
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 6);

  return (
    <div className="xai-controls" id="xai-controls">
      <div className="xai-controls-header">
        <div className="xai-controls-title">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.4" />
            <path d="M8 4v5M8 11v1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          XAI Explanations
        </div>
        <button
          className={`xai-toggle-btn ${xaiVisible ? 'xai-toggle-btn--active' : ''}`}
          onClick={onToggleXAI}
          title={xaiVisible ? 'Hide XAI overlay' : 'Show XAI overlay'}
          id="xai-toggle-global"
        >
          {xaiVisible ? 'Hide Overlay' : 'Show Overlay'}
        </button>
      </div>

      {xaiVisible && (
        <div className="xai-controls-body">
          {/* Finding selector */}
          <div className="xai-finding-selector">
            <span className="xai-selector-label">Explain finding:</span>
            <div className="xai-finding-chips">
              {meaningfulFindings.map(f => (
                <button
                  key={f.name}
                  className={`xai-finding-chip ${activeFinding === f.name ? 'xai-finding-chip--active' : ''}`}
                  onClick={() => onSelectFinding(activeFinding === f.name ? null : f.name)}
                  title={`Show XAI for ${f.name} (${(f.prob * 100).toFixed(0)}%)`}
                >
                  <span
                    className="xai-chip-dot"
                    style={{
                      background: f.prob >= 0.8 ? 'var(--color-danger)' :
                                  f.prob >= 0.5 ? 'var(--color-warning)' : 'var(--color-success)',
                    }}
                  />
                  {f.name}
                  <span className="xai-chip-prob">{(f.prob * 100).toFixed(0)}%</span>
                </button>
              ))}
              {meaningfulFindings.length === 0 && (
                <span className="xai-no-findings">No significant findings to explain</span>
              )}
            </div>
          </div>

          {/* Lead visibility controls */}
          <div className="xai-lead-controls">
            <button
              className="xai-lead-toggle"
              onClick={() => setShowLeadControls(!showLeadControls)}
              id="xai-lead-toggle"
            >
              <span>Lead Visibility</span>
              <span className={`xai-lead-chevron ${showLeadControls ? 'xai-lead-chevron--open' : ''}`}>▾</span>
            </button>
            {showLeadControls && (
              <div className="xai-lead-grid">
                {leads.map(lead => (
                  <label key={lead} className="xai-lead-checkbox">
                    <input
                      type="checkbox"
                      checked={leadVisibility[lead] !== false}
                      onChange={() => onToggleLead(lead)}
                    />
                    <span className="xai-lead-name">{lead}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Demo XAI data generator ─────────────────────────────── */

/**
 * Generate demo XAI attribution data that matches the demo ECG data.
 * This provides a realistic preview of how XAI overlays look.
 */
export function generateDemoXAIData(): XAIAttribution[] {
  const sampleRate = 500;
  const numSamples = sampleRate * 10;
  const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const hr = 72;
  const beatInterval = sampleRate * 60 / hr;

  // Generate attributions that peak around QRS and ST segments
  const buildAttribution = (leadIdx: number): number[] => {
    const attr = new Array(numSamples).fill(0);
    const amplitude = 0.3 + Math.random() * 0.7;

    for (let i = 0; i < numSamples; i++) {
      const phase = (i % beatInterval) / beatInterval;

      // QRS attribution (high)
      if (phase > 0.19 && phase < 0.25) {
        const t = (phase - 0.19) / 0.06;
        attr[i] = amplitude * Math.sin(t * Math.PI) * (1.0 + leadIdx * 0.05);
      }
      // ST segment attribution
      if (phase > 0.26 && phase < 0.36) {
        const t = (phase - 0.26) / 0.10;
        attr[i] = amplitude * 0.4 * Math.sin(t * Math.PI);
      }
      // P wave attribution (lower)
      if (phase > 0.07 && phase < 0.13) {
        const t = (phase - 0.07) / 0.06;
        attr[i] = amplitude * 0.2 * Math.sin(t * Math.PI);
      }
      // T wave
      if (phase > 0.37 && phase < 0.50) {
        const t = (phase - 0.37) / 0.13;
        attr[i] = amplitude * 0.3 * Math.sin(t * Math.PI);
      }

      // Add slight noise
      attr[i] += (Math.random() - 0.5) * 0.02;
    }
    return attr;
  };

  const buildBoundaries = (): SegmentBoundary[] => {
    const bounds: SegmentBoundary[] = [];
    for (let i = 0; i < numSamples; i += Math.round(beatInterval)) {
      const r = i + Math.round(beatInterval * 0.22);
      if (r >= numSamples) break;
      bounds.push({
        p_start: Math.max(r - 100, 0),
        p_end: Math.max(r - 40, 0),
        qrs_start: Math.max(r - 20, 0),
        qrs_end: Math.min(r + 20, numSamples - 1),
        t_start: Math.min(r + 80, numSamples - 1),
        t_end: Math.min(r + 160, numSamples - 1),
      });
    }
    return bounds;
  };

  const perLeadAttrs: Record<string, number[]> = {};
  const segmentAttrs: XAISegmentAttribution[] = [];
  const segBounds: Record<string, SegmentBoundary[]> = {};

  for (let i = 0; i < leads.length; i++) {
    const lead = leads[i];
    perLeadAttrs[lead] = buildAttribution(i);
    segBounds[lead] = buildBoundaries();
    segmentAttrs.push({
      lead,
      segments: {
        'P wave': 0.1 + Math.random() * 0.1,
        'PR interval': 0.05 + Math.random() * 0.05,
        'QRS complex': 0.5 + Math.random() * 0.3,
        'ST segment': 0.2 + Math.random() * 0.2,
        'T wave': 0.15 + Math.random() * 0.15,
        'QT/QTc': 0.4 + Math.random() * 0.2,
      },
    });
  }

  const rhythmAttr: XAIAttribution = {
    task: 'rhythm',
    per_lead_attributions: perLeadAttrs,
    segment_attributions: segmentAttrs,
    top_features: [
      { feature_name: 'QRS complex', lead: 'V1', delta_score: 0.82 },
      { feature_name: 'P wave', lead: 'II', delta_score: 0.45 },
      { feature_name: 'ST segment', lead: 'V4', delta_score: 0.31 },
    ],
    segment_boundaries: segBounds,
  };

  return [rhythmAttr];
}
