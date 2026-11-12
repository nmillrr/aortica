/**
 * Prediction output as the UI consumes it, from either inference path.
 *
 * The server (POST /api/v1/predict) already returns gated output: per task,
 * only the classes the checkpoint trained, each with its own operating point,
 * and the withheld names in `suppressed_classes`. The offline ONNX path
 * returns raw head tensors, so it is gated here with the same sidecars the
 * server uses (copied by scripts/sync_release_artifacts.py).
 *
 * Probabilities are not comparable across classes — rare findings peak far
 * below common ones — so a finding is positive only when it reaches its own
 * threshold. Nothing here falls back to invented values: a response that
 * cannot be read yields no findings.
 */

import taskClassNames from '../data/task_class_names.json';
import trainedOutputsFile from '../data/trained_outputs.json';
import classThresholds from '../data/class_thresholds.json';

/* ---------- Types -------------------------------------------------------- */

/** One task head's output, matching the API's `TaskPrediction`. */
export interface TaskPrediction {
  task: string;
  class_names: string[];
  probabilities: number[];
  thresholds: (number | null)[];
  suppressed_classes: string[];
}

export interface Finding {
  /** Canonical class name, as the model and API name it (e.g. `av_block_1st`). */
  name: string;
  /** Human-readable label. */
  label: string;
  task: string;
  prob: number;
  /** Per-class operating point; `null` when the checkpoint has none. */
  threshold: number | null;
  /** Probability reached the class's own threshold. */
  positive: boolean;
}

export interface LeadQuality {
  lead: string;
  score: number;
  classification: string;
}

export interface QualityReport {
  overall: number;
  classification: string;
  recommendation: string;
  leads: LeadQuality[];
}

export interface Suggestion {
  condition: string;
  prompt: string;
  urgency: string;
  rationale: string;
  probability: number | null;
  threshold: number | null;
}

/** Task heads shown as finding lists. `risk` is a separate (regression) head. */
export const FINDING_TASKS = ['rhythm', 'structural', 'ischaemia'] as const;

/** Sample rate every checkpoint is trained and run at (aortica/signal/preprocess.py). */
export const MODEL_SAMPLE_RATE_HZ = 100;

/* ---------- Labels ------------------------------------------------------- */

const LABELS: Record<string, string> = {
  AF: 'Atrial Fibrillation',
  AFL: 'Atrial Flutter',
  SVT: 'Supraventricular Tachycardia',
  AVRT: 'AV Re-entrant Tachycardia',
  sinus_brady: 'Sinus Bradycardia',
  sinus_tachy: 'Sinus Tachycardia',
  PAC: 'Premature Atrial Complex',
  PVC: 'Premature Ventricular Complex',
  av_block_1st: '1st Degree AV Block',
  av_block_2nd: '2nd Degree AV Block',
  av_block_3rd: '3rd Degree AV Block',
  LBBB: 'Left Bundle Branch Block',
  RBBB: 'Right Bundle Branch Block',
  LAFB: 'Left Anterior Fascicular Block',
  LPFB: 'Left Posterior Fascicular Block',
  WPW: 'Wolff-Parkinson-White Pattern',
  pacemaker_rhythm: 'Pacemaker Rhythm',
  normal_sinus_rhythm: 'Normal Sinus Rhythm',
  LVH: 'Left Ventricular Hypertrophy',
  RVH: 'Right Ventricular Hypertrophy',
  LA_enlargement: 'Left Atrial Enlargement',
  RA_enlargement: 'Right Atrial Enlargement',
  STEMI: 'ST-Elevation MI',
  posterior_MI: 'Posterior MI',
  old_MI: 'Old MI',
  digitalis_effect: 'Digitalis Effect',
  QTc_prolongation: 'QTc Prolongation',
  early_repol_vs_STEMI: 'Early Repolarisation vs STEMI',
};

/** Human-readable label for a class name; falls back to the name itself. */
export function labelFor(name: string): string {
  return LABELS[name] ?? name.replace(/_/g, ' ');
}

/* ---------- Helpers ------------------------------------------------------ */

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const num = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

const strList = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];

/* ---------- Server response ---------------------------------------------- */

/**
 * Read `predictions` from a /api/v1/predict response. Entries that do not
 * have the expected shape are dropped rather than guessed at.
 */
export function parseTaskPredictions(raw: unknown): TaskPrediction[] {
  if (!Array.isArray(raw)) return [];
  const out: TaskPrediction[] = [];
  for (const entry of raw) {
    if (!isRecord(entry) || typeof entry.task !== 'string') continue;
    const names = strList(entry.class_names);
    const probs = Array.isArray(entry.probabilities) ? entry.probabilities.map(num) : [];
    if (names.length !== probs.length || probs.some(p => p === null)) continue;
    const thresholds = Array.isArray(entry.thresholds) ? entry.thresholds.map(num) : [];
    out.push({
      task: entry.task,
      class_names: names,
      probabilities: probs as number[],
      thresholds: names.map((_, i) => thresholds[i] ?? null),
      suppressed_classes: strList(entry.suppressed_classes),
    });
  }
  return out;
}

/** Read `quality_report` from a /api/v1/predict response. */
export function parseQualityReport(raw: unknown): QualityReport | null {
  if (!isRecord(raw)) return null;
  const overall = num(raw.overall_score);
  if (overall === null) return null;
  const leads: LeadQuality[] = [];
  if (Array.isArray(raw.per_lead)) {
    for (const l of raw.per_lead) {
      if (!isRecord(l) || typeof l.lead_name !== 'string') continue;
      const score = num(l.score);
      if (score === null) continue;
      leads.push({
        lead: l.lead_name,
        score,
        classification: typeof l.classification === 'string' ? l.classification : 'unknown',
      });
    }
  }
  return {
    overall,
    classification: typeof raw.overall_classification === 'string' ? raw.overall_classification : 'unknown',
    recommendation: typeof raw.recommendation === 'string' ? raw.recommendation : '',
    leads,
  };
}

/** Read `suggestions` from a /api/v1/predict response. */
export function parseSuggestions(raw: unknown): Suggestion[] {
  if (!Array.isArray(raw)) return [];
  const out: Suggestion[] = [];
  for (const s of raw) {
    if (!isRecord(s) || typeof s.condition !== 'string' || typeof s.prompt !== 'string') continue;
    out.push({
      condition: s.condition,
      prompt: s.prompt,
      urgency: typeof s.urgency === 'string' ? s.urgency : 'routine',
      rationale: typeof s.rationale === 'string' ? s.rationale : '',
      probability: num(s.probability),
      threshold: num(s.threshold),
    });
  }
  return out;
}

/* ---------- Offline (ONNX) output ---------------------------------------- */

const TASK_CLASS_NAMES = taskClassNames as Record<string, string[]>;
const TRAINED = new Set<string>((trainedOutputsFile as { trained_outputs: string[] }).trained_outputs);
const THRESHOLDS = classThresholds as Record<string, number>;

/**
 * Gate raw ONNX head outputs the way the server does: name each value by
 * head order, keep trained classes, attach operating points.
 *
 * @param outputs task name → raw output values, in head order
 */
export function gateEdgeOutputs(outputs: Record<string, number[]>): TaskPrediction[] {
  const out: TaskPrediction[] = [];
  for (const [task, values] of Object.entries(outputs)) {
    const names = TASK_CLASS_NAMES[task];
    if (!names) continue;
    if (values.length !== names.length) {
      throw new Error(
        `Offline model '${task}' output has ${values.length} values; expected ${names.length}.`,
      );
    }
    const pred: TaskPrediction = {
      task, class_names: [], probabilities: [], thresholds: [], suppressed_classes: [],
    };
    names.forEach((name, i) => {
      if (TRAINED.has(name)) {
        pred.class_names.push(name);
        pred.probabilities.push(values[i]);
        pred.thresholds.push(THRESHOLDS[name] ?? null);
      } else {
        pred.suppressed_classes.push(name);
      }
    });
    out.push(pred);
  }
  return out;
}

/* ---------- Findings ----------------------------------------------------- */

/** Flatten task predictions into findings for the finding-list tasks. */
export function toFindings(predictions: TaskPrediction[]): Finding[] {
  const findings: Finding[] = [];
  for (const p of predictions) {
    if (!(FINDING_TASKS as readonly string[]).includes(p.task)) continue;
    p.class_names.forEach((name, i) => {
      const prob = p.probabilities[i];
      const threshold = p.thresholds[i] ?? null;
      findings.push({
        name,
        label: labelFor(name),
        task: p.task,
        prob,
        threshold,
        positive: threshold !== null && prob >= threshold,
      });
    });
  }
  return findings;
}

/** Every output the model did not report, across all tasks. */
export function suppressedOutputs(predictions: TaskPrediction[]): string[] {
  return predictions.flatMap(p => p.suppressed_classes);
}
