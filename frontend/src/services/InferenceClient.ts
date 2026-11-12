/**
 * InferenceClient — Hybrid server/edge inference with automatic fallback.
 *
 * Strategy:
 *   1. Attempt POST to local FastAPI server with a 3-second timeout
 *   2. On timeout or network error, fall back to in-browser ONNX Runtime Web
 *   3. Annotate result with inference_mode so the UI knows which path was used
 */

import {
  gateEdgeOutputs,
  parseQualityReport,
  parseSuggestions,
  parseTaskPredictions,
  type QualityReport,
  type Suggestion,
  type TaskPrediction,
} from './predictions';

/* ---------- Types -------------------------------------------------------- */

export type InferenceMode = 'server' | 'edge_wasm';

export interface PredictionResult {
  /** Which inference path produced this result */
  inference_mode: InferenceMode;
  /** Per-task predictions — trained outputs only, with operating points */
  predictions: TaskPrediction[];
  /** Signal quality report (server only) */
  quality: QualityReport | null;
  /** Clinical suggestions for positive findings (server only) */
  suggestions: Suggestion[];
  /** Uncertainty / conformal prediction data (server only) */
  uncertainty?: Record<string, unknown>;
  /** XAI attribution data (when include_xai=true) */
  xai?: unknown[];
  /** Raw response payload (server mode) or edge output tensor data */
  raw?: unknown;
}

export interface InferenceClientConfig {
  /** Base URL of the local FastAPI server */
  serverUrl?: string;
  /** Timeout in ms before falling back to edge model */
  timeoutMs?: number;
  /** Path to the ONNX edge model (relative to public/) */
  modelPath?: string;
}

/** Build a PredictionResult from one /api/v1/predict response body. */
export function fromServerResponse(data: unknown): PredictionResult {
  const d = (data ?? {}) as Record<string, unknown>;
  return {
    inference_mode: 'server',
    raw: data,
    predictions: parseTaskPredictions(d.predictions),
    quality: parseQualityReport(d.quality_report),
    suggestions: parseSuggestions(d.suggestions),
    uncertainty: d.uncertainty as Record<string, unknown> | undefined,
    xai: Array.isArray(d.xai) ? d.xai : undefined,
  };
}

/* ---------- Constants ---------------------------------------------------- */

const DEFAULT_SERVER_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 3000;
const DEFAULT_MODEL_PATH = '/models/aortica_edge_int8.onnx';

/** Number of ECG leads the edge model expects */
const NUM_LEADS = 12;
/**
 * Samples per lead the edge model expects: 100 Hz × 10 s. Must match
 * aortica/signal/preprocess.py. The ONNX graph accepts any length, so a
 * 500 Hz tensor runs without error and scores near chance (macro-AUROC
 * 0.53 vs 0.94 on PTB-XL fold 10).
 */
const NUM_SAMPLES = 1000;
/** Input tensor name baked into the exported ONNX graph. */
const INPUT_NAME = 'ecg_input';

/* ---------- Edge session cache ------------------------------------------- */

let _edgeSession: unknown | null = null;
let _edgeSessionLoading: Promise<unknown> | null = null;

/**
 * Lazily load ONNX Runtime Web and create an InferenceSession.
 * The session is cached after first load so subsequent calls are instant.
 */
async function getEdgeSession(modelPath: string): Promise<unknown> {
  if (_edgeSession) return _edgeSession;

  if (_edgeSessionLoading) return _edgeSessionLoading;

  _edgeSessionLoading = (async () => {
    // Dynamic import so bundle isn't affected when server mode is used
    const ort = await import('onnxruntime-web');

    // Prefer WASM backend (universally available)
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });

    _edgeSession = session;
    return session;
  })();

  return _edgeSessionLoading;
}

/* ---------- Server inference --------------------------------------------- */

async function inferServer(
  file: File,
  serverUrl: string,
  timeoutMs: number,
): Promise<PredictionResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${serverUrl}/api/v1/predict?include_suggestions=true`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    return fromServerResponse(await response.json());
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- Edge WASM inference ------------------------------------------ */

async function inferEdge(
  file: File,
  modelPath: string,
): Promise<PredictionResult> {
  const ort = await import('onnxruntime-web');
  const session = await getEdgeSession(modelPath);

  // The offline path takes a preprocessed tensor only: raw little-endian
  // float32, [12 leads × 1000 samples], µV, 100 Hz, not denoised. Format
  // parsing and resampling live on the server; anything else is rejected
  // rather than zero-padded into a confident-looking prediction.
  const buffer = await file.arrayBuffer();
  const expectedBytes = NUM_LEADS * NUM_SAMPLES * 4;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `Offline inference needs a preprocessed ${NUM_LEADS}x${NUM_SAMPLES} float32 ` +
        `tensor (${expectedBytes} bytes, 100 Hz, µV); got ${buffer.byteLength} bytes. ` +
        'Connect to the server to analyse ECG files.',
    );
  }
  const rawSignal = new Float32Array(buffer);

  const inputTensor = new ort.Tensor('float32', rawSignal, [1, NUM_LEADS, NUM_SAMPLES]);

  // Run inference
  const typedSession = session as { run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>> };
  const results = await typedSession.run({ [INPUT_NAME]: inputTensor });

  // Name and gate the raw head outputs the way the server does.
  const outputs: Record<string, number[]> = {};
  for (const [key, value] of Object.entries(results)) {
    outputs[key.replace(/_output$/, '')] = Array.from(value.data);
  }

  return {
    inference_mode: 'edge_wasm',
    predictions: gateEdgeOutputs(outputs),
    quality: null,
    suggestions: [],
    raw: results,
  };
}

/* ---------- Health check ------------------------------------------------- */

/**
 * Check if the local FastAPI server is reachable.
 */
export async function checkServerHealth(
  serverUrl: string = DEFAULT_SERVER_URL,
  timeoutMs: number = 2000,
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${serverUrl}/health`, {
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- Pre-warm edge session ---------------------------------------- */

/**
 * Pre-load the ONNX edge model so it's ready for instant fallback.
 * Call this on app startup so the first offline inference is fast.
 */
export async function preloadEdgeModel(
  modelPath: string = DEFAULT_MODEL_PATH,
): Promise<boolean> {
  try {
    await getEdgeSession(modelPath);
    return true;
  } catch {
    return false;
  }
}

/* ---------- Main predict function ---------------------------------------- */

/**
 * Run ECG inference with automatic server → edge fallback.
 *
 * @param file       - The ECG file to analyze
 * @param config     - Optional configuration overrides
 * @returns          - Prediction result annotated with inference_mode
 */
export async function predict(
  file: File,
  config?: InferenceClientConfig,
): Promise<PredictionResult> {
  const serverUrl = config?.serverUrl ?? DEFAULT_SERVER_URL;
  const timeoutMs = config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const modelPath = config?.modelPath ?? DEFAULT_MODEL_PATH;

  // Strategy 1: Try the local server first
  try {
    return await inferServer(file, serverUrl, timeoutMs);
  } catch {
    // Server unreachable or timed out — fall through to edge
  }

  // Strategy 2: Fall back to in-browser ONNX Runtime Web
  try {
    return await inferEdge(file, modelPath);
  } catch (edgeError) {
    // Both paths failed
    throw new Error(
      `Inference failed: server unreachable and edge model error: ${
        edgeError instanceof Error ? edgeError.message : String(edgeError)
      }`,
    );
  }
}

/* ---------- Batch inference ---------------------------------------------- */

/**
 * Submit multiple ECG files to the batch predict endpoint.
 * Falls back to sequential single-file inference if batch endpoint fails.
 *
 * @param files  - Array of ECG files to analyze
 * @param config - Optional configuration overrides
 * @returns      - Array of per-file PredictionResult (same order as input files)
 */
export async function predictBatch(
  files: File[],
  config?: InferenceClientConfig,
): Promise<PredictionResult[]> {
  const serverUrl = config?.serverUrl ?? DEFAULT_SERVER_URL;
  const timeoutMs = config?.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  // Try batch endpoint first
  const controller = new AbortController();
  const batchTimeoutMs = timeoutMs + files.length * 2000; // generous per-file budget
  const timer = setTimeout(() => controller.abort(), batchTimeoutMs);

  try {
    const formData = new FormData();
    for (const f of files) {
      formData.append('files', f);
    }

    const response = await fetch(`${serverUrl}/api/v1/predict/batch`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Batch server returned ${response.status}`);
    }

    const data = await response.json() as Record<string, unknown>;
    const results = Array.isArray(data['results']) ? data['results'] : [];

    // Each entry is a BatchFileResult; the prediction itself is under `result`.
    return results.map(r => {
      const entry = (r ?? {}) as Record<string, unknown>;
      if (entry['status'] !== 'success' || !entry['result']) {
        throw new Error(String(entry['error'] ?? 'Batch item failed'));
      }
      return fromServerResponse(entry['result']);
    });
  } catch {
    // Fall back to sequential single-file inference
    clearTimeout(timer);
  } finally {
    clearTimeout(timer);
  }

  // Sequential fallback — process each file independently. A failure
  // propagates rather than producing an empty result that reads as "normal".
  const results: PredictionResult[] = [];
  for (const f of files) {
    results.push(await predict(f, config));
  }
  return results;
}

/* ---------- Exports ------------------------------------------------------ */

export const InferenceClient = {
  predict,
  predictBatch,
  checkServerHealth,
  preloadEdgeModel,
} as const;

export default InferenceClient;
