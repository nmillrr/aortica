/**
 * InferenceClient — Hybrid server/edge inference with automatic fallback.
 *
 * Strategy:
 *   1. Attempt POST to local FastAPI server with a 3-second timeout
 *   2. On timeout or network error, fall back to in-browser ONNX Runtime Web
 *   3. Annotate result with inference_mode so the UI knows which path was used
 */

/* ---------- Types -------------------------------------------------------- */

export type InferenceMode = 'server' | 'edge_wasm';

export interface PredictionResult {
  /** Which inference path produced this result */
  inference_mode: InferenceMode;
  /** Quality report from the pipeline */
  quality?: Record<string, unknown>;
  /** Per-task predictions */
  predictions?: Record<string, unknown>;
  /** Uncertainty / conformal prediction data */
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

/* ---------- Constants ---------------------------------------------------- */

const DEFAULT_SERVER_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 3000;
const DEFAULT_MODEL_PATH = '/models/aortica_edge_int8.onnx';

/** Number of ECG leads the edge model expects */
const NUM_LEADS = 12;
/** Default sample count (500 Hz × 10 s) */
const NUM_SAMPLES = 5000;

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

    const response = await fetch(`${serverUrl}/api/v1/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data: unknown = await response.json();

    return {
      inference_mode: 'server',
      raw: data,
      predictions: (data as Record<string, unknown>).predictions as Record<string, unknown> | undefined,
      quality: (data as Record<string, unknown>).quality as Record<string, unknown> | undefined,
      uncertainty: (data as Record<string, unknown>).uncertainty as Record<string, unknown> | undefined,
      xai: (data as Record<string, unknown>).xai as unknown[] | undefined,
    };
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

  // Read file as ArrayBuffer and convert to Float32Array
  const buffer = await file.arrayBuffer();
  const rawSignal = new Float32Array(buffer.byteLength / 4);

  // If the file is a raw float32 binary, use it directly;
  // otherwise create a zero-padded input tensor
  if (buffer.byteLength === NUM_LEADS * NUM_SAMPLES * 4) {
    rawSignal.set(new Float32Array(buffer));
  } else {
    // Zero-padded placeholder — the real preprocessing would need
    // format detection, but for edge fallback we accept raw tensors
    const available = Math.min(buffer.byteLength / 4, NUM_LEADS * NUM_SAMPLES);
    rawSignal.set(new Float32Array(buffer, 0, available));
  }

  const inputTensor = new ort.Tensor('float32', rawSignal, [1, NUM_LEADS, NUM_SAMPLES]);

  // Run inference
  const typedSession = session as { run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>> };
  const results = await typedSession.run({ input: inputTensor });

  // Collect outputs
  const predictions: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(results)) {
    predictions[key.replace('_output', '')] = Array.from(value.data);
  }

  return {
    inference_mode: 'edge_wasm',
    predictions,
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
    const results = (data['results'] as unknown[]) ?? [];

    return results.map(r => ({
      inference_mode: 'server' as InferenceMode,
      raw: r,
      predictions: (r as Record<string, unknown>)['predictions'] as Record<string, unknown> | undefined,
      quality: (r as Record<string, unknown>)['quality'] as Record<string, unknown> | undefined,
      uncertainty: (r as Record<string, unknown>)['uncertainty'] as Record<string, unknown> | undefined,
      xai: (r as Record<string, unknown>)['xai'] as unknown[] | undefined,
    }));
  } catch {
    // Fall back to sequential single-file inference
    clearTimeout(timer);
  } finally {
    clearTimeout(timer);
  }

  // Sequential fallback — process each file independently
  const results: PredictionResult[] = [];
  for (const f of files) {
    try {
      const result = await predict(f, config);
      results.push(result);
    } catch (err) {
      results.push({
        inference_mode: 'edge_wasm',
        raw: null,
        predictions: undefined,
        quality: undefined,
      });
    }
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
