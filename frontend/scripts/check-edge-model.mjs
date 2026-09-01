/**
 * Fail the build when the bundled edge model is a placeholder.
 *
 * `InferenceClient` points onnxruntime-web at
 * `public/models/aortica_edge_int8.onnx` as its offline fallback — the
 * "inference continues when the server is unreachable" claim rests entirely
 * on that file. It shipped as two bytes (`08 09`) for the life of the PWA,
 * so every offline build failed at runtime, in the field, silently.
 *
 * Size is the only cheap discriminator: `08 09` is a syntactically valid
 * start to an ONNX protobuf, so parsing the header would not have caught it.
 * A quantised MobileNet-1D cannot be under 100 KB.
 *
 * Run `python3 scripts/sync_release_artifacts.py` from the repo root to
 * refresh it from `artifacts_combined/`.
 */
import { statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

const MIN_BYTES = 100_000;

const here = dirname(fileURLToPath(import.meta.url));
const modelPath = join(here, '..', 'public', 'models', 'aortica_edge_int8.onnx');
const shown = relative(process.cwd(), modelPath);

let size;
try {
  size = statSync(modelPath).size;
} catch {
  console.error(
    `\nEdge model missing: ${shown}\n` +
      'The PWA offline fallback cannot work without it.\n' +
      'Run: python3 scripts/sync_release_artifacts.py\n',
  );
  process.exit(1);
}

if (size < MIN_BYTES) {
  console.error(
    `\nEdge model is a placeholder: ${shown} is ${size} bytes ` +
      `(minimum ${MIN_BYTES}).\n` +
      'Shipping this build would leave offline inference broken.\n' +
      'Run: python3 scripts/sync_release_artifacts.py\n',
  );
  process.exit(1);
}

console.log(`Edge model OK: ${shown} (${size.toLocaleString()} bytes)`);
