/**
 * copy-edge-model.mjs — Build script to copy the latest quantized ONNX model
 * from aortica/edge/ artifacts into frontend/public/models/.
 *
 * Usage:
 *   node scripts/copy-edge-model.mjs [source_path]
 *
 * If source_path is not provided, it looks for the model at:
 *   ../aortica/edge/aortica_edge_int8.onnx
 */

import { existsSync, copyFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULT_SOURCE = resolve(__dirname, '../../aortica/edge/aortica_edge_int8.onnx');
const TARGET_DIR = resolve(__dirname, '../public/models');
const TARGET_FILE = resolve(TARGET_DIR, 'aortica_edge_int8.onnx');

const sourcePath = process.argv[2] || DEFAULT_SOURCE;

if (!existsSync(sourcePath)) {
  console.warn(`⚠  Edge model not found at: ${sourcePath}`);
  console.warn('   Skipping copy. Use a placeholder or provide the correct path.');
  console.warn(`   Usage: node scripts/copy-edge-model.mjs <path_to_model.onnx>`);
  process.exit(0);
}

mkdirSync(TARGET_DIR, { recursive: true });
copyFileSync(sourcePath, TARGET_FILE);
console.log(`✓  Copied edge model to: ${TARGET_FILE}`);
