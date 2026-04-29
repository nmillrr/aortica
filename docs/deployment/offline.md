# Offline / PWA Deployment

Aortica's web UI is a Progressive Web App (PWA) that works fully offline after first load.

## How It Works

1. **First load**: The service worker pre-caches the entire app shell, the INT8 ONNX edge model (~5–8 MB), and ONNX Runtime Web WASM binaries
2. **Subsequent loads**: Everything loads from cache — no network required
3. **Inference fallback**: When the local FastAPI server is unreachable, inference falls back to in-browser ONNX Runtime Web (WASM)

## Hybrid Inference Strategy

```
┌─────────────────────────────────────────┐
│            InferenceClient              │
│                                         │
│  1. POST /api/v1/predict (3s timeout)   │
│     ↓ success → inference_mode: server  │
│     ↓ timeout/error ↓                   │
│  2. ONNX Runtime Web (WASM) fallback    │
│     → inference_mode: edge_wasm         │
└─────────────────────────────────────────┘
```

## Connection Status

The UI shows a `ConnectionStatusBanner`:

- 🟢 **Server — Full Model** — local FastAPI server responding
- 🟡 **Offline Mode — Edge Model** — using WASM fallback

## Installing as a PWA

### Android / Chrome

1. Navigate to the Aortica web UI
2. Tap the "Add to Home Screen" prompt (or Menu → Install App)
3. The app installs with `display: standalone` — runs like a native app

### Desktop Chrome/Edge

1. Navigate to the Aortica web UI
2. Click the install icon in the address bar
3. The app opens in its own window

## Service Worker

The Vite PWA plugin generates the service worker with these caching strategies:

| Asset | Strategy | Details |
|-------|----------|---------|
| App shell (JS/CSS/HTML) | Precache | Cached at install time |
| ONNX edge model | CacheFirst | Fetched once, served from cache |
| WASM binaries | CacheFirst | ~5.9 MB gzipped |
| Google Fonts | StaleWhileRevalidate | Updated in background |
| API requests | NetworkFirst | Falls back to cache |

## Building for Offline

```bash
cd frontend
npm run build
```

The production build in `dist/` includes the service worker and all precached assets. Serve it with any static file server.
