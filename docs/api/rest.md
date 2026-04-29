# REST API

FastAPI-based REST API for ECG inference, comparison, and feedback.

## Endpoints

### Health & Info

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/info` | Model version, supported formats, enabled tasks |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/predict` | Single ECG inference (multipart file upload) |
| `POST` | `/api/v1/predict/batch` | Batch ECG inference (multiple files) |

**Query parameters for `/api/v1/predict`:**

- `format` — explicit format override (e.g., `wfdb`, `dicom`)
- `include_xai` — include XAI attribution data (default: `false`)
- `include_suggestions` — include clinical suggestions (default: `false`)

### Second Reader

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/compare` | Compare clinician interpretation vs AI |

### Feedback

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/feedback` | Submit clinician feedback on a finding |
| `GET` | `/api/v1/feedback/stats` | Aggregate feedback statistics |

### Clinical Suggestions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/suggestions/{condition}` | Get clinical suggestion for a condition |

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/auth/token` | Generate API key |

## App Factory

::: aortica.api.app.create_app

## Inference Pipeline

::: aortica.api.predict.run_inference_pipeline

## Batch Inference

::: aortica.api.batch_predict.run_batch_inference

## Comparison Logic

::: aortica.api.compare.compare_interpretations
