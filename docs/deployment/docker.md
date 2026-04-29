# Docker Deployment

Aortica ships with multi-architecture Docker images for server and edge deployment.

## Server Image (amd64)

Full platform with API server, CLI, and pre-built frontend.

### Build

```bash
docker build -f Dockerfile.server -t aortica-server .
```

### Run

```bash
docker run -d \
  -p 8000:8000 \
  -v aortica-data:/app/data \
  -v aortica-cache:/root/.cache/aortica \
  --name aortica \
  aortica-server
```

The API server is available at `http://localhost:8000`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AORTICA_HOST` | `0.0.0.0` | Server bind address |
| `AORTICA_PORT` | `8000` | Server port |
| `AORTICA_CORS_ORIGINS` | `*` | Comma-separated CORS origins |

## Edge Image (arm64)

Lightweight image for edge devices (Raspberry Pi, Jetson).

### Build

```bash
docker build -f Dockerfile.edge -t aortica-edge .
```

### Run

```bash
docker run --rm aortica-edge predict /data/patient_ecg.dat
```

## Docker Compose

For local development with API server + frontend:

```bash
docker compose up
```

This starts:

- **API server** at `http://localhost:8000` with health check
- **Frontend dev server** at `http://localhost:5173`

### Services

```yaml
services:
  api:       # Aortica API server (Dockerfile.server)
  frontend:  # React dev server (node:22-slim)
```

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `aortica-data` | Local result storage (SQLite + encrypted predictions) |
| `aortica-cache` | Model cache (downloaded checkpoints) |
