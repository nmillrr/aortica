# Docker Compose Full-Stack Quickstart

Spin up the entire Aortica stack — API, frontend, documentation, and edge
inference — with a single command. This guide gets you from a fresh clone to a
running stack in a few minutes.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker Engine | 24+ | [Install Docker](https://docs.docker.com/get-docker/) |
| Docker Compose | v2 (`docker compose`) | Bundled with modern Docker Desktop |
| GNU Make | any | Optional but recommended (`make dev`) |
| OpenSSL | any | Only for `make prod` (self-signed TLS cert) |
| Free disk | ~4 GB | Images + model cache |

Verify your toolchain:

```bash
docker --version
docker compose version
```

## First-run setup

1. **Clone and enter the repository:**

    ```bash
    git clone https://github.com/nmillrr/aortica.git
    cd aortica
    ```

2. **Create your environment file** from the template and edit as needed:

    ```bash
    cp .env.example .env
    ```

    At minimum, set `AORTICA_SECRET_KEY` to a long random string for any
    non-local deployment. All variables have sensible defaults for local dev.

3. **Start the development stack:**

    ```bash
    make dev
    # or, without Make:
    docker compose -f docker-compose.full.yml --profile dev up --build
    ```

    The first build compiles the frontend and installs Python dependencies, so
    it takes a few minutes. Subsequent starts reuse cached layers.

4. **Open the services:**

    | Service | URL | Description |
    |---------|-----|-------------|
    | API | <http://localhost:8000> | FastAPI backend (`/health`, `/docs-api`) |
    | Frontend | <http://localhost:5173> | React / Vite dev server |
    | Docs | <http://localhost:8001> | MkDocs Material site |

5. **Stop the stack:**

    ```bash
    make down          # stop, keep volumes
    make clean         # stop and remove named volumes
    ```

## Service architecture

```text
                     dev profile                         prod profile
 ┌──────────────┐                                ┌──────────────────────────────┐
 │   Browser    │                                │           Browser            │
 └──────┬───────┘                                └───────────────┬──────────────┘
        │ :5173  :8000  :8001                                     │ :8443 (HTTPS)
        ▼                                                         ▼
 ┌──────────────┐  ┌──────────┐  ┌──────────┐        ┌──────────────────────────┐
 │  frontend    │  │   api    │  │   docs   │        │   proxy (nginx + TLS)    │
 │  Vite dev    │  │ FastAPI  │  │  MkDocs  │        │  static SPA + reverse    │
 │  :5173       │  │  :8000   │  │  :8001   │        │  proxy /api, /docs       │
 └──────┬───────┘  └────┬─────┘  └────┬─────┘        └────────┬─────────┬───────┘
        │ depends_on    │             │                       │ /api    │ /docs
        └──────────────►│◄────────────┘                       ▼         ▼
                        │                              ┌──────────┐ ┌──────────┐
 ┌──────────────┐       │  (health-gated startup)      │   api    │ │   docs   │
 │    edge      │       ▼                              │ :8000    │ │ :8001    │
 │ ONNX Runtime │  ┌─────────────────────────┐        └──────────┘ └──────────┘
 │ arm64 image  │  │  ./data   ./models  ./logs (shared host volumes)          │
 └──────────────┘  └─────────────────────────────────────────────────────────┘
```

- **api** — FastAPI + ONNX Runtime, built from `Dockerfile.server`. Health check
  on `/health` gates the startup of dependent services.
- **frontend** — React app served by the Vite dev server (dev profile only).
- **docs** — MkDocs Material live server (reuses the API image + adds MkDocs).
- **edge** — arm64 edge image (`Dockerfile.edge`); models load on-demand per
  request to minimise idle power draw.
- **proxy** — production-only nginx reverse proxy that serves the built React
  bundle and reverse-proxies `/api` and `/docs`, terminating TLS.

Shared host directories are bind-mounted into the containers:

| Host path | Purpose |
|-----------|---------|
| `./data` | SQLite result databases |
| `./models` | Model checkpoints (`.pt` / `.onnx`) |
| `./logs` | Application logs |

## Production-like mode (`make prod`)

`make prod` builds production images and runs the stack behind an nginx reverse
proxy with TLS termination. A self-signed certificate is generated automatically
for local testing:

```bash
make prod
# equivalent to:
#   ./deploy/nginx/gen-cert.sh
#   docker compose -f docker-compose.full.yml -f docker-compose.prod.yml \
#       --profile prod up --build
```

Then open **<https://localhost:8443>** (accept the self-signed certificate
warning). The proxy routes:

- `/` → React production bundle
- `/api/` → FastAPI backend
- `/docs/` → MkDocs site

!!! warning "Self-signed certificate"
    The generated certificate (`deploy/nginx/certs/`) is for **local testing
    only**. Use a real CA-issued certificate for any internet-facing deployment.

## Configuration reference

All settings live in `.env` (copied from `.env.example`). Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSE_PROFILES` | `dev` | Active profile (`dev` or `prod`) |
| `AORTICA_MODEL_PATH` | *(empty)* | In-container path to a model checkpoint |
| `AORTICA_SECRET_KEY` | `change-me-...` | JWT signing secret — **change this** |
| `AORTICA_OAUTH_CLIENT_ID` | *(empty)* | OAuth client id (Google/GitHub) |
| `AORTICA_SYNC_URL` | *(empty)* | Central sync server URL for edge upload |
| `AORTICA_LOG_LEVEL` | `info` | `debug`/`info`/`warning`/`error` |
| `AORTICA_API_PORT` | `8000` | Host port for the API |
| `AORTICA_FRONTEND_PORT` | `5173` | Host port for the Vite dev server |
| `AORTICA_DOCS_PORT` | `8001` | Host port for the docs site |
| `AORTICA_HTTPS_PORT` | `8443` | Host HTTPS port for the nginx proxy |

## Troubleshooting

**`port is already allocated`**
: Another process holds the port. Change the mapping in `.env` (e.g.
  `AORTICA_API_PORT=8010`) and re-run, or stop the conflicting process.

**API container is `unhealthy` / dependents never start**
: The `frontend`, `docs`, and `proxy` services wait for the API health check to
  pass. Inspect the API logs: `docker compose -f docker-compose.full.yml logs api`.
  A slow first model load can exceed the `start_period`; give it a minute.

**`frontend` shows `npm install` errors**
: Remove the cached node modules volume and rebuild:
  `docker compose -f docker-compose.full.yml down -v && make dev`.

**`make prod` fails with a TLS/certificate error**
: Ensure OpenSSL is installed and regenerate the certificate:
  `rm -f deploy/nginx/certs/aortica.* && make certs`.

**Browser rejects the HTTPS connection in prod**
: This is expected with a self-signed certificate — accept the security warning,
  or import `deploy/nginx/certs/aortica.crt` into your trust store.

**Changes to Python code are not reflected**
: The API image bakes the package at build time. Rebuild with `make dev` (which
  passes `--build`) after changing backend code.

**Reset everything**
: `make clean` stops the stack and removes named volumes. Delete `./data`,
  `./models`, and `./logs` contents to start completely fresh.

## Validate the compose configuration

```bash
make config
# or
docker compose -f docker-compose.full.yml -f docker-compose.prod.yml \
    --profile dev --profile prod config
```

This is also exercised by the automated test suite
(`tests/deployment/test_docker_compose.py`), which validates the compose files
and lints the Dockerfiles with `hadolint`.
