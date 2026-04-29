# Development Setup

Set up a development environment for contributing to Aortica.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Node.js 18+ (for frontend)
- Git

## Clone and Install

```bash
git clone https://github.com/nmillrr/aortica.git
cd aortica
pip install -e ".[dev,torch,cli,api]"
```

## Running Tests

```bash
# All tests
pytest

# Fast tests only (skip slow integration tests)
pytest -m "not slow"

# Specific module
pytest tests/io/
pytest tests/models/

# With coverage
pytest --cov=aortica --cov-report=term-missing
```

## Type Checking

```bash
mypy aortica/
```

## Linting

```bash
# Check
ruff check aortica/ tests/

# Auto-fix
ruff check --fix aortica/ tests/
```

## Frontend Development

```bash
cd frontend
npm install
npm run dev        # Development server at localhost:5173
npm run build      # Production build
npx tsc -b --noEmit  # TypeScript type check
```

## Documentation

```bash
pip install mkdocs-material mkdocstrings[python]

# Local preview
mkdocs serve

# Build static site
mkdocs build
```

## Project Structure

```
aortica/               # Python package
├── io/                # Format readers
├── signal/            # Signal processing
├── models/            # ML models
├── xai/               # Explainability
├── evaluation/        # Benchmarking
├── edge/              # Edge deployment
├── api/               # REST API
├── cli/               # CLI tool
├── sync/              # Offline storage
├── data/              # Dataset loaders
└── utils/             # Utilities

frontend/              # React + Vite + TypeScript
├── src/
│   ├── components/    # Reusable UI components
│   ├── pages/         # Page components
│   ├── services/      # API client, inference
│   └── contexts/      # React contexts
└── public/            # Static assets

tests/                 # Test suite (mirrors source)
docs/                  # MkDocs documentation
```

## CI/CD

GitHub Actions runs on every push and PR:

1. `ruff` — lint check
2. `mypy` — type check
3. `pytest` — test suite

See `.github/workflows/` for workflow definitions.
