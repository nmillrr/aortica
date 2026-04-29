# Code Style

Aortica follows consistent code style conventions enforced by automated tooling.

## Python

### Formatter / Linter

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check aortica/ tests/
ruff check --fix aortica/ tests/
```

### Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
```

### Type Hints

All functions must have type annotations:

```python
def read_ecg(
    path: str | Path,
    target_rate: int | None = 500,
    format: str | None = None,
) -> ECGRecord:
    ...
```

### Docstrings

Use Google-style docstrings (parsed by mkdocstrings):

```python
def denoise(ecg_record: ECGRecord, methods: list[str] | None = None) -> ECGRecord:
    """Remove noise from an ECG recording.

    Applies configurable filter stages to clean the signal.

    Args:
        ecg_record: Input ECG recording.
        methods: Filter methods to apply. Options: 'baseline', 'powerline', 'highfreq'.
            Defaults to all three.

    Returns:
        A new ECGRecord with cleaned signals.

    Raises:
        ValueError: If an unknown method is specified.
    """
```

### Optional Dependencies

Aortica uses a conditional import pattern for optional dependencies:

```python
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("torch is required: pip install aortica[torch]")
```

### Test Patterns

- Use `pytest.importorskip("torch")` for tests requiring optional deps
- Synthetic data fixtures preferred over external data dependencies
- Mark slow tests with `@pytest.mark.slow`

## TypeScript (Frontend)

- Strict mode enabled (`strict: true` in `tsconfig.json`)
- Functional React components with typed props
- CSS Modules or component-level `.css` files (no Tailwind)

## Git

### Commit Messages

Follow conventional commits:

```
feat: add STEMI detection to ischaemia head
fix: correct lead normalization for non-12-lead ECGs
docs: add deployment guide for Raspberry Pi
test: add unit tests for sync engine
```

### Branching

- `main` — stable release branch
- Feature branches: `feat/us-XXX-description`
