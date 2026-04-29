# CLI Reference

Aortica provides a command-line interface via the `aortica` command.

## Global Usage

```bash
aortica [COMMAND] [OPTIONS]
```

## Commands

### `aortica predict`

Run AI inference on an ECG file.

```bash
aortica predict <file> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `table` | Output format: `table` or `json` |
| `--tasks` | all | Comma-separated task heads: `rhythm,structural,ischaemia,risk` |
| `--model` | auto | Path to model checkpoint (auto-downloads from HuggingFace Hub if omitted) |
| `--ecg-format` | auto | Explicit ECG file format override |

**Examples:**

```bash
# Basic prediction with rich table output
aortica predict patient_ecg.dat

# JSON output for scripting
aortica predict patient_ecg.dat --format json

# Rhythm analysis only
aortica predict patient_ecg.dat --tasks rhythm

# Use a custom model checkpoint
aortica predict patient_ecg.dat --model ./checkpoints/best.pt
```

---

### `aortica benchmark`

Run the evaluation harness on a dataset.

```bash
aortica benchmark <dataset_path> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `table` | Output format: `table`, `json`, or `csv` |
| `--tasks` | all | Comma-separated task heads |
| `--model` | auto | Model checkpoint path |
| `--csv-export` | — | Path to write CSV results |
| `--sampling-rate` | `500` | ECG sampling rate (Hz) |
| `--batch-size` | `32` | Batch size for evaluation |
| `--seed` | `42` | Random seed for reproducibility |

**Examples:**

```bash
# Full benchmark with table output
aortica benchmark /data/ptbxl/

# Export to CSV
aortica benchmark /data/ptbxl/ --csv-export results.csv

# Custom batch size and seed
aortica benchmark /data/ptbxl/ --batch-size 64 --seed 123
```

---

### `aortica train`

Train the multi-task model from a YAML config file.

```bash
aortica train <config.yaml> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `pytorch` | Training backend: `pytorch` or `tensorflow` |

**Example config (`config.yaml`):**

```yaml
dataset_path: /data/ptbxl/
sampling_rate: 500
epochs: 50
batch_size: 32
learning_rate: 0.001
warmup_epochs: 5
gradient_clip: 1.0
enabled_tasks:
  - rhythm
  - structural
  - ischaemia
  - risk
loss_weights:
  rhythm: 1.0
  structural: 1.0
  ischaemia: 1.0
  risk: 0.5
save_metric: rhythm_f1
checkpoint_dir: ./checkpoints/
```

---

### `aortica info`

Display information about the currently loaded model.

```bash
aortica info
```

Shows: package version, cached model version/variant, checkpoint source (Hub vs local), SHA-256 hash, and training data attribution.

---

### `aortica --help`

```bash
aortica --help
```

Lists all available commands and global options.
