"""``aortica build-index`` — build a latent-space ANN index from ECG data.

Encodes a dataset of ECGs through the AorticaModel backbone, extracts
feature vectors, and builds an approximate nearest-neighbor index for
case-based retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click


@click.command("build-index")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to ECG dataset directory (PTB-XL or compatible).",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to model checkpoint.  Defaults to latest pretrained.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Directory to write the index and metadata sidecar.",
)
@click.option(
    "--backend",
    type=click.Choice(["annoy", "faiss"], case_sensitive=False),
    default="annoy",
    show_default=True,
    help="ANN backend library.",
)
@click.option(
    "--num-trees",
    type=int,
    default=100,
    show_default=True,
    help="Number of Annoy trees (or FAISS nlist).",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size for feature extraction.",
)
@click.option(
    "--max-records",
    type=int,
    default=None,
    help="Maximum number of records to index (default: all).",
)
def build_index_cmd(
    dataset_path: str,
    model_path: Optional[str],
    output_path: str,
    backend: str,
    num_trees: int,
    batch_size: int,
    max_records: Optional[int],
) -> None:
    """Build a latent-space ANN index over ECG recordings.

    Encodes ECGs through the model backbone and builds an approximate
    nearest-neighbor index for case-based retrieval.

    Examples:

    \b
        aortica build-index --dataset ./ptbxl --output ./index
        aortica build-index --dataset ./ptbxl --model ckpt.pt --output ./index --backend faiss
    """
    try:
        from rich.console import Console
    except ImportError:
        click.echo("Error: rich is required.  pip install aortica[cli]", err=True)
        sys.exit(1)

    console = Console()

    # Load model ---------------------------------------------------------
    console.print("[bold cyan]Loading model…[/bold cyan]")
    try:
        if model_path is not None:
            import torch

            from aortica.models.aortica_model import AorticaModel

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model = AorticaModel()
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, AorticaModel):
                model = checkpoint
            else:
                model = AorticaModel()
                model.load_state_dict(checkpoint)
            model.eval()
        else:
            from aortica.models.registry import load_pretrained

            model = load_pretrained("latest")
    except Exception as exc:
        console.print(f"[bold red]Error loading model:[/bold red] {exc}")
        sys.exit(1)

    # Load dataset -------------------------------------------------------
    console.print(f"[bold cyan]Loading dataset from[/bold cyan] {dataset_path}")
    try:
        import numpy as np

        from aortica.data.ptbxl import load_ptbxl

        (train_records, train_labels), (val_records, val_labels), (test_records, test_labels) = (
            load_ptbxl(dataset_path, sampling_rate=500)
        )

        # Combine all splits for indexing
        all_records_signals: list[np.ndarray] = []
        all_labels_list: list[list[str]] = []
        all_demographics: list[dict[str, object]] = []
        all_record_ids: list[str] = []

        for split_name, records, labels_arr in [
            ("train", train_records, train_labels),
            ("val", val_records, val_labels),
            ("test", test_records, test_labels),
        ]:
            for idx, ecg_record in enumerate(records):
                all_records_signals.append(ecg_record.signals)
                # Convert numeric label vector to string labels
                label_names = []
                if labels_arr.shape[0] > idx:
                    row = labels_arr[idx]
                    for li, name in enumerate(["rhythm", "structural", "ischaemia"]):
                        if row[li] > 0.5:
                            label_names.append(name)
                all_labels_list.append(label_names)
                rec_id = f"{split_name}_{idx:06d}"
                all_record_ids.append(rec_id)
                meta = ecg_record.patient_metadata or {}
                all_demographics.append({
                    "age": meta.get("age"),
                    "sex": meta.get("sex"),
                })

        if max_records is not None and max_records < len(all_records_signals):
            all_records_signals = all_records_signals[:max_records]
            all_labels_list = all_labels_list[:max_records]
            all_record_ids = all_record_ids[:max_records]
            all_demographics = all_demographics[:max_records]

        # Stack into a single array [N, leads, samples]
        # Pad/truncate to uniform length
        target_len = 5000  # 10s at 500 Hz
        processed: list[np.ndarray] = []
        for sig in all_records_signals:
            if sig.shape[1] >= target_len:
                processed.append(sig[:, :target_len])
            else:
                pad_width = target_len - sig.shape[1]
                processed.append(
                    np.pad(sig, ((0, 0), (0, pad_width)), mode="constant")
                )
        dataset_array = np.stack(processed, axis=0).astype(np.float64)
        console.print(
            f"[green]Loaded {dataset_array.shape[0]} ECGs "
            f"(shape: {dataset_array.shape})[/green]"
        )
    except Exception as exc:
        console.print(f"[bold red]Error loading dataset:[/bold red] {exc}")
        sys.exit(1)

    # Build index --------------------------------------------------------
    console.print(
        f"[bold cyan]Building {backend} index "
        f"({num_trees} trees, batch_size={batch_size})…[/bold cyan]"
    )
    try:
        from aortica.retrieval import build_index

        report = build_index(
            model=model,
            dataset=dataset_array,
            output_path=output_path,
            labels=all_labels_list,
            record_ids=all_record_ids,
            demographics=all_demographics,
            backend=backend,
            num_trees=num_trees,
            batch_size=batch_size,
        )
    except Exception as exc:
        console.print(f"[bold red]Error building index:[/bold red] {exc}")
        sys.exit(1)

    # Report -------------------------------------------------------------
    console.print()
    console.print("[bold green]✓ Index built successfully[/bold green]")
    console.print(f"  Index file:    {report.index_path}")
    console.print(f"  Metadata file: {report.metadata_path}")
    console.print(f"  Vectors:       {report.num_vectors}")
    console.print(f"  Feature dim:   {report.feature_dim}")
    console.print(f"  Backend:       {report.backend}")
    console.print(f"  Trees/nlist:   {report.num_trees}")
    console.print(f"  Build time:    {report.build_time_seconds:.2f}s")
