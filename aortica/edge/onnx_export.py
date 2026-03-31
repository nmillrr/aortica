"""ONNX export pipeline for AorticaModel.

Provides :func:`export_onnx` to export an :class:`~aortica.models.AorticaModel`
to the ONNX format with dynamic axes for batch size and signal length.

After export, :func:`validate_onnx` verifies numerical equivalence between the
PyTorch model and the ONNX Runtime inference session (default atol=1e-4).

Example::

    from aortica.models import AorticaModel
    from aortica.edge import export_onnx, validate_onnx

    model = AorticaModel(enabled_tasks=["rhythm", "risk"])
    export_onnx(model, "aortica.onnx", opset_version=17)
    validate_onnx(model, "aortica.onnx")  # raises if outputs diverge
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime  # noqa: F401

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def _check_dependencies() -> None:
    """Ensure torch, onnx, and onnxruntime are available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ONNX export. "
            "Install with: pip install aortica[torch]"
        )
    if not HAS_ONNX:
        raise ImportError(
            "ONNX is required for model export. "
            "Install with: pip install aortica[edge]"
        )


class _OnnxWrapper(nn.Module):
    """Thin wrapper that converts AorticaModel output to a flat tuple.

    ``torch.onnx.export`` requires the forward method to return a tensor
    or a tuple/list of tensors — not a dataclass.  This wrapper calls the
    underlying model, collects outputs from all *enabled* task heads in
    a deterministic order, and returns them as a tuple.

    The output order matches ``model.enabled_tasks`` (alphabetical is *not*
    assumed; construction order is preserved).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass returning a tuple of task-head tensors."""
        output = self.model(x)
        results: list[torch.Tensor] = []
        for task in self.model.enabled_tasks:
            tensor = getattr(output, task)
            if tensor is not None:
                results.append(tensor)
        return tuple(results)


def export_onnx(
    model: nn.Module,
    output_path: Union[str, pathlib.Path],
    opset_version: int = 17,
    sample_leads: int = 12,
    sample_length: int = 5000,
) -> pathlib.Path:
    """Export an :class:`~aortica.models.AorticaModel` to ONNX format.

    Dynamic axes are configured for both batch size (dim 0) and signal
    length (dim 2) so the exported model accepts variable-length ECG
    inputs.

    Args:
        model: A trained ``AorticaModel`` instance (PyTorch).
        output_path: Destination file path for the ``.onnx`` model.
        opset_version: ONNX opset version.  Default ``17``.
        sample_leads: Number of ECG leads in the dummy input.  Default ``12``.
        sample_length: Number of samples in the dummy input.  Default ``5000``.

    Returns:
        Resolved :class:`~pathlib.Path` to the saved ONNX model.

    Raises:
        ImportError: If PyTorch or ONNX is not installed.
        ValueError: If the model has no enabled task heads.
    """
    _check_dependencies()

    enabled_tasks: list[str] = model.enabled_tasks  # type: ignore[attr-defined]
    if not enabled_tasks:
        raise ValueError("Model has no enabled task heads — nothing to export.")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap model so forward returns a flat tuple of tensors.
    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, sample_leads, sample_length)

    # Build output name list from enabled tasks
    output_names = [f"{task}_output" for task in enabled_tasks]

    # Dynamic axes: batch + signal length for input; batch for each output
    dynamic_axes: dict[str, dict[int, str]] = {
        "ecg_input": {0: "batch_size", 2: "signal_length"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        opset_version=opset_version,
        input_names=["ecg_input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Validate the exported graph is structurally sound
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return output_path.resolve()


def validate_onnx(
    model: nn.Module,
    onnx_path: Union[str, pathlib.Path],
    atol: float = 1e-4,
    sample_leads: int = 12,
    sample_length: int = 5000,
    batch_size: int = 2,
    seed: Optional[int] = 42,
) -> dict[str, float]:
    """Validate ONNX model outputs match the PyTorch model.

    Feeds the same random input through both the PyTorch model and the
    ONNX Runtime inference session, then asserts per-task outputs are
    within ``atol`` absolute tolerance.

    Args:
        model: The original PyTorch ``AorticaModel``.
        onnx_path: Path to the exported ``.onnx`` file.
        atol: Absolute tolerance for output comparison.  Default ``1e-4``.
        sample_leads: Number of ECG leads.  Default ``12``.
        sample_length: Number of signal samples.  Default ``5000``.
        batch_size: Batch size for validation input.  Default ``2``.
        seed: Random seed for reproducibility.  Default ``42``.

    Returns:
        Dictionary mapping each task name to the maximum absolute
        difference observed between PyTorch and ONNX outputs.

    Raises:
        ImportError: If onnxruntime is not installed.
        AssertionError: If any output exceeds ``atol``.
    """
    _check_dependencies()
    if not HAS_ONNXRUNTIME:
        raise ImportError(
            "onnxruntime is required for validation. "
            "Install with: pip install aortica[edge]"
        )
    import numpy as np

    onnx_path = pathlib.Path(onnx_path)

    enabled_tasks: list[str] = model.enabled_tasks  # type: ignore[attr-defined]

    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to numpy, handling version incompatibilities."""
        t = t.cpu().detach().contiguous()
        try:
            arr: np.ndarray = t.numpy()
            return arr
        except RuntimeError:
            # PyTorch compiled without numpy support for installed numpy.
            # Use DLPack zero-copy protocol (numpy >= 1.22, torch >= 1.10).
            try:
                arr = np.from_dlpack(t)  # type: ignore[arg-type]
                return arr
            except (TypeError, AttributeError):
                # Ultimate fallback via Python lists
                arr = np.array(t.tolist(), dtype=np.float32)
                return arr

    # Deterministic input
    if seed is not None:
        torch.manual_seed(seed)
    test_input = torch.randn(batch_size, sample_leads, sample_length)

    # PyTorch forward
    model.eval()
    with torch.no_grad():
        pytorch_output = model(test_input)

    pytorch_results: dict[str, np.ndarray] = {}
    for task in enabled_tasks:
        tensor = getattr(pytorch_output, task)
        if tensor is not None:
            pytorch_results[task] = _to_numpy(tensor)

    # ONNX Runtime forward
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_np = _to_numpy(test_input)
    onnx_outputs = session.run(None, {input_name: input_np})

    # Compare per-task
    max_diffs: dict[str, float] = {}
    for i, task in enumerate(enabled_tasks):
        if task in pytorch_results:
            onnx_arr = onnx_outputs[i]
            pytorch_arr = pytorch_results[task]
            max_diff = float(np.max(np.abs(onnx_arr - pytorch_arr)))
            max_diffs[task] = max_diff

            assert max_diff <= atol, (
                f"ONNX validation failed for task '{task}': "
                f"max absolute diff {max_diff:.6f} exceeds atol={atol}"
            )

    return max_diffs
