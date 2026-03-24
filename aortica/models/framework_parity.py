"""TensorFlow/Keras ↔ PyTorch weight conversion and parity validation.

Provides :func:`convert_pytorch_to_tf` to transfer weights from a trained
PyTorch :class:`AorticaModel` into its TF/Keras counterpart
(:func:`build_aortica_model_tf`), and :func:`validate_parity` to assert
that both models produce equivalent outputs for the same input.

Weight Conversion Process
-------------------------
PyTorch and TF/Keras store tensor parameters in different layouts:

* **Conv1d weights**: PyTorch ``[out, in, kernel]`` →
  TF ``[kernel, in, out]`` (transpose axes 0↔2).
* **Linear / Dense weights**: PyTorch ``[out, in]`` →
  TF ``[in, out]`` (simple transpose).
* **Bias vectors**: identical layout (1-D).
* **BatchNorm / LayerNorm**: ``gamma``, ``beta``, ``running_mean``,
  ``running_var`` are all 1-D and identical in both frameworks.

The composite TF model produced by :func:`build_aortica_model_tf`
nests three sub-models (backbone, attention, per-task heads).
This module walks the PyTorch state-dict and the TF model's layer
tree to build an explicit name mapping, then copies each parameter
with the appropriate transpose.

Included as a CI test (tagged ``@pytest.mark.slow``) in
``tests/models/test_framework_parity.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # type: ignore[attr-defined]

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _check_deps() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for framework parity validation. "
            "Install with: pip install aortica[torch]"
        )
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required for framework parity validation. "
            "Install with: pip install tensorflow"
        )


# ---------------------------------------------------------------------------
# Weight conversion helpers
# ---------------------------------------------------------------------------


def _convert_conv1d_weight(
    weight: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Convert Conv1D weight from PyTorch to TF layout.

    PyTorch: ``[out_channels, in_channels, kernel_size]``
    TF:      ``[kernel_size, in_channels, out_channels]``
    """
    return np.transpose(weight, (2, 1, 0))


def _convert_linear_weight(
    weight: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Convert Linear/Dense weight from PyTorch to TF layout.

    PyTorch: ``[out_features, in_features]``
    TF:      ``[in_features, out_features]``
    """
    return weight.T


# ---------------------------------------------------------------------------
# Name mapping between PyTorch state dict keys and TF layer names
# ---------------------------------------------------------------------------

# Backbone mapping: PyTorch param name prefix → TF layer name prefix
# The PyTorch model has ``backbone.`` prefix; the TF composite model
# nests the backbone as a sub-model called ``AorticaBackboneTF``.

_BACKBONE_MAP: list[tuple[str, str, str]] = [
    # (pytorch_key_fragment, tf_layer_name, param_type)
    # Initial conv + BN
    ("backbone.conv1.weight", "conv1", "conv_weight"),
    ("backbone.bn1.weight", "bn1", "bn_gamma"),
    ("backbone.bn1.bias", "bn1", "bn_beta"),
    ("backbone.bn1.running_mean", "bn1", "bn_mean"),
    ("backbone.bn1.running_var", "bn1", "bn_var"),
]


def _make_res_stage_map(
    stage_idx: int, block_idx: int
) -> list[tuple[str, str, str]]:
    """Generate mapping entries for a single residual block."""
    pt_prefix = f"backbone.layer{stage_idx}.{block_idx}"
    tf_prefix = f"stage{stage_idx}_block{block_idx + 1}"
    entries: list[tuple[str, str, str]] = []

    # Main path conv1, bn1, conv2, bn2
    for conv_i in [1, 2]:
        entries.append(
            (f"{pt_prefix}.conv{conv_i}.weight", f"{tf_prefix}_conv{conv_i}", "conv_weight")
        )
        entries.append(
            (f"{pt_prefix}.bn{conv_i}.weight", f"{tf_prefix}_bn{conv_i}", "bn_gamma")
        )
        entries.append(
            (f"{pt_prefix}.bn{conv_i}.bias", f"{tf_prefix}_bn{conv_i}", "bn_beta")
        )
        entries.append(
            (f"{pt_prefix}.bn{conv_i}.running_mean", f"{tf_prefix}_bn{conv_i}", "bn_mean")
        )
        entries.append(
            (f"{pt_prefix}.bn{conv_i}.running_var", f"{tf_prefix}_bn{conv_i}", "bn_var")
        )

    # Shortcut (downsample) — only present on first block of stages 2, 3
    entries.append(
        (f"{pt_prefix}.downsample.0.weight", f"{tf_prefix}_shortcut_conv", "conv_weight")
    )
    entries.append(
        (f"{pt_prefix}.downsample.1.weight", f"{tf_prefix}_shortcut_bn", "bn_gamma")
    )
    entries.append(
        (f"{pt_prefix}.downsample.1.bias", f"{tf_prefix}_shortcut_bn", "bn_beta")
    )
    entries.append(
        (f"{pt_prefix}.downsample.1.running_mean", f"{tf_prefix}_shortcut_bn", "bn_mean")
    )
    entries.append(
        (f"{pt_prefix}.downsample.1.running_var", f"{tf_prefix}_shortcut_bn", "bn_var")
    )

    return entries


def _build_full_backbone_map() -> list[tuple[str, str, str]]:
    """Build complete backbone weight mapping list."""
    mapping = list(_BACKBONE_MAP)
    for stage in [1, 2, 3]:
        for block in [0, 1]:
            mapping.extend(_make_res_stage_map(stage, block))
    return mapping


# Attention mapping: PyTorch → TF
_ATTENTION_MAP: list[tuple[str, str, str]] = [
    ("attention.q_proj.weight", "q_proj", "linear_weight"),
    ("attention.k_proj.weight", "k_proj", "linear_weight"),
    ("attention.v_proj.weight", "v_proj", "linear_weight"),
    ("attention.out_proj.weight", "out_proj", "linear_weight"),
    ("attention.layer_norm.weight", "layer_norm", "ln_gamma"),
    ("attention.layer_norm.bias", "layer_norm", "ln_beta"),
]

# Task head mapping template
_TASK_HEAD_NAMES = {
    "rhythm": ("rhythm_head", "rhythm"),
    "structural": ("structural_head", "structural"),
    "ischaemia": ("ischaemia_head", "ischaemia"),
    "risk": ("risk_head", "risk"),
}


def _make_task_head_map(
    task_name: str,
) -> list[tuple[str, str, str]]:
    """Generate mapping entries for a task head."""
    pt_head, tf_prefix = _TASK_HEAD_NAMES[task_name]
    return [
        (f"{pt_head}.classifier.0.weight", f"{tf_prefix}_fc1", "linear_weight"),
        (f"{pt_head}.classifier.0.bias", f"{tf_prefix}_fc1", "linear_bias"),
        (f"{pt_head}.classifier.3.weight", f"{tf_prefix}_logits", "linear_weight"),
        (f"{pt_head}.classifier.3.bias", f"{tf_prefix}_logits", "linear_bias"),
    ]


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------


def _find_tf_layer(
    model: "keras.Model", layer_name: str
) -> "keras.layers.Layer":
    """Recursively find a layer by name in a (potentially nested) Keras model."""
    # Direct lookup
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        # Recurse into sub-models
        if isinstance(layer, keras.Model):
            try:
                return _find_tf_layer(layer, layer_name)
            except KeyError:
                continue
    raise KeyError(f"Layer '{layer_name}' not found in model '{model.name}'.")


def _set_tf_weight(
    model: "keras.Model",
    tf_layer_name: str,
    param_type: str,
    np_value: npt.NDArray[np.floating[Any]],
) -> None:
    """Set a single weight in a TF model layer."""
    layer = _find_tf_layer(model, tf_layer_name)
    weights = layer.get_weights()

    if param_type == "conv_weight":
        converted = _convert_conv1d_weight(np_value)
        weights[0] = converted
    elif param_type == "linear_weight":
        converted = _convert_linear_weight(np_value)
        weights[0] = converted
    elif param_type == "linear_bias":
        weights[1] = np_value
    elif param_type == "bn_gamma":
        weights[0] = np_value
    elif param_type == "bn_beta":
        weights[1] = np_value
    elif param_type == "bn_mean":
        weights[2] = np_value
    elif param_type == "bn_var":
        weights[3] = np_value
    elif param_type == "ln_gamma":
        weights[0] = np_value
    elif param_type == "ln_beta":
        weights[1] = np_value
    else:
        raise ValueError(f"Unknown param type: {param_type}")

    layer.set_weights(weights)


def convert_pytorch_to_tf(
    pt_model: "torch.nn.Module",
    tf_model: "keras.Model",
    enabled_tasks: list[str] | None = None,
) -> list[str]:
    """Transfer weights from a PyTorch AorticaModel to a TF/Keras model.

    Args:
        pt_model: Trained PyTorch :class:`AorticaModel`.
        tf_model: A TF model built via :func:`build_aortica_model_tf`
            with matching architecture.
        enabled_tasks: Tasks enabled in the model.  Default: all four.

    Returns:
        List of PyTorch state-dict keys that were transferred.

    Raises:
        KeyError: If a required TF layer is not found.
    """
    _check_deps()

    if enabled_tasks is None:
        enabled_tasks = list(_TASK_HEAD_NAMES.keys())

    state_dict = pt_model.state_dict()
    transferred: list[str] = []

    # Build complete mapping
    mapping: list[tuple[str, str, str]] = []
    mapping.extend(_build_full_backbone_map())
    mapping.extend(_ATTENTION_MAP)
    for task in enabled_tasks:
        mapping.extend(_make_task_head_map(task))

    for pt_key, tf_name, param_type in mapping:
        if pt_key not in state_dict:
            # Skip optional params (e.g. downsample on blocks without it)
            continue
        np_value = state_dict[pt_key].detach().cpu().numpy()
        _set_tf_weight(tf_model, tf_name, param_type, np_value)
        transferred.append(pt_key)

    return transferred


# ---------------------------------------------------------------------------
# Parity validation
# ---------------------------------------------------------------------------


def validate_parity(
    pt_model: "torch.nn.Module",
    tf_model: "keras.Model",
    input_shape: tuple[int, ...] = (2, 12, 2500),
    atol: float = 1e-5,
    seed: int = 42,
    enabled_tasks: list[str] | None = None,
) -> dict[str, float]:
    """Validate that PyTorch and TF models produce equivalent outputs.

    Feeds the same random input through both models and checks that
    all task outputs are within floating-point tolerance.

    Args:
        pt_model: PyTorch :class:`AorticaModel` (weights already copied).
        tf_model: TF :class:`keras.Model` (weights already set via
            :func:`convert_pytorch_to_tf`).
        input_shape: Input tensor shape ``(batch, leads, samples)``.
        atol: Absolute tolerance.  Default ``1e-5``.
        seed: Random seed for reproducibility.
        enabled_tasks: Tasks to compare.  Default: all four.

    Returns:
        Dict mapping task name to max absolute difference.

    Raises:
        AssertionError: If any task output exceeds *atol*.
    """
    _check_deps()

    if enabled_tasks is None:
        enabled_tasks = list(_TASK_HEAD_NAMES.keys())

    np.random.seed(seed)
    input_np = np.random.randn(*input_shape).astype(np.float32)

    # PyTorch forward
    pt_model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(input_np)
        pt_output = pt_model(pt_input)

    # TF forward — transpose to channels-last [batch, samples, leads]
    tf_input = np.transpose(input_np, (0, 2, 1))
    tf_output = tf_model(tf_input, training=False)

    max_diffs: dict[str, float] = {}
    for task in enabled_tasks:
        pt_out = getattr(pt_output, task)
        if pt_out is None:
            continue
        pt_np = pt_out.numpy()

        if isinstance(tf_output, dict):
            tf_np: npt.NDArray[np.floating[Any]] = tf_output[task].numpy()
        else:
            tf_np = tf_output.numpy()

        max_diff = float(np.max(np.abs(pt_np - tf_np)))
        max_diffs[task] = max_diff

        assert max_diff < atol, (
            f"Parity check FAILED for task '{task}': "
            f"max_diff={max_diff:.2e} exceeds atol={atol:.0e}"
        )

    return max_diffs
