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

Layer name mapping
~~~~~~~~~~~~~~~~~~
=====================  ============================
PyTorch key prefix     TF layer name
=====================  ============================
backbone.conv1         conv1 (inside AorticaBackboneTF)
backbone.bn1           bn1
backbone.layer1.0      stage1_block1
backbone.layer1.1      stage1_block2
backbone.layer2.0      stage2_block1 (+ shortcut)
backbone.layer2.1      stage2_block2
backbone.layer3.0      stage3_block1 (+ shortcut)
backbone.layer3.1      stage3_block2
backbone.fc            fc_proj (projection Dense)
attention.q_proj       q_proj (inside CrossLeadAttentionTF)
attention.k_proj       k_proj
attention.v_proj       v_proj
attention.out_proj     out_proj
attention.layer_norm   layer_norm
{task}_head.classifier {task}_fc1 / {task}_logits
=====================  ============================

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
# Layer lookup
# ---------------------------------------------------------------------------


def _find_tf_layer(
    model: "keras.Model", layer_name: str
) -> "keras.layers.Layer":
    """Recursively find a layer by name in a (potentially nested) Keras model."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if isinstance(layer, keras.Model):
            try:
                return _find_tf_layer(layer, layer_name)
            except KeyError:
                continue
    raise KeyError(f"Layer '{layer_name}' not found in model '{model.name}'.")


def _tf_layer_exists(model: "keras.Model", layer_name: str) -> bool:
    """Return True if the layer exists in the model (or any nested sub-model)."""
    try:
        _find_tf_layer(model, layer_name)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------


def _set_bn_weights(
    model: "keras.Model",
    tf_layer_name: str,
    gamma: npt.NDArray[np.floating[Any]],
    beta: npt.NDArray[np.floating[Any]],
    running_mean: npt.NDArray[np.floating[Any]],
    running_var: npt.NDArray[np.floating[Any]],
) -> None:
    """Set all four BatchNorm parameters in one call."""
    layer = _find_tf_layer(model, tf_layer_name)
    layer.set_weights([gamma, beta, running_mean, running_var])


def _set_conv_weights(
    model: "keras.Model",
    tf_layer_name: str,
    weight: npt.NDArray[np.floating[Any]],
    bias: npt.NDArray[np.floating[Any]] | None = None,
) -> None:
    """Set Conv1D weights (and optional bias)."""
    layer = _find_tf_layer(model, tf_layer_name)
    converted = _convert_conv1d_weight(weight)
    if bias is not None:
        layer.set_weights([converted, bias])
    else:
        layer.set_weights([converted])


def _set_dense_weights(
    model: "keras.Model",
    tf_layer_name: str,
    weight: npt.NDArray[np.floating[Any]],
    bias: npt.NDArray[np.floating[Any]] | None = None,
) -> None:
    """Set Dense weights (and optional bias)."""
    layer = _find_tf_layer(model, tf_layer_name)
    converted = _convert_linear_weight(weight)
    if bias is not None:
        layer.set_weights([converted, bias])
    else:
        layer.set_weights([converted])


def _set_ln_weights(
    model: "keras.Model",
    tf_layer_name: str,
    gamma: npt.NDArray[np.floating[Any]],
    beta: npt.NDArray[np.floating[Any]],
) -> None:
    """Set LayerNorm gamma and beta."""
    layer = _find_tf_layer(model, tf_layer_name)
    layer.set_weights([gamma, beta])


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

# Task head name mapping: PT prefix → TF prefix
_TASK_HEAD_NAMES: dict[str, tuple[str, str]] = {
    "rhythm": ("rhythm_head", "rhythm"),
    "structural": ("structural_head", "structural"),
    "ischaemia": ("ischaemia_head", "ischaemia"),
    "risk": ("risk_head", "risk"),
}


def convert_pytorch_to_tf(
    pt_model: "torch.nn.Module",
    tf_model: "keras.Model",
    enabled_tasks: list[str] | None = None,
) -> list[str]:
    """Transfer weights from a PyTorch AorticaModel to a TF/Keras model.

    Each PyTorch Conv1d weight is transposed from ``[out, in, kernel]`` to
    ``[kernel, in, out]`` for TF.  Linear/Dense weights are transposed from
    ``[out, in]`` to ``[in, out]``.  BatchNorm and LayerNorm parameters are
    identical in both frameworks and copied directly.

    Args:
        pt_model: Trained PyTorch :class:`AorticaModel`.
        tf_model: A TF model built via :func:`build_aortica_model_tf`
            with matching architecture.
        enabled_tasks: Tasks enabled in the model.  Default: all four.

    Returns:
        List of PyTorch state-dict keys that were transferred.
    """
    _check_deps()

    if enabled_tasks is None:
        enabled_tasks = list(_TASK_HEAD_NAMES.keys())

    sd = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}
    transferred: list[str] = []

    def _get(key: str) -> npt.NDArray[np.floating[Any]] | None:
        return sd.get(key)

    def _mark(*keys: str) -> None:
        for k in keys:
            if k in sd:
                transferred.append(k)

    # ------------------------------------------------------------------
    # Backbone: initial conv + BN
    # ------------------------------------------------------------------
    w = _get("backbone.conv1.weight")
    if w is not None:
        _set_conv_weights(tf_model, "conv1", w)
        _mark("backbone.conv1.weight")

    g = _get("backbone.bn1.weight")
    b = _get("backbone.bn1.bias")
    m = _get("backbone.bn1.running_mean")
    v = _get("backbone.bn1.running_var")
    if all(x is not None for x in [g, b, m, v]):
        _set_bn_weights(tf_model, "bn1", g, b, m, v)  # type: ignore[arg-type]
        _mark("backbone.bn1.weight", "backbone.bn1.bias",
              "backbone.bn1.running_mean", "backbone.bn1.running_var")

    # ------------------------------------------------------------------
    # Backbone: residual stages
    # ------------------------------------------------------------------
    for stage_idx in [1, 2, 3]:
        for block_idx in [0, 1]:
            pt_pfx = f"backbone.layer{stage_idx}.{block_idx}"
            tf_pfx = f"stage{stage_idx}_block{block_idx + 1}"

            # conv1 + bn1
            w = _get(f"{pt_pfx}.conv1.weight")
            if w is not None:
                _set_conv_weights(tf_model, f"{tf_pfx}_conv1", w)
                _mark(f"{pt_pfx}.conv1.weight")

            g, b = _get(f"{pt_pfx}.bn1.weight"), _get(f"{pt_pfx}.bn1.bias")
            m_, v_ = _get(f"{pt_pfx}.bn1.running_mean"), _get(f"{pt_pfx}.bn1.running_var")
            if all(x is not None for x in [g, b, m_, v_]):
                _set_bn_weights(tf_model, f"{tf_pfx}_bn1", g, b, m_, v_)  # type: ignore[arg-type]
                _mark(f"{pt_pfx}.bn1.weight", f"{pt_pfx}.bn1.bias",
                      f"{pt_pfx}.bn1.running_mean", f"{pt_pfx}.bn1.running_var")

            # conv2 + bn2
            w = _get(f"{pt_pfx}.conv2.weight")
            if w is not None:
                _set_conv_weights(tf_model, f"{tf_pfx}_conv2", w)
                _mark(f"{pt_pfx}.conv2.weight")

            g, b = _get(f"{pt_pfx}.bn2.weight"), _get(f"{pt_pfx}.bn2.bias")
            m_, v_ = _get(f"{pt_pfx}.bn2.running_mean"), _get(f"{pt_pfx}.bn2.running_var")
            if all(x is not None for x in [g, b, m_, v_]):
                _set_bn_weights(tf_model, f"{tf_pfx}_bn2", g, b, m_, v_)  # type: ignore[arg-type]
                _mark(f"{pt_pfx}.bn2.weight", f"{pt_pfx}.bn2.bias",
                      f"{pt_pfx}.bn2.running_mean", f"{pt_pfx}.bn2.running_var")

            # Shortcut / downsample (only on blocks that have it)
            ds_conv_key = f"{pt_pfx}.downsample.0.weight"
            if ds_conv_key in sd and _tf_layer_exists(tf_model, f"{tf_pfx}_shortcut_conv"):
                _set_conv_weights(tf_model, f"{tf_pfx}_shortcut_conv", sd[ds_conv_key])
                _mark(ds_conv_key)

                g = _get(f"{pt_pfx}.downsample.1.weight")
                b = _get(f"{pt_pfx}.downsample.1.bias")
                m_ = _get(f"{pt_pfx}.downsample.1.running_mean")
                v_ = _get(f"{pt_pfx}.downsample.1.running_var")
                if all(x is not None for x in [g, b, m_, v_]):
                    _set_bn_weights(
                        tf_model, f"{tf_pfx}_shortcut_bn", g, b, m_, v_  # type: ignore[arg-type]
                    )
                    _mark(f"{pt_pfx}.downsample.1.weight",
                          f"{pt_pfx}.downsample.1.bias",
                          f"{pt_pfx}.downsample.1.running_mean",
                          f"{pt_pfx}.downsample.1.running_var")

    # ------------------------------------------------------------------
    # Backbone: optional fc projection (when feature_dim != 256)
    # ------------------------------------------------------------------
    fc_w = _get("backbone.fc.weight")
    fc_b = _get("backbone.fc.bias")
    if fc_w is not None and _tf_layer_exists(tf_model, "fc_proj"):
        _set_dense_weights(tf_model, "fc_proj", fc_w, fc_b)
        _mark("backbone.fc.weight")
        if fc_b is not None:
            _mark("backbone.fc.bias")

    # ------------------------------------------------------------------
    # Attention: Q, K, V, output projections + LayerNorm
    # ------------------------------------------------------------------
    for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        w = _get(f"attention.{proj_name}.weight")
        if w is not None:
            _set_dense_weights(tf_model, proj_name, w)
            _mark(f"attention.{proj_name}.weight")

    g = _get("attention.layer_norm.weight")
    b = _get("attention.layer_norm.bias")
    if g is not None and b is not None:
        _set_ln_weights(tf_model, "layer_norm", g, b)
        _mark("attention.layer_norm.weight", "attention.layer_norm.bias")

    # ------------------------------------------------------------------
    # Task heads
    # ------------------------------------------------------------------
    for task in enabled_tasks:
        pt_head, tf_pfx = _TASK_HEAD_NAMES[task]
        # Determine the nn.Sequential attribute name:
        # classification heads use "classifier", risk head uses "regressor"
        seq_name = "regressor" if task == "risk" else "classifier"

        # fc1 (index 0 in nn.Sequential)
        w = _get(f"{pt_head}.{seq_name}.0.weight")
        b = _get(f"{pt_head}.{seq_name}.0.bias")
        if w is not None:
            _set_dense_weights(tf_model, f"{tf_pfx}_fc1", w, b)
            _mark(f"{pt_head}.{seq_name}.0.weight")
            if b is not None:
                _mark(f"{pt_head}.{seq_name}.0.bias")

        # logits (index 3 in nn.Sequential: Linear, ReLU, Dropout, Linear)
        w = _get(f"{pt_head}.{seq_name}.3.weight")
        b = _get(f"{pt_head}.{seq_name}.3.bias")
        if w is not None:
            _set_dense_weights(tf_model, f"{tf_pfx}_logits", w, b)
            _mark(f"{pt_head}.{seq_name}.3.weight")
            if b is not None:
                _mark(f"{pt_head}.{seq_name}.3.bias")

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

    # PyTorch forward (eval mode, no dropout)
    pt_model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(input_np)
        pt_output = pt_model(pt_input)

    # TF forward — transpose to channels-last [batch, samples, leads]
    tf_input = np.transpose(input_np, (0, 2, 1))
    tf_output = tf_model(tf_input, training=False)

    max_diffs: dict[str, float] = {}
    for task in enabled_tasks:
        pt_out = getattr(pt_output, task, None)
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
