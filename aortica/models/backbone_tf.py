"""TensorFlow/Keras implementation of the shared ResNet backbone encoder.

Provides :class:`AorticaBackboneTF`, a Keras ``Model`` with architecture
identical to the PyTorch :class:`AorticaBackbone`:

* Initial Conv1D (kernel=15, stride=2) + BN + ReLU + MaxPool
* 3 residual stages at 64, 128, 256 filter widths (2 blocks each)
* Global average pooling → feature vector of size ``feature_dim``

Uses ``channels_last`` data format internally.  Input shape:
``(batch, samples, leads)`` — the caller should transpose if needed.
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow import keras  # type: ignore[attr-defined]

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _check_tf() -> None:
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required for AorticaBackboneTF. "
            "Install with: pip install tensorflow"
        )


def _residual_block_tf(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 7,
    stride: int = 1,
    name_prefix: str = "res",
) -> "tf.Tensor":
    """Functional residual block (two Conv1D + BN, shortcut projection)."""
    _check_tf()
    shortcut = x

    out = keras.layers.Conv1D(
        filters, kernel_size, strides=stride, padding="same",
        use_bias=False, name=f"{name_prefix}_conv1",
    )(x)
    out = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{name_prefix}_bn1")(out)
    out = keras.layers.ReLU(name=f"{name_prefix}_relu1")(out)

    out = keras.layers.Conv1D(
        filters, kernel_size, strides=1, padding="same",
        use_bias=False, name=f"{name_prefix}_conv2",
    )(out)
    out = keras.layers.BatchNormalization(epsilon=1e-5, name=f"{name_prefix}_bn2")(out)

    # Shortcut projection when dimensions change
    if stride != 1 or x.shape[-1] != filters:
        shortcut = keras.layers.Conv1D(
            filters, 1, strides=stride, use_bias=False,
            name=f"{name_prefix}_shortcut_conv",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            epsilon=1e-5, name=f"{name_prefix}_shortcut_bn",
        )(shortcut)

    out = keras.layers.Add(name=f"{name_prefix}_add")([out, shortcut])
    out = keras.layers.ReLU(name=f"{name_prefix}_relu2")(out)
    return out


def build_aortica_backbone_tf(
    in_channels: int = 12,
    feature_dim: int = 256,
    kernel_size: int = 7,
    input_length: int | None = None,
) -> "keras.Model":
    """Build the Keras functional-API backbone model.

    Input shape: ``(batch, samples, leads)`` (channels-last).
    Output shape: ``(batch, feature_dim)``.

    Args:
        in_channels: Number of ECG leads (input channels). Default ``12``.
        feature_dim: Output feature vector dimension. Default ``256``.
        kernel_size: Kernel size for residual blocks. Default ``7``.
        input_length: Optional fixed temporal length.  Pass ``None`` for
            variable-length inputs.

    Returns:
        A compiled ``keras.Model`` instance.
    """
    _check_tf()

    inputs = keras.Input(shape=(input_length, in_channels), name="ecg_input")

    # Initial convolution
    x = keras.layers.Conv1D(
        64, 15, strides=2, padding="same", use_bias=False, name="conv1",
    )(inputs)
    x = keras.layers.BatchNormalization(epsilon=1e-5, name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="maxpool")(x)

    # Stage 1: 64 filters, 2 blocks
    x = _residual_block_tf(x, 64, kernel_size, stride=1, name_prefix="stage1_block1")
    x = _residual_block_tf(x, 64, kernel_size, stride=1, name_prefix="stage1_block2")

    # Stage 2: 128 filters, 2 blocks
    x = _residual_block_tf(x, 128, kernel_size, stride=2, name_prefix="stage2_block1")
    x = _residual_block_tf(x, 128, kernel_size, stride=1, name_prefix="stage2_block2")

    # Stage 3: 256 filters, 2 blocks
    x = _residual_block_tf(x, 256, kernel_size, stride=2, name_prefix="stage3_block1")
    x = _residual_block_tf(x, 256, kernel_size, stride=1, name_prefix="stage3_block2")

    # Global average pooling → feature vector
    x = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Optional projection
    if feature_dim != 256:
        x = keras.layers.Dense(feature_dim, name="fc_proj")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="AorticaBackboneTF")
    return model
