"""TensorFlow/Keras implementation of the Ischaemia & Metabolic Task Head.

Provides :func:`build_ischaemia_head_tf`, a Keras functional-API model producing
19 sigmoid outputs for ischaemic and metabolic ECG pattern classification.

Input shape:  ``(batch, feature_dim)``
Output shape: ``(batch, 19)``
"""

from __future__ import annotations

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # type: ignore[attr-defined]

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _check_tf() -> None:
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required for build_ischaemia_head_tf. "
            "Install with: pip install tensorflow"
        )


NUM_ISCHAEMIA_CLASSES: int = 19


def build_ischaemia_head_tf(
    feature_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a Keras functional-API ischaemia classification head.

    Architecture:

    * Dense(hidden_dim) + ReLU + Dropout
    * Dense(19) + Sigmoid

    Args:
        feature_dim: Input feature vector dimension.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Returns:
        A ``keras.Model`` with sigmoid output of shape ``(batch, 19)``.
    """
    _check_tf()

    inputs = keras.Input(shape=(feature_dim,), name="ischaemia_features")

    x = keras.layers.Dense(hidden_dim, name="ischaemia_fc1")(inputs)
    x = keras.layers.ReLU(name="ischaemia_relu")(x)
    x = keras.layers.Dropout(dropout, name="ischaemia_dropout")(x)
    x = keras.layers.Dense(NUM_ISCHAEMIA_CLASSES, name="ischaemia_logits")(x)
    outputs = keras.layers.Activation("sigmoid", name="ischaemia_sigmoid")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="IschaemiaHeadTF",
    )
    return model
