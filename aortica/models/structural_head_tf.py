"""TensorFlow/Keras implementation of the Structural & Functional Task Head.

Provides :func:`build_structural_head_tf`, a Keras functional-API model producing
15 sigmoid outputs for structural and functional cardiac classification.

Input shape:  ``(batch, feature_dim)``
Output shape: ``(batch, 15)``
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
            "TensorFlow is required for build_structural_head_tf. "
            "Install with: pip install tensorflow"
        )


NUM_STRUCTURAL_CLASSES: int = 15


def build_structural_head_tf(
    feature_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a Keras functional-API structural classification head.

    Architecture:

    * Dense(hidden_dim) + ReLU + Dropout
    * Dense(15) + Sigmoid

    Args:
        feature_dim: Input feature vector dimension.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Returns:
        A ``keras.Model`` with sigmoid output of shape ``(batch, 15)``.
    """
    _check_tf()

    inputs = keras.Input(shape=(feature_dim,), name="structural_features")

    x = keras.layers.Dense(hidden_dim, name="structural_fc1")(inputs)
    x = keras.layers.ReLU(name="structural_relu")(x)
    x = keras.layers.Dropout(dropout, name="structural_dropout")(x)
    x = keras.layers.Dense(NUM_STRUCTURAL_CLASSES, name="structural_logits")(x)
    outputs = keras.layers.Activation("sigmoid", name="structural_sigmoid")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="StructuralHeadTF",
    )
    return model
