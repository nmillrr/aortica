"""TensorFlow/Keras implementation of the Rhythm & Conduction Task Head.

Provides :func:`build_rhythm_head_tf`, a Keras functional-API model producing
22 sigmoid outputs for rhythm and conduction classification.

Input shape:  ``(batch, feature_dim)``
Output shape: ``(batch, 22)``
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
            "TensorFlow is required for build_rhythm_head_tf. "
            "Install with: pip install tensorflow"
        )


NUM_RHYTHM_CLASSES: int = 22


def build_rhythm_head_tf(
    feature_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a Keras functional-API rhythm classification head.

    Architecture:

    * Dense(hidden_dim) + ReLU + Dropout
    * Dense(22) + Sigmoid

    Args:
        feature_dim: Input feature vector dimension.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Returns:
        A ``keras.Model`` with sigmoid output of shape ``(batch, 22)``.
    """
    _check_tf()

    inputs = keras.Input(shape=(feature_dim,), name="rhythm_features")

    x = keras.layers.Dense(hidden_dim, name="rhythm_fc1")(inputs)
    x = keras.layers.ReLU(name="rhythm_relu")(x)
    x = keras.layers.Dropout(dropout, name="rhythm_dropout")(x)
    x = keras.layers.Dense(NUM_RHYTHM_CLASSES, name="rhythm_logits")(x)
    outputs = keras.layers.Activation("sigmoid", name="rhythm_sigmoid")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="RhythmHeadTF",
    )
    return model
