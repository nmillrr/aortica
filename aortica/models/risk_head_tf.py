"""TensorFlow/Keras implementation of the Risk Prediction Task Head.

Provides :func:`build_risk_head_tf`, a Keras functional-API model producing
3 sigmoid outputs for clinical risk score prediction.

Input shape:  ``(batch, feature_dim)``
Output shape: ``(batch, 3)``
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
            "TensorFlow is required for build_risk_head_tf. "
            "Install with: pip install tensorflow"
        )


NUM_RISK_OUTPUTS: int = 3


def build_risk_head_tf(
    feature_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> "keras.Model":
    """Build a Keras functional-API risk prediction head.

    Architecture:

    * Dense(hidden_dim) + ReLU + Dropout
    * Dense(3) + Sigmoid

    Args:
        feature_dim: Input feature vector dimension.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Returns:
        A ``keras.Model`` with sigmoid output of shape ``(batch, 3)``.
    """
    _check_tf()

    inputs = keras.Input(shape=(feature_dim,), name="risk_features")

    x = keras.layers.Dense(hidden_dim, name="risk_fc1")(inputs)
    x = keras.layers.ReLU(name="risk_relu")(x)
    x = keras.layers.Dropout(dropout, name="risk_dropout")(x)
    x = keras.layers.Dense(NUM_RISK_OUTPUTS, name="risk_logits")(x)
    outputs = keras.layers.Activation("sigmoid", name="risk_sigmoid")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="RiskHeadTF",
    )
    return model
