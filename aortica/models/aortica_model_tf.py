"""TensorFlow/Keras implementation of the unified multi-task model.

Provides :func:`build_aortica_model_tf`, a Keras functional-API model that
composes the backbone, cross-lead attention, and all four task heads.

Input shape:  ``(batch, samples, leads)`` (channels-last).
Output: dict of ``{task_name: tensor}`` for each enabled task.
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
            "TensorFlow is required for build_aortica_model_tf. "
            "Install with: pip install tensorflow"
        )


# Number of classes/outputs per task head.
_TASK_OUTPUTS: dict[str, int] = {
    "rhythm": 28,
    "structural": 19,
    "ischaemia": 19,
    "risk": 6,
}


def _build_head_tf(
    features: "tf.Tensor",
    task_name: str,
    num_outputs: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> "tf.Tensor":
    """Build a single task head (Dense → ReLU → Dropout → Dense → Sigmoid)."""
    x = keras.layers.Dense(hidden_dim, name=f"{task_name}_fc1")(features)
    x = keras.layers.ReLU(name=f"{task_name}_relu")(x)
    x = keras.layers.Dropout(dropout, name=f"{task_name}_dropout")(x)
    x = keras.layers.Dense(num_outputs, name=f"{task_name}_logits")(x)
    out = keras.layers.Activation("sigmoid", name=f"{task_name}_sigmoid")(x)
    return out


def build_aortica_model_tf(
    in_channels: int = 12,
    feature_dim: int = 256,
    input_length: int | None = None,
    head_hidden_dim: int = 128,
    head_dropout: float = 0.3,
    enabled_tasks: list[str] | None = None,
) -> "keras.Model":
    """Build a Keras functional-API multi-task ECG model.

    Composes the shared backbone, cross-lead attention, and task heads
    into a single ``keras.Model``.

    Args:
        in_channels: Number of ECG leads.  Default ``12``.
        feature_dim: Backbone feature dimension.  Default ``256``.
        input_length: Optional fixed temporal length (``None`` for variable).
        head_hidden_dim: Hidden dimension for task heads.  Default ``128``.
        head_dropout: Dropout for task heads.  Default ``0.3``.
        enabled_tasks: Subset of ``['rhythm', 'structural', 'ischaemia',
            'risk']``.  Default: all four.

    Returns:
        A ``keras.Model`` with dict outputs keyed by task name.
    """
    _check_tf()

    if enabled_tasks is None:
        enabled_tasks = list(_TASK_OUTPUTS.keys())

    from aortica.models.attention_tf import build_cross_lead_attention_tf
    from aortica.models.backbone_tf import build_aortica_backbone_tf

    # Build sub-models
    backbone = build_aortica_backbone_tf(
        in_channels=in_channels,
        feature_dim=feature_dim,
        input_length=input_length,
    )

    attention_model = build_cross_lead_attention_tf(
        feature_dim=feature_dim,
        num_leads=in_channels,
    )

    # Compose: input → backbone → attention → heads
    inputs = keras.Input(
        shape=(input_length, in_channels), name="ecg_input",
    )
    features = backbone(inputs)
    enriched, _attn_weights = attention_model(features)

    # Build task heads
    outputs: dict[str, "tf.Tensor"] = {}
    for task_name in enabled_tasks:
        if task_name not in _TASK_OUTPUTS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Must be one of {list(_TASK_OUTPUTS.keys())}."
            )
        outputs[task_name] = _build_head_tf(
            enriched,
            task_name=task_name,
            num_outputs=_TASK_OUTPUTS[task_name],
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="AorticaModelTF",
    )
    return model
