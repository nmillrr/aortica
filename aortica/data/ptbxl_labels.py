"""Per-output label coverage for the public training corpora.

A dataset supplies ground truth for only a subset of the model's 72 outputs.
PTB-XL has no echocardiography, serum chemistry, or follow-up data, so the
whole structural head, most of the metabolic classes, and the entire risk
head are unlabelable from it.  Training those columns against implicit
all-zero targets teaches the heads to confidently rule the findings *out*,
which is worse than leaving them untrained.

:func:`per_task_weights` returns the per-class loss weights that zero out
every unlabelable column.  ``train_multitask`` applies it by default so a
caller cannot silently omit it.

.. warning::
   :data:`LABELABLE_OUTPUTS` describes the **PTB-XL + Chapman-Shaoxing**
   corpus that produced v0.3.0.  It is not derived from the data at run
   time — adding or swapping a dataset requires updating this list, or the
   new corpus's extra labels will be silently discarded.
"""

from __future__ import annotations

from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
from aortica.models.rhythm_head import RHYTHM_CLASSES
from aortica.models.risk_head import RISK_OUTPUTS
from aortica.models.structural_head import STRUCTURAL_CLASSES

#: Outputs with ground truth in PTB-XL 1.0.3 + Chapman-Shaoxing 1.0.0 (28 of
#: 72).  Recovered from the ``trained_outputs.json`` sidecar written by the
#: v0.3.0 training run; see ``artifacts_combined/README.md``.
LABELABLE_OUTPUTS: frozenset[str] = frozenset({
    # rhythm (18 of 28)
    "AF", "AFL", "AVRT", "LBBB", "PAC", "PVC", "RBBB", "SVT", "WPW",
    "av_block_1st", "av_block_2nd", "av_block_3rd", "normal_sinus_rhythm",
    "pacemaker_rhythm", "sinus_brady", "sinus_tachy", "LAFB", "LPFB",
    # structural (4 of 19)
    "LVH", "RVH", "LA_enlargement", "RA_enlargement",
    # ischaemia (6 of 19)
    "STEMI", "posterior_MI", "old_MI", "QTc_prolongation",
    "digitalis_effect", "early_repol_vs_STEMI",
    # risk: nothing — no outcome linkage in either cohort.
})

_TASK_CLASSES: dict[str, list[str]] = {
    "rhythm": list(RHYTHM_CLASSES),
    "structural": list(STRUCTURAL_CLASSES),
    "ischaemia": list(ISCHAEMIA_CLASSES),
    "risk": list(RISK_OUTPUTS),
}


def per_task_weights(
    labelable: frozenset[str] | set[str] | None = None,
) -> dict[str, list[float]]:
    """Return per-class loss weights that mask unlabelable outputs.

    Args:
        labelable: Output names the corpus can supply ground truth for.
            Defaults to :data:`LABELABLE_OUTPUTS`.

    Returns:
        Mapping of task name to a list of per-class weights, ``1.0`` for a
        labelable column and ``0.0`` otherwise, ordered to match each head's
        class list.
    """
    names = LABELABLE_OUTPUTS if labelable is None else frozenset(labelable)
    return {
        task: [1.0 if cls in names else 0.0 for cls in classes]
        for task, classes in _TASK_CLASSES.items()
    }
