"""Source-vocabulary to Aortica-output mappings, loaded from data files.

Every corpus speaks its own diagnostic vocabulary: PTB-XL uses SCP-ECG
statement codes, Chapman uses its own condition names, MIMIC-IV reaches
ICD-9/10, CODE-15 ships six plain-English labels.  Each has to be mapped onto
the model's 72 outputs, and each mapping is a set of clinical judgements that
somebody should be able to review without reading Python.

So the mappings live in ``aortica/data/label_maps/*.yaml`` and this module
reads them.  A mapping file records, per Aortica output, the source codes
that set it, a SNOMED CT concept id (once the ontology work lands), and a
free-text note for any judgement call — pooling two codes, excluding a
tempting one, or naming an output more confidently than its label supports.

The alternative — a dict literal inside each loader — is how the same
mapping decision gets made three slightly different ways in three files, and
how "AF" comes to mean something subtly different depending on which corpus
a record arrived from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

#: Directory holding the shipped mapping files.  Declared as package data in
#: ``pyproject.toml`` so it survives a wheel build.
LABEL_MAP_DIR: Path = Path(__file__).resolve().parent / "label_maps"


class LabelMapError(ValueError):
    """A mapping file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ClassMapping:
    """How one Aortica output is derived from a source vocabulary.

    Attributes:
        name: Aortica output name, e.g. ``"AF"``.
        task: Head the output belongs to — ``rhythm``, ``structural``,
            ``ischaemia`` or ``risk``.
        codes: Source codes whose presence sets this output to 1.
        snomed: SNOMED CT concept id, or ``None`` while unmapped.
        verified_test_pos: Positive count this mapping is known to reproduce
            on the corpus's test split, or ``None`` if unverified.
        note: Rationale for any judgement call in this entry.
        raw: The entry's whole YAML block.  Mapping files carry evidence
            fields beyond the ones modelled here — ``corpus_total``,
            ``implied_ratio``, ``acronyms``, ``published_test_pos`` — and
            which of those apply depends on how that corpus's mapping was
            established.  Keeping the block means a new kind of evidence
            does not need a schema change to be recorded or tested.
    """

    name: str
    task: str
    codes: tuple[str, ...]
    snomed: str | None = None
    verified_test_pos: int | None = None
    note: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelMap:
    """A whole source vocabulary mapped onto Aortica outputs."""

    version: int
    source_name: str
    source_release: str
    #: Fold numbers per split, for a corpus that publishes its own
    #: stratification (PTB-XL).  ``None`` when the corpus ships no folds.
    splits: dict[str, tuple[int, ...]] | None
    #: How to derive splits when the corpus publishes none — the raw
    #: ``split_policy`` block, e.g. ``{"kind": "random", "seed": 42,
    #: "fractions": {...}}``.  Empty when :attr:`splits` carries folds.
    split_policy: dict[str, Any]
    include_zero_likelihood: bool
    min_likelihood: float
    classes: dict[str, ClassMapping]
    unmapped_codes: dict[str, str]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    @property
    def code_to_classes(self) -> dict[str, tuple[str, ...]]:
        """Reverse index: source code -> Aortica outputs it contributes to."""
        index: dict[str, list[str]] = {}
        for mapping in self.classes.values():
            for code in mapping.codes:
                index.setdefault(code, []).append(mapping.name)
        return {code: tuple(names) for code, names in index.items()}

    def classes_for_task(self, task: str) -> tuple[str, ...]:
        """Aortica outputs this corpus can label within *task*."""
        return tuple(
            name for name, m in self.classes.items() if m.task == task
        )

    @property
    def labelable_outputs(self) -> frozenset[str]:
        """Every output this corpus supplies ground truth for.

        Feed this to :func:`aortica.data.ptbxl_labels.per_task_weights` to
        mask the columns the corpus cannot label.
        """
        return frozenset(self.classes)

    def outputs_for_codes(self, codes: Iterable[str]) -> set[str]:
        """Aortica outputs implied by a record's source codes."""
        index = self.code_to_classes
        found: set[str] = set()
        for code in codes:
            found.update(index.get(code, ()))
        return found


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise LabelMapError(f"{where}: missing required key '{key}'")
    return block[key]


@lru_cache(maxsize=None)
def load_label_map(name: str) -> LabelMap:
    """Load and validate the mapping file called *name*.

    Args:
        name: Basename without extension, e.g. ``"ptbxl_scp"``.

    Returns:
        The parsed :class:`LabelMap`.  Results are cached — the files are
        immutable package data.

    Raises:
        LabelMapError: If the file is absent, unparseable, or names a task
            or output that does not exist in the model.
    """
    path = LABEL_MAP_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in LABEL_MAP_DIR.glob("*.yaml"))
        raise LabelMapError(
            f"No label map named '{name}' in {LABEL_MAP_DIR}. "
            f"Available: {', '.join(available) or '(none)'}"
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed shipped data
        raise LabelMapError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise LabelMapError(f"{path}: top level must be a mapping")

    source = _require(raw, "source", str(path))
    presence = raw.get("presence_rule", {})
    # A corpus either publishes its own folds (PTB-XL) or it does not and
    # the split has to be derived (Chapman). Exactly one must be present.
    splits_raw = raw.get("splits")
    policy_raw = raw.get("split_policy")
    if (splits_raw is None) == (policy_raw is None):
        raise LabelMapError(
            f"{path}: provide exactly one of 'splits' (published folds) or "
            "'split_policy' (how to derive a split)"
        )
    classes_raw = _require(raw, "classes", str(path))

    # Validate against the model's real output names so a typo in the data
    # file fails here rather than silently training a column of zeros.
    from aortica.models.task_dims import TASK_NUM_OUTPUTS

    valid_outputs = _all_output_names()

    classes: dict[str, ClassMapping] = {}
    for output_name, body in classes_raw.items():
        where = f"{path}: classes.{output_name}"
        if output_name not in valid_outputs:
            raise LabelMapError(
                f"{where}: '{output_name}' is not an output of any model head. "
                f"Valid names come from the head class lists in aortica/models/."
            )
        task = _require(body, "task", where)
        if task not in TASK_NUM_OUTPUTS:
            raise LabelMapError(f"{where}: unknown task '{task}'")
        if valid_outputs[output_name] != task:
            raise LabelMapError(
                f"{where}: '{output_name}' belongs to the "
                f"'{valid_outputs[output_name]}' head, not '{task}'"
            )
        codes = _require(body, "codes", where)
        if not codes:
            raise LabelMapError(f"{where}: 'codes' must not be empty")

        classes[output_name] = ClassMapping(
            name=output_name,
            task=task,
            codes=tuple(str(c) for c in codes),
            snomed=body.get("snomed"),
            verified_test_pos=body.get("verified_test_pos"),
            note=body.get("note"),
            raw=dict(body),
        )

    return LabelMap(
        version=int(raw.get("version", 1)),
        source_name=str(_require(source, "name", str(path))),
        source_release=str(source.get("release", "unknown")),
        splits=(
            {k: tuple(int(f) for f in v) for k, v in splits_raw.items()}
            if splits_raw is not None
            else None
        ),
        split_policy=dict(policy_raw or {}),
        include_zero_likelihood=bool(
            presence.get("include_zero_likelihood", True)
        ),
        min_likelihood=float(presence.get("min_likelihood", 0.0)),
        classes=classes,
        unmapped_codes={
            str(k): str(v) for k, v in (raw.get("unmapped_codes") or {}).items()
        },
    )


@lru_cache(maxsize=1)
def _all_output_names() -> dict[str, str]:
    """Every model output name -> the task whose head owns it."""
    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.risk_head import RISK_OUTPUTS
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    return {
        **{name: "rhythm" for name in RHYTHM_CLASSES},
        **{name: "structural" for name in STRUCTURAL_CLASSES},
        **{name: "ischaemia" for name in ISCHAEMIA_CLASSES},
        **{name: "risk" for name in RISK_OUTPUTS},
    }
