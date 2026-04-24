"""Clinical Suggestion Prompt Data Layer.

Provides a structured, maintainable mapping of AI-detected conditions to
plain-language clinical suggestion prompts.  Suggestions are loaded from
an editable JSON file (``data/clinical_suggestions.json``) so that
clinicians can customise prompts without code changes.

The module exposes:

* :class:`ClinicalSuggestion` — dataclass for a single suggestion
* :data:`CONDITION_SUGGESTIONS` — dict mapping class names → suggestions
* :func:`get_suggestion` — lookup helper with fuzzy matching
* :func:`load_suggestions_from_json` — (re)load from JSON file
* :func:`create_suggestions_router` — FastAPI router factory
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClinicalSuggestion:
    """A single clinical suggestion prompt for an AI-detected condition.

    Attributes
    ----------
    prompt:
        Short (≤100 chars) plain-language clinical cue.
    urgency:
        One of ``'routine'``, ``'prompt'``, ``'urgent'``, or ``'emergent'``.
    rationale:
        1–2 sentence clinical justification.
    """

    prompt: str
    urgency: str  # 'routine' | 'prompt' | 'urgent' | 'emergent'
    rationale: str


# ---------------------------------------------------------------------------
# Pydantic response models (for API serialisation)
# ---------------------------------------------------------------------------


class ClinicalSuggestionResponse(BaseModel):
    """API response model for a single clinical suggestion."""

    condition: str = Field(..., description="Condition class name")
    prompt: str = Field(..., description="Short clinical cue (≤100 chars)")
    urgency: str = Field(
        ...,
        description="Urgency level: routine, prompt, urgent, or emergent",
    )
    rationale: str = Field(
        ...,
        description="1-2 sentence clinical justification",
    )


class SuggestionsListResponse(BaseModel):
    """API response model for a list of suggestions."""

    suggestions: List[ClinicalSuggestionResponse] = Field(
        ..., description="List of condition suggestions"
    )


# ---------------------------------------------------------------------------
# Valid urgency levels
# ---------------------------------------------------------------------------

VALID_URGENCY_LEVELS = frozenset({"routine", "prompt", "urgent", "emergent"})


# ---------------------------------------------------------------------------
# High-severity conditions that MUST have entries
# ---------------------------------------------------------------------------

HIGH_SEVERITY_CONDITIONS: List[str] = [
    "STEMI",
    "VT",
    "VF",
    "WPW",
    "av_block_3rd",
    "hyperkalaemia",
    "LVSD",
    "LBBB",
    "posterior_MI",
    "occlusive_NSTEMI",
]


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

# Default JSON location relative to the repo root.
_DEFAULT_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "clinical_suggestions.json"


def load_suggestions_from_json(
    json_path: Optional[Path] = None,
) -> Dict[str, ClinicalSuggestion]:
    """Load clinical suggestions from a JSON file.

    Parameters
    ----------
    json_path:
        Path to the JSON file.  Defaults to
        ``data/clinical_suggestions.json`` at the repo root.

    Returns
    -------
    Dict mapping condition class name → :class:`ClinicalSuggestion`.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON is malformed or missing required fields.
    """
    path = json_path or _DEFAULT_JSON_PATH

    if not path.exists():
        raise FileNotFoundError(f"Clinical suggestions JSON not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw: Any = json.load(fh)

    if not isinstance(raw, dict) or "suggestions" not in raw:
        raise ValueError(
            "Clinical suggestions JSON must contain a top-level 'suggestions' key"
        )

    suggestions: Dict[str, ClinicalSuggestion] = {}
    for condition_name, entry in raw["suggestions"].items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Suggestion entry for '{condition_name}' must be an object"
            )
        for required_key in ("prompt", "urgency", "rationale"):
            if required_key not in entry:
                raise ValueError(
                    f"Suggestion entry for '{condition_name}' is missing "
                    f"required key '{required_key}'"
                )
        urgency = entry["urgency"]
        if urgency not in VALID_URGENCY_LEVELS:
            raise ValueError(
                f"Invalid urgency '{urgency}' for '{condition_name}'. "
                f"Must be one of: {sorted(VALID_URGENCY_LEVELS)}"
            )

        suggestions[condition_name] = ClinicalSuggestion(
            prompt=str(entry["prompt"]),
            urgency=str(entry["urgency"]),
            rationale=str(entry["rationale"]),
        )

    return suggestions


# ---------------------------------------------------------------------------
# Module-level suggestion dict (lazy-loaded)
# ---------------------------------------------------------------------------

_SUGGESTIONS: Optional[Dict[str, ClinicalSuggestion]] = None


def _get_suggestions() -> Dict[str, ClinicalSuggestion]:
    """Return the module-level suggestion dict, loading from JSON on first call."""
    global _SUGGESTIONS  # noqa: PLW0603
    if _SUGGESTIONS is None:
        try:
            _SUGGESTIONS = load_suggestions_from_json()
        except FileNotFoundError:
            _SUGGESTIONS = {}
    return _SUGGESTIONS


def get_condition_suggestions() -> Dict[str, ClinicalSuggestion]:
    """Return the full condition → suggestion mapping.

    This is the public API for accessing the suggestion dict. It loads
    from the JSON file on first access and caches the result.
    """
    return dict(_get_suggestions())


def get_suggestion(condition_name: str) -> Optional[ClinicalSuggestion]:
    """Look up a clinical suggestion by exact condition class name.

    Parameters
    ----------
    condition_name:
        The condition class name (e.g. ``'AF'``, ``'STEMI'``).

    Returns
    -------
    The :class:`ClinicalSuggestion` for the condition, or ``None`` if
    no suggestion exists for the given name.
    """
    suggestions = _get_suggestions()
    return suggestions.get(condition_name)


def reload_suggestions(json_path: Optional[Path] = None) -> Dict[str, ClinicalSuggestion]:
    """Force-reload suggestions from JSON, clearing the cache.

    Returns
    -------
    The freshly loaded suggestion dict.
    """
    global _SUGGESTIONS  # noqa: PLW0603
    _SUGGESTIONS = load_suggestions_from_json(json_path)
    return dict(_SUGGESTIONS)


# ---------------------------------------------------------------------------
# FastAPI router factory
# ---------------------------------------------------------------------------


def create_suggestions_router() -> Any:
    """Create a FastAPI :class:`APIRouter` for suggestion endpoints.

    Routes
    ------
    ``GET /api/v1/suggestions/{condition_name}``
        Return the :class:`ClinicalSuggestionResponse` for a condition.

    ``GET /api/v1/suggestions``
        Return all suggestions.
    """
    try:
        from fastapi import APIRouter, HTTPException
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the suggestions router. "
            "Install with: pip install aortica[api]"
        ) from exc

    router = APIRouter(prefix="/api/v1/suggestions", tags=["suggestions"])

    @router.get(
        "/{condition_name}",
        response_model=ClinicalSuggestionResponse,
        summary="Get clinical suggestion for a condition",
    )
    async def get_suggestion_endpoint(condition_name: str) -> Any:
        """Return the clinical suggestion for a given condition class name."""
        suggestion = get_suggestion(condition_name)
        if suggestion is None:
            raise HTTPException(
                status_code=404,
                detail=f"No suggestion found for condition: {condition_name}",
            )
        return ClinicalSuggestionResponse(
            condition=condition_name,
            prompt=suggestion.prompt,
            urgency=suggestion.urgency,
            rationale=suggestion.rationale,
        )

    @router.get(
        "",
        response_model=SuggestionsListResponse,
        summary="List all clinical suggestions",
    )
    async def list_suggestions() -> Any:
        """Return all clinical suggestions."""
        suggestions = get_condition_suggestions()
        items = [
            ClinicalSuggestionResponse(
                condition=name,
                prompt=s.prompt,
                urgency=s.urgency,
                rationale=s.rationale,
            )
            for name, s in sorted(suggestions.items())
        ]
        return SuggestionsListResponse(suggestions=items)

    return router
