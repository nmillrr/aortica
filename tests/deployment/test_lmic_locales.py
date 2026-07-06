"""Tests for LMIC pilot localization and deployment docs (US-061b)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCALE_DIR = REPO_ROOT / "docs" / "deployment" / "locales"
DEPLOY_DIR = REPO_ROOT / "docs" / "deployment"

COMPLETE_LOCALES = ["en", "fr"]
STUB_LOCALES = ["es", "sw"]
ALL_LOCALES = COMPLETE_LOCALES + STUB_LOCALES

REQUIRED_SECTIONS = ["tiers", "chw_steps", "guide", "troubleshooting"]
REQUIRED_TIERS = ["low_risk", "refer", "urgent"]

GUIDE_DOCS = [
    "LMIC_PILOT_GUIDE.md",
    "CHW_TRAINING.md",
    "PILOT_CHECKLIST.md",
]


def _load(code: str) -> Dict[str, Any]:
    with (LOCALE_DIR / f"{code}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Docs presence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc", GUIDE_DOCS)
def test_deployment_docs_exist(doc: str) -> None:
    assert (DEPLOY_DIR / doc).is_file(), f"Missing deployment doc: {doc}"


def test_chw_training_covers_three_tiers() -> None:
    text = (DEPLOY_DIR / "CHW_TRAINING.md").read_text(encoding="utf-8")
    assert "Low risk" in text
    assert "Refer for assessment" in text
    assert "Urgent referral recommended" in text


# ---------------------------------------------------------------------------
# Locale files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ALL_LOCALES)
def test_locale_file_exists_and_parses(code: str) -> None:
    data = _load(code)
    assert data["_meta"]["code"] == code
    for section in REQUIRED_SECTIONS:
        assert section in data, f"{code}.json missing section {section}"


@pytest.mark.parametrize("code", ALL_LOCALES)
def test_locale_has_three_tiers(code: str) -> None:
    data = _load(code)
    for tier in REQUIRED_TIERS:
        assert tier in data["tiers"], f"{code}.json missing tier {tier}"


@pytest.mark.parametrize("code", COMPLETE_LOCALES)
def test_complete_locales_are_fully_translated(code: str) -> None:
    data = _load(code)
    assert data["_meta"]["status"] == "complete"
    # Every tier label and guidance is non-empty.
    for tier in REQUIRED_TIERS:
        assert data["tiers"][tier]["label"].strip()
        assert data["tiers"][tier]["guidance"].strip()
    for step_value in data["chw_steps"].values():
        assert step_value.strip()


@pytest.mark.parametrize("code", STUB_LOCALES)
def test_stub_locales_are_marked(code: str) -> None:
    data = _load(code)
    assert data["_meta"]["status"] == "stub"


def test_stub_locales_mirror_english_keys() -> None:
    """Stub locales must have the exact same key structure as English."""
    en = _load("en")

    def key_shape(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: key_shape(v) for k, v in sorted(d.items()) if k != "_meta"}
        return None

    en_shape = key_shape(en)
    for code in STUB_LOCALES:
        assert key_shape(_load(code)) == en_shape, (
            f"{code}.json key structure diverges from en.json"
        )
