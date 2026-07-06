"""Tests for duty-cycled on-demand model loading (US-061b)."""

from __future__ import annotations

import pytest

from aortica.edge.deploy_profiles import DutyCycledModelLoader, RaspberryPiProfile

# ---------------------------------------------------------------------------
# Profile duty-cycle configuration
# ---------------------------------------------------------------------------


def test_profile_has_duty_cycle_defaults() -> None:
    p = RaspberryPiProfile()
    assert p.duty_cycle_enabled is True
    assert p.inference_interval_seconds == 300
    assert p.model_unload_after_seconds == 30
    assert "Duty cycling" in p.summary()


def test_profile_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        RaspberryPiProfile(inference_interval_seconds=0)


def test_profile_rejects_negative_unload() -> None:
    with pytest.raises(ValueError):
        RaspberryPiProfile(model_unload_after_seconds=-1)


def test_profile_duty_cycle_roundtrips_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = RaspberryPiProfile(inference_interval_seconds=120, duty_cycle_enabled=False)
    path = tmp_path / "profile.json"
    p.to_json(path)
    loaded = RaspberryPiProfile.from_json(path)
    assert loaded.inference_interval_seconds == 120
    assert loaded.duty_cycle_enabled is False


# ---------------------------------------------------------------------------
# DutyCycledModelLoader
# ---------------------------------------------------------------------------


def test_loads_lazily_and_counts() -> None:
    calls = {"n": 0}

    def factory() -> object:
        calls["n"] += 1
        return object()

    loader = DutyCycledModelLoader(factory, unload_after_seconds=30)
    assert not loader.is_loaded
    m1 = loader.load()
    assert loader.is_loaded
    assert loader.load_count == 1
    # Second load reuses the resident model — no new construction.
    m2 = loader.load()
    assert m1 is m2
    assert loader.load_count == 1


def test_release() -> None:
    loader = DutyCycledModelLoader(lambda: object())
    loader.load()
    assert loader.is_loaded
    loader.release()
    assert not loader.is_loaded


def test_context_manager_immediate_unload() -> None:
    loader = DutyCycledModelLoader(lambda: object(), unload_after_seconds=0)
    with loader.session() as m:
        assert m is not None
        assert loader.is_loaded
    # unload_after_seconds == 0 → released on exit.
    assert not loader.is_loaded


def test_context_manager_deferred_unload_with_clock() -> None:
    clock = {"t": 0.0}

    loader = DutyCycledModelLoader(
        lambda: object(),
        unload_after_seconds=30,
        time_fn=lambda: clock["t"],
    )
    with loader.session():
        pass
    # Not enough idle time elapsed → still resident.
    assert loader.is_loaded

    # Advance the clock past the threshold and re-check.
    loader.load()
    clock["t"] = 100.0
    assert loader.maybe_unload() is True
    assert not loader.is_loaded


def test_maybe_unload_noop_when_unloaded() -> None:
    loader = DutyCycledModelLoader(lambda: object())
    assert loader.maybe_unload() is False


def test_rejects_negative_unload_threshold() -> None:
    with pytest.raises(ValueError):
        DutyCycledModelLoader(lambda: object(), unload_after_seconds=-1)


def test_reloads_after_release() -> None:
    calls = {"n": 0}

    def factory() -> object:
        calls["n"] += 1
        return object()

    loader = DutyCycledModelLoader(factory, unload_after_seconds=0)
    with loader.session():
        pass
    with loader.session():
        pass
    # Released and reloaded between the two context blocks.
    assert loader.load_count == 2
