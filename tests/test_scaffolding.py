"""Smoke test to verify package is importable and structured correctly."""

import aortica
import aortica.data
import aortica.evaluation
import aortica.io
import aortica.models
import aortica.signal
import aortica.utils
import aortica.xai


def test_version() -> None:
    """Package exposes a version string."""
    assert isinstance(aortica.__version__, str)
    assert aortica.__version__ == "0.1.0"


def test_subpackages_importable() -> None:
    """All subpackages can be imported without error."""
    # If we got here, the imports at the top already succeeded.
    assert aortica.io is not None
    assert aortica.signal is not None
    assert aortica.models is not None
    assert aortica.xai is not None
    assert aortica.evaluation is not None
    assert aortica.data is not None
    assert aortica.utils is not None
