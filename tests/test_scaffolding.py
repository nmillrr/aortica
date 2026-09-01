"""Smoke test to verify package is importable and structured correctly."""

import re
from importlib import metadata

import aortica
import aortica.data
import aortica.evaluation
import aortica.io
import aortica.models
import aortica.signal
import aortica.utils
import aortica.xai


def test_version() -> None:
    """Package exposes a version string that matches its distribution metadata.

    Pinned to a literal this drifted silently — it still asserted 0.1.0 long
    after the package reached 0.2.0, and no workflow ran on push to notice.
    """
    assert isinstance(aortica.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", aortica.__version__), (
        f"__version__ is not semver: {aortica.__version__!r}"
    )

    try:
        installed = metadata.version("aortica")
    except metadata.PackageNotFoundError:
        # Running from a source tree without an install; nothing to compare.
        return
    assert aortica.__version__ == installed


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
