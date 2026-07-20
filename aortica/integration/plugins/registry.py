"""Plugin registry for discovery (US-118).

A tiny name → class registry so ECG-system plugins can be registered by
vendors or hospital IT and looked up by name from configuration.
"""

from __future__ import annotations

from typing import Dict, List, Type

from aortica.integration.plugins.base import ECGSystemPlugin

_REGISTRY: Dict[str, Type[ECGSystemPlugin]] = {}


def register_plugin(name: str, cls: Type[ECGSystemPlugin]) -> None:
    """Register plugin *cls* under *name*.

    Raises:
        TypeError: If *cls* is not an :class:`ECGSystemPlugin` subclass.
        ValueError: If *name* is empty.
    """
    if not name:
        raise ValueError("plugin name must not be empty")
    if not (isinstance(cls, type) and issubclass(cls, ECGSystemPlugin)):
        raise TypeError(
            f"{cls!r} must be an ECGSystemPlugin subclass"
        )
    _REGISTRY[name] = cls


def get_plugin(name: str) -> Type[ECGSystemPlugin]:
    """Return the plugin class registered under *name*.

    Raises:
        KeyError: If no plugin is registered under *name*.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No plugin registered as {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_plugins() -> List[str]:
    """Return the sorted names of all registered plugins."""
    return sorted(_REGISTRY)


def unregister_plugin(name: str) -> None:
    """Remove *name* from the registry (mainly for tests)."""
    _REGISTRY.pop(name, None)
