"""ECG management system plugin architecture (US-118).

Provides a uniform :class:`ECGSystemPlugin` interface, a name registry, and
reference plugins (MUSE/DICOM, FHIR, file-watcher) plus a daemon runner that
polls a configured ECG management system, runs inference, and submits
results back.

Example::

    from aortica.integration.plugins import get_plugin, load_plugins_config

    cfg = load_plugins_config("plugins.yaml")
    plugin = get_plugin(cfg[0].plugin_type)()
    plugin.connect(cfg[0].config)
"""

from __future__ import annotations

from aortica.integration.plugins.base import (
    ECGSystemPlugin,
    PluginHealth,
    ProcessedECG,
    ProcessReport,
    default_critical_detector,
)
from aortica.integration.plugins.daemon import (
    PluginConfig,
    PluginDaemon,
    load_plugins_config,
)
from aortica.integration.plugins.fhir_plugin import FHIRPlugin
from aortica.integration.plugins.file_watcher import FileWatcherPlugin
from aortica.integration.plugins.muse import MusePlugin
from aortica.integration.plugins.registry import (
    get_plugin,
    list_plugins,
    register_plugin,
    unregister_plugin,
)

# Register the reference plugins.
register_plugin("file_watcher", FileWatcherPlugin)
register_plugin("muse", MusePlugin)
register_plugin("fhir", FHIRPlugin)

__all__ = [
    "ECGSystemPlugin",
    "FHIRPlugin",
    "FileWatcherPlugin",
    "MusePlugin",
    "PluginConfig",
    "PluginDaemon",
    "PluginHealth",
    "ProcessReport",
    "ProcessedECG",
    "default_critical_detector",
    "get_plugin",
    "list_plugins",
    "load_plugins_config",
    "register_plugin",
    "unregister_plugin",
]
