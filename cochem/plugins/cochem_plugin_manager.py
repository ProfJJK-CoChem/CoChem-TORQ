"""OS-Agnostic Plugin Architecture & Dynamic Lifecycle Registry.
Enforces pre-import AST security screening, safe lifecycle execution, and thread-safe registry operations.
"""

from __future__ import annotations

import abc
import ast
import importlib.util
import logging
import os
import pathlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cochem.plugins.ast_scanner import (
    PluginASTSecurityScanner,
    PluginSecurityViolationError,
)

logger = logging.getLogger("cochem.plugins")


class PluginLoadError(RuntimeError):
    """Raised when dynamic module specification loading, compilation, or class instantiation fails."""


@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata descriptor for a registered CoChem plugin."""

    name: str
    version: str
    author: str
    description: str


class PluginInterface(abc.ABC):
    """Abstract base contract for dynamically discoverable CoChem computational plugins."""

    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return immutable plugin metadata."""

    @abc.abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize plugin resources with runtime execution context."""

    @abc.abstractmethod
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scientific computation workflow on payload data."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release allocated resources and perform clean termination."""


class PluginRegistry:
    """Thread-safe dynamic registry managing plugin discovery, validation, and lifecycle."""

    def __init__(self, plugin_dir: Optional[pathlib.Path] = None) -> None:
        if plugin_dir is not None:
            self.plugin_dir: pathlib.Path = pathlib.Path(plugin_dir).resolve()
        elif "COCHEM_PLUGINS_PATH" in os.environ and os.environ["COCHEM_PLUGINS_PATH"]:
            self.plugin_dir = pathlib.Path(os.environ["COCHEM_PLUGINS_PATH"]).resolve()
        else:
            self.plugin_dir = (pathlib.Path.home() / ".cochem" / "plugins").resolve()

        self._lock: threading.RLock = threading.RLock()
        self._plugins: Dict[str, PluginInterface] = {}

    def scan_and_register(self) -> List[str]:
        """Discover, verify via AST scanner, and register valid plugins in plugin_dir."""
        with self._lock:
            if not self.plugin_dir.exists():
                self.plugin_dir.mkdir(parents=True, exist_ok=True)
                return []

            registered_names: List[str] = []
            for filepath in sorted(self.plugin_dir.glob("*.py")):
                if filepath.name.startswith("__"):
                    continue

                try:
                    plugin_instance = self._load_plugin_file(filepath)
                    if plugin_instance is not None:
                        name = plugin_instance.metadata.name
                        self._plugins[name] = plugin_instance
                        registered_names.append(name)
                except (PluginSecurityViolationError, PluginLoadError, Exception) as err:
                    logger.warning(
                        f"Skipping non-compliant plugin '{filepath.name}': {err}"
                    )

            return registered_names

    def _load_plugin_file(self, path: pathlib.Path) -> Optional[PluginInterface]:
        """Perform pre-import AST security scan, dynamically import module, and instantiate plugin."""
        code = path.read_text(encoding="utf-8")

        # Pre-import AST inspection
        tree = ast.parse(code, filename=str(path))
        scanner = PluginASTSecurityScanner()
        scanner.visit(tree)

        # Dynamic spec compilation and loading
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot load module specification for {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Interface inspection
        for attr_name in dir(module):
            cls = getattr(module, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, PluginInterface)
                and cls is not PluginInterface
            ):
                return cls()

        return None

    def get_plugin(self, name: str) -> PluginInterface:
        """Retrieve registered plugin instance by canonical name."""
        with self._lock:
            if name not in self._plugins:
                raise KeyError(f"Plugin '{name}' is not registered.")
            return self._plugins[name]

    def execute_plugin(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered plugin by name with provided input payload."""
        plugin = self.get_plugin(name)
        return plugin.execute(payload)
