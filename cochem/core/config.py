"""Unified hierarchical configuration manager for CoChem (cochem.toml).

Provenance & Specifications:
- Method Matrix [M]: Standardized calculation parameters and hardware constraints.
- Hierarchical Hierarchy [D]: CLI > ENV > Project TOML > User TOML > Legacy > Defaults.
- Cross-Platform [E]: OS-agnostic path sanitization across Windows, WSL, and POSIX.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib  # Fallback

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def sanitize_path(raw_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Sanitize and normalize paths across Windows, WSL, and POSIX environments."""
    path_str = os.path.expandvars(str(raw_path))
    expanded = Path(path_str).expanduser()
    if not expanded.is_absolute():
        base = base_dir if base_dir is not None else Path.cwd()
        expanded = base / expanded
    return expanded.resolve()


class ConfigurationParseError(Exception):
    """Raised when a TOML configuration file contains syntax or I/O errors."""

    pass


class CoreConfig(BaseModel):
    """Core runtime engine settings."""

    model_config = ConfigDict(frozen=True)

    log_level: str = "INFO"
    scratch_dir: Path = Field(default_factory=lambda: Path("./scratch").resolve())
    max_workers: int = 4
    memory_gb: int = 4


class TelemetryConfig(BaseModel):
    """Memory-mapped ring buffer and telemetry scrubber configuration."""

    model_config = ConfigDict(frozen=True)

    ring_buffer_capacity: int = 65536
    slot_size_bytes: int = 512
    mask_secrets: bool = True


class OrchestrationConfig(BaseModel):
    """Centralized SQLite task queue and heartbeat parameters."""

    model_config = ConfigDict(frozen=True)

    db_path: Path = Field(default_factory=lambda: Path("./cochem_tasks.db").resolve())
    heartbeat_interval_sec: float = 10.0
    lease_timeout_sec: float = 30.0


class DatabaseConfig(BaseModel):
    """SQLite WAL storage pragmas and timeout boundaries."""

    model_config = ConfigDict(frozen=True)

    wal_mode: bool = True
    busy_timeout_ms: int = 5000


class QmMMConfig(BaseModel):
    """Ab-initio and semi-empirical quantum chemistry package paths and memory budgets."""

    model_config = ConfigDict(frozen=True)

    orca_path: Optional[Path] = None
    crest_path: Optional[Path] = None
    xtb_path: Optional[Path] = None
    default_memory_mb: int = 4096


class CoChemRootConfig(BaseModel):
    """Top-level unified immutable CoChem configuration model."""

    model_config = ConfigDict(frozen=True)

    core: CoreConfig = Field(default_factory=CoreConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    qmmm: QmMMConfig = Field(default_factory=QmMMConfig)


class CoChemConfigManager:
    """Loads and resolves hierarchical configuration from CLI, ENV, TOML, and legacy files."""

    ENV_PREFIX: str = "COCHEM__"

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.cli_overrides = cli_overrides or {}
        self._cached_config: Optional[CoChemRootConfig] = None

    def _coerce_env_val(self, val: str) -> Any:
        """Coerce raw environment variable string into typed boolean, numeric, or string."""
        lower = val.strip().lower()
        if lower in ("true", "yes"):
            return True
        if lower in ("false", "no"):
            return False
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _extract_env_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Parse namespaced COCHEM__SECTION__FIELD environment variables."""
        overrides: Dict[str, Dict[str, Any]] = {}
        for k, v in os.environ.items():
            if k.startswith(self.ENV_PREFIX):
                remainder = k[len(self.ENV_PREFIX) :]
                parts = remainder.split("__")
                if len(parts) == 2:
                    section, field = parts[0].lower(), parts[1].lower()
                    if section not in overrides:
                        overrides[section] = {}
                    overrides[section][field] = self._coerce_env_val(v)
        return overrides

    def _load_toml_file(self, path: Path) -> Dict[str, Any]:
        """Load and parse TOML configuration file."""
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            raise ConfigurationParseError(
                f"Fatal TOML syntax error in configuration file '{path.resolve()}': {err}"
            ) from err
        except OSError as err:
            raise ConfigurationParseError(
                f"Fatal I/O or permissions error reading configuration file '{path.resolve()}': {err}"
            ) from err

    def _load_user_config(self) -> Dict[str, Any]:
        """Load user-level TOML configuration."""
        user_config_path = Path.home() / ".config" / "cochem" / "cochem.toml"
        if sys.platform == "win32" and "APPDATA" in os.environ:
            user_config_path = Path(os.environ["APPDATA"]) / "cochem" / "cochem.toml"
        return self._load_toml_file(user_config_path)

    def _load_project_config(self) -> Dict[str, Any]:
        """Load project-level TOML configuration."""
        project_toml = self.project_root / "cochem.toml"
        return self._load_toml_file(project_toml)

    def _load_legacy_config(self) -> Dict[str, Any]:
        """Check for deprecated cochem.json or config.ini and emit deprecation warning."""
        legacy_json = self.project_root / "cochem.json"
        if legacy_json.exists():
            warnings.warn(
                f"Legacy configuration format '{legacy_json.name}' is deprecated. Please migrate to cochem.toml.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                data = json.loads(legacy_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as err:
                logger.warning("Failed to parse legacy JSON config: %s", err)

        legacy_ini = self.project_root / "config.ini"
        if legacy_ini.exists():
            warnings.warn(
                f"Legacy configuration format '{legacy_ini.name}' is deprecated. Please migrate to cochem.toml.",
                DeprecationWarning,
                stacklevel=2,
            )
        return {}

    def load_config(self) -> CoChemRootConfig:
        """Resolve full hierarchical configuration pipeline."""
        # 1. Base default dictionary
        merged: Dict[str, Any] = {
            "core": {},
            "telemetry": {},
            "orchestration": {},
            "database": {},
            "qmmm": {},
        }

        # 2. Legacy configs (priority 5)
        legacy_data = self._load_legacy_config()
        for sec, fields in legacy_data.items():
            s_key = sec.lower()
            if s_key in merged and isinstance(fields, dict):
                merged[s_key].update(fields)

        # 3. User-level configuration (priority 4)
        user_data = self._load_user_config()
        for sec, fields in user_data.items():
            s_key = sec.lower()
            if s_key in merged and isinstance(fields, dict):
                merged[s_key].update(fields)
            elif s_key == "memory_gb":
                merged["core"]["memory_gb"] = int(fields)

        # 4. Project-level configuration (priority 3)
        project_data = self._load_project_config()
        for sec, fields in project_data.items():
            s_key = sec.lower()
            if s_key in merged and isinstance(fields, dict):
                merged[s_key].update(fields)
            elif s_key == "memory_gb":
                merged["core"]["memory_gb"] = int(fields)

        # 5. Environment variables (priority 2)
        env_overrides = self._extract_env_overrides()
        for sec, fields in env_overrides.items():
            s_key = sec.lower()
            if s_key in merged:
                merged[s_key].update(fields)

        # 6. Explicit CLI overrides (priority 1)
        for k, v in self.cli_overrides.items():
            if "__" in k:
                sec, field = k.split("__", 1)
                sec_lower = sec.lower()
                if sec_lower in merged:
                    merged[sec_lower][field.lower()] = v
            elif isinstance(v, dict) and k.lower() in merged:
                merged[k.lower()].update(v)

        # Sanitize path fields in core, orchestration, and qmmm
        if "scratch_dir" in merged["core"]:
            merged["core"]["scratch_dir"] = sanitize_path(merged["core"]["scratch_dir"], self.project_root)
        else:
            merged["core"]["scratch_dir"] = sanitize_path("./scratch", self.project_root)

        if "db_path" in merged["orchestration"]:
            merged["orchestration"]["db_path"] = sanitize_path(merged["orchestration"]["db_path"], self.project_root)
        else:
            merged["orchestration"]["db_path"] = sanitize_path("./cochem_tasks.db", self.project_root)

        for qm_bin in ("orca_path", "crest_path", "xtb_path"):
            if merged["qmmm"].get(qm_bin) is not None:
                merged["qmmm"][qm_bin] = sanitize_path(merged["qmmm"][qm_bin], self.project_root)

        config = CoChemRootConfig(
            core=CoreConfig(**merged["core"]),
            telemetry=TelemetryConfig(**merged["telemetry"]),
            orchestration=OrchestrationConfig(**merged["orchestration"]),
            database=DatabaseConfig(**merged["database"]),
            qmmm=QmMMConfig(**merged["qmmm"]),
        )
        self._cached_config = config
        return config

    def get_config(self) -> CoChemRootConfig:
        """Return cached configuration or load on demand."""
        if self._cached_config is None:
            return self.load_config()
        return self._cached_config


def get_workspace_config(workspace_dir: Optional[Union[str, Path]] = None) -> CoChemRootConfig:
    """Retrieve resolved configuration for target workspace directory."""
    return CoChemConfigManager(project_root=workspace_dir).get_config()

