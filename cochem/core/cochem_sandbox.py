"""Ephemeral Sandbox Context & Path Jailbreak Defense.
Strictly adheres to CoChem Anti-Spoofing Protocol v3 and Tripartite Storage Air-Gap.
"""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_QUARANTINED_PATHS: list[pathlib.Path] = []
_ACTIVE_SANDBOXES: weakref.WeakSet[SandboxContext] = weakref.WeakSet()


def _sweep_quarantine() -> None:
    for p in list(_QUARANTINED_PATHS):
        try:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            _QUARANTINED_PATHS.remove(p)
        except OSError as _e:
            logger.debug(f"Ignored exception: {_e}")


def _global_sandbox_atexit_cleanup() -> None:
    """Global atexit teardown iterating over surviving weak references."""
    for sb in list(_ACTIVE_SANDBOXES):
        try:
            sb.cleanup()
        except Exception as _e:
            logger.debug(f"Ignored exception: {_e}")


atexit.register(_sweep_quarantine)
atexit.register(_global_sandbox_atexit_cleanup)


class SandboxSecurityViolationError(PermissionError):
    """Raised when an operation attempts directory traversal, symlink escape, NTFS ADS injection, or reserved OS device access."""


class SandboxExecutionError(RuntimeError):
    """Raised when execution within the sandbox fails, is invoked uninitialized, or encounters an unhandled runtime fault."""


@dataclass(frozen=True)
class SandboxConfig:
    """Immutable configuration for ephemeral sandboxed execution contexts."""

    timeout_seconds: float = 300.0
    max_memory_mb: int = 4096
    allow_network: bool = False
    scratch_parent_dir: Optional[pathlib.Path] = None
    max_cleanup_retries: int = 5
    cleanup_backoff_base_s: float = 0.1


class SandboxContext:
    """OS-agnostic ephemeral scratch sandbox with strict jailbreak and path traversal defense."""

    RESERVED_WIN32_NAMES: re.Pattern[str] = re.compile(
        r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE
    )

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config: SandboxConfig = config if config is not None else SandboxConfig()
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.root: Optional[pathlib.Path] = None
        self._active: bool = False
        self._trap_registered: bool = False

    def __enter__(self) -> SandboxContext:
        parent_dir: Optional[pathlib.Path] = None
        if self.config.scratch_parent_dir is not None:
            parent_dir = self.config.scratch_parent_dir.resolve()
        else:
            scratch_env = (
                os.environ.get("COCHEM_SCRATCH_DIR")
                or os.environ.get("SLURM_TMPDIR")
                or os.environ.get("TMPDIR")
            )
            if scratch_env:
                parent_dir = pathlib.Path(scratch_env).resolve()

        if parent_dir is not None:
            parent_dir.mkdir(parents=True, exist_ok=True)

        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="cochem_sandbox_",
            dir=str(parent_dir) if parent_dir is not None else None,
        )
        self.root = pathlib.Path(self._temp_dir.name).resolve()
        self._active = True
        _ACTIVE_SANDBOXES.add(self)
        self._register_cleanup_traps()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        is_unwinding = exc_type is not None
        try:
            self.cleanup()
        except OSError as cleanup_err:
            if self.root is not None:
                _QUARANTINED_PATHS.append(self.root)
            if is_unwinding:
                logger.warning(
                    "Secondary OSError encountered during sandbox cleanup suppressed to preserve "
                    "primary scientific exception: %s (unreclaimed scratch path: %s)",
                    cleanup_err,
                    self.root,
                )
                return None
            raise

    def validate_path(self, target: pathlib.Path) -> pathlib.Path:
        """Validate target path against NTFS ADS, Win32 reserved names, and traversal."""
        if not self._active or self.root is None:
            raise SandboxExecutionError("Sandbox is not active.")

        # Rejection of NTFS Alternate Data Streams (colon check excluding drive letter)
        target_str = str(target)
        sanitized = target_str.replace(":\\", "").replace(":/", "")
        if ":" in sanitized:
            raise SandboxSecurityViolationError(
                f"NTFS Alternate Data Stream detected in path: {target}"
            )

        # Rejection of reserved Win32 device names
        for part in target.parts:
            if self.RESERVED_WIN32_NAMES.match(part):
                raise SandboxSecurityViolationError(
                    f"Reserved OS device name detected: {part}"
                )

        # Anchor relative paths to sandbox root
        if not target.is_absolute():
            resolved = (self.root / target).resolve()
        else:
            resolved = target.resolve()

        # Path containment verification
        try:
            resolved.relative_to(self.root)
        except ValueError as err:
            raise SandboxSecurityViolationError(
                f"Path traversal detected: {resolved} is outside sandbox root {self.root}"
            ) from err

        return resolved

    def cleanup(self) -> None:
        """Execute multi-pass directory deletion with exponential retry backoff."""
        if not self._active or self._temp_dir is None:
            return

        self._active = False
        _ACTIVE_SANDBOXES.discard(self)
        target_root = self.root

        for attempt in range(self.config.max_cleanup_retries):
            try:
                if self._temp_dir is not None:
                    self._temp_dir.cleanup()
                elif target_root is not None and target_root.exists():
                    shutil.rmtree(target_root)
                break
            except (OSError, PermissionError):
                if attempt == self.config.max_cleanup_retries - 1:
                    raise
                backoff_time = self.config.cleanup_backoff_base_s * (2**attempt)
                time.sleep(backoff_time)

    def _register_cleanup_traps(self) -> None:
        """Register OS signal handlers for robust teardown in main thread only."""
        if self._trap_registered:
            return

        if threading.current_thread() is not threading.main_thread():
            logger.debug("Bypassing signal.signal traps in non-main worker thread.")
            return

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                prev_handler = signal.getsignal(sig)

                def _signal_handler(signum: int, frame: Any) -> None:
                    self.cleanup()
                    if callable(prev_handler) and prev_handler not in (
                        signal.SIG_IGN,
                        signal.SIG_DFL,
                    ):
                        prev_handler(signum, frame)
                    sys.exit(128 + signum)

                signal.signal(sig, _signal_handler)
        except (ValueError, AttributeError) as _e:
            logger.debug(f"Ignored exception: {_e}")

        self._trap_registered = True
