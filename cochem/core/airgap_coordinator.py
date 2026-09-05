"""Tripartite Storage Air-Gap Coordinator & Concurrency Governor.
Enforces strict topological separation between Code (T_code), Artifacts (T_art), and Scratch (T_scr).
Provides OS-agnostic file locking and SQLite WAL concurrency governance.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import filelock

from cochem.core.cochem_sandbox import SandboxConfig, SandboxContext

logger = logging.getLogger(__name__)


class AirGapViolationError(PermissionError):
    """Raised when storage paths intersect, overlap, or violate air-gap containment rules."""


@dataclass(frozen=True)
class TripartiteStorageConfig:
    """Immutable topological layout configuration for tripartite storage."""

    scratch_root: pathlib.Path
    artifacts_root: pathlib.Path
    code_root: Optional[pathlib.Path] = None
    source_root: Optional[pathlib.Path] = None

    def __post_init__(self) -> None:
        effective_code = self.code_root or self.source_root
        if effective_code is not None:
            object.__setattr__(self, "code_root", effective_code)
            object.__setattr__(self, "source_root", effective_code)

    @property
    def effective_source_root(self) -> Optional[pathlib.Path]:
        return self.code_root


AirGapConfig = TripartiteStorageConfig


class TripartiteAirGapCoordinator:
    """Storage coordinator enforcing zero-overlap air-gap and sandboxed publication."""

    def __init__(self, config: TripartiteStorageConfig) -> None:
        scr = config.scratch_root.resolve()
        art = config.artifacts_root.resolve()
        code = (
            config.code_root.resolve()
            if config.code_root is not None
            else (scr.parent / "cochem_src").resolve()
        )
        self.config: TripartiteStorageConfig = TripartiteStorageConfig(
            code_root=code,
            artifacts_root=art,
            scratch_root=scr,
        )
        self.validate_disjointness()

    def validate_disjointness(self) -> None:
        """Verify strict mutual disjointness between Code, Artifacts, and Scratch roots."""
        code = self.config.code_root
        art = self.config.artifacts_root
        scr = self.config.scratch_root

        if code == art or code == scr or art == scr:
            raise AirGapViolationError(
                "Tripartite storage roots must be strictly distinct directories."
            )

        if code.is_relative_to(art) or code.is_relative_to(scr):
            raise AirGapViolationError(
                f"code_root '{code}' overlaps with artifacts or scratch root."
            )

        if art.is_relative_to(code) or art.is_relative_to(scr):
            raise AirGapViolationError(
                f"artifacts_root '{art}' overlaps with code or scratch root."
            )

        if scr.is_relative_to(code) or scr.is_relative_to(art):
            raise AirGapViolationError(
                f"scratch_root '{scr}' overlaps with code or artifacts root."
            )

        cwd = pathlib.Path.cwd().resolve()
        if cwd.is_relative_to(code):
            raise AirGapViolationError(
                f"CWD '{cwd}' resides within code_root '{code}'. Direct mutation access prohibited."
            )

    def create_sandboxed_workspace(
        self, job_id: str, timeout_seconds: float = 300.0
    ) -> SandboxContext:
        """Allocate an ephemeral sandboxed workspace confined strictly to scratch storage."""
        if ".." in job_id or "/" in job_id or "\\" in job_id:
            raise AirGapViolationError(
                f"Job scratch path traversal detected: {job_id}"
            )

        job_scratch = (self.config.scratch_root / f"cochem_exec_{job_id}").resolve()

        if not job_scratch.is_relative_to(self.config.scratch_root):
            raise AirGapViolationError(
                f"Job scratch path traversal detected: {job_id}"
            )

        if job_scratch == self.config.scratch_root:
            raise AirGapViolationError(
                f"Invalid job identifier attempting root allocation: {job_id}"
            )

        config = SandboxConfig(
            scratch_parent_dir=job_scratch,
            timeout_seconds=timeout_seconds,
        )
        return SandboxContext(config)

    def publish_artifact(
        self,
        source_path: pathlib.Path,
        relative_dest: pathlib.Path,
        compute_sha256: bool = True,
    ) -> Tuple[pathlib.Path, Optional[str]]:
        """Atomically transfer validated deliverable from scratch to append-only artifacts storage."""
        source = source_path.resolve()
        rel_path = pathlib.Path(relative_dest)
        dest = (self.config.artifacts_root / relative_dest).resolve()

        if self.config.code_root is not None:
            if (
                source.is_relative_to(self.config.code_root)
                or dest.is_relative_to(self.config.code_root)
                or (rel_path.is_absolute() and rel_path.resolve().is_relative_to(self.config.code_root))
            ):
                raise AirGapViolationError("Source tier (T_src) is strictly immutable and read-only.")

        if not source.is_relative_to(self.config.scratch_root):
            raise AirGapViolationError(
                f"Source path '{source}' resides outside scratch root '{self.config.scratch_root}'"
            )

        if not source.is_file():
            raise FileNotFoundError(f"Source artifact not found: {source}")

        if not dest.is_relative_to(self.config.artifacts_root):
            raise AirGapViolationError(
                f"Destination path '{dest}' resides outside artifacts root '{self.config.artifacts_root}'"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)

        sha256_hash: Optional[str] = None
        if compute_sha256:
            hasher = hashlib.sha256()
            with open(source, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            sha256_hash = hasher.hexdigest()

        # Phase 1: Atomic Commit
        staging_name = f"{dest.name}.tmp_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
        temp_dest = dest.with_name(staging_name)
        try:
            shutil.copy2(source, temp_dest)
            os.replace(temp_dest, dest)
        except Exception:
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
            raise

        # Phase 2: Post-Commit Teardown
        try:
            source.unlink(missing_ok=True)
        except OSError as cleanup_err:
            logger.warning(
                "Secondary scratch unlinking error suppressed: %s (unreclaimed scratch: %s)",
                cleanup_err,
                source,
            )

        return dest, sha256_hash


AirGapCoordinator = TripartiteAirGapCoordinator


def get_tier_file_lock(
    lock_path: pathlib.Path, timeout_sec: float = 30.0
) -> filelock.FileLock:
    """Instantiate cross-platform, OS-agnostic file lock adhering to HPC scratch requirements."""
    resolved_path = lock_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(str(resolved_path), timeout=timeout_sec)


def configure_sqlite_connection(
    conn: sqlite3.Connection, timeout_sec: float = 30.0
) -> None:
    """Configure SQLite connection with WAL mode and robust transaction timeout pragmas."""
    busy_timeout_ms = int(timeout_sec * 1000)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
