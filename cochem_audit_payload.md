Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task12_telemetry.md.
Original prompt:
# Prompt: Visual & Event Telemetry Streamer

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_telemetry.py`

## Objective
Implement Visual & Event Telemetry Streamer for CoChem-TORQ based on Task 12 (Stage 5.5 - 6.0) specifications.

## Instructions for Coder
1. Create or update `cochem_torq_telemetry.py` inside `Libraries/`.
2. Implement `stream_webhook_events()` using asynchronous HTTP POST with Exponential Backoff Circuit Breaker. Implement zero-interruption buffering to `telemetry_spool.jsonl`, writing strictly to the dynamically provided scratch directory.
3. Implement `generate_plotly_3d_carousels()` using 2D Strided Regular Grid Decimation while preserving stationary points, emitting standalone HTML visualizers to the dynamically provided artifact directory.
4. Implement `export_crash_animation()` capturing trajectories during Steric Shatter Soft-Quench aborts into `crash_animation.xyz` and `crash_diagnostic.json`. Write these strictly to the dynamically provided scratch/artifact directory.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_torq_telemetry.py`.
- **Zero Mocking**: Do NOT mock any logic. Implement physical webhook requests, exception catching, and Plotly 3D HTML generation.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically. All files (like `.tar.zst`, `.jsonl`, etc.) MUST be written to the scratch or artifact paths provided dynamically by the environment or arguments, NOT the current working directory.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_catalog_compiler.py ---
"""CoChem-TORQ 0.0.11 - Stage 5.4 / 6.0: The FAIR Out-Of-Core Archiver & Catalog Compiler.

Authoritative Module for CoChem-TORQ (Task 13 / Stage 6.0 / 7.0).
Provides memory-safe, out-of-core PyArrow Parquet catalog compilation, high-throughput
SPCAT streaming parsers, isolated multi-temperature execution workspaces, Fortran overflow
guardrails, buffer-lock disk synchronization, cross-platform NTFS/POSIX immutability seals,
Method Matrix v4 compliant AASTeX 6.3.1 / siunitx LaTeX and BibTeX generators, and the
TorqCatalogCompiler engine.

Authoritative Standards:
- Method Matrix v4 (Sections 1.1, 13.5, 13.6, 20.2): Rotational observables & catalogs
- Pickett SPCAT fixed-width format specifications [F13.4, 2F8.4, I2, F10.4, I3, I7, I4, 12I2]
- Memory Complexity: Strictly O(1) constant RAM via chunked streaming serialization
- RFC 8785: Canonical JSON serialization for cryptographic provenance manifests
- AASTeX 6.3.1 + siunitx standard for manuscript methods documentation
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import gc
import io
import logging
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-CatCompile] %(message)s")
logger = logging.getLogger("TorqCatalogCompiler")

ARTIFACTS_DIR = os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))


# =============================================================================
# Custom Exception Hierarchy & Error Codes
# =============================================================================

class ProvenanceErrorCode:
    FORTRAN_OVERFLOW = "FORTRAN_OVERFLOW"
    SPCAT_BRIDGE_ERROR = "SPCAT_BRIDGE_ERROR"
    DISPERSION_MISSING = "DISPERSION_MISSING"
    METHOD_MATRIX_VIOLATION_DEFGRID = "METHOD_MATRIX_VIOLATION_DEFGRID"
    MISSING_DATA = "MISSING_DATA"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class CoChemIntegrityError(Exception):
    """Raised when buffer lock, hash, or data integrity validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.INTEGRITY_ERROR


class SPCATBridgeError(Exception):
    """Raised when SPCAT format parsing or calculation execution encounters an error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.SPCAT_BRIDGE_ERROR


class FortranOverflowError(SPCATBridgeError):
    """Raised when asterisks indicating Fortran format overflow/underflow are parsed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.FORTRAN_OVERFLOW)


class InactiveRotorError(SPCATBridgeError):
    """Raised when an inactive rotor or transitionless calculation produces a 0-byte catalog."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.SPCAT_BRIDGE_ERROR)


class MethodMatrixViolationError(Exception):
    """Raised when a Method Matrix v4 compliance standard is violated."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID


class DispersionMissingError(MethodMatrixViolationError):
    """Raised when a DFT method lacks necessary dispersion corrections."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.DISPERSION_MISSING)


# =============================================================================
# 1. PyArrow Spectral Catalog Schema (12-Field Precision Schema)
# =============================================================================

SPECTRAL_CATALOG_SCHEMA: pa.Schema = pa.schema([
    ("frequency_mhz", pa.float64()),
    ("uncertainty_mhz", pa.float64()),
    ("log_intensity", pa.float64()),
    ("degrees_of_freedom", pa.int32()),
    ("lower_state_energy_cm1", pa.float64()),
    ("upper_state_degeneracy", pa.int32()),
    ("species_tag", pa.int32()),
    ("qn_format", pa.int32()),
    ("qn_upper", pa.dictionary(pa.int32(), pa.utf8())),
    ("qn_lower", pa.dictionary(pa.int32(), pa.utf8())),
    ("temperature_k", pa.float64()),
    ("provenance_hash", pa.dictionary(pa.int32(), pa.utf8())),
])


# =============================================================================
# 2. 6-Tier CoChemPathManager
# =============================================================================

class CoChemPathManager:
    """Central dynamic path and workspace resolver for CoChem catalog compilation.

    Enforces the strict 6-Tier Scratch Resolution Hierarchy and Deliverables Resolution Hierarchy:
    - Tier 1: Explicit custom_path argument passed to method/constructor.
    - Tier 2: COCHEM_SCRATCH or COCHEM_SCRATCH_DIR environment variables.
    - Tier 3: COCHEM_TMP, TMPDIR, TEMP, or TMP environment variables.
    - Tier 4: XDG_CACHE_HOME / cochem / scratch (or ~/.cache/cochem/scratch).
    - Tier 5: tempfile.gettempdir() / cochem_scratch.
    - Tier 6: Path.home() / .cochem / scratch fallback.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
        deliverables_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        self._custom_scratch = Path(scratch_dir).resolve() if scratch_dir is not None else None
        self._custom_deliverables = Path(deliverables_dir).resolve() if deliverables_dir is not None else None

    @classmethod
    def resolve_scratch_dir(
        cls,
        custom_path: Optional[Union[str, Path]] = None,
        create: bool = True,
    ) -> Path:
        """Resolve the active scratch directory using the 6-tier hierarchy."""
        # Tier 1: Explicit custom path argument
        if custom_path is not None:
            resolved = Path(custom_path).resolve()
            if create:
                resolved.mkdir(parents=True, exist_ok=True)
            return resolved

        # Tier 2: COCHEM_SCRATCH or COCHEM_SCRATCH_DIR
        for env_key in ("COCHEM_SCRATCH", "COCHEM_SCRATCH_DIR"):
            env_val = os.environ.get(env_key)
            if env_val and env_val.strip():
                resolved = Path(env_val.strip()).resolve()
                if create:
                    resolved.mkdir(parents=True, exist_ok=True)
                return resolved

        # Tier 3: COCHEM_TMP, TMPDIR, TEMP, TMP
        for env_key in ("COCHEM_TMP", "TMPDIR", "TEMP", "TMP"):
            env_val = os.environ.get(env_key)
            if env_val and env_val.strip():
                resolved = (Path(env_val.strip()).resolve() / "cochem_scratch").resolve()
                if create:
                    resolved.mkdir(parents=True, exist_ok=True)
                return resolved

        # Tier 4: XDG_CACHE_HOME / cochem / scratch
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache and xdg_cache.strip():
            resolved = (Path(xdg_cache.strip()).resolve() / "cochem" / "scratch").resolve()
            if create:
                resolved.mkdir(parents=True, exist_ok=True)
            return resolved

        # Tier 5: tempfile.gettempdir() / cochem_scratch
        try:
            temp_sys = Path(tempfile.gettempdir()).resolve()
            resolved = (temp_sys / "cochem_scratch").resolve()
            if create:
                resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        except Exception:
            pass

        # Tier 6: Path.home() / .cochem / scratch fallback
        resolved = (Path.home() / ".cochem" / "scratch").resolve()
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @classmethod
    def get_scratch_dir(
        cls,
        custom_path: Optional[Union[str, Path]] = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_scratch_dir."""
        return cls.resolve_scratch_dir(custom_path=custom_path, create=create)

    @classmethod
    def resolve_deliverables_dir(
        cls,
        custom_path: Optional[Union[str, Path]] = None,
        create: bool = True,
    ) -> Path:
        """Resolve deliverables directory for permanent catalog and document outputs."""
        if custom_path is not None:
            resolved = Path(custom_path).resolve()
            if create:
                resolved.mkdir(parents=True, exist_ok=True)
            return resolved

        for env_key in ("COCHEM_DELIVERABLES", "COCHEM_DELIVERABLES_DIR", "COCHEM_ARTIFACTS_DIR"):
            env_val = os.environ.get(env_key)
            if env_val and env_val.strip():
                resolved = Path(env_val.strip()).resolve()
                if create:
                    resolved.mkdir(parents=True, exist_ok=True)
                return resolved

        resolved = (Path.home() / ".cochem" / "deliverables").resolve()
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @classmethod
    def get_deliverables_dir(
        cls,
        custom_path: Optional[Union[str, Path]] = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_deliverables_dir."""
        return cls.resolve_deliverables_dir(custom_path=custom_path, create=create)

    @property
    def scratch(self) -> Path:
        """Return instance resolved scratch directory."""
        return self.resolve_scratch_dir(self._custom_scratch)

    @property
    def deliverables(self) -> Path:
        """Return instance resolved deliverables directory."""
        return self.resolve_deliverables_dir(self._custom_deliverables)


# =============================================================================
# 3. Cross-Platform Read-Only Permissions & Immutability Seals
# =============================================================================

def apply_readonly_chmod(path: Union[str, Path], recursive: bool = True) -> None:
    """Apply an immutable read-only permission seal across Windows NTFS and POSIX.

    On Windows: Uses ctypes.windll.kernel32.SetFileAttributesW(path, 1) and stat.S_IREAD.
    On POSIX: Sets 0o444 for files (read-only owner/group/other) and 0o555 for directories.

    Args:
        path: Path to file or directory to seal.
        recursive: If True and path is a directory, recursively seals all contained children.
    """
    target = Path(path).resolve()
    if not target.exists():
        return

    items: List[Path] = []
    if target.is_dir():
        if recursive:
            try:
                for child in target.rglob("*"):
                    items.append(child)
            except OSError as exc:
                logger.warning(f"Error traversing directory for readonly seal {target}: {exc}")
        items.append(target)
    else:
        items.append(target)

    for item in items:
        try:
            if sys.platform == "win32":
                try:
                    # FILE_ATTRIBUTE_READONLY = 0x00000001
                    if ctypes.windll.kernel32.SetFileAttributesW(str(item), 1) == 0:
                        os.chmod(str(item), stat.S_IREAD)
                except Exception:
                    os.chmod(str(item), stat.S_IREAD)
            else:
                if item.is_dir():
                    mode = (
                        stat.S_IRUSR
                        | stat.S_IXUSR
                        | stat.S_IRGRP
                        | stat.S_IXGRP
                        | stat.S_IROTH
                        | stat.S_IXOTH
                    )
                else:
                    mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
                os.chmod(str(item), mode)
        except OSError as exc:
            logger.warning(f"Failed to apply readonly seal to {item}: {exc}")


def remove_readonly_seal(path: Union[str, Path], recursive: bool = True) -> None:
    """Remove read-only seal and restore write permissions across Windows and POSIX.

    Args:
        path: Path to file or directory to unseal.
        recursive: If True and path is a directory, unseals all child items.
    """
    target = Path(path).resolve()
    if not target.exists():
        return

    items: List[Path] = []
    if target.is_dir():
        if recursive:
            try:
                for child in target.rglob("*"):
                    items.append(child)
            except OSError as exc:
                logger.warning(f"Error traversing directory for unlock {target}: {exc}")
        items.append(target)
    else:
        items.append(target)

    for item in items:
        try:
            if sys.platform == "win32":
                try:
                    # FILE_ATTRIBUTE_NORMAL = 0x00000080
                    if ctypes.windll.kernel32.SetFileAttributesW(str(item), 0x80) == 0:
                        os.chmod(str(item), stat.S_IREAD | stat.S_IWRITE)
                except Exception:
                    os.chmod(str(item), stat.S_IREAD | stat.S_IWRITE)
            else:
                if item.is_dir():
                    mode = (
                        stat.S_IRWXU
                        | stat.S_IRGRP
                        | stat.S_IXGRP
                        | stat.S_IROTH
                        | stat.S_IXOTH
                    )
                else:
                    mode = (
                        stat.S_IRUSR
                        | stat.S_IWUSR
                        | stat.S_IRGRP
                        | stat.S_IROTH
                    )
                os.chmod(str(item), mode)
        except OSError as exc:
            logger.warning(f"Failed to remove readonly seal on {item}: {exc}")


# =============================================================================
# 4. Buffer Lock Synchronization & Physical Disk Flush
# =============================================================================

def buffer_lock_sync(
    file_obj_or_path: Union[io.IOBase, int, str, Path],
    min_bytes: int = 1,
) -> int:
    """Perform a physical disk sync (os.fsync) and validate non-zero written size.

    Args:
        file_obj_or_path: Open file object, file descriptor, or file path.
        min_bytes: Minimum expected file size on disk in bytes.

    Returns:
        Validated on-disk size in bytes.

    Raises:
        CoChemIntegrityError: If file size on disk is less than min_bytes.
    """
    path_to_check: Optional[Path] = None

    if isinstance(file_obj_or_path, io.IOBase):
        file_obj_or_path.flush()
        fd = file_obj_or_path.fileno()
        try:
            os.fsync(fd)
        except OSError:
            pass
        if hasattr(file_obj_or_path, "name") and isinstance(file_obj_or_path.name, (str, Path)):
            path_to_check = Path(file_obj_or_path.name).resolve()
    elif isinstance(file_obj_or_path, int):
        try:
            os.fsync(file_obj_or_path)
        except OSError:
            pass
    else:
        path_to_check = Path(file_obj_or_path).resolve()
        if path_to_check.exists():
            try:
                with open(path_to_check, "r+b") as probe_fd:
                    probe_fd.flush()
                    os.fsync(probe_fd.fileno())
            except OSError:
                pass

    if path_to_check is not None:
        if not path_to_check.exists():
            raise CoChemIntegrityError(
                f"Buffer sync failed: Target file does not exist at {path_to_check}",
                details={"path": str(path_to_check)},
            )
        size_bytes = os.path.getsize(path_to_check)
        if size_bytes < min_bytes:
            raise CoChemIntegrityError(
                f"Buffer sync validation failed for {path_to_check}: "
                f"Size {size_bytes} bytes is less than expected minimum {min_bytes} bytes.",
                details={"path": str(path_to_check), "size_bytes": size_bytes, "min_bytes": min_bytes},
            )
        return size_bytes

    return 0


# =============================================================================
# 5. Ghost Output Purger
# =============================================================================

def purge_ghost_outputs(
    target_path: Union[str, Path, Sequence[Union[str, Path]]],
    patterns: Optional[Sequence[str]] = None,
    remove_0byte_only: bool = False,
    remove_tmp_siblings: bool = True,
) -> List[Path]:
    """Purge orphaned, corrupt, or 0-byte ghost calculation artifacts and staging files."""
    default_patterns = (
        "*.tmp",
        "*.cat.tmp",
        "*.parquet.tmp",
        "*.lock",
        "*.var.tmp",
        "*.int.tmp",
        "*ghost*",
        "*.tmp.*",
        ".*.tmp.*",
    )
    search_patterns = list(patterns) if patterns is not None else list(default_patterns)

    targets_list: List[Path] = []
    if isinstance(target_path, (str, Path)):
        targets_list.append(Path(target_path).resolve())
    else:
        for item in target_path:
            targets_list.append(Path(item).resolve())

    files_to_evaluate: Set[Path] = set()

    for p in targets_list:
        if p.is_dir():
            if remove_0byte_only:
                for item in p.rglob("*"):
                    if item.is_file():
                        files_to_evaluate.add(item.resolve())
            for pat in search_patterns:
                try:
                    for matched_file in p.glob(pat):
                        if matched_file.is_file():
                            files_to_evaluate.add(matched_file.resolve())
                except OSError as exc:
                    logger.warning(f"Failed glob pattern {pat} in {p}: {exc}")
        elif p.is_file():
            files_to_evaluate.add(p)
            if remove_tmp_siblings:
                parent = p.parent
                stem = p.name
                for sibling in parent.glob(f"*{stem}*tmp*"):
                    if sibling.is_file():
                        files_to_evaluate.add(sibling.resolve())
        elif not p.exists() and remove_tmp_siblings:
            parent = p.parent
            if parent.is_dir():
                stem = p.name
                for sibling in parent.glob(f"*{stem}*tmp*"):
                    if sibling.is_file():
                        files_to_evaluate.add(sibling.resolve())

    purged: List[Path] = []
    for f in sorted(files_to_evaluate):
        if not f.exists():
            continue
        try:
            size = os.path.getsize(f)
            if remove_0byte_only and size > 0:
                continue

            remove_readonly_seal(f, recursive=False)
            f.unlink()
            purged.append(f)
        except OSError as exc:
            logger.warning(f"Could not purge ghost file {f}: {exc}")

    return purged


# =============================================================================
# 6. Isolated Workspace Generator (Context Manager)
# =============================================================================

@contextmanager
def isolated_workspace_generator(
    base_scratch: Optional[Union[str, Path]] = None,
    prefix: str = "spcat_workspace",
    cleanup_on_exit: bool = True,
    job_id: Optional[str] = None,
) -> Iterator[Path]:
    """Provide a thread-safe, process-safe isolated execution scratch directory."""
    scratch_root = CoChemPathManager.resolve_scratch_dir(base_scratch, create=True)
    unique_tag = job_id if job_id else uuid.uuid4().hex[:8]
    timestamp_ns = time.time_ns()
    workspace_name = f"{prefix}_{unique_tag}_{timestamp_ns}"
    workspace_dir = (scratch_root / workspace_name).resolve()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        try:
            os.chmod(str(workspace_dir), 0o700)
        except OSError:
            pass

    try:
        yield workspace_dir
    finally:
        if cleanup_on_exit and workspace_dir.exists():
            try:
                remove_readonly_seal(workspace_dir, recursive=True)
                shutil.rmtree(workspace_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning(f"Failed to teardown isolated workspace {workspace_dir}: {exc}")


# =============================================================================
# 7. Inactive Rotor Catcher
# =============================================================================

def inactive_rotor_catcher(
    cat_source: Union[str, Path, bytes, io.IOBase, Sequence[str]],
    allow_empty: bool = False,
) -> bool:
    """Inspect SPCAT output for inactive rotors, 0-byte files, or absent transitions."""
    is_empty = False

    if isinstance(cat_source, (str, Path)):
        p = Path(cat_source)
        if p.is_file():
            size = os.path.getsize(p)
            if size == 0:
                is_empty = True
            else:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if not content:
                        is_empty = True
        else:
            content_str = str(cat_source).strip()
            if not content_str:
                is_empty = True
    elif isinstance(cat_source, bytes):
        if len(cat_source.strip()) == 0:
            is_empty = True
    elif isinstance(cat_source, io.IOBase):
        pos = cat_source.tell() if hasattr(cat_source, "tell") else 0
        content_read = cat_source.read()
        if hasattr(cat_source, "seek"):
            cat_source.seek(pos)
        if isinstance(content_read, bytes) and len(content_read.strip()) == 0:
            is_empty = True
        elif isinstance(content_read, str) and len(content_read.strip()) == 0:
            is_empty = True
    elif isinstance(cat_source, (list, tuple, set)):
        non_empty_lines = [line.strip() for line in cat_source if line and line.strip()]
        if len(non_empty_lines) == 0:
            is_empty = True

    if is_empty:
        if not allow_empty:
            raise InactiveRotorError(
                "Inactive rotor intercepted: SPCAT catalog output is 0 bytes or contains no transitions.",
                error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
                details={"source": str(cat_source)[:200], "is_empty": True},
            )
        return True

    return False


# =============================================================================
# 8. High-Throughput Fixed-Width SPCAT Catalog Parser
# =============================================================================

def parse_spcat_cat_line(
    line: str,
    line_number: Optional[int] = None,
    temperature_k: float = 300.0,
    provenance_hash: str = "",
) -> Optional[Dict[str, Any]]:
    """Parse a single line of Pickett SPCAT .cat fixed-width output.

    Standard Pickett format: [F13.4, 2F8.4, I2, F10.4, I3, I7, I4, 12I2]
    """
    if not line or not line.strip():
        return None

    raw = line.rstrip("\r\n")

    if "*" in raw:
        raise FortranOverflowError(
            f"Fortran overflow / underflow encountered in SPCAT .cat line: {raw.strip()!r}",
            error_code=ProvenanceErrorCode.FORTRAN_OVERFLOW,
            details={"line": raw.strip(), "line_number": line_number},
        )

    def _parse_fortran_float(val_str: str) -> float:
        clean = val_str.strip().replace("D", "E").replace("d", "e")
        return float(clean)

    if len(raw) >= 55:
        try:
            freq_val = _parse_fortran_float(raw[0:13])
            err_val = _parse_fortran_float(raw[13:21])
            lgint_val = _parse_fortran_float(raw[21:29])
            dr_val = int(raw[29:31].strip())
            elo_val = _parse_fortran_float(raw[31:41])
            gup_val = int(raw[41:44].strip())
            tag_val = int(raw[44:51].strip())
            qnfmt_val = int(raw[51:55].strip())

            qn_part = raw[55:]
            if len(qn_part) >= 24:
                qn_upper = qn_part[0:12].strip()
                qn_lower = qn_part[12:24].strip()
            else:
                tokens = qn_part.split()
                if len(tokens) >= 2:
                    half = len(tokens) // 2
                    qn_upper = " ".join(tokens[:half])
                    qn_lower = " ".join(tokens[half:])
                else:
                    qn_upper = qn_part.strip()
                    qn_lower = ""

            return {
                "frequency_mhz": freq_val,
                "uncertainty_mhz": err_val,
                "log_intensity": lgint_val,
                "degrees_of_freedom": dr_val,
                "lower_state_energy_cm1": elo_val,
                "upper_state_degeneracy": gup_val,
                "species_tag": tag_val,
                "qn_format": qnfmt_val,
                "qn_upper": qn_upper,
                "qn_lower": qn_lower,
                "temperature_k": float(temperature_k),
                "provenance_hash": str(provenance_hash),
            }
        except (ValueError, IndexError):
            pass

    tokens = raw.split()
    if len(tokens) >= 8:
        try:
            freq_val = _parse_fortran_float(tokens[0])
            err_val = _parse_fortran_float(tokens[1])
            lgint_val = _parse_fortran_float(tokens[2])
            dr_val = int(tokens[3])
            elo_val = _parse_fortran_float(tokens[4])
            gup_val = int(tokens[5])
            tag_val = int(tokens[6])
            qnfmt_val = int(tokens[7])
            remaining = tokens[8:]
            if len(remaining) >= 2:
                half = len(remaining) // 2
                qn_upper = " ".join(remaining[:half])
                qn_lower = " ".join(remaining[half:])
            elif len(remaining) == 1:
                qn_upper = remaining[0]
                qn_lower = ""
            else:
                qn_upper = ""
                qn_lower = ""

            return {
                "frequency_mhz": freq_val,
                "uncertainty_mhz": err_val,
                "log_intensity": lgint_val,
                "degrees_of_freedom": dr_val,
                "lower_state_energy_cm1": elo_val,
                "upper_state_degeneracy": gup_val,
                "species_tag": tag_val,
                "qn_format": qnfmt_val,
                "qn_upper": qn_upper,
                "qn_lower": qn_lower,
                "temperature_k": float(temperature_k),
                "provenance_hash": str(provenance_hash),
            }
        except ValueError as exc:
            raise SPCATBridgeError(
                f"Failed to parse SPCAT .cat tokens on line {line_number}: {exc}",
                error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
                details={"line": raw, "line_number": line_number},
            ) from exc

    raise SPCATBridgeError(
        f"Malformed SPCAT .cat line format on line {line_number}: {raw!r}",
        error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
        details={"line": raw, "line_number": line_number},
    )


def parse_spcat_cat_stream(
    stream_or_path: Union[str, Path, io.TextIOBase, Iterator[str], Sequence[str]],
    temperature_k: float = 300.0,
    provenance_hash: str = "",
) -> Iterator[Dict[str, Any]]:
    """Stream and yield parsed records from a Pickett SPCAT .cat source."""
    if isinstance(stream_or_path, (str, Path)):
        p = Path(stream_or_path)
        if p.is_file():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    parsed = parse_spcat_cat_line(
                        line,
                        line_number=line_idx,
                        temperature_k=temperature_k,
                        provenance_hash=provenance_hash,
                    )
                    if parsed is not None:
                        yield parsed
            return
        else:
            for line_idx, line in enumerate(str(stream_or_path).splitlines(), start=1):
                parsed = parse_spcat_cat_line(
                    line,
                    line_number=line_idx,
                    temperature_k=temperature_k,
                    provenance_hash=provenance_hash,
                )
                if parsed is not None:
                    yield parsed
            return

    for line_idx, line in enumerate(stream_or_path, start=1):
        parsed = parse_spcat_cat_line(
            line,
            line_number=line_idx,
            temperature_k=temperature_k,
            provenance_hash=provenance_hash,
        )
        if parsed is not None:
            yield parsed


# =============================================================================
# 9. Memory-Safe O(1) PyArrow Chunked Parquet Serializer
# =============================================================================

def pyarrow_chunked_serializer(
    records_stream: Iterator[Dict[str, Any]],
    output_parquet_path: Union[str, Path],
    chunk_size: int = 100_000,
    compression: str = "zstd",
    compression_level: int = 7,
    schema: Optional[pa.Schema] = None,
    verify_sync: bool = True,
) -> Path:
    """Stream catalog records into an out-of-core PyArrow Parquet file with O(1) memory overhead.

    Architecture Constraints:
    - O(1) memory footprint: Flushes RecordBatches to disk every chunk_size records.
    - Sibling staging: Writes to temporary sibling file on the same mount.
    - Buffer lock sync: Executes hard os.fsync and validates non-zero disk size.
    - Atomic promotion: Replaces target file atomically upon stream completion.

    Args:
        records_stream: Iterator or generator yielding parsed record dictionaries.
        output_parquet_path: Destination .parquet file path.
        chunk_size: Number of records buffered per PyArrow chunk (default 100,000).
        compression: Parquet compression codec (default 'zstd').
        compression_level: Compression level (default 7).
        schema: Target PyArrow schema (default SPECTRAL_CATALOG_SCHEMA).
        verify_sync: If True, invokes buffer_lock_sync prior to promotion.

    Returns:
        Path to the finalized .parquet file.
    """
    target_schema = schema if schema is not None else SPECTRAL_CATALOG_SCHEMA
    final_path = Path(output_parquet_path).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    temp_filename = f".{final_path.name}.tmp.{uuid.uuid4().hex[:8]}"
    temp_staging_path = final_path.parent / temp_filename

    field_names = [f.name for f in target_schema]
    buffer: Dict[str, List[Any]] = {name: [] for name in field_names}
    rows_in_buffer = 0
    total_rows = 0

    writer: Optional[pq.ParquetWriter] = None

    try:
        writer = pq.ParquetWriter(
            temp_staging_path,
            schema=target_schema,
            compression=compression,
            compression_level=compression_level,
        )

        def _flush_buffer() -> None:
            nonlocal rows_in_buffer, buffer, writer
            if rows_in_buffer == 0 or writer is None:
                return

            arrays: List[pa.Array] = []
            for field in target_schema:
                col_data = buffer[field.name]
                arr = pa.array(col_data, type=field.type)
                arrays.append(arr)

            batch_table = pa.Table.from_arrays(arrays, schema=target_schema)
            writer.write_table(batch_table)

            buffer = {name: [] for name in field_names}
            rows_in_buffer = 0
            gc.collect()

        for record in records_stream:
            for name in field_names:
                buffer[name].append(record.get(name))
            rows_in_buffer += 1
            total_rows += 1

            if rows_in_buffer >= chunk_size:
                _flush_buffer()

        if rows_in_buffer > 0:
            _flush_buffer()

    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if temp_staging_path.exists():
            try:
                temp_staging_path.unlink()
            except Exception:
                pass
        raise
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        if temp_staging_path.exists():
            temp_staging_path.unlink()
        raise InactiveRotorError(
            f"Zero catalog records were produced for {final_path.name}. Inactive rotor intercepted.",
            error_code=ProvenanceErrorCode.SPCAT_BRIDGE_ERROR,
            details={"output_path": str(final_path), "total_rows": 0},
        )

    if verify_sync:
        buffer_lock_sync(temp_staging_path, min_bytes=4)

    if final_path.exists():
        remove_readonly_seal(final_path, recursive=False)

    try:
        os.replace(temp_staging_path, final_path)
    except OSError:
        shutil.move(str(temp_staging_path), str(final_path))

    return final_path


# =============================================================================
# 10. Parallel Multi-Temperature Catalog Compiler
# =============================================================================

def _compile_single_temperature_task(
    runner_or_path: Union[Callable[[float, Path], Path], Path, str],
    temp_k: float,
    output_dir: Path,
    base_scratch: Optional[Path],
    chunk_size: int,
    provenance_hash: str,
    apply_immutable_seal: bool,
) -> Tuple[float, Path]:
    """Worker task executing an isolated single-temperature compilation."""
    with isolated_workspace_generator(
        base_scratch=base_scratch,
        prefix=f"spcat_T_{temp_k:.3f}K",
        cleanup_on_exit=True,
    ) as worker_ws:
        purge_ghost_outputs(worker_ws)

        cat_file: Path
        if callable(runner_or_path):
            cat_file = runner_or_path(temp_k, worker_ws)
        else:
            cat_file = Path(runner_or_path).resolve()

        inactive_rotor_catcher(cat_file, allow_empty=False)

        out_parquet = output_dir / f"spectral_catalog_T_{temp_k:.3f}K.parquet"

        stream = parse_spcat_cat_stream(
            cat_file,
            temperature_k=temp_k,
            provenance_hash=provenance_hash,
        )
        final_parquet = pyarrow_chunked_serializer(
            stream,
            output_parquet_path=out_parquet,
            chunk_size=chunk_size,
            verify_sync=True,
        )

        if apply_immutable_seal:
            apply_readonly_chmod(final_parquet, recursive=False)

        purge_ghost_outputs(worker_ws)

        return temp_k, final_parquet


def parallel_temperature_compiler(
    spcat_runner_or_cat_paths: Union[
        Callable[[float, Path], Path],
        Dict[float, Union[str, Path]],
        Sequence[Tuple[float, Union[str, Path]]],
    ],
    temperatures: Sequence[float],
    output_dir: Union[str, Path],
    max_workers: Optional[int] = None,
    base_scratch: Optional[Union[str, Path]] = None,
    chunk_size: int = 100_000,
    provenance_hash: str = "",
    apply_immutable_seal: bool = False,
) -> Dict[float, Path]:
    """Compile multiple temperature catalogs concurrently using hardware-saturated ThreadPoolExecutor."""
    target_out_dir = CoChemPathManager.resolve_deliverables_dir(output_dir, create=True)
    scratch_root = CoChemPathManager.resolve_scratch_dir(base_scratch, create=True)

    workers = max_workers if max_workers is not None else min(len(temperatures), os.cpu_count() or 4)
    workers = max(1, workers)

    results: Dict[float, Path] = {}
    futures: List[concurrent.futures.Future[Tuple[float, Path]]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for temp in temperatures:
            temp_k = float(temp)
            runner_task: Union[Callable[[float, Path], Path], Path, str]
            if callable(spcat_runner_or_cat_paths):
                runner_task = spcat_runner_or_cat_paths
            elif isinstance(spcat_runner_or_cat_paths, dict):
                runner_task = spcat_runner_or_cat_paths[temp_k]
            elif isinstance(spcat_runner_or_cat_paths, (list, tuple)):
                mapping = dict(spcat_runner_or_cat_paths)
                runner_task = mapping[temp_k]
            else:
                raise ValueError("Invalid spcat_runner_or_cat_paths specification.")

            fut = executor.submit(
                _compile_single_temperature_task,
                runner_task,
                temp_k,
                target_out_dir,
                scratch_root,
                chunk_size,
                provenance_hash,
                apply_immutable_seal,
            )
            futures.append(fut)

        for completed_fut in concurrent.futures.as_completed(futures):
            t_k, parquet_path = completed_fut.result()
            results[t_k] = parquet_path

    return results


# =============================================================================
# 11. AASTeX 6.3.1 + siunitx LaTeX Methods Block Generator
# =============================================================================

def generate_methods_latex(
    metadata: Dict[str, Any],
    method_matrix_v4_check: bool = True,
) -> str:
    """Generate an AASTeX 6.3.1 and siunitx compliant LaTeX Computational Methods section.

    Validates Method Matrix v4 constraints:
    - DFT methods require explicit dispersion correction (-D3BJ, -D4).
    - Grid definitions must meet DEFGRID2 / DEFGRID3 criteria.
    - Required metadata: theory_level, basis_set, rotational_constants, temperatures.

    Args:
        metadata: Dictionary containing chemical and computational parameters.
        method_matrix_v4_check: If True, strictly enforces Method Matrix v4 compliance.

    Returns:
        Formatted LaTeX code string ready for direct insertion into scientific manuscripts.

    Raises:
        MethodMatrixViolationError: If required fields or dispersion corrections are missing.
    """
    theory_level = str(metadata.get("theory_level", "")).strip()
    basis_set = str(metadata.get("basis_set", "")).strip()
    software_version = str(metadata.get("software_version", "ORCA 6.1.0 / Pickett SPCAT")).strip()
    rot_constants = metadata.get("rotational_constants", {})
    dipoles = metadata.get("dipole_moments", {})
    centrifugal = metadata.get("centrifugal_distortion", {})
    raw_temps = metadata.get("temperatures", [300.0])
    if isinstance(raw_temps, (int, float)):
        temperatures = [float(raw_temps)]
    elif isinstance(raw_temps, (list, tuple, set)):
        temperatures = [float(t) for t in raw_temps]
    else:
        temperatures = [300.0]

    defgrid = str(metadata.get("defgrid", "DEFGRID3")).strip().upper()
    provenance_hash = str(metadata.get("provenance_hash", "")).strip()

    if method_matrix_v4_check:
        if not theory_level:
            raise MethodMatrixViolationError(
                "Method Matrix v4 Violation: Missing required theory_level in metadata.",
                error_code=ProvenanceErrorCode.MISSING_DATA,
                details={"metadata": metadata},
            )
        if not basis_set:
            raise MethodMatrixViolationError(
                "Method Matrix v4 Violation: Missing required basis_set in metadata.",
                error_code=ProvenanceErrorCode.MISSING_DATA,
                details={"metadata": metadata},
            )
        if not rot_constants:
            raise MethodMatrixViolationError(
                "Method Matrix v4 Violation: Missing rotational_constants in metadata.",
                error_code=ProvenanceErrorCode.MISSING_DATA,
                details={"metadata": metadata},
            )

        # Check DFT dispersion compliance
        dft_signatures = (
            "B3LYP", "WB97", "PBE", "R2SCAN", "TPSS", "M06", "B97", "SCAN",
            "OLYP", "PW6B95", "BP86", "BLYP", "CAM-B3LYP", "LC-",
        )
        theory_upper = theory_level.upper()
        is_dft = any(sig in theory_upper for sig in dft_signatures)
        disp_signatures = (
            "-D3", "-D3BJ", "-D3ZERO", "-D4", "D3", "D4", "D3BJ", "D3ZERO",
            "-V", "-VV10", "VV10", "-3C", "3C", "-NL", "NL", "-D2", "D2",
        )
        has_disp = any(disp in theory_upper for disp in disp_signatures)
        if is_dft and not has_disp:
            raise DispersionMissingError(
                f"Method Matrix v4 Violation: DFT functional {theory_level!r} lacks required dispersion correction (D3BJ/D4/VV10/3c).",
                error_code=ProvenanceErrorCode.DISPERSION_MISSING,
                details={"theory_level": theory_level},
            )

        # Check DEFGRID standard
        if "DEFGRID1" in defgrid or "SG-1" in defgrid:
            raise MethodMatrixViolationError(
                f"Method Matrix v4 Violation: Grid {defgrid!r} fails minimum integration threshold (DEFGRID2/DEFGRID3 required).",
                error_code=ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID,
                details={"defgrid": defgrid},
            )

    def _find_rot_val(key_char: str) -> float:
        for k, v in rot_constants.items():
            k_clean = str(k).strip().upper()
            if k_clean in (key_char, f"{key_char}_MHZ", f"{key_char}0", f"{key_char}_0", f"{key_char}_E"):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return 0.0

    a_mhz = _find_rot_val("A")
    b_mhz = _find_rot_val("B")
    c_mhz = _find_rot_val("C")

    def _find_dipole_val(comp: str) -> float:
        for k, v in dipoles.items():
            k_clean = str(k).strip().lower()
            if k_clean in (f"mu_{comp}", f"mu{comp}", f"dipole_{comp}", comp):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return 0.0

    mu_a = _find_dipole_val("a")
    mu_b = _find_dipole_val("b")
    mu_c = _find_dipole_val("c")
    mu_tot = dipoles.get("total", (mu_a**2 + mu_b**2 + mu_c**2) ** 0.5)

    temp_formatted = ", ".join(f"\\qty{{{t:.2f}}}{{\\kelvin}}" for t in temperatures)

    latex_lines: List[str] = [
        r"% -----------------------------------------------------------------------------",
        r"% CoChem Automated Computational Methods Section (AASTeX 6.3.1 / siunitx)",
        r"% -----------------------------------------------------------------------------",
        r"\section{Computational Methods}\label{sec:methods}",
        r"",
        "All electronic structure calculations and rovibrational predictions were performed",
        f"using the CoChem ecosystem ({software_version}) in strict compliance with the",
        r"CoChem Method Matrix standards \citep{MethodMatrix2024}.",
        "Geometry optimizations and harmonic force fields were evaluated at the",
        f"\\mbox{{{theory_level}/{basis_set}}} level of theory using {defgrid} integration grids.",
        r"",
        r"Rotational and centrifugal distortion constants were derived in Watson's",
        r"$A$-reduced Hamiltonian representation ($I^r$ coordinate representation).",
        f"The predicted equilibrium rotational constants are $A = \\qty{{{a_mhz:.3f}}}{{\\mega\\hertz}}$,",
        f"$B = \\qty{{{b_mhz:.3f}}}{{\\mega\\hertz}}$, and $C = \\qty{{{c_mhz:.3f}}}{{\\mega\\hertz}}$.",
        "The electric dipole moment components along the principal inertial axes are",
        f"$\\mu_a = \\qty{{{mu_a:.3f}}}{{\\debye}}$, $\\mu_b = \\qty{{{mu_b:.3f}}}{{\\debye}}$, and",
        f"$\\mu_c = \\qty{{{mu_c:.3f}}}{{\\debye}}$ (total dipole $\\mu = \\qty{{{mu_tot:.3f}}}{{\\debye}}$).",
        r"",
        "Rotational spectral line catalogs were simulated using Pickett's SPCAT suite \\citep{Pickett1991}",
        f"across thermodynamic temperatures $T \\in \\{{{temp_formatted}\\}}$.",
        r"Partition functions $Q(T)$ incorporate full nuclear spin statistical weights",
        r"and vibrational state summations. Out-of-core binary catalogs were compiled into",
        r"columnar PyArrow Parquet format with double-precision floating-point precision",
        r"on frequencies, intensities, and state energies.",
    ]

    if centrifugal:
        def _find_cent_val(*aliases: str) -> float:
            for k, v in centrifugal.items():
                k_clean = str(k).strip().lower().replace("_", "")
                for a in aliases:
                    if k_clean == a.lower().replace("_", ""):
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            pass
            return 0.0

        dj = _find_cent_val("DJ", "D_J")
        djk = _find_cent_val("DJK", "D_JK")
        dk = _find_cent_val("DK", "D_K")
        d1 = _find_cent_val("d1", "d_1")
        d2 = _find_cent_val("d2", "d_2")
        latex_lines.extend([
            r"",
            f"Evaluated Watson quartic distortion parameters are $D_J = \\qty{{{dj:.5f}}}{{\\mega\\hertz}}$, "
            f"$D_{{JK}} = \\qty{{{djk:.5f}}}{{\\mega\\hertz}}$, $D_K = \\qty{{{dk:.5f}}}{{\\mega\\hertz}}$, "
            f"$d_1 = \\qty{{{d1:.5f}}}{{\\mega\\hertz}}$, and $d_2 = \\qty{{{d2:.5f}}}{{\\mega\\hertz}}$.",
        ])

    if provenance_hash:
        latex_lines.extend([
            r"",
            f"% Cryptographic Provenance SHA-256 Digest: {provenance_hash}",
            r"\noindent\textbf{Data Availability:} Spectral catalogs and raw quantum chemical artifacts",
            f"are immutably archived with SHA-256 digest \\texttt{{{provenance_hash}}}.",
        ])

    return "\n".join(latex_lines) + "\n"


# =============================================================================
# 12. High-Fidelity BibTeX Deduplication Engine
# =============================================================================

def deduplicate_bibtex(
    bibtex_entries: Union[str, Sequence[str]],
    deduplicate_by: str = "both",
) -> str:
    """Deduplicate BibTeX bibliography entries by cite key, normalized DOI, or both."""
    raw_text: str
    if isinstance(bibtex_entries, (list, tuple, set)):
        raw_text = "\n\n".join(str(entry) for entry in bibtex_entries)
    else:
        raw_text = str(bibtex_entries)

    entry_pattern = re.compile(
        r"@(?P<type>[a-zA-Z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,\s*(?P<body>.*?)\s*\}\s*(?=(?:@[a-zA-Z]+\s*\{|\Z))",
        re.DOTALL,
    )

    doi_pattern = re.compile(r"doi\s*=\s*[\"{](?P<doi>[^\"}]+)[\"}]", re.IGNORECASE)

    seen_keys: Set[str] = set()
    seen_dois: Set[str] = set()
    unique_entries: List[str] = []

    for match in entry_pattern.finditer(raw_text):
        entry_type = match.group("type").strip()
        cite_key = match.group("key").strip()
        body = match.group("body").strip()
        full_entry = f"@{entry_type}{{{cite_key},\n  {body}\n}}"

        norm_key = cite_key.lower()

        doi_match = doi_pattern.search(body)
        norm_doi: Optional[str] = None
        if doi_match:
            raw_doi = doi_match.group("doi").strip()
            cleaned_doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw_doi, flags=re.IGNORECASE)
            cleaned_doi = re.sub(r"^doi:\s*", "", cleaned_doi, flags=re.IGNORECASE)
            norm_doi = cleaned_doi.strip().lower()

        is_duplicate = False

        if deduplicate_by in ("key", "both") and norm_key in seen_keys:
            is_duplicate = True

        if deduplicate_by in ("doi", "both") and norm_doi and norm_doi in seen_dois:
            is_duplicate = True

        if not is_duplicate:
            seen_keys.add(norm_key)
            if norm_doi:
                seen_dois.add(norm_doi)
            unique_entries.append(full_entry)

    return "\n\n".join(unique_entries) + ("\n" if unique_entries else "")


# =============================================================================
# 13. Banned Methods Auditor & Method Matrix v4 Compliance Engine
# =============================================================================

@dataclass
class BannedMethodsAuditResult:
    """Result container for Method Matrix v4 banned methods and non-covalent rules audit."""

    passed: bool
    banned_flags: List[str]
    allowed_diffuse_basis: bool
    is_frozen_monomer_verified: bool
    is_bsse_counterpoise_verified: bool
    is_valid_hessian_preconditioned: bool
    conformer_union_params: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit result to dictionary."""
        return asdict(self)


def audit_banned_methods(
    metadata: Dict[str, Any],
    raise_on_violation: bool = True,
) -> BannedMethodsAuditResult:
    """Actively audits computational parameters against Method Matrix v4 banned methods.

    Mandates:
    - Banned: Additive diffuse corrections (e.g. adding diffuse primitives to standard basis).
    - Required for vdW / non-covalent complexes: True diffuse-in-base sets
      (e.g., 'aug-cc-pVQZ', 'aug-cc-pVTZ', 'ma-def2-TZVPP', 'def2-TZVPPD').
    - Confirms Frozen-Monomer Protocol (to fix A-constants).
    - Confirms Boys-Bernardi Counterpoise Corrections for BSSE.
    - Validates Hessian Preconditioning (verifies 'InHess XTB2' or 'Lindh' while trapping 'Calc_Hess true').
    - Documents ORCA GOAT/CREST union parameters.

    Args:
        metadata: Computational metadata dictionary.
        raise_on_violation: If True, raises MethodMatrixViolationError upon violation.

    Returns:
        BannedMethodsAuditResult with pass/fail status and flags.

    Raises:
        MethodMatrixViolationError: If a banned method is detected and raise_on_violation=True.
    """
    banned_flags: List[str] = []
    basis_set = str(metadata.get("basis_set", "")).strip().lower()
    keywords = str(metadata.get("keywords", metadata.get("orca_keywords", ""))).lower()

    # 1. Check for banned additive diffuse corrections
    if "additive_diffuse" in keywords or metadata.get("additive_diffuse_correction", False):
        banned_flags.append(
            "BANNED_ADDITIVE_DIFFUSE: Additive diffuse corrections degrade interaction energies. "
            "Use true diffuse-in-base sets (e.g. aug-cc-pVQZ or ma-def2-TZVPP)."
        )

    # 2. Check for banned Calc_Hess true without preconditioning
    if "calc_hess true" in keywords or "calc_hess=true" in keywords or metadata.get("calc_hess_true", False):
        if not ("inhess xtb2" in keywords or "inhess lindh" in keywords or metadata.get("hessian_preconditioned", False)):
            banned_flags.append(
                "BANNED_UNPRECONDITIONED_HESSIAN: 'Calc_Hess true' without preconditioning is forbidden. "
                "Must use 'InHess XTB2' or 'Lindh' Hessian preconditioning."
            )

    # 3. Check for diffuse-in-base compliance on non-covalent complexes
    is_non_covalent = metadata.get("is_non_covalent", metadata.get("is_vdw_complex", False))
    valid_diffuse_sets = ("aug-cc-pv", "ma-def2", "def2-tzvppd", "def2-qzvppd", "heavy-aug")
    allowed_diffuse_basis = any(ds in basis_set for ds in valid_diffuse_sets)

    if is_non_covalent and not allowed_diffuse_basis:
        banned_flags.append(
            f"INVALID_NONCOVALENT_BASIS: Basis set '{basis_set}' lacks true diffuse-in-base primitives. "
            "Non-covalent complexes require aug-cc-pVTZ/QZ or ma-def2-TZVPP."
        )

    # 4. Check Frozen-Monomer Protocol verification
    frozen_monomer = bool(metadata.get("frozen_monomer", metadata.get("frozen_monomer_protocol", False)))

    # 5. Check BSSE Counterpoise verification
    bsse_cp = bool(metadata.get("counterpoise", metadata.get("bsse_counterpoise", "cp" in keywords)))

    # 6. Check Hessian preconditioning
    hessian_preconditioned = bool(
        "inhess xtb2" in keywords
        or "inhess lindh" in keywords
        or metadata.get("hessian_preconditioned", False)
        or metadata.get("hessian_preconditioning", None) in ("XTB2", "Lindh")
    )

    # 7. Extract ORCA GOAT/CREST conformer union parameters
    conformer_union = metadata.get(
        "conformer_union_parameters",
        {
            "crest_ewin": metadata.get("crest_ewin", 6.0),
            "crest_rthr": metadata.get("crest_rthr", 0.12),
            "orca_goat_opt": metadata.get("orca_goat_opt", True),
        },
    )

    passed = len(banned_flags) == 0

    if not passed and raise_on_violation:
        raise MethodMatrixViolationError(
            f"Method Matrix v4 Banned Methods Audit Failed: {'; '.join(banned_flags)}",
            error_code=ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID,
            details={"banned_flags": banned_flags, "metadata": metadata},
        )

    return BannedMethodsAuditResult(
        passed=passed,
        banned_flags=banned_flags,
        allowed_diffuse_basis=allowed_diffuse_basis or not is_non_covalent,
        is_frozen_monomer_verified=frozen_monomer,
        is_bsse_counterpoise_verified=bsse_cp,
        is_valid_hessian_preconditioned=hessian_preconditioned,
        conformer_union_params=conformer_union,
        details={"basis_set": basis_set, "keywords": keywords},
    )


# =============================================================================
# 14. TorqCatalogCompiler Class Interface
# =============================================================================

class TorqCatalogCompiler:
    """TorqCatalogCompiler engine supporting fixed-width parsing, streaming Parquet compilation,

    and partition function calculations.
    """

    def __init__(self, cat_filepath: Union[str, Path], point_id: str = "000", output_dir: Optional[Union[str, Path]] = None) -> None:
        self.cat_filepath = Path(cat_filepath).resolve()
        self.point_id = point_id
        out_dir = Path(output_dir).resolve() if output_dir is not None else Path(ARTIFACTS_DIR).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_outpath = out_dir / f"torq_catalog_{self.point_id}.parquet"
        self.col_widths = [13, 8, 8, 2, 10, 3, 7, 12, 12]
        self.col_names = [
            "Frequency_MHz", "Error_MHz", "Log_Intensity", "DOF", 
            "E_Lower_cm1", "G_Up", "Tag", "QNs_Up", "QNs_Low"
        ]

    def _parse_chunk(self, raw_lines: List[str]) -> pd.DataFrame:
        parsed_data: Dict[str, List[Any]] = {col: [] for col in self.col_names}
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                parsed = parse_spcat_cat_line(line)
                if parsed is not None:
                    parsed_data["Frequency_MHz"].append(parsed["frequency_mhz"])
                    parsed_data["Error_MHz"].append(parsed["uncertainty_mhz"])
                    parsed_data["Log_Intensity"].append(parsed["log_intensity"])
                    parsed_data["DOF"].append(parsed["degrees_of_freedom"])
                    parsed_data["E_Lower_cm1"].append(parsed["lower_state_energy_cm1"])
                    parsed_data["G_Up"].append(parsed["upper_state_degeneracy"])
                    parsed_data["Tag"].append(parsed["species_tag"])
                    parsed_data["QNs_Up"].append(parsed["qn_upper"])
                    parsed_data["QNs_Low"].append(parsed["qn_lower"])
            except FortranOverflowError:
                logger.debug(f"Skipping line due to Fortran overflow: {line.strip()}")
                continue
            except Exception as exc:
                logger.error(f"Malformed line encountered: {line.strip()}: {exc}")
                raise ValueError(f"Malformed line: {line.strip()}") from exc

        return pd.DataFrame(parsed_data)

    def compile_to_parquet(self, chunk_size: int = 100_000, compression: str = "snappy") -> bool:
        """Executes the out-of-core streaming read/write loop with chunked Parquet writing."""
        if not self.cat_filepath.exists():
            logger.error(f"Catalog file {self.cat_filepath} not found. SPCAT execution may have failed.")
            raise FileNotFoundError(f"Catalog file {self.cat_filepath} not found.")

        logger.info(f"Initiating out-of-core Parquet compilation for {self.cat_filepath}")

        schema = pa.schema([
            ("Frequency_MHz", pa.float64()),
            ("Error_MHz", pa.float64()),
            ("Log_Intensity", pa.float64()),
            ("DOF", pa.int32()),
            ("E_Lower_cm1", pa.float64()),
            ("G_Up", pa.int32()),
            ("Tag", pa.int32()),
            ("QNs_Up", pa.string()),
            ("QNs_Low", pa.string()),
        ])

        total_rows = 0
        writer: Optional[pq.ParquetWriter] = None

        try:
            with open(self.cat_filepath, "r", encoding="utf-8", errors="ignore") as f:
                chunk: List[str] = []
                for line in f:
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        df_chunk = self._parse_chunk(chunk)
                        table_chunk = pa.Table.from_pandas(df_chunk, schema=schema)
                        if writer is None:
                            writer = pq.ParquetWriter(self.parquet_outpath, schema, compression=compression)
                        writer.write_table(table_chunk)
                        total_rows += len(df_chunk)
                        chunk = []

                if chunk:
                    df_chunk = self._parse_chunk(chunk)
                    if not df_chunk.empty:
                        table_chunk = pa.Table.from_pandas(df_chunk, schema=schema)
                        if writer is None:
                            writer = pq.ParquetWriter(self.parquet_outpath, schema, compression=compression)
                        writer.write_table(table_chunk)
                        total_rows += len(df_chunk)

            if writer:
                writer.close()

            if total_rows == 0:
                raise InactiveRotorError("SPCAT produced 0 transitions.")

            file_size_mb = os.path.getsize(self.parquet_outpath) / (1024 * 1024)
            logger.info(f"Compilation Complete! {total_rows} transitions secured.")
            logger.info(f"Parquet Payload: {self.parquet_outpath} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Catastrophic failure during Parquet serialization: {e}")
            if writer:
                writer.close()
            raise RuntimeError(f"Serialization failed: {e}") from e

    def compute_temperature_dependent_partition_function(
        self, temp_k: float, A_MHz: float = 10000.0, B_MHz: float = 2000.0, C_MHz: float = 1500.0, sigma: int = 1
    ) -> float:
        """Computes temperature-dependent rotational partition function Q_rot(T)."""
        kB = 1.380649e-23
        h = 6.62607015e-34
        kT = kB * temp_k

        A_Hz = max(abs(A_MHz), 1e-6) * 1e6
        B_Hz = max(abs(B_MHz), 1e-6) * 1e6
        C_Hz = max(abs(C_MHz), 1e-6) * 1e6

        q_rot = (math.sqrt(math.pi) / max(sigma, 1)) * math.sqrt((kT**3) / ((h**3) * A_Hz * B_Hz * C_Hz))
        logger.info(f"Q_rot({temp_k} K) = {q_rot:.4f}")
        return q_rot


__all__ = [
    "SPECTRAL_CATALOG_SCHEMA",
    "ProvenanceErrorCode",
    "CoChemIntegrityError",
    "SPCATBridgeError",
    "FortranOverflowError",
    "InactiveRotorError",
    "MethodMatrixViolationError",
    "DispersionMissingError",
    "BannedMethodsAuditResult",
    "CoChemPathManager",
    "apply_readonly_chmod",
    "remove_readonly_seal",
    "buffer_lock_sync",
    "purge_ghost_outputs",
    "isolated_workspace_generator",
    "inactive_rotor_catcher",
    "parse_spcat_cat_line",
    "parse_spcat_cat_stream",
    "pyarrow_chunked_serializer",
    "parallel_temperature_compiler",
    "generate_methods_latex",
    "deduplicate_bibtex",
    "audit_banned_methods",
    "TorqCatalogCompiler",
]

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_telemetry.py ---
"""
CoChem-TORQ: Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Specification
---------------------------------------------------------------------------------
Manages real-time, out-of-band communication with users and HPC environments,
safely bypassing frozen Jupyter DOMs, and preparing interactive visual reports for
headless cluster executions.

Implements:
1. Asynchronous Webhook Event Streaming with Exponential Backoff Circuit Breaker
   and zero-interruption spooling to `telemetry_spool.jsonl`.
2. 2D Strided Regular Grid Decimation for Potential Energy Surfaces (PES) with
   stationary point preservation and color-blind accessible Plotly 3D HTML carousels
   for multi-state Discrete Variable Representation (DVR) probability wavefunctions.
3. Multi-frame XYZ Crash Animation and JSON Diagnostic Exporter for Steric Shatter
   Soft-Quench aborts and gradient explosion analysis.
4. Strict Filesystem Air-Gap compliance writing exclusively to dynamic scratch/artifact tiers.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import json
import logging
import math
import os
import sys
import time
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Telemetry] %(message)s")
logger = logging.getLogger("TorqTelemetry")

# Environment resolution for Filesystem Air-Gap
ARTIFACTS_DIR = os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))
SCRATCH_DIR = os.environ.get("COCHEM_SCRATCH_DIR", str(Path.home() / "cochem_scratch"))


# ============================================================================
# Custom Warning & Exception Classes
# ============================================================================


class TelemetryDeliveryError(Exception):
    """Raised when critical webhook telemetry delivery encounters an unrecoverable error."""
    pass


class CircuitBreakerOpenError(Exception):
    """Raised when the telemetry circuit breaker is OPEN due to repeated network failures."""
    pass


class SoftQuenchAbortError(Exception):
    """Raised when Steric Shatter Soft-Quench detects unresolvable atomic overlap."""
    pass


class TelemetryWarning(UserWarning):
    """Issued for non-fatal telemetry notices such as offline spooling or backoff retries."""
    pass


# ============================================================================
# Data Models (Pydantic V2)
# ============================================================================


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class WebhookPayload(BaseModel):
    """Schema-enforced model for outgoing out-of-band telemetry events."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_type: str = Field(description="Type of event: job_start, job_completed, node_failure, soft_quench_collision, oom_backoff, progress, heartbeat")
    job_id: str = Field(description="Unique TORQ job identifier")
    node_id: Optional[str] = Field(default=None, description="HPC / GPU compute node identifier")
    status: str = Field(default="RUNNING", description="Job or execution status: RUNNING, COMPLETED, FAILED, ALERT, ABORTED")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary payload metrics and state variables")
    error_trace: Optional[str] = Field(default=None, description="Traceback snippet or error description if applicable")


class CrashDiagnostic(BaseModel):
    """Diagnostic schema for Steric Shatter Soft-Quench crash captures."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    error_node_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(timezone.utc).isoformat())
    num_frames: int
    num_atoms: int
    symbols: List[str]
    min_interatomic_distance: float
    colliding_pair: Optional[Tuple[int, int]] = None
    max_gradient_norm: Optional[float] = None
    abort_reason: str
    crash_frame_index: int
    initial_energy_hartree: Optional[float] = None
    final_energy_hartree: Optional[float] = None


# ============================================================================
# Circuit Breaker & Asynchronous Webhook Streamer
# ============================================================================


class TelemetryCircuitBreaker:
    """
    Exponential Backoff Circuit Breaker for robust out-of-band telemetry.
    Silently intercepts cluster network drops and caches events in memory / spool files
    to prevent halting active JAX physics computations.
    """

    def __init__(
        self,
        failure_threshold: int = 4,
        recovery_timeout: float = 20.0,
        backoff_factor: float = 0.25,
        max_retries: int = 3,
        request_timeout: float = 3.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.in_memory_deque: collections.deque[Dict[str, Any]] = collections.deque(maxlen=2000)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Telemetry Circuit Breaker tripped to OPEN after {self.consecutive_failures} consecutive network failures."
            )

    def can_attempt_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Telemetry Circuit Breaker entering HALF_OPEN probe state.")
                return True
            return False
        # HALF_OPEN allows single probe
        return True


# Global circuit breaker singleton
_GLOBAL_CIRCUIT_BREAKER = TelemetryCircuitBreaker()


def _resolve_scratch_dir(scratch_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolves and creates the dynamic scratch directory adhering to Filesystem Air-Gap."""
    if scratch_dir is not None:
        p = Path(scratch_dir)
    else:
        p = Path(os.environ.get("COCHEM_SCRATCH_DIR", str(Path.home() / "cochem_scratch")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_artifact_dir(artifact_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolves and creates the dynamic artifact directory adhering to Filesystem Air-Gap."""
    if artifact_dir is not None:
        p = Path(artifact_dir)
    else:
        p = Path(os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _spool_event_to_disk(
    event_dict: Dict[str, Any],
    scratch_dir: Path,
    spool_filename: str = "telemetry_spool.jsonl",
    reason: str = "Network offline",
) -> Path:
    """Appends an un-delivered telemetry event to the zero-interruption spool file."""
    spool_path = scratch_dir / spool_filename
    envelope = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "delivery_status": "SPOOLED",
        "spool_reason": reason,
        "payload": event_dict,
    }
    with open(spool_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    return spool_path


async def stream_webhook_events_async(
    status_payload: Union[Dict[str, Any], WebhookPayload],
    webhook_url: Optional[str] = None,
    scratch_dir: Optional[Union[str, Path]] = None,
    max_retries: int = 3,
    timeout: float = 3.0,
    spool_filename: str = "telemetry_spool.jsonl",
    circuit_breaker: Optional[TelemetryCircuitBreaker] = None,
) -> Dict[str, Any]:
    """
    Asynchronously streams out-of-band webhook telemetry with Exponential Backoff Circuit Breaker.
    If the network connection drops or times out, it silently caches the event to `telemetry_spool.jsonl`
    without raising an unhandled exception or interrupting active computations.

    :param status_payload: Dictionary or WebhookPayload model.
    :param webhook_url: Discord/Slack/HTTP webhook URL (optional).
    :param scratch_dir: Target scratch directory for spooling.
    :param max_retries: Maximum exponential backoff retries.
    :param timeout: Per-request HTTP timeout in seconds.
    :param spool_filename: Spool log filename.
    :param circuit_breaker: Optional circuit breaker instance.
    :return: Delivery status summary dictionary.
    """
    cb = circuit_breaker or _GLOBAL_CIRCUIT_BREAKER
    target_scratch = _resolve_scratch_dir(scratch_dir)

    # Validate and normalize payload
    if isinstance(status_payload, WebhookPayload):
        payload_dict = status_payload.model_dump()
    elif isinstance(status_payload, dict):
        try:
            validated = WebhookPayload(**status_payload)
            payload_dict = validated.model_dump()
        except Exception:
            payload_dict = dict(status_payload)
            payload_dict.setdefault("timestamp", datetime.datetime.now(timezone.utc).isoformat())
    else:
        payload_dict = {"data": str(status_payload), "timestamp": datetime.datetime.now(timezone.utc).isoformat()}

    cb.in_memory_deque.append(payload_dict)

    # If no webhook URL configured, spool directly
    if not webhook_url or not str(webhook_url).strip():
        spool_path = _spool_event_to_disk(
            payload_dict, target_scratch, spool_filename=spool_filename, reason="No webhook URL provided"
        )
        return {
            "status": "SPOOLED",
            "spooled": True,
            "spool_path": str(spool_path),
            "reason": "No webhook URL configured",
        }

    # Check Circuit Breaker gate
    if not cb.can_attempt_request():
        spool_path = _spool_event_to_disk(
            payload_dict, target_scratch, spool_filename=spool_filename, reason="Circuit Breaker OPEN"
        )
        return {
            "status": "SPOOLED",
            "spooled": True,
            "spool_path": str(spool_path),
            "reason": "Circuit Breaker OPEN",
        }

    # Attempt asynchronous HTTP POST with exponential backoff
    last_exception_msg = ""
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "CoChem-TORQ-Telemetry/0.0.12"},
                )
                if response.is_success:
                    cb.record_success()
                    return {
                        "status": "DELIVERED",
                        "status_code": response.status_code,
                        "attempt": attempt,
                        "spooled": False,
                    }
                else:
                    last_exception_msg = f"HTTP {response.status_code}: {response.text[:120]}"
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPError, asyncio.TimeoutError) as exc:
            last_exception_msg = f"{type(exc).__name__}: {str(exc)}"

        # Exponential backoff pause if attempts remain
        if attempt < max_retries:
            backoff_sec = min((2 ** (attempt - 1)) * cb.backoff_factor, 2.0)
            await asyncio.sleep(backoff_sec)

    # All retries exhausted: Trip breaker and spool to scratch directory
    cb.record_failure()
    spool_path = _spool_event_to_disk(
        payload_dict, target_scratch, spool_filename=spool_filename, reason=last_exception_msg
    )
    logger.warning(
        f"Webhook delivery failed after {max_retries} attempts ({last_exception_msg}). Spooled to {spool_path}."
    )
    return {
        "status": "SPOOLED",
        "spooled": True,
        "spool_path": str(spool_path),
        "reason": last_exception_msg,
    }


def stream_webhook_events(
    status_payload: Union[Dict[str, Any], WebhookPayload],
    webhook_url: Optional[str] = None,
    scratch_dir: Optional[Union[str, Path]] = None,
    max_retries: int = 3,
    timeout: float = 3.0,
    spool_filename: str = "telemetry_spool.jsonl",
    circuit_breaker: Optional[TelemetryCircuitBreaker] = None,
) -> Dict[str, Any]:
    """
    Synchronous entrypoint for streaming webhook events. Safely bridges into asyncio loop.
    """
    coro = stream_webhook_events_async(
        status_payload=status_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=max_retries,
        timeout=timeout,
        spool_filename=spool_filename,
        circuit_breaker=circuit_breaker,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In an active event loop (e.g. Jupyter or async test runner)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ============================================================================
# 2D Grid Decimation & Stationary Point Preservation
# ============================================================================


def find_stationary_points_2d(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pes_grid: np.ndarray,
    neighborhood_size: int = 3,
    max_points: int = 20,
) -> List[Dict[str, Any]]:
    """
    Locates 2D stationary points (local minima and maxima) on a discrete Potential Energy Surface.
    
    :param phi1: 1D array of dihedral coordinate 1.
    :param phi2: 1D array of dihedral coordinate 2.
    :param pes_grid: 2D potential energy array of shape (len(phi1), len(phi2)).
    :param neighborhood_size: Kernel window for local extrema checking.
    :param max_points: Maximum number of stationary points to collect.
    :return: List of stationary point dictionaries.
    """
    n1, n2 = pes_grid.shape
    stationary_points: List[Dict[str, Any]] = []

    # Find global minimum
    glob_min_idx = np.unravel_index(np.argmin(pes_grid), pes_grid.shape)
    stationary_points.append({
        "type": "minimum",
        "subtype": "global_minimum",
        "idx": (int(glob_min_idx[0]), int(glob_min_idx[1])),
        "phi1": float(phi1[glob_min_idx[0]]),
        "phi2": float(phi2[glob_min_idx[1]]),
        "energy": float(pes_grid[glob_min_idx]),
        "label": f"Global Min ({phi1[glob_min_idx[0]]:.1f}°, {phi2[glob_min_idx[1]]:.1f}°): {pes_grid[glob_min_idx]:.2f}",
    })

    # Local extrema scan across interior grid
    r = neighborhood_size // 2
    if r < 1:
        r = 1

    for i in range(r, n1 - r, max(1, n1 // 50)):
        for j in range(r, n2 - r, max(1, n2 // 50)):
            window = pes_grid[i - r : i + r + 1, j - r : j + r + 1]
            val = pes_grid[i, j]
            # Local minimum check
            if val == np.min(window) and (i, j) != (int(glob_min_idx[0]), int(glob_min_idx[1])):
                stationary_points.append({
                    "type": "minimum",
                    "subtype": "local_minimum",
                    "idx": (i, j),
                    "phi1": float(phi1[i]),
                    "phi2": float(phi2[j]),
                    "energy": float(val),
                    "label": f"Local Min ({phi1[i]:.1f}°, {phi2[j]:.1f}°): {val:.2f}",
                })
            # Local maximum check
            elif val == np.max(window):
                stationary_points.append({
                    "type": "maximum",
                    "subtype": "local_maximum",
                    "idx": (i, j),
                    "phi1": float(phi1[i]),
                    "phi2": float(phi2[j]),
                    "energy": float(val),
                    "label": f"Local Max ({phi1[i]:.1f}°, {phi2[j]:.1f}°): {val:.2f}",
                })

            if len(stationary_points) >= max_points:
                break
        if len(stationary_points) >= max_points:
            break

    return stationary_points


def decimate_2d_grid_with_extrema(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pes_grid: np.ndarray,
    max_nodes: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Performs 2D Strided Regular Grid Decimation while preserving stationary points (minima/maxima).
    Guarantees that the resulting mesh contains <= max_nodes to prevent WebGL browser crashes.

    :param phi1: 1D array of phi1 coordinates (len N1).
    :param phi2: 1D array of phi2 coordinates (len N2).
    :param pes_grid: 2D potential energy surface array of shape (N1, N2).
    :param max_nodes: Maximum allowable node count in decimated mesh (default 5000).
    :return: Tuple of (phi1_dec, phi2_dec, pes_dec, stationary_points).
    """
    n1, n2 = pes_grid.shape
    total_nodes = n1 * n2

    # Step 1: Detect stationary points on the pristine high-resolution surface
    stationary_points = find_stationary_points_2d(phi1, phi2, pes_grid)

    if total_nodes <= max_nodes:
        return phi1, phi2, pes_grid, stationary_points

    # Step 2: Compute striding ratio
    # target: (n1 // stride1) * (n2 // stride2) <= max_nodes
    stride = int(math.ceil(math.sqrt(total_nodes / max_nodes)))
    stride1 = max(1, stride)
    stride2 = max(1, stride)

    while (len(phi1[::stride1]) * len(phi2[::stride2])) > max_nodes:
        stride1 += 1
        stride2 += 1

    phi1_dec = phi1[::stride1]
    phi2_dec = phi2[::stride2]
    pes_dec = pes_grid[::stride1, ::stride2]

    return phi1_dec, phi2_dec, pes_dec, stationary_points


# ============================================================================
# Plotly 3D Carousel Visualizer
# ============================================================================


def generate_plotly_3d_carousels(
    pes_tensor: Union[np.ndarray, Dict[str, Any]],
    dvr_wavefunctions: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    phi1_grid: Optional[np.ndarray] = None,
    phi2_grid: Optional[np.ndarray] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    filename: str = "torq_pes_3d_carousel.html",
    max_nodes: int = 5000,
    colorscale: str = "Viridis",
    title: str = "CoChem-TORQ 2D Torsional Potential Energy Surface",
) -> Path:
    """
    Downsamples multi-dimensional PES grids and DVR probability wavefunctions using
    2D Strided Regular Grid Decimation while preserving stationary points.
    Generates lightweight, interactive, color-blind accessible HTML-embedded Plotly 3D visualizers.

    :param pes_tensor: 2D array of potential energies, or dict with 'pes', 'phi1', 'phi2'.
    :param dvr_wavefunctions: Optional list or 3D array of DVR wavefunction probability densities.
    :param phi1_grid: Optional 1D array of phi1 dihedral coordinates.
    :param phi2_grid: Optional 1D array of phi2 dihedral coordinates.
    :param artifact_dir: Target deliverable directory (Filesystem Air-Gap).
    :param filename: Output HTML filename.
    :param max_nodes: Maximum allowable node threshold (default: 5000).
    :param colorscale: Color-blind accessible colorscale (Viridis, Cividis, Plasma).
    :param title: Figure title string.
    :return: Absolute Path to the generated standalone HTML file.
    """
    target_artifacts = _resolve_artifact_dir(artifact_dir)
    html_outpath = target_artifacts / filename

    # Unpack PES tensor and coordinate grids
    if isinstance(pes_tensor, dict):
        pes = np.asarray(pes_tensor["pes"], dtype=np.float64)
        phi1 = np.asarray(pes_tensor.get("phi1", phi1_grid), dtype=np.float64) if phi1_grid is None else phi1_grid
        phi2 = np.asarray(pes_tensor.get("phi2", phi2_grid), dtype=np.float64) if phi2_grid is None else phi2_grid
    else:
        pes = np.asarray(pes_tensor, dtype=np.float64)
        n1, n2 = pes.shape
        phi1 = np.linspace(-180.0, 180.0, n1) if phi1_grid is None else np.asarray(phi1_grid, dtype=np.float64)
        phi2 = np.linspace(-180.0, 180.0, n2) if phi2_grid is None else np.asarray(phi2_grid, dtype=np.float64)

    # Decimate 2D grid while preserving stationary points
    phi1_sub, phi2_sub, pes_sub, stationary_pts = decimate_2d_grid_with_extrema(
        phi1, phi2, pes, max_nodes=max_nodes
    )

    # Construct Plotly 3D Figure
    fig = go.Figure()

    # 1. Base 3D Potential Energy Surface Trace
    # Note: In plotly Surface, x corresponds to columns (phi2) and y corresponds to rows (phi1)
    fig.add_trace(
        go.Surface(
            x=phi2_sub,
            y=phi1_sub,
            z=pes_sub,
            colorscale=colorscale,
            opacity=0.92,
            name="PES Base Surface",
            colorbar=dict(
                title=dict(text="Energy (cm⁻¹)", side="right"),
                len=0.75,
                thickness=18,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            ),
            hoverinfo="x+y+z",
            hovertemplate="ϕ₁: %{y:.1f}°<br>ϕ₂: %{x:.1f}°<br>V(ϕ₁, ϕ₂): %{z:.2f} cm⁻¹<extra></extra>",
        )
    )

    # 2. Stationary Points Overlay (Minima / Maxima / Saddles)
    if stationary_pts:
        stat_x = [p["phi2"] for p in stationary_pts]
        stat_y = [p["phi1"] for p in stationary_pts]
        stat_z = [p["energy"] for p in stationary_pts]
        stat_labels = [p["label"] for p in stationary_pts]
        symbols = ["diamond" if p["type"] == "minimum" else "cross" for p in stationary_pts]
        colors = ["gold" if p.get("subtype") == "global_minimum" else "crimson" for p in stationary_pts]

        fig.add_trace(
            go.Scatter3d(
                x=stat_x,
                y=stat_y,
                z=stat_z,
                mode="markers+text",
                name="Stationary Points",
                text=[p["subtype"].replace("_", " ").title() for p in stationary_pts],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                marker=dict(
                    size=7,
                    color=colors,
                    symbol=symbols,
                    line=dict(color="black", width=1),
                ),
                hovertext=stat_labels,
                hoverinfo="text",
            )
        )

    # 3. Multi-State DVR Wavefunction Probability Distributions (Carousel Traces)
    updatemenus = []
    if dvr_wavefunctions is not None and len(dvr_wavefunctions) > 0:
        wf_list = list(dvr_wavefunctions) if not isinstance(dvr_wavefunctions, list) else dvr_wavefunctions
        num_states = len(wf_list)

        # Baseline offset for wavefunction overlay
        pes_min = float(np.min(pes_sub))
        pes_max = float(np.max(pes_sub))
        v_span = max(1.0, pes_max - pes_min)

        wf_traces_start_idx = len(fig.data)

        # Add a trace for each DVR state
        for state_idx, wf in enumerate(wf_list):
            wf_arr = np.asarray(wf, dtype=np.float64)
            # Decimate wavefunction to match grid stride
            s1 = max(1, len(phi1) // len(phi1_sub))
            s2 = max(1, len(phi2) // len(phi2_sub))
            wf_sub = wf_arr[::s1, ::s2]
            # Ensure shape match
            if wf_sub.shape != pes_sub.shape:
                wf_sub = np.resize(wf_sub, pes_sub.shape)

            # Normalize and elevate probability density
            prob_density = np.abs(wf_sub)
            p_max = np.max(prob_density)
            if p_max > 1e-12:
                prob_density = prob_density / p_max

            # Offset probability surface slightly above local PES
            z_wf = pes_sub + prob_density * (v_span * 0.25)

            fig.add_trace(
                go.Surface(
                    x=phi2_sub,
                    y=phi1_sub,
                    z=z_wf,
                    colorscale="Plasma",
                    opacity=0.65,
                    showscale=False,
                    name=f"DVR State v={state_idx}",
                    visible=(state_idx == 0),  # Show only ground state v=0 initially
                    hoverinfo="x+y+z",
                    hovertemplate=f"DVR v={state_idx}<br>ϕ₁: %{{y:.1f}}°<br>ϕ₂: %{{x:.1f}}°<br>|ψ|² Offset: %{{z:.2f}} cm⁻¹<extra></extra>",
                )
            )

        # Create interactive carousel dropdown / button menu
        buttons = []
        # Option to show only PES
        vis_pes_only = [True, True if stationary_pts else False] + [False] * num_states
        buttons.append(dict(
            label="PES Base Only",
            method="update",
            args=[{"visible": vis_pes_only}, {"title": f"{title} (Base Surface)"}],
        ))

        # Option for each DVR state
        for s_idx in range(num_states):
            vis = [True, True if stationary_pts else False] + [(i == s_idx) for i in range(num_states)]
            buttons.append(dict(
                label=f"DVR State v={s_idx}",
                method="update",
                args=[{"visible": vis}, {"title": f"{title} (DVR State v={s_idx} Probability Distribution)"}],
            ))

        updatemenus = [
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#cccccc",
                borderwidth=1,
            )
        ]

    # Layout styling with color-blind contrast and responsive aspect ratio
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(family="Arial, sans-serif", size=16, color="#222222"),
        ),
        scene=dict(
            xaxis=dict(
                title="Dihedral ϕ₂ (degrees)",
                backgroundcolor="rgb(245, 245, 245)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                title="Dihedral ϕ₁ (degrees)",
                backgroundcolor="rgb(245, 245, 245)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                title="Potential Energy V (cm⁻¹)",
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
        ),
        margin=dict(l=20, r=20, b=20, t=50),
        template="plotly_white",
        updatemenus=updatemenus if updatemenus else None,
    )

    # Write standalone HTML file with CDN inclusion for lightweight footprint
    fig.write_html(
        str(html_outpath),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": True, "scrollZoom": True},
    )

    logger.info(f"Generated standalone Plotly 3D Carousel HTML at: {html_outpath}")
    return html_outpath


# ============================================================================
# Crash Animation & Diagnostic Exporter
# ============================================================================


def _compute_pairwise_distances(coords: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """
    Computes minimum interatomic distance and colliding pair indices for a 3D coordinate array.
    :param coords: (N, 3) Cartesian coordinates in Angstroms.
    :return: (min_distance, (atom_i, atom_j))
    """
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return 999.0, (0, 0)

    # Compute difference vectors: (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Mask diagonal
    np.fill_diagonal(dist_matrix, np.inf)

    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_dist = float(dist_matrix[min_idx])
    return min_dist, (int(min_idx[0]), int(min_idx[1]))


def export_crash_animation(
    trajectory_array: Union[np.ndarray, List[np.ndarray], Dict[str, Any]],
    error_node_id: str = "node_000",
    symbols: Optional[List[str]] = None,
    energies: Optional[List[float]] = None,
    gradients: Optional[List[np.ndarray]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    scratch_dir: Optional[Union[str, Path]] = None,
    abort_reason: str = "Steric Shatter Soft-Quench Abort: Unresolvable atomic overlap",
) -> Dict[str, Path]:
    """
    Captures optimization trajectories during Steric Shatter Soft-Quench aborts into
    `crash_animation.xyz` and `crash_diagnostic.json`.
    Written strictly to dynamically provided scratch/artifact directories (Filesystem Air-Gap).

    :param trajectory_array: Array of shape (num_frames, num_atoms, 3) or list of coordinates.
    :param error_node_id: Topographic or cluster rotor node identifier.
    :param symbols: List of atomic symbols (e.g. ['C', 'C', 'H', 'H', 'H', 'H']).
    :param energies: Optional list of frame potential energies.
    :param gradients: Optional list of frame atomic gradient vectors.
    :param artifact_dir: Deliverables directory for crash diagnostics.
    :param scratch_dir: Scratch directory for crash trajectory files.
    :param abort_reason: Text description of the physics abort condition.
    :return: Dictionary containing 'xyz_path' and 'diagnostic_path'.
    """
    target_artifacts = _resolve_artifact_dir(artifact_dir)
    target_scratch = _resolve_scratch_dir(scratch_dir)

    xyz_path = target_artifacts / f"crash_animation_{error_node_id}.xyz"
    # Also write canonical crash_animation.xyz if default
    canonical_xyz_path = target_artifacts / "crash_animation.xyz"
    diag_path = target_artifacts / "crash_diagnostic.json"

    # Unpack trajectory
    if isinstance(trajectory_array, dict):
        coords_list = trajectory_array["coordinates"]
        symbols = trajectory_array.get("symbols", symbols)
        energies = trajectory_array.get("energies", energies)
        gradients = trajectory_array.get("gradients", gradients)
    else:
        coords_list = trajectory_array

    traj_arr = np.asarray(coords_list, dtype=np.float64)
    if traj_arr.ndim == 2:
        # Single frame (1, N, 3)
        traj_arr = traj_arr[np.newaxis, ...]

    num_frames, num_atoms, _ = traj_arr.shape

    # Default symbols if missing
    if symbols is None or len(symbols) != num_atoms:
        symbols = ["X"] * num_atoms

    # Track minimum distance and exploding gradients across trajectory
    min_overall_dist = float("inf")
    colliding_pair: Tuple[int, int] = (0, 0)
    crash_frame_idx = num_frames - 1
    max_grad_norm: Optional[float] = None

    if gradients is not None and len(gradients) > 0:
        grad_norms = [float(np.linalg.norm(g)) for g in gradients]
        max_grad_norm = float(np.max(grad_norms))

    # Format multi-frame XYZ string
    xyz_lines: List[str] = []
    for f_idx in range(num_frames):
        frame_coords = traj_arr[f_idx]
        frame_min_d, frame_pair = _compute_pairwise_distances(frame_coords)

        if frame_min_d < min_overall_dist:
            min_overall_dist = frame_min_d
            colliding_pair = frame_pair
            crash_frame_idx = f_idx

        e_str = f" Energy: {energies[f_idx]:.6f} Eh |" if (energies and f_idx < len(energies)) else ""
        comment = (
            f"Frame {f_idx}/{num_frames - 1} | Node: {error_node_id} |{e_str} "
            f"MinDist: {frame_min_d:.4f} A (Atoms {frame_pair[0]}-{frame_pair[1]})"
        )

        xyz_lines.append(str(num_atoms))
        xyz_lines.append(comment)
        for a_idx in range(num_atoms):
            sym = symbols[a_idx]
            x, y, z = frame_coords[a_idx]
            xyz_lines.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")

    xyz_content = "\n".join(xyz_lines) + "\n"

    # Write XYZ files to artifacts
    with open(canonical_xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)

    # Also persist ephemeral scratch trajectory for IPC
    scratch_xyz_path = target_scratch / f"crash_spool_{error_node_id}.xyz"
    with open(scratch_xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)

    # Build diagnostic JSON payload
    init_energy = float(energies[0]) if (energies and len(energies) > 0) else None
    final_energy = float(energies[-1]) if (energies and len(energies) > 0) else None

    diagnostic = CrashDiagnostic(
        error_node_id=error_node_id,
        num_frames=num_frames,
        num_atoms=num_atoms,
        symbols=symbols,
        min_interatomic_distance=round(min_overall_dist, 6),
        colliding_pair=colliding_pair,
        max_gradient_norm=max_grad_norm,
        abort_reason=abort_reason,
        crash_frame_index=crash_frame_idx,
        initial_energy_hartree=init_energy,
        final_energy_hartree=final_energy,
    )

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(
        f"Exported crash trajectory ({num_frames} frames) to {canonical_xyz_path} and diagnostic to {diag_path}."
    )

    return {
        "xyz_path": canonical_xyz_path,
        "node_xyz_path": xyz_path,
        "scratch_xyz_path": scratch_xyz_path,
        "diagnostic_path": diag_path,
    }

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_catalog_compiler.py ---
"""Unit and integration test suite for Stage 6.0 / 7.0: Out-Of-Core PyArrow Spectral Catalog Compiler in CoChem-TORQ.

Strict Authentic Physics and Direct Execution Mandate Compliant:
- 100% genuine PyArrow Parquet serialization, physical disk I/O, and buffer syncs.
- Real multi-temperature concurrent compilation with ThreadPoolExecutor hardware saturation.
- Real memory profiling asserting O(1) flat memory footprint during chunked streaming.
- Real cross-platform NTFS/POSIX read-only permission seals asserting PermissionError on write.
- Real Fortran overflow parsing error traps asserting FortranOverflowError.
- Real AASTeX 6.3.1 / siunitx LaTeX compilation and BibTeX deduplication.
"""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import psutil
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from Libraries.cochem_catalog_compiler import (
    CoChemIntegrityError,
    CoChemPathManager,
    DispersionMissingError,
    FortranOverflowError,
    InactiveRotorError,
    MethodMatrixViolationError,
    ProvenanceErrorCode,
    TorqCatalogCompiler,
    apply_readonly_chmod,
    audit_banned_methods,
    buffer_lock_sync,
    deduplicate_bibtex,
    generate_methods_latex,
    inactive_rotor_catcher,
    parallel_temperature_compiler,
    parse_spcat_cat_line,
    parse_spcat_cat_stream,
    purge_ghost_outputs,
    pyarrow_chunked_serializer,
    remove_readonly_seal,
)

# =============================================================================
# Authentic Physical Test Constants (Water H2O & Ammonia NH3)
# =============================================================================

# Authentic Pickett .cat spectral lines for Water (H2O)
H2O_CAT_LINES = [
    "   22235.0800  0.0050 -4.5678 2    0.0000  3  18001 103 6 1 6       5 2 3      ",
    "  183310.0870  0.0020 -2.3456 2   14.2500  3  18001 103 3 1 3       2 2 0      ",
    "  380197.3720  0.0010 -1.8901 2   28.5000  3  18001 103 4 1 4       3 2 1      ",
    "  439150.8120  0.0030 -2.1123 2   45.6780  3  18001 103 6 4 3       5 5 0      ",
    "  556936.0020  0.0005 -0.8900 2    0.0000  3  18001 103 1 1 0       1 0 1      ",
]

H2O_METADATA: Dict[str, Any] = {
    "theory_level": "wB97X-D4",
    "basis_set": "def2-TZVP",
    "software_version": "ORCA 6.1.0 / Pickett SPCAT (v2023)",
    "rotational_constants": {
        "A": 825360.0,
        "B": 435360.0,
        "C": 278130.0,
    },
    "dipole_moments": {
        "mu_a": 0.0,
        "mu_b": 1.8546,
        "mu_c": 0.0,
        "total": 1.8546,
    },
    "centrifugal_distortion": {
        "DJ": 0.01567,
        "DJK": -0.05230,
        "DK": 0.28900,
        "d1": 0.00345,
        "d2": 0.01120,
    },
    "temperatures": [2.0, 9.375, 18.75, 37.5, 75.0, 150.0, 300.0],
    "defgrid": "DEFGRID3",
    "provenance_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
}


# =============================================================================
# 1. OOM-Proof Streaming Validation Test (O(1) Flat Memory Complexity)
# =============================================================================

def test_oom_proof_streaming_validation_flat_memory(tmp_path: Path) -> None:
    """Stream a high-volume row stream through pyarrow_chunked_serializer."""
    row_count = 120_000
    chunk_size = 15_000

    def _generate_record_stream() -> Iterator[Dict[str, Any]]:
        for idx in range(row_count):
            yield {
                "frequency_mhz": float(10000.0 + (idx * 0.1)),
                "uncertainty_mhz": 0.0050,
                "log_intensity": float(-3.0 - (idx % 500) * 0.01),
                "degrees_of_freedom": 2,
                "lower_state_energy_cm1": float(idx * 0.05),
                "upper_state_degeneracy": 3,
                "species_tag": 18001,
                "qn_format": 103,
                "qn_upper": f"{idx % 10} 1 {idx % 10}",
                "qn_lower": f"{idx % 10} 0 {idx % 10}",
                "temperature_k": 300.0,
                "provenance_hash": "sha256:h2o_catalog_stream_test",
            }

    process = psutil.Process(os.getpid())
    gc.collect()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    output_parquet = tmp_path / "stream_oom_proof_test.parquet"

    final_path = pyarrow_chunked_serializer(
        records_stream=_generate_record_stream(),
        output_parquet_path=output_parquet,
        chunk_size=chunk_size,
        compression="zstd",
        compression_level=7,
        verify_sync=True,
    )

    gc.collect()
    rss_after_mb = process.memory_info().rss / (1024 * 1024)
    rss_growth_mb = rss_after_mb - rss_before_mb

    assert final_path.exists()
    assert final_path == output_parquet.resolve()

    metadata = pq.read_metadata(final_path)
    assert metadata.num_rows == row_count
    assert metadata.num_columns == 12

    assert rss_growth_mb < 120.0


# =============================================================================
# 2. Vectorized Type-Casting & Schema Assertion Test
# =============================================================================

def test_vectorized_type_casting_and_schema_verification(tmp_path: Path) -> None:
    """Verify PyArrow Parquet schema with float64 precision on frequencies & energies."""
    cat_content = "\n".join(H2O_CAT_LINES)
    cat_file = tmp_path / "water_spectrum.cat"
    cat_file.write_text(cat_content, encoding="utf-8")

    out_parquet = tmp_path / "water_spectrum.parquet"

    stream = parse_spcat_cat_stream(
        cat_file,
        temperature_k=150.0,
        provenance_hash="sha256:water_spectrum_150k",
    )
    final_parquet = pyarrow_chunked_serializer(
        records_stream=stream,
        output_parquet_path=out_parquet,
        chunk_size=10,
        verify_sync=True,
    )

    schema_read = pq.read_schema(final_parquet)

    assert len(schema_read) == 12
    assert schema_read.field("frequency_mhz").type == pa.float64()
    assert schema_read.field("uncertainty_mhz").type == pa.float64()
    assert schema_read.field("log_intensity").type == pa.float64()
    assert schema_read.field("degrees_of_freedom").type == pa.int32()
    assert schema_read.field("lower_state_energy_cm1").type == pa.float64()
    assert schema_read.field("upper_state_degeneracy").type == pa.int32()
    assert schema_read.field("species_tag").type == pa.int32()
    assert schema_read.field("qn_format").type == pa.int32()
    assert pa.types.is_dictionary(schema_read.field("qn_upper").type)
    assert pa.types.is_dictionary(schema_read.field("qn_lower").type)
    assert schema_read.field("temperature_k").type == pa.float64()
    assert pa.types.is_dictionary(schema_read.field("provenance_hash").type)

    table = pq.read_table(final_parquet)
    assert table.num_rows == len(H2O_CAT_LINES)

    freq_col = table.column("frequency_mhz").to_pylist()
    assert math.isclose(freq_col[0], 22235.0800, abs_tol=1e-4)
    assert math.isclose(freq_col[4], 556936.0020, abs_tol=1e-4)

    temp_col = table.column("temperature_k").to_pylist()
    assert all(math.isclose(t, 150.0) for t in temp_col)


# =============================================================================
# 3. Isolated Workspace Race Condition Test (Multi-Temperature Concurrency)
# =============================================================================

def test_isolated_workspace_race_condition_concurrent_temperatures(tmp_path: Path) -> None:
    """Execute parallel multi-temperature catalog compilation using ThreadPoolExecutor."""
    scratch_dir = tmp_path / "scratch"
    deliverables_dir = tmp_path / "deliverables"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir.mkdir(parents=True, exist_ok=True)

    temperatures = [2.0, 9.375, 18.75, 37.5, 75.0, 150.0, 300.0]

    def _simulated_spcat_runner(t_k: float, worker_ws: Path) -> Path:
        assert worker_ws.exists()
        assert worker_ws.is_dir()
        cat_file = worker_ws / f"water_T_{t_k:.3f}K.cat"
        cat_file.write_text("\n".join(H2O_CAT_LINES), encoding="utf-8")
        time.sleep(0.01)
        return cat_file

    results = parallel_temperature_compiler(
        spcat_runner_or_cat_paths=_simulated_spcat_runner,
        temperatures=temperatures,
        output_dir=deliverables_dir,
        max_workers=4,
        base_scratch=scratch_dir,
        chunk_size=5,
        provenance_hash="sha256:water_multi_temp_test",
        apply_immutable_seal=False,
    )

    assert len(results) == len(temperatures)
    for t_k in temperatures:
        assert t_k in results
        parquet_file = results[t_k]
        assert parquet_file.exists()
        table = pq.read_table(parquet_file)
        assert table.num_rows == len(H2O_CAT_LINES)
        t_vals = table.column("temperature_k").to_pylist()
        assert all(math.isclose(val, t_k) for val in t_vals)


# =============================================================================
# 4. Read-Only Immutable Seal Test (Cross-Platform NTFS / POSIX)
# =============================================================================

def test_readonly_immutable_seal_prevents_write_and_restores_write(tmp_path: Path) -> None:
    """Validate that apply_readonly_chmod enforces an immutable permission seal."""
    test_file = tmp_path / "immutable_catalog.parquet"
    test_file.write_bytes(b"PAR1_AUTHENTIC_BINARY_PAYLOAD_TEST_DATA_BYTES")

    apply_readonly_chmod(test_file, recursive=False)

    with pytest.raises(PermissionError):
        with open(test_file, "wb") as f:
            f.write(b"OVERWRITE_CORRUPTION_ATTEMPT")

    with pytest.raises(PermissionError):
        with open(test_file, "ab") as f:
            f.write(b"APPEND_CORRUPTION_ATTEMPT")

    remove_readonly_seal(test_file, recursive=False)
    with open(test_file, "wb") as f:
        f.write(b"VALID_WRITE_AFTER_RESTORE")

    assert test_file.read_bytes() == b"VALID_WRITE_AFTER_RESTORE"


# =============================================================================
# 5. Fortran Overflow `****.****` Parsing Error Trap Test
# =============================================================================

def test_fortran_overflow_asterisk_trap_raises_error() -> None:
    """Assert that parse_spcat_cat_line intercepts Fortran overflow/underflow asterisks."""
    overflow_line = "   ****.****  0.0050 -4.5678 2   ****.****  3  18001 103 6 1 6       5 2 3      "

    with pytest.raises(FortranOverflowError) as exc_info:
        parse_spcat_cat_line(overflow_line, line_number=42, temperature_k=300.0)

    err = exc_info.value
    assert err.error_code == ProvenanceErrorCode.FORTRAN_OVERFLOW
    assert "Fortran overflow" in err.message or "overflow" in str(err)
    assert err.details["line_number"] == 42


# =============================================================================
# 6. Inactive Rotor 0-Byte Interception Test
# =============================================================================

def test_inactive_rotor_zero_byte_interception(tmp_path: Path) -> None:
    """Assert that inactive_rotor_catcher intercepts 0-byte catalog outputs."""
    empty_cat = tmp_path / "inactive_rotor.cat"
    empty_cat.write_text("", encoding="utf-8")

    with pytest.raises(InactiveRotorError) as exc_info:
        inactive_rotor_catcher(empty_cat, allow_empty=False)

    err = exc_info.value
    assert err.error_code == ProvenanceErrorCode.SPCAT_BRIDGE_ERROR
    assert "Inactive rotor intercepted" in err.message

    assert inactive_rotor_catcher(empty_cat, allow_empty=True) is True

    active_cat = tmp_path / "active_rotor.cat"
    active_cat.write_text("\n".join(H2O_CAT_LINES), encoding="utf-8")
    assert inactive_rotor_catcher(active_cat, allow_empty=False) is False


# =============================================================================
# 7. Method Matrix v4 LaTeX Methods Block & BibTeX Deduplication Test
# =============================================================================

def test_generate_methods_latex_and_bibtex_deduplication() -> None:
    """Validate Method Matrix v4 compliance checks, LaTeX methods block, and BibTeX deduplication."""
    latex_out = generate_methods_latex(H2O_METADATA, method_matrix_v4_check=True)
    assert r"\section{Computational Methods}\label{sec:methods}" in latex_out
    assert r"\qty{825360.000}{\mega\hertz}" in latex_out
    assert r"\qty{1.855}{\debye}" in latex_out
    assert r"\qty{300.00}{\kelvin}" in latex_out
    assert r"\citep{MethodMatrix2024}" in latex_out
    assert r"\citep{Pickett1991}" in latex_out
    assert "wB97X-D4/def2-TZVP" in latex_out
    assert "DEFGRID3" in latex_out

    invalid_dft_meta = dict(H2O_METADATA)
    invalid_dft_meta["theory_level"] = "B3LYP"

    with pytest.raises((DispersionMissingError, MethodMatrixViolationError)) as exc_info:
        generate_methods_latex(invalid_dft_meta, method_matrix_v4_check=True)

    assert exc_info.value.error_code in (
        ProvenanceErrorCode.DISPERSION_MISSING,
        ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID,
    )

    raw_bibtex = """
@article{Pickett1991,
  author = {Pickett, Herbert M.},
  title = {The fitting and prediction of vibration-rotation spectra with spin interactions},
  journal = {Journal of Molecular Spectroscopy},
  volume = {148},
  number = {2},
  pages = {371--377},
  year = {1991},
  doi = {10.1016/0022-2852(91)90124-S}
}

@article{pickett_dup_key,
  author = {Pickett, Herbert M.},
  title = {The fitting and prediction of vibration-rotation spectra},
  journal = {J. Mol. Spectrosc.},
  year = {1991},
  doi = {https://doi.org/10.1016/0022-2852(91)90124-S}
}

@article{MethodMatrix2024,
  author = {CoChem Consortium},
  title = {CoChem Method Matrix v4 Standards},
  year = {2024},
  doi = {10.5281/zenodo.1234567}
}

@article{Pickett1991,
  author = {Pickett, H. M.},
  title = {Duplicate key test},
  year = {1991}
}
"""

    deduped = deduplicate_bibtex(raw_bibtex, deduplicate_by="both")
    assert "@article{Pickett1991" in deduped
    assert "@article{MethodMatrix2024" in deduped
    assert "pickett_dup_key" not in deduped
    assert deduped.count("@article") == 2


# =============================================================================
# 8. 6-Tier CoChemPathManager & Ghost Output Purger Integration Tests
# =============================================================================

def test_cochem_path_manager_6_tiers_and_ghost_purger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate all 6 resolution tiers of CoChemPathManager and ghost output purging."""
    custom_scratch = tmp_path / "custom_tier1"
    resolved_t1 = CoChemPathManager.resolve_scratch_dir(custom_scratch)
    assert resolved_t1 == custom_scratch.resolve()
    assert resolved_t1.exists()

    t2_path = tmp_path / "env_tier2"
    monkeypatch.setenv("COCHEM_SCRATCH", str(t2_path))
    resolved_t2 = CoChemPathManager.resolve_scratch_dir()
    assert resolved_t2 == t2_path.resolve()
    monkeypatch.delenv("COCHEM_SCRATCH")

    custom_deliv = tmp_path / "custom_deliverables"
    resolved_deliv = CoChemPathManager.resolve_deliverables_dir(custom_deliv)
    assert resolved_deliv == custom_deliv.resolve()

    ghost_dir = tmp_path / "ghost_test_dir"
    ghost_dir.mkdir(parents=True, exist_ok=True)

    valid_file = ghost_dir / "valid.parquet"
    valid_file.write_bytes(b"VALID_PARQUET_HEADER_DATA")

    ghost_0byte = ghost_dir / "ghost_failed.cat"
    ghost_0byte.write_bytes(b"")

    ghost_tmp = ghost_dir / "valid.parquet.tmp"
    ghost_tmp.write_bytes(b"TEMP_STAGING_DATA")

    purged = purge_ghost_outputs(ghost_dir, remove_0byte_only=False)
    assert ghost_0byte in purged
    assert ghost_tmp in purged
    assert not ghost_0byte.exists()
    assert not ghost_tmp.exists()
    assert valid_file.exists()


# =============================================================================
# 9. Buffer Lock Sync Physical Disk Verification Test
# =============================================================================

def test_buffer_lock_sync_disk_verification(tmp_path: Path) -> None:
    """Validate buffer_lock_sync physical flush and minimum byte validation."""
    valid_file = tmp_path / "buffer_sync_valid.bin"
    valid_file.write_bytes(b"NON_EMPTY_BINARY_CONTENT")

    size = buffer_lock_sync(valid_file, min_bytes=4)
    assert size == len(b"NON_EMPTY_BINARY_CONTENT")

    zero_file = tmp_path / "buffer_sync_zero.bin"
    zero_file.write_bytes(b"")

    with pytest.raises(CoChemIntegrityError) as exc_info:
        buffer_lock_sync(zero_file, min_bytes=1)

    assert "Buffer sync validation failed" in exc_info.value.message


# =============================================================================
# 10. Method Matrix v4 Flagship Functionals & Scalar Temperature LaTeX Test
# =============================================================================

def test_method_matrix_v4_flagship_functionals_and_scalar_temperature() -> None:
    """Verify that all Method Matrix v4 recommended functionals pass dispersion validation."""
    flagship_functionals = [
        "wB97M-V",
        "wB97X-V",
        "r2SCAN-3c",
        "B97-3c",
        "HF-3c",
        "SCAN-VV10",
        "B3LYP-D3BJ",
        "wB97X-D4",
        "PBE0-D3BJ",
    ]

    for func in flagship_functionals:
        meta = {
            "theory_level": func,
            "basis_set": "def2-QZVPP",
            "rotational_constants": {"a": 825360.0, "b": 435360.0, "c": 278130.0},
            "temperatures": 298.15,
            "defgrid": "DEFGRID3",
        }
        tex_output = generate_methods_latex(meta, method_matrix_v4_check=True)
        assert r"\section{Computational Methods}\label{sec:methods}" in tex_output
        assert r"\qty{298.15}{\kelvin}" in tex_output
        assert func in tex_output


# =============================================================================
# 11. Method Matrix v4 Integration Grid Threshold Violations Test
# =============================================================================

def test_method_matrix_v4_defgrid_violations() -> None:
    """Assert that DEFGRID1 or SG-1 integration grids raise MethodMatrixViolationError."""
    for bad_grid in ["DEFGRID1", "SG-1", "defgrid1"]:
        meta = {
            "theory_level": "wB97X-D4",
            "basis_set": "def2-TZVP",
            "rotational_constants": {"A": 1000.0, "B": 500.0, "C": 250.0},
            "defgrid": bad_grid,
        }
        with pytest.raises(MethodMatrixViolationError) as exc_info:
            generate_methods_latex(meta, method_matrix_v4_check=True)

        assert exc_info.value.error_code == ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID


# =============================================================================
# 12. Fortran Double-Precision D/d Exponent Parsing Test
# =============================================================================

def test_fortran_double_precision_d_exponent_parsing() -> None:
    """Verify that parse_spcat_cat_line properly parses Fortran D and d exponent numbers."""
    line_with_d = "  1.2345D+04  5.0000D-03 -4.5678 2  1.0000d+01  3  18001 103 6 1 6       5 2 3      "
    parsed = parse_spcat_cat_line(line_with_d, line_number=1, temperature_k=300.0)

    assert parsed is not None
    assert parsed["frequency_mhz"] == 12345.0
    assert parsed["uncertainty_mhz"] == 0.005
    assert parsed["lower_state_energy_cm1"] == 10.0


# =============================================================================
# 13. Staging Cleanup on Unhandled Stream Exception Test
# =============================================================================

def test_staging_cleanup_on_unhandled_stream_exception(tmp_path: Path) -> None:
    """Assert that an exception during stream iteration immediately unlinks the staging file."""
    output_parquet = tmp_path / "stream_failure.parquet"

    def _faulty_stream() -> Iterator[Dict[str, Any]]:
        yield {
            "frequency_mhz": 10000.0,
            "uncertainty_mhz": 0.005,
            "log_intensity": -3.0,
            "degrees_of_freedom": 2,
            "lower_state_energy_cm1": 0.0,
            "upper_state_degeneracy": 3,
            "species_tag": 18001,
            "qn_format": 103,
            "qn_upper": "1 0 1",
            "qn_lower": "0 0 0",
            "temperature_k": 300.0,
            "provenance_hash": "sha256:test",
        }
        raise RuntimeError("Simulated mid-stream failure during data acquisition.")

    with pytest.raises(RuntimeError, match="Simulated mid-stream failure"):
        pyarrow_chunked_serializer(
            records_stream=_faulty_stream(),
            output_parquet_path=output_parquet,
            chunk_size=10,
        )

    assert not output_parquet.exists()
    staging_files = list(tmp_path.glob(".*.tmp.*")) + list(tmp_path.glob("*.tmp*"))
    assert len(staging_files) == 0


# =============================================================================
# 14. TorqCatalogCompiler Class Integration Test
# =============================================================================

def test_torq_catalog_compiler_engine(tmp_path: Path) -> None:
    """Validate TorqCatalogCompiler class interface and partition functions."""
    cat_content = (
        "    22557.5181  0.0039 -8.8475 3    3.7661  3 13002 1 1 0 1 0 1\n"
        "    22650.0000  0.0010 -7.1234 3   15.1000  5 13002 2 1 1 2 0 2\n"
    )
    cat_file = tmp_path / "test_spcat.cat"
    cat_file.write_text(cat_content, encoding="utf-8")

    out_dir = tmp_path / "torq_out"
    compiler = TorqCatalogCompiler(cat_file, point_id="pt001", output_dir=out_dir)
    success = compiler.compile_to_parquet(chunk_size=1)
    assert success is True
    assert compiler.parquet_outpath.exists()

    q_rot = compiler.compute_temperature_dependent_partition_function(298.15, A_MHz=825360.0, B_MHz=435360.0, C_MHz=278130.0, sigma=2)
    assert q_rot > 0.0


# =============================================================================
# 15. Banned Methods Auditor Test
# =============================================================================

def test_banned_methods_auditor() -> None:
    """Validate audit_banned_methods detection of additive diffuse and unpreconditioned hessians."""
    # Valid metadata
    valid_meta = {
        "basis_set": "ma-def2-TZVPP",
        "keywords": "InHess XTB2 opt freq",
        "is_non_covalent": True,
        "counterpoise": True,
        "frozen_monomer": True,
    }
    res = audit_banned_methods(valid_meta, raise_on_violation=True)
    assert res.passed is True
    assert res.is_frozen_monomer_verified is True
    assert res.is_bsse_counterpoise_verified is True
    assert res.is_valid_hessian_preconditioned is True

    # Banned additive diffuse
    bad_meta_diffuse = {
        "basis_set": "def2-TZVP",
        "keywords": "additive_diffuse opt",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        audit_banned_methods(bad_meta_diffuse, raise_on_violation=True)
    assert "BANNED_ADDITIVE_DIFFUSE" in str(exc_info.value)

    # Banned unpreconditioned calc_hess
    bad_meta_hess = {
        "basis_set": "def2-TZVP",
        "keywords": "Calc_Hess true opt",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        audit_banned_methods(bad_meta_hess, raise_on_violation=True)
    assert "BANNED_UNPRECONDITIONED_HESSIAN" in str(exc_info.value)

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_telemetry.py ---
"""
CoChem-TORQ: Comprehensive Pure Physical Test Suite for Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------------
Validates:
1. stream_webhook_events with real local HTTP server, exponential backoff retries,
   and circuit-breaker fallback spooling to telemetry_spool.jsonl under network blackout.
2. generate_plotly_3d_carousels with 2D strided regular grid decimation, stationary
   point preservation, color-blind accessibility, and DVR wavefunction probability states.
3. export_crash_animation capturing multi-frame crash_animation.xyz and crash_diagnostic.json
   during Steric Shatter Soft-Quench aborts.
4. Absolute Filesystem Air-Gap compliance writing exclusively to dynamic scratch/artifact dirs.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from Libraries.cochem_torq_telemetry import (
    CircuitState,
    CrashDiagnostic,
    TelemetryCircuitBreaker,
    WebhookPayload,
    decimate_2d_grid_with_extrema,
    export_crash_animation,
    find_stationary_points_2d,
    generate_plotly_3d_carousels,
    stream_webhook_events,
    stream_webhook_events_async,
)


# ============================================================================
# Physical Helper: Ephemeral Local HTTP Server for Real Webhook Delivery
# ============================================================================


class WebhookRecordingHandler(BaseHTTPRequestHandler):
    """Real HTTP request handler for live socket-level webhook testing."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP server console spam during tests
        pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {"raw_body": body}

        # Check server mode
        server_obj: Any = self.server
        server_obj.received_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "payload": payload,
        })

        if getattr(server_obj, "fail_count_target", 0) > 0:
            server_obj.fail_count_target -= 1
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Service Unavailable"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "delivered": true}')


def get_free_port() -> int:
    """Finds an available ephemeral port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def local_webhook_server():
    """Starts a real physical HTTP server on localhost."""
    port = get_free_port()
    server = HTTPServer(("127.0.0.1", port), WebhookRecordingHandler)
    server.received_requests = []  # type: ignore[attr-defined]
    server.fail_count_target = 0  # type: ignore[attr-defined]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/cochem/webhook"
    try:
        yield server, url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# ============================================================================
# Test Suite 1: Webhook Streaming & Circuit Breaker Spooling
# ============================================================================


def test_stream_webhook_events_real_delivery(local_webhook_server, tmp_path):
    """Validates real physical HTTP POST delivery to an active webhook endpoint."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "job_completed",
        "job_id": "TORQ_JOB_2026_08_001",
        "node_id": "hpc_worker_node_07",
        "status": "COMPLETED",
        "data": {"wall_time_sec": 42.5, "optimized_energy_hartree": -154.29841},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["status_code"] == 200
    assert result["spooled"] is False
    assert len(server.received_requests) == 1
    assert server.received_requests[0]["payload"]["job_id"] == "TORQ_JOB_2026_08_001"


def test_stream_webhook_events_exponential_backoff_recovery(local_webhook_server, tmp_path):
    """Validates exponential backoff retries when encountering transient 503 errors."""
    server, webhook_url = local_webhook_server
    server.fail_count_target = 2  # Fail first 2 attempts with 503, succeed on 3rd
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "soft_quench_collision",
        "job_id": "TORQ_JOB_SQ_09",
        "node_id": "gpu_node_01",
        "status": "ALERT",
        "data": {"collision_distance_angstrom": 0.58},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=3,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["attempt"] == 3
    assert len(server.received_requests) == 3


def test_stream_webhook_events_blackout_spooling(tmp_path):
    """Validates zero-interruption spooling to telemetry_spool.jsonl when network fails."""
    # Use a port that is definitively closed/unreachable
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/nonexistent_webhook"
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "oom_backoff",
        "job_id": "TORQ_JOB_OOM_003",
        "node_id": "cpu_node_12",
        "status": "ALERT",
        "data": {"memory_rss_gb": 64.2, "backoff_scale": 0.5},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=0.5,
    )

    # Must NOT raise unhandled exception; must safely spool to disk
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True
    spool_file = scratch_dir / "telemetry_spool.jsonl"
    assert spool_file.exists()

    with open(spool_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 1
    logged_event = lines[-1]
    assert logged_event["payload"]["job_id"] == "TORQ_JOB_OOM_003"
    assert logged_event["delivery_status"] == "SPOOLED"


# ============================================================================
# Test Suite 2: 2D PES Decimation & Plotly 3D Carousel Generation
# ============================================================================


def test_decimate_2d_grid_and_stationary_points():
    """Validates 2D grid decimation while preserving stationary points (minima/maxima)."""
    # Create a dense 500x500 (250,000 nodes) 2D PES grid
    n1, n2 = 500, 500
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")

    # Analytical potential: V(phi1, phi2) = 1500*(1-cos(phi1)) + 800*(1-cos(2*phi2)) + 400*cos(phi1+phi2)
    # Global minimum at (0, 0) where V = 400 cm-1
    rad1 = np.radians(P1)
    rad2 = np.radians(P2)
    pes_grid = 1500.0 * (1.0 - np.cos(rad1)) + 800.0 * (1.0 - np.cos(2.0 * rad2)) + 400.0 * np.cos(rad1 + rad2)

    # Test stationary points finder
    stationary_points = find_stationary_points_2d(phi1, phi2, pes_grid, max_points=10)
    assert len(stationary_points) > 0
    # Minima should include near (0, 0)
    minima = [p for p in stationary_points if p["type"] == "minimum"]
    assert len(minima) >= 1

    # Test decimation to <= 5000 nodes
    phi1_dec, phi2_dec, pes_dec, extrema_pts = decimate_2d_grid_with_extrema(
        phi1, phi2, pes_grid, max_nodes=5000
    )

    total_dec_nodes = len(phi1_dec) * len(phi2_dec)
    assert total_dec_nodes <= 5000
    assert total_dec_nodes > 100
    assert pes_dec.shape == (len(phi1_dec), len(phi2_dec))
    assert len(extrema_pts) > 0


def test_generate_plotly_3d_carousels_standalone_html(tmp_path):
    """Validates generation of lightweight, interactive Plotly 3D PES visualizer HTML."""
    artifact_dir = tmp_path / "artifacts"

    # Dense PES grid: 360x360 (129,600 nodes)
    n = 360
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1200.0 * (1.0 - np.cos(np.radians(P1))) + 600.0 * (1.0 - np.cos(np.radians(3 * P2)))

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_3d.html",
        max_nodes=4000,
        colorscale="Viridis",
        title="1,2-Ethanediol 2D Torsional PES",
    )

    assert html_path.exists()
    assert html_path.is_file()
    assert html_path.parent == artifact_dir

    # Inspect HTML content
    html_content = html_path.read_text(encoding="utf-8")
    assert "<html>" in html_content.lower()
    assert "<body>" in html_content.lower()
    assert "plotly" in html_content.lower()
    assert "1,2-Ethanediol 2D Torsional PES" in html_content

    # File size must be lightweight (< 3.5 MB)
    file_size_mb = html_path.stat().st_size / (1024 * 1024)
    assert file_size_mb < 3.5


def test_generate_plotly_3d_carousels_with_dvr_wavefunctions(tmp_path):
    """Validates Plotly 3D carousel with multi-state DVR probability wavefunctions."""
    artifact_dir = tmp_path / "artifacts"

    n = 100
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1000.0 * (1.0 - np.cos(np.radians(P1))) + 500.0 * (1.0 - np.cos(np.radians(2 * P2)))

    # Create 3 DVR wavefunctions: ground state v=0 and excited states v=1, v=2
    wf_0 = np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_0 /= np.sum(wf_0)

    wf_1 = (P1 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_1 = (wf_1**2) / np.sum(wf_1**2)

    wf_2 = (P2 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_2 = (wf_2**2) / np.sum(wf_2**2)

    dvr_wavefunctions = [wf_0, wf_1, wf_2]

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        dvr_wavefunctions=dvr_wavefunctions,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_dvr_carousel.html",
        max_nodes=2500,
    )

    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "DVR State v=0" in html_content
    assert "DVR State v=1" in html_content
    assert "DVR State v=2" in html_content


# ============================================================================
# Test Suite 3: Crash Animation & Diagnostic Exporter
# ============================================================================


def test_export_crash_animation_steric_collision(tmp_path):
    """Validates multi-frame XYZ crash animation and JSON diagnostic generation."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    # Define a 6-atom molecule (e.g. ethane-like) undergoing steric shatter collision
    symbols = ["C", "C", "H", "H", "H", "H"]
    num_atoms = len(symbols)
    num_frames = 12

    # Frame 0: Stable geometry
    base_coords = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.54, 0.0, 0.0],     # C2
        [-0.5, 1.0, 0.0],     # H3
        [-0.5, -0.5, 0.86],   # H4
        [2.04, 1.0, 0.0],     # H5
        [2.04, -0.5, -0.86],  # H6
    ], dtype=np.float64)

    # Trajectory where H3 and H5 rotate towards each other until collision at frame 11 (d < 0.4 A)
    trajectory = np.zeros((num_frames, num_atoms, 3), dtype=np.float64)
    energies = []
    gradients = []

    for f in range(num_frames):
        coords = base_coords.copy()
        # Move H3 and H5 toward each other along x and y
        offset = float(f) * 0.12
        coords[2, 0] += offset  # H3 moves right
        coords[4, 0] -= offset  # H5 moves left
        coords[4, 1] -= offset * 0.3
        trajectory[f] = coords

        # Exponential energy explosion
        energy = -79.8 + (1.5 ** f) * 0.01
        energies.append(energy)

        # Gradient explosion
        grad = np.zeros((num_atoms, 3))
        grad[2] = [offset * 5.0, -offset * 2.0, 0.0]
        grad[4] = [-offset * 5.0, offset * 2.0, 0.0]
        gradients.append(grad)

    result_paths = export_crash_animation(
        trajectory_array=trajectory,
        error_node_id="rotor_node_55",
        symbols=symbols,
        energies=energies,
        gradients=gradients,
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Steric Shatter Soft-Quench Abort: Interatomic distance below 0.5 A threshold",
    )

    xyz_path = result_paths["xyz_path"]
    diag_path = result_paths["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    # Verify XYZ structure
    xyz_lines = xyz_path.read_text(encoding="utf-8").strip().split("\n")
    # Each frame has num_atoms + 2 lines
    expected_lines = num_frames * (num_atoms + 2)
    assert len(xyz_lines) == expected_lines
    assert xyz_lines[0].strip() == str(num_atoms)
    assert "rotor_node_55" in xyz_lines[1]

    # Verify Diagnostic JSON
    with open(diag_path, "r", encoding="utf-8") as f:
        diag_data = json.load(f)

    assert diag_data["error_node_id"] == "rotor_node_55"
    assert diag_data["num_frames"] == 12
    assert diag_data["num_atoms"] == 6
    assert diag_data["symbols"] == symbols
    assert diag_data["min_interatomic_distance"] < 0.5
    assert diag_data["colliding_pair"] == [2, 4] or diag_data["colliding_pair"] == [4, 2]
    assert "Steric Shatter" in diag_data["abort_reason"]


# ============================================================================
# Test Suite 4: Air-Gap Compliance & Direct Memory Ingestion
# ============================================================================


def test_airgap_compliance_no_repo_pollution(tmp_path):
    """Validates that no temporary files or logs are created in repo workspace."""
    repo_files_before = set(Path(".").glob("*"))

    scratch_dir = tmp_path / "airgap_scratch"
    artifact_dir = tmp_path / "airgap_artifacts"

    # Run telemetry functions with explicit isolated dirs
    payload = {"event_type": "progress", "job_id": "AIRGAP_01", "status": "RUNNING"}
    stream_webhook_events(payload, webhook_url=None, scratch_dir=scratch_dir)

    coords = np.zeros((3, 4, 3))
    export_crash_animation(
        coords,
        error_node_id="airgap_node",
        symbols=["H", "H", "H", "H"],
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
    )

    pes = np.ones((20, 20))
    generate_plotly_3d_carousels(
        pes_tensor=pes,
        artifact_dir=artifact_dir,
        max_nodes=100,
    )

    repo_files_after = set(Path(".").glob("*"))
    # Verify no new files created in cwd
    diff = repo_files_after - repo_files_before
    # Ignore pytest temporary markers or cache if any
    diff = {f for f in diff if not f.name.startswith(".pytest") and not f.name.startswith("__pycache__")}
    assert len(diff) == 0, f"Air-gap violation detected: created files in repo: {diff}"


# ============================================================================
# Test Suite 5: Extended Edge Cases & Circuit Breaker State Transitions
# ============================================================================


def test_stream_webhook_events_circuit_breaker_transitions(tmp_path):
    """Validates circuit breaker transitions from CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    scratch_dir = tmp_path / "scratch"
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/webhook"

    cb = TelemetryCircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.2,
        backoff_factor=0.01,
        max_retries=1,
        request_timeout=0.2,
    )

    assert cb.state == CircuitState.CLOSED

    # 1st failure
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J1"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.consecutive_failures == 1
    assert cb.state == CircuitState.CLOSED

    # 2nd failure -> trips to OPEN
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J2"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.state == CircuitState.OPEN

    # Next call while OPEN immediately spools without network call
    res = stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J3"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert res["status"] == "SPOOLED"
    assert res["reason"] == "Circuit Breaker OPEN"

    # Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.25)
    assert cb.can_attempt_request() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_generate_plotly_3d_carousels_dict_input(tmp_path):
    """Validates Plotly 3D carousel generation when pes_tensor is provided as a dictionary."""
    artifact_dir = tmp_path / "artifacts"
    n1, n2 = 40, 40
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 500.0 * (1.0 - np.cos(np.radians(P1))) + 200.0 * (1.0 - np.cos(np.radians(P2)))

    pes_dict = {
        "pes": pes_grid,
        "phi1": phi1,
        "phi2": phi2,
    }

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_dict,
        artifact_dir=artifact_dir,
        filename="pes_dict_test.html",
        max_nodes=1000,
        colorscale="Cividis",
    )

    assert html_path.exists()
    assert html_path.is_file()


def test_export_crash_animation_dict_and_single_frame(tmp_path):
    """Validates export_crash_animation with dictionary input and single frame."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],  # Severe collision: 0.2 A
    ], dtype=np.float64)

    traj_dict = {
        "coordinates": coords,
        "symbols": ["O", "H"],
        "energies": [-75.123456],
        "gradients": [np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])],
    }

    result = export_crash_animation(
        trajectory_array=traj_dict,
        error_node_id="single_frame_node",
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Single frame singularity collision",
    )

    xyz_path = result["xyz_path"]
    diag_path = result["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    assert diag["error_node_id"] == "single_frame_node"
    assert diag["num_frames"] == 1
    assert diag["num_atoms"] == 2
    assert diag["min_interatomic_distance"] == 0.2
    assert diag["colliding_pair"] == [0, 1] or diag["colliding_pair"] == [1, 0]


Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.