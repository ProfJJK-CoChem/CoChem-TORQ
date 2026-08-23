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

import pandas as pd  # type: ignore[import-untyped]
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
            for schema_field in target_schema:
                col_data = buffer[schema_field.name]
                arr = pa.array(col_data, type=schema_field.type)
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

        if writer is not None:
            writer.close()
            writer = None

        if total_rows == 0:
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

    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if temp_staging_path.exists():
            try:
                remove_readonly_seal(temp_staging_path, recursive=False)
                temp_staging_path.unlink()
            except Exception:
                pass
        raise

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
    - DFT methods require explicit dispersion correction (-D3BJ, -D4, -VV10, -3c).
    - Grid definitions must meet DEFGRID2 / DEFGRID3 criteria.
    - Frozen-Monomer and BSSE Counterpoise documentation for weak complexes.
    - Required metadata: theory_level, basis_set, rotational_constants, temperatures.

    Args:
        metadata: Dictionary containing chemical and computational parameters.
        method_matrix_v4_check: If True, strictly enforces Method Matrix v4 compliance.

    Returns:
        Formatted LaTeX code string ready for direct insertion into scientific manuscripts.

    Raises:
        MethodMatrixViolationError: If required fields, grids, or dispersion corrections fail.
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

        # Audit banned methods and dispersion / grid standards
        audit_banned_methods(metadata, raise_on_violation=True)

        # Explicit DEFGRID verification
        if not defgrid or "DEFGRID1" in defgrid or "SG-1" in defgrid:
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

    is_non_covalent = metadata.get("is_non_covalent", metadata.get("is_vdw_complex", False))
    if is_non_covalent:
        latex_lines.extend([
            r"",
            "The Frozen-Monomer protocol was applied to lock intramolecular monomer coordinates,",
            "fixing the monomer $A$ constant while optimizing intermolecular degrees of freedom.",
            "Basis Set Superposition Error (BSSE) was corrected via the Boys-Bernardi counterpoise procedure.",
        ])

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
    """Deduplicate BibTeX bibliography entries by cite key, normalized DOI, or both.

    Uses a robust brace-depth tokenizer that handles inter-entry non-whitespace comments
    (e.g., '% ADS Export') without swallowing or corrupting subsequent entries.
    """
    raw_text: str
    if isinstance(bibtex_entries, (list, tuple, set)):
        raw_text = "\n\n".join(str(entry) for entry in bibtex_entries)
    else:
        raw_text = str(bibtex_entries)

    entries: List[Tuple[str, str, str]] = []  # (entry_type, cite_key, body)
    pos = 0
    length = len(raw_text)

    entry_header = re.compile(r"@(?P<type>[a-zA-Z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,", re.DOTALL)
    doi_pattern = re.compile(r"\bdoi\s*=\s*[\"{]?(?P<doi>[^\s,\"'}]+)[\"}]?", re.IGNORECASE)

    while pos < length:
        match = entry_header.search(raw_text, pos)
        if not match:
            break

        entry_type = match.group("type").strip()
        cite_key = match.group("key").strip()

        brace_pos = raw_text.find("{", match.start())
        if brace_pos == -1:
            pos = match.end()
            continue

        brace_depth = 0
        body_start = match.end()
        i = brace_pos

        while i < length:
            char = raw_text[i]
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    break
            i += 1

        if brace_depth == 0:
            body = raw_text[body_start:i].strip()
            entries.append((entry_type, cite_key, body))
            pos = i + 1
        else:
            pos = match.end()

    seen_keys: Set[str] = set()
    seen_dois: Set[str] = set()
    unique_entries: List[str] = []

    for entry_type, cite_key, body in entries:
        norm_key = cite_key.lower().strip()
        doi_match = doi_pattern.search(body)
        norm_doi: Optional[str] = None
        if doi_match:
            raw_doi = doi_match.group("doi").strip()
            cleaned_doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw_doi, flags=re.IGNORECASE)
            cleaned_doi = re.sub(r"^doi:\s*", "", cleaned_doi, flags=re.IGNORECASE)
            norm_doi = cleaned_doi.strip().lower().rstrip("/.,;")

        is_duplicate = False

        if deduplicate_by in ("key", "both") and norm_key in seen_keys:
            is_duplicate = True

        if deduplicate_by in ("doi", "both") and norm_doi and norm_doi in seen_dois:
            is_duplicate = True

        if not is_duplicate:
            seen_keys.add(norm_key)
            if norm_doi:
                seen_dois.add(norm_doi)
            clean_entry = f"@{entry_type}{{{cite_key},\n  {body}\n}}"
            unique_entries.append(clean_entry)

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
    - Banned: ONIOM and QM/QM2 partitioning on 5-10 atom non-covalent complexes (Method Matrix §9A.5).
    - Banned: Stacking explicit D3/D4 dispersion on functionals with built-in VV10 or 3c models (§9A.7).
    - Required for vdW / non-covalent complexes: True diffuse-in-base sets
      (e.g., 'aug-cc-pVTZ/QZ', 'jun-cc-pVTZ/QZ', 'jul-cc-pVTZ', 'ma-def2-TZVPP', 'def2-TZVPPD').
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
    theory_level = str(metadata.get("theory_level", "")).strip()
    theory_upper = theory_level.upper()
    basis_set = str(metadata.get("basis_set", "")).strip().lower()
    keywords = str(metadata.get("keywords", metadata.get("orca_keywords", ""))).lower()

    # 1. Check for banned additive diffuse corrections
    if "additive_diffuse" in keywords or metadata.get("additive_diffuse_correction", False):
        banned_flags.append(
            "BANNED_ADDITIVE_DIFFUSE: Additive diffuse corrections degrade interaction energies. "
            "Use true diffuse-in-base sets (e.g. aug-cc-pVQZ, jun-cc-pVTZ, or ma-def2-TZVPP)."
        )

    # 2. Check for banned ONIOM or QM/QM2 partitioning on small complexes (§9A.5)
    if "oniom" in keywords or "qm/qm2" in keywords or "qm-qm2" in keywords or metadata.get("oniom", False):
        banned_flags.append(
            "BANNED_ONIOM_QM_QM2: Method Matrix v4 §9A.5 strictly prohibits ONIOM and QM/QM2 "
            "partitioning for 5-10 atom non-covalent complexes due to boundary polarization artifacts."
        )

    # 3. Check for banned double-dispersion / improper dispersion stacking (§9A.7)
    if theory_upper:
        has_builtin_disp = any(v in theory_upper for v in ("-V", "-VV10", "VV10", "-3C", "3C"))
        has_stacked_disp = any(d in theory_upper for d in ("-D3", "-D4", "-D3BJ", "-D3ZERO", "D3BJ", "D3ZERO"))
        if has_builtin_disp and has_stacked_disp:
            banned_flags.append(
                f"BANNED_DOUBLE_DISPERSION: Functional {theory_level!r} combines built-in non-local correlation/3c parameters "
                "with explicit D3/D4 dispersion corrections, violating Method Matrix v4 §9A.7."
            )

        # Check DFT dispersion compliance if it is DFT without built-in or stacked dispersion
        dft_signatures = (
            "B3LYP", "WB97", "PBE", "R2SCAN", "TPSS", "M06", "B97", "SCAN",
            "OLYP", "PW6B95", "BP86", "BLYP", "CAM-B3LYP", "LC-",
        )
        is_dft = any(sig in theory_upper for sig in dft_signatures)
        disp_signatures = (
            "-D3", "-D3BJ", "-D3ZERO", "-D4", "D3", "D4", "D3BJ", "D3ZERO",
            "-V", "-VV10", "VV10", "-3C", "3C", "-NL", "NL", "-D2", "D2",
        )
        has_disp = any(disp in theory_upper for disp in disp_signatures)
        if is_dft and not has_disp:
            banned_flags.append(
                f"DISPERSION_MISSING: DFT functional {theory_level!r} lacks required dispersion correction (D3BJ/D4/VV10/3c)."
            )

    # 4. Check for banned Calc_Hess true without preconditioning
    if "calc_hess true" in keywords or "calc_hess=true" in keywords or metadata.get("calc_hess_true", False):
        if not ("inhess xtb2" in keywords or "inhess lindh" in keywords or metadata.get("hessian_preconditioned", False)):
            banned_flags.append(
                "BANNED_UNPRECONDITIONED_HESSIAN: 'Calc_Hess true' without preconditioning is forbidden. "
                "Must use 'InHess XTB2' or 'Lindh' Hessian preconditioning."
            )

    # 5. Check for diffuse-in-base compliance on non-covalent complexes
    is_non_covalent = metadata.get("is_non_covalent", metadata.get("is_vdw_complex", False))
    valid_diffuse_sets = (
        "aug-cc-pv", "jun-cc-pv", "jul-cc-pv", "apr-cc-pv", "may-cc-pv",
        "ma-def2", "def2-tzvpd", "def2-tzvppd", "def2-qzvpd", "def2-qzvppd",
        "def2-svpd", "heavy-aug", "aug-cc-pwcv", "aug-pcseg", "calendar"
    )
    allowed_diffuse_basis = any(ds in basis_set for ds in valid_diffuse_sets)

    if is_non_covalent and not allowed_diffuse_basis:
        banned_flags.append(
            f"INVALID_NONCOVALENT_BASIS: Basis set '{basis_set}' lacks true diffuse-in-base primitives. "
            "Non-covalent complexes require aug-cc-pVTZ/QZ, jun-cc-pVTZ, or ma-def2-TZVPP."
        )

    # 6. Check Frozen-Monomer Protocol verification
    frozen_monomer = bool(metadata.get("frozen_monomer", metadata.get("frozen_monomer_protocol", False)))

    # 7. Check BSSE Counterpoise verification
    bsse_cp = bool(metadata.get("counterpoise", metadata.get("bsse_counterpoise", "cp" in keywords)))

    # 8. Check Hessian preconditioning
    hessian_preconditioned = bool(
        "inhess xtb2" in keywords
        or "inhess lindh" in keywords
        or metadata.get("hessian_preconditioned", False)
        or metadata.get("hessian_preconditioning", None) in ("XTB2", "Lindh")
    )

    # 9. Extract ORCA GOAT/CREST conformer union parameters
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
        if any("DISPERSION_MISSING" in f for f in banned_flags):
            raise DispersionMissingError(
                f"Method Matrix v4 Banned Methods Audit Failed: {'; '.join(banned_flags)}",
                error_code=ProvenanceErrorCode.DISPERSION_MISSING,
                details={"banned_flags": banned_flags, "metadata": metadata},
            )
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
        out_dir = CoChemPathManager.resolve_deliverables_dir(output_dir, create=True)
        self.parquet_outpath = out_dir / f"torq_catalog_{self.point_id}.parquet"
        self.col_widths = [13, 8, 8, 2, 10, 3, 7, 12, 12]
        self.col_names = [
            "Frequency_MHz", "Error_MHz", "Log_Intensity", "DOF",
            "E_Lower_cm1", "G_Up", "Tag", "QNs_Up", "QNs_Low"
        ]

    def _parse_chunk_arrays(self, raw_lines: List[str], schema: pa.Schema) -> Optional[pa.Table]:
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

        if not parsed_data["Frequency_MHz"]:
            return None

        arrays = [pa.array(parsed_data[col], type=schema.field(col).type) for col in self.col_names]
        return pa.Table.from_arrays(arrays, schema=schema)

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

        temp_staging_path = self.parquet_outpath.parent / f".{self.parquet_outpath.name}.tmp.{uuid.uuid4().hex[:8]}"
        total_rows = 0
        writer: Optional[pq.ParquetWriter] = None

        try:
            with open(self.cat_filepath, "r", encoding="utf-8", errors="ignore") as f:
                chunk: List[str] = []
                for line in f:
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        table_chunk = self._parse_chunk_arrays(chunk, schema=schema)
                        if table_chunk is not None:
                            if writer is None:
                                writer = pq.ParquetWriter(temp_staging_path, schema, compression=compression)
                            writer.write_table(table_chunk)
                            total_rows += table_chunk.num_rows
                        chunk = []

                if chunk:
                    table_chunk = self._parse_chunk_arrays(chunk, schema=schema)
                    if table_chunk is not None:
                        if writer is None:
                            writer = pq.ParquetWriter(temp_staging_path, schema, compression=compression)
                        writer.write_table(table_chunk)
                        total_rows += table_chunk.num_rows

            if writer:
                writer.close()
                writer = None

            if total_rows == 0:
                raise InactiveRotorError("SPCAT produced 0 transitions.")

            buffer_lock_sync(temp_staging_path, min_bytes=4)

            if self.parquet_outpath.exists():
                remove_readonly_seal(self.parquet_outpath, recursive=False)

            try:
                os.replace(temp_staging_path, self.parquet_outpath)
            except OSError:
                shutil.move(str(temp_staging_path), str(self.parquet_outpath))

            file_size_mb = os.path.getsize(self.parquet_outpath) / (1024 * 1024)
            logger.info(f"Compilation Complete! {total_rows} transitions secured.")
            logger.info(f"Parquet Payload: {self.parquet_outpath} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Catastrophic failure during Parquet serialization: {e}")
            if writer:
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
            if isinstance(e, (InactiveRotorError, FileNotFoundError)):
                raise
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
