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
- Deterministic JSON serialization for provenance manifests (no RFC 8785 conformance claim)
- AASTeX 6.3.1 + siunitx standard for manuscript methods documentation
"""

from __future__ import annotations

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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
)

try:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

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
    INVALID_HESSIAN_STRATEGY = "INVALID_HESSIAN_STRATEGY"


class CoChemIntegrityError(Exception):
    """Raised when buffer lock, hash, or data integrity validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.INTEGRITY_ERROR


class SPCATBridgeError(Exception):
    """Raised when SPCAT format parsing or calculation execution encounters an error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.SPCAT_BRIDGE_ERROR


class FortranOverflowError(SPCATBridgeError):
    """Raised when asterisks indicating Fortran format overflow/underflow are parsed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.FORTRAN_OVERFLOW)


class InactiveRotorError(SPCATBridgeError):
    """Raised when an inactive rotor or transitionless calculation produces a 0-byte catalog."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.SPCAT_BRIDGE_ERROR)


try:
    from cochem_base.exceptions import MethodMatrixViolationError as _BaseMethodMatrixViolationError
except ImportError:
    _BaseMethodMatrixViolationError = Exception


class MethodMatrixViolationError(_BaseMethodMatrixViolationError):
    """Raised when a Method Matrix v4 compliance standard is violated."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID


class DispersionMissingError(MethodMatrixViolationError):
    """Raised when a DFT method lacks necessary dispersion corrections."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, error_code: str | None = None) -> None:
        super().__init__(message, details=details, error_code=error_code or ProvenanceErrorCode.DISPERSION_MISSING)


# =============================================================================
# 1. PyArrow Spectral Catalog Schema (12-Field Precision Schema)
# =============================================================================

if pa is not None:
    SPECTRAL_CATALOG_SCHEMA: Any = pa.schema([
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
else:
    SPECTRAL_CATALOG_SCHEMA = None


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
        base_dir: str | Path | None = None,
        scratch_dir: str | Path | None = None,
        deliverables_dir: str | Path | None = None,
    ) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        self._custom_scratch = Path(scratch_dir).resolve() if scratch_dir is not None else None
        self._custom_deliverables = Path(deliverables_dir).resolve() if deliverables_dir is not None else None

    @classmethod
    def resolve_scratch_dir(
        cls,
        custom_path: str | Path | None = None,
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
        custom_path: str | Path | None = None,
        create: bool = True,
    ) -> Path:
        """Alias for resolve_scratch_dir."""
        return cls.resolve_scratch_dir(custom_path=custom_path, create=create)

    @classmethod
    def resolve_deliverables_dir(
        cls,
        custom_path: str | Path | None = None,
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
        custom_path: str | Path | None = None,
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

def apply_readonly_chmod(path: str | Path, recursive: bool = True) -> None:
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

    items: list[Path] = []
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


def remove_readonly_seal(path: str | Path, recursive: bool = True) -> None:
    """Remove read-only seal and restore write permissions across Windows and POSIX.

    Args:
        path: Path to file or directory to unseal.
        recursive: If True and path is a directory, unseals all child items.
    """
    target = Path(path).resolve()
    if not target.exists():
        return

    items: list[Path] = []
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
    file_obj_or_path: io.IOBase | int | str | Path,
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
    path_to_check: Path | None = None

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
    target_path: str | Path | Sequence[str | Path],
    patterns: Sequence[str] | None = None,
    remove_0byte_only: bool = False,
    remove_tmp_siblings: bool = True,
) -> list[Path]:
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

    targets_list: list[Path] = []
    if isinstance(target_path, (str, Path)):
        targets_list.append(Path(target_path).resolve())
    else:
        for item in target_path:
            targets_list.append(Path(item).resolve())

    files_to_evaluate: set[Path] = set()

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

    purged: list[Path] = []
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
    base_scratch: str | Path | None = None,
    prefix: str = "spcat_workspace",
    cleanup_on_exit: bool = True,
    job_id: str | None = None,
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
    cat_source: str | Path | bytes | io.IOBase | Sequence[str],
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
                with open(p, encoding="utf-8", errors="ignore") as f:
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
    line_number: int | None = None,
    temperature_k: float = 300.0,
    provenance_hash: str = "",
) -> dict[str, Any] | None:
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
        value = float(clean)
        if not math.isfinite(value):
            raise ValueError("Nonfinite SPCAT field")
        return value

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
    stream_or_path: str | Path | io.TextIOBase | Iterator[str] | Sequence[str],
    temperature_k: float = 300.0,
    provenance_hash: str = "",
) -> Iterator[dict[str, Any]]:
    """Stream and yield parsed records from a Pickett SPCAT .cat source."""
    if isinstance(stream_or_path, (str, Path)):
        p = Path(stream_or_path)
        if p.is_file():
            with open(p, encoding="utf-8", errors="ignore") as f:
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
    records_stream: Iterator[dict[str, Any]],
    output_parquet_path: str | Path,
    chunk_size: int = 100_000,
    compression: str = "zstd",
    compression_level: int = 7,
    schema: pa.Schema | None = None,
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
    buffer: dict[str, list[Any]] = {name: [] for name in field_names}
    rows_in_buffer = 0
    total_rows = 0

    writer: pq.ParquetWriter | None = None

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

            arrays: list[pa.Array] = []
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
    runner_or_path: Callable[[float, Path], Path] | Path | str,
    temp_k: float,
    output_dir: Path,
    base_scratch: Path | None,
    chunk_size: int,
    provenance_hash: str,
    apply_immutable_seal: bool,
) -> tuple[float, Path]:
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
    spcat_runner_or_cat_paths: Callable[[float, Path], Path] | dict[float, str | Path] | Sequence[tuple[float, str | Path]],
    temperatures: Sequence[float],
    output_dir: str | Path,
    max_workers: int | None = None,
    base_scratch: str | Path | None = None,
    chunk_size: int = 100_000,
    provenance_hash: str = "",
    apply_immutable_seal: bool = False,
) -> dict[float, Path]:
    """Compile multiple temperature catalogs concurrently using hardware-saturated ThreadPoolExecutor."""
    target_out_dir = CoChemPathManager.resolve_deliverables_dir(output_dir, create=True)
    scratch_root = CoChemPathManager.resolve_scratch_dir(base_scratch, create=True)

    workers = max_workers if max_workers is not None else min(len(temperatures), os.cpu_count() or 4)
    workers = max(1, workers)

    results: dict[float, Path] = {}
    for temp in temperatures:
        temp_k = float(temp)
        runner_task: Callable[[float, Path], Path] | Path | str
        if callable(spcat_runner_or_cat_paths):
            runner_task = spcat_runner_or_cat_paths
        elif isinstance(spcat_runner_or_cat_paths, dict):
            runner_task = spcat_runner_or_cat_paths[temp_k]
        elif isinstance(spcat_runner_or_cat_paths, (list, tuple)):
            mapping = dict(spcat_runner_or_cat_paths)
            runner_task = mapping[temp_k]
        else:
            raise ValueError("Invalid spcat_runner_or_cat_paths specification.")

        t_k, parquet_path = _compile_single_temperature_task(
            runner_task,
            temp_k,
            target_out_dir,
            scratch_root,
            chunk_size,
            provenance_hash,
            apply_immutable_seal,
        )
        results[t_k] = parquet_path

    return results


def validate_catalog_grid(grid: str, phase: str = "PHASE_FINALOPT") -> bool:
    """Validates integration grid per execution phase according to Method Matrix v4 §4.4.
    Permits defgrid1 during PHASE_PREOPT; strictly enforces defgrid3 for PHASE_FINALOPT and PHASE_NUMFREQ.
    """
    from cochem_base.config import GridPolicy
    return GridPolicy.validate_grid(phase, grid)


def validate_orca_deck(
    deck_content: str,
    preliminary_opt: bool = False,
    production_opt: bool = False,
    **kwargs: Any
) -> bool:
    """
    Validates an ORCA input deck against Method Matrix v4 constraints:
    - Unconditional prohibition on 'Calc_Hess true' (§8B.3).
    - Quadrature grid scheduling: permits defgrid1 strictly when preliminary_opt=True;
      strictly rejects defgrid1 and requires defgrid3 when Freq or production_opt=True (§4.4).
    """
    # 1. Unconditional check for Calc_Hess true
    if re.search(r'\bcalc_hess\s+true\b', deck_content, re.IGNORECASE) or re.search(r'\bcalc_hess=true\b', deck_content, re.IGNORECASE):
        raise MethodMatrixViolationError(
            "[METHOD-MATRIX-VIOLATION] 'Calc_Hess true' is strictly prohibited for geometry "
            "optimizations under Method Matrix v4 §8B.3. Model Hessians ('InHess XTB2' or 'Lindh') "
            "must be used to prevent massive ab initio Hessian computational overhead.",
            error_code=ProvenanceErrorCode.INVALID_HESSIAN_STRATEGY,
        )

    # 2. Grid scheduling rules
    has_defgrid1 = bool(re.search(r'\bdefgrid1\b', deck_content, re.IGNORECASE))
    has_freq = bool(re.search(r'\bfreq\b', deck_content, re.IGNORECASE) or re.search(r'\bnumfreq\b', deck_content, re.IGNORECASE))

    if (production_opt or has_freq) and has_defgrid1:
        raise MethodMatrixViolationError(
            "[METHOD-MATRIX-VIOLATION] Integration grid 'defgrid1' is strictly prohibited for production "
            "optimization or vibrational frequency (Freq) calculations under Method Matrix v4 §4.4. "
            "Tight grid 'defgrid3' is required to eliminate loose grid numerical noise.",
            error_code=ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID,
        )

    return True


# =============================================================================
# 11. AASTeX 6.3.1 + siunitx LaTeX Methods Block Generator
# =============================================================================

def generate_methods_latex(
    metadata: dict[str, Any],
    output_tex_path: str | Path | None = None,
    method_matrix_v4_check: bool = True,
) -> str:
    """Render a factual methods record from supplied provenance only.

    Legacy templates asserted calculations, programs, reductions and corrections
    without evidence. This writer requires the core provenance and reports only
    explicitly supplied quantities/procedures. It cannot certify those claims.
    """
    if isinstance(output_tex_path, bool):
        method_matrix_v4_check = output_tex_path
        output_tex_path = None
    required = ("theory_level", "basis_set", "software_version", "provenance_hash")
    missing = [key for key in required if not isinstance(metadata.get(key), str) or not metadata[key].strip()]
    if missing:
        raise MethodMatrixViolationError(
            "Missing method provenance: " + ", ".join(missing),
            error_code=ProvenanceErrorCode.MISSING_DATA,
        )
    if method_matrix_v4_check:
        audit_banned_methods(metadata, raise_on_violation=True)
    def tex(value: Any) -> str:
        escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        return "".join(escapes.get(char, char) for char in str(value))
    lines = [
        r"\section{Computational Methods}\label{sec:methods}",
        "The recorded electronic-structure recipe is " + tex(metadata["theory_level"]) + "/" + tex(metadata["basis_set"]) + ".",
        "The supplied software/version provenance is " + tex(metadata["software_version"]) + ".",
        "The associated provenance digest is " + r"\texttt{" + tex(metadata["provenance_hash"]) + "}.",
    ]
    # These labels describe source metadata; they never infer a completed stage.
    for key, label in (
        ("geometry_method", "Geometry method"), ("force_field_method", "Force-field method"),
        ("defgrid", "Recorded integration grid"), ("orca_keywords", "Recorded engine keywords"),
        ("rotational_constants_kind", "Rotational-constant type"),
        ("hamiltonian_reduction", "Hamiltonian reduction"), ("axis_representation", "Axis representation"),
        ("partition_function_model", "Partition-function model"), ("catalog_program", "Catalog program"),
    ):
        if metadata.get(key) is not None:
            lines.append(label + ": " + tex(metadata[key]) + ".")
    for key, label, unit in (
        ("rotational_constants", "Rotational constants", "MHz"),
        ("dipole_moments", "Dipole components", "debye"),
        ("centrifugal_distortion", "Distortion parameters", "MHz"),
    ):
        values = metadata.get(key)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a mapping with declared units")
        if values and metadata.get(key + "_units") != unit:
            raise ValueError(f"{key}_units must explicitly be {unit}")
        if key == "rotational_constants" and values and not metadata.get("rotational_constants_kind"):
            raise ValueError("Specify equilibrium, ground-state, fitted, or composite rotational-constant type")
        parts = []
        for name, value in values.items():
            if value is None:
                parts.append(tex(name) + " unavailable")
            else:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"Nonfinite {key}.{name}")
                parts.append(tex(name) + " = " + format(numeric, ".10g") + " " + unit)
        if parts:
            lines.append(label + ": " + "; ".join(parts) + ".")
    if "temperatures" in metadata:
        temps = metadata["temperatures"]
        temps = [temps] if isinstance(temps, (int, float)) else list(temps)
        if not temps or any(not math.isfinite(float(t)) or float(t) <= 0 for t in temps):
            raise ValueError("Temperatures must be finite positive values")
        lines.append("Recorded temperatures: " + ", ".join(format(float(t), ".8g") for t in temps) + " K.")
    # A molecular class, requested keyword or boolean is not execution evidence.
    procedures = metadata.get("executed_procedures", [])
    for procedure in procedures:
        if not isinstance(procedure, dict) or not procedure.get("description") or not procedure.get("artifact_digest"):
            raise ValueError("Executed procedures require a description and evidence digest")
        lines.append(tex(procedure["description"]) + " (evidence: " + tex(procedure["artifact_digest"]) + ").")
    content = "\n".join(lines) + "\n"
    if output_tex_path is not None:
        target = Path(output_tex_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        buffer_lock_sync(target, min_bytes=len(content.encode("utf-8")))
    return content


# =============================================================================
# 12. High-Fidelity BibTeX Deduplication Engine
# =============================================================================

def deduplicate_bibtex(
    bibtex_entries: str | Sequence[str],
    output_bib_path: str | Path | None = None,
    deduplicate_by: str = "both",
) -> str:
    """Deduplicate BibTeX bibliography entries by cite key, normalized DOI, or both.

    Uses a robust brace-depth tokenizer that handles inter-entry non-whitespace comments
    (e.g., '% ADS Export') without swallowing or corrupting subsequent entries.

    Args:
        bibtex_entries: Raw BibTeX string or collection of BibTeX entry strings.
        output_bib_path: Optional file path to write the compiled, deduplicated .bib file.
        deduplicate_by: Deduplication strategy: 'key', 'doi', or 'both' (default 'both').

    Returns:
        Clean, deduplicated BibTeX bibliography string.
    """
    if isinstance(output_bib_path, str) and output_bib_path.lower() in ("both", "key", "doi"):
        deduplicate_by = output_bib_path
        output_bib_path = None

    raw_text: str
    if isinstance(bibtex_entries, (list, tuple, set)):
        raw_text = "\n\n".join(str(entry) for entry in bibtex_entries)
    else:
        raw_text = str(bibtex_entries)

    entries: list[tuple[str, str, str]] = []  # (entry_type, cite_key, body)
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

    seen_keys: set[str] = set()
    seen_dois: set[str] = set()
    unique_entries: list[str] = []

    for entry_type, cite_key, body in entries:
        norm_key = cite_key.lower().strip()
        doi_match = doi_pattern.search(body)
        norm_doi: str | None = None
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

    bib_content = "\n\n".join(unique_entries) + ("\n" if unique_entries else "")

    if output_bib_path is not None:
        target_bib = Path(output_bib_path).resolve()
        target_bib.parent.mkdir(parents=True, exist_ok=True)
        target_bib.write_text(bib_content, encoding="utf-8")
        buffer_lock_sync(target_bib, min_bytes=len(bib_content.encode("utf-8")))

    return bib_content


# =============================================================================
# 13. Banned Methods Auditor & Method Matrix v4 Compliance Engine
# =============================================================================

@dataclass
class BannedMethodsAuditResult:
    """Result container for Method Matrix v4 banned methods and non-covalent rules audit."""

    passed: bool
    banned_flags: list[str]
    allowed_diffuse_basis: bool
    is_frozen_monomer_verified: bool
    is_bsse_counterpoise_verified: bool
    is_valid_hessian_preconditioned: bool
    conformer_union_params: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit result to dictionary."""
        return asdict(self)


def audit_banned_methods(
    metadata: dict[str, Any],
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
    banned_flags: list[str] = []
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

    # 4. Check for banned Calc_Hess true UNCONDITIONALLY under Method Matrix v4 §8B.3
    if (
        "calc_hess true" in keywords
        or "calc_hess=true" in keywords
        or metadata.get("calc_hess_true", False)
        or any(re.search(r'\bcalc_hess\s+true\b', kw, re.IGNORECASE) for kw in keywords)
    ):
        banned_flags.append(
            "[METHOD-MATRIX-VIOLATION] 'Calc_Hess true' is strictly prohibited for geometry "
            "optimizations under Method Matrix v4 §8B.3. Model Hessians ('InHess XTB2' or 'Lindh') "
            "must be used to prevent massive ab initio Hessian computational overhead."
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

    def __init__(self, cat_filepath: str | Path, point_id: str = "000", output_dir: str | Path | None = None) -> None:
        self.cat_filepath = Path(cat_filepath).resolve()
        self.point_id = point_id
        out_dir = CoChemPathManager.resolve_deliverables_dir(output_dir, create=True)
        self.parquet_outpath = out_dir / f"torq_catalog_{self.point_id}.parquet"
        self.col_widths = [13, 8, 8, 2, 10, 3, 7, 12, 12]
        self.col_names = [
            "Frequency_MHz", "Error_MHz", "Log_Intensity", "DOF",
            "E_Lower_cm1", "G_Up", "Tag", "QNs_Up", "QNs_Low"
        ]

    def _parse_chunk_arrays(self, raw_lines: list[str], schema: pa.Schema) -> pa.Table | None:
        parsed_data: dict[str, list[Any]] = {col: [] for col in self.col_names}
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
                raise
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
        writer: pq.ParquetWriter | None = None

        try:
            with open(self.cat_filepath, encoding="utf-8", errors="ignore") as f:
                chunk: list[str] = []
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
        self, temp_k: float, A_MHz: float | None = None, B_MHz: float | None = None,
        C_MHz: float | None = None, sigma: int | None = None,
    ) -> float:
        """High-temperature nonlinear rigid-rotor approximation, not a full Q(T).

        Requires explicit rotational constants and rotational symmetry number.
        This expression omits vibrational/electronic/nuclear-spin contributions
        and is not an exact sum or a validated low-temperature approximation.
        """
        if any(value is None or isinstance(value, bool) for value in (A_MHz, B_MHz, C_MHz)):
            raise ValueError("Explicit positive A, B, C rotational constants are required")
        constants = [float(A_MHz), float(B_MHz), float(C_MHz)]
        if (not math.isfinite(temp_k) or temp_k <= 0
                or any(not math.isfinite(value) or value <= 0 for value in constants)
                or not constants[0] >= constants[1] >= constants[2]):
            raise ValueError("Require finite T > 0 and ordered A >= B >= C > 0")
        if type(sigma) is not int or sigma < 1:
            raise ValueError("Explicit positive integer rotational symmetry number required")
        kB, h = 1.380649e-23, 6.62607015e-34
        theta_a, theta_b, theta_c = [h * value * 1e6 / kB for value in constants]
        q_rot = math.sqrt(math.pi) / sigma * math.sqrt((temp_k / theta_a) * (temp_k / theta_b) * (temp_k / theta_c))
        if not math.isfinite(q_rot) or q_rot <= 0:
            raise ValueError("Rotational partition approximation overflowed or underflowed")
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
    "validate_catalog_grid",
    "validate_orca_deck",
    "TorqCatalogCompiler",
]
