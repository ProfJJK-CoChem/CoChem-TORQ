#!/usr/bin/env python3
"""Pre-flight Cross-Platform Python-Based Repository MIME and Entropy Air-Gap Inspection in CI/CD.

Mandated by SRS Doc 2 Part 1 (§1.1) for pre-flight cross-platform Python-based
repository MIME and entropy air-gap inspection in CI/CD pipelines.

Architectural Context & Invariants:
-----------------------------------
1. Method Matrix v4 Alignment (Sections 8A.5, 8C.1-8C.3):
   Guarantees that the Static Execution Tier (immutable Git tree) is strictly
   decoupled from dynamic runtime datasets (~/CoChem_Artifacts or $SCRATCH).
2. Zero-Compute on Cloud CI Constraint:
   Prevents runtime state pollution, proprietary binary contamination, and heavy
   electronic structure output from leaking into Git history.
3. Multi-OS Cross-Platform Execution:
   Platform-abstracted inspection using 'git ls-files -z' with a robust pure-Python
   recursive directory walk fallback. Avoids host shell 'find' / 'FIND.EXE' collisions.
4. Comprehensive MIME, Magic Number & Shannon Entropy Analysis:
   - Identifies disguised binary payloads (.h5, .npy, .sqlite, .parquet, ELF, PE, zip).
   - Computes Shannon entropy (H in bits/byte) to catch encrypted/compressed blobs.
   - Detects raw atomic coordinates (.xyz format) and QM log output signatures.
   - Audits system configuration files for localized user path leaks and active job pollution.

Provenance Tags:
- [M] Measured Shannon entropy bits/byte and wall-clock execution metrics.
- [D] Derived MIME byte signatures and blocked format specifications.
- [E] Estimated resource limits and entropy thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# Blocked extensions that violate the Tripartite Workspace Air-Gap [D]
FORBIDDEN_EXTENSIONS: FrozenSet[str] = frozenset({
    # Quantum chemistry calculation data & orbitals
    ".h5",
    ".hdf5",
    ".gbw",
    ".xyz",
    ".chk",
    ".opt",
    ".hess",
    ".mol",
    ".smi",
    ".pdb",
    ".cif",
    ".molden",
    ".fchk",
    ".wfn",
    ".wfx",
    # Binary array & database containers
    ".npy",
    ".npz",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".arrow",
    ".feather",
    # Ephemeral runtime logs & state files
    ".lock",
    ".log",
    ".out",
    ".err",
    ".tmp",
    ".parsl",
})

# Forbidden exact filenames & mutable registries [D]
FORBIDDEN_EXACT_FILENAMES: FrozenSet[str] = frozenset({
    "cochem_audit_log.json",
    "topos_runtime_state.json",
})

# Forbidden directory patterns that must never be committed to static git tree [D]
FORBIDDEN_DIRECTORY_PATTERNS: FrozenSet[str] = frozenset({
    "cochem_artifacts",
    "runinfo",
    "cochem_exec_",
    "scratch_ram",
})

# Directories excluded from filesystem scans by default [D]
DEFAULT_EXCLUDED_DIRS: FrozenSet[str] = frozenset({
    ".git",
    ".venv",
    ".conda",
    ".trash",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".tox",
    "dist",
    "build",
    ".eggs",
    ".agents",
})

# Suffix patterns for excluded directories (e.g. *.egg-info, *.conda_trash) [D]
EXCLUDED_DIR_SUFFIXES: Tuple[str, ...] = (
    ".egg-info",
    ".conda_trash",
)

# Files excluded from scanning (e.g. coverage db or static amnesty) [D]
DEFAULT_EXCLUDED_FILES: FrozenSet[str] = frozenset({
    ".coverage",
    ".anti_spoof_amnesty.json",
})

# File suffix patterns to ignore (e.g. *.conda_trash files) [D]
EXCLUDED_FILE_SUFFIXES: Tuple[str, ...] = (
    ".conda_trash",
)

# Legitimate media, font, and document extensions exempt from Shannon entropy violation [D]
EXEMPT_ENTROPY_EXTENSIONS: FrozenSet[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".pdf",
    ".zip",
    ".gz",
})

# Recognized source, config, and documentation extensions exempt from QM log / XYZ text heuristics [D]
RECOGNIZED_CODE_DOC_EXTENSIONS: FrozenSet[str] = frozenset({
    ".py",
    ".pyi",
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".rst",
    ".ipynb",
    ".sh",
    ".bash",
    ".zsh",
    ".slurm",
    ".bat",
    ".ps1",
    ".cmd",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".cjs",
    ".mjs",
    ".xml",
    ".svg",
})

# Magic Number Byte Signatures for Disguised Binary Detection [D]
MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89HDF\r\n\x1a\n", "HDF5 Container (.h5 / .hdf5)"),
    (b"SQLite format 3\x00", "SQLite Database (.db / .sqlite)"),
    (b"\x93NUMPY", "NumPy Binary Array (.npy)"),
    (b"PK\x03\x04", "ZIP / Binary Archive (.zip / .jar)"),
    (b"PAR1", "Apache Parquet File (.parquet)"),
    (b"\x7fELF", "Linux ELF Executable Binary"),
    (b"MZ", "Windows PE Executable Binary"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip Archive (.7z)"),
    (b"\x1f\x8b", "GZIP Compressed Stream (.gz)"),
    (b"BZh", "BZIP2 Compressed Stream (.bz2)"),
    (b"\xfd7zXZ\x00", "XZ Compressed Stream (.xz)"),
    (b"ARROW1", "Apache Arrow Stream"),
)

# Quantum Chemistry Simulation Log Output Signatures [D]
QM_LOG_PATTERNS: Tuple[Tuple[bytes, str], ...] = (
    (b"* O R C A *", "ORCA calculation banner"),
    (b"FINAL SINGLE POINT ENERGY", "ORCA/Gaussian energy convergence record"),
    (b"ORCA TERMINATED NORMALLY", "ORCA completion log"),
    (b"TOTAL RUN TIME:", "Ab initio total run time accounting"),
    (b"Gaussian, Inc.", "Gaussian calculation banner"),
    (b"NWChem", "NWChem calculation banner"),
    (b"Q-Chem", "Q-Chem calculation banner"),
    (b"TURBOMOLE", "TURBOMOLE calculation banner"),
    (b"PSI4: An Open-Source Ab Initio Electronic Structure Package", "Psi4 calculation banner"),
    (b"ACES2 / CFOUR", "CFOUR calculation banner"),
    (b"-- CFOUR --", "CFOUR calculation banner"),
    (b"CFOUR Executable", "CFOUR execution log"),
    (b"SLURM_JOB_ID=", "HPC Slurm job environment dump"),
    (b"SLURM_JOB_ID:", "HPC Slurm job environment dump"),
)

# Localized Host Path Leak Substrings [D]
HOST_PATH_LEAKS: Tuple[str, ...] = (
    "/users/",
    "/home/",
    "c:\\users",
    "c:/users",
    "c:\\\\users",
    "d:\\__cochem",
    "d:/__cochem",
    "d:\\\\__cochem",
)

# Default Shannon Entropy Threshold (bits per byte) [E]
DEFAULT_ENTROPY_THRESHOLD: float = 7.8


@dataclass(frozen=True)
class AirgapViolation:
    """Represents a single detected air-gap security violation [D]."""

    file_path: Path
    relative_path: str
    violation_type: str
    detail: str
    severity: str = "ERROR"
    entropy: Optional[float] = None
    matched_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to serializable dictionary."""
        return {
            "file_path": str(self.file_path),
            "relative_path": self.relative_path,
            "violation_type": self.violation_type,
            "detail": self.detail,
            "severity": self.severity,
            "entropy": self.entropy,
            "matched_pattern": self.matched_pattern,
        }


@dataclass
class AirgapSweepSummary:
    """Aggregated summary of repository air-gap sweep [M]."""

    repo_root: Path
    is_clean: bool
    scanned_files_count: int = 0
    scanned_directories_count: int = 0
    violations: List[AirgapViolation] = field(default_factory=list)
    scan_method: str = "unknown"
    execution_time_seconds: float = 0.0
    max_entropy_observed: float = 0.0
    high_entropy_files_count: int = 0

    @property
    def violation_count(self) -> int:
        """Return total violation count."""
        return len(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to serializable dictionary."""
        return {
            "repo_root": str(self.repo_root),
            "is_clean": self.is_clean,
            "violation_count": self.violation_count,
            "scanned_files_count": self.scanned_files_count,
            "scanned_directories_count": self.scanned_directories_count,
            "scan_method": self.scan_method,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "max_entropy_observed": round(self.max_entropy_observed, 4),
            "high_entropy_files_count": self.high_entropy_files_count,
            "violations": [v.to_dict() for v in self.violations],
        }


def calculate_shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of a byte sequence in bits per byte [M].

    Shannon entropy: H = -sum(p(x) * log2(p(x))) for all byte values present.
    Theoretical range: 0.0 (uniform single byte) to 8.0 (completely random / compressed).

    Args:
        data: Raw byte sequence to evaluate.

    Returns:
        Entropy value in bits per byte in the range [0.0, 8.0].
    """
    if not data:
        return 0.0
    length = len(data)
    counts = Counter(data)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def is_restricted_extension(
    file_path: Path,
    blocked_extensions: AbstractSet[str],
) -> Tuple[bool, str]:
    """Check whether a file matches any blocked extension (case-insensitively) [D].

    Handles standard extensions, multi-part extensions (e.g. .h5.bak),
    case variations (.H5, .XYZ), and dotfiles (e.g. .tmp, .log).

    Args:
        file_path: The file path to inspect.
        blocked_extensions: Set of lowercased blocked extensions with leading dots.

    Returns:
        A tuple of (is_blocked, matched_extension).
    """
    name_lower = file_path.name.lower()
    if name_lower in blocked_extensions:
        return True, name_lower

    suffix = file_path.suffix.lower()
    if suffix in blocked_extensions:
        return True, suffix

    for s in file_path.suffixes:
        s_lower = s.lower()
        if s_lower in blocked_extensions:
            return True, s_lower

    parts = name_lower.split(".")
    for part in parts[1:]:
        ext = f".{part}"
        if ext in blocked_extensions:
            return True, ext

    return False, ""


def is_restricted_directory(
    dir_name: str,
    blocked_patterns: AbstractSet[str],
) -> Tuple[bool, str]:
    """Check whether a directory name matches any restricted directory pattern [D].

    Args:
        dir_name: The directory name to inspect.
        blocked_patterns: Set of lowercased blocked directory name patterns.

    Returns:
        A tuple of (is_restricted, matched_pattern).
    """
    dir_lower = dir_name.lower().strip("/\\")
    for pat in blocked_patterns:
        pat_lower = pat.lower()
        if dir_lower == pat_lower or dir_lower.startswith(pat_lower) or dir_lower.endswith(pat_lower):
            return True, pat
    return False, ""


def is_excluded_path_segment(path_str: str) -> bool:
    """Check if any segment of a path belongs to an excluded directory pattern [D]."""
    parts = Path(path_str).parts
    for part in parts:
        part_lower = part.lower()
        if part_lower in DEFAULT_EXCLUDED_DIRS:
            return True
        if any(part_lower.endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES):
            return True
    return False


def inspect_magic_number(header: bytes) -> Optional[str]:
    """Inspect the binary header of a file against known magic signatures [D].

    Args:
        header: The first 8192 bytes of the file.

    Returns:
        Description of detected binary signature, or None if benign.
    """
    for magic, desc in MAGIC_SIGNATURES:
        if header.startswith(magic):
            return desc
    return None


def check_qm_log_signatures(header: bytes) -> Optional[str]:
    """Inspect file content for quantum chemistry calculation output patterns [D].

    Args:
        header: The first 8192 bytes of the file.

    Returns:
        Decoded string of matched QM pattern description if detected, or None.
    """
    for qm_pat, desc in QM_LOG_PATTERNS:
        if qm_pat in header:
            return f"{desc} ({qm_pat.decode('utf-8', errors='ignore')})"
    return None


def is_xyz_coordinate_payload(data: bytes) -> bool:
    """Inspect text payload to detect standard XYZ molecular coordinates format [D].

    Standard XYZ format consists of:
    Line 1: Number of atoms N (positive integer)
    Line 2: Comment / Title line
    Lines 3..N+2: Atomic symbol followed by 3 floating-point Cartesian coordinates.

    Args:
        data: Byte payload to inspect.

    Returns:
        True if valid XYZ format is detected, False otherwise.
    """
    try:
        text = data.decode("utf-8", errors="ignore").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False

        atom_count = int(lines[0])
        if atom_count <= 0 or atom_count > 100000:
            return False

        valid_coord_lines = 0
        sample_limit = min(atom_count, 10)
        for line in lines[2 : 2 + sample_limit]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    valid_coord_lines += 1
                except ValueError:
                    continue

        return valid_coord_lines >= min(atom_count, 3)
    except Exception:
        return False


def check_config_pollution(file_path: Path) -> List[str]:
    """Check configuration JSON files for active execution state or localized path leaks [D].

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        List of identified issue descriptions.
    """
    issues: List[str] = []
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)

        if isinstance(data, dict):
            if "active_jobs" in data and bool(data["active_jobs"]):
                issues.append("Active jobs present in configuration file (must be empty in static tier)")

        json_str = json.dumps(data).lower()
        for leak in HOST_PATH_LEAKS:
            if leak in json_str:
                issues.append(f"Localized system path leak detected: '{leak}'")
                break
    except Exception as exc:
        issues.append(f"Configuration parse error: {exc}")
    return issues


def inspect_file_airgap(
    full_path: Path,
    repo_root: Path,
    blocked_exts: AbstractSet[str],
    entropy_threshold: float,
) -> Tuple[List[AirgapViolation], float]:
    """Perform deep air-gap inspection on a single file [M][D].

    Args:
        full_path: Absolute path to the file.
        repo_root: Absolute path to the repository root.
        blocked_exts: Set of blocked file extensions.
        entropy_threshold: Threshold for flagging high Shannon entropy.

    Returns:
        A tuple of (list of AirgapViolation, observed_entropy).
    """
    violations: List[AirgapViolation] = []
    try:
        rel_path = full_path.relative_to(repo_root).as_posix()
    except ValueError:
        rel_path = str(full_path).replace("\\", "/")

    lower_name = full_path.name.lower()
    suffix = full_path.suffix.lower()

    # 1. Exact forbidden filename check (e.g. cochem_audit_log.json)
    if lower_name in FORBIDDEN_EXACT_FILENAMES:
        violations.append(
            AirgapViolation(
                file_path=full_path,
                relative_path=rel_path,
                violation_type="MUTABLE_REGISTRY_POLLUTION",
                detail=f"Mutable runtime registry '{lower_name}' must not be committed to static repository tier",
                matched_pattern=lower_name,
            )
        )

    # 2. Blocked extension check
    blocked, matched_ext = is_restricted_extension(full_path, blocked_exts)
    if blocked:
        violations.append(
            AirgapViolation(
                file_path=full_path,
                relative_path=rel_path,
                violation_type="FORBIDDEN_EXTENSION",
                detail=f"File matches forbidden runtime extension '{matched_ext}'",
                matched_pattern=matched_ext,
            )
        )
        return violations, 0.0

    # 3. Localized config pollution inspection
    if lower_name == "cochem_system_config.json":
        config_issues = check_config_pollution(full_path)
        for issue in config_issues:
            violations.append(
                AirgapViolation(
                    file_path=full_path,
                    relative_path=rel_path,
                    violation_type="CONFIG_POLLUTION",
                    detail=issue,
                )
            )

    # Read binary header and full sample for deep inspection
    try:
        with open(full_path, "rb") as fh:
            header = fh.read(8192)
            fh.seek(0)
            full_sample = fh.read(65536)
    except (OSError, PermissionError):
        return violations, 0.0

    # 4. Magic-number / MIME signature inspection
    magic_desc = inspect_magic_number(header)
    if magic_desc:
        violations.append(
            AirgapViolation(
                file_path=full_path,
                relative_path=rel_path,
                violation_type="MAGIC_NUMBER_VIOLATION",
                detail=f"Disguised binary header detected: {magic_desc}",
                matched_pattern=magic_desc,
            )
        )

    # 5. Quantum chemistry simulation log inspection (.log, .out signatures)
    is_safe_code_doc = suffix in RECOGNIZED_CODE_DOC_EXTENSIONS
    if not is_safe_code_doc:
        qm_pattern_desc = check_qm_log_signatures(header)
        if qm_pattern_desc:
            violations.append(
                AirgapViolation(
                    file_path=full_path,
                    relative_path=rel_path,
                    violation_type="QM_LOG_VIOLATION",
                    detail=f"Quantum chemistry calculation output signature detected: {qm_pattern_desc}",
                    matched_pattern=qm_pattern_desc,
                )
            )

    # 6. Coordinate format inspection (.xyz format in non-code files)
    if not is_safe_code_doc and is_xyz_coordinate_payload(header):
        violations.append(
            AirgapViolation(
                file_path=full_path,
                relative_path=rel_path,
                violation_type="XYZ_FORMAT_VIOLATION",
                detail="Restricted XYZ atomic coordinate dataset detected in static tree",
            )
        )

    # 7. Shannon Entropy Analysis [M]
    observed_entropy = 0.0
    if len(full_sample) > 512:
        observed_entropy = calculate_shannon_entropy(full_sample)
        is_exempt_media = suffix in EXEMPT_ENTROPY_EXTENSIONS
        if observed_entropy >= entropy_threshold and not is_exempt_media:
            violations.append(
                AirgapViolation(
                    file_path=full_path,
                    relative_path=rel_path,
                    violation_type="HIGH_ENTROPY_VIOLATION",
                    detail=(
                        f"High Shannon entropy ({observed_entropy:.3f} bits/byte >= {entropy_threshold:.1f}) "
                        "indicates disguised binary, compressed, or encrypted artifact"
                    ),
                    entropy=round(observed_entropy, 3),
                )
            )

    return violations, observed_entropy


def scan_git_repository(
    repo_root: Path,
    blocked_exts: AbstractSet[str],
    entropy_threshold: float,
) -> Tuple[List[AirgapViolation], int, int, float, int]:
    """Scan tracked and untracked repository files using 'git ls-files -z' [M][D].

    Args:
        repo_root: Absolute path to the git repository root.
        blocked_exts: Set of lowercased blocked extensions.
        entropy_threshold: Threshold for Shannon entropy analysis.

    Returns:
        Tuple of (violations, files_count, dir_count, max_entropy, high_entropy_count).
    """
    git_cmd = [
        "git",
        "-C",
        str(repo_root),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]

    try:
        proc = subprocess.run(
            git_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"Git scanning failed: {exc}") from exc

    raw_output = proc.stdout
    if not raw_output:
        return [], 0, 0, 0.0, 0

    entries = raw_output.split(b"\x00")
    violations: List[AirgapViolation] = []
    scanned_count = 0
    max_entropy = 0.0
    high_entropy_count = 0
    scanned_dirs: Set[str] = set()

    for raw_entry in entries:
        if not raw_entry:
            continue
        try:
            rel_str = raw_entry.decode("utf-8")
        except UnicodeDecodeError:
            rel_str = raw_entry.decode("utf-8", errors="replace")

        if is_excluded_path_segment(rel_str):
            continue

        full_path = repo_root / rel_str
        if full_path.name in DEFAULT_EXCLUDED_FILES or any(full_path.name.endswith(sfx) for sfx in EXCLUDED_FILE_SUFFIXES):
            continue

        parent_rel = Path(rel_str).parent.as_posix()
        if parent_rel != ".":
            scanned_dirs.add(parent_rel)

        # Check restricted directory path segments
        for part in Path(rel_str).parts[:-1]:
            is_restr_dir, pat = is_restricted_directory(part, FORBIDDEN_DIRECTORY_PATTERNS)
            if is_restr_dir:
                violations.append(
                    AirgapViolation(
                        file_path=full_path,
                        relative_path=rel_str,
                        violation_type="RESTRICTED_DIRECTORY",
                        detail=f"File resides in prohibited directory pattern '{pat}'",
                        matched_pattern=pat,
                    )
                )

        scanned_count += 1
        if full_path.is_file():
            file_violations, ent = inspect_file_airgap(
                full_path=full_path,
                repo_root=repo_root,
                blocked_exts=blocked_exts,
                entropy_threshold=entropy_threshold,
            )
            violations.extend(file_violations)
            if ent > max_entropy:
                max_entropy = ent
            if ent >= entropy_threshold and full_path.suffix.lower() not in EXEMPT_ENTROPY_EXTENSIONS:
                high_entropy_count += 1

    return violations, scanned_count, len(scanned_dirs), max_entropy, high_entropy_count


def scan_filesystem_tree(
    repo_root: Path,
    blocked_exts: AbstractSet[str],
    excluded_dirs: AbstractSet[str],
    entropy_threshold: float,
) -> Tuple[List[AirgapViolation], int, int, float, int]:
    """Scan directory tree recursively using pure-Python filesystem traversal [M][D].

    Args:
        repo_root: Root directory to scan.
        blocked_exts: Set of lowercased blocked extensions.
        excluded_dirs: Set of directory names to skip (e.g. {'.git', '.venv'}).
        entropy_threshold: Threshold for Shannon entropy analysis.

    Returns:
        Tuple of (violations, files_count, dir_count, max_entropy, high_entropy_count).
    """
    violations: List[AirgapViolation] = []
    scanned_files = 0
    scanned_dirs = 0
    max_entropy = 0.0
    high_entropy_count = 0

    for root, dirs, files in os.walk(repo_root, topdown=True):
        # Inspect and prune restricted directories
        for d in list(dirs):
            is_restr_dir, pat = is_restricted_directory(d, FORBIDDEN_DIRECTORY_PATTERNS)
            if is_restr_dir:
                dir_path = Path(root) / d
                try:
                    rel_d = dir_path.relative_to(repo_root).as_posix()
                except ValueError:
                    rel_d = str(dir_path).replace("\\", "/")
                violations.append(
                    AirgapViolation(
                        file_path=dir_path,
                        relative_path=rel_d,
                        violation_type="RESTRICTED_DIRECTORY",
                        detail=f"Prohibited dynamic directory '{d}' detected in static workspace repository",
                        matched_pattern=pat,
                    )
                )

        # Prune excluded directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in excluded_dirs and not any(d.lower().endswith(sfx) for sfx in EXCLUDED_DIR_SUFFIXES)
        ]
        scanned_dirs += len(dirs)

        root_path = Path(root)

        for filename in files:
            if filename in DEFAULT_EXCLUDED_FILES or any(filename.lower().endswith(sfx) for sfx in EXCLUDED_FILE_SUFFIXES):
                continue

            full_path = root_path / filename
            scanned_files += 1

            file_violations, ent = inspect_file_airgap(
                full_path=full_path,
                repo_root=repo_root,
                blocked_exts=blocked_exts,
                entropy_threshold=entropy_threshold,
            )
            violations.extend(file_violations)
            if ent > max_entropy:
                max_entropy = ent
            if ent >= entropy_threshold and full_path.suffix.lower() not in EXEMPT_ENTROPY_EXTENSIONS:
                high_entropy_count += 1

    return violations, scanned_files, scanned_dirs, max_entropy, high_entropy_count


def run_airgap_sweep(
    repo_root: str | Path = ".",
    blocked_extensions: Optional[Sequence[str] | Set[str]] = None,
    excluded_dirs: Optional[Sequence[str] | Set[str]] = None,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    use_git: bool = True,
) -> AirgapSweepSummary:
    """Run comprehensive cross-platform repository MIME, Magic Number, and Entropy Air-Gap Sweep [M][D].

    Args:
        repo_root: Path to the repository or directory root to scan.
        blocked_extensions: Custom blocked extensions. Defaults to FORBIDDEN_EXTENSIONS.
        excluded_dirs: Custom excluded directory names. Defaults to DEFAULT_EXCLUDED_DIRS.
        entropy_threshold: Threshold for high Shannon entropy detection.
        use_git: If True, uses 'git ls-files -z' if available, falling back to filesystem walk.

    Returns:
        AirgapSweepSummary containing clean status, violations list, and execution telemetry.
    """
    start_time = time.perf_counter()
    root = Path(repo_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository root path is not a directory: {root}")

    if blocked_extensions is None:
        blocked_set: Set[str] = set(FORBIDDEN_EXTENSIONS)
    else:
        blocked_set = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in blocked_extensions
        }

    if excluded_dirs is None:
        excluded_set: Set[str] = set(DEFAULT_EXCLUDED_DIRS)
    else:
        excluded_set = set(excluded_dirs)

    is_git_repo = (root / ".git").exists()
    scan_method = "filesystem_traversal"

    if use_git and is_git_repo:
        try:
            violations, scanned_files, scanned_dirs, max_ent, high_ent = scan_git_repository(
                repo_root=root,
                blocked_exts=blocked_set,
                entropy_threshold=entropy_threshold,
            )
            scan_method = "git_ls_files"
        except Exception:
            violations, scanned_files, scanned_dirs, max_ent, high_ent = scan_filesystem_tree(
                repo_root=root,
                blocked_exts=blocked_set,
                excluded_dirs=excluded_set,
                entropy_threshold=entropy_threshold,
            )
            scan_method = "filesystem_fallback"
    else:
        violations, scanned_files, scanned_dirs, max_ent, high_ent = scan_filesystem_tree(
            repo_root=root,
            blocked_exts=blocked_set,
            excluded_dirs=excluded_set,
            entropy_threshold=entropy_threshold,
        )
        scan_method = "filesystem_traversal"

    elapsed = time.perf_counter() - start_time
    is_clean = len(violations) == 0

    return AirgapSweepSummary(
        repo_root=root,
        is_clean=is_clean,
        scanned_files_count=scanned_files,
        scanned_directories_count=scanned_dirs,
        violations=violations,
        scan_method=scan_method,
        execution_time_seconds=elapsed,
        max_entropy_observed=max_ent,
        high_entropy_files_count=high_ent,
    )


def format_airgap_report(summary: AirgapSweepSummary) -> str:
    """Format an air-gap sweep summary into structured human-readable console output [D].

    Args:
        summary: The AirgapSweepSummary from run_airgap_sweep.

    Returns:
        Formatted multi-line report string.
    """
    lines: List[str] = []
    bar = "=" * 78
    lines.append(bar)
    lines.append(" [CI AIR-GAP SWEEP] Tripartite Workspace Pre-Flight Security Gate")
    lines.append(bar)
    lines.append(f" Target Repository : {summary.repo_root}")
    lines.append(f" Inspection Mode   : {summary.scan_method}")
    lines.append(f" Files Inspected   : {summary.scanned_files_count}")
    lines.append(f" Dirs Traversed    : {summary.scanned_directories_count}")
    lines.append(f" Max Entropy (H)   : {summary.max_entropy_observed:.3f} bits/byte [M]")
    lines.append(f" Execution Time    : {summary.execution_time_seconds:.4f} seconds [M]")

    if summary.is_clean:
        lines.append(" Status            : [PASSED] CLEAN (0 violations)")
        lines.append(" Verdict           : Static Execution Tier is 100% immutable and air-gap compliant.")
        lines.append(bar)
    else:
        lines.append(f" Status            : [FAILED] AIR-GAP BREACH DETECTED ({summary.violation_count} violations)")
        lines.append(" Verdict           : Prohibited dynamic or binary artifacts detected in static repository.")
        lines.append("-" * 78)
        lines.append(" DETECTED AIR-GAP VIOLATIONS:")
        for idx, v in enumerate(summary.violations, 1):
            lines.append(f"   [{idx}] Type    : {v.violation_type} [{v.severity}]")
            lines.append(f"       File    : {v.relative_path}")
            lines.append(f"       Detail  : {v.detail}")
            if v.entropy is not None:
                lines.append(f"       Entropy : {v.entropy:.3f} bits/byte")
        lines.append("-" * 78)
        lines.append(" MANDATED REMEDIATION PROTOCOL (SRS Doc 2 Part 1 §1.1):")
        lines.append("   1. Move all quantum chemical data (.h5, .gbw, .xyz, .hess) to ~/CoChem_Artifacts.")
        lines.append("   2. Purge calculation logs (.log, .out, .err) and ephemeral locks from Git history.")
        lines.append("   3. Scrub localized absolute system paths from cochem_system_config.json.")
        lines.append("   4. Re-run 'python ci_tools/ci_airgap_sweep.py --repo-root .' before submitting CI PR.")
        lines.append(bar)

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for ci_airgap_sweep.py [D]."""
    parser = argparse.ArgumentParser(
        prog="ci_airgap_sweep",
        description=(
            "Pre-flight cross-platform Python-based repository MIME, Magic Number, "
            "and Shannon Entropy air-gap inspection in CI/CD (SRS Doc 2 Part 1 §1.1)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        "-r",
        dest="repo_root",
        type=str,
        default=".",
        help="Path to the repository or directory root to inspect (default: current working directory).",
    )
    parser.add_argument(
        "--no-git",
        dest="no_git",
        action="store_true",
        default=False,
        help="Force pure-Python filesystem traversal instead of git ls-files.",
    )
    parser.add_argument(
        "--entropy-threshold",
        "-e",
        dest="entropy_threshold",
        type=float,
        default=DEFAULT_ENTROPY_THRESHOLD,
        help=f"Shannon entropy threshold in bits/byte to flag disguised binaries (default: {DEFAULT_ENTROPY_THRESHOLD}).",
    )
    parser.add_argument(
        "--add-extension",
        dest="add_extensions",
        action="append",
        default=[],
        help="Additional blocked file extension to enforce (can be specified multiple times).",
    )
    parser.add_argument(
        "--json",
        "--output-json",
        dest="output_json",
        action="store_true",
        default=False,
        help="Output structured JSON summary report instead of formatted plain text.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose output logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point for CI Air-Gap Sweep [D].

    Args:
        argv: Optional sequence of CLI argument strings.

    Returns:
        0 if repository is clean, 1 if violations are detected or on error.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    blocked_exts = set(FORBIDDEN_EXTENSIONS)
    for ext in args.add_extensions:
        ext_clean = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        blocked_exts.add(ext_clean)

    use_git = not args.no_git

    try:
        summary = run_airgap_sweep(
            repo_root=args.repo_root,
            blocked_extensions=blocked_exts,
            entropy_threshold=args.entropy_threshold,
            use_git=use_git,
        )
    except Exception as exc:
        print(f"[CI AIR-GAP SWEEP ERROR] Sweep execution failed: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        print(json.dumps(summary.to_dict(), indent=2), file=sys.stdout if summary.is_clean else sys.stderr)
    else:
        report = format_airgap_report(summary)
        if summary.is_clean:
            print(report, file=sys.stdout)
        else:
            print(report, file=sys.stderr)

    return 0 if summary.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
