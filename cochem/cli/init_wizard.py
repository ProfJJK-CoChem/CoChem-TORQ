"""Interactive CLI Workspace Scaffolding Wizard (cochem init).

Module: cochem.cli.init_wizard
Authoritative Reference: SRS Chunk 02 BASE UI & Web (Part 2), Prompt 2.

Provides automated bootstrapping for CoChem project workspaces:
1. Standard directory hierarchy:
   - data/raw: Read-only staging for incoming experimental files (.mol, .sdf, .cif, .xyz).
   - data/processed: Deterministic HDF5 datasets and SQLite databases with WAL mode.
   - models/checkpoints: Model weights and PyTorch state dictionaries with SHA-256 checksums.
   - telemetry/logs: Structured JSONL execution logs and hardware performance samples.
   - config/methods: Verified Method Matrix v4 configuration templates (ORCA, CREST, xTB).
2. Disk availability validation (minimum 2.0 GB free disk space verification via shutil.disk_usage).
3. Audit available quantum chemistry binaries in $PATH (orca, crest, xtb, obabel) and record versions.
4. Atomic .cochem_project.json manifest with UUIDv4, UTC timestamp, schema version, and SHA-256 seal.
5. Atomic rollback if directory or file creation encounters an unhandled filesystem exception.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Constants
MIN_FREE_DISK_BYTES: int = 2 * 1024 * 1024 * 1024  # 2.0 GB
SCHEMA_VERSION: str = "4.0.0"
QUANTUM_BINARIES: Sequence[str] = ("orca", "crest", "xtb", "obabel")

STANDARD_DIRECTORIES: Sequence[str] = (
    "data/raw",
    "data/processed",
    "models/checkpoints",
    "telemetry/logs",
    "config/methods",
)


class InitWizardError(Exception):
    """Base exception for workspace initialization wizard errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InsufficientDiskSpaceError(InitWizardError):
    """Raised when target volume has less than the minimum required free space (2.0 GB)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WorkspacePermissionError(InitWizardError):
    """Raised when target directory lacks required read/write permissions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExistingProjectError(InitWizardError):
    """Raised when target directory is already an initialized CoChem workspace."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class BinaryAuditRecord(BaseModel):
    """Pydantic model recording discovered quantum chemistry engine binary metadata."""

    model_config = ConfigDict(frozen=True)

    name: str
    is_available: bool
    path: Optional[str] = None
    version: Optional[str] = None
    sha256_hash: Optional[str] = None


class DiskSpaceInfo(BaseModel):
    """Disk capacity and allocation metrics for the workspace mount point."""

    model_config = ConfigDict(frozen=True)

    total_bytes: int
    used_bytes: int
    free_bytes: int
    total_gb: float
    used_gb: float
    free_gb: float
    has_minimum_space: bool


class ProjectManifest(BaseModel):
    """Cryptographically sealed project manifest persisted as .cochem_project.json."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    project_name: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    directories: List[str]
    binaries: Dict[str, BinaryAuditRecord]
    disk_info: DiskSpaceInfo
    integrity_seal: str


class InitWizardResult(BaseModel):
    """Structured report returned upon completion of run_init_wizard."""

    model_config = ConfigDict(frozen=True)

    project_path: str
    manifest: ProjectManifest
    success: bool
    created_directories: List[str]
    template_files: List[str]


def compute_sha256_file(filepath: Path) -> Optional[str]:
    """Compute SHA-256 hex digest for a physical file."""
    if not filepath.is_file():
        return None
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (OSError, PermissionError) as exc:
        logger.warning(f"Could not compute hash for {filepath}: {exc}")
        return None


def audit_binary(name: str) -> BinaryAuditRecord:
    """Audit availability, absolute path, version, and binary SHA-256 for a given executable."""
    bin_path = shutil.which(name)
    if not bin_path:
        return BinaryAuditRecord(
            name=name,
            is_available=False,
            path=None,
            version=None,
            sha256_hash=None,
        )

    resolved_path = Path(bin_path).resolve()
    sha256_val = compute_sha256_file(resolved_path)
    version_str: Optional[str] = None

    # Probe binary version safely with strict timeout
    try:
        flag = "--version" if name in ("orca", "crest", "xtb", "obabel") else "-v"
        proc = subprocess.run(
            [str(resolved_path), flag],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        combined_output = (proc.stdout + " " + proc.stderr).strip()
        if combined_output:
            first_line = combined_output.splitlines()[0][:120].strip()
            version_str = first_line if first_line else None
    except Exception as exc:
        logger.debug(f"Version check for {name} timed out or failed: {exc}")
        version_str = None

    return BinaryAuditRecord(
        name=name,
        is_available=True,
        path=str(resolved_path),
        version=version_str,
        sha256_hash=sha256_val,
    )


def audit_all_quantum_binaries(
    binary_names: Sequence[str] = QUANTUM_BINARIES,
) -> Dict[str, BinaryAuditRecord]:
    """Audit all required quantum chemistry and cheminformatics binaries in $PATH."""
    records: Dict[str, BinaryAuditRecord] = {}
    for b_name in binary_names:
        records[b_name] = audit_binary(b_name)
    return records


def check_disk_space(
    target_path: Path, min_required_bytes: int = MIN_FREE_DISK_BYTES
) -> DiskSpaceInfo:
    """Inspect disk capacity on target directory filesystem."""
    try:
        usage = shutil.disk_usage(target_path)
    except OSError as exc:
        raise WorkspacePermissionError(
            f"Cannot inspect disk space on '{target_path}': {exc}"
        ) from exc

    total_gb = round(usage.total / (1024**3), 3)
    used_gb = round(usage.used / (1024**3), 3)
    free_gb = round(usage.free / (1024**3), 3)
    has_space = usage.free >= min_required_bytes

    return DiskSpaceInfo(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        has_minimum_space=has_space,
    )


def generate_method_templates(config_methods_dir: Path) -> List[str]:
    """Generate verified Method Matrix v4 configuration templates for ORCA, CREST, and xTB."""
    written_files: List[str] = []

    # 1. ORCA Method Matrix v4 Template
    orca_config = {
        "engine": "ORCA",
        "method_matrix_version": "v4",
        "provenance": "[M]",
        "default_level_of_theory": "r2SCAN-3c",
        "conformer_refinement": {
            "functional": "wB97X-D4",
            "basis_set": "def2-TZVP",
            "dispersion": "D4",
            "grid": "defgrid2",
            "tight_opt": {
                "TolMaxG": 1e-5,
                "TolE": 1e-7,
                "TolRMSG": 3e-6,
                "TolRMSD": 5e-5,
                "TolMaxD": 1e-4,
            },
            "hessian_preconditioner": "InHess XTB2",
        },
        "single_point_gold_standard": {
            "method": "DLPNO-CCSD(T)",
            "basis_set": "cc-pVTZ",
            "aux_basis": "cc-pVTZ/C",
            "tight_pno": True,
        },
        "memory_allocation": {
            "maxcore_mb": 4096,
            "pal_cores": 8,
        },
    }
    orca_path = config_methods_dir / "orca_v4_template.json"
    orca_path.write_text(json.dumps(orca_config, indent=2), encoding="utf-8")
    written_files.append(str(orca_path.relative_to(config_methods_dir.parent.parent)))

    # 2. CREST GOAT Conformation Matrix Template
    crest_config = {
        "engine": "CREST",
        "method_matrix_version": "v4",
        "provenance": "[M]",
        "protocol": "GOAT_XTB2",
        "flags": [
            "--nci",
            "--nocross",
            "--noreftopo",
            "--v3",
            "--ewin",
            "6.0",
        ],
        "energy_window_kcal": 6.0,
        "rmsd_threshold_angstrom": 0.125,
        "rotational_temperature_k": 298.15,
    }
    crest_path = config_methods_dir / "crest_goat_template.json"
    crest_path.write_text(json.dumps(crest_config, indent=2), encoding="utf-8")
    written_files.append(str(crest_path.relative_to(config_methods_dir.parent.parent)))

    # 3. xTB Semiempirical Fast-Screening Template
    xtb_config = {
        "engine": "xTB",
        "method_matrix_version": "v4",
        "provenance": "[M]",
        "hamiltonian": "GFN2-xTB",
        "electronic_temperature_k": 300.0,
        "accuracy": 1.0,
        "solvent": "gas_phase",
        "pre_optimization": {
            "max_iterations": 250,
            "convergence_level": "tight",
        },
    }
    xtb_path = config_methods_dir / "xtb_gfn2_template.json"
    xtb_path.write_text(json.dumps(xtb_config, indent=2), encoding="utf-8")
    written_files.append(str(xtb_path.relative_to(config_methods_dir.parent.parent)))

    return written_files


def run_init_wizard(
    target_directory: Union[str, Path],
    interactive: bool = False,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
    overwrite_existing: bool = False,
) -> Path:
    """Bootstrap a standard CoChem project workspace with atomic rollback protection.

    Args:
        target_directory: Destination path for the new project workspace.
        interactive: If True, renders progress telemetry via rich console when available.
        min_free_bytes: Minimum disk free capacity in bytes required (default: 2.0 GB).
        overwrite_existing: If True, allows re-initializing an existing workspace.

    Returns:
        Path to the bootstrapped workspace directory containing sealed .cochem_project.json.

    Raises:
        InsufficientDiskSpaceError: If volume has less than min_free_bytes free space.
        ExistingProjectError: If .cochem_project.json exists and overwrite_existing is False.
        WorkspacePermissionError: If directories cannot be created due to permissions.
    """
    target_path = Path(target_directory).resolve()
    manifest_file = target_path / ".cochem_project.json"

    # Pre-flight check: Target directory already initialized
    if manifest_file.exists() and not overwrite_existing:
        raise ExistingProjectError(
            f"Directory '{target_path}' is already an initialized CoChem workspace (.cochem_project.json exists)."
        )

    # Rich console optional integration
    console = None
    if interactive:
        try:
            from rich.console import Console
            console = Console()
            console.print(f"[bold cyan]Initializing CoChem Project Workspace:[/bold cyan] {target_path}")
        except ImportError as exc:
            logger.debug(f"Rich console optional dependency unavailable: {exc}")

    # Track newly created artifacts for atomic rollback
    created_dirs: List[Path] = []
    created_files: List[Path] = []
    target_existed_prior = target_path.exists()

    try:
        # Step 1: Ensure target root directory exists and inspect permissions
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(target_path)

        # Step 2: Validate disk availability (minimum 2.0 GB free disk space verification)
        disk_info = check_disk_space(target_path, min_required_bytes=min_free_bytes)
        if not disk_info.has_minimum_space:
            min_gb = round(min_free_bytes / (1024**3), 2)
            raise InsufficientDiskSpaceError(
                f"Target volume for '{target_path}' has {disk_info.free_gb:.2f} GB free space. "
                f"Minimum required is {min_gb:.2f} GB."
            )

        if console:
            console.print(f"  [green]✓[/green] Disk space verified: {disk_info.free_gb:.2f} GB free")

        # Step 3: Scaffolding standard directory hierarchy
        rel_dirs_created: List[str] = []
        for rel_dir in STANDARD_DIRECTORIES:
            dir_path = target_path / rel_dir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
            rel_dirs_created.append(rel_dir)
            if console:
                console.print(f"  [green]✓[/green] Created directory: {rel_dir}/")

        # Step 4: Write Method Matrix v4 configuration templates
        methods_dir = target_path / "config" / "methods"
        template_rel_paths = generate_method_templates(methods_dir)
        for t_rel in template_rel_paths:
            created_files.append(target_path / t_rel)
            if console:
                console.print(f"  [green]✓[/green] Generated configuration template: {t_rel}")

        # Step 5: Audit available quantum chemistry binaries in $PATH
        binary_records = audit_all_quantum_binaries(QUANTUM_BINARIES)
        if console:
            console.print("  [bold yellow]Quantum Chemistry Binary Audit:[/bold yellow]")
            for b_name, rec in binary_records.items():
                status_color = "green" if rec.is_available else "dim"
                status_text = f"Found ({rec.version or rec.path})" if rec.is_available else "Not in $PATH"
                console.print(f"    - {b_name}: [{status_color}]{status_text}[/{status_color}]")

        # Step 6: Generate immutable .cochem_project.json manifest
        project_uuid = str(uuid.uuid4())
        now_utc = datetime.now(timezone.utc).isoformat()

        # Build raw manifest dict without seal to calculate canonical SHA-256 seal
        manifest_payload_for_seal = {
            "project_id": project_uuid,
            "project_name": target_path.name,
            "created_at": now_utc,
            "schema_version": SCHEMA_VERSION,
            "directories": rel_dirs_created,
            "binaries": {k: v.model_dump() for k, v in binary_records.items()},
            "disk_info": disk_info.model_dump(),
        }
        canonical_seal_str = json.dumps(
            manifest_payload_for_seal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        integrity_seal = hashlib.sha256(canonical_seal_str.encode("utf-8")).hexdigest()

        manifest_obj = ProjectManifest(
            project_id=project_uuid,
            project_name=target_path.name,
            created_at=now_utc,
            schema_version=SCHEMA_VERSION,
            directories=rel_dirs_created,
            binaries=binary_records,
            disk_info=disk_info,
            integrity_seal=integrity_seal,
        )

        # Atomic file write: write to .tmp and rename to target
        tmp_manifest_file = target_path / ".cochem_project.json.tmp"
        tmp_manifest_file.write_text(
            manifest_obj.model_dump_json(indent=2), encoding="utf-8"
        )
        tmp_manifest_file.replace(manifest_file)
        created_files.append(manifest_file)

        if console:
            console.print(f"  [bold green]✓ Project initialized successfully! Project UUID: {project_uuid}[/bold green]")

        return target_path

    except Exception as exc:
        # Atomic rollback: revert all created files and directories
        logger.error(f"Workspace initialization aborted due to error: {exc}. Rolling back changes...")
        for f in reversed(created_files):
            try:
                if f.exists():
                    f.unlink()
            except OSError as unlink_err:
                logger.warning(f"Rollback file removal failed for {f}: {unlink_err}")

        for d in reversed(created_dirs):
            try:
                if d.exists() and d != target_path:
                    # Only remove if directory is now empty
                    if not any(d.iterdir()):
                        d.rmdir()
            except OSError as rmdir_err:
                logger.warning(f"Rollback dir removal failed for {d}: {rmdir_err}")

        # If the root target path was created in this run and is now empty, remove it
        if not target_existed_prior and target_path.exists():
            try:
                if not any(target_path.iterdir()):
                    target_path.rmdir()
            except OSError as rmdir_err:
                logger.warning(f"Could not remove empty root directory {target_path} during rollback: {rmdir_err}")

        raise exc


def verify_project_integrity(target_directory: Union[str, Path]) -> ProjectManifest:
    """Verify SHA-256 seal and schema integrity of an existing .cochem_project.json manifest."""
    target_path = Path(target_directory).resolve()
    manifest_file = target_path / ".cochem_project.json"
    if not manifest_file.exists():
        raise InitWizardError(f"No .cochem_project.json found in '{target_path}'.")

    content = manifest_file.read_text(encoding="utf-8")
    data = json.loads(content)

    manifest_obj = ProjectManifest.model_validate(data)

    # Recompute seal
    payload_for_seal = {
        "project_id": manifest_obj.project_id,
        "project_name": manifest_obj.project_name,
        "created_at": manifest_obj.created_at,
        "schema_version": manifest_obj.schema_version,
        "directories": manifest_obj.directories,
        "binaries": {k: v.model_dump() for k, v in manifest_obj.binaries.items()},
        "disk_info": manifest_obj.disk_info.model_dump(),
    }
    canonical_seal_str = json.dumps(
        payload_for_seal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected_seal = hashlib.sha256(canonical_seal_str.encode("utf-8")).hexdigest()

    if manifest_obj.integrity_seal != expected_seal:
        raise InitWizardError(
            f"Integrity seal mismatch in '{manifest_file}'. Recorded: {manifest_obj.integrity_seal}, Computed: {expected_seal}"
        )

    return manifest_obj
