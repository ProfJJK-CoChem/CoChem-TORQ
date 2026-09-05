"""
Unit test suite for CoChem Orchestrator Golden System Registry Validator (cochem_system_config.py).
Strict Zero-Mock Mandate: Real filesystem operations, genuine hardware interrogations,
deterministic SHA-256 checksums, and rigorous Pydantic V2 schema validations per SRS Doc 2 Part 2 §3.11.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from orchestrator.cochem_system_config import (
    AdaptiveRoutingPolicy,
    AirGapViolationError,
    CoChemConfig,
    CoChemSystemConfig,
    CorePinningConfig,
    EngineInfo,
    EnginePaths,
    EngineStatus,
    EngineTrack,
    EnvironmentSchema,
    GPUComputeSchema,
    GPUDevice,
    GoldenSystemRegistry,
    HardwareConfig,
    HardwareProfile,
    HardwareSchema,
    HPCConfig,
    HPCProfile,
    HPCSchedulerType,
    IntegrityCheckError,
    MPSConfig,
    OSProfile,
    OSTarget,
    QuantumEngineRegistry,
    QuantumSettings,
    RegistryLockError,
    RoutingPolicy,
    SchemaValidationError,
    SiloAuditItem,
    SiloConfig,
    SiloPathsSchema,
    SiloRegistry,
    SiloStatus,
    SiloType,
    StorageMode,
    StorageTopology,
    SystemConfigError,
    WSL9PMountError,
    audit_wsl_mount_traps,
    discover_engines,
    discover_host_hardware,
    discover_host_os,
    discover_storage_topology,
    get_mendeleev_isotopic_masses,
    interrogate_system_config,
    load_golden_registry,
    main,
    resolve_golden_registry_path,
    save_golden_registry,
    validate_system_config,
    verify_golden_registry_integrity,
    verify_ieee754_subnormal_precision,
)


# =============================================================================
# 1. PYDANTIC V2 SCHEMA CONSTRAINTS & EXTRA FIELD ENFORCEMENT
# =============================================================================


def test_extra_fields_strictly_forbidden() -> None:
    """Verify that all schemas reject hallucinated or extra fields."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OSProfile(
            system="Linux",
            release="6.1",
            version="#1",
            machine="x86_64",
            hallucinated_os_flag="bad",
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HardwareProfile(ram_gb=16.0, physical_cpu_cores=4, rogue_hw="bad")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GPUComputeSchema(gpu_profile="NVIDIA", rogue_gpu="bad")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MPSConfig(enabled=True, rogue_mps=123)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QuantumSettings(charge=0, multiplicity=1, rogue_setting="bad")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RoutingPolicy(max_concurrent_mace_threads=4, rogue_route=True)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HPCConfig(scheduler="slurm", rogue_hpc="bad")


def test_schema_assignment_validation() -> None:
    """Verify that property assignment triggers strict Pydantic validation."""
    hw = HardwareProfile(ram_gb=32.0, physical_cpu_cores=8)
    hw.ram_gb = 64.0
    assert hw.ram_gb == 64.0

    with pytest.raises(ValidationError):
        hw.ram_gb = 0.0  # Must be > 0.0

    with pytest.raises(ValidationError):
        hw.physical_cpu_cores = 0  # Must be >= 1


# =============================================================================
# 2. MENDELEEV MASS AUTHORITY TESTS
# =============================================================================


def test_mendeleev_dynamic_mass_loading() -> None:
    """Verify Mendeleev dynamic mass retrieval without hardcoded CODATA constants."""
    masses = get_mendeleev_isotopic_masses()
    assert isinstance(masses, dict)
    assert "C" in masses
    assert "13C" in masses
    assert "12C" in masses
    assert abs(masses["13C"] - 13.00335) < 0.01
    assert abs(masses["C"] - 12.011) < 0.05


# =============================================================================
# 3. HARDWARE & OS DISCOVERY TESTS
# =============================================================================


def test_discover_host_os_live() -> None:
    """Verify host OS interrogation returns real platform metrics."""
    os_prof = discover_host_os()
    assert isinstance(os_prof, OSProfile)
    assert os_prof.system == platform.system()
    assert os_prof.machine == platform.machine()
    assert Path(os_prof.python_executable).exists()
    assert len(os_prof.python_version) > 0


def test_discover_host_hardware_live() -> None:
    """Verify host hardware discovery returns positive RAM and CPU counts."""
    hw_prof = discover_host_hardware()
    assert isinstance(hw_prof, HardwareProfile)
    assert hw_prof.physical_cpu_cores >= 1
    assert hw_prof.logical_cpu_cores >= hw_prof.physical_cpu_cores
    assert hw_prof.ram_gb > 0.0
    assert hw_prof.maxcore_mb is not None and hw_prof.maxcore_mb >= 500
    assert isinstance(hw_prof.subnormal_precision_trap, bool)


def test_ieee754_subnormal_precision_live() -> None:
    """Verify IEEE-754 subnormal precision test executes without crashing."""
    trap = verify_ieee754_subnormal_precision()
    assert isinstance(trap, bool)


# =============================================================================
# 4. QUANTUM ENGINE & SILO REGISTRY TESTS
# =============================================================================


def test_quantum_engine_registry_helpers() -> None:
    """Test QuantumEngineRegistry helper functions and models."""
    orca_info = EngineInfo(
        status="found",
        path=sys.executable,
        version="6.1.1",
        hash="abcdef123456",
        gpu_support=False,
    )
    reg = QuantumEngineRegistry(orca=orca_info)
    assert reg.is_available("orca") is True
    assert reg.is_bypassed("orca") is False
    assert reg.get_engine("orca") == orca_info

    bypassed_info = EngineInfo(status="bypassed", path="BYPASSED")
    reg_bypassed = QuantumEngineRegistry(cfour=bypassed_info)
    assert reg_bypassed.is_bypassed("cfour") is True
    assert reg_bypassed.is_available("cfour") is False


def test_silo_registry_helpers(tmp_path: Path) -> None:
    """Test SiloRegistry and SiloAuditItem models."""
    silo_item = SiloAuditItem(
        name="cochem_core_silo",
        silo_type=SiloType.CORE,
        path=str(tmp_path / "core_silo"),
        python_executable=sys.executable,
        status=SiloStatus.PROVISIONED,
        is_available=True,
    )
    reg = SiloRegistry(core_silo=silo_item, torq_silo_active=True)
    assert reg.is_silo_active("core_silo") is True
    assert reg.is_silo_active("torq") is True
    assert reg.get_silo("core_silo") == silo_item


# =============================================================================
# 5. STORAGE TOPOLOGY & AIR-GAP TESTS
# =============================================================================


def test_storage_topology_valid(tmp_path: Path) -> None:
    """Test StorageTopology path expansions and validation."""
    art_dir = tmp_path / "Artifacts"
    scratch_dir = tmp_path / "Scratch"
    art_dir.mkdir()
    scratch_dir.mkdir()

    topo = StorageTopology(
        artifacts_dir=str(art_dir),
        scratch_dir=str(scratch_dir),
        hdf5_pes_store_path=str(art_dir / "pes_store.h5"),
    )
    assert Path(topo.artifacts_dir).resolve() == art_dir.resolve()
    assert Path(topo.scratch_dir).resolve() == scratch_dir.resolve()


def test_storage_topology_airgap_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that targeting immutable $COCHEM_ROOT for write stores triggers AirGapViolationError."""
    cochem_root = tmp_path / "CoChem_Root"
    cochem_root.mkdir()
    monkeypatch.setenv("COCHEM_ROOT", str(cochem_root))

    with pytest.raises(ValidationError):
        StorageTopology(
            artifacts_dir=str(tmp_path / "Artifacts"),
            hdf5_pes_store_path=str(cochem_root / "illegal_store.h5"),
        )


# =============================================================================
# 6. COCHEM SYSTEM CONFIG LIFECYCLE & CHECKSUM TESTS
# =============================================================================


def test_cochem_system_config_checksum_lifecycle(tmp_path: Path) -> None:
    """Test SHA-256 checksum calculation, verification, and file persistence."""
    hw = HardwareProfile(ram_gb=16.0, physical_cpu_cores=4)
    cfg = CoChemSystemConfig(hardware=hw)

    checksum1 = cfg.compute_checksum()
    assert len(checksum1) == 64
    cfg.update_checksum()
    assert cfg.registry_checksum == checksum1
    assert cfg.verify_checksum() is True

    # Mutate a field and verify checksum mismatch
    cfg.hardware.ram_gb = 32.0
    assert cfg.verify_checksum() is False

    # Persist to disk and load back
    target_file = tmp_path / "cochem_system_config.json"
    saved_path = cfg.to_file(target_file)
    assert saved_path.exists()

    loaded_cfg = CoChemSystemConfig.from_file(saved_path)
    assert loaded_cfg.hardware.ram_gb == 32.0
    assert loaded_cfg.verify_checksum() is True


def test_cochem_system_config_legacy_migration() -> None:
    """Test migration of legacy flat JSON structure into nested Pydantic architecture."""
    legacy_data = {
        "schema_version": "1.0.0",
        "physical_cpu_cores": 8,
        "logical_cpu_cores": 16,
        "ram_gb": 32.0,
        "avx512_support": True,
        "gpu_profile": "None",
        "vram_gb": 0.0,
        "subnormal_precision_trap": False,
        "os_target": "Local-Windows",
        "engines": {
            "orca": {"status": "missing", "path": None, "version": None, "hash": None},
        },
        "silos": {"torq_silo_active": True, "gpu_silo_active": False},
        "adaptive_routing": "[MISSING DATA]",
        "hpc": {"scheduler": "local", "default_partition": "compute"},
        "active_jobs": {},
    }

    cfg = CoChemSystemConfig.model_validate(legacy_data)
    assert cfg.hardware.physical_cpu_cores == 8
    assert cfg.hardware.ram_gb == 32.0
    assert cfg.hardware.avx512_support is True
    assert cfg.adaptive_routing is None
    assert cfg.hpc.scheduler == "local"


# =============================================================================
# 7. INTERROGATION & INTEGRITY CHECK TESTS
# =============================================================================


def test_interrogate_system_config_end_to_end(tmp_path: Path) -> None:
    """Perform end-to-end host interrogation and verify constructed config."""
    out_file = tmp_path / "Registry" / "cochem_system_config.json"
    cfg = interrogate_system_config(workspace_dir=tmp_path, output_path=out_file)

    assert isinstance(cfg, CoChemSystemConfig)
    assert cfg.status == "INITIALIZED"
    assert out_file.exists()
    assert cfg.verify_checksum() is True


def test_verify_golden_registry_integrity_valid(tmp_path: Path) -> None:
    """Test verify_golden_registry_integrity on a valid configuration."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)
    out_file = tmp_path / "cochem_system_config.json"
    cfg.to_file(out_file)

    valid, issues = verify_golden_registry_integrity(out_file)
    assert valid is True
    assert len(issues) == 0


def test_verify_golden_registry_integrity_corrupted(tmp_path: Path) -> None:
    """Test verify_golden_registry_integrity detects mutated payload with mismatched checksum."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)
    out_file = tmp_path / "cochem_system_config.json"
    cfg.to_file(out_file)

    # Manually tamper with JSON file
    raw = json.loads(out_file.read_text(encoding="utf-8"))
    raw["hardware"]["ram_gb"] = 999.0  # Tampered without recalculating checksum
    out_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    valid, issues = verify_golden_registry_integrity(out_file)
    assert valid is False
    assert any("checksum mismatch" in i.lower() for i in issues)


# =============================================================================
# 8. CLI INTERFACE TESTS
# =============================================================================


def test_cli_validate_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI main() with --validate flag."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)
    out_file = tmp_path / "cochem_system_config.json"
    cfg.to_file(out_file)

    exit_code = main(argv=["--validate", str(out_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SUCCESS: System configuration" in captured.out


def test_cli_interrogate_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI main() with --interrogate and --output flags."""
    out_file = tmp_path / "Registry" / "cochem_system_config.json"
    exit_code = main(argv=["--interrogate", "--output", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()
    captured = capsys.readouterr()
    assert "SUCCESS: Interrogated Golden System Registry" in captured.out


def test_cli_check_integrity_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI main() with --check-integrity flag."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)
    out_file = tmp_path / "cochem_system_config.json"
    cfg.to_file(out_file)

    exit_code = main(argv=["--check-integrity", str(out_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "is VALID and intact" in captured.out


# =============================================================================
# 9. COMPREHENSIVE PUBLIC API & METHOD MATRIX VALIDATION TESTS
# =============================================================================


def test_audit_wsl_mount_traps_live(tmp_path: Path) -> None:
    """Verify WSL2 9P / drvfs mount detection on physical paths."""
    is_9p, details = audit_wsl_mount_traps(tmp_path)
    assert isinstance(is_9p, bool)
    assert isinstance(details, str)
    assert len(details) > 0


def test_discover_engines_live() -> None:
    """Verify physical engine discovery returns a validated QuantumEngineRegistry."""
    engines_reg = discover_engines()
    assert isinstance(engines_reg, QuantumEngineRegistry)
    assert isinstance(engines_reg.orca, EngineInfo)
    assert isinstance(engines_reg.mpirun, EngineInfo)
    assert isinstance(engines_reg.xtb, EngineInfo)


def test_discover_storage_topology_live(tmp_path: Path) -> None:
    """Verify physical storage discovery returns a validated StorageTopology."""
    topo = discover_storage_topology(workspace_dir=tmp_path)
    assert isinstance(topo, StorageTopology)
    assert Path(topo.artifacts_dir).is_absolute()
    assert topo.scratch_dir is not None and Path(topo.scratch_dir).is_absolute()


def test_resolve_golden_registry_path_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 4-tier path resolution for cochem_system_config.json."""
    custom_target = tmp_path / "custom" / "cochem_system_config.json"
    assert resolve_golden_registry_path(custom_target) == custom_target.resolve()

    dir_target = tmp_path / "dir_config"
    dir_target.mkdir()
    assert resolve_golden_registry_path(dir_target) == (dir_target / "cochem_system_config.json").resolve()

    env_cfg = tmp_path / "env_config.json"
    env_cfg.touch()
    monkeypatch.setenv("COCHEM_CONFIG", str(env_cfg))
    assert resolve_golden_registry_path() == env_cfg.resolve()


def test_save_and_load_golden_registry(tmp_path: Path) -> None:
    """Verify atomic saving and loading of the Golden System Registry."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)
    out_file = tmp_path / "Registry" / "cochem_system_config.json"
    saved = save_golden_registry(cfg, path=out_file, lock=False)
    assert saved.exists()

    loaded = load_golden_registry(saved)
    assert loaded.status == "INITIALIZED"
    assert loaded.verify_checksum() is True

    # Loading non-existent path raises FileNotFoundError
    missing_file = tmp_path / "nonexistent" / "config.json"
    with pytest.raises(FileNotFoundError):
        load_golden_registry(missing_file)


def test_validate_system_config_polymorphic(tmp_path: Path) -> None:
    """Verify polymorphic validation on model, dict, file path, and JSON string."""
    cfg = interrogate_system_config(workspace_dir=tmp_path)

    # 1. From model
    val1 = validate_system_config(cfg)
    assert val1.verify_checksum() is True

    # 2. From dict
    cfg_dict = cfg.to_dict()
    val2 = validate_system_config(cfg_dict)
    assert val2.schema_version == "4.0.0"

    # 3. From file
    out_file = tmp_path / "poly_config.json"
    cfg.to_file(out_file)
    val3 = validate_system_config(out_file)
    assert val3.hardware.ram_gb == cfg.hardware.ram_gb

    # 4. From JSON string
    json_str = cfg.to_json()
    val4 = validate_system_config(json_str)
    assert val4.hardware.physical_cpu_cores == cfg.hardware.physical_cpu_cores

    # 5. Invalid source type raises TypeError
    with pytest.raises(TypeError):
        validate_system_config(12345)  # type: ignore


def test_quantum_settings_and_environment_schema(tmp_path: Path) -> None:
    """Verify QuantumSettings and EnvironmentSchema constraints."""
    qs = QuantumSettings(charge=0, multiplicity=1, implicit_solvation="CPCM", integration_grid="defgrid3")
    assert qs.implicit_solvation == "CPCM"
    assert qs.integration_grid == "defgrid3"

    with pytest.raises(ValidationError):
        QuantumSettings(implicit_solvation="INVALID_SOLVENT")

    with pytest.raises(ValidationError):
        QuantumSettings(integration_grid="invalid_grid")

    env = EnvironmentSchema(
        os_target="Local-Windows",
        artifacts_dir=str(tmp_path / "Artifacts"),
        scratch_dir=str(tmp_path / "Scratch"),
    )
    assert env.isotopic_mass_locking is True
    assert "C" in env.isotopic_masses
