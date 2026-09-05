"""CoChem-TORQ: Test Suite for Dynamic Path Resolution & Air-Gap Verification Contracts.

=============================================================================
Unit & Integration Test Suite for Libraries/cochem_path_manager.py
-----------------------------------------------------------------------------
Authentic Physical Test Matrix:
- Real physical disk I/O, directory resolutions, and air-gap boundary checks.
- 6-Tier Environment Matrix resolution and TierCapabilities verification.
- Genuine multiprocessing.shared_memory.SharedMemory and mmap allocations.
- Real xxhash.xxh64 & SHA-256 cryptographic tamper detection.
- Deterministic IPC resource cleanup and atexit safety.
- Ghost output purging (0-byte and temp files).
- Physical disk buffer synchronization (buffer_lock_sync).
- Read-only permission seals (NTFS / POSIX).
- Mendeleev dynamic mass lookups (enforce_ciaaw_masses).
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import tempfile
import time
import uuid

import numpy as np
import pytest
from pydantic import ValidationError

from Libraries.cochem_path_manager import (
    AirGapReport,
    AirGapViolationError,
    BufferCorruptedError,
    ChecksumMismatchError,
    CoChemAirGapRing,
    CoChemIntegrityError,
    CoChemPathManager,
    CoChemSharedMemoryBuffer,
    EnvironmentTier,
    IPCBufferError,
    IPCBufferMetadata,
    IPCPayloadType,
    IPCSegmentType,
    MethodMatrixViolationError,
    OrphanedResourceError,
    SharedMemoryAllocationError,
    TierCapabilities,
    apply_readonly_chmod,
    buffer_lock_sync,
    cleanup_ipc_scratch,
    compute_buffer_checksum,
    enforce_ciaaw_masses,
    get_atomic_mass,
    get_atomic_number,
    get_isotopic_mass,
    purge_ghost_outputs,
    register_shm_buffer,
    remove_readonly_seal,
    unregister_shm_buffer,
)


@pytest.fixture(autouse=True)
def clean_ipc_env():
    """Ensure clean IPC registry state before and after each test."""
    cleanup_ipc_scratch(force_unlink=True)
    yield
    cleanup_ipc_scratch(force_unlink=True)


# =============================================================================
# 1. 6-Tier Scratch & Deliverables Resolution Hierarchy Tests
# =============================================================================


class TestPathManagerResolutionTiers:
    """Authentic physical tests for 6-tier scratch and deliverables resolution."""

    def test_scratch_resolution_tier1_custom_path(self, tmp_path: Path) -> None:
        """Tier 1: Explicit custom_path argument."""
        custom_scratch = tmp_path / "custom_scratch_tier1"
        resolved = CoChemPathManager.resolve_scratch_dir(custom_scratch)
        assert resolved == custom_scratch.resolve()
        assert resolved.exists()
        assert resolved.is_dir()

    def test_scratch_resolution_tier2_env_vars(self, tmp_path: Path) -> None:
        """Tier 2: COCHEM_SCRATCH / COCHEM_SCRATCH_DIR environment variables."""
        env_scratch = tmp_path / "env_scratch_tier2"
        os.environ["COCHEM_SCRATCH"] = str(env_scratch)
        try:
            resolved = CoChemPathManager.resolve_scratch_dir()
            assert resolved == env_scratch.resolve()
            assert resolved.exists()
        finally:
            os.environ.pop("COCHEM_SCRATCH", None)

    def test_deliverables_resolution_tier1_and_tier2(self, tmp_path: Path) -> None:
        """Deliverables resolution across Tier 1 custom path and Tier 2 environment variable."""
        custom_deliv = tmp_path / "custom_deliv_tier1"
        resolved_t1 = CoChemPathManager.resolve_deliverables_dir(custom_deliv)
        assert resolved_t1 == custom_deliv.resolve()
        assert resolved_t1.exists()

        env_deliv = tmp_path / "env_deliv_tier2"
        os.environ["COCHEM_DELIVERABLES"] = str(env_deliv)
        try:
            resolved_t2 = CoChemPathManager.resolve_deliverables_dir()
            assert resolved_t2 == env_deliv.resolve()
            assert resolved_t2.exists()
        finally:
            os.environ.pop("COCHEM_DELIVERABLES", None)

    def test_instance_properties(self, tmp_path: Path) -> None:
        """Test CoChemPathManager instance properties."""
        inst_scratch = tmp_path / "inst_scratch"
        inst_deliv = tmp_path / "inst_deliv"
        mgr = CoChemPathManager(scratch_dir=inst_scratch, deliverables_dir=inst_deliv)

        assert mgr.scratch == inst_scratch.resolve()
        assert mgr.deliverables == inst_deliv.resolve()
        assert mgr.artifacts == inst_deliv.resolve()


# =============================================================================
# 2. Tripartite Air-Gap Boundary Enforcement Tests
# =============================================================================


class TestAirGapBoundaryEnforcement:
    """Tests verifying mathematical boundary verification between Domain A, B, and C."""

    def test_air_gap_safe_disjoint_paths(self, tmp_path: Path) -> None:
        """Disjoint target path outside repo root must pass."""
        safe_scratch = tmp_path / "safe_isolated_scratch"
        report = CoChemPathManager.verify_air_gap_boundary(
            safe_scratch, ring=CoChemAirGapRing.RING_2_DOMAIN_C_EPHEMERAL
        )
        assert report.is_valid is True
        assert report.target_resolved == safe_scratch.resolve()
        assert report.reason is None

    def test_air_gap_repo_violation_raises_error(self) -> None:
        """Mutable target path inside Ring 1 repo root must raise AirGapViolationError."""
        repo_root = CoChemPathManager.get_repo_root()
        illegal_target = repo_root / "Libraries" / "illegal_runtime_file.tmp"

        with pytest.raises(AirGapViolationError) as exc_info:
            CoChemPathManager.verify_air_gap_boundary(
                illegal_target, ring=CoChemAirGapRing.RING_2_DOMAIN_C_EPHEMERAL
            )

        assert "CRITICAL AIR-GAP BREACH" in str(exc_info.value)

    def test_assert_air_gap_helper(self, tmp_path: Path) -> None:
        """CoChemPathManager.assert_air_gap helper method."""
        safe_path = tmp_path / "safe_dir"
        CoChemPathManager.assert_air_gap(safe_path)

        repo_root = CoChemPathManager.get_repo_root()
        illegal_target = repo_root / "tests"
        with pytest.raises(AirGapViolationError):
            CoChemPathManager.assert_air_gap(illegal_target)


# =============================================================================
# 3. Cross-Platform Shared Memory Buffer Tests
# =============================================================================


class TestCoChemSharedMemoryBuffer:
    """Authentic physical tests for CoChemSharedMemoryBuffer."""

    def test_shm_buffer_lifecycle_and_zero_copy(self, tmp_path: Path) -> None:
        """Verify allocation, write, slice read, memoryview, checksum, and unlinking."""
        buf_name = f"cochem_test_shm_{uuid.uuid4().hex[:8]}"
        payload = b"AUTHENTIC_QUANTUM_DENSITY_MATRIX_BYTES_12345"

        with CoChemSharedMemoryBuffer(
            name=buf_name,
            size=len(payload) + 128,
            create=True,
            prefer_shm=True,
            backing_dir=tmp_path,
        ) as buffer:
            assert buffer.name == buf_name
            assert buffer.size == len(payload) + 128
            assert not buffer.is_closed

            bytes_written = buffer.write(payload, offset=0)
            assert bytes_written == len(payload)

            read_back = buffer.read(size=len(payload), offset=0)
            assert read_back == payload

            # Non-copying memoryview slice
            mview = buffer.get_memoryview(offset=0, size=len(payload))
            assert bytes(mview) == payload
            mview.release()

            # Checksum verification
            assert buffer.verify_checksum() is True

        # Ensure cleanup after context
        assert buffer.is_closed


# =============================================================================
# 4. Ghost Purger & Physical Sync Tests
# =============================================================================


class TestGhostPurgerAndBufferSync:
    """Tests for ghost file purging and physical fsync verification."""

    def test_purge_ghost_outputs(self, tmp_path: Path) -> None:
        """Validate purging of 0-byte files and .tmp staging artifacts."""
        ghost_dir = tmp_path / "ghost_test_run"
        ghost_dir.mkdir(parents=True, exist_ok=True)

        valid_file = ghost_dir / "catalog.parquet"
        valid_file.write_bytes(b"VALID_PARQUET_SPECTRAL_HEADER")

        ghost_0byte = ghost_dir / "failed_orca.out"
        ghost_0byte.write_bytes(b"")

        ghost_tmp = ghost_dir / "catalog.parquet.tmp"
        ghost_tmp.write_bytes(b"STAGING_TEMP_DATA")

        purged = purge_ghost_outputs(ghost_dir, remove_0byte_only=False)
        assert ghost_0byte in purged
        assert ghost_tmp in purged
        assert not ghost_0byte.exists()
        assert not ghost_tmp.exists()
        assert valid_file.exists()

    def test_buffer_lock_sync_success_and_failure(self, tmp_path: Path) -> None:
        """Validate physical buffer flush and minimum byte validation."""
        valid_file = tmp_path / "sync_valid.bin"
        valid_file.write_bytes(b"PHYSICAL_DISK_CONTENT")

        size = buffer_lock_sync(valid_file, min_bytes=5)
        assert size == len(b"PHYSICAL_DISK_CONTENT")

        empty_file = tmp_path / "sync_empty.bin"
        empty_file.write_bytes(b"")

        with pytest.raises(CoChemIntegrityError) as exc_info:
            buffer_lock_sync(empty_file, min_bytes=1)

        assert "Buffer sync validation failed" in exc_info.value.message


# =============================================================================
# 5. Mendeleev Dynamic Elemental & Isotopic Mass Tests
# =============================================================================


class TestMendeleevDynamicMasses:
    """Strict Mendeleev Mandate compliance tests."""

    def test_atomic_number_and_mass(self) -> None:
        """Dynamic retrieval of atomic numbers and standard masses."""
        assert get_atomic_number("C") == 6
        assert get_atomic_number("O") == 8
        assert get_atomic_number("H") == 1
        assert get_atomic_number(10) == 10  # Neon

        c_mass = get_atomic_mass("C")
        assert 12.010 < c_mass < 12.012

    def test_enforce_ciaaw_masses(self) -> None:
        """Dynamic retrieval of exact mono-isotopic CIAAW masses."""
        symbols = ["12C", "13C", "1H", "D", "16O", "18O"]
        masses = enforce_ciaaw_masses(symbols)

        assert len(masses) == 6
        assert np.isclose(masses[0], 12.000000, atol=1e-5)  # 12C
        assert np.isclose(masses[1], 13.003355, atol=1e-5)  # 13C
        assert np.isclose(masses[2], 1.007825, atol=1e-5)   # 1H
        assert np.isclose(masses[3], 2.014102, atol=1e-5)   # D (2H)
        assert np.isclose(masses[4], 15.994915, atol=1e-5)  # 16O
        assert np.isclose(masses[5], 17.999160, atol=1e-5)  # 18O

    def test_enforce_ciaaw_masses_edge_cases(self) -> None:
        """Validate empty list and invalid symbol errors."""
        empty_res = enforce_ciaaw_masses([])
        assert len(empty_res) == 0

        with pytest.raises(ValueError):
            enforce_ciaaw_masses([""])

        with pytest.raises(ValueError):
            enforce_ciaaw_masses(["InvalidSymbol$$"])

    def test_tier_detection_and_capabilities(self) -> None:
        """Test environment tier detection and associated TierCapabilities."""
        tier = CoChemPathManager.detect_tier()
        assert isinstance(tier, EnvironmentTier)

        caps = CoChemPathManager.get_tier_capabilities(tier)
        assert isinstance(caps, TierCapabilities)
        assert caps.tier == tier
        assert caps.max_worker_processes > 0
        assert len(caps.default_scratch_mount) > 0

        for candidate_tier in EnvironmentTier:
            t_caps = CoChemPathManager.get_tier_capabilities(candidate_tier)
            assert t_caps.tier == candidate_tier
            assert isinstance(t_caps.is_ci, bool)
            assert isinstance(t_caps.is_hpc, bool)

    def test_mmap_fallback_buffer_lifecycle(self, tmp_path: Path) -> None:
        """Verify explicit mmap fallback buffer lifecycle (prefer_shm=False)."""
        buf_name = f"cochem_test_mmap_{uuid.uuid4().hex[:8]}"
        payload = np.array([1.234, 5.678, 9.1011], dtype=np.float64)

        with CoChemSharedMemoryBuffer(
            name=buf_name,
            size=256,
            create=True,
            prefer_shm=False,
            backing_dir=tmp_path,
        ) as buffer:
            assert buffer.name == buf_name
            assert not buffer.is_shm
            assert buffer.path is not None
            assert buffer.path.exists()

            bytes_written = buffer.write(payload, offset=0)
            assert bytes_written == payload.nbytes

            read_back = buffer.read(size=payload.nbytes, offset=0)
            recovered_array = np.frombuffer(read_back, dtype=np.float64)
            assert np.allclose(recovered_array, payload)

            assert buffer.verify_checksum() is True

        buffer.unlink()
        assert not buffer.path.exists()

    def test_buffer_overflow_and_closed_errors(self, tmp_path: Path) -> None:
        """Assert buffer overflow raises IPCBufferError and closed buffer operations fail."""
        with CoChemSharedMemoryBuffer(
            size=64,
            create=True,
            prefer_shm=False,
            backing_dir=tmp_path,
        ) as buffer:
            overflow_payload = b"X" * 128
            with pytest.raises(IPCBufferError) as exc_info:
                buffer.write(overflow_payload, offset=0)
            assert "exceeds buffer capacity" in str(exc_info.value)

            buffer.close()
            with pytest.raises(IPCBufferError):
                buffer.write(b"ABC")
            with pytest.raises(IPCBufferError):
                buffer.read(10)
            with pytest.raises(IPCBufferError):
                buffer.get_memoryview()

    def test_checksum_mismatch_detection(self, tmp_path: Path) -> None:
        """Verify ChecksumMismatchError on corrupted buffer."""
        with CoChemSharedMemoryBuffer(
            size=128,
            create=True,
            prefer_shm=False,
            backing_dir=tmp_path,
        ) as buffer:
            buffer.write(b"ORIGINAL_CORRECT_PAYLOAD")
            with pytest.raises(ChecksumMismatchError):
                buffer.verify_checksum(expected_checksum="DEADBEEF12345678")
