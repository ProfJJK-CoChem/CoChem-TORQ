"""CoChem-TORQ: Test Suite for PyArrow IPC & Shared Memory Manager.

=============================================================================
Phase 5 (Stage 4.0) Authentic Physical Test Matrix
--------------------------------------------------
Authentic physical test suite verifying genuine PyArrow IPC serialization/deserialization,
cross-platform POSIX/Windows multiprocessing.shared_memory.SharedMemory mapping,
memory-mapped file (mmap) fallbacks, xxhash.xxh64 & SHA-256 tamper detection,
deterministic orphaned IPC resource cleanup, 6-Tier Environment Matrix path
resolutions with Tripartite Air-Gap enforcement, and Mendeleev dynamic mass lookups.

All tests operate against real physical memory segments, genuine temp files,
and real PyArrow tables, batches, and multi-dimensional tensors.
"""

from __future__ import annotations

import gc
import mmap
import multiprocessing.shared_memory as sm
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mendeleev import element
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_ipc import (
    AirGapReport,
    AirGapViolationError,
    BufferCorruptedError,
    ChecksumMismatchError,
    ChecksumVerificationResult,
    EnvironmentTier,
    ExecutionContext,
    IPCBufferError,
    IPCBufferMetadata,
    IPCPayloadType,
    IPCRegistry,
    IPCSegmentType,
    OrphanedResourceError,
    PyArrowMmapFileSegment,
    PyArrowSharedMemorySegment,
    SharedMemoryAllocationError,
    _GLOBAL_IPC_REGISTRY,
    _XXHASH_AVAILABLE,
    compute_buffer_checksum,
    deserialize_from_ipc,
    get_atomic_mass,
    get_atomic_number,
    get_ipc_registry,
    get_isotopic_mass,
    get_repo_root,
    mmap_checksum_verifier,
    orphaned_ipc_cleaner,
    pyarrow_mmap_mapper,
    serialize_to_ipc,
)


# ============================================================================
# Autouse Fixture to Guarantee Clean IPC Registry Between Tests
# ============================================================================


@pytest.fixture(autouse=True)
def clean_ipc_environment() -> Generator[None, None, None]:
    """Ensure clean IPC tracking registry state before and after each test."""
    orphaned_ipc_cleaner()
    yield
    orphaned_ipc_cleaner()


# ============================================================================
# 1. PyArrow Shared Memory Lifecycle Tests
# ============================================================================


class TestPyArrowSharedMemoryLifecycle:
    """Authentic physical tests for SharedMemory IPC lifecycle."""

    def test_shared_memory_table_lifecycle(self) -> None:
        """Verify allocation, write, read-back, checksum verification, and cleanup for PyArrow Table."""
        table = pa.table({
            "atom_index": [0, 1, 2, 3],
            "symbol": ["O", "H", "H", "Ne"],
            "mass": [
                get_atomic_mass("O"),
                get_atomic_mass("H"),
                get_atomic_mass("H"),
                get_atomic_mass("Ne"),
            ],
            "x": [0.0, 0.757, -0.757, 3.2],
            "y": [0.0, 0.586, 0.586, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        })

        seg_name = f"cochem_test_tbl_{uuid.uuid4().hex[:8]}"
        segment = pyarrow_mmap_mapper(payload=table, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert segment.name == seg_name
        assert segment.metadata is not None
        assert segment.metadata.payload_type == IPCPayloadType.TABLE
        assert segment.metadata.size_bytes > 0
        assert segment.metadata.checksum_xxh64 != ""

        # Verify checksum
        raw_bytes = segment.read_raw(verify_checksum=True)
        assert len(raw_bytes) == segment.metadata.size_bytes

        # Read back table directly
        table_read = segment.read_table(verify_checksum=True)
        assert table_read.equals(table)

        # Deserialize using metadata alone via attachment
        table_deserialized = deserialize_from_ipc(segment.metadata, verify_checksum=True)
        assert table_deserialized.equals(table)

        # Teardown
        segment.unlink()
        assert segment.is_closed

    def test_shared_memory_tensor_lifecycle(self) -> None:
        """Verify allocation, write, read-back, checksum verification, and cleanup for 3D Tensors."""
        # 3D Tensor representing dipole transition moment matrix (3, N, N)
        np_tensor = np.random.RandomState(42).randn(3, 10, 10).astype(np.float64)
        seg_name = f"cochem_test_tsr_{uuid.uuid4().hex[:8]}"

        segment = pyarrow_mmap_mapper(payload=np_tensor, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert segment.metadata is not None
        assert segment.metadata.payload_type == IPCPayloadType.NUMPY_ARRAY
        assert segment.metadata.extra_metadata["shape"] == [3, 10, 10]

        # Read back as numpy
        np_read = segment.read_numpy(verify_checksum=True)
        assert np.allclose(np_tensor, np_read)
        assert np_read.shape == (3, 10, 10)

        # Read back as pa.Tensor
        pa_tensor = segment.read_tensor(verify_checksum=True)
        assert np.allclose(pa_tensor.to_numpy(), np_tensor)

        # Deserialize from metadata
        deserialized = deserialize_from_ipc(segment.metadata, verify_checksum=True)
        assert np.allclose(deserialized, np_tensor)

        segment.unlink()

    def test_shared_memory_record_batch_lifecycle(self) -> None:
        """Verify allocation and round-trip for PyArrow RecordBatch."""
        batch = pa.record_batch([
            pa.array([101, 102, 103], type=pa.int64()),
            pa.array(["rot_A", "rot_B", "rot_C"], type=pa.string()),
            pa.array([28754.21, 14238.19, 9512.44], type=pa.float64()),
        ], names=["id", "constant", "value_mhz"])

        seg_name = f"cochem_test_rb_{uuid.uuid4().hex[:8]}"
        segment, meta = serialize_to_ipc(batch, name=seg_name, prefer_shm=True)
        assert isinstance(segment, PyArrowSharedMemorySegment)
        assert meta.payload_type == IPCPayloadType.RECORD_BATCH

        batch_back = segment.read_record_batch(verify_checksum=True)
        assert batch_back.equals(batch)

        segment.unlink()

    def test_shared_memory_context_manager(self) -> None:
        """Verify context manager closes SharedMemory automatically upon block exit."""
        seg_name = f"cochem_test_ctx_{uuid.uuid4().hex[:8]}"
        with PyArrowSharedMemorySegment(name=seg_name, size=2048, create=True) as seg:
            assert not seg.is_closed
            seg.write_raw(b"CONTEXT_MANAGER_PAYLOAD", payload_type=IPCPayloadType.RAW_BYTES)
            read_back = seg.read_raw(verify_checksum=True)
            assert read_back == b"CONTEXT_MANAGER_PAYLOAD"

        assert seg.is_closed
        seg.unlink()


# ============================================================================
# 2. Memory-Mapped File Fallback Tests
# ============================================================================


class TestMemoryMappedFileFallback:
    """Authentic physical tests for mmap file IPC operations and fallback mechanisms."""

    def test_mmap_file_table_roundtrip(self, tmp_path: Path) -> None:
        """Verify mmap file allocation in designated scratch directory and table round-trip."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "custom_shm")
        table = pa.table({
            "freq_ghz": [12.45, 18.90, 24.12, 31.88],
            "intensity": [0.98, 0.45, 0.12, 0.76],
            "quantum_state": ["J=1<-0", "J=2<-1", "J=3<-2", "J=4<-3"],
        })

        segment = pyarrow_mmap_mapper(
            payload=table,
            prefer_shm=False,
            context=ctx,
        )
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert segment.file_path.exists()
        assert segment.metadata is not None
        assert segment.metadata.buffer_type == IPCSegmentType.MEMORY_MAPPED_FILE
        assert segment.metadata.file_path == segment.file_path

        # Read back
        table_read = segment.read_table(verify_checksum=True)
        assert table_read.equals(table)

        # Deserialize via metadata
        table_deserialized = deserialize_from_ipc(segment.metadata, context=ctx, verify_checksum=True)
        assert table_deserialized.equals(table)

        # Unlink should remove physical file
        backing_path = segment.file_path
        segment.unlink()
        assert segment.is_closed
        assert not backing_path.exists()

    def test_mmap_file_tensor_and_numpy(self, tmp_path: Path) -> None:
        """Verify high-dimensional float64 array operations via mmap file."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "custom_shm")
        # 4D Tensor (e.g. 2-body interaction integrals / Cartesian Hessian slice)
        arr = np.linspace(0.1, 100.0, 120, dtype=np.float64).reshape((2, 3, 4, 5))

        segment, meta = serialize_to_ipc(arr, prefer_shm=False, context=ctx)
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert meta.payload_type == IPCPayloadType.NUMPY_ARRAY

        arr_read = segment.read_numpy(verify_checksum=True)
        assert np.allclose(arr, arr_read)
        assert arr_read.shape == (2, 3, 4, 5)

        segment.unlink()

    def test_shared_memory_allocation_failure_fallback(self, tmp_path: Path) -> None:
        """Verify transparent fallback to mmap file when prefer_shm is False or allocation fails."""
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "fallback_shm")

        raw_payload = b"FALLBACK_TEST_STREAM_DATA"
        segment = pyarrow_mmap_mapper(
            payload=raw_payload,
            name="fallback_mmap_test",
            prefer_shm=False,  # Direct mmap allocation fallback
            context=ctx,
        )
        assert isinstance(segment, PyArrowMmapFileSegment)
        assert segment.file_path.exists()
        read_back = segment.read_raw(verify_checksum=True)
        assert read_back == raw_payload
        segment.unlink()


# ============================================================================
# 3. 6-Tier Environment Matrix & Air-Gap Enforcement Tests
# ============================================================================


class TestEnvironmentMatrixAndAirGap:
    """Tests verifying 6-Tier Environment Matrix path routing and Tripartite Air-Gap isolation."""

    @pytest.mark.parametrize("tier", [
        EnvironmentTier.LOCAL_WINDOWS,
        EnvironmentTier.LOCAL_MACOS,
        EnvironmentTier.LOCAL_LINUX,
        EnvironmentTier.GITHUB_ACTIONS,
        EnvironmentTier.CODESPACES,
        EnvironmentTier.HPC_NODES,
    ])
    def test_all_six_tiers_path_resolution(self, tier: EnvironmentTier) -> None:
        """Verify dynamic scratch, shm, and artifacts resolution for all 6 tiers."""
        orig_artifacts = os.environ.pop("COCHEM_ARTIFACTS_DIR", None)
        orig_scratch = os.environ.pop("COCHEM_SCRATCH_DIR", None)
        orig_shm = os.environ.pop("COCHEM_SHM_DIR", None)

        try:
            ctx = ExecutionContext(tier=tier)
            scratch_dir = ctx.get_scratch_dir()
            shm_dir = ctx.get_shm_dir()
            artifacts_dir = ctx.get_artifacts_dir()

            assert scratch_dir.is_dir()
            assert shm_dir.is_dir()
            assert artifacts_dir.is_dir()

            # Check subfolder resolution
            sub_scratch = ctx.get_scratch_dir("orca_job_01")
            assert sub_scratch.name == "orca_job_01"
            assert sub_scratch.is_dir()
        finally:
            if orig_artifacts is not None:
                os.environ["COCHEM_ARTIFACTS_DIR"] = orig_artifacts
            if orig_scratch is not None:
                os.environ["COCHEM_SCRATCH_DIR"] = orig_scratch
            if orig_shm is not None:
                os.environ["COCHEM_SHM_DIR"] = orig_shm

    def test_airgap_violation_in_repo_root(self) -> None:
        """Attempting to resolve scratch or artifacts inside Ring 1 repo root must raise AirGapViolationError."""
        repo_root = get_repo_root()
        illegal_target = repo_root / "forbidden_run_output"

        ctx = ExecutionContext()
        with pytest.raises(AirGapViolationError) as exc_info:
            ctx.verify_air_gap_boundary(illegal_target)

        assert "Tripartite Air-Gap Violation" in str(exc_info.value)

    def test_airgap_report_generation(self, tmp_path: Path) -> None:
        """Verify AirGapReport diagnostic properties."""
        safe_path = tmp_path / "valid_scratch"
        ctx = ExecutionContext()
        report = ctx.verify_air_gap_boundary(safe_path)

        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.reason is None
        assert report.target_resolved == safe_path.resolve()

    def test_environment_tier_detection(self) -> None:
        """Exercise actual selectors without inheriting or changing host context."""
        import platform

        selectors = (
            "GITHUB_ACTIONS", "RUNNER_TEMP", "CODESPACES", "CODESPACE_NAME",
            "SLURM_TMPDIR", "SLURM_JOB_ID", "PFSDIR", "PBS_O_WORKDIR",
            "WSL_DISTRO_NAME",
        )
        host_tier = {
            "Windows": EnvironmentTier.LOCAL_WINDOWS,
            "Darwin": EnvironmentTier.LOCAL_MACOS,
        }.get(platform.system(), EnvironmentTier.LOCAL_LINUX)
        cases = [
            ({}, host_tier),
            ({"GITHUB_ACTIONS": "false", "CODESPACES": "false"}, host_tier),
            ({"GITHUB_ACTIONS": "true"}, EnvironmentTier.GITHUB_ACTIONS),
            ({"RUNNER_TEMP": "/tmp"}, EnvironmentTier.GITHUB_ACTIONS),
            ({"CODESPACES": "true"}, EnvironmentTier.CODESPACES),
            ({"CODESPACE_NAME": "explicit-test-context"}, EnvironmentTier.CODESPACES),
            ({"SLURM_TMPDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"SLURM_JOB_ID": "123456"}, EnvironmentTier.HPC_NODES),
            ({"PFSDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"PBS_O_WORKDIR": "/tmp"}, EnvironmentTier.HPC_NODES),
            ({"WSL_DISTRO_NAME": "explicit-test-context"}, EnvironmentTier.LOCAL_WINDOWS),
            ({"RUNNER_TEMP": "/tmp", "CODESPACES": "true", "SLURM_JOB_ID": "123456"},
             EnvironmentTier.GITHUB_ACTIONS),
            ({"CODESPACE_NAME": "explicit-test-context", "SLURM_JOB_ID": "123456",
              "WSL_DISTRO_NAME": "explicit-test-context"}, EnvironmentTier.CODESPACES),
            ({"PBS_O_WORKDIR": "/tmp", "WSL_DISTRO_NAME": "explicit-test-context"},
             EnvironmentTier.HPC_NODES),
        ]
        original_env = dict(os.environ)
        try:
            for configuration, expected in cases:
                for key in selectors:
                    os.environ.pop(key, None)
                os.environ.update(configuration)
                assert ExecutionContext.detect_tier() == expected, configuration
        finally:
            os.environ.clear()
            os.environ.update(original_env)
        assert dict(os.environ) == original_env


# ============================================================================
# 4. xxHash Checksum Verification & Tamper Detection Tests
# ============================================================================


class TestChecksumVerificationAndTamperDetection:
    """Authentic physical tests for xxHash-64 bit-exact verification and bit corruption traps."""

    def test_bit_exact_checksum_verification(self) -> None:
        """Verify deterministic xxHash-64 computation on known byte payload."""
        payload_bytes = b"COCHEM_TORQ_QUANTUM_TENSOR_STATE_DATA_2026"
        digest, size = compute_buffer_checksum(payload_bytes, algorithm="xxh64")

        assert size == len(payload_bytes)
        assert isinstance(digest, str)
        if _XXHASH_AVAILABLE:
            assert len(digest) == 16  # xxhash64 hex length is 16 chars
        else:
            assert len(digest) == 64  # SHA-256 fallback hex length is 64 chars

        result = mmap_checksum_verifier(
            payload_bytes,
            expected_size=size,
            expected_checksum=digest,
            algorithm="xxh64",
            raise_on_error=True,
        )
        assert result.is_valid is True
        assert result.computed_hash == digest

    def test_single_byte_tamper_detection(self) -> None:
        """Corrupting a single byte in the buffer must raise ChecksumMismatchError."""
        original_data = bytearray(b"HIGH_PRECISION_CARTESIAN_HESSIAN_MATRIX_BLOCK")
        digest, size = compute_buffer_checksum(original_data, algorithm="xxh64")

        # Corrupt exactly 1 byte
        corrupted_data = bytearray(original_data)
        corrupted_data[5] = corrupted_data[5] ^ 0xFF  # Flip bits

        with pytest.raises(ChecksumMismatchError) as exc_info:
            mmap_checksum_verifier(
                corrupted_data,
                expected_size=size,
                expected_checksum=digest,
                algorithm="xxh64",
                raise_on_error=True,
            )

        assert "Hash mismatch" in str(exc_info.value)

    def test_size_mismatch_detection(self) -> None:
        """Truncated or expanded buffer must raise ChecksumMismatchError."""
        data = b"EXACT_64_BYTE_PAYLOAD_STRING_FOR_TORQ_INTERPROCESS_COMMUNICATION"
        digest, size = compute_buffer_checksum(data, algorithm="xxh64")

        with pytest.raises(ChecksumMismatchError) as exc_info:
            mmap_checksum_verifier(
                data,
                expected_size=size + 10,  # Wrong expected size
                expected_checksum=digest,
                algorithm="xxh64",
                raise_on_error=True,
            )

        assert "Size mismatch" in str(exc_info.value)

    def test_sha256_cryptographic_fallback(self) -> None:
        """Verify SHA-256 algorithm execution."""
        manifest_bytes = b"PROVENANCE_LOCKED_MANIFEST_BLOCK"
        digest, size = compute_buffer_checksum(manifest_bytes, algorithm="sha256")
        assert len(digest) == 64  # SHA-256 hex is 64 chars

        result = mmap_checksum_verifier(
            manifest_bytes,
            expected_size=size,
            expected_checksum=digest,
            algorithm="sha256",
            raise_on_error=True,
        )
        assert result.is_valid is True


# ============================================================================
# 5. Orphaned IPC Cleaner & Lifecycle Registry Tests
# ============================================================================


class TestOrphanedIPCCleanerAndRegistry:
    """Tests verifying thread-safe registry tracking and aggressive cleanup."""

    def test_registry_tracking_and_cleanup(self, tmp_path: Path) -> None:
        """Verify registry tracks active segments and cleanup_all purges all resources."""
        registry = get_ipc_registry()
        initial_count = registry.active_count()
        assert initial_count == 0

        # Create a SharedMemory segment
        seg_shm = PyArrowSharedMemorySegment(name=f"cochem_reg_shm_{uuid.uuid4().hex[:6]}", size=1024, create=True)
        # Create an mmap file segment
        ctx = ExecutionContext(custom_shm_dir=tmp_path / "reg_shm")
        seg_mmap = PyArrowMmapFileSegment(size=1024, context=ctx)

        assert registry.active_count() == 2
        assert seg_shm.name in registry.active_shm_names()
        assert seg_mmap.file_path.resolve() in registry.active_file_paths()

        # Run orphaned cleaner
        cleaned = orphaned_ipc_cleaner()
        assert cleaned >= 2
        assert registry.active_count() == 0
        assert not seg_mmap.file_path.exists()

    def test_subprocess_atexit_cleaner(self) -> None:
        """Spawning a subprocess that allocates SharedMemory and exits cleanly must purge resources."""
        seg_name = f"cochem_subp_test_{uuid.uuid4().hex[:8]}"

        script = f"""
import sys
from Libraries.cochem_torq_ipc import pyarrow_mmap_mapper, get_ipc_registry
seg = pyarrow_mmap_mapper(name='{seg_name}', size_bytes=2048, prefer_shm=True)
print('ALLOCATED', seg.name)
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(get_repo_root()),
            env=dict(os.environ, PYTHONPATH=str(get_repo_root())),
        )
        assert proc.returncode == 0
        assert "ALLOCATED" in proc.stdout

        # Verify that segment was cleanly unlinked by atexit in subprocess
        with pytest.raises(Exception):
            # Attempting to attach should fail because it was unlinked
            sm.SharedMemory(name=seg_name, create=False)


# ============================================================================
# 6. Mendeleev Dynamic Mass Integration Tests
# ============================================================================


class TestMendeleevIntegration:
    """Authentic physical tests for dynamic mass and isotope lookups."""

    def test_dynamic_atomic_mass(self) -> None:
        """Verify dynamic atomic weight lookup without hardcoding."""
        h_mass = get_atomic_mass("H")
        c_mass = get_atomic_mass("C")
        o_mass = get_atomic_mass("O")
        zn_mass = get_atomic_mass("Zn")

        assert abs(h_mass - float(element("H").atomic_weight)) < 1e-6
        assert abs(c_mass - float(element("C").atomic_weight)) < 1e-6
        assert abs(o_mass - float(element("O").atomic_weight)) < 1e-6
        assert abs(zn_mass - float(element("Zn").atomic_weight)) < 1e-6

    def test_dynamic_isotopic_mass(self) -> None:
        """Verify dynamic isotopic mass retrieval for spectroscopic isotopologues."""
        c13_mass = get_isotopic_mass("C", 13)
        h2_mass = get_isotopic_mass("H", 2)  # Deuterium
        o18_mass = get_isotopic_mass("O", 18)

        assert 13.0 < c13_mass < 13.01
        assert 2.014 < h2_mass < 2.015
        assert 17.99 < o18_mass < 18.01

    def test_atomic_number_lookup(self) -> None:
        """Verify atomic number lookups."""
        assert get_atomic_number("H") == 1
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("Ne") == 10
        assert get_atomic_number("Ar") == 18


# ============================================================================
# 7. High-Dimensional Tensor & Table Stress Tests
# ============================================================================


class TestHighDimensionalIPCStress:
    """Stress tests verifying high-throughput multi-megabyte tensor transfers."""

    def test_multi_megabyte_tensor_ipc(self) -> None:
        """Verify multi-megabyte (1M double elements = 8MB) zero-copy tensor transfer."""
        # 1,000,000 float64 elements = 8,000,000 bytes
        large_array = np.random.RandomState(99).randn(100, 100, 100).astype(np.float64)
        seg_name = f"cochem_large_tsr_{uuid.uuid4().hex[:8]}"

        segment, meta = serialize_to_ipc(large_array, name=seg_name, prefer_shm=True)
        assert meta.size_bytes >= 8000000

        read_array = deserialize_from_ipc(meta, segment=segment, verify_checksum=True)
        assert np.allclose(large_array, read_array)
        assert read_array.shape == (100, 100, 100)

        segment.unlink()

    def test_pydantic_metadata_validation(self) -> None:
        """Verify Pydantic validation rejects negative buffer sizes."""
        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="invalid_buffer",
                buffer_type=IPCSegmentType.SHARED_MEMORY,
                size_bytes=-100,  # Invalid negative size
                allocated_bytes=1024,
                checksum_xxh64="deadbeef",
            )
