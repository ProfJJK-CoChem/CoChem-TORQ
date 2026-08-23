"""CoChem-TORQ: Test Suite for Environment Bootstrapper & IPC Resource Manager.

=============================================================================
Phase 1 (Stage 0.0) Test Suite
------------------------------
Zero-Mock test suite verifying physical air-gap mathematical boundary checks,
dynamic artifact directory routing, Pydantic v2 model validations,
cross-platform memory-mapped buffers (mmap & SharedMemory), deterministic
IPC resource reclamation, and live subprocess atexit exit handlers.

All tests operate against real physical files, shared memory segments,
and subprocess executions within pytest `tmp_path`.
"""

from __future__ import annotations

import mmap
import multiprocessing.shared_memory as sm
import os
import subprocess
import sys
import tempfile
import time
import uuid

# ============================================================================
# Autouse Fixture to Clean IPC Registry Between Tests
# ============================================================================
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_init import (
    _ACTIVE_BUFFERS,
    _ACTIVE_FILE_DESCRIPTORS,
    _ACTIVE_MMAP_OBJECTS,
    _ACTIVE_SCRATCH_PATHS,
    _ACTIVE_SHM_NAMES,
    _ACTIVE_SHM_OBJECTS,
    AirGapReport,
    AirGapViolationError,
    BootstrapperConfig,
    IPCBufferError,
    IPCBufferMetadata,
    bootstrap_environment,
    check_airgap,
    cleanup_ipc_scratch,
    create_ipc_scratch_buffer,
    get_artifact_directory,
    get_scratch_directory,
    register_ipc_cleanup,
    register_mmap_buffer,
    unregister_ipc_cleanup,
    verify_airgap,
)


@pytest.fixture(autouse=True)
def clean_ipc_state() -> Generator[None, None, None]:
    """Ensure clean IPC tracking registry state before and after each test."""
    cleanup_ipc_scratch()
    _ACTIVE_SCRATCH_PATHS.clear()
    _ACTIVE_SHM_NAMES.clear()
    _ACTIVE_SHM_OBJECTS.clear()
    _ACTIVE_MMAP_OBJECTS.clear()
    _ACTIVE_BUFFERS.clear()
    _ACTIVE_FILE_DESCRIPTORS.clear()
    yield
    cleanup_ipc_scratch()
    _ACTIVE_SCRATCH_PATHS.clear()
    _ACTIVE_SHM_NAMES.clear()
    _ACTIVE_SHM_OBJECTS.clear()
    _ACTIVE_MMAP_OBJECTS.clear()
    _ACTIVE_BUFFERS.clear()
    _ACTIVE_FILE_DESCRIPTORS.clear()


# ============================================================================
# 1. Air-Gap Verification Tests
# ============================================================================


class TestAirGapVerification:
    """Tests verifying the mathematical air-gap boundary checks."""

    def test_verify_airgap_disjoint_paths(self, tmp_path: Path) -> None:
        """Disjoint working directory and artifact directory must pass."""
        repo_dir = tmp_path / "repo_root"
        artifacts_dir = tmp_path / "external_artifacts"
        repo_dir.mkdir(parents=True)
        artifacts_dir.mkdir(parents=True)

        assert verify_airgap(cwd=repo_dir, artifacts_dir=artifacts_dir) is True

    def test_verify_airgap_identical_path(self, tmp_path: Path) -> None:
        """Identical working directory and artifact dir must raise error."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=shared_dir, artifacts_dir=shared_dir)
        err_str = str(exc_info.value).lower()
        assert "intersect" in err_str or "overlap" in err_str or "identical" in err_str

    def test_verify_airgap_artifacts_inside_cwd(self, tmp_path: Path) -> None:
        """Artifacts dir inside working directory must raise violation error."""
        repo_dir = tmp_path / "repo_root"
        nested_artifacts = repo_dir / "build" / "artifacts"
        nested_artifacts.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=repo_dir, artifacts_dir=nested_artifacts)
        err_str = str(exc_info.value).lower()
        assert "inside" in err_str or "intersect" in err_str or "overlap" in err_str

    def test_verify_airgap_cwd_inside_artifacts(self, tmp_path: Path) -> None:
        """Working directory inside artifact dir must raise violation error."""
        artifacts_dir = tmp_path / "cochem_artifacts"
        nested_cwd = artifacts_dir / "subproject" / "repo"
        nested_cwd.mkdir(parents=True)

        with pytest.raises(AirGapViolationError) as exc_info:
            verify_airgap(cwd=nested_cwd, artifacts_dir=artifacts_dir)
        err_str = str(exc_info.value).lower()
        assert "inside" in err_str or "intersect" in err_str or "overlap" in err_str

    def test_verify_airgap_string_and_path_inputs(self, tmp_path: Path) -> None:
        """verify_airgap must accept str, Path, and resolve accurately."""
        repo_dir = tmp_path / "repo"
        artifacts_dir = tmp_path / "artifacts"
        repo_dir.mkdir()
        artifacts_dir.mkdir()

        assert (
            verify_airgap(cwd=str(repo_dir), artifacts_dir=str(artifacts_dir))
            is True
        )
        assert verify_airgap(cwd=repo_dir, artifacts_dir=str(artifacts_dir)) is True
        assert verify_airgap(cwd=str(repo_dir), artifacts_dir=artifacts_dir) is True

    def test_verify_airgap_default_resolution(self) -> None:
        """Default verify_airgap call resolves cwd and dynamic artifact dir."""
        result = verify_airgap()
        assert isinstance(result, bool)

    def test_check_airgap_report(self, tmp_path: Path) -> None:
        """check_airgap returns AirGapReport Pydantic model with diagnostics."""
        repo_dir = tmp_path / "repo"
        artifacts_dir = tmp_path / "artifacts"
        repo_dir.mkdir()
        artifacts_dir.mkdir()

        report = check_airgap(cwd=repo_dir, artifacts_dir=artifacts_dir)
        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.cwd_resolved == repo_dir.resolve()
        assert report.artifacts_resolved == artifacts_dir.resolve()
        assert report.reason is None

        # Failing case report
        fail_report = check_airgap(cwd=repo_dir, artifacts_dir=repo_dir)
        assert isinstance(fail_report, AirGapReport)
        assert fail_report.is_valid is False
        assert fail_report.reason is not None


# ============================================================================
# 2. Dynamic Artifact & Scratch Directory Mapping Tests
# ============================================================================


class TestDirectoryMapping:
    """Tests verifying dynamic host environment directory mapping without hardcoding."""

    def test_get_artifact_directory_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """get_artifact_directory must respect host environment variable."""
        custom_dir = tmp_path / "env_artifacts"
        monkeypatch.setenv("COCHEM_ARTIFACTS", str(custom_dir))

        resolved = get_artifact_directory(env_var="COCHEM_ARTIFACTS")
        assert resolved == custom_dir.resolve()
        assert resolved.exists()
        assert resolved.is_dir()

    def test_get_artifact_directory_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """get_artifact_directory must use fallback_dir when env var is unset."""
        monkeypatch.delenv("COCHEM_ARTIFACTS", raising=False)
        fallback = tmp_path / "fallback_artifacts"

        resolved = get_artifact_directory(
            env_var="COCHEM_ARTIFACTS", fallback_dir=fallback
        )
        assert resolved == fallback.resolve()
        assert resolved.exists()
        assert resolved.is_dir()

    def test_get_artifact_directory_default_temp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_artifact_directory falls back to system temp when unset."""
        monkeypatch.delenv("COCHEM_ARTIFACTS", raising=False)

        resolved = get_artifact_directory(
            env_var="COCHEM_ARTIFACTS", fallback_dir=None
        )
        expected_parent = Path(tempfile.gettempdir()).resolve()
        assert (
            resolved.parent == expected_parent
            or str(resolved).startswith(str(expected_parent))
        )
        assert "cochem_artifacts" in resolved.name
        assert resolved.exists()

    def test_get_scratch_directory_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """get_scratch_directory must respect host environment variable."""
        scratch = tmp_path / "fast_scratch"
        monkeypatch.setenv("COCHEM_SCRATCH", str(scratch))

        resolved = get_scratch_directory(env_var="COCHEM_SCRATCH")
        assert resolved == scratch.resolve()
        assert resolved.exists()

    def test_get_scratch_directory_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_scratch_directory creates PID-isolated temp scratch folder."""
        monkeypatch.delenv("COCHEM_SCRATCH", raising=False)

        resolved = get_scratch_directory(env_var="COCHEM_SCRATCH")
        assert "cochem_scratch" in str(resolved)
        assert resolved.exists()
        assert resolved.is_dir()


# ============================================================================
# 3. Pydantic v2 Models Validation Tests
# ============================================================================


class TestPydanticModels:
    """Tests verifying Pydantic v2 metadata and configuration validation models."""

    def test_airgap_report_model(self, tmp_path: Path) -> None:
        """AirGapReport model must be frozen and properly serialize."""
        report = AirGapReport(
            is_valid=True,
            cwd_resolved=tmp_path / "cwd",
            artifacts_resolved=tmp_path / "artifacts",
            verified_at=time.time(),
            reason=None,
        )
        assert report.is_valid is True
        with pytest.raises(ValidationError):
            # Model is frozen
            setattr(report, "is_valid", False)

    def test_ipc_buffer_metadata_model(self, tmp_path: Path) -> None:
        """IPCBufferMetadata model validates types, sizes, and buffer categories."""
        meta = IPCBufferMetadata(
            name="test_buffer",
            buffer_type="mmap",
            size_bytes=1024,
            file_path=tmp_path / "buf.dat",
        )
        assert meta.size_bytes == 1024
        assert meta.buffer_type == "mmap"
        assert meta.is_active is True
        assert isinstance(meta.buffer_id, str)
        assert meta.pid == os.getpid()

    def test_ipc_buffer_metadata_invalid_size_or_type(self) -> None:
        """IPCBufferMetadata must reject non-positive size and invalid buffer types."""
        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="bad_size",
                buffer_type="mmap",
                size_bytes=0,
            )

        with pytest.raises(ValidationError):
            IPCBufferMetadata(
                name="bad_type",
                buffer_type="unsupported_type",  # type: ignore[arg-type]
                size_bytes=1024,
            )

    def test_bootstrapper_config_model(self, tmp_path: Path) -> None:
        """BootstrapperConfig model validates filesystem paths and settings."""
        cfg = BootstrapperConfig(
            artifacts_dir=tmp_path / "artifacts",
            scratch_dir=tmp_path / "scratch",
            env_var="COCHEM_ARTIFACTS",
            enforce_airgap=True,
            clean_on_exit=True,
        )
        assert cfg.enforce_airgap is True
        assert cfg.clean_on_exit is True
        assert cfg.artifacts_dir == tmp_path / "artifacts"


# ============================================================================
# 4. IPC Buffer Creation, Read/Write, and Context Manager Tests
# ============================================================================


class TestIPCScratchBuffer:
    """Tests verifying real memory-mapped files and SharedMemory buffers."""

    def test_create_ipc_scratch_buffer_mmap(self, tmp_path: Path) -> None:
        """create_ipc_scratch_buffer creates a physical mmap buffer."""
        buf = create_ipc_scratch_buffer(
            name="test_mmap_buffer",
            size=2048,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        try:
            assert buf.metadata.buffer_type == "mmap"
            assert buf.size == 2048
            assert buf.path is not None
            assert buf.path.exists()
            assert buf.path.stat().st_size == 2048

            # Write and read data
            payload = b"CoChem-TORQ-Physical-IPC-Data-Payload"
            written = buf.write(payload, offset=0)
            assert written == len(payload)

            read_data = buf.read(size=len(payload), offset=0)
            assert read_data == payload
        finally:
            buf.close()
            buf.unlink()

    def test_create_ipc_scratch_buffer_shm(self) -> None:
        """create_ipc_scratch_buffer creates a POSIX/Windows SharedMemory segment."""
        unique_name = f"cochem_shm_{uuid.uuid4().hex[:8]}"
        buf = create_ipc_scratch_buffer(
            name=unique_name,
            size=1024,
            buffer_type="shm",
            auto_register=True,
        )
        try:
            assert buf.metadata.buffer_type == "shm"
            assert buf.size == 1024
            assert buf.name == unique_name

            payload = b"Quantum-Chemistry-SWMR-Buffer-Test"
            written = buf.write(payload, offset=16)
            assert written == len(payload)

            read_data = buf.read(size=len(payload), offset=16)
            assert read_data == payload
        finally:
            buf.close()
            buf.unlink()

    def test_ipc_buffer_context_manager_mmap(self, tmp_path: Path) -> None:
        """IPCScratchBuffer context manager manages lifecycle and cleanup."""
        target_path: Path | None = None
        with create_ipc_scratch_buffer(
            name="ctx_mmap",
            size=512,
            buffer_type="mmap",
            scratch_dir=tmp_path,
        ) as buf:
            target_path = buf.path
            assert target_path is not None and target_path.exists()
            buf.write(b"Inside-Context-Manager")
            assert buf.read(22) == b"Inside-Context-Manager"

        assert buf.is_closed is True

    def test_ipc_buffer_context_manager_shm(self) -> None:
        """IPCScratchBuffer context manager manages SharedMemory lifecycle."""
        shm_name = f"cochem_shm_ctx_{uuid.uuid4().hex[:8]}"
        with create_ipc_scratch_buffer(
            name=shm_name,
            size=512,
            buffer_type="shm",
        ) as buf:
            buf.write(b"SHM-Context-Payload")
            assert buf.read(19) == b"SHM-Context-Payload"

        assert buf.is_closed is True

    def test_ipc_buffer_boundary_checks(self, tmp_path: Path) -> None:
        """Writing or reading outside buffer boundaries must raise IPCBufferError."""
        with create_ipc_scratch_buffer(size=128, scratch_dir=tmp_path) as buf:
            # Writing payload that exceeds buffer size
            overflow_data = b"X" * 200
            with pytest.raises(IPCBufferError):
                buf.write(overflow_data, offset=0)

            # Writing with offset that exceeds buffer size
            with pytest.raises(IPCBufferError):
                buf.write(b"Hello", offset=150)

            # Reading with invalid offset
            with pytest.raises(IPCBufferError):
                buf.read(size=10, offset=200)

    def test_register_mmap_buffer_helper(self, tmp_path: Path) -> None:
        """register_mmap_buffer must register standalone mmap and backing file."""
        file_path = tmp_path / "raw_mmap.dat"
        file_path.write_bytes(b"\x00" * 256)

        f = open(file_path, "r+b")
        try:
            mm = mmap.mmap(f.fileno(), 256)
        finally:
            f.close()

        register_mmap_buffer(mm, backing_path=file_path)

        mm.write(b"Standalone-MMAP-Payload")
        mm.seek(0)
        assert mm.read(23) == b"Standalone-MMAP-Payload"

        # Execute cleanup
        cleanup_summary = cleanup_ipc_scratch()
        assert cleanup_summary["mmaps_closed"] >= 1
        assert cleanup_summary["files_removed"] >= 1
        assert not file_path.exists()


# ============================================================================
# 5. Deterministic Resource Reclamation & Exit Handlers
# ============================================================================


class TestDeterministicCleanup:
    """Tests verifying deterministic resource reclamation and tracking."""

    def test_cleanup_ipc_scratch_files(self, tmp_path: Path) -> None:
        """cleanup_ipc_scratch removes registered physical files and directories."""
        f1 = tmp_path / "scratch_1.tmp"
        f2 = tmp_path / "scratch_2.tmp"
        d1 = tmp_path / "scratch_dir"
        d1.mkdir()
        (d1 / "nested.tmp").write_text("test")

        f1.write_text("data1")
        f2.write_text("data2")

        register_ipc_cleanup(scratch_paths=[f1, f2, d1])
        assert f1.exists() and f2.exists() and d1.exists()

        summary = cleanup_ipc_scratch()
        assert summary["files_removed"] >= 2
        assert summary["directories_removed"] >= 1
        assert not f1.exists()
        assert not f2.exists()
        assert not d1.exists()

    def test_cleanup_ipc_scratch_shm(self) -> None:
        """cleanup_ipc_scratch unlinks registered SharedMemory segments."""
        shm_name = f"cochem_shm_clean_{uuid.uuid4().hex[:8]}"
        shm_obj = sm.SharedMemory(name=shm_name, create=True, size=256)

        register_ipc_cleanup(shm_names=[shm_name], shm_objects=[shm_obj])
        summary = cleanup_ipc_scratch()
        assert summary["shm_unlinked"] >= 1

        # Verifying segment is unlinked: attempting to open should fail
        with pytest.raises(FileNotFoundError):
            sm.SharedMemory(name=shm_name, create=False)

    def test_cleanup_ipc_scratch_idempotent(self, tmp_path: Path) -> None:
        """cleanup_ipc_scratch must be completely safe to invoke repeatedly."""
        f1 = tmp_path / "idempotent.tmp"
        f1.write_text("test")
        register_ipc_cleanup(scratch_paths=[f1])

        first_summary = cleanup_ipc_scratch()
        assert first_summary["files_removed"] >= 1

        second_summary = cleanup_ipc_scratch()
        assert second_summary["files_removed"] == 0
        assert second_summary["shm_unlinked"] == 0

    def test_targeted_cleanup_does_not_destroy_unrelated_buffers(
        self, tmp_path: Path
    ) -> None:
        """Targeted cleanup must only destroy requested targets, leaving active buffers intact."""
        # Create an active buffer
        buf = create_ipc_scratch_buffer(
            name="persistent_buffer",
            size=512,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        assert buf.is_closed is False
        assert buf in _ACTIVE_BUFFERS

        # Create a targeted scratch file
        targeted_file = tmp_path / "targeted_scratch.tmp"
        targeted_file.write_text("to be deleted")
        register_ipc_cleanup(scratch_paths=[targeted_file])

        # Run targeted cleanup for the single file
        summary = cleanup_ipc_scratch(scratch_paths=[targeted_file])
        assert summary["files_removed"] == 1
        assert not targeted_file.exists()

        # The active buffer MUST NOT have been destroyed or closed
        assert buf.is_closed is False
        assert buf in _ACTIVE_BUFFERS
        assert buf.read(5) == b"\x00\x00\x00\x00\x00"

        # Now clean up the buffer
        buf.unlink()
        assert buf.is_closed is True
        assert buf not in _ACTIVE_BUFFERS

    def test_buffer_unlink_deregisters_from_global_state(
        self, tmp_path: Path
    ) -> None:
        """Calling unlink on a buffer must remove its references from global tracking sets."""
        buf = create_ipc_scratch_buffer(
            name="unlink_dereg_test",
            size=256,
            buffer_type="mmap",
            scratch_dir=tmp_path,
            auto_register=True,
        )
        assert buf in _ACTIVE_BUFFERS
        assert buf.path in _ACTIVE_SCRATCH_PATHS

        buf.unlink()

        # Global registries must be cleared of this buffer and its path
        assert buf not in _ACTIVE_BUFFERS
        assert buf.path not in _ACTIVE_SCRATCH_PATHS


# ============================================================================
# 6. Live Subprocess Exit Handler (atexit) Verification
# ============================================================================


class TestSubprocessExitHandlers:
    """Tests executing real child processes to verify atexit cleanup upon exit."""

    def test_live_subprocess_atexit_cleanup_mmap(self, tmp_path: Path) -> None:
        """Child process creates an mmap IPC buffer; on exit, buffer is cleaned up."""
        scratch_dir = tmp_path / "proc_scratch"
        scratch_dir.mkdir()

        script = f"""
import sys
from pathlib import Path
from Libraries.cochem_torq_init import create_ipc_scratch_buffer

buf = create_ipc_scratch_buffer(
    name="child_proc_mmap",
    size=1024,
    buffer_type="mmap",
    scratch_dir=r"{scratch_dir}",
    auto_register=True
)
buf.write(b"Subprocess-Test-Data")
print("BUFFER_CREATED:" + str(buf.path))
sys.stdout.flush()
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=15,
        )
        assert proc.returncode == 0, f"Process failed: {proc.stderr}"
        assert "BUFFER_CREATED:" in proc.stdout

        for line in proc.stdout.splitlines():
            if line.startswith("BUFFER_CREATED:"):
                created_path = Path(line.split(":", 1)[1].strip())
                assert not created_path.exists(), (
                    f"Scratch file {created_path} still exists after process exit!"
                )

    def test_live_subprocess_atexit_cleanup_shm(self) -> None:
        """Child process creates SharedMemory buffer; on exit, shm is unlinked."""
        shm_name = f"cochem_subproc_shm_{uuid.uuid4().hex[:8]}"

        script = f"""
import sys
from Libraries.cochem_torq_init import create_ipc_scratch_buffer

buf = create_ipc_scratch_buffer(
    name="{shm_name}",
    size=512,
    buffer_type="shm",
    auto_register=True
)
buf.write(b"SHM-Subprocess-Payload")
print("SHM_CREATED:" + "{shm_name}")
sys.stdout.flush()
sys.exit(0)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=15,
        )
        assert proc.returncode == 0, f"Process failed: {proc.stderr}"
        assert "SHM_CREATED:" in proc.stdout

        with pytest.raises(FileNotFoundError):
            sm.SharedMemory(name=shm_name, create=False)


# ============================================================================
# 7. Environment Bootstrapper Integration Tests
# ============================================================================


class TestBootstrapEnvironment:
    """Tests verifying the full bootstrap_environment lifecycle."""

    def test_bootstrap_environment_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """bootstrap_environment resolves paths and registers IPC cleanup."""
        artifacts_dir = tmp_path / "artifacts"
        scratch_dir = tmp_path / "scratch"
        monkeypatch.setenv("COCHEM_ARTIFACTS", str(artifacts_dir))
        monkeypatch.setenv("COCHEM_SCRATCH", str(scratch_dir))

        config = bootstrap_environment(
            artifacts_env="COCHEM_ARTIFACTS",
            scratch_env="COCHEM_SCRATCH",
            enforce_airgap=True,
        )
        assert isinstance(config, BootstrapperConfig)
        assert config.artifacts_dir == artifacts_dir.resolve()
        assert config.scratch_dir == scratch_dir.resolve()
        assert config.artifacts_dir.exists()
        assert config.scratch_dir.exists()

    def test_bootstrap_environment_airgap_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """bootstrap_environment raises AirGapViolationError if artifacts in cwd."""
        repo_cwd = Path.cwd()
        nested_artifacts = repo_cwd / "test_nested_artifacts_violation"
        monkeypatch.setenv("COCHEM_ARTIFACTS", str(nested_artifacts))

        with pytest.raises(AirGapViolationError):
            bootstrap_environment(
                artifacts_env="COCHEM_ARTIFACTS",
                enforce_airgap=True,
            )

    def test_bootstrap_environment_no_enforce(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """bootstrap_environment proceeds without error if enforce_airgap=False."""
        repo_cwd = Path.cwd()
        nested_artifacts = repo_cwd / "test_nested_artifacts_no_enforce"
        monkeypatch.setenv("COCHEM_ARTIFACTS", str(nested_artifacts))

        config = bootstrap_environment(
            artifacts_env="COCHEM_ARTIFACTS",
            enforce_airgap=False,
        )
        assert config.artifacts_dir == nested_artifacts.resolve()
        assert config.enforce_airgap is False
        if nested_artifacts.exists():
            nested_artifacts.rmdir()

    def test_bootstrap_environment_tripartite_scratch_inside_artifacts_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """bootstrap_environment raises AirGapViolationError if scratch is inside artifacts."""
        artifacts_dir = tmp_path / "artifacts"
        nested_scratch = artifacts_dir / "nested_scratch"
        monkeypatch.setenv("COCHEM_ARTIFACTS", str(artifacts_dir))
        monkeypatch.setenv("COCHEM_SCRATCH", str(nested_scratch))

        with pytest.raises(AirGapViolationError):
            bootstrap_environment(
                artifacts_env="COCHEM_ARTIFACTS",
                scratch_env="COCHEM_SCRATCH",
                enforce_airgap=True,
            )


# ============================================================================
# 8. Air-Gap Compliance Runtime Test
# ============================================================================


class TestAirGapRepositoryIntegrity:
    """Asserts that no test or bootstrapper logic wrote runtime files to repo."""

    def test_no_runtime_writes_to_repo(self, tmp_path: Path) -> None:
        """Verify that scratch buffers do not produce repository artifacts."""
        repo_path = Path(__file__).parent.parent.resolve()
        initial_repo_files = {
            p for p in repo_path.glob("**/*") if not p.name.startswith(".")
        }

        scratch_dir = tmp_path / "airgap_scratch"
        with create_ipc_scratch_buffer(
            name="integrity_test",
            size=1024,
            buffer_type="mmap",
            scratch_dir=scratch_dir,
        ) as buf:
            buf.write(b"Air-gap runtime data")

        current_repo_files = {
            p for p in repo_path.glob("**/*") if not p.name.startswith(".")
        }
        new_repo_files = {
            f
            for f in (current_repo_files - initial_repo_files)
            if "__pycache__" not in str(f) and ".pytest_cache" not in str(f)
        }
        assert (
            len(new_repo_files) == 0
        ), f"Unexpected runtime files in repository: {new_repo_files}"
