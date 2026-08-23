"""CoChem-TORQ: Test Suite for SWMR Zombie Lock Reaper & HDF5 Healer.

=================================================================
Phase 1 (Stage 0.0) Test Suite
------------------------------
Zero-Mock test suite verifying physical SWMR lock management, multi-tier zombie
detection (local PID existence, PID recycling, and Slurm HPC states), live subprocess
crash simulations, atomic collision handling, and superblock recovery.

All tests operate against real physical files within pytest `tmp_path`.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import psutil
import pytest

from Libraries.cochem_h5_healer import (
    SLURM_ACTIVE_STATES,
    SLURM_TERMINAL_STATES,
    LockMetadata,
    SWMRWriteContext,
    _check_local_pid,
    _check_slurm_job_status,
    _is_local_host,
    acquire_swmr_lock,
    detect_zombie_pids,
    force_release_swmr,
    get_lock_path,
    heal_swmr_database,
    read_lock_metadata,
    release_swmr_lock,
    verify_h5_swmr_integrity,
)

# ============================================================================
# Helpers & Fixtures
# ============================================================================


def _find_unused_pid() -> int:
    """Find a non-existent PID in the current operating system process table."""
    candidate = 999000
    while psutil.pid_exists(candidate):
        candidate += 1
    return candidate


# ============================================================================
# 1. Lock Path Resolution & Pydantic Serialization Tests
# ============================================================================


class TestLockMetadataAndPaths:
    """Tests for lock file pathing and Pydantic v2 LockMetadata model."""

    def test_get_lock_path_formats(self, tmp_path: Path) -> None:
        """Verify lock path suffixing for Path and str inputs."""
        h5_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(h5_path)
        assert lock_path == tmp_path / "landscape.h5.lock"

        str_path = str(tmp_path / "quantum_grid.h5")
        lock_str_path = get_lock_path(str_path)
        assert lock_str_path == Path(str_path + ".lock")

    def test_lock_metadata_serialization_roundtrip(
        self, tmp_path: Path
    ) -> None:
        """Verify Pydantic v2 serialization, JSON writing, and deserialization."""
        meta = LockMetadata(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            slurm_job_id="123456",
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
            extra={"cluster": "stampede3", "queue": "gpu-a100"},
        )

        lock_path = tmp_path / "test.h5.lock"
        lock_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        loaded = read_lock_metadata(lock_path)
        assert loaded is not None
        assert loaded.pid == meta.pid
        assert loaded.hostname == meta.hostname
        assert loaded.slurm_job_id == "123456"
        assert loaded.session_id == meta.session_id
        assert loaded.extra["cluster"] == "stampede3"

    def test_read_lock_metadata_nonexistent(self, tmp_path: Path) -> None:
        """Verify reading nonexistent lock file safely returns None."""
        assert read_lock_metadata(tmp_path / "absent.lock") is None

    def test_read_lock_metadata_empty_file(self, tmp_path: Path) -> None:
        """Verify reading an empty 0-byte lock file safely returns None."""
        empty_lock = tmp_path / "empty.lock"
        empty_lock.write_text("", encoding="utf-8")
        assert read_lock_metadata(empty_lock) is None

    def test_read_lock_metadata_corrupted_json(self, tmp_path: Path) -> None:
        """Verify reading invalid JSON returns None without unhandled exceptions."""
        bad_json = tmp_path / "bad.lock"
        bad_json.write_text(
            '{"pid": 1234, "hostname": "broken...', encoding="utf-8"
        )
        assert read_lock_metadata(bad_json) is None

    def test_read_lock_metadata_schema_mismatch(self, tmp_path: Path) -> None:
        """Verify reading JSON that violates LockMetadata schema returns None."""
        invalid_schema = tmp_path / "invalid.lock"
        invalid_schema.write_text(
            '{"pid": "not_an_int", "hostname": 12345}', encoding="utf-8"
        )
        assert read_lock_metadata(invalid_schema) is None

        non_dict = tmp_path / "array.lock"
        non_dict.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_lock_metadata(non_dict) is None

    def test_read_lock_metadata_os_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify read_lock_metadata handles PermissionError and OSError gracefully."""
        lock_path = tmp_path / "perm.lock"
        lock_path.write_text("{}", encoding="utf-8")

        def mock_read_perm(*args: object, **kwargs: object) -> str:
            raise PermissionError("Access denied")

        monkeypatch.setattr(Path, "read_text", mock_read_perm)
        assert read_lock_metadata(lock_path) is None

        def mock_read_os(*args: object, **kwargs: object) -> str:
            raise OSError("I/O error")

        monkeypatch.setattr(Path, "read_text", mock_read_os)
        assert read_lock_metadata(lock_path) is None


# ============================================================================
# 2. Process Validation & Zombie Detection Tests (Zero-Mock)
# ============================================================================


class TestZombieDetection:
    """Tests for multi-tier zombie PID detection under real OS processes."""

    def test_detect_no_lock(self, tmp_path: Path) -> None:
        """Verify detect_zombie_pids returns (False, None) when no lock exists."""
        db_path = tmp_path / "landscape.h5"
        is_zombie, meta = detect_zombie_pids(db_path)
        assert is_zombie is False
        assert meta is None

    def test_detect_active_current_process(self, tmp_path: Path) -> None:
        """Verify detect_zombie_pids identifies running process as active."""
        db_path = tmp_path / "landscape.h5"
        meta = acquire_swmr_lock(db_path)

        try:
            is_zombie, detected_meta = detect_zombie_pids(db_path)
            assert is_zombie is False
            assert detected_meta is not None
            assert detected_meta["pid"] == os.getpid()
            assert detected_meta["hostname"] == socket.gethostname()
            assert "actively running" in detected_meta["zombie_reason"].lower()
        finally:
            release_swmr_lock(db_path, session_id=meta.session_id)

    def test_detect_zombie_nonexistent_pid(self, tmp_path: Path) -> None:
        """Verify detect_zombie_pids detects a dead/nonexistent PID as a zombie."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)
        dead_pid = _find_unused_pid()

        stale_meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            created_at=time.time() - 100.0,
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(
            stale_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
        assert detected["pid"] == dead_pid
        assert "does not exist" in detected["zombie_reason"]

    def test_detect_zombie_recycled_pid(self, tmp_path: Path) -> None:
        """Verify PID recycling detection against process create_time."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)

        curr_proc = psutil.Process(os.getpid())
        proc_start = curr_proc.create_time()

        # Set lock creation timestamp well BEFORE the process was started
        recycled_lock_time = proc_start - 3600.0

        recycled_meta = LockMetadata(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            created_at=recycled_lock_time,
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(
            recycled_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
        assert "recycled" in detected["zombie_reason"].lower()

    def test_detect_zombie_corrupted_lock(self, tmp_path: Path) -> None:
        """Verify a corrupted lock file is classified as a zombie lock."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)
        lock_path.write_text("{corrupt-garbage-bytes", encoding="utf-8")

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
        assert detected.get("corrupted") is True

    def test_detect_remote_host_without_slurm_defaults_to_alive(
        self, tmp_path: Path
    ) -> None:
        """Verify remote lock without Slurm queue safely defaults to active."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)

        remote_meta = LockMetadata(
            pid=4567,
            hostname="unreachable-remote-hpc-node.internal",
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(
            remote_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is False
        assert detected is not None
        assert "assumed alive" in detected["zombie_reason"]

    def test_detect_remote_host_with_slurm_job(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify detect_zombie_pids handles distributed HPC remote jobs."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)

        remote_meta = LockMetadata(
            pid=4567,
            hostname="compute-node-104.cluster",
            slurm_job_id="998877",
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(
            remote_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        # Terminal slurm state -> zombie
        monkeypatch.setattr(shutil, "which", lambda _exe: "/usr/bin/squeue")

        class MockCompletedProcessTerminal:
            returncode = 0
            stdout = "COMPLETED\n"
            stderr = ""

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: MockCompletedProcessTerminal(),
        )

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
        assert "terminal state" in detected["zombie_reason"].lower()

        # Active slurm state -> not zombie
        class MockCompletedProcessActive:
            returncode = 0
            stdout = "RUNNING\n"
            stderr = ""

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: MockCompletedProcessActive(),
        )

        is_zombie_act, detected_act = detect_zombie_pids(db_path)
        assert is_zombie_act is False
        assert detected_act is not None
        assert "active in queue" in detected_act["zombie_reason"].lower()

    def test_check_local_pid_access_denied_simulation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AccessDenied when inspecting a process defaults to alive."""

        def mock_process(pid: int) -> psutil.Process:
            raise psutil.AccessDenied(pid=pid)

        monkeypatch.setattr(psutil, "Process", mock_process)
        monkeypatch.setattr(psutil, "pid_exists", lambda _pid: True)

        is_zombie, reason = _check_local_pid(os.getpid(), time.time())
        assert is_zombie is False
        assert "access denied" in reason.lower()

    def test_check_local_pid_no_such_process_simulation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify NoSuchProcess during inspection is classified as zombie."""

        def mock_process(pid: int) -> psutil.Process:
            raise psutil.NoSuchProcess(pid=pid)

        monkeypatch.setattr(psutil, "Process", mock_process)
        monkeypatch.setattr(psutil, "pid_exists", lambda _pid: True)

        is_zombie, reason = _check_local_pid(999999, time.time())
        assert is_zombie is True
        assert "terminated during inspection" in reason.lower()

    def test_check_local_pid_zombie_status_simulation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify process reporting zombie/dead status is classified as zombie."""

        class MockProc:
            def create_time(self) -> float:
                return time.time() - 10.0

            def status(self) -> str:
                return str(getattr(psutil, "STATUS_ZOMBIE", "zombie"))

        monkeypatch.setattr(psutil, "Process", lambda _pid: MockProc())
        monkeypatch.setattr(psutil, "pid_exists", lambda _pid: True)

        is_zombie, reason = _check_local_pid(12345, time.time())
        assert is_zombie is True
        assert "status" in reason.lower()


# ============================================================================
# 3. Live Subprocess Crash & SWMR Recovery Simulation (Zero-Mock)
# ============================================================================


class TestSubprocessCrashRecovery:
    """Zero-Mock crash simulation killing live worker subprocess."""

    def test_subprocess_crash_and_zombie_reaper_recovery(
        self, tmp_path: Path
    ) -> None:
        """Physical crash test.

        1. Subprocess opens HDF5 in SWMR mode and acquires lock.
        2. Writes real data and flushes.
        3. Forcefully killed with proc.kill().
        4. detect_zombie_pids identifies dead process.
        5. force_release_swmr clears lock and flushes superblock.
        6. verify_h5_swmr_integrity validates file and data readability.
        """
        db_path = tmp_path / "crashed_landscape.h5"
        worker_script = tmp_path / "swmr_worker.py"
        proj_root = Path(__file__).resolve().parent.parent

        # Worker script that creates SWMR dataset, acquires lock, and hangs
        script_code = f"""
import os
import sys
import time
import h5py
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r"{proj_root}")
from Libraries.cochem_h5_healer import acquire_swmr_lock

db_file = r"{db_path}"
with h5py.File(db_file, mode="w", libver="latest") as f:
    dset = f.create_dataset("energies", (100,), dtype="float64")
    dset[:] = np.linspace(-100.0, 0.0, 100)
    f.swmr_mode = True
    f.flush()

meta = acquire_swmr_lock(db_file, session_id="crash-worker-session")
sys.stdout.write("LOCK_ACQUIRED\\n")
sys.stdout.flush()

# Hang until forcefully terminated by test harness
time.sleep(120)
"""
        worker_script.write_text(script_code, encoding="utf-8")

        # Launch real subprocess
        proc = subprocess.Popen(
            [sys.executable, str(worker_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for worker to signal lock acquisition
            assert proc.stdout is not None
            line = proc.stdout.readline().strip()
            assert line == "LOCK_ACQUIRED"

            # Check that the lock is held and recognized as active
            is_zombie, meta = detect_zombie_pids(db_path)
            assert is_zombie is False
            assert meta is not None
            assert meta["pid"] == proc.pid

            # Attempting force_release without force_override must raise BlockingIOError
            with pytest.raises(BlockingIOError) as exc_info:
                force_release_swmr(db_path, force_override=False)
            assert f"actively locked by PID {proc.pid}" in str(exc_info.value)

            # Forcefully kill the subprocess (SIGKILL / TerminateProcess)
            proc.kill()
            proc.wait(timeout=5)

            # Assert process is terminated
            assert not psutil.pid_exists(proc.pid) or proc.poll() is not None

            # detect_zombie_pids must now classify the lock as a zombie
            is_zombie_after, meta_after = detect_zombie_pids(db_path)
            assert is_zombie_after is True
            assert meta_after is not None

            # Reaping the zombie lock must succeed
            healed = force_release_swmr(db_path)
            assert healed is True

            # Lock file must be unlinked
            assert not get_lock_path(db_path).exists()

            # Physical HDF5 integrity check
            assert verify_h5_swmr_integrity(db_path) is True

            # Read back physical data written before crash
            with h5py.File(db_path, mode="r", libver="latest", swmr=True) as f:
                assert "energies" in f
                dset = f["energies"][:]
                assert len(dset) == 100
                assert dset[0] == pytest.approx(-100.0)
                assert dset[-1] == pytest.approx(0.0)

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


# ============================================================================
# 4. Atomic Lock Acquisition, Collision, & Timeout Tests
# ============================================================================


class TestAtomicLockingAndHealing:
    """Tests for atomic acquisition, collision handling, and timeout semantics."""

    def test_acquire_and_release_lifecycle(self, tmp_path: Path) -> None:
        """Verify standard lock acquisition and clean release."""
        db_path = tmp_path / "landscape.h5"
        meta = acquire_swmr_lock(db_path, session_id="test-session-123")

        lock_path = get_lock_path(db_path)
        assert lock_path.exists()
        assert meta.session_id == "test-session-123"
        assert meta.pid == os.getpid()

        # Release lock cleanly
        released = release_swmr_lock(db_path, session_id="test-session-123")
        assert released is True
        assert not lock_path.exists()

    def test_acquire_nested_directory_creation(self, tmp_path: Path) -> None:
        """Verify acquire_swmr_lock creates non-existent parent directories."""
        nested_db = tmp_path / "deep" / "nested" / "sub" / "landscape.h5"
        meta = acquire_swmr_lock(nested_db, session_id="nested-session")
        assert get_lock_path(nested_db).exists()
        release_swmr_lock(nested_db, session_id=meta.session_id)

    def test_acquire_timeout_on_active_lock(self, tmp_path: Path) -> None:
        """Verify TimeoutError is raised when another active session holds lock."""
        db_path = tmp_path / "landscape.h5"
        meta1 = acquire_swmr_lock(db_path, session_id="session-primary")

        try:
            with pytest.raises(TimeoutError) as exc_info:
                acquire_swmr_lock(
                    db_path,
                    timeout=0.4,
                    retry_interval=0.1,
                    auto_heal=False,
                    session_id="session-secondary",
                )
            assert "Timed out" in str(exc_info.value)
        finally:
            release_swmr_lock(db_path, session_id=meta1.session_id)

    def test_acquire_auto_heal_stale_lock(self, tmp_path: Path) -> None:
        """Verify acquire_swmr_lock automatically reaps stale lock."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)
        dead_pid = _find_unused_pid()

        stale_meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            created_at=time.time() - 50.0,
            session_id="stale-dead-session",
        )
        lock_path.write_text(
            stale_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        new_meta = acquire_swmr_lock(
            db_path, timeout=2.0, auto_heal=True, session_id="fresh-session"
        )

        assert new_meta.pid == os.getpid()
        assert new_meta.session_id == "fresh-session"
        assert lock_path.exists()

        # Clean up
        release_swmr_lock(db_path, session_id=new_meta.session_id)
        assert not lock_path.exists()

    def test_release_swmr_lock_session_mismatch(self, tmp_path: Path) -> None:
        """Verify release_swmr_lock rejects release if session ID mismatches."""
        db_path = tmp_path / "landscape.h5"
        acquire_swmr_lock(db_path, session_id="correct-session")

        try:
            with pytest.raises(PermissionError) as exc_info:
                release_swmr_lock(
                    db_path, session_id="wrong-session", force=False
                )
            assert "session ID mismatch" in str(exc_info.value)

            # Releasing with force=True must bypass session check
            assert (
                release_swmr_lock(
                    db_path, session_id="wrong-session", force=True
                )
                is True
            )
            assert not get_lock_path(db_path).exists()
        finally:
            if get_lock_path(db_path).exists():
                release_swmr_lock(db_path, force=True)

    def test_release_swmr_lock_pid_mismatch_without_session(
        self, tmp_path: Path
    ) -> None:
        """Verify release_swmr_lock rejects release if caller PID mismatches."""
        db_path = tmp_path / "landscape.h5"
        lock_path = get_lock_path(db_path)

        other_pid = os.getpid() + 1
        other_meta = LockMetadata(
            pid=other_pid,
            hostname=socket.gethostname(),
            created_at=time.time(),
            session_id=None,
        )
        lock_path.write_text(
            other_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        with pytest.raises(PermissionError) as exc_info:
            release_swmr_lock(db_path, session_id=None, force=False)
        assert f"owned by PID {other_pid}" in str(exc_info.value)

        # Force release cleans it up
        assert release_swmr_lock(db_path, force=True) is True
        assert not lock_path.exists()

    def test_force_release_override_on_active_lock(
        self, tmp_path: Path
    ) -> None:
        """Verify force_release_swmr with force_override=True clears locks."""
        db_path = tmp_path / "landscape.h5"
        acquire_swmr_lock(db_path)

        assert get_lock_path(db_path).exists()
        released = force_release_swmr(db_path, force_override=True)
        assert released is True
        assert not get_lock_path(db_path).exists()

    def test_force_release_nonexistent_lock_and_absent_db(
        self, tmp_path: Path
    ) -> None:
        """Verify force_release_swmr returns True when files are absent."""
        db_path = tmp_path / "nonexistent.h5"
        assert force_release_swmr(db_path) is True


# ============================================================================
# 5. SWMRWriteContext Context Manager Tests
# ============================================================================


class TestSWMRWriteContext:
    """Tests for the SWMRWriteContext manager."""

    def test_swmr_write_context_basic(self, tmp_path: Path) -> None:
        """Verify context manager acquires lock and cleans up on exit."""
        db_path = tmp_path / "context_landscape.h5"
        lock_path = get_lock_path(db_path)

        with SWMRWriteContext(db_path, mode="w") as f:
            assert lock_path.exists()
            assert isinstance(f, h5py.File)
            dset = f.create_dataset("grid_points", (50, 3), dtype="float32")
            dset[:] = np.ones((50, 3), dtype="float32")
            f.flush()

        # After context exit, lock must be released and file must be closed
        assert not lock_path.exists()

        # Reopen in read mode to verify data
        with h5py.File(db_path, mode="r") as f:
            assert "grid_points" in f
            assert f["grid_points"].shape == (50, 3)

    def test_swmr_write_context_exception_cleanup(
        self, tmp_path: Path
    ) -> None:
        """Verify lock is released even if exception occurs within context."""
        db_path = tmp_path / "context_error.h5"
        lock_path = get_lock_path(db_path)

        with pytest.raises(RuntimeError, match="Simulation Failure"):
            with SWMRWriteContext(db_path, mode="w") as f:
                f.create_dataset("test", data=[1, 2, 3])
                assert lock_path.exists()
                raise RuntimeError("Simulation Failure")

        assert not lock_path.exists()

    def test_swmr_write_context_auto_heal_on_enter(
        self, tmp_path: Path
    ) -> None:
        """Verify context manager automatically heals stale lock upon entry."""
        db_path = tmp_path / "autoheal_context.h5"
        lock_path = get_lock_path(db_path)
        dead_pid = _find_unused_pid()

        stale_meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            created_at=time.time() - 60.0,
            session_id="dead-session",
        )
        lock_path.write_text(
            stale_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        with SWMRWriteContext(db_path, mode="w", auto_heal=True) as f:
            f.create_dataset("recovered", data=[42])
            # Lock should now belong to current process
            curr_meta = read_lock_metadata(lock_path)
            assert curr_meta is not None
            assert curr_meta.pid == os.getpid()

        assert not lock_path.exists()

    def test_swmr_write_context_failure_on_open_releases_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify that if opening h5py.File fails, the acquired lock is released."""
        db_path = tmp_path / "open_fail.h5"

        def mock_h5_file(*args: object, **kwargs: object) -> h5py.File:
            raise OSError("Corrupted file header")

        monkeypatch.setattr(h5py, "File", mock_h5_file)

        ctx = SWMRWriteContext(db_path)
        with pytest.raises(OSError, match="Corrupted file header"):
            with ctx:
                pass

        assert not get_lock_path(db_path).exists()


# ============================================================================
# 6. Concurrent Multi-Reader SWMR Validation (Zero-Mock)
# ============================================================================


class TestSWMRConcurrentMultiReader:
    """Zero-Mock verification of concurrent readers accessing live SWMR writer."""

    def test_swmr_concurrent_multi_reader(self, tmp_path: Path) -> None:
        """Verify SWMR single-writer concurrent multiple-reader protocol.

        1. Writer creates expandable dataset in SWMR mode and flushes.
        2. Reader 1 and Reader 2 open file concurrently in read SWMR mode.
        3. Writer extends dataset and flushes.
        4. Readers refresh and observe newly appended data.
        """
        db_path = tmp_path / "swmr_stream.h5"

        with SWMRWriteContext(db_path, mode="w") as writer:
            # Create chunked, extendable dataset
            dset = writer.create_dataset(
                "stream",
                shape=(10, 4),
                maxshape=(None, 4),
                chunks=(10, 4),
                dtype="float64",
            )
            dset[:] = np.full((10, 4), 1.0)
            writer.flush()

            # Reader 1 opens file
            reader1 = h5py.File(db_path, mode="r", libver="latest", swmr=True)
            # Reader 2 opens file
            reader2 = h5py.File(db_path, mode="r", libver="latest", swmr=True)

            try:
                assert reader1["stream"].shape == (10, 4)
                assert reader2["stream"].shape == (10, 4)

                # Writer appends additional rows
                dset.resize((20, 4))
                dset[10:20] = np.full((10, 4), 2.0)
                dset.flush()

                # Readers refresh dataset view
                reader1["stream"].refresh()
                reader2["stream"].refresh()

                assert reader1["stream"].shape == (20, 4)
                assert reader2["stream"].shape == (20, 4)
                assert reader1["stream"][15, 0] == pytest.approx(2.0)
                assert reader2["stream"][15, 0] == pytest.approx(2.0)

            finally:
                reader1.close()
                reader2.close()


# ============================================================================
# 7. Slurm State Parsing & Distributed Edge Cases
# ============================================================================


class TestSlurmStateHandling:
    """Tests for Slurm scheduler state evaluation and HPC edge cases."""

    def test_slurm_terminal_states_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify terminal Slurm states evaluate to zombie status."""
        for state in SLURM_TERMINAL_STATES:
            monkeypatch.setattr(
                shutil,
                "which",
                lambda exe: "/usr/bin/squeue" if exe == "squeue" else None,
            )

            class MockCompletedProcess:
                returncode = 0
                stdout = f"{state}\n"
                stderr = ""

            monkeypatch.setattr(
                subprocess,
                "run",
                lambda *args, **kwargs: MockCompletedProcess(),
            )

            is_zombie, reason = _check_slurm_job_status("999111")
            assert (
                is_zombie is True
            ), f"State {state} should be classified as terminal zombie"
            assert "terminal state" in reason.lower()

    def test_slurm_active_states_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify active Slurm states evaluate to non-zombie status."""
        for state in SLURM_ACTIVE_STATES:
            monkeypatch.setattr(
                shutil,
                "which",
                lambda exe: "/usr/bin/squeue" if exe == "squeue" else None,
            )

            class MockCompletedProcess:
                returncode = 0
                stdout = f"{state}\n"
                stderr = ""

            monkeypatch.setattr(
                subprocess,
                "run",
                lambda *args, **kwargs: MockCompletedProcess(),
            )

            is_zombie, reason = _check_slurm_job_status("999222")
            assert (
                is_zombie is False
            ), f"State {state} should be classified as active"
            assert "active in queue" in reason.lower()

    def test_slurm_job_not_in_queue_is_zombie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify job missing from squeue is classified as zombie."""
        monkeypatch.setattr(
            shutil,
            "which",
            lambda exe: "/usr/bin/squeue" if exe == "squeue" else None,
        )

        class MockCompletedProcess:
            returncode = 1
            stdout = ""
            stderr = "slurm_load_jobs error: Invalid job id specified"

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: MockCompletedProcess(),
        )

        is_zombie, reason = _check_slurm_job_status("999333")
        assert is_zombie is True
        assert "not found in active squeue" in reason.lower()

    def test_slurm_squeue_missing_defaults_safely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify when squeue is absent, status safely defaults to not zombie."""
        monkeypatch.setattr(shutil, "which", lambda _exe: None)
        is_zombie, reason = _check_slurm_job_status("12345")
        assert is_zombie is False
        assert "not available" in reason.lower()

    def test_slurm_squeue_timeout_defaults_safely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify when squeue query times out, status safely defaults to alive."""
        monkeypatch.setattr(shutil, "which", lambda _exe: "/usr/bin/squeue")

        def mock_timeout(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=["squeue"], timeout=10)

        monkeypatch.setattr(subprocess, "run", mock_timeout)
        is_zombie, reason = _check_slurm_job_status("12345")
        assert is_zombie is False
        assert "timed out" in reason.lower()

    def test_slurm_squeue_subprocess_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify subprocess execution error in squeue is handled cleanly."""
        monkeypatch.setattr(shutil, "which", lambda _exe: "/usr/bin/squeue")

        def mock_error(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.SubprocessError("Failed to execute")

        monkeypatch.setattr(subprocess, "run", mock_error)
        is_zombie, reason = _check_slurm_job_status("12345")
        assert is_zombie is False
        assert "execution error" in reason.lower()

    def test_slurm_squeue_unrecognized_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify unrecognized status from squeue safely defaults to alive."""
        monkeypatch.setattr(shutil, "which", lambda _exe: "/usr/bin/squeue")

        class MockUnrecognized:
            returncode = 0
            stdout = "MYSTERIOUS_STATUS\n"
            stderr = ""

        monkeypatch.setattr(
            subprocess, "run", lambda *args, **kwargs: MockUnrecognized()
        )
        is_zombie, reason = _check_slurm_job_status("12345")
        assert is_zombie is False
        assert "unrecognized status" in reason.lower()

    def test_is_local_host_variants(self) -> None:
        """Verify hostname comparison for localhost, FQDNs, and remote nodes."""
        curr_host = socket.gethostname()
        assert _is_local_host(curr_host) is True
        assert _is_local_host("localhost") is True
        assert _is_local_host("127.0.0.1") is True
        assert _is_local_host("::1") is True
        assert _is_local_host(curr_host.split(".")[0]) is True
        assert _is_local_host("completely-different-cluster-node-99") is False


# ============================================================================
# 8. Integrity Verification & Full Healing Tests
# ============================================================================


class TestIntegrityAndHealDatabase:
    """Tests for physical HDF5 integrity checking and full database healing."""

    def test_verify_h5_swmr_integrity_valid(self, tmp_path: Path) -> None:
        """Verify integrity check succeeds on a valid HDF5 file."""
        db_path = tmp_path / "valid.h5"
        with h5py.File(db_path, mode="w", libver="latest") as f:
            f.create_dataset("test", data=[1, 2, 3])
            f.attrs["version"] = "1.0.0"

        assert verify_h5_swmr_integrity(db_path) is True

    def test_verify_h5_swmr_integrity_corrupted_bytes(
        self, tmp_path: Path
    ) -> None:
        """Verify integrity check returns False on corrupted/garbage files."""
        db_path = tmp_path / "corrupted.h5"
        db_path.write_bytes(
            b"\x89HDF\r\n\x1a\n\x00\x00\x00CORRUPT_SUPERBLOCK_GARBAGE"
        )
        assert verify_h5_swmr_integrity(db_path) is False

    def test_verify_h5_swmr_integrity_nonexistent(self, tmp_path: Path) -> None:
        """Verify integrity check returns False for missing file."""
        assert verify_h5_swmr_integrity(tmp_path / "does_not_exist.h5") is False

    def test_heal_swmr_database_end_to_end(self, tmp_path: Path) -> None:
        """Verify heal_swmr_database reaps stale lock and confirms integrity."""
        db_path = tmp_path / "healed_landscape.h5"
        with h5py.File(db_path, mode="w", libver="latest") as f:
            f.create_dataset("coordinates", (10, 3), dtype="float64")

        # Create stale lock
        lock_path = get_lock_path(db_path)
        dead_pid = _find_unused_pid()
        stale_meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            created_at=time.time() - 100.0,
            session_id="dead-worker",
        )
        lock_path.write_text(
            stale_meta.model_dump_json(indent=2), encoding="utf-8"
        )

        assert lock_path.exists()

        # Perform healing
        success = heal_swmr_database(db_path)
        assert success is True
        assert not lock_path.exists()
        assert verify_h5_swmr_integrity(db_path) is True
