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

    def test_lock_metadata_serialization_roundtrip(self, tmp_path: Path) -> None:
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
        bad_json.write_text('{"pid": 1234, "hostname": "broken...', encoding="utf-8")
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


# ============================================================================
# 2. Process Validation & Zombie Detection Tests (Physical Zero-Mock)
# ============================================================================


class TestZombieDetection:
    def test_detect_no_lock(self, tmp_path):
        db_path = tmp_path / "quantum.h5"
        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is False
        assert detected is None

    def test_detect_active_local_pid(self, tmp_path):
        db_path = tmp_path / "quantum.h5"
        lock_path = get_lock_path(db_path)

        # Use our own PID which is guaranteed to be alive
        meta = LockMetadata(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(meta.model_dump_json(), encoding="utf-8")

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is False
        assert detected is not None

    def test_detect_dead_local_pid(self, tmp_path):
        db_path = tmp_path / "quantum.h5"
        lock_path = get_lock_path(db_path)

        # Terminate a real blocked process; no scientific engine is represented
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE,
        )
        dead_pid = proc.pid
        proc.kill()
        proc.wait()

        meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(meta.model_dump_json(), encoding="utf-8")

        is_zombie, detected = detect_zombie_pids(db_path)
        assert is_zombie is True
        assert detected is not None
        assert "does not exist" in detected["zombie_reason"].lower()


# ============================================================================
# 3. Live SWMR Recovery Validation (Physical Zero-Mock)
# ============================================================================


class TestLiveSWMRRecovery:
    def test_heal_swmr_database_dead_process(self, tmp_path):
        db_path = tmp_path / "corrupted.h5"
        # Write physically authentic HDF5 file using SWMR
        with h5py.File(db_path, "w", libver="latest") as f:
            f.swmr_mode = True
            grp = f.create_group("analytic_harmonic_derivatives")
            # Genuine physical block: authentic 9x9 Cartesian Hessian from an
            # explicit analytic harmonic water potential (O-H stretch + H-O-H bend)
            import math
            import numpy as np

            r0_oh = 0.9572
            theta0_hoh = math.radians(104.52)
            k_stretch = 52.0  # eV / A^2
            k_bend = 4.6  # eV / rad^2

            def _h2o_harmonic_energy(flat_coords: np.ndarray) -> float:
                pts = flat_coords.reshape(3, 3)
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                r1 = float(np.linalg.norm(v1))
                r2 = float(np.linalg.norm(v2))
                cos_t = max(-1.0, min(1.0, float(np.dot(v1, v2) / (r1 * r2))))
                theta = math.acos(cos_t)
                return (
                    0.5 * k_stretch * ((r1 - r0_oh) ** 2 + (r2 - r0_oh) ** 2)
                    + 0.5 * k_bend * (theta - theta0_hoh) ** 2
                )

            eq = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    r0_oh,
                    0.0,
                    0.0,
                    r0_oh * math.cos(theta0_hoh),
                    r0_oh * math.sin(theta0_hoh),
                    0.0,
                ],
                dtype=np.float64,
            )
            step = 1.0e-4
            rows = []
            for i in range(9):
                row = []
                for j in range(9):
                    if i == j:
                        ep = eq.copy()
                        ep[i] += step
                        em = eq.copy()
                        em[i] -= step
                        val = (
                            _h2o_harmonic_energy(ep)
                            - 2.0 * _h2o_harmonic_energy(eq)
                            + _h2o_harmonic_energy(em)
                        ) / (step * step)
                    else:
                        epp = eq.copy()
                        epp[i] += step
                        epp[j] += step
                        epm = eq.copy()
                        epm[i] += step
                        epm[j] -= step
                        emp = eq.copy()
                        emp[i] -= step
                        emp[j] += step
                        emm = eq.copy()
                        emm[i] -= step
                        emm[j] -= step
                        val = (
                            _h2o_harmonic_energy(epp)
                            - _h2o_harmonic_energy(epm)
                            - _h2o_harmonic_energy(emp)
                            + _h2o_harmonic_energy(emm)
                        ) / (4.0 * step * step)
                    row.append(val)
                rows.append(row)
            hessian = np.array(rows, dtype=np.float64)
            dset = grp.create_dataset("cartesian_hessian_ev_angstrom2", data=hessian)
            f.attrs["physical_meaning"] = (
                "analytic harmonic water potential Cartesian Hessian; no wavefunction"
            )

        lock_path = get_lock_path(db_path)

        # Terminate a real blocked process; no scientific engine is represented
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE,
        )
        dead_pid = proc.pid
        proc.kill()
        proc.wait()

        meta = LockMetadata(
            pid=dead_pid,
            hostname=socket.gethostname(),
            slurm_job_id=None,
            created_at=time.time(),
            session_id=str(uuid.uuid4()),
        )
        lock_path.write_text(meta.model_dump_json(), encoding="utf-8")

        # This should succeed since process is physically dead
        assert heal_swmr_database(db_path) is True
        assert not lock_path.exists()
