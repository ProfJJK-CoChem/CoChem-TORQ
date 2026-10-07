"""Real process/file checks; no scientific engine or result is simulated."""

import asyncio
import math
import sqlite3
import sys

import filelock
import h5py
import pytest

from cochem.runners.async_process_runner import AsyncProcessRunner
from cochem.hpc.models import MpiProcessSupervisorError


def runner_at(tmp_path):
    return AsyncProcessRunner(src_dir=tmp_path / "source",
                              artifacts_dir=tmp_path / "artifacts",
                              scratch_dir=tmp_path / "scratch")


def test_reinitialize_and_resume_keep_committed_unknown_energy_rows(tmp_path):
    path = tmp_path / "artifacts" / "telemetry.h5"
    with runner_at(tmp_path) as runner:
        runner.init_telemetry(path)
        runner.record_telemetry_metric(path, step=0, energy=None, walltime=0)
        runner.init_telemetry(path)
        runner.close_telemetry(path)
        runner.init_telemetry(path)
        runner.record_telemetry_metric(path, step=1, energy=None, walltime=0)
    with h5py.File(path, "r") as handle:
        group = handle["telemetry"]
        assert group["committed_rows"][()] == 2
        assert list(group["step"][:]) == [0, 1]
        assert not group["energy_available"][:].any()
        assert all(math.isnan(value) for value in group["energy"][:])


def test_unrecognized_existing_file_is_preserved(tmp_path):
    path = tmp_path / "unrelated.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("user_record", data="preserve this record")
    original = path.read_bytes()
    with runner_at(tmp_path) as runner:
        with pytest.raises(ValueError, match="explicit migration"):
            runner.init_telemetry(path)
    assert path.read_bytes() == original


def test_writer_ownership_covers_handle_lifetime(tmp_path):
    path = tmp_path / "shared.h5"
    first, second = runner_at(tmp_path), runner_at(tmp_path)
    try:
        first.init_telemetry(path)
        with pytest.raises(filelock.Timeout):
            second.init_telemetry(path)
        first.close_telemetry(path)
        second.init_telemetry(path)
    finally:
        first.close()
        second.close()


def test_actual_subprocess_creates_database_without_invented_energy(tmp_path):
    database = tmp_path / "artifacts" / "child.sqlite"
    script = (
        "import sqlite3, sys\n"
        "with sqlite3.connect(sys.argv[1]) as connection:\n"
        "    connection.execute('CREATE TABLE executed (process_id INTEGER)')\n"
        "    connection.execute('INSERT INTO executed VALUES (?)', (__import__('os').getpid(),))\n"
    )
    with runner_at(tmp_path) as runner:
        result = asyncio.run(runner.dispatch_task(
            task_name="create_database", binary_args=[sys.executable, "-c", script, str(database)]))
        assert result["exit_code"] == 0
        assert result["energy_hartree"] is None
        assert result["energy_status"] == "unavailable"
        assert result["scientific_validation"] == "not_performed_by_process_runner"
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM executed").fetchone()[0] == 1
        with h5py.File(result["telemetry_file"], "r") as handle:
            assert handle["telemetry/committed_rows"][()] == 1
            assert not handle["telemetry/energy_available"][0]
            assert math.isnan(handle["telemetry/energy"][0])
            assert handle["telemetry/walltime"][0] >= 0


def test_actual_failed_process_does_not_leave_writer_lease(tmp_path):
    with runner_at(tmp_path) as runner:
        with pytest.raises(MpiProcessSupervisorError, match="return code 7"):
            asyncio.run(runner.dispatch_task(
                task_name="failure", binary_args=[sys.executable, "-c", "raise SystemExit(7)"]))
        assert not runner._active_writers
        assert not runner._writer_locks
        telemetry = next((tmp_path / "artifacts" / "telemetry").glob("*.h5"))
        with h5py.File(telemetry, "r") as handle:
            assert handle["telemetry/committed_rows"][()] == 0


def test_cleanup_preserves_unowned_scratch_files(tmp_path):
    with runner_at(tmp_path) as runner:
        original = runner.scratch_dir / "user-work.txt"
        original.write_text("existing data")
        runner.cleanup_scratch()
        assert original.read_text() == "existing data"
