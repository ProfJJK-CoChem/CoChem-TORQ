"""Real SQLite/HDF5 integrity checks using arithmetic arrays, never engine output.

Processes, file locks, crash interruption and storage I/O are real. The arrays
exercise byte storage and have no asserted physical interpretation.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sqlite3
import time

import h5py
import numpy as np
import pytest
from filelock import Timeout

from cochem.orchestration.sqlite_queue import LeaseLostError, LeaseRequiredError, SQLiteTaskQueue
from cochem.storage.cochem_core_pes_store import StorageSchemaError, SWMRPESStore
from Libraries.cochem_torq_storage import HDF5StorageManager


def _credentials(task):
    return {'lease_token': task.lease_token, 'lease_generation': task.lease_generation}


def _consume(db_path):
    completed = []
    with SQLiteTaskQueue(db_path) as queue:
        while (task := queue.lease_task(os.getpid(), 'local-integrity-test')) is not None:
            queue.complete_task(task.task_id, {'index': task.payload['index']}, **_credentials(task))
            completed.append(task.task_id)
    return completed


def _attempt_writer(path):
    store = SWMRPESStore(path, n_atoms=2, writer_timeout_sec=0.1)
    try:
        store.open_writer()
    except Timeout:
        return 'ownership rejected'
    else:
        store.close_writer()
        return 'incorrect concurrent ownership'


def _read_snapshots(path, connection):
    store = SWMRPESStore(path, n_atoms=2)
    connection.send('ready')
    snapshots = []
    for _ in range(25):
        data = store.read_trajectory()
        n = len(data['energies'])
        expected = np.arange(n, dtype=float)
        assert np.array_equal(data['energies'], expected)
        assert np.array_equal(data['coordinates'], np.broadcast_to(expected[:, None, None], (n, 2, 3)))
        snapshots.append(n)
        time.sleep(0.01)
    connection.send(snapshots)
    connection.close()


def _write_uncommitted_then_wait(path, connection):
    with SWMRPESStore(path, n_atoms=2) as store:
        store.append_frames(np.ones((1, 2, 3)), [1.0])
        # Real partial write boundary: resize both, write coordinates only, flush,
        # then let the parent kill this process before publication.
        f = store._writer_file
        f['coordinates'].resize((2, 2, 3))
        f['energies'].resize((2,))
        f['coordinates'][1] = np.full((2, 3), 2.0)
        f.flush()
        connection.send('uncommitted tail flushed')
        connection.recv()


def test_stale_attempt_cannot_heartbeat_complete_or_fail(tmp_path):
    with SQLiteTaskQueue(tmp_path / 'queue.db') as queue:
        task_id = queue.enqueue_task('storage-check', {'index': 1})
        old = queue.lease_task(7, 'host-a')
        assert old.lease_generation == 1 and len(old.lease_token) >= 32
        assert queue.reclaim_orphaned_tasks(0) == [task_id]
        current = queue.lease_task(7, 'host-b')  # Same PID, different ownership.
        assert current.lease_generation == old.lease_generation + 1
        assert current.lease_token != old.lease_token
        assert not queue.heartbeat(task_id, 7, **_credentials(old))
        with pytest.raises(LeaseLostError):
            queue.complete_task(task_id, {'index': -1}, **_credentials(old))
        with pytest.raises(LeaseLostError):
            queue.fail_task(task_id, 'late worker failure', **_credentials(old))
        assert queue.heartbeat(task_id, 7, **_credentials(current))
        queue.complete_task(task_id, {'index': 1}, **_credentials(current))
        timestamp = queue.get_task(task_id).completed_at
        queue.complete_task(task_id, {'index': 1}, **_credentials(current))
        assert queue.get_task(task_id).completed_at == timestamp
        with pytest.raises(LeaseLostError):
            queue.complete_task(task_id, {'index': 2}, **_credentials(current))
        with pytest.raises(LeaseLostError):
            queue.fail_task(task_id, 'cannot overwrite success', **_credentials(current))
        assert queue.get_task(task_id).result == {'index': 1}


def test_credential_free_legacy_mutations_fail_explicitly(tmp_path):
    with SQLiteTaskQueue(tmp_path / 'queue.db') as queue:
        task_id = queue.enqueue_task('storage-check', {})
        queue.lease_task(1, 'host')
        with pytest.raises(LeaseRequiredError):
            queue.heartbeat(task_id, 1)
        with pytest.raises(LeaseRequiredError):
            queue.complete_task(task_id, {})
        with pytest.raises(LeaseRequiredError):
            queue.fail_task(task_id, 'legacy call')
        assert queue.get_task(task_id).state == 'RUNNING'


def test_wrong_generation_and_token_cannot_mutate(tmp_path):
    with SQLiteTaskQueue(tmp_path / 'queue.db') as queue:
        task_id = queue.enqueue_task('storage-check', {})
        task = queue.lease_task(1, 'host')
        for credentials in ({'lease_token': task.lease_token, 'lease_generation': 2},
                            {'lease_token': 'not-the-issued-token', 'lease_generation': 1}):
            assert not queue.heartbeat(task_id, **credentials)
            with pytest.raises(LeaseLostError):
                queue.complete_task(task_id, {}, **credentials)
            with pytest.raises(LeaseLostError):
                queue.fail_task(task_id, 'unowned', **credentials)


def test_retry_invalidates_token_and_advances_generation(tmp_path):
    with SQLiteTaskQueue(tmp_path / 'queue.db') as queue:
        task_id = queue.enqueue_task('storage-check', {}, max_retries=2)
        first = queue.lease_task(1, 'host')
        queue.fail_task(task_id, 'real failure transition', **_credentials(first))
        assert queue.get_task(task_id).lease_token is None
        second = queue.lease_task(1, 'host')
        assert second.lease_generation == 2
        queue.fail_task(task_id, 'retry exhausted', **_credentials(second))
        assert queue.get_task(task_id).state == 'FAILED'
        assert queue.get_task(task_id).lease_token is None


def test_legacy_database_migration_invalidates_unowned_running_task(tmp_path):
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE tasks (task_id TEXT PRIMARY KEY, task_type TEXT,
            state TEXT, priority INTEGER, payload_json TEXT, result_json TEXT,
            error_message TEXT, retry_count INTEGER, max_retries INTEGER,
            locked_by_pid INTEGER, locked_by_host TEXT, created_at REAL,
            heartbeat_ts REAL, completed_at REAL)''')
        conn.execute("INSERT INTO tasks VALUES ('legacy','check','RUNNING',0,'{}',NULL,NULL,0,3,1,'host',?, ?, NULL)", (time.time(), time.time()))
    with SQLiteTaskQueue(path) as queue:
        migrated = queue.get_task('legacy')
        assert migrated.state == 'PENDING' and migrated.retry_count == 1
        assert 'migration' in migrated.error_message.lower()
        new = queue.lease_task(2, 'host')
        assert new.lease_generation == 1 and new.lease_token
    with SQLiteTaskQueue(path) as reopened:
        assert reopened.get_task('legacy').state == 'RUNNING'
        assert reopened.get_task('legacy').lease_token == new.lease_token


def test_real_processes_claim_each_task_only_once(tmp_path):
    path = tmp_path / 'concurrent.db'
    with SQLiteTaskQueue(path) as queue:
        task_ids = {queue.enqueue_task('storage-check', {'index': i}) for i in range(30)}
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as pool:
        completed = [task for group in pool.map(_consume, [path] * 3) for task in group]
    assert len(completed) == len(set(completed)) == len(task_ids)
    assert set(completed) == task_ids
    with SQLiteTaskQueue(path) as queue:
        assert all(queue.get_task(task).state == 'COMPLETED' for task in task_ids)


def test_hdf5_rejects_concurrent_persistent_writer(tmp_path):
    path = tmp_path / 'writer.h5'
    with SWMRPESStore(path, n_atoms=2) as owner:
        owner.append_frames(np.ones((1, 2, 3)), [1.0])
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
            assert pool.submit(_attempt_writer, path).result(timeout=15) == 'ownership rejected'
    with SWMRPESStore(path, n_atoms=2) as successor:
        assert successor._writer_generation == 2
        successor.append_frames(np.full((1, 2, 3), 2.0), [2.0])
    np.testing.assert_array_equal(successor.read_trajectory()['energies'], [1, 2])


def test_real_swmr_reader_observes_matching_committed_payloads(tmp_path):
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    path = tmp_path / 'readers.h5'
    with SWMRPESStore(path, n_atoms=2) as store:
        store.append_frames(np.zeros((1, 2, 3)), [0.0])
        process = ctx.Process(target=_read_snapshots, args=(path, child))
        process.start()
        try:
            assert parent.poll(15) and parent.recv() == 'ready'
            for value in range(1, 12):
                store.append_frames(np.full((1, 2, 3), value, dtype=float), [float(value)])
                time.sleep(0.01)
            assert parent.poll(15)
            snapshots = parent.recv()
            assert snapshots == sorted(snapshots)
            assert snapshots[-1] == 12
            process.join(15)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
            parent.close()
            child.close()


def test_kill_after_partial_write_exposes_only_committed_frames(tmp_path):
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    path = tmp_path / 'crash.h5'
    process = ctx.Process(target=_write_uncommitted_then_wait, args=(path, child))
    process.start()
    try:
        assert parent.poll(15) and parent.recv() == 'uncommitted tail flushed'
        process.kill()
        process.join(15)
        assert process.exitcode != 0
        data = SWMRPESStore(path, n_atoms=2).read_trajectory()
        np.testing.assert_array_equal(data['coordinates'], np.ones((1, 2, 3)))
        np.testing.assert_array_equal(data['energies'], [1.0])
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
        child.close()


def test_pes_store_rejects_legacy_shapes_and_nonfinite_values(tmp_path):
    store = SWMRPESStore(tmp_path / 'valid.h5', n_atoms=2)
    with pytest.raises(ValueError, match='Expected coords'):
        store.append_frames(np.ones((1, 3, 3)), [1])
    with pytest.raises(ValueError, match='finite'):
        store.append_frames(np.ones((1, 2, 3)), [np.nan])
    path = tmp_path / 'legacy.h5'
    with h5py.File(path, 'w', libver='latest') as f:
        f.create_dataset('coordinates', data=np.ones((1, 2, 3)))
        f.create_dataset('energies', data=[1.0])
    with pytest.raises(StorageSchemaError):
        SWMRPESStore(path, n_atoms=2).read_trajectory()


def test_batch_storage_preserves_atoms_and_unknown_uncertainty(tmp_path):
    store = HDF5StorageManager(tmp_path / 'batch.h5')
    coords = np.arange(12, dtype=float).reshape(2, 2, 3)
    store.append_batch(coords, np.array([1., 2.]), -coords, atom_ids=['a', 'b'])
    data = store.read_batch()
    np.testing.assert_array_equal(data['coordinates'], coords)
    np.testing.assert_array_equal(data['forces'], -coords)
    assert np.isnan(data['uncertainties']).all()
    assert not data['uncertainty_available'].any()
    for n_atoms in (1, 3):
        array = np.ones((1, n_atoms, 3))
        with pytest.raises(ValueError, match='cropping and padding'):
            store.append_batch(array, np.ones(1), array)
    with pytest.raises(ValueError, match='Atom identifiers'):
        store.append_batch(coords, np.array([1., 2.]), -coords, atom_ids=['b', 'a'])
    with pytest.raises(ValueError, match='Atom identifiers'):
        store.append_batch(coords, np.array([1., 2.]), -coords)
    assert len(store.read_batch()['energies']) == 2
    store.append_batch(coords, np.array([1., 2.]), -coords, np.array([0.0, np.nan]), atom_ids=['a', 'b'])
    np.testing.assert_array_equal(store.read_batch()['uncertainty_available'], [False, False, True, False])


def test_batch_storage_never_preallocates_results_or_accepts_bad_shapes(tmp_path):
    store = HDF5StorageManager(tmp_path / 'batch.h5')
    with pytest.raises(ValueError, match='initial_samples'):
        store.initialize_swmr_datasets(max_atoms=2, initial_samples=4)
    with pytest.raises(ValueError, match='force shape'):
        store.append_batch(np.ones((1, 2, 3)), np.ones(1), np.ones((1, 1, 3)))
    with pytest.raises(ValueError, match='finite'):
        store.append_batch(np.ones((1, 2, 3)), np.array([np.nan]), np.ones((1, 2, 3)))
    assert not store.filepath.exists()


def test_uncommitted_batch_tail_is_never_returned(tmp_path):
    path = tmp_path / 'batch.h5'
    store = HDF5StorageManager(path)
    coords = np.ones((1, 2, 3))
    store.append_batch(coords, np.ones(1), coords)
    with h5py.File(path, 'r+', libver='latest', locking=True) as f:
        f['coordinates'].resize((2, 2, 3))
        f['coordinates'][1] = 9
        f.flush()
    assert len(store.read_batch()['coordinates']) == 1
    store.append_batch(coords * 2, np.array([2.0]), coords * 2)
    np.testing.assert_array_equal(store.read_batch()['energies'], [1, 2])


@pytest.mark.parametrize('credentials_state', ['current', 'stale', 'unavailable'])
def test_process_reaper_uses_only_original_attempt_credentials(tmp_path, credentials_state):
    import subprocess
    import sys
    import psutil
    from cochem.core.process_reaper import ProcessTreeManager, ZombieReaperDaemon

    # Real harmless child processes exercise lifecycle handling, not QM engines.
    exited_parent = subprocess.Popen([sys.executable, '-c', 'pass'])
    exited_parent.wait(timeout=10)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        with SQLiteTaskQueue(tmp_path / 'reaper.db') as queue:
            task_id = queue.enqueue_task('lifecycle-check', {})
            original = queue.lease_task(child.pid, 'local')
            tree = ProcessTreeManager()
            supplied = {} if credentials_state == 'unavailable' else _credentials(original)
            meta = tree.register_process(child.pid, task_id, **supplied)
            assert meta.lease_token == supplied.get('lease_token')
            if credentials_state == 'stale':
                queue.reclaim_orphaned_tasks(0)
                current = queue.lease_task(os.getpid(), 'replacement')
            else:
                current = original
            daemon = ZombieReaperDaemon(queue, tree, parent_pid=exited_parent.pid, grace_period_sec=60)
            assert daemon.sweep_orphans() == [child.pid]
            state = queue.get_task(task_id)
            if credentials_state == 'current':
                assert state.state == 'PENDING' and state.lease_token is None
            else:
                assert state.state == 'RUNNING'
                assert state.lease_token == current.lease_token
            # psutil.wait_procs has already reaped this PID; Popen cannot recover
            # its exit code after that waitpid. Check actual process absence.
            assert not psutil.pid_exists(child.pid)
            assert child.pid not in tree.get_tracked_pids()
            child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_reader_rejects_inconsistent_uncertainty_validity(tmp_path):
    path = tmp_path / 'invalid-mask.h5'
    store = HDF5StorageManager(path)
    store.append_batch(np.ones((1, 2, 3)), np.ones(1), np.ones((1, 2, 3)))
    with h5py.File(path, 'r+', libver='latest', locking=True) as f:
        f['uncertainty_available'][0] = True
    with pytest.raises(ValueError, match='availability mask'):
        store.read_batch()


@pytest.mark.cpu_ml
def test_training_store_preserves_native_locking_identity_and_committed_prefix(tmp_path):
    from Libraries.cochem_torq_training_persistence import HDF5DatasetManager
    before = os.environ.get('HDF5_USE_FILE_LOCKING')
    manager = HDF5DatasetManager(tmp_path / 'training.h5')
    coords = np.arange(6, dtype=float).reshape(1, 2, 3)
    manager.write_trajectory_batch('molecule', coords, [1, 8], np.array([1.0]), -coords)
    assert os.environ.get('HDF5_USE_FILE_LOCKING') == before
    with pytest.raises(ValueError, match='identity'):
        manager.write_trajectory_batch('molecule', coords, [8, 1], np.array([1.0]), -coords)
    with h5py.File(manager.filepath, 'r+', libver='latest', locking=True) as f:
        f['molecule/coordinates'].resize((2, 2, 3))
        f['molecule/coordinates'][1] = 99
    data = manager.read_trajectory_batch('molecule')
    np.testing.assert_array_equal(data['coordinates'], coords)
    manager.write_trajectory_batch('molecule', coords * 2, [1, 8], np.array([2.0]), -coords * 2)
    np.testing.assert_array_equal(manager.read_trajectory_batch('molecule')['energies'], [1, 2])
    with pytest.raises(ValueError, match='finite'):
        manager.write_trajectory_batch('molecule', coords, [1, 8], np.array([np.nan]), -coords)


@pytest.mark.cpu_ml
def test_md_trajectory_commits_original_step_and_never_truncates_existing_file(tmp_path):
    import torch
    from Libraries.cochem_torq_md_schemas import TrajectoryFrame
    from Libraries.cochem_torq_trajectory import HDF5TrajectoryReader, HDF5TrajectoryWriter
    path = tmp_path / 'md.h5'
    coordinates = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    frame = TrajectoryFrame(12, 1.0, torch.tensor([1, 8]), coordinates, coordinates / 2,
                            -coordinates, 3.0, 2.0)
    with HDF5TrajectoryWriter(path, 2, [1, 8]) as writer:
        writer.append_frame(frame)
        with pytest.raises(ValueError, match='atomic identity'):
            writer.append_frame(frame._replace(atomic_numbers=torch.tensor([8, 1])))
        writer.coordinates.resize((2, 2, 3))
        writer.coordinates[1] = 10
        writer.file.flush()
        with HDF5TrajectoryReader(path) as reader:
            assert len(reader) == 1
            restored = reader.read_frame(0)
            assert restored.step == 12
            assert torch.equal(restored.coordinates, coordinates)
            assert reader.read_all_coordinates().shape == (1, 2, 3)
            assert reader.read_all_energies().shape == (1, 3)
            with pytest.raises(IndexError):
                reader.read_frame(1)
        assert np.isnan(writer.file['temperatures'][0])
        assert not writer.file['temperature_available'][0]
    with pytest.raises(FileExistsError):
        HDF5TrajectoryWriter(path, 2, [1, 8])
    with HDF5TrajectoryReader(path) as reader:
        assert len(reader) == 1


@pytest.mark.cpu_ml
def test_md_trajectory_validates_before_allocating_and_records_explicit_temperature_dof(tmp_path):
    import torch
    from Libraries.cochem_torq_md_schemas import TrajectoryFrame
    from Libraries.cochem_torq_trajectory import HDF5TrajectoryWriter
    vectors = torch.ones((2, 3), dtype=torch.float64)
    frame = TrajectoryFrame(1, 0.5, torch.tensor([1, 8]), vectors, vectors, vectors, 1.0, 2.0)
    with HDF5TrajectoryWriter(tmp_path / 'md.h5', 2, [1, 8], temperature_dof=3) as writer:
        with pytest.raises(ValueError, match='finite'):
            writer.append_frame(frame._replace(potential_energy_ev=float('nan')))
        assert writer.coordinates.shape[0] == 0
        assert writer.file['committed_frames'][0] == 0
        writer.append_frame(frame)
        assert writer.file['temperature_available'][0]
        assert np.isfinite(writer.file['temperatures'][0])
