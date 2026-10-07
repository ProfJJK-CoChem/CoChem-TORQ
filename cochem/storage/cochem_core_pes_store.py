"""Local-filesystem HDF5 SWMR trajectories with a single fenced writer.

A commit marker publishes only complete coordinate/energy batches. HDF5 and OS
file locks remain enabled. Network filesystems require separate qualification;
SWMR does not provide a distributed transaction or power-loss guarantee.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
from filelock import FileLock

from cochem.storage.hdf5_zstd import HDF5ZstdConfig


class StorageSchemaError(ValueError):
    """A file lacks the committed-frame contract; explicit migration is required."""


class SWMRPESStore:
    """One persistent writer and committed-prefix readers on a local filesystem."""

    def __init__(self, h5_path: Union[str, Path], n_atoms: int = 1,
                 zstd_config: Optional[HDF5ZstdConfig] = None,
                 enable_swmr: bool = True, writer_timeout_sec: float = 10.0) -> None:
        if type(n_atoms) is not int or n_atoms < 1:
            raise ValueError('n_atoms must be a positive integer')
        if os.environ.get('HDF5_USE_FILE_LOCKING', '').strip().upper() in {'FALSE', '0'}:
            raise ValueError('HDF5 file locking must be enabled; remove HDF5_USE_FILE_LOCKING=FALSE')
        self.h5_path = Path(h5_path).resolve()
        self.lock_path = Path(str(self.h5_path) + '.writer.lock')
        self.n_atoms = n_atoms
        self.zstd_config = zstd_config or HDF5ZstdConfig()
        self.enable_swmr = enable_swmr
        self._writer_file: Optional[h5py.File] = None
        self._writer_lock = FileLock(str(self.lock_path), timeout=writer_timeout_sec,
                                     thread_local=False)
        self._thread_lock = threading.RLock()
        self._writer_pid: Optional[int] = None
        self._writer_generation: Optional[int] = None
        self._preallocate_schema()

    def _preallocate_schema(self) -> None:
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        if self.h5_path.exists():
            return
        with self._writer_lock:
            if self.h5_path.exists():
                return
            with h5py.File(self.h5_path, 'x', libver='latest', locking=True) as f:
                f.create_dataset('coordinates', shape=(0, self.n_atoms, 3),
                                 maxshape=(None, self.n_atoms, 3), dtype='float64',
                                 **self.zstd_config.get_dataset_kwargs(
                                     self.zstd_config.resolve_chunk_shape_3d(self.n_atoms, 3, 8)))
                f.create_dataset('energies', shape=(0,), maxshape=(None,), dtype='float64',
                                 **self.zstd_config.get_dataset_kwargs(
                                     self.zstd_config.resolve_chunk_shape_1d(8)))
                f.create_dataset('committed_frames', data=np.array([0], dtype='uint64'),
                                 chunks=(1,), fletcher32=True)
                f.create_dataset('writer_generation', data=np.array([0], dtype='uint64'))
                f.attrs['schema_version'] = '2.0.0'
                f.attrs['n_atoms'] = self.n_atoms
                # This numerical trajectory API has no molecular identity input.
                # Consumers must join an independently validated molecule record.
                f.attrs['molecular_identity_status'] = 'unavailable; external molecule record required'
                f.flush()

    def _validate_schema(self, f: h5py.File, committed: Optional[int] = None) -> int:
        if (f.attrs.get('schema_version') != '2.0.0' or
                not {'coordinates', 'energies', 'committed_frames', 'writer_generation'} <= set(f)):
            raise StorageSchemaError('Legacy/uncommitted PES schema: independently validate and explicitly migrate; no frames are implicitly trusted')
        if (int(f.attrs['n_atoms']) != self.n_atoms or f['coordinates'].shape[1:] != (self.n_atoms, 3)
                or f['energies'].ndim != 1 or f['committed_frames'].shape != (1,)
                or f['writer_generation'].shape != (1,)):
            raise StorageSchemaError('Atom count differs from the immutable stored atom dimension')
        if committed is None:
            committed = int(f['committed_frames'][0])
        if committed > min(f['coordinates'].shape[0], f['energies'].shape[0]):
            raise StorageSchemaError('Commit marker exceeds available complete frames; file requires recovery')
        return committed

    def open_writer(self) -> None:
        """Hold exclusive ownership until close; concurrent writer attempts time out."""
        with self._thread_lock:
            if self._writer_file is not None:
                self._assert_writer()
                return
            self._writer_lock.acquire()
            f = None
            try:
                f = h5py.File(self.h5_path, 'r+', libver='latest', locking=True)
                committed = self._validate_schema(f)
                # Discard only unpublished tails left by interrupted/failed appends.
                f['coordinates'].resize((committed, self.n_atoms, 3))
                f['energies'].resize((committed,))
                generation = int(f['writer_generation'][0]) + 1
                f['writer_generation'][0] = generation
                f.flush()
                if self.enable_swmr:
                    f.swmr_mode = True
                self._writer_file = f
                self._writer_pid = os.getpid()
                self._writer_generation = generation
            except Exception:
                if f is not None:
                    f.close()
                self._writer_lock.release()
                raise

    def _assert_writer(self) -> h5py.File:
        if self._writer_file is None or self._writer_pid != os.getpid():
            raise RuntimeError('No writer ownership in this process; inherited handles cannot write')
        if int(self._writer_file['writer_generation'][0]) != self._writer_generation:
            raise RuntimeError('Stale writer fencing generation')
        return self._writer_file

    def close_writer(self) -> None:
        with self._thread_lock:
            if self._writer_file is not None:
                f = self._assert_writer()
                try:
                    f.close()
                finally:
                    self._writer_file = None
                    self._writer_pid = None
                    self._writer_generation = None
                    self._writer_lock.release()

    def __enter__(self) -> SWMRPESStore:
        self.open_writer()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close_writer()

    def append_frames(self, coords: Union[np.ndarray, List[Any]],
                      energies: Union[np.ndarray, float, List[float]]) -> None:
        coords_arr = np.asarray(coords, dtype=np.float64)
        energies_arr = np.asarray(energies, dtype=np.float64)
        if coords_arr.ndim == 2:
            coords_arr = coords_arr[None, :, :]
        if energies_arr.ndim == 0:
            energies_arr = energies_arr[None]
        if coords_arr.ndim != 3 or coords_arr.shape[1:] != (self.n_atoms, 3):
            raise ValueError(f'Expected coords shape (N, {self.n_atoms}, 3), got {coords_arr.shape}')
        if energies_arr.shape != (len(coords_arr),):
            raise ValueError('One scalar energy is required for every coordinate frame')
        if not np.isfinite(coords_arr).all() or not np.isfinite(energies_arr).all():
            raise ValueError('Coordinates and energies must be finite; unavailable values cannot be appended as results')
        with self._thread_lock:
            transient = self._writer_file is None
            if transient:
                self.open_writer()
            try:
                f = self._assert_writer()
                start = self._validate_schema(f)
                end = start + len(coords_arr)
                f['coordinates'].resize((end, self.n_atoms, 3))
                f['energies'].resize((end,))
                f['coordinates'][start:end] = coords_arr
                f['energies'][start:end] = energies_arr
                f['coordinates'].flush()
                f['energies'].flush()
                f.flush()
                # Publish after both payloads, never derive truth from allocated size.
                f['committed_frames'][0] = end
                f['committed_frames'].flush()
                f.flush()
            finally:
                if transient:
                    self.close_writer()

    def read_trajectory(self, max_retries: int = 5,
                        retry_delay_s: float = 0.05) -> Dict[str, np.ndarray]:
        if max_retries < 1:
            raise ValueError('max_retries must be positive')
        for attempt in range(max_retries):
            try:
                with h5py.File(self.h5_path, 'r', libver='latest', swmr=self.enable_swmr,
                               locking=True) as f:
                    if 'committed_frames' not in f:
                        raise StorageSchemaError('Missing commit marker: explicit migration required')
                    # Snapshot marker first, then refresh payloads. A later append
                    # cannot change the already committed prefix being returned.
                    f['committed_frames'].refresh()
                    committed = int(f['committed_frames'][0])
                    f['coordinates'].refresh()
                    f['energies'].refresh()
                    self._validate_schema(f, committed=committed)
                    coords = f['coordinates'][:committed].copy()
                    energies = f['energies'][:committed].copy()
                    if not np.isfinite(coords).all() or not np.isfinite(energies).all():
                        raise StorageSchemaError('Nonfinite committed payload; scientific results unavailable')
                    return {'coordinates': coords, 'energies': energies}
            except (RuntimeError, OSError) as err:
                if attempt == max_retries - 1:
                    raise RuntimeError('Unable to read a consistent committed trajectory') from err
                time.sleep(retry_delay_s * 2**attempt)
        raise RuntimeError('Unable to read a consistent committed trajectory')
