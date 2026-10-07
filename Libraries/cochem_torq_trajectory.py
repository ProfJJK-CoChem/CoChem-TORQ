"""Single-owner molecular-dynamics trajectories with committed-frame visibility.

Existing paths are never truncated. Atomic identity and the original integration
step are preserved. Temperature requires an explicitly supplied degree count;
without it temperature is unavailable rather than based on an assumed constraint.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Optional, Sequence, Union

from filelock import FileLock
import h5py
import numpy as np
from scipy.constants import physical_constants
import torch

from Libraries.cochem_torq_md_schemas import TrajectoryFrame

_FRAME_FIELDS = ('coordinates', 'velocities', 'forces', 'energies', 'temperatures',
                 'temperature_available', 'time_fs', 'step')


class HDF5TrajectoryWriter:
    """One writer per new file; publication follows complete, finite payloads."""

    def __init__(self, file_path: Union[str, Path], n_atoms: int,
                 atomic_numbers: Union[torch.Tensor, Sequence[int]], chunk_size: int = 100,
                 compression: Optional[str] = 'gzip', *,
                 temperature_dof: Optional[int] = None, writer_timeout_seconds: float = 10.0) -> None:
        if type(n_atoms) is not int or n_atoms < 1 or type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('n_atoms and chunk_size must be positive integers')
        if temperature_dof is not None and (type(temperature_dof) is not int or temperature_dof < 1):
            raise ValueError('temperature_dof must be an explicitly justified positive integer')
        if os.environ.get('HDF5_USE_FILE_LOCKING', '').strip().upper() in {'FALSE', '0'}:
            raise ValueError('HDF5 file locking must remain enabled')
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_atoms = n_atoms
        self.chunk_size = chunk_size
        self.compression = compression
        self.temperature_dof = temperature_dof
        z = atomic_numbers.detach().cpu().numpy() if isinstance(atomic_numbers, torch.Tensor) else np.asarray(atomic_numbers)
        if (z.shape != (n_atoms,) or not np.issubdtype(z.dtype, np.integer)
                or np.any(z < 1) or np.any(z > 118)):
            raise ValueError('Atomic numbers must be an ordered integer vector in 1..118')
        self.atomic_numbers_arr = z.astype(np.int64)
        self._thread_lock = threading.RLock()
        self._writer_lock = FileLock(str(self.file_path) + '.writer.lock',
                                     timeout=writer_timeout_seconds, thread_local=False)
        self._writer_pid = os.getpid()
        self._writer_lock.acquire()
        try:
            self.file = h5py.File(self.file_path, 'x', libver='latest', locking=True)
            self._create_schema()
        except Exception:
            if hasattr(self, 'file'):
                self.file.close()
            self._writer_lock.release()
            raise

    def _create_schema(self) -> None:
        for key in ('coordinates', 'velocities', 'forces'):
            dataset = self.file.create_dataset(key, shape=(0, self.n_atoms, 3),
                maxshape=(None, self.n_atoms, 3), dtype='float64',
                chunks=(self.chunk_size, self.n_atoms, 3), compression=self.compression,
                fletcher32=True, fillvalue=np.nan)
            setattr(self, key, dataset)
        self.energies = self.file.create_dataset('energies', shape=(0, 3), maxshape=(None, 3),
            dtype='float64', chunks=(self.chunk_size, 3), compression=self.compression,
            fletcher32=True, fillvalue=np.nan)
        for key in ('temperatures', 'time_fs'):
            dataset = self.file.create_dataset(key, shape=(0,), maxshape=(None,), dtype='float64',
                chunks=(self.chunk_size,), compression=self.compression, fletcher32=True, fillvalue=np.nan)
            setattr(self, key, dataset)
        self.file.create_dataset('step', shape=(0,), maxshape=(None,), dtype='int64',
                                 chunks=(self.chunk_size,), fletcher32=True)
        self.file.create_dataset('temperature_available', shape=(0,), maxshape=(None,), dtype='bool',
                                 chunks=(self.chunk_size,), fletcher32=True)
        self.file.create_dataset('atomic_numbers', data=self.atomic_numbers_arr)
        self.file.create_dataset('committed_frames', data=np.array([0], dtype='uint64'),
                                 chunks=(1,), fletcher32=True)
        self.file.attrs['schema_version'] = '2.0.0'
        self.file.attrs['temperature_dof_status'] = 'supplied' if self.temperature_dof else 'unavailable'
        if self.temperature_dof is not None:
            self.file.attrs['temperature_dof'] = self.temperature_dof
        self.file.flush()
        self.file.swmr_mode = True

    def append_frame(self, frame: TrajectoryFrame) -> None:
        if self._writer_pid != os.getpid():
            raise RuntimeError('An inherited trajectory handle cannot write')
        z = frame.atomic_numbers.detach().cpu().numpy()
        if not np.array_equal(z, self.atomic_numbers_arr):
            raise ValueError('Trajectory frame changes ordered atomic identity')
        payload = {key: getattr(frame, key).detach().cpu().numpy().astype(np.float64)
                   for key in ('coordinates', 'velocities', 'forces')}
        if any(array.shape != (self.n_atoms, 3) or not np.isfinite(array).all() for array in payload.values()):
            raise ValueError('Frame vectors must have exact atom shape and finite values')
        potential, kinetic, timestamp = float(frame.potential_energy_ev), float(frame.kinetic_energy_ev), float(frame.time_fs)
        if (not np.isfinite([potential, kinetic, timestamp]).all() or kinetic < 0 or timestamp < 0
                or type(frame.step) is not int or frame.step < 0):
            raise ValueError('Frame scalars must be finite with nonnegative kinetic energy, time and integer step')
        total = potential + kinetic
        if not np.isfinite(total):
            raise ValueError('Total energy overflows')
        available = self.temperature_dof is not None
        temperature = (2 * kinetic / (self.temperature_dof * physical_constants['Boltzmann constant in eV/K'][0])
                       if available else np.nan)
        payload.update(energies=np.array([potential, kinetic, total]), temperatures=temperature,
                       temperature_available=available, time_fs=timestamp, step=frame.step)
        with self._thread_lock:
            if not self.file.id.valid:
                raise RuntimeError('Trajectory writer is closed')
            index = int(self.file['committed_frames'][0])
            for key, values in payload.items():
                dataset = self.file[key]
                dataset.resize((index + 1,) + dataset.shape[1:])
                dataset[index] = values
                dataset.flush()
            self.file.flush()
            self.file['committed_frames'][0] = index + 1
            self.file['committed_frames'].flush()
            self.file.flush()

    def close(self) -> None:
        if self._writer_pid != os.getpid():
            raise RuntimeError('An inherited trajectory handle cannot release writer ownership')
        with self._thread_lock:
            if hasattr(self, 'file') and self.file.id.valid:
                try:
                    self.file.close()
                finally:
                    self._writer_lock.release()

    def __enter__(self) -> HDF5TrajectoryWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class HDF5TrajectoryReader:
    """Read complete frames only; unmarked legacy data requires explicit migration."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        self.file_path = Path(file_path).resolve()
        self.file = h5py.File(self.file_path, 'r', libver='latest', swmr=True, locking=True)
        try:
            if self.file.attrs.get('schema_version') != '2.0.0' or 'committed_frames' not in self.file:
                raise ValueError('Legacy trajectory requires explicit validated migration')
            self.atomic_numbers = torch.tensor(self.file['atomic_numbers'][:], dtype=torch.int64)
            self.n_atoms = len(self.atomic_numbers)
            self.refresh()
        except Exception:
            self.file.close()
            raise

    def refresh(self) -> None:
        self.file['committed_frames'].refresh()
        committed = int(self.file['committed_frames'][0])
        for key in _FRAME_FIELDS:
            self.file[key].refresh()
            if len(self.file[key]) < committed:
                raise ValueError('Trajectory commit marker exceeds complete frame data')
        self._committed = committed

    def __len__(self) -> int:
        self.refresh()
        return self._committed

    def read_frame(self, index: int) -> TrajectoryFrame:
        self.refresh()
        if index < 0 or index >= self._committed:
            raise IndexError(f'Frame {index} is outside the committed prefix of length {self._committed}')
        vectors = {key: torch.tensor(self.file[key][index], dtype=torch.float64)
                   for key in ('coordinates', 'velocities', 'forces')}
        energies = self.file['energies'][index]
        if (any(vector.shape != (self.n_atoms, 3) or not torch.isfinite(vector).all() for vector in vectors.values())
                or not np.isfinite(energies).all()):
            raise ValueError('Nonfinite or malformed committed frame')
        return TrajectoryFrame(step=int(self.file['step'][index]),
            time_fs=float(self.file['time_fs'][index]), atomic_numbers=self.atomic_numbers,
            potential_energy_ev=float(energies[0]), kinetic_energy_ev=float(energies[1]), **vectors)

    def read_all_coordinates(self) -> torch.Tensor:
        self.refresh()
        return torch.tensor(self.file['coordinates'][:self._committed], dtype=torch.float64)

    def read_all_energies(self) -> torch.Tensor:
        self.refresh()
        return torch.tensor(self.file['energies'][:self._committed], dtype=torch.float64)

    def close(self) -> None:
        if self.file.id.valid:
            self.file.close()

    def __enter__(self) -> HDF5TrajectoryReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
