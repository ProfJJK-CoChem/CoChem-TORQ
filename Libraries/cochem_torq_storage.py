"""Committed HDF5 batches with exact atom shape and explicit missing uncertainty.

Local-filesystem advisory and HDF5 locks serialize writers. Unknown values are
never represented as zero or published merely because space was allocated.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Dict, Optional, Sequence, Union

import filelock
import h5py
import numpy as np


class HDF5StorageManager:
    """Append complete batches; each file has one fixed, ordered atom dimension."""

    def __init__(self, filepath: Union[str, Path], timeout_seconds: float = 30.0,
                 compression: str = 'gzip', compression_opts: int = 4) -> None:
        if os.environ.get('HDF5_USE_FILE_LOCKING', '').strip().upper() in {'FALSE', '0'}:
            raise ValueError('HDF5 file locking must be enabled')
        self.filepath = Path(filepath).resolve()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.compression = compression
        self.compression_opts = int(compression_opts)
        self.lock_path = Path(str(self.filepath) + '.writer.lock')
        self._thread_lock = threading.RLock()
        self._file_lock = filelock.FileLock(str(self.lock_path), timeout=self.timeout_seconds)

    @staticmethod
    def _ids(atom_ids: Optional[Sequence[str]], n_atoms: int) -> Optional[list[str]]:
        if atom_ids is None:
            return None
        ids = list(atom_ids)
        if (len(ids) != n_atoms or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != n_atoms):
            raise ValueError('atom_ids must contain one distinct nonempty identifier per atom')
        return ids

    @staticmethod
    def _validate_file(f: h5py.File, committed: Optional[int] = None) -> int:
        expected = {'coordinates', 'energies', 'forces', 'uncertainties',
                    'uncertainty_available', 'committed_frames'}
        if f.attrs.get('schema_version') != '2.0.0' or not expected <= set(f):
            raise ValueError('Legacy or incomplete storage schema: explicit validated migration required')
        if committed is None:
            committed = int(f['committed_frames'][0])
        if any(f[key].shape[0] < committed for key in expected - {'committed_frames'}):
            raise ValueError('Commit marker exceeds payload length; recovery required')
        return committed

    def initialize_swmr_datasets(self, max_atoms: int = 100, initial_samples: int = 0,
                                 *, atom_ids: Optional[Sequence[str]] = None) -> None:
        """Create an empty store. max_atoms is the exact immutable atom count.

        initial_samples must be zero: allocated zero-fill is not measured data.
        If atom IDs are unavailable this is recorded, never inferred as identity.
        """
        if type(max_atoms) is not int or max_atoms < 1:
            raise ValueError('max_atoms must be a positive integer exact atom count')
        if initial_samples != 0:
            raise ValueError('initial_samples must be zero; append actual results to create samples')
        ids = self._ids(atom_ids, max_atoms)
        with self._thread_lock, self._file_lock:
            if self.filepath.exists():
                with h5py.File(self.filepath, 'r', libver='latest', swmr=True, locking=True) as f:
                    self._validate_file(f)
                    if f['coordinates'].shape[1] != max_atoms:
                        raise ValueError('Cannot change immutable atom dimension')
                    self._validate_mapping(f, ids)
                return
            with h5py.File(self.filepath, 'x', libver='latest', locking=True) as f:
                for key in ('coordinates', 'forces'):
                    f.create_dataset(key, shape=(0, max_atoms, 3), maxshape=(None, max_atoms, 3),
                                     chunks=(32, max_atoms, 3), dtype=np.float64,
                                     compression=self.compression, compression_opts=self.compression_opts,
                                     shuffle=True, fletcher32=True, fillvalue=np.nan)
                for key in ('energies', 'uncertainties'):
                    f.create_dataset(key, shape=(0,), maxshape=(None,), chunks=(32,), dtype=np.float64,
                                     compression=self.compression, compression_opts=self.compression_opts,
                                     shuffle=True, fletcher32=True, fillvalue=np.nan)
                f.create_dataset('uncertainty_available', shape=(0,), maxshape=(None,),
                                 chunks=(32,), dtype=np.bool_, fletcher32=True)
                f.create_dataset('committed_frames', data=np.array([0], dtype='uint64'),
                                 chunks=(1,), fletcher32=True)
                f.attrs['schema_version'] = '2.0.0'
                f.attrs['molecular_identity_status'] = 'atom_ids_supplied' if ids else 'unavailable'
                f['uncertainties'].attrs['missing_value_semantics'] = 'NaN with uncertainty_available=False'
                if ids is not None:
                    f.create_dataset('atom_ids', data=ids, dtype=h5py.string_dtype('utf-8'))
                f.flush()

    @staticmethod
    def _validate_mapping(f: h5py.File, ids: Optional[list[str]]) -> None:
        stored = f['atom_ids'].asstr()[:].tolist() if 'atom_ids' in f else None
        if stored != ids:
            raise ValueError('Atom identifiers/order differ or are missing; explicit atom mapping is required')

    def append_batch(self, coordinates: np.ndarray, energies: np.ndarray, forces: np.ndarray,
                     uncertainties: Optional[np.ndarray] = None, *,
                     atom_ids: Optional[Sequence[str]] = None) -> None:
        coords = np.asarray(coordinates, dtype=np.float64)
        energy = np.asarray(energies, dtype=np.float64)
        force = np.asarray(forces, dtype=np.float64)
        if coords.ndim != 3 or coords.shape[2] != 3 or coords.shape[1] < 1:
            raise ValueError('coordinates must have shape (samples, atoms, 3)')
        n_samples, n_atoms = coords.shape[:2]
        if energy.shape != (n_samples,) or force.shape != coords.shape:
            raise ValueError('Energy count and full force shape must exactly match coordinate samples')
        if not all(np.isfinite(x).all() for x in (coords, energy, force)):
            raise ValueError('Coordinates, energy and force results must be finite')
        ids = self._ids(atom_ids, n_atoms)
        uncertainty = (np.full(n_samples, np.nan) if uncertainties is None
                       else np.asarray(uncertainties, dtype=np.float64))
        if uncertainty.shape != (n_samples,) or np.isinf(uncertainty).any() or np.any(uncertainty < 0):
            raise ValueError('Uncertainty must have one nonnegative finite value or NaN per sample')
        available = np.isfinite(uncertainty)
        with self._thread_lock, self._file_lock:
            if not self.filepath.exists():
                self.initialize_swmr_datasets(max_atoms=n_atoms, atom_ids=ids)
            with h5py.File(self.filepath, 'r+', libver='latest', locking=True) as f:
                start = self._validate_file(f)
                if coords.shape[1:] != f['coordinates'].shape[1:]:
                    raise ValueError('Atom count mismatch: cropping and padding are prohibited')
                self._validate_mapping(f, ids)
                payloads = {'coordinates': coords, 'energies': energy, 'forces': force,
                            'uncertainties': uncertainty, 'uncertainty_available': available}
                # Remove unpublished tails before entering SWMR or appending.
                for key in payloads:
                    f[key].resize((start,) + f[key].shape[1:])
                f.swmr_mode = True
                end = start + n_samples
                for key, values in payloads.items():
                    f[key].resize((end,) + f[key].shape[1:])
                    f[key][start:end] = values
                    f[key].flush()
                f.flush()
                f['committed_frames'][0] = end
                f['committed_frames'].flush()
                f.flush()

    def read_batch(self) -> Dict[str, np.ndarray]:
        """Read the committed prefix, including an explicit uncertainty validity mask."""
        with h5py.File(self.filepath, 'r', libver='latest', swmr=True, locking=True) as f:
            if 'committed_frames' not in f:
                raise ValueError('Missing commit marker: explicit validated migration required')
            f['committed_frames'].refresh()
            committed = int(f['committed_frames'][0])
            keys = ('coordinates', 'energies', 'forces', 'uncertainties', 'uncertainty_available')
            for key in keys:
                f[key].refresh()
            self._validate_file(f, committed=committed)
            data = {key: f[key][:committed].copy() for key in keys}
            if not all(np.isfinite(data[key]).all() for key in ('coordinates', 'energies', 'forces')):
                raise ValueError('Nonfinite committed result; stored batch requires review')
            uncertainty = data['uncertainties']
            if (np.isinf(uncertainty).any() or np.any(uncertainty < 0)
                    or not np.array_equal(data['uncertainty_available'], np.isfinite(uncertainty))):
                raise ValueError('Uncertainty values and availability mask are inconsistent')
            return data


HDF5TorqStorage = HDF5StorageManager
