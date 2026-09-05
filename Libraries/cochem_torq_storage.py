"""Thread-Safe and Multi-Process Safe HDF5 SWMR Storage Engine for CoChem-TORQ.

Authoritative Standards:
- Method Matrix v4 §12: Thread-Safe HDF5 Storage [M]
- HPC Distributed Lock Prohibition [M]: Cross-platform filelock.FileLock anchored on local mounts.
- SWMR Preallocation Invariant: Extensible chunk boundaries must be initialized prior to swmr_mode = True.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import filelock
import h5py
import numpy as np

# Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical


class HDF5StorageManager:
    """Thread-safe and process-safe HDF5 storage manager enforcing SWMR preallocation invariant [M]."""

    def __init__(
        self,
        filepath: Union[str, Path],
        timeout_seconds: float = 30.0,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        self.filepath = Path(filepath).resolve()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.compression = compression
        self.compression_opts = int(compression_opts)

        # Cross-platform advisory lock anchored next to dataset or local scratch [M]
        self.lock_path = self.filepath.with_suffix(".h5.lock")
        self._thread_lock = threading.RLock()
        self._file_lock = filelock.FileLock(str(self.lock_path), timeout=self.timeout_seconds)

    def initialize_swmr_datasets(
        self,
        max_atoms: int = 100,
        initial_samples: int = 0,
    ) -> None:
        """Preallocate extensible datasets with chunk boundaries before enabling SWMR mode [M].

        Preallocation Invariant: HDF5 SWMR specifications prohibit structural modifications
        (group/dataset creation) once swmr_mode is active. All extensible datasets
        ('coordinates', 'energies', 'forces', 'uncertainties') are created beforehand.
        """
        with self._thread_lock:
            with self._file_lock:
                with h5py.File(self.filepath, "a", libver="latest") as f:
                    chunk_samples = 32

                    # Preallocate extensible coordinates dataset
                    if "coordinates" not in f:
                        f.create_dataset(
                            "coordinates",
                            shape=(initial_samples, max_atoms, 3),
                            maxshape=(None, max_atoms, 3),
                            chunks=(chunk_samples, max_atoms, 3),
                            dtype=np.float64,
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                            shuffle=True,
                            fletcher32=True,
                        )

                    # Preallocate extensible energies dataset
                    if "energies" not in f:
                        f.create_dataset(
                            "energies",
                            shape=(initial_samples,),
                            maxshape=(None,),
                            chunks=(chunk_samples,),
                            dtype=np.float64,
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                            shuffle=True,
                            fletcher32=True,
                        )

                    # Preallocate extensible forces dataset
                    if "forces" not in f:
                        f.create_dataset(
                            "forces",
                            shape=(initial_samples, max_atoms, 3),
                            maxshape=(None, max_atoms, 3),
                            chunks=(chunk_samples, max_atoms, 3),
                            dtype=np.float64,
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                            shuffle=True,
                            fletcher32=True,
                        )

                    # Preallocate extensible uncertainties dataset
                    if "uncertainties" not in f:
                        f.create_dataset(
                            "uncertainties",
                            shape=(initial_samples,),
                            maxshape=(None,),
                            chunks=(chunk_samples,),
                            dtype=np.float64,
                            compression=self.compression,
                            compression_opts=self.compression_opts,
                            shuffle=True,
                            fletcher32=True,
                        )

                    # Enable Single-Writer Multiple-Reader (SWMR) mode
                    if not f.swmr_mode:
                        f.swmr_mode = True

    def append_batch(
        self,
        coordinates: np.ndarray,
        energies: np.ndarray,
        forces: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> None:
        """Append batch of molecular geometries and observables under dual locking [M]."""
        coords_arr = np.asarray(coordinates, dtype=np.float64)
        energies_arr = np.asarray(energies, dtype=np.float64)
        forces_arr = np.asarray(forces, dtype=np.float64)

        n_samples = coords_arr.shape[0]
        n_atoms = coords_arr.shape[1]

        if uncertainties is not None:
            uncert_arr = np.asarray(uncertainties, dtype=np.float64)
        else:
            uncert_arr = np.asarray([0.0] * n_samples, dtype=np.float64)

        with self._thread_lock:
            with self._file_lock:
                if not self.filepath.exists():
                    self.initialize_swmr_datasets(max_atoms=n_atoms)

                with h5py.File(self.filepath, "a", libver="latest") as f:
                    if "coordinates" not in f:
                        self.initialize_swmr_datasets(max_atoms=n_atoms)

                    if not f.swmr_mode:
                        f.swmr_mode = True

                    dset_coords = f["coordinates"]
                    dset_energies = f["energies"]
                    dset_forces = f["forces"]
                    dset_uncert = f["uncertainties"]

                    curr_len = dset_coords.shape[0]
                    new_len = curr_len + n_samples

                    # Resize extensible boundaries
                    dset_coords.resize((new_len, dset_coords.shape[1], 3))
                    dset_energies.resize((new_len,))
                    dset_forces.resize((new_len, dset_forces.shape[1], 3))
                    dset_uncert.resize((new_len,))

                    # Pad or crop atomic coordinates to preallocated atom dimension
                    target_atoms = dset_coords.shape[1]
                    if n_atoms < target_atoms:
                        padded_coords = np.pad(coords_arr, ((0, 0), (0, target_atoms - n_atoms), (0, 0)), mode="constant")
                        padded_forces = np.pad(forces_arr, ((0, 0), (0, target_atoms - n_atoms), (0, 0)), mode="constant")
                        dset_coords[curr_len:new_len] = padded_coords
                        dset_forces[curr_len:new_len] = padded_forces
                    else:
                        dset_coords[curr_len:new_len] = coords_arr[:, :target_atoms, :]
                        dset_forces[curr_len:new_len] = forces_arr[:, :target_atoms, :]

                    dset_energies[curr_len:new_len] = energies_arr
                    dset_uncert[curr_len:new_len] = uncert_arr

                    f.flush()


# Backwards compatibility alias
HDF5TorqStorage = HDF5StorageManager
