"""High-Concurrency HDF5 SWMR PES Trajectory Store.

Enforces cross-platform file locking during schema initialization, driver-level
filesystem lock bypass for network mounts, and lockless Single-Writer Multiple-Reader
(SWMR) high-throughput trajectory persistence.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set driver-level bypass before importing h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from filelock import FileLock  # noqa: E402

from cochem.storage.hdf5_zstd import HDF5ZstdConfig  # noqa: E402


class SWMRPESStore:
    """Single-Writer Multiple-Reader (SWMR) store for PES trajectories."""

    def __init__(
        self,
        h5_path: Union[str, Path],
        n_atoms: int = 1,
        zstd_config: Optional[HDF5ZstdConfig] = None,
        enable_swmr: bool = True,
    ) -> None:
        """Initialize PES trajectory store, pre-allocating extensible schemas.

        Args:
            h5_path: Path to target HDF5 file.
            n_atoms: Number of atoms per system frame.
            zstd_config: Compression configuration (defaults to HDF5ZstdConfig()).
            enable_swmr: Whether to activate SWMR access mode.
        """
        self.h5_path = Path(h5_path).resolve()
        self.lock_path = self.h5_path.with_suffix(".lock")
        self.n_atoms = n_atoms
        self.zstd_config = zstd_config or HDF5ZstdConfig()
        self.enable_swmr = enable_swmr
        self._writer_file: Optional[h5py.File] = None

        self._preallocate_schema()

    def _preallocate_schema(self) -> None:
        """Pre-allocate extensible coordinate and energy datasets under exclusive file lock."""
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path), timeout=10.0):
            if not self.h5_path.exists():
                with h5py.File(self.h5_path, "w", libver="latest") as f:
                    coords_chunk = self.zstd_config.resolve_chunk_shape_3d(
                        self.n_atoms, 3, 8
                    )
                    energy_chunk = self.zstd_config.resolve_chunk_shape_1d(8)

                    coords_kwargs = self.zstd_config.get_dataset_kwargs(
                        coords_chunk
                    )
                    energy_kwargs = self.zstd_config.get_dataset_kwargs(
                        energy_chunk
                    )

                    f.create_dataset(
                        "coordinates",
                        shape=(0, self.n_atoms, 3),
                        maxshape=(None, self.n_atoms, 3),
                        dtype="float64",
                        **coords_kwargs,
                    )
                    f.create_dataset(
                        "energies",
                        shape=(0,),
                        maxshape=(None,),
                        dtype="float64",
                        **energy_kwargs,
                    )

                    f.attrs["schema_version"] = "1.0.0"
                    f.attrs["n_atoms"] = self.n_atoms
                    f.flush()

    def open_writer(self) -> None:
        """Open persistent writer file handle and activate SWMR mode."""
        if self._writer_file is None:
            self._writer_file = h5py.File(
                self.h5_path, "r+", libver="latest"
            )
            if self.enable_swmr and not self._writer_file.swmr_mode:
                self._writer_file.swmr_mode = True

    def close_writer(self) -> None:
        """Flush and close persistent writer handle."""
        if self._writer_file is not None:
            self._writer_file.flush()
            self._writer_file.close()
            self._writer_file = None

    def __enter__(self) -> "SWMRPESStore":
        """Enter context manager, opening writer handle."""
        self.open_writer()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context manager, closing writer handle."""
        self.close_writer()

    def append_frames(
        self,
        coords: Union[np.ndarray, List[Any]],
        energies: Union[np.ndarray, float, List[float]],
    ) -> None:
        """Append trajectory coordinate frames and energies atomically.

        Supports active persistent writer handle or acquires short-lived exclusive lock.

        Args:
            coords: Array of shape (N, n_atoms, 3) or (n_atoms, 3).
            energies: Array of shape (N,) or scalar float.
        """
        coords_arr = np.asarray(coords, dtype=np.float64)  # type: ignore[type-var,arg-type]
        energies_arr = np.asarray(energies, dtype=np.float64)  # type: ignore[type-var,arg-type]

        if coords_arr.ndim == 2:
            coords_arr = np.expand_dims(coords_arr, axis=0)

        if energies_arr.ndim == 0:
            energies_arr = np.expand_dims(energies_arr, axis=0)

        if (
            coords_arr.ndim != 3
            or coords_arr.shape[1] != self.n_atoms
            or coords_arr.shape[2] != 3
        ):
            raise ValueError(
                f"Expected coords shape (N, {self.n_atoms}, 3), got {coords_arr.shape}"
            )

        if (
            energies_arr.ndim != 1
            or energies_arr.shape[0] != coords_arr.shape[0]
        ):
            raise ValueError(
                f"Frame count mismatch: coords has {coords_arr.shape[0]} frames, "
                f"energies has {energies_arr.shape[0]} points"
            )

        n_new = coords_arr.shape[0]

        if self._writer_file is not None:
            ds_coords = self._writer_file["coordinates"]
            ds_energy = self._writer_file["energies"]
            cur_len = ds_coords.shape[0]

            ds_coords.resize((cur_len + n_new, self.n_atoms, 3))
            ds_energy.resize((cur_len + n_new,))

            ds_coords[cur_len : cur_len + n_new] = coords_arr
            ds_energy[cur_len : cur_len + n_new] = energies_arr

            ds_coords.flush()
            ds_energy.flush()
            self._writer_file.flush()
        else:
            with FileLock(str(self.lock_path), timeout=30.0):
                with h5py.File(self.h5_path, "r+", libver="latest") as f:
                    if self.enable_swmr and not f.swmr_mode:
                        f.swmr_mode = True

                    ds_coords = f["coordinates"]
                    ds_energy = f["energies"]
                    cur_len = ds_coords.shape[0]

                    ds_coords.resize((cur_len + n_new, self.n_atoms, 3))
                    ds_energy.resize((cur_len + n_new,))

                    ds_coords[cur_len : cur_len + n_new] = coords_arr
                    ds_energy[cur_len : cur_len + n_new] = energies_arr

                    ds_coords.flush()
                    ds_energy.flush()
                    f.flush()

    def read_trajectory(
        self, max_retries: int = 5, retry_delay_s: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """Perform lockless non-blocking read with SWMR dataset refresh synchronization.

        Args:
            max_retries: Maximum refresh retry attempts during writer flush boundaries.
            retry_delay_s: Initial delay between retries (exponential backoff).

        Returns:
            Dictionary with 'coordinates' (N, n_atoms, 3) and 'energies' (N,) arrays.
        """
        for attempt in range(max_retries):
            try:
                with h5py.File(
                    self.h5_path, "r", libver="latest", swmr=self.enable_swmr
                ) as f:
                    if self.enable_swmr:
                        f["coordinates"].refresh()
                        f["energies"].refresh()

                    coords_copy = f["coordinates"][:].copy()
                    energies_copy = f["energies"][:].copy()
                    return {
                        "coordinates": coords_copy,
                        "energies": energies_copy,
                    }
            except (RuntimeError, OSError, KeyError) as err:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_s * (2**attempt))
                else:
                    raise RuntimeError(
                        "SWMR reader refresh failed to synchronize."
                    ) from err

        raise RuntimeError("SWMR reader refresh failed to synchronize.")
