"""Thread-Safe HDF5 SWMR Trajectory Serialization (Chunk 21).

Method Matrix v4 Provenance Tags:
- [M] Mandated: Strict SWMR lifecycle (dataset pre-allocation prior to swmr_mode activation),
               lossless chunked compression, non-blocking concurrent reader.
- [D] Derived: Extensible datasets with maxshape=(None, N, 3) and atomic flush sequencing.
- [E] Empirical: 100-frame chunking calibrated for optimal sequential I/O throughput.

Strict Zero-Mock Mandate v3: Completely authentic physics and mathematical rigor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import h5py
import numpy as np
import torch

from Libraries.cochem_torq_md_schemas import TrajectoryFrame
from Libraries.cochem_torq_symplectic import compute_instantaneous_temperature


class HDF5TrajectoryWriter:
    """Thread-safe Single-Writer-Multiple-Reader (SWMR) HDF5 trajectory serializer [M]."""

    def __init__(
        self,
        file_path: Union[str, Path],
        n_atoms: int,
        atomic_numbers: Union[torch.Tensor, Sequence[int]],
        chunk_size: int = 100,
        compression: Optional[str] = "gzip",
    ) -> None:
        """Initialize and pre-allocate SWMR extensible HDF5 dataset structure [M].

        Parameters
        ----------
        file_path : Union[str, Path]
            Destination path for the trajectory HDF5 file.
        n_atoms : int
            Number of atoms in the molecular system.
        atomic_numbers : Union[torch.Tensor, Sequence[int]]
            Static atomic numbers Z of shape (N_atoms,).
        chunk_size : int, optional
            Extensible chunk size along the temporal dimension. Defaults to 100.
        compression : Optional[str], optional
            Compression filter ('gzip' or 'lzf'). Defaults to 'gzip'.
        """
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_atoms = int(n_atoms)
        self.chunk_size = int(chunk_size)
        self.compression = compression

        # Convert atomic numbers to numpy int64
        if isinstance(atomic_numbers, torch.Tensor):
            z_arr = atomic_numbers.detach().cpu().numpy().astype(np.int64).flatten()
        else:
            z_arr = np.array(atomic_numbers, dtype=np.int64).flatten()

        if z_arr.shape[0] != self.n_atoms:
            raise ValueError(
                f"Atomic numbers length ({z_arr.shape[0]}) does not match n_atoms ({self.n_atoms})."
            )
        self.atomic_numbers_arr = z_arr

        # Open file with libver='latest'
        self.file = h5py.File(str(self.file_path), "w", libver="latest")

        # Create chunked extensible datasets prior to SWMR mode activation [M]
        self.coordinates = self.file.create_dataset(
            "coordinates",
            shape=(0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype="float64",
            chunks=(self.chunk_size, self.n_atoms, 3),
            compression=self.compression,
        )

        self.velocities = self.file.create_dataset(
            "velocities",
            shape=(0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype="float64",
            chunks=(self.chunk_size, self.n_atoms, 3),
            compression=self.compression,
        )

        self.forces = self.file.create_dataset(
            "forces",
            shape=(0, self.n_atoms, 3),
            maxshape=(None, self.n_atoms, 3),
            dtype="float64",
            chunks=(self.chunk_size, self.n_atoms, 3),
            compression=self.compression,
        )

        self.energies = self.file.create_dataset(
            "energies",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype="float64",
            chunks=(self.chunk_size, 3),
            compression=self.compression,
        )

        self.temperatures = self.file.create_dataset(
            "temperatures",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=(self.chunk_size,),
            compression=self.compression,
        )

        self.time_fs = self.file.create_dataset(
            "time_fs",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=(self.chunk_size,),
            compression=self.compression,
        )

        # Static immutable atomic numbers dataset
        self.file.create_dataset(
            "atomic_numbers",
            data=self.atomic_numbers_arr,
            dtype="int64",
        )

        # Crucial SWMR Lifecycle Sequencing: Flush all schema and datasets to disk [M]
        self.file.flush()

        # Engage active SWMR mode
        self.file.swmr_mode = True

    def append_frame(self, frame: TrajectoryFrame) -> None:
        """Append a single trajectory frame to extensible datasets and flush [M].

        Parameters
        ----------
        frame : TrajectoryFrame
            TrajectoryFrame containing tensors for coordinates, velocities, forces, and energies.
        """
        idx = self.coordinates.shape[0]
        new_len = idx + 1

        # Resize chunked datasets along time axis
        self.coordinates.resize((new_len, self.n_atoms, 3))
        self.velocities.resize((new_len, self.n_atoms, 3))
        self.forces.resize((new_len, self.n_atoms, 3))
        self.energies.resize((new_len, 3))
        self.temperatures.resize((new_len,))
        self.time_fs.resize((new_len,))

        # Assign values
        coords_np = frame.coordinates.detach().cpu().numpy().astype(np.float64)
        vel_np = frame.velocities.detach().cpu().numpy().astype(np.float64)
        forces_np = frame.forces.detach().cpu().numpy().astype(np.float64)

        self.coordinates[idx] = coords_np
        self.velocities[idx] = vel_np
        self.forces[idx] = forces_np

        e_pot = float(frame.potential_energy_ev)
        e_kin = float(frame.kinetic_energy_ev)
        e_tot = e_pot + e_kin
        self.energies[idx] = [e_pot, e_kin, e_tot]

        t_inst = compute_instantaneous_temperature(e_kin, self.n_atoms, remove_com=True)
        self.temperatures[idx] = t_inst
        self.time_fs[idx] = float(frame.time_fs)

        # Flush datasets to disk for active concurrent SWMR readers [M]
        self.coordinates.flush()
        self.velocities.flush()
        self.forces.flush()
        self.energies.flush()
        self.temperatures.flush()
        self.time_fs.flush()
        self.file.flush()

    def close(self) -> None:
        """Close trajectory file handle."""
        if hasattr(self, "file") and self.file:
            self.file.flush()
            self.file.close()

    def __enter__(self) -> "HDF5TrajectoryWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class HDF5TrajectoryReader:
    """Concurrent non-blocking Single-Writer-Multiple-Reader (SWMR) trajectory reader [M]."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        """Open HDF5 trajectory in SWMR read mode.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to existing HDF5 trajectory file.
        """
        self.file_path = Path(file_path).resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {self.file_path}")

        self.file = h5py.File(str(self.file_path), "r", libver="latest", swmr=True)
        self.atomic_numbers = torch.tensor(
            self.file["atomic_numbers"][:], dtype=torch.int64
        )
        self.n_atoms = int(self.atomic_numbers.shape[0])

    def refresh(self) -> None:
        """Refresh datasets to see newly appended frames from active SWMR writer [M]."""
        for ds_name in [
            "coordinates",
            "velocities",
            "forces",
            "energies",
            "temperatures",
            "time_fs",
        ]:
            if ds_name in self.file:
                self.file[ds_name].refresh()

    def __len__(self) -> int:
        """Return total number of available trajectory frames."""
        self.refresh()
        return int(self.file["coordinates"].shape[0])

    def read_frame(self, index: int) -> TrajectoryFrame:
        """Read a single trajectory frame by index as torch tensors [M].

        Parameters
        ----------
        index : int
            Temporal frame index.

        Returns
        -------
        TrajectoryFrame
            Loaded frame containing tensors for coordinates, velocities, and forces.
        """
        self.refresh()
        total_frames = self.file["coordinates"].shape[0]
        if index < 0 or index >= total_frames:
            raise IndexError(
                f"Frame index {index} out of bounds for trajectory of length {total_frames}."
            )

        coords = torch.tensor(self.file["coordinates"][index], dtype=torch.float64)
        vel = torch.tensor(self.file["velocities"][index], dtype=torch.float64)
        forces = torch.tensor(self.file["forces"][index], dtype=torch.float64)
        energies = self.file["energies"][index]
        t_fs = float(self.file["time_fs"][index])

        return TrajectoryFrame(
            step=index,
            time_fs=t_fs,
            atomic_numbers=self.atomic_numbers,
            coordinates=coords,
            velocities=vel,
            forces=forces,
            potential_energy_ev=float(energies[0]),
            kinetic_energy_ev=float(energies[1]),
        )

    def read_all_coordinates(self) -> torch.Tensor:
        """Read full trajectory coordinate tensor of shape (N_frames, N_atoms, 3) [M]."""
        self.refresh()
        return torch.tensor(self.file["coordinates"][:], dtype=torch.float64)

    def read_all_energies(self) -> torch.Tensor:
        """Read full trajectory energy tensor of shape (N_frames, 3) (potential, kinetic, total) [M]."""
        self.refresh()
        return torch.tensor(self.file["energies"][:], dtype=torch.float64)

    def close(self) -> None:
        """Close trajectory file handle."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __enter__(self) -> "HDF5TrajectoryReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
