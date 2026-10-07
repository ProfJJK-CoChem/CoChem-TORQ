"""Thread-Safe HDF5 Storage & Atomic Checkpointing Engine (REQ-TORQ-TRAIN-097 [M]).

Features:
- Chunked HDF5 storage with gzip (level 4), shuffle, and fletcher32 checksums.
- Cross-process advisory locking with filelock placed in the dataset mount directory.
- Native HDF5 locking remains enabled; shared network filesystems require qualification.
- Multi-worker PyTorch DataLoader worker initialization hook.
- Atomic checkpoint serialization (.pt.tmp -> fsync -> os.replace -> .sha256 verification).
- CheckpointCorruptionError and HDF5LockTimeoutError enforcement.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, Sequence, Tuple

import filelock
import h5py
import numpy as np
import torch

from Libraries.cochem_torq_training_errors import (
    CheckpointCorruptionError,
    HDF5LockTimeoutError,
)


def configure_cluster_hdf5_environment() -> None:
    """Validate native locking without changing process-wide HDF5 behavior."""
    if os.environ.get("HDF5_USE_FILE_LOCKING", "").strip().upper() in {"FALSE", "0"}:
        raise ValueError("HDF5 file locking must remain enabled; use qualified local storage")


class HDF5DatasetManager:
    """Thread-safe and multi-process safe HDF5 manager for MLFF trajectory storage. [M]"""

    def __init__(
        self,
        filepath: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.filepath = Path(filepath)
        configure_cluster_hdf5_environment()
        # Advisory lock resides in the shared cluster mount directory containing the dataset
        self.lock_path = self.filepath.with_suffix(".h5.lock")
        self.timeout_seconds = timeout_seconds
        self._file_lock = filelock.FileLock(str(self.lock_path), timeout=self.timeout_seconds)
        self._swmr_enabled = (sys.platform != "win32")

    def write_trajectory_batch(
        self, group_name: str, coordinates: np.ndarray, atomic_numbers: Sequence[int],
        energies: np.ndarray, forces: np.ndarray,
    ) -> None:
        """Append a validated batch under exclusive ownership, then publish its length."""
        coords = np.asarray(coordinates, dtype=np.float64)
        energy = np.asarray(energies, dtype=np.float64)
        force = np.asarray(forces, dtype=np.float64)
        z = np.asarray(atomic_numbers)
        if coords.ndim != 3 or coords.shape[2] != 3 or coords.shape[1] < 1:
            raise ValueError("coordinates must have shape (samples, atoms, 3)")
        if force.shape != coords.shape or energy.shape != (len(coords),):
            raise ValueError("Force shape and energy count must exactly match coordinate samples")
        if (z.shape != (coords.shape[1],) or not np.issubdtype(z.dtype, np.integer)
                or np.any(z < 1) or np.any(z > 118)):
            raise ValueError("Atomic numbers must be an ordered integer vector in 1..118")
        if not all(np.isfinite(array).all() for array in (coords, energy, force)):
            raise ValueError("Training coordinates, forces and energies must be finite")
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._file_lock:
                # Group creation is incompatible with a preallocated SWMR schema.
                # This API serializes complete transactions, including readers.
                with h5py.File(self.filepath, "a", libver="latest", locking=True) as f:
                    if group_name in f:
                        grp = f[group_name]
                        start = self._committed_length(grp)
                        if not np.array_equal(grp["atomic_numbers"][:], z):
                            raise ValueError("Cannot change ordered atomic identity within a training group")
                        if grp["coordinates"].shape[1:] != coords.shape[1:]:
                            raise ValueError("Cannot change training group atom dimension")
                    else:
                        grp = f.create_group(group_name)
                        grp.attrs["schema_version"] = "2.0.0"
                        grp.create_dataset("atomic_numbers", data=z.astype(np.int32))
                        for key, values in (("coordinates", coords), ("energies", energy), ("forces", force)):
                            grp.create_dataset(key, shape=(0,) + values.shape[1:],
                                maxshape=(None,) + values.shape[1:],
                                chunks=(32,) + values.shape[1:], dtype="float64", compression="gzip",
                                compression_opts=4, shuffle=True, fletcher32=True, fillvalue=np.nan)
                        grp.create_dataset("committed_frames", data=np.array([0], dtype="uint64"),
                                           chunks=(1,), fletcher32=True)
                        start = 0
                    end = start + len(coords)
                    for key, values in (("coordinates", coords), ("energies", energy), ("forces", force)):
                        grp[key].resize((end,) + values.shape[1:])
                        grp[key][start:end] = values
                        grp[key].flush()
                    f.flush()
                    grp["committed_frames"][0] = end
                    grp["committed_frames"].flush()
                    f.flush()
        except filelock.Timeout as exc:
            raise HDF5LockTimeoutError(
                f"Timed out waiting for HDF5 ownership at {self.lock_path}.",
                diagnostics={"lock_path": str(self.lock_path), "timeout": self.timeout_seconds},
            ) from exc

    @staticmethod
    def _committed_length(group: h5py.Group) -> int:
        if group.attrs.get("schema_version") != "2.0.0" or "committed_frames" not in group:
            raise ValueError("Legacy training group requires explicit validated migration")
        length = int(group["committed_frames"][0])
        if any(len(group[key]) < length for key in ("coordinates", "energies", "forces")):
            raise ValueError("Training commit marker exceeds complete payloads")
        return length

    def read_trajectory_batch(self, group_name: str) -> Dict[str, np.ndarray]:
        """Read only a group's committed prefix using the same ownership lock."""
        with self._file_lock:
            with h5py.File(self.filepath, "r", libver="latest", locking=True) as f:
                group = f[group_name]
                length = self._committed_length(group)
                data = {key: group[key][:length].copy() for key in ("coordinates", "energies", "forces")}
                data["atomic_numbers"] = group["atomic_numbers"][:].copy()
                if not all(np.isfinite(data[key]).all() for key in ("coordinates", "energies", "forces")):
                    raise ValueError("Nonfinite committed training data")
                return data


def worker_init_fn(worker_id: int) -> None:
    """PyTorch DataLoader worker initialization hook ensuring independent worker file handles. [M]"""
    configure_cluster_hdf5_environment()
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Re-initialize dataset handles per worker process
        if hasattr(dataset, "reopen_handles"):
            dataset.reopen_handles(worker_id)


def compute_tensor_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a saved checkpoint file. [M]"""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_atomic_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_path: Path,
) -> Tuple[Path, Path]:
    """Atomically serialize PyTorch checkpoint and write accompanying SHA-256 digest. [M]
    
    Pipeline:
    1. Write payload to isolated temporary file checkpoint_path.tmp
    2. Flush OS buffers to physical non-volatile storage via os.fsync()
    3. Atomically rename temporary file to destination via os.replace()
    4. Generate and save matching .sha256 digest file
    
    Returns
    -------
    Tuple[Path, Path]
        (saved_checkpoint_path, saved_sha256_path)
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ckpt_path.with_name(f"{ckpt_path.name}.tmp")
    sha_path = ckpt_path.with_name(f"{ckpt_path.name}.sha256")

    # Step 1: Serialize to temporary location
    torch.save(state_dict, temp_path)

    # Step 2: Flush to physical storage
    with open(temp_path, "a+b") as f:
        f.flush()
        os.fsync(f.fileno())

    # Step 3: Compute SHA-256 digest
    sha256_digest = compute_tensor_sha256(temp_path)

    # Step 4: Atomic replacement
    os.replace(temp_path, ckpt_path)

    # Step 5: Write digest file
    sha_path.write_text(sha256_digest.strip() + "\n", encoding="utf-8")

    return ckpt_path, sha_path


def load_atomic_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Verify cryptographic SHA-256 integrity and reload model state dict. [M]
    
    Raises CheckpointCorruptionError if file is missing, checksum mismatches,
    or bytes are corrupted.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

    sha_path = ckpt_path.with_name(f"{ckpt_path.name}.sha256")
    if not sha_path.exists():
        raise CheckpointCorruptionError(
            f"Missing SHA-256 digest file for checkpoint: {sha_path}",
            diagnostics={"checkpoint_path": str(ckpt_path), "missing_digest": str(sha_path)},
        )

    expected_sha256 = sha_path.read_text(encoding="utf-8").strip()
    actual_sha256 = compute_tensor_sha256(ckpt_path)

    if actual_sha256.lower() != expected_sha256.lower():
        raise CheckpointCorruptionError(
            f"Cryptographic hash mismatch for checkpoint {ckpt_path.name}. "
            f"Expected {expected_sha256}, got {actual_sha256}.",
            diagnostics={
                "checkpoint_path": str(ckpt_path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            },
        )

    try:
        state_dict = dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        return state_dict
    except Exception as exc:
        raise CheckpointCorruptionError(
            f"Failed to unpickle checkpoint payload from {ckpt_path}: {exc}",
            diagnostics={"checkpoint_path": str(ckpt_path), "underlying_error": str(exc)},
        ) from exc
