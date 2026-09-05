"""Thread-Safe HDF5 Storage & Atomic Checkpointing Engine (REQ-TORQ-TRAIN-097 [M]).

Features:
- Chunked HDF5 storage with gzip (level 4), shuffle, and fletcher32 checksums.
- Cross-process advisory locking with filelock placed in the dataset mount directory.
- Dynamic cluster file locking override: HDF5_USE_FILE_LOCKING="FALSE".
- Multi-worker PyTorch DataLoader worker initialization hook.
- Atomic checkpoint serialization (.pt.tmp -> fsync -> os.replace -> .sha256 verification).
- CheckpointCorruptionError and HDF5LockTimeoutError enforcement.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import filelock
import h5py
import numpy as np
import torch

from Libraries.cochem_torq_training_errors import (
    CheckpointCorruptionError,
    HDF5LockTimeoutError,
)


def configure_cluster_hdf5_environment() -> None:
    """Disable native HDF5 kernel file locking to prevent Lustre/NFS cluster deadlocks. [M]"""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


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
        self,
        group_name: str,
        coordinates: np.ndarray,
        atomic_numbers: Sequence[int],
        energies: np.ndarray,
        forces: np.ndarray,
    ) -> None:
        """Atomically append a trajectory batch with chunking, shuffle, and fletcher32. [M]"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._file_lock:
                h5_kwargs = (
                    {"libver": "latest", "swmr": True}
                    if sys.platform != "win32"
                    else {}
                )
                with h5py.File(self.filepath, "a", **h5_kwargs) as f:
                    grp = f.require_group(group_name)

                    # Store atomic numbers
                    if "atomic_numbers" not in grp:
                        grp.create_dataset(
                            "atomic_numbers",
                            data=np.array(atomic_numbers, dtype=np.int32),
                        )

                    n_samples = coordinates.shape[0]
                    chunk_size = min(32, max(1, n_samples))

                    # Store coordinates
                    if "coordinates" not in grp:
                        grp.create_dataset(
                            "coordinates",
                            data=coordinates,
                            maxshape=(None, *coordinates.shape[1:]),
                            chunks=(chunk_size, *coordinates.shape[1:]),
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                            fletcher32=True,
                        )
                    else:
                        dset = grp["coordinates"]
                        curr_len = dset.shape[0]
                        dset.resize((curr_len + n_samples, *coordinates.shape[1:]))
                        dset[curr_len : curr_len + n_samples] = coordinates

                    # Store energies
                    if "energies" not in grp:
                        grp.create_dataset(
                            "energies",
                            data=energies,
                            maxshape=(None,),
                            chunks=(chunk_size,),
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                            fletcher32=True,
                        )
                    else:
                        dset = grp["energies"]
                        curr_len = dset.shape[0]
                        dset.resize((curr_len + n_samples,))
                        dset[curr_len : curr_len + n_samples] = energies

                    # Store forces
                    if "forces" not in grp:
                        grp.create_dataset(
                            "forces",
                            data=forces,
                            maxshape=(None, *forces.shape[1:]),
                            chunks=(chunk_size, *forces.shape[1:]),
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                            fletcher32=True,
                        )
                    else:
                        dset = grp["forces"]
                        curr_len = dset.shape[0]
                        dset.resize((curr_len + n_samples, *forces.shape[1:]))
                        dset[curr_len : curr_len + n_samples] = forces

                    f.flush()
        except filelock.Timeout as exc:
            raise HDF5LockTimeoutError(
                f"Timed out after {self.timeout_seconds}s waiting for advisory HDF5 lock at {self.lock_path}.",
                diagnostics={"lock_path": str(self.lock_path), "timeout": self.timeout_seconds},
            ) from exc


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
            f"Missing cryptographic signature file for checkpoint: {sha_path}",
            diagnostics={"checkpoint_path": str(ckpt_path), "missing_signature": str(sha_path)},
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
