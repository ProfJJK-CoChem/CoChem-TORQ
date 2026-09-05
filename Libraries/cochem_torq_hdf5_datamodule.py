"""Chunked HDF5 PyTorch Lightning DataModule with SWMR concurrency and jagged graph collation.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic IO, worker isolation, and jagged batch assembly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from Libraries.cochem_torq_inference_errors import HDF5DataModuleLockError
from Libraries.cochem_torq_inference_schemas import ChunkedHDF5DataModuleConfig


def h5_worker_init_fn(worker_id: int) -> None:
    """Worker initialization hook ensuring each subprocess instantiates an isolated HDF5 handle. [M]"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        raw_dataset = worker_info.dataset
        root_ds = getattr(raw_dataset, "dataset", raw_dataset)
        is_posix = sys.platform != "win32"
        use_locking = (
            os.environ.get("HDF5_USE_FILE_LOCKING", "TRUE").upper() != "FALSE"
        )
        enable_swmr = is_posix and use_locking
        cache_bytes = getattr(root_ds, "chunk_cache_bytes", 16 * 1024 * 1024)
        cache_slots = getattr(root_ds, "chunk_cache_slots", 10007)

        root_ds.handle = h5py.File(
            str(root_ds.h5_path),
            mode="r",
            swmr=enable_swmr,
            libver="latest" if enable_swmr else "earliest",
            rdcc_nbytes=cache_bytes,
            rdcc_nslots=cache_slots,
        )


def jagged_graph_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized molecular graphs into a unified contiguous batch representation. [D]
    
    Constructs:
    - Node batch vector b in {0 ... B-1}^{sum N_k}
    - Shifted edge index e'_k = e_k + sum_{j < k} N_j concatenated into shape (2, sum E_k)
    - Concatenated coordinates, atomic numbers, and forces
    - Stacked energies
    """
    if not batch:
        return {}

    b_size = len(batch)
    coords_list: List[torch.Tensor] = []
    z_list: List[torch.Tensor] = []
    energy_list: List[torch.Tensor] = []
    forces_list: List[torch.Tensor] = []
    shifted_edges: List[torch.Tensor] = []
    batch_ptrs: List[torch.Tensor] = []

    cumulative_nodes = 0

    for graph_idx, sample in enumerate(batch):
        coords = sample["coordinates"]
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(dtype=torch.float32)

        z = sample["atomic_numbers"]
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.to(dtype=torch.int64)

        energy = sample["energy"]
        if isinstance(energy, (float, int, np.number)):
            energy = torch.tensor(float(energy), dtype=torch.float64)
        elif isinstance(energy, np.ndarray):
            energy = torch.from_numpy(energy).to(dtype=torch.float64)
        elif isinstance(energy, torch.Tensor):
            energy = energy.to(dtype=torch.float64)

        forces = sample["forces"]
        if isinstance(forces, np.ndarray):
            forces = torch.from_numpy(forces)
        forces = forces.to(dtype=torch.float32)

        edge_index = sample.get("edge_index", None)
        if edge_index is not None:
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.from_numpy(edge_index)
            edge_index = edge_index.to(dtype=torch.int64)

        n_atoms = coords.shape[0]
        coords_list.append(coords)
        z_list.append(z)
        energy_list.append(energy.view(-1))
        forces_list.append(forces)

        # Batch assignment pointer b_i = graph_idx
        b_vec = torch.full((n_atoms,), graph_idx, dtype=torch.int64)
        batch_ptrs.append(b_vec)

        # Shift edge index
        if edge_index is not None and edge_index.numel() > 0:
            shifted_edges.append(edge_index + cumulative_nodes)

        cumulative_nodes += n_atoms

    out_coords = torch.cat(coords_list, dim=0)
    out_z = torch.cat(z_list, dim=0)
    out_energy = torch.cat(energy_list, dim=0)
    out_forces = torch.cat(forces_list, dim=0)
    out_batch = torch.cat(batch_ptrs, dim=0)

    if shifted_edges:
        out_edge_index = torch.cat(shifted_edges, dim=1)
    else:
        out_edge_index = torch.empty((2, 0), dtype=torch.int64)

    return {
        "coordinates": out_coords,
        "atomic_numbers": out_z,
        "energy": out_energy,
        "forces": out_forces,
        "edge_index": out_edge_index,
        "batch": out_batch,
        "num_graphs": b_size,
        "num_nodes": cumulative_nodes,
    }


class ChunkedHDF5Dataset(Dataset):
    """Authentic HDF5 dataset reader supporting both group-based and array-based chunked schemas. [M]"""

    def __init__(
        self,
        h5_path: Union[str, Path],
        chunk_cache_bytes: int = 16 * 1024 * 1024,
        chunk_cache_slots: int = 10007,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.chunk_cache_bytes = chunk_cache_bytes
        self.chunk_cache_slots = chunk_cache_slots
        self.handle: Optional[h5py.File] = None
        self._len: Optional[int] = None
        self._is_group_format: Optional[bool] = None

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        # Quick inspection of length in main process
        with h5py.File(str(self.h5_path), mode="r") as f:
            if "coordinates" in f:
                self._is_group_format = False
                self._len = len(f["coordinates"])
            else:
                self._is_group_format = True
                self._len = len([k for k in f.keys() if k.startswith("mol_") or k.startswith("sample_")])

    def _open_handle(self) -> h5py.File:
        """Lazily instantiate thread-safe HDF5 handle in worker or main process. [D]"""
        if self.handle is None:
            is_posix = sys.platform != "win32"
            use_locking = (
                os.environ.get("HDF5_USE_FILE_LOCKING", "TRUE").upper() != "FALSE"
            )
            enable_swmr = is_posix and use_locking
            self.handle = h5py.File(
                str(self.h5_path),
                mode="r",
                swmr=enable_swmr,
                libver="latest" if enable_swmr else "earliest",
                rdcc_nbytes=self.chunk_cache_bytes,
                rdcc_nslots=self.chunk_cache_slots,
            )
        return self.handle

    def __len__(self) -> int:
        if self._len is not None:
            return self._len
        h5 = self._open_handle()
        if "coordinates" in h5:
            self._len = len(h5["coordinates"])
        else:
            self._len = len([k for k in h5.keys() if k.startswith("mol_") or k.startswith("sample_")])
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5 = self._open_handle()
        if not self._is_group_format and "coordinates" in h5:
            coords = torch.from_numpy(h5["coordinates"][idx]).float()
            z = torch.from_numpy(h5["atomic_numbers"][idx]).long()
            energy = float(h5["energy"][idx])
            forces = torch.from_numpy(h5["forces"][idx]).float()
            if "edge_index" in h5:
                edge_index = torch.from_numpy(h5["edge_index"][idx]).long()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.int64)
        else:
            key = f"sample_{idx}" if f"sample_{idx}" in h5 else f"mol_{idx}"
            grp = h5[key]
            coords = torch.from_numpy(grp["coordinates"][()]).float()
            z = torch.from_numpy(grp["atomic_numbers"][()]).long()
            energy = float(grp["energy"][()])
            forces = torch.from_numpy(grp["forces"][()]).float()
            if "edge_index" in grp:
                edge_index = torch.from_numpy(grp["edge_index"][()]).long()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.int64)

        return {
            "coordinates": coords,
            "atomic_numbers": z,
            "energy": energy,
            "forces": forces,
            "edge_index": edge_index,
        }

    def close(self) -> None:
        """Close open file handle cleanly. [M]"""
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:
                self.handle = None
            self.handle = None


class ChunkedHDF5DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule orchestrating partitioned HDF5 ingestion with SWMR safety. [M]"""

    def __init__(self, config: ChunkedHDF5DataModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.dataset: Optional[ChunkedHDF5Dataset] = None
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def prepare_data(self) -> None:
        """Verify dataset presence and SHA-256 integrity without allocating memory in parent. [M]"""
        path = self.config.h5_path
        if not path.exists():
            raise FileNotFoundError(f"Configured HDF5 path does not exist: {path}")

        # Check for matching .sha256 digest file
        digest_path = path.with_suffix(path.suffix + ".sha256")
        if digest_path.exists():
            expected_hash = digest_path.read_text().strip().split()[0]
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            computed_hash = sha.hexdigest()
            if computed_hash.lower() != expected_hash.lower():
                raise HDF5DataModuleLockError(
                    f"HDF5 checksum mismatch: expected {expected_hash}, computed {computed_hash}.",
                    diagnostics={"h5_path": str(path), "expected": expected_hash, "computed": computed_hash},
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Partition index offsets into deterministic train, val, and test Subset splits. [M]"""
        self.dataset = ChunkedHDF5Dataset(
            h5_path=self.config.h5_path,
            chunk_cache_bytes=self.config.chunk_cache_bytes,
            chunk_cache_slots=self.config.chunk_cache_slots,
        )

        total = len(self.dataset)
        splits = self.config.train_val_test_split

        n_train = int(total * splits[0])
        n_val = int(total * splits[1])
        n_test = total - n_train - n_val

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total, generator=generator).tolist()

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)

    def train_dataloader(self) -> DataLoader:
        """Return configured train DataLoader with jagged collation and worker isolation. [M]"""
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None

        ctx = "spawn" if (self.config.num_workers > 0 and sys.platform != "win32") else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=jagged_graph_collate,
            worker_init_fn=h5_worker_init_fn if self.config.num_workers > 0 else None,
            multiprocessing_context=ctx,
        )

    def val_dataloader(self) -> DataLoader:
        """Return configured validation DataLoader. [M]"""
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None

        ctx = "spawn" if (self.config.num_workers > 0 and sys.platform != "win32") else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=jagged_graph_collate,
            worker_init_fn=h5_worker_init_fn if self.config.num_workers > 0 else None,
            multiprocessing_context=ctx,
        )

    def test_dataloader(self) -> DataLoader:
        """Return configured test DataLoader. [M]"""
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None

        ctx = "spawn" if (self.config.num_workers > 0 and sys.platform != "win32") else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=jagged_graph_collate,
            worker_init_fn=h5_worker_init_fn if self.config.num_workers > 0 else None,
            multiprocessing_context=ctx,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Cleanly close open HDF5 file handles upon shutdown. [M]"""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
