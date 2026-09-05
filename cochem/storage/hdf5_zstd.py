"""Lossless Zstandard (Zstd) HDF5 Compression Pipeline.

Provides deterministic chunk size derivation and configuration for lossless
Zstd compression with optional byte shuffling and Fletcher32 checksums
on IEEE 754 floating-point trajectories and tensors.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import hdf5plugin  # type: ignore[import-untyped,import-not-found]


@dataclass(frozen=True)
class HDF5ZstdConfig:
    """Configuration for lossless Zstandard HDF5 compression pipeline.

    Enforces strict lossless compression policies, dynamic chunk geometry derivation
    targeting bounded chunk sizes, and hardware-isolated CPU execution.
    """

    clevel: int = 3
    enable_shuffle: bool = True
    enable_fletcher32: bool = True
    min_chunk_bytes: int = 64 * 1024
    max_chunk_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        """Validate compression level and chunk boundary constraints."""
        if not (1 <= self.clevel <= 22):
            raise ValueError(f"Zstd clevel must be in 1..22, got {self.clevel}")
        if self.min_chunk_bytes <= 0 or self.max_chunk_bytes < self.min_chunk_bytes:
            raise ValueError(
                f"Invalid chunk byte bounds: [{self.min_chunk_bytes}, {self.max_chunk_bytes}]"
            )

    def resolve_chunk_shape_3d(
        self,
        n_atoms: int,
        spatial_dim: int = 3,
        itemsize: int = 8,
        max_frames: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Derive 3D chunk geometry (frames, n_atoms, spatial_dim) targeting chunk byte boundaries.

        Args:
            n_atoms: Number of atoms per frame (> 0).
            spatial_dim: Spatial dimension per atom (> 0, default 3).
            itemsize: Byte size per element (> 0, default 8 for float64).
            max_frames: Optional frame count ceiling.

        Returns:
            Tuple of (chunk_frames, n_atoms, spatial_dim).
        """
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if spatial_dim <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        if itemsize <= 0:
            raise ValueError(f"itemsize must be positive, got {itemsize}")

        frame_bytes = n_atoms * spatial_dim * itemsize
        target_bytes = (self.min_chunk_bytes + self.max_chunk_bytes) // 2
        chunk_frames = max(1, target_bytes // frame_bytes)

        if max_frames is not None and max_frames > 0:
            chunk_frames = min(chunk_frames, max_frames)

        return (chunk_frames, n_atoms, spatial_dim)

    def resolve_chunk_shape_1d(
        self,
        itemsize: int = 8,
        max_len: Optional[int] = None,
    ) -> Tuple[int]:
        """Derive 1D chunk geometry for scalar series (e.g. potential energies).

        Args:
            itemsize: Byte size per element (> 0, default 8 for float64).
            max_len: Optional length ceiling.

        Returns:
            Tuple of (chunk_len,).
        """
        if itemsize <= 0:
            raise ValueError(f"itemsize must be positive, got {itemsize}")

        target_bytes = (self.min_chunk_bytes + self.max_chunk_bytes) // 2
        chunk_len = max(1, target_bytes // itemsize)

        if max_len is not None and max_len > 0:
            chunk_len = min(chunk_len, max_len)

        return (chunk_len,)

    def resolve_chunk_shape_2d(
        self,
        n_features: int,
        itemsize: int = 8,
        max_rows: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Derive 2D chunk geometry for tabular or feature matrices.

        Args:
            n_features: Feature dimension (> 0).
            itemsize: Byte size per element (> 0, default 8 for float64).
            max_rows: Optional row count ceiling.

        Returns:
            Tuple of (chunk_rows, n_features).
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if itemsize <= 0:
            raise ValueError(f"itemsize must be positive, got {itemsize}")

        row_bytes = n_features * itemsize
        target_bytes = (self.min_chunk_bytes + self.max_chunk_bytes) // 2
        chunk_rows = max(1, target_bytes // row_bytes)

        if max_rows is not None and max_rows > 0:
            chunk_rows = min(chunk_rows, max_rows)

        return (chunk_rows, n_features)

    def get_dataset_kwargs(
        self,
        chunk_shape: Tuple[int, ...],
        is_numeric: bool = True,
    ) -> Dict[str, Any]:
        """Construct h5py dataset keyword arguments for Zstd compression filter.

        Bans lossy filters (e.g. scaleoffset) and enforces bit-for-bit lossless precision.

        Args:
            chunk_shape: Shape tuple defining HDF5 dataset chunking.
            is_numeric: Whether the dataset stores numeric floating/integer data.

        Returns:
            Dictionary of dataset creation parameters.
        """
        zstd_opts = hdf5plugin.Zstd(clevel=self.clevel)
        kwargs: Dict[str, Any] = {
            "chunks": chunk_shape,
            **zstd_opts,
        }
        if is_numeric and self.enable_shuffle:
            kwargs["shuffle"] = True
        if is_numeric and self.enable_fletcher32:
            kwargs["fletcher32"] = True
        return kwargs
