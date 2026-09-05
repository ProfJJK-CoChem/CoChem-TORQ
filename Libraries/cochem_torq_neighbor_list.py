"""GPU-Accelerated Spatial Neighbor-List Generator with Seamless CPU/MPS Fallback.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic spatial metrics, zero self-interaction, exact symmetry.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from Libraries.cochem_torq_inference_errors import HardwareDispatchError
from Libraries.cochem_torq_inference_schemas import NeighborListResult

# Guarded Triton import for Windows NT / macOS / CPU portability
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_neighbor_list_kernel(
        coords_ptr,
        dist_out_ptr,
        mask_out_ptr,
        n_atoms,
        cutoff_sq,
        stride_n,
        stride_c,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Block-tiled pairwise spatial distance evaluation in Triton. [D]"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        mask_m = offs_m < n_atoms
        mask_n = offs_n < n_atoms

        # Load coordinates
        x_m = tl.load(coords_ptr + offs_m * stride_n + 0 * stride_c, mask=mask_m, other=0.0)
        y_m = tl.load(coords_ptr + offs_m * stride_n + 1 * stride_c, mask=mask_m, other=0.0)
        z_m = tl.load(coords_ptr + offs_m * stride_n + 2 * stride_c, mask=mask_m, other=0.0)

        x_n = tl.load(coords_ptr + offs_n * stride_n + 0 * stride_c, mask=mask_n, other=0.0)
        y_n = tl.load(coords_ptr + offs_n * stride_n + 1 * stride_c, mask=mask_n, other=0.0)
        z_n = tl.load(coords_ptr + offs_n * stride_n + 2 * stride_c, mask=mask_n, other=0.0)

        # Coordinate differences
        dx = x_n[None, :] - x_m[:, None]
        dy = y_n[None, :] - y_m[:, None]
        dz = z_n[None, :] - z_m[:, None]

        dist_sq = dx * dx + dy * dy + dz * dz

        # Exclude self-interaction (offs_m == offs_n) and filter dist_sq <= cutoff_sq
        valid = (dist_sq <= cutoff_sq) & (dist_sq > 0.0) & mask_m[:, None] & mask_n[None, :]

        # Store output
        out_idx = offs_m[:, None] * n_atoms + offs_n[None, :]
        valid_store_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(dist_out_ptr + out_idx, tl.sqrt(dist_sq), mask=valid_store_mask)
        tl.store(mask_out_ptr + out_idx, valid.to(tl.int8), mask=valid_store_mask)


def _build_neighbor_list_triton(
    coordinates: torch.Tensor,
    cutoff_radius: float,
) -> NeighborListResult:
    """Execute Triton block-tiled GPU kernel for neighbor list search. [D]"""
    n_atoms = coordinates.shape[0]
    device = coordinates.device
    dtype = coordinates.dtype

    if n_atoms < 2:
        return NeighborListResult(
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=device),
            edge_vector=torch.empty((0, 3), dtype=dtype, device=device),
            edge_distance=torch.empty((0,), dtype=dtype, device=device),
        )

    coords_c = coordinates.contiguous()
    dist_matrix = torch.zeros((n_atoms, n_atoms), dtype=dtype, device=device)
    mask_matrix = torch.zeros((n_atoms, n_atoms), dtype=torch.int8, device=device)

    BLOCK_SIZE = 32
    grid = (
        triton.cdiv(n_atoms, BLOCK_SIZE),
        triton.cdiv(n_atoms, BLOCK_SIZE),
    )

    _triton_neighbor_list_kernel[grid](
        coords_c,
        dist_matrix,
        mask_matrix,
        n_atoms,
        float(cutoff_radius * cutoff_radius),
        coords_c.stride(0),
        coords_c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    valid_mask = mask_matrix.to(torch.bool)
    edge_index = torch.nonzero(valid_mask, as_tuple=False).T  # [2, num_edges]

    # Reciprocal vectors: r_j - r_i
    i_idx = edge_index[0]
    j_idx = edge_index[1]
    edge_vector = coords_c[j_idx] - coords_c[i_idx]
    edge_distance = dist_matrix[i_idx, j_idx]

    return NeighborListResult(
        edge_index=edge_index.to(dtype=torch.int64, device=device),
        edge_vector=edge_vector.to(dtype=dtype, device=device),
        edge_distance=edge_distance.to(dtype=dtype, device=device),
    )


def _build_neighbor_list_scipy(
    coordinates: torch.Tensor,
    cutoff_radius: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
) -> NeighborListResult:
    """Portable CPU / MPS neighbor list generator using SciPy cKDTree and vectorized PyTorch. [D]"""
    device = coordinates.device
    dtype = coordinates.dtype
    n_atoms = coordinates.shape[0]

    if n_atoms < 2:
        return NeighborListResult(
            edge_index=torch.empty((2, 0), dtype=torch.int64, device=device),
            edge_vector=torch.empty((0, 3), dtype=dtype, device=device),
            edge_distance=torch.empty((0,), dtype=dtype, device=device),
        )

    try:
        # Transfer coordinates to CPU for cKDTree spatial query
        coords_np = coordinates.detach().cpu().numpy().astype(np.float64)

        boxsize = None
        if cell is not None and pbc is not None and torch.all(pbc):
            # If orthogonal unit cell
            diag = torch.diagonal(cell)
            boxsize = diag.detach().cpu().numpy().astype(np.float64)

        tree = cKDTree(coords_np, boxsize=boxsize)
        # Query pairwise neighbors within cutoff radius
        pair_set = tree.query_pairs(r=float(cutoff_radius), output_type="ndarray")

        if len(pair_set) == 0:
            return NeighborListResult(
                edge_index=torch.empty((2, 0), dtype=torch.int64, device=device),
                edge_vector=torch.empty((0, 3), dtype=dtype, device=device),
                edge_distance=torch.empty((0,), dtype=dtype, device=device),
            )

        pairs_i = pair_set[:, 0]
        pairs_j = pair_set[:, 1]

        # Enforce exact reciprocity: include both (i, j) and (j, i)
        src = np.concatenate([pairs_i, pairs_j])
        dst = np.concatenate([pairs_j, pairs_i])

        edge_index_cpu = torch.from_numpy(np.stack([src, dst], axis=0)).to(dtype=torch.int64)

        # Compute displacements on target device and dtype
        coords_dev = coordinates.contiguous()
        i_idx = edge_index_cpu[0].to(device=device)
        j_idx = edge_index_cpu[1].to(device=device)

        edge_vector = coords_dev[j_idx] - coords_dev[i_idx]

        # Handle periodic wrapping if applicable
        if cell is not None and pbc is not None and torch.any(pbc):
            diag = torch.diagonal(cell)
            edge_vector = edge_vector - diag * torch.round(edge_vector / diag)

        edge_distance = torch.norm(edge_vector, dim=-1)

        # Filter strictly positive distance
        valid = edge_distance > 0.0
        edge_index = torch.stack([i_idx[valid], j_idx[valid]], dim=0)
        edge_vector = edge_vector[valid]
        edge_distance = edge_distance[valid]

        return NeighborListResult(
            edge_index=edge_index.to(dtype=torch.int64, device=device),
            edge_vector=edge_vector.to(dtype=dtype, device=device),
            edge_distance=edge_distance.to(dtype=dtype, device=device),
        )
    except Exception as exc:
        raise HardwareDispatchError(
            f"Hardware dispatch neighbor-list calculation failed: {exc}",
            diagnostics={"n_atoms": n_atoms, "device": str(device)},
        ) from exc


def build_neighbor_list(
    coordinates: torch.Tensor,
    cutoff_radius: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
) -> NeighborListResult:
    r"""Build spatial neighbor list with dynamic hardware dispatch. [M]

    Dispatches to Triton GPU block-tiled kernel on CUDA accelerators when Triton is available;
    otherwise seamlessly routes to optimized SciPy cKDTree / PyTorch vectorized routines on CPU,
    Windows NT, and Apple Silicon MPS.

    Parameters
    ----------
    coordinates : torch.Tensor
        Cartesian coordinates [N, 3] in Angstroms.
    cutoff_radius : float
        Radial cutoff radius r_cut in Angstroms.
    cell : Optional[torch.Tensor]
        Unit cell vectors for periodic boundary conditions.
    pbc : Optional[torch.Tensor]
        Periodic boundary flags [pbc_x, pbc_y, pbc_z].

    Returns
    -------
    NeighborListResult
        NamedTuple containing edge_index [2, M], edge_vector [M, 3], and edge_distance [M].
    """
    if coordinates.is_cuda and TRITON_AVAILABLE and cell is None:
        try:
            return _build_neighbor_list_triton(coordinates, cutoff_radius)
        except Exception:
            # Fall back to robust CPU/SciPy routine if Triton encounters hardware or kernel errors
            pass

    return _build_neighbor_list_scipy(coordinates, cutoff_radius, cell=cell, pbc=pbc)
