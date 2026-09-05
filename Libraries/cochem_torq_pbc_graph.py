"""Periodic boundary condition (PBC) minimum image radial graph engine and virial stress tensor.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic lattice geometry, self-image exclusion, and analytical virial stress.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from Libraries.cochem_torq_inference_errors import PBCGraphError
from Libraries.cochem_torq_inference_schemas import PBCRadialGraphConfig


@dataclass(frozen=True)
class PBCGraph:
    """Immutable container for periodic molecular graph representations. [M]"""

    edge_index: torch.Tensor       # Shape (2, E)
    edge_shifts: torch.Tensor      # Shape (E, 3) integer unit cell shifts
    edge_vectors: torch.Tensor     # Shape (E, 3) Cartesian displacement vectors r_{ij, S}
    edge_distances: torch.Tensor   # Shape (E,) interatomic distances
    cell_volume: float             # Unit cell volume Omega = |det(L)|
    fractional_coords: torch.Tensor # Shape (N, 3) coordinates in [0, 1)^3


def compute_cell_volume(lattice: Union[torch.Tensor, np.ndarray]) -> float:
    """Compute crystal unit cell volume Omega = |det(L)|. [D]"""
    if isinstance(lattice, np.ndarray):
        lat_t = torch.from_numpy(lattice).double()
    else:
        lat_t = lattice.double()
    
    if lat_t.shape != (3, 3):
        raise PBCGraphError(
            f"Lattice matrix must have shape (3, 3), got {lat_t.shape}.",
            diagnostics={"lattice_shape": list(lat_t.shape)},
        )
    vol = float(torch.abs(torch.linalg.det(lat_t)).item())
    if vol <= 1e-12:
        raise PBCGraphError(
            f"Singular or degenerate lattice matrix with volume Omega={vol:.3e}.",
            diagnostics={"cell_volume": vol},
        )
    return vol


def cartesian_to_fractional(
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Transform Cartesian coordinates into fractional coordinates s = L^{-1} r (mod 1). [D]
    
    Assumes lattice vectors a1, a2, a3 are either the rows or columns of L.
    By standard convention L = [a1, a2, a3]^T (rows) so r = s @ L, s = r @ L^{-1}.
    """
    lat = lattice.to(dtype=coordinates.dtype, device=coordinates.device)
    # Solve s @ lat = coordinates
    # s = coordinates @ lat^{-1}
    inv_lat = torch.linalg.inv(lat)
    # Check if r = s @ lat or lat @ s
    # Standard crystallographic: r_cart = s_frac @ lat
    frac = torch.matmul(coordinates, inv_lat)
    return frac % 1.0


def fractional_to_cartesian(
    fractional: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Transform fractional coordinates to Cartesian coordinates r = s @ L. [D]"""
    lat = lattice.to(dtype=fractional.dtype, device=fractional.device)
    return torch.matmul(fractional, lat)


def compute_interplanar_spacings(
    lattice: torch.Tensor,
    rc: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute perpendicular interplanar spacings d_perp and cell search ranges n_max. [D]
    
    Using cyclic permutations (k, i, j) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
    d_{perp, k} = Omega / ||a_i x a_j||_2
    n_k in [-ceil(rc / d_{perp, k}), ceil(rc / d_{perp, k})]
    """
    vol = compute_cell_volume(lattice)
    lat = lattice.double()

    # Lattice vectors as rows a0, a1, a2
    a = [lat[0], lat[1], lat[2]]
    spacings = []
    n_max_list = []

    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for k, i, j in cyclic:
        cross_prod = torch.linalg.cross(a[i], a[j])
        cross_norm = float(torch.norm(cross_prod).item())
        if cross_norm <= 1e-12:
            raise PBCGraphError(
                f"Degenerate collinear lattice vectors for direction k={k}.",
                diagnostics={"k": k, "cross_norm": cross_norm},
            )
        d_perp = vol / cross_norm
        spacings.append(d_perp)
        n_max = int(math.ceil(rc / d_perp))
        n_max_list.append(n_max)

    d_perp_tensor = torch.tensor(spacings, dtype=lattice.dtype, device=lattice.device)
    n_max_tensor = torch.tensor(n_max_list, dtype=torch.int64, device=lattice.device)
    return d_perp_tensor, n_max_tensor


def build_pbc_radial_graph(
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    rc: float = 5.0,
) -> PBCGraph:
    """Construct authentic minimum-image periodic graph with strict self-image exclusion. [M]/[D]"""
    if rc <= 0.0:
        raise PBCGraphError(
            f"Cutoff radius rc={rc} must be strictly positive.",
            diagnostics={"rc": rc},
        )
    num_atoms = coordinates.shape[0]
    if num_atoms == 0:
        raise PBCGraphError("Cannot construct PBC graph with zero atoms.")

    vol = compute_cell_volume(lattice)
    lat = lattice.to(dtype=coordinates.dtype, device=coordinates.device)

    # Fractional coordinates
    frac_coords = cartesian_to_fractional(coordinates, lat)
    # Wrapped cartesian coordinates in unit cell
    wrapped_coords = fractional_to_cartesian(frac_coords, lat)

    # Determine integer search range for each lattice vector
    _, n_max = compute_interplanar_spacings(lat, rc)
    nx, ny, nz = int(n_max[0].item()), int(n_max[1].item()), int(n_max[2].item())

    # Build shift grid
    shift_ranges = [
        torch.arange(-nx, nx + 1, device=coordinates.device, dtype=torch.int64),
        torch.arange(-ny, ny + 1, device=coordinates.device, dtype=torch.int64),
        torch.arange(-nz, nz + 1, device=coordinates.device, dtype=torch.int64),
    ]
    grid_shifts = torch.cartesian_prod(*shift_ranges)  # (N_shifts, 3)

    # Cartesian shifts: S @ lat
    cart_shifts = torch.matmul(grid_shifts.to(dtype=lat.dtype), lat)  # (N_shifts, 3)

    # Pairwise evaluation across atoms and shifts
    # r_j - r_i + S
    diff_cart = wrapped_coords.unsqueeze(0) - wrapped_coords.unsqueeze(1) # (N, N, 3) = r_j - r_i

    # Expand with shifts: (N, N, N_shifts, 3)
    # To keep memory bounded, iterate over shifts
    src_list: list[torch.Tensor] = []
    dst_list: list[torch.Tensor] = []
    shifts_list: list[torch.Tensor] = []
    vectors_list: list[torch.Tensor] = []
    distances_list: list[torch.Tensor] = []

    for s_idx in range(grid_shifts.shape[0]):
        shift_int = grid_shifts[s_idx]
        shift_vec = cart_shifts[s_idx]

        disp = diff_cart + shift_vec  # (N, N, 3): r_j - r_i + S
        dist = torch.norm(disp, dim=-1)  # (N, N)

        # Self-Image Exclusion Invariant:
        # Construct graph edge (i, j, S) iff dist <= rc AND ((i != j) or (S != (0, 0, 0)))
        is_zero_shift = (shift_int[0] == 0) and (shift_int[1] == 0) and (shift_int[2] == 0)
        
        mask = dist <= rc
        if is_zero_shift:
            # Exclude self-loop i == j
            eye_mask = torch.eye(num_atoms, dtype=torch.bool, device=coordinates.device)
            mask = mask & (~eye_mask)

        # Nonzero entries
        i_indices, j_indices = torch.where(mask)
        if i_indices.numel() > 0:
            src_list.append(i_indices)
            dst_list.append(j_indices)
            shifts_list.append(shift_int.unsqueeze(0).expand(i_indices.numel(), -1))
            vectors_list.append(disp[i_indices, j_indices])
            distances_list.append(dist[i_indices, j_indices])

    if len(src_list) == 0:
        # Empty graph within cutoff
        edge_index = torch.empty((2, 0), dtype=torch.int64, device=coordinates.device)
        edge_shifts = torch.empty((0, 3), dtype=torch.int64, device=coordinates.device)
        edge_vectors = torch.empty((0, 3), dtype=coordinates.dtype, device=coordinates.device)
        edge_distances = torch.empty((0,), dtype=coordinates.dtype, device=coordinates.device)
    else:
        src = torch.cat(src_list, dim=0)
        dst = torch.cat(dst_list, dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        edge_shifts = torch.cat(shifts_list, dim=0)
        edge_vectors = torch.cat(vectors_list, dim=0)
        edge_distances = torch.cat(distances_list, dim=0)

    # Verify self-image exclusion: no edge with distance == 0 and same source/target
    zero_dist_mask = (edge_distances <= 1e-12) & (edge_index[0] == edge_index[1])
    if zero_dist_mask.any():
        raise PBCGraphError(
            "Self-image exclusion violated: detected zero-distance self-loop.",
            diagnostics={"num_violations": int(zero_dist_mask.sum().item())},
        )

    return PBCGraph(
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        edge_vectors=edge_vectors,
        edge_distances=edge_distances,
        cell_volume=vol,
        fractional_coords=frac_coords,
    )


def compute_virial_stress_tensor(
    edge_vectors: torch.Tensor,
    pair_forces_or_gradients: torch.Tensor,
    cell_volume: float,
    is_force: bool = False,
) -> torch.Tensor:
    """Compute analytical symmetric unit cell virial stress tensor Xi in R^{3x3}. [D]
    
    Xi_{alpha, beta} = - 1 / (2 * Omega) * sum_e (r_e)_{alpha} * (grad_{r_e} E)_{beta}
    If is_force=True, pair_forces_or_gradients = f_e = - grad_{r_e} E:
    Xi_{alpha, beta} = 1 / (2 * Omega) * sum_e (r_e)_{alpha} * (f_e)_{beta}
    """
    if cell_volume <= 1e-12:
        raise PBCGraphError(
            f"Invalid cell volume {cell_volume} for virial stress calculation.",
            diagnostics={"cell_volume": cell_volume},
        )
    if edge_vectors.shape[0] == 0:
        return torch.zeros((3, 3), dtype=edge_vectors.dtype, device=edge_vectors.device)

    # Outer product sum: sum_e r_e * g_e^T
    # edge_vectors: (E, 3), pair_forces_or_gradients: (E, 3)
    outer = torch.matmul(
        edge_vectors.unsqueeze(-1),
        pair_forces_or_gradients.unsqueeze(-2),
    ) # (E, 3, 3)
    virial_sum = torch.sum(outer, dim=0) # (3, 3)

    factor = (1.0 if is_force else -1.0) / (2.0 * cell_volume)
    xi = factor * virial_sum

    # Symmetrize stress tensor
    xi_sym = 0.5 * (xi + xi.t())
    
    # Check numerical symmetry
    asym_err = float(torch.max(torch.abs(xi - xi.t())).item())
    if asym_err > 1e-6:
        raise PBCGraphError(
            f"Virial stress tensor violates symmetry constraint: asymmetry={asym_err:.3e} exceeds 1e-6.",
            diagnostics={"asymmetry": asym_err},
        )

    return xi_sym


def compute_hydrostatic_pressure(virial_stress: torch.Tensor) -> float:
    """Compute scalar isotropic hydrostatic pressure P = -1/3 * Tr(Xi). [D]"""
    trace = float(torch.trace(virial_stress).item())
    return - (1.0 / 3.0) * trace


class PBCRadialGraphEngine(nn.Module):
    """PyTorch module for periodic graph construction and virial stress computation. [M]/[D]"""

    def __init__(self, config: Optional[PBCRadialGraphConfig] = None) -> None:
        super().__init__()
        self.config = config or PBCRadialGraphConfig()
        self.rc = float(self.config.cutoff_radius_rc)

    def forward(
        self, coordinates: torch.Tensor, lattice: torch.Tensor
    ) -> PBCGraph:
        """Construct PBC graph from coordinates and lattice matrix. [M]"""
        return build_pbc_radial_graph(coordinates, lattice, rc=self.rc)

    def compute_virial(
        self,
        edge_vectors: torch.Tensor,
        pair_forces: torch.Tensor,
        cell_volume: float,
        is_force: bool = True,
    ) -> torch.Tensor:
        """Compute symmetric virial stress tensor. [D]"""
        return compute_virial_stress_tensor(
            edge_vectors, pair_forces, cell_volume=cell_volume, is_force=is_force
        )

    def compute_pressure(self, virial_stress: torch.Tensor) -> float:
        """Compute scalar hydrostatic pressure from virial stress tensor. [D]"""
        return compute_hydrostatic_pressure(virial_stress)
