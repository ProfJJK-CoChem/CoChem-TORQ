"""C^2-Smooth Graph Pruning & Reciprocal Sparse Topology (REQ-TORQ-TRAIN-100 [D]).

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic physics, continuous forces, and Newton's Third Law.
Features:
- Quintic polynomial C^2-smooth switching envelope preventing Dirac-delta force spikes.
- Strict undirected edge reciprocity (j in N(i) <=> i in N(j)) conserving Newton's Third Law.
- Pairwise force antisymmetry (F_ij = -F_ji) and net momentum conservation (sum F_i = 0).
- Covalent core degree invariant (degree >= 1 within r_cov = 1.5 Angstroms).
- Ban on stochastic edge dropout on spatial coordinate graphs.
- Dynamic atomic masses from mendeleev for center-of-mass momentum calculations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from Libraries.cochem_torq_masses import get_monoisotopic_masses_tensor
from Libraries.cochem_torq_training_errors import (
    DiscontinuousForceError,
    NonReciprocalGraphError,
)
from Libraries.cochem_torq_training_schemas import C2GraphPrunerConfig


def quintic_c2_switching(
    r: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Evaluate quintic polynomial C^2 cutoff switching envelope f_cut(r). [D]

    f_cut(r) = 1 - 10*(r/r_c)^3 + 15*(r/r_c)^4 - 6*(r/r_c)^5  if r <= r_c
             = 0                                              if r > r_c
    """
    if cutoff <= 0.0:
        raise ValueError(f"Cutoff radius must be strictly positive, got {cutoff}.")

    u = torch.clamp(r / cutoff, min=0.0, max=1.0)
    envelope = 1.0 - 10.0 * (u**3) + 15.0 * (u**4) - 6.0 * (u**5)
    return torch.where(r <= cutoff, envelope, torch.zeros_like(r))


def quintic_c2_derivative(
    r: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Analytical first spatial derivative df_cut/dr. [D]

    df/dr = (-30*(r/r_c)^2 + 60*(r/r_c)^3 - 30*(r/r_c)^4) / r_c  if r <= r_c
          = 0                                                    if r > r_c
    """
    u = torch.clamp(r / cutoff, min=0.0, max=1.0)
    poly = -30.0 * (u**2) + 60.0 * (u**3) - 30.0 * (u**4)
    d_env = poly / cutoff
    return torch.where(r <= cutoff, d_env, torch.zeros_like(r))


def quintic_c2_second_derivative(
    r: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Analytical second spatial derivative d^2 f_cut / dr^2. [D]

    d^2 f/dr^2 = (-60*(r/r_c) + 180*(r/r_c)^2 - 120*(r/r_c)^3) / r_c^2  if r <= r_c
               = 0                                                      if r > r_c
    """
    u = torch.clamp(r / cutoff, min=0.0, max=1.0)
    poly = -60.0 * u + 180.0 * (u**2) - 120.0 * (u**3)
    d2_env = poly / (cutoff**2)
    return torch.where(r <= cutoff, d2_env, torch.zeros_like(r))


def verify_c2_continuity_boundary(cutoff: float = 5.0) -> Dict[str, float]:
    """Verify that quintic switching envelope satisfies f(r_c)=0, f'(r_c)=0, f''(r_c)=0. [D]"""
    r_boundary = torch.tensor([cutoff], dtype=torch.float64)
    val = float(quintic_c2_switching(r_boundary, cutoff).item())
    d1 = float(quintic_c2_derivative(r_boundary, cutoff).item())
    d2 = float(quintic_c2_second_derivative(r_boundary, cutoff).item())

    # Check tolerances
    if abs(val) > 1e-7 or abs(d1) > 1e-7 or abs(d2) > 1e-7:
        raise DiscontinuousForceError(
            f"Switching envelope violates C^2 continuity at r_c={cutoff}: "
            f"val={val}, d1={d1}, d2={d2}",
            diagnostics={"val": val, "d1": d1, "d2": d2, "cutoff": cutoff},
        )

    return {"f_rc": val, "df_rc": d1, "d2f_rc": d2}


def check_reciprocal_topology(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> bool:
    """Check whether edge_index satisfies undirected reciprocity: j in N(i) <=> i in N(j). [D]"""
    if edge_index.numel() == 0:
        return True

    src = edge_index[0]
    dst = edge_index[1]

    # Map directed edges (u, v) into set of tuples
    edge_set = set(zip(src.tolist(), dst.tolist()))

    for u, v in edge_set:
        if (v, u) not in edge_set:
            return False

    return True


def enforce_graph_reciprocity(
    edge_index: torch.Tensor,
    raise_on_asymmetry: bool = True,
) -> torch.Tensor:
    """Enforce edge reciprocity via symmetric intersection: A_pruned = A & A^T. [D]"""
    if edge_index.numel() == 0:
        return edge_index

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    edge_set = set(zip(src, dst))

    symmetric_edges: List[Tuple[int, int]] = []
    has_asymmetry = False

    for u, v in edge_set:
        if (v, u) in edge_set:
            symmetric_edges.append((u, v))
        else:
            has_asymmetry = True

    if has_asymmetry and raise_on_asymmetry:
        raise NonReciprocalGraphError(
            "Asymmetric edges detected in graph topology, breaking Newton's Third Law.",
            diagnostics={
                "num_edges": len(edge_set),
                "num_symmetric": len(symmetric_edges),
            },
        )

    if not symmetric_edges:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    new_src = [e[0] for e in symmetric_edges]
    new_dst = [e[1] for e in symmetric_edges]
    return torch.tensor([new_src, new_dst], dtype=edge_index.dtype, device=edge_index.device)


def build_c2_reciprocal_graph(
    coordinates: torch.Tensor,
    species: Sequence[int] | torch.Tensor,
    config: Optional[C2GraphPrunerConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct C^2 continuous reciprocal molecular graph within cutoff_radius. [D]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (edge_index (2, E), edge_weights (E,))
    """
    cfg = config if config is not None else C2GraphPrunerConfig()

    # Banned stochastic dropout on spatial coordinate graphs
    if cfg.stochastic_edge_dropout > 0.0:
        raise DiscontinuousForceError(
            "Stochastic edge dropout is strictly banned on spatial coordinate graphs [D]."
        )

    n_atoms = coordinates.shape[0]
    if n_atoms <= 1:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=coordinates.device)
        empty_weights = torch.empty((0,), dtype=coordinates.dtype, device=coordinates.device)
        return empty_edges, empty_weights

    # Compute pairwise Euclidean distance matrix
    diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, p=2, dim=-1)  # (N, N)

    # Exclude self-loops
    mask_no_self = ~torch.eye(n_atoms, dtype=torch.bool, device=coordinates.device)

    # Core covalent mask (guarantee degree >= 1 for atoms with core contacts)
    core_mask = (dist <= cfg.covalent_core_radius_angstrom) & mask_no_self
    cutoff_mask = (dist <= cfg.cutoff_radius_angstrom) & mask_no_self

    # Combined candidate edges
    candidate_mask = cutoff_mask | core_mask

    # Symmetrical pruning: A_pruned = A & A^T
    reciprocal_mask = candidate_mask & candidate_mask.t()

    # Verify covalent core degree invariant
    for i in range(n_atoms):
        has_core_neighbor = core_mask[i].any().item()
        if has_core_neighbor:
            degree_in_graph = reciprocal_mask[i].sum().item()
            if degree_in_graph < 1:
                # Force retain nearest neighbor within core
                min_core_idx = int(torch.argmin(torch.where(core_mask[i], dist[i], 1e9)).item())
                reciprocal_mask[i, min_core_idx] = True
                reciprocal_mask[min_core_idx, i] = True

    src_idx, dst_idx = torch.where(reciprocal_mask)
    edge_index = torch.stack([src_idx, dst_idx], dim=0)

    # Verify reciprocity
    if cfg.enforce_reciprocal_edges and not check_reciprocal_topology(edge_index):
        raise NonReciprocalGraphError(
            "Generated topology failed undirected reciprocity invariant [D]."
        )

    # Compute C^2 switching envelope weights
    edge_distances = dist[src_idx, dst_idx]
    edge_weights = quintic_c2_switching(edge_distances, cutoff=cfg.cutoff_radius_angstrom)

    return edge_index, edge_weights


def compute_pairwise_conservative_forces(
    coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    cutoff: float = 5.0,
    k_spring: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise interatomic forces modulated by C^2 switching envelope. [D]

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (pairwise_forces (E, 3), atomic_forces (N, 3))
    """
    n_atoms = coordinates.shape[0]
    src = edge_index[0]
    dst = edge_index[1]

    # r_ij vector pointing from j to i (displacement)
    r_ij = coordinates[src] - coordinates[dst]  # (E, 3)
    dist = torch.norm(r_ij + 1e-12, p=2, dim=-1, keepdim=True)  # (E, 1)
    unit_r = r_ij / dist

    # Model potential V(r) = 0.5 * k * (r - r_0)^2 * f_cut(r)
    # Pairwise scalar force: F_ij = -dV/dr * unit_r
    f_env = quintic_c2_switching(dist.squeeze(-1), cutoff=cutoff).unsqueeze(-1)
    df_env = quintic_c2_derivative(dist.squeeze(-1), cutoff=cutoff).unsqueeze(-1)

    r_0 = 1.5
    v_base = 0.5 * k_spring * torch.square(dist - r_0)
    dv_base = k_spring * (dist - r_0)

    # Product rule: d(V * f_cut) / dr = dv * f + v * df
    scalar_force = -(dv_base * f_env + v_base * df_env)
    pairwise_f = scalar_force * unit_r  # (E, 3)

    # Accumulate atomic forces: F_i = sum_{j in N(i)} F_ij
    atomic_f = torch.zeros_like(coordinates)
    atomic_f.index_add_(0, src, pairwise_f)

    return pairwise_f, atomic_f


def evaluate_momentum_and_antisymmetry(
    atomic_forces: torch.Tensor,
    edge_index: torch.Tensor,
    pairwise_forces: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate Newton's Third Law parity error and total external force drift. [D]"""
    # 1. Net external force drift: ||sum_i F_i||_2
    net_force = torch.sum(atomic_forces, dim=0)
    net_force_norm = float(torch.norm(net_force, p=2).item())

    # 2. Pairwise antisymmetry: ||F_ij + F_ji||_inf
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    edge_to_idx = { (u, v): idx for idx, (u, v) in enumerate(zip(src, dst)) }

    max_asym = 0.0
    for (u, v), idx_uv in edge_to_idx.items():
        if (v, u) in edge_to_idx:
            idx_vu = edge_to_idx[(v, u)]
            pair_sum = pairwise_forces[idx_uv] + pairwise_forces[idx_vu]
            asym = float(torch.max(torch.abs(pair_sum)).item())
            if asym > max_asym:
                max_asym = asym

    return {
        "net_force_drift": net_force_norm,
        "pairwise_antisymmetry_error": max_asym,
    }


def compute_center_of_mass(
    coordinates: torch.Tensor,
    species: Sequence[int],
) -> Tuple[torch.Tensor, float]:
    """Compute molecular center-of-mass dynamically using mendeleev monoisotopic masses. [M]"""
    masses = get_monoisotopic_masses_tensor(
        species, dtype=coordinates.dtype, device=coordinates.device
    )
    total_mass = float(torch.sum(masses).item())
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive.")

    com = torch.sum(coordinates * masses.unsqueeze(-1), dim=0) / total_mass
    return com, total_mass
