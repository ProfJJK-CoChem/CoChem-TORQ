"""Macromolecular Coarse-Graining Subsystem (Graph Crusher).

Implements discrete graph quotient partitioning with strict mass and formal charge
conservation, boundary edge contraction, and isolated Stage 2 center-of-mass evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from cochem.topos.exceptions import TopologyError
from cochem.topos.graph import TopologyGraph

logger = logging.getLogger("cochem.topos.coarse_grain")


@dataclass(frozen=True)
class GraphCrusherConfig:
    """Configuration parameters for macromolecular coarse-graining."""

    partition_strategy: str = "residue"
    preserve_boundary_bonds: bool = True
    mass_tolerance: float = 1e-5
    charge_tolerance: float = 1e-5


def crush_macromolecule(
    graph: TopologyGraph,
    partition_map: dict[int, int],
    coords: np.ndarray | None = None,
    config: GraphCrusherConfig | None = None,
) -> tuple[TopologyGraph, np.ndarray | None]:
    """Coarse-grains an all-atom molecular topology into quotient beads with physical conservation.
    
    Stage 1:
      - Validates topological contiguity for each bead quotient partition.
      - Enforces mass and charge conservation across the transformation.
      - Contracts boundary edges and aggregates bond orders.
      
    Stage 2:
      - Calculates center-of-mass coordinates for each bead when Cartesian coords are provided.
      - Maintains strict isolation of spatial metrics from topological node attributes.
    """
    if config is None:
        config = GraphCrusherConfig()

    # 1. Validation of partition completeness
    missing_nodes = set(graph.nodes()) - set(partition_map.keys())
    if missing_nodes:
        raise TopologyError(f"Missing partition mapping for nodes: {sorted(missing_nodes)}")

    # 2. Group nodes by bead
    bead_to_nodes: dict[int, list[int]] = {}
    for node_id, bead_id in partition_map.items():
        bead_to_nodes.setdefault(bead_id, []).append(node_id)

    # 3. Stage 1: Topological contiguity verification
    for bead_id, bead_nodes in bead_to_nodes.items():
        induced_subgraph = graph.subgraph(bead_nodes)
        if not nx.is_connected(induced_subgraph):
            raise TopologyError(
                f"Partition block {bead_id} is not topologically contiguous. "
                f"Components: {list(nx.connected_components(induced_subgraph))}"
            )

    # 4. Net mass and charge calculation & conservation
    orig_total_mass: float = sum(float(graph.nodes[n]["mass"]) for n in graph.nodes())
    orig_total_charge: float = sum(float(graph.nodes[n]["formal_charge"]) for n in graph.nodes())

    bead_masses: dict[int, float] = {}
    bead_charges: dict[int, float] = {}

    for bead_id, bead_nodes in bead_to_nodes.items():
        b_mass = sum(float(graph.nodes[n]["mass"]) for n in bead_nodes)
        b_charge = sum(float(graph.nodes[n]["formal_charge"]) for n in bead_nodes)
        bead_masses[bead_id] = b_mass
        bead_charges[bead_id] = b_charge

    cg_total_mass = sum(bead_masses.values())
    cg_total_charge = sum(bead_charges.values())

    if abs(cg_total_mass - orig_total_mass) > config.mass_tolerance:
        raise TopologyError(
            f"Mass conservation failure: original={orig_total_mass:.6f}, CG={cg_total_mass:.6f}, "
            f"delta={abs(cg_total_mass - orig_total_mass):.2e} > tol={config.mass_tolerance:.2e}"
        )

    if abs(cg_total_charge - orig_total_charge) > config.charge_tolerance:
        raise TopologyError(
            f"Charge conservation failure: original={orig_total_charge:.6f}, CG={cg_total_charge:.6f}, "
            f"delta={abs(cg_total_charge - orig_total_charge):.2e} > tol={config.charge_tolerance:.2e}"
        )

    # 5. Construct quotient TopologyGraph
    cg_graph = TopologyGraph()
    sorted_bead_ids = sorted(bead_to_nodes.keys())

    for bead_id in sorted_bead_ids:
        b_nodes = bead_to_nodes[bead_id]
        all_in_ring = all(bool(graph.nodes[n].get("in_ring", False)) for n in b_nodes)
        cg_graph.add_chemical_node(
            node_id=bead_id,
            symbol=f"BEAD_{bead_id}",
            formal_charge=int(round(bead_charges[bead_id])),
            hybridization="coarse_grained",
            in_ring=all_in_ring,
            mass=bead_masses[bead_id],
            atomic_number=0,
            bead_atoms=b_nodes,
        )

    # 6. Boundary Edge Contraction
    aggregated_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        b_u = partition_map[u]
        b_v = partition_map[v]
        if b_u != b_v:
            pair = (min(b_u, b_v), max(b_u, b_v))
            if pair not in aggregated_edges:
                aggregated_edges[pair] = {
                    "bond_order": 0.0,
                    "aromatic": False,
                    "in_ring": False,
                    "stereo": None,
                }
            aggregated_edges[pair]["bond_order"] += float(data.get("bond_order", 1.0))
            aggregated_edges[pair]["aromatic"] = aggregated_edges[pair]["aromatic"] or bool(data.get("aromatic", False))
            aggregated_edges[pair]["in_ring"] = aggregated_edges[pair]["in_ring"] or bool(data.get("in_ring", False))

    for (b1, b2), edge_attrs in aggregated_edges.items():
        cg_graph.add_chemical_edge(
            b1,
            b2,
            bond_order=edge_attrs["bond_order"],
            aromatic=edge_attrs["aromatic"],
            in_ring=edge_attrs["in_ring"],
            stereo=edge_attrs["stereo"],
        )

    # 7. Stage 2: Spatial bead center-of-mass evaluation
    cg_coords: np.ndarray | None = None
    if coords is not None:
        if coords.shape[0] != graph.number_of_nodes() or coords.shape[1] != 3:
            raise TopologyError(
                f"Coordinates shape {coords.shape} incompatible with graph node count {graph.number_of_nodes()}"
            )

        sorted_orig_nodes = sorted(graph.nodes())
        orig_node_to_idx = {n: idx for idx, n in enumerate(sorted_orig_nodes)}

        cg_coords_list: list[np.ndarray] = []
        for bead_id in sorted_bead_ids:
            b_nodes = bead_to_nodes[bead_id]
            b_indices = [orig_node_to_idx[n] for n in b_nodes]
            masses = np.array([float(graph.nodes[n]["mass"]) for n in b_nodes], dtype=float)
            total_b_mass = np.sum(masses)
            if total_b_mass <= 0:
                raise TopologyError(f"Total mass of bead {bead_id} must be strictly positive.")

            r_nodes = coords[b_indices]
            r_com = np.sum(masses[:, np.newaxis] * r_nodes, axis=0) / total_b_mass
            cg_coords_list.append(r_com)

        cg_coords = np.array(cg_coords_list, dtype=float)

    return cg_graph, cg_coords
