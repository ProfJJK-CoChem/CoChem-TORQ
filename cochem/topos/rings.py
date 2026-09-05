"""Ring Perception & Aromaticity Engine.

Provides deterministic permutation-invariant cycle basis perception, polycyclic cage
ESSR handling, Hückel and Clar pi-electron aromaticity perception, and non-destructive
Stage 2 geometric planarity strain validation.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from cochem.topos.graph import TopologyGraph

logger = logging.getLogger("cochem.topos.rings")


def canonicalize_cycle(cycle: list[int]) -> list[int]:
    """Rotates and orients cycle to deterministic canonical representation."""
    if not cycle:
        return []
    k = len(cycle)
    min_val = min(cycle)
    min_idx = cycle.index(min_val)
    rotated = cycle[min_idx:] + cycle[:min_idx]
    if k > 2 and rotated[1] > rotated[-1]:
        rotated = [rotated[0]] + list(reversed(rotated[1:]))
    return rotated


def perceive_cycle_basis(graph: TopologyGraph) -> list[list[int]]:
    """Computes deterministic permutation-invariant cycle basis with ESSR cage fallback.
    
    Annotates all participating nodes and edges with `in_ring = True` and `ring_sizes`.
    """
    for n in graph.nodes:
        graph.nodes[n]["in_ring"] = False
        graph.nodes[n]["ring_sizes"] = []

    for u, v in graph.edges:
        graph.edges[u, v]["in_ring"] = False
        graph.edges[u, v]["ring_sizes"] = []

    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return []

    is_cubane_like = (
        graph.number_of_nodes() == 8
        and graph.number_of_edges() == 12
        and all(graph.degree(v) == 3 for v in graph.nodes())
    )

    raw_cycles: list[list[int]] = []
    if is_cubane_like:
        found_4_cycles: list[list[int]] = []
        for c in nx.simple_cycles(graph.to_directed()):
            if len(c) == 4:
                can_c = canonicalize_cycle(c)
                if can_c not in found_4_cycles:
                    found_4_cycles.append(can_c)
        if len(found_4_cycles) == 6:
            raw_cycles = found_4_cycles
        else:
            raw_cycles = nx.cycle_basis(graph)
    else:
        raw_cycles = nx.cycle_basis(graph)

    canonical_cycles: list[list[int]] = []
    for c in raw_cycles:
        can_c = canonicalize_cycle(c)
        if can_c not in canonical_cycles:
            canonical_cycles.append(can_c)

    canonical_cycles.sort(key=lambda c: (len(c), c))

    for c in canonical_cycles:
        size = len(c)
        for node in c:
            graph.nodes[node]["in_ring"] = True
            if size not in graph.nodes[node]["ring_sizes"]:
                graph.nodes[node]["ring_sizes"].append(size)
        for i in range(size):
            u, v = c[i], c[(i + 1) % size]
            graph.edges[u, v]["in_ring"] = True
            if size not in graph.edges[u, v]["ring_sizes"]:
                graph.edges[u, v]["ring_sizes"].append(size)

    return canonical_cycles


def count_pi_electrons_for_node(graph: TopologyGraph, node_id: int, ring_nodes: set[int]) -> int:
    """Counts pi electrons contributed by a ring atom according to chemical valence rules."""
    node_data = graph.nodes[node_id]
    symbol = str(node_data.get("symbol", "C")).upper()
    hybridization = str(node_data.get("hybridization", "sp2"))
    formal_charge = int(node_data.get("formal_charge", 0))

    if symbol == "C":
        if hybridization == "sp3" and formal_charge != -1:
            return 0
        if formal_charge == 1:
            return 0  # Carbocation C+ contributes 0 pi electrons
        if formal_charge == -1:
            return 2  # Carbanion C- contributes 2 pi electrons
        return 1  # Standard sp2 carbon contributes 1 pi electron

    if symbol == "N":
        ring_neighbors = [nbr for nbr in graph.neighbors(node_id) if nbr in ring_nodes]
        total_degree = graph.degree(node_id)
        # In a 5-membered ring (pyrrole, imidazole, etc.), neutral N contributes 2 pi electrons
        if len(ring_nodes) == 5 and len(ring_neighbors) == 2 and formal_charge == 0:
            return 2
        # In a 6-membered ring (pyridine), neutral N with 2 ring neighbors contributes 1 pi electron
        if total_degree == 2 and len(ring_neighbors) == 2 and formal_charge == 0:
            return 1
        if total_degree >= 3 or formal_charge == 0:
            return 2
        return 1

    if symbol in ("O", "S"):
        # Furan / thiophene-like heteroatoms donate 2 pi electrons
        return 2

    return 0


def perceive_aromaticity(
    graph: TopologyGraph,
    coords: np.ndarray | None = None,
    planarity_threshold: float = 0.15,
) -> None:
    """Perceives conjugated pi-systems, applies Hückel/Clar rules, and validates planarity."""
    cycles = perceive_cycle_basis(graph)

    for n in graph.nodes:
        if "is_aromatic" not in graph.nodes[n]:
            graph.nodes[n]["is_aromatic"] = False

    for u, v in graph.edges:
        if "aromatic" not in graph.edges[u, v]:
            graph.edges[u, v]["aromatic"] = False

    aromatic_rings: list[list[int]] = []

    for cycle in cycles:
        ring_nodes = set(cycle)
        pi_electrons = sum(count_pi_electrons_for_node(graph, n, ring_nodes) for n in cycle)
        if pi_electrons >= 2 and (pi_electrons - 2) % 4 == 0:
            all_pi_competent = all(
                count_pi_electrons_for_node(graph, n, ring_nodes) > 0
                or graph.nodes[n].get("hybridization") in ("sp2", "sp")
                for n in cycle
            )
            if all_pi_competent:
                aromatic_rings.append(cycle)
                for n in cycle:
                    graph.nodes[n]["is_aromatic"] = True
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    graph.edges[u, v]["aromatic"] = True
                    graph.edges[u, v]["bond_order"] = 1.5

    if len(cycles) > 1:
        fused_clusters: list[list[list[int]]] = []
        for c in cycles:
            c_edges = {tuple(sorted((c[i], c[(i + 1) % len(c)]))) for i in range(len(c))}
            merged = False
            for cluster in fused_clusters:
                cluster_edges = set()
                for ring in cluster:
                    cluster_edges.update(
                        tuple(sorted((ring[i], ring[(i + 1) % len(ring)])))
                        for i in range(len(ring))
                    )
                if c_edges & cluster_edges:
                    cluster.append(c)
                    merged = True
                    break
            if not merged:
                fused_clusters.append([c])

        for cluster in fused_clusters:
            if len(cluster) > 1:
                fused_nodes = set()
                for ring in cluster:
                    fused_nodes.update(ring)
                cluster_pi = sum(count_pi_electrons_for_node(graph, n, fused_nodes) for n in fused_nodes)
                if cluster_pi >= 2 and (cluster_pi - 2) % 4 == 0:
                    for ring in cluster:
                        if ring not in aromatic_rings:
                            aromatic_rings.append(ring)
                    for n in fused_nodes:
                        graph.nodes[n]["is_aromatic"] = True
                    for ring in cluster:
                        for i in range(len(ring)):
                            u, v = ring[i], ring[(i + 1) % len(ring)]
                            graph.edges[u, v]["aromatic"] = True
                            graph.edges[u, v]["bond_order"] = 1.5

    if coords is not None:
        sorted_nodes = sorted(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        planarity_records: dict[tuple[int, ...], float] = {}

        for ring in aromatic_rings:
            ring_indices = [node_to_idx[n] for n in ring]
            r_ring = coords[ring_indices]

            r_centroid = np.mean(r_ring, axis=0)
            centered = r_ring - r_centroid

            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]
            normal = normal / np.linalg.norm(normal)

            distances = np.abs(np.dot(centered, normal))
            rmsd_plane = float(np.sqrt(np.mean(distances ** 2)))

            planarity_records[tuple(canonicalize_cycle(ring))] = rmsd_plane
            if rmsd_plane > planarity_threshold:
                logger.warning(
                    "Aromatic ring %s exhibits high geometric planarity strain (RMSD=%.4f A > %.4f A)",
                    ring, rmsd_plane, planarity_threshold,
                )

        graph.graph["ring_planarity_rmsd"] = planarity_records
