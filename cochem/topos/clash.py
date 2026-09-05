"""Graph-Based Geometric Clash Detector.

Implements dynamic Van der Waals queries with 5-tier transactinide fallback hierarchy,
topological exclusion masks (1-2, 1-3), threshold scaling (1-4, non-bonded, h-bond),
and spatial acceleration via scipy.spatial.cKDTree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from mendeleev import element
from scipy.spatial import cKDTree

from cochem.topos.exceptions import StericClashError
from cochem.topos.graph import TopologyGraph

logger = logging.getLogger("cochem.topos.clash")


@dataclass(frozen=True)
class ClashPair:
    """Records geometric steric clash between an atom pair."""

    atom_i: int
    atom_j: int
    distance: float
    clash_threshold: float
    pair_type: str  # "1-4", "non-bonded", "h-bond"


class GeometricClashDetector:
    """Detects geometric steric clashes adhering to dynamic Mendeleev lookups and topological exclusions."""

    def __init__(
        self,
        k_clash: float = 0.75,
        k_hbond: float = 0.55,
        k_14: float = 0.60,
    ) -> None:
        self.k_clash = k_clash
        self.k_hbond = k_hbond
        self.k_14 = k_14
        self._vdw_cache: dict[str, float] = {}

    def get_vdw_radius(self, symbol: str) -> float:
        """Dynamically resolves Van der Waals radius in Angstroms via 5-tier Mendeleev hierarchy."""
        sym_clean = symbol.strip()
        if sym_clean in self._vdw_cache:
            return self._vdw_cache[sym_clean]

        try:
            elem = element(sym_clean)
        except Exception as exc:
            raise StericClashError(f"Undefined Van der Waals and covalent radii for element '{symbol}': {exc}") from exc

        # 1. Primary Query: elem.vdw_radius
        if elem.vdw_radius is not None:
            r_vdw = float(elem.vdw_radius) / 100.0
            self._vdw_cache[sym_clean] = r_vdw
            return r_vdw

        # 2. Fallback 1: elem.vdw_radius_alvarez
        if elem.vdw_radius_alvarez is not None:
            r_vdw = float(elem.vdw_radius_alvarez) / 100.0
            self._vdw_cache[sym_clean] = r_vdw
            return r_vdw

        # 3. Fallback 2: elem.vdw_radius_bondi
        if elem.vdw_radius_bondi is not None:
            r_vdw = float(elem.vdw_radius_bondi) / 100.0
            self._vdw_cache[sym_clean] = r_vdw
            return r_vdw

        # 4. Fallback 3 (Superheavy Transactinides Z in [104, 118]): 1.60 * r_cov
        if elem.covalent_radius_pyykko is not None:
            r_vdw = 1.60 * (float(elem.covalent_radius_pyykko) / 100.0)
            self._vdw_cache[sym_clean] = r_vdw
            return r_vdw

        if elem.covalent_radius_cordero is not None:
            r_vdw = 1.60 * (float(elem.covalent_radius_cordero) / 100.0)
            self._vdw_cache[sym_clean] = r_vdw
            return r_vdw

        # 5. Exception Handling
        raise StericClashError(
            f"Undefined Van der Waals and covalent radii for element {symbol} (Z={elem.atomic_number})"
        )

    def detect_clashes(self, coords: np.ndarray, topology: TopologyGraph) -> list[ClashPair]:
        """Detects all steric clashes using topological masks and cKDTree spatial acceleration."""
        num_nodes = topology.number_of_nodes()
        if coords.shape[0] != num_nodes or coords.shape[1] != 3:
            raise StericClashError(
                f"Coordinates shape {coords.shape} incompatible with topology node count {num_nodes}"
            )

        sorted_nodes = sorted(topology.nodes())
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

        # Precompute elemental radii and atomic numbers
        radii = np.array([self.get_vdw_radius(str(topology.nodes[n]["symbol"])) for n in sorted_nodes], dtype=float)
        atomic_numbers = [int(topology.nodes[n].get("atomic_number", 0)) for n in sorted_nodes]

        # Topological shortest path lengths for exclusion masks
        spl = dict(nx.all_pairs_shortest_path_length(topology))

        # Spatial search radius
        max_vdw = float(np.max(radii)) if len(radii) > 0 else 2.0
        max_multiplier = max(self.k_clash, self.k_14, self.k_hbond)
        search_radius = 2.0 * max_vdw * max_multiplier

        tree = cKDTree(coords)
        candidate_pairs = tree.query_pairs(r=search_radius)

        clashes: list[ClashPair] = []

        for i, j in candidate_pairs:
            u, v = sorted_nodes[i], sorted_nodes[j]
            top_dist = spl.get(u, {}).get(v, 999)

            # Exclusion masks: 1-2 (bonded) and 1-3 (geminal)
            if top_dist <= 2:
                continue

            z_i, z_j = atomic_numbers[i], atomic_numbers[j]
            r_sum = radii[i] + radii[j]

            # Determine pair type and threshold
            if top_dist == 3:
                pair_type = "1-4"
                threshold = self.k_14 * r_sum
            else:
                # Check for polar hydrogen bond pair (H with O, N, or F)
                is_hbond = (z_i == 1 and z_j in (7, 8, 9)) or (z_j == 1 and z_i in (7, 8, 9))
                if is_hbond:
                    pair_type = "h-bond"
                    threshold = self.k_hbond * r_sum
                else:
                    pair_type = "non-bonded"
                    threshold = self.k_clash * r_sum

            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < threshold:
                clashes.append(
                    ClashPair(
                        atom_i=u,
                        atom_j=v,
                        distance=dist,
                        clash_threshold=threshold,
                        pair_type=pair_type,
                    )
                )

        # Sort clashes by distance ascending (most critical first)
        clashes.sort(key=lambda c: c.distance)
        return clashes
