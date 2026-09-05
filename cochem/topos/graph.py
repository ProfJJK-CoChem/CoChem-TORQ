"""Unified TopologyGraph Subsystem.

Provides pure topological molecular graph representation subclassing networkx.Graph,
VF2 subgraph isomorphism searching, QCSchema dictionary export, and thread-safe persistence.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
from filelock import FileLock
from mendeleev import element

from cochem.topos.exceptions import TopologyError

logger = logging.getLogger("cochem.topos.graph")

ANGSTROM_TO_BOHR: float = 1.8897261246257702
VALID_HYBRIDIZATIONS: frozenset[str] = frozenset({
    "sp", "sp2", "sp3", "sp3d", "sp3d2", "coarse_grained"
})
FORBIDDEN_COORDINATE_KEYS: frozenset[str] = frozenset({
    "coords", "coordinates", "x", "y", "z", "pos"
})


class TopologyGraph(nx.Graph):
    """Unified chemical topology graph subclassing networkx.Graph.
    
    Invariants:
    1. Topologically Pure Node Schema: Node attributes strictly exclude Cartesian coordinates.
    2. Dynamic Mendeleev Elemental Queries: Masses and atomic numbers queried dynamically.
    3. Strict Chemical Attribute Typing on nodes and edges.
    """

    def __init__(self, incoming_graph_data: Any = None, **attr: Any) -> None:
        super().__init__(incoming_graph_data, **attr)

    def add_chemical_node(
        self,
        node_id: int,
        symbol: str,
        formal_charge: int = 0,
        hybridization: str = "sp3",
        in_ring: bool = False,
        mass: float | None = None,
        atomic_number: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Validates chemical identity and dynamically sets mass via mendeleev."""
        for forbidden in FORBIDDEN_COORDINATE_KEYS:
            if forbidden in kwargs:
                raise TopologyError(
                    f"Cartesian coordinate key '{forbidden}' is strictly forbidden in topological node attributes."
                )

        if hybridization not in VALID_HYBRIDIZATIONS:
            raise TopologyError(
                f"Invalid hybridization state '{hybridization}'. Must be one of {sorted(VALID_HYBRIDIZATIONS)}"
            )

        # Dynamic Mendeleev resolution
        resolved_symbol = symbol.strip()
        if resolved_symbol.startswith("BEAD_") or resolved_symbol in ("X", "CG"):
            calc_atomic_num = 0 if atomic_number is None else atomic_number
            calc_mass = 0.0 if mass is None else mass
        else:
            try:
                elem = element(resolved_symbol)
                calc_atomic_num = int(elem.atomic_number) if atomic_number is None else atomic_number
                calc_mass = float(elem.mass) if mass is None else mass
            except Exception as exc:
                raise TopologyError(f"Unknown element symbol '{resolved_symbol}': {exc}") from exc

        super().add_node(
            node_id,
            symbol=resolved_symbol,
            atomic_number=calc_atomic_num,
            mass=calc_mass,
            formal_charge=int(formal_charge),
            hybridization=hybridization,
            in_ring=bool(in_ring),
            **kwargs,
        )

    def add_chemical_edge(
        self,
        u: int,
        v: int,
        bond_order: float = 1.0,
        aromatic: bool = False,
        in_ring: bool = False,
        stereo: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Validates connectivity and registers edge attributes."""
        if u not in self.nodes:
            raise TopologyError(f"Node {u} not found in graph.")
        if v not in self.nodes:
            raise TopologyError(f"Node {v} not found in graph.")
        if bond_order <= 0:
            raise TopologyError(f"Invalid bond_order {bond_order}. Must be strictly positive.")

        super().add_edge(
            u,
            v,
            bond_order=float(bond_order),
            aromatic=bool(aromatic),
            in_ring=bool(in_ring),
            stereo=stereo,
            **kwargs,
        )

    def substructure_search(self, query: TopologyGraph) -> list[dict[int, int]]:
        """Executes VF2 subgraph isomorphism search matching atomic and edge invariants."""
        def node_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
            return (
                n1.get("atomic_number") == n2.get("atomic_number")
                and n1.get("formal_charge") == n2.get("formal_charge")
                and n1.get("hybridization") == n2.get("hybridization")
            )

        def edge_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
            return (
                abs(float(e1.get("bond_order", 1.0)) - float(e2.get("bond_order", 1.0))) < 1e-3
                and bool(e1.get("aromatic", False)) == bool(e2.get("aromatic", False))
            )

        matcher = nx.isomorphism.GraphMatcher(
            self,
            query,
            node_match=node_match,
            edge_match=edge_match,
        )
        return list(matcher.subgraph_isomorphisms_iter())

    def to_qcschema_dict(self, geometry: np.ndarray | None = None) -> dict[str, Any]:
        """Exports all-atom topologies to QCSchema-compliant JSON dictionary."""
        sorted_nodes = sorted(self.nodes())
        symbols: list[str] = [str(self.nodes[n]["symbol"]) for n in sorted_nodes]
        atomic_numbers: list[int] = [int(self.nodes[n]["atomic_number"]) for n in sorted_nodes]
        masses: list[float] = [float(self.nodes[n]["mass"]) for n in sorted_nodes]
        molecular_charge: int = sum(int(self.nodes[n]["formal_charge"]) for n in sorted_nodes)

        # Node re-indexing map to [0..N-1]
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

        connectivity: list[list[Any]] = []
        for u, v, data in self.edges(data=True):
            connectivity.append([node_to_idx[u], node_to_idx[v], float(data.get("bond_order", 1.0))])

        geom_list: list[float] = []
        if geometry is not None:
            if geometry.shape[0] != len(sorted_nodes) or geometry.shape[1] != 3:
                raise TopologyError(
                    f"Geometry shape {geometry.shape} does not match node count ({len(sorted_nodes)}, 3)"
                )
            bohr_coords = geometry * ANGSTROM_TO_BOHR
            geom_list = [float(val) for val in bohr_coords.flatten()]

        # Extras dictionary preserving topological attributes
        hybridization_dict = {n: str(self.nodes[n]["hybridization"]) for n in sorted_nodes}
        in_ring_dict = {n: bool(self.nodes[n]["in_ring"]) for n in sorted_nodes}
        aromatic_bonds = [
            (node_to_idx[u], node_to_idx[v], float(data.get("bond_order", 1.5)))
            for u, v, data in self.edges(data=True)
            if data.get("aromatic", False)
        ]
        stereo_dict = {
            f"{node_to_idx[u]}-{node_to_idx[v]}": data.get("stereo")
            for u, v, data in self.edges(data=True)
            if data.get("stereo") is not None
        }

        schema: dict[str, Any] = {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "symbols": symbols,
            "atomic_numbers": atomic_numbers,
            "masses": masses,
            "molecular_charge": molecular_charge,
            "molecular_multiplicity": 1,
            "connectivity": connectivity,
            "geometry": geom_list,
            "extras": {
                "cochem_topology": {
                    "hybridization": hybridization_dict,
                    "in_ring": in_ring_dict,
                    "aromatic_bonds": aromatic_bonds,
                    "stereo": stereo_dict,
                }
            },
        }
        return schema

    def save_to_disk(self, filepath: Path | str) -> None:
        """Persists graph representation with atomic file replacement protected by FileLock."""
        target_path = Path(filepath).resolve()
        lock_path = target_path.with_suffix(target_path.suffix + ".lock")

        data = {
            "nodes": [
                {"id": n, **self.nodes[n]}
                for n in self.nodes()
            ],
            "edges": [
                {"u": u, "v": v, **self.edges[u, v]}
                for u, v in self.edges()
            ],
            "graph": dict(self.graph),
        }

        with FileLock(str(lock_path), timeout=10.0):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = target_path.with_suffix(f"{target_path.suffix}.tmp_{os.getpid()}")
            temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(temp_file, target_path)

    @classmethod
    def load_from_disk(cls, filepath: Path | str) -> TopologyGraph:
        """Loads serialized graph representation protected by FileLock."""
        target_path = Path(filepath).resolve()
        lock_path = target_path.with_suffix(target_path.suffix + ".lock")

        if not target_path.exists():
            raise TopologyError(f"Target topology file {target_path} does not exist.")

        with FileLock(str(lock_path), timeout=10.0):
            raw_text = target_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)

        graph = cls()
        graph.graph.update(data.get("graph", {}))

        for n_info in data.get("nodes", []):
            nid = n_info.pop("id")
            symbol = n_info.pop("symbol")
            formal_charge = n_info.pop("formal_charge", 0)
            hybridization = n_info.pop("hybridization", "sp3")
            in_ring = n_info.pop("in_ring", False)
            mass = n_info.pop("mass", None)
            atomic_number = n_info.pop("atomic_number", None)
            graph.add_chemical_node(
                node_id=nid,
                symbol=symbol,
                formal_charge=formal_charge,
                hybridization=hybridization,
                in_ring=in_ring,
                mass=mass,
                atomic_number=atomic_number,
                **n_info,
            )

        for e_info in data.get("edges", []):
            u = e_info.pop("u")
            v = e_info.pop("v")
            bond_order = e_info.pop("bond_order", 1.0)
            aromatic = e_info.pop("aromatic", False)
            in_ring = e_info.pop("in_ring", False)
            stereo = e_info.pop("stereo", None)
            graph.add_chemical_edge(
                u=u,
                v=v,
                bond_order=bond_order,
                aromatic=aromatic,
                in_ring=in_ring,
                stereo=stereo,
                **e_info,
            )

        return graph
