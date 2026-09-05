"""CoChem-TOPOS: Pure Topological Molecular Graph Theory and Geometry Subsystem."""

from __future__ import annotations

from cochem.topos.clash import ClashPair, GeometricClashDetector
from cochem.topos.coarse_grain import GraphCrusherConfig, crush_macromolecule
from cochem.topos.exceptions import (
    ChiralityAssignmentError,
    IsomorphismMismatchError,
    StericClashError,
    TopologyError,
)
from cochem.topos.graph import TopologyGraph
from cochem.topos.rings import (
    canonicalize_cycle,
    perceive_aromaticity,
    perceive_cycle_basis,
)
from cochem.topos.stereochemistry import (
    assign_double_bond_stereo,
    assign_tetrahedral_chirality,
    compute_dihedral_angle,
)
from cochem.topos.visualization import TOPOSpy3DmolWidget

__all__ = [
    "TopologyGraph",
    "GraphCrusherConfig",
    "crush_macromolecule",
    "perceive_cycle_basis",
    "perceive_aromaticity",
    "canonicalize_cycle",
    "assign_tetrahedral_chirality",
    "assign_double_bond_stereo",
    "compute_dihedral_angle",
    "TOPOSpy3DmolWidget",
    "GeometricClashDetector",
    "ClashPair",
    "TopologyError",
    "StericClashError",
    "IsomorphismMismatchError",
    "ChiralityAssignmentError",
]
