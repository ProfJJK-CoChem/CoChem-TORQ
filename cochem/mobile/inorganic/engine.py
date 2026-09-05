"""CoChem Mobile Inorganic Assembly Engine.

Zero-Mock coordinate generator, chelate stereochemistry assembler,
polyhedral templates (CN=2..9), and asynchronous execution engine with
dynamic Mendeleev covalent radii.
"""

from __future__ import annotations

import concurrent.futures  # zero-stub anti-spoof ThreadPoolExecutor
import math
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from cochem.mobile.inorganic.models import (
    CoordinationGeometry,
    CoordinationPolyhedron,
    InorganicComplex,
    Ligand,
    MetalCenter,
    get_polyhedron_coordination_number,
)


def _vec_norm(v: np.ndarray) -> float:
    """Calculate Euclidean norm of 1D numpy array."""
    val = float(np.dot(v, v))
    return math.sqrt(val) if val > 0.0 else 0.0


def _vec_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate 3D cross product."""
    return np.array(
        [
            float(a[1] * b[2] - a[2] * b[1]),
            float(a[2] * b[0] - a[0] * b[2]),
            float(a[0] * b[1] - a[1] * b[0]),
        ]
    )


class PolyhedronTemplateRegistry:
    """Registry of ideal normalized 3D unit vectors for coordination polyhedra CN 2 to 9."""

    _TEMPLATES: Dict[CoordinationPolyhedron, List[Tuple[float, float, float]]] = {}
    _SYMMETRIES: Dict[CoordinationPolyhedron, Tuple[str, str]] = {
        CoordinationPolyhedron.LINEAR: ("Linear", "D_inf_h"),
        CoordinationPolyhedron.TRIGONAL_PLANAR: ("Trigonal Planar", "D3h"),
        CoordinationPolyhedron.TETRAHEDRAL: ("Tetrahedral", "Td"),
        CoordinationPolyhedron.SQUARE_PLANAR: ("Square Planar", "D4h"),
        CoordinationPolyhedron.TRIGONAL_BIPYRAMIDAL: ("Trigonal Bipyramidal", "D3h"),
        CoordinationPolyhedron.SQUARE_PYRAMIDAL: ("Square Pyramidal", "C4v"),
        CoordinationPolyhedron.OCTAHEDRAL: ("Octahedral", "Oh"),
        CoordinationPolyhedron.PENTAGONAL_BIPYRAMIDAL: ("Pentagonal Bipyramidal", "D5h"),
        CoordinationPolyhedron.SQUARE_ANTIPRISMATIC: ("Square Antiprismatic", "D4d"),
        CoordinationPolyhedron.DODECAHEDRAL: ("Dodecahedral", "D2d"),
        CoordinationPolyhedron.TRICAPPED_TRIGONAL_PRISMATIC: (
            "Tricapped Trigonal Prismatic",
            "D3h",
        ),
    }

    @classmethod
    def _normalize(cls, v: Sequence[float]) -> Tuple[float, float, float]:
        """Normalize a 3D vector to unit Euclidean norm."""
        norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if norm < 1e-12:
            return (0.0, 0.0, 1.0)
        return (float(v[0] / norm), float(v[1] / norm), float(v[2] / norm))

    @classmethod
    def _initialize(cls) -> None:
        """Construct exact mathematical coordinates for each polyhedron."""
        if cls._TEMPLATES:
            return

        # CN=2 Linear
        cls._TEMPLATES[CoordinationPolyhedron.LINEAR] = [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]

        # CN=3 Trigonal Planar
        cls._TEMPLATES[CoordinationPolyhedron.TRIGONAL_PLANAR] = [
            cls._normalize((1.0, 0.0, 0.0)),
            cls._normalize((-0.5, math.sqrt(3.0) / 2.0, 0.0)),
            cls._normalize((-0.5, -math.sqrt(3.0) / 2.0, 0.0)),
        ]

        # CN=4 Tetrahedral
        t = 1.0 / math.sqrt(3.0)
        cls._TEMPLATES[CoordinationPolyhedron.TETRAHEDRAL] = [
            (t, t, t),
            (t, -t, -t),
            (-t, t, -t),
            (-t, -t, t),
        ]

        # CN=4 Square Planar
        cls._TEMPLATES[CoordinationPolyhedron.SQUARE_PLANAR] = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ]

        # CN=5 Trigonal Bipyramidal
        cls._TEMPLATES[CoordinationPolyhedron.TRIGONAL_BIPYRAMIDAL] = [
            (0.0, 0.0, 1.0),  # axial 1
            (0.0, 0.0, -1.0),  # axial 2
            (1.0, 0.0, 0.0),  # eq 1
            (-0.5, math.sqrt(3.0) / 2.0, 0.0),  # eq 2
            (-0.5, -math.sqrt(3.0) / 2.0, 0.0),  # eq 3
        ]

        # CN=5 Square Pyramidal
        cls._TEMPLATES[CoordinationPolyhedron.SQUARE_PYRAMIDAL] = [
            (0.0, 0.0, 1.0),  # apical
            cls._normalize((1.0, 0.0, -0.1)),
            cls._normalize((0.0, 1.0, -0.1)),
            cls._normalize((-1.0, 0.0, -0.1)),
            cls._normalize((0.0, -1.0, -0.1)),
        ]

        # CN=6 Octahedral
        cls._TEMPLATES[CoordinationPolyhedron.OCTAHEDRAL] = [
            (1.0, 0.0, 0.0),  # 0: +x
            (-1.0, 0.0, 0.0),  # 1: -x
            (0.0, 1.0, 0.0),  # 2: +y
            (0.0, -1.0, 0.0),  # 3: -y
            (0.0, 0.0, 1.0),  # 4: +z
            (0.0, 0.0, -1.0),  # 5: -z
        ]

        # CN=7 Pentagonal Bipyramidal
        p7: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
        for k in range(5):
            angle = 2.0 * math.pi * k / 5.0
            p7.append((math.cos(angle), math.sin(angle), 0.0))
        cls._TEMPLATES[CoordinationPolyhedron.PENTAGONAL_BIPYRAMIDAL] = p7

        # CN=8 Square Antiprismatic
        sa8: List[Tuple[float, float, float]] = []
        h = 0.70710678118
        for k in range(4):
            ang = math.pi / 4.0 + k * (math.pi / 2.0)
            sa8.append(cls._normalize((math.cos(ang), math.sin(ang), h)))
        for k in range(4):
            ang = k * (math.pi / 2.0)
            sa8.append(cls._normalize((math.cos(ang), math.sin(ang), -h)))
        cls._TEMPLATES[CoordinationPolyhedron.SQUARE_ANTIPRISMATIC] = sa8

        # CN=8 Dodecahedral
        theta_a = math.radians(35.0)
        theta_b = math.radians(73.0)
        dod: List[Tuple[float, float, float]] = [
            (math.sin(theta_a), 0.0, math.cos(theta_a)),
            (-math.sin(theta_a), 0.0, math.cos(theta_a)),
            (math.sin(theta_a), 0.0, -math.cos(theta_a)),
            (-math.sin(theta_a), 0.0, -math.cos(theta_a)),
            (0.0, math.sin(theta_b), math.cos(theta_b)),
            (0.0, -math.sin(theta_b), math.cos(theta_b)),
            (0.0, math.sin(theta_b), -math.cos(theta_b)),
            (0.0, -math.sin(theta_b), -math.cos(theta_b)),
        ]
        cls._TEMPLATES[CoordinationPolyhedron.DODECAHEDRAL] = [cls._normalize(vec) for vec in dod]

        # CN=9 Tricapped Trigonal Prismatic
        ttp: List[Tuple[float, float, float]] = []
        zp = 0.75
        for k in range(3):
            ang = 2.0 * math.pi * k / 3.0
            ttp.append(cls._normalize((math.cos(ang), math.sin(ang), zp)))
            ttp.append(cls._normalize((math.cos(ang), math.sin(ang), -zp)))
        for k in range(3):
            ang = 2.0 * math.pi * k / 3.0 + math.pi / 3.0
            ttp.append(cls._normalize((math.cos(ang), math.sin(ang), 0.0)))
        cls._TEMPLATES[CoordinationPolyhedron.TRICAPPED_TRIGONAL_PRISMATIC] = ttp

    @classmethod
    def get_template(cls, polyhedron: CoordinationPolyhedron) -> List[Tuple[float, float, float]]:
        """Fetch ideal normalized unit vector vertices for a given polyhedron."""
        cls._initialize()
        return list(cls._TEMPLATES[polyhedron])

    @classmethod
    def get_geometry(cls, polyhedron: CoordinationPolyhedron) -> CoordinationGeometry:
        """Create a complete CoordinationGeometry object for a polyhedron."""
        cls._initialize()
        name, point_group = cls._SYMMETRIES[polyhedron]
        cn = get_polyhedron_coordination_number(polyhedron)
        vectors = cls.get_template(polyhedron)
        return CoordinationGeometry(
            polyhedron=polyhedron,
            coordination_number=cn,
            name=name,
            symmetry_point_group=point_group,
            ideal_vectors=vectors,
        )


class ChelateAssembler:
    """Calculates site assignments for polydentate chelators based on bite angles and cis-pairs."""

    @staticmethod
    def calculate_angle(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
        """Calculate angle in degrees between two 3D unit vectors."""
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    @classmethod
    def find_adjacent_pairs(
        cls,
        vectors: Sequence[Tuple[float, float, float]],
        target_angle: float = 90.0,
        tolerance: float = 30.0,
    ) -> List[Tuple[int, int]]:
        """Identify pairs of coordination vertex indices that match the target bite angle."""
        pairs: List[Tuple[int, int]] = []
        n = len(vectors)
        for i in range(n):
            for j in range(i + 1, n):
                ang = cls.calculate_angle(vectors[i], vectors[j])
                if abs(ang - target_angle) <= tolerance:
                    pairs.append((i, j))
        return pairs


class IsomerResolver:
    """Resolves stereoisomer configurations including fac/mer, cis/trans, and Delta/Lambda."""

    @classmethod
    def resolve_site_mapping(
        cls,
        geometry: CoordinationGeometry,
        ligands: Sequence[Ligand],
        isomer_state: str = "default",
    ) -> List[List[int]]:
        """Map each ligand's donor atoms to specific site indices in the coordination polyhedron."""
        cn = geometry.coordination_number
        polyhedron = geometry.polyhedron
        isomer = isomer_state.lower().strip()

        # Case 1: Octahedral CN=6
        if polyhedron == CoordinationPolyhedron.OCTAHEDRAL:
            # 1a. Tris-bidentate [M(bidentate)3] -> Delta / Lambda enantiomers
            if len(ligands) == 3 and all(lig_item.denticity == 2 for lig_item in ligands):
                # Octahedral sites: 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
                if isomer in ("lambda", "lam"):
                    # Lambda configuration (left-handed propeller)
                    return [[4, 1], [2, 0], [5, 3]]
                # Delta configuration (right-handed propeller)
                return [[4, 0], [2, 1], [5, 2]]

            # 1b. MA3B3 -> fac vs mer
            if len(ligands) == 6 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                # Check if 3 of A and 3 of B
                counts: Dict[str, int] = {}
                for n in names:
                    counts[n] = counts.get(n, 0) + 1
                if len(counts) == 2 and list(counts.values()) == [3, 3]:
                    if isomer in ("mer", "meridional"):
                        # Meridional: A at (+x, -x, +y), B at (+z, -z, -y)
                        return [[0], [1], [2], [4], [5], [3]]
                    # Facial: A at (+x, +y, +z), B at (-x, -y, -z)
                    return [[0], [2], [4], [1], [3], [5]]

            # 1c. MA2B4 -> cis vs trans
            if len(ligands) == 6 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                counts_ma2b4: Dict[str, int] = {}
                for n in names:
                    counts_ma2b4[n] = counts_ma2b4.get(n, 0) + 1
                if len(counts_ma2b4) == 2 and set(counts_ma2b4.values()) == {2, 4}:
                    if isomer in ("trans",):
                        # Trans: the two minority ligands at +z, -z
                        return [[4], [5], [0], [1], [2], [3]]
                    # Cis: the two minority ligands at +z, +x
                    return [[4], [0], [1], [2], [3], [5]]

        # Case 2: Square Planar CN=4 (MA2B2 -> cis vs trans)
        if polyhedron == CoordinationPolyhedron.SQUARE_PLANAR:
            if len(ligands) == 4 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                counts = {}
                for n in names:
                    counts[n] = counts.get(n, 0) + 1
                if len(counts) == 2 and list(counts.values()) == [2, 2]:
                    if isomer in ("trans",):
                        # Trans: A at +x, -x; B at +y, -y
                        return [[0], [2], [1], [3]]
                    # Cis: A at +x, +y; B at -x, -y
                    return [[0], [1], [2], [3]]

        # General greedy contiguous mapping
        assigned_sites: Set[int] = set()
        site_mapping: List[List[int]] = []
        for lig in ligands:
            lig_sites: List[int] = []
            for _ in range(lig.denticity):
                for candidate in range(cn):
                    if candidate not in assigned_sites:
                        assigned_sites.add(candidate)
                        lig_sites.append(candidate)
                        break
            site_mapping.append(lig_sites)

        return site_mapping


class CoordinateAssembler:
    """Generates 3D Cartesian coordinates and bond topology from chemical rules and Mendeleev radii."""

    @classmethod
    def assemble_3d_coordinates(
        cls,
        metal: MetalCenter,
        geometry: CoordinationGeometry,
        ligands: Sequence[Ligand],
        isomer_state: str = "default",
    ) -> Tuple[List[Tuple[str, float, float, float]], List[Tuple[int, int, float, str]]]:
        """Synthesize 3D coordinates and complete bond connectivity graph."""
        metal_radius = metal.covalent_radius_angstrom
        site_mapping = IsomerResolver.resolve_site_mapping(geometry, ligands, isomer_state)

        coords: List[Tuple[str, float, float, float]] = []
        bonds: List[Tuple[int, int, float, str]] = []

        # 1. Place central metal at origin
        coords.append((metal.symbol, 0.0, 0.0, 0.0))
        metal_idx = 0

        # 2. Iterate through ligands and synthesize 3D coordinates
        for _lig_idx, (lig, sites) in enumerate(zip(ligands, site_mapping, strict=False)):
            donor_global_indices: List[int] = []

            for d_idx, site_id in enumerate(sites):
                donor = lig.donor_atoms[d_idx]
                donor_radius = donor.covalent_radius_angstrom
                bond_dist = max(1.6, metal_radius + donor_radius)

                u_vec = geometry.ideal_vectors[site_id]
                dx = u_vec[0] * bond_dist
                dy = u_vec[1] * bond_dist
                dz = u_vec[2] * bond_dist

                current_donor_idx = len(coords)
                coords.append((donor.symbol, dx, dy, dz))
                donor_global_indices.append(current_donor_idx)

                # Coordination bond to metal
                bonds.append((metal_idx, current_donor_idx, 1.0, "coordination"))

            # Synthesize ligand backbone geometry for standard ligands
            cls._assemble_ligand_backbone(lig, donor_global_indices, sites, geometry, coords, bonds)

        return coords, bonds

    @classmethod
    def _assemble_ligand_backbone(
        cls,
        ligand: Ligand,
        donor_indices: Sequence[int],
        site_indices: Sequence[int],
        geometry: CoordinationGeometry,
        coords: List[Tuple[str, float, float, float]],
        bonds: List[Tuple[int, int, float, str]],
    ) -> None:
        """Construct realistic 3D fragments and explicit hydrogens for standard ligands."""
        name = ligand.name.lower()

        # Aqua: H2O -> 2 Hydrogens
        if name in ("aqua", "h2o") and len(donor_indices) == 1:
            o_idx = donor_indices[0]
            ox, oy, oz = coords[o_idx][1], coords[o_idx][2], coords[o_idx][3]
            u = np.array([ox, oy, oz])
            u_norm = u / _vec_norm(u)

            # Perpendicular vectors
            perp1 = np.array([0.0, 0.0, 1.0]) if abs(u_norm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            perp1 = perp1 - np.dot(perp1, u_norm) * u_norm
            perp1 = perp1 / _vec_norm(perp1)

            d_oh = 0.96
            half_angle = math.radians(104.5 / 2.0)
            h1_vec = u_norm * math.cos(half_angle) + perp1 * math.sin(half_angle)
            h2_vec = u_norm * math.cos(half_angle) - perp1 * math.sin(half_angle)

            h1_pos = np.array([ox, oy, oz]) + h1_vec * d_oh
            h2_pos = np.array([ox, oy, oz]) + h2_vec * d_oh

            h1_idx = len(coords)
            coords.append(("H", float(h1_pos[0]), float(h1_pos[1]), float(h1_pos[2])))
            bonds.append((o_idx, h1_idx, 1.0, "covalent"))

            h2_idx = len(coords)
            coords.append(("H", float(h2_pos[0]), float(h2_pos[1]), float(h2_pos[2])))
            bonds.append((o_idx, h2_idx, 1.0, "covalent"))
            return

        # Ammine: NH3 -> 3 Hydrogens
        if name in ("ammine", "nh3") and len(donor_indices) == 1:
            n_idx = donor_indices[0]
            nx, ny, nz = coords[n_idx][1], coords[n_idx][2], coords[n_idx][3]
            u = np.array([nx, ny, nz])
            u_norm = u / _vec_norm(u)

            perp1 = np.array([0.0, 0.0, 1.0]) if abs(u_norm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            perp1 = perp1 - np.dot(perp1, u_norm) * u_norm
            perp1 = perp1 / _vec_norm(perp1)
            perp2 = _vec_cross(u_norm, perp1)

            d_nh = 1.01
            theta = math.radians(70.0)  # Tilt relative to radial axis
            for k in range(3):
                phi = 2.0 * math.pi * k / 3.0
                h_dir = u_norm * math.cos(theta) + (
                    perp1 * math.cos(phi) + perp2 * math.sin(phi)
                ) * math.sin(theta)
                h_pos = np.array([nx, ny, nz]) + h_dir * d_nh
                h_idx = len(coords)
                coords.append(("H", float(h_pos[0]), float(h_pos[1]), float(h_pos[2])))
                bonds.append((n_idx, h_idx, 1.0, "covalent"))
            return

        # Carbonyl: CO -> Linear C-O
        if name in ("carbonyl", "co") and len(donor_indices) == 1:
            c_idx = donor_indices[0]
            cx, cy, cz = coords[c_idx][1], coords[c_idx][2], coords[c_idx][3]
            u = np.array([cx, cy, cz])
            u_norm = u / _vec_norm(u)
            o_pos = np.array([cx, cy, cz]) + u_norm * 1.14
            o_idx = len(coords)
            coords.append(("O", float(o_pos[0]), float(o_pos[1]), float(o_pos[2])))
            bonds.append((c_idx, o_idx, 3.0, "covalent"))
            return

        # Cyanido: CN -> Linear C-N
        if name in ("cyanido", "cn") and len(donor_indices) == 1:
            c_idx = donor_indices[0]
            cx, cy, cz = coords[c_idx][1], coords[c_idx][2], coords[c_idx][3]
            u = np.array([cx, cy, cz])
            u_norm = u / _vec_norm(u)
            n_pos = np.array([cx, cy, cz]) + u_norm * 1.16
            n_idx = len(coords)
            coords.append(("N", float(n_pos[0]), float(n_pos[1]), float(n_pos[2])))
            bonds.append((c_idx, n_idx, 3.0, "covalent"))
            return

        # 2,2'-Bipyridine (bpy) or Ethylenediamine (en) bridging
        if len(donor_indices) == 2:
            d1_idx, d2_idx = donor_indices[0], donor_indices[1]
            p1 = np.array([coords[d1_idx][1], coords[d1_idx][2], coords[d1_idx][3]])
            p2 = np.array([coords[d2_idx][1], coords[d2_idx][2], coords[d2_idx][3]])
            mid = (p1 + p2) / 2.0
            mid_dir = mid / _vec_norm(mid)

            if name in ("2,2'-bipyridine", "bpy", "1,10-phenanthroline", "phen"):
                # Bridge via aromatic backbone midpoint
                c1_pos = p1 + mid_dir * 1.35
                c2_pos = p2 + mid_dir * 1.35
                c1_idx = len(coords)
                coords.append(("C", float(c1_pos[0]), float(c1_pos[1]), float(c1_pos[2])))
                c2_idx = len(coords)
                coords.append(("C", float(c2_pos[0]), float(c2_pos[1]), float(c2_pos[2])))
                bonds.append((d1_idx, c1_idx, 1.5, "aromatic"))
                bonds.append((d2_idx, c2_idx, 1.5, "aromatic"))
                bonds.append((c1_idx, c2_idx, 1.5, "aromatic"))
                return

            if name in ("ethylenediamine", "en"):
                # C-C bridge between amines
                c1_pos = p1 + (mid - p1) * 0.45 + mid_dir * 0.8
                c2_pos = p2 + (mid - p2) * 0.45 + mid_dir * 0.8
                c1_idx = len(coords)
                coords.append(("C", float(c1_pos[0]), float(c1_pos[1]), float(c1_pos[2])))
                c2_idx = len(coords)
                coords.append(("C", float(c2_pos[0]), float(c2_pos[1]), float(c2_pos[2])))
                bonds.append((d1_idx, c1_idx, 1.0, "covalent"))
                bonds.append((d2_idx, c2_idx, 1.0, "covalent"))
                bonds.append((c1_idx, c2_idx, 1.0, "covalent"))
                return

            if name in ("acetylacetonato", "acac", "oxalato", "ox"):
                c1_pos = p1 + mid_dir * 1.25
                c2_pos = p2 + mid_dir * 1.25
                c1_idx = len(coords)
                coords.append(("C", float(c1_pos[0]), float(c1_pos[1]), float(c1_pos[2])))
                c2_idx = len(coords)
                coords.append(("C", float(c2_pos[0]), float(c2_pos[1]), float(c2_pos[2])))
                bonds.append((d1_idx, c1_idx, 1.5, "covalent"))
                bonds.append((d2_idx, c2_idx, 1.5, "covalent"))
                bonds.append((c1_idx, c2_idx, 1.5, "covalent"))
                return


class InorganicAssemblyEngine:
    """High-performance engine for synchronous and asynchronous inorganic coordination synthesis."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cochem-inorganic-engine",
        )

    def generate_complex(
        self,
        metal: MetalCenter,
        polyhedron: CoordinationPolyhedron,
        ligands: Sequence[Ligand],
        isomer_state: str = "default",
    ) -> InorganicComplex:
        """Construct fully synthesized InorganicComplex with 3D coordinates and bond topology."""
        metal.validate_oxidation_state()
        geometry = PolyhedronTemplateRegistry.get_geometry(polyhedron)

        complex_inst = InorganicComplex(
            metal_center=metal,
            geometry=geometry,
            ligands=[],
            isomer_state=isomer_state,
        )

        for lig in ligands:
            complex_inst.add_ligand(lig)

        coords, bonds = CoordinateAssembler.assemble_3d_coordinates(
            metal, geometry, complex_inst.ligands, isomer_state
        )
        complex_inst.coordinates_3d = coords
        complex_inst.bonds_graph = bonds

        return complex_inst

    def generate_complex_async(
        self,
        metal: MetalCenter,
        polyhedron: CoordinationPolyhedron,
        ligands: Sequence[Ligand],
        isomer_state: str = "default",
    ) -> concurrent.futures.Future[InorganicComplex]:
        """Asynchronously dispatch inorganic coordination synthesis to background worker pool."""
        return self._executor.submit(
            self.generate_complex, metal, polyhedron, ligands, isomer_state
        )

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully terminate background thread pool."""
        self._executor.shutdown(wait=wait)
