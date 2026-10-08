"""
CoChem-TORQ 0.0.11
Stage 1.0: Torsional Topology & Dihedral Matrix
-----------------------------------------------
Provides heuristic dihedral candidates and radius-based graph proposals.
Legacy method-recipe suggestions do not establish engine support or scientific
qualification; calculations require the validated dispatcher and real evidence.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from mendeleev import element
from scipy.spatial.distance import cdist

from Libraries.cochem_isotopes import isotope_mass

ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ] %(message)s"
)
logger = logging.getLogger("TorqTopology")


# =============================================================================
# DYNAMIC ATOMIC NUMBER, MASS, AND RADII INTEGRATION (Mendeleev Mandate)
# =============================================================================


def _parse_element_and_mass(symbol: str) -> tuple[str, int | None]:
    """
    Parse an element or isotope ('13C', 'C-13', 'C13', 'D', 'T', '18O', 'H:1')
    into a standardized element symbol and an optional validated mass number.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("An explicit element/isotope string is required.")
    clean_sym = symbol.strip()
    if clean_sym in ("D", "d", "2H", "2h", "H2", "H-2", "H:2", "H_2"):
        return _validated_isotope_tag("H", 2)
    if clean_sym in ("T", "t", "3H", "3h", "H3", "H-3", "H:3", "H_3"):
        return _validated_isotope_tag("H", 3)
    # Leading mass number: e.g. 13C, 18O, 13-C
    match_leading = re.fullmatch(r"(\d+)[-_:]?([A-Za-z]{1,2})", clean_sym)
    if match_leading:
        mass_num = int(match_leading.group(1))
        elem_sym = match_leading.group(2).capitalize()
        return _validated_isotope_tag(elem_sym, mass_num)
    # Trailing mass number: e.g. C13, C-13, C:13, O18
    match_trailing = re.fullmatch(r"([A-Za-z]{1,2})[-_:]?(\d+)", clean_sym)
    if match_trailing:
        elem_sym = match_trailing.group(1).capitalize()
        mass_num = int(match_trailing.group(2))
        return _validated_isotope_tag(elem_sym, mass_num)
    if not re.fullmatch(r"[A-Za-z]{1,2}", clean_sym):
        raise ValueError(f"Invalid element/isotope notation: {symbol!r}.")
    return clean_sym.capitalize(), None


def _validated_isotope_tag(symbol: str, mass_number: int) -> tuple[str, int]:
    """Element-level lookups cannot launder a nonexistent isotope label."""
    if mass_number <= 0:
        raise ValueError("An explicit isotope mass number must be a positive integer.")
    isotope_mass(f"{mass_number}{symbol}")
    return symbol, mass_number


def _positive_tabulated(value: Any, *, symbol: str, definition: str) -> float:
    """Require the requested database quantity without changing its definition."""
    if value is None:
        raise ValueError(f"No tabulated {definition} for {symbol!r}.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"No finite positive tabulated {definition} for {symbol!r}.")
    return result


@functools.lru_cache(maxsize=256)
def get_atomic_mass(symbol: str) -> float:
    """
    Return the database atomic_weight for a bare element, in u.

    This element-level quantity is not an isotope mass. For elements without a
    standard atomic weight the database may store a conventional representative
    value; isotope-dependent calculations must resolve an actual isotope instead.
    Explicit isotope notation uses its measured/tabulated mass, not a mass number
    or a fallback standard atomic weight. Tabulated isotope masses are not exact.
    """
    elem_sym, mass_num = _parse_element_and_mass(symbol)
    if mass_num is not None:
        return get_isotopic_mass(elem_sym, mass_num)
    el = element(elem_sym)
    return _positive_tabulated(
        el.atomic_weight, symbol=symbol, definition="atomic_weight"
    )


@functools.lru_cache(maxsize=256, typed=True)
def get_isotopic_mass(symbol: str, mass_number: int | None = None) -> float:
    """
    Return an actual resolved isotope mass in u using the shared strict resolver.

    A bare element selects its most abundant tabulated natural isotope. An
    element without that natural-abundance default requires an explicit isotope.
    An explicit argument must agree with any mass number already in the symbol.
    """
    elem_sym, parsed_mass = _parse_element_and_mass(symbol)
    if mass_number is not None and (type(mass_number) is not int or mass_number <= 0):
        raise ValueError("An explicit isotope mass number must be a positive integer.")
    if (
        mass_number is not None
        and parsed_mass is not None
        and mass_number != parsed_mass
    ):
        raise ValueError("Explicit isotope tag and mass-number argument conflict.")
    target_mass = mass_number if mass_number is not None else parsed_mass
    if target_mass is not None and target_mass <= 0:
        raise ValueError("An explicit isotope mass number must be a positive integer.")
    label = elem_sym if target_mass is None else f"{target_mass}{elem_sym}"
    return _positive_tabulated(
        isotope_mass(label), symbol=label, definition="isotope mass"
    )


@functools.lru_cache(maxsize=256)
def get_atomic_number(symbol: str) -> int:
    """
    Dynamically retrieves atomic number (Z) using mendeleev.
    Handles isotopic notations and aliases (e.g. 13C, D, T).
    """
    elem_sym, _ = _parse_element_and_mass(symbol)
    el = element(elem_sym)
    return int(el.atomic_number)


@functools.lru_cache(maxsize=256)
def get_pyykko_radius(symbol: str) -> float:
    """
    Retrieve the actual covalent_radius_pyykko single-bond radius in Angstrom.
    Mendeleev supplies picometers; conversion to Angstrom is division by100.
    """
    elem_sym, _ = _parse_element_and_mass(symbol)
    el = element(elem_sym)
    return (
        _positive_tabulated(
            el.covalent_radius_pyykko,
            symbol=symbol,
            definition="covalent_radius_pyykko",
        )
        / 100.0
    )


@functools.lru_cache(maxsize=256)
def get_vdw_radius(symbol: str) -> float:
    """
    Dynamically retrieves van der Waals radius in Angstroms using mendeleev.
    (Mendeleev provides vdw_radius in picometers, converted to Å via / 100.0).

    Only the database field named vdw_radius is requested. Other radius tables
    have distinct definitions and are never silently substituted.
    """
    elem_sym, _ = _parse_element_and_mass(symbol)
    el = element(elem_sym)
    return (
        _positive_tabulated(el.vdw_radius, symbol=symbol, definition="vdw_radius")
        / 100.0
    )


class TorqTopology:
    """Distance/radius heuristic candidates, not verified chemical topology.

    Coordinates and database radii are in Angstrom. Inferred edges and rough
    bond-order labels require independent chemical validation before constraints,
    torsional paths or electronic-state assignments use them as authoritative.
    """

    def __init__(
        self,
        symbols: list[str],
        coordinates: list[list[float]] | np.ndarray,
        is_complex: bool = False,
    ) -> None:
        """
        Initialize the structural topology engine.
        """
        self.symbols = symbols
        if np.iscomplexobj(coordinates):
            raise ValueError("Real Cartesian geometry in Angstrom is required.")
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.num_atoms = len(symbols)
        if (
            self.num_atoms == 0
            or self.coordinates.shape != (self.num_atoms, 3)
            or not np.all(np.isfinite(self.coordinates))
        ):
            raise ValueError(
                "Finite ordered [atoms,3] geometry in Angstrom is required."
            )
        if len(np.unique(self.coordinates, axis=0)) != self.num_atoms:
            raise ValueError("Coincident nuclei do not define a molecular graph.")
        self.is_complex = is_complex
        self.graph = nx.Graph()

        self._build_covalent_graph()

    def _build_covalent_graph(self, tolerance_multiplier: float = 1.15) -> None:
        """
        Build heuristic distance/radius edges and rough bond-order candidates.
        """
        if (
            isinstance(tolerance_multiplier, bool)
            or not np.isfinite(tolerance_multiplier)
            or tolerance_multiplier <= 0
        ):
            raise ValueError(
                "A finite positive radius tolerance multiplier is required."
            )
        dist_matrix = cdist(self.coordinates, self.coordinates)
        radii = np.array(
            [get_pyykko_radius(sym) for sym in self.symbols], dtype=np.float64
        )
        self.graph.graph.update(
            evidence_class="geometric_radius_heuristic",
            independently_verified=False,
            coordinate_unit="angstrom",
            radius_definition="Mendeleev covalent_radius_pyykko",
            radius_unit="angstrom",
            tolerance_multiplier=tolerance_multiplier,
            bond_order_status="heuristic_candidate",
            bond_order_distance_ratios={"double": 0.95, "triple": 0.88},
        )

        summed_radii_matrix = (radii[:, None] + radii[None, :]) * tolerance_multiplier

        for i, sym in enumerate(self.symbols):
            self.graph.add_node(i, element=sym, coords=self.coordinates[i])

        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                d = dist_matrix[i, j]
                r_sum_tol = summed_radii_matrix[i, j]
                if d < r_sum_tol:
                    r_sum = radii[i] + radii[j]
                    # Estimate bond order tolerance
                    bond_order = 1
                    if d < r_sum * 0.88:
                        bond_order = 3
                    elif d < r_sum * 0.95:
                        bond_order = 2
                    self.graph.add_edge(i, j, weight=d, bond_order=bond_order)

        logger.info(
            "Heuristic graph built with Pyykkö radii: %s nodes, %s edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # =========================================================================
    # THE 5-OPTION DIHEDRAL DETECTION ENGINE
    # =========================================================================

    def detect_via_zmatrix_diff(self, ref_coords: np.ndarray) -> list[int]:
        """
        Option 1: Z-Matrix Internal Coordinate Diffing.
        Analyzes the variation in the distance matrix to identify the cluster
        of atoms moving relative to the frame.
        """
        ref_dist = cdist(ref_coords, ref_coords)
        curr_dist = cdist(self.coordinates, self.coordinates)
        variance = np.abs(curr_dist - ref_dist)

        # Atoms with highest variance in distance to the rest of the molecule
        moving_atoms = np.where(np.sum(variance, axis=0) > 0.1)[0]
        logger.info(f"[Z-Matrix Diff] Detected moving subset: {moving_atoms}")
        return moving_atoms.tolist()

    def detect_via_kabsch_rmsd(self, ref_coords: np.ndarray) -> list[int]:
        """
        Option 2: Kabsch RMSD Heatmap.
        Aligns the backbone and subtracts the matrices, isolating the atoms
        with the largest physical displacement vector.
        """
        # Centering
        centroid_ref = np.mean(ref_coords, axis=0)
        centroid_curr = np.mean(self.coordinates, axis=0)
        p = ref_coords - centroid_ref
        q = self.coordinates - centroid_curr

        # Covariance matrix and SVD
        covariance = p.T @ q
        left, _, right_transposed = np.linalg.svd(covariance)

        # Collinearity / Reflection trap prevention
        d = np.sign(np.linalg.det(right_transposed.T @ left.T))
        right_transposed[2, :] *= d

        rotation = right_transposed.T @ left.T
        aligned_q = (q @ rotation) + centroid_ref

        displacements = np.linalg.norm(ref_coords - aligned_q, axis=1)
        moving_atoms = np.where(displacements > 0.25)[0]
        logger.info(f"[Kabsch RMSD] Detected moving subset: {moving_atoms}")
        return moving_atoms.tolist()

    def detect_via_graph_theory(
        self, bond_to_sever: tuple[int, int]
    ) -> list[list[int]]:
        """
        Option 3: Graph-Theory Edge Severing.
        Sever a candidate bridge to propose spinning-top/frame subsets.
        :param bond_to_sever: tuple of (atom_idx_1, atom_idx_2)
        """
        temp_graph = self.graph.copy()
        if temp_graph.has_edge(*bond_to_sever):
            temp_graph.remove_edge(*bond_to_sever)
            subgraphs = list(nx.connected_components(temp_graph))
            if len(subgraphs) == 2:
                logger.info(
                    f"[Graph Theory] Top 1: {subgraphs[0]} | Top 2: {subgraphs[1]}"
                )
                return [list(subgraphs[0]), list(subgraphs[1])]
            else:
                raise ValueError(
                    f"Bond severing resulted in {len(subgraphs)} subgraphs, expected 2."
                )
        raise ValueError(f"Bond {bond_to_sever} not found in graph.")

    def detect_via_coulomb_variance(self, ref_coords: np.ndarray) -> list[int]:
        """
        Option 4: Coulomb Matrix Variance.
        Compare translation-invariant labelled Coulomb-matrix rows heuristically.
        This does not calculate an eigenspectrum or verify a torsional mechanism.
        """

        def build_coulomb(coords: np.ndarray) -> np.ndarray:
            dist = cdist(coords, coords)
            np.fill_diagonal(dist, 1.0)  # Prevent div by zero
            charges = np.array(
                [get_atomic_number(sym) for sym in self.symbols], dtype=np.float64
            )
            q_mat = charges[:, None] * charges[None, :]
            c_mat = q_mat / dist
            np.fill_diagonal(c_mat, 0.5 * charges**2.4)
            return c_mat

        c_ref = build_coulomb(ref_coords)
        c_curr = build_coulomb(self.coordinates)
        diff = np.sum(np.abs(c_curr - c_ref), axis=1)

        moving_atoms = np.where(diff > np.mean(diff) + np.std(diff))[0]
        logger.info(f"[Coulomb Variance] Detected moving subset: {moving_atoms}")
        return moving_atoms.tolist()

    def detect_via_override(self, indices: list[int]) -> list[int]:
        """
        Option 5: Manual User Override.
        Bypasses algorithms and accepts exact 4-atom dihedral indices.
        """
        if len(indices) != 4:
            raise ValueError(
                "Manual override requires exactly 4 indices defining a dihedral."
            )
        logger.info(f"[Manual Override] Dihedral set to: {indices}")
        return indices

    # =========================================================================
    # CASCADE METHODOLOGY INJECTION & TRACK ROUTING
    # =========================================================================

    def generate_cascade_parameters(
        self,
        tier: str = "T3-1h",
        basis_set: str | None = None,
        method: str | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Write non-executable documentary tier requests for explicit review.

        This legacy helper has no validated MPQC/CFOUR input compiler. Historical
        names are planning requests, not exact registered recipes or scientific
        qualifications. The canonical dispatcher accepts separately defined,
        supported profiles after its actual capability/approval checks.
        """
        aliases = {
            "t1-10s": "T1-10s",
            "t1": "T1-10s",
            "low": "T1-10s",
            "t2-1m": "T2-1m",
            "t2": "T2-1m",
            "medium": "T2-1m",
            "t3-1h": "T3-1h",
            "t3": "T3-1h",
            "high": "T3-1h",
            "t4-1mo": "T4-1mo",
            "t4": "T4-1mo",
            "ultra": "T4-1mo",
        }
        defaults = {
            "T1-10s": ("r2SCAN-3c", "r2SCAN-3c", "MPQC"),
            "T2-1m": ("wB97X-D4", "def2-TZVP", "MPQC"),
            "T3-1h": ("CCSD(T)-F12", "cc-pVTZ-F12", "MPQC"),
            "T4-1mo": ("CCSD(T)", "cc-pVTZ", "CFOUR"),
        }
        if not isinstance(tier, str) or tier.strip().casefold() not in aliases:
            raise ValueError("An explicit known historical tier is required.")
        tier_key = aliases[tier.strip().casefold()]
        for value in (method, basis_set):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    "Explicit method/basis requests must be nonempty strings."
                )
        default_method, default_basis, documentary_engine = defaults[tier_key]
        method_name = default_method if method is None else method.strip()
        basis_name = default_basis if basis_set is None else basis_set.strip()
        f12_requested = "F12" in method_name.upper() or "F12" in basis_name.upper()
        params: dict[str, Any] = {
            "schema_version": "cochem.torq.legacy-cascade-planning/1",
            "status": "planning_only",
            "dispatch_authorized": False,
            "scientific_qualification": "unqualified",
            "tier": tier_key,
            "wall_time_tier": tier_key,
            "tier_runtime_guarantee": None,
            "proposal": {
                "requested_method": method_name,
                "requested_basis": basis_name,
                "documentary_preferred_engine": documentary_engine,
                "counterpoise_policy_suggestion": (
                    self.is_complex
                    and should_apply_counterpoise(basis_name, method_name)
                ),
                "counterpoise_policy_status": "legacy_policy_only_unvalidated",
                "anharmonic_model_request": "VPT2",
            },
            "engine_input": None,
            # Null runtime fields prevent legacy readers silently selecting a
            # different default method when given this planning document.
            "engine": None,
            "method": None,
            "basis_set": None,
            "keywords": [],
            "anharmonicity": None,
            "dispersion": None,
            "bsse_correction": None,
            "cabs_mappings": None,
            "auxiliary_basis": {
                "status": "unavailable" if f12_requested else "not_requested",
                "values": None,
                "reason": (
                    "Exact CABS/JKFIT/MP2FIT definitions require an authenticated "
                    "method/basis recipe and supported engine compiler; names "
                    "cannot be inferred by adding filename suffixes."
                    if f12_requested
                    else "No F12 auxiliary basis requested."
                ),
            },
            "blocked_reason": (
                "No validated executable recipe in this legacy helper. Select "
                "an explicit supported profile using the canonical dispatcher; "
                "a method-matrix identity does not authorize execution."
            ),
        }
        if output_path is not None:
            target_file = Path(output_path)
        else:
            target_file = Path(ARTIFACTS_DIR) / "torq_cascade_plan.json"

        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)

        logger.info(
            "Non-executable cascade plan written to %s at %s.", target_file, tier_key
        )
        return params


def should_apply_counterpoise(basis_set: str | None, method: str | None) -> bool:
    """
    Determines whether BSSE Counterpoise (CP) correction should be applied (§4.7, §9A).
    - Restricted to non-augmented triple-zeta bases, including cc-pVTZ and def2-TZVP.
    - Prohibited for augmented/diffuse bases ('aug-', 'ma-', 'jun-', 'apr-', 'may-',
      'jul-' or trailing diffuse designations 'd', 'tzvpd', 'tzvppd').
    - Prohibited for composite rows ('CBS', 'W1', 'HEAT', 'COMPOSITE').
    """
    basis_lower = (basis_set or "").lower().strip()
    method_upper = (method or "").upper().strip()

    # 1. Prohibit on CBS-extrapolated composite rows
    if any(cbs_kw in method_upper for cbs_kw in ["CBS", "W1", "HEAT", "COMPOSITE"]):
        return False

    # 2. Prohibit on augmented or diffuse basis sets
    aug_diffuse_prefixes = ["aug-", "aug", "ma-", "jun-", "apr-", "may-", "jul-"]
    if any(pref in basis_lower for pref in aug_diffuse_prefixes):
        return False

    if basis_lower.endswith("d") or "tzvpd" in basis_lower or "tzvppd" in basis_lower:
        return False

    # 3. Restrict to non-augmented triple-zeta basis sets
    valid_tz_bases = ["cc-pvtz", "def2-tzvp", "def2-tzvpp", "tzvp", "tzvpp"]
    clean_basis = basis_lower.split("/")[-1]
    is_non_aug_tz = clean_basis in valid_tz_bases

    return is_non_aug_tz


def route_method_track(method: str | None, is_anharmonic: bool, n_atoms: int) -> str:
    """
    Suggest legacy CFOUR/MPQC tracks using 36N^2 displacement arithmetic (§9).
    This suggestion does not establish an executable/qualified engine capability.
    - Route CCSD(T) VPT2/analytic Hessians to CFOUR track.
    - Route DFT/SCF/F12 to MPQC track.
    - Abort MPQC CCSD(T)-F12 numerical VPT2 for N > 6 due to 36N^2 displacement penalty.
    """
    m_upper = (method or "").upper()

    # 1. Coupled-Cluster Anharmonicity / Analytic Hessians without F12 -> CFOUR Track
    if (
        ("CCSD(T)" in m_upper or "CFOUR" in m_upper)
        and "F12" not in m_upper
        and is_anharmonic
    ):
        logger.info(f"Routing CCSD(T) VPT2 calculation (N={n_atoms}) to CFOUR track.")
        return "CFOUR"

    # 2. CCSD(T)-F12 Numerical VPT2 check
    if "F12" in m_upper and is_anharmonic:
        if n_atoms > 6:
            modes = 3 * n_atoms - 6 if n_atoms >= 3 else 1
            vpt2_displacements = 2 * modes + 1
            points_per_hess = (6 * n_atoms) ** 2
            total_points = vpt2_displacements * points_per_hess
            err_msg = (
                "MPQC CCSD(T)-F12 numerical VPT2 calculation aborted "
                f"for system size N={n_atoms} > 6 due to 36N^2 displacement "
                f"penalty ({total_points} single points required). "
                f"Route to CFOUR track or limit N <= 6."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info(
                "Routing MPQC CCSD(T)-F12 numerical VPT2 (N=%s <= 6) to MPQC track.",
                n_atoms,
            )
            return "MPQC"

    # 3. Standard DFT/SCF/F12/Harmonic -> MPQC Track
    logger.info(
        f"Routing {method} (is_anharmonic={is_anharmonic}, N={n_atoms}) to MPQC track."
    )
    return "MPQC"


if __name__ == "__main__":
    # Self-test payload
    test_coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    test_syms = ["C", "C", "O", "H"]

    topos = TorqTopology(test_syms, test_coords, is_complex=False)
    topos.detect_via_override([0, 1, 2, 3])
    topos.generate_cascade_parameters(tier="T3-1h")

    logger.info(f"Track route CCSD(T) VPT2: {route_method_track('CCSD(T)', True, 5)}")
    logger.info(
        "Track route CCSD(T)-F12 harmonic: %s",
        route_method_track("CCSD(T)-F12", False, 10),
    )
