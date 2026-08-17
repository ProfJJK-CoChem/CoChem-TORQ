"""
CoChem-TORQ 0.0.11
Stage 1.0: Torsional Topology & Dihedral Matrix
-----------------------------------------------
Provides the 5-Option Dihedral Detection Engine and Covalent Radii 
Summation graph builder. Prepares the downstream Method Matrix Cascade 
configuration (r2SCAN-3c -> wB97X-D4 -> CCSD(T)-F12 + BSSE/VPT2) 
for the torsional grid scan.
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import json
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ] %(message)s")
logger = logging.getLogger("TorqTopology")

ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54
}

# Pyykkö Covalent Radii (Å) for single, double, triple bonds
PYYKKO_SINGLE_RADII = {
    "H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64,
    "Ne": 0.67, "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11,
    "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17,
    "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.48, "Nb": 1.37, "Mo": 1.36, "Tc": 1.26, "Ru": 1.26, "Rh": 1.25,
    "Pd": 1.25, "Ag": 1.28, "Cd": 1.36, "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31
}

class TorqTopology:
    def __init__(self, symbols: list[str], coordinates: list[list[float]] | np.ndarray, is_complex: bool = False) -> None:
        """
        Initialize the structural topology engine.
        """
        self.symbols = symbols
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.num_atoms = len(symbols)
        self.is_complex = is_complex
        self.graph = nx.Graph()
        
        self._build_covalent_graph()

    def _build_covalent_graph(self, tolerance_multiplier: float = 1.15) -> None:
        """
        Builds the molecular graph using Pyykkö covalent radii and bond-order tolerances.
        """
        dist_matrix = cdist(self.coordinates, self.coordinates)
        radii = np.array([PYYKKO_SINGLE_RADII.get(sym, 1.40) for sym in self.symbols])
        
        summed_radii_matrix = (radii[:, None] + radii[None, :]) * tolerance_multiplier
        
        for i, sym in enumerate(self.symbols):
            self.graph.add_node(i, element=sym, coords=self.coordinates[i])
            
        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                d = dist_matrix[i, j]
                r_sum = radii[i] + radii[j]
                if d < r_sum * tolerance_multiplier:
                    # Estimate bond order tolerance
                    bond_order = 1
                    if d < r_sum * 0.88:
                        bond_order = 3
                    elif d < r_sum * 0.95:
                        bond_order = 2
                    self.graph.add_edge(i, j, weight=d, bond_order=bond_order)
                    
        logger.info(f"Covalent graph built with Pyykkö radii: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

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
        H = p.T @ q
        U, S, Vt = np.linalg.svd(H)
        
        # Collinearity / Reflection trap prevention
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Vt[2, :] *= d
        
        rotation = Vt.T @ U.T
        aligned_q = (q @ rotation) + centroid_ref
        
        displacements = np.linalg.norm(ref_coords - aligned_q, axis=1)
        moving_atoms = np.where(displacements > 0.25)[0]
        logger.info(f"[Kabsch RMSD] Detected moving subset: {moving_atoms}")
        return moving_atoms.tolist()

    def detect_via_graph_theory(self, bond_to_sever: tuple[int, int]) -> list[list[int]]:
        """
        Option 3: Graph-Theory Edge Severing.
        Systematically severs a bridge bond and isolates the spinning top from the frame.
        :param bond_to_sever: tuple of (atom_idx_1, atom_idx_2)
        """
        temp_graph = self.graph.copy()
        if temp_graph.has_edge(*bond_to_sever):
            temp_graph.remove_edge(*bond_to_sever)
            subgraphs = list(nx.connected_components(temp_graph))
            if len(subgraphs) == 2:
                logger.info(f"[Graph Theory] Top 1: {subgraphs[0]} | Top 2: {subgraphs[1]}")
                return [list(subgraphs[0]), list(subgraphs[1])]
            else:
                raise ValueError(f"Bond severing resulted in {len(subgraphs)} subgraphs, expected 2.")
        raise ValueError(f"Bond {bond_to_sever} not found in graph.")

    def detect_via_coulomb_variance(self, ref_coords: np.ndarray) -> list[int]:
        """
        Option 4: Coulomb Matrix Variance.
        Calculates the translation-invariant Coulomb eigenspectrum variance.
        """
        def build_coulomb(coords: np.ndarray) -> np.ndarray:
            dist = cdist(coords, coords)
            np.fill_diagonal(dist, 1.0) # Prevent div by zero
            charges = np.array([ATOMIC_NUMBERS.get(sym, 6.0) for sym in self.symbols])
            q_mat = charges[:, None] * charges[None, :]
            c_mat = q_mat / dist
            np.fill_diagonal(c_mat, 0.5 * charges ** 2.4)
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
        assert len(indices) == 4, "Manual override requires exactly 4 indices defining a dihedral."
        logger.info(f"[Manual Override] Dihedral set to: {indices}")
        return indices

    # =========================================================================
    # CASCADE METHODOLOGY INJECTION & TRACK ROUTING
    # =========================================================================

    def generate_cascade_parameters(self, tier: str = "T3-1h", basis_set: str | None = None, method: str | None = None) -> dict[str, Any]:
        """
        Applies the CoChem Method Matrix Cascade parameters for the torsional scan.
        Refactored to map onto v4 T1-T4 tier rows ('T1-10s'..'T4-1mo') (§4.4, §9).
        Restricts BSSE Counterpoise correction per §4.7 / §9A rules.
        """
        tier_str = str(tier).strip()
        
        # Tier Mapping
        if tier_str in ["T1-10s", "low", "T1"]:
            tier_key = "T1-10s"
            method_name = method or "r2SCAN-3c"
            basis_name = basis_set or "r2SCAN-3c"
            keywords = ["! r2SCAN-3c", "TightSCF", "defgrid1", "Opt"]
        elif tier_str in ["T2-1m", "medium", "T2"]:
            tier_key = "T2-1m"
            method_name = method or "wB97X-D4"
            basis_name = basis_set or "def2-TZVP"
            keywords = ["! wB97X-D4", "def2-TZVP", "def2/J", "TightSCF", "defgrid1", "Opt"]
        elif tier_str in ["T3-1h", "high", "T3"]:
            tier_key = "T3-1h"
            method_name = method or "CCSD(T)-F12"
            basis_name = basis_set or "cc-pVTZ-F12"
            keywords = ["! CCSD(T)-F12", "cc-pVTZ-F12", "def2/J", "def2/C", "ExtremeSCF", "defgrid1", "Opt"]
        elif tier_str in ["T4-1mo", "ultra", "T4"]:
            tier_key = "T4-1mo"
            method_name = method or "CCSD(T)"
            basis_name = basis_set or "cc-pVTZ"
            keywords = ["! CCSD(T)", "cc-pVTZ", "ExtremeSCF", "Opt"]
        else:
            tier_key = tier_str
            method_name = method or "r2SCAN-3c"
            basis_name = basis_set or "def2-TZVP"
            keywords = [f"! {method_name}", basis_name, "TightSCF", "Opt"]

        params = {
            "tier": tier_key,
            "wall_time_tier": tier_key,
            "engine": "CFOUR" if tier_key == "T4-1mo" else "MPQC",
            "method": method_name,
            "basis_set": basis_name,
            "keywords": keywords,
            "anharmonicity": "! VPT2",
            "dispersion": "D4",
            "bsse_correction": None
        }

        # Enforce Counterpoise Correction Rules (§4.7, §9A)
        if self.is_complex:
            if should_apply_counterpoise(basis_name, method_name):
                logger.info(f"Complex identified with non-aug TZ basis ({basis_name}). Appending BSSE Counterpoise Correction.")
                params["bsse_correction"] = "Counterpoise"
                if "! CP" not in params["keywords"]:
                    params["keywords"].append("! CP")
            else:
                logger.info(f"Complex identified but CP addition prohibited for basis='{basis_name}'/method='{method_name}' per §4.7/§9A.")

        # CABS basis set mappings for F12 methods
        if "F12" in method_name.upper() or "F12" in basis_name.upper():
            params["cabs_mappings"] = {
                "OptRI": f"{basis_name}-OptRI",
                "JKFIT": f"{basis_name}-JKFIT",
                "MP2FIT": f"{basis_name}-MP2FIT"
            }

        with open("torq_run_params.json", "w") as f:
            json.dump(params, f, indent=4)
        
        logger.info(f"Cascade parameters written to torq_run_params.json at Tier: {tier_key}")
        return params

def should_apply_counterpoise(basis_set: str | None, method: str | None) -> bool:
    """
    Determines whether BSSE Counterpoise (CP) correction should be applied (§4.7, §9A).
    - Restricted to non-augmented triple-zeta basis sets (e.g. cc-pVTZ, def2-TZVP, def2-TZVPP).
    - Prohibited for augmented/diffuse basis sets (containing 'aug-', 'ma-', 'jun-', 'apr-', 'may-', 'jul-', or trailing diffuse designations like 'd', 'tzvpd', 'tzvppd').
    - Prohibited for CBS-extrapolated composite rows (containing 'CBS', 'W1', 'HEAT', 'COMPOSITE').
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
    Routes calculations between CFOUR and MPQC tracks based on $36N^2$ displacement arithmetic (§9).
    - Route CCSD(T) VPT2/analytic Hessians to CFOUR track.
    - Route DFT/SCF/F12 to MPQC track.
    - Abort MPQC CCSD(T)-F12 numerical VPT2 for N > 6 due to 36N^2 displacement penalty.
    """
    m_upper = (method or "").upper()

    # 1. Coupled-Cluster Anharmonicity / Analytic Hessians without F12 -> CFOUR Track
    if ("CCSD(T)" in m_upper or "CFOUR" in m_upper) and "F12" not in m_upper and is_anharmonic:
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
                f"MPQC CCSD(T)-F12 numerical VPT2 calculation aborted for system size N={n_atoms} > 6 "
                f"due to 36N^2 displacement penalty ({total_points} single points required). "
                f"Route to CFOUR track or limit N <= 6."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info(f"Routing MPQC CCSD(T)-F12 numerical VPT2 (N={n_atoms} <= 6) to MPQC track.")
            return "MPQC"

    # 3. Standard DFT/SCF/F12/Harmonic -> MPQC Track
    logger.info(f"Routing {method} (is_anharmonic={is_anharmonic}, N={n_atoms}) to MPQC track.")
    return "MPQC"

if __name__ == "__main__":
    # Self-test payload
    test_coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    test_syms = ["C", "C", "O", "H"]
    
    topos = TorqTopology(test_syms, test_coords, is_complex=False)
    topos.detect_via_override([0, 1, 2, 3])
    topos.generate_cascade_parameters(tier="T3-1h")
    
    logger.info("Track route CCSD(T) VPT2:", route_method_track("CCSD(T)", True, 5))
    logger.info("Track route CCSD(T)-F12 harmonic:", route_method_track("CCSD(T)-F12", False, 10))
