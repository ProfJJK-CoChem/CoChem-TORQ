"""
CoChem-TORQ 0.0.11
Stage 1.0: Torsional Topology & Dihedral Matrix
-----------------------------------------------
Provides the 5-Option Dihedral Detection Engine and Covalent Radii 
Summation graph builder. Prepares the downstream Method Matrix Cascade 
configuration (r2SCAN-3c -> wB97X-D4 -> DLPNO-CCSD(T) + BSSE/VPT2) 
for the torsional grid scan.
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ] %(message)s")
logger = logging.getLogger("TorqTopology")

# Standard Covalent Radii (Å) for graph connectivity summation
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39
}

class TorqTopology:
    def __init__(self, symbols, coordinates, is_complex=False):
        """
        Initialize the structural topology engine.
        :param symbols: List of atomic symbols (e.g., ['C', 'H', 'H', ...])
        :param coordinates: Nx3 numpy array of Cartesian coordinates
        :param is_complex: Boolean flag indicating if this is a vdW complex (triggers BSSE)
        """
        self.symbols = symbols
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.num_atoms = len(symbols)
        self.is_complex = is_complex
        self.graph = nx.Graph()
        
        self._build_covalent_graph()

    def _build_covalent_graph(self, tolerance_multiplier=1.15):
        """
        Alternative 1 (Covalent Radii Summation): 
        Builds the molecular graph by computing the pairwise distance matrix and 
        comparing it against the sum of covalent radii with a breathing tolerance.
        """
        dist_matrix = cdist(self.coordinates, self.coordinates)
        radii = np.array([COVALENT_RADII.get(sym, 1.50) for sym in self.symbols])
        
        # Matrix of summed radii (r_i + r_j) * 1.15
        summed_radii_matrix = (radii[:, None] + radii[None, :]) * tolerance_multiplier
        
        # Add nodes
        for i, sym in enumerate(self.symbols):
            self.graph.add_node(i, element=sym, coords=self.coordinates[i])
            
        # Add edges where distance < summed radii (ignoring diagonal)
        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                if dist_matrix[i, j] < summed_radii_matrix[i, j]:
                    self.graph.add_edge(i, j, weight=dist_matrix[i, j])
                    
        logger.info(f"Covalent graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    # =========================================================================
    # THE 5-OPTION DIHEDRAL DETECTION ENGINE
    # =========================================================================

    def detect_via_zmatrix_diff(self, ref_coords):
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

    def detect_via_kabsch_rmsd(self, ref_coords):
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
        aligned_q = (q @ rotation.T) + centroid_ref
        
        displacements = np.linalg.norm(ref_coords - aligned_q, axis=1)
        moving_atoms = np.where(displacements > 0.25)[0]
        logger.info(f"[Kabsch RMSD] Detected moving subset: {moving_atoms}")
        return moving_atoms.tolist()

    def detect_via_graph_theory(self, bond_to_sever):
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
                logger.warning("Bond severing resulted in rings or fragmentation > 2.")
        return []

    def detect_via_coulomb_variance(self, ref_coords):
        """
        Option 4: Coulomb Matrix Variance.
        Calculates the translation-invariant Coulomb eigenspectrum variance.
        """
        def build_coulomb(coords):
            dist = cdist(coords, coords)
            np.fill_diagonal(dist, 1.0) # Prevent div by zero
            charges = np.array([1.0] * len(coords)) # Simplified for topology isolation
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

    def detect_via_override(self, indices):
        """
        Option 5: Manual User Override.
        Bypasses algorithms and accepts exact 4-atom dihedral indices.
        """
        assert len(indices) == 4, "Manual override requires exactly 4 indices defining a dihedral."
        logger.info(f"[Manual Override] Dihedral set to: {indices}")
        return indices

    # =========================================================================
    # CASCADE METHODOLOGY INJECTION
    # =========================================================================

    def generate_cascade_parameters(self, tier="high"):
        """
        Applies the CoChem Method Matrix Cascade parameters for the torsional scan.
        Injects BSSE and Dispersion (D4) corrections rigorously.
        """
        params = {
            "tier": tier,
            "engine": "ORCA_6.1.1",
            "keywords": [],
            "anharmonicity": "! VPT2",  # Native rotational-vibrational coupling
            "dispersion": "D4"
        }

        if tier == "low":
            params["keywords"] = ["! r2SCAN-3c", "TightSCF", "DefGrid3", "Opt"]
        elif tier == "medium":
            params["keywords"] = ["! wB97X-D4", "def2-TZVP", "def2/J", "TightSCF", "DefGrid3", "Opt"]
        elif tier == "high":
            params["keywords"] = ["! DLPNO-CCSD(T)", "def2-TZVPP", "def2/J", "def2/C", "ExtremeSCF", "DefGrid3", "Opt"]

        # Enforce Counterpoise Correction for Weak Interactions
        if self.is_complex:
            logger.info("Complex identified. Appending BSSE Counterpoise Correction block.")
            params["bsse_correction"] = "Counterpoise"
            params["keywords"].append("! CP")

        with open("torq_run_params.json", "w") as f:
            json.dump(params, f, indent=4)
        
        logger.info(f"Cascade parameters written to torq_run_params.json at Tier: {tier}")
        return params

if __name__ == "__main__":
    # Self-test payload
    mock_coords = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    mock_syms = ["C", "C", "O", "H"]
    
    topos = TorqTopology(mock_syms, mock_coords, is_complex=False)
    topos.detect_via_manual_override([0, 1, 2, 3])
    topos.generate_cascade_parameters(tier="high")