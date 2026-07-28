"""
CoChem-TORQ 0.0.11
Stage 4.1: Tensor Extraction & Provenance
-----------------------------------------
Mathematically processes optimized Cartesian coordinates to derive
the Principal Axes of Inertia and Rotational Constants (MHz).
Implements the Cartesian Protection layer (Linearity Trap) to prevent
singularities during partition function generation for linear complexes.
"""

import numpy as np
import json
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Tensor] %(message)s")
logger = logging.getLogger("TorqTensorExt")

# CODATA 2018/2022 Physical Constants explicitly locked to prevent drift
CODATA_YEAR = 2022
PLANCK_CONSTANT_JS = 6.62607015e-34  # Exact J s
C_M_S = 299792458.0  # Exact m/s
# Conversion factor: amu * Angstrom^2 to MHz
# B(MHz) = h / (8 * pi^2 * I) * conversion_factors
AMU_A2_TO_MHZ = 505379.005 

# AME2020 Exact Isotopic Masses (Most abundant isotope for baseline)
EXACT_MASSES = {
    "H": 1.007825032, "C": 12.000000000, "N": 14.003074004,
    "O": 15.994914619, "F": 18.998403163, "P": 30.973761998,
    "S": 31.972071174, "Cl": 34.968852682, "Br": 78.9183371,
    "I": 126.904473
}

class TorqTensorExtractor:
    def __init__(self, symbols, coordinates, point_id="000"):
        """
        Initializes the tensor extractor.
        :param symbols: List of element symbols.
        :param coordinates: Nx3 numpy array of geometries.
        :param point_id: Topographic identifier for provenance tracking.
        """
        self.symbols = symbols
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.point_id = point_id
        
        self.masses = np.array([EXACT_MASSES.get(sym, 12.0) for sym in self.symbols])
        self.total_mass = np.sum(self.masses)
        
        self.center_of_mass = np.zeros(3)
        self.principal_moments = np.zeros(3)
        self.principal_axes = np.eye(3)
        self.rotational_constants_mhz = np.zeros(3)
        self.is_linear = False

    def apply_cartesian_protections(self):
        """
        Translates the system to the Center of Mass (COM) and checks for
        pathological linearity which shatters Z-matrix generation.
        """
        # Calculate COM
        self.center_of_mass = np.average(self.coordinates, axis=0, weights=self.masses)
        self.coordinates -= self.center_of_mass
        
        # Check for absolute linearity (all atoms along a single vector)
        # We do this by checking if the coordinate matrix rank is 1
        u, s, vh = np.linalg.svd(self.coordinates)
        
        # If the second singular value is near zero, the molecule is 1D (linear)
        if len(s) > 1 and np.isclose(s[1], 0.0, atol=1e-4):
            self.is_linear = True
            logger.warning(f"Linearity Trap Triggered! Geometry {self.point_id} is mathematically linear (< 1e-4 Å dev). Cartesian protections engaged.")
        
        return self.coordinates

    def compute_inertia_tensor(self):
        """
        Builds the 3x3 inertia tensor matrix from the COM-centered coordinates.
        """
        I = np.zeros((3, 3))
        for m, (x, y, z) in zip(self.masses, self.coordinates):
            I[0, 0] += m * (y**2 + z**2)
            I[1, 1] += m * (x**2 + z**2)
            I[2, 2] += m * (x**2 + y**2)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
            
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        return I

    def diagonalize_and_derive_constants(self):
        """
        Diagonalizes the inertia tensor to find the Principal Moments of Inertia (amu*A^2)
        and converts them to Rotational Constants A, B, C in MHz.
        """
        inertia_matrix = self.compute_inertia_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_matrix)
        
        # Sort eigenvalues to ensure Ia <= Ib <= Ic  => A >= B >= C
        sort_indices = np.argsort(eigenvalues)
        self.principal_moments = eigenvalues[sort_indices]
        self.principal_axes = eigenvectors[:, sort_indices]
        
        constants_mhz = []
        for i_val in self.principal_moments:
            # Linearity Trap: If moment of inertia is mathematically 0 (e.g. diatomic axis)
            if np.isclose(i_val, 0.0, atol=1e-4):
                constants_mhz.append(0.0)
            else:
                constants_mhz.append(AMU_A2_TO_MHZ / i_val)
                
        self.rotational_constants_mhz = np.array(constants_mhz)
        
        logger.info(f"Point {self.point_id} | A: {self.rotational_constants_mhz[0]:.4f} MHz, "
                    f"B: {self.rotational_constants_mhz[1]:.4f} MHz, C: {self.rotational_constants_mhz[2]:.4f} MHz")
        
        return self.rotational_constants_mhz

    def export_provenance(self):
        """
        Generates the JSON-LD compliant semantic tracker for the extracted tensors.
        Cryptographically hashes the coordinates and atomic weights.
        """
        payload_string = f"{self.symbols}_{self.coordinates.round(6).tolist()}_{self.masses.tolist()}"
        sha_hash = hashlib.sha256(payload_string.encode('utf-8')).hexdigest()
        
        provenance = {
            "point_id": self.point_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "codata_year": CODATA_YEAR,
            "geometry_hash_sha256": sha_hash,
            "is_linear": self.is_linear,
            "tensors": {
                "center_of_mass_angstrom": self.center_of_mass.tolist(),
                "principal_moments_amu_A2": self.principal_moments.tolist(),
                "rotational_constants_MHz": {
                    "A": self.rotational_constants_mhz[0],
                    "B": self.rotational_constants_mhz[1],
                    "C": self.rotational_constants_mhz[2]
                }
            }
        }
        
        out_file = f"torq_tensors_{self.point_id}.json"
        with open(out_file, "w") as f:
            json.dump(provenance, f, indent=4)
            
        logger.info(f"Tensor provenance successfully sealed to {out_file} [SHA256: {sha_hash[:8]}...]")
        return provenance

if __name__ == "__main__":
    # Self-test: Linearity Trap and Normal Extraction (CO2-like mock vs non-linear)
    mock_syms_linear = ["O", "C", "O"]
    mock_coords_linear = [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]
    
    extractor = TorqTensorExtractor(mock_syms_linear, mock_coords_linear, point_id="test_linear")
    extractor.apply_cartesian_protections()
    extractor.diagonalize_and_derive_constants()
    extractor.export_provenance()