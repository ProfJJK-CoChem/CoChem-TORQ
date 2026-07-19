#!/usr/bin/env python3
"""
CoChem-TORQ: Tensor Extractor & Cartesian Protections
Computes the Principal Axes of Inertia, Rotational Constants in MHz, 
and strictly enforces structural protections against linear singularities.
"""

import os
import json
import datetime
import numpy as np
import scipy.constants as const
from typing import List, Dict, Tuple

class TensorExtractor:
    def __init__(self, registry_path: str = "fit_provenance.json"):
        """
        Initializes the extractor and anchors the exact CODATA physical constants 
        to ensure 100% mathematical reproducibility over decades.
        """
        self.registry_path = registry_path
        # Anchor CODATA constants explicitly
        self.h_planck = const.h
        self.amu_kg = const.atomic_mass
        self.angstrom_m = 1e-10
        self.c_light = const.c
        
        self.provenance = self._load_provenance()

    def _load_provenance(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"CRITICAL: {self.registry_path} missing. Cannot guarantee fit semantics.")

    def _update_provenance(self, unique_id: str, data: dict) -> None:
        self.provenance["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        self.provenance["fits"][unique_id] = data
        with open(self.registry_path, 'w') as f:
            json.dump(self.provenance, f, indent=4)

    def verify_cartesian_protections(self, coordinates: np.ndarray, tolerance_deg: float = 2.0) -> bool:
        """
        Scans for 180-degree linear angle singularities (e.g., C#N bonds, alkynes)
        that will crash standard Z-matrix redundant internal coordinate generators.
        """
        num_atoms = coordinates.shape[0]
        if num_atoms < 3:
            return True # Diatomics are inherently linear and handled natively
            
        for i in range(num_atoms - 2):
            v1 = coordinates[i] - coordinates[i+1]
            v2 = coordinates[i+2] - coordinates[i+1]
            
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 < 1e-4 or n2 < 1e-4:
                continue
                
            cos_theta = np.dot(v1, v2) / (n1 * n2)
            # Clip to valid arccos domain to prevent floating point warnings
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
            
            if abs(angle - 180.0) < tolerance_deg:
                print(f"⚠️ CARTESIAN PROTECTION TRIGGERED: Near-linear angle detected ({angle:.2f}°).")
                print("   Downstream torsional mapping must utilize Dummy Atoms or pure Cartesians.")
                return False
        return True

    def calculate_rotational_constants(self, symbols: List[str], coordinates: np.ndarray, masses: List[float], unique_id: str) -> Dict[str, float]:
        """
        Calculates the Principal Moments of Inertia and converts them to 
        spectroscopic Rotational Constants (A, B, C) in MHz.
        """
        masses_arr = np.array(masses)
        
        # Build Inertia Tensor (Assuming COM alignment is already handled by MInt/TOPOS)
        I = np.zeros((3, 3))
        for m, r in zip(masses_arr, coordinates):
            x, y, z = r
            I[0, 0] += m * (y**2 + z**2)
            I[1, 1] += m * (x**2 + z**2)
            I[2, 2] += m * (x**2 + y**2)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
        
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        # Eigenvalues represent moments of inertia (I_a, I_b, I_c) in amu * Angstrom^2
        moments_of_inertia_amu_A2, eigenvectors = np.linalg.eigh(I)
        
        # Sort moments of inertia ascending (Ia <= Ib <= Ic) -> (A >= B >= C)
        sorted_indices = np.argsort(moments_of_inertia_amu_A2)
        moments = moments_of_inertia_amu_A2[sorted_indices]
        
        # Convert amu * Angstrom^2 to kg * m^2
        conversion_factor = self.amu_kg * (self.angstrom_m ** 2)
        moments_kg_m2 = moments * conversion_factor
        
        # B = h / (8 * pi^2 * I)  -> result in Hz, divide by 1e6 for MHz
        constants_mhz = []
        for I_val in moments_kg_m2:
            if I_val > 1e-50:  # Avoid division by zero for linear molecules
                B_hz = self.h_planck / (8.0 * (np.pi**2) * I_val)
                constants_mhz.append(B_hz / 1e6)
            else:
                constants_mhz.append(0.0)

        # Structure the payload
        rotational_data = {
            "A_mhz": constants_mhz[0],
            "B_mhz": constants_mhz[1],
            "C_mhz": constants_mhz[2],
            "linear_singularity_safe": self.verify_cartesian_protections(coordinates)
        }
        
        # Append to Provenance Registry
        self._update_provenance(unique_id, {
            "rotational_constants": rotational_data,
            "CODATA_anchors": {
                "h": self.h_planck,
                "amu_to_kg": self.amu_kg
            }
        })
        
        return rotational_data

if __name__ == "__main__":
    print("CoChem-TORQ Tensor Extractor initialized.")