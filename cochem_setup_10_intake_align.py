#!/usr/bin/env python3
"""
CoChem Setup Phase 10: Master Engine Part 1 - Intake & Alignment
Implements geometry folder generation, file ingestion, Center of Mass (COM) translation, 
Principal Axis (Eckart Frame) alignment, and Point Group detection via MolSym.
"""
import os
import sys
import json
import glob
import numpy as np
from typing import List, Dict, Tuple, Any

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success": print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning": print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail": print(f"  {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else: print(f"  {Colors.OKCYAN}➡️ {msg}{Colors.ENDC}")

# ---------------------------------------------------------
# REGISTRY INJECTION
# ---------------------------------------------------------
def load_cochem_registry() -> dict:
    config_path = os.path.join("cochem_setup", "cochem_system_config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print_status(f"CRITICAL: Registry {config_path} missing. Run Phase 5.", "fail")
        sys.exit(1)

def mount_molsym(registry: dict) -> bool:
    """Dynamically injects the MolSym silo path into the kernel."""
    molsym_path = registry.get("silo_registry", {}).get("molsym_path")
    if molsym_path and os.path.exists(molsym_path):
        if molsym_path not in sys.path:
            sys.path.append(molsym_path)
        print_status("MolSym library dynamically mounted from registry.", "success")
        return True
    print_status("MolSym path missing from registry. Symmetry detection disabled.", "warning")
    return False

# ---------------------------------------------------------
# INTAKE & INGESTION
# ---------------------------------------------------------
def initialize_geometry_folder() -> List[str]:
    """Generates the intake folder and returns detected input files."""
    intake_dir = "01_Input_Geometries"
    os.makedirs(intake_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(intake_dir, "*.xyz")) + glob.glob(os.path.join(intake_dir, "*.out"))
    if not files:
        print_status(f"Folder '{intake_dir}' initialized. Awaiting geometries.", "info")
    else:
        print_status(f"Ingested {len(files)} geometries from {intake_dir}.", "success")
    return files

# ---------------------------------------------------------
# MATHEMATICAL ALIGNMENT (ECKART FRAME)
# ---------------------------------------------------------
def align_principal_axes(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Translates molecule to Center of Mass (COM) and aligns to Principal Axes of Inertia
    to separate internal rotation kinetic energy from overall tumbling.
    """
    # 1. Translate to Center of Mass
    com = np.average(positions, axis=0, weights=masses)
    centered_positions = positions - com
    
    # 2. Construct Inertia Tensor
    I = np.zeros((3, 3))
    for i in range(len(masses)):
        x, y, z = centered_positions[i]
        m = masses[i]
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z
        
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    
    # 3. Diagonalize Inertia Tensor to find Principal Axes
    evals, evecs = np.linalg.eigh(I)
    
    # 4. Rotate geometry into the Eckart Frame
    aligned_positions = np.dot(centered_positions, evecs)
    return aligned_positions

# ---------------------------------------------------------
# SYMMETRY DETECTION
# ---------------------------------------------------------
def detect_point_group(elements: List[str], positions: np.ndarray) -> str:
    """Utilizes MolSym to detect rotational symmetry numbers / point groups."""
    try:
        import molsym
        # Convert to MolSym molecule object
        # Note: Syntax dependent on specific MolSym implementation API
        from molsym import Molecule
        mol = Molecule(elements, positions)
        pg = mol.find_point_group()
        return pg.symbol
    except ImportError:
        return "C1 (Default/Fallback)"
    except Exception as e:
        print_status(f"MolSym detection failed: {e}", "warning")
        return "C1"

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 10: Geometry Intake & Mathematical Alignment ---{Colors.ENDC}")
    reg = load_cochem_registry()
    mount_molsym(reg)
    initialize_geometry_folder()
    print_status("Phase 10 Module Ready for downstream imports.", "success")

if __name__ == "__main__":
    main()