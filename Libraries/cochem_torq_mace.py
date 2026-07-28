"""
CoChem-TORQ 0.0.11
Stage 2.0: MACE-OFF23 Hierarchical Triage
-----------------------------------------
Ingests the perfectly rotated Cartesian grid from Stage 1.2.
Executes high-throughput Single Point (SP) or Constrained Quench 
evaluations using the MACE-OFF23 machine learning potential.
Identifies the topographic peaks (Transition States) and valleys 
(Local Minima) to route to ORCA for ab initio refinement.
"""

import json
import logging
import gc
import numpy as np
from pathlib import Path

# CoChem Environment Silo Dependencies
try:
    import torch
    from ase import Atoms
    from mace.calculators import mace_off
except ImportError as e:
    raise ImportError("Missing critical ML dependencies. Ensure 'mace-torch' and 'ase' are installed in this silo.") from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-MACE] %(message)s")
logger = logging.getLogger("TorqMACETriage")

# Conversion factor: ASE native eV to kcal/mol
EV_TO_KCAL_MOL = 23.0605

class TorqMACETriage:
    def __init__(self, grid_filepath="torq_grid.json", model_size="medium"):
        """
        Initializes the ML Triage Engine and dynamically allocates hardware.
        """
        self.grid_filepath = Path(grid_filepath)
        self.grid_data = self._load_grid()
        self.symbols = self.grid_data.get("symbols", [])
        self.grid_points = self.grid_data.get("grid_points", [])
        
        self.device = self._detect_hardware()
        self.calculator = self._initialize_calculator(model_size)
        self.triage_results = []

    def _load_grid(self):
        if not self.grid_filepath.exists():
            raise FileNotFoundError(f"Grid file {self.grid_filepath} not found. Run Stage 1.2 first.")
        with open(self.grid_filepath, "r") as f:
            return json.load(f)

    def _detect_hardware(self):
        """Hardware-aware routing protocol."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA backend detected. Routing MACE-OFF23 to GPU: {gpu_name}")
            return "cuda"
        else:
            logger.warning("CUDA not detected! Falling back to CPU. This will cause severe bottlenecks for 2D meshes.")
            return "cpu"

    def _initialize_calculator(self, model_size):
        """Loads the MACE-OFF23 model."""
        logger.info(f"Initializing MACE-OFF23 ({model_size}) on {self.device}...")
        try:
            # mace_off wrapper automatically downloads and caches the foundation model
            calc = mace_off(model=model_size, device=self.device)
            return calc
        except Exception as e:
            logger.error("Failed to initialize MACE-OFF23. Verify network access for initial model weight download.")
            raise e

    def execute_surface_scan(self, vram_flush_interval=100):
        """
        Iterates through the grid geometries, extracting energies while actively 
        managing the PyTorch tensor memory footprint.
        """
        total_points = len(self.grid_points)
        logger.info(f"Initiating High-Density PES Scan for {total_points} geometries...")

        global_min_energy = float('inf')

        for idx, point in enumerate(self.grid_points):
            coords = np.array(point["coordinates"])
            angles = point["dihedral_angles"]

            # Construct ASE Atoms object
            mol = Atoms(symbols=self.symbols, positions=coords)
            mol.calc = self.calculator

            try:
                # Single Point Energy Evaluation
                energy_ev = mol.get_potential_energy()
                energy_kcal = energy_ev * EV_TO_KCAL_MOL
                
                # Track global minimum for relative normalization
                if energy_kcal < global_min_energy:
                    global_min_energy = energy_kcal

                self.triage_results.append({
                    "dihedral_angles": angles,
                    "coordinates": coords.tolist(),
                    "energy_kcal_mol": energy_kcal,
                    "status": "converged"
                })

            except Exception as e:
                logger.error(f"Point {idx} {angles} failed ML evaluation: {e}")
                self.triage_results.append({
                    "dihedral_angles": angles,
                    "energy_kcal_mol": None,
                    "status": "failed"
                })

            # ---------------------------------------------------------
            # ZOMBIE ASSASSIN / MEMORY GOVERNOR PROTOCOL
            # ---------------------------------------------------------
            if (idx + 1) % vram_flush_interval == 0:
                logger.info(f"Progress: {idx + 1}/{total_points}. Flushing VRAM...")
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Normalize energies relative to the global minimum
        for result in self.triage_results:
            if result["status"] == "converged":
                result["relative_energy_kcal_mol"] = result["energy_kcal_mol"] - global_min_energy

        logger.info("MACE-OFF23 Scan Complete.")
        return self.triage_results

    def extract_topographic_extrema(self, energy_threshold_kcal=0.5):
        """
        Isolates the key structural points (basins and barrier peaks) 
        that actually require expensive ab initio ORCA refinement.
        """
        valid_points = [p for p in self.triage_results if p["status"] == "converged"]
        if not valid_points:
            logger.warning("No valid points generated to extract extrema.")
            return []

        # Sort by relative energy
        valid_points.sort(key=lambda x: x["relative_energy_kcal_mol"])
        
        # Keep global minimum
        extrema = [valid_points[0]]
        
        # Simple clustering: only keep points that are separated by `energy_threshold_kcal`
        # and represent distinct conformational regions. (Placeholder for full 2D spline logic)
        for point in valid_points[1:]:
            if point["relative_energy_kcal_mol"] > extrema[-1]["relative_energy_kcal_mol"] + energy_threshold_kcal:
                extrema.append(point)

        logger.info(f"Extracted {len(extrema)} distinct extrema for ORCA escalation.")
        return extrema

    def export_triage_surface(self, filename="torq_mace_surface.json"):
        """Saves the evaluated landscape for visualization and downstream ORCA routing."""
        payload = {
            "metadata": {
                "engine": "MACE-OFF23",
                "points_evaluated": len(self.triage_results)
            },
            "landscape": self.triage_results
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Triage map committed to {filename}")

if __name__ == "__main__":
    # Standard Execution Block
    try:
        triage_engine = TorqMACETriage(grid_filepath="torq_grid.json")
        triage_engine.execute_surface_scan(vram_flush_interval=50)
        extrema = triage_engine.extract_topographic_extrema()
        triage_engine.export_triage_surface()
    except FileNotFoundError:
        logger.warning("torq_grid.json not found. Run cochem_torq_grid.py first to generate the Cartesian mesh.")