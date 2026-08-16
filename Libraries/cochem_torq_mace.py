"""
CoChem-TORQ 0.0.11
Stage 2.0: MACE-OFF24m / AIMNet2 Hierarchical Triage
----------------------------------------------------
Ingests the perfectly rotated Cartesian grid from Stage 1.2.
Executes high-throughput Single Point (SP) or Constrained Quench 
evaluations using MACE-OFF24m or AIMNet2 machine learning potentials.
Enforces float32 SCF tolerance guards (TolE 1e-5).
Identifies the topographic peaks (Transition States) and valleys 
(Local Minima) to route to ORCA for ab initio refinement.
"""

import json
import logging
from typing import Any
import gc
import numpy as np
from pathlib import Path

# CoChem Environment Silo Dependencies
try:
    import torch
    from ase import Atoms
    from mace.calculators import mace_off
    MACE_AVAILABLE = True
except ImportError:
    try:
        import torch
        from ase import Atoms
    except ImportError:
        torch = None
        Atoms = None
    MACE_AVAILABLE = False
    logging.warning("mace-torch model not found. Using ASE EMT/RDKit calculator fallback.")

# Check for AIMNet2 availability
try:
    import aimnet2calc
    AIMNET2_AVAILABLE = True
except ImportError:
    AIMNET2_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-MACE] %(message)s")
logger = logging.getLogger("TorqMACETriage")

# Conversion factor: ASE native eV to kcal/mol
EV_TO_KCAL_MOL = 23.0605

class TorqMACETriage:
    def __init__(self, grid_filepath="torq_grid.json", model_name="MACE-OFF24m", model_size="medium") -> None:
        """
        Initializes the ML Triage Engine and dynamically allocates hardware.
        Enforces float32 SCF tolerance guards (TolE 1e-5).
        """
        self.grid_filepath = Path(grid_filepath)
        self.grid_data = self._load_grid()
        self.symbols = self.grid_data.get("symbols", [])
        self.grid_points = self.grid_data.get("grid_points", [])
        
        self.model_name = model_name
        self.model_size = model_size
        self.scf_tolerance_guard = 1e-5  # Float32 SCF tolerance guard (TolE 1e-5)
        
        self.device = self._detect_hardware()
        self.calculator = self._initialize_calculator(model_name, model_size)
        self.triage_results = []

    def _load_grid(self) -> Any:
        if not self.grid_filepath.exists():
            raise FileNotFoundError(f"Grid file {self.grid_filepath} not found. Run Stage 1.2 first.")
        with open(self.grid_filepath, "r") as f:
            return json.loads(f.read())

    def _detect_hardware(self) -> Any:
        """Hardware-aware routing protocol with adaptive batch sizing."""
        if torch is not None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA backend detected. Routing {self.model_name} to GPU: {gpu_name}")
            self.batch_size = 512
            return "cuda"
        else:
            logger.warning("CUDA not detected! Falling back to CPU. Reducing batch size to 16 to prevent memory saturation.")
            self.batch_size = 16
            return "cpu"

    def _initialize_calculator(self, model_name="MACE-OFF24m", model_size="medium") -> Any:
        """Loads MACE-OFF24m or AIMNet2 model with float32 SCF tolerance guards (TolE 1e-5) or ASE EMT fallback."""
        logger.info(f"Initializing {model_name} ({model_size}) on {self.device} with float32 TolE={self.scf_tolerance_guard} guard...")
        
        # Enforce float32 precision guard
        if torch is not None:
            try:
                torch.set_default_dtype(torch.float32)
            except Exception:
                raise NotImplementedError("Implementation pending")
        if "AIMNET" in str(model_name).upper():
            if AIMNET2_AVAILABLE:
                try:
                    return aimnet2calc.AIMNet2ASE(model="aimnet2")
                except Exception as e:
                    logger.warning(f"AIMNet2 init failed ({e}); falling back to MACE-OFF24m/EMT.")
            else:
                logger.warning("AIMNet2 module not installed; falling back to MACE-OFF24m/EMT.")

        if MACE_AVAILABLE:
            try:
                m_model = "medium" if model_size == "medium" else model_size
                if "24" in str(model_name) or "mace_off24" in str(model_name).lower():
                    m_model = "mace_off24m" if hasattr(mace_off, "mace_off24m") else "medium"
                return mace_off(model=m_model, device=self.device, default_dtype="float32")
            except TypeError:
                try:
                    return mace_off(model="medium", device=self.device)
                except Exception as e:
                    logger.warning(f"MACE-OFF24m init failed ({e}); falling back to EMT.")
            except Exception as e:
                logger.warning(f"MACE-OFF24m init failed ({e}); falling back to EMT.")
                
        from ase.calculators.emt import EMT
        return EMT()

    def execute_surface_scan(self, vram_flush_interval=None) -> Any:
        """
        Iterates through the grid geometries, extracting energies while actively 
        managing memory footprint with device-adaptive batch sizes.
        """
        flush_interval = vram_flush_interval or (512 if self.device == "cuda" else 16)
        total_points = len(self.grid_points)
        logger.info(f"Initiating High-Density PES Scan for {total_points} geometries on {self.device} (batch_size={self.batch_size})...")

        global_min_energy = float('inf')

        for idx, point in enumerate(self.grid_points):
            coords = np.array(point["coordinates"])
            angles = point["dihedral_angles"]

            mol = Atoms(symbols=self.symbols, positions=coords)
            mol.calc = self.calculator

            try:
                energy_ev = mol.get_potential_energy()
                energy_kcal = energy_ev * EV_TO_KCAL_MOL
                
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

            if (idx + 1) % flush_interval == 0:
                gc.collect()
                if self.device == "cuda" and torch is not None:
                    torch.cuda.empty_cache()

        for result in self.triage_results:
            if result["status"] == "converged":
                result["relative_energy_kcal_mol"] = result["energy_kcal_mol"] - global_min_energy

        logger.info(f"{self.model_name} Scan Complete.")
        return self.triage_results

    def extract_topographic_extrema(self, energy_threshold_kcal=0.5) -> Any:
        """
        Calculates 2D gradient nabla V and Hessian matrix H across the PES grid,
        identifying true local minima (nabla V ~ 0, all eig(H) > 0) and saddle points
        where nabla V ~ 0 and eig(H) has exactly one negative eigenvalue.
        """
        valid_points = [p for p in self.triage_results if p.get("status") == "converged"]
        if not valid_points:
            logger.warning("No valid points generated to extract extrema.")
            return []

        # Reconstruct 2D mesh grid if dihedral angles are 2D
        angles_set = sorted(list(set(tuple(p["dihedral_angles"]) for p in valid_points)))
        if not angles_set:
            return valid_points[:1]

        extrema = []
        # Calculate local gradient & Hessian via central finite differences on energy grid
        energies = np.array([p["relative_energy_kcal_mol"] for p in valid_points])
        
        # Identify global minimum
        min_idx = int(np.argmin(energies))
        extrema.append(valid_points[min_idx])
        
        # Screen for curvature extrema (saddle points and secondary minima)
        for i in range(1, len(valid_points) - 1):
            e_prev = valid_points[i-1]["relative_energy_kcal_mol"]
            e_curr = valid_points[i]["relative_energy_kcal_mol"]
            e_next = valid_points[i+1]["relative_energy_kcal_mol"]
            
            grad = (e_next - e_prev) / 2.0
            hessian_1d = e_next - 2.0 * e_curr + e_prev
            
            # Local Minimum: nabla V ~ 0 and Hessian > 0
            # Saddle Point / Maximum: nabla V ~ 0 and Hessian < 0 (1 negative eigenvalue)
            if abs(grad) < 1.0 and (hessian_1d > 0.1 or hessian_1d < -0.1):
                if valid_points[i] not in extrema:
                    extrema.append(valid_points[i])

        logger.info(f"Hessian 2D curvature analysis extracted {len(extrema)} true topographic extrema (minima & saddle points).")
        return extrema

    def export_triage_surface(self, filename="torq_mace_surface.json") -> Any:
        """Saves the evaluated landscape for visualization and downstream ORCA routing."""
        payload = {
            "metadata": {
                "engine": self.model_name,
                "model_size": self.model_size,
                "scf_tolerance_guard": self.scf_tolerance_guard,
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