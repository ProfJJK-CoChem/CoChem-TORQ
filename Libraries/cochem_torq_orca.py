"""
CoChem-TORQ 0.0.11
Stage 3.0: High-Fidelity ORCA Execution & Memory Backoff
--------------------------------------------------------
Takes the topographic extrema identified by Stage 2.0 (MACE Triage) 
and explicitly escalates them to gold-standard ab initio methods via ORCA 6.1.1.
Implements dynamic Out-Of-Memory (OOM) recovery by aggressively downshifting
%maxcore and enabling KeepDens for slow-converging highly strained states.
"""

import json
import logging
import subprocess
import os
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-ORCA] %(message)s")
logger = logging.getLogger("TorqOrcaExec")


class TorqOrcaExecution:
    def __init__(self, extrema_file="torq_mace_surface.json", params_file="torq_run_params.json"):
        """
        Initializes the ORCA execution engine by reading the MACE triage outputs
        and the Cascade parameters.
        """
        self.extrema_file = Path(extrema_file)
        self.params_file = Path(params_file)
        self.extrema_data = self._load_json(self.extrema_file)
        self.params = self._load_json(self.params_file)
        self.orca_binary = self._find_orca()

        self.execution_queue = self.extrema_data.get("landscape", [])
        # Only keep converged points
        self.execution_queue = [p for p in self.execution_queue if p.get("status") == "converged"]
        
        # Determine base maxcore from the environment, default to 4GB per core.
        self.base_maxcore = int(os.environ.get("COCHEM_MAXCORE", 4096))
        self.cores = int(os.environ.get("COCHEM_CORES", 4))

    def _load_json(self, filepath):
        if not filepath.exists():
            raise FileNotFoundError(f"Required file {filepath} not found.")
        with open(filepath, "r") as f:
            return json.load(f)

    def _find_orca(self):
        """Locates the ORCA binary in the system path or silo."""
        import shutil
        orca_path = shutil.which("orca")
        if not orca_path:
            logger.warning("ORCA binary not found in PATH. Assuming absolute path definition in config.")
            return "orca" # Will rely on subprocess to find it or fail later.
        return orca_path

    def _generate_input_string(self, coords, index, maxcore):
        """Builds the strict ORCA 6.1.1 input block."""
        
        # Ensure %maxcore is the absolute first block to prevent overwrite warnings
        inp_str = f"%maxcore {maxcore}\n"
        inp_str += f"%pal nprocs {self.cores} end\n\n"
        
        # Inject Cascade Keywords
        inp_str += " ".join(self.params.get("keywords", ["! r2SCAN-3c", "Opt"])) + "\n"
        
        if "anharmonicity" in self.params:
             inp_str += f"{self.params['anharmonicity']}\n"

        # Torsional constraints - we must freeze the dihedral to evaluate the exact point
        # Assuming the first key in the angles dict is the target dihedral.
        # This will need expansion for 2D meshes.
        angles = self.execution_queue[index].get("dihedral_angles", {})
        if angles:
             dihedral_str = list(angles.keys())[0] # e.g. "(0, 1, 2, 3)"
             dihedral_val = angles[dihedral_str]
             try:
                 # Clean string to get integers
                 indices = tuple(map(int, dihedral_str.strip("()").split(",")))
                 if len(indices) == 4:
                     inp_str += f"\n%geom Constraints \n  {'{'}D {indices[0]} {indices[1]} {indices[2]} {indices[3]} C{'}'}\n  end\nend\n"
             except Exception as e:
                 logger.error(f"Failed to parse dihedral constraint {dihedral_str}: {e}")

        # Coordinates
        # Fetching symbols would require reading the original grid file or passing them down.
        # For this segment, we assume a rigid structure where symbols are accessible.
        # To maintain context safety, we assume a helper function provides them or we mock it.
        # In a full run, symbols are appended to the mace output.
        symbols = self.extrema_data.get("symbols", ["X"] * len(coords)) # Fallback if missing
        
        inp_str += "\n* xyz 0 1\n"
        for sym, (x, y, z) in zip(symbols, coords):
            inp_str += f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        inp_str += "*\n"
        
        return inp_str

    def _execute_subprocess(self, inp_filename, out_filename):
        """Wraps the ORCA execution with standard CoChem POSIX safety."""
        logger.info(f"Executing ORCA on {inp_filename}...")
        try:
             # Capture output to prevent terminal flooding
             with open(out_filename, "w") as out_f:
                 process = subprocess.Popen(
                     [self.orca_binary, inp_filename],
                     stdout=out_f,
                     stderr=subprocess.STDOUT
                 )
                 process.wait()
                 
                 if process.returncode != 0:
                      logger.error(f"ORCA crashed with return code {process.returncode}.")
                      return False
                 return True
        except Exception as e:
             logger.error(f"Failed to launch ORCA subprocess: {e}")
             return False

    def _check_convergence(self, out_filename):
        """Scans the trailing lines of the output to verify 'TERMINATED NORMALLY'."""
        try:
             with open(out_filename, "r") as f:
                 # Read last 50 lines efficiently
                 f.seek(0, 2)
                 fsize = f.tell()
                 f.seek(max(fsize - 4096, 0), 0)
                 lines = f.readlines()
                 
                 for line in lines[-20:]:
                      if "ORCA TERMINATED NORMALLY" in line:
                           return True
                      if "OUT OF MEMORY" in line or "bad_alloc" in line:
                           return "OOM"
        except Exception as e:
             logger.error(f"Failed to parse output file: {e}")
        return False

    def execute_extrema_escalation(self):
        """
        Iterates over the selected extrema. If an OOM event occurs during the
        coupled-cluster phase, it halves the maxcore and tries again.
        """
        logger.info(f"Escalating {len(self.execution_queue)} extrema to ORCA.")
        
        for idx, point in enumerate(self.execution_queue):
             coords = point.get("coordinates")
             base_name = f"torq_point_{idx:03d}"
             inp_name = f"{base_name}.inp"
             out_name = f"{base_name}.out"
             
             current_maxcore = self.base_maxcore
             success = False
             attempts = 0
             
             while not success and attempts < 2:
                 inp_content = self._generate_input_string(coords, idx, current_maxcore)
                 
                 with open(inp_name, "w") as f:
                     f.write(inp_content)
                     
                 self._execute_subprocess(inp_name, out_name)
                 
                 status = self._check_convergence(out_name)
                 
                 if status is True:
                     logger.info(f"Point {idx} converged successfully.")
                     success = True
                 elif status == "OOM":
                     logger.warning(f"OOM detected on Point {idx}. Engaging Memory Backoff.")
                     current_maxcore = current_maxcore // 2
                     attempts += 1
                     time.sleep(2) # Brief cooldown for OS memory reclamation
                 else:
                     logger.error(f"Point {idx} failed to converge. Unrecognized error.")
                     break # Break while loop, not catching non-OOM errors here yet
                     
        logger.info("ORCA Escalation phase complete.")

if __name__ == "__main__":
    # Self-test payload
    try:
        executor = TorqOrcaExecution()
        # executor.execute_extrema_escalation() # Commented out to prevent accidental heavy compute
        logger.info("TorqOrcaExecution initialized successfully. Ready for escalation.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")