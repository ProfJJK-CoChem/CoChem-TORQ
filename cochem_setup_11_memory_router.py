#!/usr/bin/env python3
"""
CoChem Setup Phase 11: Master Engine Part 2 - Memory Router & Handlers
Implements the OpenMPI Subprocess Wrapper (fixing the Shell Trap), verifies MACE-OFF23 
capability, estimates VRAM footprints, and enacts Hardware-Aware Adaptive Tiering.
"""
import os
import sys
import json
import subprocess
import psutil
from typing import List, Dict, Optional

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

def load_cochem_registry() -> dict:
    config_path = os.path.join("cochem_setup", "cochem_system_config.json")
    try:
        with open(config_path, "r") as f: return json.load(f)
    except FileNotFoundError:
        print_status("CRITICAL: Registry missing.", "fail")
        sys.exit(1)

# ---------------------------------------------------------
# OPEN MPI SUBPROCESS WRAPPER
# ---------------------------------------------------------
def run_orca_with_mpi_wrapper(registry: dict, input_file: str, output_file: str) -> bool:
    """
    Solves the 'Subprocess Shell Trap' on Fedora/Linux. Clones OS environment, 
    injects precise OpenMPI binary/library paths, and passes it to ORCA so it doesn't 
    crash when generating parallel nodes.
    """
    orca_path = registry.get("engine_paths", {}).get("orca_path")
    mpi_path = registry.get("engine_paths", {}).get("mpi_path")
    
    if not orca_path:
        print_status("ORCA path undefined in registry.", "fail")
        return False
        
    env = os.environ.copy()
    if mpi_path:
        # Dynamically map the bin and lib paths
        mpi_bin_dir = os.path.dirname(mpi_path)
        mpi_lib_dir = os.path.join(os.path.dirname(mpi_bin_dir), "lib")
        
        # Inject directly to bypass bashrc omissions
        env["PATH"] = f"{mpi_bin_dir}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{mpi_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        
    print_status(f"Executing ORCA natively with mapped MPI environment...", "info")
    try:
        with open(output_file, "w") as out_f:
            res = subprocess.run([orca_path, input_file], env=env, stdout=out_f, stderr=subprocess.STDOUT)
        
        # Safe Checkpoint Reader
        with open(output_file, "r") as check_f:
            if "ORCA TERMINATED NORMALLY" in check_f.read():
                print_status("ORCA Terminated Normally.", "success")
                return True
            else:
                print_status("ORCA crashed or was killed. Inspect output file.", "fail")
                return False
    except Exception as e:
        print_status(f"ORCA Execution Error: {e}", "fail")
        return False

# ---------------------------------------------------------
# MACE CHECKER & MEMORY ESTIMATORS
# ---------------------------------------------------------
def verify_mace_compatibility(registry: dict) -> bool:
    """Checks the micro-silo registry for MACE-Torch acceleration compatibility."""
    mace_status = registry.get("silo_registry", {}).get("mace-torch")
    if mace_status:
        print_status(f"MACE-OFF23 Compatibility Verified (Status: {mace_status})", "success")
        return True
    print_status("MACE-Torch missing from registry. Neural Network workflows disabled.", "warning")
    return False

def calculate_theoretical_vram(num_atoms: int, method: str = "DFT") -> float:
    """
    Mathematical footprint estimator to prevent OOM.
    Returns estimated memory required in Gigabytes (GB).
    """
    if method.upper() == "MACE":
        # MACE scales approx O(N) natively, rough heuristic 25MB per atom per batch
        footprint_gb = (num_atoms * 25.0) / 1024.0
    elif method.upper() == "DFT":
        # PySCF/ORCA DFT scales O(N^3) roughly. 
        # Standard basis set heuristic (def2-TZVP scale):
        footprint_gb = (num_atoms ** 3) * 1.5e-5
    else:
        footprint_gb = 1.0
        
    return round(footprint_gb, 3)

def get_system_vram() -> float:
    """Probes nvidia-smi for total VRAM. Falls back to System RAM if CPU."""
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
        vram_mb = int(res.strip().split('\n')[0])
        return vram_mb / 1024.0
    except Exception:
        # Fallback to standard system RAM
        return psutil.virtual_memory().total / (1024.0 ** 3)

# ---------------------------------------------------------
# ADAPTIVE TIER ROUTING
# ---------------------------------------------------------
def hardware_aware_router(num_atoms: int, registry: dict) -> str:
    """
    Adaptive Tiers: Routes downstream calculations to GPU (PySCF/MACE) 
    or CPU (ORCA) based on VRAM limitations to prevent crashes.
    """
    vram_available = get_system_vram()
    dft_footprint = calculate_theoretical_vram(num_atoms, "DFT")
    mace_footprint = calculate_theoretical_vram(num_atoms, "MACE")
    
    print_status(f"System Capacity: {vram_available:.1f} GB Available.", "info")
    print_status(f"Theoretical Load for {num_atoms} Atoms: DFT={dft_footprint}GB, MACE={mace_footprint}GB.", "info")
    
    if dft_footprint < (vram_available * 0.8):
        print_status("Safeguard Check Passed. Routing to GPU (PySCF / High-Accuracy).", "success")
        return "GPU_PYSCF"
    elif mace_footprint < (vram_available * 0.8):
        print_status("DFT Exceeds Memory Threshold. Downgrading and routing to MACE-OFF23 Neural Net.", "warning")
        return "GPU_MACE"
    else:
        print_status("VRAM Threshold Exceeded. Delegating to CPU Cluster (ORCA / Disk-Backed).", "warning")
        return "CPU_ORCA"

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 11: Hardware Routing & Execution Wrappers ---{Colors.ENDC}")
    reg = load_cochem_registry()
    verify_mace_compatibility(reg)
    hardware_aware_router(num_atoms=50, registry=reg) # Diagnostic Test
    print_status("Phase 11 Module Ready for downstream imports.", "success")

if __name__ == "__main__":
    main()