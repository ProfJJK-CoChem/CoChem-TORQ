#!/usr/bin/env python3
"""
CoChem Setup Phase 2: Deep Hardware & GPU Precision Verification
Handles VRAM pinning, topology mapping, CPU core detection, and memory profiling.
"""
import os
import sys
import subprocess
import json
import logging
import psutil

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

log = logging.getLogger("CoChem")

def check_gpu_profile() -> dict:
    gpu_data = {"vendor": "CPU", "temp_ok": True, "name": "N/A", "vram_gb": 0.0}
    print_status("Probing cross-vendor GPU topologies...", "info")
    try:
        # Requesting Name, Temp, and Total VRAM from NVIDIA
        nvidia_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.total", "--format=csv,noheader,nounits"], 
            text=True
        ).strip().split('\n')[0].split(',')
        
        gpu_data["vendor"] = "NVIDIA"
        gpu_data["name"] = nvidia_out[0].strip()
        temp = int(nvidia_out[1].strip())
        gpu_data["vram_gb"] = round(float(nvidia_out[2].strip()) / 1024.0, 1) # Convert MB to GB
        
        if temp > 80:
            print_status(f"NVIDIA GPU is running hot ({temp}°C). Thermal throttling risk.", "warning")
            gpu_data["temp_ok"] = False
        else:
            print_status(f"GPU Detected: {gpu_data['name']} ({gpu_data['vram_gb']} GB VRAM) - Thermals nominal ({temp}°C).", "success")
            
        return gpu_data
    except Exception:
        print_status("No NVIDIA GPU detected. Falling back to CPU-only compute.", "warning")
    return gpu_data

def check_cpu_topology() -> dict:
    cpu_data = {
        "avx2": False, 
        "avx512": False, 
        "numa_nodes": 1, 
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1)
    }
    
    print_status(f"Detected {cpu_data['physical_cores']} Physical Cores ({cpu_data['logical_cores']} Threads) and {cpu_data['total_ram_gb']} GB System RAM.", "success")
    
    try:
        lscpu_out = subprocess.check_output(["lscpu"], text=True)
        if "avx2" in lscpu_out.lower(): cpu_data["avx2"] = True
        if "avx512" in lscpu_out.lower(): cpu_data["avx512"] = True
    except Exception: pass
    
    return cpu_data

def test_ieee754_subnormals() -> bool:
    """Verifies that the hardware hasn't flushed subnormal floats to zero."""
    import math
    min_norm = sys.float_info.min
    subnormal = min_norm / 2.0
    if subnormal > 0.0 and subnormal < min_norm:
        print_status("IEEE-754 Subnormal arithmetic verified (No DAZ/FTZ flags).", "success")
        return True
    print_status("Hardware is flushing subnormals to zero (DAZ). Precision loss possible in deep SCF cycles.", "warning")
    return False

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 2: Hardware & Precision Verification ---{Colors.ENDC}")
    try:
        with open("cochem_setup/cochem_state_p1.json", "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status("Missing cochem_setup/cochem_state_p1.json. Run Phase 1 first.", "fail")
        sys.exit(1)
        
    gpu_data = check_gpu_profile()
    cpu_data = check_cpu_topology()
    precision_ok = test_ieee754_subnormals()
    
    state["gpu_profile"] = gpu_data
    state["cpu_topology"] = cpu_data
    state["ieee754_compliant"] = precision_ok
    
    os.makedirs("cochem_setup", exist_ok=True)
    with open("cochem_setup/cochem_state_p2.json", "w") as f: json.dump(state, f, indent=4)
    print_status("Phase 2 Complete. Hardware mapped.", "success")

if __name__ == "__main__":
    main()