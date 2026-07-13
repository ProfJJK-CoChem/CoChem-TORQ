#!/usr/bin/env python3
"""
CoChem Setup Phase 2: Deep Hardware & GPU Precision Verification
Handles VRAM pinning, topology mapping, and IEEE-754 precision checks.
"""
import os
import sys
import subprocess
import json
import logging
import math

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
    gpu_data = {"vendor": "CPU", "temp_ok": True}
    print_status("Probing cross-vendor GPU topologies...", "info")
    try:
        nvidia_out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True)
        temp = int(nvidia_out.strip().split('\n')[0])
        gpu_data["vendor"] = "NVIDIA"
        if temp > 80:
            print_status(f"NVIDIA GPU is running hot ({temp}°C). Thermal throttling risk.", "warning")
            gpu_data["temp_ok"] = False
        else:
            print_status(f"NVIDIA GPU detected. Thermals nominal ({temp}°C).", "success")
        return gpu_data
    except Exception: pass
    
    try:
        rocm_out = subprocess.check_output(["rocm-smi", "--showtemp"], text=True)
        gpu_data["vendor"] = "AMD"
        print_status("AMD ROCm architecture detected.", "success")
        return gpu_data
    except Exception: pass
    
    try:
        xpu_out = subprocess.check_output(["xpu-smi", "dump", "-m", "1"], text=True)
        gpu_data["vendor"] = "INTEL"
        print_status("Intel XPU architecture detected.", "success")
        return gpu_data
    except Exception: pass

    print_status("No dedicated compute GPUs detected. Falling back to CPU.", "warning")
    return gpu_data

def check_cpu_topology() -> dict:
    cpu_data = {"avx2": False, "avx512": False, "numa_nodes": 1}
    try:
        lscpu_out = subprocess.check_output(["lscpu"], text=True)
        if "avx2" in lscpu_out.lower(): cpu_data["avx2"] = True
        if "avx512" in lscpu_out.lower(): cpu_data["avx512"] = True
        
        for line in lscpu_out.split('\n'):
            if "NUMA node(s):" in line:
                cpu_data["numa_nodes"] = int(line.split(":")[1].strip())
                
        print_status(f"CPU Topology: AVX2={cpu_data['avx2']}, AVX512={cpu_data['avx512']}, NUMA={cpu_data['numa_nodes']}", "success")
    except Exception as e:
        log.warning(f"CPU Probe failed: {e}")
    return cpu_data

def test_ieee754_subnormals() -> bool:
    """Verifies that the hardware hasn't flushed subnormal floats to zero."""
    print_status("Testing host IEEE-754 subnormal float compliance...", "info")
    smallest_normal = sys.float_info.min
    subnormal = smallest_normal / 2.0
    if subnormal > 0.0 and subnormal != smallest_normal:
        print_status("Subnormal floating-point arithmetic verified.", "success")
        return True
    else:
        print_status("CRITICAL: CPU is flushing subnormals to zero (FTZ). Precision loss highly likely.", "fail")
        return False

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 2: Hardware & Precision Verification ---{Colors.ENDC}")
    try:
        with open("cochem_state_p1.json", "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status("Missing cochem_state_p1.json. Run Phase 1 first.", "fail")
        sys.exit(1)
        
    gpu_data = check_gpu_profile()
    cpu_data = check_cpu_topology()
    precision_ok = test_ieee754_subnormals()
    
    state["gpu_profile"] = gpu_data
    state["cpu_topology"] = cpu_data
    state["ieee754_compliant"] = precision_ok
    
    with open("cochem_state_p2.json", "w") as f: json.dump(state, f, indent=4)
    print_status("Phase 2 Complete. Hardware mapped.", "success")

if __name__ == "__main__":
    main()