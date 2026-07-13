#!/usr/bin/env python3
"""
CoChem Setup Phase 1: Core System Auditing, Hypervisor Profiling & Caching
Executes baseline OS checks, memory auditing, and network profiling.
"""

import os
import sys
import subprocess
import shutil
import logging
from logging.handlers import RotatingFileHandler
import json

# ---------------------------------------------------------
# UI & LOGGING PROTOCOLS
# ---------------------------------------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
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

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("CoChem")
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler('cochem_execution.log', maxBytes=5*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

log = setup_logging()

# ---------------------------------------------------------
# SYSTEM AUDIT FUNCTIONS
# ---------------------------------------------------------
def sanitize_environment() -> None:
    print_status("Sanitizing ghost dependencies ($LD_LIBRARY_PATH, $PYTHONPATH)...", "info")
    for var in ['LD_LIBRARY_PATH', 'PYTHONPATH']:
        if var in os.environ:
            log.warning(f"Cleared {var} from environment to prevent silo pollution (Was: {os.environ[var]}).")
            del os.environ[var]

def check_session_multiplexer() -> None:
    if not os.environ.get("TMUX") and "screen" not in os.environ.get("TERM", ""):
        print_status("No TMUX/Screen detected. An SSH drop will kill this compilation.", "warning")
        log.warning("Session multiplexer not found.")
    else:
        print_status("Session multiplexer (TMUX/Screen) active.", "success")

def check_filesystem_and_space() -> dict:
    fs_stats = {"space_gb": 0.0, "fs_type": "unknown", "safe": True}
    try:
        total, used, free = shutil.disk_usage(".")
        fs_stats["space_gb"] = free / (1024**3)
        if fs_stats["space_gb"] < 15.0:
            print_status(f"CRITICAL: Only {fs_stats['space_gb']:.1f}GB free. MLFF wheels require >15GB.", "fail")
            sys.exit(1)
        
        df_out = subprocess.check_output(["df", "-T", "."], text=True)
        if "nfs" in df_out.lower():
            print_status("NFS drive detected. ORCA scratch IO will be severely bottlenecked.", "warning")
            fs_stats["fs_type"] = "NFS"
        else:
            fs_stats["fs_type"] = "Local"
            print_status(f"Local storage verified ({fs_stats['space_gb']:.1f} GB free).", "success")
    except Exception as e:
        log.error(f"Filesystem check failed: {e}")
    return fs_stats

def detect_hypervisor() -> dict:
    sys_env = {"type": "Bare Metal", "shm_gb": 0.0}
    try:
        if os.path.exists("/.dockerenv"):
            sys_env["type"] = "Docker"
        elif "CODESPACES" in os.environ:
            sys_env["type"] = "GitHub Codespaces"
        elif os.path.exists("/proc/1/cgroup"):
            with open("/proc/1/cgroup", "r") as f:
                if "lxc" in f.read():
                    sys_env["type"] = "Proxmox/LXC"
    except: pass
    
    try:
        shm_stat = os.statvfs("/dev/shm")
        sys_env["shm_gb"] = (shm_stat.f_bsize * shm_stat.f_blocks) / (1024**3)
        if sys_env["shm_gb"] < 2.0 and sys_env["type"] != "Bare Metal":
            print_status(f"Restricted /dev/shm ({sys_env['shm_gb']:.2f}GB) in {sys_env['type']}. PyTorch multiprocessing may crash.", "warning")
        else:
            print_status(f"Hypervisor Profile: {sys_env['type']} (/dev/shm: {sys_env['shm_gb']:.2f}GB)", "success")
    except Exception as e:
        log.error(f"SHM check failed: {e}")
    return sys_env

def check_memory_and_ecc() -> dict:
    mem_stats = {"ram_gb": 0.0, "swap_gb": 0.0, "ecc": "Unknown"}
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        mem_stats["ram_gb"] = mem.total / (1024**3)
        mem_stats["swap_gb"] = swap.total / (1024**3)
        
        if mem_stats["swap_gb"] < (mem_stats["ram_gb"] * 0.25):
            print_status("Low Swap Space. Linux OOM killer is highly likely to terminate heavy jobs.", "warning")
            
        ecc_out = subprocess.run(["sudo", "-n", "dmidecode", "-t", "memory"], capture_output=True, text=True)
        if "Error Correction Type: Multi-bit ECC" in ecc_out.stdout:
            mem_stats["ecc"] = "Verified"
        elif "Error Correction Type: None" in ecc_out.stdout:
            mem_stats["ecc"] = "None"
            print_status("Non-ECC memory detected. Heavy float accumulations risk silent bit-flip corruption.", "warning")
            
        print_status(f"Memory Profile: {mem_stats['ram_gb']:.1f}GB RAM / {mem_stats['swap_gb']:.1f}GB Swap.", "success")
    except ImportError:
        print_status("psutil missing. Skipping deep memory profiling.", "info")
    return mem_stats

def test_network_latency() -> bool:
    print_status("Testing Conda/PyPI packet health...", "info")
    try:
        ping_res = subprocess.run(["ping", "-c", "4", "pypi.org"], capture_output=True, text=True)
        if "0% packet loss" in ping_res.stdout:
            print_status("Network streams are stable. No packet loss detected.", "success")
            return True
        else:
            print_status("High latency or packet loss detected. PyTorch wheel downloads may corrupt.", "warning")
            return False
    except:
        print_status("Ping command unavailable. Proceeding with caution.", "warning")
        return False

def main() -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} CoChem: Phase 1 - System Audit & Hypervisor Profiling {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}")
    log.info("--- STARTED PHASE 1 ---")
    
    sanitize_environment()
    check_session_multiplexer()
    fs_stats = check_filesystem_and_space()
    sys_env = detect_hypervisor()
    mem_stats = check_memory_and_ecc()
    net_health = test_network_latency()
    
    state = {
        "filesystem": fs_stats,
        "hypervisor": sys_env,
        "memory": mem_stats,
        "network_stable": net_health
    }
    
    with open("cochem_state_p1.json", "w") as f:
        json.dump(state, f, indent=4)
        
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🏁 Phase 1 Complete. State cached in cochem_state_p1.json.{Colors.ENDC}")

if __name__ == "__main__":
    main()