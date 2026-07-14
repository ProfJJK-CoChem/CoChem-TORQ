#!/usr/bin/env python3
"""
CoChem Master Deployment Orchestrator
Sequentially triggers the CoChem environment initialization.
Halts gracefully if any sub-phase throws a fatal error.
"""
import os
import sys
import subprocess
import time

class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"\n{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}       CoChem Pipeline: Master Initialization         {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}\n")

def run_phase(script_name: str, phase_desc: str) -> bool:
    """Executes a setup phase and monitors its return code."""
    if not os.path.exists(script_name):
        print(f"{Colors.FAIL}❌ FATAL: Cannot find {script_name} in the root directory.{Colors.ENDC}")
        return False

    print(f"{Colors.BOLD}>>> Initiating {phase_desc} ({script_name})...{Colors.ENDC}")
    time.sleep(1) # Brief pause for UI readability
    
    try:
        # Execute the phase directly without external project args
        result = subprocess.run([sys.executable, script_name], check=False)
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✅ {phase_desc} Completed Successfully.\n{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}❌ FATAL: {phase_desc} failed with exit code {result.returncode}.{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Pipeline halted. Please resolve the errors above before re-running setup.py.{Colors.ENDC}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}❌ Initialization aborted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error during {script_name} execution: {e}{Colors.ENDC}")
        return False

def main():
    print_banner()
    
    # The definitive 7-Phase execution chain
    phases = [
        ("cochem_setup_1_sys.py", "Phase 1: System Auditing"),
        ("cochem_setup_2_hw.py", "Phase 2: Hardware & Precision Mapping"),
        ("cochem_setup_3_engines.py", "Phase 3: Deep Engine Compilations"),
        ("cochem_setup_4_silos.py", "Phase 4: Dynamic Silo Orchestration"),
        ("cochem_setup_5_finalize.py", "Phase 5: Configuration & Workspace Sweep"),
        ("cochem_setup_10_intake_align.py", "Phase 10: Geometry & Symmetry Intake"),
        ("cochem_setup_11_memory_router.py", "Phase 11: Hardware Routing Wrappers")
    ]
    
    start_time = time.time()
    
    for script, desc in phases:
        success = run_phase(script, desc)
        if not success:
            sys.exit(1)
            
    elapsed = round(time.time() - start_time, 2)
    print(f"{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 CoChem FULLY DEPLOYED in {elapsed} seconds.{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}======================================================{Colors.ENDC}")
    print(f"You may now launch your Jupyter Notebooks. The master config is locked at:")
    print(f"  -> ./cochem_setup/cochem_system_config.json\n")

if __name__ == "__main__":
    # Ensure the user is running this from the correct directory
    if not os.path.exists("cochem_setup_1_sys.py"):
        print(f"{Colors.FAIL}Error: setup.py must be run from the directory containing the cochem_setup_*.py scripts.{Colors.ENDC}")
        sys.exit(1)
        
    main()