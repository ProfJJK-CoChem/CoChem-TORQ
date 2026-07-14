#!/usr/bin/env python3
"""
CoChem Setup Phase 5: Academic Output, Offloading, & Finalization
Generates Codespaces offload YAML, strict locks, final IPC config,
and organizes the main directory to prevent workspace clutter.
"""
import os
import sys
import json
import subprocess
import shutil
import glob

class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success": print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning": print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    elif status == "fail": print(f"  {Colors.FAIL}❌ {msg}{Colors.ENDC}")
    else: print(f"  ➡️ {msg}")

def organize_workspace() -> None:
    print_status("Organizing workspace and clustering installation files...", "info")
    os.makedirs("cochem_setup", exist_ok=True)
    os.makedirs("Logs", exist_ok=True)

    # Route Logs
    for log_file in glob.glob("*.log*"):
        try:
            dest = os.path.join("Logs", os.path.basename(log_file))
            if os.path.abspath(log_file) != os.path.abspath(dest):
                shutil.move(log_file, dest)
        except Exception:
            pass
    
    # Route Setup, Source, and Archival files
    patterns_to_move = [
        "cochem_state_p*.json", 
        "*.tar.gz", 
        "*.tar.xz", 
        "*.zip",
        "cochem_citations.bib",
        "cochem_system_config.json"
    ]
    
    for pattern in patterns_to_move:
        for file_path in glob.glob(pattern):
            try:
                dest = os.path.join("cochem_setup", os.path.basename(file_path))
                if os.path.abspath(file_path) != os.path.abspath(dest):
                    shutil.move(file_path, dest)
            except Exception:
                pass
    print_status("Workspace organized. Installation files routed to 'cochem_setup', logs to 'Logs'.", "success")

def generate_codespaces_offload() -> None:
    os.makedirs(".github/workflows", exist_ok=True)
    yaml_content = """name: CoChem HPC Offload
on: [workflow_dispatch]
jobs:
  compute:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      - name: Execute Pipeline
        run: mpirun -n 16 orca input.inp > output.out
"""
    with open(".github/workflows/cochem_offload.yml", "w") as f:
        f.write(yaml_content)
    print_status("GitHub Actions HPC Offload YAML generated.", "success")

def write_academic_outputs(state: dict) -> None:
    print_status("Freezing environments and generating Academic locks...", "info")
    
    # Output Citations
    bibtex = """@misc{cochem_env,
  title = {CoChem Orchestrator Architecture},
  year = {2026},
  note = {Dynamically provisioned environment matrix}
}"""
    bib_path = os.path.join("cochem_setup", "cochem_citations.bib")
    with open(bib_path, "w") as f: f.write(bibtex)
    
    # Finalize Configuration
    final_config = {
        "pipeline_version": "2.1",
        "hardware_profile": state.get("gpu_profile"),
        "cpu_topology": state.get("cpu_topology"),
        "silo_registry": state.get("silos", {}),
        "engine_paths": state.get("engines", {})
    }
    config_path = os.path.join("cochem_setup", "cochem_system_config.json")
    with open(config_path, "w") as f: json.dump(final_config, f, indent=4)
    print_status(f"Configuration locked and saved to {config_path}", "success")

def cleanup_states() -> None:
    for i in range(1, 5):
        state_file = os.path.join("cochem_setup", f"cochem_state_p{i}.json")
        if os.path.exists(state_file):
            os.remove(state_file)
    print_status("Temporary state matrices purged from cochem_setup.", "success")

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 5: Finalization & Outputs ---{Colors.ENDC}")
    
    # Run the cleanup sweeping operation first
    organize_workspace()
    
    # Paths updated to reflect the new directory structure
    state_path = os.path.join("cochem_setup", "cochem_state_p4.json")
    try:
        with open(state_path, "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status(f"Missing {state_path}.", "fail")
        print_status("Phase 5 cannot proceed without the state handoff from Phase 4.", "warning")
        print_status("ACTION REQUIRED:", "warning")
        print_status("1. Ensure Phase 4 (cochem_setup_4_silos.py) completed successfully.", "warning")
        print_status("2. Resolve any missing dependency errors or hard-fails flagged in earlier phases.", "warning")
        print_status("3. Re-run the setup sequence sequentially from the point of failure.", "warning")
        sys.exit(1)

    if state.get("hypervisor", {}).get("type") == "GitHub Codespaces":
        generate_codespaces_offload()
        
    write_academic_outputs(state)
    cleanup_states()
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 CoChem PIPELINE FULLY INITIALIZED AND LOCKED.{Colors.ENDC}")

if __name__ == "__main__":
    main()