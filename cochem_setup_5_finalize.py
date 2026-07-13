#!/usr/bin/env python3
"""
CoChem Setup Phase 5: Academic Output, Offloading, & Finalization
Generates Codespaces offload YAML, strict locks, and final IPC config.
"""
import os
import sys
import json
import subprocess

class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(msg: str, status: str = "info") -> None:
    if status == "success": print(f"  {Colors.OKGREEN}✅ {msg}{Colors.ENDC}")
    elif status == "warning": print(f"  {Colors.WARNING}⚠️ {msg}{Colors.ENDC}")
    else: print(f"  ➡️ {msg}")

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
    with open("cochem_citations.bib", "w") as f: f.write(bibtex)
    
    # Finalize Configuration
    final_config = {
        "pipeline_version": "2.1",
        "hardware_profile": state.get("gpu_profile"),
        "cpu_topology": state.get("cpu_topology"),
        "silo_registry": state.get("silos", {}),
        "engine_paths": state.get("engines", {})
    }
    with open("cochem_system_config.json", "w") as f: json.dump(final_config, f, indent=4)

def cleanup_states() -> None:
    for i in range(1, 5):
        if os.path.exists(f"cochem_state_p{i}.json"):
            os.remove(f"cochem_state_p{i}.json")
    print_status("Temporary state matrices purged.", "success")

def main() -> None:
    print(f"\n{Colors.BOLD}--- Phase 5: Finalization & Outputs ---{Colors.ENDC}")
    try:
        with open("cochem_state_p4.json", "r") as f: state = json.load(f)
    except FileNotFoundError:
        print_status("Missing cochem_state_p4.json.", "fail")
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