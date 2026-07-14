# Append this to CoChem-MInt_Engine.py

import os
import json
import time
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print_status("CRITICAL: 'matplotlib' is missing. GUI rendering disabled.", "fail")
    plt = None

# Attempt to load MolSym (assuming Stage 0.0 sys.path injection was successful)
try:
    import molsym
    MOLSYM_AVAILABLE = True
except ImportError:
    MOLSYM_AVAILABLE = False
    print_status("MolSym not found in path. Defaulting to C1 symmetry.", "warning")

# ---------------------------------------------------------
# STAGE 4.0: SYMMETRY & GUI
# ---------------------------------------------------------
class SymmetryAndUI:
    def __init__(self, atoms, filepath: str):
        self.atoms = atoms
        self.filepath = filepath
        self.basename = os.path.basename(filepath)
        self.point_group = "C1"

    def detect_symmetry(self) -> str:
        """Invokes MolSym to calculate the rigorous point group."""
        if not MOLSYM_AVAILABLE:
            return "C1"
        
        try:
            # Convert ASE atoms to MolSym format (requires symbols and coords)
            symbols = self.atoms.get_chemical_symbols()
            coords = self.atoms.positions
            
            # Create a basic string representation compatible with MolSym ingestion
            xyz_string = f"{len(symbols)}\n\n"
            for s, c in zip(symbols, coords):
                xyz_string += f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
                
            mol = molsym.Molecule.from_string(xyz_string)
            self.point_group = mol.find_point_group().symbol
            print_status(f"MolSym detected Point Group: {self.point_group} for {self.basename}", "success")
        except Exception as e:
            print_status(f"MolSym evaluation failed for {self.basename}: {e}. Defaulting to C1.", "warning")
            self.point_group = "C1"
            
        return self.point_group

    def render_geometry(self):
        """Displays a 3D scatter plot of the renumbered molecule."""
        if plt is None:
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        coords = self.atoms.positions
        symbols = self.atoms.get_chemical_symbols()
        
        # Plot atoms
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=200, c='cyan', edgecolors='k')
        
        # Annotate with new canonical indices
        for i, (sym, coord) in enumerate(zip(symbols, coords)):
            ax.text(coord[0], coord[1], coord[2], f"{sym}{i}", size=10, zorder=1, color='k', weight='bold')

        ax.set_title(f"Canonical Geometry: {self.basename}\nDetected Symmetry: {self.point_group}", pad=20)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        
        # Use block=False if running in a notebook, or block=True for standalone scripts.
        # We draw and pause to allow the terminal prompt to appear simultaneously.
        plt.ion()
        plt.show()
        plt.pause(0.1)

    def prompt_symmetry_override(self) -> str:
        """Provides the batch override mechanism via terminal interaction."""
        print(f"\n{Colors.BOLD}--- Symmetry Confirmation: {self.basename} ---{Colors.ENDC}")
        print(f"Detected: {Colors.OKGREEN}{self.point_group}{Colors.ENDC}")
        override = input(f"Enter new Point Group to override (or press Enter to accept): ").strip()
        
        if override:
            self.point_group = override
            print_status(f"Symmetry manually overridden to: {self.point_group}", "info")
            
        if plt is not None:
            plt.close() # Close plot after decision
            
        return self.point_group

# ---------------------------------------------------------
# STAGE 5.0: REGISTRY & FINALIZATION
# ---------------------------------------------------------
class RegistryCommitter:
    def __init__(self, project_name: str, local_reg_path: str, global_reg_path: str):
        self.project_name = project_name
        self.local_reg_path = local_reg_path
        self.global_reg_path = global_reg_path

    def commit_file_data(self, file_hash: str, original_file: str, point_group: str, mapping: dict):
        """Updates the local project registry.json with provenance data."""
        try:
            with open(self.local_reg_path, "r") as f:
                local_reg = json.load(f)
        except FileNotFoundError:
            local_reg = {"ingested_files": {}}

        local_reg.setdefault("ingested_files", {})[file_hash] = {
            "original_file": os.path.basename(original_file),
            "point_group": point_group,
            "canonical_mapping": mapping,
            "timestamp": datetime.now().isoformat(),
            "source": "CoChem-MInt (Standard)" if "BuildAMol" not in original_file else "BuildAMol"
        }

        with open(self.local_reg_path, "w") as f:
            json.dump(local_reg, f, indent=4)
        print_status(f"Committed {file_hash[:8]}... to local registry.", "success")

    def update_global_config(self):
        """Registers the active project within the authoritative cochem_system_config.json."""
        try:
            with open(self.global_reg_path, "r") as f:
                global_reg = json.load(f)
                
            global_reg.setdefault("active_projects", {})[self.project_name] = {
                "local_registry": self.local_reg_path,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.global_reg_path, "w") as f:
                json.dump(global_reg, f, indent=4)
            print_status("Global cochem_system_config.json updated.", "success")
        except Exception as e:
            print_status(f"Failed to update global registry: {e}", "warning")

# ---------------------------------------------------------
# MInt EXECUTION HOOK (Example Usage)
# ---------------------------------------------------------
# if __name__ == "__main__":
#     # Assuming engine processed a file and returned an ASE 'atoms' object and 'mapping'
#     sym_ui = SymmetryAndUI(atoms, filepath)
#     sym_ui.detect_symmetry()
#     sym_ui.render_geometry()
#     final_sym = sym_ui.prompt_symmetry_override()
#     
#     committer = RegistryCommitter("CoChem-MInt-Test", "./Processed/registry.json", "cochem_setup/cochem_system_config.json")
#     committer.commit_file_data(file_hash, filepath, final_sym, mapping)
#     committer.update_global_config()