# %% [markdown]
# <details open>
# <summary><h2 style="display:inline;">▶ Stage 1.0: CoChem-MInt (Molecular Intake & Canonicalization)</h2></summary>
# 
# ### 🧪 Physical Chemistry Context: Standardizing the Ensemble
# Before executing PES scans or property predictions, the molecular ensemble must be rigorously standardized:
# 1. **Canonical Renumbering:** Isomer 0 is sorted by Atomic Weight, then Euclidean distance from the heaviest atom. 
# 2. **Hungarian Alignment:** All subsequent isomers are perfectly mapped to Isomer 0 using a Euclidean cost matrix to prevent atom-index scrambling during trajectory mapping.
# 3. **Symmetry Detection:** `molsym` evaluates the point group.
# </details>

# %%
import os
import sys
import json
import glob
import subprocess
import hashlib
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from ase.io import read, write
    from scipy.optimize import linear_sum_assignment
except ImportError:
    display(HTML("<b style='color:red;'>⚠️ FATAL: 'ase' and 'scipy' are required. Please run conda install conda-forge::ase scipy</b>"))
    sys.exit(1)

try:
    import molsym
    MOLSYM_AVAILABLE = True
except ImportError:
    MOLSYM_AVAILABLE = False

class CoChemMInt:
    def __init__(self):
        self.registry = self._load_system_registry()
        self.isomers = []  # Holds dicts of {file, atoms, point_group}
        self._build_dashboard()

    def _load_system_registry(self):
        """Computer Scan: Authenticate Hardware Registry from Stage 0."""
        config_path = os.path.join("cochem_setup", "cochem_system_config.json")
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            display(HTML("<b style='color:orange;'>⚠️ Warning: cochem_system_config.json missing. Hardware Matrix will display defaults.</b>"))
            return {}

    def _hash_file(self, filepath):
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ---------------------------------------------------------
    # MATH: Canonicalization & Hungarian Alignment
    # ---------------------------------------------------------
    def canonicalize_reference(self, atoms):
        """Renumber Isomer 0: Ordered by weight, then distance from heaviest atom."""
        masses = atoms.get_masses()
        heaviest_idx = np.argmax(masses)
        
        # Shift heaviest atom to origin (0,0,0)
        shift = np.copy(atoms.positions[heaviest_idx])
        atoms.positions -= shift
        
        distances = np.linalg.norm(atoms.positions, axis=1)
        
        # Sort criteria: Mass (Descending) -> Distance (Ascending)
        sort_indices = np.lexsort((distances, -masses))
        
        return atoms[sort_indices], shift

    def align_and_map_isomer(self, ref_atoms, tgt_atoms):
        """Map Isomer N to Isomer 0 using the Hungarian Algorithm to guarantee index preservation."""
        # 1. CoM Alignment
        ref_com = ref_atoms.get_center_of_mass()
        tgt_com = tgt_atoms.get_center_of_mass()
        tgt_atoms.positions += (ref_com - tgt_com)
        
        mapped_tgt_indices = np.zeros(len(ref_atoms), dtype=int)
        
        # 2. Map elements via Distance Cost Matrix
        for element in set(ref_atoms.get_chemical_symbols()):
            ref_idx = [i for i, a in enumerate(ref_atoms) if a.symbol == element]
            tgt_idx = [i for i, a in enumerate(tgt_atoms) if a.symbol == element]
            
            if len(ref_idx) != len(tgt_idx):
                raise ValueError(f"Isotope/Element mismatch detected for {element}!")
                
            cost = np.linalg.norm(
                ref_atoms.positions[ref_idx][:, None, :] - 
                tgt_atoms.positions[tgt_idx][None, :, :], 
                axis=2
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            
            for r, c in zip(row_ind, col_ind):
                mapped_tgt_indices[ref_idx[r]] = tgt_idx[c]
                
        tgt_mapped = tgt_atoms[mapped_tgt_indices]
        
        # 3. Kabsch Rotation Alignment (Overlay Isomer N perfectly onto Isomer 0)
        H = np.dot(tgt_mapped.positions.T, ref_atoms.positions)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U) * np.linalg.det(Vt)
        if d < 0.0:
            Vt[-1, :] *= -1
        R = np.dot(U, Vt)
        tgt_mapped.positions = np.dot(tgt_mapped.positions, R)
        
        return tgt_mapped

    def detect_symmetry(self, atoms):
        if not MOLSYM_AVAILABLE: return "C1"
        try:
            xyz_string = f"{len(atoms)}\n\n"
            for s, c in zip(atoms.get_chemical_symbols(), atoms.positions):
                xyz_string += f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
            mol = molsym.Molecule.from_string(xyz_string)
            return mol.find_point_group().symbol
        except Exception:
            return "C1"

    # ---------------------------------------------------------
    # INTERFACE: TORQ 1.0 UI Style & Bridge
    # ---------------------------------------------------------
    def _build_dashboard(self):
        # 1. Computer Scan Matrix (TORQ Style)
        hw = self.registry.get("hardware_profile", {})
        cpu_cores = hw.get("physical_cores", "N/A")
        ram_gb = hw.get("total_ram_gb", "N/A")
        avx512 = hw.get("avx512_support", False)
        avx_text = "Yes" if avx512 else "No"
        
        cpu_text = f"{cpu_cores} Cores | {ram_gb} GB RAM | AVX-512: {avx_text}"
        gpu_text = self.registry.get("gpu_profile", {}).get("name", "CPU Only / Not Detected")
        
        engine_paths = self.registry.get("engine_paths", {})
        orca_path = engine_paths.get("orca_path", "Not Found")
        mpi_path = engine_paths.get("mpi_path", "Not Found")
        
        matrix_html = f"""
        <div style="background-color: #2E3440; padding: 20px; border-radius: 8px; font-family: sans-serif; color: #D8DEE9; max-width: 800px;">
            <h2 style="color: #88C0D0; margin-top: 0;">CoChem-MInt: Ingestion Protocol</h2>
            <hr style="border-color: #4C566A;">
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px;">
                <tr><td style="padding: 8px; border-bottom: 1px solid #4C566A;"><b>CPU Topology & Memory:</b></td><td style="padding: 8px; border-bottom: 1px solid #4C566A;">{cpu_text}</td></tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #4C566A;"><b>GPU Acceleration:</b></td><td style="padding: 8px; border-bottom: 1px solid #4C566A;">{gpu_text}</td></tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #4C566A;"><b>Quantum Engine (ORCA):</b></td><td style="padding: 8px; border-bottom: 1px solid #4C566A; word-break: break-all;">{orca_path}</td></tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #4C566A;"><b>MPI Framework:</b></td><td style="padding: 8px; border-bottom: 1px solid #4C566A; word-break: break-all;">{mpi_path}</td></tr>
            </table>
        </div>
        """
        
        # 2. Inputs & Folder Scan
        self.w_proj = widgets.Text(value='New_Project', description='Project Name:', style={'description_width': 'initial'})
        self.w_path = widgets.Text(value='./Input_Files', description='Scan Folder:', style={'description_width': 'initial'})
        
        self.btn_upload = widgets.FileUpload(accept='.xyz', multiple=True, description='Upload Geometries', button_style='primary', icon='upload')
        self.btn_scan = widgets.Button(description='Scan & Canonicalize', button_style='info', icon='search')
        self.btn_build = widgets.Button(description='Build Molecule', button_style='warning', icon='cube')
        self.btn_save = widgets.Button(description='Register Ensemble', button_style='success', icon='save')
        
        # Observers & Callbacks
        self.btn_upload.observe(self.handle_upload, names='value')
        self.btn_scan.on_click(self.run_folder_scan)
        self.btn_build.on_click(self.launch_buildamol)
        self.btn_save.on_click(self.save_registry)

        self.out_console = widgets.Output(layout={'border': '1px solid #4C566A', 'height': '150px', 'overflow': 'auto', 'padding': '10px'})
        self.out_plots = widgets.Output()

        # Display Assembly
        display(HTML(matrix_html))
        display(widgets.HBox([self.w_proj, self.w_path]))
        display(widgets.HBox([self.btn_upload, self.btn_scan, self.btn_build, self.btn_save]))
        display(self.out_console)
        display(widgets.HTML("<hr><h3 style='margin:0; font-family: sans-serif;'>Canonical Isomer Alignment</h3>"))
        display(self.out_plots)

    # ---------------------------------------------------------
    # EXECUTION LOGIC
    # ---------------------------------------------------------
    def handle_upload(self, change):
        """Catches the FileUpload byte stream and writes valid XYZs to the target directory."""
        folder = self.w_path.value
        os.makedirs(folder, exist_ok=True)
        uploaded_files = change['new']
        
        with self.out_console:
            # ipywidgets v8+ returns a tuple of dicts, older versions return a dict
            if isinstance(uploaded_files, dict):
                for name, file_info in uploaded_files.items():
                    with open(os.path.join(folder, name), 'wb') as f:
                        f.write(file_info['content'])
                    print(f"✅ Uploaded and extracted: {name}")
            else:
                for file_info in uploaded_files:
                    with open(os.path.join(folder, file_info['name']), 'wb') as f:
                        f.write(file_info['content'])
                    print(f"✅ Uploaded and extracted: {file_info['name']}")
            
            # Reset widget value to allow re-uploading the same files if needed
            self.btn_upload.value = () if not isinstance(uploaded_files, dict) else {}

    def launch_buildamol(self, b):
        with self.out_console:
            print("Launching BuildAMol UI in isolated sub-process...")
            print("Please save output directly to your target directory.")
            os.makedirs(self.w_path.value, exist_ok=True)
            subprocess.Popen([sys.executable, "-m", "buildamol"], cwd=self.w_path.value)

    def run_folder_scan(self, b):
        self.out_console.clear_output()
        self.out_plots.clear_output()
        self.isomers = []
        
        folder = self.w_path.value
        if not os.path.exists(folder):
            os.makedirs(folder)
            with self.out_console: print(f"Created folder: {folder}. Please upload .xyz files.")
            return

        xyz_files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        if not xyz_files:
            with self.out_console: print(f"No .xyz files found in {folder}.")
            return

        with self.out_console:
            print(f"Found {len(xyz_files)} files. Initiating Canonical Engine...")
            
            try:
                # Load and Canonicalize Reference (Isomer 0)
                ref_raw = read(xyz_files[0])
                ref_canon, _ = self.canonicalize_reference(ref_raw)
                ref_sym = self.detect_symmetry(ref_canon)
                self.isomers.append({'file': xyz_files[0], 'atoms': ref_canon, 'sym': ref_sym})
                print(f"Reference Isomer set to: {os.path.basename(xyz_files[0])}")

                # Align remaining isomers
                for f in xyz_files[1:]:
                    tgt_raw = read(f)
                    tgt_mapped = self.align_and_map_isomer(ref_canon, tgt_raw)
                    tgt_sym = self.detect_symmetry(tgt_mapped)
                    self.isomers.append({'file': f, 'atoms': tgt_mapped, 'sym': tgt_sym})
                    print(f"Aligned and mapped: {os.path.basename(f)}")
            except Exception as e:
                print(f"❌ Canonicalization Failed: {e}")
                return

        self._render_isomer_tabs()

    def _render_isomer_tabs(self):
        """TORQ Style: Renders interactive Matplotlib windows inside IPywidget Tabs."""
        tabs = widgets.Tab()
        tab_children = []
        
        for i, iso in enumerate(self.isomers):
            out = widgets.Output()
            with out:
                # 3D Matplotlib Render
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                
                coords = iso['atoms'].positions
                syms = iso['atoms'].get_chemical_symbols()
                
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=150, c='cyan', edgecolors='k')
                for idx, (s, c) in enumerate(zip(syms, coords)):
                    ax.text(c[0], c[1], c[2], f"{s}{idx}", size=9, zorder=1, color='black', weight='bold')
                
                ax.set_title(f"Canonical Atoms: {os.path.basename(iso['file'])}")
                ax.axis('off')
                plt.show()

            # Symmetry Selector
            sym_opts = ["C1", "Cs", "C2", "C2v", "C3v", "D2h", "D3h", "D2d", "Td", "Oh", "Ih"]
            if iso['sym'] not in sym_opts: sym_opts.append(iso['sym'])
            
            w_sym = widgets.Dropdown(options=sym_opts, value=iso['sym'], description='Point Group:')
            iso['sym_widget'] = w_sym # Bind widget to isomer memory
            
            tab_children.append(widgets.VBox([out, w_sym]))
            
        tabs.children = tab_children
        for i, iso in enumerate(self.isomers):
            tabs.set_title(i, os.path.basename(iso['file']))
            
        with self.out_plots:
            display(tabs)

    def save_registry(self, b):
        with self.out_console:
            if not self.isomers:
                print("⚠️ No canonicalized isomers to save. Run Scan first.")
                return

            proj_dir = f"CoChem-{self.w_proj.value}"
            proc_dir = os.path.join(proj_dir, "Processed")
            os.makedirs(proc_dir, exist_ok=True)
            
            reg_data = {"project": self.w_proj.value, "ingested_files": {}}
            
            for iso in self.isomers:
                final_sym = iso['sym_widget'].value
                file_hash = self._hash_file(iso['file'])
                out_path = os.path.join(proc_dir, f"canonical_{file_hash[:8]}.xyz")
                
                write(out_path, iso['atoms'])
                
                reg_data["ingested_files"][file_hash] = {
                    "original_file": os.path.basename(iso['file']),
                    "canonical_path": out_path,
                    "point_group": final_sym
                }
                print(f"Exported Canonical Geometry -> {out_path} ({final_sym})")
                
            reg_path = os.path.join(proc_dir, "registry.json")
            with open(reg_path, "w") as f:
                json.dump(reg_data, f, indent=4)
            print(f"\n✅ Registry Locked: {reg_path}")

# Execute the CoChem-MInt initialization block automatically when run inside the cell
if __name__ == "__main__":
    MInt_Dashboard = CoChemMInt()

# Append this to CoChem-MInt.py

import os
import json
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print_status("CRITICAL: 'watchdog' library missing from main environment. Cannot build data bridge.", "fail")
    sys.exit(1)

# ---------------------------------------------------------
# STAGE 1.0: PROJECT SCAFFOLDING
# ---------------------------------------------------------
class ProjectScaffold:
    def __init__(self, project_name: str):
        self.project_name = f"CoChem-{project_name}"
        self.base_dir = os.path.abspath(self.project_name)
        self.input_dir = os.path.join(self.base_dir, "Input_Files")
        self.processed_dir = os.path.join(self.base_dir, "Processed")
        self.logs_dir = os.path.join(self.base_dir, "Logs")
        self.registry_path = os.path.join(self.processed_dir, "registry.json")

    def build_workspace(self) -> bool:
        """Constructs the rigid directory tree and initializes the local registry."""
        try:
            for directory in [self.input_dir, self.processed_dir, self.logs_dir]:
                os.makedirs(directory, exist_ok=True)
            
            if not os.path.exists(self.registry_path):
                initial_registry = {
                    "project_name": self.project_name,
                    "created_at": datetime.now().isoformat(),
                    "ingested_files": {},
                    "canonical_templates": {}
                }
                with open(self.registry_path, "w") as f:
                    json.dump(initial_registry, f, indent=4)
            
            print_status(f"Workspace configured at: {self.base_dir}", "success")
            return True
        except PermissionError:
            print_status(f"Permission denied creating workspace at {self.base_dir}", "fail")
            return False

# ---------------------------------------------------------
# STAGE 1.2: INTERACTIVE BRIDGE (BUILDAMOL)
# ---------------------------------------------------------
class MoleculeBuildHandler(FileSystemEventHandler):
    """Listens for new .xyz files saved by BuildAMol."""
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.xyz'):
            print_status(f"New geometry detected from builder: {os.path.basename(event.src_path)}", "info")
            self.callback(event.src_path)

class BuildAMolBridge:
    def __init__(self, silo_path: str, input_dir: str):
        self.silo_path = silo_path
        self.input_dir = input_dir
        self.python_bin = os.path.join(self.silo_path, "bin", "python")
        self.observer = None

    def _trigger_ingestion(self, file_path: str):
        """Callback fired when a new molecule is saved."""
        # This will link directly to Stage 2.0 (Intake & Hashing)
        print_status(f"Queuing {file_path} for canonical ingestion...", "success")

    def launch_and_watch(self):
        """Launches BuildAMol in a subprocess and watches the input directory."""
        if not os.path.exists(self.python_bin):
            print_status("BuildAMol Python binary missing. Silo may be corrupted.", "fail")
            return

        print_status("Initializing Data Bridge Watchdog...", "info")
        event_handler = MoleculeBuildHandler(self._trigger_ingestion)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_dir, recursive=False)
        self.observer.start()

        print_status("Launching BuildAMol interactive module...", "info")
        # Launching as a subprocess to keep the CoChem-MInt loop active
        env = os.environ.copy()
        # Ensure the subprocess saves directly to our project input folder
        env["WORKDIR"] = self.input_dir 
        
        try:
            # Command assumes BuildAMol can be invoked as a module or script
            subprocess.run(
                [self.python_bin, "-m", "buildamol"], 
                env=env, 
                cwd=self.input_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print_status(f"BuildAMol terminated unexpectedly: {e}", "warning")
        finally:
            print_status("Closing Data Bridge and halting Watchdog.", "info")
            self.observer.stop()
            self.observer.join()

# Append this to CoChem-MInt_Bridge.py

import os
import json
import hashlib
import numpy as np

try:
    import networkx as nx
    from ase.io import read, write
    from ase.neighborlist import neighbor_list, natural_cutoffs
except ImportError as e:
    print_status(f"CRITICAL: Engine dependency missing. {e}", "fail")
    sys.exit(1)

# ---------------------------------------------------------
# STAGE 2.0: INTAKE & HASHING (INCREMENTAL RESCAN)
# ---------------------------------------------------------
class IntakeEngine:
    def __init__(self, input_dir: str, registry_path: str):
        self.input_dir = input_dir
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def hash_file(self, filepath: str) -> str:
        """Calculates SHA-256 for deterministic file tracking."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_pending_files(self) -> list:
        """Returns only new or modified XYZ files not present in the registry."""
        pending = []
        for file in os.listdir(self.input_dir):
            if file.endswith(".xyz"):
                full_path = os.path.join(self.input_dir, file)
                file_hash = self.hash_file(full_path)
                
                if file_hash not in self.registry.get("ingested_files", {}):
                    pending.append((full_path, file_hash))
        return pending

# ---------------------------------------------------------
# STAGE 3.0: CANONICAL RENUMBERING ENGINE
# ---------------------------------------------------------
class CanonicalRenumberer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        try:
            self.atoms = read(filepath)
            self.success = True
        except Exception as e:
            print_status(f"Failed to parse {filepath}: {e}", "warning")
            self.success = False

    def _build_molecular_graph(self) -> nx.Graph:
        """Generates connectivity graph using ASE natural cutoffs."""
        cutoffs = natural_cutoffs(self.atoms)
        i_indices, j_indices = neighbor_list('ij', self.atoms, cutoffs)
        
        G = nx.Graph()
        G.add_nodes_from(range(len(self.atoms)))
        G.add_edges_from(zip(i_indices, j_indices))
        return G

    def renumber(self) -> dict:
        """Executes the Mass-Weighted Shell-Traversal algorithm."""
        if not self.success:
            return {}

        # 1. Center of Mass
        com = self.atoms.get_center_of_mass()
        self.atoms.positions -= com

        # 2. Identify Root (Heaviest Mass -> Furthest from CoM)
        masses = self.atoms.get_masses()
        distances_to_com = np.linalg.norm(self.atoms.positions, axis=1)
        
        # Sort criteria: -mass, -distance (Negative for descending sort)
        root_idx = max(range(len(self.atoms)), key=lambda i: (masses[i], distances_to_com[i]))

        # 3. Build Graph & Traverse Shells
        G = self._build_molecular_graph()
        
        if not nx.is_connected(G):
            print_status(f"Warning: Disconnected fragments detected in {os.path.basename(self.filepath)}", "warning")
            # For strict rigor, isolated atoms are assigned to a "disconnected" shell
            path_lengths = nx.single_source_shortest_path_length(G, root_idx)
            for i in range(len(self.atoms)):
                if i not in path_lengths:
                    path_lengths[i] = 999 
        else:
            path_lengths = nx.single_source_shortest_path_length(G, root_idx)

        # 4. Group by Shell and Sort (The Branch-Priority Rule)
        # Sort: Shell Level (ASC) -> Mass (DESC) -> Valence/Degree (DESC) -> Dist to Root (DESC)
        root_pos = self.atoms.positions[root_idx]
        distances_to_root = np.linalg.norm(self.atoms.positions - root_pos, axis=1)
        degrees = dict(G.degree())

        sorted_indices = sorted(
            range(len(self.atoms)),
            key=lambda i: (
                path_lengths[i],       # Primary: Shell depth (0, 1, 2...)
                -masses[i],            # Secondary: Atomic Mass
                -degrees[i],           # Tertiary: Valence connectivity
                -distances_to_root[i]  # Quaternary: Euclidean spread from root
            )
        )

        # 5. Apply reordering mapping
        mapping_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        self.atoms = self.atoms[sorted_indices]

        return mapping_old_to_new

    def export(self, export_path: str):
        """Saves the canonicalized geometry."""
        write(export_path, self.atoms)

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

                        