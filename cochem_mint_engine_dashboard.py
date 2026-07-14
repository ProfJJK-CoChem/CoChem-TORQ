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
        self.isomers = []  
        self.ui_elements = []
        
        # Initialize output areas first so logging can use them
        self.out_console = widgets.Output(layout={'border': '1px solid #4C566A', 'background-color': '#3B4252', 'color': '#A3BE8C', 'height': '180px', 'overflow': 'auto', 'padding': '10px'})
        self.out_plots = widgets.Output()
        
        self._build_dashboard()
        display(widgets.VBox(self.ui_elements))

    def _load_system_registry(self):
        """Authenticates Hardware Registry from Stage 0."""
        config_path = os.path.join("cochem_setup", "cochem_system_config.json")
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _recursive_find(self, d, target_key):
        """Helper to find keys in deeply nested config dicts."""
        if target_key in d: return d[target_key]
        for k, v in d.items():
            if isinstance(v, dict):
                res = self._recursive_find(v, target_key)
                if res is not None: return res
        return None

    def _find_path(self, bin_name):
        """Aggressively hunts for engine paths in the registry string values."""
        def _search(d):
            for k, v in d.items():
                if isinstance(v, str) and bin_name in v.lower() and ('/' in v or '\\' in v):
                    return v
                if isinstance(v, dict):
                    res = _search(v)
                    if res: return res
            return None
        val = _search(self.registry)
        return val if val else "Not Found"

    def _hash_file(self, filepath):
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def print_log(self, msg):
        """Safe logging router attached to the IPyWidget console."""
        with self.out_console:
            print(msg)

    # ---------------------------------------------------------
    # MATH: Canonicalization & Hungarian Alignment
    # ---------------------------------------------------------
    def canonicalize_reference(self, atoms):
        """Renumber Isomer 0: Ordered by weight, then distance from heaviest atom."""
        masses = atoms.get_masses()
        heaviest_idx = np.argmax(masses)
        
        shift = np.copy(atoms.positions[heaviest_idx])
        atoms.positions -= shift
        distances = np.linalg.norm(atoms.positions, axis=1)
        
        sort_indices = np.lexsort((distances, -masses))
        return atoms[sort_indices], shift

    def align_and_map_isomer(self, ref_atoms, tgt_atoms):
        """Map Isomer N to Isomer 0 using Hungarian Cost Matrix."""
        ref_com = ref_atoms.get_center_of_mass()
        tgt_com = tgt_atoms.get_center_of_mass()
        tgt_atoms.positions += (ref_com - tgt_com)
        
        mapped_tgt_indices = np.zeros(len(ref_atoms), dtype=int)
        
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
    # INTERFACE: TORQ 1.0 UI Style
    # ---------------------------------------------------------
    def _build_dashboard(self):
        # 1. Computer Scan Matrix
        cores = self._recursive_find(self.registry, "physical_cores") or "N/A"
        ram = self._recursive_find(self.registry, "total_ram_gb") or "N/A"
        avx = "Detected" if self._recursive_find(self.registry, "avx512_support") else "Not Detected"
        gpu = self._recursive_find(self.registry, "name") or "CPU Only / Not Detected"
        
        orca_path = self._find_path("orca")
        mpi_path = self._find_path("mpirun")
        
        matrix_html = f"""
        <div style="background-color: #2E3440; padding: 20px; border-radius: 8px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #D8DEE9; max-width: 850px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h2 style="color: #88C0D0; margin-top: 0; border-bottom: 2px solid #4C566A; padding-bottom: 10px;">CoChem-MInt: Intake & Triage State</h2>
            <h4 style="color: #ECEFF4; margin-bottom: 5px;">CoChem System Matrix</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr>
                    <td style="padding: 6px 0; border-bottom: 1px solid #4C566A; width: 50%;"><b>CPU:</b> {cores} Cores | <b>Available Memory:</b> {ram} GB | <b>AVX-512:</b> {avx}</td>
                    <td style="padding: 6px 0; border-bottom: 1px solid #4C566A;"><b>GPU:</b> {gpu}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; border-bottom: 1px solid #4C566A;"><b>ORCA Engine:</b> <span style="color:#A3BE8C;">{orca_path}</span></td>
                    <td style="padding: 6px 0; border-bottom: 1px solid #4C566A;"><b>MPI Framework:</b> <span style="color:#A3BE8C;">{mpi_path}</span></td>
                </tr>
            </table>
        </div>
        """
        
        # 2. Inputs & Folder Scan
        self.w_proj = widgets.Text(value='New_Project', description='Project Name:', style={'description_width': 'initial'})
        self.w_path = widgets.Text(value='./Input_Files', description='Scan Folder:', style={'description_width': 'initial'})
        
        self.btn_scan = widgets.Button(description='Scan Folder & Canonicalize', button_style='info', icon='search')
        self.btn_build = widgets.Button(description='Build Molecule', button_style='warning', icon='cube')
        self.btn_save = widgets.Button(description='Register Ensemble', button_style='success', icon='save')
        
        self.btn_scan.on_click(self.run_folder_scan)
        self.btn_build.on_click(self.launch_buildamol)
        self.btn_save.on_click(self.save_registry)

        # Build UI Elements array (TORQ methodology)
        self.ui_elements.append(widgets.HTML(matrix_html))
        self.ui_elements.append(widgets.HBox([self.w_proj, self.w_path]))
        self.ui_elements.append(widgets.HBox([self.btn_scan, self.btn_build, self.btn_save]))
        self.ui_elements.append(self.out_console)
        self.ui_elements.append(widgets.HTML("<hr><h3 style='margin:0; font-family: sans-serif; color: #4C566A;'>Canonical Isomer Alignment</h3>"))
        self.ui_elements.append(self.out_plots)

        self.print_log("▶ Initiating CoChem-MInt Protocol... Waiting for user input.")

    # ---------------------------------------------------------
    # EXECUTION LOGIC
    # ---------------------------------------------------------
    def launch_buildamol(self, b):
        self.print_log("\n🚀 Launching BuildAMol UI in isolated sub-process...")
        self.print_log("⚠️ Note: Save your built geometry directly to the './Input_Files' directory.")
        os.makedirs(self.w_path.value, exist_ok=True)
        try:
            # Assumes buildamol is pip installed in the active silo
            subprocess.Popen([sys.executable, "-m", "buildamol"], cwd=self.w_path.value)
        except Exception as e:
            self.print_log(f"❌ Failed to launch builder: {e}")

    def run_folder_scan(self, b):
        self.out_console.clear_output()
        self.out_plots.clear_output()
        self.isomers = []
        
        folder = self.w_path.value
        if not os.path.exists(folder):
            os.makedirs(folder)
            self.print_log(f"📁 Created folder: {folder}. Please place .xyz files inside and click Scan again.")
            return

        xyz_files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        if not xyz_files:
            self.print_log(f"⚠️ No .xyz files found in {folder}.")
            return

        self.print_log(f"🔍 Found {len(xyz_files)} files. Initiating Canonical Engine...")
        
        try:
            # Process Isomer 0
            ref_raw = read(xyz_files[0])
            ref_canon, _ = self.canonicalize_reference(ref_raw)
            ref_sym = self.detect_symmetry(ref_canon)
            self.isomers.append({'file': xyz_files[0], 'atoms': ref_canon, 'sym': ref_sym})
            self.print_log(f"✅ Reference Isomer [0] locked: {os.path.basename(xyz_files[0])}")

            # Align remaining isomers via Hungarian Cost Matrix
            for f in xyz_files[1:]:
                tgt_raw = read(f)
                tgt_mapped = self.align_and_map_isomer(ref_canon, tgt_raw)
                tgt_sym = self.detect_symmetry(tgt_mapped)
                self.isomers.append({'file': f, 'atoms': tgt_mapped, 'sym': tgt_sym})
                self.print_log(f"✅ Aligned and mapped: {os.path.basename(f)}")
                
        except Exception as e:
            self.print_log(f"❌ Canonicalization Failed: {e}")
            return

        self._render_isomer_tabs()

    def _render_isomer_tabs(self):
        """Renders interactive Matplotlib windows inside IPywidget Tabs."""
        tabs = widgets.Tab()
        tab_children = []
        
        for i, iso in enumerate(self.isomers):
            out = widgets.Output()
            with out:
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

            # Dynamic Symmetry Options
            sym_opts = ["C1", "Cs", "C2", "C2v", "C3v", "D2h", "D3h", "D2d", "Td", "Oh", "Ih"]
            if iso['sym'] not in sym_opts: sym_opts.append(iso['sym'])
            
            w_sym = widgets.Dropdown(options=sym_opts, value=iso['sym'], description='Point Group:')
            iso['sym_widget'] = w_sym 
            
            tab_children.append(widgets.VBox([out, w_sym]))
            
        tabs.children = tab_children
        for i, iso in enumerate(self.isomers):
            tabs.set_title(i, os.path.basename(iso['file']))
            
        with self.out_plots:
            display(tabs)

    def save_registry(self, b):
        with self.out_console:
            if not self.isomers:
                print("\n⚠️ No canonicalized isomers to save. Run Scan first.")
                return

            proj_dir = f"CoChem-{self.w_proj.value}"
            proc_dir = os.path.join(proj_dir, "Processed")
            os.makedirs(proc_dir, exist_ok=True)
            
            reg_data = {"project": self.w_proj.value, "ingested_files": {}}
            
            print(f"\n[Registering Ensemble -> {proc_dir}]")
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
                print(f" -> Exported Geometry: {out_path} ({final_sym})")
                
            reg_path = os.path.join(proc_dir, "registry.json")
            with open(reg_path, "w") as f:
                json.dump(reg_data, f, indent=4)
            print(f"✅ Registry Locked successfully!")

# Auto-execute when run in Jupyter
if __name__ == "__main__":
    MInt_Dashboard = CoChemMInt()