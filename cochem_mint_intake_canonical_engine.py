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