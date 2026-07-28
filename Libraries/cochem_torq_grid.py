"""
CoChem-TORQ 0.0.11
Stage 1.2: Grid Definition & Torsional Mesh
-------------------------------------------
Generates 1D/2D torsional sampling grids. Uses NetworkX to isolate
the rotating fragment and applies explicit Cartesian rotations around the 
central bond vector (Rodrigues' rotation formula) to seed the downstream
MACE-OFF23 triage with optimal starting geometries.
"""

import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [CoChem-TORQ] %(message)s")
logger = logging.getLogger("TorqGrid")

class TorqGrid:
    def __init__(self, symbols, coordinates, graph):
        """
        Initialize the grid generator.
        :param symbols: List of atomic symbols.
        :param coordinates: Nx3 numpy array of Cartesian coordinates.
        :param graph: networkx.Graph representing molecular connectivity (from Stage 1.0).
        """
        self.symbols = symbols
        self.base_coords = np.array(coordinates, dtype=np.float64)
        self.graph = graph
        self.num_atoms = len(symbols)
        self.generated_grid = []

    def _isolate_rotating_top(self, dihedral):
        """
        Sever the central bond (B-C) of the dihedral (A-B-C-D) to find which 
        atoms rotate with the C-D fragment.
        """
        a, b, c, d = dihedral
        temp_graph = self.graph.copy()
        
        if temp_graph.has_edge(b, c):
            temp_graph.remove_edge(b, c)
        else:
            logger.error(f"Central bond {b}-{c} not found in graph. Check topology.")
            return []

        # Find the connected component containing atom 'c' (and 'd')
        subgraphs = list(nx.connected_components(temp_graph))
        for sg in subgraphs:
            if c in sg:
                logger.info(f"Isolated rotating top for dihedral {dihedral}: {list(sg)}")
                return list(sg)
                
        return []

    def _rotate_cartesian(self, coords, rotating_indices, axis_start, axis_end, angle_deg):
        """
        Rotate specific atoms around a defined bond axis vector.
        :param coords: Nx3 coordinate array.
        :param rotating_indices: List of atom indices to rotate.
        :param axis_start: Atom index B (origin of rotation axis).
        :param axis_end: Atom index C (direction of rotation axis).
        :param angle_deg: Angle to rotate in degrees.
        """
        new_coords = coords.copy()
        
        # Define rotation axis
        p1 = coords[axis_start]
        p2 = coords[axis_end]
        axis_vector = p2 - p1
        axis_vector /= np.linalg.norm(axis_vector) # Normalize
        
        # Calculate rotation using Rodrigues' formula (via scipy Rotation)
        rot_vector = axis_vector * np.radians(angle_deg)
        rotation = R.from_rotvec(rot_vector)
        
        # Translate to origin, rotate, translate back
        to_rotate = new_coords[rotating_indices] - p1
        rotated = rotation.apply(to_rotate)
        new_coords[rotating_indices] = rotated + p1
        
        return new_coords

    def generate_1d_grid(self, dihedral, resolution_deg=10):
        """
        Generates a 1D torsional sweep.
        :param dihedral: tuple/list of 4 atom indices (A, B, C, D).
        :param resolution_deg: Step size in degrees.
        """
        logger.info(f"Generating 1D grid for dihedral {dihedral} with {resolution_deg}° steps.")
        rotating_top = self._isolate_rotating_top(dihedral)
        
        if not rotating_top:
            raise ValueError("Failed to isolate spinning top. Cannot build grid.")

        angles = np.arange(0, 360, resolution_deg)
        self.generated_grid = []

        for angle in angles:
            rotated_coords = self._rotate_cartesian(
                self.base_coords, rotating_top, dihedral[1], dihedral[2], angle
            )
            self.generated_grid.append({
                "dihedral_angles": {str(dihedral): float(angle)},
                "coordinates": rotated_coords.tolist()
            })
            
        logger.info(f"Successfully generated {len(self.generated_grid)} grid points.")
        return self.generated_grid

    def generate_2d_grid(self, dihedral_1, dihedral_2, resolution_deg=15):
        """
        Generates a 2D torsional mesh. Includes O(N^2) combinatorial warnings.
        """
        points_per_axis = int(360 / resolution_deg)
        total_points = points_per_axis ** 2
        
        logger.warning(f"Generating 2D grid. Combinatorial scaling alert: {total_points} total geometries.")
        if total_points > 1000:
            logger.warning("Grid exceeds 1000 points. MACE-OFF23 highly recommended for Stage 2.0 triage.")

        top_1 = self._isolate_rotating_top(dihedral_1)
        top_2 = self._isolate_rotating_top(dihedral_2)

        angles_1 = np.arange(0, 360, resolution_deg)
        angles_2 = np.arange(0, 360, resolution_deg)
        self.generated_grid = []

        for a1 in angles_1:
            # Apply first rotation
            temp_coords = self._rotate_cartesian(
                self.base_coords, top_1, dihedral_1[1], dihedral_1[2], a1
            )
            for a2 in angles_2:
                # Apply second rotation cumulatively
                final_coords = self._rotate_cartesian(
                    temp_coords, top_2, dihedral_2[1], dihedral_2[2], a2
                )
                self.generated_grid.append({
                    "dihedral_angles": {str(dihedral_1): float(a1), str(dihedral_2): float(a2)},
                    "coordinates": final_coords.tolist()
                })

        logger.info(f"Successfully generated 2D mesh containing {len(self.generated_grid)} points.")
        return self.generated_grid

    def export_grid(self, filename="torq_grid.json"):
        """Exports the Cartesian grid coordinates for downstream evaluation."""
        if not self.generated_grid:
            logger.warning("No grid generated to export.")
            return

        with open(filename, "w") as f:
            json.dump({
                "symbols": self.symbols,
                "grid_points": self.generated_grid,
                "num_points": len(self.generated_grid)
            }, f, indent=2)
        logger.info(f"Grid exported to {filename} for downstream ML/DFT triage.")

if __name__ == "__main__":
    # Self-test payload (Ethane-like mock)
    mock_syms = ["C", "C", "H", "H", "H", "H", "H", "H"]
    mock_coords = [
        [ 0.0, 0.0, 0.0], [ 1.5, 0.0, 0.0],
        [-0.5, 1.0, 0.0], [-0.5,-0.5, 0.8], [-0.5,-0.5,-0.8],
        [ 2.0, 1.0, 0.0], [ 2.0,-0.5, 0.8], [ 2.0,-0.5,-0.8]
    ]
    
    # Mocking graph for test
    mock_graph = nx.Graph()
    mock_graph.add_edges_from([(0,1), (0,2), (0,3), (0,4), (1,5), (1,6), (1,7)])
    
    gridder = TorqGrid(mock_syms, mock_coords, mock_graph)
    
    # 1D test
    grid_1d = gridder.generate_1d_grid(dihedral=(2, 0, 1, 5), resolution_deg=30)
    gridder.export_grid()