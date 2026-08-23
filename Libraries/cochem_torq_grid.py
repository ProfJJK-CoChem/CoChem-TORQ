"""
CoChem-TORQ: 1D and Multidimensional Potential Energy Grid Engine
Stage 3: Sinc-DVR Hamiltonian Construction & Variational Nuclear Dynamics
Compliant with Method Matrix v4 (§4.4, §8C, Table 2).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ARTIFACTS_DIR = os.environ.get('COCHEM_ARTIFACTS_DIR', str(Path.home() / 'cochem_artifacts'))
from typing import Any
import numpy as np
import networkx as nx

logger = logging.getLogger("TorqGrid")


class TorqGrid:
    """
    Grid engine for torsional scanning, PES generation, and discrete variable
    representation (sinc-DVR) nuclear dynamics.
    """

    def __init__(
        self,
        symbols: list[str],
        coordinates: list[list[float]] | np.ndarray,
        connectivity_graph: nx.Graph | None = None,
    ) -> None:
        """
        Initialize TorqGrid with molecular geometry and connectivity.

        :param symbols: List of atomic element symbols.
        :param coordinates: Cartesian atomic coordinates (N_atoms x 3).
        :param connectivity_graph: NetworkX graph representing covalent connectivity.
        """
        self.symbols = list(symbols)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.connectivity_graph = connectivity_graph
        self.num_atoms = len(self.symbols)

    def construct_sinc_dvr_hamiltonian(
        self,
        grid_points: list[float] | np.ndarray,
        energies: list[float] | np.ndarray,
        mass_amu: float = 1.0,
    ) -> dict[str, Any]:
        """
        Constructs a 1D sinc-DVR (Colbert-Miller) kinetic energy matrix and diagonal
        potential energy matrix, diagonalizes the resulting Hamiltonian matrix,
        and returns the eigenvalues, eigenvectors, full Hamiltonian, and point count.

        :param grid_points: 1D coordinate array (uniform grid points).
        :param energies: Potential energies at each grid point.
        :param mass_amu: Effective reduced mass in atomic mass units.
        :return: Dictionary containing 'hamiltonian', 'energy_levels', 'wavefunctions', and 'num_points'.
        """
        pts = np.asarray(grid_points, dtype=np.float64)
        pot = np.asarray(energies, dtype=np.float64)
        n_pts = len(pts)

        if n_pts == 0:
            raise ValueError("Grid points and energies cannot be empty.")
        if len(pot) != n_pts:
            raise ValueError(
                f"Mismatch between number of grid points ({n_pts}) and energies ({len(pot)})."
            )

        if n_pts > 1:
            delta_x = float(pts[1] - pts[0])
            if delta_x == 0.0:
                delta_x = 1.0
        else:
            delta_x = 1.0

        # Sinc-DVR Kinetic Energy Matrix (Colbert-Miller formalism)
        # T_ii = (hbar^2 / (2 * m * dx^2)) * (pi^2 / 3)
        # T_ij = (hbar^2 / (2 * m * dx^2)) * (2 * (-1)^(i-j) / (i-j)^2)
        factor = 1.0 / (2.0 * float(mass_amu) * (delta_x**2))

        t_matrix = np.zeros((n_pts, n_pts), dtype=np.float64)
        for i in range(n_pts):
            for j in range(n_pts):
                if i == j:
                    t_matrix[i, i] = factor * (np.pi**2 / 3.0)
                else:
                    diff = i - j
                    t_matrix[i, j] = factor * (2.0 * ((-1.0) ** diff) / (diff**2))

        v_matrix = np.diag(pot)
        h_matrix = t_matrix + v_matrix

        eigenvalues, eigenvectors = np.linalg.eigh(h_matrix)

        logger.info(
            f"Constructed sinc-DVR Hamiltonian for {n_pts} grid points; ground state energy = {eigenvalues[0]:.6f}"
        )

        return {
            "hamiltonian": h_matrix,
            "energy_levels": eigenvalues,
            "wavefunctions": eigenvectors,
            "num_points": n_pts,
        }
