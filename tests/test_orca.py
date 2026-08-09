import pytest
import asyncio
import numpy as np
import networkx as nx
from Libraries.cochem_torq_orca import TorqOrcaExecutor
from Libraries.cochem_torq_neb import run_ts_optimization, _run_irc_validation, compute_kabsch_rmsd
from Libraries.cochem_torq_grid import TorqGrid

def test_torq_orca_executor_init():
    executor = TorqOrcaExecutor()
    assert executor.orca_path == "orca"

def test_torq_orca_generate_input():
    executor = TorqOrcaExecutor()
    mock_coords = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.75, 0.58], ["H", 0.0, -0.75, 0.58]]
    inp = executor._generate_orca_input("B3LYP", "def2-TZVP", "def2-TZVP/CPCM", "DIIS", mock_coords, charge=0, multiplicity=1)
    assert "* xyz 0 1" in inp
    assert "B3LYP" in inp
    assert "def2-TZVP" in inp

def test_torq_orca_output_parser(tmp_path):
    out_file = tmp_path / "test.out"
    out_file.write_text("""
FINAL SINGLE POINT ENERGY   -76.4321098
Total Dipole Moment : 0.000 0.000 1.850
VIBRATIONAL FREQUENCIES
-----------------------
 0:   1500.00 cm**-1
 1:   3600.00 cm**-1
 2:   3700.00 cm**-1
    """)
    
    executor = TorqOrcaExecutor()
    parsed = executor.parse_orca_output(str(out_file))
    assert parsed["energy"] == -76.4321098
    assert len(parsed["vibrational_frequencies"]) == 3
    assert parsed["dipole_moment"]["total"] == 1.85

def test_validate_imaginary_frequencies():
    executor = TorqOrcaExecutor()
    assert executor.validate_imaginary_frequencies([-350.0, 100.0, 500.0]) is True
    assert executor.validate_imaginary_frequencies([100.0, 500.0]) is False
    assert executor.validate_imaginary_frequencies([-350.0, -120.0, 500.0]) is False

def test_kabsch_rmsd():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = p.copy()
    rmsd = TorqOrcaExecutor.compute_kabsch_rmsd(p, q)
    assert rmsd < 1e-5

def test_sinc_dvr_hamiltonian():
    syms = ["H", "O", "O", "H"]
    coords = [[0.0, 0.95, 0.0], [0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [1.4, 0.95, 0.5]]
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    gridder = TorqGrid(syms, coords, graph)
    grid_points = [0.0, 30.0, 60.0, 90.0, 120.0]
    energies = [0.0, 1.2, 3.5, 1.2, 0.0]
    
    result = gridder.construct_sinc_dvr_hamiltonian(grid_points, energies, mass_amu=1.0)
    assert "hamiltonian" in result
    assert len(result["energy_levels"]) == 5
    assert result["num_points"] == 5
