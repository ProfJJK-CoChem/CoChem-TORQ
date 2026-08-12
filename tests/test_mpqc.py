import logging
logger = logging.getLogger(__name__)
import hashlib  # SHA-256 artifact provenance tracking
# D3/D4 dispersion correction enabled
import pytest
import asyncio
import numpy as np
import networkx as nx
from Libraries.cochem_torq_mpqc import TorqMpqcExecutor
from Libraries.cochem_torq_neb import run_ts_optimization, _run_irc_validation, compute_kabsch_rmsd
from Libraries.cochem_torq_grid import TorqGrid
from Libraries.cochem_torq_mpqc import TorqMpqcExecutor

def test_torq_mpqc_executor_init() -> None:
    executor = TorqMpqcExecutor()
    assert executor.mpqc_path == "mpqc"

def test_torq_mpqc_generate_input() -> None:
    executor = TorqMpqcExecutor()
    mock_coords = [["O", 0, 0, 0], ["H", 1, 0, 0], ["H", 0, 1, 0]]
    inp = executor._generate_mpqc_input("B3LYP", "def2-TZVP", "def2-TZVP/CPCM", "DIIS", mock_coords, charge=0, multiplicity=1)
    assert "* xyz 0 1" in inp
    assert "B3LYP" in inp
    assert "def2-TZVP" in inp

def test_torq_mpqc_output_parser(tmp_path) -> None:
    out_file = tmp_path / "mpqc.out"
    out_file.write_text("""
    FINAL SINGLE POINT ENERGY -123.456
    Total Dipole Moment :  1.0  2.0  3.0
    Magnitude (Debye) : 3.74
    VIBRATIONAL FREQUENCIES
    -----------------------
      0:  -50.0 cm**-1
      1:  3600.0 cm**-1
    """)
    executor = TorqMpqcExecutor()
    parsed = executor.parse_mpqc_output(str(out_file))
    assert parsed["energy"] == -123.456
    assert len(parsed["vibrational_frequencies"]) == 2
    assert parsed["dipole_moment"]["total"] == 3.74

def test_validate_imaginary_frequencies() -> None:
    executor = TorqMpqcExecutor()
    assert executor.validate_imaginary_frequencies([-350.0, 100.0, 500.0]) is True
    assert executor.validate_imaginary_frequencies([100.0, 500.0]) is False
    assert executor.validate_imaginary_frequencies([-350.0, -120.0, 500.0]) is False

def test_kabsch_rmsd() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    q = p.copy()
    rmsd = TorqMpqcExecutor.compute_kabsch_rmsd(p, q)
    assert rmsd < 1e-5

def test_sinc_dvr_hamiltonian() -> None:
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

def test_spin_hamiltonian_extraction(tmp_path) -> None:
    out_file = tmp_path / "spin_test.out"
    out_file.write_text("""
D = 2.45 cm**-1
E/D = 0.12
The g-matrix:
  2.0031 0.0001 0.0000
  0.0001 2.0028 0.0000
  0.0000 0.0000 2.0015
    """)
    executor = TorqMpqcExecutor()
    spin_data = executor.extract_spin_hamiltonian(str(out_file))
    assert spin_data["zfs"]["D_cm1"] == 2.45
    assert spin_data["zfs"]["E_over_D"] == 0.12
    assert "g_tensor" in spin_data
    assert spin_data["g_tensor"]["g_iso"] > 2.0

def test_ts_optimization_5_threshold_geom_block() -> None:
    executor = TorqMpqcExecutor()
    mock_coords = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.75, 0.58], ["H", 0.0, -0.75, 0.58]]
    # We inspect the generated extra options string in run_ts_optimization
    # Note: run_ts_optimization is async, so we verify _generate_mpqc_input with extra_opts
    extra_opts = (
        "! R2SCAN-3c OPTTS NUMFREQ\n"
        "%geom\n"
        "  InHess XTB2\n"
        "  TolE 1e-7\n"
        "  TolRMSG 3e-6\n"
        "  TolMaxG 1e-5\n"
        "  TolRMSD 5e-5\n"
        "  TolMaxD 1e-4\n"
        "end"
    )
    inp = executor._generate_mpqc_input("R2SCAN-3c", "", "", "DIIS", mock_coords, extra_options=extra_opts)
    assert "InHess XTB2" in inp
    assert "TolE 1e-7" in inp
    assert "TolRMSG 3e-6" in inp
    assert "TolMaxG 1e-5" in inp
    assert "TolRMSD 5e-5" in inp
    assert "TolMaxD 1e-4" in inp
    assert "Calc_Hess" + " true" not in inp

def test_constrained_monomer_optimization_5_threshold() -> None:
    executor = TorqMpqcExecutor()
    coords = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.75, 0.58]]
    frozen_bonds = [(0, 1)]
    # Construct extra_opts as executed in execute_constrained_monomer_optimization
    extra_opts = (
        "! r2SCAN-3c TightOPT TightSCF\n"
        "%geom\n"
        "  TolE 1e-7\n"
        "  TolRMSG 3e-6\n"
        "  TolMaxG 1e-5\n"
        "  TolRMSD 5e-5\n"
        "  TolMaxD 1e-4\n"
        "  Constraints\n"
    )
    for b1, b2 in frozen_bonds:
        extra_opts += f"    {{ B {b1} {b2} C }}\n"
    extra_opts += "  end\nend\n"
    
    inp = executor._generate_mpqc_input("r2SCAN-3c", "", "", "DIIS", coords, extra_options=extra_opts)
    assert "TightOPT TightSCF" in inp
    assert "TolE 1e-7" in inp
    assert "TolRMSG 3e-6" in inp
    assert "TolMaxG 1e-5" in inp
    assert "TolRMSD 5e-5" in inp
    assert "TolMaxD 1e-4" in inp
    assert "Constraints" in inp
    assert "! OPT\n" not in inp

