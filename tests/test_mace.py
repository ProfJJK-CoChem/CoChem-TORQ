import pytest
import json
from pathlib import Path
from Libraries.cochem_torq_mace import TorqMACETriage

def test_mace_triage_init(tmp_path):
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {"dihedral_angles": [0], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]},
            {"dihedral_angles": [30], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]]}
        ]
    }
    grid_file.write_text(json.dumps(grid_data))
    
    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")
    assert triage.symbols == ["H", "H"]
    assert len(triage.grid_points) == 2
    assert triage.batch_size in (16, 512)
    assert triage.model_name == "MACE-OFF24m"
    assert triage.scf_tolerance_guard == 1e-5
    assert triage.device in ["cpu", "cuda"]

def test_aimnet2_triage_init(tmp_path):
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {"dihedral_angles": [0], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]}
        ]
    }
    grid_file.write_text(json.dumps(grid_data))
    
    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="AIMNet2")
    assert triage.model_name == "AIMNet2"
    assert triage.scf_tolerance_guard == 1e-5

def test_extract_topographic_extrema():
    # Populate test triage results directly
    triage = TorqMACETriage.__new__(TorqMACETriage)
    triage.triage_results = [
        {"dihedral_angles": [0], "status": "converged", "relative_energy_kcal_mol": 0.0},
        {"dihedral_angles": [30], "status": "converged", "relative_energy_kcal_mol": 5.2},
        {"dihedral_angles": [60], "status": "converged", "relative_energy_kcal_mol": 1.1}
    ]
    extrema = triage.extract_topographic_extrema()
    assert len(extrema) >= 1

