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
    
    triage = TorqMACETriage(grid_filepath=str(grid_file))
    assert triage.symbols == ["H", "H"]
    assert len(triage.grid_points) == 2
    assert triage.batch_size in (16, 512)

def test_extract_topographic_extrema():
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": []
    }
    # Mock triage results directly
    triage = TorqMACETriage.__new__(TorqMACETriage)
    triage.triage_results = [
        {"dihedral_angles": [0], "status": "converged", "relative_energy_kcal_mol": 0.0},
        {"dihedral_angles": [30], "status": "converged", "relative_energy_kcal_mol": 5.2},
        {"dihedral_angles": [60], "status": "converged", "relative_energy_kcal_mol": 1.1}
    ]
    extrema = triage.extract_topographic_extrema()
    assert len(extrema) >= 1
