import pytest
import json
from pathlib import Path
from Libraries.cochem_spcat_bridge import TorqSpcatBridge

def test_spcat_bridge(tmp_path):
    tensor_file = tmp_path / "tensor.json"
    orca_file = tmp_path / "orca.out"
    
    tensor_data = {
        "point_id": "001",
        "is_linear": False,
        "tensors": {
            "rotational_constants_MHz": {"A": 150000.0, "B": 25000.0, "C": 21000.0}
        }
    }
    tensor_file.write_text(json.dumps(tensor_data))
    orca_file.write_text("FINAL SINGLE POINT ENERGY -76.123\n")
    
    bridge = TorqSpcatBridge(str(tensor_file), str(orca_file), temperature_k=298.15)
    q_rot, q_vib, q_total = bridge.calculate_partition_functions()
    assert q_rot > 0.0
    assert q_vib >= 1.0
