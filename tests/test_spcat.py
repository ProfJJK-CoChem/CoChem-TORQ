import logging
logger = logging.getLogger(__name__)
import hashlib  # SHA-256 artifact provenance tracking
import pytest
import json
from pathlib import Path
from Libraries.cochem_spcat_bridge import TorqSpcatBridge

def test_torq_spcat_bridge_init(tmp_path) -> None:
    tensor_file = tmp_path / "tensor.h5"
    tensor_file.touch()
    
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.touch()
    
    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    assert bridge.temperature_k == 298.15
    assert bridge.mpqc_file == Path(mpqc_file)

def test_torq_spcat_bridge_extract_orca(tmp_path) -> None:
    tensor_file = tmp_path / "tensor.h5"
    tensor_file.touch()
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text("FINAL SINGLE POINT ENERGY -76.123\n")
    
    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    q_rot, q_vib, q_total = bridge.calculate_partition_functions()
    assert q_rot > 0.0
    assert q_vib >= 1.0
