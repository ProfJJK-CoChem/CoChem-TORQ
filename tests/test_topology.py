import logging
logger = logging.getLogger(__name__)
import pytest
import glob
import re
from pathlib import Path
from Libraries.cochem_torq_topology import TorqTopology, should_apply_counterpoise, route_method_track

def test_v4_tier_mapping() -> None:
    syms = ["C", "H", "H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.09], [1.02, 0.0, -0.36], [-0.51, 0.89, -0.36]]
    topo = TorqTopology(syms, coords, is_complex=False)
    
    t1 = topo.generate_cascade_parameters(tier="T1-10s")
    assert t1["tier"] == "T1-10s"
    assert t1["engine"] == "MPQC"
    assert "! r2SCAN-3c" in t1["keywords"]
    
    t2 = topo.generate_cascade_parameters(tier="T2-1m")
    assert t2["tier"] == "T2-1m"
    assert "! wB97X-D4" in t2["keywords"]
    
    t3 = topo.generate_cascade_parameters(tier="T3-1h")
    assert t3["tier"] == "T3-1h"
    assert "! CCSD(T)-F12" in t3["keywords"]
    
    t4 = topo.generate_cascade_parameters(tier="T4-1mo")
    assert t4["tier"] == "T4-1mo"
    assert t4["engine"] == "CFOUR"
    assert "! CCSD(T)" in t4["keywords"]

def test_counterpoise_rules() -> None:
    # Non-augmented triple-zeta -> True
    assert should_apply_counterpoise("cc-pVTZ", "B3LYP") is True
    assert should_apply_counterpoise("def2-TZVP", "wB97X-D4") is True
    assert should_apply_counterpoise("def2-TZVPP", "r2SCAN") is True
    
    # Augmented/diffuse basis set -> False
    assert should_apply_counterpoise("aug-cc-pVTZ", "B3LYP") is False
    assert should_apply_counterpoise("aug-cc-pVQZ", "wB97X-D4") is False
    assert should_apply_counterpoise("def2-TZVPd", "DFT") is False
    assert should_apply_counterpoise("ma-def2-TZVP", "DFT") is False
    assert should_apply_counterpoise("jun-cc-pVTZ", "DFT") is False
    assert should_apply_counterpoise("def2-TZVP", "DFT") is True
    assert should_apply_counterpoise("cc-pVTZ", "DFT") is True
    
    # CBS composite rows -> False
    assert should_apply_counterpoise("cc-pVTZ-F12", "CCSD(T)-F12/CBS") is False
    assert should_apply_counterpoise("cc-pVTZ", "W1-F12") is False

def test_topology_cascade_counterpoise_integration() -> None:
    syms = ["O", "H", "H", "O", "H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58],
              [3.0, 0.0, 0.0], [3.0, 0.75, 0.58], [3.0, -0.75, 0.58]]
    topo_complex = TorqTopology(syms, coords, is_complex=True)
    
    # Non-aug TZ basis set -> CP appended
    p1 = topo_complex.generate_cascade_parameters(tier="T2-1m", basis_set="def2-TZVP", method="wB97X-D4")
    assert p1["bsse_correction"] == "Counterpoise"
    assert "! CP" in p1["keywords"]
    
    # Augmented basis set -> CP prohibited
    p2 = topo_complex.generate_cascade_parameters(tier="T2-1m", basis_set="aug-cc-pVTZ", method="wB97X-D4")
    assert p2["bsse_correction"] is None
    assert "! CP" not in p2["keywords"]

    # CBS composite row -> CP prohibited
    p3 = topo_complex.generate_cascade_parameters(tier="T3-1h", basis_set="cc-pVTZ-F12", method="CCSD(T)-F12/CBS")
    assert p3["bsse_correction"] is None
    assert "! CP" not in p3["keywords"]

def test_route_method_track() -> None:
    # CCSD(T) VPT2/analytic Hessians -> CFOUR Track
    assert route_method_track("CCSD(T)", is_anharmonic=True, n_atoms=5) == "CFOUR"
    assert route_method_track("CFOUR", is_anharmonic=True, n_atoms=4) == "CFOUR"
    
    # DFT / SCF / F12 harmonic -> MPQC Track
    assert route_method_track("r2SCAN-3c", is_anharmonic=True, n_atoms=10) == "MPQC"
    assert route_method_track("wB97X-D4", is_anharmonic=False, n_atoms=15) == "MPQC"
    assert route_method_track("CCSD(T)-F12", is_anharmonic=False, n_atoms=10) == "MPQC"
    
    # CCSD(T)-F12 numerical VPT2 for N <= 6 -> MPQC Track
    assert route_method_track("CCSD(T)-F12", is_anharmonic=True, n_atoms=5) == "MPQC"
    
    # CCSD(T)-F12 numerical VPT2 for N > 6 -> ValueError (penalty abortion)
    with pytest.raises(ValueError) as exc_info:
        route_method_track("CCSD(T)-F12", is_anharmonic=True, n_atoms=10)
    assert "aborted for system size N=10 > 6" in str(exc_info.value)
    assert "36N^2" in str(exc_info.value)

def test_zero_mock_code_in_libraries() -> None:
    lib_dir = Path(__file__).parent.parent / "Libraries"
    py_files = list(lib_dir.glob("*.py"))
    assert len(py_files) > 0, "No library python files found!"
    
    prohibited_patterns = [
        r"\bmock\b",
        r"\bplaceholder\b",
        r"\bdummy_data\b",
        r"return\s+42\b",
        r"return\s+['\"]mock['\"]"
    ]
    
    violations = []
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern in prohibited_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Exclude comments explaining prohibition
                    if "# Exclude" in line or "prohibit" in line.lower():
                        continue
                    violations.append(f"{py_file.name}:{line_num}: {line.strip()}")
                    
    assert len(violations) == 0, f"Found mock code violations in Libraries: {violations}"
