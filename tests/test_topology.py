import ast
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_torq_topology import (
    TorqTopology,
    route_method_track,
    should_apply_counterpoise,
)


def test_v4_tier_mapping() -> None:
    syms = ["C", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.02, 0.0, -0.36],
        [-0.51, 0.89, -0.36],
    ]
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
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.75, 0.58],
        [0.0, -0.75, 0.58],
        [3.0, 0.0, 0.0],
        [3.0, 0.75, 0.58],
        [3.0, -0.75, 0.58],
    ]
    topo_complex = TorqTopology(syms, coords, is_complex=True)

    # Non-aug TZ basis set -> CP appended
    p1 = topo_complex.generate_cascade_parameters(
        tier="T2-1m", basis_set="def2-TZVP", method="wB97X-D4"
    )
    assert p1["bsse_correction"] == "Counterpoise"
    assert "! CP" in p1["keywords"]

    # Augmented basis set -> CP prohibited
    p2 = topo_complex.generate_cascade_parameters(
        tier="T2-1m", basis_set="aug-cc-pVTZ", method="wB97X-D4"
    )
    assert p2["bsse_correction"] is None
    assert "! CP" not in p2["keywords"]

    # CBS composite row -> CP prohibited
    p3 = topo_complex.generate_cascade_parameters(
        tier="T3-1h", basis_set="cc-pVTZ-F12", method="CCSD(T)-F12/CBS"
    )
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
    """Inspect executable imports/calls rather than scientific-policy prose."""
    lib_dir = Path(__file__).parent.parent / "Libraries"
    py_files = list(lib_dir.glob("*.py"))
    assert len(py_files) > 0, "No library python files found!"

    violations = []
    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported_modules = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            for name in imported_modules:
                if name == "mock" or name.startswith("unittest.mock"):
                    violations.append(f"{py_file.name}:{node.lineno}: import {name}")
            if isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.id
                    if isinstance(called, ast.Name)
                    else called.attr
                    if isinstance(called, ast.Attribute)
                    else ""
                )
                if name in {"Mock", "MagicMock", "AsyncMock"}:
                    violations.append(f"{py_file.name}:{node.lineno}: {name}()")

    assert not violations, f"Found executable replacement code: {violations}"


def test_mendeleev_dynamic_queries_heavy_elements() -> None:
    from Libraries.cochem_torq_topology import (
        get_atomic_mass,
        get_atomic_number,
        get_isotopic_mass,
        get_pyykko_radius,
        get_vdw_radius,
    )

    # Test heavy elements Z > 54
    assert get_atomic_number("Au") == 79
    assert get_atomic_number("Pt") == 78
    assert get_atomic_number("U") == 92
    assert get_atomic_number("Xe") == 54
    assert get_atomic_number("H") == 1
    assert get_atomic_number("C") == 6

    # Test Pyykko radii
    r_au = get_pyykko_radius("Au")
    assert isinstance(r_au, float)
    assert 1.0 < r_au < 2.0
    r_h = get_pyykko_radius("H")
    assert 0.25 < r_h < 0.40

    # Test atomic masses
    m_c = get_atomic_mass("C")
    assert 12.0 < m_c < 12.02
    m_au = get_atomic_mass("Au")
    assert 196.0 < m_au < 198.0

    # Test VdW radii
    vdw_c = get_vdw_radius("C")
    assert 1.5 < vdw_c < 2.0

    # Test isotope queries
    assert get_atomic_number("13C") == 6
    assert get_atomic_number("C-13") == 6
    assert get_atomic_number("C13") == 6
    assert get_atomic_number("D") == 1
    assert get_atomic_number("T") == 1
    assert get_atomic_number("H-2") == 1
    m_13c = get_atomic_mass("13C")
    assert 13.00 < m_13c < 13.01
    assert get_atomic_mass("C-13") == m_13c
    assert get_atomic_mass("C13") == m_13c
    m_d = get_atomic_mass("D")
    assert 2.01 < m_d < 2.02
    assert get_atomic_mass("H-2") == m_d
    m_t = get_atomic_mass("T")
    assert 3.01 < m_t < 3.02
    assert get_isotopic_mass("C", 13) == m_13c
    assert get_isotopic_mass("H", 2) == m_d
    m_18o = get_atomic_mass("18O")
    assert 17.99 < m_18o < 18.01
    assert get_atomic_mass("O-18") == m_18o
    assert get_atomic_mass("O18") == m_18o

    # Test invalid symbols raise ValueError
    with pytest.raises(Exception):
        get_atomic_mass("FakeElement")
    with pytest.raises(Exception):
        get_atomic_number("FakeElement")
    with pytest.raises(Exception):
        get_pyykko_radius("FakeElement")
    with pytest.raises(Exception):
        get_vdw_radius("FakeElement")


def test_dihedral_detection_5_options() -> None:
    syms = ["C", "C", "C", "C"]
    coords_ref = np.array(
        [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.0, 1.45, 0.0], [3.5, 1.45, 0.0]],
        dtype=np.float64,
    )

    # Rotate 4th atom around C2-C3 bond
    coords_rotated = coords_ref.copy()
    coords_rotated[3] = [2.0, 1.45, 1.5]

    topo = TorqTopology(syms, coords_rotated)

    # Option 1: Z-Matrix diff
    moving_z = topo.detect_via_zmatrix_diff(coords_ref)
    assert isinstance(moving_z, list)
    assert len(moving_z) > 0

    # Option 2: Kabsch RMSD
    moving_k = topo.detect_via_kabsch_rmsd(coords_ref)
    assert isinstance(moving_k, list)

    # Option 3: Graph theory (sever C1-C2 edge at (0, 1))
    subgraphs = topo.detect_via_graph_theory((0, 1))
    assert len(subgraphs) == 2
    assert set(subgraphs[0] + subgraphs[1]) == {0, 1, 2, 3}

    # Graph theory on non-existent edge raises ValueError
    with pytest.raises(ValueError, match="not found in graph"):
        topo.detect_via_graph_theory((0, 3))

    # Option 4: Coulomb variance
    moving_c = topo.detect_via_coulomb_variance(coords_ref)
    assert isinstance(moving_c, list)

    # Option 5: Manual override
    override = topo.detect_via_override([0, 1, 2, 3])
    assert override == [0, 1, 2, 3]

    # Manual override with invalid length raises ValueError (not AssertionError)
    with pytest.raises(ValueError, match="Manual override requires exactly 4 indices"):
        topo.detect_via_override([0, 1, 2])
    with pytest.raises(ValueError, match="Manual override requires exactly 4 indices"):
        topo.detect_via_override([0, 1, 2, 3, 4])


def test_heavy_element_coulomb_and_graph_support() -> None:
    # Test molecule with heavy elements (e.g. cis-platin / organogold complex)
    syms = ["Pt", "Cl", "Cl", "N", "N", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [0.0, 2.3, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [-2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    topo = TorqTopology(syms, coords)
    assert topo.graph.number_of_nodes() == 6

    coords_perturbed = coords.copy()
    coords_perturbed[5] = [-2.0, 0.0, 1.5]

    moving = topo.detect_via_coulomb_variance(coords_perturbed)
    assert isinstance(moving, list)


def test_cascade_parameters_custom_output_path(tmp_path: Path) -> None:
    syms = ["C", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.02, 0.0, -0.36],
        [-0.51, 0.89, -0.36],
    ]
    topo = TorqTopology(syms, coords)

    custom_file = tmp_path / "custom_run_params.json"
    res = topo.generate_cascade_parameters(tier="T2-1m", output_path=custom_file)
    assert custom_file.exists()
    assert res["tier"] == "T2-1m"
