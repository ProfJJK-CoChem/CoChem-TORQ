"""
Unit and Integration Tests for CoChem-TORQ Quench & Clash Evasion System
========================================================================
Validates Stage 2.0-2.1 Clash Evasion System, Soft Quench relaxation,
Jiggle Quench micro-randomization, and Mendeleev dynamic property resolution.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_torq_quench import (
    TorqQuenchGovernor,
    detect_covalent_clashes,
    execute_jiggle_quench,
    execute_soft_quench,
    get_atomic_mass,
    get_covalent_radius,
)


def test_get_covalent_radius_and_atomic_mass() -> None:
    """Verifies dynamic Mendeleev property retrieval and table consistency."""
    r_c = get_covalent_radius("C")
    assert 0.60 <= r_c <= 0.85
    r_h = get_covalent_radius("H")
    assert 0.25 <= r_h <= 0.45

    mass_c = get_atomic_mass("C")
    assert pytest.approx(12.011, rel=1e-2) == mass_c
    mass_h = get_atomic_mass("H")
    assert pytest.approx(1.008, rel=1e-2) == mass_h


def test_detect_covalent_clashes_clean() -> None:
    """Ensures no false positive clashes on equilibrium ethane structure."""
    symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [-1.15, 1.02, 0.0],
            [-1.15, -0.51, 0.88],
            [-1.15, -0.51, -0.88],
            [1.15, 1.02, 0.0],
            [1.15, -0.51, 0.88],
            [1.15, -0.51, -0.88],
        ],
        dtype=np.float64,
    )
    clashes = detect_covalent_clashes(symbols, coords, clash_ratio=0.70)
    assert len(clashes) == 0


def test_detect_covalent_clashes_and_soft_quench() -> None:
    """Verifies clash detection and soft quench relaxation."""
    symbols = ["C", "C", "H", "H"]
    coords = np.array(
        [
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.00, 0.20, 0.0],
            [0.00, 0.35, 0.0],
        ],
        dtype=np.float64,
    )

    initial_clashes = detect_covalent_clashes(symbols, coords, clash_ratio=0.70)
    assert len(initial_clashes) >= 1
    i, j, dist, thresh = initial_clashes[0]
    assert i == 2
    assert j == 3
    assert dist == pytest.approx(0.15, abs=1e-3)
    assert dist < thresh

    quench_res = execute_soft_quench(
        symbols=symbols,
        coordinates=coords,
        frozen_dihedrals=[(2, 0, 1, 3)],
        max_steps=50,
        damping=0.2,
    )

    assert quench_res["converged"] is True
    assert quench_res["final_clash_count"] == 0
    assert quench_res["steps_taken"] > 0
    relaxed_coords = quench_res["relaxed_coordinates"]
    relaxed_dist = np.linalg.norm(relaxed_coords[2] - relaxed_coords[3])
    assert relaxed_dist > 0.40


def test_execute_soft_quench_bypass() -> None:
    """Verifies that an unclashed system bypasses relaxation cleanly."""
    symbols = ["C", "H"]
    coords = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    res = execute_soft_quench(symbols, coords)
    assert res["converged"] is True
    assert res["steps_taken"] == 0
    assert res["method"] == "soft_quench_bypass"


def test_execute_jiggle_quench() -> None:
    """Verifies jiggle quench micro-randomization and clash relief."""
    symbols = ["C", "C", "H", "H"]
    coords = np.array(
        [
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.00, 0.20, 0.0],
            [0.00, 0.35, 0.0],
        ],
        dtype=np.float64,
    )

    jiggle_res = execute_jiggle_quench(
        symbols=symbols,
        coordinates=coords,
        jiggle_amplitude=0.03,
        max_steps=50,
        seed=12345,
    )
    assert (
        jiggle_res["final_clash_count"] < jiggle_res["initial_clash_count"]
        or jiggle_res["converged"]
    )
    assert jiggle_res["method"] == "jiggle_quench"


def test_quench_governor_class() -> None:
    """Verifies OOP TorqQuenchGovernor interface."""
    symbols = ["C", "C", "H", "H"]
    coords = np.array(
        [
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.00, 0.20, 0.0],
            [0.00, 0.35, 0.0],
        ],
        dtype=np.float64,
    )
    gov = TorqQuenchGovernor(symbols=symbols, frozen_dihedrals=[(2, 0, 1, 3)])
    clashes = gov.check_clashes(coords)
    assert len(clashes) >= 1

    res = gov.quench(coords, max_steps=50)
    assert res["converged"] is True
    assert res["final_clash_count"] == 0

    jres = gov.jiggle_quench(coords, jiggle_amplitude=0.02, max_steps=50)
    assert jres["final_clash_count"] < jres["initial_clash_count"] or jres["converged"]


def test_dimension_mismatch_validation() -> None:
    """Verifies that invalid dimension shapes raise ValueError."""
    symbols = ["C", "H"]
    invalid_coords = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        detect_covalent_clashes(symbols, invalid_coords)


def test_anti_spoofing_ast_integrity() -> None:
    """AST audit to verify zero mocks, stubs, or empty pass blocks."""
    quench_file = (
        Path(__file__).resolve().parent.parent / "Libraries" / "cochem_torq_quench.py"
    )
    assert quench_file.exists()
    tree = ast.parse(quench_file.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Name) and node.exc.id == "NotImplementedError":
                pytest.fail("Found forbidden NotImplementedError")
            if (
                isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "NotImplementedError"
            ):
                pytest.fail("Found forbidden NotImplementedError call")
        if isinstance(node, ast.FunctionDef):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                pytest.fail(f"Found empty pass block in function {node.name}")


@pytest.mark.parametrize("symbol", ["H", "C", "O", "Fe", "Og"])
def test_named_pyykko_radius_matches_actual_database_and_retains_identity(symbol):
    """Genuine provider lookup with actual version, file digest and named units."""
    import hashlib
    from importlib.metadata import version

    from mendeleev import element
    from mendeleev.db import get_package_dbpath

    from Libraries.cochem_torq_quench import get_covalent_radius_record

    actual = element(symbol)
    record = get_covalent_radius_record(symbol)
    assert record["property"] == "covalent_radius_pyykko"
    assert record["database_value_pm"] == float(actual.covalent_radius_pyykko)
    assert record["radius_angstrom"] == float(actual.covalent_radius_pyykko) / 100.0
    assert get_covalent_radius(symbol) == record["radius_angstrom"]
    assert record["uncertainty_angstrom"] is None
    assert record["source"]["distribution_version"] == version("mendeleev")
    assert (
        record["source"]["database_sha256"]
        == hashlib.sha256(Path(get_package_dbpath()).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("value", [None, 0.0, -1.0, np.nan, np.inf, True, 1j])
def test_absent_or_invalid_named_radius_never_gets_a_substitute(value):
    """Pure numeric/absence contract checks; no provider is replaced or emulated."""
    from Libraries.cochem_torq_quench import _validate_pyykko_radius

    with pytest.raises(ValueError, match="Named Pyykkö"):
        _validate_pyykko_radius(value, "C")


def test_unknown_element_has_no_hardcoded_radius():
    with pytest.raises(ValueError):
        get_covalent_radius("UnknownElement")
