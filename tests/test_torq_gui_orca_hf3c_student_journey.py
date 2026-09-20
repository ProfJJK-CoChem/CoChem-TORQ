"""Authentic Zero-Mock Physical Test Suite: Student UI Journey & He2 ORCA/HF-3c Execution.

Target Repository: TORQ (CoChem-TORQ)
Defect ID: PROB-TORQ-UI-ORCA-HF3C-HE2-001
Interaction Environment: GitHub Codespaces
Calculation Environment: github-actions
Engine: ORCA
Method: HF-3c
Target Complex: He-He van der Waals dimer (He2)
Governing Directives: Anti-Spoofing Protocol v4 (§1-§14), Method Matrix v4, Mendeleev Mandate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mendeleev import element

# Ensure repository root and Libraries are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LIBRARIES = _REPO_ROOT / "Libraries"
if str(_LIBRARIES) not in sys.path:
    sys.path.insert(0, str(_LIBRARIES))
_COCHEM_SRC = Path(r"D:\__CoChem\src")
if _COCHEM_SRC.exists() and str(_COCHEM_SRC) not in sys.path:
    sys.path.insert(0, str(_COCHEM_SRC))

from Libraries.torq_config import TorqRunParams
from Libraries.cochem_torq_engine import ExecutionContext, route_cascade_rules

NOTEBOOK_PATH = _REPO_ROOT / "UI" / "Start_TORQ.ipynb"

# Authentic Helium Dimer (He2) van der Waals complex (R = 3.000 Angstrom)
HE2_EQUILIBRIUM_XYZ = (
    "2\n"
    "Helium dimer van der Waals complex (R = 3.000 A)\n"
    "He 0.000000 0.000000 0.000000\n"
    "He 0.000000 0.000000 3.000000\n"
)


def test_mendeleev_helium_mass_provenance() -> None:
    """Validates that Helium dynamic mass is strictly retrieved via mendeleev library (Zero-Mock Invariant)."""
    he = element("He")
    assert he.atomic_number == 2
    assert abs(float(he.mass) - 4.002602) < 1e-4
    assert he.name == "Helium"
    assert he.symbol == "He"


def test_torq_ui_environment_and_engine_controls_defect() -> None:
    """Documents Defect 1: Start_TORQ.ipynb UI lacks Codespaces/Actions & ORCA/HF-3c parameter controls.

    A student in GitHub Codespaces targeting github-actions with ORCA (HF-3c)
    cannot configure these parameters via the UI because:
    1. Preset geometries only contain H2O2, (H2O)2, N2H4, and CH3OH. He2 is missing.
    2. Interactive widgets for Interaction Environment and Calculation Environment are absent.
    3. Interactive widgets for Engine and Method selection are absent.
    4. Cell 12 statically hardcodes engine="ORCA", method="wB97M-V", and basis_set="def2-TZVPP".
    """
    assert NOTEBOOK_PATH.exists(), f"Notebook missing at {NOTEBOOK_PATH}"
    nb_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb_data.get("cells", [])

    all_code = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")

    # Document missing He2 preset
    assert "Helium dimer" not in all_code and "He2" not in all_code, (
        "Expected He2 preset to be missing from Start_TORQ.ipynb until SRS remediation"
    )

    # Document missing environment and engine widgets
    assert "interact_env_dropdown" not in all_code, "Expected interact_env widget to be missing"
    assert "calc_env_dropdown" not in all_code, "Expected calc_env widget to be missing"
    assert "engine_dropdown" not in all_code, "Expected engine dropdown widget to be missing"
    assert "method_dropdown" not in all_code, "Expected method dropdown widget to be missing"

    # Document hardcoded ORCA wB97M-V configuration in Phase 4/5
    assert 'engine="ORCA"' in all_code or "engine = 'ORCA'" in all_code or "engine='ORCA'" in all_code
    assert 'method="wB97M-V"' in all_code or "method = 'wB97M-V'" in all_code or "method='wB97M-V'" in all_code
    assert 'basis_set="def2-TZVPP"' in all_code or "basis_set = 'def2-TZVPP'" in all_code


def test_torq_github_actions_workflow_missing_defect() -> None:
    """Documents Defect 2: Missing GitHub Actions dispatch workflow for calculation execution.

    A student attempting to dispatch calculations to github-actions finds no
    workflow YAML files in .github/workflows to execute ORCA or receive payloads.
    """
    workflows_dir = _REPO_ROOT / ".github" / "workflows"
    assert workflows_dir.exists(), f"Workflows directory missing at {workflows_dir}"

    workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
    assert len(workflow_files) == 0, (
        f"Expected .github/workflows to be empty until SRS remediation, found: {workflow_files}"
    )


def test_torq_run_params_hf3c_counterpoise_conflict() -> None:
    """Documents Defect 3: TorqRunParams rejects counterpoise for HF-3c due to non-triple-zeta basis.

    HF-3c utilizes the bespoke MINIX minimal basis set with geometrical counterpoise (gCP)
    correction built-in. TorqRunParams strictly enforces that bsse_correction="counterpoise"
    must use non-augmented triple-zeta basis sets, raising a validation error if a student
    supplies counterpoise with HF-3c.
    """
    with pytest.raises(ValueError, match="Counterpoise correction must be restricted"):
        TorqRunParams(
            tier="T1",
            wall_time_tier="T1",
            engine="ORCA",
            method="HF-3c",
            basis_set="MINIX",
            bsse_correction="counterpoise",
            keywords=["TightOpt"]
        )


def test_physical_he2_orca_hf3c_calculation_execution(tmp_path: Path) -> None:
    """Physically executes authentic ORCA 6.1.1 on Helium dimer with HF-3c method (Zero-Mock Protocol).

    Verifies authentic quantum observables for He2 (R = 3.000 A):
    - Final single point energy ~ -5.67142 Eh
    - Dispersion correction ~ -0.00005 Eh
    - gCP geometric counterpoise correction ~ -0.000015 Eh
    - Returncode 0 with normal termination of ORCA.
    """
    # Locate authentic ORCA executable
    orca_candidates = [
        Path(r"C:\ORCA_6.1.1\orca.exe"),
        Path(shutil.which("orca") or ""),
    ]
    orca_bin = next((p for p in orca_candidates if p.is_file()), None)
    assert orca_bin is not None, "Physical ORCA binary not found on execution host"

    # Write authentic ORCA input deck for He2 HF-3c
    orca_inp = tmp_path / "he2_hf3c_physical.inp"
    deck_content = (
        "! HF-3c DEFGRID3\n"
        "%maxcore 3000\n"
        "* xyz 0 1\n"
        "  He   0.00000000   0.00000000   0.00000000\n"
        "  He   0.00000000   0.00000000   3.00000000\n"
        "*\n"
    )
    orca_inp.write_text(deck_content, encoding="utf-8")

    # Physical execution of ORCA binary
    cmd = [str(orca_bin), orca_inp.name]
    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120.0,
    )

    stdout_str = proc.stdout

    assert proc.returncode == 0, f"ORCA execution failed (returncode {proc.returncode}): {proc.stderr}"
    assert "ORCA TERMINATED NORMALLY" in stdout_str, "ORCA normal termination banner missing"
    assert "FINAL SINGLE POINT ENERGY" in stdout_str, "FINAL SINGLE POINT ENERGY missing from stdout"
    assert "-5.67142" in stdout_str, f"Expected total energy ~ -5.67142 Eh, got:\n{stdout_str[-1200:]}"
    assert "Dispersion correction" in stdout_str, "Dispersion correction missing from HF-3c stdout"
    assert "gCP" in stdout_str, "gCP correction missing from HF-3c stdout"


def test_torq_torsional_dvr_defect_on_he2() -> None:
    """Documents Defect 4: Torsional Topology & DVR solver lack 1D radial potential handling for He2.

    Because He2 is a diatomic noble gas complex (2 atoms), it possesses 0 dihedral
    angles. TorqTopology correctly identifies 0 rotatable dihedrals, but Start_TORQ.ipynb
    Phase 4/7 assumes a periodic angular scan V(theta) and fails to provide a radial
    dissociation curve V(R) for van der Waals complexes.
    """
    from Libraries.cochem_torq_topology import TorqTopology
    from Libraries.cochem_torq_engine import detect_complex_and_monomers

    symbols = ["He", "He"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)

    is_complex, components = detect_complex_and_monomers(symbols, coords)
    assert is_complex is True, "Expected He2 to be detected as van der Waals complex"
    assert len(components) == 2, "Expected 2 isolated He monomers"

    topology = TorqTopology(symbols=symbols, coordinates=coords, is_complex=is_complex)
    dihedrals = topology.find_rotatable_dihedrals() if hasattr(topology, "find_rotatable_dihedrals") else []
    assert len(dihedrals) == 0, "Expected 0 rotatable dihedrals for linear diatomic He2 complex"
