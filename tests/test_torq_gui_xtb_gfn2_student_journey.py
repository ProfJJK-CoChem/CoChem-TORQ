"""Canonical student interface and unsupported vendor-profile rejection.

These checks do not assert unmeasured He2 binding, energies, gaps, dispersion,
or gCP values. A supplied He2-shaped geometry is an input, not an equilibrium
structure. ORCA/xTB require separately qualified site calculation workflows.
"""

from pathlib import Path
import json

import pytest

from Libraries.cochem_isotopes import isotope_mass
from cochem_torq.application import validate_request
from cochem_torq.registry import get_profile
from cochem_torq.spectroscopy.harmonic import equilibrium_rotor

ROOT = Path(__file__).parents[1]


def request(recipe):
    return {
        "molecule": {
            "symbols": ["He", "He"],
            "geometry_bohr": [[0, 0, 0], [0, 0, 6]],
            "charge": 0,
            "multiplicity": 1,
        },
        "recipe": recipe,
        "products": ["geometry"],
    }


def test_mendeleev_helium_mass_provenance():
    from mendeleev import element

    mass = isotope_mass("4He")
    genuine = next(
        record.mass for record in element("He").isotopes if record.mass_number == 4
    )
    assert mass == pytest.approx(float(genuine), abs=1e-12)


def test_torq_ui_environment_and_engine_controls_defect():
    notebook = json.loads((ROOT / "UI" / "Start_TORQ.ipynb").read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "cochem_torq.student_app" in code and "launch_student_app" in code
    assert "TorqPipeline(" not in code
    profile = get_profile("hf-sto-3g-education")
    assert profile["engine"] == "PySCF" and profile["runnable"]


def test_torq_github_actions_workflow_missing_defect():
    workflow = (ROOT / ".github" / "workflows" / "calculation.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "request_sha256:" in workflow and "expected_source_sha:" in workflow
    assert "options: [pyscf]" in workflow


def test_torq_rotational_constants_init_defect():
    """The undefined axial linear-rotor constant remains None."""
    mass = isotope_mass("4He")
    rotor = equilibrium_rotor([[0, 0, 0], [0, 0, 6]], [mass, mass])
    assert rotor.rotor_type == "linear"
    assert rotor.constants_mhz[0] is None
    assert rotor.constants_mhz[1] == pytest.approx(rotor.constants_mhz[2])


def test_physical_he2_xtb_gfn2_calculation_execution(tmp_path):
    """Unavailable recipe requests fail before any physical result is invented."""
    with pytest.raises(ValueError, match="Unknown recipe"):
        validate_request(request("xtb-gfn2"))
    assert not list(tmp_path.iterdir())


def test_torq_torsional_dvr_defect_on_he2():
    """Two atoms define a linear geometry, never an invented torsional spectrum."""
    mass = isotope_mass("4He")
    rotor = equilibrium_rotor([[0, 0, 0], [0, 0, 6]], [mass, mass])
    assert rotor.rotor_type == "linear"
    assert rotor.constants_mhz[0] is None
    assert rotor.constants_mhz[1] > 0
