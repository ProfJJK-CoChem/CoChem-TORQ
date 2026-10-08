"""Real NH3 saddle, displaced minima and authenticated checkpoint evidence.

The HF/STO-3G quantities are numerical software checks, never experimental
accuracy references or a substitute for a qualified IRC/connectivity campaign.
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.engines.diagnostics import (
    electronic_diagnostics,
    stationary_point_diagnostics,
)
from cochem_torq.engines.pathway import PathwayRequest, locate_saddle_candidate
from cochem_torq.engines.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.real_engine


def ammonia_request():
    radius = 1.85
    angles = np.arange(3) * 2 * np.pi / 3
    return {
        "molecule": {
            "symbols": ["N", "H", "H", "H"],
            "geometry_bohr": [
                [0.0, 0.0, 0.0],
                *[
                    [float(radius * np.cos(angle)), float(radius * np.sin(angle)), 0.0]
                    for angle in angles
                ],
            ],
            "charge": 0,
            "multiplicity": 1,
            "atom_ids": ["n", "h1", "h2", "h3"],
            "isotopes": [14, 1, 1, 1],
        },
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
        },
    }


@pytest.fixture(scope="module")
def saddle_result(tmp_path_factory):
    result = locate_saddle_candidate(
        ammonia_request(), tmp_path_factory.mktemp("cochem_exec_nh3_saddle_actual")
    )
    assert result["status"] == "available", result["errors"]
    return result


def test_actual_saddle_hessian_stationarity_and_explicit_irc_limit(saddle_result):
    result = saddle_result
    diagnostic = result["stationary_evidence"]["saddle"]
    assert result["optimizer"]["converged"]
    assert result["optimizer"]["evaluation_count"] > 1
    assert diagnostic["classification"] == "first_order_saddle_candidate"
    assert diagnostic["maximum_gradient_hartree_bohr"] < 1.5e-5
    assert diagnostic["external_residual_relative"] < 1e-4
    assert diagnostic["imaginary_mode_count"] == 1
    assert len(diagnostic["signed_frequencies_cm1"]) == 6
    assert (
        np.count_nonzero(np.asarray(diagnostic["signed_frequencies_cm1"]) < -0.1) == 1
    )
    assert result["ts_verified"] is False
    assert result["irc"]["status"] == "unavailable"
    assert "integrator" in result["irc"]["reason"]
    assert result["components"]["saddle"]["stability"]["status"] == "stable"
    assert result["method"]["name"] == "hf"


def test_actual_both_direction_minima_have_opposite_umbrella_geometry(saddle_result):
    signed_heights = []
    energies = []
    for label in ["negative", "positive"]:
        endpoint = saddle_result["downhill_endpoints"][label]
        native = saddle_result["components"][f"endpoint_{label}"]
        assert native["optimization"]["converged"]
        assert native["optimization"]["final_gradient_verified"]
        assert (
            saddle_result["stationary_evidence"][f"endpoint_{label}"]["classification"]
            == "minimum"
        )
        assert endpoint["electronic_barrier_hartree"] > 0
        assert endpoint["irc_trajectory"] is False
        assert (
            endpoint["mapped_chemical_endpoint_identity"]
            == "unverified_requires_review"
        )
        geometry = np.asarray(endpoint["geometry_bohr"])
        plane_normal = np.cross(geometry[2] - geometry[1], geometry[3] - geometry[1])
        signed_heights.append(
            float(
                np.dot(geometry[0] - geometry[1], plane_normal)
                / np.linalg.norm(plane_normal)
            )
        )
        energies.append(endpoint["energy_hartree"])
        assert native["method"] == saddle_result["components"]["saddle"]["method"]
        assert native["molecule"]["atom_ids"] == ["n", "h1", "h2", "h3"]
    assert signed_heights[0] * signed_heights[1] < 0
    assert abs(signed_heights[0]) > 0.5
    assert energies[0] == pytest.approx(energies[1], abs=2e-8)


def test_actual_checkpoint_diagnostics_use_actual_orbitals_and_preserve_missing_cc(
    saddle_result,
):
    from pyscf import lib

    native = saddle_result["components"]["saddle"]
    diagnostic = electronic_diagnostics(native)
    checkpoint = Path(native["manifest_path"]).parent / "wavefunction.chk"
    scf_data = lib.chkfile.load(str(checkpoint), "scf")
    assert diagnostic["occupations"] == np.asarray(scf_data["mo_occ"]).tolist()
    assert (
        diagnostic["orbital_energies_hartree"]
        == np.asarray(scf_data["mo_energy"]).tolist()
    )
    assert diagnostic["electron_count"] == 10
    assert diagnostic["homo_lumo_gap_hartree"] > 0
    assert diagnostic["spin_squared"] == 0.0
    assert "determinant identity" in diagnostic["spin_squared_definition"]
    assert diagnostic["automatic_model_or_state_changes"] is False
    assert set(diagnostic["unavailable"]) == {
        "T1",
        "D1",
        "correlated_natural_occupations",
    }


def test_changed_native_claim_rejected_against_authentic_retained_result(saddle_result):
    native = deepcopy(saddle_result["components"]["saddle"])
    native["energy_hartree"] = np.nextafter(native["energy_hartree"], np.inf)
    with pytest.raises(RuntimeError, match="authenticated retained result"):
        electronic_diagnostics(native)


def test_corrupt_native_checkpoint_rejected_before_scientific_diagnostics(
    saddle_result, tmp_path
):
    native = deepcopy(saddle_result["components"]["saddle"])
    original = Path(native["manifest_path"]).parent
    copied = tmp_path / "copied-native-artifacts"
    shutil.copytree(original, copied)
    native["manifest_path"] = str(copied / "manifest.json")
    checkpoint = copied / "wavefunction.chk"
    with checkpoint.open("ab") as stream:
        stream.write(b"deliberately-corrupted-integrity-test")
    with pytest.raises(RuntimeError, match="artifact bytes"):
        electronic_diagnostics(native)


@pytest.mark.parametrize(
    "change",
    ["unsupported_method", "implicit_atom_ids", "unchecked_stability", "budget"],
)
def test_pathway_rejects_unsupported_state_or_unbounded_requests(change):
    request = ammonia_request()
    if change == "unsupported_method":
        request["method"]["name"] = "mp2"
    elif change == "implicit_atom_ids":
        request["molecule"].pop("atom_ids")
    elif change == "unchecked_stability":
        request["method"]["settings"] = {"check_stability": False}
    else:
        request["maxsteps"] = 151
    with pytest.raises(ValueError):
        PathwayRequest.model_validate(request)


def test_actual_nonstationary_hessian_never_certifies_a_saddle(tmp_path):
    from Libraries.cochem_isotopes import isotope_mass

    request = ammonia_request()
    for coordinate in request["molecule"]["geometry_bohr"][1:]:
        coordinate[0] *= 0.9
        coordinate[1] *= 0.9
    native = PySCFBackend().evaluate(
        {**request, "properties": ["energy", "gradient", "hessian"]},
        tmp_path / "actual-nonstationary",
    )
    assert native["status"] == "complete", native["errors"]
    diagnostic = stationary_point_diagnostics(
        native, [isotope_mass("14N"), *[isotope_mass("1H")] * 3]
    )
    assert diagnostic["maximum_gradient_hartree_bohr"] > 1.5e-5
    assert diagnostic["classification"] == "not_stationary"
    assert diagnostic["ts_verified"] is False


def test_actual_saddle_bundle_retains_initial_hessian_and_optimizer_trajectory(
    saddle_result,
):
    directory = Path(saddle_result["manifest_path"]).parent
    raw_hessian = np.loadtxt(directory / "solver-initial-hessian-hartree-bohr2.txt")
    initial = np.asarray(
        saddle_result["components"]["initial"]["hessian_hartree_bohr2"]
    )
    assert np.allclose(raw_hessian, initial, atol=1e-10, rtol=0)
    trajectory = json.loads(
        (directory / "saddle-optimization-trajectory.json").read_text()
    )
    assert len(trajectory) == saddle_result["optimizer"]["evaluation_count"]
    assert len(trajectory) > 1
