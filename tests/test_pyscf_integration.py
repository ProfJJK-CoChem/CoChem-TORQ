"""Mandatory genuine-engine evidence for the bounded CPU teaching profile.

These tests invoke installed PySCF/geomeTRIC/DFTD4 and preserve authentic logs,
native checkpoints, tensors and displacement calculations in pytest's output
directory. No engine or numerical result is mocked. Passing these software and
derivative checks does not establish a universal chemical accuracy claim.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("geometric")

from cochem_torq.engines.pyscf_backend import BackendInputError, PySCFBackend


pytestmark = pytest.mark.real_engine


def request(symbols=None, geometry=None, *, name="hf", basis="sto-3g", properties=None):
    return {
        "molecule": {"symbols": symbols or ["H", "H"],
                     "geometry_bohr": geometry or [[0, 0, -0.7], [0, 0, 0.7]],
                     "charge": 0, "multiplicity": 1},
        "method": {"name": name, "basis": basis, "reference": "restricted", "frozen_core": False},
        "properties": properties or ["energy", "gradient"],
        "settings": {"threads": 1},
    }


def complete(result):
    assert result["status"] == "complete", result["errors"]
    assert result["scf"]["converged"] is True
    assert result["stability"]["status"] == "stable"
    return result


@pytest.fixture(scope="module")
def backend():
    return PySCFBackend()


@pytest.fixture(scope="module")
def h2_result(backend, tmp_path_factory):
    directory = tmp_path_factory.mktemp("cochem_exec_h2_actual")
    return complete(backend.evaluate(request(properties=["energy", "gradient", "hessian", "dipole"]), directory))


@pytest.fixture(scope="module")
def water_result(backend, tmp_path_factory):
    directory = tmp_path_factory.mktemp("cochem_exec_water_actual")
    data = request(["O", "H", "H"], [[0, 0, 0], [0, 0, 2.15], [1.9, 0, -0.5]],
                   properties=["energy", "gradient", "hessian", "dipole"])
    return complete(backend.optimize(data, directory))


def test_h2_energy_matches_independent_two_electron_rhf_formula(h2_result):
    """Compute the two-electron closed-shell energy directly from AO integrals.

    At the symmetric H2 geometry the normalized bonding orbital is known by
    symmetry. This independent contraction checks the backend's energy label
    and units without recycling its SCF orbital coefficients or energy.
    """
    from pyscf import gto
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", unit="Bohr", basis="sto-3g", verbose=0)
    overlap = mol.intor("int1e_ovlp")
    coeff = np.ones(2) / np.sqrt(2 * (1 + overlap[0, 1]))
    one_electron = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e")
    expected = (2 * np.einsum("i,ij,j", coeff, one_electron, coeff)
                + np.einsum("i,j,k,l,ijkl", coeff, coeff, coeff, coeff, eri)
                + mol.energy_nuc())
    assert h2_result["energy_hartree"] == pytest.approx(expected, abs=2e-11)
    assert h2_result["energy_hartree"] == pytest.approx(-1.11671432506255, abs=2e-10)
    assert np.linalg.norm(h2_result["dipole_debye"]) < 1e-9


def test_actual_artifact_manifest_covers_native_checkpoint_and_arrays(h2_result):
    directory = Path(h2_result["manifest_path"]).parent
    manifest = json.loads((directory / "manifest.json").read_text())
    assert h2_result["manifest_sha256"] == sha256((directory / "manifest.json").read_bytes()).hexdigest()
    names = {entry["path"] for entry in manifest["artifacts"]}
    assert {"pyscf.log", "wavefunction.chk", "geometry-bohr.npy", "gradient-hartree-bohr.npy",
            "hessian-hartree-bohr2.npy", "result.json", "request.json"} <= names
    for entry in manifest["artifacts"]:
        path = directory / entry["path"]
        assert entry["sha256"] == sha256(path.read_bytes()).hexdigest()
        assert entry["size_bytes"] == path.stat().st_size
    assert "converged SCF energy" in (directory / "pyscf.log").read_text()
    assert np.array_equal(np.load(directory / "hessian-hartree-bohr2.npy"),
                          np.asarray(h2_result["hessian_hartree_bohr2"]))


def test_hf_gradient_and_hessian_against_two_independent_difference_scales(backend, h2_result, tmp_path):
    geometry = np.asarray(h2_result["geometry_bohr"])
    expected_gradient = np.asarray(h2_result["gradient_hartree_bohr"]).ravel()
    expected_hessian = np.asarray(h2_result["hessian_hartree_bohr2"])
    residuals = []
    for step in (0.002, 0.001):
        gradient = np.empty(6)
        hessian = np.empty((6, 6))
        for component in range(6):
            actual = []
            for sign in (-1, 1):
                coords = geometry.copy().ravel()
                coords[component] += sign * step
                data = request(geometry=coords.reshape(2, 3).tolist())
                output = complete(backend.evaluate(data, tmp_path / f"h-{step}-c-{component}-s-{sign}"))
                actual.append(output)
            gradient[component] = (actual[1]["energy_hartree"] - actual[0]["energy_hartree"]) / (2 * step)
            hessian[:, component] = (np.asarray(actual[1]["gradient_hartree_bohr"]).ravel()
                                      - np.asarray(actual[0]["gradient_hartree_bohr"]).ravel()) / (2 * step)
        error_g = np.max(np.abs(gradient - expected_gradient))
        error_h = np.max(np.abs(hessian - expected_hessian))
        assert error_g < 4e-6
        assert error_h < 1e-5
        residuals.append((error_g, error_h))
    assert residuals[1][0] < residuals[0][0]
    assert residuals[1][1] < residuals[0][1]
    np.save(tmp_path / "independent-derivative-errors.npy", residuals)


def test_real_non_equilibrium_water_optimization_and_minimum(water_result):
    optimization = water_result["optimization"]
    assert optimization["converged"] is True
    assert optimization["final_gradient_verified"] is True
    assert optimization["evaluations"] > 1
    coordinates = np.asarray(water_result["geometry_bohr"])
    assert not np.allclose(coordinates, optimization["initial_geometry_bohr"])
    trajectory = json.loads((Path(water_result["manifest_path"]).parent / "optimization-trajectory.json").read_text())
    assert water_result["energy_hartree"] < trajectory[0]["energy_hartree"]
    assert np.max(np.abs(water_result["gradient_hartree_bohr"])) < 1.5e-5
    hessian = np.asarray(water_result["hessian_hartree_bohr2"])
    # Independent rigid-body projection verifies exactly 3 vibrational curvatures.
    from Libraries.cochem_torq_spectroscopy import resolve_isotopic_masses
    _, masses, _ = resolve_isotopic_masses(["16O", "1H", "1H"])
    root_mass = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / root_mass[:, None] / root_mass[None, :]
    center = np.average(coordinates, axis=0, weights=masses)
    displacement = coordinates - center
    rigid = []
    for axis in np.eye(3):
        rigid.append((np.tile(axis, (3, 1)) * np.sqrt(masses[:, None])).ravel())
        rigid.append((np.cross(displacement, axis) * np.sqrt(masses[:, None])).ravel())
    _, singular_values, vh = np.linalg.svd(np.asarray(rigid), full_matrices=True)
    assert sum(singular_values > 1e-8) == 6
    complement = vh[6:].T
    curvatures = np.linalg.eigvalsh(complement.T @ weighted @ complement)
    assert curvatures.shape == (3,)
    assert np.all(curvatures > 0)
    assert np.linalg.norm(water_result["dipole_debye"]) > 1
    from pyscf.lib import chkfile
    artifact_root = Path(water_result["manifest_path"]).parent
    native_initial = chkfile.load_mol(str(artifact_root / "initial" / "wavefunction.chk"))
    native_final = chkfile.load_mol(str(artifact_root / "final" / "wavefunction.chk"))
    assert np.allclose(native_initial.atom_coords(unit="Bohr"), optimization["initial_geometry_bohr"], atol=1e-12)
    assert np.allclose(native_final.atom_coords(unit="Bohr"), coordinates, atol=1e-12)


def test_real_h2_isotope_remassing_uses_same_born_oppenheimer_hessian(backend, h2_result, tmp_path):
    isotopic = complete(backend.evaluate(request(["2H", "2H"]), tmp_path / "deuterium"))
    assert isotopic["energy_hartree"] == pytest.approx(h2_result["energy_hartree"], abs=1e-12)
    from Libraries.cochem_torq_spectroscopy import resolve_isotopic_masses
    _, hydrogen, _ = resolve_isotopic_masses(["1H", "1H"])
    _, deuterium, _ = resolve_isotopic_masses(["2H", "2H"])
    k = h2_result["hessian_hartree_bohr2"][2][2]
    hydrogen_curvature = k * (1 / hydrogen[0] + 1 / hydrogen[1])
    deuterium_curvature = k * (1 / deuterium[0] + 1 / deuterium[1])
    assert hydrogen_curvature > deuterium_curvature > 0
    assert np.sqrt(deuterium_curvature / hydrogen_curvature) == pytest.approx(
        np.sqrt(hydrogen[0] / deuterium[0]), rel=1e-12)


def test_hf_cc_pvdz_is_real_separately_named_profile(backend, tmp_path):
    result = complete(backend.evaluate(request(basis="cc-pvdz", properties=["energy", "gradient", "hessian", "dipole"]), tmp_path))
    assert result["method"]["basis"] == "cc-pvdz"
    assert result["energy_hartree"] < -1.12
    assert result["hessian_evidence"]["derivative"] == "analytic"


@pytest.mark.parametrize("name", ["pbe-d4", "b3lyp-d4"])
def test_real_dft_d4_def2_svp_gradients_and_hessian(backend, tmp_path, name):
    pytest.importorskip("dftd4")
    data = request(["O", "H", "H"], [[0, 0, 0], [0, 0, 1.8], [1.7, 0, -0.6]],
                   name=name, basis="def2-svp", properties=["energy", "gradient", "hessian", "dipole"])
    result = complete(backend.evaluate(data, tmp_path / "parent"))
    assert result["method"]["dispersion"] == "d4"
    assert result["dispersion"]["variant"] == "D4(BJ)-EEQ-ATM"
    assert result["dispersion"]["parameters"]["s9"] == 1
    assert result["dispersion"]["energy_hartree"] < 0
    assert result["hessian_evidence"]["derivative"] == "centered_difference_of_analytic_gradient"
    geometry = np.asarray(data["molecule"]["geometry_bohr"])
    for step in (0.002, 0.001):
        outputs = []
        for sign in (-1, 1):
            displaced = deepcopy(data)
            coords = geometry.copy()
            coords[1, 2] += sign * step
            displaced["molecule"]["geometry_bohr"] = coords.tolist()
            displaced["properties"] = ["energy", "gradient"]
            outputs.append(complete(backend.evaluate(displaced, tmp_path / f"h-{step}-s-{sign}")))
        grad_fd = (outputs[1]["energy_hartree"] - outputs[0]["energy_hartree"]) / (2 * step)
        assert grad_fd == pytest.approx(result["gradient_hartree_bohr"][1][2], abs=1e-5)
        column_fd = (np.asarray(outputs[1]["gradient_hartree_bohr"]).ravel()
                     - np.asarray(outputs[0]["gradient_hartree_bohr"]).ravel()) / (2 * step)
        assert np.max(np.abs(column_fd - np.asarray(result["hessian_hartree_bohr2"])[:, 5])) < 2e-4


def test_real_mp2_cc_pvdz_gradient_and_numerical_hessian(backend, tmp_path):
    data = request(name="mp2", basis="cc-pvdz", properties=["energy", "gradient", "hessian"])
    result = complete(backend.evaluate(data, tmp_path / "parent"))
    assert result["mp2"]["correlation_energy_hartree"] < 0
    assert result["dipole_debye"] is None
    assert result["hessian_evidence"]["derivative"] == "centered_difference_of_analytic_gradient"
    assert result["hessian_evidence"]["displacement_count"] == 24
    geometry = np.asarray(data["molecule"]["geometry_bohr"])
    for step in (0.002, 0.001):
        outputs = []
        for sign in (-1, 1):
            displaced = deepcopy(data)
            coords = geometry.copy()
            coords[1, 2] += sign * step
            displaced["molecule"]["geometry_bohr"] = coords.tolist()
            displaced["properties"] = ["energy", "gradient"]
            outputs.append(complete(backend.evaluate(displaced, tmp_path / f"h-{step}-s-{sign}")))
        gradient_fd = (outputs[1]["energy_hartree"] - outputs[0]["energy_hartree"]) / (2 * step)
        assert gradient_fd == pytest.approx(result["gradient_hartree_bohr"][1][2], abs=5e-6)


def test_actual_scf_failure_never_returns_zero_energy_or_gradient(backend, tmp_path):
    data = request(["O", "H", "H"], [[0, 0, 0], [0, 0, 2.15], [1.9, 0, -.5]])
    data["settings"]["scf_max_cycle"] = 1
    result = backend.evaluate(data, tmp_path)
    assert result["status"] == "failed"
    assert result["scf"]["converged"] is False
    assert result["energy_hartree"] is None
    assert result["gradient_hartree_bohr"] is None
    assert result["errors"]
    assert Path(result["manifest_path"]).is_file()


def test_actual_restricted_instability_is_visible(backend, tmp_path):
    result = backend.evaluate(request(geometry=[[0, 0, -2], [0, 0, 2]]), tmp_path)
    assert result["scf"]["converged"] is True
    assert result["stability"]["status"] == "unstable"
    assert result["status"] == "partial"
    assert result["gradient_hartree_bohr"] is None
    assert result["errors"]


def test_actual_unconverged_optimization_retains_only_partial_result(backend, tmp_path):
    data = request(["O", "H", "H"], [[0, 0, 0], [0, 0, 2.15], [1.9, 0, -.5]])
    data["optimization"] = {"maxsteps": 1}
    result = backend.optimize(data, tmp_path)
    assert result["status"] == "partial"
    assert result["optimization"]["converged"] is False
    assert result["energy_hartree"] is not None
    assert result["optimization"]["stationary_point"] == "unclassified_until_hessian_analysis"
    assert result["errors"]


def test_actual_ion_dipole_translation_has_net_charge_origin_term(backend, tmp_path):
    from pyscf.data import nist
    data = request(["He", "H"], [[0, 0, 0], [0, 0, 1.5]],
                   properties=["energy", "gradient", "dipole"])
    data["molecule"]["charge"] = 1
    original = complete(backend.evaluate(data, tmp_path / "original"))
    shift = np.array([1.0, -2.0, 3.0])
    translated = deepcopy(data)
    translated["molecule"]["geometry_bohr"] = (np.asarray(data["molecule"]["geometry_bohr"]) + shift).tolist()
    result = complete(backend.evaluate(translated, tmp_path / "translated"))
    assert result["energy_hartree"] == pytest.approx(original["energy_hartree"], abs=2e-11)
    assert np.asarray(result["dipole_debye"]) - np.asarray(original["dipole_debye"]) == pytest.approx(
        shift * nist.AU2DEBYE, abs=2e-9)


@pytest.mark.parametrize("update", [
    {"method": {"name": "revdsd", "basis": "cc-pvdz"}},
    {"method": {"name": "mp2", "basis": "cc-pvdz"}, "properties": ["dipole"]},
    {"molecule": {"symbols": ["H"], "geometry_bohr": [[0, 0, 0]], "multiplicity": 2}},
])
def test_unsupported_tuple_rejected_before_engine_execution(backend, tmp_path, update):
    data = request()
    data.update(update)
    with pytest.raises(BackendInputError):
        backend.evaluate(data, tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonempty_workspace_rejected_without_overwriting_actual_artifacts(backend, h2_result):
    directory = Path(h2_result["manifest_path"]).parent
    original = (directory / "wavefunction.chk").read_bytes()
    with pytest.raises(BackendInputError, match="empty"):
        backend.evaluate(request(), directory)
    assert (directory / "wavefunction.chk").read_bytes() == original
