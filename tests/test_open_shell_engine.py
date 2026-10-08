"""Genuine open-shell calculations and strict molecular/state admission."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.engines.open_shell import OpenShellPySCFBackend
from cochem_torq.engines.pyscf_backend import BackendInputError

pytestmark = pytest.mark.research


def _request(name="hf", *, oxygen=False, stability=True):
    return {
        "molecule": {
            "symbols": ["O", "O"] if oxygen else ["O", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.3 if oxygen else 1.9]],
            "charge": 0,
            "multiplicity": 3 if oxygen else 2,
        },
        "method": {"name": name, "basis": "sto-3g", "reference": "unrestricted"},
        "properties": ["energy", "gradient"],
        "settings": {
            "threads": 1,
            "scf_max_cycle": 300,
            "scf_energy_tolerance": 1e-12,
            "scf_gradient_tolerance": 1e-9,
            "dft_grid_level": 4,
            "check_stability": stability,
        },
    }


@pytest.fixture(scope="module")
def oh_result(tmp_path_factory):
    result = OpenShellPySCFBackend().evaluate(
        _request(), tmp_path_factory.mktemp("actual_oh_unrestricted")
    )
    assert result["status"] == "complete", result["errors"]
    return result


@pytest.mark.real_engine
def test_actual_oh_spin_occupations_stability_and_checkpoint_readback(oh_result):
    from pyscf import scf

    result = oh_result
    assert result["scf"]["electron_count"] == 9
    assert result["scf"]["alpha_electrons"] == 5
    assert result["scf"]["beta_electrons"] == 4
    assert result["scf"]["native_molecular_spin"] == 1
    assert result["molecule"]["multiplicity"] == 2
    assert result["spin"]["spin_squared"] > 0.75
    assert result["spin"]["spin_projection_applied"] is False
    assert result["spin"]["effective_multiplicity_is_state_label"] is False
    assert result["lowest_electronic_state_certified"] is False
    stability = result["stability"]
    assert stability["status"] == "stable"
    assert stability["reference_unchanged"] is True
    assert stability["candidate_orbitals_applied"] is False
    assert stability["checkpoint_sha256_before"] == stability["checkpoint_sha256_after"]
    assert stability["orbital_sha256_before"] == stability["orbital_sha256_after"]
    assert (
        stability["real_to_complex_status"]
        == "native_log_only_not_separately_extracted"
    )
    root = Path(result["manifest_path"]).parent
    checkpoint = root / "wavefunction.chk"
    saved = scf.chkfile.load(str(checkpoint), "scf")
    assert np.asarray(saved["mo_coeff"]).shape[0] == 2
    assert np.asarray(saved["mo_occ"]).sum(axis=1).tolist() == [5.0, 4.0]
    assert saved["e_tot"] == pytest.approx(result["energy_hartree"], abs=1e-12)
    for entry in result["artifacts"]["artifacts"]:
        path = root / entry["path"]
        assert sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]
    assert "converged SCF energy" in (root / "pyscf.log").read_text()
    assert result["hessian_hartree_bohr2"] is None
    assert result["unavailable_properties"]["spin_projected_energy"]["value"] is None


@pytest.mark.real_engine
def test_oh_analytic_gradient_against_two_full_energy_difference_scales(
    oh_result, tmp_path
):
    backend = OpenShellPySCFBackend()
    original = np.asarray(_request()["molecule"]["geometry_bohr"])
    for step in (0.002, 0.001):
        numerical = np.empty(6)
        for q in range(6):
            energies = []
            for sign in (-1, 1):
                request = _request(stability=False)
                geometry = original.copy().ravel()
                geometry[q] += sign * step
                request["molecule"]["geometry_bohr"] = geometry.reshape((2, 3)).tolist()
                request["properties"] = ["energy"]
                result = backend.evaluate(
                    request, tmp_path / f"step-{step}-q{q}-s{sign}"
                )
                assert result["status"] == "complete", result["errors"]
                assert result["scf"]["requested_multiplicity"] == 2
                energies.append(result["energy_hartree"])
            numerical[q] = (energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(
            numerical,
            np.asarray(oh_result["gradient_hartree_bohr"]).ravel(),
            atol=3e-6,
            rtol=0,
        )


@pytest.mark.real_engine
@pytest.mark.parametrize("name", ["pbe-d4", "b3lyp-d4"])
def test_triplet_uks_d4_actual_gradient_and_dispersion_against_energy_differences(
    tmp_path, name
):
    from dftd4.interface import DampingParam, DispersionModel

    backend = OpenShellPySCFBackend()
    request = _request(name, oxygen=True, stability=False)
    reference = backend.evaluate(request, tmp_path / "reference")
    assert reference["status"] == "complete", reference["errors"]
    assert reference["scf"]["alpha_electrons"] == 9
    assert reference["scf"]["beta_electrons"] == 7
    assert reference["spin"]["requested_spin_squared"] == 2.0
    assert reference["density_functional"]["gradient_grid_response"] is True
    assert reference["stability"]["status"] == "not_requested"
    assert (
        "unrestricted_reference_stability_not_requested" in reference["quality_flags"]
    )
    geometry = np.asarray(request["molecule"]["geometry_bohr"])
    native = DispersionModel(np.asarray([8, 8]), geometry, charge=0.0).get_dispersion(
        DampingParam(method=name[:-3], atm=True), grad=True
    )
    assert reference["dispersion"]["energy_hartree"] == pytest.approx(
        float(native["energy"]), abs=1e-13
    )
    np.testing.assert_allclose(
        reference["dispersion"]["gradient_hartree_bohr"],
        native["gradient"],
        atol=1e-13,
    )
    assert reference["dispersion"]["implementation"]["parameter_table_sha256"]
    for step in (0.002, 0.001):
        energies = []
        for sign in (-1, 1):
            displaced = deepcopy(request)
            displaced["molecule"]["geometry_bohr"][1][2] += sign * step
            displaced["properties"] = ["energy"]
            result = backend.evaluate(displaced, tmp_path / f"step{step}-s{sign}")
            assert result["status"] == "complete", result["errors"]
            energies.append(result["energy_hartree"])
        numerical = (energies[1] - energies[0]) / (2 * step)
        assert numerical == pytest.approx(
            reference["gradient_hartree_bohr"][1][2], abs=3e-5
        )


@pytest.mark.real_engine
def test_actual_external_instability_is_retained_without_state_replacement(tmp_path):
    result = OpenShellPySCFBackend().evaluate(
        _request("pbe", oxygen=True), tmp_path / "unstable-triplet"
    )
    assert result["status"] == "partial"
    assert result["energy_hartree"] is not None
    assert result["gradient_hartree_bohr"] is not None
    assert result["stability"]["status"] == "unstable"
    assert result["stability"]["internal_stable"] is True
    assert result["stability"]["external_stable"] is False
    assert result["stability"]["reference_unchanged"] is True
    assert result["stability"]["candidate_orbitals_applied"] is False
    assert result["molecule"]["multiplicity"] == 3
    assert result["spin"]["spin_projection_applied"] is False
    assert result["errors"][0]["code"] == "UNRESTRICTED_STATE_NOT_QUALIFIED"


@pytest.mark.real_engine
def test_actual_uhf_optimizer_preserves_state_and_native_step_checkpoints(tmp_path):
    request = _request()
    request["molecule"]["geometry_bohr"][1][2] = 2.3
    original = deepcopy(request)
    result = OpenShellPySCFBackend().optimize(request, tmp_path / "optimization")
    assert request == original
    assert result["status"] == "complete", result["errors"]
    assert result["optimization"]["converged"] is True
    assert result["optimization"]["final_gradient_verified"] is True
    assert result["optimization"]["initial_artifacts_unchanged"] is True
    assert (
        result["optimization"]["stationary_point"]
        == "unclassified_until_validated_unrestricted_hessian"
    )
    assert result["hessian_hartree_bohr2"] is None
    root = Path(result["manifest_path"]).parent
    trajectory = json.loads((root / "optimization-trajectory.json").read_text())
    assert len(trajectory) == result["optimization"]["evaluations"]
    assert len(trajectory) > 1
    for observation in trajectory:
        assert observation["alpha_electrons"] == 5
        assert observation["beta_electrons"] == 4
        assert observation["requested_multiplicity"] == 2
        assert observation["spin_projection_applied"] is False
        checkpoint = (
            Path(observation["checkpoint_artifact_directory"]) / "wavefunction.chk"
        )
        assert (
            sha256(checkpoint.read_bytes()).hexdigest()
            == observation["checkpoint_sha256"]
        )
    assert result["energy_hartree"] < trajectory[0]["energy_hartree"]
    assert result["spin"]["requested_multiplicity"] == 2
    assert np.max(np.abs(result["gradient_hartree_bohr"])) < 1.5e-5


@pytest.mark.real_engine
def test_nonconvergence_retains_raw_uhf_without_accepted_values(tmp_path):
    request = _request()
    request["settings"]["scf_max_cycle"] = 1
    result = OpenShellPySCFBackend().evaluate(request, tmp_path / "nonconverged")
    assert result["status"] == "failed"
    assert result["scf"]["converged"] is False
    assert result["energy_hartree"] is None
    assert result["gradient_hartree_bohr"] is None
    root = Path(result["manifest_path"]).parent
    assert (root / "pyscf.log").stat().st_size > 100
    assert (root / "wavefunction.chk").stat().st_size > 100


@pytest.mark.parametrize(
    "change",
    [
        ("molecule", "multiplicity", 1),
        ("molecule", "multiplicity", 3),
        ("molecule", "multiplicity", True),
        ("molecule", "charge", 3),
        ("molecule", "charge", True),
        ("molecule", "symbols", ["O", "Cl"]),
        ("molecule", "symbols", ["16O", "H"]),
        ("molecule", "isotopes", [9, 1]),
        ("method", "reference", "restricted"),
        ("method", "reference", "uks"),
        ("method", "name", "revdsd-pbep86-d4"),
        ("method", "density_fitting", True),
        ("method", "projection", "yamaguchi"),
        ("settings", "threads", True),
        ("settings", "threads", 9),
        ("settings", "memory_mb", 40000),
        ("settings", "scf_energy_tolerance", np.nan),
        ("settings", "spin_contamination_tolerance", False),
    ],
)
def test_unsupported_state_method_or_resource_fails_before_execution(tmp_path, change):
    request = _request()
    section, key, value = change
    request[section][key] = value
    with pytest.raises(BackendInputError):
        OpenShellPySCFBackend().evaluate(request, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_unsupported_hessian_and_complex_coordinates_have_no_values(tmp_path):
    request = _request()
    request["properties"] = ["energy", "hessian"]
    with pytest.raises(BackendInputError, match="Hessians"):
        OpenShellPySCFBackend().evaluate(request, tmp_path / "unsupported-hessian")
    request = _request()
    request["molecule"]["geometry_bohr"] = np.asarray(
        request["molecule"]["geometry_bohr"], dtype=complex
    )
    with pytest.raises(BackendInputError, match="complex"):
        OpenShellPySCFBackend().evaluate(request, tmp_path / "complex-geometry")
    assert list(tmp_path.iterdir()) == []
