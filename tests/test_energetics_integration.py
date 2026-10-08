"""Genuine PySCF fragment, ghost-response and optimized energy identities.

Atomic reference contractions and finite differences qualify this bounded
software profile; they do not establish experimental binding accuracy.
"""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import k, physical_constants

from cochem_torq.energetics import (
    InteractionRequest,
    evaluate_interaction,
    rrho_thermochemistry,
)
from cochem_torq.spectroscopy.harmonic import analyze_hessian

pytestmark = pytest.mark.real_engine


def helium_request(
    *, protocol="counterpoise_single_point", method="hf", separation=5.0
):
    return {
        "molecule": {
            "symbols": ["He", "He"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, separation]],
            "charge": 0,
            "multiplicity": 1,
            "atom_ids": ["he_a", "he_b"],
        },
        "method": {
            "name": method,
            "basis": "cc-pvdz",
            "reference": "restricted",
            "frozen_core": False,
        },
        "fragments": [
            {"fragment_id": "a", "atom_ids": ["he_a"], "charge": 0, "multiplicity": 1},
            {"fragment_id": "b", "atom_ids": ["he_b"], "charge": 0, "multiplicity": 1},
        ],
        "geometry_protocol": protocol,
    }


def accepted(result):
    assert result.status == "available", result.errors
    for component in result.components.values():
        assert component["status"] == "complete", component["errors"]
        assert component["scf"]["converged"]
        assert component["stability"]["status"] == "stable"
    return result


@pytest.fixture(scope="module")
def helium_cp(tmp_path_factory):
    return accepted(
        evaluate_interaction(
            helium_request(), tmp_path_factory.mktemp("cochem_exec_real_he_cp")
        )
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("overlap", "disjoint"),
        ("unassigned", "partition"),
        ("charge", "sum"),
        ("spin", "singlet"),
        ("implicit_ids", "atom IDs"),
        ("singlepoint_opt", "Single-point"),
    ],
)
def test_invalid_fragment_or_protocol_contract_rejected_before_engine(change, match):
    request = helium_request()
    if change == "overlap":
        request["fragments"][1]["atom_ids"] = ["he_a", "he_b"]
    elif change == "unassigned":
        request["fragments"][1]["atom_ids"] = ["missing"]
    elif change == "charge":
        request["fragments"][0]["charge"] = 1
    elif change == "spin":
        request["fragments"][0]["multiplicity"] = 3
    elif change == "implicit_ids":
        request["molecule"].pop("atom_ids")
    else:
        request["optimization"] = {"maxsteps": 40}
    with pytest.raises(ValueError, match=match):
        InteractionRequest.model_validate(request)


def test_real_ghost_basis_matches_independent_native_pyscf_and_cp_identity(helium_cp):
    from pyscf import gto, lib, scf

    lib.num_threads(1)
    positions = np.asarray(helium_cp.molecule.geometry_bohr)
    independent = []
    for active in [None, 0, 1]:
        molecule = gto.M(
            atom=[
                ("He" if active is None or index == active else "ghost-He", coordinate)
                for index, coordinate in enumerate(positions)
            ],
            unit="Bohr",
            basis="cc-pvdz",
            charge=0,
            spin=0,
            verbose=0,
        )
        reference = scf.RHF(molecule)
        reference.conv_tol = 1e-12
        reference.kernel()
        assert reference.converged
        independent.append(reference.e_tot)
    cp = helium_cp.quantities["interaction_counterpoise"].value_hartree
    assert cp == pytest.approx(
        independent[0] - independent[1] - independent[2], abs=2e-10
    )
    raw = helium_cp.quantities["interaction_uncorrected"].value_hartree
    bsse = helium_cp.quantities["bsse_correction"].value_hartree
    assert cp == pytest.approx(raw + bsse, abs=1e-12)
    assert bsse > 0
    for key in ["a_full_complex_basis", "b_full_complex_basis"]:
        component = helium_cp.components[key]
        directory = Path(component["manifest_path"]).parent
        charges = np.load(directory / "nuclear-charges.npy")
        assert sorted(charges.tolist()) == [0, 2]
        assert component["scf"]["electron_count"] == 2
        assert component["basis_function_count"] == 10
        assert (
            len(component["physical_atom_ids"]) == len(component["ghost_atom_ids"]) == 1
        )
        assert component["scf"]["spin_squared"] == 0.0


def test_real_all_center_ghost_gradient_matches_two_step_independent_energy_differences(
    helium_cp, tmp_path
):
    expected = np.asarray(helium_cp.gradients_hartree_bohr["interaction_counterpoise"])
    assert np.linalg.norm(expected.sum(axis=0)) < 1e-9
    errors = []
    for step in [0.002, 0.001]:
        values = []
        for sign in [-1, 1]:
            request = helium_request()
            request["molecule"]["geometry_bohr"][1][2] += sign * step
            result = accepted(
                evaluate_interaction(request, tmp_path / f"h-{step}-sign-{sign}")
            )
            values.append(result.quantities["interaction_counterpoise"].value_hartree)
        numerical = (values[1] - values[0]) / (2 * step)
        error = abs(numerical - expected[1, 2])
        assert error < 2e-7
        errors.append(error)
    assert errors[1] <= errors[0] + 2e-10
    ghost = helium_cp.components["a_full_complex_basis"]
    gradient = np.asarray(ghost["gradient_hartree_bohr"])
    # Basis-center response is physically nonzero. Dropping the ghost's
    # derivative would destroy translational invariance and the CP force.
    assert np.linalg.norm(gradient[1]) > 1e-8
    assert np.linalg.norm(gradient.sum(axis=0)) < 1e-10
    np.save(tmp_path / "independent-cp-gradient-errors.npy", errors)


def test_real_mp2_counterpoise_components_match_independent_mp2(tmp_path):
    from pyscf import gto, lib, mp, scf

    lib.num_threads(1)
    result = accepted(
        evaluate_interaction(helium_request(method="mp2"), tmp_path / "mp2-cp")
    )
    molecule = gto.M(
        atom="He 0 0 0; ghost-He 0 0 5", unit="Bohr", basis="cc-pvdz", spin=0, verbose=0
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-12
    reference.kernel()
    correlated = mp.MP2(reference, frozen=None)
    correlated.kernel()
    assert result.components["a_full_complex_basis"]["energy_hartree"] == pytest.approx(
        correlated.e_tot, abs=2e-10
    )
    assert (
        result.components["a_full_complex_basis"]["mp2"]["correlation_energy_hartree"]
        < 0
    )
    expected = np.asarray(result.gradients_hartree_bohr["interaction_counterpoise"])[
        1, 2
    ]
    errors = []
    for step in [0.002, 0.001]:
        values = []
        for sign in [-1, 1]:
            displaced = helium_request(method="mp2")
            displaced["molecule"]["geometry_bohr"][1][2] += sign * step
            actual = accepted(
                evaluate_interaction(displaced, tmp_path / f"mp2-h-{step}-s-{sign}")
            )
            values.append(actual.quantities["interaction_counterpoise"].value_hartree)
        error = abs((values[1] - values[0]) / (2 * step) - expected)
        assert error < 2e-7
        errors.append(error)
    np.save(tmp_path / "independent-mp2-cp-gradient-errors.npy", errors)


def test_frozen_monomers_never_claim_de_d0_or_optimized_geometry(tmp_path):
    result = accepted(
        evaluate_interaction(
            helium_request(protocol="supplied_frozen_monomers"), tmp_path / "frozen"
        )
    )
    assert len(result.components) == 3
    assert result.stationary_evidence == {}
    assert result.quantities["De"].status == "unavailable"
    assert result.quantities["D0_harmonic"].value_hartree is None
    assert "interaction_counterpoise" not in result.quantities
    assert (
        result.molecule.geometry_bohr == helium_request()["molecule"]["geometry_bohr"]
    )


@pytest.mark.parametrize("method_name", ["pbe-d4", "b3lyp-d4"])
def test_real_dft_d4_ghost_uses_physical_fragment_dispersion_only(
    tmp_path, method_name
):
    request = helium_request()
    request["molecule"] = {
        "symbols": ["H", "H", "H", "H"],
        "geometry_bohr": [
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
            [0.0, 6.0, -0.7],
            [0.0, 6.0, 0.7],
        ],
        "charge": 0,
        "multiplicity": 1,
        "atom_ids": ["a1", "a2", "b1", "b2"],
    }
    request["fragments"][0]["atom_ids"] = ["a1", "a2"]
    request["fragments"][1]["atom_ids"] = ["b1", "b2"]
    request["method"] = {
        "name": method_name,
        "basis": "def2-svp",
        "reference": "restricted",
        "frozen_core": False,
    }
    result = accepted(evaluate_interaction(request, tmp_path / f"{method_name}-cp"))
    for fragment_id in ["a", "b"]:
        native = result.components[f"{fragment_id}_native_at_complex"]
        ghost = result.components[f"{fragment_id}_full_complex_basis"]
        assert ghost["dispersion"]["energy_hartree"] == pytest.approx(
            native["dispersion"]["energy_hartree"], abs=1e-14
        )
        assert len(ghost["dispersion"]["gradient_hartree_bohr"]) == 2
        assert ghost["energy_hartree"] == pytest.approx(
            ghost["electronic_energy_hartree"] + ghost["dispersion"]["energy_hartree"],
            abs=1e-12,
        )
        assert ghost["scf"]["electron_count"] == 2
        assert result.basis_identity_evidence[f"{fragment_id}_full_complex_basis"][
            "same_angular_exponents_and_contractions"
        ]
    expected = float(
        np.asarray(result.gradients_hartree_bohr["interaction_counterpoise"])[
            2:, 1
        ].sum()
    )
    errors = []
    for step in [0.002, 0.001]:
        values = []
        for sign in [-1, 1]:
            displaced = deepcopy(request)
            for coordinate in displaced["molecule"]["geometry_bohr"][2:]:
                coordinate[1] += sign * step
            actual = accepted(
                evaluate_interaction(displaced, tmp_path / f"dft-h-{step}-s-{sign}")
            )
            values.append(actual.quantities["interaction_counterpoise"].value_hartree)
        error = abs((values[1] - values[0]) / (2 * step) - expected)
        assert error < 3e-6
        errors.append(error)
    np.save(tmp_path / "independent-dft-d4-cp-gradient-errors.npy", errors)


def water_dimer_request():
    angstrom_per_bohr = physical_constants["Bohr radius"][0] / 1e-10
    coordinates_angstrom = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.93, 0.0, -0.24],
            [0.0, 0.0, 2.8],
            [0.0, 0.75, 3.38],
            [0.0, -0.75, 3.38],
        ]
    )
    return {
        "molecule": {
            "symbols": ["O", "H", "H", "O", "H", "H"],
            "geometry_bohr": (coordinates_angstrom / angstrom_per_bohr).tolist(),
            "charge": 0,
            "multiplicity": 1,
            "atom_ids": ["o_a", "h_a1", "h_a2", "o_b", "h_b1", "h_b2"],
            "isotopes": [16, 1, 1, 16, 1, 1],
        },
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
        },
        "fragments": [
            {
                "fragment_id": "a",
                "atom_ids": ["o_a", "h_a1", "h_a2"],
                "charge": 0,
                "multiplicity": 1,
            },
            {
                "fragment_id": "b",
                "atom_ids": ["o_b", "h_b1", "h_b2"],
                "charge": 0,
                "multiplicity": 1,
            },
        ],
        "geometry_protocol": "fully_relaxed",
        "rrho": {
            "temperature_kelvin": 298.15,
            "standard_pressure_pa": 100000.0,
            "rotational_symmetry_numbers": {"complex": 1, "a": 2, "b": 2},
        },
    }


@pytest.fixture(scope="module")
def relaxed_water_dimer(tmp_path_factory):
    return accepted(
        evaluate_interaction(
            water_dimer_request(),
            tmp_path_factory.mktemp("cochem_exec_water_relaxed_binding"),
        )
    )


def test_real_relaxed_binding_deformation_de_and_harmonic_d0_identities(
    relaxed_water_dimer,
):
    result = relaxed_water_dimer
    assert len(result.stationary_evidence) == 3
    assert not np.allclose(
        result.molecule.geometry_bohr,
        water_dimer_request()["molecule"]["geometry_bohr"],
    )
    values = {
        key: quantity.value_hartree
        for key, quantity in result.quantities.items()
        if quantity.status == "available"
    }
    assert values["deformation"] >= -2e-9
    assert values["binding_electronic"] == pytest.approx(
        values["interaction_uncorrected"] + values["deformation"], abs=1e-11
    )
    assert values["De"] == pytest.approx(-values["binding_electronic"], abs=1e-12)
    assert values["De"] > 0
    assert values["D0_harmonic"] == pytest.approx(
        values["De"] - values["zpe_binding_correction"], abs=1e-12
    )
    assert "interaction_counterpoise" not in values
    assert all(
        evidence["classification"] == "positive_definite_vibrational_hessian"
        for evidence in result.stationary_evidence.values()
    )
    assert all(
        evidence["independent_final_gradient_verified"]
        for evidence in result.stationary_evidence.values()
    )
    assert len(result.stationary_evidence["complex"]["harmonic_frequencies_cm1"]) == 12
    assert len(result.stationary_evidence["a"]["harmonic_frequencies_cm1"]) == 3
    assert result.thermochemistry["complex"]["zpe_included_exactly_once"]


def test_actual_rrho_standard_pressure_scaling_and_h_minus_ts_identity(
    relaxed_water_dimer,
):
    differences = {}
    temperature = 298.15
    for key, component_id in [
        ("complex", "complex"),
        ("a", "a_relaxed"),
        ("b", "b_relaxed"),
    ]:
        component = relaxed_water_dimer.components[component_id]
        masses = [
            record["mass_u"]
            for record in relaxed_water_dimer.stationary_evidence[key]["isotopes"]
        ]
        harmonic = analyze_hessian(
            component["geometry_bohr"], masses, component["hessian_hartree_bohr2"]
        )
        reference = relaxed_water_dimer.thermochemistry[key]
        twice_pressure = rrho_thermochemistry(
            harmonic,
            temperature_kelvin=temperature,
            standard_pressure_pa=200000.0,
            rotational_symmetry_number=reference["rotational_symmetry_number"],
        )
        differences[key] = (
            twice_pressure["gibbs_correction_hartree"]
            - reference["gibbs_correction_hartree"]
        )
        assert reference["gibbs_correction_hartree"] == pytest.approx(
            reference["enthalpy_correction_hartree"]
            - temperature * reference["entropy_hartree_per_kelvin"],
            abs=1e-13,
        )
    expected = -k * temperature * np.log(2) / physical_constants["Hartree energy"][0]
    assert differences["complex"] - differences["a"] - differences[
        "b"
    ] == pytest.approx(expected, abs=1e-13)


def test_interaction_bundle_hashes_every_real_component_and_rejects_stale_workspace(
    helium_cp,
):
    directory = Path(helium_cp.manifest_path).parent
    manifest = json.loads(Path(helium_cp.manifest_path).read_text())
    assert (
        helium_cp.manifest_sha256
        == sha256(Path(helium_cp.manifest_path).read_bytes()).hexdigest()
    )
    assert any(
        entry["path"].endswith("wavefunction.chk") for entry in manifest["artifacts"]
    )
    assert any(
        entry["path"].endswith("nuclear-charges.npy") for entry in manifest["artifacts"]
    )
    for entry in manifest["artifacts"]:
        assert (
            sha256((directory / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    with pytest.raises(ValueError, match="empty"):
        evaluate_interaction(deepcopy(helium_request()), directory)
