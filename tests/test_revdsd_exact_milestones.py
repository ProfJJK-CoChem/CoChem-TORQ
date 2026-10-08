"""Genuine HF-response milestones; exact revDSD remains independently gated."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.engines.revdsd import (
    ExperimentalDoubleHybrid,
    RecipeNotQualifiedError,
    ResearchCalculationError,
    ResearchRecipe,
    exact_revdsd_pbep86_d4_recipe,
)

pytestmark = pytest.mark.research
WATER = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])


def _evaluator(*, os_scale=0.6, ss_scale=0.04, frozen=(), d4=None):
    return ExperimentalDoubleHybrid(
        ResearchRecipe(
            "explicit HF spin-scaled correlation derivative milestone",
            "HF",
            "HF",
            os_scale,
            ss_scale,
            frozen_occupied_orbitals=frozen,
            d4_method=d4,
        ),
        basis="sto-3g",
        check_reference_stability=False,
        scf_energy_tolerance=1e-13,
        scf_gradient_tolerance=1e-10,
    )


def test_analytic_research_milestone_does_not_open_named_revdsd():
    with pytest.raises(RecipeNotQualifiedError, match="original article/SI"):
        exact_revdsd_pbep86_d4_recipe()


def test_analytic_gks_response_is_explicitly_unavailable_before_execution(tmp_path):
    evaluator = ExperimentalDoubleHybrid(
        ResearchRecipe("explicit PBE0 research", "PBE0", "PBE0", 0.6, 0.04),
        basis="sto-3g",
    )
    with pytest.raises(ResearchCalculationError, match="moving-grid GKS"):
        evaluator.analytic_gradient(
            ["O", "H", "H"],
            WATER,
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "controls",
    [
        {"response_tolerance": 0.0},
        {"response_tolerance": np.nan},
        {"response_tolerance": True},
        {"orbital_separation_tolerance": -0.1},
        {"response_max_cycle": 0},
        {"response_max_cycle": True},
        {"max_ao_tensor_bytes": 0},
        {"max_ao_tensor_bytes": True},
    ],
)
def test_invalid_analytic_controls_reject_without_engine_calculation(
    tmp_path, controls
):
    with pytest.raises(ValueError):
        _evaluator().analytic_gradient(
            ["O", "H", "H"],
            WATER,
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            **controls,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.real_engine
@pytest.mark.parametrize("frozen", [(), (0,)])
def test_analytic_canonical_mp2_against_native_gradient_with_core_response(
    tmp_path, frozen
):
    from pyscf import gto, lib, mp, scf

    result = _evaluator(os_scale=1.0, ss_scale=1.0, frozen=frozen).analytic_gradient(
        ["O", "H", "H"],
        WATER,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    with lib.with_omp_threads(1):
        mol = gto.M(
            atom=list(zip(["O", "H", "H"], WATER)),
            basis="sto-3g",
            unit="Bohr",
            verbose=0,
        )
        reference = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
        native = mp.MP2(reference, frozen=list(frozen)).run()
        expected = native.nuc_grad_method().kernel()
    np.testing.assert_allclose(result["gradient"], expected, atol=2e-9, rtol=0)
    assert result["energy_hartree"] == pytest.approx(native.e_tot, abs=2e-10)
    assert result["finite_difference_energy_calculations"] == 0
    assert result["executed_energy_calculations"] == 1
    assert result["exact_revdsd_qualification"] is False
    assert result["independent_method_validation"] == "unavailable"
    assert result["GKS_analytic_response_available"] is False
    assert result["provenance"]["frozen_occupied_orbitals"] == list(frozen)
    diagnostics = result["response_diagnostics"]
    assert diagnostics["maximum_native_cphf_equation_residual_hartree_per_bohr"] < 1e-9
    assert diagnostics["maximum_native_occupied_response_difference_per_bohr"] < 1e-9
    assert diagnostics["response_level_shift"] == 0
    assert diagnostics["reference_orbitals_unchanged"] is True
    parent = result["parent"]
    directory = Path(parent["artifact_directory"])
    assert parent["parent_artifact_hashes_unchanged"] is True
    for name, digest in parent["parent_artifact_hashes"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest
    native_log = (directory / "pyscf.log").read_text()
    assert "converged SCF energy" in native_log
    assert (directory / "analytic-response.log").stat().st_size > 100
    saved = json.loads(Path(result["artifact_path"]).read_text())
    assert saved["gradient"] == result["gradient"]


@pytest.mark.real_engine
@pytest.mark.parametrize("frozen", [(), (0,)])
def test_separate_os_ss_response_against_complete_displaced_energy_components(
    tmp_path, frozen
):
    evaluator = _evaluator(frozen=frozen)
    analytic = evaluator.analytic_gradient(
        ["O", "H", "H"],
        WATER,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    for step in (0.0006, 0.0003):
        numerical_os, numerical_ss, numerical_total = (
            np.empty_like(WATER),
            np.empty_like(WATER),
            np.empty_like(WATER),
        )
        for atom in range(3):
            for axis in range(3):
                energies = []
                for sign in (1, -1):
                    displaced = WATER.copy()
                    displaced[atom, axis] += sign * step
                    energies.append(
                        evaluator.evaluate(
                            ["O", "H", "H"],
                            displaced,
                            charge=0,
                            multiplicity=1,
                            artifact_directory=tmp_path,
                        )
                    )
                plus, minus = energies
                numerical_os[atom, axis] = (
                    plus["components"]["opposite_spin_pt2_hartree"]
                    - minus["components"]["opposite_spin_pt2_hartree"]
                ) / (2 * step)
                numerical_ss[atom, axis] = (
                    plus["components"]["same_spin_pt2_hartree"]
                    - minus["components"]["same_spin_pt2_hartree"]
                ) / (2 * step)
                numerical_total[atom, axis] = (
                    plus["energy_hartree"] - minus["energy_hartree"]
                ) / (2 * step)
        np.testing.assert_allclose(
            analytic["gradient_components"]["unscaled_opposite_spin_pt2"],
            numerical_os,
            atol=2e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            analytic["gradient_components"]["unscaled_same_spin_pt2"],
            numerical_ss,
            atol=1e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            analytic["gradient"], numerical_total, atol=9e-8, rtol=0
        )


@pytest.mark.real_engine
def test_analytic_scs_d4_gradient_and_cartesian_covariance(tmp_path):
    from dftd4.interface import DampingParam, DispersionModel

    evaluator = _evaluator(d4="pbe0")
    reference = evaluator.analytic_gradient(
        ["O", "H", "H"],
        WATER,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    native_dispersion = DispersionModel(np.asarray([8, 1, 1]), WATER, charge=0.0)
    expected = native_dispersion.get_dispersion(
        DampingParam(method="pbe0", atm=True), grad=True
    )["gradient"]
    np.testing.assert_allclose(
        reference["gradient_components"]["dispersion"], expected, atol=1e-13, rtol=0
    )
    angle = 0.437
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    moved = evaluator.analytic_gradient(
        ["O", "H", "H"],
        WATER @ rotation.T + np.asarray([1.6, -0.4, 0.2]),
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
    )
    np.testing.assert_allclose(
        moved["gradient"], np.asarray(reference["gradient"]) @ rotation.T, atol=3e-9
    )
    for result in (reference, moved):
        np.testing.assert_allclose(
            result["translational_gradient_sum_hartree_per_bohr"], 0.0, atol=2e-9
        )
        np.testing.assert_allclose(
            result["rotational_gradient_torque_hartree"], 0.0, atol=2e-9
        )
        assert result["dispersion_status"] == "native_DFTD4_gradient_evaluated"


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "controls,match",
    [
        ({"max_ao_tensor_bytes": 1}, "workspace exceeds"),
        ({"orbital_separation_tolerance": 0.5}, "nondegenerate orbitals"),
    ],
)
def test_unsupported_response_retains_energy_and_actual_derivative_failure(
    tmp_path, controls, match
):
    with pytest.raises(ResearchCalculationError, match=match):
        _evaluator().analytic_gradient(
            ["O", "H", "H"],
            WATER,
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            **controls,
        )
    parent = next(tmp_path.glob("double-hybrid-*"))
    assert json.loads((parent / "result.json").read_text())["status"] == "success"
    failure_path = next(tmp_path.glob("analytic-gradient-*.json"))
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert (
        failure["parent_result_sha256"]
        == sha256((parent / "result.json").read_bytes()).hexdigest()
    )
    assert "gradient" not in failure


def _hessian_controls():
    return {
        "steps_bohr": (0.003, 0.0015),
        "symmetry_tolerance_hartree_per_bohr2": 1e-5,
        "step_difference_tolerance_hartree_per_bohr2": 1e-5,
        "translation_tolerance_hartree_per_bohr2": 1e-5,
        "rotation_tolerance_hartree_per_bohr": 1e-5,
        "max_gradient_calculations": 25,
    }


@pytest.mark.real_engine
def test_response_gradient_hessian_matches_native_mp2_d4_gradient_stencil(tmp_path):
    from dftd4.interface import DampingParam, DispersionModel
    from pyscf import gto, lib, mp, scf

    coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    result = _evaluator(
        os_scale=1.0, ss_scale=1.0, d4="pbe0"
    ).hessian_from_analytic_gradients(
        ["H", "H"],
        coords,
        charge=0,
        multiplicity=1,
        artifact_directory=tmp_path,
        **_hessian_controls(),
    )
    independent_stencil = np.empty((6, 6))
    step = 0.0003
    with lib.with_omp_threads(1):
        for q in range(6):
            gradients = []
            for sign in (1, -1):
                geometry = coords.copy().ravel()
                geometry[q] += sign * step
                geometry = geometry.reshape((2, 3))
                mol = gto.M(
                    atom=list(zip(["H", "H"], geometry)),
                    unit="Bohr",
                    basis="sto-3g",
                    verbose=0,
                )
                hf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
                native_mp2 = mp.MP2(hf).run().nuc_grad_method().kernel()
                native_d4 = DispersionModel(
                    np.asarray([1, 1]), geometry, charge=0.0
                ).get_dispersion(DampingParam(method="pbe0", atm=True), grad=True)[
                    "gradient"
                ]
                gradients.append(native_mp2 + native_d4)
            independent_stencil[:, q] = (gradients[0] - gradients[1]).ravel() / (
                2 * step
            )
    np.testing.assert_allclose(
        result["hessian"], independent_stencil, atol=3e-6, rtol=0
    )
    assert result["executed_gradient_calculations"] == 25
    assert result["executed_energy_calculations"] == 25
    assert result["selected_step_bohr"] == 0.0015
    assert result["matrix_symmetrized"] is False
    assert result["symmetry_by_construction"] is False
    assert result["numerical_consistency_accepted"] is True
    assert all(
        gate["passed"] for gate in result["numerical_consistency_gates"].values()
    )
    assert result["derivative_convergence_qualified"] is False
    assert result["stationarity_qualified"] is False
    assert result["harmonic_analysis_qualified"] is False
    assert result["exact_revdsd_qualification"] is False
    assert result["formal_discretization_error_bound"] is None
    assert np.linalg.norm(result["reference_gradient_hartree_per_bohr"]) > 1e-3
    for raw_matrix, components in zip(
        result["raw_hessians_by_step"], result["component_hessians_by_step"]
    ):
        reconstructed = sum(np.asarray(value) for value in components.values())
        np.testing.assert_allclose(raw_matrix, reconstructed, atol=2e-12, rtol=0)
    for parent in result["parents"]:
        artifact = Path(parent["gradient_artifact_path"])
        assert (
            sha256(artifact.read_bytes()).hexdigest()
            == parent["gradient_result_sha256"]
        )
        run = json.loads(artifact.read_text())
        assert run["parent"]["parent_artifact_hashes_unchanged"] is True
        assert parent["checkpoint_sha256"] == run["parent"]["checkpoint_sha256_after"]
        energy = Path(parent["energy_artifact_directory"]) / "result.json"
        assert sha256(energy.read_bytes()).hexdigest() == parent["energy_result_sha256"]


@pytest.mark.real_engine
def test_response_hessian_rejects_actual_step_disagreement_without_symmetrizing(
    tmp_path,
):
    controls = _hessian_controls()
    controls["step_difference_tolerance_hartree_per_bohr2"] = 1e-12
    with pytest.raises(ResearchCalculationError, match="cross_step_agreement"):
        _evaluator().hessian_from_analytic_gradients(
            ["H", "H"],
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            **controls,
        )
    record = json.loads(
        next(tmp_path.glob("analytic-gradient-hessian-*.json")).read_text()
    )
    assert record["status"] == "failed"
    assert record["executed_gradient_calculations"] == 25
    assert (
        record["observations"]["numerical_consistency_gates"]["cross_step_agreement"][
            "passed"
        ]
        is False
    )
    assert len(record["observations"]["raw_hessians_by_step"]) == 2
    assert len(record["parents"]) == 25
    assert "hessian" not in record


@pytest.mark.parametrize(
    "changed,expected",
    [
        ({"max_gradient_calculations": 24}, "budget permits 24"),
        ({"steps_bohr": (0.001,)}, "two distinct"),
        ({"symmetry_tolerance_hartree_per_bohr2": True}, "thresholds"),
        ({"translation_tolerance_hartree_per_bohr2": np.inf}, "thresholds"),
    ],
)
def test_response_hessian_controls_fail_before_execution(tmp_path, changed, expected):
    controls = {**_hessian_controls(), **changed}
    with pytest.raises((ValueError, ResearchCalculationError), match=expected):
        _evaluator().hessian_from_analytic_gradients(
            ["H", "H"],
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            **controls,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.real_engine
def test_response_hessian_failed_gradient_counts_completed_energy_separately(tmp_path):
    with pytest.raises(ResearchCalculationError, match="workspace exceeds"):
        _evaluator().hessian_from_analytic_gradients(
            ["H", "H"],
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
            charge=0,
            multiplicity=1,
            artifact_directory=tmp_path,
            max_ao_tensor_bytes=1,
            **_hessian_controls(),
        )
    record = json.loads(
        next(tmp_path.glob("analytic-gradient-hessian-*.json")).read_text()
    )
    assert record["status"] == "failed"
    assert record["attempted_gradient_calculations"] == 1
    assert record["executed_gradient_calculations"] == 0
    assert record["attempted_energy_calculations"] == 1
    assert record["completed_energy_calculations"] == 1
    assert len(record["failed_nodes"]) == 1
    node = record["failed_nodes"][0]
    assert node["completed_energy_calculations"] == 1
    assert node["retained_result_and_failure_hashes"]
    for relative, digest in node["retained_result_and_failure_hashes"].items():
        path = Path(node["artifact_directory"]) / relative
        assert sha256(path.read_bytes()).hexdigest() == digest
    assert "hessian" not in record
