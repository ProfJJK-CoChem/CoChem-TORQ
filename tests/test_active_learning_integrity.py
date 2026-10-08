"""Real numerical/electronic calculations and explicit unavailable-resource checks.

The geometries and least-squares training function are declared mathematical
inputs, not claimed optimized structures, quantum labels, or trained QM models.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cochem_torq.units import SPEED_OF_LIGHT_METRE_SECOND, atomic_mass, h, pi
from Libraries.cochem_torq_active_learning import (
    ActiveLearningOrchestrator,
    ActiveLearningSampler,
    assess_stage_b_rotational_redundancy,
    check_stage_b_rotational_redundancy,
    compute_rotational_constants,
    route_qm_tier,
)
from Libraries.cochem_torq_inference_errors import ActiveLearningSelectionError
from Libraries.cochem_torq_inference_schemas import ActiveLearningOrchestratorConfig
from Libraries.cochem_torq_masses import get_monoisotopic_mass


def test_linear_rotor_has_no_a_axis_constant_and_analytic_bc():
    length_angstrom = 0.74
    coordinates = [[0, 0, -length_angstrom / 2], [0, 0, length_angstrom / 2]]
    constants = compute_rotational_constants(coordinates, [1, 1])
    inertia_si = (
        get_monoisotopic_mass(1) * atomic_mass * (length_angstrom * 1e-10) ** 2 / 2
    )
    expected_cm1 = h / (8 * pi**2 * SPEED_OF_LIGHT_METRE_SECOND * inertia_si) / 100
    assert constants[0] is None
    assert constants[1:] == pytest.approx((expected_cm1, expected_cm1), rel=1e-12)


def test_atom_has_no_rotational_constants():
    assert compute_rotational_constants([[3, -2, 7]], [2]) == (None, None, None)


def test_bent_and_spherical_declared_geometries_have_applicable_constants():
    bent = compute_rotational_constants([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [8, 1, 1])
    assert all(value is not None and np.isfinite(value) and value > 0 for value in bent)
    assert bent[0] > bent[1] > bent[2]
    tetrahedron = [[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    spherical = compute_rotational_constants(tetrahedron, [6, 1, 1, 1, 1])
    assert spherical[0] == pytest.approx(spherical[1], rel=1e-12)
    assert spherical[0] == pytest.approx(spherical[2], rel=1e-12)


def test_rotational_constants_are_rigid_translation_rotation_invariant():
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    before = compute_rotational_constants(coordinates, [8, 1, 1])
    after = compute_rotational_constants(coordinates @ rotation + [3, -2, 8], [8, 1, 1])
    assert before == pytest.approx(after, rel=1e-12)


@pytest.mark.parametrize(
    "coordinates,numbers",
    [
        ([[complex(0, 1), 0, 0]], [1]),
        ([[float("nan"), 0, 0]], [1]),
        ([[0, 0, 0]], [1.2]),
        ([[0, 0, 0]], [0]),
        ([[0, 0, 0]], [119]),
    ],
)
def test_invalid_physical_inputs_are_rejected(coordinates, numbers):
    with pytest.raises(ActiveLearningSelectionError):
        compute_rotational_constants(coordinates, numbers)


def test_missing_required_rotational_axis_prevents_complete_comparison():
    assessment = assess_stage_b_rotational_redundancy(
        (None, 2.0, 2.0), (None, 2.0, 2.0)
    )
    assert not assessment.complete
    assert assessment.redundant is None
    assert assessment.unavailable_required_axes == ("A",)
    assert assessment.compared_axes == ("B", "C")
    assert assessment.relative_changes == (None, 0.0, 0.0)
    assert not check_stage_b_rotational_redundancy((None, 2.0, 2.0), (None, 2.0, 2.0))


def test_linear_bc_only_comparison_is_explicit_and_not_an_a_axis_claim():
    assessment = assess_stage_b_rotational_redundancy(
        (None, 2.0, 2.0), (None, 2.0001, 2.0001), required_axes=("B", "C")
    )
    assert assessment.complete
    assert assessment.redundant
    assert assessment.required_axes == ("B", "C")
    assert assessment.relative_changes[0] is None
    assert assessment.relative_changes[1] == pytest.approx(0.00005)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, 2 + 1j])
def test_available_rotational_constants_must_be_real_finite_positive(value):
    with pytest.raises(ActiveLearningSelectionError):
        assess_stage_b_rotational_redundancy((value, 2.0, 1.0), (3.0, 2.0, 1.0))


@pytest.mark.parametrize(
    "force,tier,missing",
    [
        (0.1, "T3-10s", ["xtb"]),
        (0.3, "T3O-1h", ["orca"]),
        (1.0, "T3O-12h", ["orca", "cfour"]),
    ],
)
def test_missing_engine_blocks_exact_planned_tier(force, tier, missing):
    with pytest.raises(
        ActiveLearningSelectionError, match="no alternative method"
    ) as error:
        route_qm_tier(force, available_engines=[])
    assert error.value.diagnostics["nominal_tier"] == tier
    assert error.value.diagnostics["missing_engines"] == missing


def test_insufficient_budget_does_not_change_method():
    with pytest.raises(ActiveLearningSelectionError) as error:
        route_qm_tier(
            1.0, available_engines=["orca", "cfour", "xtb"], compute_budget_hours=1.0
        )
    assert error.value.diagnostics["nominal_tier"] == "T3O-12h"
    assert error.value.diagnostics["budget_exceeded"]


def test_available_engines_are_planning_inputs_not_scientific_qualification():
    assert (
        route_qm_tier(
            1.0, available_engines=["orca", "cfour"], compute_budget_hours=12.0
        )
        == "T3O-12h"
    )


def test_manifest_preserves_missing_constants_and_advisory_status(tmp_path):
    orchestrator = ActiveLearningOrchestrator(
        ActiveLearningOrchestratorConfig(staging_manifest_dir=tmp_path)
    )
    # Declared vectors exercise committee statistics and manifest serialization;
    # these values do not claim to be engine predictions.
    orchestrator.evaluate_pool(
        [np.array([[0, 0, -0.37], [0, 0, 0.37]])],
        [[1, 1]],
        [[0.0, 0.2]],
        [np.zeros((2, 2, 3))],
    )
    assert len(orchestrator.select_active_batch()) == 1
    path, digest = orchestrator.emit_air_gapped_manifest()
    payload = json.loads(path.read_text())
    assert payload["candidates"][0]["rotational_constants"][0] is None
    assert payload["candidates"][0]["metadata"]["rotational_unavailable_axes"] == ["A"]
    assert (
        payload["candidates"][0]["metadata"]["rotational_constants_kind"]
        == "rigid_rotor_at_input_geometry"
    )
    assert payload["scientific_qualification"] == "not_established"
    assert payload["advisory_only"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


class LeastSquaresMathematicalCorrection:
    """Actual fitted regression of a declared mathematical training function."""

    def __init__(self):
        points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.coefficients = np.linalg.lstsq(
            np.column_stack((np.ones(len(points)), points)),
            1.0 + 2.0 * points,
            rcond=None,
        )[0]

    def predict(self, coordinates):
        return float(np.array([1.0, np.asarray(coordinates)[0, 0]]) @ self.coefficients)


def test_actual_fitted_mathematical_surrogate_is_never_labelled_quantum_anchor():
    sampler = ActiveLearningSampler(delta_ml_model=LeastSquaresMathematicalCorrection())
    candidates = [
        {"coordinates": [[0.25, 0, 0]], "scout_energy": -1.0, "sigma_mev": 1.0}
    ]
    result = sampler.sample_pes_grid(candidates)[0]
    assert result["energy_hartree"] == pytest.approx(0.5)
    assert result["surrogate_correction_hartree"] == pytest.approx(1.5)
    assert result["energy_kind"] == "surrogate_prediction"
    assert result["method_identity"] is None
    assert not result["accuracy_qualified"]
    assert not result["used_for_final_scientific_observables"]
    assert not result["eligible_for_pruning"]
    assert candidates[0] == {
        "coordinates": [[0.25, 0, 0]],
        "scout_energy": -1.0,
        "sigma_mev": 1.0,
    }


@pytest.mark.real_engine
def test_genuine_pyscf_callback_requires_provenance_to_qualify_anchor():
    from pyscf import gto, scf

    def calculate_energy(coordinates):
        molecule = gto.M(
            atom=[("H", position) for position in coordinates],
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )
        method = scf.RHF(molecule)
        energy = method.kernel()
        if not method.converged:
            raise RuntimeError("Actual RHF calculation did not converge")
        return energy

    sampler = ActiveLearningSampler()
    result = sampler.sample_pes_grid(
        [
            {
                "coordinates": [[0, 0, -0.37], [0, 0, 0.37]],
                "scout_energy": -1.0,
                "sigma_mev": 20.0,
            }
        ],
        anchor_evaluator=calculate_energy,
    )[0]
    assert np.isfinite(result["energy_hartree"])
    assert result["query_anchor"]
    assert result["energy_kind"] == "external_evaluator_unverified"
    assert result["method_identity"] is None
    assert result["surrogate_correction_hartree"] is None
    assert result["provenance_scope"] == "method_matrix_policy_tag_only"
    assert not result["accuracy_qualified"]
    assert not result["used_for_final_scientific_observables"]
