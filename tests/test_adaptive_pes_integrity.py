"""Declared mathematics and genuine bounded, approved H2 adaptive calculations.

No callback, engine, energy, density, gradient or Hessian is mocked. The native
profile is a local finite-design demonstration, not chemical accuracy evidence.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.adaptive import (
    AdaptiveScanPlan,
    AdaptiveScanResult,
    InterpolationChallenge,
    StationaryCandidate,
    interpolate_energy,
    maximum_design_distance,
    next_maximin_index,
    run_adaptive_scan,
    validate_candidate_refinement,
    verify_candidate_minimum,
)
from cochem_torq.application import execute_request
from cochem_torq.artifacts import file_digest
from cochem_torq.candidate_ledger import CandidateLedger
from cochem_torq.domain import (
    CalculationRequest,
    Molecule,
    Resources,
    digest,
    read_json,
)
from cochem_torq.scan import (
    ApprovedScanExecutor,
    CoordinateDomain,
    InternalCoordinate,
    ScanBudget,
    ScanPlan,
)
from cochem_torq.scientific_contracts import (
    ObservableTarget,
    ProductDeclaration,
    ScientificGoal,
)
from cochem_torq.service import approve_plan, plan_request


def goal(tolerance: float) -> ScientificGoal:
    return ScientificGoal(
        products=(
            ProductDeclaration(
                product_class="A", observables=("numerical_interpolation_energy",)
            ),
        ),
        chemical_domain="neutral singlet H2 finite fixed-bond teaching design",
        allowed_approximations=("restricted HF/STO-3G", "linear interpolation"),
        target_observables=("numerical_interpolation_energy",),
        error_targets=(
            ObservableTarget(
                observable="numerical_interpolation_energy",
                unit="hartree",
                target_kind="maximum_absolute_error",
                maximum=tolerance,
                rationale="Declared interpolation residual, not chemical accuracy",
            ),
        ),
        reference_policy="de_novo_no_calibration",
        unknown_achievable_accuracy_reason="No independent chemical benchmark",
    )


def request(
    *, maximum_calls: int = 7, spacing: float = 0.26, scf_max_cycle: int = 100
) -> tuple[CalculationRequest, AdaptiveScanPlan]:
    scan = ScanPlan(
        coordinates=(
            InternalCoordinate(
                coordinate_id="H1-H2",
                kind="bond",
                atom_indices=(0, 1),
                unit="bohr",
                domain=CoordinateDomain(minimum=1.0, maximum=1.8, periodic=False),
            ),
        ),
        grid=tuple((value,) for value in (1.0, 1.2, 1.4, 1.6, 1.8)),
        sampling_strategy="bounded_adaptive",
        passes=(),
        coordinate_treatment="fixed",
        unscanned_coordinates="frozen_cartesian",
        additional_constraints=(),
        initial_guess_policy="independent_pyscf_minao",
        budget=ScanBudget(max_physical_calls=maximum_calls, per_point_wall_seconds=20),
        energy_recheck_tolerance_hartree=1e-9,
        density_recheck_tolerance=1e-8,
        scf_max_cycle=scf_max_cycle,
    )
    adaptive = AdaptiveScanPlan(
        scan_sha256=digest(scan.model_dump(mode="json")),
        seed_indices=(0, 4),
        held_out_indices=(2,),
        max_normalized_design_distance=spacing,
        energy_interpolation_tolerance_hartree=0.02,
        scientific_goal=goal(0.02),
    )
    model = CalculationRequest(
        molecule=Molecule(
            symbols=["H", "H"],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            atom_ids=["H1", "H2"],
            charge=0,
            multiplicity=1,
        ),
        recipe="hf-sto-3g-pes-validation",
        products=["pes_scan"],
        resources=Resources(cores=1, memory_mb=512, wall_seconds=maximum_calls * 20),
        scientific_goal=adaptive.scientific_goal,
        source_provenance={
            "pes_scan": scan.model_dump(mode="json"),
            "adaptive_scan": adaptive.model_dump(mode="json"),
        },
    )
    return model, adaptive


def approval(model: CalculationRequest) -> dict[str, object]:
    return approve_plan(
        plan_request(model, execution="local_validation"),
        actor=f"local-os:{os.getuid()}",
    )


def test_declared_exact_geometric_distance_and_deterministic_diversity():
    axis = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert maximum_design_distance(axis, ()) is None
    assert maximum_design_distance(axis, (0, 4)) == 0.5
    assert next_maximin_index(axis, (0, 4), (2,)) == 1
    assert next_maximin_index(axis, (0, 1, 4), (2,)) == 3
    assert maximum_design_distance(axis, (0, 1, 3, 4)) == 0.25
    assert next_maximin_index(axis, (0, 1, 3, 4), (2,)) is None


@pytest.mark.parametrize(
    "axis",
    (
        [0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, np.nan],
        [0.0, np.inf],
        [-1e308, 1e308],
        [0.0, 1.0j],
    ),
)
def test_invalid_math_design_rejected(axis):
    with pytest.raises(ValueError):
        maximum_design_distance(axis, (0,))


@pytest.mark.parametrize("indices", ((-1,), (5,), (True,), (0, 0)))
def test_invalid_design_indices_rejected(indices):
    with pytest.raises(ValueError):
        maximum_design_distance([0.0, 1.0, 2.0, 3.0, 4.0], indices)
    with pytest.raises(ValueError):
        next_maximin_index([0.0, 1.0, 2.0, 3.0, 4.0], (0, 4), indices)


def test_interpolation_is_declared_math_with_brackets_and_no_missing_fill():
    assert interpolate_energy(((0.0, 0.0), (2.0, 4.0)), 1.0) == 2.0
    for anchors, value in (
        (((0.0, 0.0),), 0.0),
        (((0.0, 0.0), (2.0, 4.0)), 3.0),
        (((0.0, 0.0), (2.0, np.nan)), 1.0),
        (((0.0, 0.0), (2.0, 1.0j)), 1.0),
        (((0.0, 0.0), (2.0, 4.0)), np.nan),
    ):
        with pytest.raises(ValueError):
            interpolate_energy(anchors, value)


def test_plan_has_predeclared_disjoint_challenges_and_explicit_accuracy_scope():
    _, adaptive = request()
    assert adaptive.surrogate_model is None
    assert adaptive.scientific_goal.accuracy_status == "unqualified"
    for changes in (
        {"held_out_indices": (0,)},
        {"seed_indices": (0, 0)},
        {"energy_interpolation_tolerance_hartree": 0.01},
    ):
        with pytest.raises(ValidationError):
            AdaptiveScanPlan.model_validate({**adaptive.model_dump(), **changes})


def test_numeric_challenge_residual_cannot_be_invented():
    declared_math = {
        "sample_index": 1,
        "point_id": "declared-math-input-only",
        "coordinate_value": 1.0,
        "challenge_kind": "held_out_interior",
        "observed_energy_hartree": 4.0,
        "numerical_estimate_hartree": 3.0,
        "absolute_residual_hartree": 0.0,
        "parent_anchor_point_ids": ("declared-left", "declared-right"),
        "status": "available",
    }
    with pytest.raises(ValidationError, match="Residual"):
        InterpolationChallenge.model_validate(declared_math)


@pytest.fixture(scope="module")
def actual_scan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cochem_exec_adaptive_actual")
    model, adaptive = request()
    ledger_path = directory / "candidates.sqlite"
    with CandidateLedger(ledger_path) as ledger:
        with ApprovedScanExecutor(approval(model), directory / "scan") as executor:
            result = run_adaptive_scan(
                executor, adaptive, ledger=ledger, actor=f"local-os:{os.getuid()}"
            )
    return directory, ledger_path, result


@pytest.mark.real_engine
def test_actual_h2_loop_qualifies_only_finite_design_challenges(actual_scan):
    _, _, result = actual_scan
    assert result.stop_reason == "criteria_met", result.stop_detail
    assert result.physical_calls == 6
    assert result.physical_calls < result.surface.scan.budget.max_physical_calls
    assert len(result.anchor_point_ids) == 3
    assert len(result.challenges) == 3
    assert result.surrogate_model is None
    assert not result.automatic_pruning_enabled
    assert result.search_completeness == "not_established"
    assert result.accuracy_qualified is False
    points = {point.point_id: point for point in result.surface.points}
    assert {points[key].sample_index for key in result.anchor_point_ids} == {0, 1, 4}
    assert all(point.status == "available" for point in points.values())
    assert all(point.energy_hartree is not None for point in points.values())
    for challenge in result.challenges:
        assert challenge.absolute_residual_hartree == abs(
            challenge.observed_energy_hartree - challenge.numerical_estimate_hartree
        )
        assert (
            challenge.estimate_kind
            == "numerical_interpolation_not_physical_observation"
        )
        assert challenge.accuracy_qualified is False


@pytest.mark.real_engine
def test_actual_observations_retain_native_identity_and_summary_integrity(actual_scan):
    directory, _, result = actual_scan
    for point in result.surface.points:
        manifest = directory / "scan" / point.native_manifest_path
        assert file_digest(manifest) == point.native_manifest_sha256
        native = read_json(manifest.parent / "result.json")
        assert native["energy_hartree"] == point.energy_hartree
        assert native["scf"]["converged"] is True
        assert native["stability"]["status"] == "stable"
    for changes in ({"physical_calls": 0}, {"anchor_point_ids": ("absent",)}):
        with pytest.raises(ValidationError):
            AdaptiveScanResult.model_validate({**result.model_dump(), **changes})


@pytest.mark.real_engine
def test_actual_candidates_are_reversible_quarantined_inputs_not_stationary(
    actual_scan,
):
    _, ledger_path, result = actual_scan
    assert result.candidates
    with CandidateLedger(ledger_path) as ledger:
        for candidate in result.candidates:
            record = ledger.inspect(candidate.candidate_id)
            assert record["selection_state"] == "quarantined"
            assert record["quality"]["evidence"] == "input_only"
            assert not candidate.verified_stationary
            assert candidate.constraint_release_requires_review
            excluded = ledger.exclude(
                candidate.candidate_id,
                expected_revision=record["revision"],
                actor="actual-local-review",
                reason="Exercise reversible review",
            )
            restored = ledger.restore(
                candidate.candidate_id,
                expected_revision=excluded["revision"],
                actor="actual-local-review",
                reason="Restore original input selection",
            )
            assert restored["request"] == record["request"]
            assert restored["selection_state"] == "quarantined"


@pytest.mark.real_engine
def test_native_candidate_requires_ledger_review_before_refinement(actual_scan):
    _, ledger_path, result = actual_scan
    candidate = result.candidates[0]
    approved = approval(candidate.candidate_request)
    with CandidateLedger(ledger_path) as ledger:
        record = ledger.inspect(candidate.candidate_id)
        with pytest.raises(ValueError, match="retain"):
            validate_candidate_refinement(
                candidate,
                ledger=ledger,
                expected_revision=record["revision"],
                approved_plan=approved,
            )


@pytest.mark.real_engine
def test_genuine_candidate_optimization_hessian_and_adversarial_origin(actual_scan):
    directory, ledger_path, result = actual_scan
    candidate = min(
        result.candidates, key=lambda item: abs(item.molecule.geometry_bohr[1][2] - 1.4)
    )
    approved = approval(candidate.candidate_request)
    with CandidateLedger(ledger_path) as ledger:
        old = ledger.inspect(candidate.candidate_id)
        retained = ledger.retain(
            candidate.candidate_id,
            expected_revision=old["revision"],
            actor="actual-local-review",
            reason="Review fixed-scan constraint release",
        )
        candidate_request = validate_candidate_refinement(
            candidate,
            ledger=ledger,
            expected_revision=retained["revision"],
            approved_plan=approved,
        )
        changed = candidate.model_dump(mode="json")
        forged_request = candidate_request.model_dump(mode="json")
        changed["point_id"] = "deliberately-altered-origin-rejection-case"
        forged_request["source_provenance"]["adaptive_candidate"]["point_id"] = changed[
            "point_id"
        ]
        changed["candidate_request_json"] = json.dumps(forged_request)
        changed["candidate_request_sha256"] = digest(forged_request)
        forged = StationaryCandidate.model_validate(changed)
        with pytest.raises(ValueError, match="immutable ledger"):
            validate_candidate_refinement(
                forged,
                ledger=ledger,
                expected_revision=retained["revision"],
                approved_plan=approved,
            )
        shard = directory / "target-minimum"
        observed = execute_request(candidate_request, shard, approved_plan=approved)
        assert observed["status"] == "complete", observed.get("errors")
        verified = verify_candidate_minimum(
            candidate,
            ledger=ledger,
            expected_revision=retained["revision"],
            approved_plan=approved,
            result_shard=shard,
            actor="actual-local-review",
        )
        assert verified.verified_stationary is True
        assert verified.transition_state_established is False
        assert verified.accuracy_qualified is False
        assert verified.verified_result_candidate_id != candidate.candidate_id
        assert (
            ledger.inspect(candidate.candidate_id)["quality"]["evidence"]
            == "input_only"
        )
        assert candidate.verified_stationary is False


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "parameters,reason,calls",
    (
        ({"maximum_calls": 2}, "budget_exhausted", 2),
        ({"maximum_calls": 7, "spacing": 0.01}, "insufficient_coverage", 4),
        ({"maximum_calls": 2, "scf_max_cycle": 1}, "failed", 1),
    ),
)
def test_genuine_budget_coverage_and_native_failure_stops(
    tmp_path, parameters, reason, calls
):
    model, adaptive = request(**parameters)
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        with ApprovedScanExecutor(approval(model), tmp_path / "scan") as executor:
            result = run_adaptive_scan(
                executor, adaptive, ledger=ledger, actor="actual-local-review"
            )
    assert result.stop_reason == reason, result.stop_detail
    assert result.physical_calls == calls
    if reason == "failed":
        assert result.surface.points[0].status == "failed"
        assert result.surface.points[0].energy_hartree is None
        assert not result.candidates
    else:
        assert all(point.energy_hartree is not None for point in result.surface.points)
    assert not result.accuracy_qualified
    assert result.search_completeness == "not_established"


@pytest.mark.real_engine
def test_generic_execution_routes_approved_adaptation_and_persists_summary(tmp_path):
    model, _ = request(maximum_calls=2)
    directory = tmp_path / "generic-adaptive"
    observed = execute_request(model, directory, approved_plan=approval(model))
    assert observed["adaptive"]["stop_reason"] == "budget_exhausted"
    assert observed["physical_call_count"] == 2
    saved = AdaptiveScanResult.model_validate(
        read_json(directory / "adaptive-result.json")
    )
    assert saved.model_dump(mode="json") == observed["adaptive"]
