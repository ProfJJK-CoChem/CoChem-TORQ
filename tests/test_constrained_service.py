"""Finite constrained plans and genuine approved native calculation shards."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from cochem_torq.application import execute_request, validate_request
from cochem_torq.artifacts import verify_shard
from cochem_torq.constrained_service import constrained_specification
from cochem_torq.domain import CalculationRequest, StageResult, digest, read_json
from cochem_torq.engines.constrained_optimization import (
    RECIPE_ID,
    ConstrainedOptimizationResult,
)
from cochem_torq.internal_coordinates import coordinate_target, coordinate_value
from tests.test_constrained_optimization import specification, water


def _request(*, curvature=False):
    return CalculationRequest.model_validate(
        {
            "molecule": water().model_dump(mode="json"),
            "recipe": RECIPE_ID,
            "products": ["constrained_geometry"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 180},
            "source_provenance": {
                "constrained_optimization": specification(
                    curvature=curvature, max_calls=40, wall=30
                ).model_dump(mode="json")
            },
        }
    )


def test_finite_constrained_plan_binds_exact_request_and_physical_call_ceiling():
    request = _request()
    checked = validate_request(request, execution="local_validation")
    assert checked["executable"], checked["blocking_reasons"]
    spec = constrained_specification(request)
    plan = checked["plan"]
    assert plan["request"] == request.model_dump(mode="json")
    assert plan["constraints"] == spec.model_dump(mode="json")
    assert plan["request_sha256"] == checked["request_sha256"]
    assert plan["scientific_cache_key"] == checked["scientific_cache_key"]
    assert plan["plan_sha256"] == digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    assert len(plan["tasks"]) == plan["planned_physical_call_ceiling"] == 40
    assert plan["curvature_requested"] is False
    assert plan["unconstrained_equilibrium_claimed"] is False
    assert plan["search_completeness_claimed"] is False
    for index, task in enumerate(plan["tasks"]):
        assert task["id"] == f"constrained-evaluation-{index:04d}"
        assert task["depends_on"] == (
            [] if index == 0 else [f"constrained-evaluation-{index - 1:04d}"]
        )
        assert task["conditional"] is True
        assert task["recipe_sha256"] == checked["recipe"]["recipe_sha256"]


def test_curvature_plan_declares_and_authorizes_independent_final_hessian():
    checked = validate_request(_request(curvature=True), execution="local_validation")
    assert checked["executable"], checked["blocking_reasons"]
    plan = checked["plan"]
    assert plan["curvature_requested"] is True
    assert plan["curvature_derivative"] == "analytic_HF_hessian"
    assert len(plan["tasks"]) == plan["planned_physical_call_ceiling"] == 40
    for task in plan["tasks"]:
        assert task["native_properties"] == ["energy", "gradient"]
        assert task["fresh_final_verification_properties"] == [
            "energy",
            "gradient",
            "hessian",
        ]
    assert any(
        receipt["tuple"]["property"] == "hessian"
        and receipt["tuple"]["derivative"] == "analytic"
        for receipt in checked["capabilities"]["resolved"]
    )


@pytest.mark.parametrize(
    "change", ["bond_target", "angle_target", "movable", "curvature", "call_limit"]
)
def test_constrained_cache_and_plan_bind_targets_movable_atoms_and_full_spec(change):
    request = _request().model_dump(mode="json")
    original = validate_request(request, execution="local_validation")
    changed = deepcopy(request)
    spec = changed["source_provenance"]["constrained_optimization"]
    if change == "bond_target":
        spec["constraints"][0]["target"] = 2.0
    elif change == "angle_target":
        spec["constraints"][1]["target"] = 105.5
    elif change == "movable":
        spec["movable_atom_indices"] = [0, 1, 2]
    elif change == "curvature":
        spec["request_curvature"] = True
    else:
        spec["max_physical_calls"] = 41
    checked = validate_request(changed, execution="local_validation")
    assert checked["executable"], checked["blocking_reasons"]
    assert checked["scientific_cache_key"] != original["scientific_cache_key"]
    assert checked["plan"]["plan_sha256"] != original["plan"]["plan_sha256"]
    assert checked["plan"]["constraints"] == spec


@pytest.mark.parametrize(
    "problem",
    [
        "missing_spec",
        "wrong_recipe",
        "mixed_products",
        "movable_outside",
        "coordinate_outside",
        "cannot_move_target",
        "dependent_constraints",
        "wall_ceiling",
        "target_outside",
    ],
)
def test_invalid_constrained_declarations_fail_before_engine_work(problem):
    raw = _request().model_dump(mode="json")
    spec = raw["source_provenance"]["constrained_optimization"]
    if problem == "missing_spec":
        raw["source_provenance"] = {}
    elif problem == "wrong_recipe":
        raw["recipe"] = "hf-sto-3g-education"
    elif problem == "mixed_products":
        raw["products"].append("geometry")
    elif problem == "movable_outside":
        spec["movable_atom_indices"] = [1, 3]
    elif problem == "coordinate_outside":
        spec["constraints"][0]["coordinate"]["atom_indices"] = [0, 3]
    elif problem == "cannot_move_target":
        spec["movable_atom_indices"] = [2]
    elif problem == "dependent_constraints":
        duplicate = deepcopy(spec["constraints"][0])
        duplicate["coordinate"]["coordinate_id"] = "same-bond-distinct-label"
        spec["constraints"] = [spec["constraints"][0], duplicate]
    elif problem == "wall_ceiling":
        spec["per_evaluation_wall_seconds"] = 181
    else:
        spec["constraints"][0]["target"] = 1.4
    model = CalculationRequest.model_validate(raw)
    with pytest.raises(ValueError):
        constrained_specification(model)


def test_constrained_product_is_local_and_does_not_leak_into_equilibrium_geometry():
    checked = validate_request(_request(), execution="github_actions")
    assert checked["executable"] is False
    assert any("local validation" in reason for reason in checked["blocking_reasons"])
    raw = _request().model_dump(mode="json")
    raw["products"] = ["geometry"]
    checked = validate_request(raw, execution="local_validation")
    assert checked["executable"] is False
    assert any("separate product" in reason for reason in checked["blocking_reasons"])


@pytest.mark.real_engine
def test_actual_approved_constrained_water_worker_and_immutable_shard(tmp_path):
    pytest.importorskip("pyscf")
    from cochem_torq.service import approve_plan, plan_request

    request = _request().model_dump(mode="json")
    planned = plan_request(request, execution="local_validation")
    approved = approve_plan(planned, actor="genuine-constrained-water-service-test")
    directory = tmp_path / "approved-constrained-water"
    result = execute_request(request, directory, approved_plan=approved)
    assert result["status"] == "complete", result["errors"]
    manifest = verify_shard(directory)
    assert manifest["request_sha256"] == result["request_sha256"]
    assert manifest["recipe_sha256"] == result["recipe_sha256"]
    electronic = StageResult.model_validate(result["stages"]["electronic_structure"])
    constrained = StageResult.model_validate(result["stages"]["constrained_geometry"])
    assert electronic.status == constrained.status == "available"
    assert constrained.observable == "constrained_stationary_geometry"
    observation = ConstrainedOptimizationResult.model_validate(constrained.value)
    assert observation.final_molecule is not None
    assert observation.status == "available"
    assert observation.equilibrium_geometry_claimed is False
    assert observation.unconstrained_minimum_claimed is False
    assert observation.independent_scientific_qualification is False
    assert observation.physical_call_count <= 40
    assert observation.physical_call_count == len(observation.evaluations)
    assert observation.evaluations[-1].purpose == "independent_final_verification"
    assert observation.curvature.status == "unavailable"
    assert "not requested" in observation.curvature.reason.lower()
    assert observation.final_molecule.atom_ids == water().atom_ids
    np.testing.assert_array_equal(
        observation.final_molecule.geometry_bohr[0], water().geometry_bohr[0]
    )
    for item in observation.specification.constraints:
        actual = coordinate_value(
            observation.final_molecule.geometry_bohr, item.coordinate
        )
        assert actual == pytest.approx(
            coordinate_target(item.target, item.coordinate), abs=1e-8
        )
    assert (
        max(abs(value) for value in observation.tangent_gradient_hartree_bohr) <= 1e-5
    )
    for evaluation in observation.evaluations:
        assert evaluation.status == "available"
        native_manifest = (
            directory / "constrained-engine" / evaluation.native_manifest_path
        )
        assert (
            sha256(native_manifest.read_bytes()).hexdigest()
            == evaluation.native_manifest_sha256
        )
        native = read_json(native_manifest.parent / "result.json")
        assert native["energy_hartree"] == evaluation.electronic.energy_hartree
        assert native["scf"]["converged"] is True
        assert (native_manifest.parent / "wavefunction.chk").is_file()
    for name in ("equilibrium_geometry", "equilibrium_constants", "harmonic_analysis"):
        assert result["stages"][name]["absence_kind"] == "not_requested"
    assert result["experimental_accuracy_established"] is False
    assert result["identification_ready"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        execute_request(request, directory)
