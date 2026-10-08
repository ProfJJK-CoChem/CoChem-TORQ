"""Bounded real constrained water scans and independent contract/KKT checks.

Physical acceptance observes actual owned PySCF workers. Negative declarations
exercise intake mathematics and process contracts without invented engine data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from cochem_torq.application import execute_request, source_identity, validate_request
from cochem_torq.domain import CalculationRequest, canonical_json, digest, read_json
from cochem_torq.registry import get_profile, profile_capabilities
from cochem_torq.relaxed_scan import INNER_RECIPE, point_relaxation_request
from cochem_torq.scan import (
    RELAXED_SCAN_RECIPE,
    RelaxedPointEvidence,
    ScanPlan,
    ScanSurface,
    geometry_for_sample,
    scan_definition,
)
from cochem_torq.service import approve_plan, plan_request


def water_request(*, native_ceiling=40):
    scan = {
        "coordinates": [
            {
                "coordinate_id": "fixed-OH1",
                "kind": "bond",
                "atom_indices": [0, 1],
                "unit": "bohr",
                "domain": {"minimum": 1.5, "maximum": 2.5, "periodic": False},
            }
        ],
        "grid": [[1.9]],
        "sampling_strategy": "full_grid",
        "passes": [
            {"purpose": purpose, "sample_indices": [0]}
            for purpose in ("forward", "reverse", "challenge")
        ],
        "coordinate_treatment": "relaxed",
        "unscanned_coordinates": "constrained_relaxation",
        "cartesian_movable_atom_indices": [1, 2],
        "additional_constraints": [],
        "initial_guess_policy": "independent_pyscf_minao",
        "budget": {
            "max_physical_calls": 3 * native_ceiling,
            "per_point_wall_seconds": 60,
        },
        "energy_recheck_tolerance_hartree": 1e-9,
        "density_recheck_tolerance": 1e-7,
        "relaxation": {
            "max_native_evaluations_per_point": native_ceiling,
            "per_evaluation_wall_seconds": 10,
            "max_optimizer_iterations": 30,
            "geometry_recheck_tolerance_bohr": 1e-6,
        },
    }
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["O", "H", "H"],
                "geometry_bohr": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.8],
                    [1.7, 0.0, -0.4],
                ],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["water-O", "water-H1", "water-H2"],
            },
            "recipe": RELAXED_SCAN_RECIPE,
            "products": ["pes_scan"],
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 180},
            "source_provenance": {"pes_scan": scan},
        }
    )


def test_exact_local_relaxed_route_and_distinct_inner_method_identity():
    request = water_request()
    review = plan_request(request, execution="local_validation")
    assert review["executable"] is True, review["blocking_reasons"]
    assert validate_request(request, execution="github_actions")["executable"] is False
    plan = review["plan"]
    assert plan["execution_scope"] == "experimental_local_constrained_relaxed_pes"
    assert plan["point_attempt_ceiling"] == 3
    assert plan["physical_call_ceiling"] == 120
    assert (
        plan["inner_solver_recipe_sha256"] == get_profile(INNER_RECIPE)["recipe_sha256"]
    )
    assert plan["recipe_sha256"] != plan["inner_solver_recipe_sha256"]
    assert len(plan["tasks"]) == 3
    assert all(
        task["operation"] == "constrained_coordinate_energy_relaxation"
        for task in plan["tasks"]
    )
    assert {
        record.tuple_definition.property
        for record in profile_capabilities(RELAXED_SCAN_RECIPE)
    } == {"energy", "gradient", "optimization"}
    body = dict(plan)
    assert body.pop("plan_sha256") == digest(body)
    scan = scan_definition(request)
    operation = point_relaxation_request(request, scan, 0)
    assert operation["specification"]["constraints"][0]["target"] == 1.9
    assert operation["specification"]["request_curvature"] is False
    assert operation["inner_recipe_sha256"] == plan["inner_solver_recipe_sha256"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("resources", "cores", 2),
        ("resources", "memory_mb", 2048),
        ("resources", "wall_seconds", 181),
        ("molecule", "atom_ids", None),
        ("scan", "coordinate_treatment", "fixed"),
        ("scan", "unscanned_coordinates", "minimum_displacement_embedding"),
        ("scan", "cartesian_movable_atom_indices", [0, 1, 2]),
        ("scan", "initial_guess_policy", "previous_point_density"),
        ("scan", "additional_constraints", ["undeclared constraint"]),
        ("scan", "scf_max_cycle", 1),
        ("scan", "relaxation", None),
    ],
)
def test_unapproved_scope_rejects_before_native_dispatch(section, key, value):
    declaration = water_request().model_dump(mode="json")
    target = (
        declaration["source_provenance"]["pes_scan"]
        if section == "scan"
        else declaration[section]
    )
    target[key] = value
    checked = validate_request(declaration, execution="local_validation")
    assert checked["executable"] is False
    assert checked["blocking_reasons"]


def test_inner_native_ceiling_cannot_be_mistaken_for_three_point_attempts():
    declaration = deepcopy(water_request().source_provenance["pes_scan"])
    declaration["budget"]["max_physical_calls"] = 3
    with pytest.raises(ValueError, match="physical calls"):
        ScanPlan.model_validate(declaration)
    declaration = water_request().model_dump(mode="json")
    declaration["resources"]["wall_seconds"] = 179
    with pytest.raises(ValueError, match="wall budget"):
        validate_request(declaration, execution="local_validation")


def test_no_more_than_three_independent_scheduled_relaxation_attempts():
    declaration = water_request().model_dump(mode="json")
    scan = declaration["source_provenance"]["pes_scan"]
    scan["grid"].append([2.0])
    scan["budget"]["max_physical_calls"] = 240
    scan["budget"]["per_point_wall_seconds"] = 30
    scan["relaxation"]["per_evaluation_wall_seconds"] = 10
    scan["passes"] = [
        {"purpose": "forward", "sample_indices": [0, 1]},
        {"purpose": "reverse", "sample_indices": [1, 0]},
        {"purpose": "challenge", "sample_indices": [0, 1]},
    ]
    checked = validate_request(declaration, execution="local_validation")
    assert checked["executable"] is False
    assert any("three scheduled" in reason for reason in checked["blocking_reasons"])


def test_initial_embedding_is_only_coordinate_mathematics():
    request = water_request()
    initial = geometry_for_sample(request, scan_definition(request), 0)
    geometry = np.asarray(initial.geometry_bohr)
    assert np.linalg.norm(geometry[1] - geometry[0]) == pytest.approx(1.9, abs=1e-8)
    np.testing.assert_array_equal(geometry[0], request.molecule.geometry_bohr[0])
    np.testing.assert_array_equal(geometry[2], request.molecule.geometry_bohr[2])
    assert initial.atom_ids == tuple(request.molecule.atom_ids)


@pytest.mark.parametrize("index", [False, -1, 1, "0"])
def test_relaxed_point_requires_an_actual_approved_index(index):
    request = water_request()
    with pytest.raises(ValueError, match="approved relaxed grid index"):
        point_relaxation_request(request, scan_definition(request), index)


def test_relaxed_point_helper_rejects_another_outer_recipe():
    declaration = water_request().model_dump(mode="json")
    declaration["recipe"] = "hf-sto-3g-internal-pes-validation"
    request = CalculationRequest.model_validate(declaration)
    with pytest.raises(ValueError, match="exact separate outer recipe"):
        point_relaxation_request(request, scan_definition(request), 0)


@pytest.mark.parametrize("singular_initial", [False, True])
def test_physical_angle_targets_and_singular_constraints_reject_at_preflight(
    singular_initial,
):
    declaration = water_request().model_dump(mode="json")
    scan = declaration["source_provenance"]["pes_scan"]
    scan["coordinates"] = [
        {
            "coordinate_id": "fixed-HOH",
            "kind": "angle",
            "atom_indices": [1, 0, 2],
            "unit": "degree",
            "domain": {"minimum": 80.0, "maximum": 180.0, "periodic": False},
        }
    ]
    scan["grid"] = [[105.0 if singular_initial else 180.0]]
    if singular_initial:
        declaration["molecule"]["geometry_bohr"][2] = [0.0, 0.0, -1.8]
    review = plan_request(declaration, execution="local_validation")
    assert review["executable"] is False
    assert any(
        "invalid inner constrained" in item for item in review["blocking_reasons"]
    )


def test_empty_relaxation_metadata_cannot_qualify_stationarity():
    with pytest.raises(ValueError):
        RelaxedPointEvidence.model_validate({})


@pytest.fixture(scope="module")
def genuine_relaxed_scan(tmp_path_factory):
    root = tmp_path_factory.mktemp("genuine-relaxed-water")
    request = water_request()
    approval = approve_plan(
        plan_request(request, execution="local_validation"),
        actor="test-local-user",
        max_scratch_bytes=16 * 1024**2,
    )
    request_path = root / "request.json"
    approval_path = root / "approval.json"
    request_path.write_bytes(canonical_json(request.model_dump(mode="json")))
    approval_path.write_bytes(canonical_json(approval))
    output = root / "scan"
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    command = [
        sys.executable,
        "-m",
        "cochem_torq",
        "execute",
        "--request",
        str(request_path),
        "--approved-plan",
        str(approval_path),
        "--output-dir",
        str(output),
        "--json",
    ]
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=210,
        check=False,
    )
    (root / "cli-stdout.txt").write_text(completed.stdout)
    (root / "cli-stderr.txt").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    response = json.loads(completed.stdout)
    assert response["status"] == "complete"
    assert response["data"]["identification_ready"] is False
    result = ScanSurface.model_validate(read_json(output / "result.json"))
    yield output, request, approval, result


@pytest.mark.real_engine
def test_genuine_free_coordinate_relaxation_through_existing_cli(genuine_relaxed_scan):
    _, request, _, result = genuine_relaxed_scan
    assert result.status == "complete", result.stop_reason
    assert result.point_attempt_count == 3
    assert result.unique_planned_samples == 1
    assert result.unique_physically_attempted_samples == 1
    assert [point.purpose for point in result.points] == [
        "forward",
        "reverse",
        "challenge",
    ]
    assert result.physical_call_count is not None
    assert 3 < result.physical_call_count <= 120
    assert result.native_evaluation_attempt_lower_bound == result.physical_call_count
    assert result.stationary_points_verified is False
    assert result.surface_completeness_established is False
    assert result.independent_scientific_qualification is False
    assert result.identification_ready is False
    for point in result.points:
        assert point.geometry_status == "constrained_stationary_sample"
        assert point.initial_molecule is not None
        assert point.molecule != point.initial_molecule
        np.testing.assert_array_equal(
            point.molecule.geometry_bohr[0], request.molecule.geometry_bohr[0]
        )
        initial = np.asarray(point.initial_molecule.geometry_bohr)
        final = np.asarray(point.molecule.geometry_bohr)
        # OH1 is constrained, while OH2 is independently allowed to relax.
        assert np.linalg.norm(final[1] - final[0]) == pytest.approx(1.9, abs=1e-8)
        assert (
            abs(
                np.linalg.norm(final[2] - final[0])
                - np.linalg.norm(initial[2] - initial[0])
            )
            > 1e-4
        )


@pytest.mark.real_engine
def test_native_final_gradient_has_independent_analytic_bond_kkt(genuine_relaxed_scan):
    _, _, _, result = genuine_relaxed_scan
    for point in result.points:
        geometry = np.asarray(point.molecule.geometry_bohr)
        gradient = np.asarray(point.gradient_hartree_bohr)[1:].ravel()
        bond = geometry[1] - geometry[0]
        analytic_row = np.r_[bond / np.linalg.norm(bond), np.zeros(3)]
        projected = gradient - analytic_row * float(analytic_row @ gradient)
        assert np.linalg.norm(projected) <= np.sqrt(5) * 1e-5
        evidence = point.relaxation_evidence
        assert evidence is not None
        assert evidence["constraint_rank"] == 1
        assert evidence["tangent_dimension"] == 5
        assert (
            max(abs(value) for value in evidence["constraint_residuals_bohr_or_radian"])
            <= 1e-8
        )
        assert evidence["stationary_character"] == "constrained_stationary"
        assert evidence["equilibrium_geometry_claimed"] is False


@pytest.mark.real_engine
def test_original_native_evaluations_and_owned_waits_are_retained(genuine_relaxed_scan):
    output, _, approval, result = genuine_relaxed_scan
    count = 0
    for point in result.points:
        assert point.relaxation_evidence is not None
        assert point.native_manifest_path is not None
        native = output / point.native_manifest_path
        assert native.name == "manifest.json"
        assert "/constrained-engine/evaluations/" in native.as_posix()
        assert sha256(native.read_bytes()).hexdigest() == point.native_manifest_sha256
        original = read_json(native.parent / "result.json")
        assert original["energy_hartree"] == point.energy_hartree
        assert original["gradient_hartree_bohr"] == [
            list(row) for row in point.gradient_hartree_bohr
        ]
        point_root = output / "points" / f"point-{point.sequence_index:04d}"
        constrained = point_root / "constrained-engine"
        original_result_path = constrained / "result.json"
        assert (
            sha256(original_result_path.read_bytes()).hexdigest()
            == (point.relaxation_evidence["result_sha256"])
        )
        original_result = read_json(original_result_path)
        assert original_result["recipe"] == INNER_RECIPE
        assert original_result["evaluations"][-1]["purpose"] == (
            "independent_final_verification"
        )
        assert original_result["physical_call_count"] == len(
            original_result["evaluations"]
        )
        count += original_result["physical_call_count"]
        assert point.host_allocation is not None
        assert point.host_allocation.process_death_proof in {
            "actual_pid_absent",
            "actual_pid_reuse",
            "actual_zombie_state",
        }
        assert point.request_sha256 == approval["plan"]["request_sha256"]
        assert point.source_identity_sha256 == digest(approval["source_identity"])
        for index, observation in enumerate(original_result["evaluations"]):
            evaluation = constrained / "evaluations" / f"evaluation-{index:04d}"
            assert read_json(evaluation / "evaluation.json") == observation
            binding_path = evaluation / "owner-binding.json"
            binding = read_json(binding_path)
            parent = read_json(evaluation / "parent-process-observation.json")
            waited = read_json(evaluation / "worker-process.json")
            assert binding["owner_pid"] == point.host_allocation.worker_pid
            assert binding["owner_create_time"] == (
                point.host_allocation.worker_create_time
            )
            assert waited["worker_pid"] == parent["worker_pid"] == binding["worker_pid"]
            assert (
                waited["worker_create_time"]
                == parent["worker_create_time"]
                == binding["worker_create_time"]
            )
            assert waited["wait_completed"] is True
            assert waited["returncode"] == 0
            assert (
                waited["owner_binding_sha256"]
                == sha256(binding_path.read_bytes()).hexdigest()
            )
        assert not (point_root / "native").exists()
    assert count == result.physical_call_count
    assert digest(source_identity()) == digest(approval["source_identity"])


@pytest.mark.real_engine
def test_density_comparisons_respect_actual_relaxed_centers(genuine_relaxed_scan):
    _, _, _, result = genuine_relaxed_scan
    assert len(result.comparisons) == 2
    by_id = {point.point_id: point for point in result.points}
    for comparison in result.comparisons:
        first = by_id[comparison.first_point_id]
        recheck = by_id[comparison.recheck_point_id]
        assert comparison.geometry_difference_max_bohr is not None
        if first.molecule == recheck.molecule:
            assert comparison.density_difference_frobenius is not None
            assert comparison.scope == (
                "independent_restart_energy_and_AO_density_same_geometry"
            )
        else:
            assert comparison.density_difference_frobenius is None
            assert comparison.scope == (
                "independent_constrained_restarts_same_targets_density_unavailable"
            )
        assert comparison.inconsistent is False


@pytest.mark.real_engine
def test_genuine_inner_budget_exhaustion_retains_missing_stationarity(tmp_path):
    request = water_request(native_ceiling=2)
    approval = approve_plan(
        plan_request(request, execution="local_validation"),
        actor="test-local-user",
        max_scratch_bytes=16 * 1024**2,
    )
    output = tmp_path / "exhausted-relaxed-scan"
    result = ScanSurface.model_validate(
        execute_request(request, output, approved_plan=approval)
    )
    assert result.status == "failed"
    assert result.point_attempt_count == 3
    assert result.physical_call_count is not None
    assert 0 < result.physical_call_count <= 6
    for point in result.points:
        assert point.status == "failed"
        assert point.energy_hartree is None
        assert point.gradient_hartree_bohr is None
        assert point.density is None
        assert point.geometry_status == "constrained_relaxation_unavailable"
        assert point.molecule == point.initial_molecule
        assert point.reason
        evidence = point.relaxation_evidence
        assert evidence is not None
        assert evidence["stationary_character"] is None
        assert evidence["constraint_residuals_bohr_or_radian"] is None
        point_root = output / "points" / f"point-{point.sequence_index:04d}"
        assert (point_root / "constrained-engine" / "result.json").is_file()
        assert list((point_root / "constrained-engine" / "evaluations").iterdir())
    assert result.identification_ready is False
