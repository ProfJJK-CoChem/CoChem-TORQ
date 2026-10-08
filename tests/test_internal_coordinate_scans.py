"""Coordinate mathematics and genuine approved multidimensional molecular scans."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cochem_torq.application import validate_request
from cochem_torq.domain import CalculationRequest, digest, read_json
from cochem_torq.internal_coordinates import (
    constraint_jacobian,
    coordinate_residual,
    coordinate_value,
    embed_fixed_coordinates,
)
from cochem_torq.scan import (
    ApprovedScanExecutor,
    InternalCoordinate,
    ScanPlan,
    geometry_for_sample,
    scan_definition,
)
from cochem_torq.service import approve_plan, plan_request


def coordinate(kind, atoms):
    angular = kind != "bond"
    periodic = kind == "dihedral"
    return InternalCoordinate.model_validate(
        {
            "coordinate_id": kind + str(atoms),
            "kind": kind,
            "atom_indices": atoms,
            "unit": "radian" if angular else "bohr",
            "domain": {
                "minimum": -math.pi if periodic else 0.2,
                "maximum": math.pi if periodic else 3.0,
                "periodic": periodic,
                **({"period": 2 * math.pi} if periodic else {}),
            },
        }
    )


def water_request(*, initial_guess_policy="independent_pyscf_minao"):
    scan = ScanPlan.model_validate(
        {
            "coordinates": [
                coordinate("bond", [0, 1]).model_dump(mode="json"),
                coordinate("angle", [1, 0, 2]).model_dump(mode="json"),
            ],
            "grid": [[1.8, 1.8], [1.9, 1.9]],
            "sampling_strategy": "full_grid",
            "passes": [
                {"purpose": "forward", "sample_indices": [0, 1]},
                {"purpose": "reverse", "sample_indices": [1, 0]},
                {"purpose": "challenge", "sample_indices": [0, 1]},
            ],
            "coordinate_treatment": "fixed",
            "unscanned_coordinates": "minimum_displacement_embedding",
            "cartesian_movable_atom_indices": [1, 2],
            "additional_constraints": [],
            "initial_guess_policy": initial_guess_policy,
            "budget": {"max_physical_calls": 6, "per_point_wall_seconds": 30},
            "energy_recheck_tolerance_hartree": 1e-9,
            "density_recheck_tolerance": 1e-7,
        }
    )
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["O", "H", "H"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.7, 0.0, -0.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["oxygen", "hydrogen1", "hydrogen2"],
            },
            "recipe": "hf-sto-3g-internal-pes-validation",
            "products": ["pes_scan"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 240},
            "source_provenance": {"pes_scan": scan.model_dump(mode="json")},
        }
    )


def test_independent_bond_jacobian_and_angle_geometry():
    c = coordinate("bond", [0, 1])
    g = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    assert coordinate_value(g, c) == 3.0
    expected = np.r_[-g[1] / 3.0, g[1] / 3.0]
    np.testing.assert_allclose(constraint_jacobian(g, [c])[0], expected, atol=1e-10)
    a = coordinate("angle", [0, 1, 2])
    assert coordinate_value(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], a
    ) == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("torsion", [-3.13, -1.2, 0.0, 1.2, 3.13])
def test_signed_torsion_matches_independent_cartesian_definition(torsion):
    c = coordinate("dihedral", [0, 1, 2, 3])
    geometry = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [math.cos(torsion), math.sin(torsion), 1.0],
    ]
    assert coordinate_value(geometry, c) == pytest.approx(torsion)
    assert coordinate_residual(torsion + 2 * math.pi, torsion, c) == pytest.approx(
        0.0, abs=5e-15
    )


def test_coupled_embedding_preserves_frozen_rows_and_explicit_math_identity():
    request = water_request()
    scan = scan_definition(request)
    point = geometry_for_sample(request, scan, 1)
    np.testing.assert_array_equal(
        point.geometry_bohr[0], request.molecule.geometry_bohr[0]
    )
    for c, v in zip(scan.coordinates, scan.grid[1]):
        assert coordinate_value(point.geometry_bohr, c) == pytest.approx(v, abs=1e-8)
    data = embed_fixed_coordinates(
        request.molecule.geometry_bohr, scan.coordinates, scan.grid[1], (1, 2)
    )
    assert data["evidence_class"] == "derived_coordinate_mathematics"
    assert data["stationary_point_claimed"] is False
    assert data["jacobian_rank"] == 2


def test_periodic_embedding_handles_branch_cut_without_duplicate_endpoint():
    c = coordinate("dihedral", [0, 1, 2, 3])
    geometry = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.02, 1.0]]
    result = embed_fixed_coordinates(geometry, [c], [-3.12], (3,))
    assert coordinate_value(result["geometry_bohr"], c) == pytest.approx(
        -3.12, abs=1e-8
    )
    np.testing.assert_array_equal(np.array(result["geometry_bohr"])[:3], geometry[:3])


@pytest.mark.parametrize("target", [1.2, 1.3, 1.4, 1.5, 1.6])
def test_translated_hydrogen_embedding_uses_independent_residual_and_kkt(target):
    reference = [[0.3, 0.4, 0.0], [0.3, 0.4, 1.4]]
    c = coordinate("bond", [0, 1])
    result = embed_fixed_coordinates(reference, [c], [target], (1,))
    np.testing.assert_array_equal(result["geometry_bohr"][0], reference[0])
    assert coordinate_value(result["geometry_bohr"], c) == pytest.approx(
        target, abs=1e-8
    )
    assert abs(result["residuals_bohr_or_radian"][0]) <= 1e-8
    assert result["objective_stationarity_residual"] <= 1e-7
    assert result["stationary_point_claimed"] is False


@pytest.mark.parametrize("indices", [(), (0, 0), (-1,), (3,), (True,)])
def test_invalid_movable_rows_reject(indices):
    with pytest.raises(ValueError):
        embed_fixed_coordinates(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [coordinate("bond", [0, 1])],
            [1.4],
            indices,
        )


def test_dependent_coordinates_and_undefined_torsion_reject():
    c = coordinate("bond", [0, 1])
    with pytest.raises(ValueError, match="dependent"):
        embed_fixed_coordinates(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [c, c], [1.4, 1.5], (1,)
        )
    with pytest.raises(ValueError, match="Collinear"):
        coordinate_value(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 0.0, 3.0]],
            coordinate("dihedral", [0, 1, 2, 3]),
        )


def test_local_only_internal_scan_approval_and_digest():
    request = water_request()
    checked = validate_request(request, execution="local_validation")
    assert checked["executable"] is True
    assert validate_request(request, execution="github_actions")["executable"] is False
    plan = plan_request(request, execution="local_validation")["plan"]
    assert plan["coordinate_construction"] == "minimum_displacement_embedding"
    assert plan["physical_call_ceiling"] == 6
    body = dict(plan)
    declared = body.pop("plan_sha256")
    assert declared == digest(body)


def test_checkpoint_schedule_declares_actual_parent_dependencies():
    request = water_request(initial_guess_policy="previous_point_density")
    review = plan_request(request, execution="local_validation")
    assert review["executable"] is True
    tasks = review["plan"]["tasks"]
    assert [task["depends_on"] for task in tasks] == [
        [],
        ["point-0000"],
        [],
        ["point-0002"],
        [],
        [],
    ]
    assert [task["initial_guess"] for task in tasks] == [
        "independent_pyscf_minao",
        "previous_point_density",
        "independent_pyscf_minao",
        "previous_point_density",
        "independent_pyscf_minao",
        "independent_pyscf_minao",
    ]


@pytest.mark.real_engine
def test_genuine_approved_coupled_water_scan(tmp_path):
    request = water_request()
    approved = approve_plan(
        plan_request(request, execution="local_validation"), actor="test-local-user"
    )
    with ApprovedScanExecutor(approved, tmp_path / "water") as executor:
        result = executor.run()
    assert result.status == "complete"
    assert result.physical_call_count == 6
    assert result.stationary_points_verified is False
    assert result.independent_scientific_qualification is False
    assert not any(c.inconsistent for c in result.comparisons)
    for point in result.points:
        assert point.energy_hartree is not None
        native = tmp_path / "water" / point.native_manifest_path
        assert native.is_file()
        assert (
            read_json(native.parent / "result.json")["energy_hartree"]
            == point.energy_hartree
        )
        np.testing.assert_array_equal(
            point.molecule.geometry_bohr[0], request.molecule.geometry_bohr[0]
        )


@pytest.mark.real_engine
def test_genuine_water_forward_reverse_checkpoint_and_independent_challenges(tmp_path):
    request = water_request(initial_guess_policy="previous_point_density")
    approved = approve_plan(
        plan_request(request, execution="local_validation"), actor="test-local-user"
    )
    with ApprovedScanExecutor(approved, tmp_path / "continued-water") as executor:
        result = executor.run()
    assert result.status == "complete"
    assert result.stop_reason is None
    assert result.physical_call_count == 6
    for index, point in enumerate(result.points):
        native = tmp_path / "continued-water" / point.native_manifest_path
        record = read_json(native.parent / "result.json")
        if index in {1, 3}:
            parent = result.points[index - 1]
            assert point.parent_point_ids == (parent.point_id,)
            consumed = record["checkpoint_consumption"]
            assert consumed["engine_checkpoint_reused"] is True
            assert consumed["fresh_scf_kernel_called"] is True
            assert consumed["scf_iteration_state_resumed"] is False
            assert consumed["source_manifest_sha256"] == parent.native_manifest_sha256
            assert (
                consumed["source_checkpoint_sha256"]
                == parent.density.native_checkpoint_sha256
            )
            assert consumed["changed_fields"] == ["geometry_bohr"]
        else:
            assert point.parent_point_ids == ()
            assert "checkpoint_consumption" not in record
    assert result.independent_scientific_qualification is False
