"""Genuine constrained HF optimization and independent native/math readback.

No engine behavior or physical quantities are substituted. Invalid declarations
are explicit contract inputs; physical acceptance uses actual PySCF observations.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from hashlib import sha256

import numpy as np
import psutil
import pytest
from pydantic import ValidationError

from cochem_torq.domain import Molecule, Resources, canonical_json, read_json
from cochem_torq.engines.constrained_optimization import (
    RECIPE_ID,
    ConstrainedOptimizationResult,
    ConstrainedOptimizationSpec,
    ConstraintTarget,
    _coordinate_hessians,
    _jacobian,
    execute_constrained_optimization,
)
from cochem_torq.engines.pyscf_backend import PySCFBackend
from cochem_torq.internal_coordinates import coordinate_target, coordinate_value
from cochem_torq.registry import get_profile


def water():
    return Molecule.model_validate(
        {
            "symbols": ["O", "H", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.7, 0.0, -0.4]],
            "charge": 0,
            "multiplicity": 1,
            "atom_ids": ["water-O", "water-H1", "water-H2"],
        }
    )


def specification(*, max_calls=40, curvature=False, wall=30):
    return ConstrainedOptimizationSpec.model_validate(
        {
            "constraints": [
                {
                    "coordinate": {
                        "coordinate_id": "fixed-OH1",
                        "kind": "bond",
                        "atom_indices": [0, 1],
                        "unit": "bohr",
                        "domain": {"minimum": 1.5, "maximum": 2.5, "periodic": False},
                    },
                    "target": 1.9,
                },
                {
                    "coordinate": {
                        "coordinate_id": "fixed-HOH",
                        "kind": "angle",
                        "atom_indices": [1, 0, 2],
                        "unit": "degree",
                        "domain": {
                            "minimum": 80.0,
                            "maximum": 130.0,
                            "periodic": False,
                        },
                    },
                    "target": 104.5,
                },
            ],
            "movable_atom_indices": [1, 2],
            "max_physical_calls": max_calls,
            "per_evaluation_wall_seconds": wall,
            "request_curvature": curvature,
        }
    )


def resources():
    return Resources(cores=1, memory_mb=2048, wall_seconds=1200)


def backend_request(molecule, properties):
    profile = get_profile(RECIPE_ID)
    return {
        "molecule": molecule.model_dump(mode="json"),
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
            "dispersion": None,
        },
        "properties": properties,
        "settings": {**profile["numerical"], "threads": 1, "memory_mb": 2048},
    }


@pytest.fixture(scope="module")
def optimized_water(tmp_path_factory):
    root = tmp_path_factory.mktemp("genuine-constrained-water")
    initial = water()
    result = execute_constrained_optimization(
        initial,
        specification(curvature=True),
        root / "optimization",
        resources=resources(),
        profile=get_profile(RECIPE_ID),
    )
    yield root, initial, result


@pytest.mark.real_engine
def test_actual_water_energy_relaxation_and_independent_final_readback(optimized_water):
    root, initial, result = optimized_water
    assert result.status == "available", result.reason
    assert result.final_molecule is not None and result.electronic is not None
    assert result.stationary_character == "constrained_stationary"
    assert result.optimizer["engine"] == "SciPy SLSQP"
    assert result.optimizer["version"] == "1.18.1"
    assert len(result.optimizer["installed_implementation_sha256"]) == 64
    assert result.physical_call_count == len(result.evaluations)
    assert result.physical_call_count <= result.specification.max_physical_calls
    assert result.evaluations[-1].purpose == "independent_final_verification"
    assert result.evaluations[0].molecule == initial
    assert initial == water()
    np.testing.assert_array_equal(
        result.final_molecule.geometry_bohr[0], initial.geometry_bohr[0]
    )
    assert result.final_molecule.geometry_bohr != initial.geometry_bohr
    assert result.unconstrained_minimum_claimed is False
    assert result.equilibrium_geometry_claimed is False
    assert result.independent_scientific_qualification is False
    assert (
        max(abs(value) for value in result.constraint_residuals_bohr_or_radian) <= 1e-8
    )
    assert max(abs(value) for value in result.tangent_gradient_hartree_bohr) <= 1e-5
    for item in result.specification.constraints:
        actual = coordinate_value(result.final_molecule.geometry_bohr, item.coordinate)
        assert actual == pytest.approx(
            coordinate_target(item.target, item.coordinate), abs=1e-8
        )
    assert result.constraint_rank == 2 and result.tangent_dimension == 4
    assert result.lagrange_multiplier_units == ("hartree/bohr", "hartree/radian")
    for observation in result.evaluations:
        assert observation.status == "available"
        native_manifest = root / "optimization" / observation.native_manifest_path
        assert (
            sha256(native_manifest.read_bytes()).hexdigest()
            == observation.native_manifest_sha256
        )
        original = read_json(native_manifest.parent / "result.json")
        assert original["energy_hartree"] == observation.electronic.energy_hartree
        assert original["gradient_hartree_bohr"] == [
            list(row) for row in observation.gradient_hartree_bohr
        ]
        parent_observation = read_json(
            native_manifest.parent.parent / "parent-process-observation.json"
        )
        binding_path = native_manifest.parent.parent / "owner-binding.json"
        binding = read_json(binding_path)
        waited = read_json(native_manifest.parent.parent / "worker-process.json")
        assert parent_observation["worker_pid"] == binding["worker_pid"]
        assert parent_observation["worker_create_time"] == binding["worker_create_time"]
        assert waited["worker_pid"] == binding["worker_pid"]
        assert waited["worker_create_time"] == binding["worker_create_time"]
        assert waited["wait_completed"] is True and waited["returncode"] == 0
        assert (
            waited["owner_binding_sha256"]
            == sha256(binding_path.read_bytes()).hexdigest()
        )
    # A fresh same-engine call verifies numerical software consistency. It is
    # expressly not an independent cross-engine or experimental accuracy claim.
    independent = PySCFBackend().evaluate(
        backend_request(result.final_molecule, ["energy", "gradient"]),
        root / "independent-final-reference",
    )
    assert independent["status"] == "complete"
    assert independent["energy_hartree"] == pytest.approx(
        result.electronic.energy_hartree, abs=1e-10
    )
    np.testing.assert_allclose(
        independent["gradient_hartree_bohr"],
        result.gradient_hartree_bohr,
        atol=1e-9,
        rtol=0,
    )
    # Independent SCF energies at exact radial variations of the free OH bond
    # establish a derivative check without changing either fixed constraint.
    geometry = np.asarray(result.final_molecule.geometry_bohr)
    direction = geometry[2] - geometry[0]
    direction /= np.linalg.norm(direction)
    step = 1e-3
    energies = []
    for sign in (-1, 1):
        payload = result.final_molecule.model_dump(mode="json")
        shifted = geometry.copy()
        shifted[2] += sign * step * direction
        payload["geometry_bohr"] = shifted.tolist()
        calculation = PySCFBackend().evaluate(
            backend_request(Molecule.model_validate(payload), ["energy"]),
            root / f"radial-energy-reference-{sign}",
        )
        assert calculation["status"] == "complete"
        energies.append(calculation["energy_hartree"])
    derivative = (energies[1] - energies[0]) / (2 * step)
    analytic = float(np.asarray(result.gradient_hartree_bohr)[2] @ direction)
    assert derivative == pytest.approx(analytic, abs=2e-6)


def test_actual_constrained_curvature_has_lagrangian_terms_and_real_hessian(
    optimized_water,
):
    _, _, result = optimized_water
    assert result.status == "available", result.reason
    assert result.curvature.status == "available", result.curvature.reason
    assert result.native_result["hessian_evidence"]["derivative"] == "analytic"
    tangent = np.asarray(result.curvature.tangent_basis_columns)
    hessian = np.asarray(result.curvature.lagrangian_hessian_hartree_bohr2)
    jacobian = _jacobian(
        np.asarray(result.final_molecule.geometry_bohr), result.specification
    )
    np.testing.assert_allclose(jacobian @ tangent, 0, atol=1e-9)
    np.testing.assert_allclose(tangent.T @ tangent, np.eye(4), atol=1e-10)
    projected = tangent.T @ hessian @ tangent
    np.testing.assert_allclose(
        projected, result.curvature.projected_hessian_hartree_bohr2, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(projected),
        result.curvature.eigenvalues_hartree_bohr2,
        atol=1e-12,
    )
    assert result.curvature.two_scale_max_difference_hartree_bohr2 <= 1e-5
    assert result.curvature.unconstrained_minimum_claimed is False
    assert result.curvature.coordinate_hessians_two_steps["steps_bohr"] == [1e-4, 5e-5]
    assert result.curvature.character != "negative_curvature"


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_rows",
        "negative_row",
        "boolean_row",
        "one_call",
        "boolean_curvature",
        "duplicate_coordinate",
        "outside_target",
        "linear_target",
    ],
)
def test_invalid_optimizer_declaration_is_rejected(change):
    raw = specification().model_dump(mode="json")
    if change == "duplicate_rows":
        raw["movable_atom_indices"] = [1, 1]
    elif change == "negative_row":
        raw["movable_atom_indices"] = [-1]
    elif change == "boolean_row":
        raw["movable_atom_indices"] = [True]
    elif change == "one_call":
        raw["max_physical_calls"] = 1
    elif change == "boolean_curvature":
        raw["request_curvature"] = 1
    elif change == "duplicate_coordinate":
        raw["constraints"][1]["coordinate"]["coordinate_id"] = raw["constraints"][0][
            "coordinate"
        ]["coordinate_id"]
    elif change == "outside_target":
        raw["constraints"][0]["target"] = 0.5
    else:
        raw["constraints"][1]["coordinate"]["domain"] = {
            "minimum": 0.0,
            "maximum": 180.0,
            "periodic": False,
        }
        raw["constraints"][1]["target"] = 180.0
    with pytest.raises(ValidationError):
        ConstrainedOptimizationSpec.model_validate(raw)


def test_periodic_constraint_uses_physical_half_open_domain():
    raw = {
        "coordinate": {
            "coordinate_id": "exact-torsion",
            "kind": "dihedral",
            "atom_indices": [0, 1, 2, 3],
            "unit": "radian",
            "domain": {
                "minimum": -math.pi,
                "maximum": math.pi,
                "periodic": True,
                "period": 2 * math.pi,
            },
        },
        "target": -math.pi,
    }
    ConstraintTarget.model_validate(raw)
    raw["target"] = math.pi
    with pytest.raises(ValidationError):
        ConstraintTarget.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["basis", "optimizer_version", "dispersion", "method"]
)
def test_unreviewed_profile_is_rejected_before_native_dispatch(tmp_path, change):
    profile = deepcopy(get_profile(RECIPE_ID))
    profile[change] = "explicit mismatched protocol input"
    with pytest.raises(ValueError, match="exact registered"):
        execute_constrained_optimization(
            water(),
            specification(),
            tmp_path / "not-dispatched",
            resources=resources(),
            profile=profile,
        )
    assert not (tmp_path / "not-dispatched").exists()


def test_dependent_constraints_are_rejected_before_engine_dispatch(tmp_path):
    raw = specification().model_dump(mode="json")
    duplicate = deepcopy(raw["constraints"][0])
    duplicate["coordinate"]["coordinate_id"] = "independent-label-same-bond"
    raw["constraints"] = [raw["constraints"][0], duplicate]
    with pytest.raises(ValueError, match="singular or dependent"):
        execute_constrained_optimization(
            water(),
            ConstrainedOptimizationSpec.model_validate(raw),
            tmp_path / "dependent",
            resources=resources(),
            profile=get_profile(RECIPE_ID),
        )
    assert not (tmp_path / "dependent").exists()


def test_invalid_atom_mapping_is_rejected_before_engine_dispatch(tmp_path):
    raw = specification().model_dump(mode="json")
    raw["movable_atom_indices"] = [3]
    with pytest.raises(ValueError, match="atom indices"):
        execute_constrained_optimization(
            water(),
            ConstrainedOptimizationSpec.model_validate(raw),
            tmp_path / "invalid-map",
            resources=resources(),
            profile=get_profile(RECIPE_ID),
        )


def test_missing_chemical_atom_ids_is_rejected_before_engine_dispatch(tmp_path):
    payload = water().model_dump(mode="json")
    payload["atom_ids"] = None
    with pytest.raises(ValueError, match="stable atom IDs"):
        execute_constrained_optimization(
            Molecule.model_validate(payload),
            specification(),
            tmp_path / "missing-map",
            resources=resources(),
            profile=get_profile(RECIPE_ID),
        )


def test_constraint_hessian_matches_independent_analytic_bond_definition():
    raw = specification().model_dump(mode="json")
    raw["constraints"] = raw["constraints"][:1]
    spec = ConstrainedOptimizationSpec.model_validate(raw)
    geometry = np.asarray(water().geometry_bohr)
    matrices = _coordinate_hessians(geometry, spec, 1e-4)
    vector = geometry[1] - geometry[0]
    length = np.linalg.norm(vector)
    direction = vector / length
    exact = (np.eye(3) - np.outer(direction, direction)) / length
    np.testing.assert_allclose(matrices[0, :3, :3], exact, atol=1e-8)
    np.testing.assert_array_equal(matrices[0, 3:, :], np.zeros((3, 6)))
    np.testing.assert_array_equal(matrices[0, :, 3:], np.zeros((6, 3)))


@pytest.mark.real_engine
def test_actual_budget_stop_preserves_available_observations_without_stationarity(
    tmp_path,
):
    result = execute_constrained_optimization(
        water(),
        specification(max_calls=2),
        tmp_path / "bounded",
        resources=resources(),
        profile=get_profile(RECIPE_ID),
    )
    assert result.status == "partial"
    assert result.physical_call_count == 1
    assert result.evaluations[0].status == "available"
    assert result.evaluations[0].electronic is not None
    assert result.final_molecule is None and result.electronic is None
    assert result.stationary_character is None
    assert "budget exhausted" in result.reason
    assert result.curvature.status == "unavailable"
    assert (
        ConstrainedOptimizationResult.model_validate(
            read_json(tmp_path / "bounded" / "result.json")
        )
        == result
    )


@pytest.mark.real_engine
def test_actual_worker_wall_limit_returns_missing_results(tmp_path):
    result = execute_constrained_optimization(
        water(),
        specification(wall=1),
        tmp_path / "wall-limited",
        resources=resources(),
        profile=get_profile(RECIPE_ID),
    )
    assert result.status == "failed"
    assert result.physical_call_count == 1
    assert result.evaluations[0].status == "failed"
    assert result.evaluations[0].electronic is None
    assert result.final_molecule is None and result.native_result is None
    assert result.curvature.status == "unavailable"
    assert "wall ceiling" in result.reason
    assert (
        tmp_path / "wall-limited" / "evaluations" / "evaluation-0000" / "worker.log"
    ).is_file()


@pytest.mark.real_engine
def test_actual_owner_sigkill_terminates_observed_native_worker(tmp_path):
    # A separated eight-water cluster makes the actual SCF process observable;
    # this process-lifecycle test makes no chemical-accuracy claim for its state.
    molecule = water().model_dump(mode="json")
    single_geometry = np.asarray(molecule["geometry_bohr"])
    molecule["symbols"] = molecule["symbols"] * 8
    molecule["atom_ids"] = [f"cluster-atom-{index}" for index in range(24)]
    molecule["geometry_bohr"] = np.concatenate(
        [single_geometry + [8.0 * index, 0.0, 0.0] for index in range(8)]
    ).tolist()
    declaration = {
        "molecule": molecule,
        "specification": specification().model_dump(mode="json"),
        "resources": resources().model_dump(mode="json"),
    }
    request = tmp_path / "owner-input.json"
    request.write_bytes(canonical_json(declaration))
    owner_code = """
import sys
from cochem_torq.domain import Molecule, Resources, read_json
from cochem_torq.engines.constrained_optimization import (
    ConstrainedOptimizationSpec, RECIPE_ID, execute_constrained_optimization,
)
from cochem_torq.registry import get_profile
payload = read_json(sys.argv[1])
execute_constrained_optimization(
    Molecule.model_validate(payload['molecule']),
    ConstrainedOptimizationSpec.model_validate(payload['specification']),
    sys.argv[2], resources=Resources.model_validate(payload['resources']),
    profile=get_profile(RECIPE_ID),
)
"""
    workspace = tmp_path / "owner-workspace"
    child = None
    observed = None
    with (tmp_path / "owner.log").open("wb") as log:
        owner = subprocess.Popen(
            [sys.executable, "-c", owner_code, str(request), str(workspace)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            binding = workspace / "evaluations/evaluation-0000/owner-binding.json"
            engine_log = workspace / "evaluations/evaluation-0000/native/pyscf.log"
            while time.monotonic() < deadline:
                if binding.is_file() and engine_log.is_file():
                    observed = read_json(binding)
                    child = psutil.Process(observed["worker_pid"])
                    assert child.create_time() == observed["worker_create_time"]
                    assert child.status() != psutil.STATUS_ZOMBIE
                    break
                assert owner.poll() is None, (tmp_path / "owner.log").read_text()
                time.sleep(0.005)
            assert child is not None and observed is not None
            assert observed["owner_pid"] == owner.pid
            assert (
                observed["owner_create_time"] == psutil.Process(owner.pid).create_time()
            )
            assert observed["parent_death_signal"] == int(signal.SIGKILL)
            owner.kill()
            assert owner.wait(timeout=5) == -signal.SIGKILL
            deadline = time.monotonic() + 10
            termination = None
            while time.monotonic() < deadline:
                try:
                    state = child.status()
                except psutil.NoSuchProcess:
                    termination = "process_absent"
                    break
                if state == psutil.STATUS_ZOMBIE:
                    termination = "observed_dead_zombie"
                    break
                time.sleep(0.01)
            (tmp_path / "owner-death-observation.json").write_bytes(
                canonical_json(
                    {
                        "binding": observed,
                        "owner_returncode": owner.returncode,
                        "child_termination_observation": termination,
                        "native_engine_log_observed_before_kill": True,
                    }
                )
            )
            assert termination is not None, (
                "An actual native engine survived owner death."
            )
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)
            if child is not None:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        os.killpg(child.pid, signal.SIGKILL)
                except psutil.NoSuchProcess:
                    pass
