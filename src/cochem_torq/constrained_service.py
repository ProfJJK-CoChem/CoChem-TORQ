"""Request and plan binding for explicitly constrained local energy optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .domain import CalculationRequest, digest, scientific_cache_key
from .engines.constrained_optimization import ConstrainedOptimizationSpec
from .internal_coordinates import constraint_jacobian


def constrained_specification(
    request: CalculationRequest,
) -> ConstrainedOptimizationSpec:
    raw = request.source_provenance.get("constrained_optimization")
    if not isinstance(raw, dict):
        raise ValueError(
            "An explicit constrained_optimization specification is required."
        )
    spec = ConstrainedOptimizationSpec.model_validate(raw)
    if request.recipe != "hf-sto-3g-constrained-pes-validation" or request.products != [
        "constrained_geometry"
    ]:
        raise ValueError(
            "Use the standalone constrained geometry product and exact recipe."
        )
    count = len(request.molecule.symbols)
    if any(atom >= count for atom in spec.movable_atom_indices) or any(
        max(item.coordinate.atom_indices) >= count for item in spec.constraints
    ):
        raise ValueError(
            "Coordinate and movable atom indices exceed the original molecule."
        )
    if spec.per_evaluation_wall_seconds > request.resources.wall_seconds:
        raise ValueError("A native evaluation exceeds the total approved wall ceiling.")
    coordinates = tuple(item.coordinate for item in spec.constraints)
    columns = [
        3 * atom + axis for atom in spec.movable_atom_indices for axis in range(3)
    ]
    jacobian = constraint_jacobian(request.molecule.geometry_bohr, coordinates)[
        :, columns
    ]
    if np.linalg.matrix_rank(jacobian, tol=spec.rank_tolerance) != len(coordinates):
        raise ValueError(
            "Declared constraints are dependent or cannot move their targets."
        )
    return spec


def constrained_plan_for_request(
    request: CalculationRequest, profile: dict[str, Any]
) -> dict[str, Any]:
    """Declare every possible native call without inventing an optimizer trajectory."""
    spec = constrained_specification(request)
    tasks = [
        {
            "id": f"constrained-evaluation-{index:04d}",
            "depends_on": []
            if index == 0
            else [f"constrained-evaluation-{index - 1:04d}"],
            "operation": "genuine_energy_and_analytic_gradient",
            "engine": profile["engine"],
            "recipe_sha256": profile["recipe_sha256"],
            "conditional": True,
            "geometry_policy": "actual constrained optimizer trajectory",
            "native_properties": ["energy", "gradient"],
            "fresh_final_verification_properties": ["energy", "gradient"]
            + (["hessian"] if spec.request_curvature else []),
        }
        for index in range(spec.max_physical_calls)
    ]
    plan = {
        "schema_version": "cochem.torq.plan/1",
        "serialization_profile": request.serialization_profile,
        "capability_purpose": "controlled_validation",
        "request": request.model_dump(mode="json"),
        "request_sha256": digest(request.model_dump(mode="json")),
        "scientific_cache_key": scientific_cache_key(request, profile),
        "scientific_recipe": profile,
        "requested_products": request.products,
        "tasks": tasks,
        "resources": request.resources.model_dump(mode="json"),
        "constraints": spec.model_dump(mode="json"),
        "planned_physical_call_ceiling": spec.max_physical_calls,
        "curvature_requested": spec.request_curvature,
        "curvature_derivative": "analytic_HF_hessian"
        if spec.request_curvature
        else "not_requested",
        "final_verification_policy": (
            "reserve one physical call inside the declared ceiling for an "
            "independent energy/gradient and requested analytic Hessian"
        ),
        "recovery": "same_exact_model_and_constraints_only",
        "execution_environment": "isolated_linux_cpu_worker",
        "unconstrained_equilibrium_claimed": False,
        "search_completeness_claimed": False,
    }
    return {**plan, "plan_sha256": digest(plan)}


def execute_constrained_worker(
    request: CalculationRequest, directory: Path
) -> dict[str, Any]:
    """Preserve authentic constrained results in the ordinary immutable shard."""
    from .application import _checkpoint, _empty_result, _stage
    from .engines.constrained_optimization import execute_constrained_optimization
    from .registry import get_profile

    profile = get_profile(request.recipe)
    spec = constrained_specification(request)
    result = _empty_result(
        request,
        profile,
        "This constrained PES product does not establish equilibrium spectroscopy.",
    )
    for name in result["stages"]:
        result["stages"][name] = _stage(
            "blocked",
            name,
            reason="Equilibrium spectroscopy is outside this constrained PES request.",
            absence_kind="not_requested",
        )
    result["stages"]["constrained_geometry"] = _stage(
        "blocked",
        "constrained_stationary_geometry",
        reason="The declared constrained native optimization has not completed.",
    )
    _checkpoint(directory, result)
    observation = execute_constrained_optimization(
        request.molecule,
        spec,
        directory / "constrained-engine",
        resources=request.resources,
        profile=profile,
    )
    result["constrained_optimization"] = observation.model_dump(mode="json")
    result["native_result"] = observation.native_result or {}
    if observation.electronic is not None:
        result["stages"]["electronic_structure"] = _stage(
            "available",
            "electronic_energy",
            value=observation.electronic.model_dump(mode="json"),
            flags=["constrained_PES_geometry", "accuracy_uncalibrated"],
        )
    if observation.status == "available":
        result["stages"]["constrained_geometry"] = _stage(
            "available",
            "constrained_stationary_geometry",
            value=observation.model_dump(mode="json"),
            parents=["electronic_structure"],
            flags=list(observation.quality_flags),
        )
        result["status"] = "complete"
        if spec.request_curvature and observation.curvature.status != "available":
            result["status"] = "partial"
            result["errors"].append(
                {
                    "code": "CONSTRAINED_CURVATURE_UNAVAILABLE",
                    "message": observation.curvature.reason,
                }
            )
    else:
        result["stages"]["constrained_geometry"] = _stage(
            "failed" if observation.status == "failed" else "unavailable",
            "constrained_stationary_geometry",
            reason=observation.reason,
            flags=list(observation.quality_flags),
        )
        result["status"] = "partial" if observation.electronic is not None else "failed"
        result["errors"].append(
            {
                "code": "CONSTRAINED_OPTIMIZATION_UNAVAILABLE",
                "message": observation.reason,
            }
        )
    _checkpoint(directory, result)
    return result
