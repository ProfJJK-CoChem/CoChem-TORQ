"""The student-facing application: explicit plans, real engines and durable evidence."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .artifacts import publish_shard, seal_shard, verify_shard
from .domain import (
    PRODUCT_TO_STAGE,
    CalculationRequest,
    PrerequisiteError,
    StageResult,
    canonical_json,
    digest,
    read_json,
    scientific_cache_key,
)
from .domain import SPECTROSCOPY_STAGES as STAGES
from .registry import get_profile

if TYPE_CHECKING:
    from .service import ApprovedPlan

_ADVANCED = {
    "anharmonic_force_field",
    "vpt2",
    "ground_state_constants",
    "identification_catalog",
}


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


RUNTIME_SOURCE_MODULES = (
    "Libraries.cochem_isotopes",
    "cochem.orchestration.campaign",
    "cochem.orchestration.campaign_backup",
    "cochem.orchestration.host_allocation",
    "cochem.storage.fenced_pes",
)


def source_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "distribution": "CoChem-TORQ",
        "code_sha256": None,
        "git_commit": None,
        "working_tree_dirty": None,
    }
    records = []
    for path in sorted(package_root.rglob("*.py")):
        records.append(
            {
                "path": f"cochem_torq/{path.relative_to(package_root).as_posix()}",
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    import importlib.util

    for module in RUNTIME_SOURCE_MODULES:
        specification = importlib.util.find_spec(module)
        if specification is None or not specification.origin:
            raise RuntimeError(
                f"Required scientific runtime module is unavailable: {module}"
            )
        records.append(
            {
                "path": module,
                "sha256": sha256(Path(specification.origin).read_bytes()).hexdigest(),
            }
        )
    identity["code_sha256"] = digest(records)
    repository = next(
        (path for path in package_root.parents if (path / ".git").exists()), None
    )
    if repository is not None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        identity["git_commit"] = result.stdout.strip()
        state = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        identity["working_tree_dirty"] = bool(state.stdout)
    identity["declared_workflow_commit"] = os.environ.get("COCHEM_SOURCE_COMMIT")
    identity["declared_engine_image"] = os.environ.get("COCHEM_IMAGE_ID")
    return identity


def validate_request(
    request: dict[str, Any] | CalculationRequest, *, execution: str = "github_actions"
) -> dict[str, Any]:
    model = (
        request
        if isinstance(request, CalculationRequest)
        else CalculationRequest.model_validate(request)
    )
    profile = get_profile(model.recipe)
    from .registry import route_profile_capabilities

    capabilities = route_profile_capabilities(
        model.recipe, model.products, execution=execution
    )
    _masses(
        model
    )  # Fail before expensive work if a requested isotope lacks real mass data.
    reasons = list(capabilities["blocking_reasons"])
    if "constrained_geometry" in model.products:
        from .constrained_service import constrained_specification

        constrained_specification(model)
        if execution != "local_validation":
            reasons.append(
                "Constrained optimization requires explicit local validation."
            )
    elif (
        model.recipe == "hf-sto-3g-constrained-pes-validation"
        or "constrained_optimization" in model.source_provenance
    ):
        reasons.append(
            "Constrained specifications require their separate product and recipe."
        )
    if "pes_scan" in model.products:
        from .scan import scan_blocking_reasons, scan_definition

        if model.products != ["pes_scan"]:
            reasons.append(
                "A PES scan is a separate product from equilibrium spectroscopy."
            )
        if model.recipe not in {
            "hf-sto-3g-pes-validation",
            "hf-sto-3g-internal-pes-validation",
        }:
            reasons.append(
                "PES scans require their explicitly named local validation recipe."
            )
        reasons.extend(scan_blocking_reasons(model, scan_definition(model)))
        if execution != "local_validation":
            reasons.append("PES scans require the explicit local validation workflow.")
    elif model.recipe in {
        "hf-sto-3g-pes-validation",
        "hf-sto-3g-internal-pes-validation",
    } or any(key in model.source_provenance for key in ("pes_scan", "adaptive_scan")):
        reasons.append(
            "A scan declaration requires the separate pes_scan product and recipe."
        )
    if model.scientific_goal is not None and any(
        declaration.product_class in {"B", "C", "BENCHMARK"}
        for declaration in model.scientific_goal.products
    ):
        reasons.append(
            "Anchored, difference and independent benchmark goals require "
            "their separately qualified campaign implementations; this "
            "single-molecule worker cannot supply them."
        )
    if not profile.get("runnable"):
        reasons.append(profile["reason"])
    if profile.get("runnable"):
        if len(model.molecule.symbols) > profile["max_atoms"]:
            reasons.append(
                f"This bounded profile permits at most {profile['max_atoms']} atoms."
            )
        if set(model.molecule.symbols) - set(profile["elements"]):
            reasons.append(
                "This recipe requires a separately validated basis/element scope "
                "for those atoms."
            )
        if profile.get("reference") == "unrestricted":
            if model.molecule.multiplicity not in profile["multiplicities"]:
                reasons.append(
                    "The explicit unrestricted multiplicity exceeds "
                    "this validation profile."
                )
            if model.molecule.charge not in profile["charges"]:
                reasons.append(
                    "The explicit unrestricted charge exceeds this validation profile."
                )
            if model.products != ["geometry"]:
                reasons.append(
                    "Unrestricted validation supports only the separately typed "
                    "unclassified geometry product."
                )
            if execution != "local_validation":
                reasons.append(
                    "Unrestricted profiles require explicit local validation."
                )
        elif model.molecule.multiplicity != profile["multiplicity"]:
            reasons.append(
                "Only the explicitly supported restricted closed-shell singlet is "
                "runnable."
            )
    if execution == "github_actions":
        if (
            model.resources.cores > 2
            or model.resources.memory_mb > 2048
            or model.resources.wall_seconds > 600
        ):
            reasons.append(
                "The public Actions classroom worker permits at most 2 CPU cores, "
                "2048 MB and 600 wall seconds."
            )
        if model.recipe not in {"hf-sto-3g-education", "hf-cc-pvdz-research"}:
            reasons.append(
                "This recipe requires a separate qualified calculation workflow."
            )
        if (
            abs(model.molecule.charge) > 4
            or np.max(np.abs(model.molecule.geometry_bohr)) > 1e4
        ):
            reasons.append(
                "The public Actions worker requires bounded charge and coordinates."
            )
    elif execution != "local_validation":
        raise ValueError("Unknown execution environment.")
    blocked = sorted(set(model.products) - set(profile.get("products", [])))
    normalized = model.model_dump(mode="json")
    if execution == "github_actions" and len(canonical_json(normalized)) > 16384:
        reasons.append(
            "The normalized Actions request exceeds its 16 KiB transport limit."
        )
    return {
        "valid": True,
        "executable": not reasons,
        "request": normalized,
        "request_sha256": digest(normalized),
        "scientific_cache_key": scientific_cache_key(model, profile),
        "recipe": profile,
        "capabilities": capabilities,
        "blocked_products": blocked,
        "blocking_reasons": reasons,
        "execution": execution,
        "warnings": ([profile["accuracy"]] if "accuracy" in profile else [])
        + (
            [f"Requested stages remain blocked: {', '.join(blocked)}"]
            if blocked
            else []
        ),
        "plan": create_plan(model, profile),
    }


def create_plan(request: CalculationRequest, profile: dict[str, Any]) -> dict[str, Any]:
    """Immutable dependency closure; no unplanned method or engine switching."""
    if profile.get("reference") == "unrestricted":
        from .open_shell_service import open_shell_plan_for_request

        return open_shell_plan_for_request(request, profile)
    if "pes_scan" in request.products:
        from .scan import scan_plan_for_request

        return scan_plan_for_request(request, profile)
    if "constrained_geometry" in request.products:
        from .constrained_service import constrained_plan_for_request

        return constrained_plan_for_request(request, profile)
    from .registry import profile_capabilities

    needs_hessian = bool(
        set(request.products)
        & ({"harmonic", "equilibrium_constants", "rigid_rotor_catalog"} | _ADVANCED)
    )
    tasks = [
        {
            "id": "optimize",
            "depends_on": [],
            "operation": "optimize",
            "engine": profile["engine"],
            "recipe_sha256": profile["recipe_sha256"],
        },
        {
            "id": "rotor",
            "depends_on": ["optimize"] + (["harmonic"] if needs_hessian else []),
            "operation": "equilibrium_constants",
        },
    ]
    if needs_hessian:
        tasks.append(
            {
                "id": "harmonic",
                "depends_on": ["optimize"],
                "operation": "projected_hessian_analysis",
            }
        )
    if "rigid_rotor_catalog" in request.products:
        tasks.append(
            {
                "id": "rigid_catalog",
                "depends_on": ["rotor", "harmonic"],
                "operation": "exact_rigid_rotor_teaching_catalog",
            }
        )
    chain = [
        "anharmonic_force_field",
        "resonance_analysis",
        "vpt2",
        "ground_state_constants",
        "identification_catalog",
    ]
    advanced = set(request.products) & _ADVANCED
    if advanced:
        last = max(chain.index(product) for product in advanced)
        for index, product in enumerate(chain[: last + 1]):
            dependencies = ["harmonic"] if index == 0 else [chain[index - 1]]
            if product == "ground_state_constants":
                dependencies += ["rotor"]
                if profile.get("rovibrational"):
                    dependencies = ["vibration_rotation_corrections", "rotor"]
            tasks.append(
                {
                    "id": product,
                    "depends_on": dependencies,
                    "operation": product,
                    "qualification": (
                        "explicit_experimental_validation_only"
                        if profile.get("anharmonic")
                        and product
                        in {
                            "anharmonic_force_field",
                            "resonance_analysis",
                            "vpt2",
                            "ground_state_constants",
                        }
                        else "blocked_pending_independent_scientific_validation"
                    ),
                }
            )
        if profile.get("rovibrational"):
            for name, parents in (
                ("rovibrational_precursors", ["harmonic", "rotor"]),
                (
                    "vibration_rotation_corrections",
                    ["rovibrational_precursors", "anharmonic_force_field"],
                ),
                ("semirigid_vpt2", ["vibration_rotation_corrections"]),
                ("centrifugal_distortion", ["vibration_rotation_corrections"]),
            ):
                tasks.append(
                    {
                        "id": name,
                        "depends_on": parents,
                        "operation": name,
                        "conditional": True,
                        "qualification": "explicit_experimental_validation_only",
                        "applicability": "bounded nonresonant Watson model",
                        "native_calls": 0,
                    }
                )
    plan = {
        "schema_version": "cochem.torq.plan/1",
        "capability_purpose": "controlled_validation",
        "declared_capability_records": [
            record.model_dump(mode="json")
            for record in profile_capabilities(profile["id"])
        ],
        "serialization_profile": request.serialization_profile,
        "request": request.model_dump(mode="json"),
        "request_sha256": digest(request.model_dump(mode="json")),
        "scientific_cache_key": scientific_cache_key(request, profile),
        "tasks": tasks,
        "resources": request.resources.model_dump(),
        "scientific_recipe": profile,
        "requested_products": request.products,
        "recovery": "same_model_only",
        "constraints": "none; fully relaxed isolated-molecule optimization",
        "execution_environment": "isolated_linux_cpu_worker",
    }
    return {**plan, "plan_sha256": digest(plan)}


def _stage(
    status: Literal["available", "blocked", "unavailable", "failed"],
    observable: str,
    value: Any = None,
    reason: str | None = None,
    parents: Sequence[str] = (),
    flags: Sequence[str] = (),
    absence_kind: Literal[
        "not_requested", "unsupported", "not_applicable", "failed", "not_computed"
    ]
    | None = None,
) -> dict[str, Any]:
    resolved_absence: (
        Literal[
            "not_requested", "unsupported", "not_applicable", "failed", "not_computed"
        ]
        | None
    )
    if status == "available":
        resolved_absence = None
    elif absence_kind is not None:
        resolved_absence = absence_kind
    elif status == "failed":
        resolved_absence = "failed"
    else:
        resolved_absence = "not_computed"
    return StageResult(
        status=status,
        observable=observable,
        value=_plain(value),
        reason=reason,
        absence_kind=resolved_absence,
        parents=list(parents),
        quality_flags=list(flags),
    ).model_dump(mode="json")


def _empty_result(
    request: CalculationRequest, profile: dict[str, Any], reason: str
) -> dict[str, Any]:
    from .units import constants_provenance

    return {
        "schema_version": "cochem.torq.result/1",
        "request_id": str(request.request_id),
        "request_sha256": digest(request.model_dump(mode="json")),
        "recipe_sha256": profile["recipe_sha256"],
        "recipe": profile,
        "status": "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity(),
        "constants": constants_provenance(),
        "plan": create_plan(request, profile),
        "molecule": request.molecule.model_dump(mode="json"),
        "scientific_goal": request.scientific_goal.model_dump(mode="json")
        if request.scientific_goal
        else None,
        "stages": {
            name: _stage("blocked", name, reason=reason)
            for name in STAGES
            + (
                (
                    "rovibrational_precursors",
                    "vibration_rotation_corrections",
                    "semirigid_vpt2",
                    "centrifugal_distortion",
                )
                if profile.get("rovibrational")
                else ()
            )
        },
        "errors": [],
        "experimental_accuracy_established": False,
        "identification_ready": False,
    }


def _checkpoint(directory: Path, result: dict[str, Any]) -> None:
    temporary = directory / "result.json.checkpoint"
    with temporary.open("wb") as stream:
        stream.write(canonical_json(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "result.json")


def _masses(request: CalculationRequest) -> list[float]:
    from Libraries.cochem_isotopes import isotope_mass

    isotopes = request.molecule.isotopes or [None] * len(request.molecule.symbols)
    return [
        isotope_mass(f"{mass}{symbol}" if mass is not None else symbol)
        for symbol, mass in zip(request.molecule.symbols, isotopes)
    ]


def _isotope_provenance(request: CalculationRequest) -> list[dict[str, Any]]:
    from Libraries.cochem_isotopes import isotope_record

    return [
        isotope_record(f"{mass}{symbol}" if mass is not None else symbol)
        for symbol, mass in zip(
            request.molecule.symbols,
            request.molecule.isotopes or [None] * len(request.molecule.symbols),
        )
    ]


def _qcschema(
    request: CalculationRequest,
    profile: dict[str, Any],
    native: dict[str, Any],
    directory: Path,
    *,
    optimization: dict[str, Any],
) -> None:
    """Validate the actual single-point derivative at the optimized coordinates."""
    from qcelemental.models import AtomicResult

    driver = (
        "hessian" if native.get("hessian_hartree_bohr2") is not None else "gradient"
    )
    data = (
        native["hessian_hartree_bohr2"]
        if driver == "hessian"
        else native["gradient_hartree_bohr"]
    )
    molecule = {
        "symbols": request.molecule.symbols,
        "geometry": np.asarray(native["geometry_bohr"]).ravel().tolist(),
        "molecular_charge": request.molecule.charge,
        "molecular_multiplicity": request.molecule.multiplicity,
        "masses": _masses(request),
        "fix_com": True,
        "fix_orientation": True,
    }
    atomic = AtomicResult(
        molecule=molecule,
        driver=driver,
        model={"method": profile["method"], "basis": profile["basis"]},
        keywords=native["settings"],
        properties={"return_energy": native["energy_hartree"]},
        return_result=data,
        provenance={
            "creator": "PySCF",
            "version": native["engine_version"],
            "routine": "TORQ PySCFBackend native single-point at independently "
            "optimized geometry",
        },
        success=True,
        extras={
            "torq_recipe_sha256": profile["recipe_sha256"],
            "optimization_is_separate_provenance": optimization,
            "dispersion": native.get("dispersion"),
        },
    )
    (directory / "atomic-result.json").write_text(
        atomic.json() + "\n", encoding="utf-8"
    )


def worker_execute(request: CalculationRequest, directory: Path) -> dict[str, Any]:
    if "pes_scan" in request.products:
        raise PrerequisiteError(
            "PES sampling requires the approved scan executor; "
            "the spectroscopy worker cannot execute it."
        )
    validated = validate_request(request, execution="local_validation")
    profile = validated["recipe"]
    result = _empty_result(
        request, profile, "An earlier required stage has not completed."
    )
    _checkpoint(directory, result)
    if not validated["executable"]:
        result["errors"] = validated["blocking_reasons"]
        return result
    if "constrained_geometry" in request.products:
        from .constrained_service import execute_constrained_worker

        return execute_constrained_worker(request, directory)
    if profile.get("reference") == "unrestricted":
        from .open_shell_service import execute_open_shell_worker

        return execute_open_shell_worker(request, directory)
    from .engines.pyscf_backend import PySCFBackend

    probe = PySCFBackend.probe()
    if (
        not probe["available"]
        or probe.get("version") != profile["engine_version"]
        or probe.get("geometric_version") != profile["optimizer_version"]
    ):
        result["errors"].append(
            f"Recipe requires PySCF {profile['engine_version']}; actual probe: {probe}"
        )
        return result
    if (
        profile.get("dispersion") == "d4"
        and probe.get("dftd4_version") != profile["dispersion_version"]
    ):
        result["errors"].append(
            "This recipe requires its exact DFTD4 version; no bare-DFT "
            "substitution is allowed."
        )
        return result
    properties = ["energy", "gradient"] + (
        ["dipole"] if profile["method"] != "mp2" else []
    )
    want_hessian = any(task["id"] == "harmonic" for task in validated["plan"]["tasks"])
    backend_request: dict[str, Any] = {
        "molecule": request.molecule.model_dump(mode="json"),
        "method": {
            key: profile[key]
            for key in ("basis", "reference", "frozen_core", "dispersion")
        },
        "properties": properties,
        "settings": {
            **profile["numerical"],
            "threads": request.resources.cores,
            "memory_mb": request.resources.memory_mb,
        },
    }
    backend_request["method"]["name"] = profile["method"]
    native = PySCFBackend().optimize(backend_request, directory / "engine")
    result["native_result"] = native
    result["errors"].extend(native.get("errors", []))
    stages = result["stages"]
    if (
        native.get("energy_hartree") is None
        or not native.get("scf", {}).get("converged")
        or native.get("stability", {}).get("status") != "stable"
    ):
        stages["electronic_structure"] = _stage(
            "failed",
            "electronic_energy",
            reason="SCF energy and stable restricted reference were not both "
            "established.",
        )
        return result
    stages["electronic_structure"] = _stage(
        "available",
        "electronic_energy",
        {
            "energy_hartree": native["energy_hartree"],
            "engine": native["engine"],
            "engine_version": native["engine_version"],
            "method": native["method"],
            "scf": native["scf"],
            "stability": native["stability"],
            "native_manifest_sha256": native["manifest_sha256"],
        },
        flags=native["quality_flags"],
    )
    result["status"] = "partial"
    _checkpoint(directory, result)
    optimization = native.get("optimization", {})
    if not optimization.get("converged") or not optimization.get(
        "final_gradient_verified"
    ):
        stages["equilibrium_geometry"] = _stage(
            "failed",
            "optimized_geometry",
            reason="Optimizer convergence and independent final gradient checks did "
            "not both pass.",
            parents=["electronic_structure"],
        )
        return result
    stages["equilibrium_geometry"] = _stage(
        "available",
        "stationary_geometry",
        {
            "symbols": request.molecule.symbols,
            "geometry_bohr": native["geometry_bohr"],
            "atom_ids": request.molecule.atom_ids,
            "optimization": optimization,
            "stationary_character": "unclassified_until_harmonic_analysis",
        },
        parents=["electronic_structure"],
        flags=["geometry_accuracy_uncalibrated", "minimum_not_yet_characterized"],
    )
    _checkpoint(directory, result)
    from .spectroscopy.harmonic import analyze_hessian, equilibrium_rotor

    masses = _masses(request)
    result["resolved_isotopes"] = _isotope_provenance(request)
    rotor = equilibrium_rotor(native["geometry_bohr"], masses)
    stages["equilibrium_constants"] = _stage(
        "available",
        "rotational_constants_at_stationary_geometry",
        rotor,
        parents=["equilibrium_geometry"],
        flags=[
            "vibrational_correction_not_applied",
            "accuracy_uncalibrated",
            "minimum_not_yet_characterized",
        ],
    )
    stages["equilibrium_constants"]["value"]["observable"] = (
        "rotational_constants_at_stationary_geometry"
    )
    stages["stationary_rotational_constants"] = dict(stages["equilibrium_constants"])
    stages["equilibrium_constants"] = _stage(
        "blocked",
        "Be",
        reason="Equilibrium constants require an independently verified minimum.",
        parents=["equilibrium_geometry", "harmonic_analysis"],
    )
    _checkpoint(directory, result)
    harmonic = None
    derivative_native = native
    if want_hessian:
        derivative_request = {
            **backend_request,
            "molecule": {
                **backend_request["molecule"],
                "geometry_bohr": native["geometry_bohr"],
            },
            "properties": ["energy", "gradient", "hessian"],
        }
        derivative_native = PySCFBackend().evaluate(
            derivative_request, directory / "harmonic-engine"
        )
        result["harmonic_native_result"] = derivative_native
        result["errors"].extend(derivative_native.get("errors", []))
        _checkpoint(directory, result)
    if (
        derivative_native.get("hessian_hartree_bohr2") is not None
        and derivative_native.get("status") == "complete"
    ):
        try:
            if (
                derivative_native["geometry_bohr"] != native["geometry_bohr"]
                or abs(derivative_native["energy_hartree"] - native["energy_hartree"])
                > 1e-8
            ):
                raise ValueError(
                    "The derivative calculation differs from its exact geometry or "
                    "same-model energy parent."
                )
            harmonic = analyze_hessian(
                native["geometry_bohr"],
                masses,
                derivative_native["hessian_hartree_bohr2"],
                symmetry_tolerance=profile.get(
                    "harmonic_symmetry_relative_tolerance", 1e-8
                ),
            )
            stages["harmonic_analysis"] = _stage(
                "available",
                "harmonic_frequencies",
                harmonic,
                parents=["equilibrium_geometry"],
                flags=["harmonic_approximation", "accuracy_uncalibrated"],
            )
            stages["equilibrium_geometry"]["value"]["stationary_character"] = (
                harmonic.stationary_character
            )
            invariance_ok = (
                harmonic.external_residual_relative
                <= profile["harmonic_external_residual_relative_tolerance"]
            )
            if not invariance_ok:
                stages["harmonic_analysis"]["quality_flags"].append(
                    "hessian_external_invariance_failed"
                )
                result["errors"].append(
                    {
                        "code": "HESSIAN_EXTERNAL_INVARIANCE_FAILED",
                        "message": "The actual Hessian failed the recipe's explicit "
                        "translation/rotation residual gate.",
                    }
                )
            if (
                harmonic.stationary_character == "positive_definite_vibrational_hessian"
                and invariance_ok
            ):
                stages["equilibrium_geometry"]["observable"] = "re"
                stages["equilibrium_geometry"]["quality_flags"].remove(
                    "minimum_not_yet_characterized"
                )
                stages["equilibrium_constants"] = _stage(
                    "available",
                    "Be",
                    rotor,
                    parents=["equilibrium_geometry", "harmonic_analysis"],
                    flags=[
                        "vibrational_correction_not_applied",
                        "accuracy_uncalibrated",
                    ],
                )
            else:
                stages["equilibrium_geometry"]["quality_flags"].append(
                    "nonminimum_or_unresolved_stationary_point"
                )
                stages["harmonic_analysis"]["quality_flags"].append(
                    "nonminimum_or_unresolved_stationary_point"
                )
                result["errors"].append(
                    {
                        "code": "MINIMUM_NOT_ESTABLISHED",
                        "message": "Signed harmonic modes are retained; "
                        "equilibrium/minimum-dependent products cannot be "
                        "accepted.",
                    }
                )
        except ValueError as exc:
            stages["harmonic_analysis"] = _stage(
                "failed",
                "harmonic_frequencies",
                reason=str(exc),
                parents=["equilibrium_geometry"],
            )
    elif want_hessian:
        stages["harmonic_analysis"] = _stage(
            "unavailable",
            "harmonic_frequencies",
            reason="The real engine did not produce an accepted Hessian.",
            parents=["equilibrium_geometry"],
        )
    else:
        stages["harmonic_analysis"] = _stage(
            "unavailable",
            "harmonic_frequencies",
            reason="No Hessian was requested in this immutable plan.",
            parents=["equilibrium_geometry"],
            absence_kind="not_requested",
        )
    _checkpoint(directory, result)
    for name in (
        "anharmonic_force_field",
        "resonance_analysis",
        "vpt2",
        "ground_state_constants",
        "identification_catalog",
    ):
        stages[name] = _stage(
            "blocked",
            name,
            reason="Independent molecular "
            "anharmonic/rotation–vibration/distortion/intensity benchmarks "
            "and method qualification are required. No harmonic substitute is "
            "supplied.",
            absence_kind="unsupported" if name in request.products else "not_requested",
            parents=["harmonic_analysis"]
            if name == "anharmonic_force_field"
            else [STAGES[STAGES.index(name) - 1]],
        )
    if profile.get("anharmonic") and any(
        task["id"] == "anharmonic_force_field" for task in result["plan"]["tasks"]
    ):
        if (
            harmonic is not None
            and harmonic.stationary_character == "positive_definite_vibrational_hessian"
            and stages["equilibrium_constants"]["status"] == "available"
        ):
            from .research_pipeline import execute_anharmonic_validation

            try:
                research = execute_anharmonic_validation(
                    request,
                    harmonic,
                    backend_request,
                    directory / "anharmonic-research",
                )
                result["anharmonic_validation"] = research
                result["errors"].extend(research["errors"])
                for name, source_name, observable, parents in (
                    (
                        "anharmonic_force_field",
                        "anharmonic_force_field",
                        "anharmonic_force_field",
                        ["harmonic_analysis"],
                    ),
                    (
                        "resonance_analysis",
                        "resonance_analysis",
                        "resonance_analysis",
                        ["anharmonic_force_field"],
                    ),
                    (
                        "vpt2",
                        "vibrational_vpt2",
                        "vibrational_only_vpt2",
                        ["anharmonic_force_field", "resonance_analysis"],
                    ),
                    (
                        "rovibrational_precursors",
                        "rovibrational_precursors",
                        "rovibrational_precursors",
                        ["harmonic_analysis", "equilibrium_constants"],
                    ),
                    (
                        "vibration_rotation_corrections",
                        "vibration_rotation_corrections",
                        "vibration_rotation_corrections",
                        ["rovibrational_precursors", "anharmonic_force_field"],
                    ),
                    (
                        "semirigid_vpt2",
                        "semirigid_vpt2",
                        "semirigid_vpt2",
                        ["vibration_rotation_corrections", "resonance_analysis"],
                    ),
                    (
                        "centrifugal_distortion",
                        "centrifugal_distortion",
                        "unreduced_harmonic_distortion",
                        ["vibration_rotation_corrections"],
                    ),
                    (
                        "ground_state_constants",
                        "ground_state_constants",
                        "nonresonant_Watson_model_B0",
                        ["vibration_rotation_corrections", "centrifugal_distortion"],
                    ),
                ):
                    observation = research["stages"][source_name]
                    stages[name] = _stage(
                        observation["status"],
                        observable,
                        value=observation.get("value"),
                        reason=observation.get("reason"),
                        parents=parents,
                        flags=[
                            "experimental_unqualified",
                            "full_resonant_GVPT2_unavailable",
                            "identification_accuracy_not_established",
                            *observation.get("quality_flags", []),
                        ],
                    )
            except (ValueError, RuntimeError) as exc:
                if stages["anharmonic_force_field"]["status"] != "available":
                    stages["anharmonic_force_field"] = _stage(
                        "unavailable",
                        "anharmonic_force_field",
                        reason=str(exc),
                        parents=["harmonic_analysis"],
                    )
                result["errors"].append(
                    {
                        "code": "ANHARMONIC_STAGE_MAPPING_FAILED",
                        "message": str(exc),
                    }
                )
        _checkpoint(directory, result)
    if "rigid_rotor_catalog" in request.products:
        stages["rigid_rotor_catalog"] = _stage(
            "blocked",
            "rigid_rotor_transitions",
            reason="Rigid-rotor teaching catalog requires a verified minimum and the "
            "implemented exact-rotor solver.",
            parents=["equilibrium_constants", "harmonic_analysis"],
        )
        if (
            harmonic is not None
            and harmonic.stationary_character == "positive_definite_vibrational_hessian"
            and native.get("dipole_debye") is not None
            and stages["equilibrium_constants"]["status"] == "available"
        ):
            # For ions the rotational dipole is defined at the center of mass.
            from .spectroscopy import rigid_rotor_catalog
            from .units import (
                BOHR_METRE,
                DEBYE_COULOMB_METRE,
                ELEMENTARY_CHARGE_COULOMB,
            )

            center_bohr = np.average(native["geometry_bohr"], axis=0, weights=masses)
            mu_com = (
                np.asarray(native["dipole_debye"])
                - request.molecule.charge
                * center_bohr
                * ELEMENTARY_CHARGE_COULOMB
                * BOHR_METRE
                / DEBYE_COULOMB_METRE
            )
            dipole = mu_com @ rotor.principal_axes_columns
            try:
                catalog = rigid_rotor_catalog(
                    rotor.constants_mhz,
                    dipole,
                    temperature_kelvin=request.catalog.temperature_kelvin,
                    J_max=request.catalog.max_j,
                    constant_observable="Be",
                )
                from .spectroscopy.results import make_scientific_context

                catalog_value = _plain(catalog)
                catalog_value["dipole_origin"] = "center_of_mass"
                catalog_value["scientific_context"] = make_scientific_context(
                    molecule=request.molecule.model_dump(mode="json"),
                    geometry_bohr=native["geometry_bohr"],
                    isotope_provenance=result["resolved_isotopes"],
                    recipe_sha256=profile["recipe_sha256"],
                    protocol_sha256=result["plan"]["plan_sha256"],
                    evidence_class="engine_calculation",
                    parent_artifact_sha256=[
                        native["manifest_sha256"],
                        derivative_native["manifest_sha256"],
                    ],
                    harmonic=harmonic,
                    principal_axes_columns=rotor.principal_axes_columns,
                )
                stages["rigid_rotor_catalog"] = _stage(
                    "available",
                    "rigid_rotor_transitions",
                    catalog_value,
                    parents=["equilibrium_constants", "harmonic_analysis"],
                    flags=[
                        "rigid_rotor_approximation",
                        "Be_not_B0",
                        "identification_accuracy_not_established",
                    ],
                )
                if not catalog.partition_converged_at_requested_tolerance:
                    stages["rigid_rotor_catalog"]["quality_flags"].append(
                        "finite_J_partition_not_converged"
                    )
                    result["errors"].append(
                        {
                            "code": "CATALOG_PARTITION_NOT_CONVERGED",
                            "message": "Computed finite-J lines are retained; "
                            "increase max_j within the explicit budget to "
                            "qualify thermal intensities.",
                        }
                    )
            except ValueError as exc:
                stages["rigid_rotor_catalog"] = _stage(
                    "unavailable",
                    "rigid_rotor_transitions",
                    reason=str(exc),
                    parents=["equilibrium_constants", "harmonic_analysis"],
                )
    _checkpoint(directory, result)
    try:
        _qcschema(
            request,
            profile,
            derivative_native
            if derivative_native.get("status") == "complete"
            else native,
            directory,
            optimization=optimization,
        )
    except (ValueError, ImportError) as exc:
        result["errors"].append(
            {"code": "QCSCHEMA_EXPORT_UNAVAILABLE", "message": str(exc)}
        )
    if request.scientific_goal is not None:
        result["goal_evaluation"] = {
            "status": "unavailable",
            "accuracy_status": "unqualified",
            "reason": "The goal declaration is retained; independent "
            "accuracy/error-budget validation has not been executed.",
        }
        result["errors"].append(
            {
                "code": "SCIENTIFIC_GOAL_NOT_ESTABLISHED",
                "message": result["goal_evaluation"]["reason"],
            }
        )
    if not result["errors"] and all(
        stages[PRODUCT_TO_STAGE[product]]["status"] == "available"
        for product in request.products
    ):
        result["status"] = "complete"
    return result


def execute_request(
    request: dict[str, Any] | CalculationRequest,
    output_directory: str | Path,
    *,
    approved_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Owned subprocess, wall timeout, fenced completion, then immutable publication."""
    checked = validate_request(request, execution="local_validation")
    model = CalculationRequest.model_validate(checked["request"])
    if not checked["executable"]:
        raise PrerequisiteError("; ".join(checked["blocking_reasons"]))
    if "pes_scan" in model.products:
        from .scan import execute_scan_request

        return execute_scan_request(
            model, output_directory, approved_plan=approved_plan
        )
    destination = Path(output_directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Select a new output directory; published runs are immutable."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    from .operations import admit_artifact_work
    from .service import approve_plan, plan_request, validate_approved_plan

    if approved_plan is None:
        # An explicit local execution call authorizes this exact finite request.
        # The observed OS identity is recorded; no remote user identity is inferred.
        approved_plan = approve_plan(
            plan_request(model.model_dump(mode="json"), execution="local_validation"),
            actor=f"local-os:{os.getuid()}@{socket.gethostname()}",
        )
    approved = validate_approved_plan(approved_plan, request=checked["request"])
    scratch_bytes = approved.approval.max_scratch_bytes
    with admit_artifact_work(destination.parent, incoming_bytes=scratch_bytes):
        return _execute_admitted_request(checked, model, destination, approved)


def _execute_admitted_request(
    checked: dict[str, Any],
    model: CalculationRequest,
    destination: Path,
    approved: ApprovedPlan,
    *,
    host_allocation_directory: str | Path | None = None,
) -> dict[str, Any]:
    import psutil

    from cochem.orchestration.campaign import (
        Allocation,
        CampaignBudget,
        CampaignCoordinator,
        MeasuredUsage,
    )
    from cochem.orchestration.host_allocation import (
        HostAllocationRequest,
        HostAuthorityError,
        ObservedHostUsage,
        bootstrap_host_allocation,
    )

    from .operations import (
        ArtifactBackpressureError,
        artifact_usage,
        enforce_artifact_budget,
    )
    from .scan import _host_provenance, _verify_host_provenance

    scratch_bytes = approved.approval.max_scratch_bytes
    actor = f"local-os:{os.getuid()}@{socket.gethostname()}"
    queue = CampaignCoordinator(destination.parent / ".torq-campaign.sqlite")
    attempt = None
    try:
        receipt = queue.create_campaign(
            approved.model_dump(mode="json"),
            CampaignBudget(
                max_wall_seconds=approved.approval.max_wall_seconds,
                max_cpu_core_seconds=approved.approval.max_cpu_core_seconds,
                max_tasks=1,
                max_concurrent_workers=1,
                max_cpu_cores=model.resources.cores,
                max_memory_mb=approved.approval.max_memory_mb,
                max_scratch_mb=(scratch_bytes + 1024**2 - 1) // 1024**2,
                expires_at_unix=approved.approval.expires_at.timestamp(),
                allowed_engines=[checked["recipe"]["engine"]],
                allowed_recipes=[model.recipe],
                permitted_retries=approved.approval.permitted_retries,
            ),
            actor=actor,
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.worker-", dir=destination.parent
            )
        )
        (staging / "request.json").write_bytes(
            canonical_json(checked["request"]) + b"\n"
        )
        task = queue.register_task(
            receipt["campaign_id"],
            {
                "request_id": str(model.request_id),
                "request_sha256": checked["request_sha256"],
                "output": str(destination),
                "staging": str(staging),
            },
            engine=checked["recipe"]["engine"],
            recipe=model.recipe,
            plan_sha256=receipt["plan_sha256"],
            authority=receipt["authority"],
            actor=actor,
        )
        task_id = task["task_id"]
        attempt = queue.new_attempt(
            task_id, authority=receipt["authority"], actor=actor
        )
        attempt = queue.transition(
            attempt["id"],
            attempt["revision"],
            "validated",
            authority=receipt["authority"],
            actor=actor,
            reason="Validated exact request, profile, reviewed plan and finite budget.",
        )
        attempt = queue.reserve(
            attempt["id"],
            attempt["revision"],
            Allocation(
                wall_seconds=model.resources.wall_seconds,
                cores=model.resources.cores,
                memory_mb=model.resources.memory_mb,
                scratch_mb=(scratch_bytes + 1024**2 - 1) // 1024**2,
            ),
            authority=receipt["authority"],
            actor=actor,
        )
    except BaseException:
        try:
            if attempt is not None:
                current = queue.attempt(attempt["id"])
                queue.cancel(
                    attempt["id"],
                    current["revision"],
                    authority=receipt["authority"],
                    actor=actor,
                    reason="Setup failed before any worker was launched.",
                )
                if any(
                    record["attempt_id"] == attempt["id"]
                    for record in queue.accounting(receipt["campaign_id"])
                ):
                    queue.reconcile(
                        attempt["id"],
                        MeasuredUsage(
                            wall_seconds=0.0,
                            measurement_source=(
                                "not dispatched; setup failed before Popen"
                            ),
                        ),
                        authority=receipt["authority"],
                        actor=actor,
                        reason=(
                            "No child process was launched; "
                            "released undispatched resources."
                        ),
                    )
        finally:
            queue.close()
        raise
    allowed_environment = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "COCHEM_SOURCE_COMMIT",
        "COCHEM_IMAGE_ID",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in allowed_environment
    }
    environment.update(
        {
            "OMP_NUM_THREADS": str(model.resources.cores),
            "OPENBLAS_NUM_THREADS": str(model.resources.cores),
            "MKL_NUM_THREADS": str(model.resources.cores),
        }
    )
    deadline = time.monotonic() + model.resources.wall_seconds
    started = time.monotonic()
    timed_out = False
    budget_failure = None
    sampled_peak_memory_mb = None
    measured_peak_scratch_bytes = 0
    process = None
    published = False
    launched = False
    host_ledger = None
    acquired_host = None
    host_receipt = None
    host_provenance = None
    host_scratch_observed = False
    try:
        host_ledger = bootstrap_host_allocation(
            actor=f"local-os-user:{os.getuid()}",
            directory=host_allocation_directory,
        )
        acquired_host = host_ledger.acquire(
            campaign_id=receipt["campaign_id"],
            attempt_id=attempt["id"],
            workspace=staging,
            request=HostAllocationRequest(
                cores=model.resources.cores,
                memory_mb=model.resources.memory_mb,
                scratch_mb=(scratch_bytes + 1024**2 - 1) // 1024**2,
                child_inventory_scope=(
                    "constrained-evaluation-v1"
                    if "constrained_geometry" in model.products
                    else "single_owned_worker"
                ),
            ),
            lease_seconds=60.0,
        )
        host_receipt = acquired_host.receipt
        with (staging / "worker.log").open("wb") as log:
            host_receipt = host_ledger.begin_dispatch(
                host_receipt["allocation_id"],
                host_receipt["revision"],
                acquired_host.lease_token,
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "cochem_torq.cli",
                    "_worker",
                    "--request",
                    str(staging / "request.json"),
                    "--output-dir",
                    str(staging),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            launched = True
            host_receipt = host_ledger.bind_dispatch(
                host_receipt["allocation_id"],
                host_receipt["revision"],
                acquired_host.lease_token,
                pid=process.pid,
            )
            attempt = queue.record_dispatch(
                attempt["id"],
                attempt["revision"],
                pid=process.pid,
                authority=receipt["authority"],
                actor=actor,
            )
            lease = queue.lease(
                attempt["id"],
                attempt["revision"],
                pid=process.pid,
                authority=receipt["authority"],
                actor=actor,
            )
            attempt = lease
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    _stop_owned_process(process)
                    break
                try:
                    usage = enforce_artifact_budget(
                        staging, baseline_bytes=0, max_growth_bytes=scratch_bytes
                    )
                    measured_peak_scratch_bytes = max(
                        measured_peak_scratch_bytes, usage["owned_bytes"]
                    )
                    host_scratch_observed = True
                    owned = psutil.Process(process.pid)
                    actual_rss_samples = []
                    for member in [owned, *owned.children(recursive=True)]:
                        try:
                            actual_rss_samples.append(member.memory_info().rss)
                        except psutil.NoSuchProcess:
                            pass
                    if actual_rss_samples:
                        rss = sum(actual_rss_samples)
                        sampled_peak_memory_mb = max(
                            sampled_peak_memory_mb or 0.0, rss / 1024**2
                        )
                        if rss > approved.approval.max_memory_mb * 1024**2:
                            raise ArtifactBackpressureError(
                                "Owned worker process tree exceeded its approved "
                                "resident-memory ceiling."
                            )
                except psutil.NoSuchProcess:
                    pass
                except ArtifactBackpressureError as exc:
                    budget_failure = str(exc)
                    _stop_owned_process(process)
                    break
                try:
                    queue.heartbeat(
                        attempt["id"],
                        lease_token=lease["lease_token"],
                        lease_generation=lease["lease_generation"],
                    )
                except RuntimeError:
                    _stop_owned_process(process)
                    raise RuntimeError(
                        "Coordinator lease lost; stale worker artifacts will not be "
                        "published."
                    )
                try:
                    host_receipt = host_ledger.heartbeat(
                        host_receipt["allocation_id"],
                        host_receipt["revision"],
                        acquired_host.lease_token,
                    )
                except HostAuthorityError:
                    host_receipt = host_ledger.observe_worker_exit(
                        host_receipt["allocation_id"],
                        host_receipt["revision"],
                        acquired_host.lease_token,
                    )
                    # The ledger proves actual PID/create-time death (including
                    # a zombie) and current authority; poll may still race exit.
                    process.wait()
                    break
                time.sleep(0.2)
        process.wait()
        host_receipt = host_ledger.reconcile(
            host_receipt["allocation_id"],
            host_receipt["revision"],
            reason="Actual spectroscopy child wait completed before shared "
            "host resource reconciliation.",
            usage=ObservedHostUsage(
                peak_memory_mb=sampled_peak_memory_mb,
                peak_scratch_mb=measured_peak_scratch_bytes / 1024**2
                if host_scratch_observed
                else None,
                measurement_source="Actual sampled owned-process RSS and owned "
                "staging byte observations; missing peaks unavailable.",
            ),
        )
        host_provenance = _host_provenance(host_ledger, host_receipt)
        if (
            timed_out
            or budget_failure
            or not (staging / "result.json").is_file()
            or process.returncode not in (0, 4, 5)
        ):
            reason = (
                budget_failure
                if budget_failure
                else "Owned calculation process exceeded its wall-time budget."
                if timed_out or budget_failure
                else "Worker failed before producing a typed result; authentic worker "
                "log retained."
            )
            result = (
                _read_worker_result(staging / "result.json")
                if (staging / "result.json").is_file()
                else _empty_result(model, checked["recipe"], reason)
            )
            result["status"] = (
                "partial"
                if any(
                    stage["status"] == "available"
                    for stage in result["stages"].values()
                )
                else "failed"
            )
            result["errors"].append(
                {
                    "code": "RESOURCE_BUDGET_EXCEEDED"
                    if budget_failure
                    else "WALL_TIMEOUT"
                    if timed_out
                    else "WORKER_FAILED",
                    "message": reason,
                }
            )
            (staging / "result.json").write_bytes(canonical_json(result) + b"\n")
        result = _read_worker_result(staging / "result.json")
        wall_seconds = time.monotonic() - started
        measured_peak_scratch_bytes = max(
            measured_peak_scratch_bytes, artifact_usage(staging)["owned_bytes"]
        )
        result["campaign"] = {
            "campaign_id": receipt["campaign_id"],
            "task_id": task_id,
            "attempt_id": attempt["id"],
            "approved_plan_sha256": approved.approval.plan_sha256,
            "approval_sha256": approved.approval_sha256,
            "worker_wall_seconds": wall_seconds,
            "sampled_peak_memory_mb": sampled_peak_memory_mb,
            "memory_measurement": "sampled resident memory; exact peak unavailable"
            if sampled_peak_memory_mb is not None
            else "unavailable; no sample completed",
            "observed_peak_scratch_bytes": measured_peak_scratch_bytes,
            "cpu_core_seconds": None,
            "cpu_accounting": "conservative_reserved_ceiling; "
            "exact CPU measurement unavailable",
            "coordinator": "single_authority_campaign_CAS_and_fenced_publication",
            "host_allocation": host_provenance.model_dump(mode="json"),
        }
        if wall_seconds > approved.approval.max_wall_seconds and not timed_out:
            result["status"] = (
                "partial"
                if any(
                    stage["status"] == "available"
                    for stage in result["stages"].values()
                )
                else "failed"
            )
            result["errors"].append(
                {
                    "code": "RESOURCE_BUDGET_EXCEEDED",
                    "message": (
                        "Observed wall consumption exceeded the approved "
                        "campaign ceiling."
                    ),
                }
            )
        (staging / "result.json").write_bytes(canonical_json(result) + b"\n")
        attempt = queue.worker_transition(
            attempt["id"],
            attempt["revision"],
            "collecting",
            lease_token=lease["lease_token"],
            lease_generation=lease["lease_generation"],
            actor=actor,
            reason=(
                "Owned child stopped and native handles closed; "
                "collecting authentic artifacts."
            ),
        )
        attempt = queue.worker_transition(
            attempt["id"],
            attempt["revision"],
            "validating",
            lease_token=lease["lease_token"],
            lease_generation=lease["lease_generation"],
            actor=actor,
            reason=(
                "Checking typed stages, identities and complete immutable inventory."
            ),
        )
        # Only the coordinator seals after the child has closed all native handles.
        seal_shard(
            staging,
            request_sha256=checked["request_sha256"],
            request_id=str(model.request_id),
            recipe_sha256=checked["recipe"]["recipe_sha256"],
            source_identity=result["source_identity"],
            worker_id=task_id,
        )
        verify_shard(staging, expected_request_sha256=checked["request_sha256"])

        def publish_with_host_authority() -> None:
            if host_ledger is None or host_provenance is None:
                raise ValueError("Publication requires actual shared host authority.")
            _verify_host_provenance(host_ledger, host_provenance)
            publish_shard(staging, destination)

        queue.publish(
            attempt["id"],
            attempt["revision"],
            {
                "status": result["status"],
                "request_sha256": checked["request_sha256"],
                "artifact_manifest_sha256": sha256(
                    (staging / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            MeasuredUsage(
                wall_seconds=wall_seconds,
                peak_memory_mb=None,
                peak_scratch_mb=measured_peak_scratch_bytes / 1024**2,
                measurement_source=(
                    "owned child monotonic wall; observed byte accounting; "
                    "exact CPU/GPU and memory peak unavailable"
                ),
            ),
            publish_with_host_authority,
            state="succeeded" if result["status"] == "complete" else result["status"],
            lease_token=lease["lease_token"],
            lease_generation=lease["lease_generation"],
            actor=actor,
        )
        published = True
        return result
    finally:
        try:
            if process is not None and process.poll() is None:
                _stop_owned_process(process)
            if process is not None:
                process.wait()
            if not published:
                current_attempt = queue.attempt(attempt["id"])
                if current_attempt["state"] not in {
                    "succeeded",
                    "partial",
                    "failed",
                    "cancelled",
                }:
                    queue.cancel(
                        attempt["id"],
                        current_attempt["revision"],
                        authority=receipt["authority"],
                        actor=actor,
                        reason=(
                            "Owned local execution ended without validated "
                            "publication; authentic staging retained."
                        ),
                    )
            if host_ledger is not None and host_receipt is not None:
                if host_receipt["state"] == "running":
                    host_receipt = host_ledger.reconcile(
                        host_receipt["allocation_id"],
                        host_receipt["revision"],
                        reason="Actual owned child wait completed after "
                        "unsuccessful spectroscopy publication.",
                        usage=ObservedHostUsage(
                            peak_memory_mb=sampled_peak_memory_mb,
                            peak_scratch_mb=measured_peak_scratch_bytes / 1024**2
                            if host_scratch_observed
                            else None,
                            measurement_source="Actual sampled RSS/staging bytes; "
                            "missing observations unavailable.",
                        ),
                    )
                elif host_receipt["state"] == "reserved" and acquired_host is not None:
                    host_ledger.release_undispatched(
                        host_receipt["allocation_id"],
                        host_receipt["revision"],
                        acquired_host.lease_token,
                        reason="The live owner ended before dispatch started.",
                    )
                # Unbound dispatching records remain unknown/reserved.
                if host_receipt["state"] == "dispatching":
                    raise HostAuthorityError(
                        "Dispatch identity is unknown; shared resources and "
                        "scientific cost accounting remain held for recovery."
                    )
            if not published:
                # The shared ledger verifies every known nested child before
                # any physical-cost settlement. Outer process death alone is
                # insufficient for a constrained adapter's separate sessions.
                queue.reconcile(
                    attempt["id"],
                    MeasuredUsage(
                        wall_seconds=time.monotonic() - started if launched else 0.0,
                        measurement_source=(
                            "observed local and nested termination; exact CPU/GPU "
                            "and memory peak unavailable"
                        )
                        if launched
                        else "not dispatched; Popen did not launch a worker",
                    ),
                    authority=receipt["authority"],
                    actor=actor,
                    reason=(
                        "Reconciled actual stopped local process after "
                        "unsuccessful publication."
                    ),
                )
        finally:
            queue.close()
            if host_ledger is not None:
                host_ledger.close()


def _read_worker_result(path: Path) -> dict[str, Any]:
    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError("The actual worker result must be a JSON object.")
    return result


def _stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def submit_request(
    request: dict[str, Any] | CalculationRequest,
    execution: str = "github_actions",
    *,
    approved_plan: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    from .service import validate_approved_plan

    if approved_plan is None or not idempotency_key:
        raise ValueError(
            "Submission requires an explicitly approved plan and idempotency key."
        )
    normalized = (
        request.model_dump(mode="json")
        if isinstance(request, CalculationRequest)
        else request
    )
    approved = validate_approved_plan(approved_plan, request=normalized)
    checked = validate_request(request)
    if not checked["executable"]:
        raise PrerequisiteError("; ".join(checked["blocking_reasons"]))
    if execution != "github_actions":
        raise ValueError(
            "Student submissions use GitHub Actions; explicit local "
            "validation uses execute_request."
        )
    from .github import GitHubActions

    return GitHubActions.from_environment().submit(
        checked["request"],
        idempotency_key=idempotency_key,
        approved_plan_sha256=approved.approval.plan_sha256,
        approved_plan=approved.model_dump(mode="json"),
    )


def get_run_status(run_id: str | int) -> dict[str, Any]:
    from .github import GitHubActions

    return GitHubActions.from_environment().status(str(run_id))


def download_run_results(run_id: str | int, destination: str | Path) -> list[Path]:
    from .github import GitHubActions

    return GitHubActions.from_environment().download(str(run_id), Path(destination))
