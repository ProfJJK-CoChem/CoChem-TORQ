"""The student-facing application: explicit plans, real engines and durable evidence."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

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
)
from .domain import SPECTROSCOPY_STAGES as STAGES
from .registry import get_profile

_ADVANCED = {
    "anharmonic_force_field",
    "vpt2",
    "ground_state_constants",
    "identification_catalog",
}


def _plain(value: Any) -> Any:
    if is_dataclass(value):
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

    for module in ("Libraries.cochem_isotopes", "cochem.orchestration.sqlite_queue"):
        specification = importlib.util.find_spec(module)
        if specification is not None and specification.origin:
            records.append(
                {
                    "path": module,
                    "sha256": sha256(
                        Path(specification.origin).read_bytes()
                    ).hexdigest(),
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
    _masses(
        model
    )  # Fail before expensive work if a requested isotope lacks real mass data.
    reasons = []
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
        if model.molecule.multiplicity != profile["multiplicity"]:
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
        "recipe": profile,
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
            tasks.append(
                {
                    "id": product,
                    "depends_on": dependencies,
                    "operation": product,
                    "qualification": (
                        "explicit_experimental_validation_only"
                        if profile.get("anharmonic")
                        and product in {"anharmonic_force_field", "resonance_analysis"}
                        else "blocked_pending_independent_scientific_validation"
                    ),
                }
            )
    plan = {
        "schema_version": "cochem.torq.plan/1",
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
    status: str, observable: str, value=None, reason=None, parents=(), flags=()
) -> dict[str, Any]:
    return StageResult(
        status=status,
        observable=observable,
        value=_plain(value),
        reason=reason,
        parents=list(parents),
        quality_flags=list(flags),
    ).model_dump(mode="json")


def _empty_result(
    request: CalculationRequest, profile: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "schema_version": "cochem.torq.result/1",
        "request_id": str(request.request_id),
        "request_sha256": digest(request.model_dump(mode="json")),
        "recipe_sha256": profile["recipe_sha256"],
        "recipe": profile,
        "status": "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity(),
        "plan": create_plan(request, profile),
        "molecule": request.molecule.model_dump(mode="json"),
        "stages": {name: _stage("blocked", name, reason=reason) for name in STAGES},
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
):
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
    validated = validate_request(request, execution="local_validation")
    profile = validated["recipe"]
    result = _empty_result(
        request, profile, "An earlier required stage has not completed."
    )
    _checkpoint(directory, result)
    if not validated["executable"]:
        result["errors"] = validated["blocking_reasons"]
        return result
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
    backend_request = {
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
                for name in ("anharmonic_force_field", "resonance_analysis"):
                    observation = research["stages"][name]
                    stages[name] = _stage(
                        observation["status"],
                        name,
                        value=observation.get("value"),
                        reason=observation.get("reason"),
                        parents=["harmonic_analysis"]
                        if name == "anharmonic_force_field"
                        else ["anharmonic_force_field"],
                        flags=[
                            "experimental_unqualified",
                            "rotation_vibration_not_implemented",
                            *observation.get("quality_flags", []),
                        ],
                    )
            except (ValueError, RuntimeError) as exc:
                stages["anharmonic_force_field"] = _stage(
                    "unavailable",
                    "anharmonic_force_field",
                    reason=str(exc),
                    parents=["harmonic_analysis"],
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
            from scipy.constants import elementary_charge, physical_constants

            from .spectroscopy import rigid_rotor_catalog

            bohr_m = physical_constants["Bohr radius"][0]
            center_bohr = np.average(native["geometry_bohr"], axis=0, weights=masses)
            mu_com = (
                np.asarray(native["dipole_debye"])
                - request.molecule.charge
                * center_bohr
                * elementary_charge
                * bohr_m
                / 3.33564e-30
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
                stages["rigid_rotor_catalog"] = _stage(
                    "available",
                    "rigid_rotor_transitions",
                    catalog,
                    parents=["equilibrium_constants", "harmonic_analysis"],
                    flags=[
                        "rigid_rotor_approximation",
                        "Be_not_B0",
                        "identification_accuracy_not_established",
                    ],
                )
                stages["rigid_rotor_catalog"]["value"]["dipole_origin"] = (
                    "center_of_mass"
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
    if not result["errors"] and all(
        stages[PRODUCT_TO_STAGE[product]]["status"] == "available"
        for product in request.products
    ):
        result["status"] = "complete"
    return result


def execute_request(
    request: dict[str, Any] | CalculationRequest, output_directory: str | Path
) -> dict[str, Any]:
    """Owned subprocess, wall timeout, fenced completion, then immutable publication."""
    checked = validate_request(request, execution="local_validation")
    model = CalculationRequest.model_validate(checked["request"])
    if not checked["executable"]:
        raise PrerequisiteError("; ".join(checked["blocking_reasons"]))
    destination = Path(output_directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Select a new output directory; published runs are immutable."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.worker-", dir=destination.parent)
    )
    (staging / "request.json").write_bytes(canonical_json(checked["request"]) + b"\n")
    from cochem.orchestration.sqlite_queue import SQLiteTaskQueue

    coordinator = destination.parent / ".torq-coordinator.sqlite"
    queue = SQLiteTaskQueue(coordinator)
    task_id = queue.enqueue_task(
        "torq_calculation",
        {
            "request_id": str(model.request_id),
            "request_sha256": checked["request_sha256"],
            "output": str(destination),
            "staging": str(staging),
        },
        max_retries=1,
    )
    lease = queue.lease_task(os.getpid(), socket.gethostname(), task_id=task_id)
    if lease is None or lease.task_id != task_id:
        queue.close()
        raise RuntimeError(
            "Coordinator lease did not match this planned task; use a "
            "separate output root for independent coordinators."
        )
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
    timed_out = False
    process = None
    try:
        with (staging / "worker.log").open("wb") as log:
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
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    _stop_owned_process(process)
                    break
                if not queue.heartbeat(
                    task_id,
                    lease_token=lease.lease_token,
                    lease_generation=lease.lease_generation,
                ):
                    _stop_owned_process(process)
                    raise RuntimeError(
                        "Coordinator lease lost; stale worker artifacts will not be "
                        "published."
                    )
                time.sleep(0.2)
        if (
            timed_out
            or not (staging / "result.json").is_file()
            or process.returncode not in (0, 4, 5)
        ):
            reason = (
                "Owned calculation process exceeded its wall-time budget."
                if timed_out
                else "Worker failed before producing a typed result; authentic worker "
                "log retained."
            )
            result = (
                read_json(staging / "result.json")
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
                    "code": "WALL_TIMEOUT" if timed_out else "WORKER_FAILED",
                    "message": reason,
                }
            )
            (staging / "result.json").write_bytes(canonical_json(result) + b"\n")
        result = read_json(staging / "result.json")
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
        queue.complete_task_with_publication(
            task_id,
            {
                "status": result["status"],
                "request_sha256": checked["request_sha256"],
                "artifact_manifest_sha256": sha256(
                    (staging / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            lambda: publish_shard(staging, destination),
            lease_token=lease.lease_token,
            lease_generation=lease.lease_generation,
        )
        return result
    finally:
        if process is not None and process.poll() is None:
            _stop_owned_process(process)
        queue.close()


def _stop_owned_process(process: subprocess.Popen) -> None:
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
    request: dict[str, Any] | CalculationRequest, execution: str = "github_actions"
) -> dict[str, Any]:
    checked = validate_request(request)
    if not checked["executable"]:
        raise PrerequisiteError("; ".join(checked["blocking_reasons"]))
    if execution != "github_actions":
        raise ValueError(
            "Student submissions use GitHub Actions; explicit local "
            "validation uses execute_request."
        )
    from .github import GitHubActions

    return GitHubActions.from_environment().submit(checked["request"])


def get_run_status(run_id: str | int) -> dict[str, Any]:
    from .github import GitHubActions

    return GitHubActions.from_environment().status(str(run_id))


def download_run_results(run_id: str | int, destination: str | Path) -> list[Path]:
    from .github import GitHubActions

    return GitHubActions.from_environment().download(str(run_id), Path(destination))
