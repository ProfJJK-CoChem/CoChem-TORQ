"""Owned local constrained-relaxation samples, separate from fixed PES nodes.

Original native evaluations remain in the constrained optimizer's own inventory.
Only actual independently read-back constrained stationarity becomes a point;
this experimental route does not establish an equilibrium minimum or accuracy.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cochem.orchestration.host_allocation import _valid_process_start_time

from .domain import CalculationRequest, Molecule, Resources, digest
from .registry import get_profile
from .scan import (
    RELAXED_SCAN_RECIPE,
    NativeScanEvidence,
    ScanMolecule,
    ScanPlan,
    _native_scan_evidence,
)

INNER_RECIPE = "hf-sto-3g-constrained-pes-validation"


def relaxed_blocking_reasons(
    request: CalculationRequest, scan: ScanPlan
) -> tuple[str, ...]:
    """Fail before launch outside the explicitly bounded first experiment."""
    reasons = []
    if request.recipe != RELAXED_SCAN_RECIPE:
        reasons.append(
            "Constrained relaxation requires its exact separate outer recipe."
        )
    if request.products != ["pes_scan"]:
        reasons.append("Relaxed scans expose only their separately named PES product.")
    if (
        request.molecule.symbols != ["O", "H", "H"]
        or request.molecule.charge != 0
        or request.molecule.multiplicity != 1
    ):
        reasons.append(
            "This first relaxed profile is restricted to neutral singlet "
            "water in O,H,H atom order."
        )
    if request.molecule.atom_ids is None:
        reasons.append("Relaxed samples require explicit stable atom IDs.")
    if (
        request.resources.cores != 1
        or request.resources.memory_mb > 1024
        or request.resources.wall_seconds > 180
    ):
        reasons.append(
            "The first relaxed experiment requires one core, at most "
            "1024 MiB and at most 180 seconds."
        )
    if (
        scan.coordinate_treatment != "relaxed"
        or scan.unscanned_coordinates != "constrained_relaxation"
    ):
        reasons.append(
            "The relaxed recipe requires explicitly constrained energy relaxation."
        )
    if scan.relaxation is None:
        reasons.append("Supply the typed inner relaxation ceilings.")
    elif (
        scan.relaxation.per_evaluation_wall_seconds > scan.budget.per_point_wall_seconds
    ):
        reasons.append(
            "An inner evaluation ceiling cannot exceed the whole-point deadline."
        )
    if len(scan.coordinates) != 1 or any(
        coordinate.kind not in {"bond", "angle"} or coordinate.domain.periodic
        for coordinate in scan.coordinates
    ):
        reasons.append(
            "Only one explicitly nonperiodic water bond or angle target is supported."
        )
    if scan.cartesian_movable_atom_indices != (1, 2):
        reasons.append(
            "The first profile freezes oxygen and moves the two "
            "explicitly ordered hydrogen rows."
        )
    if scan.additional_constraints:
        reasons.append(
            "Additional textual constraints have no typed executable definition."
        )
    if scan.initial_guess_policy != "independent_pyscf_minao":
        reasons.append(
            "Each relaxed point starts independently; density/optimizer "
            "continuation is not enabled."
        )
    if scan.sampling_strategy != "full_grid" or scan.planned_point_attempts > 3:
        reasons.append(
            "The first experiment permits at most three scheduled "
            "full-grid optimizer attempts."
        )
    if scan.scf_max_cycle != 100:
        reasons.append(
            "The exact inner profile uses the native default 100-cycle SCF ceiling."
        )
    if not reasons:
        reasons.extend(_inner_preflight_reasons(request, scan))
    return tuple(reasons)


def _inner_preflight_reasons(
    request: CalculationRequest, scan: ScanPlan
) -> tuple[str, ...]:
    from .engines.constrained_optimization import (
        ConstrainedOptimizationSpec,
    )
    from .internal_coordinates import constraint_jacobian

    # Existing model and simple profile guards remain authoritative. Do not
    # evaluate invalid indices or silently normalize an unsupported declaration.
    if scan.relaxation is None:
        return ()
    count = len(request.molecule.symbols)
    if (
        len(scan.coordinates) != 1
        or any(
            coordinate.kind not in {"bond", "angle"} or coordinate.domain.periodic
            for coordinate in scan.coordinates
        )
        or scan.cartesian_movable_atom_indices != (1, 2)
        or count != 3
        or any(
            atom >= count
            for coordinate in scan.coordinates
            for atom in coordinate.atom_indices
        )
    ):
        return ()

    columns = [
        3 * atom + axis
        for atom in scan.cartesian_movable_atom_indices
        for axis in range(3)
    ]
    for sample_index, values in enumerate(scan.grid):
        try:
            specification = ConstrainedOptimizationSpec.model_validate(
                {
                    "constraints": [
                        {
                            "coordinate": coordinate.model_dump(mode="json"),
                            "target": target,
                        }
                        for coordinate, target in zip(
                            scan.coordinates, values, strict=True
                        )
                    ],
                    "movable_atom_indices": scan.cartesian_movable_atom_indices,
                    "max_physical_calls": (
                        scan.relaxation.max_native_evaluations_per_point
                    ),
                    "per_evaluation_wall_seconds": (
                        scan.relaxation.per_evaluation_wall_seconds
                    ),
                    "max_optimizer_iterations": (
                        scan.relaxation.max_optimizer_iterations
                    ),
                    "request_curvature": False,
                }
            )
            if (
                specification.per_evaluation_wall_seconds
                > request.resources.wall_seconds
            ):
                raise ValueError(
                    "An inner evaluation exceeds the overall approved wall budget."
                )
            jacobian = constraint_jacobian(
                request.molecule.geometry_bohr,
                tuple(target.coordinate for target in specification.constraints),
            )[:, columns]
            if np.linalg.matrix_rank(jacobian, tol=specification.rank_tolerance) != len(
                specification.constraints
            ):
                raise ValueError(
                    "The initial constrained coordinate Jacobian is singular."
                )
        except (ValueError, IndexError, np.linalg.LinAlgError) as error:
            return (
                f"Relaxed grid sample {sample_index} has an invalid "
                f"inner constrained declaration: {error}",
            )
    return ()


def point_relaxation_request(
    request: CalculationRequest, scan: ScanPlan, index: int
) -> dict[str, Any]:
    from .engines.constrained_optimization import ConstrainedOptimizationSpec

    if type(index) is not int or not 0 <= index < len(scan.grid):
        raise ValueError("Select an actual approved relaxed grid index.")
    declared = ScanPlan.model_validate(request.source_provenance.get("pes_scan"))
    if declared != scan:
        raise ValueError(
            "The relaxed point declaration differs from its bound request."
        )
    reasons = relaxed_blocking_reasons(request, scan)
    if reasons:
        raise ValueError("; ".join(reasons))
    if scan.relaxation is None:
        raise ValueError("Missing typed inner algorithm settings.")
    specification = ConstrainedOptimizationSpec.model_validate(
        {
            "constraints": [
                {"coordinate": coordinate.model_dump(mode="json"), "target": value}
                for coordinate, value in zip(
                    scan.coordinates, scan.grid[index], strict=True
                )
            ],
            "movable_atom_indices": scan.cartesian_movable_atom_indices,
            "max_physical_calls": scan.relaxation.max_native_evaluations_per_point,
            "per_evaluation_wall_seconds": scan.relaxation.per_evaluation_wall_seconds,
            "max_optimizer_iterations": scan.relaxation.max_optimizer_iterations,
            "request_curvature": False,
        }
    )
    profile = get_profile(INNER_RECIPE)
    return {
        "schema_version": "cochem.torq.relaxed-point-operation/1",
        "outer_recipe": RELAXED_SCAN_RECIPE,
        "inner_recipe": INNER_RECIPE,
        "inner_recipe_sha256": profile["recipe_sha256"],
        "specification": specification.model_dump(mode="json"),
        "specification_sha256": digest(specification.model_dump(mode="json")),
        "resources": {
            "cores": request.resources.cores,
            "memory_mb": request.resources.memory_mb,
            "wall_seconds": scan.budget.per_point_wall_seconds,
        },
    }


def execute_relaxed_point_worker(backend_request: dict[str, Any], staging: Path) -> int:
    """Use the unchanged inner owned algorithm under the outer point's lease."""
    from .engines.constrained_optimization import (
        ConstrainedOptimizationSpec,
        execute_constrained_optimization,
    )

    operation = backend_request["relaxed_optimization"]
    if (
        not isinstance(operation, dict)
        or operation.get("schema_version") != "cochem.torq.relaxed-point-operation/1"
        or operation.get("outer_recipe") != RELAXED_SCAN_RECIPE
        or operation.get("inner_recipe") != INNER_RECIPE
    ):
        raise ValueError("Unknown or mismatched relaxed native operation.")
    specification = ConstrainedOptimizationSpec.model_validate(
        operation["specification"]
    )
    profile = get_profile(INNER_RECIPE)
    if (
        digest(specification.model_dump(mode="json"))
        != operation["specification_sha256"]
        or profile["recipe_sha256"] != operation["inner_recipe_sha256"]
    ):
        raise ValueError(
            "Relaxed inner algorithm differs from its exact approved request."
        )
    resources = Resources.model_validate(operation["resources"])
    if resources.cores != 1 or not hasattr(os, "sched_getaffinity"):
        raise ValueError(
            "The relaxed validation worker requires Linux one-core containment."
        )
    available = os.sched_getaffinity(0)
    if not available:
        raise ValueError("No actual permitted CPU is available.")
    os.sched_setaffinity(0, {min(available)})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    observation = execute_constrained_optimization(
        Molecule.model_validate(backend_request["molecule"]),
        specification,
        staging / "constrained-engine",
        resources=resources,
        profile=profile,
    )
    return 0 if observation.status == "available" else 4


def _collection_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise ValueError(
            "Whole-point deadline expired during original evidence collection."
        )


def _original_json(
    path: Path,
    root: Path,
    deadline_monotonic: float | None,
    *,
    maximum_bytes: int = 16 * 1024**2,
) -> tuple[Any, str]:
    """Parse exactly the bounded original bytes whose SHA is returned."""
    _collection_deadline(deadline_monotonic)
    if not path.is_relative_to(root) or any(
        ancestor.is_symlink() for ancestor in (path, *path.parents)
    ):
        raise ValueError("Original evidence cannot escape or traverse symlinks.")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or not 0 < observed.st_size <= maximum_bytes
        ):
            raise ValueError("Original JSON evidence is not a bounded regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
        path_observed = os.stat(path, follow_symlinks=False)
        if (path_observed.st_dev, path_observed.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise ValueError("Original evidence path changed while being read.")
        if len(raw) != observed.st_size or (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Original evidence changed while being read.")
    finally:
        os.close(descriptor)
    _collection_deadline(deadline_monotonic)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate original JSON fields are invalid.")
            value[key] = item
        return value

    def constant(value: str) -> None:
        raise ValueError(f"Nonfinite original JSON constant {value} is invalid.")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant), sha256(
        raw
    ).hexdigest()


def _retained_identity(
    molecule: Molecule, initial: ScanMolecule, specification: Any
) -> None:
    original = initial.model_dump(mode="json")
    candidate = molecule.model_dump(mode="json")
    if any(
        candidate[key] != value
        for key, value in original.items()
        if key != "geometry_bohr"
    ):
        raise ValueError(
            "Original evaluation changed atom order, state or isotope identity."
        )
    frozen = set(range(len(initial.symbols))) - set(specification.movable_atom_indices)
    if any(
        candidate["geometry_bohr"][row] != original["geometry_bohr"][row]
        for row in frozen
    ):
        raise ValueError("Original evaluation moved a declared frozen atom row.")
    displacement = np.abs(
        np.asarray(molecule.geometry_bohr) - np.asarray(initial.geometry_bohr)
    )
    if np.max(displacement) > specification.max_cartesian_displacement_bohr + 1e-8:
        raise ValueError(
            "Original evaluation exceeds its declared Cartesian search bound."
        )


def _launch_receipts(
    directory: Path,
    root: Path,
    owner_pid: int,
    owner_create_time: float,
    deadline_monotonic: float | None,
) -> tuple[bool, bool, dict[str, Any] | None]:
    """A launch lower bound needs genuine bound owner evidence; exactness needs wait."""
    specifications = {
        "owner-binding.json": (
            "cochem.torq.native-owner-binding/1",
            {
                "owner_pid",
                "owner_create_time",
                "worker_pid",
                "worker_create_time",
                "parent_death_signal",
                "platform",
            },
        ),
        "parent-process-observation.json": (
            "cochem.torq.native-parent-observation/1",
            {
                "owner_pid",
                "owner_create_time",
                "worker_pid",
                "worker_create_time",
                "identity_observation",
            },
        ),
        "worker-process.json": (
            "cochem.torq.native-worker-wait/1",
            {
                "worker_pid",
                "worker_create_time",
                "wait_completed",
                "returncode",
                "owner_binding_sha256",
                "parent_observed_limit_failure",
            },
        ),
    }
    receipts: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, (schema, fields) in specifications.items():
        try:
            value, hashed = _original_json(
                directory / name, root, deadline_monotonic, maximum_bytes=16384
            )
        except FileNotFoundError:
            continue
        if (
            not isinstance(value, dict)
            or set(value) != fields | {"schema_version"}
            or value["schema_version"] != schema
        ):
            raise ValueError("Unexpected original native-process receipt schema.")
        receipts[name], hashes[name] = value, hashed
    binding = receipts.get("owner-binding.json")
    parent = receipts.get("parent-process-observation.json")
    waited = receipts.get("worker-process.json")

    def valid_start(value: Any, nullable: bool = False) -> bool:
        return _valid_process_start_time(value, nullable=nullable)

    for observed in (binding, parent):
        if observed is not None and (
            type(observed["owner_pid"]) is not int
            or observed["owner_pid"] != owner_pid
            or observed["owner_create_time"] != owner_create_time
            or not valid_start(observed["owner_create_time"])
            or type(observed["worker_pid"]) is not int
            or observed["worker_pid"] <= 0
            or not valid_start(
                observed["worker_create_time"], nullable=observed is parent
            )
        ):
            raise ValueError(
                "Native launch belongs to a foreign/unknown outer worker identity."
            )
    if binding is not None and (
        type(binding["parent_death_signal"]) is not int
        or binding["parent_death_signal"] != 9
        or binding["platform"] != "linux"
    ):
        raise ValueError("Native owner containment receipt is invalid.")
    if parent is not None and parent["identity_observation"] != (
        "observed" if parent["worker_create_time"] is not None else "unavailable"
    ):
        raise ValueError("Original parent process observation is inconsistent.")
    if (
        binding is not None
        and parent is not None
        and (
            binding["worker_pid"] != parent["worker_pid"]
            or (
                parent["worker_create_time"] is not None
                and binding["worker_create_time"] != parent["worker_create_time"]
            )
        )
    ):
        raise ValueError("Original launch and parent worker identities conflict.")
    if waited is not None and (
        parent is None
        or waited["worker_pid"] != parent["worker_pid"]
        or type(waited["worker_pid"]) is not int
        or waited["worker_create_time"] != parent["worker_create_time"]
        or waited["wait_completed"] is not True
        or type(waited["returncode"]) is not int
        or waited["owner_binding_sha256"] != hashes.get("owner-binding.json")
        or not (
            waited["parent_observed_limit_failure"] is None
            or isinstance(waited["parent_observed_limit_failure"], str)
        )
    ):
        raise ValueError("Original owned wait receipt is inconsistent.")
    return (
        binding is not None or parent is not None,
        parent is not None and waited is not None,
        waited,
    )


def collect_relaxed_point(
    staging: Path,
    initial: ScanMolecule,
    operation: dict[str, Any],
    *,
    execution_failure: str | None,
    owner_pid: int,
    owner_create_time: float,
    deadline_monotonic: float | None = None,
) -> tuple[ScanMolecule, NativeScanEvidence, dict[str, Any], str]:
    """Re-read original source/launch/native bytes without inventing exact counts."""
    from .engines.constrained_optimization import (
        ConstrainedEvaluation,
        ConstrainedOptimizationResult,
        ConstrainedOptimizationSpec,
        _kkt,
    )
    from .engines.pyscf_backend import _normalize
    from .internal_coordinates import (
        coordinate_residual,
        coordinate_target,
        coordinate_value,
    )

    root = staging / "constrained-engine"
    metadata: dict[str, Any] = {
        "schema_version": "cochem.torq.relaxed-point-evidence/1",
        "inner_recipe": INNER_RECIPE,
        "inner_recipe_sha256": operation["inner_recipe_sha256"],
        "specification_sha256": operation["specification_sha256"],
        "result_path": None,
        "result_sha256": None,
        "native_evaluation_attempts": None,
        "native_evaluation_attempt_lower_bound": 0,
        "stationary_character": None,
        "constraint_residuals_bohr_or_radian": None,
        "tangent_gradient_hartree_bohr": None,
        "equilibrium_geometry_claimed": False,
        "independent_scientific_qualification": False,
        "original_evaluation_receipts": [],
    }
    native_relative = "constrained-engine/no-available-final-native-manifest"
    try:
        if (
            type(owner_pid) is not int
            or owner_pid <= 0
            or not _valid_process_start_time(owner_create_time)
        ):
            raise ValueError("Actual bound outer worker process identity is required.")
        if deadline_monotonic is not None and (
            type(deadline_monotonic) is not float
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("The actual parent deadline must be finite.")
        _collection_deadline(deadline_monotonic)
        if any(ancestor.is_symlink() for ancestor in (root, *root.parents)):
            raise ValueError("Original constrained inventory cannot traverse symlinks.")
        specification = ConstrainedOptimizationSpec.model_validate(
            operation["specification"]
        )
        profile = get_profile(INNER_RECIPE)
        if (
            profile["recipe_sha256"] != operation["inner_recipe_sha256"]
            or digest(specification.model_dump(mode="json"))
            != operation["specification_sha256"]
        ):
            raise ValueError("Original inner scientific operation identity changed.")
        original_request, request_hash = _original_json(
            root / "request.json", root, deadline_monotonic
        )
        if original_request != {
            "molecule": initial.model_dump(mode="json"),
            "specification": specification.model_dump(mode="json"),
            "resources": operation["resources"],
            "profile": profile,
        }:
            raise ValueError(
                "Original optimizer request differs from its admitted operation."
            )
        metadata["original_request_sha256"] = request_hash
        result = None
        result_error = None
        try:
            original_result, result_hash = _original_json(
                root / "result.json", root, deadline_monotonic
            )
            metadata.update(
                result_path="constrained-engine/result.json", result_sha256=result_hash
            )
            result = ConstrainedOptimizationResult.model_validate(original_result)
            if (
                result.initial_molecule.model_dump(mode="json")
                != initial.model_dump(mode="json")
                or result.recipe != INNER_RECIPE
                or result.recipe_sha256 != operation["inner_recipe_sha256"]
                or result.specification != specification
            ):
                raise ValueError(
                    "Original constrained result differs from admitted model/geometry."
                )
        except (OSError, ValueError, KeyError) as error:
            result_error = str(error)
            result = None
        inventory = root / "evaluations"
        if inventory.is_symlink() or not inventory.is_dir():
            raise ValueError("Original evaluation inventory is missing or unsafe.")
        directories: list[Path] = []
        with os.scandir(inventory) as entries:
            for entry in entries:
                _collection_deadline(deadline_monotonic)
                if len(directories) >= specification.max_physical_calls + 1:
                    raise ValueError(
                        "Original launch inventory exceeds its bounded enumeration."
                    )
                if (
                    not re.fullmatch(r"evaluation-[0-9]{4}", entry.name)
                    or entry.is_symlink()
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    raise ValueError(
                        "Unexpected or unsafe original evaluation inventory entry."
                    )
                directories.append(Path(entry.path))
        directories.sort()
        complete = True
        observed_records = []
        for sequence, directory in enumerate(directories):
            _collection_deadline(deadline_monotonic)
            if (
                directory.name != f"evaluation-{sequence:04d}"
                or sequence >= specification.max_physical_calls
            ):
                raise ValueError(
                    "Original evaluations violate their contiguous approved ceiling."
                )
            backend, backend_hash = _original_json(
                directory / "backend-request.json", root, deadline_monotonic
            )
            molecule = Molecule.model_validate(backend["molecule"])
            _retained_identity(molecule, initial, specification)
            expected_backend = {
                "molecule": molecule.model_dump(mode="json"),
                "method": {
                    "name": "hf",
                    "basis": "sto-3g",
                    "reference": "restricted",
                    "frozen_core": False,
                    "dispersion": None,
                },
                "properties": ["energy", "gradient"],
                "settings": {
                    **profile["numerical"],
                    "threads": operation["resources"]["cores"],
                    "memory_mb": operation["resources"]["memory_mb"],
                },
            }
            if backend != expected_backend:
                raise ValueError(
                    "Original native request differs from the exact admitted recipe."
                )
            launched, waited, wait_receipt = _launch_receipts(
                directory, root, owner_pid, owner_create_time, deadline_monotonic
            )
            if launched:
                metadata["native_evaluation_attempt_lower_bound"] += 1
            complete = complete and launched and waited
            try:
                original_evaluation, evaluation_hash = _original_json(
                    directory / "evaluation.json", root, deadline_monotonic
                )
            except FileNotFoundError:
                complete = False
                continue
            observed = ConstrainedEvaluation.model_validate(original_evaluation)
            if observed.sequence_index != sequence or observed.molecule != molecule:
                raise ValueError(
                    "Original evaluation identity differs from its launch input."
                )
            if not launched or not waited:
                complete = False
            expected_path = f"evaluations/{directory.name}/native/manifest.json"
            manifest = directory / "native/manifest.json"
            if (
                observed.native_manifest_path is not None
                and observed.native_manifest_path != expected_path
            ):
                raise ValueError(
                    "Original evaluation points to a foreign native manifest."
                )
            native_request_path = directory / "native/request.json"
            if native_request_path.exists():
                native_request, _ = _original_json(
                    native_request_path, root, deadline_monotonic
                )
                if native_request != _normalize(expected_backend):
                    raise ValueError(
                        "Original normalized native request differs from admitted "
                        "input."
                    )
            if observed.status == "available":
                if (
                    wait_receipt is None
                    or wait_receipt["returncode"] != 0
                    or wait_receipt["parent_observed_limit_failure"] is not None
                ):
                    raise ValueError(
                        "Available evaluation has no successful genuine owned wait."
                    )
                _collection_deadline(deadline_monotonic)
                evidence = _native_scan_evidence(
                    manifest.parent,
                    ScanMolecule.model_validate(molecule.model_dump(mode="json")),
                )
                _collection_deadline(deadline_monotonic)
                if (
                    evidence.status != "available"
                    or evidence.electronic != observed.electronic
                    or evidence.gradient != observed.gradient_hartree_bohr
                    or evidence.density != observed.density
                    or evidence.native_manifest_sha256
                    != observed.native_manifest_sha256
                ):
                    raise ValueError(
                        "Original evaluation differs from actual native readback."
                    )
                _, manifest_hash = _original_json(manifest, root, deadline_monotonic)
                if manifest_hash != observed.native_manifest_sha256:
                    raise ValueError(
                        "Original native manifest bytes differ from evaluation hash."
                    )
            if (
                observed.purpose == "independent_final_verification"
                and sequence != len(directories) - 1
            ):
                raise ValueError(
                    "Independent final verification is not the last actual evaluation."
                )
            observed_records.append(observed)
            metadata["original_evaluation_receipts"].append(
                {
                    "sequence_index": sequence,
                    "backend_request_sha256": backend_hash,
                    "evaluation_sha256": evaluation_hash,
                    "launch_verified": launched,
                    "owned_wait_verified": waited,
                }
            )
        if result is None:
            raise ValueError(result_error or "Original constrained result unavailable.")
        if (
            len(directories) != len(result.evaluations)
            or tuple(observed_records) != result.evaluations
        ):
            complete = False
        if complete:
            metadata["native_evaluation_attempts"] = len(directories)
        if not complete:
            raise ValueError(
                "Original native attempt inventory is incomplete; exact "
                "total unavailable."
            )
        if execution_failure is not None:
            raise ValueError(execution_failure)
        if (
            result.status != "available"
            or result.stationary_character != "constrained_stationary"
            or result.final_molecule is None
            or not observed_records
        ):
            raise ValueError(result.reason or "Constrained stationarity unavailable.")
        final = observed_records[-1]
        expected_final = (
            f"evaluations/evaluation-{final.sequence_index:04d}/native/manifest.json"
        )
        if (
            final.purpose != "independent_final_verification"
            or final.status != "available"
            or final.native_manifest_path != expected_final
            or result.final_native_manifest_path != expected_final
        ):
            raise ValueError(
                "Final result does not reference the actual fresh final "
                "evaluation manifest."
            )
        _retained_identity(result.final_molecule, initial, specification)
        native_manifest = root / expected_final
        _, actual_manifest_hash = _original_json(
            native_manifest, root, deadline_monotonic
        )
        if (
            actual_manifest_hash != result.final_native_manifest_sha256
            or actual_manifest_hash != final.native_manifest_sha256
        ):
            raise ValueError(
                "Final original manifest bytes do not match both preserved "
                "observations."
            )
        final_molecule = ScanMolecule.model_validate(
            result.final_molecule.model_dump(mode="json")
        )
        evidence = _native_scan_evidence(native_manifest.parent, final_molecule)
        _collection_deadline(deadline_monotonic)
        if (
            evidence.status != "available"
            or evidence.electronic != result.electronic
            or evidence.gradient != result.gradient_hartree_bohr
            or evidence.density != final.density
        ):
            raise ValueError(
                evidence.reason or "Original final result differs from native readback."
            )
        geometry = np.asarray(final_molecule.geometry_bohr)
        residuals = [
            coordinate_residual(
                coordinate_value(geometry, target.coordinate),
                coordinate_target(target.target, target.coordinate),
                target.coordinate,
            )
            for target in specification.constraints
        ]
        kkt = _kkt(geometry, np.asarray(evidence.gradient), specification)
        if max(
            abs(value) for value in residuals
        ) > specification.constraint_tolerance_bohr_or_radian or (
            len(kkt["tangent_gradient"])
            and np.max(np.abs(kkt["tangent_gradient"]))
            > specification.tangent_gradient_tolerance_hartree_bohr
        ):
            raise ValueError("Recomputed final constraint/tangent checks failed.")
        _collection_deadline(deadline_monotonic)
        metadata.update(
            stationary_character="constrained_stationary",
            constraint_residuals_bohr_or_radian=residuals,
            tangent_gradient_hartree_bohr=kkt["tangent_gradient"].tolist(),
            constraint_rank=kkt["rank"],
            tangent_dimension=len(kkt["tangent_gradient"]),
            optimizer=result.optimizer,
        )
        return (
            final_molecule,
            evidence,
            metadata,
            "constrained-engine/" + expected_final,
        )
    except (KeyError, ValueError, OSError) as error:
        evidence = NativeScanEvidence(
            status="failed",
            electronic=None,
            gradient=None,
            density=None,
            native_manifest_sha256=None,
            reason=f"Actual constrained point unavailable: {error}",
        )
        return initial, evidence, metadata, native_relative
