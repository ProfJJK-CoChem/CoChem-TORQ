"""Genuine approved scan observations to immutable coordinator-fenced PES data.

The native bundle preserves original request, result, log, basis, geometry and
checkpoint bytes. Neither a caller-supplied energy nor a provenance label can
authorize a native PES sample: publication revalidates checkpoint quantities
and the exact source commit inside the coordinator transaction. This establishes
traceable electronic-model observations, never independent molecular accuracy.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import UUID

import h5py
import numpy as np
from pydantic import Field

from cochem.orchestration.campaign import (
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
)
from cochem.storage.fenced_pes import (
    PESIdentity,
    publish_pes_shard,
    shard_task_payload,
    verify_committed_pes,
    verify_pes_artifact,
)

from .domain import CalculationRequest, Contract, canonical_json, digest, read_json
from .registry import get_profile
from .scan import (
    SCAN_RECIPE,
    ScanPointResult,
    _native_scan_evidence,
    _rename_no_replace,
    _verify_native,
    geometry_for_sample,
    scan_definition,
    scan_plan_for_request,
)
from .service import ApprovedPlan

SOURCE_SCHEMA = "cochem.torq.native-pes-source/1"
BUNDLE_SCHEMA = "cochem.torq.native-pes-bundle/1"
SUPPORTED_SCAN_RECIPES = {SCAN_RECIPE, "hf-sto-3g-internal-pes-validation"}


class NativePESCollection(Contract):
    """A sealed native inventory shared by all shards of one approved scan."""

    schema_version: Literal["cochem.torq.native-pes-collection/1"] = (
        "cochem.torq.native-pes-collection/1"
    )
    identity: PESIdentity
    frame_ids: tuple[str, ...] = Field(min_length=1)
    bundle_directory: str
    bundle_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    independent_scientific_qualification: Literal[False] = False


def _hash(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _canonical_equal(first: Any, second: Any) -> bool:
    """Scientific JSON identity distinguishes booleans from numeric substitutes."""
    return canonical_json(first) == canonical_json(second)


def _regular_path(path: str | Path, *, directory: bool = False) -> Path:
    value = Path(path).absolute()
    if any(parent.is_symlink() for parent in (value, *value.parents)):
        raise ValueError("Native evidence paths cannot traverse symlinks.")
    if not (value.is_dir() if directory else value.is_file()):
        raise ValueError("Native evidence requires actual regular files/directories.")
    return value


def _confined(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        not isinstance(relative, str)
        or value.is_absolute()
        or not value.parts
        or ".." in value.parts
        or value.as_posix() != relative
    ):
        raise ValueError("Native evidence inventory paths must be canonical/confined.")
    return _regular_path(root.joinpath(*value.parts))


def _source_receipt(
    coordinator: CampaignCoordinator,
    point: ScanPointResult,
    approved: ApprovedPlan,
    *,
    transactional: bool,
) -> dict[str, Any]:
    """Check settled/latest ownership, with atomic admission during publication."""
    attempt = coordinator.attempt(point.attempt_id)
    task = coordinator.connection.execute(
        "SELECT * FROM campaign_tasks WHERE id=?", (point.task_id,)
    ).fetchone()
    campaign = coordinator.connection.execute(
        "SELECT * FROM campaigns WHERE id=?", (point.campaign_id,)
    ).fetchone()
    if (
        task is None
        or campaign is None
        or task["engine"] != "PySCF"
        or task["recipe"] != point.recipe
        or not _canonical_equal(
            json.loads(task["payload_json"]), _point_definition(point)
        )
        or not _canonical_equal(
            json.loads(campaign["plan_json"]), approved.model_dump(mode="json")
        )
        or campaign["plan_sha256"] != digest(approved.model_dump(mode="json"))
    ):
        raise CampaignError("Native source task/campaign differs from its approval.")
    if transactional:
        current = coordinator.committed_attempt_for_admission(
            point.attempt_id, expected_revision=attempt["revision"]
        )
    else:
        latest = coordinator.connection.execute(
            "SELECT id FROM campaign_attempts WHERE task_id=? "
            "ORDER BY number DESC LIMIT 1",
            (point.task_id,),
        ).fetchone()
        reservation = coordinator.connection.execute(
            "SELECT state FROM campaign_reservations WHERE attempt_id=?",
            (point.attempt_id,),
        ).fetchone()
        if (
            task is None
            or latest is None
            or latest["id"] != point.attempt_id
            or attempt["state"] != "succeeded"
            or attempt["generation"] <= 0
            or attempt["generation"] != task["next_generation"]
            or reservation is None
            or reservation["state"] != "settled"
            or not attempt["result_json"]
        ):
            raise CampaignError(
                "Native point lacks current settled committed authority."
            )
        current = {
            "attempt_id": point.attempt_id,
            "task_id": task["id"],
            "campaign_id": task["campaign_id"],
            "revision": attempt["revision"],
            "lease_generation": attempt["generation"],
            "state": attempt["state"],
            "result": json.loads(attempt["result_json"]),
        }
    if (
        not _canonical_equal(current["result"], point.model_dump(mode="json"))
        or current["campaign_id"] != point.campaign_id
        or current["task_id"] != point.task_id
        or current["attempt_id"] != point.attempt_id
    ):
        raise CampaignError("Native point differs from the exact coordinator commit.")
    return current


def _validate_approved(approved: ApprovedPlan) -> CalculationRequest:
    request = CalculationRequest.model_validate(approved.plan["request"])
    if request.recipe not in SUPPORTED_SCAN_RECIPES or not _canonical_equal(
        approved.plan, scan_plan_for_request(request, get_profile(request.recipe))
    ):
        raise ValueError("Native bridge supports only the exact reviewed scan profile.")
    return request


def _point_definition(point: ScanPointResult) -> dict[str, Any]:
    fields = (
        "point_id",
        "sample_index",
        "coordinate_values",
        "purpose",
        "sequence_index",
        "parent_point_ids",
        "molecule",
        "scan_sha256",
        "request_sha256",
        "recipe_sha256",
    )
    record = point.model_dump(mode="json")
    return {field: record[field] for field in fields}


def _validate_point(
    approved: ApprovedPlan, point: ScanPointResult, directory: Path
) -> None:
    """Re-read real engine output and reconstruct its RHF checkpoint quantities."""
    if point.status != "available":
        raise ValueError("Failed/unavailable points cannot become native PES frames.")
    if str(UUID(point.point_id)) != point.point_id:
        raise ValueError("Native point identifiers must be canonical UUIDs.")
    request = _validate_approved(approved)
    if point.geometry_status != "fixed_nonstationary_sample":
        raise ValueError(
            "This fixed-native HDF5 bridge cannot admit relaxed stationary points; "
            "retain the original relaxed JSON/native artifacts."
        )
    expected_geometry = geometry_for_sample(
        request, scan_definition(request), point.sample_index
    )
    if (
        point.molecule != expected_geometry
        or point.coordinate_values != scan_definition(request).grid[point.sample_index]
        or point.recipe != request.recipe
        or point.recipe_sha256 != approved.plan["recipe_sha256"]
        or point.scan_sha256 != approved.plan["scan_sha256"]
        or point.request_sha256 != approved.plan["request_sha256"]
        or point.source_identity_sha256 != digest(approved.source_identity)
    ):
        raise ValueError(
            "Point geometry/state/source/recipe differs from its approval."
        )
    if not _canonical_equal(
        read_json(_regular_path(directory / "point-result.json")),
        point.model_dump(mode="json"),
    ) or not _canonical_equal(
        read_json(_regular_path(directory / "point-definition.json")),
        _point_definition(point),
    ):
        raise ValueError("Preserved original point bytes differ from the observation.")
    native_directory = _regular_path(directory / "native", directory=True)
    native, manifest_sha = _verify_native(native_directory)
    evidence = _native_scan_evidence(native_directory, point.molecule)
    if (
        evidence.status != "available"
        or evidence.electronic is None
        or evidence.electronic.energy_hartree != point.energy_hartree
        or evidence.gradient != point.gradient_hartree_bohr
        or evidence.density != point.density
        or manifest_sha != point.native_manifest_sha256
    ):
        raise ValueError("Actual native quantities differ from the committed point.")
    expected_units = {
        "geometry": "bohr",
        "energy": "hartree",
        "gradient": "hartree/bohr",
        "hessian": "hartree/bohr^2",
        "dipole": "debye",
    }
    if native.get("units") != expected_units:
        raise ValueError(
            "Native geometry/energy/derivative units differ from the bridge."
        )
    backend_request = read_json(_regular_path(directory / "backend-request.json"))
    native_request = read_json(_regular_path(native_directory / "request.json"))
    if (
        not isinstance(backend_request, dict)
        or not isinstance(native_request, dict)
        or not _canonical_equal(native.get("settings"), native_request.get("settings"))
    ):
        raise ValueError(
            "Native request/settings differ from the actual point request."
        )
    expected_backend = {
        "molecule": point.molecule.model_dump(mode="json"),
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
            "dispersion": None,
        },
        "properties": ["energy", "gradient"],
        "settings": {
            **get_profile(request.recipe)["numerical"],
            "threads": request.resources.cores,
            "memory_mb": request.resources.memory_mb,
            "scf_max_cycle": scan_definition(request).scf_max_cycle,
        },
    }
    physical_backend = {
        key: value for key, value in backend_request.items() if key != "initial_guess"
    }
    if not _canonical_equal(physical_backend, expected_backend):
        raise ValueError(
            "Point backend request differs from the approved exact recipe."
        )
    from .engines.pyscf_backend import _normalize

    if not _canonical_equal(native_request, _normalize(expected_backend)):
        raise ValueError("Native normalized request differs from the approved recipe.")
    _validate_initial_guess(approved, point, directory, backend_request, native)
    required = {
        "wavefunction.chk",
        "geometry-bohr.npy",
        "basis-definition.json",
        "engine-installation.json",
    }
    inventory = read_json(native_directory / "manifest.json")
    if not required <= {entry["path"] for entry in inventory["artifacts"]}:
        raise ValueError(
            "Original native state/basis/installation evidence is missing."
        )
    installation = read_json(native_directory / "engine-installation.json")
    if installation.get("digest") != native.get("engine_installation_sha256"):
        raise ValueError("Native engine installation identity is inconsistent.")
    actual_geometry = np.load(
        native_directory / "geometry-bohr.npy", allow_pickle=False
    )
    if not np.array_equal(actual_geometry, point.molecule.geometry_bohr):
        raise ValueError("Native geometry array differs from ordered point nuclei.")
    _verify_checkpoint_quantities(native_directory, point)


def _continuation_required(approved: ApprovedPlan, point: ScanPointResult) -> bool:
    scan = scan_definition(CalculationRequest.model_validate(approved.plan["request"]))
    if scan.initial_guess_policy != "previous_point_density":
        return False
    if scan.sampling_strategy != "full_grid":
        raise ValueError("Native continuation requires the exact finite pass schedule.")
    schedule = [
        (scan_pass.purpose, index)
        for scan_pass in scan.passes
        for index in scan_pass.sample_indices
    ]
    if point.sequence_index >= len(schedule) or schedule[point.sequence_index] != (
        point.purpose,
        point.sample_index,
    ):
        raise ValueError(
            "Native point differs from its approved physical-call schedule."
        )
    return (
        point.purpose in {"forward", "reverse"}
        and point.sequence_index > 0
        and schedule[point.sequence_index - 1][0] == point.purpose
    )


def _verify_lineage(
    approved: ApprovedPlan, points: tuple[ScanPointResult, ...]
) -> None:
    available = {point.point_id: point for point in points}
    for point in points:
        if _continuation_required(approved, point):
            if (
                len(point.parent_point_ids) != 1
                or point.parent_point_ids[0] not in available
            ):
                raise ValueError(
                    "Native collections require every actual continued ancestor."
                )
            parent = available[point.parent_point_ids[0]]
            if (
                parent.sequence_index != point.sequence_index - 1
                or parent.purpose != point.purpose
                or parent.status != "available"
            ):
                raise ValueError(
                    "Continued parent must be the actual preceding same-pass point."
                )
        elif (
            scan_definition(
                CalculationRequest.model_validate(approved.plan["request"])
            ).initial_guess_policy
            == "previous_point_density"
            and point.parent_point_ids
        ):
            raise ValueError(
                "Independent pass starts/challenges cannot claim continuation parents."
            )


def _validate_initial_guess(
    approved: ApprovedPlan,
    point: ScanPointResult,
    directory: Path,
    backend_request: dict[str, Any],
    native: dict[str, Any],
) -> None:
    """Bind actual consumed density, preserved input and preceding point identity."""
    reference = backend_request.get("initial_guess")
    consumed = native.get("checkpoint_consumption")
    if not _continuation_required(approved, point):
        if reference is not None or consumed is not None:
            raise ValueError(
                "Independent pass/challenge point cannot claim checkpoint reuse."
            )
        return
    if (
        not isinstance(reference, dict)
        or set(reference)
        != {
            "policy",
            "source_native_directory",
            "source_manifest_sha256",
            "recipe_sha256",
            "parent_point_id",
        }
        or reference["policy"] != "previous_point_density"
    ):
        raise ValueError(
            "Continued points require their exact approved checkpoint reference."
        )
    parent_native_directory = _regular_path(
        reference["source_native_directory"], directory=True
    )
    parent_directory = _regular_path(parent_native_directory.parent, directory=True)
    parent = ScanPointResult.model_validate(
        read_json(parent_directory / "point-result.json")
    )
    if (
        point.parent_point_ids != (parent.point_id,)
        or reference["parent_point_id"] != parent.point_id
        or reference["source_manifest_sha256"] != parent.native_manifest_sha256
        or reference["recipe_sha256"] != point.recipe_sha256
        or parent.status != "available"
        or parent.recipe != point.recipe
        or parent.recipe_sha256 != point.recipe_sha256
        or parent.request_sha256 != point.request_sha256
        or parent.scan_sha256 != point.scan_sha256
        or parent.source_identity_sha256 != point.source_identity_sha256
        or parent.campaign_id != point.campaign_id
        or parent.purpose != point.purpose
        or parent.sequence_index != point.sequence_index - 1
        or parent_directory.name != f"point-{parent.sequence_index:04d}"
        or parent.native_manifest_path
        != f"points/point-{parent.sequence_index:04d}/native/manifest.json"
    ):
        raise ValueError(
            "Checkpoint source differs from the exact preceding approved point."
        )
    parent_native, parent_manifest_sha = _verify_native(parent_native_directory)
    if parent_manifest_sha != parent.native_manifest_sha256 or parent.density is None:
        raise ValueError("Original parent native state is missing or altered.")
    required_consumption = {
        "schema_version": "cochem.torq.native-checkpoint-consumption/1",
        "purpose": "initial_guess",
        "engine_checkpoint_reused": True,
        "new_physical_calculation_required": True,
        "fresh_scf_kernel_called": True,
        "scf_iteration_state_resumed": False,
        "final_result_reused": False,
        "chemical_accuracy_established": False,
        "registered_recipe_id": point.recipe,
        "recipe_sha256": point.recipe_sha256,
        "source_manifest_sha256": parent.native_manifest_sha256,
        "source_checkpoint_sha256": parent.density.native_checkpoint_sha256,
        "project_orbitals": True,
        "immutable_inputs_verified": True,
    }
    if not isinstance(consumed, dict) or any(
        consumed.get(key) != value
        or (type(value) is bool and type(consumed.get(key)) is not bool)
        for key, value in required_consumption.items()
    ):
        raise ValueError(
            "Actual checkpoint consumption does not match the parent/state/model."
        )
    child_native = directory / "native"
    if not _canonical_equal(
        read_json(_regular_path(child_native / "checkpoint-consumption.json")),
        consumed,
    ):
        raise ValueError("Native checkpoint consumption bytes differ from its result.")
    checkpoint_sha = parent.density.native_checkpoint_sha256
    for name in (
        "sealed-input.chk",
        "restart-input.chk",
        "source-snapshot/wavefunction.chk",
    ):
        if _hash(_regular_path(child_native / name)) != checkpoint_sha:
            raise ValueError(
                "Retained checkpoint input differs from actual parent bytes."
            )
    if (
        _hash(_regular_path(child_native / "source-snapshot/source-manifest.json"))
        != parent_manifest_sha
    ):
        raise ValueError(
            "Retained source snapshot manifest differs from actual parent."
        )
    parent_inventory = read_json(parent_native_directory / "manifest.json")["artifacts"]
    for entry in parent_inventory:
        copied = _confined(child_native / "source-snapshot", entry["path"])
        if (
            _hash(copied) != entry["sha256"]
            or copied.stat().st_size != entry["size_bytes"]
        ):
            raise ValueError(
                "Retained source snapshot differs from parent inventory bytes."
            )
    initial_density = np.load(
        _regular_path(child_native / "initial-density.npy"), allow_pickle=False
    )
    expected_dimension = len(point.density.matrix) if point.density is not None else 0
    if (
        np.iscomplexobj(initial_density)
        or initial_density.shape != (expected_dimension, expected_dimension)
        or not np.isfinite(initial_density).all()
        or not np.allclose(initial_density, initial_density.T, rtol=0, atol=1e-10)
    ):
        raise ValueError(
            "Consumed native density must be a finite real symmetric AO matrix."
        )
    density_bytes = np.asarray(initial_density, dtype="<f8").tobytes(order="C")
    density_sha = sha256(density_bytes).hexdigest()
    if (
        consumed.get("initial_density_sha256") != density_sha
        or _regular_path(child_native / "initial-density.raw").read_bytes()
        != density_bytes
    ):
        raise ValueError(
            "Native initial density bytes differ from observed consumption."
        )
    observations = consumed.get("startup_observations")
    if (
        not isinstance(observations, list)
        or len(observations) != 1
        or not isinstance(observations[0], dict)
        or observations[0].get("first_cycle_input_density_sha256") != density_sha
        or type(observations[0].get("first_cycle")) is not int
        or observations[0]["first_cycle"] < 1
    ):
        raise ValueError(
            "An actual first SCF iteration must consume the retained density."
        )
    for field in ("initial_energy_hartree", "first_cycle_energy_hartree"):
        observed = observations[0].get(field)
        if type(observed) is not float or not math.isfinite(observed):
            raise ValueError(
                "Observed SCF startup energies require actual finite floats."
            )
    electron_count = consumed.get("initial_electron_count")
    if (
        type(electron_count) is not float
        or not math.isfinite(electron_count)
        or electron_count <= 0
    ):
        raise ValueError(
            "Observed initial electron count requires a finite positive float."
        )
    _verify_consumed_fingerprints(approved, point, parent_native, native, consumed)
    from pyscf import scf
    from pyscf.lib import chkfile

    molecule = chkfile.load_mol(str(child_native / "wavefunction.chk"))
    mean_field = scf.RHF(molecule)
    mean_field.verbose = 0
    expected_density = mean_field.init_guess_by_chkfile(
        str(child_native / "restart-input.chk"), project=True
    )
    if not np.allclose(initial_density, expected_density, rtol=0, atol=1e-12):
        raise ValueError(
            "Stored initial density differs from actual projected parent orbitals."
        )
    electrons = float(np.einsum("ij,ji->", initial_density, mean_field.get_ovlp()))
    if not np.isclose(electrons, electron_count, rtol=0, atol=1e-10):
        raise ValueError(
            "Actual initial-density electron count differs from consumption."
        )


def _verify_consumed_fingerprints(
    approved: ApprovedPlan,
    point: ScanPointResult,
    parent_native: dict[str, Any],
    native: dict[str, Any],
    consumed: dict[str, Any],
) -> None:
    from .capabilities import RestartFingerprint

    molecule = parent_native["molecule"]
    source = RestartFingerprint.model_validate(
        {
            "geometry_bohr": parent_native["geometry_bohr"],
            "symbols": molecule["symbols"],
            "atom_ids": molecule["atom_ids"],
            "isotope_symbols": molecule["isotope_symbols"],
            "charge": molecule["charge"],
            "multiplicity": molecule["multiplicity"],
            "electronic_reference": parent_native["method"]["reference"],
            "basis": parent_native["method"]["basis"],
            "basis_definition_sha256": parent_native["basis_definition_sha256"],
            "recipe_sha256": point.recipe_sha256,
            "engine": parent_native["engine"],
            "engine_version": parent_native["engine_version"],
            "engine_installation_sha256": parent_native["engine_installation_sha256"],
            "adapter_source_sha256": parent_native["adapter_source_sha256"],
            "source_code_sha256": approved.source_identity["code_sha256"],
            "method_definition_json": canonical_json(parent_native["method"]).decode(),
            "numerical_settings_json": canonical_json(
                parent_native["settings"]
            ).decode(),
        }
    )
    fields = source.model_dump(mode="json")
    fields.update(
        geometry_bohr=native["geometry_bohr"],
        basis_definition_sha256=native["basis_definition_sha256"],
        method_definition_json=canonical_json(native["method"]).decode(),
        numerical_settings_json=canonical_json(native["settings"]).decode(),
    )
    target = RestartFingerprint.model_validate(fields)
    changed = [
        field
        for field in source.model_fields
        if source.model_dump(mode="json")[field]
        != target.model_dump(mode="json")[field]
    ]
    if (
        consumed.get("source_fingerprint_sha256") != source.identity_sha256
        or consumed.get("target_fingerprint_sha256") != target.identity_sha256
        or consumed.get("changed_fields") != changed
        or set(changed) - {"geometry_bohr"}
    ):
        raise ValueError(
            "Consumed checkpoint source/target fingerprints differ from native states."
        )


def _verify_checkpoint_quantities(directory: Path, point: ScanPointResult) -> None:
    """Independent readback of the stored state, without rerunning an SCF search."""
    from pyscf import scf
    from pyscf.lib import chkfile

    checkpoint = str(directory / "wavefunction.chk")
    molecule = chkfile.load_mol(checkpoint)
    state = chkfile.load(checkpoint, "scf")
    if (
        molecule.charge != point.molecule.charge
        or molecule.spin != point.molecule.multiplicity - 1
        or [molecule.atom_symbol(index) for index in range(molecule.natm)]
        != list(point.molecule.symbols)
        or state.get("e_tot") != point.energy_hartree
        or canonical_json(molecule._basis)
        != canonical_json(read_json(directory / "basis-definition.json"))
    ):
        raise ValueError("Native checkpoint state/basis/energy identity differs.")
    if not np.array_equal(
        molecule.atom_coords(unit="Bohr"), point.molecule.geometry_bohr
    ):
        raise ValueError(
            "Native checkpoint geometry differs from ordered point nuclei."
        )
    mean_field = scf.RHF(molecule)
    mean_field.verbose = 0
    mean_field.mo_coeff = state["mo_coeff"]
    mean_field.mo_occ = state["mo_occ"]
    mean_field.mo_energy = state["mo_energy"]
    density = mean_field.make_rdm1()
    energy = float(mean_field.energy_tot(dm=density))
    gradient = mean_field.nuc_grad_method().kernel()
    if point.energy_hartree is None or not np.isclose(
        energy, point.energy_hartree, rtol=0, atol=1e-10
    ):
        raise ValueError("RHF checkpoint reconstruction rejects the supplied energy.")
    if not np.allclose(gradient, point.gradient_hartree_bohr, rtol=0, atol=1e-10):
        raise ValueError("RHF checkpoint reconstruction rejects the supplied gradient.")


def _inventory(directory: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("Native bundles forbid symlinks and special files.")
        if path.is_file() and path != directory / "manifest.json":
            entries.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": _hash(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return entries


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _copy_exact(source: Path, destination: Path) -> None:
    source = _regular_path(source)
    original_hash, original_size = _hash(source), source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    destination.chmod(0o600)
    if (
        _hash(destination) != original_hash
        or destination.stat().st_size != original_size
    ):
        raise ValueError("Native bytes changed while creating an immutable snapshot.")


def prepare_native_scan_collection(
    coordinator: CampaignCoordinator,
    *,
    workspace: str | Path,
    points: tuple[ScanPointResult, ...],
    destination: str | Path,
) -> NativePESCollection:
    """Snapshot actual observations before explicitly authorizing storage tasks.

    Original engine leases need not remain alive after their committed results.
    The source coordinator must be the same durable authority used by subsequent
    storage/merge tasks; a copied database cannot provide distributed authority.
    """
    root = _regular_path(workspace, directory=True)
    approved_path = _regular_path(root / "approved-plan.json")
    approved = ApprovedPlan.model_validate(read_json(approved_path))
    request = _validate_approved(approved)
    if not points or len({point.point_id for point in points}) != len(points):
        raise ValueError(
            "Select a nonempty set of distinct actual native observations."
        )
    if tuple(point.sequence_index for point in points) != tuple(
        sorted({point.sequence_index for point in points})
    ):
        raise ValueError(
            "Native collection must preserve increasing physical-call order."
        )
    _verify_lineage(approved, points)
    target = Path(destination).absolute()
    if any(parent.is_symlink() for parent in (target, *target.parents)):
        raise ValueError("Native bundle destinations cannot traverse symlinks.")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".native-pes-", dir=target.parent))
    try:
        _copy_exact(approved_path, staging / "approved-plan.json")
        records = []
        for point in points:
            receipt = _source_receipt(coordinator, point, approved, transactional=False)
            expected_path = (
                f"points/point-{point.sequence_index:04d}/native/manifest.json"
            )
            if point.native_manifest_path != expected_path:
                raise ValueError(
                    "Native point path differs from its original call order."
                )
            source = _confined(root, expected_path).parent.parent
            _validate_point(approved, point, source)
            copied = staging / "points" / point.point_id
            for name in (
                "point-result.json",
                "point-definition.json",
                "backend-request.json",
                "native/manifest.json",
            ):
                _copy_exact(_confined(source, name), copied / name)
            inventory = read_json(source / "native" / "manifest.json")
            for entry in inventory["artifacts"]:
                name = f"native/{entry['path']}"
                _copy_exact(_confined(source, name), copied / name)
            _validate_point(approved, point, copied)
            records.append(
                {
                    "point_id": point.point_id,
                    "point_result_sha256": _hash(copied / "point-result.json"),
                    "original_point_directory": str(source),
                    "native_manifest_sha256": point.native_manifest_sha256,
                    "committed_receipt": receipt,
                }
            )
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "approved_plan_sha256": digest(approved.model_dump(mode="json")),
            "profile": get_profile(request.recipe),
            "points": records,
            "files": _inventory(staging),
            "independent_scientific_qualification": False,
        }
        _write_json(staging / "manifest.json", manifest)
        _rename_no_replace(staging, target)
        identity = _bundle_identity(target, manifest, approved)
        collection = NativePESCollection(
            identity=identity,
            frame_ids=tuple(point.point_id for point in points),
            bundle_directory=str(target),
            bundle_manifest_sha256=_hash(target / "manifest.json"),
        )
        verify_native_collection(collection)
        return collection
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _bundle_identity(
    root: Path, manifest: dict[str, Any], approved: ApprovedPlan
) -> PESIdentity:
    request = _validate_approved(approved)
    return PESIdentity(
        molecule=request.molecule,
        recipe=manifest["profile"],
        source_identity={
            "schema_version": SOURCE_SCHEMA,
            "approved_plan_sha256": manifest["approved_plan_sha256"],
            "request_sha256": approved.plan["request_sha256"],
            "scan_sha256": approved.plan["scan_sha256"],
            "implementation_source_identity": approved.source_identity,
            "implementation_source_sha256": digest(approved.source_identity),
            "native_bundle_directory": str(root),
            "native_bundle_manifest_sha256": _hash(root / "manifest.json"),
            "energy_reference": "absolute_electronic_energy_hartree",
            "geometry_status": "fixed_nonstationary_sample",
            "original_native_bytes_preserved": True,
            "independent_scientific_qualification": False,
        },
        evidence_class="native_engine_observation",
        coordinate_unit="bohr",
        energy_unit="hartree",
        engine="PySCF",
    )


def verify_native_collection(
    collection: NativePESCollection,
) -> tuple[dict[str, Any], tuple[ScanPointResult, ...]]:
    """Verify original snapshots, approval/model identity and real state quantities."""
    root = _regular_path(collection.bundle_directory, directory=True)
    manifest_path = _regular_path(root / "manifest.json")
    if _hash(manifest_path) != collection.bundle_manifest_sha256:
        raise ValueError(
            "Native bundle manifest no longer matches its frozen identity."
        )
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "approved_plan_sha256",
            "profile",
            "points",
            "files",
            "independent_scientific_qualification",
        }
        or manifest["schema_version"] != BUNDLE_SCHEMA
        or manifest["independent_scientific_qualification"] is not False
        or not _canonical_equal(manifest["files"], _inventory(root))
    ):
        raise ValueError("Native bundle schema/inventory/bytes are invalid.")
    approved = ApprovedPlan.model_validate(read_json(root / "approved-plan.json"))
    request = _validate_approved(approved)
    if (
        manifest["approved_plan_sha256"] != digest(approved.model_dump(mode="json"))
        or not _canonical_equal(manifest["profile"], get_profile(request.recipe))
        or not _canonical_equal(
            collection.identity.record(),
            _bundle_identity(root, manifest, approved).record(),
        )
    ):
        raise ValueError("Native bundle approval/recipe/source/units identity differs.")
    points = []
    for record in manifest["points"]:
        point_id = record["point_id"]
        if not isinstance(point_id, str) or str(UUID(point_id)) != point_id:
            raise ValueError("Native source point IDs must be canonical UUIDs.")
        directory = _regular_path(root / "points" / point_id, directory=True)
        point = ScanPointResult.model_validate(
            read_json(directory / "point-result.json")
        )
        if (
            point.point_id != point_id
            or record["point_result_sha256"] != _hash(directory / "point-result.json")
            or record["native_manifest_sha256"] != point.native_manifest_sha256
            or not _canonical_equal(
                record["committed_receipt"]["result"], point.model_dump(mode="json")
            )
        ):
            raise ValueError(
                "Native point bytes/result/receipt binding is inconsistent."
            )
        _validate_point(approved, point, directory)
        points.append(point)
    if tuple(point.point_id for point in points) != collection.frame_ids or tuple(
        point.sequence_index for point in points
    ) != tuple(sorted({point.sequence_index for point in points})):
        raise ValueError("Native source frame identity/order is inconsistent.")
    _verify_lineage(approved, tuple(points))
    return manifest, tuple(points)


def _collection_from_identity(identity: PESIdentity) -> NativePESCollection:
    if identity.evidence_class != "native_engine_observation":
        raise ValueError("Native bridge requires the actual native evidence class.")
    source = identity.source_identity
    root = _regular_path(source["native_bundle_directory"], directory=True)
    manifest = read_json(root / "manifest.json")
    return NativePESCollection(
        identity=identity,
        frame_ids=tuple(record["point_id"] for record in manifest["points"]),
        bundle_directory=str(root),
        bundle_manifest_sha256=source["native_bundle_manifest_sha256"],
    )


def native_shard_task_payload(
    collection: NativePESCollection, frame_ids: list[str]
) -> dict[str, Any]:
    """The exact payload to register before allocating and leasing a shard writer."""
    _, points = verify_native_collection(collection)
    _select_frames(points, frame_ids)
    return shard_task_payload(collection.identity, frame_ids)


def _select_frames(
    points: tuple[ScanPointResult, ...], frame_ids: list[str]
) -> tuple[ScanPointResult, ...]:
    if (
        not isinstance(frame_ids, list)
        or not frame_ids
        or any(not isinstance(value, str) for value in frame_ids)
    ):
        raise ValueError("Select actual nonempty native frame IDs in original order.")
    available = {point.point_id: point for point in points}
    if len(set(frame_ids)) != len(frame_ids) or set(frame_ids) - available.keys():
        raise ValueError("Native frame IDs must be actual, unique source observations.")
    selected = tuple(available[frame_id] for frame_id in frame_ids)
    if tuple(point.sequence_index for point in selected) != tuple(
        sorted(point.sequence_index for point in selected)
    ):
        raise ValueError("Native shards/merges preserve original physical-call order.")
    return selected


def _admission_lineage(
    approved: ApprovedPlan,
    points: tuple[ScanPointResult, ...],
    selected: tuple[ScanPointResult, ...],
) -> tuple[ScanPointResult, ...]:
    """Admit consumed parents even when a shard selects only a child frame."""
    available = {point.point_id: point for point in points}
    admitted = {point.point_id: point for point in selected}
    pending = list(selected)
    while pending:
        point = pending.pop()
        if _continuation_required(approved, point):
            parent_id = point.parent_point_ids[0]
            if parent_id not in admitted:
                admitted[parent_id] = available[parent_id]
                pending.append(available[parent_id])
    return tuple(sorted(admitted.values(), key=lambda point: point.sequence_index))


def publish_native_scan_shard(
    coordinator: CampaignCoordinator,
    attempt_id: str,
    expected_revision: int,
    *,
    collection: NativePESCollection,
    frame_ids: list[str],
    destination: str | Path,
    lease_token: str,
    lease_generation: int,
    usage: MeasuredUsage,
    actor: str,
) -> Path:
    """Publish only actual stored coordinates/energies, never caller quantities."""
    _, points = verify_native_collection(collection)
    selected = _select_frames(points, frame_ids)
    return publish_pes_shard(
        coordinator,
        attempt_id,
        expected_revision,
        identity=collection.identity,
        coordinates=[point.molecule.geometry_bohr for point in selected],
        energies=[point.energy_hartree for point in selected],
        frame_ids=frame_ids,
        destination=destination,
        lease_token=lease_token,
        lease_generation=lease_generation,
        usage=usage,
        actor=actor,
    )


def admit_native_scan_points(
    coordinator: CampaignCoordinator, destination: str | Path
) -> None:
    """Mandatory transaction-only admission for native shards, merges and recovery.

    This function is invoked by fenced_pes, not as a caller-controlled callback.
    Authority checks keep original settled results valid after lease expiry, and
    reject a superseded generation, absent attempt, altered receipt or native file.
    """
    artifact = verify_pes_artifact(destination)
    identity = PESIdentity.model_validate(artifact["identity"])
    collection = _collection_from_identity(identity)
    manifest, points = verify_native_collection(collection)
    selected = _select_frames(points, artifact["frame_ids"])
    records = {record["point_id"]: record for record in manifest["points"]}
    with h5py.File(Path(destination) / "samples.h5", "r", locking=True) as store:
        if not np.array_equal(
            store["coordinates"][:],
            np.asarray([point.molecule.geometry_bohr for point in selected]),
        ) or not np.array_equal(
            store["energies"][:],
            np.asarray([point.energy_hartree for point in selected]),
        ):
            raise ValueError(
                "Native HDF5 quantities differ from original native frames."
            )
    approved = ApprovedPlan.model_validate(
        read_json(Path(collection.bundle_directory) / "approved-plan.json")
    )
    for point in _admission_lineage(approved, points, selected):
        original = _regular_path(
            records[point.point_id]["original_point_directory"], directory=True
        )
        _validate_point(approved, point, original)
        receipt = _source_receipt(coordinator, point, approved, transactional=True)
        if not _canonical_equal(receipt, records[point.point_id]["committed_receipt"]):
            raise CampaignError("Native source ownership changed since bundle sealing.")


def verify_committed_native_pes(
    coordinator: CampaignCoordinator, destination: str | Path
) -> dict[str, Any]:
    """Verify the durable storage receipt and current source identity read-only."""
    artifact = verify_committed_pes(coordinator, destination)
    identity = PESIdentity.model_validate(artifact["identity"])
    manifest, points = verify_native_collection(_collection_from_identity(identity))
    selected = _select_frames(points, artifact["frame_ids"])
    records = {record["point_id"]: record for record in manifest["points"]}
    approved = ApprovedPlan.model_validate(
        read_json(
            Path(identity.source_identity["native_bundle_directory"])
            / "approved-plan.json"
        )
    )
    for point in _admission_lineage(approved, points, selected):
        if not _canonical_equal(
            _source_receipt(coordinator, point, approved, transactional=False),
            records[point.point_id]["committed_receipt"],
        ):
            raise CampaignError("Native source no longer has its exact current commit.")
    return artifact
