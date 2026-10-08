"""Approved, bounded experimental PES observations from genuine fixed geometries.

The first executable profile is a neutral H2 singlet bond scan at RHF/STO-3G.
Coordinates define fixed geometries, never optimized stationary points. Ordered
independent-restart passes can reveal discrepancies; they do not establish
continuation-branch completeness or an experimentally accurate potential.
"""

from __future__ import annotations

import ctypes
import math
import os
import signal
import socket
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import psutil
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from .domain import (
    CalculationRequest,
    Contract,
    Molecule,
    canonical_json,
    digest,
    read_json,
)
from .engines.pyscf_backend import PySCFBackend
from .registry import get_profile
from .scientific_values import ElectronicEnergy
from .service import ApprovedPlan, validate_approved_plan
from .units import convert, real_values

ScanPurpose = Literal[
    "forward", "reverse", "challenge", "adaptive_seed", "adaptive_acquisition"
]
SCAN_RECIPE = "hf-sto-3g-pes-validation"


class CoordinateDomain(Contract):
    minimum: StrictFloat
    maximum: StrictFloat
    periodic: StrictBool
    period: StrictFloat | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def ordered_domain(self) -> Self:
        if self.maximum <= self.minimum:
            raise ValueError("Coordinate domains must increase.")
        if self.periodic != (self.period is not None):
            raise ValueError("Periodicity requires an explicit period, and vice versa.")
        if self.period is not None and not math.isclose(
            self.maximum - self.minimum, self.period, rel_tol=1e-13, abs_tol=0
        ):
            raise ValueError("A periodic domain must span exactly its declared period.")
        return self


class InternalCoordinate(Contract):
    coordinate_id: str = Field(min_length=1, max_length=128)
    kind: Literal["bond", "angle", "dihedral"]
    atom_indices: tuple[StrictInt, ...]
    indexing: Literal["zero_based_request_atom_order"] = "zero_based_request_atom_order"
    unit: Literal["bohr", "angstrom", "radian", "degree"]
    domain: CoordinateDomain

    @model_validator(mode="after")
    def coordinate_definition(self) -> Self:
        expected = {"bond": 2, "angle": 3, "dihedral": 4}[self.kind]
        if (
            len(self.atom_indices) != expected
            or len(set(self.atom_indices)) != expected
        ):
            raise ValueError(
                "Internal coordinates require distinct, correctly counted atoms."
            )
        if min(self.atom_indices) < 0:
            raise ValueError("Coordinate atom indices cannot be negative.")
        if self.kind == "bond":
            if self.unit not in {"bohr", "angstrom"} or self.domain.minimum <= 0:
                raise ValueError(
                    "Bond domains require positive lengths in bohr or angstrom."
                )
            if self.domain.periodic:
                raise ValueError("A bond length is not a periodic coordinate.")
        elif self.unit not in {"radian", "degree"}:
            raise ValueError("Angular coordinates require explicit angular units.")
        if self.kind == "angle" and self.domain.periodic:
            raise ValueError(
                "A valence angle cannot be assigned torsional periodicity."
            )
        if self.kind == "dihedral":
            expected_period = 2 * math.pi if self.unit == "radian" else 360.0
            if (
                not self.domain.periodic
                or self.domain.period is None
                or not math.isclose(
                    self.domain.period, expected_period, rel_tol=1e-13, abs_tol=0
                )
            ):
                raise ValueError(
                    "A torsion requires its full physical 2pi/360-degree period."
                )
        return self


class ScanBudget(Contract):
    max_physical_calls: StrictInt = Field(ge=1, le=1000)
    per_point_wall_seconds: StrictInt = Field(ge=1, le=1800)


class ScanPass(Contract):
    purpose: Literal["forward", "reverse", "challenge"]
    sample_indices: tuple[StrictInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_indices(self) -> Self:
        if min(self.sample_indices) < 0 or len(set(self.sample_indices)) != len(
            self.sample_indices
        ):
            raise ValueError("A pass must select distinct nonnegative grid indices.")
        return self


class ScanPlan(Contract):
    schema_version: Literal["cochem.torq.pes-scan/1"] = "cochem.torq.pes-scan/1"
    coordinates: tuple[InternalCoordinate, ...] = Field(min_length=1, max_length=6)
    grid: tuple[tuple[StrictFloat, ...], ...] = Field(min_length=1, max_length=1000)
    sampling_rule: Literal["explicit_finite_grid"] = "explicit_finite_grid"
    sampling_strategy: Literal["full_grid", "bounded_adaptive"]
    passes: tuple[ScanPass, ...]
    coordinate_treatment: Literal["fixed", "relaxed"]
    unscanned_coordinates: Literal["frozen_cartesian", "constrained_relaxation"]
    additional_constraints: tuple[str, ...]
    initial_guess_policy: Literal["independent_pyscf_minao", "previous_point_density"]
    energy_reference: Literal["absolute_electronic_energy_hartree"] = (
        "absolute_electronic_energy_hartree"
    )
    imposed_symmetry: Literal["none"] = "none"
    budget: ScanBudget
    energy_recheck_tolerance_hartree: StrictFloat = Field(gt=0)
    density_recheck_tolerance: StrictFloat = Field(gt=0)
    scf_max_cycle: StrictInt = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def finite_design(self) -> Self:
        if len({item.coordinate_id for item in self.coordinates}) != len(
            self.coordinates
        ):
            raise ValueError("Coordinate identities must be distinct.")
        if len(set(self.grid)) != len(self.grid):
            raise ValueError(
                "Grid samples must be unique; no symmetry division is applied."
            )
        for values in self.grid:
            if len(values) != len(self.coordinates):
                raise ValueError("Every sample requires every declared coordinate.")
            for value, coordinate in zip(values, self.coordinates):
                domain = coordinate.domain
                if not domain.minimum <= value <= domain.maximum or (
                    domain.periodic and value == domain.maximum
                ):
                    raise ValueError(
                        "Samples must lie in their domain; "
                        "periodic grids are half-open."
                    )
        if any(
            index >= len(self.grid)
            for scan_pass in self.passes
            for index in scan_pass.sample_indices
        ):
            raise ValueError("A pass selects a sample outside the approved grid.")
        if self.sampling_strategy == "full_grid":
            purposes = [item.purpose for item in self.passes]
            if (
                purposes.count("forward") != 1
                or purposes.count("reverse") != 1
                or purposes.count("challenge") != 1
            ):
                raise ValueError(
                    "A full scan requires one forward, reverse "
                    "and independent challenge pass."
                )
            for scan_pass in self.passes:
                expected = tuple(range(len(self.grid)))
                if scan_pass.purpose == "reverse":
                    expected = tuple(reversed(expected))
                if (
                    scan_pass.purpose != "challenge"
                    and scan_pass.sample_indices != expected
                ):
                    raise ValueError(
                        "Forward/reverse passes must traverse "
                        "the full declared grid in order."
                    )
            if (
                sum(len(item.sample_indices) for item in self.passes)
                > self.budget.max_physical_calls
            ):
                raise ValueError(
                    "The actual declared physical calls exceed the scan budget."
                )
        elif self.passes:
            raise ValueError(
                "Adaptive scheduling belongs to its separately "
                "approved adaptive declaration."
            )
        return self

    @property
    def unique_sample_count(self) -> int:
        return len(self.grid)

    @property
    def planned_physical_calls(self) -> int:
        if self.sampling_strategy == "bounded_adaptive":
            return self.budget.max_physical_calls
        return sum(len(item.sample_indices) for item in self.passes)


def scan_definition(request: CalculationRequest) -> ScanPlan:
    """Read only the exact explicit declaration bound into a reviewed request."""
    raw = request.source_provenance.get("pes_scan")
    if not isinstance(raw, dict):
        raise ValueError(
            "A scan requires source_provenance.pes_scan as a complete ScanPlan."
        )
    return ScanPlan.model_validate(raw)


def scan_blocking_reasons(
    request: CalculationRequest, scan: ScanPlan
) -> tuple[str, ...]:
    reasons = []
    if request.recipe != SCAN_RECIPE or request.products != ["pes_scan"]:
        reasons.append(
            "Use only the explicit hf-sto-3g-pes-validation "
            "recipe and pes_scan product."
        )
    if (
        request.molecule.symbols != ["H", "H"]
        or request.molecule.charge != 0
        or request.molecule.multiplicity != 1
    ):
        reasons.append("This experimental executor supports neutral singlet H2 only.")
    if request.molecule.atom_ids is None:
        reasons.append(
            "Stable explicit atom_ids are required for atom-indexed scan lineage."
        )
    if (
        len(scan.coordinates) != 1
        or scan.coordinates[0].kind != "bond"
        or scan.coordinates[0].atom_indices != (0, 1)
        or scan.coordinates[0].domain.periodic
    ):
        reasons.append(
            "Only the nonperiodic atom-0/atom-1 H2 bond coordinate is executable."
        )
    if (
        scan.coordinate_treatment != "fixed"
        or scan.unscanned_coordinates != "frozen_cartesian"
        or scan.additional_constraints
    ):
        reasons.append(
            "Constrained relaxation/additional constraints "
            "have no validated optimizer profile."
        )
    if scan.initial_guess_policy != "independent_pyscf_minao":
        reasons.append(
            "Checkpoint-density continuation is not implemented by this native adapter."
        )
    if scan.sampling_strategy == "bounded_adaptive" and not isinstance(
        request.source_provenance.get("adaptive_scan"), dict
    ):
        reasons.append(
            "Adaptive execution requires its exact separately "
            "declared adaptive_scan policy."
        )
    return tuple(reasons)


def scan_plan_for_request(
    request: CalculationRequest, profile: dict[str, Any]
) -> dict[str, Any]:
    """Build the real scan task envelope for the standard plan/approval service."""
    scan = scan_definition(request)
    if any(
        index >= len(request.molecule.symbols)
        for item in scan.coordinates
        for index in item.atom_indices
    ):
        raise ValueError(
            "Coordinate indices exceed the immutable request atom mapping."
        )
    if (
        scan.planned_physical_calls * scan.budget.per_point_wall_seconds
        > request.resources.wall_seconds
    ):
        raise ValueError(
            "The requested point ceilings exceed the total reviewed wall budget."
        )
    if (
        profile.get("id") != SCAN_RECIPE
        or profile.get("engine") != "PySCF"
        or profile.get("method") != "hf"
        or profile.get("basis") != "sto-3g"
        or profile.get("dispersion") is not None
    ):
        raise ValueError(
            "The fixed scan requires its exact named RHF/STO-3G energy model."
        )
    scan_sha256 = digest(scan.model_dump(mode="json"))
    tasks: list[dict[str, Any]] = []
    if scan.sampling_strategy == "full_grid":
        for scan_pass in scan.passes:
            for index in scan_pass.sample_indices:
                tasks.append(
                    {
                        "id": f"point-{len(tasks):04d}",
                        "operation": "fixed_coordinate_energy_gradient",
                        "sample_index": index,
                        "purpose": scan_pass.purpose,
                        "depends_on": [],
                        "engine": "PySCF",
                        "recipe_sha256": profile["recipe_sha256"],
                    }
                )
    else:
        for index in range(scan.budget.max_physical_calls):
            tasks.append(
                {
                    "id": f"physical-call-slot-{index:04d}",
                    "operation": "bounded_adaptive_domain_energy_gradient",
                    "depends_on": [],
                    "engine": "PySCF",
                    "recipe_sha256": profile["recipe_sha256"],
                    "approved_grid_sha256": digest(scan.grid),
                    "completion_status": "not_executed",
                }
            )
    plan: dict[str, Any] = {
        "schema_version": "cochem.torq.plan/1",
        "request": request.model_dump(mode="json"),
        "request_sha256": digest(request.model_dump(mode="json")),
        "recipe_sha256": profile["recipe_sha256"],
        "resources": request.resources.model_dump(mode="json"),
        "tasks": tasks,
        "scan_sha256": scan_sha256,
        "unique_sample_count": scan.unique_sample_count,
        "physical_call_ceiling": scan.planned_physical_calls,
        "energy_reference": scan.energy_reference,
        "stationary_points_claimed": False,
        "execution_scope": "experimental_local_fixed_coordinate_pes",
    }
    plan["plan_sha256"] = digest(plan)
    return plan


class ScanMolecule(Contract):
    symbols: tuple[str, ...]
    geometry_bohr: tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...]
    charge: StrictInt
    multiplicity: StrictInt
    atom_ids: tuple[str, ...] | None
    isotopes: tuple[StrictInt | None, ...] | None

    @model_validator(mode="after")
    def physical_identity(self) -> Self:
        Molecule.model_validate(self.model_dump(mode="json"))
        return self


class DensityObservation(Contract):
    representation: Literal["RHF_spin_summed_AO_density_in_native_basis"] = (
        "RHF_spin_summed_AO_density_in_native_basis"
    )
    matrix: tuple[tuple[StrictFloat, ...], ...]
    mo_occupations: tuple[StrictFloat, ...]
    electron_count: StrictInt = Field(gt=0)
    native_checkpoint_sha256: str = Field(pattern="^[0-9a-f]{64}$")


class ScanPointResult(Contract):
    schema_version: Literal["cochem.torq.scan-point/1"] = "cochem.torq.scan-point/1"
    point_id: str
    sample_index: StrictInt = Field(ge=0)
    coordinate_values: tuple[StrictFloat, ...]
    purpose: ScanPurpose
    sequence_index: StrictInt = Field(ge=0)
    parent_point_ids: tuple[str, ...]
    status: Literal["available", "failed", "unavailable"]
    energy_hartree: StrictFloat | None
    gradient_hartree_bohr: (
        tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...] | None
    )
    reason: str | None
    molecule: ScanMolecule
    recipe: str
    recipe_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    scan_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    source_identity_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    native_manifest_path: str | None
    native_manifest_sha256: str | None = Field(pattern="^[0-9a-f]{64}$")
    density: DensityObservation | None
    campaign_id: str
    task_id: str
    attempt_id: str
    wall_seconds: StrictFloat = Field(ge=0)
    observed_peak_scratch_bytes: StrictInt | None = Field(ge=0)
    sampled_peak_memory_mb: StrictFloat | None = Field(ge=0)
    cpu_core_seconds: None = None
    telemetry_scope: Literal[
        "observed_parent_wall_and_sampled_resources_exact_cpu_unavailable"
    ] = "observed_parent_wall_and_sampled_resources_exact_cpu_unavailable"
    geometry_status: Literal["fixed_nonstationary_sample"] = (
        "fixed_nonstationary_sample"
    )
    independent_scientific_qualification: Literal[False] = False

    @model_validator(mode="after")
    def no_replacement_energy(self) -> Self:
        if self.status == "available":
            if (
                self.energy_hartree is None
                or self.gradient_hartree_bohr is None
                or self.reason is not None
                or self.native_manifest_sha256 is None
                or self.density is None
            ):
                raise ValueError(
                    "Available samples require actual validated native "
                    "energy, gradient and state evidence."
                )
            if len(self.gradient_hartree_bohr) != len(self.molecule.symbols):
                raise ValueError(
                    "Gradient atoms must match the scanned molecular identity."
                )
        elif (
            self.energy_hartree is not None
            or self.gradient_hartree_bohr is not None
            or not self.reason
        ):
            raise ValueError(
                "Failed/unavailable scan nodes remain missing with an explicit reason."
            )
        return self


class NativeScanEvidence(Contract):
    """Accepted native quantities or an explicit failed evidence validation."""

    status: Literal["available", "failed"]
    electronic: ElectronicEnergy | None
    gradient: tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...] | None
    density: DensityObservation | None
    native_manifest_sha256: str | None
    reason: str | None


class ScanComparison(Contract):
    sample_index: StrictInt
    first_point_id: str
    recheck_point_id: str
    purpose: Literal["reverse", "challenge"]
    energy_difference_hartree: StrictFloat
    density_difference_frobenius: StrictFloat
    inconsistent: bool
    scope: Literal["independent_restart_energy_and_AO_density_same_geometry"] = (
        "independent_restart_energy_and_AO_density_same_geometry"
    )


class ScanNode(Contract):
    sample_index: StrictInt
    coordinate_values: tuple[StrictFloat, ...]
    status: Literal["available", "failed", "not_evaluated"]
    representative_point_id: str | None
    energy_hartree: StrictFloat | None
    observation_ids: tuple[str, ...]
    failed_observation_ids: tuple[str, ...]


class ScanSurface(Contract):
    schema_version: Literal["cochem.torq.scan-surface/1"] = "cochem.torq.scan-surface/1"
    scan: ScanPlan
    scan_sha256: str
    request_sha256: str
    recipe: str
    recipe_sha256: str
    source_identity_sha256: str
    energy_reference: Literal["absolute_electronic_energy_hartree"]
    points: tuple[ScanPointResult, ...]
    nodes: tuple[ScanNode, ...]
    comparisons: tuple[ScanComparison, ...]
    outcome: Literal["completed_declared_calls", "partial", "failed"]
    unique_planned_samples: StrictInt
    unique_physically_attempted_samples: StrictInt
    physical_call_count: StrictInt
    status: Literal["complete", "partial", "failed"]
    request_id: str
    identification_ready: Literal[False] = False
    experimental_accuracy_established: Literal[False] = False
    representative_policy: Literal[
        "first_available_actual_observation_failed_attempts_retained"
    ] = "first_available_actual_observation_failed_attempts_retained"
    hysteresis_scope: Literal[
        "ordered_independent_restarts_no_continuation_branch_completeness"
    ] = "ordered_independent_restarts_no_continuation_branch_completeness"
    surface_completeness_established: Literal[False] = False
    stationary_points_verified: Literal[False] = False
    independent_scientific_qualification: Literal[False] = False


def geometry_for_sample(
    request: CalculationRequest, scan: ScanPlan, index: int
) -> ScanMolecule:
    if type(index) is not int or not 0 <= index < len(scan.grid):
        raise ValueError("Select an actual approved grid index.")
    reasons = scan_blocking_reasons(request, scan)
    if reasons:
        raise ValueError("; ".join(reasons))
    reference = np.asarray(request.molecule.geometry_bohr, dtype=np.float64)
    axis = reference[1] - reference[0]
    length = float(np.linalg.norm(axis))
    if length <= 0:
        raise ValueError("A zero reference bond does not define a scan axis.")
    target = float(convert(scan.grid[index][0], scan.coordinates[0].unit, "bohr"))
    reference[1] = reference[0] + target * axis / length
    payload = request.molecule.model_dump(mode="json")
    payload["geometry_bohr"] = reference.tolist()
    return ScanMolecule.model_validate(payload)


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".checkpoint")
    with temporary.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_native(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("A genuine immutable native artifact manifest is required.")
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "cochem-torq.engine-artifacts.v1"
    ):
        raise ValueError("Unsupported native manifest schema.")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Native manifest must inventory actual artifacts.")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Malformed native artifact reference.")
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or entry["path"] in seen
        ):
            raise ValueError("Native artifact paths must be unique and confined.")
        seen.add(entry["path"])
        candidate = directory.joinpath(*relative.parts)
        if any(
            parent.is_symlink()
            for parent in (candidate, *candidate.parents)
            if parent == directory or directory in parent.parents
        ):
            raise ValueError("Native artifacts must not traverse symlinks.")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != entry.get("size_bytes")
            or sha256(candidate.read_bytes()).hexdigest() != entry.get("sha256")
        ):
            raise ValueError(
                "Native artifact bytes do not match their immutable inventory."
            )
    if not {"request.json", "result.json", "pyscf.log"} <= seen:
        raise ValueError(
            "Required genuine native request/result/log artifacts are missing."
        )
    record = read_json(directory / "result.json")
    if not isinstance(record, dict):
        raise ValueError("Native result must be a result object.")
    return record, sha256(manifest_path.read_bytes()).hexdigest()


def _density_observation(directory: Path) -> DensityObservation:
    from pyscf.lib import chkfile

    checkpoint = directory / "wavefunction.chk"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError(
            "Actual native MO checkpoint is required for state comparisons."
        )
    values = chkfile.load(str(checkpoint), "scf")
    if not isinstance(values, dict):
        raise ValueError("Native checkpoint has no actual SCF orbital record.")
    coefficients = real_values(values["mo_coeff"])
    occupations = real_values(values["mo_occ"])
    if (
        coefficients.ndim != 2
        or occupations.shape != (coefficients.shape[1],)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(occupations).all()
    ):
        raise ValueError("Native MO dimensions/values are invalid.")
    if not np.all(np.isin(occupations, [0.0, 2.0])):
        raise ValueError(
            "The declared restricted singlet requires native closed-shell occupations."
        )
    density = (coefficients * occupations) @ coefficients.T
    return DensityObservation.model_validate(
        {
            "matrix": density.tolist(),
            "mo_occupations": occupations.tolist(),
            "electron_count": int(np.sum(occupations)),
            "native_checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        }
    )


def _native_scan_evidence(
    directory: Path,
    molecule: ScanMolecule,
    *,
    execution_failure: str | None = None,
) -> NativeScanEvidence:
    """Retain failed actual evidence without substituting a physical quantity."""
    native_digest: str | None = None
    try:
        manifest = directory / "manifest.json"
        if manifest.is_file() and not manifest.is_symlink():
            # A rejected inventory still has observed bytes worth retaining.
            native_digest = sha256(manifest.read_bytes()).hexdigest()
        native, native_digest = _verify_native(directory)
        if execution_failure is not None:
            raise ValueError(execution_failure)
        if native.get("status") != "complete":
            raise ValueError(
                "Native calculation failed its numerical/state checks: "
                + str(native.get("errors"))
            )
        identity = native.get("molecule")
        method = native.get("method")
        expected = molecule.model_dump(mode="json")
        if (
            not isinstance(identity, dict)
            or any(identity.get(field) != expected[field] for field in expected)
            or native.get("engine_version") != "2.14.0"
            or not isinstance(method, dict)
            or method.get("name") != "hf"
            or method.get("basis") != "sto-3g"
            or method.get("reference") != "restricted"
            or method.get("dispersion") is not None
            or method.get("frozen_core") is not False
        ):
            raise ValueError("Native point state/model differs from its approval.")
        electronic = ElectronicEnergy.model_validate(
            {
                "energy_hartree": native["energy_hartree"],
                "engine": native["engine"],
                "engine_version": native["engine_version"],
                "method": method,
                "native_manifest_sha256": native_digest,
                "scf": native["scf"],
                "stability": native["stability"],
            }
        )
        density = _density_observation(directory)
        if density.electron_count != electronic.scf.electron_count:
            raise ValueError("Actual checkpoint and SCF electron counts differ.")
        gradient = real_values(native["gradient_hartree_bohr"])
        if gradient.shape != (len(molecule.symbols), 3):
            raise ValueError("The actual gradient has inconsistent atom dimensions.")
        return NativeScanEvidence.model_validate(
            {
                "status": "available",
                "electronic": electronic,
                "gradient": gradient.tolist(),
                "density": density,
                "native_manifest_sha256": native_digest,
                "reason": None,
            }
        )
    except (KeyError, OSError, ValueError) as exc:
        return NativeScanEvidence(
            status="failed",
            electronic=None,
            gradient=None,
            density=None,
            native_manifest_sha256=native_digest,
            reason=f"Native point evidence validation failed: {exc}",
        )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically admit an immutable Linux artifact without overwriting a rival."""
    if sys.platform != "linux":
        raise RuntimeError("Immutable scan publication requires Linux renameat2.")
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        raise RuntimeError("Linux renameat2 RENAME_NOREPLACE is unavailable.")
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if source.is_file() and not source.is_symlink():
        with source.open("rb") as stream:
            os.fsync(stream.fileno())
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Immutable scientific files cannot be symlinks.")
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.{uuid4()}.pending")
    _write(pending, value)
    _rename_no_replace(pending, path)


class ApprovedScanExecutor:
    """One coordinator authority; each actual physical call has a fenced attempt.

    Approval is supplied by the existing TORQ plan service. Unknown exact CPU,
    memory and scratch peaks remain null; observed samples and conservative
    reservation charges are separate records. Native engines run in owned
    subprocesses with a real wall limit and measured resource samples.
    """

    def __init__(self, approved: ApprovedPlan | dict[str, Any], workspace: str | Path):
        from cochem.orchestration.campaign import CampaignBudget, CampaignCoordinator

        value = (
            approved.model_dump(mode="json")
            if isinstance(approved, ApprovedPlan)
            else approved
        )
        checked = validate_approved_plan(value)
        self._approved_json = checked.model_dump_json()
        self._request_json = canonical_json(checked.plan["request"])
        request = self.request
        self._scan = scan_definition(request)
        reasons = scan_blocking_reasons(request, self.scan)
        if reasons:
            raise ValueError("; ".join(reasons))
        if checked.plan != scan_plan_for_request(request, get_profile(request.recipe)):
            raise ValueError(
                "The approved task envelope is not this exact finite scan."
            )
        self._actor = f"local-os:{os.getuid()}@{socket.gethostname()}"
        self.workspace = Path(workspace).absolute()
        if self.workspace.is_symlink() or any(
            parent.is_symlink() for parent in self.workspace.parents
        ):
            raise ValueError("Scan workspaces cannot traverse symlinks.")
        if self.workspace.exists() and any(self.workspace.iterdir()):
            raise FileExistsError(
                "Use an empty scan workspace; observations are immutable."
            )
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.workspace / "points").mkdir(mode=0o700)
        _write(self.workspace / "request.json", request.model_dump(mode="json"))
        _write(self.workspace / "approved-plan.json", checked.model_dump(mode="json"))
        self._queue = CampaignCoordinator(
            self.workspace.parent / f".{self.workspace.name}.scan-campaign.sqlite"
        )
        self._receipt = self._queue.create_campaign(
            checked.model_dump(mode="json"),
            CampaignBudget(
                max_wall_seconds=float(checked.approval.max_wall_seconds),
                max_cpu_core_seconds=float(checked.approval.max_cpu_core_seconds),
                max_tasks=checked.approval.max_tasks,
                max_concurrent_workers=1,
                max_cpu_cores=request.resources.cores,
                max_memory_mb=checked.approval.max_memory_mb,
                max_scratch_mb=(checked.approval.max_scratch_bytes + 1024**2 - 1)
                // 1024**2,
                expires_at_unix=checked.approval.expires_at.timestamp(),
                allowed_engines=["PySCF"],
                allowed_recipes=[request.recipe],
                permitted_retries=0,
            ),
            actor=self._actor,
        )
        self._points: list[ScanPointResult] = []
        self._closed = False

    @property
    def request(self) -> CalculationRequest:
        return CalculationRequest.model_validate_json(self._request_json)

    @property
    def scan(self) -> ScanPlan:
        return self._scan

    @property
    def points(self) -> tuple[ScanPointResult, ...]:
        return tuple(self._points)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> Literal[False]:
        self.close()
        return False

    def close(self) -> None:
        if not self._closed:
            self._queue.close()
            self._closed = True

    def evaluate(
        self,
        index: int,
        *,
        purpose: ScanPurpose,
        parent_point_ids: tuple[str, ...] = (),
    ) -> ScanPointResult:
        from cochem.orchestration.campaign import (
            Allocation,
            BudgetExceededError,
            MeasuredUsage,
        )

        from .operations import ArtifactBackpressureError, enforce_artifact_budget

        if self._closed:
            raise RuntimeError("The scan executor is closed.")
        checked = validate_approved_plan(
            ApprovedPlan.model_validate_json(self._approved_json).model_dump(
                mode="json"
            )
        )
        request = self.request
        molecule = geometry_for_sample(request, self.scan, index)
        if len(self._points) >= self.scan.planned_physical_calls:
            raise BudgetExceededError(
                "The actual physical-call ceiling has been exhausted."
            )
        if len(set(parent_point_ids)) != len(parent_point_ids) or set(
            parent_point_ids
        ) - {point.point_id for point in self._points}:
            raise ValueError(
                "Lineage parents must be distinct actual preceding observations."
            )
        if self.scan.sampling_strategy == "full_grid":
            schedule = [
                (entry.purpose, sample)
                for entry in self.scan.passes
                for sample in entry.sample_indices
            ]
            if (purpose, index) != schedule[len(self._points)]:
                raise ValueError(
                    "The physical call differs from the exact approved pass order."
                )
        elif purpose not in {"adaptive_seed", "adaptive_acquisition", "challenge"}:
            raise ValueError(
                "Adaptive physical calls require their approved "
                "acquisition/challenge purpose."
            )
        profile = get_profile(request.recipe)
        sequence = len(self._points)
        name = f"point-{sequence:04d}"
        staging = self.workspace / f".{name}.work"
        staging.mkdir(mode=0o700)
        destination = self.workspace / "points" / name
        point_id = str(uuid4())
        payload = {
            "point_id": point_id,
            "sample_index": index,
            "coordinate_values": list(self.scan.grid[index]),
            "purpose": purpose,
            "sequence_index": sequence,
            "parent_point_ids": list(parent_point_ids),
            "molecule": molecule.model_dump(mode="json"),
            "scan_sha256": checked.plan["scan_sha256"],
            "request_sha256": checked.plan["request_sha256"],
            "recipe_sha256": profile["recipe_sha256"],
        }
        task = self._queue.register_task(
            self._receipt["campaign_id"],
            payload,
            engine="PySCF",
            recipe=request.recipe,
            plan_sha256=self._receipt["plan_sha256"],
            authority=self._receipt["authority"],
            actor=self._actor,
        )
        attempt = self._queue.new_attempt(
            task["task_id"], authority=self._receipt["authority"], actor=self._actor
        )
        attempt = self._queue.transition(
            attempt["id"],
            attempt["revision"],
            "validated",
            authority=self._receipt["authority"],
            actor=self._actor,
            reason="Validated exact approved coordinate/state/model "
            "and explicit point ceiling.",
        )
        attempt = self._queue.reserve(
            attempt["id"],
            attempt["revision"],
            Allocation(
                wall_seconds=float(self.scan.budget.per_point_wall_seconds),
                cores=request.resources.cores,
                memory_mb=request.resources.memory_mb,
                scratch_mb=(checked.approval.max_scratch_bytes + 1024**2 - 1)
                // 1024**2,
            ),
            authority=self._receipt["authority"],
            actor=self._actor,
        )
        backend_request = {
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
                "threads": request.resources.cores,
                "memory_mb": request.resources.memory_mb,
                "scf_max_cycle": self.scan.scf_max_cycle,
            },
        }
        _write(staging / "backend-request.json", backend_request)
        _write(staging / "point-definition.json", payload)
        started = time.monotonic()
        memory_samples: list[float] = []
        scratch_samples: list[int] = []
        failure: str | None = None
        process: subprocess.Popen[bytes] | None = None
        lease_token: str | None = None
        lease_generation: int | None = None
        try:
            with (staging / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cochem_torq.scan",
                        "_native_worker",
                        str(staging / "backend-request.json"),
                        str(staging / "native"),
                        str(staging / "dispatch-permit.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt = self._queue.record_dispatch(
                    attempt["id"],
                    attempt["revision"],
                    pid=process.pid,
                    authority=self._receipt["authority"],
                    actor=self._actor,
                )
                lease = self._queue.lease(
                    attempt["id"],
                    attempt["revision"],
                    pid=process.pid,
                    authority=self._receipt["authority"],
                    actor=self._actor,
                    heartbeat_seconds=1.0,
                    lease_seconds=10.0,
                )
                attempt = lease
                lease_token, lease_generation = (
                    lease["lease_token"],
                    lease["lease_generation"],
                )
                _write(
                    staging / "dispatch-permit.json",
                    {
                        "backend_request_sha256": digest(backend_request),
                        "approved_plan_sha256": checked.approval.plan_sha256,
                        "point_id": point_id,
                    },
                )
                next_heartbeat = time.monotonic() + 1.0
                while process.poll() is None:
                    if (
                        time.monotonic() - started
                        > self.scan.budget.per_point_wall_seconds
                    ):
                        failure = (
                            "The owned native worker exceeded "
                            "its approved point wall limit."
                        )
                        self._stop(process)
                        break
                    try:
                        memory = psutil.Process(process.pid).memory_info().rss / 1024**2
                        memory_samples.append(memory)
                        if memory > checked.approval.max_memory_mb:
                            failure = (
                                "Actual sampled native resident memory "
                                "exceeded its approved ceiling."
                            )
                            self._stop(process)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # No observation is added; exact peak remains unavailable.
                    try:
                        usage = enforce_artifact_budget(
                            self.workspace,
                            baseline_bytes=0,
                            max_growth_bytes=checked.approval.max_scratch_bytes,
                        )
                        scratch_samples.append(usage["owned_bytes"])
                    except ArtifactBackpressureError as exc:
                        failure = (
                            f"Actual scan scratch accounting blocked execution: {exc}"
                        )
                        self._stop(process)
                        break
                    if time.monotonic() >= next_heartbeat:
                        self._queue.heartbeat(
                            attempt["id"],
                            lease_token=lease_token,
                            lease_generation=lease_generation,
                        )
                        next_heartbeat = time.monotonic() + 1.0
                    time.sleep(0.02)
                process.wait()
                if process.returncode != 0 and failure is None:
                    failure = (
                        "The genuine native worker exited with code "
                        f"{process.returncode}; worker.log is retained."
                    )
        except BaseException:
            if process is not None and process.poll() is None:
                self._stop(process)
            raise
        if lease_token is None or lease_generation is None:
            raise RuntimeError(
                "A physical observation cannot be published "
                "without an actual worker lease."
            )
        evidence = _native_scan_evidence(
            staging / "native", molecule, execution_failure=failure
        )
        native_digest = evidence.native_manifest_sha256
        electronic, density, gradient = (
            evidence.electronic,
            evidence.density,
            evidence.gradient,
        )
        failure = evidence.reason
        point = ScanPointResult.model_validate(
            {
                **payload,
                "status": "available"
                if failure is None and electronic is not None
                else "failed",
                "energy_hartree": electronic.energy_hartree
                if failure is None and electronic is not None
                else None,
                "gradient_hartree_bohr": gradient if failure is None else None,
                "reason": failure,
                "recipe": request.recipe,
                "source_identity_sha256": digest(checked.source_identity),
                "native_manifest_path": f"points/{name}/native/manifest.json"
                if native_digest is not None
                else None,
                "native_manifest_sha256": native_digest,
                "density": density.model_dump(mode="json")
                if failure is None and density is not None
                else None,
                "campaign_id": self._receipt["campaign_id"],
                "task_id": task["task_id"],
                "attempt_id": attempt["id"],
                "wall_seconds": time.monotonic() - started,
                "observed_peak_scratch_bytes": max(scratch_samples)
                if scratch_samples
                else None,
                "sampled_peak_memory_mb": max(memory_samples)
                if memory_samples
                else None,
            }
        )
        _write(staging / "point-result.json", point.model_dump(mode="json"))
        for state in ("collecting", "validating"):
            attempt = self._queue.worker_transition(
                attempt["id"],
                attempt["revision"],
                state,
                lease_token=lease_token,
                lease_generation=lease_generation,
                actor=self._actor,
                reason="Preserved actual native files and validated "
                "the typed observation/missingness.",
            )
        self._queue.publish(
            attempt["id"],
            attempt["revision"],
            point.model_dump(mode="json"),
            MeasuredUsage(
                wall_seconds=point.wall_seconds,
                measurement_source="actual parent monotonic difference; "
                "exact CPU/resource peaks unavailable",
            ),
            lambda: _rename_no_replace(staging, destination),
            state="succeeded" if point.status == "available" else "failed",
            lease_token=lease_token,
            lease_generation=lease_generation,
            actor=self._actor,
        )
        self._points.append(point)
        return point

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def run(self) -> ScanSurface:
        if self.scan.sampling_strategy != "full_grid":
            raise ValueError(
                "Adaptive execution requires its approved adaptive runner."
            )
        for scan_pass in self.scan.passes:
            previous: str | None = None
            for index in scan_pass.sample_indices:
                point = self.evaluate(
                    index,
                    purpose=scan_pass.purpose,
                    parent_point_ids=(previous,) if previous is not None else (),
                )
                previous = point.point_id
        return self.finish()

    def finish(self) -> ScanSurface:
        """Preserve all observations and explicit masks; never interpolate energies."""
        request = self.request
        approved = ApprovedPlan.model_validate_json(self._approved_json)
        nodes: list[ScanNode] = []
        comparisons: list[ScanComparison] = []
        for index, coordinates in enumerate(self.scan.grid):
            observations = [
                point for point in self._points if point.sample_index == index
            ]
            available = [point for point in observations if point.status == "available"]
            representative = available[0] if available else None
            nodes.append(
                ScanNode(
                    sample_index=index,
                    coordinate_values=coordinates,
                    status="available"
                    if available
                    else "failed"
                    if observations
                    else "not_evaluated",
                    representative_point_id=representative.point_id
                    if representative
                    else None,
                    energy_hartree=representative.energy_hartree
                    if representative
                    else None,
                    observation_ids=tuple(point.point_id for point in observations),
                    failed_observation_ids=tuple(
                        point.point_id
                        for point in observations
                        if point.status != "available"
                    ),
                )
            )
            if (
                representative is not None
                and representative.energy_hartree is not None
                and representative.density is not None
            ):
                for point in available[1:]:
                    if (
                        point.purpose in {"reverse", "challenge"}
                        and point.energy_hartree is not None
                        and point.density is not None
                    ):
                        energy_difference = (
                            point.energy_hartree - representative.energy_hartree
                        )
                        density_difference = float(
                            np.linalg.norm(
                                np.asarray(point.density.matrix)
                                - np.asarray(representative.density.matrix)
                            )
                        )
                        comparisons.append(
                            ScanComparison(
                                sample_index=index,
                                first_point_id=representative.point_id,
                                recheck_point_id=point.point_id,
                                purpose=point.purpose,
                                energy_difference_hartree=energy_difference,
                                density_difference_frobenius=density_difference,
                                inconsistent=abs(energy_difference)
                                > self.scan.energy_recheck_tolerance_hartree
                                or density_difference
                                > self.scan.density_recheck_tolerance
                                or point.density.electron_count
                                != representative.density.electron_count,
                            )
                        )
        complete = (
            len(self._points) == self.scan.planned_physical_calls
            and all(point.status == "available" for point in self._points)
            and all(node.status == "available" for node in nodes)
            and not any(item.inconsistent for item in comparisons)
        )
        any_available = any(point.status == "available" for point in self._points)
        surface = ScanSurface(
            scan=self.scan,
            scan_sha256=approved.plan["scan_sha256"],
            request_sha256=approved.plan["request_sha256"],
            recipe=request.recipe,
            recipe_sha256=approved.plan["recipe_sha256"],
            source_identity_sha256=digest(approved.source_identity),
            energy_reference=self.scan.energy_reference,
            points=tuple(self._points),
            nodes=tuple(nodes),
            comparisons=tuple(comparisons),
            outcome="completed_declared_calls"
            if complete
            else "partial"
            if any_available
            else "failed",
            unique_planned_samples=self.scan.unique_sample_count,
            unique_physically_attempted_samples=len(
                {point.sample_index for point in self._points}
            ),
            physical_call_count=len(self._points),
            status="complete" if complete else "partial" if any_available else "failed",
            request_id=str(request.request_id),
        )
        _write(self.workspace / "result.json", surface.model_dump(mode="json"))
        _write(
            self.workspace / "campaign-events.json",
            self._queue.events(self._receipt["campaign_id"]),
        )
        _write(
            self.workspace / "campaign-accounting.json",
            self._queue.accounting(self._receipt["campaign_id"]),
        )
        from .operations import enforce_artifact_budget

        enforce_artifact_budget(
            self.workspace,
            baseline_bytes=0,
            max_growth_bytes=approved.approval.max_scratch_bytes,
        )
        return surface


def execute_scan_request(
    request: CalculationRequest,
    destination: str | Path,
    *,
    approved_plan: ApprovedPlan | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route only the explicit local scan through the existing approval service."""
    if approved_plan is None:
        from .service import approve_plan, plan_request

        approved_plan = approve_plan(
            plan_request(request, execution="local_validation"),
            actor=f"local-os:{os.getuid()}@{socket.gethostname()}",
        )
    value = (
        approved_plan.model_dump(mode="json")
        if isinstance(approved_plan, ApprovedPlan)
        else approved_plan
    )
    validated = validate_approved_plan(value, request=request.model_dump(mode="json"))
    from .operations import admit_artifact_work

    with admit_artifact_work(
        Path(destination).absolute().parent,
        incoming_bytes=validated.approval.max_scratch_bytes,
    ):
        return _execute_approved_scan(request, destination, validated)


def _execute_approved_scan(
    request: CalculationRequest,
    destination: str | Path,
    validated: ApprovedPlan,
) -> dict[str, Any]:
    with ApprovedScanExecutor(validated, destination) as executor:
        if executor.request.model_dump(mode="json") != request.model_dump(mode="json"):
            raise ValueError(
                "The submitted scan request differs from its exact reviewed approval."
            )
        if executor.scan.sampling_strategy == "bounded_adaptive":
            from .adaptive import AdaptiveScanPlan, run_adaptive_scan
            from .candidate_ledger import CandidateLedger

            adaptive_plan = AdaptiveScanPlan.model_validate(
                executor.request.source_provenance["adaptive_scan"]
            )
            with CandidateLedger(
                executor.workspace / "candidate-ledger.sqlite"
            ) as ledger:
                outcome = run_adaptive_scan(
                    executor,
                    adaptive_plan,
                    ledger=ledger,
                    actor=validated.approval.actor,
                )
            _write(
                executor.workspace / "adaptive-result.json",
                outcome.model_dump(mode="json"),
            )
            candidate_root = executor.workspace / "candidates"
            request_root = executor.workspace / "candidate-requests"
            candidate_root.mkdir(mode=0o700)
            request_root.mkdir(mode=0o700)
            for candidate in outcome.candidates:
                _write_immutable(
                    candidate_root / f"{candidate.candidate_id}.json",
                    candidate.model_dump(mode="json"),
                )
                _write_immutable(
                    request_root / f"{candidate.candidate_id}.json",
                    candidate.candidate_request.model_dump(mode="json"),
                )
            from .operations import enforce_artifact_budget

            enforce_artifact_budget(
                executor.workspace,
                baseline_bytes=0,
                max_growth_bytes=validated.approval.max_scratch_bytes,
            )
            return {
                **outcome.surface.model_dump(mode="json"),
                "adaptive": outcome.model_dump(mode="json"),
            }
        return executor.run().model_dump(mode="json")


def _native_worker(request_path: Path, directory: Path, permit_path: Path) -> int:
    """Private gated native subprocess, never a substitute engine executable."""
    started = time.monotonic()
    while not permit_path.is_file():
        if time.monotonic() - started > 60:
            raise RuntimeError("No current fenced dispatch permit was supplied.")
        time.sleep(0.02)
    request = read_json(request_path)
    permit = read_json(permit_path)
    if (
        not isinstance(request, dict)
        or not isinstance(permit, dict)
        or permit.get("backend_request_sha256") != digest(request)
    ):
        raise ValueError(
            "Native request differs from its actual leased dispatch permit."
        )
    from pyscf.scf.hf import SCF

    if SCF.init_guess != "minao":
        raise RuntimeError(
            "The pinned independent minao initial-guess policy is unavailable."
        )
    probe = PySCFBackend.probe()
    if not probe.get("available") or probe.get("version") != "2.14.0":
        raise RuntimeError("The actual pinned PySCF 2.14.0 engine is required.")
    result = PySCFBackend().evaluate(request, directory)
    return 0 if result["status"] == "complete" else 4


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "_native_worker":
        raise SystemExit(
            "Use the reviewed TORQ scan API; "
            "only the private native worker is exposed here."
        )
    raise SystemExit(
        _native_worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    )
