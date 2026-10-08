"""Bounded actual HF energy optimization under explicit internal constraints.

Every distinct objective geometry is evaluated by a genuine owned PySCF worker.
Independent native final energy/gradient readback qualifies only constrained
stationarity on the selected Cartesian rows; it never proves an unconstrained
minimum, an equilibrium geometry, chemical accuracy or global search completeness.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import psutil
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator
from scipy.optimize import minimize
from typing_extensions import Self

from ..domain import Contract, Molecule, Resources, canonical_json, digest, read_json
from ..internal_coordinates import (
    constraint_jacobian,
    coordinate_residual,
    coordinate_target,
    coordinate_value,
)
from ..registry import get_profile
from ..scan import (
    DensityObservation,
    InternalCoordinate,
    ScanMolecule,
    _native_scan_evidence,
    _verify_native,
)
from ..scientific_values import ElectronicEnergy
from ..units import real_values

RECIPE_ID = "hf-sto-3g-constrained-pes-validation"
SCIPY_VERSION = "1.18.1"

# Activate containment before importing this module or any scientific engine.
# Linux checks are mandatory; other platforms receive an explicit failure.
_OWNED_WORKER_LAUNCHER = """
import ctypes, json, os, signal, sys
if sys.platform != 'linux':
    raise RuntimeError('Native constrained workers require Linux owner containment.')
owner_pid, owner_start = int(sys.argv[1]), float(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
if os.getppid() != owner_pid:
    raise RuntimeError('Native owner died or changed during containment activation.')
import psutil
if psutil.Process(owner_pid).create_time() != owner_start:
    raise RuntimeError('Native owner process identity changed.')
receipt = {
    'schema_version': 'cochem.torq.native-owner-binding/1',
    'owner_pid': owner_pid,
    'owner_create_time': owner_start,
    'worker_pid': os.getpid(),
    'worker_create_time': psutil.Process().create_time(),
    'parent_death_signal': int(signal.SIGKILL),
    'platform': sys.platform,
}
with open(sys.argv[5], 'x') as stream:
    json.dump(receipt, stream, sort_keys=True)
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(sys.argv[5], 0o600)
import runpy
sys.argv = ['constrained_optimization', '_native_worker', sys.argv[3], sys.argv[4]]
runpy.run_module('cochem_torq.engines.constrained_optimization', run_name='__main__')
"""


class ConstraintTarget(Contract):
    coordinate: InternalCoordinate
    target: StrictFloat

    @model_validator(mode="after")
    def physical_target(self) -> Self:
        domain = self.coordinate.domain
        if not domain.minimum <= self.target <= domain.maximum or (
            domain.periodic and self.target == domain.maximum
        ):
            raise ValueError(
                "Constraint targets must lie in their exact declared domain."
            )
        value = coordinate_target(self.target, self.coordinate)
        if self.coordinate.kind == "angle" and not 0 < value < math.pi:
            raise ValueError("Constrained valence angles must lie strictly in (0,pi).")
        return self


class ConstrainedOptimizationSpec(Contract):
    schema_version: Literal["cochem.torq.constrained-optimization/1"] = (
        "cochem.torq.constrained-optimization/1"
    )
    constraints: tuple[ConstraintTarget, ...] = Field(min_length=1, max_length=6)
    movable_atom_indices: tuple[StrictInt, ...] = Field(min_length=1, max_length=100)
    max_physical_calls: StrictInt = Field(ge=2, le=1000)
    per_evaluation_wall_seconds: StrictInt = Field(ge=1, le=1800)
    max_optimizer_iterations: StrictInt = Field(default=100, ge=1, le=1000)
    constraint_tolerance_bohr_or_radian: StrictFloat = Field(
        default=1e-8, gt=0, le=1e-6
    )
    tangent_gradient_tolerance_hartree_bohr: StrictFloat = Field(
        default=1e-5, gt=0, le=1e-3
    )
    rank_tolerance: StrictFloat = Field(default=1e-9, gt=0, le=1e-5)
    optimizer_energy_tolerance_hartree: StrictFloat = Field(
        default=1e-10, gt=0, le=1e-6
    )
    max_cartesian_displacement_bohr: StrictFloat = Field(default=2.0, gt=0, le=10)
    request_curvature: StrictBool = False
    coordinate_hessian_step_bohr: StrictFloat = Field(default=1e-4, gt=1e-7, le=1e-2)
    curvature_convergence_tolerance_hartree_bohr2: StrictFloat = Field(
        default=1e-5, gt=0, le=1e-3
    )
    curvature_zero_tolerance_hartree_bohr2: StrictFloat = Field(
        default=1e-5, gt=0, le=1e-3
    )

    @model_validator(mode="after")
    def unique_declaration(self) -> Self:
        if (
            len(set(self.movable_atom_indices)) != len(self.movable_atom_indices)
            or min(self.movable_atom_indices) < 0
        ):
            raise ValueError("Movable atom rows must be distinct nonnegative integers.")
        if len({item.coordinate.coordinate_id for item in self.constraints}) != len(
            self.constraints
        ):
            raise ValueError("Constrained coordinate identities must be distinct.")
        return self


class ConstrainedEvaluation(Contract):
    sequence_index: StrictInt = Field(ge=0)
    purpose: Literal["optimizer_objective", "independent_final_verification"]
    molecule: Molecule
    status: Literal["available", "failed"]
    electronic: ElectronicEnergy | None
    gradient_hartree_bohr: (
        tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...] | None
    )
    density: DensityObservation | None
    reason: str | None
    native_manifest_path: str | None
    native_manifest_sha256: str | None
    wall_seconds: StrictFloat = Field(ge=0)
    sampled_peak_memory_mb: StrictFloat | None = Field(ge=0)
    exact_cpu_core_seconds: None = None

    @model_validator(mode="after")
    def actual_or_missing(self) -> Self:
        if self.status == "available":
            if (
                self.electronic is None
                or self.gradient_hartree_bohr is None
                or self.density is None
                or self.native_manifest_path is None
                or self.native_manifest_sha256 is None
                or self.reason is not None
            ):
                raise ValueError(
                    "Available evaluations require actual native observations."
                )
        elif (
            self.electronic is not None
            or self.gradient_hartree_bohr is not None
            or self.density is not None
            or not self.reason
        ):
            raise ValueError(
                "Failed optimization evaluations must retain explicit missingness."
            )
        return self


class ConstrainedCurvature(Contract):
    status: Literal["available", "unavailable"]
    reason: str | None
    convention: Literal["L=E-lambda.c;unweighted_selected_Cartesian_nullspace"] = (
        "L=E-lambda.c;unweighted_selected_Cartesian_nullspace"
    )
    lagrangian_hessian_hartree_bohr2: tuple[tuple[StrictFloat, ...], ...] | None
    tangent_basis_columns: tuple[tuple[StrictFloat, ...], ...] | None
    projected_hessian_hartree_bohr2: tuple[tuple[StrictFloat, ...], ...] | None
    eigenvalues_hartree_bohr2: tuple[StrictFloat, ...] | None
    coordinate_hessians_two_steps: dict[str, Any] | None
    two_scale_max_difference_hartree_bohr2: StrictFloat | None
    character: (
        Literal[
            "positive_definite",
            "nonnegative_with_unresolved_zero_modes",
            "negative_curvature",
        ]
        | None
    )
    unconstrained_minimum_claimed: Literal[False] = False

    @model_validator(mode="after")
    def actual_or_unavailable(self) -> Self:
        quantities = (
            self.lagrangian_hessian_hartree_bohr2,
            self.tangent_basis_columns,
            self.projected_hessian_hartree_bohr2,
            self.eigenvalues_hartree_bohr2,
            self.coordinate_hessians_two_steps,
            self.two_scale_max_difference_hartree_bohr2,
            self.character,
        )
        if self.status == "available":
            if self.reason is not None or any(value is None for value in quantities):
                raise ValueError(
                    "Available curvature requires actual complete observations."
                )
        elif not self.reason or any(value is not None for value in quantities):
            raise ValueError("Unavailable curvature retains explicit missingness.")
        return self


class ConstrainedOptimizationResult(Contract):
    schema_version: Literal["cochem.torq.constrained-result/1"] = (
        "cochem.torq.constrained-result/1"
    )
    recipe: str
    recipe_sha256: str
    specification: ConstrainedOptimizationSpec
    initial_molecule: Molecule
    final_molecule: Molecule | None
    status: Literal["available", "partial", "failed"]
    reason: str | None
    stationary_character: Literal["constrained_stationary"] | None
    electronic: ElectronicEnergy | None
    gradient_hartree_bohr: (
        tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...] | None
    )
    constraint_residuals_bohr_or_radian: tuple[StrictFloat, ...] | None
    lagrange_multipliers: tuple[StrictFloat, ...] | None
    lagrange_multiplier_units: tuple[str, ...]
    tangent_gradient_hartree_bohr: tuple[StrictFloat, ...] | None
    constraint_rank: StrictInt | None
    tangent_dimension: StrictInt | None
    evaluations: tuple[ConstrainedEvaluation, ...]
    physical_call_count: StrictInt = Field(ge=0)
    exact_geometry_cache_hits: StrictInt = Field(ge=0)
    optimizer: dict[str, Any]
    curvature: ConstrainedCurvature
    native_result: dict[str, Any] | None
    final_native_manifest_path: str | None
    final_native_manifest_sha256: str | None
    wall_seconds: StrictFloat = Field(ge=0)
    quality_flags: tuple[str, ...]
    rejected_proposals: tuple[dict[str, Any], ...]
    independent_scientific_qualification: Literal[False] = False
    unconstrained_minimum_claimed: Literal[False] = False
    equilibrium_geometry_claimed: Literal[False] = False

    @model_validator(mode="after")
    def qualified_geometry(self) -> Self:
        if self.physical_call_count != len(self.evaluations):
            raise ValueError(
                "Physical-call counts must match actual evaluation records."
            )
        if self.status == "available":
            if (
                self.reason is not None
                or self.final_molecule is None
                or self.electronic is None
                or self.gradient_hartree_bohr is None
                or self.stationary_character != "constrained_stationary"
                or self.native_result is None
                or self.final_native_manifest_path is None
                or self.final_native_manifest_sha256 is None
                or self.constraint_residuals_bohr_or_radian is None
                or self.lagrange_multipliers is None
                or self.tangent_gradient_hartree_bohr is None
                or self.constraint_rank is None
                or self.tangent_dimension is None
            ):
                raise ValueError(
                    "Available constrained geometry requires native qualification."
                )
            if (
                not self.evaluations
                or self.evaluations[-1].purpose != "independent_final_verification"
                or self.evaluations[-1].status != "available"
                or self.evaluations[-1].molecule != self.final_molecule
                or self.evaluations[-1].electronic != self.electronic
                or self.evaluations[-1].gradient_hartree_bohr
                != self.gradient_hartree_bohr
                or self.evaluations[-1].native_manifest_sha256
                != self.final_native_manifest_sha256
            ):
                raise ValueError(
                    "Qualified results must bind their actual final evaluation."
                )
        elif (
            not self.reason
            or self.final_molecule is not None
            or self.electronic is not None
            or self.stationary_character is not None
            or self.gradient_hartree_bohr is not None
            or self.constraint_residuals_bohr_or_radian is not None
            or self.lagrange_multipliers is not None
            or self.tangent_gradient_hartree_bohr is not None
            or self.constraint_rank is not None
            or self.tangent_dimension is not None
            or self.native_result is not None
            or self.final_native_manifest_path is not None
            or self.final_native_manifest_sha256 is not None
        ):
            raise ValueError("Unqualified constrained geometries remain unavailable.")
        return self


class _StopOptimizationError(RuntimeError):
    """An actual resource, engine or mathematical limitation stops this attempt."""


@lru_cache(maxsize=1)
def _scipy_fingerprint() -> dict[str, Any]:
    import scipy

    if version("scipy") != SCIPY_VERSION:
        raise ValueError(
            f"This constrained profile requires actual SciPy {SCIPY_VERSION}."
        )
    root = Path(scipy.__file__).parent
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or ".so" in path.name):
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {
        "engine": "SciPy SLSQP",
        "version": SCIPY_VERSION,
        "installed_implementation_sha256": digest(records),
        "fingerprint_scheme": "sha256 installed SciPy Python/native library bytes",
    }


def _write(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


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
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _unavailable_curvature(reason: str) -> ConstrainedCurvature:
    return ConstrainedCurvature(
        status="unavailable",
        reason=reason,
        lagrangian_hessian_hartree_bohr2=None,
        tangent_basis_columns=None,
        projected_hessian_hartree_bohr2=None,
        eigenvalues_hartree_bohr2=None,
        coordinate_hessians_two_steps=None,
        two_scale_max_difference_hartree_bohr2=None,
        character=None,
    )


def _coordinates(
    specification: ConstrainedOptimizationSpec,
) -> tuple[InternalCoordinate, ...]:
    return tuple(item.coordinate for item in specification.constraints)


def _residual(
    geometry: np.ndarray, specification: ConstrainedOptimizationSpec
) -> np.ndarray:
    return np.asarray(
        [
            coordinate_residual(
                coordinate_value(geometry, item.coordinate),
                coordinate_target(item.target, item.coordinate),
                item.coordinate,
            )
            for item in specification.constraints
        ],
        dtype=float,
    )


def _columns(specification: ConstrainedOptimizationSpec) -> np.ndarray:
    return np.asarray(
        [
            3 * atom + axis
            for atom in specification.movable_atom_indices
            for axis in range(3)
        ]
    )


def _jacobian(
    geometry: np.ndarray,
    specification: ConstrainedOptimizationSpec,
    *,
    step_bohr: float = 1e-5,
) -> np.ndarray:
    return np.asarray(
        constraint_jacobian(geometry, _coordinates(specification), step_bohr=step_bohr)[
            :, _columns(specification)
        ],
        dtype=float,
    )


def _kkt(
    geometry: np.ndarray,
    gradient: np.ndarray,
    specification: ConstrainedOptimizationSpec,
) -> dict[str, Any]:
    jacobian = _jacobian(geometry, specification)
    rank = int(np.linalg.matrix_rank(jacobian, tol=specification.rank_tolerance))
    if rank != len(specification.constraints):
        raise ValueError(
            "Constrained geometry has a singular/dependent coordinate Jacobian."
        )
    selected_gradient = gradient.reshape(-1)[_columns(specification)]
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=True)
    relative_rank_cutoff = specification.rank_tolerance / float(singular_values[0])
    multipliers = np.linalg.lstsq(
        jacobian.T, selected_gradient, rcond=relative_rank_cutoff
    )[0]
    tangent = right_vectors[rank:].T
    expected_dimension = len(selected_gradient) - rank
    if (
        tangent.shape != (len(selected_gradient), expected_dimension)
        or not np.allclose(jacobian @ tangent, 0, rtol=0, atol=1e-9)
        or not np.allclose(
            tangent.T @ tangent, np.eye(expected_dimension), rtol=0, atol=1e-10
        )
    ):
        raise ValueError(
            "The independent Cartesian constraint tangent basis is invalid."
        )
    tangent_gradient = tangent.T @ selected_gradient
    projected_residual = selected_gradient - jacobian.T @ multipliers
    if not np.allclose(
        tangent @ tangent_gradient, projected_residual, rtol=0, atol=1e-9
    ):
        raise ValueError(
            "Independent tangent projection and multiplier residual differ."
        )
    return {
        "jacobian": jacobian,
        "rank": rank,
        "multipliers": multipliers,
        "tangent": tangent,
        "tangent_gradient": tangent_gradient,
        "projected_residual": projected_residual,
    }


def _coordinate_hessians(
    geometry: np.ndarray, specification: ConstrainedOptimizationSpec, step: float
) -> np.ndarray:
    columns = _columns(specification)
    matrices = np.empty((len(specification.constraints), len(columns), len(columns)))
    for position, column in enumerate(columns):
        plus, minus = geometry.copy(), geometry.copy()
        plus.flat[column] += step
        minus.flat[column] -= step
        matrices[:, :, position] = (
            _jacobian(plus, specification) - _jacobian(minus, specification)
        ) / (2 * step)
    if not np.isfinite(matrices).all():
        raise ValueError(
            "Actual coordinate Hessian differentiation produced nonfinite values."
        )
    return matrices


def _curvature(
    native: dict[str, Any],
    geometry: np.ndarray,
    kkt: dict[str, Any],
    specification: ConstrainedOptimizationSpec,
) -> ConstrainedCurvature:
    try:
        hessian = real_values(native["hessian_hartree_bohr2"])
        if (
            hessian.shape != (geometry.size, geometry.size)
            or native.get("hessian_evidence", {}).get("derivative") != "analytic"
        ):
            raise ValueError(
                "Actual analytic HF Hessian is unavailable or has invalid dimensions."
            )
        columns = _columns(specification)
        energy_hessian = hessian[np.ix_(columns, columns)]
        step = specification.coordinate_hessian_step_bohr
        coarse = _coordinate_hessians(geometry, specification, step)
        fine = _coordinate_hessians(geometry, specification, step / 2)
        multipliers = kkt["multipliers"]
        coarse_lagrangian = energy_hessian - np.einsum("i,ijk->jk", multipliers, coarse)
        fine_lagrangian = energy_hessian - np.einsum("i,ijk->jk", multipliers, fine)
        error = float(np.max(np.abs(coarse_lagrangian - fine_lagrangian)))
        asymmetry = float(np.max(np.abs(fine_lagrangian - fine_lagrangian.T)))
        if (
            max(error, asymmetry)
            > specification.curvature_convergence_tolerance_hartree_bohr2
        ):
            raise ValueError(
                "Constrained curvature failed actual two-step/symmetry checks."
            )
        lagrangian = (fine_lagrangian + fine_lagrangian.T) / 2
        tangent = kkt["tangent"]
        projected = tangent.T @ lagrangian @ tangent
        eigenvalues = np.linalg.eigvalsh(projected)
        zero = specification.curvature_zero_tolerance_hartree_bohr2
        character = (
            "negative_curvature"
            if np.any(eigenvalues < -zero)
            else "positive_definite"
            if len(eigenvalues) and np.all(eigenvalues > zero)
            else "nonnegative_with_unresolved_zero_modes"
        )
        return ConstrainedCurvature.model_validate(
            {
                "status": "available",
                "reason": None,
                "lagrangian_hessian_hartree_bohr2": lagrangian.tolist(),
                "tangent_basis_columns": tangent.tolist(),
                "projected_hessian_hartree_bohr2": projected.tolist(),
                "eigenvalues_hartree_bohr2": eigenvalues.tolist(),
                "coordinate_hessians_two_steps": {
                    "steps_bohr": [step, step / 2],
                    "raw_coordinate_hessians": [coarse.tolist(), fine.tolist()],
                    "raw_lagrangian_hessians": [
                        coarse_lagrangian.tolist(),
                        fine_lagrangian.tolist(),
                    ],
                    "max_fine_lagrangian_asymmetry_hartree_bohr2": asymmetry,
                },
                "two_scale_max_difference_hartree_bohr2": error,
                "character": character,
            }
        )
    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
        return _unavailable_curvature(
            f"Actual constrained curvature unavailable: {exc}"
        )


class _Evaluator:
    def __init__(
        self,
        molecule: Molecule,
        specification: ConstrainedOptimizationSpec,
        destination: Path,
        resources: Resources,
        profile: dict[str, Any],
    ):
        self.molecule, self.specification, self.destination = (
            molecule,
            specification,
            destination,
        )
        self.resources, self.profile = resources, profile
        self.started = time.monotonic()
        self.evaluations: list[ConstrainedEvaluation] = []
        self.cache: dict[str, ConstrainedEvaluation] = {}
        self.cache_hits = 0
        self.rejected: list[dict[str, Any]] = []

    def candidate(self, geometry: np.ndarray) -> Molecule:
        payload = self.molecule.model_dump(mode="json")
        payload["geometry_bohr"] = geometry.tolist()
        try:
            return Molecule.model_validate(payload)
        except ValueError as exc:
            rejected = {
                "geometry_bohr": geometry.tolist(),
                "status": "rejected_nonphysical_proposal",
                "reason": str(exc),
                "physical_engine_dispatched": False,
            }
            self.rejected.append(rejected)
            _write(
                self.destination
                / f"rejected-proposal-{len(self.rejected) - 1:04d}.json",
                rejected,
            )
            raise _StopOptimizationError(
                "The optimizer proposed invalid/coincident nuclei."
            ) from exc

    def evaluate(
        self, geometry: np.ndarray, *, final: bool = False
    ) -> ConstrainedEvaluation:
        molecule = self.candidate(geometry)
        key = digest(
            {
                "molecule": molecule.model_dump(mode="json"),
                "recipe_sha256": self.profile["recipe_sha256"],
            }
        )
        if not final and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        maximum = (
            self.specification.max_physical_calls
            if final
            else self.specification.max_physical_calls - 1
        )
        if len(self.evaluations) >= maximum:
            raise _StopOptimizationError(
                "Physical-call budget exhausted; final verification remains reserved."
            )
        remaining = self.resources.wall_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise _StopOptimizationError(
                "Actual overall constrained-optimization wall budget exhausted."
            )
        try:
            memory = psutil.virtual_memory().available / 1024**2
        except (OSError, psutil.Error):
            raise _StopOptimizationError(
                "Host memory availability could not be observed."
            ) from None
        if memory < 256:
            raise _StopOptimizationError(
                "Actual available host memory is below the engine startup reserve."
            )
        sequence = len(self.evaluations)
        root = self.destination / "evaluations" / f"evaluation-{sequence:04d}"
        root.mkdir(mode=0o700)
        backend_request = {
            "molecule": molecule.model_dump(mode="json"),
            "method": {
                "name": "hf",
                "basis": "sto-3g",
                "reference": "restricted",
                "frozen_core": False,
                "dispersion": None,
            },
            "properties": ["energy", "gradient", "hessian"]
            if final and self.specification.request_curvature
            else ["energy", "gradient"],
            "settings": {
                **self.profile["numerical"],
                "threads": self.resources.cores,
                "memory_mb": self.resources.memory_mb,
            },
        }
        _write(root / "backend-request.json", backend_request)
        started = time.monotonic()
        memory_samples: list[float] = []
        failure: str | None = None
        ceiling = min(float(self.specification.per_evaluation_wall_seconds), remaining)
        process: subprocess.Popen[bytes] | None = None
        observed_process_start: float | None = None
        try:
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _OWNED_WORKER_LAUNCHER,
                        str(os.getpid()),
                        repr(psutil.Process().create_time()),
                        str(root / "backend-request.json"),
                        str(root / "native"),
                        str(root / "owner-binding.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    observed_process_start = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    pass
                _write(
                    root / "parent-process-observation.json",
                    {
                        "schema_version": "cochem.torq.native-parent-observation/1",
                        "owner_pid": os.getpid(),
                        "owner_create_time": psutil.Process().create_time(),
                        "worker_pid": process.pid,
                        "worker_create_time": observed_process_start,
                        "identity_observation": "observed"
                        if observed_process_start is not None
                        else "unavailable",
                    },
                )
                while process.poll() is None:
                    if time.monotonic() - started > ceiling:
                        failure = (
                            "Actual native worker exceeded its reviewed wall ceiling."
                        )
                        _stop(process)
                        break
                    try:
                        observed = (
                            psutil.Process(process.pid).memory_info().rss / 1024**2
                        )
                        memory_samples.append(observed)
                        if observed > self.resources.memory_mb:
                            failure = (
                                "Actual sampled worker memory exceeded its ceiling."
                            )
                            _stop(process)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # Unavailable memory observations remain missing.
                    time.sleep(0.02)
                process.wait()
                binding = root / "owner-binding.json"
                _write(
                    root / "worker-process.json",
                    {
                        "schema_version": "cochem.torq.native-worker-wait/1",
                        "worker_pid": process.pid,
                        "worker_create_time": observed_process_start,
                        "wait_completed": True,
                        "returncode": process.returncode,
                        "owner_binding_sha256": sha256(binding.read_bytes()).hexdigest()
                        if binding.is_file()
                        else None,
                        "parent_observed_limit_failure": failure,
                    },
                )
                if process.returncode != 0 and failure is None:
                    failure = (
                        f"Native worker exited {process.returncode}; log retained."
                    )
        except BaseException:
            if process is not None and process.poll() is None:
                _stop(process)
            raise
        evidence = _native_scan_evidence(
            root / "native",
            ScanMolecule.model_validate(molecule.model_dump(mode="json")),
            execution_failure=failure,
        )
        available = evidence.status == "available"
        native_path = root / "native" / "manifest.json"
        record = ConstrainedEvaluation(
            sequence_index=sequence,
            purpose="independent_final_verification"
            if final
            else "optimizer_objective",
            molecule=molecule,
            status="available" if available else "failed",
            electronic=evidence.electronic if available else None,
            gradient_hartree_bohr=evidence.gradient if available else None,
            density=evidence.density if available else None,
            reason=(f"{failure} {evidence.reason}" if failure else evidence.reason)
            if not available
            else None,
            native_manifest_path=native_path.relative_to(self.destination).as_posix()
            if native_path.is_file()
            else None,
            native_manifest_sha256=evidence.native_manifest_sha256,
            wall_seconds=time.monotonic() - started,
            sampled_peak_memory_mb=max(memory_samples) if memory_samples else None,
        )
        self.evaluations.append(record)
        _write(root / "evaluation.json", record.model_dump(mode="json"))
        if not available:
            raise _StopOptimizationError(
                record.reason or "Actual native observation unavailable."
            )
        self.cache[key] = record
        return record


def execute_constrained_optimization(
    molecule: Molecule,
    specification: ConstrainedOptimizationSpec,
    destination: str | Path,
    *,
    resources: Resources,
    profile: dict[str, Any],
) -> ConstrainedOptimizationResult:
    """Execute bounded actual HF optimization and preserve every observation."""
    molecule = Molecule.model_validate(molecule.model_dump(mode="json"))
    specification = ConstrainedOptimizationSpec.model_validate(
        specification.model_dump(mode="json")
    )
    resources = Resources.model_validate(resources.model_dump(mode="json"))
    if molecule.multiplicity != 1 or molecule.atom_ids is None:
        raise ValueError(
            "Constrained HF requires a closed-shell state and stable atom IDs."
        )
    if (
        profile.get("id") != RECIPE_ID
        or canonical_json(profile) != canonical_json(get_profile(RECIPE_ID))
        or profile.get("engine") != "PySCF"
        or profile.get("method") != "hf"
        or profile.get("basis") != "sto-3g"
        or profile.get("dispersion") is not None
    ):
        raise ValueError(
            "Use the exact registered HF/STO-3G constrained optimization profile."
        )
    optimizer_identity = _scipy_fingerprint()
    if (
        profile.get("optimizer_version") != optimizer_identity["version"]
        or profile.get("optimizer") != "SciPy SLSQP"
    ):
        raise ValueError(
            "The registered constrained optimizer differs from actual SciPy SLSQP."
        )
    if max(specification.movable_atom_indices) >= len(molecule.symbols) or any(
        max(item.coordinate.atom_indices) >= len(molecule.symbols)
        for item in specification.constraints
    ):
        raise ValueError(
            "Constrained/movable atom indices exceed the immutable molecule order."
        )
    reference = real_values(molecule.geometry_bohr)
    if np.linalg.matrix_rank(
        _jacobian(reference, specification), tol=specification.rank_tolerance
    ) != len(specification.constraints):
        raise ValueError("Initial constraints are singular or dependent.")
    root = Path(destination).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError(
            "Constrained optimization workspaces cannot traverse symlinks."
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new empty constrained optimization workspace.")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "evaluations").mkdir(mode=0o700)
    _write(
        root / "request.json",
        {
            "molecule": molecule.model_dump(mode="json"),
            "specification": specification.model_dump(mode="json"),
            "resources": resources.model_dump(mode="json"),
            "profile": profile,
        },
    )
    evaluator = _Evaluator(molecule, specification, root, resources, profile)
    rows = list(specification.movable_atom_indices)
    origin = reference[rows].reshape(-1)

    def geometry(vector: np.ndarray) -> np.ndarray:
        value = reference.copy()
        value[rows] = np.asarray(vector).reshape(-1, 3)
        return cast(np.ndarray, value)

    def objective(vector: np.ndarray) -> float:
        observed = evaluator.evaluate(geometry(vector))
        if observed.electronic is None:
            raise _StopOptimizationError("Actual objective energy unavailable.")
        return observed.electronic.energy_hartree

    def gradient(vector: np.ndarray) -> np.ndarray:
        observed = evaluator.evaluate(geometry(vector))
        if observed.gradient_hartree_bohr is None:
            raise _StopOptimizationError("Actual objective gradient unavailable.")
        return cast(
            np.ndarray,
            np.asarray(observed.gradient_hartree_bohr).reshape(-1)[
                _columns(specification)
            ],
        )

    final: ConstrainedEvaluation | None = None
    diagnostics: dict[str, Any] | None = None
    optimizer_outcome: dict[str, Any] = {
        **optimizer_identity,
        "method": "SLSQP",
        "objective": "actual absolute electronic energy hartree",
        "energy_tolerance_hartree": specification.optimizer_energy_tolerance_hartree,
        "source_file_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    reason: str | None = None
    native: dict[str, Any] | None = None
    curvature = _unavailable_curvature(
        "Constrained curvature was not requested or final stationarity is unavailable."
    )
    try:
        bound = specification.max_cartesian_displacement_bohr
        result = minimize(
            objective,
            origin,
            jac=gradient,
            constraints={
                "type": "eq",
                "fun": lambda vector: _residual(geometry(vector), specification),
                "jac": lambda vector: _jacobian(geometry(vector), specification),
            },
            bounds=[(float(value - bound), float(value + bound)) for value in origin],
            method="SLSQP",
            options={
                "ftol": specification.optimizer_energy_tolerance_hartree,
                "maxiter": specification.max_optimizer_iterations,
            },
        )
        optimizer_outcome.update(
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            optimizer_function_calls=int(result.nfev),
            optimizer_gradient_calls=int(result.njev),
            final_proposed_geometry_bohr=geometry(result.x).tolist(),
        )
        if not result.success:
            raise _StopOptimizationError(
                f"Constrained energy optimizer did not converge: {result.message}"
            )
        final = evaluator.evaluate(geometry(result.x), final=True)
        actual_geometry = real_values(final.molecule.geometry_bohr)
        actual_gradient = real_values(final.gradient_hartree_bohr)
        residuals = _residual(actual_geometry, specification)
        if (
            float(np.max(np.abs(residuals)))
            > specification.constraint_tolerance_bohr_or_radian
        ):
            raise _StopOptimizationError(
                "Actual final geometry violates the declared coordinate tolerance."
            )
        diagnostics = _kkt(actual_geometry, actual_gradient, specification)
        tangent_gradient = diagnostics["tangent_gradient"]
        if (
            len(tangent_gradient)
            and float(np.max(np.abs(tangent_gradient)))
            > specification.tangent_gradient_tolerance_hartree_bohr
        ):
            raise _StopOptimizationError(
                "Final actual tangent gradient failed its declared tolerance."
            )
        if np.any(np.abs(result.x - origin) >= bound - 1e-8):
            raise _StopOptimizationError(
                "Cartesian search bound reached; stationarity remains unqualified."
            )
        if final.native_manifest_path is None:
            raise _StopOptimizationError("Actual final native manifest is unavailable.")
        native, native_sha = _verify_native((root / final.native_manifest_path).parent)
        if native_sha != final.native_manifest_sha256:
            raise _StopOptimizationError(
                "Final native bytes changed during independent readback."
            )
        if specification.request_curvature:
            curvature = _curvature(native, actual_geometry, diagnostics, specification)
    except (_StopOptimizationError, ValueError, np.linalg.LinAlgError) as exc:
        reason = str(exc)
    qualified = (
        reason is None
        and final is not None
        and diagnostics is not None
        and native is not None
    )
    flags = [
        "electronic_model_error_uncalibrated",
        "constrained_stationarity_only",
        "no_global_search_completeness",
        "fixed_atom_forces_are_not_stationarity_conditions",
        "actual_parent_wall_and_sampled_worker_memory_exact_cpu_unavailable",
    ]
    if qualified and curvature.status == "available":
        flags += [
            "coordinate_hessians_numerical_two_step",
            "unweighted_cartesian_tangent_curvature",
        ]
    payload = {
        "recipe": profile["id"],
        "recipe_sha256": profile["recipe_sha256"],
        "specification": specification.model_dump(mode="json"),
        "initial_molecule": molecule.model_dump(mode="json"),
        "final_molecule": final.molecule.model_dump(mode="json")
        if qualified and final is not None
        else None,
        "status": "available"
        if qualified
        else "partial"
        if any(item.status == "available" for item in evaluator.evaluations)
        else "failed",
        "reason": None
        if qualified
        else reason or "Constrained stationarity could not be qualified.",
        "stationary_character": "constrained_stationary" if qualified else None,
        "electronic": final.electronic.model_dump(mode="json")
        if qualified and final is not None and final.electronic is not None
        else None,
        "gradient_hartree_bohr": final.gradient_hartree_bohr
        if qualified and final is not None
        else None,
        "constraint_residuals_bohr_or_radian": _residual(
            real_values(final.molecule.geometry_bohr), specification
        ).tolist()
        if qualified and final is not None
        else None,
        "lagrange_multipliers": diagnostics["multipliers"].tolist()
        if qualified and diagnostics is not None
        else None,
        "lagrange_multiplier_units": [
            "hartree/bohr" if item.coordinate.kind == "bond" else "hartree/radian"
            for item in specification.constraints
        ],
        "tangent_gradient_hartree_bohr": diagnostics["tangent_gradient"].tolist()
        if qualified and diagnostics is not None
        else None,
        "constraint_rank": diagnostics["rank"]
        if qualified and diagnostics is not None
        else None,
        "tangent_dimension": diagnostics["tangent"].shape[1]
        if qualified and diagnostics is not None
        else None,
        "evaluations": [item.model_dump(mode="json") for item in evaluator.evaluations],
        "physical_call_count": len(evaluator.evaluations),
        "exact_geometry_cache_hits": evaluator.cache_hits,
        "optimizer": optimizer_outcome,
        "curvature": curvature.model_dump(mode="json"),
        "native_result": native if qualified else None,
        "final_native_manifest_path": final.native_manifest_path
        if qualified and final is not None
        else None,
        "final_native_manifest_sha256": final.native_manifest_sha256
        if qualified and final is not None
        else None,
        "wall_seconds": time.monotonic() - evaluator.started,
        "quality_flags": flags,
        "rejected_proposals": evaluator.rejected,
    }
    checked = ConstrainedOptimizationResult.model_validate(payload)
    _write(root / "result.json", checked.model_dump(mode="json"))
    return checked


def _native_worker(request_path: str, native_path: str) -> None:
    from .pyscf_backend import PySCFBackend

    PySCFBackend().evaluate(read_json(request_path), native_path)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "_native_worker":
        raise SystemExit(
            "Use only the actual constrained optimization native worker entry."
        )
    _native_worker(sys.argv[2], sys.argv[3])
