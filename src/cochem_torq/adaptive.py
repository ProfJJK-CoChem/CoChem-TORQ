"""Bounded physical-anchor PES acquisition and ledger-linked candidates.

The first profile uses one nonperiodic coordinate, deterministic maximin
acquisition, and independent numerical-interpolation challenges. It supplies no
trained ML model, calibrated error interval, global completeness proof or TS
claim. Every energy observation comes from the approved scan executor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from cochem.orchestration.campaign import AuthorityError, BudgetExceededError

from .candidate_ledger import CandidateLedger
from .domain import CalculationRequest, Contract, Molecule, digest, read_json
from .registry import get_profile
from .scan import ApprovedScanExecutor, ScanMolecule, ScanPointResult, ScanSurface
from .scientific_contracts import ScientificGoal
from .service import validate_approved_plan

StopReason = Literal[
    "criteria_met", "budget_exhausted", "insufficient_coverage", "failed"
]
CandidateKind = Literal[
    "lowest_observed_point", "sampled_neighbor_minimum", "sampled_neighbor_maximum"
]


class AdaptiveScanPlan(Contract):
    schema_version: Literal["cochem.torq.adaptive-scan-plan/1"] = (
        "cochem.torq.adaptive-scan-plan/1"
    )
    scan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    seed_indices: tuple[StrictInt, ...] = Field(min_length=2)
    held_out_indices: tuple[StrictInt, ...] = Field(min_length=1)
    max_normalized_design_distance: StrictFloat = Field(gt=0, le=1)
    energy_interpolation_tolerance_hartree: StrictFloat = Field(gt=0)
    scientific_goal: ScientificGoal
    candidate_refinement_recipe: str = "hf-sto-3g-education"
    acquisition: Literal["deterministic_maximin_coordinate_distance/1"] = (
        "deterministic_maximin_coordinate_distance/1"
    )
    interpolation: Literal["piecewise_linear_numerical_estimate/1"] = (
        "piecewise_linear_numerical_estimate/1"
    )
    surrogate_model: None = None
    challenge_policy: Literal["fresh_endpoints_and_frozen_interior_once/1"] = (
        "fresh_endpoints_and_frozen_interior_once/1"
    )
    coverage_scope: Literal["declared_finite_design_only"] = (
        "declared_finite_design_only"
    )

    @model_validator(mode="after")
    def declared_policy(self) -> Self:
        for values in (self.seed_indices, self.held_out_indices):
            if len(set(values)) != len(values) or min(values) < 0:
                raise ValueError(
                    "Seed/held-out indices must be unique nonnegative indices."
                )
        if set(self.seed_indices) & set(self.held_out_indices):
            raise ValueError("Held-out interior points cannot be seed/training points.")
        if self.scientific_goal.reference_policy != "de_novo_no_calibration":
            raise ValueError(
                "This within-scan numerical profile does not implement "
                "independent molecular benchmarks or experimental inference."
            )
        targets = [
            target
            for target in self.scientific_goal.error_targets
            if target.observable == "numerical_interpolation_energy"
        ]
        if (
            len(targets) != 1
            or targets[0].target_kind != "maximum_absolute_error"
            or targets[0].unit != "hartree"
            or targets[0].maximum != self.energy_interpolation_tolerance_hartree
        ):
            raise ValueError(
                "The goal must declare the exact numerical-interpolation energy "
                "tolerance; this is not chemical accuracy."
            )
        return self


class InterpolationChallenge(Contract):
    sample_index: StrictInt = Field(ge=0)
    point_id: str
    coordinate_value: StrictFloat
    challenge_kind: Literal["boundary_repeat", "held_out_interior"]
    observed_energy_hartree: StrictFloat | None
    numerical_estimate_hartree: StrictFloat | None
    absolute_residual_hartree: StrictFloat | None
    parent_anchor_point_ids: tuple[str, ...]
    status: Literal["available", "unavailable"]
    absence_reason: str | None = None
    estimate_kind: Literal["numerical_interpolation_not_physical_observation"] = (
        "numerical_interpolation_not_physical_observation"
    )
    accuracy_qualified: Literal[False] = False

    @model_validator(mode="after")
    def truthful(self) -> Self:
        values = (
            self.observed_energy_hartree,
            self.numerical_estimate_hartree,
            self.absolute_residual_hartree,
        )
        if self.status == "available":
            if (
                any(value is None for value in values)
                or self.absence_reason is not None
            ):
                raise ValueError(
                    "An available challenge requires real observation and numerical "
                    "estimate."
                )
            if (
                self.absolute_residual_hartree is not None
                and self.absolute_residual_hartree < 0
            ):
                raise ValueError("Challenge residual cannot be negative.")
            if (
                self.observed_energy_hartree is None
                or self.numerical_estimate_hartree is None
                or self.absolute_residual_hartree
                != abs(self.observed_energy_hartree - self.numerical_estimate_hartree)
                or len(self.parent_anchor_point_ids) < 2
                or len(set(self.parent_anchor_point_ids))
                != len(self.parent_anchor_point_ids)
            ):
                raise ValueError(
                    "Residual and distinct actual anchor identities must agree."
                )
        elif self.absolute_residual_hartree is not None or not self.absence_reason:
            raise ValueError("Unavailable challenge residual is absent with a reason.")
        return self


class StationaryCandidate(Contract):
    candidate_id: UUID
    ledger_revision: StrictInt = Field(ge=1)
    point_id: str
    sample_index: StrictInt
    molecule: ScanMolecule
    native_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    constraints_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_kind: CandidateKind
    native_manifest_path: str
    point_result_path: str
    point_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_request_json: str
    candidate_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    geometry_status: Literal["candidate"] = "candidate"
    verified_stationary: Literal[False] = False
    transition_state_established: Literal[False] = False
    constraint_release_requires_review: Literal[True] = True
    review_status: Literal["needs_review"] = "needs_review"

    @model_validator(mode="after")
    def request_identity(self) -> Self:
        request = CalculationRequest.model_validate_json(self.candidate_request_json)
        if self.candidate_request_sha256 != digest(
            request.model_dump(mode="json")
        ) or request.molecule.model_dump(mode="json") != self.molecule.model_dump(
            mode="json"
        ):
            raise ValueError(
                "Candidate request must bind its exact molecule and recorded identity."
            )
        origin = request.source_provenance.get("adaptive_candidate")
        if not isinstance(origin, dict) or any(
            origin.get(field) != getattr(self, field)
            for field in (
                "scan_sha256",
                "constraints_sha256",
                "point_id",
                "sample_index",
                "native_manifest_sha256",
                "native_manifest_path",
                "point_result_path",
                "point_result_sha256",
                "candidate_kind",
            )
        ):
            raise ValueError(
                "Candidate fields must bind their immutable request origin."
            )
        return self

    @property
    def candidate_request(self) -> CalculationRequest:
        return CalculationRequest.model_validate_json(self.candidate_request_json)


class StationaryVerification(Contract):
    original_candidate_id: UUID
    verified_result_candidate_id: UUID
    verified_result_ledger_revision: StrictInt
    classification: Literal["verified_minimum"]
    target_recipe: str
    result_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_constraints_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    approval_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verified_stationary: Literal[True] = True
    transition_state_established: Literal[False] = False
    accuracy_qualified: Literal[False] = False


class AdaptiveScanResult(Contract):
    schema_version: Literal["cochem.torq.adaptive-scan-result/1"] = (
        "cochem.torq.adaptive-scan-result/1"
    )
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    plan: AdaptiveScanPlan
    scan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    stop_reason: StopReason
    stop_detail: str
    physical_calls: StrictInt = Field(ge=0)
    maximum_normalized_design_distance: StrictFloat | None
    anchor_point_ids: tuple[str, ...]
    challenges: tuple[InterpolationChallenge, ...]
    candidates: tuple[StationaryCandidate, ...]
    surface: ScanSurface
    review_required: bool = False
    numerical_criteria_scope: Literal[
        "finite_design_spacing_and_inspected_challenge_residuals_only"
    ] = "finite_design_spacing_and_inspected_challenge_residuals_only"
    surrogate_model: None = None
    automatic_pruning_enabled: Literal[False] = False
    search_completeness: Literal["not_established"] = "not_established"
    accuracy_qualified: Literal[False] = False
    limits: tuple[str, ...] = (
        "No bound on unsampled between-node behavior or missed wells.",
        "The within-scan challenge is not a family-held-out chemical benchmark.",
        "No calibrated error probabilities, intervals or experimental accuracy.",
        "Grid extrema require separate approved stationary-point calculations.",
        "No automatic change of electronic state, method, domain or constraints.",
    )

    @model_validator(mode="after")
    def actual_observations(self) -> Self:
        if (
            self.plan_sha256 != digest(self.plan.model_dump(mode="json"))
            or self.scan_sha256 != self.plan.scan_sha256
            or self.scan_sha256 != self.surface.scan_sha256
            or self.physical_calls != self.surface.physical_call_count
        ):
            raise ValueError("Adaptive summary must bind its actual plan and surface.")
        points = {point.point_id: point for point in self.surface.points}
        if len(set(self.anchor_point_ids)) != len(self.anchor_point_ids):
            raise ValueError("Anchor identities must be distinct observations.")
        for point_id in self.anchor_point_ids:
            point = points.get(point_id)
            if (
                point is None
                or point.status != "available"
                or point.purpose
                not in {
                    "adaptive_seed",
                    "adaptive_acquisition",
                }
            ):
                raise ValueError(
                    "Training anchors must be actual successful acquisitions."
                )
        distance = maximum_design_distance(
            [row[0] for row in self.surface.scan.grid],
            [points[point_id].sample_index for point_id in self.anchor_point_ids],
        )
        if distance != self.maximum_normalized_design_distance:
            raise ValueError(
                "Reported design distance must equal the actual anchor spacing."
            )
        for candidate in self.candidates:
            point = points.get(candidate.point_id)
            if (
                point is None
                or candidate.point_id not in self.anchor_point_ids
                or candidate.sample_index != point.sample_index
                or candidate.molecule != point.molecule
                or candidate.native_manifest_sha256 != point.native_manifest_sha256
                or candidate.scan_sha256 != self.scan_sha256
                or candidate.constraints_sha256
                != digest(self.surface.scan.model_dump(mode="json"))
            ):
                raise ValueError(
                    "Candidate summaries must bind actual training observations."
                )
        for challenge in self.challenges:
            point = points.get(challenge.point_id)
            if (
                point is None
                or point.purpose != "challenge"
                or point.sample_index != challenge.sample_index
                or point.coordinate_values != (challenge.coordinate_value,)
                or point.energy_hartree != challenge.observed_energy_hartree
                or (challenge.status == "available") != (point.status == "available")
                or set(challenge.parent_anchor_point_ids) - set(self.anchor_point_ids)
            ):
                raise ValueError(
                    "Challenge summaries must bind their physical observations."
                )
        if self.stop_reason == "criteria_met" and (
            self.maximum_normalized_design_distance is None
            or self.maximum_normalized_design_distance
            > self.plan.max_normalized_design_distance
            or len(self.challenges) != len(self.plan.held_out_indices) + 2
            or tuple(challenge.sample_index for challenge in self.challenges)
            != (0, len(self.surface.scan.grid) - 1, *self.plan.held_out_indices)
            or len({challenge.point_id for challenge in self.challenges})
            != len(self.challenges)
            or any(
                challenge.status != "available"
                or challenge.absolute_residual_hartree is None
                or challenge.absolute_residual_hartree
                > self.plan.energy_interpolation_tolerance_hartree
                for challenge in self.challenges
            )
            or any(comparison.inconsistent for comparison in self.surface.comparisons)
        ):
            raise ValueError("Criteria require every predeclared numerical challenge.")
        return self


def _finite_axis(values: Sequence[float]) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError("Coordinate values must be real.")
    result = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.diff(result)
        span = result[-1] - result[0] if result.ndim == 1 and len(result) else np.nan
    if (
        result.ndim != 1
        or len(result) < 2
        or not np.isfinite(result).all()
        or not np.isfinite(differences).all()
        or not np.isfinite(span)
        or np.any(differences <= 0)
    ):
        raise ValueError(
            "A finite strictly increasing one-dimensional design is required."
        )
    return result


def _validate_indices(indices: Sequence[int], size: int) -> None:
    if any(type(index) is not int or not 0 <= index < size for index in indices):
        raise ValueError("Indices must belong to the finite design.")
    if len(set(indices)) != len(indices):
        raise ValueError("Design indices must be distinct.")


def maximum_design_distance(
    values: Sequence[float], anchor_indices: Sequence[int]
) -> float | None:
    """Exact normalized distance of the declared finite design to observed anchors.

    This geometric statistic is not prediction uncertainty or continuous coverage.
    """
    axis = _finite_axis(values)
    if not anchor_indices:
        return None
    _validate_indices(anchor_indices, len(axis))
    distances = np.abs(axis[:, None] - axis[list(anchor_indices)][None, :]) / (
        axis[-1] - axis[0]
    )
    return float(np.max(np.min(distances, axis=1)))


def next_maximin_index(
    values: Sequence[float],
    anchor_indices: Sequence[int],
    held_out_indices: Sequence[int],
) -> int | None:
    """Choose a remaining training point deterministically, never pruning the design."""
    axis = _finite_axis(values)
    maximum_design_distance(values, anchor_indices)
    _validate_indices(held_out_indices, len(axis))
    if set(anchor_indices) & set(held_out_indices):
        raise ValueError("Inspected challenge points cannot become training anchors.")
    excluded = set(anchor_indices) | set(held_out_indices)
    remaining = [index for index in range(len(axis)) if index not in excluded]
    if not remaining:
        return None
    if not anchor_indices:
        return remaining[0]
    return max(
        remaining,
        key=lambda index: (
            min(abs(float(axis[index] - axis[parent])) for parent in anchor_indices),
            -index,
        ),
    )


def interpolate_energy(
    anchors: Sequence[tuple[float, float]], coordinate: float
) -> float:
    """An explicitly numerical estimate from declared mathematical/physical inputs."""
    if len(anchors) < 2:
        raise ValueError(
            "Interpolation requires two actual/declaratively mathematical anchors."
        )
    if np.iscomplexobj(anchors) or np.iscomplexobj(coordinate):
        raise ValueError("Interpolation requires real inputs.")
    values = _finite_axis([item[0] for item in anchors])
    energies = np.asarray([item[1] for item in anchors], dtype=np.float64)
    if (
        not np.isfinite(energies).all()
        or not np.isfinite(coordinate)
        or coordinate < values[0]
        or coordinate > values[-1]
    ):
        raise ValueError(
            "Real finite bracketed inputs are required; no extrapolation or "
            "missing-energy fill."
        )
    result = float(np.interp(coordinate, values, energies))
    if not np.isfinite(result):
        raise ValueError("Interpolation exceeded finite numeric range.")
    return result


def _physical_energy(point: ScanPointResult) -> float:
    if point.status != "available" or point.energy_hartree is None:
        raise ValueError("An available physical energy is required.")
    return point.energy_hartree


def _candidate_nodes(
    anchors: Sequence[ScanPointResult],
) -> list[tuple[ScanPointResult, CandidateKind]]:
    available = sorted(
        [
            point
            for point in anchors
            if point.status == "available" and point.energy_hartree is not None
        ],
        key=lambda point: point.sample_index,
    )
    if not available:
        return []
    lowest = min(available, key=_physical_energy)
    chosen: dict[str, tuple[ScanPointResult, CandidateKind]] = {
        lowest.point_id: (lowest, "lowest_observed_point")
    }
    for left, middle, right in zip(available, available[1:], available[2:]):
        if (
            middle.energy_hartree is None
            or left.energy_hartree is None
            or right.energy_hartree is None
        ):
            raise ValueError(
                "Candidate extraction requires available physical energies."
            )
        if middle.energy_hartree < min(left.energy_hartree, right.energy_hartree):
            chosen[middle.point_id] = (middle, "sampled_neighbor_minimum")
        elif middle.energy_hartree > max(left.energy_hartree, right.energy_hartree):
            chosen[middle.point_id] = (middle, "sampled_neighbor_maximum")
    return list(chosen.values())


def _register_candidates(
    executor: ApprovedScanExecutor,
    plan: AdaptiveScanPlan,
    anchors: Sequence[ScanPointResult],
    ledger: CandidateLedger,
    actor: str,
) -> tuple[StationaryCandidate, ...]:
    records = []
    constraints_sha256 = digest(executor.scan.model_dump(mode="json"))
    from .artifacts import file_digest

    for point, kind in _candidate_nodes(anchors):
        if point.native_manifest_sha256 is None or point.native_manifest_path is None:
            raise ValueError(
                "A physical candidate requires its retained native manifest."
            )
        manifest_path = executor.workspace / point.native_manifest_path
        point_result_path = manifest_path.parent.parent / "point-result.json"
        point_result_sha256 = file_digest(point_result_path)
        request = CalculationRequest(
            molecule=Molecule.model_validate(point.molecule.model_dump(mode="json")),
            recipe=plan.candidate_refinement_recipe,
            products=["geometry", "harmonic"],
            resources=executor.request.resources,
            source_provenance={
                "adaptive_candidate": {
                    "scan_sha256": plan.scan_sha256,
                    "constraints_sha256": constraints_sha256,
                    "point_id": point.point_id,
                    "sample_index": point.sample_index,
                    "native_manifest_sha256": point.native_manifest_sha256,
                    "native_manifest_path": str(manifest_path),
                    "point_result_path": str(point_result_path),
                    "point_result_sha256": point_result_sha256,
                    "candidate_kind": kind,
                    "verified_stationary": False,
                    "constraint_release_review": "required",
                }
            },
        )
        candidate_id = uuid5(
            NAMESPACE_URL,
            f"TORQ-adaptive:{digest(plan.model_dump(mode='json'))}:{point.point_id}",
        )
        registered = ledger.register_request(
            request.model_dump(mode="json"),
            actor=actor,
            reason=(
                "Observed finite-scan candidate only; target optimization and "
                "Hessian remain uncomputed."
            ),
            candidate_id=candidate_id,
        )
        quarantined = ledger.quarantine(
            candidate_id,
            expected_revision=registered["revision"],
            actor=actor,
            reason=(
                "Releasing scan constraints and checking stationarity requires a "
                "reviewed target calculation."
            ),
        )
        records.append(
            StationaryCandidate(
                candidate_id=candidate_id,
                ledger_revision=quarantined["revision"],
                point_id=point.point_id,
                sample_index=point.sample_index,
                molecule=point.molecule,
                native_manifest_sha256=point.native_manifest_sha256,
                scan_sha256=plan.scan_sha256,
                constraints_sha256=constraints_sha256,
                candidate_kind=kind,
                native_manifest_path=str(manifest_path),
                point_result_path=str(point_result_path),
                point_result_sha256=point_result_sha256,
                candidate_request_json=request.model_dump_json(),
                candidate_request_sha256=digest(request.model_dump(mode="json")),
            )
        )
    return tuple(records)


def run_adaptive_scan(
    executor: ApprovedScanExecutor,
    plan: AdaptiveScanPlan,
    *,
    ledger: CandidateLedger,
    actor: str,
) -> AdaptiveScanResult:
    """Run the approved finite loop; scalar callbacks cannot supply physical anchors."""
    if not isinstance(executor, ApprovedScanExecutor):
        raise TypeError("Use the real approval/coordinator-backed scan executor.")
    if executor.points:
        raise ValueError(
            "Adaptive execution must begin with a fresh executor; inspected "
            "challenge reuse is forbidden."
        )
    if digest(
        executor.scan.model_dump(mode="json")
    ) != plan.scan_sha256 or executor.request.source_provenance.get(
        "adaptive_scan"
    ) != plan.model_dump(mode="json"):
        raise ValueError(
            "The immutable approval must bind this exact scan and adaptive policy."
        )
    if not actor.strip():
        raise ValueError("A candidate ledger actor is required.")
    if (
        executor.request.scientific_goal is None
        or executor.request.scientific_goal != plan.scientific_goal
    ):
        raise ValueError(
            "The approved request and adaptive observable goal must agree."
        )
    if (
        len(executor.scan.coordinates) != 1
        or executor.scan.coordinates[0].domain.periodic
        or executor.scan.sampling_strategy != "bounded_adaptive"
    ):
        raise ValueError("Only one-dimensional nonperiodic adaptation is implemented.")
    target = get_profile(plan.candidate_refinement_recipe)
    source = get_profile(executor.request.recipe)
    if any(
        target.get(field) != source.get(field)
        for field in ("engine", "method", "basis", "reference", "dispersion")
    ):
        raise ValueError(
            "A different candidate-refinement physical model requires "
            "a separately approved scope change."
        )
    axis = _finite_axis([row[0] for row in executor.scan.grid])
    if (
        max((*plan.seed_indices, *plan.held_out_indices)) >= len(axis)
        or {0, len(axis) - 1} - set(plan.seed_indices)
        or any(index in {0, len(axis) - 1} for index in plan.held_out_indices)
    ):
        raise ValueError(
            "Seed endpoints and disjoint frozen interior challenge indices "
            "must belong to the design."
        )
    anchors: list[ScanPointResult] = []
    challenges: list[InterpolationChallenge] = []
    stop: StopReason = "failed"
    detail = "No valid seed design was completed."
    review_required = False
    try:
        for index in plan.seed_indices:
            point = executor.evaluate(
                index,
                purpose="adaptive_seed",
                parent_point_ids=tuple(item.point_id for item in anchors),
            )
            if point.status != "available" or point.energy_hartree is None:
                detail = "A real seed calculation failed or remains unavailable."
                break
            anchors.append(point)
        else:
            while True:
                distance = maximum_design_distance(
                    axis.tolist(), [point.sample_index for point in anchors]
                )
                if (
                    distance is not None
                    and distance <= plan.max_normalized_design_distance
                ):
                    break
                acquired_index = next_maximin_index(
                    axis.tolist(),
                    [point.sample_index for point in anchors],
                    plan.held_out_indices,
                )
                if acquired_index is None:
                    stop, detail = (
                        "insufficient_coverage",
                        "The finite training design cannot meet the declared spacing "
                        "criterion; request a reviewed denser design.",
                    )
                    break
                point = executor.evaluate(
                    acquired_index,
                    purpose="adaptive_acquisition",
                    parent_point_ids=tuple(item.point_id for item in anchors),
                )
                if point.status != "available" or point.energy_hartree is None:
                    detail = (
                        "A real acquisition calculation failed or remains unavailable."
                    )
                    break
                anchors.append(point)
            else:
                raise RuntimeError("Unreachable adaptive loop state.")
            distance = maximum_design_distance(
                axis.tolist(), [point.sample_index for point in anchors]
            )
            if distance is not None and distance <= plan.max_normalized_design_distance:
                ordered = sorted(anchors, key=lambda point: point.coordinate_values[0])
                energy_pairs = [
                    (float(point.coordinate_values[0]), float(point.energy_hartree))
                    for point in ordered
                    if point.energy_hartree is not None
                ]
                for index in (0, len(axis) - 1, *plan.held_out_indices):
                    estimate = interpolate_energy(energy_pairs, float(axis[index]))
                    point = executor.evaluate(
                        index,
                        purpose="challenge",
                        parent_point_ids=tuple(item.point_id for item in ordered),
                    )
                    if point.status != "available" or point.energy_hartree is None:
                        challenges.append(
                            InterpolationChallenge(
                                sample_index=index,
                                point_id=point.point_id,
                                coordinate_value=float(axis[index]),
                                challenge_kind="boundary_repeat"
                                if index in {0, len(axis) - 1}
                                else "held_out_interior",
                                observed_energy_hartree=None,
                                numerical_estimate_hartree=estimate,
                                absolute_residual_hartree=None,
                                parent_anchor_point_ids=tuple(
                                    item.point_id for item in ordered
                                ),
                                status="unavailable",
                                absence_reason=(
                                    "The real challenge calculation failed or is "
                                    "unavailable."
                                ),
                            )
                        )
                        detail = (
                            "A genuine final challenge failed; numerical criteria "
                            "cannot be "
                            "established."
                        )
                        break
                    residual = abs(point.energy_hartree - estimate)
                    challenges.append(
                        InterpolationChallenge(
                            sample_index=index,
                            point_id=point.point_id,
                            coordinate_value=float(axis[index]),
                            challenge_kind="boundary_repeat"
                            if index in {0, len(axis) - 1}
                            else "held_out_interior",
                            observed_energy_hartree=point.energy_hartree,
                            numerical_estimate_hartree=estimate,
                            absolute_residual_hartree=residual,
                            parent_anchor_point_ids=tuple(
                                item.point_id for item in ordered
                            ),
                            status="available",
                        )
                    )
                else:
                    residuals = [
                        challenge.absolute_residual_hartree for challenge in challenges
                    ]
                    if all(
                        value is not None
                        and value <= plan.energy_interpolation_tolerance_hartree
                        for value in residuals
                    ):
                        stop, detail = (
                            "criteria_met",
                            "Declared finite-design spacing and newly calculated "
                            "challenge "
                            "residuals passed; no continuous/global or "
                            "chemical-accuracy "
                            "claim.",
                        )
                    else:
                        stop, detail = (
                            "insufficient_coverage",
                            "Independent numerical-interpolation challenge residual "
                            "exceeded "
                            "its predeclared tolerance; no adaptation using this "
                            "inspected "
                            "validation set.",
                        )
    except BudgetExceededError as error:
        stop, detail = "budget_exhausted", str(error)
    except AuthorityError as error:
        stop, detail, review_required = "failed", str(error), True
    surface = executor.finish()
    if stop == "criteria_met" and any(
        comparison.inconsistent for comparison in surface.comparisons
    ):
        stop, detail = (
            "failed",
            "Actual independent endpoint energy or density rechecks failed "
            "their approved comparison tolerance.",
        )
    return AdaptiveScanResult(
        plan_sha256=digest(plan.model_dump(mode="json")),
        plan=plan,
        scan_sha256=plan.scan_sha256,
        stop_reason=stop,
        stop_detail=detail,
        physical_calls=len(executor.points),
        maximum_normalized_design_distance=maximum_design_distance(
            axis.tolist(), [point.sample_index for point in anchors]
        ),
        anchor_point_ids=tuple(point.point_id for point in anchors),
        challenges=tuple(challenges),
        candidates=_register_candidates(executor, plan, anchors, ledger, actor),
        surface=surface,
        review_required=review_required,
    )


def validate_candidate_refinement(
    candidate: StationaryCandidate,
    *,
    ledger: CandidateLedger,
    expected_revision: int,
    approved_plan: dict[str, Any],
) -> CalculationRequest:
    """Authenticate ledger origin and exact target approval before physical work."""
    from .artifacts import file_digest
    from .scan import _verify_native

    candidate = StationaryCandidate.model_validate(candidate.model_dump(mode="json"))
    record = ledger.inspect(candidate.candidate_id)
    if (
        record["revision"] != expected_revision
        or record["selection_state"] != "retained"
    ):
        raise ValueError(
            "Inspect and explicitly retain the current candidate revision "
            "before refinement verification."
        )
    original_request = candidate.candidate_request.model_dump(mode="json")
    if (
        record["request"] != original_request
        or record["origin"] != "supplied_input"
        or record["geometry_status"] != "input"
        or record["molecule"] != candidate.molecule.model_dump(mode="json")
        or digest(record["request"]) != candidate.candidate_request_sha256
    ):
        raise ValueError(
            "Candidate must match its immutable ledger request and origin."
        )
    native_path = Path(candidate.native_manifest_path)
    point_path = Path(candidate.point_result_path)
    for path in (native_path, point_path):
        if (
            not path.is_absolute()
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
        ):
            raise ValueError(
                "Candidate native evidence must use original absolute paths."
            )
    if (
        file_digest(native_path) != candidate.native_manifest_sha256
        or file_digest(point_path) != candidate.point_result_sha256
        or native_path.parent.parent / "point-result.json" != point_path
    ):
        raise ValueError(
            "Candidate original physical evidence failed hash/path validation."
        )
    point = ScanPointResult.model_validate(read_json(point_path))
    native, native_sha = _verify_native(native_path.parent)
    source_recipe = get_profile(point.recipe)
    target_recipe = get_profile(candidate.candidate_request.recipe)
    if (
        point.point_id != candidate.point_id
        or point.sample_index != candidate.sample_index
        or point.scan_sha256 != candidate.scan_sha256
        or point.molecule != candidate.molecule
        or point.status != "available"
        or native.get("status") != "complete"
        or native_sha != candidate.native_manifest_sha256
        or point.native_manifest_sha256 != candidate.native_manifest_sha256
        or native.get("energy_hartree") != point.energy_hartree
        or native.get("molecule", {}).get("geometry_bohr")
        != candidate.molecule.model_dump(mode="json")["geometry_bohr"]
        or native.get("molecule", {}).get("charge") != candidate.molecule.charge
        or native.get("molecule", {}).get("multiplicity")
        != candidate.molecule.multiplicity
        or candidate.candidate_request.products != ["geometry", "harmonic"]
        or any(
            target_recipe.get(field) != source_recipe.get(field)
            for field in ("engine", "method", "basis", "reference", "dispersion")
        )
    ):
        raise ValueError(
            "Candidate origin is not its successful actual scan observation."
        )
    validate_approved_plan(approved_plan, request=original_request)
    return candidate.candidate_request


def verify_candidate_minimum(
    candidate: StationaryCandidate,
    *,
    ledger: CandidateLedger,
    expected_revision: int,
    approved_plan: dict[str, Any],
    result_shard: str | Path,
    actor: str,
) -> StationaryVerification:
    """Register a separate minimum record only after actual approved qualification."""
    from .artifacts import file_digest, verify_shard
    from .scientific_values import HarmonicData, StationaryGeometry

    request = validate_candidate_refinement(
        candidate,
        ledger=ledger,
        expected_revision=expected_revision,
        approved_plan=approved_plan,
    )
    original_request = request.model_dump(mode="json")
    approved = validate_approved_plan(approved_plan, request=original_request)
    root = Path(result_shard)
    verify_shard(root)
    supplied_request = CalculationRequest.model_validate(
        read_json(root / "request.json")
    )
    if supplied_request.model_dump(mode="json") != original_request:
        raise ValueError(
            "The optimized result must bind the exact approved candidate "
            "atom/state/recipe/constraint-release request."
        )
    result = read_json(root / "result.json")
    if result.get("status") != "complete" or result.get("errors"):
        raise ValueError("A failed or partial target result cannot qualify a minimum.")
    stages = result["stages"]
    geometry_stage, harmonic_stage = (
        stages["equilibrium_geometry"],
        stages["harmonic_analysis"],
    )
    if (
        geometry_stage["status"] != "available"
        or harmonic_stage["status"] != "available"
    ):
        raise ValueError(
            "Actual optimized geometry and target harmonic characterization "
            "are required."
        )
    geometry = StationaryGeometry.model_validate(geometry_stage["value"])
    harmonic = HarmonicData.model_validate(harmonic_stage["value"])
    if (
        harmonic.stationary_character != "positive_definite_vibrational_hessian"
        or geometry.stationary_character != "positive_definite_vibrational_hessian"
        or harmonic.external_residual_relative
        > get_profile(request.recipe)["harmonic_external_residual_relative_tolerance"]
        or "hessian_external_invariance_failed" in harmonic_stage["quality_flags"]
        or "nonminimum_or_unresolved_stationary_point"
        in geometry_stage["quality_flags"]
    ):
        raise ValueError("The actual target Hessian did not verify a minimum.")
    verified = ledger.register_shard(
        root,
        actor=actor,
        reason=(
            "Target optimization and actual harmonic Hessian verify a minimum "
            f"linked to grid candidate {candidate.candidate_id}."
        ),
    )
    return StationaryVerification(
        original_candidate_id=candidate.candidate_id,
        verified_result_candidate_id=UUID(verified["candidate_id"]),
        verified_result_ledger_revision=verified["revision"],
        classification="verified_minimum",
        target_recipe=supplied_request.recipe,
        result_manifest_sha256=file_digest(root / "manifest.json"),
        original_constraints_sha256=candidate.constraints_sha256,
        approval_sha256=approved.approval_sha256,
    )
