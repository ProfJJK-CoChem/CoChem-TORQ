"""Research-grade ML screening contracts and reversible QC scheduling.

All energies are relative eV for the explicit reference_id and target_method.
An ensemble standard deviation alone is never a calibrated interval.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[float, Field(ge=0)]


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ModelIdentity(Contract):
    name: Text
    version: Text
    checkpoint_sha256: Digest


class CalibrationEvidence(Contract):
    """Held-out calibration plus independent validation, bound to the task."""

    calibration_id: Text
    model: ModelIdentity
    domain_id: Text
    reference_id: Text
    target_method: Text
    quantity: Literal["relative_energy_ev", "barrier_ev"]
    dataset_sha256: Digest
    validation_sha256: Digest
    calibration_count: int = Field(ge=1)
    validation_count: int = Field(ge=1)
    coverage: float = Field(gt=0, lt=1)
    observed_coverage: float = Field(ge=0, le=1)
    half_width_ev: Nonnegative
    held_out: bool
    exchangeability_supported: bool


class DomainAssessment(Contract):
    """Domain ID must cover composition, charge, spin and geometry regime."""

    domain_id: Text
    status: Literal["in_domain", "out_of_domain", "unknown"] = "unknown"
    ood_score: Nonnegative | None = None
    detector_version: Text | None = None
    reason: Text


class MLPrediction(Contract):
    model: ModelIdentity
    value_ev: float
    quantity: Literal["relative_energy_ev", "barrier_ev"]
    reference_id: Text
    target_method: Text
    domain: DomainAssessment
    raw_uncertainty_ev: Nonnegative | None = None
    calibration: CalibrationEvidence | None = None


class QCConfirmation(Contract):
    """A parsed, validated QC result; failed searches do not prove absence."""

    candidate_id: Text
    geometry_sha256: Digest
    calculation_id: Text
    artifact_sha256: Digest
    engine: Text
    method: Text
    reference_id: Text
    converged: bool
    physical_checks_passed: bool
    stationary_point_validated: bool = False
    quantity: Literal["relative_energy_ev", "barrier_ev"]
    value_ev: float
    uncertainty_ev: Nonnegative
    imaginary_modes: int = Field(default=0, ge=0)
    intended_endpoints_connected: bool = False


class ScreeningCandidate(Contract):
    candidate_id: Text
    geometry_sha256: Digest
    kind: Literal["conformer", "reaction_path"] = "conformer"
    prediction: MLPrediction | None = None
    qc: QCConfirmation | None = None

    @model_validator(mode="after")
    def bind_qc(self) -> ScreeningCandidate:
        if self.qc and (
            self.qc.candidate_id != self.candidate_id
            or self.qc.geometry_sha256 != self.geometry_sha256
        ):
            raise ValueError("QC evidence does not match candidate and geometry")
        return self


class ScreeningPolicy(Contract):
    version: Text = "ml-safeguards-v1"
    mode: Literal["prioritize", "defer"] = "prioritize"
    energy_cutoff_ev: Nonnegative | None = None
    max_interval_ev: Nonnegative | None = None
    max_ood_score: Nonnegative | None = None
    reference_id: Text | None = None
    target_method: Text | None = None
    required_coverage: float = Field(default=0.95, gt=0, lt=1)
    min_calibration_count: int = Field(default=20, ge=1)
    min_validation_count: int = Field(default=20, ge=1)
    sentinel_fraction: float = Field(default=0.05, gt=0, le=1)
    min_sentinels: int = Field(default=1, ge=1)
    sentinel_seed: Text = "torq-v1"

    @model_validator(mode="after")
    def explicit_defer_thresholds(self) -> ScreeningPolicy:
        if self.mode == "defer" and any(
            value is None
            for value in (
                self.energy_cutoff_ev,
                self.max_interval_ev,
                self.max_ood_score,
                self.reference_id,
                self.target_method,
            )
        ):
            raise ValueError(
                "Deferral requires explicit limits, reference and QC method"
            )
        return self


class Decision(Contract):
    candidate: ScreeningCandidate
    policy: ScreeningPolicy
    action: Literal["prioritize", "retain_low_priority", "query_qc", "qc_excluded"]
    reason: Text
    would_defer: bool = False
    sentinel: bool = False
    sentinel_rank: int | None = None
    lower_bound_ev: float | None = None
    path_status: Literal["not_applicable", "unresolved", "qc_confirmed"]


class ScreeningBatch(Contract):
    """Every input appears exactly once; QC exclusion still retains its record."""

    decisions: tuple[Decision, ...]
    audit_failure: bool = False
    history: tuple[tuple[Decision, ...], ...] = ()

    @property
    def low_priority_pool(self) -> tuple[ScreeningCandidate, ...]:
        return tuple(d.candidate for d in self.decisions if d.would_defer)

    @property
    def qc_queue(self) -> tuple[ScreeningCandidate, ...]:
        return tuple(d.candidate for d in self.decisions if d.action == "query_qc")

    def persist(self, path: Path) -> None:
        """Create an immutable audit snapshot; never overwrite an earlier decision."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(self.model_dump_json(indent=2))


def calibration_gate(prediction: MLPrediction, policy: ScreeningPolicy) -> str | None:
    """Return a QC escalation reason, or None when the calibration is usable."""
    domain = prediction.domain
    if (
        policy.reference_id is not None
        and policy.reference_id != prediction.reference_id
        or policy.target_method is not None
        and policy.target_method != prediction.target_method
    ):
        return "prediction_scope_mismatch"
    if domain.status != "in_domain":
        return "ood_or_unknown_domain"
    if domain.ood_score is None or domain.detector_version is None:
        return "missing_ood_evidence"
    if policy.max_ood_score is None:
        return "missing_ood_limit"
    if domain.ood_score > policy.max_ood_score:
        return "ood_threshold_exceeded"
    evidence = prediction.calibration
    if evidence is None or prediction.raw_uncertainty_ev is None:
        return "missing_calibrated_uncertainty"
    if (
        evidence.model != prediction.model
        or evidence.domain_id != domain.domain_id
        or evidence.reference_id != prediction.reference_id
        or evidence.target_method != prediction.target_method
        or evidence.quantity != prediction.quantity
    ):
        return "calibration_scope_mismatch"
    if not evidence.held_out or not evidence.exchangeability_supported:
        return "unvalidated_calibration_domain"
    rank = math.ceil((evidence.calibration_count + 1) * evidence.coverage)
    if (
        evidence.calibration_count < policy.min_calibration_count
        or rank > evidence.calibration_count
        or evidence.validation_count < policy.min_validation_count
        or evidence.coverage < policy.required_coverage
        or evidence.observed_coverage < evidence.coverage
    ):
        return "calibration_coverage_gate_failed"
    if policy.max_interval_ev is None:
        return "missing_interval_limit"
    if evidence.half_width_ev > policy.max_interval_ev:
        return "uncertainty_threshold_exceeded"
    return None


def _decision(candidate: ScreeningCandidate, policy: ScreeningPolicy) -> Decision:
    path_status: Literal["not_applicable", "unresolved", "qc_confirmed"] = (
        "unresolved" if candidate.kind == "reaction_path" else "not_applicable"
    )
    qc = candidate.qc
    pred = candidate.prediction
    valid_qc = bool(
        qc
        and qc.converged
        and qc.physical_checks_passed
        and qc.stationary_point_validated
    )
    if candidate.kind == "reaction_path":
        confirmed = bool(
            valid_qc
            and qc
            and qc.quantity == "barrier_ev"
            and qc.imaginary_modes == 1
            and qc.intended_endpoints_connected
            and (
                pred is None
                or (
                    qc.method == pred.target_method
                    and qc.reference_id == pred.reference_id
                )
            )
        )
        return Decision(
            candidate=candidate,
            policy=policy,
            action="prioritize" if confirmed else "query_qc",
            reason="qc_path_confirmed" if confirmed else "qc_path_validation_required",
            path_status="qc_confirmed" if confirmed else "unresolved",
        )
    if (
        valid_qc
        and qc
        and qc.quantity == "relative_energy_ev"
        and qc.imaginary_modes == 0
    ):
        matches = pred is None or (
            qc.method == pred.target_method and qc.reference_id == pred.reference_id
        )
        matches = (
            matches
            and (policy.target_method is None or qc.method == policy.target_method)
            and (policy.reference_id is None or qc.reference_id == policy.reference_id)
        )
        if matches:
            lower = qc.value_ev - qc.uncertainty_ev
            exclude = (
                policy.energy_cutoff_ev is not None
                and lower > policy.energy_cutoff_ev
                and qc.reference_id == policy.reference_id
                and qc.method == policy.target_method
            )
            return Decision(
                candidate=candidate,
                policy=policy,
                action="qc_excluded" if exclude else "prioritize",
                reason="qc_confirmed_above_window" if exclude else "qc_retained",
                lower_bound_ev=lower,
                path_status=path_status,
            )
    if qc is not None:
        return Decision(
            candidate=candidate,
            policy=policy,
            action="query_qc",
            reason="qc_failed_or_scope_mismatch",
            path_status=path_status,
        )
    failure = "missing_prediction" if pred is None else calibration_gate(pred, policy)
    if pred and pred.quantity != "relative_energy_ev":
        failure = "prediction_quantity_mismatch"
    if failure:
        return Decision(
            candidate=candidate,
            policy=policy,
            action="query_qc",
            reason=failure,
            path_status=path_status,
        )
    assert pred is not None and pred.calibration is not None
    lower = pred.value_ev - pred.calibration.half_width_ev
    above = policy.energy_cutoff_ev is not None and lower > policy.energy_cutoff_ev
    defer = policy.mode == "defer" and above
    return Decision(
        candidate=candidate,
        policy=policy,
        action="retain_low_priority" if defer else "prioritize",
        reason="calibrated_lower_bound_above_window"
        if defer
        else "ml_prioritization_only",
        would_defer=defer,
        lower_bound_ev=lower,
        path_status=path_status,
    )


def screen_candidates(
    candidates: Sequence[ScreeningCandidate],
    policy: ScreeningPolicy | None = None,
    *,
    previous_batch: ScreeningBatch | None = None,
) -> ScreeningBatch:
    """Partition scheduling, sample would-be exclusions, retain all candidates."""
    policy = policy or ScreeningPolicy()
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique within a screening batch")
    decisions = [_decision(c, policy) for c in candidates]
    if previous_batch is not None and previous_batch.audit_failure:
        reopened = []
        for decision in decisions:
            if decision.would_defer:
                data = decision.model_dump()
                data.update(action="query_qc", reason="prior_sentinel_audit_failure")
                decision = Decision.model_validate(data)
            reopened.append(decision)
        return ScreeningBatch(
            decisions=tuple(reopened),
            audit_failure=True,
            history=(*previous_batch.history, previous_batch.decisions),
        )
    deferred = [d for d in decisions if d.would_defer]

    def key(decision: Decision) -> tuple[str, str]:
        token = json.dumps(
            [
                policy.sentinel_seed,
                policy.version,
                decision.candidate.candidate_id,
                decision.candidate.geometry_sha256,
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(
            token.encode()
        ).hexdigest(), decision.candidate.candidate_id

    ordered = sorted(deferred, key=key)
    count = min(
        len(ordered),
        max(policy.min_sentinels, math.ceil(len(ordered) * policy.sentinel_fraction)),
    )
    ranks = {d.candidate.candidate_id: i + 1 for i, d in enumerate(ordered)}
    revised = []
    for d in decisions:
        rank = ranks.get(d.candidate.candidate_id)
        if rank is not None:
            # Re-validate every transition through the typed constructor.
            data = d.model_dump()
            data.update(sentinel_rank=rank, sentinel=rank <= count)
            if rank <= count:
                data.update(action="query_qc", reason="sentinel_qc_validation")
            d = Decision.model_validate(data)
        revised.append(d)
    return ScreeningBatch(decisions=tuple(revised))


def confirm_qc(
    batch: ScreeningBatch, confirmations: Sequence[QCConfirmation]
) -> ScreeningBatch:
    """Append QC evidence in a new snapshot; sentinel false negatives reopen pool."""
    by_id = {q.candidate_id: q for q in confirmations}
    if len(by_id) != len(confirmations):
        raise ValueError("Duplicate QC confirmations")
    known = {d.candidate.candidate_id for d in batch.decisions}
    if not set(by_id) <= known:
        raise ValueError("QC confirmation for an unknown candidate")
    revised = []
    audit_failure = batch.audit_failure
    for previous in batch.decisions:
        qc = by_id.get(previous.candidate.candidate_id)
        if qc is None:
            revised.append(previous)
            continue
        data = previous.candidate.model_dump()
        data["qc"] = qc.model_dump()
        candidate = ScreeningCandidate.model_validate(data)
        current = _decision(candidate, previous.policy)
        data = current.model_dump()
        data.update(
            would_defer=previous.would_defer,
            sentinel=previous.sentinel,
            sentinel_rank=previous.sentinel_rank,
        )
        current = Decision.model_validate(data)
        if previous.sentinel and current.action == "prioritize":
            audit_failure = True
        revised.append(current)
    if audit_failure:
        reopened = []
        for d in revised:
            if d.would_defer and d.action in ("retain_low_priority", "query_qc"):
                data = d.model_dump()
                data.update(
                    action="query_qc", reason="sentinel_false_negative_reopen_pool"
                )
                d = Decision.model_validate(data)
            reopened.append(d)
        revised = reopened
    return ScreeningBatch(
        decisions=tuple(revised),
        audit_failure=audit_failure,
        history=(*batch.history, batch.decisions),
    )


class CalibrationObservation(Contract):
    sample_id: Text
    predicted_ev: float
    reference_ev: float


def calibrate_energy_interval(
    *,
    calibration_id: str,
    model: ModelIdentity,
    domain_id: str,
    reference_id: str,
    target_method: str,
    quantity: Literal["relative_energy_ev", "barrier_ev"],
    calibration: Sequence[CalibrationObservation],
    validation: Sequence[CalibrationObservation],
    training_ids: Sequence[str],
    coverage: float,
    exchangeability_supported: bool,
) -> CalibrationEvidence:
    """Split conformal absolute-residual interval with independent validation.

    Independence/exchangeability of chemical systems remains a scientific input;
    distinct IDs alone cannot establish it. No OOD coverage guarantee is implied.
    """
    if not 0 < coverage < 1 or not calibration or not validation:
        raise ValueError("Require coverage in (0,1) and two nonempty held-out sets")
    c_ids = [o.sample_id for o in calibration]
    v_ids = [o.sample_id for o in validation]
    if (
        len(set(c_ids)) != len(c_ids)
        or len(set(v_ids)) != len(v_ids)
        or set(c_ids) & set(v_ids)
        or (set(c_ids) | set(v_ids)) & set(training_ids)
    ):
        raise ValueError(
            "Training, calibration and validation systems must be disjoint"
        )
    rank = math.ceil((len(calibration) + 1) * coverage)
    if rank > len(calibration):
        raise ValueError("Too few independent systems for a finite conformal interval")
    residuals = sorted(abs(o.reference_ev - o.predicted_ev) for o in calibration)
    half_width = residuals[rank - 1]
    observed = sum(
        abs(o.reference_ev - o.predicted_ev) <= half_width for o in validation
    ) / len(validation)

    def digest(observations: Sequence[CalibrationObservation]) -> str:
        payload = [
            o.model_dump() for o in sorted(observations, key=lambda o: o.sample_id)
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    return CalibrationEvidence(
        calibration_id=calibration_id,
        model=model,
        domain_id=domain_id,
        reference_id=reference_id,
        target_method=target_method,
        quantity=quantity,
        dataset_sha256=digest(calibration),
        validation_sha256=digest(validation),
        calibration_count=len(calibration),
        validation_count=len(validation),
        coverage=coverage,
        observed_coverage=observed,
        half_width_ev=half_width,
        held_out=True,
        exchangeability_supported=exchangeability_supported,
    )
