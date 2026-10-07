"""Contract/math tests: test observations do not certify a production model."""

import hashlib

import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_ml_policy import (
    CalibrationEvidence,
    CalibrationObservation,
    DomainAssessment,
    MLPrediction,
    ModelIdentity,
    QCConfirmation,
    ScreeningBatch,
    ScreeningCandidate,
    ScreeningPolicy,
    calibrate_energy_interval,
    confirm_qc,
    screen_candidates,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def model():
    return ModelIdentity(
        name="contract-test", version="1", checkpoint_sha256=digest("model")
    )


@pytest.fixture
def policy():
    return ScreeningPolicy(
        mode="defer",
        energy_cutoff_ev=1.0,
        max_interval_ev=0.2,
        max_ood_score=0.5,
        reference_id="reference",
        target_method="QC/method/basis",
        sentinel_fraction=0.25,
    )


@pytest.fixture
def prediction(model):
    return MLPrediction(
        model=model,
        value_ev=2.0,
        quantity="relative_energy_ev",
        reference_id="reference",
        target_method="QC/method/basis",
        raw_uncertainty_ev=0.001,
        domain=DomainAssessment(
            domain_id="validated-domain",
            status="in_domain",
            ood_score=0.1,
            detector_version="detector-1",
            reason="composition/state/geometry supported",
        ),
        calibration=CalibrationEvidence(
            calibration_id="held-out-report",
            model=model,
            domain_id="validated-domain",
            reference_id="reference",
            target_method="QC/method/basis",
            quantity="relative_energy_ev",
            dataset_sha256=digest("calibration"),
            validation_sha256=digest("validation"),
            calibration_count=100,
            validation_count=100,
            coverage=0.95,
            observed_coverage=0.96,
            half_width_ev=0.1,
            held_out=True,
            exchangeability_supported=True,
        ),
    )


def changed(record, **kwargs):
    return type(record).model_validate({**record.model_dump(), **kwargs})


def candidates(prediction, count=12, kind="conformer"):
    return [
        ScreeningCandidate(
            candidate_id=f"c{i}",
            geometry_sha256=digest(f"g{i}"),
            kind=kind,
            prediction=prediction,
        )
        for i in range(count)
    ]


def qc(candidate, **kwargs):
    data = dict(
        candidate_id=candidate.candidate_id,
        geometry_sha256=candidate.geometry_sha256,
        calculation_id="qc-1",
        artifact_sha256=digest("qc-artifact"),
        engine="QC",
        method="QC/method/basis",
        reference_id="reference",
        converged=True,
        physical_checks_passed=True,
        stationary_point_validated=True,
        quantity="relative_energy_ev",
        value_ev=2.0,
        uncertainty_ev=0.1,
    )
    return QCConfirmation(**{**data, **kwargs})


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("calibration", None, "missing_calibrated_uncertainty"),
        ("raw_uncertainty_ev", None, "missing_calibrated_uncertainty"),
        ("reference_id", "other", "prediction_scope_mismatch"),
        ("target_method", "other", "prediction_scope_mismatch"),
        ("quantity", "barrier_ev", "prediction_quantity_mismatch"),
    ],
)
def test_missing_and_mismatched_prediction_escalates(
    prediction, policy, field, value, reason
):
    batch = screen_candidates(candidates(changed(prediction, **{field: value})), policy)
    assert all(d.action == "query_qc" and d.reason == reason for d in batch.decisions)
    assert not batch.low_priority_pool


@pytest.mark.parametrize(
    "field,value",
    [
        ("held_out", False),
        ("exchangeability_supported", False),
        ("observed_coverage", 0.94),
        ("coverage", 0.90),
        ("calibration_count", 10),
        ("validation_count", 10),
        ("half_width_ev", 0.3),
        ("domain_id", "other"),
        ("reference_id", "other"),
        ("quantity", "barrier_ev"),
    ],
)
def test_uncalibrated_evidence_cannot_defer(prediction, policy, field, value):
    evidence = changed(prediction.calibration, **{field: value})
    batch = screen_candidates(
        candidates(changed(prediction, calibration=evidence)), policy
    )
    assert len(batch.qc_queue) == len(batch.decisions)
    assert not batch.low_priority_pool


def test_stale_checkpoint(prediction, policy):
    model = changed(prediction.model, checkpoint_sha256=digest("updated"))
    batch = screen_candidates(candidates(changed(prediction, model=model)), policy)
    assert all(d.reason == "calibration_scope_mismatch" for d in batch.decisions)


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "unknown"},
        {"status": "out_of_domain"},
        {"ood_score": None},
        {"ood_score": 0.6},
        {"detector_version": None},
    ],
)
def test_ood_escalation_even_when_confident(prediction, policy, fields):
    pred = changed(prediction, domain=changed(prediction.domain, **fields))
    batch = screen_candidates(candidates(pred), policy)
    assert len(batch.qc_queue) == len(batch.decisions)
    assert not batch.low_priority_pool


@pytest.mark.parametrize("energy", [0.8, 1.0, 1.1])
def test_overlap_and_equality_keep_priority(prediction, policy, energy):
    batch = screen_candidates(candidates(changed(prediction, value_ev=energy)), policy)
    assert all(d.action == "prioritize" for d in batch.decisions)
    assert not batch.low_priority_pool


def test_prioritization_is_not_pruning(prediction, policy):
    batch = screen_candidates(
        candidates(prediction), changed(policy, mode="prioritize")
    )
    assert len(batch.decisions) == 12
    assert not batch.low_priority_pool
    assert all(d.action == "prioritize" for d in batch.decisions)


def test_default_fails_closed(prediction):
    batch = screen_candidates(candidates(prediction))
    assert len(batch.qc_queue) == 12


def test_retention_and_sentinel_reproducibility(prediction, policy):
    inputs = candidates(prediction)
    first = screen_candidates(inputs, policy)
    second = screen_candidates(list(reversed(inputs)), policy)

    def flags(batch):
        return {
            d.candidate.candidate_id: (d.sentinel, d.sentinel_rank)
            for d in batch.decisions
        }

    assert flags(first) == flags(second)
    assert len(first.low_priority_pool) == len(inputs)
    assert len(first.qc_queue) == 3
    assert sum(d.action == "retain_low_priority" for d in first.decisions) == 9
    assert flags(first) != flags(
        screen_candidates(inputs, changed(policy, sentinel_seed="other"))
    )


def test_small_pool_always_has_sentinel(prediction, policy):
    batch = screen_candidates(candidates(prediction, 1), policy)
    assert batch.decisions[0].sentinel
    assert len(batch.low_priority_pool) == len(batch.qc_queue) == 1
    assert screen_candidates([], policy).decisions == ()


def test_complete_audit(prediction, policy):
    batch = screen_candidates(
        candidates(prediction), changed(policy, sentinel_fraction=1)
    )
    assert all(d.sentinel and d.action == "query_qc" for d in batch.decisions)


@pytest.mark.parametrize(
    "fields",
    [
        {"mode": "defer"},
        {"sentinel_fraction": 0},
        {"min_sentinels": 0},
        {"required_coverage": 1},
        {"max_ood_score": float("nan")},
        {"energy_cutoff_ev": float("inf")},
    ],
)
def test_invalid_policy_rejected(fields):
    with pytest.raises(ValidationError):
        ScreeningPolicy(**fields)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_uncertainty_rejected(prediction, value):
    with pytest.raises(ValidationError):
        changed(prediction, raw_uncertainty_ev=value)


def test_duplicate_candidate_ids_rejected(prediction, policy):
    inputs = candidates(prediction, 1)
    with pytest.raises(ValueError, match="unique"):
        screen_candidates(inputs * 2, policy)


def test_provenance_roundtrip_and_history(prediction, policy, tmp_path):
    original = screen_candidates(candidates(prediction), policy)
    path = tmp_path / "initial.json"
    original.persist(path)
    assert ScreeningBatch.model_validate_json(path.read_text()) == original
    with pytest.raises(FileExistsError):
        original.persist(path)
    sentinel = original.qc_queue[0]
    updated = confirm_qc(original, [qc(sentinel)])
    confirmed = next(
        d
        for d in updated.decisions
        if d.candidate.candidate_id == sentinel.candidate_id
    )
    assert confirmed.action == "qc_excluded" and confirmed.sentinel
    assert confirmed.candidate.qc.artifact_sha256 == digest("qc-artifact")
    assert updated.history == (original.decisions,)
    assert original.decisions != updated.decisions
    assert ScreeningBatch.model_validate_json(updated.model_dump_json()) == updated


def test_sentinel_false_negative_reopens_pool(prediction, policy):
    original = screen_candidates(candidates(prediction), policy)
    sentinel = original.qc_queue[0]
    updated = confirm_qc(original, [qc(sentinel, value_ev=0.5)])
    assert updated.audit_failure
    assert len(updated.low_priority_pool) == 12
    assert all(d.action in ("query_qc", "prioritize") for d in updated.decisions)
    assert not any(d.action == "retain_low_priority" for d in updated.decisions)
    assert confirm_qc(updated, []).audit_failure
    next_batch = screen_candidates(
        candidates(prediction), policy, previous_batch=updated
    )
    assert next_batch.audit_failure
    assert all(d.action == "query_qc" for d in next_batch.decisions)
    assert next_batch.history[-1] == updated.decisions


@pytest.mark.parametrize(
    "fields",
    [
        {"converged": False},
        {"physical_checks_passed": False},
        {"method": "wrong"},
        {"reference_id": "wrong"},
        {"imaginary_modes": 1},
    ],
)
def test_invalid_qc_cannot_exclude(prediction, policy, fields):
    original = screen_candidates(candidates(prediction), policy)
    sentinel = original.qc_queue[0]
    updated = confirm_qc(original, [qc(sentinel, **fields)])
    decision = next(
        d
        for d in updated.decisions
        if d.candidate.candidate_id == sentinel.candidate_id
    )
    assert decision.action == "query_qc"
    assert not updated.audit_failure


def test_qc_evidence_binds_geometry_and_candidate(prediction, policy):
    original = screen_candidates(candidates(prediction), policy)
    sentinel = original.qc_queue[0]
    with pytest.raises(ValidationError, match="geometry"):
        confirm_qc(original, [qc(sentinel, geometry_sha256=digest("other"))])
    with pytest.raises(ValueError, match="unknown"):
        confirm_qc(original, [qc(sentinel, candidate_id="unknown")])
    with pytest.raises(ValueError, match="Duplicate"):
        confirm_qc(original, [qc(sentinel)] * 2)


def test_ml_barrier_never_establishes_absence(prediction, policy):
    pred = changed(prediction, quantity="barrier_ev", value_ev=10000)
    batch = screen_candidates(candidates(pred, kind="reaction_path"), policy)
    assert len(batch.qc_queue) == 12
    assert not batch.low_priority_pool
    assert all(d.path_status == "unresolved" for d in batch.decisions)


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, "qc_confirmed"),
        ({"converged": False}, "unresolved"),
        ({"physical_checks_passed": False}, "unresolved"),
        ({"imaginary_modes": 0}, "unresolved"),
        ({"imaginary_modes": 2}, "unresolved"),
        ({"intended_endpoints_connected": False}, "unresolved"),
        ({"method": "wrong"}, "unresolved"),
    ],
)
def test_path_qc_requires_ts_and_irc(prediction, policy, fields, expected):
    batch = screen_candidates(candidates(prediction, 1, "reaction_path"), policy)
    confirmation = qc(
        batch.qc_queue[0],
        **{
            **dict(
                quantity="barrier_ev",
                imaginary_modes=1,
                intended_endpoints_connected=True,
            ),
            **fields,
        },
    )
    updated = confirm_qc(batch, [confirmation])
    assert updated.decisions[0].path_status == expected
    assert updated.decisions[0].action != "qc_excluded"


def observations(prefix, count, residual):
    return [
        CalibrationObservation(
            sample_id=f"{prefix}-{i}", predicted_ev=0, reference_ev=residual
        )
        for i in range(count)
    ]


def calibrate(model, **kwargs):
    options = dict(
        calibration_id="report",
        model=model,
        domain_id="validated-domain",
        reference_id="reference",
        target_method="QC/method/basis",
        quantity="relative_energy_ev",
        training_ids=["trained-system"],
        coverage=0.95,
        exchangeability_supported=True,
        calibration=observations("c", 20, 0.1),
        validation=observations("v", 20, 0.05),
    )
    return calibrate_energy_interval(**{**options, **kwargs})


def test_calibration_order_statistic_and_validation(model, prediction, policy):
    report = calibrate(model)
    assert report.half_width_ev == 0.1 and report.observed_coverage == 1
    assert report.dataset_sha256 != report.validation_sha256
    batch = screen_candidates(
        candidates(changed(prediction, calibration=report)), policy
    )
    assert len(batch.low_priority_pool) == 12
    failed = calibrate(model, validation=observations("v", 20, 0.2))
    assert failed.observed_coverage == 0
    assert (
        len(
            screen_candidates(
                candidates(changed(prediction, calibration=failed)), policy
            ).qc_queue
        )
        == 12
    )


def test_finite_sample_rank_cannot_be_clamped(model):
    with pytest.raises(ValueError, match="Too few"):
        calibrate(model, calibration=observations("c", 5, 0.1))


@pytest.mark.parametrize(
    "fields",
    [
        {"validation": observations("c", 20, 0.1)},
        {"training_ids": ["c-0"]},
        {"calibration": observations("c", 1, 0.1) * 20},
        {"validation": []},
        {"coverage": 1},
    ],
)
def test_calibration_independence_and_invalid_inputs(model, fields):
    with pytest.raises(ValueError):
        calibrate(model, **fields)
