"""Reject malformed approval artifacts through actual public service contracts."""

from copy import deepcopy
from pathlib import Path

import pytest

from cochem_torq.domain import CalculationRequest, read_json
from cochem_torq.service import (
    approve_plan,
    doctor,
    plan_request,
    validate_approved_plan,
)

INPUT = Path(__file__).resolve().parents[1] / "examples/student/water-hf-teaching.json"


@pytest.mark.parametrize("value", [None, [], 1, "request"])
def test_plan_requires_a_request_object(value):
    with pytest.raises(ValueError, match="request object"):
        plan_request(value)


def test_plan_accepts_actual_typed_request_without_inferred_execution():
    typed = CalculationRequest.model_validate(read_json(INPUT))
    review = plan_request(typed)
    assert review["plan"]["request"] == typed.model_dump(mode="json")
    assert review["approval"] is None


@pytest.mark.parametrize("execution", ["unknown", "", None, []])
def test_unsupported_environment_never_reports_available(execution):
    with pytest.raises(ValueError, match="Unknown execution"):
        doctor(execution=execution)
    with pytest.raises(ValueError, match="Unknown execution"):
        plan_request(read_json(INPUT), execution=execution)


@pytest.mark.parametrize(
    "review",
    [
        None,
        [],
        {},
        {"schema_version": "cochem.torq.plan-review/1"},
        {"schema_version": "cochem.torq.plan-review/1", "plan": None},
        {
            "schema_version": "cochem.torq.plan-review/1",
            "plan": {"request": []},
            "execution": "github_actions",
        },
    ],
)
def test_malformed_review_raises_stable_validation_error(review):
    with pytest.raises(ValueError):
        approve_plan(review, actor="explicit local validation")


@pytest.mark.parametrize(
    "corruption", ["request", "request_resources", "resources", "tasks", "task_shape"]
)
def test_malformed_approved_plan_cannot_escape_with_key_or_type_error(corruption):
    approved = approve_plan(
        plan_request(read_json(INPUT)), actor="explicit local validation"
    )
    altered = deepcopy(approved)
    if corruption == "request":
        altered["plan"].pop("request")
    elif corruption == "request_resources":
        altered["plan"]["request"].pop("resources")
    elif corruption == "resources":
        altered["plan"].pop("resources")
    elif corruption == "tasks":
        altered["plan"]["tasks"] = None
    else:
        altered["plan"]["tasks"] = [None]
    with pytest.raises(ValueError, match="complete request/resources"):
        validate_approved_plan(altered)
