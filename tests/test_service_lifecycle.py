"""Actual plan/approval/CLI/file lifecycle checks; no invented hosted responses."""

import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cochem_torq.application import submit_request
from cochem_torq.domain import PrerequisiteError, canonical_json, read_json
from cochem_torq.github import GitHubAccessError, GitHubActions
from cochem_torq.service import (
    approve_plan,
    doctor,
    plan_request,
    validate_approved_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def request():
    return read_json(ROOT / "examples/student/water-hf-teaching.json")


def test_plan_binds_molecule_recipe_products_and_resources():
    review = plan_request(request())
    assert review["approval"] is None
    assert review["plan"]["request"]["molecule"] == request()["molecule"]
    approved = approve_plan(review, actor="local review test")
    assert validate_approved_plan(approved).approval.actor == "local review test"
    assert approved["approval"]["max_cpu_core_seconds"] == 1200
    altered = deepcopy(approved)
    altered["plan"]["request"]["molecule"]["charge"] = 2
    with pytest.raises(ValueError, match="digest"):
        validate_approved_plan(altered)


def test_changed_request_cannot_use_old_approval():
    review = plan_request(request())
    approved = approve_plan(review, actor="local review test")
    altered = deepcopy(review["plan"]["request"])
    altered["molecule"]["geometry_bohr"][0][2] += 0.01
    with pytest.raises(ValueError, match="differs"):
        validate_approved_plan(approved, request=altered)


def test_unqualified_recipe_cannot_receive_an_executable_approval():
    value = request()
    value["recipe"] = "revdsd-pbep86-d4-experimental"
    review = plan_request(value)
    assert not review["executable"]
    with pytest.raises(PrerequisiteError):
        approve_plan(review, actor="local review test")


def test_approval_expiry_and_budget_are_enforced():
    review = plan_request(request())
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    with pytest.raises(ValueError, match="increasing"):
        approve_plan(review, actor="local review test", expires_at=past)
    with pytest.raises(ValueError):
        approve_plan(review, actor=" ")
    approved = approve_plan(review, actor="local review test")
    approved["approval"]["max_memory_mb"] = 256
    from cochem_torq.domain import digest

    approved["approval_sha256"] = digest(approved["approval"])
    with pytest.raises(ValueError, match="budget"):
        validate_approved_plan(approved)


def test_missing_approval_or_idempotency_never_dispatches():
    with pytest.raises(ValueError, match="approved plan"):
        submit_request(request())
    with pytest.raises(ValueError, match="approved plan"):
        submit_request(request(), approved_plan={})


def test_cancellation_requires_owned_receipt_before_network(tmp_path):
    client = GitHubActions("ProfJJK-CoChem/CoChem-TORQ", state_directory=tmp_path)
    with pytest.raises(GitHubAccessError, match="owned submission receipt"):
        client.cancel("123456", reason="explicit test of missing ownership")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "arguments", [["validate"], ["unknown"], ["approve-plan", "--plan", "absent"]]
)
def test_cli_invalid_arguments_have_versioned_json_errors(arguments):
    completed = subprocess.run(
        [sys.executable, "-m", "cochem_torq.cli", "--json", *arguments],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 2
    import json

    response = json.loads(completed.stdout)
    assert response["schema_version"] == "cochem.torq.cli-response/1"
    assert response["status"] == "error"
    assert response["errors"][0]["code"] == "INVALID_REQUEST"
    assert "data" in response and "request_id" in response


def test_cli_plan_only_writes_requested_artifact(tmp_path):
    source = tmp_path / "request.json"
    source.write_bytes(canonical_json(request()))
    target = tmp_path / "plan.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.cli",
            "plan",
            "--json",
            "--request",
            str(source),
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout
    review = read_json(target)
    assert review["approval"] is None and review["executable"]
    assert {path.name for path in tmp_path.iterdir()} == {"request.json", "plan.json"}
    approved = tmp_path / "approved.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.cli",
            "approve-plan",
            "--json",
            "--plan",
            str(target),
            "--actor",
            "explicit local review test",
            "--output",
            str(approved),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout
    assert validate_approved_plan(read_json(approved))


def test_doctor_reports_observed_dependencies_and_unchecked_hosted_permissions():
    report = doctor()
    assert report["dependencies"]["pydantic"] is not None
    assert report["github"]["status"] == "not_checked"
    assert not report["github"]["calculation_dispatch_verified"]
    assert not report["full_srs_release_qualified"]
