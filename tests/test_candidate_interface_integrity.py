"""Actual SQLite and CLI selection changes preserve input and require review."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cochem_torq.domain import read_json
from cochem_torq.student_app import StudentSession

ROOT = Path(__file__).resolve().parents[1]


def test_selection_changes_preserve_originals_and_invalidate_approval(tmp_path):
    session = StudentSession(tmp_path / "downloads")
    session.load_request(ROOT / "examples/student/water-hf-teaching.json")
    session.review_plan()
    session.approve(actor="explicit selection integration check")
    original = json.loads(json.dumps(session.request))
    candidate = session.register_input_candidate(
        actor="explicit selection integration check", reason="retain supplied input"
    )
    assert session.approved_plan is None
    assert session.request["request_id"] != original["request_id"]
    inspected = session.inspect_candidate(candidate["candidate_id"])
    assert inspected["candidate"]["request"] == original
    session.review_plan()
    session.approve(actor="explicit selection integration check")
    excluded = session.change_candidate_selection(
        candidate["candidate_id"],
        revision=candidate["revision"],
        action="exclude",
        actor="explicit selection integration check",
        reason="manual input selection decision; no computed energy exists",
    )
    assert excluded["selection_state"] == "excluded"
    assert session.approved_plan is None
    with pytest.raises(RuntimeError, match="excluded"):
        session.review_plan()
    restored = session.change_candidate_selection(
        candidate["candidate_id"],
        revision=excluded["revision"],
        action="restore",
        actor="explicit selection integration check",
        reason="undo the manual exclusion",
    )
    assert restored["selection_state"] == "retained"
    final = session.inspect_candidate(candidate["candidate_id"])
    assert final["candidate"]["request"] == original
    assert len(final["history"]) == 3
    assert final["selection"]["needs_review"]
    assert final["selection"]["search_completeness"] == "not_established"
    assert session.request["molecule"] == original["molecule"]


def test_headless_candidate_commands_have_actual_reversible_state(tmp_path):
    request_path = ROOT / "examples/student/water-hf-teaching.json"
    ledger = tmp_path / "actual-candidates.sqlite"
    common = [
        sys.executable,
        "-m",
        "cochem_torq.cli",
        "--json",
        "candidates",
    ]
    registration = subprocess.run(
        [
            *common,
            "register",
            "--ledger",
            str(ledger),
            "--request",
            str(request_path),
            "--actor",
            "explicit headless selection test",
            "--reason",
            "register a genuine supplied request file",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert registration.returncode == 0, registration.stdout + registration.stderr
    record = json.loads(registration.stdout)["data"]
    assert record["request"]["molecule"] == read_json(request_path)["molecule"]
    exclusion = subprocess.run(
        [
            *common,
            "exclude",
            "--ledger",
            str(ledger),
            "--candidate-id",
            record["candidate_id"],
            "--revision",
            str(record["revision"]),
            "--actor",
            "explicit headless selection test",
            "--reason",
            "manual exclusion has no inferred uncertainty",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert exclusion.returncode == 0, exclusion.stdout + exclusion.stderr
    assert json.loads(exclusion.stdout)["data"]["selection_state"] == "excluded"


def test_external_candidate_exclusion_blocks_submission_before_network(tmp_path):
    from cochem_torq.candidate_ledger import CandidateLedger

    session = StudentSession(tmp_path / "downloads")
    session.load_request(ROOT / "examples/student/water-hf-teaching.json")
    candidate = session.register_input_candidate(
        actor="first actual ledger operator", reason="retain a supplied input"
    )
    session.review_plan()
    session.approve(actor="first actual ledger operator")
    with CandidateLedger(session.candidate_ledger_path) as ledger:
        ledger.exclude(
            candidate["candidate_id"],
            expected_revision=candidate["revision"],
            actor="second actual ledger operator",
            reason="record an independent manual exclusion decision",
        )
    with pytest.raises(RuntimeError, match="excluded"):
        session.submit()
    assert session.approved_plan is None
    assert session.submission is None
