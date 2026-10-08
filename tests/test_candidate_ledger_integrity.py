"""Actual supplied inputs, verified failed shards, SQLite and OS-process evidence.

No conformer, quantum result, device, executable reply, or model is invented.
Failure shards are produced by the real application rejecting an unqualified
method; they remain scientifically unavailable after ledger registration.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from copy import deepcopy
from uuid import uuid4

import pytest

from cochem_torq.candidate_ledger import (
    CandidateLedger,
    CandidateLedgerError,
    CandidateRevisionConflictError,
)
from cochem_torq.domain import CalculationRequest, digest
from tests.test_application_contracts import actual_failure_shard


def supplied_request():
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["nucleus-a", "nucleus-b"],
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry"],
            "source_provenance": {"producer": "explicit_local_test_input"},
        }
    ).model_dump(mode="json")


def register(ledger):
    return ledger.register_request(
        supplied_request(),
        actor="local integrity test",
        reason="Supplied molecular input",
    )


def test_supplied_input_has_no_invented_properties_and_private_real_db(tmp_path):
    path = tmp_path / "candidates.sqlite"
    with CandidateLedger(path) as ledger:
        record = register(ledger)
        assert record["origin"] == "supplied_input"
        assert record["geometry_status"] == "input"
        assert record["selection_state"] == "retained"
        assert record["review_status"] == "needs_review"
        assert record["quality"]["uncertainty"] is None
        assert record["quality"]["uncertainty_status"] == "not_computed"
        assert record["quality"]["stage_statuses"] == {}
        assert record["native_artifact_hashes"] == []
        assert record["result_reference"] is None
        assert record["source_identity"]["code_sha256"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert ledger.verify_integrity()["candidates"] == 1
        assert not ledger.verify_integrity()["authenticated_signature"]
    with pytest.raises(CandidateLedgerError, match="closed"):
        ledger.inspect(record["candidate_id"])
    with CandidateLedger(path) as reopened:
        assert reopened.inspect(record["candidate_id"]) == record


def test_exclude_restore_preserves_definition_quality_origin_and_raw_request(tmp_path):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        original = register(ledger)
        snapshot = ledger.selection_snapshot()
        excluded = ledger.exclude(
            original["candidate_id"],
            expected_revision=1,
            actor="manual reviewer",
            reason="Temporarily inspect another candidate",
            decision_mode="assignment_assisted",
        )
        assert excluded["selection_state"] == "excluded"
        assert ledger.candidates(include_excluded=False) == []
        assert ledger.candidates()[0]["candidate_id"] == original["candidate_id"]
        changed = ledger.selection_snapshot()
        assert changed["snapshot_sha256"] != snapshot["snapshot_sha256"]
        assert changed["needs_review"]
        assert not changed["automatic_pruning_enabled"]
        assert changed["search_completeness"] == "not_established"
        restored = ledger.restore(
            original["candidate_id"],
            expected_revision=2,
            actor="manual reviewer",
            reason="Restore original selection",
        )
        assert restored["selection_state"] == "retained"
        for field in (
            "candidate_id",
            "request",
            "molecule",
            "quality",
            "origin",
            "content_sha256",
            "recipe",
            "source_identity",
        ):
            assert restored[field] == original[field]
        history = ledger.history(original["candidate_id"])
        assert [event["revision"] for event in history] == [1, 2, 3]
        assert [event["action"] for event in history] == [
            "register",
            "exclude",
            "restore",
        ]
        assert history[1]["decision_mode"] == "assignment_assisted"
        assert history[2]["previous_selection_state"] == "excluded"
        assert all(
            event["actor"] and event["reason"] and event["time_utc"]
            for event in history
        )
        assert ledger.verify_integrity()["events"] == 3


def test_quarantine_exclusion_restores_prior_quality_state(tmp_path):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        original = register(ledger)
        quarantined = ledger.quarantine(
            original["candidate_id"],
            expected_revision=1,
            actor="reviewer",
            reason="Await external reference review",
        )
        assert quarantined["selection_state"] == "quarantined"
        assert ledger.selection_snapshot()["quarantined_candidate_ids"] == [
            original["candidate_id"]
        ]
        ledger.exclude(
            original["candidate_id"],
            expected_revision=2,
            actor="reviewer",
            reason="Temporarily hide quarantined input",
        )
        restored = ledger.restore(
            original["candidate_id"],
            expected_revision=3,
            actor="reviewer",
            reason="Restore previous disposition",
        )
        assert restored["selection_state"] == "quarantined"
        retained = ledger.retain(
            original["candidate_id"],
            expected_revision=4,
            actor="reviewer",
            reason="Explicitly retain input for later review",
        )
        assert retained["selection_state"] == "retained"
        assert retained["quality"] == original["quality"]
        assert retained["review_status"] == "needs_review"


def test_duplicate_uuid_cannot_overwrite_a_physical_candidate(tmp_path):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        original = register(ledger)
        altered = deepcopy(original["request"])
        altered["molecule"]["geometry_bohr"][1][2] += 0.01
        with pytest.raises(CandidateRevisionConflictError, match="UUID"):
            ledger.register_request(
                altered,
                candidate_id=original["candidate_id"],
                actor="reviewer",
                reason="Attempt incompatible identity reuse",
            )
        assert ledger.inspect(original["candidate_id"]) == original
        second = ledger.register_request(
            altered, actor="reviewer", reason="Explicit different supplied geometry"
        )
        assert second["content_sha256"] != original["content_sha256"]
        assert second["candidate_id"] != original["candidate_id"]


@pytest.mark.parametrize("operation", ["retain", "exclude", "restore", "quarantine"])
def test_stale_revision_is_rejected_without_new_event(tmp_path, operation):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        record = register(ledger)
        ledger.exclude(
            record["candidate_id"],
            expected_revision=1,
            actor="reviewer",
            reason="Actual first decision",
        )
        with pytest.raises(CandidateRevisionConflictError):
            getattr(ledger, operation)(
                record["candidate_id"],
                expected_revision=1,
                actor="stale reviewer",
                reason="Stale selection decision",
            )
        assert len(ledger.history(record["candidate_id"])) == 2


@pytest.mark.parametrize("table", ["candidate_definitions", "candidate_events"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_sql_triggers_protect_immutable_rows(tmp_path, table, operation):
    path = tmp_path / "ledger.sqlite"
    with CandidateLedger(path) as ledger:
        register(ledger)
        with sqlite3.connect(path) as connection:
            sql = f"DELETE FROM {table}"
            if operation == "UPDATE":
                field = (
                    "definition_json"
                    if table == "candidate_definitions"
                    else "event_json"
                )
                sql = f"UPDATE {table} SET {field}='{{}}'"
            with pytest.raises(sqlite3.IntegrityError, match="immutable|deleted"):
                connection.execute(sql)
        assert ledger.verify_integrity()["events"] == 1


def test_changed_bytes_rejected_after_actual_sql_corruption(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with CandidateLedger(path) as ledger:
        record = register(ledger)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        event = ledger._decode(
            connection.execute("SELECT event_json FROM candidate_events").fetchone()[0]
        )
        event["reason"] = "Changed bytes without their original hash"
        connection.execute(
            "UPDATE candidate_events SET event_json=?", (json.dumps(event),)
        )
    with pytest.raises(CandidateLedgerError, match="integrity"):
        CandidateLedger(path)
    assert record["quality"]["uncertainty"] is None


def test_genuine_failed_result_remains_quarantined_with_actual_artifact_hashes(
    tmp_path,
):
    shard = actual_failure_shard(tmp_path / "rejected-actual-method")
    original_files = {
        path.relative_to(shard).as_posix(): path.read_bytes()
        for path in shard.rglob("*")
        if path.is_file()
    }
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        record = ledger.register_shard(
            shard, actor="reviewer", reason="Retain actual failed method evidence"
        )
        assert record["origin"] == "verified_result"
        assert record["selection_state"] == "quarantined"
        assert record["geometry_status"] == "input"
        assert record["quality"]["scientific_status"] == "failed"
        assert not record["quality"]["identification_ready"]
        assert record["native_artifact_hashes"] == []
        assert record["result_reference"]["inventory"]
        assert all(
            stage["status"] == "blocked"
            for stage in record["quality"]["stage_statuses"].values()
        )
        ledger.exclude(
            record["candidate_id"],
            expected_revision=1,
            actor="reviewer",
            reason="Temporarily exclude failed attempt",
        )
        restored = ledger.restore(
            record["candidate_id"],
            expected_revision=2,
            actor="reviewer",
            reason="Restore failed evidence",
        )
        assert restored["selection_state"] == "quarantined"
    after_files = {
        path.relative_to(shard).as_posix(): path.read_bytes()
        for path in shard.rglob("*")
        if path.is_file()
    }
    assert after_files == original_files


def test_unverified_shard_is_never_registered(tmp_path):
    shard = actual_failure_shard(tmp_path / "rejected")
    (shard / "result.json").write_text("{}")
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        with pytest.raises(ValueError):
            ledger.register_shard(
                shard, actor="reviewer", reason="Reject corrupt evidence"
            )
        assert ledger.candidates() == []


@pytest.mark.parametrize(
    "field,value", [("actor", " "), ("reason", ""), ("actor", 1), ("reason", None)]
)
def test_invalid_decision_metadata_creates_no_candidate(tmp_path, field, value):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        arguments = {"actor": "reviewer", "reason": "Supplied input"}
        arguments[field] = value
        with pytest.raises(ValueError):
            ledger.register_request(supplied_request(), **arguments)
        assert ledger.candidates() == []


def test_unknown_candidate_and_invalid_identity_are_explicit_errors(tmp_path):
    with CandidateLedger(tmp_path / "ledger.sqlite") as ledger:
        with pytest.raises(KeyError):
            ledger.inspect(str(uuid4()))
        with pytest.raises(ValueError):
            ledger.inspect("not-a-UUID")
        with pytest.raises(ValueError):
            ledger.candidates(include_excluded=1)
        record = register(ledger)
        with pytest.raises(ValueError, match="positive integer"):
            ledger.exclude(
                record["candidate_id"],
                expected_revision=True,
                actor="reviewer",
                reason="Invalid revision type",
            )


def test_existing_nonprivate_file_and_symlink_are_rejected(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with CandidateLedger(path):
        assert path.exists()
    path.chmod(0o644)
    with pytest.raises(CandidateLedgerError, match="private"):
        CandidateLedger(path)
    path.chmod(0o600)
    symlink = tmp_path / "linked.sqlite"
    symlink.symlink_to(path)
    with pytest.raises(CandidateLedgerError, match="private regular"):
        CandidateLedger(symlink)


def test_real_cross_process_concurrent_cas_commits_exactly_one_event(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with CandidateLedger(path) as ledger:
        record = register(ledger)
    release = tmp_path / "release"
    script = """
import sys,time
from pathlib import Path
from cochem_torq.candidate_ledger import CandidateLedger, CandidateRevisionConflictError
path,candidate,ready,release=sys.argv[1:]
with CandidateLedger(path) as ledger:
    Path(ready).write_text('ready')
    deadline=time.monotonic()+10
    while not Path(release).exists():
        if time.monotonic()>deadline:
            raise SystemExit(10)
        time.sleep(0.01)
    try:
        ledger.exclude(candidate,expected_revision=1,actor='actual child process',
                       reason='Concurrent explicit selection')
    except CandidateRevisionConflictError:
        raise SystemExit(4)
"""
    children = []
    ready_paths = [tmp_path / "ready-1", tmp_path / "ready-2"]
    try:
        for ready in ready_paths:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(path),
                        record["candidate_id"],
                        str(ready),
                        str(release),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ.copy(),
                )
            )
        deadline = time.monotonic() + 15
        while not all(ready.exists() for ready in ready_paths):
            assert time.monotonic() < deadline, "Actual child readiness timed out"
            assert all(child.poll() is None for child in children)
            time.sleep(0.01)
        release.write_text("allow actual concurrent SQLite operations")
        outputs = [child.communicate(timeout=15) for child in children]
        assert sorted(child.returncode for child in children) == [0, 4], outputs
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=5)
    with CandidateLedger(path) as ledger:
        assert ledger.inspect(record["candidate_id"])["revision"] == 2
        assert ledger.verify_integrity()["events"] == 2
        snapshot = ledger.selection_snapshot()
        content = {
            key: value for key, value in snapshot.items() if key != "snapshot_sha256"
        }
        assert digest(content) == snapshot["snapshot_sha256"]
