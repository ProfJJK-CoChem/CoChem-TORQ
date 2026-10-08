"""Real SQLite/process/file tests for campaign authority; no physical stand-ins."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from cochem.orchestration.campaign import (
    Allocation,
    AuthorityError,
    BudgetExceeded,
    CampaignBudget,
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
    RevisionConflict,
)

ACTOR = f"local-os-user:{os.getuid()}"


def budget(**changes):
    values = {
        "max_wall_seconds": 120.0,
        "max_cpu_core_seconds": 120.0,
        "max_tasks": 8,
        "max_concurrent_workers": 1,
        "max_cpu_cores": 1,
        "max_memory_mb": 256,
        "max_scratch_mb": 32,
        "expires_at_unix": time.time() + 60,
        "allowed_engines": ["software-process-check"],
        "allowed_recipes": ["integer-arithmetic-process"],
    }
    values.update(changes)
    return CampaignBudget(**values)


def fixture_campaign(tmp_path, **changes):
    store = CampaignCoordinator(tmp_path / "campaign.sqlite")
    approved = store.create_campaign(
        {"purpose": "real operating-system and mathematical checks"},
        budget(**changes),
        actor=ACTOR,
    )
    return store, approved


def task_attempt(store, approved, label="one"):
    task = store.register_task(
        approved["campaign_id"],
        {"software_check": label},
        engine="software-process-check",
        recipe="integer-arithmetic-process",
        plan_sha256=approved["plan_sha256"],
        authority=approved["authority"],
        actor=ACTOR,
    )
    attempt = store.new_attempt(
        task["task_id"], authority=approved["authority"], actor=ACTOR
    )
    return task, store.transition(
        attempt["id"],
        attempt["revision"],
        "validated",
        authority=approved["authority"],
        actor=ACTOR,
        reason="Validated actual software-process configuration.",
    )


def allocate(store, approved, attempt, **changes):
    values = {"wall_seconds": 10.0, "cores": 1, "memory_mb": 128, "scratch_mb": 8}
    values.update(changes)
    return store.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(**values),
        authority=approved["authority"],
        actor=ACTOR,
    )


def running(store, approved, attempt, *, pid=None, **lease_settings):
    queued = allocate(store, approved, attempt)
    return store.lease(
        queued["id"],
        queued["revision"],
        pid=pid or os.getpid(),
        authority=approved["authority"],
        actor=ACTOR,
        **lease_settings,
    )


def advance_worker(store, attempt, state):
    return {
        **store.worker_transition(
            attempt["id"],
            attempt["revision"],
            state,
            lease_token=attempt["lease_token"],
            lease_generation=attempt["lease_generation"],
            actor=ACTOR,
            reason=f"Actual software worker entered {state}.",
        ),
        "lease_token": attempt["lease_token"],
        "lease_generation": attempt["lease_generation"],
    }


def observed_usage(start_wall, start_cpu):
    return MeasuredUsage(
        wall_seconds=time.monotonic() - start_wall,
        cpu_core_seconds=time.process_time() - start_cpu,
        measurement_source="actual Python monotonic and process-time differences",
    )


def test_cas_events_restart_and_fenced_actual_file_publication(tmp_path):
    start_wall, start_cpu = time.monotonic(), time.process_time()
    store, approved = fixture_campaign(tmp_path)
    _, attempt = task_attempt(store, approved)
    store.close()
    store = CampaignCoordinator(tmp_path / "campaign.sqlite")
    assert store.attempt(attempt["id"])["state"] == "validated"
    with pytest.raises(RevisionConflict):
        store.transition(
            attempt["id"],
            0,
            "cancelled",
            authority=approved["authority"],
            actor=ACTOR,
            reason="Stale caller.",
        )
    attempt = running(store, approved, attempt)
    store.heartbeat(
        attempt["id"],
        lease_token=attempt["lease_token"],
        lease_generation=attempt["lease_generation"],
    )
    attempt = advance_worker(store, attempt, "collecting")
    store.close()
    store = CampaignCoordinator(tmp_path / "campaign.sqlite")
    assert store.attempt(attempt["id"])["state"] == "collecting"
    attempt = advance_worker(store, attempt, "validating")
    original = tmp_path / "actual-software-result.txt"
    original.write_text(str(sum(range(101))))
    destination = tmp_path / "published.txt"
    result = store.publish(
        attempt["id"],
        attempt["revision"],
        {"kind": "software", "filename": destination.name},
        observed_usage(start_wall, start_cpu),
        lambda: original.rename(destination),
        state="succeeded",
        lease_token=attempt["lease_token"],
        lease_generation=attempt["lease_generation"],
        actor=ACTOR,
    )
    assert destination.read_text() == "5050"
    assert result["state"] == "succeeded"
    assert store.accounting(approved["campaign_id"])[0]["state"] == "settled"
    events = store.events(approved["campaign_id"])
    transitions = [
        json.loads(event["details_json"])["to"]
        for event in events
        if event["kind"] == "transition"
    ]
    assert transitions == [
        "draft",
        "validated",
        "queued",
        "running",
        "collecting",
        "validating",
        "succeeded",
    ]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connection.execute("DELETE FROM campaign_events")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        store.connection.execute(
            "UPDATE campaign_attempts SET state='running' WHERE id=?", (attempt["id"],)
        )
    store.close()


def test_task_identity_deduplicates_exact_requests_and_rejects_changed_plan(tmp_path):
    store, approved = fixture_campaign(tmp_path)
    arguments = dict(
        engine="software-process-check",
        recipe="integer-arithmetic-process",
        plan_sha256=approved["plan_sha256"],
        authority=approved["authority"],
        actor=ACTOR,
    )
    first = store.register_task(approved["campaign_id"], {"input": "same"}, **arguments)
    second = store.register_task(
        approved["campaign_id"], {"input": "same"}, **arguments
    )
    assert first["task_id"] == second["task_id"] and not second["created"]
    with pytest.raises(AuthorityError, match="exact immutable plan"):
        store.register_task(
            approved["campaign_id"],
            {"input": "same"},
            **{**arguments, "plan_sha256": "0" * 64},
        )
    with pytest.raises(AuthorityError, match="scientific model scope"):
        store.register_task(
            approved["campaign_id"],
            {"input": "same"},
            **{**arguments, "recipe": "changed-model"},
        )
    store.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"wall_seconds": 121.0},
        {"cores": 2},
        {"memory_mb": 257},
        {"scratch_mb": 33},
        {"gpu_count": 1},
    ],
)
def test_every_allocation_ceiling_is_enforced_before_dispatch(tmp_path, changes):
    store, approved = fixture_campaign(tmp_path)
    _, attempt = task_attempt(store, approved)
    with pytest.raises(BudgetExceeded):
        allocate(store, approved, attempt, **changes)
    assert store.attempt(attempt["id"])["state"] == "validated"
    assert not store.accounting(approved["campaign_id"])
    store.close()


def _race_reservation(arguments):
    database, attempt_id, revision, authority = arguments
    store = CampaignCoordinator(database)
    try:
        store.reserve(
            attempt_id,
            revision,
            Allocation(wall_seconds=10, cores=1, memory_mb=128, scratch_mb=8),
            authority=authority,
            actor=ACTOR,
        )
        return "granted"
    except BudgetExceeded:
        return "denied"
    finally:
        store.close()


def test_real_concurrent_processes_cannot_double_spend_remaining_capacity(tmp_path):
    store, approved = fixture_campaign(tmp_path)
    _, first = task_attempt(store, approved, "first")
    _, second = task_attempt(store, approved, "second")
    store.close()
    arguments = [
        (
            str(tmp_path / "campaign.sqlite"),
            item["id"],
            item["revision"],
            approved["authority"],
        )
        for item in (first, second)
    ]
    with multiprocessing.get_context("spawn").Pool(2) as workers:
        outcomes = workers.map(_race_reservation, arguments)
    assert sorted(outcomes) == ["denied", "granted"]
    store = CampaignCoordinator(tmp_path / "campaign.sqlite")
    assert len(store.accounting(approved["campaign_id"])) == 1
    store.close()


def test_operator_cancellation_invalidates_worker_without_releasing_live_resources(
    tmp_path,
):
    store, approved = fixture_campaign(tmp_path, permitted_retries=1)
    task, draft = task_attempt(store, approved)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        attempt = running(store, approved, draft, pid=child.pid)
        with pytest.raises(AuthorityError):
            store.cancel(
                attempt["id"],
                attempt["revision"],
                authority=attempt["lease_token"],
                actor=ACTOR,
                reason="Wrong authority kind.",
            )
        cancelled = store.cancel(
            attempt["id"],
            attempt["revision"],
            authority=approved["authority"],
            actor=ACTOR,
            reason="Authenticated operator stopped owned work.",
        )
        with pytest.raises(AuthorityError, match="stale attempt lease"):
            store.heartbeat(
                attempt["id"],
                lease_token=attempt["lease_token"],
                lease_generation=attempt["lease_generation"],
            )
        usage = MeasuredUsage(
            wall_seconds=time.monotonic() - started,
            measurement_source="actual parent monotonic; CPU unavailable",
        )
        with pytest.raises(CampaignError, match="original owned process is alive"):
            store.reconcile(
                cancelled["id"],
                usage,
                authority=approved["authority"],
                actor=ACTOR,
                reason="Process still alive.",
            )
        assert store.accounting(approved["campaign_id"])[0]["state"] == "reserved"
        child.communicate(timeout=10)
        usage = MeasuredUsage(
            wall_seconds=time.monotonic() - started,
            measurement_source="actual parent monotonic; CPU unavailable",
        )
        store.reconcile(
            cancelled["id"],
            usage,
            authority=approved["authority"],
            actor=ACTOR,
            reason="Verified original child stopped.",
        )
        record = store.accounting(approved["campaign_id"])[0]
        assert json.loads(record["usage_json"])["cpu_core_seconds"] is None
        assert record["charged_cpu"] == 10.0  # Accounting ceiling, not measured CPU.
        resumed = store.new_attempt(
            task["task_id"],
            authority=approved["authority"],
            actor=ACTOR,
            parent_attempt_id=cancelled["id"],
            compatibility_digest=task["task_key"],
        )
        assert (
            resumed["id"] != cancelled["id"]
            and resumed["parent_attempt_id"] == cancelled["id"]
        )
        assert store.attempt(cancelled["id"])["state"] == "cancelled"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        store.close()


def test_expired_lease_cannot_publish_or_release_its_unreconciled_budget(tmp_path):
    store, approved = fixture_campaign(tmp_path)
    _, draft = task_attempt(store, approved)
    attempt = running(
        store, approved, draft, heartbeat_seconds=0.01, lease_seconds=0.03
    )
    time.sleep(0.05)
    with pytest.raises(AuthorityError, match="expired/stale"):
        store.heartbeat(
            attempt["id"],
            lease_token=attempt["lease_token"],
            lease_generation=attempt["lease_generation"],
        )
    _, other = task_attempt(store, approved, "other")
    with pytest.raises(BudgetExceeded):
        allocate(store, approved, other)
    assert store.accounting(approved["campaign_id"])[0]["state"] == "reserved"
    store.close()


def test_publication_failure_rolls_back_completion_and_settlement(tmp_path):
    start_wall, start_cpu = time.monotonic(), time.process_time()
    store, approved = fixture_campaign(tmp_path)
    _, draft = task_attempt(store, approved)
    attempt = advance_worker(
        store,
        advance_worker(store, running(store, approved, draft), "collecting"),
        "validating",
    )
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("actual software artifact bytes")
    target.write_text("immutable existing bytes")
    with pytest.raises(FileExistsError):
        store.publish(
            attempt["id"],
            attempt["revision"],
            {"kind": "software"},
            observed_usage(start_wall, start_cpu),
            lambda: os.link(source, target),
            state="succeeded",
            lease_token=attempt["lease_token"],
            lease_generation=attempt["lease_generation"],
            actor=ACTOR,
        )
    assert store.attempt(attempt["id"])["state"] == "validating"
    assert store.accounting(approved["campaign_id"])[0]["state"] == "reserved"
    assert target.read_text() == "immutable existing bytes"
    store.close()


def test_unrun_dependencies_and_revoked_future_work_are_blocked(tmp_path):
    store, approved = fixture_campaign(tmp_path)
    parent, _ = task_attempt(store, approved)
    child = store.register_task(
        approved["campaign_id"],
        {"child": "actual dependency check"},
        engine="software-process-check",
        recipe="integer-arithmetic-process",
        plan_sha256=approved["plan_sha256"],
        authority=approved["authority"],
        actor=ACTOR,
        dependencies=[parent["task_id"]],
    )
    attempt = store.new_attempt(
        child["task_id"], authority=approved["authority"], actor=ACTOR
    )
    attempt = store.transition(
        attempt["id"],
        attempt["revision"],
        "validated",
        authority=approved["authority"],
        actor=ACTOR,
        reason="Validated child schema.",
    )
    with pytest.raises(CampaignError, match="unsuccessful/unrun dependency"):
        allocate(store, approved, attempt)
    store.revoke(
        approved["campaign_id"],
        authority=approved["authority"],
        actor=ACTOR,
        reason="Revoked future dispatch authorization.",
    )
    with pytest.raises(AuthorityError, match="revoked or expired"):
        allocate(store, approved, attempt)
    assert store.attempt(attempt["id"])["state"] == "validated"
    store.close()


@pytest.mark.parametrize(
    "limits,allocation",
    [
        ({"max_wall_seconds": 5.0}, {}),
        ({"max_cpu_core_seconds": 5.0}, {}),
        ({"max_gpus": 1, "max_gpu_seconds": 5.0}, {"gpu_count": 1}),
    ],
)
def test_consumption_limits_are_distinct_from_concurrent_allocation(
    limits, allocation, tmp_path
):
    store, approved = fixture_campaign(tmp_path, **limits)
    _, attempt = task_attempt(store, approved)
    with pytest.raises(BudgetExceeded):
        allocate(store, approved, attempt, **allocation)
    assert not store.accounting(approved["campaign_id"])
    store.close()


@pytest.mark.parametrize(
    "limits,second_allocation",
    [
        ({"max_memory_mb": 255}, {}),
        ({"max_scratch_mb": 15}, {}),
        ({"max_tasks": 1}, {}),
    ],
)
def test_aggregate_ram_scratch_and_task_count_are_reserved_atomically(
    limits, second_allocation, tmp_path
):
    store, approved = fixture_campaign(
        tmp_path, max_concurrent_workers=2, max_cpu_cores=2, **limits
    )
    _, first = task_attempt(store, approved, "first")
    _, second = task_attempt(store, approved, "second")
    allocate(store, approved, first)
    with pytest.raises(BudgetExceeded):
        allocate(store, approved, second, **second_allocation)
    assert len(store.accounting(approved["campaign_id"])) == 1
    store.close()


def test_classified_retries_use_new_attempts_and_stop_after_two_additional_attempts(
    tmp_path,
):
    store, approved = fixture_campaign(tmp_path, permitted_retries=2)
    task, attempt = task_attempt(store, approved)
    previous = None
    for number in range(1, 4):
        if previous is not None:
            attempt = store.new_attempt(
                task["task_id"],
                authority=approved["authority"],
                actor=ACTOR,
                parent_attempt_id=previous["id"],
                compatibility_digest=task["task_key"],
                automatic=True,
                failure_class="engine_interrupted",
            )
            attempt = store.transition(
                attempt["id"],
                attempt["revision"],
                "validated",
                authority=approved["authority"],
                actor=ACTOR,
                reason=(
                    "Retained exact model after observed controlled process failure."
                ),
            )
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(); sys.exit(7)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started = time.monotonic()
        try:
            attempt = running(store, approved, attempt, pid=child.pid)
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 7
            attempt = advance_worker(
                store, advance_worker(store, attempt, "collecting"), "validating"
            )
            actual = tmp_path / f"attempt-{number}.log"

            def write_observed_failure():
                actual.write_bytes(stdout + stderr)

            previous = store.publish(
                attempt["id"],
                attempt["revision"],
                {
                    "kind": "controlled_failure_process",
                    "observed_exit": child.returncode,
                },
                MeasuredUsage(
                    wall_seconds=time.monotonic() - started,
                    measurement_source=(
                        "observed parent monotonic; terminated-child CPU unavailable"
                    ),
                ),
                write_observed_failure,
                state="failed",
                lease_token=attempt["lease_token"],
                lease_generation=attempt["lease_generation"],
                actor=ACTOR,
            )
            assert previous["number"] == number and previous["state"] == "failed"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    with pytest.raises(CampaignError, match="at most two additional attempts"):
        store.new_attempt(
            task["task_id"],
            authority=approved["authority"],
            actor=ACTOR,
            parent_attempt_id=previous["id"],
            compatibility_digest=task["task_key"],
            automatic=True,
            failure_class="engine_interrupted",
        )
    assert len(store.accounting(approved["campaign_id"])) == 3
    store.close()


def test_revocation_stops_live_worker_mutations_but_preserves_reconciliation_authority(
    tmp_path,
):
    store, approved = fixture_campaign(tmp_path)
    _, draft = task_attempt(store, approved)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        attempt = running(store, approved, draft, pid=child.pid)
        store.revoke(
            approved["campaign_id"],
            authority=approved["authority"],
            actor=ACTOR,
            reason="Revoked live mutation authority.",
        )
        with pytest.raises(AuthorityError, match="revoked or expired"):
            store.heartbeat(
                attempt["id"],
                lease_token=attempt["lease_token"],
                lease_generation=attempt["lease_generation"],
            )
        with pytest.raises(AuthorityError, match="revoked or expired"):
            advance_worker(store, attempt, "collecting")
        child.communicate(timeout=10)
        result = store.reconcile(
            attempt["id"],
            MeasuredUsage(
                wall_seconds=time.monotonic() - started,
                measurement_source="observed parent monotonic; CPU unavailable",
            ),
            authority=approved["authority"],
            actor=ACTOR,
            reason="Reconciled observed stopped revoked worker.",
        )
        assert result["state"] == "failed"
        assert store.accounting(approved["campaign_id"])[0]["state"] == "settled"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        store.close()


def test_database_and_live_sqlite_sidecars_are_private(tmp_path):
    import stat

    store, approved = fixture_campaign(tmp_path)
    for name in ("campaign.sqlite", "campaign.sqlite-wal", "campaign.sqlite-shm"):
        path = tmp_path / name
        assert path.is_file(), name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, name
    assert approved["authority"] not in json.dumps(
        store.events(approved["campaign_id"])
    )
    assert (
        approved["authority"].encode()
        not in (tmp_path / "campaign.sqlite-wal").read_bytes()
    )
    store.close()


@pytest.mark.parametrize("expiration", ["lease", "approval"])
def test_publication_callback_cannot_commit_after_actual_authority_expiration(
    tmp_path, expiration
):
    started = time.monotonic()
    approval_deadline = time.time() + (1.5 if expiration == "approval" else 60)
    store, approved = fixture_campaign(tmp_path, expires_at_unix=approval_deadline)
    _, draft = task_attempt(store, approved)
    attempt = running(
        store,
        approved,
        draft,
        heartbeat_seconds=0.05,
        lease_seconds=1.0 if expiration == "lease" else 60,
    )
    attempt = advance_worker(store, attempt, "collecting")
    attempt = advance_worker(store, attempt, "validating")
    deadline = attempt["lease_expires"] if expiration == "lease" else approval_deadline
    artifact = tmp_path / "actual-publication-time.json"

    def actual_filesystem_publication():
        entered_at = time.time()
        assert entered_at < deadline
        # Observe real time passing inside publication; neither clocks nor
        # engines are replaced by a test double.
        time.sleep(max(0.0, deadline - time.time()) + 0.05)
        artifact.write_text(
            json.dumps({"entered_at": entered_at, "written_at": time.time()})
        )

    try:
        with pytest.raises(AuthorityError, match="expired"):
            store.publish(
                attempt["id"],
                attempt["revision"],
                {"kind": "software", "filename": artifact.name},
                MeasuredUsage(
                    wall_seconds=time.monotonic() - started,
                    measurement_source="observed setup time before publication",
                ),
                actual_filesystem_publication,
                state="succeeded",
                lease_token=attempt["lease_token"],
                lease_generation=attempt["lease_generation"],
                actor=ACTOR,
            )
        observation = json.loads(artifact.read_text())
        assert observation["entered_at"] < deadline < observation["written_at"]
        # Filesystem writes cannot be rolled back with SQLite. Retain those
        # authentic bytes, but reject success and keep resources unreconciled.
        retained = store.attempt(attempt["id"])
        assert retained["state"] == "validating"
        assert retained["revision"] == attempt["revision"]
        assert retained["result_json"] is None
        assert store.accounting(approved["campaign_id"])[0]["state"] == "reserved"
        assert not any(
            event["kind"] == "transition"
            and json.loads(event["details_json"])["to"] == "succeeded"
            for event in store.events(approved["campaign_id"])
        )
        store.connection.execute("BEGIN")
        try:
            with pytest.raises(AuthorityError, match="publication transaction"):
                store.committed_attempt_for_admission(
                    attempt["id"], expected_revision=attempt["revision"]
                )
        finally:
            store.connection.execute("ROLLBACK")
    finally:
        store.close()


@pytest.mark.parametrize(
    ("source_state", "supersede"),
    [("succeeded", False), ("partial", False), ("failed", False), ("partial", True)],
)
def test_merger_admission_checks_committed_latest_source_in_same_transaction(
    tmp_path, source_state, supersede
):
    started_wall, started_cpu = time.monotonic(), time.process_time()
    store, approved = fixture_campaign(tmp_path, permitted_retries=1)
    task, draft = task_attempt(store, approved, "actual source arithmetic")
    source_attempt = running(store, approved, draft)
    source_attempt = advance_worker(store, source_attempt, "collecting")
    source_attempt = advance_worker(store, source_attempt, "validating")
    original = tmp_path / "source-arithmetic.txt"
    original.write_text(str(sum(range(101))))
    published_source = tmp_path / "published-source-arithmetic.txt"
    source_result = {
        "evidence_class": "software_process_check",
        "filename": published_source.name,
        "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    source_record = store.publish(
        source_attempt["id"],
        source_attempt["revision"],
        source_result,
        observed_usage(started_wall, started_cpu),
        lambda: original.rename(published_source),
        state=source_state,
        lease_token=source_attempt["lease_token"],
        lease_generation=source_attempt["lease_generation"],
        actor=ACTOR,
    )
    if supersede:
        resumed = store.new_attempt(
            task["task_id"],
            authority=approved["authority"],
            actor=ACTOR,
            parent_attempt_id=source_record["id"],
            compatibility_digest=task["task_key"],
        )
        assert resumed["id"] != source_record["id"]
    with pytest.raises(AuthorityError, match="publication transaction"):
        store.committed_attempt_for_admission(
            source_record["id"], expected_revision=source_record["revision"]
        )
    store.connection.execute("BEGIN")
    try:
        with pytest.raises(AuthorityError, match="publication transaction"):
            store.committed_attempt_for_admission(
                source_record["id"], expected_revision=source_record["revision"]
            )
    finally:
        store.connection.execute("ROLLBACK")
    _, merger_draft = task_attempt(store, approved, "actual merger admission")
    merger = running(store, approved, merger_draft)
    merger = advance_worker(store, merger, "collecting")
    merger = advance_worker(store, merger, "validating")
    merged = tmp_path / "actual-admission-inspection.json"

    def inspect_committed_source():
        with pytest.raises(AuthorityError, match="Recursive"):
            store.publish(
                merger["id"],
                merger["revision"],
                {"evidence_class": "software_process_check"},
                observed_usage(started_wall, started_cpu),
                inspect_committed_source,
                state="succeeded",
                lease_token=merger["lease_token"],
                lease_generation=merger["lease_generation"],
                actor=ACTOR,
            )
        with pytest.raises(RevisionConflict):
            store.committed_attempt_for_admission(
                source_record["id"], expected_revision=source_record["revision"] - 1
            )
        options = {"expected_revision": source_record["revision"]}
        admissible = source_state in {"succeeded", "partial"} and not supersede
        if source_state != "succeeded":
            with pytest.raises(AuthorityError, match="committed result"):
                store.committed_attempt_for_admission(source_record["id"], **options)
        if admissible:
            retained = store.committed_attempt_for_admission(
                source_record["id"], allow_partial=True, **options
            )
            assert retained["result"] == source_result
            assert retained["lease_generation"] == source_attempt["lease_generation"]
            assert retained["campaign_id"] == approved["campaign_id"]
            merged.write_text(json.dumps(retained))
        else:
            with pytest.raises(AuthorityError, match="committed result"):
                store.committed_attempt_for_admission(
                    source_record["id"], allow_partial=True, **options
                )
            merged.write_text(json.dumps({"admitted": False, "state": source_state}))

    try:
        record = store.publish(
            merger["id"],
            merger["revision"],
            {"evidence_class": "software_process_check", "filename": merged.name},
            observed_usage(started_wall, started_cpu),
            inspect_committed_source,
            state="succeeded",
            lease_token=merger["lease_token"],
            lease_generation=merger["lease_generation"],
            actor=ACTOR,
        )
        assert record["state"] == "succeeded" and merged.is_file()
        assert published_source.read_text() == "5050"
        assert store.attempt(source_record["id"]) == source_record
    finally:
        store.close()


def test_actual_immediate_exit_is_recorded_and_settled_without_inventing_process_birth(
    tmp_path,
):
    store, approved = fixture_campaign(tmp_path)
    _, attempt = task_attempt(store, approved)
    attempt = allocate(store, approved, attempt)
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    assert child.wait(timeout=10) == 7
    dispatched = store.record_dispatch(
        attempt["id"],
        attempt["revision"],
        pid=child.pid,
        authority=approved["authority"],
        actor=ACTOR,
    )
    assert dispatched["state"] == "queued"
    assert dispatched["pid"] == child.pid
    assert dispatched["process_start"] is None
    assert store.accounting(approved["campaign_id"])[0]["dispatched"] == 1
    cancelled = store.cancel(
        dispatched["id"],
        dispatched["revision"],
        authority=approved["authority"],
        actor=ACTOR,
        reason="Observed immediate child failure before lease acquisition.",
    )
    store.reconcile(
        cancelled["id"],
        MeasuredUsage(
            wall_seconds=time.monotonic() - started,
            measurement_source=(
                "actual Popen/wait/monotonic observation; CPU unavailable"
            ),
        ),
        authority=approved["authority"],
        actor=ACTOR,
        reason=(
            "Reconciled the actual exited child without claiming scientific execution."
        ),
    )
    reservation = store.accounting(approved["campaign_id"])[0]
    assert reservation["state"] == "settled" and reservation["charged_wall"] > 0
    store.close()
