"""Genuine SQLite/WAL/process recovery checks; no synthetic scientific evidence."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cochem.orchestration.campaign import (
    Allocation,
    AuthorityError,
    CampaignBudget,
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
)
from cochem.orchestration.campaign_backup import (
    CampaignBackupError,
    adopt_campaign_schema_version,
    create_campaign_backup,
    inspect_campaign_database,
    restore_campaign_backup,
    verify_campaign_backup,
)

ACTOR = f"local-os-user:{os.getuid()}"
SESSION = "actual-process-check-session"
REASON = "Recover actual SQLite history; worker liveness and usage remain unobserved."


def _budget():
    return CampaignBudget(
        max_wall_seconds=120.0,
        max_cpu_core_seconds=120.0,
        max_tasks=100,
        max_concurrent_workers=2,
        max_cpu_cores=2,
        max_memory_mb=512,
        max_scratch_mb=64,
        expires_at_unix=time.time() + 120,
        allowed_engines=["software-process-check"],
        allowed_recipes=["integer-arithmetic-process"],
    )


def _store(tmp_path):
    store = CampaignCoordinator(tmp_path / "live.sqlite")
    approved = store.create_campaign(
        {"purpose": "actual mathematical/software checks"}, _budget(), actor=ACTOR
    )
    return store, approved


def _attempt(store, approved, label="one"):
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
        reason="Validated real mathematical process configuration.",
    )


def _backup(tmp_path, name="backup"):
    return create_campaign_backup(
        tmp_path / "live.sqlite",
        tmp_path / name,
        source_session=SESSION,
        actor=ACTOR,
        reason=REASON,
    )


def _restore(tmp_path, backup, **changes):
    values = {
        "expected_manifest_sha256": backup.receipt["manifest_sha256"],
        "recovery_authority": backup.recovery_authority,
        "source_session": SESSION,
        "actor": ACTOR,
        "reason": REASON,
    }
    values.update(changes)
    return restore_campaign_backup(
        tmp_path / "backup", tmp_path / "restored.sqlite", **values
    )


def _adopt(tmp_path, backup, **changes):
    values = {
        "expected_manifest_sha256": backup.receipt["manifest_sha256"],
        "recovery_authority": backup.recovery_authority,
        "source_session": SESSION,
        "actor": ACTOR,
        "reason": REASON,
    }
    values.update(changes)
    return adopt_campaign_schema_version(
        tmp_path / "live.sqlite", tmp_path / "backup", **values
    )


def test_online_snapshot_includes_committed_wal_and_has_private_bounded_receipt(
    tmp_path,
):
    store, approved = _store(tmp_path)
    store.connection.execute("PRAGMA wal_autocheckpoint=0")
    _, attempt = _attempt(store, approved)
    assert (tmp_path / "live.sqlite-wal").stat().st_size > 0
    before = inspect_campaign_database(tmp_path / "live.sqlite")
    backup = _backup(tmp_path)
    manifest = verify_campaign_backup(
        tmp_path / "backup", expected_manifest_sha256=backup.receipt["manifest_sha256"]
    )
    assert manifest["inventory"] == before
    assert manifest["inventory"]["tables"]["campaign_attempts"]["rows"] == 1
    assert store.attempt(attempt["id"])["state"] == "validated"
    assert store.connection.execute("SELECT revoked FROM campaigns").fetchone()[0] == 0
    assert (tmp_path / "backup").stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == 0o600 for path in (tmp_path / "backup").iterdir()
    )
    public = json.dumps(backup.receipt)
    assert approved["authority"] not in public
    assert backup.recovery_authority not in public
    assert ACTOR not in public and SESSION not in public
    assert str(tmp_path) not in public
    assert backup.recovery_authority not in repr(backup)
    store.close()


def test_restore_fences_running_worker_and_preserves_unsettled_resources(tmp_path):
    store, approved = _store(tmp_path)
    task, attempt = _attempt(store, approved)
    queued = store.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(wall_seconds=10.0, cores=1, memory_mb=128, scratch_mb=8),
        authority=approved["authority"],
        actor=ACTOR,
    )
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        running = store.lease(
            queued["id"],
            queued["revision"],
            pid=process.pid,
            authority=approved["authority"],
            actor=ACTOR,
        )
        reservation_before = store.accounting(approved["campaign_id"])
        generation_before = store.connection.execute(
            "SELECT next_generation FROM campaign_tasks WHERE id=?", (task["task_id"],)
        ).fetchone()[0]
        backup = _backup(tmp_path)
        receipt = _restore(tmp_path, backup)
        assert receipt["cancelled_unfinished_attempts"] == 1
        assert receipt["worker_liveness_observed"] is False
        assert receipt["accounting_released"] is False
        assert process.poll() is None
        # The source continues under its original legitimate lease untouched.
        store.heartbeat(
            running["id"],
            lease_token=running["lease_token"],
            lease_generation=running["lease_generation"],
        )
        assert store.attempt(running["id"])["state"] == "running"
        recovered = CampaignCoordinator(tmp_path / "restored.sqlite")
        try:
            row = recovered.attempt(running["id"])
            assert row["state"] == "cancelled"
            assert row["revision"] == running["revision"] + 1
            assert recovered.accounting(approved["campaign_id"]) == reservation_before
            assert (
                recovered.connection.execute(
                    "SELECT next_generation FROM campaign_tasks WHERE id=?",
                    (task["task_id"],),
                ).fetchone()[0]
                == generation_before + 1
            )
            with pytest.raises(AuthorityError, match="revoked"):
                recovered.heartbeat(
                    running["id"],
                    lease_token=running["lease_token"],
                    lease_generation=running["lease_generation"],
                )
            with pytest.raises(AuthorityError, match="revoked"):
                recovered.new_attempt(
                    task["task_id"], authority=approved["authority"], actor=ACTOR
                )
            usage = MeasuredUsage(
                wall_seconds=0.0,
                measurement_source=(
                    "No usage assertion: process is still alive; "
                    "reconciliation must reject."
                ),
            )
            with pytest.raises(CampaignError, match="process is alive"):
                recovered.reconcile(
                    running["id"],
                    usage,
                    authority=approved["authority"],
                    actor=ACTOR,
                    reason=REASON,
                )
            events = recovered.events(approved["campaign_id"])
            recovery = [
                event
                for event in events
                if event["kind"] == "offline_recovery_revocation"
            ]
            assert len(recovery) == 1
            assert (
                json.loads(recovery[0]["details_json"])["worker_liveness_observed"]
                is False
            )
            # New, explicit campaign approval is independent of recovered tokens.
            new = recovered.create_campaign(
                {"purpose": "new actual process campaign"}, _budget(), actor=ACTOR
            )
            assert new["authority"] != approved["authority"]
            _attempt(recovered, new, label="fresh")
        finally:
            recovered.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
        store.close()


def test_stopped_process_requires_actual_measured_reconciliation(tmp_path):
    store, approved = _store(tmp_path)
    _, attempt = _attempt(store, approved)
    queued = store.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(wall_seconds=10.0, cores=1, memory_mb=128, scratch_mb=8),
        authority=approved["authority"],
        actor=ACTOR,
    )
    start = time.monotonic()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    running = store.lease(
        queued["id"],
        queued["revision"],
        pid=process.pid,
        authority=approved["authority"],
        actor=ACTOR,
    )
    backup = _backup(tmp_path)
    _restore(tmp_path, backup)
    process.terminate()
    process.wait(timeout=10)
    usage = MeasuredUsage(
        wall_seconds=time.monotonic() - start,
        measurement_source=(
            "Actual monotonic elapsed interval through observed Popen termination; "
            "CPU/GPU unavailable."
        ),
    )
    recovered = CampaignCoordinator(tmp_path / "restored.sqlite")
    try:
        with pytest.raises(AuthorityError):
            recovered.reconcile(
                running["id"],
                usage,
                authority=backup.recovery_authority,
                actor=ACTOR,
                reason=REASON,
            )
        recovered.reconcile(
            running["id"],
            usage,
            authority=approved["authority"],
            actor=ACTOR,
            reason="Observed actual owned process termination.",
        )
        assert recovered.accounting(approved["campaign_id"])[0]["state"] == "settled"
        assert recovered.attempt(running["id"])["state"] == "cancelled"
    finally:
        recovered.close()
        store.close()


def test_completed_results_events_and_reservations_are_preserved_exactly(tmp_path):
    store, approved = _store(tmp_path)
    _, attempt = _attempt(store, approved)
    queued = store.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(wall_seconds=10.0, cores=1, memory_mb=128, scratch_mb=8),
        authority=approved["authority"],
        actor=ACTOR,
    )
    running = store.lease(
        queued["id"],
        queued["revision"],
        pid=os.getpid(),
        authority=approved["authority"],
        actor=ACTOR,
    )
    start_wall, start_cpu = time.monotonic(), time.process_time()
    for state in ("collecting", "validating"):
        running = {
            **running,
            **store.worker_transition(
                running["id"],
                running["revision"],
                state,
                lease_token=running["lease_token"],
                lease_generation=running["lease_generation"],
                actor=ACTOR,
                reason=f"Actual integer arithmetic result entered {state}.",
            ),
        }
    result_file = tmp_path / "actual-arithmetic.txt"
    mathematical_result = str(sum(range(101)))
    result = {
        "kind": "mathematical-process",
        "sha256": hashlib.sha256(mathematical_result.encode()).hexdigest(),
    }
    store.publish(
        running["id"],
        running["revision"],
        result,
        MeasuredUsage(
            wall_seconds=time.monotonic() - start_wall,
            cpu_core_seconds=time.process_time() - start_cpu,
            measurement_source="Actual process-time/monotonic differences.",
        ),
        lambda: result_file.write_text(mathematical_result),
        state="succeeded",
        lease_token=running["lease_token"],
        lease_generation=running["lease_generation"],
        actor=ACTOR,
    )
    terminal_before = dict(
        store.connection.execute("SELECT * FROM campaign_attempts").fetchone()
    )
    accounting_before = store.accounting(approved["campaign_id"])
    events_before = store.events(approved["campaign_id"])
    backup = _backup(tmp_path)
    receipt = _restore(tmp_path, backup)
    recovered = CampaignCoordinator(tmp_path / "restored.sqlite")
    try:
        assert (
            dict(
                recovered.connection.execute(
                    "SELECT * FROM campaign_attempts"
                ).fetchone()
            )
            == terminal_before
        )
        assert recovered.accounting(approved["campaign_id"]) == accounting_before
        assert (
            recovered.events(approved["campaign_id"])[: len(events_before)]
            == events_before
        )
        assert result_file.read_text() == "5050"
        assert receipt["cancelled_unfinished_attempts"] == 0
    finally:
        recovered.close()
        store.close()


@pytest.mark.parametrize(
    "field,value,message",
    [
        (
            "recovery_authority",
            "unrelated-operator-or-worker-token",
            "recovery capability",
        ),
        ("source_session", "another-source-session", "source session"),
        ("expected_manifest_sha256", "0" * 64, "trusted pin"),
        ("expected_source_logical_sha256", "0" * 64, "stale"),
        ("actor", "", "actor"),
        ("reason", "", "reason"),
    ],
)
def test_restore_rejects_untrusted_authority_or_source_binding(
    tmp_path, field, value, message
):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    with pytest.raises(CampaignBackupError, match=message):
        _restore(tmp_path, backup, **{field: value})
    assert not (tmp_path / "restored.sqlite").exists()
    store.close()


@pytest.mark.parametrize("artifact", ["campaign.sqlite", "manifest.json"])
def test_tampered_snapshot_or_manifest_is_rejected(tmp_path, artifact):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    path = tmp_path / "backup" / artifact
    with path.open("ab") as stream:
        stream.write(b"actual file tampering")
    with pytest.raises(CampaignBackupError, match="changed|trusted pin"):
        _restore(tmp_path, backup)
    assert not (tmp_path / "restored.sqlite").exists()
    store.close()


def test_schema_or_foreign_key_damage_cannot_be_admitted_as_backup(tmp_path):
    store, approved = _store(tmp_path)
    _attempt(store, approved)
    store.connection.execute("CREATE TABLE unrelated_shape(id TEXT)")
    with pytest.raises(CampaignBackupError, match="schema"):
        _backup(tmp_path)
    store.connection.execute("DROP TABLE unrelated_shape")
    store.connection.execute("PRAGMA foreign_keys=OFF")
    store.connection.execute("UPDATE campaign_tasks SET campaign_id='missing-parent'")
    with pytest.raises(CampaignBackupError, match="foreign-key"):
        _backup(tmp_path)
    assert not (tmp_path / "backup").exists()
    store.close()


def test_unknown_user_version_or_plan_digest_is_rejected(tmp_path):
    store, _ = _store(tmp_path)
    store.connection.execute("PRAGMA user_version=9000")
    with pytest.raises(CampaignBackupError, match="schema/version"):
        _backup(tmp_path)
    store.connection.execute("PRAGMA user_version=0")
    store.connection.execute("UPDATE campaigns SET plan_json='{}'")
    with pytest.raises(CampaignBackupError, match="immutable digest"):
        _backup(tmp_path)
    store.close()


@pytest.mark.parametrize(
    "kind", ["existing-target", "target-symlink", "parent-symlink", "orphan-wal"]
)
def test_restore_never_overwrites_or_follows_symlinks(tmp_path, kind):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    target = tmp_path / "restored.sqlite"
    if kind == "existing-target":
        target.write_bytes(b"keep original content")
    elif kind == "target-symlink":
        target.symlink_to(tmp_path / "live.sqlite")
    elif kind == "orphan-wal":
        Path(str(target) + "-wal").write_bytes(b"unrelated sidecar")
    else:
        parent = tmp_path / "linked"
        parent.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(CampaignBackupError, match="symlink"):
            restore_campaign_backup(
                tmp_path / "backup",
                parent / "restored.sqlite",
                expected_manifest_sha256=backup.receipt["manifest_sha256"],
                recovery_authority=backup.recovery_authority,
                source_session=SESSION,
                actor=ACTOR,
                reason=REASON,
            )
        store.close()
        return
    with pytest.raises(CampaignBackupError, match="absent|symlink"):
        _restore(tmp_path, backup)
    if kind == "existing-target":
        assert target.read_bytes() == b"keep original content"
    store.close()


def test_public_permissions_and_unexpected_bundle_files_are_rejected(tmp_path):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    (tmp_path / "backup" / "campaign.sqlite").chmod(0o644)
    with pytest.raises(CampaignBackupError, match="private"):
        _restore(tmp_path, backup)
    (tmp_path / "backup" / "campaign.sqlite").chmod(0o600)
    (tmp_path / "backup" / "extra").write_text("extra")
    with pytest.raises(CampaignBackupError, match="unexpected"):
        _restore(tmp_path, backup)
    store.close()


def test_known_schema_adoption_is_atomic_and_rollback_is_fenced(tmp_path):
    store, approved = _store(tmp_path)
    _, attempt = _attempt(store, approved)
    before = inspect_campaign_database(tmp_path / "live.sqlite")
    backup = _backup(tmp_path)
    receipt = _adopt(tmp_path, backup)
    assert receipt["from_user_version"] == 0 and receipt["to_user_version"] == 1
    assert receipt["before_inventory"] == before
    assert receipt["before_inventory"]["tables"] == {
        key: value
        for key, value in receipt["after_inventory"]["tables"].items()
        if key != "campaign_recovery_metadata"
    }
    assert store.attempt(attempt["id"])["state"] == "validated"
    assert store.connection.execute("SELECT revoked FROM campaigns").fetchone()[0] == 0
    restored = _restore(tmp_path, backup)
    assert restored["restored_inventory"]["user_version"] == 0
    assert restored["campaign_approvals_revoked"] == 1
    second = _backup(tmp_path, name="version-one-backup")
    assert (
        verify_campaign_backup(
            tmp_path / "version-one-backup",
            expected_manifest_sha256=second.receipt["manifest_sha256"],
        )["inventory"]["user_version"]
        == 1
    )
    store.close()


def test_migration_rejects_source_changed_since_backup(tmp_path):
    store, approved = _store(tmp_path)
    backup = _backup(tmp_path)
    _attempt(store, approved)
    changed = inspect_campaign_database(tmp_path / "live.sqlite")
    with pytest.raises(CampaignBackupError, match="changed"):
        _adopt(tmp_path, backup)
    assert inspect_campaign_database(tmp_path / "live.sqlite") == changed
    assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 0
    store.close()


def _pause_at_phase(pipe, wanted, phase):
    if phase == wanted:
        pipe.send(phase)
        os.kill(os.getpid(), signal.SIGSTOP)


def _crash_operation(tmp_path, receipt, authority, operation, pipe):
    common = dict(
        expected_manifest_sha256=receipt["manifest_sha256"],
        recovery_authority=authority,
        source_session=SESSION,
        actor=ACTOR,
        reason=REASON,
    )
    if operation == "migration":
        adopt_campaign_schema_version(
            tmp_path / "live.sqlite",
            tmp_path / "backup",
            **common,
            progress=lambda phase: _pause_at_phase(
                pipe, "schema_adoption_before_commit", phase
            ),
        )
    elif operation == "restore":
        restore_campaign_backup(
            tmp_path / "backup",
            tmp_path / "restored.sqlite",
            **common,
            progress=lambda phase: _pause_at_phase(
                pipe, "restore_fenced_before_publication", phase
            ),
        )
    else:
        create_campaign_backup(
            tmp_path / "live.sqlite",
            tmp_path / "crashed-backup",
            source_session=SESSION,
            actor=ACTOR,
            reason=REASON,
            progress=lambda phase: _pause_at_phase(
                pipe, "backup_verified_before_publication", phase
            ),
        )


@pytest.mark.parametrize("operation", ["backup", "restore", "migration"])
def test_actual_process_crash_never_publishes_partial_or_commits_migration(
    tmp_path, operation
):
    store, approved = _store(tmp_path)
    _attempt(store, approved)
    backup = _backup(tmp_path)
    before = inspect_campaign_database(tmp_path / "live.sqlite")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_crash_operation,
        args=(tmp_path, backup.receipt, backup.recovery_authority, operation, child),
    )
    process.start()
    try:
        assert parent.poll(30), (
            "actual child did not reach the recovery transaction phase"
        )
        parent.recv()
        process.kill()
        process.join(timeout=10)
        assert process.exitcode == -signal.SIGKILL
        assert not (tmp_path / "crashed-backup").exists()
        assert not (tmp_path / "restored.sqlite").exists()
        assert inspect_campaign_database(tmp_path / "live.sqlite") == before
        assert (
            verify_campaign_backup(
                tmp_path / "backup",
                expected_manifest_sha256=backup.receipt["manifest_sha256"],
            )["inventory"]
            == before
        )
        # Interrupted private stages are not authoritative and are never discovered
        # or resumed automatically; only the named verified bundle is admissible.
        assert all(
            path.stat().st_mode & 0o777 == 0o700
            for path in tmp_path.glob(".campaign-*-*")
        )
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()
        store.close()


def _writer(path, approved, ready, complete):
    store = CampaignCoordinator(path)
    store.connection.execute("PRAGMA wal_autocheckpoint=0")
    ready.set()
    for index in range(20):
        _attempt(store, approved, label=f"real-concurrent-{index}")
    store.close()
    complete.set()


def test_multiprocess_live_wal_writer_and_snapshot_have_consistent_inventory(tmp_path):
    store, approved = _store(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready, complete = context.Event(), context.Event()
    writer = context.Process(
        target=_writer, args=(tmp_path / "live.sqlite", approved, ready, complete)
    )
    writer.start()
    try:
        assert ready.wait(30)
        backup = _backup(tmp_path)
        manifest = verify_campaign_backup(
            tmp_path / "backup",
            expected_manifest_sha256=backup.receipt["manifest_sha256"],
        )
        count = manifest["inventory"]["tables"]["campaign_attempts"]["rows"]
        assert 0 <= count <= 20
        with sqlite3.connect(tmp_path / "backup" / "campaign.sqlite") as snapshot:
            assert snapshot.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert snapshot.execute("PRAGMA foreign_key_check").fetchall() == []
            assert (
                snapshot.execute("SELECT count(*) FROM campaign_tasks").fetchone()[0]
                >= count
            )
            assert snapshot.execute("SELECT count(*) FROM campaigns").fetchone()[0] == 1
        assert complete.wait(30)
        writer.join(timeout=10)
        assert writer.exitcode == 0
        assert (
            store.connection.execute(
                "SELECT count(*) FROM campaign_attempts"
            ).fetchone()[0]
            == 20
        )
    finally:
        if writer.is_alive():
            writer.kill()
            writer.join(timeout=10)
        store.close()


def _racing_backup(path, source, gate, result):
    try:
        backup = create_campaign_backup(
            source,
            path,
            source_session=SESSION,
            actor=ACTOR,
            reason=REASON,
            progress=lambda _: gate.wait(timeout=15),
        )
        result.put(("published", backup.receipt))
    except CampaignBackupError:
        result.put(("rejected", None))


def test_multiprocess_backup_publication_cannot_overwrite_a_rival(tmp_path):
    store, _ = _store(tmp_path)
    context = multiprocessing.get_context("spawn")
    gate = context.Barrier(2)
    results = context.Queue()
    children = [
        context.Process(
            target=_racing_backup,
            args=(tmp_path / "backup", tmp_path / "live.sqlite", gate, results),
        )
        for _ in range(2)
    ]
    for child in children:
        child.start()
    observed = [results.get(timeout=30) for _ in children]
    for child in children:
        child.join(timeout=10)
        assert child.exitcode == 0
    assert sorted(item[0] for item in observed) == ["published", "rejected"]
    published = next(item[1] for item in observed if item[0] == "published")
    verify_campaign_backup(
        tmp_path / "backup", expected_manifest_sha256=published["manifest_sha256"]
    )
    store.close()


def test_backup_source_symlink_and_duplicate_publication_are_rejected(tmp_path):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    with pytest.raises(CampaignBackupError, match="already exists"):
        _backup(tmp_path)
    linked = tmp_path / "linked.sqlite"
    linked.symlink_to(tmp_path / "live.sqlite")
    with pytest.raises(CampaignBackupError, match="symlink"):
        create_campaign_backup(
            linked,
            tmp_path / "second",
            source_session=SESSION,
            actor=ACTOR,
            reason=REASON,
        )
    assert (
        verify_campaign_backup(
            tmp_path / "backup",
            expected_manifest_sha256=backup.receipt["manifest_sha256"],
        )["snapshot_sha256"]
        == backup.receipt["snapshot_sha256"]
    )
    store.close()


def test_copy_of_snapshot_at_a_different_location_is_not_a_migration_target(tmp_path):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    alternate = tmp_path / "different.sqlite"
    shutil.copyfile(tmp_path / "backup" / "campaign.sqlite", alternate)
    alternate.chmod(0o600)
    with pytest.raises(CampaignBackupError, match="another source location"):
        adopt_campaign_schema_version(
            alternate,
            tmp_path / "backup",
            expected_manifest_sha256=backup.receipt["manifest_sha256"],
            recovery_authority=backup.recovery_authority,
            source_session=SESSION,
            actor=ACTOR,
            reason=REASON,
        )
    assert inspect_campaign_database(alternate)["user_version"] == 0
    store.close()


def test_schema_adoption_rejects_replaced_source_inode(tmp_path):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    store.close()
    copied = tmp_path / "copy.sqlite"
    shutil.copyfile(tmp_path / "backup" / "campaign.sqlite", copied)
    copied.chmod(0o600)
    copied.replace(tmp_path / "live.sqlite")
    with pytest.raises(CampaignBackupError, match="file identity changed"):
        _adopt(tmp_path, backup)


def test_version_one_snapshot_restores_and_second_adoption_is_rejected(tmp_path):
    store, _ = _store(tmp_path)
    first = _backup(tmp_path, name="before-adoption")
    adopt_campaign_schema_version(
        tmp_path / "live.sqlite",
        tmp_path / "before-adoption",
        expected_manifest_sha256=first.receipt["manifest_sha256"],
        recovery_authority=first.recovery_authority,
        source_session=SESSION,
        actor=ACTOR,
        reason=REASON,
    )
    current = _backup(tmp_path)
    with pytest.raises(CampaignBackupError, match="unversioned-to-v1"):
        _adopt(tmp_path, current)
    restored = _restore(tmp_path, current)
    assert restored["restored_inventory"]["user_version"] == 1
    assert (
        restored["restored_inventory"]["tables"]["campaign_recovery_metadata"]
        == current.receipt["inventory"]["tables"]["campaign_recovery_metadata"]
    )
    store.close()


@pytest.mark.parametrize(
    "kind", ["task-key", "missing-dependency", "event-lineage", "reservation-lineage"]
)
def test_semantically_conflicting_sqlite_rows_are_not_qualified(tmp_path, kind):
    store, approved = _store(tmp_path)
    task, attempt = _attempt(store, approved)
    if kind == "task-key":
        store.connection.execute("UPDATE campaign_tasks SET task_key=?", ("0" * 64,))
    elif kind == "missing-dependency":
        store.connection.execute(
            "UPDATE campaign_tasks SET dependencies_json='[\"absent-task\"]'"
        )
    elif kind == "event-lineage":
        store.connection.execute(
            "INSERT INTO campaign_events(campaign_id,revision,actor,timestamp,"
            "kind,reason,details_json) VALUES('absent-campaign',0,?,?,'actual-damage',"
            "'actual corruption','{}')",
            (ACTOR, time.time()),
        )
    else:
        other = store.create_campaign(
            {"purpose": "another actual software campaign"}, _budget(), actor=ACTOR
        )
        store.reserve(
            attempt["id"],
            attempt["revision"],
            Allocation(wall_seconds=10.0, cores=1, memory_mb=128, scratch_mb=8),
            authority=approved["authority"],
            actor=ACTOR,
        )
        store.connection.execute(
            "UPDATE campaign_reservations SET campaign_id=?", (other["campaign_id"],)
        )
    with pytest.raises(CampaignBackupError, match="immutable|identity|lineage"):
        _backup(tmp_path)
    assert not (tmp_path / "backup").exists()
    store.close()


def test_repinning_a_malformed_manifest_does_not_authorize_unknown_fields(tmp_path):
    store, _ = _store(tmp_path)
    backup = _backup(tmp_path)
    path = tmp_path / "backup" / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.pop("source_file_identity_sha256")
    path.write_text(json.dumps(manifest))
    repin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CampaignBackupError, match="manifest fields"):
        _restore(tmp_path, backup, expected_manifest_sha256=repin)
    store.close()
