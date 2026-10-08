"""Actual shared SQLite/process host admission; no synthetic capacity probes."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from pydantic import ValidationError

from cochem.orchestration.host_allocation import (
    MIB,
    HostAllocationError,
    HostAllocationLedger,
    HostAllocationPolicy,
    HostAllocationRequest,
    HostAuthorityError,
    HostCapacityError,
    HostRevisionConflict,
    ObservedHostUsage,
    bootstrap_host_allocation,
    probe_host_resources,
)

ACTOR = f"local-os-user:{os.getuid()}"


def _policy(tmp_path, **changes):
    probe = probe_host_resources(tmp_path)
    values = dict(
        max_cpu_cores=min(2, probe["cpu_cores"]),
        max_memory_mb=16,
        max_scratch_mb=8,
        max_workers=2,
        gpu_ids=[],
        scratch_probe_root=str(tmp_path),
    )
    values.update(changes)
    return HostAllocationPolicy(**values)


def _ledger(tmp_path, **changes):
    policy = _policy(tmp_path, **changes)
    created = HostAllocationLedger.initialize(
        tmp_path / "ledger.sqlite", policy, actor=ACTOR
    )
    return HostAllocationLedger(
        tmp_path / "ledger.sqlite", authority=created.authority, actor=ACTOR
    ), created


def _workspace(tmp_path, name="work"):
    path = tmp_path / name
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def _acquire(ledger, tmp_path, label="one", **changes):
    values = dict(
        campaign_id=f"campaign-{label}",
        attempt_id=f"attempt-{label}",
        workspace=_workspace(tmp_path, name=f"work-{label}"),
        request=HostAllocationRequest(cores=1, memory_mb=2, scratch_mb=1),
        pid=os.getpid(),
    )
    values.update(changes)
    return ledger.acquire(**values)


def _finish(ledger, acquired, **changes):
    values = dict(
        reason="Actual owned worker completed its bounded software operation."
    )
    values.update(changes)
    return ledger.finish(
        acquired.receipt["allocation_id"],
        acquired.receipt["revision"],
        acquired.lease_token,
        **values,
    )


def test_actual_probe_and_private_bootstrap_preserve_reviewed_policy(tmp_path):
    policy = _policy(tmp_path)
    root = tmp_path / "common-private-root"
    first = bootstrap_host_allocation(actor=ACTOR, directory=root, policy=policy)
    try:
        probe = probe_host_resources(tmp_path)
        assert probe["cpu_cores"] >= 1
        assert probe["memory_total_mb"] >= policy.max_memory_mb
        assert probe["scratch_free_mb"] >= policy.max_scratch_mb
        assert (
            probe["host"]["boot_id"]
            == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
        second = bootstrap_host_allocation(actor=ACTOR, directory=root)
        try:
            assert first.path == second.path
            assert first.policy == second.policy == policy
            assert first.path.stat().st_mode & 0o777 == 0o600
            assert first.path.parent.stat().st_mode & 0o777 == 0o700
            capability = first.path.parent / "operator-capability.txt"
            assert capability.stat().st_mode & 0o777 == 0o600
            with pytest.raises(HostCapacityError, match="pinned"):
                bootstrap_host_allocation(
                    actor=ACTOR,
                    directory=root,
                    policy=policy.model_copy(update={"max_workers": 1}),
                )
        finally:
            second.close()
    finally:
        first.close()


def test_same_process_finish_keeps_missing_peaks_unavailable_and_fences_terminal(
    tmp_path,
):
    ledger, created = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path)
        assert acquired.receipt["resource_peaks"] is None
        assert acquired.lease_token not in repr(acquired)
        assert created.authority not in repr(created)
        assert acquired.lease_token not in json.dumps(acquired.receipt)
        assert created.authority not in json.dumps(created.receipt)
        assert ledger.accounting()["active_allocations"] == 1
        finished = _finish(ledger, acquired)
        assert finished["state"] == "finished"
        assert finished["resource_peaks"] is None
        assert finished["final_observation"]["process_stopped"] is False
        assert ledger.accounting()["active_allocations"] == 0
        with pytest.raises(HostAuthorityError, match="terminal"):
            ledger.heartbeat(
                finished["allocation_id"], finished["revision"], acquired.lease_token
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            ledger.connection.execute("UPDATE allocations SET state='running'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.connection.execute("DELETE FROM events")
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "resource", ["cores", "memory_mb", "scratch_mb", "worker_slots"]
)
def test_each_shared_host_limit_blocks_a_distinct_campaign(tmp_path, resource):
    probe = probe_host_resources(tmp_path)
    limits = dict(
        max_cpu_cores=min(2, probe["cpu_cores"]),
        max_memory_mb=8,
        max_scratch_mb=4,
        max_workers=2,
    )
    ledger, _ = _ledger(tmp_path, **limits)
    capacities = dict(
        cores=limits["max_cpu_cores"], memory_mb=8, scratch_mb=4, worker_slots=2
    )
    first_request = dict(cores=1, memory_mb=1, scratch_mb=0, worker_slots=1)
    first_request[resource] = capacities[resource]
    second_request = dict(cores=1, memory_mb=1, scratch_mb=0, worker_slots=1)
    if resource == "scratch_mb":
        second_request[resource] = 1
    try:
        first = _acquire(
            ledger,
            tmp_path,
            label="first",
            request=HostAllocationRequest(**first_request),
        )
        before = ledger.accounting()
        with pytest.raises(HostCapacityError, match="capacity is exhausted"):
            _acquire(
                ledger,
                tmp_path,
                label="different",
                request=HostAllocationRequest(**second_request),
            )
        assert ledger.accounting() == before
        _finish(ledger, first)
        second = _acquire(
            ledger,
            tmp_path,
            label="different",
            request=HostAllocationRequest(**second_request),
        )
        _finish(ledger, second)
    finally:
        ledger.close()


def test_current_revision_and_lease_are_required_for_every_worker_mutation(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path)
        with pytest.raises(HostAuthorityError):
            ledger.heartbeat(acquired.receipt["allocation_id"], 0, "unrelated-token")
        updated = ledger.heartbeat(
            acquired.receipt["allocation_id"], 0, acquired.lease_token
        )
        assert updated["revision"] == 1
        with pytest.raises(HostRevisionConflict):
            _finish(ledger, acquired)
        ledger.finish(
            updated["allocation_id"],
            updated["revision"],
            acquired.lease_token,
            reason="Actual CAS heartbeat completed.",
        )
    finally:
        ledger.close()


def test_expired_live_lease_holds_resources_and_cannot_be_reconciled_as_dead(tmp_path):
    ledger, _ = _ledger(tmp_path, max_workers=1, max_cpu_cores=1)
    try:
        acquired = _acquire(ledger, tmp_path, lease_seconds=0.01)
        time.sleep(0.025)
        with pytest.raises(HostAuthorityError, match="expired"):
            ledger.heartbeat(acquired.receipt["allocation_id"], 0, acquired.lease_token)
        with pytest.raises(HostAllocationError, match="still alive"):
            ledger.reconcile(
                acquired.receipt["allocation_id"],
                0,
                reason="Actual liveness observation must retain resources.",
            )
        with pytest.raises(HostCapacityError):
            _acquire(ledger, tmp_path, label="other")
        assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.close()


def test_actual_child_dispatch_wait_and_expired_reconciliation(tmp_path):
    ledger, _ = _ledger(tmp_path)
    process = None
    try:
        acquired = _acquire(ledger, tmp_path, pid=None, lease_seconds=0.1)
        dispatching = ledger.begin_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token
        )
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        running = ledger.bind_dispatch(
            dispatching["allocation_id"],
            dispatching["revision"],
            acquired.lease_token,
            pid=process.pid,
        )
        with pytest.raises(HostAllocationError, match="still alive"):
            ledger.finish(
                running["allocation_id"],
                running["revision"],
                acquired.lease_token,
                reason="A parent cannot declare a live child stopped.",
            )
        with pytest.raises(HostAllocationError, match="still alive"):
            ledger.reconcile(
                running["allocation_id"],
                running["revision"],
                reason="A live process retains resources.",
            )
        process.terminate()
        process.wait(timeout=10)
        time.sleep(0.11)
        reconciled = ledger.reconcile(
            running["allocation_id"],
            running["revision"],
            reason="Actual Popen wait completed; original PID is absent.",
        )
        assert reconciled["state"] == "reconciled"
        assert reconciled["resource_peaks"] is None
        assert reconciled["final_observation"]["proof"] == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        ledger.close()


def test_actual_exit_racing_heartbeat_keeps_exact_authority_and_holds_resources(
    tmp_path,
):
    ledger, _ = _ledger(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    try:
        acquired = _acquire(ledger, tmp_path, pid=None)
        running = ledger.bind_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token, pid=process.pid
        )
        with pytest.raises(HostAllocationError, match="still alive"):
            ledger.observe_worker_exit(
                running["allocation_id"], running["revision"], acquired.lease_token
            )
        process.communicate(b"x", timeout=10)
        with pytest.raises(HostAuthorityError):
            ledger.heartbeat(
                running["allocation_id"], running["revision"], acquired.lease_token
            )
        with pytest.raises(HostAuthorityError):
            ledger.observe_worker_exit(
                running["allocation_id"], running["revision"], "wrong-token"
            )
        observed = ledger.observe_worker_exit(
            running["allocation_id"], running["revision"], acquired.lease_token
        )
        assert observed["state"] == "running"
        assert observed["resource_peaks"] is None
        assert ledger.accounting()["active_allocations"] == 1
        with pytest.raises(HostRevisionConflict):
            ledger.observe_worker_exit(
                running["allocation_id"], running["revision"], acquired.lease_token
            )
        reconciled = ledger.reconcile(
            observed["allocation_id"],
            observed["revision"],
            reason="Actual child wait/death verified after the current-lease race.",
        )
        assert reconciled["final_observation"]["proof"] == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        ledger.close()


def test_expired_authority_is_not_restored_by_genuine_worker_exit(tmp_path):
    ledger, _ = _ledger(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    try:
        acquired = _acquire(ledger, tmp_path, pid=None, lease_seconds=0.1)
        running = ledger.bind_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token, pid=process.pid
        )
        process.communicate(b"x", timeout=10)
        time.sleep(0.11)
        with pytest.raises(HostAuthorityError, match="expired"):
            ledger.observe_worker_exit(
                running["allocation_id"], running["revision"], acquired.lease_token
            )
        assert ledger.accounting()["active_allocations"] == 1
        ledger.reconcile(
            running["allocation_id"],
            running["revision"],
            reason="Only operator actual-death recovery may settle the expired lease.",
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        ledger.close()


def test_actual_zombie_exit_is_verified_before_the_owned_wait(tmp_path):
    ledger, _ = _ledger(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    try:
        acquired = _acquire(ledger, tmp_path, pid=None)
        running = ledger.bind_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token, pid=process.pid
        )
        assert process.stdin is not None
        process.stdin.write(b"x")
        process.stdin.flush()
        deadline = time.monotonic() + 10
        child = psutil.Process(process.pid)
        while child.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # No Popen poll/wait has yet reaped the actual child. Its real zombie
        # identity is stronger evidence than a cached returncode of None.
        assert process.returncode is None
        with pytest.raises(HostAuthorityError, match="not a live"):
            ledger.heartbeat(
                running["allocation_id"], running["revision"], acquired.lease_token
            )
        observed = ledger.observe_worker_exit(
            running["allocation_id"], running["revision"], acquired.lease_token
        )
        event = ledger.connection.execute(
            "SELECT details_json FROM events WHERE kind='worker_exit_observed'"
        ).fetchone()
        assert json.loads(event[0])["observation"]["proof"] == "actual_zombie_state"
        assert ledger.accounting()["active_allocations"] == 1
        assert process.wait(timeout=10) == 0
        reconciled = ledger.reconcile(
            observed["allocation_id"],
            observed["revision"],
            reason="Actual zombie death verified first; genuine owned wait completed.",
        )
        assert reconciled["final_observation"]["proof"] == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if process.stdin is not None:
            process.stdin.close()
        ledger.close()


@pytest.mark.parametrize("inventory", ["absent", "empty"])
def test_pinned_nested_scope_cannot_release_from_absent_or_empty_inventory(
    tmp_path,
    inventory,
):
    ledger, _ = _ledger(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    try:
        workspace = _workspace(tmp_path)
        if inventory == "empty":
            (workspace / "constrained-engine").mkdir(mode=0o700)
            (workspace / "constrained-engine/evaluations").mkdir(mode=0o700)
        acquired = _acquire(
            ledger,
            tmp_path,
            pid=None,
            workspace=workspace,
            request=HostAllocationRequest(
                cores=1,
                memory_mb=2,
                scratch_mb=1,
                child_inventory_scope="constrained-evaluation-v1",
            ),
        )
        running = ledger.bind_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token, pid=process.pid
        )
        process.communicate(b"x", timeout=10)
        with pytest.raises(HostAuthorityError, match="inventory"):
            ledger.finish(
                running["allocation_id"],
                running["revision"],
                acquired.lease_token,
                reason="An authenticated worker cannot invent missing child evidence.",
            )
        with pytest.raises(HostAuthorityError, match="inventory"):
            ledger.reconcile(
                running["allocation_id"],
                running["revision"],
                reason="An absent/empty pinned inventory is no nested death proof.",
            )
        assert ledger.allocation(running["allocation_id"])["state"] == "running"
        assert ledger.accounting()["active_allocations"] == 1
        assert ledger.allocation(running["allocation_id"])["resource_peaks"] is None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        ledger.close()


def test_only_live_owner_can_release_before_dispatch_and_unknown_launch_stays_reserved(
    tmp_path,
):
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path, pid=None)
        released = ledger.release_undispatched(
            acquired.receipt["allocation_id"],
            0,
            acquired.lease_token,
            reason="Current owner completed preflight before any launch.",
        )
        assert released["state"] == "undispatched_released"
        next_one = _acquire(ledger, tmp_path, label="next", pid=None)
        unknown = ledger.begin_dispatch(
            next_one.receipt["allocation_id"], 0, next_one.lease_token
        )
        with pytest.raises(HostAllocationError, match="before any dispatch"):
            ledger.release_undispatched(
                unknown["allocation_id"],
                unknown["revision"],
                next_one.lease_token,
                reason="Cannot manufacture no-launch proof.",
            )
        with pytest.raises(HostAllocationError, match="ownership is unknown"):
            ledger.reconcile(
                unknown["allocation_id"],
                unknown["revision"],
                reason="Dispatch did not bind a provable worker.",
            )
        assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.close()


def test_actual_observed_samples_are_preserved_without_filling_missing_peaks(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path)
        process = psutil.Process(os.getpid())
        actual_samples = [
            process.memory_info().rss / MIB,
            process.memory_info().rss / MIB,
        ]
        usage = ObservedHostUsage(
            peak_memory_mb=max(actual_samples),
            measurement_source=(
                "Maximum of two genuine RSS samples; "
                "lifetime peak and scratch peak unavailable."
            ),
        )
        result = _finish(ledger, acquired, usage=usage)
        assert result["resource_peaks"]["peak_memory_mb"] == max(actual_samples)
        assert result["resource_peaks"]["peak_scratch_mb"] is None
    finally:
        ledger.close()


def test_private_operator_capability_actor_and_host_binding_are_checked(tmp_path):
    ledger, created = _ledger(tmp_path)
    ledger.close()
    with pytest.raises(HostAuthorityError, match="capability"):
        HostAllocationLedger(
            tmp_path / "ledger.sqlite",
            authority="not-the-deployment-token",
            actor=ACTOR,
        )
    with pytest.raises(HostAuthorityError, match="actor"):
        HostAllocationLedger(
            tmp_path / "ledger.sqlite",
            authority=created.authority,
            actor="unrelated-actor",
        )
    with sqlite3.connect(tmp_path / "ledger.sqlite") as damaged:
        host = json.loads(
            damaged.execute("SELECT host_json FROM deployment").fetchone()[0]
        )
        host["boot_id"] = "persisted-corruption-of-actual-boot-binding"
        damaged.execute("UPDATE deployment SET host_json=?", (json.dumps(host),))
    with pytest.raises(HostAuthorityError, match="boot"):
        HostAllocationLedger(
            tmp_path / "ledger.sqlite", authority=created.authority, actor=ACTOR
        )


@pytest.mark.parametrize(
    "kind",
    ["workspace-symlink", "workspace-replaced", "public-ledger", "public-capability"],
)
def test_symlinks_replacements_and_public_private_files_are_rejected(tmp_path, kind):
    if kind == "public-capability":
        ledger = bootstrap_host_allocation(
            actor=ACTOR, directory=tmp_path / "shared", policy=_policy(tmp_path)
        )
        capability = ledger.path.parent / "operator-capability.txt"
        ledger.close()
        capability.chmod(0o644)
        with pytest.raises(HostAuthorityError, match="private"):
            bootstrap_host_allocation(actor=ACTOR, directory=tmp_path / "shared")
        return
    ledger, created = _ledger(tmp_path)
    try:
        if kind == "public-ledger":
            ledger.close()
            ledger.path.chmod(0o644)
            with pytest.raises(HostAuthorityError, match="private"):
                HostAllocationLedger(
                    ledger.path, authority=created.authority, actor=ACTOR
                )
        elif kind == "workspace-symlink":
            path = _workspace(tmp_path)
            linked = tmp_path / "linked"
            linked.symlink_to(path, target_is_directory=True)
            with pytest.raises(HostAuthorityError, match="symlink"):
                _acquire(ledger, tmp_path, workspace=linked)
        else:
            acquired = _acquire(ledger, tmp_path)
            workspace = tmp_path / "work-one"
            workspace.rename(tmp_path / "old-work")
            workspace.mkdir(mode=0o700)
            with pytest.raises(HostAuthorityError, match="identity changed"):
                _finish(ledger, acquired)
            assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.close()


def test_cpu_only_admission_never_asserts_or_grants_fake_gpu_slots(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        cpu = _acquire(ledger, tmp_path)
        _finish(ledger, cpu)
        with pytest.raises(HostCapacityError, match="GPU"):
            _acquire(
                ledger,
                tmp_path,
                label="gpu",
                request=HostAllocationRequest(
                    cores=1,
                    memory_mb=2,
                    scratch_mb=1,
                    gpu_ids=["GPU-unobserved-fiction"],
                ),
            )
        with pytest.raises(HostCapacityError, match="actual probed"):
            HostAllocationLedger.initialize(
                tmp_path / "gpu-policy.sqlite",
                _policy(tmp_path, gpu_ids=["GPU-unobserved-fiction"]),
                actor=ACTOR,
            )
        assert not (tmp_path / "gpu-policy.sqlite").exists()
    finally:
        ledger.close()


@pytest.mark.parametrize("lease", [float("nan"), float("inf"), 0, -1, True, 3601])
def test_invalid_or_unbounded_lease_duration_is_rejected(tmp_path, lease):
    ledger, _ = _ledger(tmp_path)
    try:
        with pytest.raises(HostAllocationError, match="duration"):
            _acquire(ledger, tmp_path, lease_seconds=lease)
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        ledger.close()


def test_request_types_and_missing_usage_are_strict(tmp_path):
    with pytest.raises(ValidationError):
        HostAllocationRequest(cores=True, memory_mb=2, scratch_mb=1)
    with pytest.raises(ValidationError):
        ObservedHostUsage(
            peak_memory_mb=float("nan"), measurement_source="invalid nonfinite input"
        )
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path)
        with pytest.raises(HostAllocationError):
            _acquire(ledger, tmp_path)
        assert ledger.accounting()["active_allocations"] == 1
        _finish(ledger, acquired)
    finally:
        ledger.close()


def _crashing_owner(path, authority, workspace, pipe, bound):
    ledger = HostAllocationLedger(path, authority=authority, actor=ACTOR)
    acquired = ledger.acquire(
        campaign_id="actual-crash-campaign",
        attempt_id="actual-crash-attempt",
        workspace=workspace,
        request=HostAllocationRequest(cores=1, memory_mb=2, scratch_mb=1),
        pid=os.getpid() if bound else None,
    )
    receipt = acquired.receipt
    if not bound:
        receipt = ledger.begin_dispatch(
            receipt["allocation_id"], receipt["revision"], acquired.lease_token
        )
    pipe.send(receipt)
    os.kill(os.getpid(), signal.SIGSTOP)


@pytest.mark.parametrize("bound", [True, False])
def test_actual_sigkill_keeps_resources_until_provable_bound_process_recovery(
    tmp_path, bound
):
    ledger, created = _ledger(tmp_path, max_workers=1, max_cpu_cores=1)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_crashing_owner,
        args=(ledger.path, created.authority, _workspace(tmp_path), child, bound),
    )
    process.start()
    try:
        assert parent.poll(30)
        receipt = parent.recv()
        process.kill()
        process.join(timeout=10)
        assert process.exitcode == -signal.SIGKILL
        assert ledger.accounting()["active_allocations"] == 1
        with pytest.raises(HostCapacityError):
            _acquire(ledger, tmp_path, label="another-campaign")
        if bound:
            reconciled = ledger.reconcile(
                receipt["allocation_id"],
                receipt["revision"],
                reason="Actual child SIGKILL and wait proved original PID absent.",
            )
            assert reconciled["final_observation"]["proof"] == "actual_pid_absent"
            assert ledger.accounting()["active_allocations"] == 0
        else:
            with pytest.raises(HostAllocationError, match="unknown"):
                ledger.reconcile(
                    receipt["allocation_id"],
                    receipt["revision"],
                    reason="Owner death does not prove unbound dispatch ownership.",
                )
            assert ledger.accounting()["active_allocations"] == 1
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()
        ledger.close()


def _racing_acquire(path, authority, workspace, gate, queue, label):
    ledger = HostAllocationLedger(path, authority=authority, actor=ACTOR)
    try:
        gate.wait(timeout=20)
        acquired = ledger.acquire(
            campaign_id=f"separate-campaign-{label}",
            attempt_id=f"separate-attempt-{label}",
            workspace=workspace,
            request=HostAllocationRequest(cores=1, memory_mb=2, scratch_mb=1),
            pid=os.getpid(),
        )
        queue.put(("acquired", acquired.receipt))
    except HostCapacityError:
        queue.put(("capacity-rejected", None))
    finally:
        ledger.close()


def test_multiprocess_distinct_campaigns_cannot_double_allocate_one_host_slot(tmp_path):
    ledger, created = _ledger(tmp_path, max_workers=1, max_cpu_cores=1)
    context = multiprocessing.get_context("spawn")
    gate, queue = context.Barrier(2), context.Queue()
    children = [
        context.Process(
            target=_racing_acquire,
            args=(
                ledger.path,
                created.authority,
                _workspace(tmp_path, name=f"worker-{index}"),
                gate,
                queue,
                str(index),
            ),
        )
        for index in range(2)
    ]
    for child in children:
        child.start()
    try:
        results = [queue.get(timeout=30) for _ in children]
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(item[0] for item in results) == ["acquired", "capacity-rejected"]
        assert ledger.accounting()["reserved_cpu_cores"] == 1
        winning = next(item[1] for item in results if item[0] == "acquired")
        ledger.reconcile(
            winning["allocation_id"],
            winning["revision"],
            reason="Actual winning child completed and parent wait observed its exit.",
        )
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=10)
        ledger.close()


def _racing_heartbeat(path, authority, allocation, token, gate, queue):
    ledger = HostAllocationLedger(path, authority=authority, actor=ACTOR)
    try:
        gate.wait(timeout=20)
        result = ledger.heartbeat(allocation, 0, token)
        queue.put(("updated", result["revision"]))
    except HostRevisionConflict:
        queue.put(("revision-rejected", None))
    finally:
        ledger.close()


def test_real_multiprocess_heartbeat_cas_has_exactly_one_winner(tmp_path):
    ledger, created = _ledger(tmp_path)
    acquired = _acquire(ledger, tmp_path)
    context = multiprocessing.get_context("spawn")
    gate, queue = context.Barrier(2), context.Queue()
    children = [
        context.Process(
            target=_racing_heartbeat,
            args=(
                ledger.path,
                created.authority,
                acquired.receipt["allocation_id"],
                acquired.lease_token,
                gate,
                queue,
            ),
        )
        for _ in range(2)
    ]
    for child in children:
        child.start()
    try:
        results = [queue.get(timeout=30) for _ in children]
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(item[0] for item in results) == ["revision-rejected", "updated"]
        ledger.finish(
            acquired.receipt["allocation_id"],
            1,
            acquired.lease_token,
            reason="Actual CAS race completed in the live bound worker.",
        )
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=10)
        ledger.close()


def _racing_bootstrap(root, policy, gate, queue):
    gate.wait(timeout=20)
    ledger = bootstrap_host_allocation(actor=ACTOR, directory=root, policy=policy)
    try:
        queue.put((str(ledger.path), ledger.policy.model_dump()))
    finally:
        ledger.close()


def test_actual_multiprocess_bootstrap_admits_one_private_capability_database_pair(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    gate, queue = context.Barrier(2), context.Queue()
    root = tmp_path / "shared-root"
    policy = _policy(tmp_path)
    children = [
        context.Process(target=_racing_bootstrap, args=(root, policy, gate, queue))
        for _ in range(2)
    ]
    for child in children:
        child.start()
    results = [queue.get(timeout=30) for _ in children]
    for child in children:
        child.join(timeout=10)
        assert child.exitcode == 0
    assert results[0] == results[1]
    assert len(list(root.glob("*/ledger.sqlite"))) == 1
    assert len(list(root.glob("*/operator-capability.txt"))) == 1
    ledger = bootstrap_host_allocation(actor=ACTOR, directory=root)
    ledger.close()


def test_child_that_exits_before_binding_stays_unknown_without_false_release(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path, pid=None)
        pending = ledger.begin_dispatch(
            acquired.receipt["allocation_id"], 0, acquired.lease_token
        )
        child = subprocess.Popen([sys.executable, "-c", "sum(range(101))"])
        child.wait(timeout=10)
        with pytest.raises(HostAuthorityError, match="identity is unavailable"):
            ledger.bind_dispatch(
                pending["allocation_id"],
                pending["revision"],
                acquired.lease_token,
                pid=child.pid,
            )
        assert ledger.allocation(pending["allocation_id"])["state"] == "dispatching"
        assert ledger.accounting()["active_allocations"] == 1
        with pytest.raises(HostAllocationError, match="unknown"):
            ledger.reconcile(
                pending["allocation_id"],
                pending["revision"],
                reason=(
                    "Unbound process creation identity "
                    "is not established by a bare PID."
                ),
            )
    finally:
        ledger.close()


def test_workspace_permissions_changed_after_acquire_reject_mutation(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        acquired = _acquire(ledger, tmp_path)
        (tmp_path / "work-one").chmod(0o777)
        with pytest.raises(HostAuthorityError, match="ownership is unsafe"):
            _finish(ledger, acquired)
        assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.close()


def test_copied_frozen_models_cannot_bypass_resource_validation(tmp_path):
    ledger, _ = _ledger(tmp_path)
    try:
        invalid = HostAllocationRequest(cores=1, memory_mb=2, scratch_mb=1).model_copy(
            update={"cores": -1}
        )
        with pytest.raises(ValidationError):
            _acquire(ledger, tmp_path, request=invalid)
        assert ledger.accounting()["active_allocations"] == 0
        invalid_policy = _policy(tmp_path).model_copy(update={"max_cpu_cores": -1})
        with pytest.raises(ValidationError):
            HostAllocationLedger.initialize(
                tmp_path / "invalid-policy.sqlite", invalid_policy, actor=ACTOR
            )
        assert not (tmp_path / "invalid-policy.sqlite").exists()
    finally:
        ledger.close()
