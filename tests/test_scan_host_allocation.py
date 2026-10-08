"""Genuine native worker admission, common-ledger competition and death proofs."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import sqlite3
import time
from pathlib import Path

import psutil
import pytest
from pydantic import ValidationError

from cochem.orchestration.campaign import CampaignCoordinator, CampaignError
from cochem.orchestration.host_allocation import (
    HostAllocationPolicy,
    HostAllocationRequest,
    HostAuthorityError,
    HostCapacityError,
    bootstrap_host_allocation,
)
from cochem_torq.application import _execute_admitted_request, validate_request
from cochem_torq.artifacts import verify_shard
from cochem_torq.domain import CalculationRequest, digest, read_json
from cochem_torq.pes_storage import _source_receipt
from cochem_torq.scan import (
    ApprovedScanExecutor,
    HostAllocationProvenance,
    ScanPointResult,
    _verify_host_provenance,
)
from cochem_torq.service import ApprovedPlan, approve_plan, plan_request

ROOT = Path(__file__).resolve().parents[1]
HOST_ACTOR = f"local-os-user:{os.getuid()}"


def _scan_request(*, scf_max_cycle=100):
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["designed-H2-H1", "designed-H2-H2"],
            },
            "recipe": "hf-sto-3g-pes-validation",
            "products": ["pes_scan"],
            "resources": {"cores": 1, "memory_mb": 512, "wall_seconds": 90},
            "source_provenance": {
                "pes_scan": {
                    "coordinates": [
                        {
                            "coordinate_id": "H-H",
                            "kind": "bond",
                            "atom_indices": [0, 1],
                            "unit": "bohr",
                            "domain": {
                                "minimum": 1.2,
                                "maximum": 1.6,
                                "periodic": False,
                            },
                        }
                    ],
                    "grid": [[1.4]],
                    "sampling_strategy": "full_grid",
                    "passes": [
                        {"purpose": purpose, "sample_indices": [0]}
                        for purpose in ("forward", "reverse", "challenge")
                    ],
                    "coordinate_treatment": "fixed",
                    "unscanned_coordinates": "frozen_cartesian",
                    "additional_constraints": [],
                    "initial_guess_policy": "independent_pyscf_minao",
                    "budget": {"max_physical_calls": 3, "per_point_wall_seconds": 20},
                    "energy_recheck_tolerance_hartree": 1e-9,
                    "density_recheck_tolerance": 1e-8,
                    "scf_max_cycle": scf_max_cycle,
                }
            },
        }
    )


def _approved(request, *, scratch_bytes=64 * 1024**2):
    return approve_plan(
        plan_request(request, execution="local_validation"),
        actor=f"actual-host-integration-os:{os.getuid()}",
        max_scratch_bytes=scratch_bytes,
    )


def _bootstrap(directory, scratch_root, *, max_workers=1):
    return bootstrap_host_allocation(
        actor=HOST_ACTOR,
        directory=directory,
        policy=HostAllocationPolicy(
            max_cpu_cores=1,
            max_memory_mb=4096,
            max_scratch_mb=256,
            max_workers=max_workers,
            gpu_ids=[],
            scratch_probe_root=str(scratch_root),
        ),
    )


def _ordinary_request(*, constrained=False):
    data = {
        "molecule": _scan_request().molecule.model_dump(mode="json"),
        "recipe": "hf-sto-3g-education",
        "products": ["geometry", "harmonic", "equilibrium_constants"],
        "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
        "source_provenance": {"producer": "explicitly_designed_H2_process_regression"},
    }
    if constrained:
        data.update(
            recipe="hf-sto-3g-constrained-pes-validation",
            products=["constrained_geometry"],
            source_provenance={
                "constrained_optimization": {
                    "constraints": [
                        {
                            "coordinate": _scan_request().source_provenance["pes_scan"][
                                "coordinates"
                            ][0],
                            "target": 1.4,
                        }
                    ],
                    "movable_atom_indices": [1],
                    "max_physical_calls": 4,
                    "per_evaluation_wall_seconds": 20,
                    "request_curvature": False,
                }
            },
        )
    return CalculationRequest.model_validate(data)


def _ordinary_driver(approved, request, destination, host_directory, output):
    try:
        model = CalculationRequest.model_validate(request)
        result = _execute_admitted_request(
            validate_request(model, execution="local_validation"),
            model,
            destination,
            ApprovedPlan.model_validate(approved),
            host_allocation_directory=host_directory,
        )
        output.put(("result", result))
    except Exception as exc:
        output.put(("error", {"type": type(exc).__name__, "message": str(exc)}))


def _running_worker(ledger, driver, *, permit=None):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = ledger.connection.execute(
            "SELECT * FROM allocations WHERE state='running'"
        ).fetchone()
        if row is not None and (permit is None or permit.is_file()):
            return dict(row)
        assert driver.is_alive(), "Actual driver exited before observed dispatch."
        time.sleep(0.01)
    raise AssertionError("Actual native dispatch was not observed.")


def _wait_actual_dead(pid, process_start, *, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            worker = psutil.Process(pid)
            if (
                worker.create_time() != process_start
                or worker.status() == psutil.STATUS_ZOMBIE
            ):
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.01)
    raise AssertionError("The actual native worker did not terminate.")


def _stop_exact_worker(row):
    try:
        worker = psutil.Process(row["pid"])
        if (
            worker.create_time() == row["process_start"]
            and worker.uids().effective == os.getuid()
            and worker.status() != psutil.STATUS_ZOMBIE
        ):
            worker.kill()
    except psutil.NoSuchProcess:
        pass


@pytest.fixture(scope="module")
def actual_host_scan(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-host-scan")
    host_directory = root / "shared-host"
    ledger = _bootstrap(host_directory, root)
    ledger.close()
    request = _scan_request()
    approved = _approved(request)
    with ApprovedScanExecutor(
        approved, root / "scan", host_allocation_directory=host_directory
    ) as executor:
        point = executor.evaluate(0, purpose="forward")
        assert point.status == "available", point.reason
        campaign_db = (
            executor.workspace.parent
            / f".{executor.workspace.name}.scan-campaign.sqlite"
        )
    return root, host_directory, approved, point, campaign_db


@pytest.mark.real_engine
def test_actual_native_scan_has_exact_terminal_host_resource_provenance(
    actual_host_scan,
):
    root, host_directory, _, point, _ = actual_host_scan
    provenance = point.host_allocation
    assert provenance is not None
    assert provenance.campaign_id == point.campaign_id
    assert provenance.attempt_id == point.attempt_id
    assert provenance.state == "reconciled"
    assert provenance.process_death_proof == "actual_pid_absent"
    ledger = bootstrap_host_allocation(actor=HOST_ACTOR, directory=host_directory)
    try:
        _verify_host_provenance(ledger, provenance)
        receipt = ledger.allocation(provenance.allocation_id)
        assert digest(receipt) == provenance.terminal_receipt_sha256
        assert ledger.accounting()["active_allocations"] == 0
        assert receipt["request"]["cores"] == 1
        assert receipt["request"]["memory_mb"] == 512
        assert point.native_manifest_path is not None
        assert (root / "scan" / point.native_manifest_path).is_file()
        assert point.energy_hartree is not None
        assert point.independent_scientific_qualification is False
        private = (ledger.path.parent / "operator-capability.txt").read_text()
        assert private not in point.model_dump_json()
        assert "lease_hash" not in point.model_dump_json()
    finally:
        ledger.close()


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "field", ["policy_sha256", "terminal_receipt_sha256", "worker_pid"]
)
def test_changed_host_provenance_cannot_be_admitted_as_actual_source(
    actual_host_scan, field
):
    _, host_directory, approved, point, campaign_db = actual_host_scan
    raw = point.model_dump(mode="json")
    assert raw["host_allocation"] is not None
    raw["host_allocation"][field] = (
        raw["host_allocation"][field] + 1 if field == "worker_pid" else "0" * 64
    )
    tampered = ScanPointResult.model_validate(raw)
    ledger = bootstrap_host_allocation(actor=HOST_ACTOR, directory=host_directory)
    coordinator = CampaignCoordinator(campaign_db)
    try:
        assert tampered.host_allocation is not None
        with pytest.raises(ValueError, match="durable receipt"):
            _verify_host_provenance(ledger, tampered.host_allocation)
        with pytest.raises(CampaignError, match="exact coordinator commit"):
            _source_receipt(
                coordinator,
                tampered,
                ApprovedPlan.model_validate(approved),
                transactional=False,
            )
    finally:
        ledger.close()
        coordinator.close()


@pytest.mark.real_engine
def test_legacy_absent_optional_provenance_serializes_without_inventing_a_record(
    actual_host_scan,
):
    _, _, approved, point, campaign_db = actual_host_scan
    legacy_shape = point.model_dump(mode="json")
    legacy_shape.pop("host_allocation")
    loaded = ScanPointResult.model_validate(legacy_shape)
    assert loaded.host_allocation is None
    assert loaded.model_dump(mode="json") == legacy_shape
    # Readability does not authorize omission from a genuine new source commit.
    coordinator = CampaignCoordinator(campaign_db)
    try:
        with pytest.raises(CampaignError, match="exact coordinator commit"):
            _source_receipt(
                coordinator,
                loaded,
                ApprovedPlan.model_validate(approved),
                transactional=False,
            )
    finally:
        coordinator.close()


@pytest.mark.real_engine
def test_host_record_from_another_attempt_is_rejected(actual_host_scan):
    _, _, _, point, _ = actual_host_scan
    raw = point.model_dump(mode="json")
    raw["host_allocation"]["attempt_id"] = "different-operational-attempt"
    with pytest.raises(ValidationError, match="another campaign attempt"):
        ScanPointResult.model_validate(raw)


def test_host_capacity_failure_cancels_undispatched_campaign_without_native_observation(
    tmp_path,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    hold = tmp_path / "actual-owned-held-workspace"
    hold.mkdir(mode=0o700)
    acquired = ledger.acquire(
        campaign_id="actual-resource-owner",
        attempt_id="actual-resource-attempt",
        workspace=hold,
        request=HostAllocationRequest(cores=1, memory_mb=512, scratch_mb=64),
        pid=os.getpid(),
    )
    try:
        with ApprovedScanExecutor(
            _approved(_scan_request()),
            tmp_path / "blocked",
            host_allocation_directory=host_directory,
        ) as executor:
            with pytest.raises(HostCapacityError):
                executor.evaluate(0, purpose="forward")
            assert executor.points == ()
            assert not list(executor.workspace.glob("**/wavefunction.chk"))
            attempts = executor._queue.connection.execute(
                "SELECT state FROM campaign_attempts"
            ).fetchall()
            assert [row[0] for row in attempts] == ["cancelled"]
            accounting = executor._queue.accounting(executor._receipt["campaign_id"])
            assert accounting[0]["state"] == "settled"
            assert accounting[0]["charged_wall"] == 0.0
            assert accounting[0]["dispatched"] == 0
            assert (
                "operational setup"
                in json.loads(accounting[0]["usage_json"])["measurement_source"]
            )
        assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.finish(
            acquired.receipt["allocation_id"],
            acquired.receipt["revision"],
            acquired.lease_token,
            reason="Actual held resource software check completed.",
        )
        ledger.close()


@pytest.mark.real_engine
def test_actual_scratch_failure_keeps_native_quantities_absent_and_releases_dead_child(
    tmp_path,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    ledger.close()
    approved = _approved(_scan_request(), scratch_bytes=1024**2)
    with ApprovedScanExecutor(
        approved, tmp_path / "quota", host_allocation_directory=host_directory
    ) as executor:
        (executor.workspace / "actual-byte-allocation.bin").write_bytes(b"x" * 1024**2)
        point = executor.evaluate(0, purpose="forward")
        assert point.status == "failed"
        assert point.energy_hartree is None
        assert point.gradient_hartree_bohr is None
        assert point.host_allocation is not None
        assert "scratch" in point.reason.lower()
        assert executor._host_ledger.accounting()["active_allocations"] == 0


def _scan_driver(approved, workspace, host_directory, gate, output):
    try:
        with ApprovedScanExecutor(
            approved, workspace, host_allocation_directory=host_directory
        ) as executor:
            if gate is not None:
                gate.wait(timeout=30)
            point = executor.evaluate(0, purpose="forward")
            output.put(("point", point.model_dump(mode="json")))
    except HostCapacityError:
        output.put(("capacity-rejected", None))
    except Exception as exc:
        output.put(("error", {"type": type(exc).__name__, "message": str(exc)}))


@pytest.mark.real_engine
def test_two_actual_scan_campaigns_share_one_host_slot(tmp_path):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    approved = _approved(_scan_request())
    context = multiprocessing.get_context("spawn")
    gate, output = context.Barrier(2), context.Queue()
    children = [
        context.Process(
            target=_scan_driver,
            args=(approved, tmp_path / f"scan-{index}", host_directory, gate, output),
        )
        for index in range(2)
    ]
    for child in children:
        child.start()
    try:
        results = [output.get(timeout=90) for _ in children]
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(item[0] for item in results) == ["capacity-rejected", "point"], (
            results
        )
        actual = next(item[1] for item in results if item[0] == "point")
        assert actual["status"] == "available", actual["reason"]
        assert actual["host_allocation"]["process_death_proof"] == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
        assert (
            ledger.connection.execute("SELECT count(*) FROM allocations").fetchone()[0]
            == 1
        )
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=10)
        ledger.close()


@pytest.mark.real_engine
def test_real_native_worker_sigkill_is_not_success_and_resources_follow_actual_wait(
    tmp_path,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    approved = _approved(_scan_request())
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    workspace = tmp_path / "victim"
    driver = context.Process(
        target=_scan_driver, args=(approved, workspace, host_directory, None, output)
    )
    driver.start()
    try:
        deadline = time.monotonic() + 30
        victim_pid = None
        while time.monotonic() < deadline:
            row = ledger.connection.execute(
                "SELECT pid FROM allocations WHERE state='running'"
            ).fetchone()
            if (
                row is not None
                and (workspace / ".point-0000.work" / "dispatch-permit.json").is_file()
            ):
                victim_pid = row[0]
                break
            if not driver.is_alive():
                break
            time.sleep(0.01)
        assert victim_pid is not None, "actual native dispatch was not observed"
        os.kill(victim_pid, signal.SIGKILL)
        status, result = output.get(timeout=60)
        driver.join(timeout=10)
        assert driver.exitcode == 0
        assert status == "point", result
        assert result["status"] == "failed"
        assert result["energy_hartree"] is None
        assert result["gradient_hartree_bohr"] is None
        assert result["host_allocation"]["process_death_proof"] == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if driver.is_alive():
            driver.kill()
            driver.join(timeout=10)
        ledger.close()


@pytest.mark.real_engine
def test_ordinary_local_spectroscopy_worker_uses_the_same_terminal_host_proof(tmp_path):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    request_data = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request_data["resources"].update(cores=1, memory_mb=1024, wall_seconds=120)
    model = CalculationRequest.model_validate(request_data)
    checked = validate_request(model, execution="local_validation")
    approved = ApprovedPlan.model_validate(_approved(model))
    destination = tmp_path / "actual-water"
    try:
        result = _execute_admitted_request(
            checked,
            model,
            destination,
            approved,
            host_allocation_directory=host_directory,
        )
        assert result["status"] == "complete", result["errors"]
        record = HostAllocationProvenance.model_validate(
            result["campaign"]["host_allocation"]
        )
        _verify_host_provenance(ledger, record)
        assert record.campaign_id == result["campaign"]["campaign_id"]
        assert record.attempt_id == result["campaign"]["attempt_id"]
        assert record.process_death_proof == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
        verify_shard(destination)
    finally:
        ledger.close()


def test_ordinary_spectroscopy_cannot_bypass_another_campaigns_host_reservation(
    tmp_path,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    workspace = tmp_path / "held"
    workspace.mkdir(mode=0o700)
    acquired = ledger.acquire(
        campaign_id="actual-independent-resource-campaign",
        attempt_id="actual-independent-resource-attempt",
        workspace=workspace,
        request=HostAllocationRequest(cores=1, memory_mb=512, scratch_mb=64),
        pid=os.getpid(),
    )
    request_data = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request_data["resources"].update(cores=1, memory_mb=1024, wall_seconds=120)
    model = CalculationRequest.model_validate(request_data)
    checked = validate_request(model, execution="local_validation")
    approved = ApprovedPlan.model_validate(_approved(model))
    try:
        with pytest.raises(HostCapacityError):
            _execute_admitted_request(
                checked,
                model,
                tmp_path / "blocked-spectroscopy",
                approved,
                host_allocation_directory=host_directory,
            )
        assert not (tmp_path / "blocked-spectroscopy").exists()
        assert not list(tmp_path.glob("**/wavefunction.chk"))
        with sqlite3.connect(tmp_path / ".torq-campaign.sqlite") as campaign:
            assert (
                campaign.execute("SELECT state FROM campaign_attempts").fetchone()[0]
                == "cancelled"
            )
            assert (
                campaign.execute("SELECT state FROM campaign_reservations").fetchone()[
                    0
                ]
                == "settled"
            )
        assert ledger.accounting()["active_allocations"] == 1
    finally:
        ledger.finish(
            acquired.receipt["allocation_id"],
            acquired.receipt["revision"],
            acquired.lease_token,
            reason="Actual current resource owner completed.",
        )
        ledger.close()


@pytest.mark.real_engine
@pytest.mark.parametrize("dispatch", ["scan", "spectroscopy"])
def test_actual_native_exit_during_sqlite_blocked_host_heartbeat_is_collectable(
    tmp_path,
    dispatch,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    workspace = tmp_path / "actual-worker"
    if dispatch == "scan":
        approved = _approved(_scan_request())
        driver = context.Process(
            target=_scan_driver,
            args=(approved, workspace, host_directory, None, output),
        )
        permit = workspace / ".point-0000.work" / "dispatch-permit.json"
    else:
        model = _ordinary_request()
        approved = _approved(model)
        driver = context.Process(
            target=_ordinary_driver,
            args=(
                approved,
                model.model_dump(mode="json"),
                workspace,
                host_directory,
                output,
            ),
        )
        permit = None
    driver.start()
    row = None
    blocker = None
    try:
        row = _running_worker(ledger, driver, permit=permit)
        os.kill(row["pid"], signal.SIGSTOP)
        blocker = sqlite3.connect(ledger.path, isolation_level=None, timeout=15)
        blocker.execute("BEGIN IMMEDIATE")
        # Real SQLite contention forces the real host heartbeat to wait while
        # the actual engine is paused. Scientific code is never substituted.
        time.sleep(1.5)
        assert driver.is_alive()
        os.kill(row["pid"], signal.SIGCONT)
        _wait_actual_dead(
            row["pid"],
            row["process_start"],
            timeout=6 if dispatch == "scan" else 12,
        )
        blocker.rollback()
        blocker.close()
        blocker = None
        status, result = output.get(timeout=60)
        driver.join(timeout=10)
        assert driver.exitcode == 0
        assert status == ("point" if dispatch == "scan" else "result"), result
        assert result["status"] == (
            "available" if dispatch == "scan" else "complete"
        ), result
        assert (
            ledger.connection.execute(
                "SELECT count(*) FROM events WHERE kind='worker_exit_observed'"
            ).fetchone()[0]
            == 1
        )
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if row is not None:
            try:
                os.kill(row["pid"], signal.SIGCONT)
            except ProcessLookupError:
                pass
        if driver.is_alive():
            if row is not None:
                _stop_exact_worker(row)
            driver.join(timeout=10)
            if driver.is_alive():
                driver.kill()
                driver.join(timeout=10)
        ledger.close()


@pytest.mark.real_engine
def test_actual_constrained_children_have_independent_host_death_proofs(tmp_path):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    model = _ordinary_request(constrained=True)
    destination = tmp_path / "actual-constrained"
    try:
        result = _execute_admitted_request(
            validate_request(model, execution="local_validation"),
            model,
            destination,
            ApprovedPlan.model_validate(_approved(model)),
            host_allocation_directory=host_directory,
        )
        assert result["status"] == "complete", result["errors"]
        proof = HostAllocationProvenance.model_validate(
            result["campaign"]["host_allocation"]
        )
        _verify_host_provenance(ledger, proof)
        assert len(proof.nested_process_observations) >= 2
        for child in proof.nested_process_observations:
            assert child.owner_binding_sha256 is not None
            assert child.parent_observation_sha256 is not None
            assert child.worker_wait_sha256 is not None
            assert child.process_death_proof == "actual_pid_absent"
        assert ledger.accounting()["active_allocations"] == 0
        verify_shard(destination)
    finally:
        ledger.close()


@pytest.mark.real_engine
@pytest.mark.parametrize("loss", ["receipts", "whole_inventory"])
def test_missing_real_nested_launch_receipts_keep_resources_and_costs_unknown(
    tmp_path,
    loss,
):
    host_directory = tmp_path / "shared"
    ledger = _bootstrap(host_directory, tmp_path)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    model = _ordinary_request(constrained=True)
    driver = context.Process(
        target=_ordinary_driver,
        args=(
            _approved(model),
            model.model_dump(mode="json"),
            tmp_path / "victim",
            host_directory,
            output,
        ),
    )
    driver.start()
    row = None
    original = {}
    try:
        row = _running_worker(ledger, driver)
        assert (
            json.loads(row["request_json"])["child_inventory_scope"]
            == "constrained-evaluation-v1"
        )
        evaluation = (
            Path(row["workspace"]) / "constrained-engine/evaluations/evaluation-0000"
        )
        deadline = time.monotonic() + 30
        while not (evaluation / "owner-binding.json").is_file():
            assert time.monotonic() < deadline
            assert driver.is_alive()
            time.sleep(0.005)
        # Stop the actual owner so it cannot produce a later wait receipt;
        # the genuine independently sessioned native child remains observable.
        os.kill(row["pid"], signal.SIGSTOP)
        child = read_json(evaluation / "owner-binding.json")
        for name in (
            "owner-binding.json",
            "parent-process-observation.json",
            "worker-process.json",
        ):
            path = evaluation / name
            if path.exists():
                original[name] = path.read_bytes()
                # Deliberate evidence loss, never replacement physical results.
                path.unlink()
        if loss == "whole_inventory":
            shutil.rmtree(evaluation.parent)
        os.kill(row["pid"], signal.SIGKILL)
        _wait_actual_dead(child["worker_pid"], child["worker_create_time"], timeout=10)
        status, result = output.get(timeout=60)
        driver.join(timeout=10)
        assert driver.exitcode == 0
        assert status == "error", result
        assert result["type"] == "HostAuthorityError"
        assert "unknown" in result["message"]
        assert ledger.accounting()["active_allocations"] == 1
        current = ledger.allocation(row["id"])
        assert current["resource_peaks"] is None
        with sqlite3.connect(tmp_path / ".torq-campaign.sqlite") as campaign:
            assert (
                campaign.execute("SELECT state FROM campaign_attempts").fetchone()[0]
                == "cancelled"
            )
            assert (
                campaign.execute("SELECT state FROM campaign_reservations").fetchone()[
                    0
                ]
                == "reserved"
            )
            assert (
                campaign.execute(
                    "SELECT usage_json FROM campaign_reservations"
                ).fetchone()[0]
                is None
            )
        # Restore the exact observed receipts, then explicit operator recovery
        # must independently prove each actual child dead before release.
        if loss == "whole_inventory":
            # Restoring only part of a deleted native tree would invent an
            # incomplete recovery inventory. It stays held without that proof.
            assert "inventory" in result["message"]
            with pytest.raises(HostAuthorityError, match="inventory"):
                ledger.reconcile(
                    row["id"],
                    current["revision"],
                    reason="Missing entire native inventory cannot prove "
                    "child termination.",
                )
            assert ledger.accounting()["active_allocations"] == 1
            return
        for name, raw in original.items():
            path = evaluation / name
            path.write_bytes(raw)
            path.chmod(0o600)
        recovered = ledger.reconcile(
            row["id"],
            current["revision"],
            reason="Exact genuine lost receipts restored; each original PID "
            "independently observed dead.",
        )
        assert recovered["resource_peaks"] is None
        assert recovered["final_observation"]["nested_process_observations"]
        assert ledger.accounting()["active_allocations"] == 0
    finally:
        if row is not None:
            try:
                os.kill(row["pid"], signal.SIGCONT)
            except ProcessLookupError:
                pass
        if driver.is_alive():
            if row is not None:
                _stop_exact_worker(row)
            driver.join(timeout=10)
            if driver.is_alive():
                driver.kill()
                driver.join(timeout=10)
        ledger.close()
