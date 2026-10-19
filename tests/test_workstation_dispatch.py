"""Workstation lane: routing, TPO ledger, approved submission and verified ingestion.

Scientific results here are genuine: the returned shard comes from TORQ's own
``worker_execute`` (real PySCF) for the exact submitted request, sealed with
``seal_shard``. Only the workstation job runner's protocol files (status.json
and the results-archive summary) are written by the test, as the runner would;
the runner's result signature is a real Ed25519 signature made with a key
generated here in place of the workstation's.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from cochem_torq.application import validate_request, worker_execute
from cochem_torq.artifacts import seal_shard, verify_shard
from cochem_torq.cli import main
from cochem_torq.domain import CalculationRequest, PrerequisiteError, canonical_json
from cochem_torq.service import approve_plan, plan_request
from cochem_torq.workstation import (
    CANCELLED,
    FAILED,
    INGESTED,
    OPEN_STATES,
    PENDING,
    REJECTED,
    RUNNING,
    WorkstationError,
    WorkstationQueue,
    recommend_execution,
)

WORKSTATION_KEY = Ed25519PrivateKey.generate()  # stands in for the workstation key


def fingerprint(key: Ed25519PrivateKey = WORKSTATION_KEY) -> str:
    raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def attestation(statement: dict, key: Ed25519PrivateKey = WORKSTATION_KEY) -> dict:
    """What the runner's ``Signer.attest`` publishes in status.json."""
    payload = json.dumps(
        {"schema": "cochem.workstation-attestation/1", **statement},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "schema": "cochem.workstation-attestation/1",
        "algorithm": "ed25519",
        "public_key": base64.b64encode(public).decode(),
        "key_fingerprint": hashlib.sha256(public).hexdigest(),
        "statement": base64.b64encode(payload).decode(),
        "signature": base64.b64encode(key.sign(payload)).decode(),
    }


def h2_request(cores: int = 1) -> dict:
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0, 0, 0], [0, 0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry"],
            "resources": {"cores": cores, "memory_mb": 2048, "wall_seconds": 600},
        }
    ).model_dump(mode="json")


def approved(request: dict, *, hours: float = 48) -> dict:
    review = plan_request(request, execution="local_validation")
    return approve_plan(
        review,
        actor="student",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=hours),
    )


def queue(tmp_path: Path, trusted: list[str] | None = None) -> WorkstationQueue:
    (tmp_path / "Drive").mkdir(exist_ok=True)
    station = WorkstationQueue(
        tmp_path / "Drive" / "alice",
        student_id="alice",
        ledger_path=tmp_path / "state" / "tpo_ledger.sqlite",
        trusted_keys=[fingerprint()] if trusted is None else trusted,
    )
    station.assign()
    return station


def pick_up(station: WorkstationQueue, receipt: dict) -> tuple[Path, Path]:
    label = receipt["job_name"] + "__0123abcd"
    job = station.folder / "jobs" / label
    job.mkdir(parents=True)
    (station.folder / "inbox" / receipt["job_name"]).rename(job / "input")
    work = station.folder.parent.parent / "scratch" / label
    shutil.copytree(job / "input", work)
    return job, work


def genuine_shard(work: Path, request: dict) -> Path:
    """What `cochem-torq execute` publishes: a sealed shard for this exact request."""
    shard = work / "result"
    shard.mkdir()
    checked = validate_request(request, execution="local_validation")
    model = CalculationRequest.model_validate(checked["request"])
    result = worker_execute(model, shard)
    (shard / "request.json").write_bytes(canonical_json(checked["request"]))
    (shard / "result.json").write_bytes(canonical_json(result))
    seal_shard(
        shard,
        request_sha256=checked["request_sha256"],
        request_id=str(model.request_id),
        recipe_sha256=checked["recipe"]["recipe_sha256"],
        source_identity=result["source_identity"],
        worker_id="workstation-contract-check",
    )
    return shard


def publish(
    job: Path,
    receipt: dict,
    state: str,
    work: Path | None = None,
    reported: dict | None = None,
    key: Ed25519PrivateKey = WORKSTATION_KEY,
    **extra,
) -> None:
    status = {
        "schema": "cochem.workstation-job-status/1",
        "label": job.name,
        "client_job_id": receipt["client_job_id"],
        "state": state,
        "message": f"workstation reports {state.lower()}",
        "attempts": 1,
        "workstation": "lab-ws",
        "result_file": None,
        **extra,
    }
    if work is not None:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            for path in sorted(work.rglob("*")):
                relative = path.relative_to(work).as_posix()
                if path.is_file() and relative not in {
                    "request.json",
                    "approved_plan.json",
                    "job.json",
                }:
                    bundle.writestr(relative, path.read_bytes())
            bundle.writestr(
                "_runner/job_summary.json",
                json.dumps(
                    {
                        "schema": "cochem.workstation-job-summary/1",
                        "client_job_id": receipt["client_job_id"],
                        "input_hashes": reported
                        or {
                            name: hashlib.sha256((work / name).read_bytes()).hexdigest()
                            for name in ("request.json", "approved_plan.json")
                        },
                    }
                ),
            )
        archive = stream.getvalue()
        status["result_file"] = f"{job.name}_results.zip"
        status["result_sha256"] = hashlib.sha256(archive).hexdigest()
        (job / status["result_file"]).write_bytes(archive)
        inputs = reported or {
            name: hashlib.sha256((work / name).read_bytes()).hexdigest()
            for name in ("request.json", "approved_plan.json", "job.json")
        }
        status["attestation"] = attestation(
            {
                "client_job_id": receipt["client_job_id"],
                "label": job.name,
                "state": state,
                "workstation": "lab-ws",
                "input_hashes": inputs,
                "result_file": status["result_file"],
                "result_sha256": status["result_sha256"],
            },
            key,
        )
    (job / "status.json").write_text(json.dumps(status))


def test_heavy_requests_are_routed_to_the_workstation() -> None:
    assert recommend_execution(h2_request(1))["execution"] == "github_actions"
    heavy = recommend_execution(h2_request(4))
    assert heavy["execution"] == "workstation"
    assert any("2 CPU cores" in reason for reason in heavy["reasons"])
    assert "non-canonical" in heavy["evidence_lane"]


def test_submission_is_ledgered_idempotent_and_bound(tmp_path: Path) -> None:
    station = queue(tmp_path)
    request = h2_request(4)
    plan = approved(request)
    receipt = station.submit(request, approved_plan=plan, idempotency_key="lab-1")
    assert receipt["state"] == PENDING and receipt["execution"] == "workstation"
    inbox = station.folder / "inbox" / receipt["job_name"]
    manifest = json.loads((inbox / "job.json").read_text())
    assert manifest["engine"] == "torq_execute" and manifest["input"] == "request.json"
    assert manifest["resources"] == {"cores": 4, "memory_gb": 2.0, "max_hours": 2}
    assert (
        manifest["student_id"] == "alice"
        and manifest["job_id"] == receipt["client_job_id"]
    )
    checked = validate_request(request, execution="local_validation")
    assert (inbox / "request.json").read_bytes() == canonical_json(checked["request"])
    assert (
        manifest["files"]["approved_plan.json"]
        == hashlib.sha256((inbox / "approved_plan.json").read_bytes()).hexdigest()
    )
    again = station.submit(request, approved_plan=plan, idempotency_key="lab-1")
    assert again["client_job_id"] == receipt["client_job_id"]
    assert len(list((station.folder / "inbox").iterdir())) == 1
    other = h2_request(2)
    with pytest.raises(ValueError, match="different request"):
        station.submit(other, approved_plan=approved(other), idempotency_key="lab-1")
    with pytest.raises(PrerequisiteError, match="24 hours"):
        station.submit(
            other, approved_plan=approved(other, hours=1), idempotency_key="lab-2"
        )
    events = station.ledger.events(receipt["client_job_id"])
    assert [event["state"] for event in events] == [PENDING]
    with pytest.raises(sqlite3.DatabaseError):
        station.ledger.db.execute("DELETE FROM workstation_events")
    station.close()


def test_round_trip_ingests_only_the_verified_shard(tmp_path: Path) -> None:
    station = queue(tmp_path)
    request = h2_request(1)
    receipt = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-3"
    )
    job, work = pick_up(station, receipt)
    publish(job, receipt, "RUNNING", progress={"stage": "scf"})
    assert station.status(receipt["client_job_id"])["state"] == RUNNING
    genuine_shard(work, request)
    publish(job, receipt, "COMPLETED")  # finished, archive not yet published
    assert station.poll(tmp_path / "ingested")[0]["state"] == RUNNING
    publish(job, receipt, "COMPLETED", work)
    outcome = station.poll(tmp_path / "ingested")[0]
    assert outcome["state"] == INGESTED, outcome
    (published,) = [Path(path) for path in outcome["result_paths"]]
    manifest = verify_shard(
        published, expected_request_sha256=receipt["request_sha256"]
    )
    result = json.loads((published / "result.json").read_text())
    assert (
        manifest["request_id"] == receipt["request_id"]
        and result["status"] == "complete"
    )
    states = [
        event["state"] for event in station.ledger.events(receipt["client_job_id"])
    ]
    assert states[0] == PENDING and RUNNING in states and states[-1] == INGESTED
    assert station.poll(tmp_path / "ingested") == []  # nothing left open
    assert station.download(receipt["client_job_id"], tmp_path / "elsewhere") == [
        published
    ]
    station.close()


def test_results_for_another_request_are_rejected(tmp_path: Path) -> None:
    station = queue(tmp_path)
    request = h2_request(1)
    receipt = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-4"
    )
    job, work = pick_up(station, receipt)
    different = h2_request(1)  # same chemistry, different request identity
    genuine_shard(work, different)
    publish(job, receipt, "COMPLETED", work)
    with pytest.raises(WorkstationError, match="different request"):
        station.download(receipt["client_job_id"], tmp_path / "ingested")
    assert station.ledger.get(receipt["client_job_id"])["state"] == REJECTED
    assert not (tmp_path / "ingested" / f"workstation-{receipt['request_id']}").exists()
    station.close()


def test_rejected_and_withdrawn_submissions_are_final(tmp_path: Path) -> None:
    station = queue(tmp_path)
    request = h2_request(1)
    first = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-5"
    )
    job, _work = pick_up(station, first)
    rejected = {
        "attempts": 0,
        "message": "Rejected: Unknown engine 'torq_execute'",
    }
    publish(job, first, "FAILED", **rejected)  # unsigned: anyone could write this
    assert station.status(first["client_job_id"])["state"] == RUNNING
    publish(
        job,
        first,
        "FAILED",
        attestation=attestation(
            {
                "client_job_id": first["client_job_id"],
                "label": job.name,
                "state": "FAILED",
                "input_hashes": {},
                "result_file": None,
                "result_sha256": None,
            }
        ),
        **rejected,
    )
    assert station.status(first["client_job_id"])["state"] == FAILED
    assert station.ledger.get(first["client_job_id"])["attested_by"] == fingerprint()
    second_request = h2_request(1)
    second = station.submit(
        second_request, approved_plan=approved(second_request), idempotency_key="lab-6"
    )
    assert (
        station.cancel(second["client_job_id"], reason="no longer needed")["state"]
        == CANCELLED
    )
    assert not (station.folder / "inbox" / second["job_name"]).exists()
    station.close()


def test_cli_recommend_and_submit(tmp_path: Path, capsys) -> None:
    request = h2_request(4)
    (tmp_path / "request.json").write_text(json.dumps(request))
    assert (
        main(
            [
                "workstation",
                "recommend",
                "--request",
                str(tmp_path / "request.json"),
                "--json",
            ]
        )
        == 0
    )
    assert '"workstation"' in capsys.readouterr().out
    small = h2_request(1)
    (tmp_path / "small.json").write_text(json.dumps(small))
    (tmp_path / "plan.json").write_text(json.dumps(approved(small)))
    (tmp_path / "Drive").mkdir()
    common = [
        "--folder",
        str(tmp_path / "Drive" / "bob"),
        "--student",
        "bob",
        "--ledger",
        str(tmp_path / "tpo.sqlite"),
    ]
    assert main(["workstation", "assign", *common, "--json"]) == 0
    assert (
        main(
            [
                "workstation",
                "submit",
                *common,
                "--request",
                str(tmp_path / "small.json"),
                "--approved-plan",
                str(tmp_path / "plan.json"),
                "--idempotency-key",
                "cli-1",
                "--json",
            ]
        )
        == 0
    )
    assert PENDING in capsys.readouterr().out
    assert len(list((tmp_path / "Drive" / "bob" / "inbox").iterdir())) == 1


def test_status_with_a_foreign_label_is_ignored(tmp_path: Path) -> None:
    station = queue(tmp_path)
    request = h2_request(1)
    receipt = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-7"
    )
    job, _work = pick_up(station, receipt)
    publish(job, receipt, "RUNNING")
    status = json.loads((job / "status.json").read_text())
    status["label"] = "../../elsewhere"
    (job / "status.json").write_text(json.dumps(status))
    assert station.status(receipt["client_job_id"])["state"] == PENDING
    station.close()


def test_results_under_another_approval_or_implementation_are_rejected(
    tmp_path: Path,
) -> None:
    station = queue(tmp_path)
    request = h2_request(1)
    receipt = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-8"
    )
    job, work = pick_up(station, receipt)
    genuine_shard(work, request)
    other_plan = hashlib.sha256(b"a different approved plan").hexdigest()
    publish(
        job,
        receipt,
        "COMPLETED",
        work,
        reported={
            "request.json": hashlib.sha256(
                (work / "request.json").read_bytes()
            ).hexdigest(),
            "approved_plan.json": other_plan,
        },
    )
    with pytest.raises(
        WorkstationError, match="different submission or approval"
    ):  # signed
        station.download(receipt["client_job_id"], tmp_path / "ingested")
    assert station.ledger.get(receipt["client_job_id"])["state"] == REJECTED

    second = h2_request(1)
    later = station.submit(
        second, approved_plan=approved(second), idempotency_key="lab-9"
    )
    # The ledger keeps the implementation approved at submission; a shard from
    # any other implementation is refused even if it matches the local install.
    station.ledger.db.execute(
        "UPDATE workstation_jobs SET code_sha256 = ? WHERE client_job_id = ?",
        ("0" * 64, later["client_job_id"]),
    )
    job, work = pick_up(station, later)
    genuine_shard(work, second)
    publish(job, later, "COMPLETED", work)
    with pytest.raises(WorkstationError, match="different TORQ implementation"):
        station.download(later["client_job_id"], tmp_path / "ingested")
    station.close()


def test_interrupted_delivery_and_publication_are_recovered(tmp_path: Path) -> None:
    (tmp_path / "Drive").mkdir()
    station = WorkstationQueue(
        tmp_path / "Drive" / "carol",
        student_id="carol",
        ledger_path=tmp_path / "state" / "tpo_ledger.sqlite",
        trusted_keys=[fingerprint()],
    )
    request = h2_request(1)
    plan = approved(request)
    with pytest.raises(WorkstationError, match="assign"):
        station.submit(request, approved_plan=plan, idempotency_key="lab-10")
    (row,) = station.ledger.jobs()
    assert row["state"] == PENDING and row["dispatched_at"] is None
    assert "same idempotency key" in station.status(row["client_job_id"])["message"]
    station.assign()
    receipt = station.submit(request, approved_plan=plan, idempotency_key="lab-10")
    assert (station.folder / "inbox" / receipt["job_name"]).is_dir()
    assert station.ledger.get(receipt["client_job_id"])["dispatched_at"]
    station.submit(request, approved_plan=plan, idempotency_key="lab-10")
    assert len(list((station.folder / "inbox").iterdir())) == 1

    job, work = pick_up(station, receipt)
    genuine_shard(work, request)
    publish(job, receipt, "COMPLETED", work)
    published = station.download(receipt["client_job_id"], tmp_path / "ingested")
    # Interrupted after publishing but before the ledger recorded it:
    station.ledger.db.execute(
        "UPDATE workstation_jobs SET state = ?, result_paths = '[]' "
        "WHERE client_job_id = ?",
        (RUNNING, receipt["client_job_id"]),
    )
    assert (
        station.download(receipt["client_job_id"], tmp_path / "ingested") == published
    )
    assert station.ledger.get(receipt["client_job_id"])["state"] == INGESTED
    station.close()


def test_student_environment_variable_applies_without_student_option(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COCHEM_WORKSTATION_STUDENT", "dana")
    (tmp_path / "Drive").mkdir()
    folder = tmp_path / "Drive" / "dana"
    assert (
        main(
            [
                "workstation",
                "assign",
                "--folder",
                str(folder),
                "--ledger",
                str(tmp_path / "tpo.sqlite"),
                "--json",
            ]
        )
        == 0
    )
    marker = json.loads((folder / "cochem_workstation_folder.json").read_text())
    assert marker["student_id"] == "dana"


def test_only_results_signed_by_a_trusted_workstation_key_are_ingested(
    tmp_path: Path,
) -> None:
    station = queue(tmp_path, trusted=[])
    request = h2_request(1)
    receipt = station.submit(
        request, approved_plan=approved(request), idempotency_key="lab-11"
    )
    job, work = pick_up(station, receipt)
    genuine_shard(work, request)
    publish(job, receipt, "COMPLETED", work)
    with pytest.raises(WorkstationError, match=f"key {fingerprint()}, which is not"):
        station.download(receipt["client_job_id"], tmp_path / "ingested")
    assert station.ledger.get(receipt["client_job_id"])["state"] in OPEN_STATES

    impostor = Ed25519PrivateKey.generate()
    publish(job, receipt, "COMPLETED", work, key=impostor)
    trusting = WorkstationQueue(
        station.folder,
        student_id="alice",
        ledger_path=station.ledger.path,
        trusted_keys=f"sha256:{fingerprint()}",
    )
    with pytest.raises(WorkstationError, match="not trusted"):
        trusting.download(receipt["client_job_id"], tmp_path / "ingested")

    publish(job, receipt, "COMPLETED", work)
    status = json.loads((job / "status.json").read_text())
    status["result_sha256"] = hashlib.sha256(b"edited").hexdigest()
    (job / "status.json").write_text(json.dumps(status))
    with pytest.raises(WorkstationError, match="still syncing|different results"):
        trusting.download(receipt["client_job_id"], tmp_path / "ingested")

    publish(job, receipt, "COMPLETED", work)
    (published,) = trusting.download(receipt["client_job_id"], tmp_path / "ingested")
    row = trusting.ledger.get(receipt["client_job_id"])
    assert row["state"] == INGESTED and row["attested_by"] == fingerprint()
    saved = (
        json.loads(
            (published.parent / "_runner" / "workstation-attestation.json").read_text()
        )
        if (published.parent / "_runner").is_dir()
        else None
    )
    assert saved is None or saved["key_fingerprint"] == fingerprint()
    station.close()
    trusting.close()


def test_cli_trusted_key_option(tmp_path: Path, capsys) -> None:
    (tmp_path / "Drive").mkdir()
    assert (
        main(
            [
                "workstation",
                "assign",
                "--folder",
                str(tmp_path / "Drive" / "carol"),
                "--student",
                "carol",
                "--ledger",
                str(tmp_path / "tpo.sqlite"),
                "--trusted-key",
                fingerprint(),
                "--json",
            ]
        )
        == 0
    )
    assert "carol" in capsys.readouterr().out
    with pytest.raises(ValueError, match="not a workstation key fingerprint"):
        WorkstationQueue(
            tmp_path / "Drive" / "carol",
            ledger_path=tmp_path / "tpo.sqlite",
            trusted_keys="not-a-key",
        )
