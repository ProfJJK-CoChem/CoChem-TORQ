"""Heavy TORQ requests on the lab workstation through an assigned Drive folder.

Requests too large for the public Actions classroom worker (more cores, memory
or time, or research recipes) can be queued to the lab workstation. The
workstation runs the CoChem workstation job runner
(https://github.com/ProfJJK-CoChem/cochem_workstation_job_runner) whose
``torq_execute`` template performs the ordinary explicit local validation run::

    cochem-torq execute --request request.json --approved-plan approved_plan.json \
        --output-dir result

Evidence lane: GitHub Actions remains TORQ's canonical calculation
environment. A workstation result is a separate, explicitly labelled lane and
is accepted only after ``verify_shard`` (request identity, byte inventory)
and a matching implementation identity. A workstation "COMPLETED" status is
not scientific success by itself.

Every submission is recorded in ``tpo_ledger.sqlite`` as PENDING_WORKSTATION
and followed until INGESTED (or failed/cancelled/rejected); its event table
is append-only. Nothing hard-codes the Drive folder: it is passed explicitly
or taken from ``COCHEM_TORQ_WORKSTATION_FOLDER`` / ``COCHEM_WORKSTATION_FOLDER``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .domain import CalculationRequest, PrerequisiteError, canonical_json, read_json

JOB_SCHEMA = "cochem.workstation-job/1"
STATUS_SCHEMA = "cochem.workstation-job-status/1"
SUMMARY_SCHEMA = "cochem.workstation-job-summary/1"
FOLDER_SCHEMA = "cochem.workstation-folder/1"
FOLDER_MARKER = "cochem_workstation_folder.json"
SHARD_SCHEMA = "cochem.torq.shard/1"

PENDING = "PENDING_WORKSTATION"
PAUSED = "PAUSED_WORKSTATION"
RUNNING = "RUNNING_WORKSTATION"
INGESTED = "INGESTED"
FAILED = "FAILED_WORKSTATION"
CANCELLED = "CANCELLED_WORKSTATION"
REJECTED = "REJECTED_RESULT"
OPEN_STATES = (PENDING, PAUSED, RUNNING)
FINAL_STATES = (INGESTED, FAILED, CANCELLED, REJECTED)
EVIDENCE_LANE = "workstation (non-canonical; GitHub Actions remains canonical)"
MAX_RESULT_BYTES = 4 * 1024 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_ledger_path() -> Path:
    return Path.home() / ".local/state/cochem-torq/tpo_ledger.sqlite"


class WorkstationError(RuntimeError):
    """The workstation queue is unavailable or returned untrustworthy results."""


# ---------------------------------------------------------------------------
# tpo_ledger.sqlite
# ---------------------------------------------------------------------------
class TpoLedger:
    """Workstation dispatch ledger with an append-only event history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        states = ", ".join(f"'{state}'" for state in (*OPEN_STATES, *FINAL_STATES))
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS workstation_jobs (
              client_job_id TEXT PRIMARY KEY,
              idempotency_key TEXT NOT NULL UNIQUE,
              request_id TEXT NOT NULL,
              request_sha256 TEXT NOT NULL,
              approved_plan_sha256 TEXT NOT NULL,
              code_sha256 TEXT NOT NULL,
              folder TEXT NOT NULL,
              student_id TEXT,
              job_name TEXT NOT NULL,
              template TEXT NOT NULL,
              state TEXT NOT NULL CHECK (state IN ({states})),
              workstation_state TEXT,
              message TEXT NOT NULL DEFAULT '',
              label TEXT,
              submitted_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              result_paths TEXT NOT NULL DEFAULT '[]',
              input_hashes TEXT NOT NULL DEFAULT '{{}}',
              dispatched_at TEXT
            );
            CREATE TABLE IF NOT EXISTS workstation_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              client_job_id TEXT NOT NULL,
              at TEXT NOT NULL,
              state TEXT NOT NULL,
              workstation_state TEXT,
              message TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS workstation_events_no_update
              BEFORE UPDATE ON workstation_events
              BEGIN SELECT RAISE(ABORT, 'workstation events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS workstation_events_no_delete
              BEFORE DELETE ON workstation_events
              BEGIN SELECT RAISE(ABORT, 'workstation events are append-only'); END;
            """
        )
        columns = {
            row["name"]
            for row in self.db.execute("PRAGMA table_info(workstation_jobs)")
        }
        for name, definition in (
            ("input_hashes", "TEXT NOT NULL DEFAULT '{}'"),
            ("dispatched_at", "TEXT"),
        ):
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE workstation_jobs ADD COLUMN {name} {definition}"
                )

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _event(
        self, client_job_id: str, state: str, observed: str | None, message: str
    ) -> None:
        self.db.execute(
            "INSERT INTO workstation_events "
            "(client_job_id, at, state, workstation_state, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (client_job_id, _now(), state, observed, message),
        )

    def add(self, row: dict[str, Any]) -> None:
        columns = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._transaction():
            self.db.execute(
                f"INSERT INTO workstation_jobs ({columns}) VALUES ({marks})",
                list(row.values()),
            )
            self._event(row["client_job_id"], row["state"], None, "submitted")

    def mark_dispatched(self, client_job_id: str) -> None:
        with self._transaction():
            self.db.execute(
                "UPDATE workstation_jobs "
                "SET dispatched_at = COALESCE(dispatched_at, ?) "
                "WHERE client_job_id = ?",
                (_now(), client_job_id),
            )

    def get(self, client_job_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM workstation_jobs WHERE client_job_id = ?", (client_job_id,)
        ).fetchone()
        return dict(row) if row else None

    def by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM workstation_jobs WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def jobs(self, *states: str) -> list[dict[str, Any]]:
        if states:
            marks = ", ".join("?" for _ in states)
            query = f"SELECT * FROM workstation_jobs WHERE state IN ({marks})"
            rows = self.db.execute(query + " ORDER BY submitted_at", states)
        else:
            rows = self.db.execute(
                "SELECT * FROM workstation_jobs ORDER BY submitted_at"
            )
        return [dict(row) for row in rows]

    def events(self, client_job_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT at, state, workstation_state, message FROM workstation_events "
            "WHERE client_job_id = ? ORDER BY id",
            (client_job_id,),
        )
        return [dict(row) for row in rows]

    def update(
        self,
        client_job_id: str,
        *,
        state: str,
        observed: str | None = None,
        message: str = "",
        label: str | None = None,
        result_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._transaction():
            current = self.get(client_job_id)
            if current is None:
                raise KeyError(client_job_id)
            if current["state"] in FINAL_STATES and state != current["state"]:
                raise WorkstationError(
                    f"Workstation job {client_job_id} is already {current['state']}."
                )
            changed = (
                current["state"],
                current["workstation_state"],
                current["message"],
            ) != (
                state,
                observed,
                message,
            )
            self.db.execute(
                "UPDATE workstation_jobs SET state = ?, workstation_state = ?, "
                "message = ?, label = COALESCE(?, label), "
                "result_paths = COALESCE(?, result_paths), updated_at = ? "
                "WHERE client_job_id = ?",
                (
                    state,
                    observed,
                    message,
                    label,
                    json.dumps(result_paths) if result_paths is not None else None,
                    _now(),
                    client_job_id,
                ),
            )
            if changed:
                self._event(client_job_id, state, observed, message)
        updated = self.get(client_job_id)
        assert updated is not None
        return updated


# ---------------------------------------------------------------------------
# The assigned folder (Google Drive for desktop sync)
# ---------------------------------------------------------------------------
class LocalFolder:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder).expanduser()

    def ensure(self, student_id: str | None) -> None:
        if not self.folder.parent.is_dir():
            raise WorkstationError(
                f"{self.folder.parent} does not exist; "
                "is Google Drive for desktop running?"
            )
        (self.folder / "inbox").mkdir(parents=True, exist_ok=True)
        (self.folder / "jobs").mkdir(parents=True, exist_ok=True)
        marker = self.folder / FOLDER_MARKER
        if student_id and not marker.is_file():
            marker.write_text(
                json.dumps(
                    {
                        "schema": FOLDER_SCHEMA,
                        "student_id": student_id,
                        "created_by": "CoChem-TORQ",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def submit(self, job_name: str, files: dict[str, bytes]) -> None:
        inbox = self.folder / "inbox"
        if not inbox.is_dir():
            raise WorkstationError(
                f"{self.folder} is not an assigned workstation folder "
                "(missing inbox/); "
                "run `cochem-torq workstation assign` first"
            )
        staging = inbox / f".cochem-upload-{uuid.uuid4().hex[:12]}"
        staging.mkdir()
        try:
            for name, contents in files.items():
                (staging / name).write_bytes(contents)
            os.replace(staging, inbox / job_name)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def pending(self, job_name: str) -> bool:
        return (self.folder / "inbox" / job_name).is_dir()

    def delivered(self, job_name: str) -> bool:
        """The job is in inbox/ or the workstation has already moved it to jobs/."""
        return self.pending(job_name) or any(
            (self.folder / "jobs").glob(job_name + "__*")
        )

    def status(self, job_name: str, client_job_id: str) -> dict[str, Any] | None:
        for path in sorted((self.folder / "jobs").glob(job_name + "__*/status.json")):
            try:
                document = read_json(path)
            except (OSError, ValueError):
                continue
            if (
                isinstance(document, dict)
                and document.get("schema") == STATUS_SCHEMA
                and document.get("client_job_id") == client_job_id
                # The label is used in paths: it must name this very job folder.
                and document.get("label") == path.parent.name
            ):
                return document
        return None

    def result_archive(self, status: dict[str, Any]) -> Path:
        name = str(status.get("result_file") or "")
        path = self.folder / "jobs" / str(status["label"]) / name
        if (
            not name
            or "/" in name
            or "\\" in name
            or path.is_symlink()
            or not path.is_file()
        ):
            raise WorkstationError("The results archive has not finished syncing yet.")
        return path

    def cancel(self, job_name: str, status: dict[str, Any] | None) -> None:
        if status is None and self.pending(job_name):
            shutil.rmtree(self.folder / "inbox" / job_name)
        elif status is not None:
            cancel = self.folder / "jobs" / str(status["label"]) / "CANCEL"
            cancel.write_text("cancel requested\n", encoding="utf-8")


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if sum(member.file_size for member in members) > MAX_RESULT_BYTES:
            raise WorkstationError("The results archive is larger than TORQ ingests.")
        for member in members:
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename.replace("\\", "/"))
            link = (member.external_attr >> 16) & 0o170000 == 0o120000
            if path.is_absolute() or ".." in path.parts or not path.parts or link:
                raise ValueError("The results archive contains an unsafe entry.")
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)


# ---------------------------------------------------------------------------
# Routing: when is the workstation the right lane?
# ---------------------------------------------------------------------------
def recommend_execution(request: dict[str, Any] | CalculationRequest) -> dict[str, Any]:
    """Recommend GitHub Actions when possible, else the workstation if it can run it."""
    from .application import validate_request

    hosted = validate_request(request, execution="github_actions")
    if hosted["executable"]:
        return {"execution": "github_actions", "reasons": []}
    local = validate_request(request, execution="local_validation")
    if local["executable"]:
        return {
            "execution": "workstation",
            "reasons": hosted["blocking_reasons"],
            "evidence_lane": EVIDENCE_LANE,
        }
    return {"execution": None, "reasons": local["blocking_reasons"]}


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
class WorkstationQueue:
    def __init__(
        self,
        folder: str | Path,
        *,
        student_id: str | None = None,
        ledger_path: str | Path | None = None,
        template: str = "torq_execute",
        min_approval_hours: float = 24.0,
    ) -> None:
        self.folder = Path(folder).expanduser()
        self.student_id = student_id
        self.transport = LocalFolder(self.folder)
        self.ledger = TpoLedger(ledger_path or default_ledger_path())
        self.template = template
        self.min_approval_hours = min_approval_hours

    @classmethod
    def from_environment(
        cls, folder: str | Path | None = None, **kwargs: Any
    ) -> WorkstationQueue:
        chosen = (
            folder
            or os.environ.get("COCHEM_TORQ_WORKSTATION_FOLDER")
            or os.environ.get("COCHEM_WORKSTATION_FOLDER")
        )
        if not chosen:
            raise PrerequisiteError(
                "Assign the lab workstation Drive folder: pass --folder or set "
                "COCHEM_WORKSTATION_FOLDER."
            )
        if kwargs.get("student_id") is None:  # the CLI passes None without --student
            kwargs["student_id"] = os.environ.get(
                "COCHEM_TORQ_WORKSTATION_STUDENT"
            ) or os.environ.get("COCHEM_WORKSTATION_STUDENT")
        return cls(chosen, **kwargs)

    def close(self) -> None:
        self.ledger.close()

    def assign(self) -> dict[str, Any]:
        self.transport.ensure(self.student_id)
        return {
            "folder": str(self.folder),
            "student_id": self.student_id,
            "ready": True,
        }

    def submit(
        self,
        request: dict[str, Any] | CalculationRequest,
        *,
        approved_plan: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        from .application import validate_request
        from .service import validate_approved_plan

        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 256
        ):
            raise ValueError("An idempotency key must contain 1..256 characters.")
        checked = validate_request(request, execution="local_validation")
        if not checked["executable"]:
            raise PrerequisiteError("; ".join(checked["blocking_reasons"]))
        approved = validate_approved_plan(approved_plan, request=checked["request"])
        existing = self.ledger.by_idempotency_key(idempotency_key)
        if existing is not None and (
            existing["request_sha256"] != checked["request_sha256"]
            or existing["approved_plan_sha256"] != approved.approval.plan_sha256
        ):
            raise ValueError("This idempotency key was used for a different request.")
        if existing is not None and (
            existing["dispatched_at"] or existing["state"] not in OPEN_STATES
        ):
            return self._receipt(existing)
        remaining = approved.approval.expires_at - datetime.now(timezone.utc)
        if remaining.total_seconds() < self.min_approval_hours * 3600:
            raise PrerequisiteError(
                "The workstation may hold a job while its owner uses the machine; "
                "approve the plan with --expires-at at least "
                f"{self.min_approval_hours:g} hours ahead."
            )
        request_bytes = canonical_json(checked["request"])
        plan_bytes = canonical_json(approved.model_dump(mode="json"))
        request_id = str(checked["request"]["request_id"])
        client_job_id = "cochem-torq:" + request_id
        job_name = "torq-" + request_id.replace("-", "")[:12]
        resources = CalculationRequest.model_validate(checked["request"]).resources
        input_hashes = {
            "request.json": hashlib.sha256(request_bytes).hexdigest(),
            "approved_plan.json": hashlib.sha256(plan_bytes).hexdigest(),
        }
        manifest = {
            "schema": JOB_SCHEMA,
            "engine": self.template,
            "name": job_name,
            "input": "request.json",
            "resources": {
                "cores": resources.cores,
                "memory_gb": round(resources.memory_mb / 1024, 3),
                "max_hours": min(168, math.ceil(resources.wall_seconds / 3600) + 1),
            },
            "job_id": client_job_id,
            "files": input_hashes,
        }
        if self.student_id:
            manifest["student_id"] = self.student_id
        if existing is None:
            # Ledger first: a crash before delivery leaves a recoverable PENDING row.
            now = _now()
            self.ledger.add(
                {
                    "client_job_id": client_job_id,
                    "idempotency_key": idempotency_key,
                    "request_id": request_id,
                    "request_sha256": checked["request_sha256"],
                    "approved_plan_sha256": approved.approval.plan_sha256,
                    "code_sha256": approved.source_identity["code_sha256"],
                    "folder": str(self.folder),
                    "student_id": self.student_id,
                    "job_name": job_name,
                    "template": self.template,
                    "state": PENDING,
                    "message": "Waiting for the workstation to pick the job up",
                    "submitted_at": now,
                    "updated_at": now,
                    "input_hashes": json.dumps(input_hashes, sort_keys=True),
                }
            )
        self._dispatch(
            client_job_id,
            job_name,
            {
                "request.json": request_bytes,
                "approved_plan.json": plan_bytes,
                "job.json": json.dumps(manifest, indent=2, sort_keys=True).encode(),
            },
        )
        return self._receipt(self.ledger.get(client_job_id))

    def _dispatch(
        self, client_job_id: str, job_name: str, files: dict[str, bytes]
    ) -> None:
        """Deliver the job to the folder once; repeating a submission resumes here."""
        try:
            if not self.transport.delivered(job_name):
                self.transport.submit(job_name, files)
        except (OSError, WorkstationError) as exc:
            self.ledger.update(
                client_job_id,
                state=PENDING,
                message=f"Not delivered to the workstation folder yet ({exc}); "
                "repeat the submission with the same idempotency key.",
            )
            raise
        self.ledger.mark_dispatched(client_job_id)

    @staticmethod
    def _receipt(row: dict[str, Any] | None) -> dict[str, Any]:
        assert row is not None
        return {
            "client_job_id": row["client_job_id"],
            "request_id": row["request_id"],
            "state": row["state"],
            "workstation_state": row["workstation_state"],
            "message": row["message"],
            "folder": row["folder"],
            "job_name": row["job_name"],
            "request_sha256": row["request_sha256"],
            "approved_plan_sha256": row["approved_plan_sha256"],
            "result_paths": json.loads(row["result_paths"]),
            "execution": "workstation",
            "evidence_lane": EVIDENCE_LANE,
        }

    def _row(self, client_job_id: str) -> dict[str, Any]:
        row = self.ledger.get(client_job_id)
        if row is None:
            raise KeyError(
                f"No workstation submission {client_job_id} in the TPO ledger."
            )
        return row

    def status(self, client_job_id: str) -> dict[str, Any]:
        """Refresh one submission from the folder and record any change."""
        row = self._row(client_job_id)
        if row["state"] in FINAL_STATES:
            return self._receipt(row)
        document = self.transport.status(row["job_name"], client_job_id)
        if document is None:
            if self.transport.pending(row["job_name"]):
                message = "Waiting for the workstation to pick the job up"
            elif not row["dispatched_at"]:
                message = (
                    "Not delivered to the workstation folder yet; repeat the "
                    "submission with the same idempotency key."
                )
            else:
                message = "Between inbox and jobs/ (Drive may still be syncing)"
            return self._receipt(
                self.ledger.update(client_job_id, state=row["state"], message=message)
            )
        observed = str(document.get("state"))
        message = str(document.get("message") or "")
        label = document.get("label")
        if observed in {"COMPLETED", "FAILED", "CANCELLED"} and not document.get(
            "result_file"
        ):
            if document.get("attempts", 0) == 0:
                final = CANCELLED if observed == "CANCELLED" else FAILED
                return self._receipt(
                    self.ledger.update(
                        client_job_id,
                        state=final,
                        observed=observed,
                        message=message,
                        label=label,
                    )
                )
            state = RUNNING  # finished, results archive not yet published
        elif observed in {"COMPLETED", "FAILED", "CANCELLED"}:
            state = row["state"] if row["state"] in OPEN_STATES else RUNNING
        else:
            state = {"QUEUED": PENDING, "PAUSED": PAUSED, "SUSPENDED": PAUSED}.get(
                observed, RUNNING
            )
        receipt = self._receipt(
            self.ledger.update(
                client_job_id,
                state=state,
                observed=observed,
                message=message,
                label=label,
            )
        )
        receipt["result_ready"] = bool(
            observed in {"COMPLETED", "FAILED", "CANCELLED"}
            and document.get("result_file")
        )
        receipt["progress"] = document.get("progress") or {}
        receipt["repairs"] = document.get("repairs") or []
        return receipt

    def _verified_shards(self, results: Path, row: dict[str, Any]) -> list[Path]:
        """Shard roots in an extracted results folder bound to this exact submission."""
        from .artifacts import verify_shard

        summary = read_json(results / "_runner" / "job_summary.json")
        expected = json.loads(row["input_hashes"])
        reported = summary.get("input_hashes") or {}
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("client_job_id") != row["client_job_id"]
            or not expected
            or any(reported.get(name) != digest for name, digest in expected.items())
        ):
            raise ValueError(
                "The returned results belong to a different submission or approval."
            )
        roots = sorted(
            path.parent
            for path in results.rglob("manifest.json")
            if isinstance(read_json(path), dict)
            and read_json(path).get("schema_version") == SHARD_SCHEMA
        )
        if not roots:
            raise ValueError("The workstation returned no sealed TORQ result shard.")
        for root in roots:
            manifest = verify_shard(root, expected_request_sha256=row["request_sha256"])
            # The implementation approved at submission, not whatever is installed now.
            if manifest["source_identity"].get("code_sha256") != row["code_sha256"]:
                raise ValueError(
                    "The workstation ran a different TORQ implementation than the "
                    "approved plan; update the workstation's TORQ and resubmit."
                )
        return roots

    def _ingested(
        self,
        row: dict[str, Any],
        document: dict[str, Any] | None,
        published: list[Path],
    ) -> list[Path]:
        self.ledger.update(
            row["client_job_id"],
            state=INGESTED,
            observed=str((document or {}).get("state") or row["workstation_state"]),
            message=f"Verified and ingested {len(published)} TORQ result shard(s) "
            f"({EVIDENCE_LANE}).",
            label=(document or {}).get("label"),
            result_paths=[str(path) for path in published],
        )
        return published

    def download(self, client_job_id: str, destination: str | Path) -> list[Path]:
        """Verify and publish the returned sealed TORQ shards (immutable target)."""
        row = self._row(client_job_id)
        if row["state"] == INGESTED:
            return [Path(path) for path in json.loads(row["result_paths"])]
        target = Path(destination).absolute() / f"workstation-{row['request_id']}"
        if target.exists():
            if row["state"] not in OPEN_STATES:
                raise FileExistsError(
                    "Ingested workstation results are immutable; "
                    "choose a new destination."
                )
            try:  # published before an interruption: finish recording it
                roots = self._verified_shards(target, row)
            except (OSError, ValueError, KeyError) as exc:
                raise FileExistsError(
                    f"{target} already exists and is not this job's verified result "
                    f"({exc}); choose a new destination."
                ) from exc
            return self._ingested(row, None, roots)
        document = self.transport.status(row["job_name"], client_job_id)
        if document is None or not document.get("result_file"):
            raise WorkstationError(
                "The workstation has not published results for this job yet."
            )
        expected = document.get("result_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise WorkstationError(
                "The workstation has not published the results archive checksum yet."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".torq-workstation-", dir=target.parent))
        try:
            archive = staging / "results.zip"
            shutil.copyfile(self.transport.result_archive(document), archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
                raise WorkstationError(
                    "The results archive is still syncing (checksum differs)."
                )
            extracted = staging / "results"
            try:
                _safe_extract(archive, extracted)
                roots = self._verified_shards(extracted, row)
            except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                self.ledger.update(
                    client_job_id,
                    state=REJECTED,
                    observed=str(document.get("state")),
                    message=f"Returned results were not accepted: {exc} "
                    f"(workstation: {document.get('message', '')})",
                    label=document.get("label"),
                )
                raise WorkstationError(
                    f"Returned results were not accepted: {exc}"
                ) from exc
            extracted.rename(target)
            published = [target / root.relative_to(extracted) for root in roots]
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self._ingested(row, document, published)

    def poll(self, destination: str | Path) -> list[dict[str, Any]]:
        """Refresh every open submission and ingest those whose results arrived."""
        outcomes = []
        for row in self.ledger.jobs(*OPEN_STATES):
            receipt = self.status(row["client_job_id"])
            if receipt.get("result_ready"):
                try:
                    receipt["result_paths"] = [
                        str(path)
                        for path in self.download(row["client_job_id"], destination)
                    ]
                    receipt["state"] = INGESTED
                except WorkstationError as exc:
                    receipt["error"] = str(exc)
                    receipt["state"] = self._row(row["client_job_id"])["state"]
            outcomes.append(receipt)
        return outcomes

    def cancel(self, client_job_id: str, *, reason: str) -> dict[str, Any]:
        row = self._row(client_job_id)
        if row["state"] in FINAL_STATES:
            return self._receipt(row)
        document = self.transport.status(row["job_name"], client_job_id)
        self.transport.cancel(row["job_name"], document)
        if document is None:
            return self._receipt(
                self.ledger.update(
                    client_job_id, state=CANCELLED, message=f"Withdrawn: {reason}"
                )
            )
        return self._receipt(
            self.ledger.update(
                client_job_id,
                state=row["state"],
                observed=row["workstation_state"],
                message=f"Cancellation requested: {reason}",
            )
        )
