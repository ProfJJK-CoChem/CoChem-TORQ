"""Local campaign authority with CAS events, fenced attempts and resource accounting.

This is an independent coordinator implementation, not a second state authority
to layer over SQLiteTaskQueue. Applications must select one authority. Actual
scientific validation and native checkpoint compatibility belong to the caller;
this service never derives or substitutes a scientific observation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import secrets
import socket
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from cochem_torq.domain import canonical_json, digest

STATES = {
    "draft",
    "validated",
    "needs_review",
    "queued",
    "running",
    "collecting",
    "validating",
    "succeeded",
    "partial",
    "failed",
    "cancelled",
    "blocked",
}
TERMINAL = {"succeeded", "partial", "failed", "cancelled"}
EDGES = {
    "draft": {"validated", "blocked", "cancelled"},
    "validated": {"queued", "needs_review", "blocked", "cancelled"},
    "needs_review": {"validated", "cancelled"},
    "blocked": {"validated", "cancelled"},
    "queued": {"running", "blocked", "cancelled"},
    "running": {"collecting", "failed", "partial", "cancelled"},
    "collecting": {"validating", "failed", "partial", "cancelled"},
    "validating": {"succeeded", "needs_review", "failed", "partial", "cancelled"},
}


class CampaignError(RuntimeError):
    """Invalid authority, lifecycle revision, dependency or budget."""


class RevisionConflictError(CampaignError):
    pass


class BudgetExceededError(CampaignError):
    pass


class AuthorityError(CampaignError):
    pass


# Retain public imports while giving the exception classes explicit error names.
RevisionConflict = RevisionConflictError
BudgetExceeded = BudgetExceededError


class AccountingModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, strict=True
    )


class CampaignBudget(AccountingModel):
    # Wall here is cumulative task wall time, explicitly separate from expiry.
    max_wall_seconds: float = Field(gt=0)
    max_cpu_core_seconds: float = Field(gt=0)
    max_gpu_seconds: float = Field(default=0, ge=0)
    max_tasks: StrictInt = Field(ge=1)
    permitted_retries: StrictInt = Field(default=0, ge=0, le=2)
    max_concurrent_workers: StrictInt = Field(ge=1)
    max_cpu_cores: StrictInt = Field(ge=1)
    max_gpus: StrictInt = Field(default=0, ge=0)
    max_memory_mb: StrictInt = Field(ge=1)
    max_scratch_mb: StrictInt = Field(ge=1)
    expires_at_unix: float = Field(gt=0)
    allowed_engines: list[str] = Field(min_length=1)
    allowed_recipes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_choices(self):
        for names in (self.allowed_engines, self.allowed_recipes):
            if len(set(names)) != len(names) or any(not name.strip() for name in names):
                raise ValueError(
                    "Allowed engine/recipe identities must be explicit and unique."
                )
        return self


class Allocation(AccountingModel):
    wall_seconds: float = Field(gt=0)
    cores: StrictInt = Field(ge=1)
    gpu_count: StrictInt = Field(default=0, ge=0)
    memory_mb: StrictInt = Field(ge=1)
    scratch_mb: StrictInt = Field(ge=0)


class MeasuredUsage(AccountingModel):
    wall_seconds: float = Field(ge=0)
    cpu_core_seconds: float | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)
    peak_scratch_mb: float | None = Field(default=None, ge=0)
    measurement_source: str = Field(min_length=1)


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class CampaignCoordinator:
    """Trusted local application service; opaque operator authority is not a lease.

    The database is created privately (0600). Operator tokens must be retained
    outside public artifacts. Remote adapters must authenticate their caller
    before invoking this service; a passed actor string alone is not identity.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise AuthorityError("A coordinator database cannot be a symlink.")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(descriptor)
        os.chmod(path, 0o600)
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY, actor TEXT NOT NULL, plan_json TEXT NOT NULL,
                plan_sha256 TEXT NOT NULL, budget_json TEXT NOT NULL,
                authority_hash TEXT NOT NULL, created_at REAL NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS campaign_tasks (
                id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                task_key TEXT NOT NULL, payload_json TEXT NOT NULL,
                engine TEXT NOT NULL,
                recipe TEXT NOT NULL, dependencies_json TEXT NOT NULL,
                allow_partial_dependencies INTEGER NOT NULL DEFAULT 0,
                next_generation INTEGER NOT NULL DEFAULT 0,
                UNIQUE(campaign_id, task_key));
            CREATE TABLE IF NOT EXISTS campaign_attempts (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES campaign_tasks(id),
                number INTEGER NOT NULL,
                parent_attempt_id TEXT REFERENCES campaign_attempts(id),
                state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                lease_hash TEXT, generation INTEGER NOT NULL DEFAULT 0,
                lease_expires REAL, heartbeat_seconds REAL, lease_seconds REAL,
                host TEXT, pid INTEGER, process_start REAL, result_json TEXT,
                created_at REAL NOT NULL, UNIQUE(task_id, number));
            CREATE TABLE IF NOT EXISTS campaign_reservations (
                attempt_id TEXT PRIMARY KEY REFERENCES campaign_attempts(id),
                campaign_id TEXT NOT NULL REFERENCES campaigns(id),
                allocation_json TEXT NOT NULL,
                state TEXT NOT NULL, dispatched INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT, charged_wall REAL NOT NULL DEFAULT 0,
                charged_cpu REAL NOT NULL DEFAULT 0,
                charged_gpu REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS campaign_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id TEXT NOT NULL,
                attempt_id TEXT, revision INTEGER NOT NULL, actor TEXT NOT NULL,
                timestamp REAL NOT NULL, kind TEXT NOT NULL, reason TEXT NOT NULL,
                details_json TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS campaign_events_no_update
                BEFORE UPDATE ON campaign_events
                BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS campaign_events_no_delete
                BEFORE DELETE ON campaign_events
                BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS campaign_attempts_terminal_immutable
                BEFORE UPDATE ON campaign_attempts
                WHEN OLD.state IN ('succeeded','partial','failed','cancelled')
                BEGIN SELECT RAISE(ABORT,'Terminal attempts are immutable'); END;
        """)

    def close(self) -> None:
        self.connection.close()

    @contextlib.contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _event(
        self,
        campaign: str,
        attempt: str | None,
        revision: int,
        actor: str,
        kind: str,
        reason: str,
        details: dict,
    ) -> None:
        if not actor.strip() or not reason.strip():
            raise CampaignError("Lifecycle events require an actor and reason.")
        self.connection.execute(
            "INSERT INTO campaign_events(campaign_id,attempt_id,revision,actor,"
            "timestamp,kind,reason,details_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                campaign,
                attempt,
                revision,
                actor,
                time.time(),
                kind,
                reason,
                canonical_json(details).decode(),
            ),
        )

    def _campaign(self, campaign_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM campaigns WHERE id=?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise CampaignError("Campaign is absent.")
        return row

    def _operator(self, campaign_id: str, authority: str) -> sqlite3.Row:
        row = self._campaign(campaign_id)
        if not isinstance(authority, str) or not secrets.compare_digest(
            row["authority_hash"], _token_hash(authority)
        ):
            raise AuthorityError(
                "A separate authenticated operator authority is required."
            )
        return row

    def _approved(self, campaign_id: str) -> tuple[sqlite3.Row, CampaignBudget]:
        campaign = self._campaign(campaign_id)
        budget = CampaignBudget.model_validate_json(campaign["budget_json"])
        if campaign["revoked"] or time.time() >= budget.expires_at_unix:
            raise AuthorityError(
                "Campaign approval was revoked or expired; no new work is authorized."
            )
        return campaign, budget

    def create_campaign(
        self, plan: dict, budget: CampaignBudget, *, actor: str
    ) -> dict:
        if not isinstance(plan, dict) or not plan:
            raise CampaignError("An explicit immutable approved plan is required.")
        budget = CampaignBudget.model_validate(budget)
        if budget.expires_at_unix <= time.time():
            raise CampaignError("A campaign approval cannot start expired.")
        campaign_id, authority = str(uuid4()), secrets.token_urlsafe(32)
        with self._transaction():
            self.connection.execute(
                "INSERT INTO campaigns(id,actor,plan_json,plan_sha256,budget_json,"
                "authority_hash,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    campaign_id,
                    actor,
                    canonical_json(plan).decode(),
                    digest(plan),
                    budget.model_dump_json(),
                    _token_hash(authority),
                    time.time(),
                ),
            )
            self._event(
                campaign_id,
                None,
                0,
                actor,
                "approval",
                "Approved explicit plan and finite budget.",
                {
                    "plan_sha256": digest(plan),
                    "budget_sha256": digest(budget.model_dump()),
                },
            )
        return {
            "campaign_id": campaign_id,
            "plan_sha256": digest(plan),
            "authority": authority,
        }

    def register_task(
        self,
        campaign_id: str,
        payload: dict,
        *,
        engine: str,
        recipe: str,
        plan_sha256: str,
        authority: str,
        actor: str,
        dependencies: list[str] | None = None,
        allow_partial_dependencies: bool = False,
    ) -> dict:
        dependencies = list(dependencies or [])
        if len(dependencies) != len(set(dependencies)):
            raise CampaignError("Dependencies must be unique immutable task IDs.")
        identity = {
            "payload": payload,
            "engine": engine,
            "recipe": recipe,
            "dependencies": dependencies,
            "allow_partial_dependencies": allow_partial_dependencies,
            "plan_sha256": plan_sha256,
        }
        key = digest(identity)
        with self._transaction():
            self._operator(campaign_id, authority)
            campaign, budget = self._approved(campaign_id)
            if plan_sha256 != campaign["plan_sha256"]:
                raise AuthorityError(
                    "Task approval does not bind this exact immutable plan."
                )
            if (
                engine not in budget.allowed_engines
                or recipe not in budget.allowed_recipes
            ):
                raise AuthorityError(
                    "Engine/recipe is outside the approved scientific model scope."
                )
            for parent in dependencies:
                row = self.connection.execute(
                    "SELECT campaign_id FROM campaign_tasks WHERE id=?", (parent,)
                ).fetchone()
                if row is None or row["campaign_id"] != campaign_id:
                    raise CampaignError(
                        "Dependencies must be existing tasks of this campaign."
                    )
            old = self.connection.execute(
                "SELECT id FROM campaign_tasks WHERE campaign_id=? AND task_key=?",
                (campaign_id, key),
            ).fetchone()
            if old is not None:
                return {"task_id": old["id"], "task_key": key, "created": False}
            task_id = str(uuid4())
            self.connection.execute(
                (
                    "INSERT INTO "
                    "campaign_tasks(id,campaign_id,task_key,payload_json,engine,recipe,dependencies_json,allow_partial_dependencies)"
                    " VALUES(?,?,?,?,?,?,?,?)"
                ),
                (
                    task_id,
                    campaign_id,
                    key,
                    canonical_json(payload).decode(),
                    engine,
                    recipe,
                    canonical_json(dependencies).decode(),
                    int(allow_partial_dependencies),
                ),
            )
            self._event(
                campaign_id,
                None,
                0,
                actor,
                "task_registered",
                "Registered immutable scientific task.",
                {"task_id": task_id, "task_key": key},
            )
        return {"task_id": task_id, "task_key": key, "created": True}

    def new_attempt(
        self,
        task_id: str,
        *,
        authority: str,
        actor: str,
        parent_attempt_id: str | None = None,
        compatibility_digest: str | None = None,
        automatic: bool = False,
        failure_class: str | None = None,
    ) -> dict:
        with self._transaction():
            task = self.connection.execute(
                "SELECT * FROM campaign_tasks WHERE id=?", (task_id,)
            ).fetchone()
            if task is None:
                raise CampaignError("Task is absent.")
            self._operator(task["campaign_id"], authority)
            _, budget = self._approved(task["campaign_id"])
            previous = self.connection.execute(
                (
                    "SELECT * FROM campaign_attempts WHERE task_id=? ORDER "
                    "BY number DESC LIMIT 1"
                ),
                (task_id,),
            ).fetchone()
            if previous is not None:
                if previous["state"] not in {"partial", "failed", "cancelled"}:
                    raise CampaignError(
                        "Only a terminal incomplete task may create a resumed attempt."
                    )
                if (
                    parent_attempt_id != previous["id"]
                    or compatibility_digest != task["task_key"]
                ):
                    raise CampaignError(
                        "Resume must retain exact scientific compatibility and "
                        "parent attempt."
                    )
                reservation = self.connection.execute(
                    "SELECT state FROM campaign_reservations WHERE attempt_id=?",
                    (previous["id"],),
                ).fetchone()
                if reservation is not None and reservation["state"] != "settled":
                    raise CampaignError(
                        "Reconcile prior owned work and consumption before resume."
                    )
                if previous["number"] >= 1 + budget.permitted_retries:
                    raise CampaignError(
                        "The approved retry ceiling permits at most two "
                        "additional attempts; no further attempt is authorized."
                    )
                if automatic and (
                    previous["number"] >= 3
                    or failure_class
                    not in {
                        "transient_io",
                        "temporary_resource",
                        "engine_interrupted",
                        "approved_numerical_recovery",
                    }
                    or previous["state"] == "cancelled"
                ):
                    raise CampaignError(
                        "Automatic retry requires a classified permitted failure"
                        " and at most two additional attempts."
                    )
            elif parent_attempt_id is not None:
                raise CampaignError(
                    "A first attempt cannot claim a nonexistent parent."
                )
            elif automatic:
                raise CampaignError("A first attempt is not a classified retry.")
            attempt_id = str(uuid4())
            number = previous["number"] + 1 if previous is not None else 1
            self.connection.execute(
                (
                    "INSERT INTO "
                    "campaign_attempts(id,task_id,number,parent_attempt_id,state,created_at)"
                    " VALUES(?,?,?,?,?,?)"
                ),
                (attempt_id, task_id, number, parent_attempt_id, "draft", time.time()),
            )
            self._event(
                task["campaign_id"],
                attempt_id,
                0,
                actor,
                "transition",
                "Created immutable attempt lineage.",
                {"from": None, "to": "draft", "parent_attempt_id": parent_attempt_id},
            )
        return self.attempt(attempt_id)

    def _current(
        self, attempt_id: str, expected_revision: int
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        row = self.connection.execute(
            "SELECT * FROM campaign_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise CampaignError("Attempt is absent.")
        if type(expected_revision) is not int or row["revision"] != expected_revision:
            raise RevisionConflict(
                "The durable attempt revision changed; refresh before mutation."
            )
        task = self.connection.execute(
            "SELECT * FROM campaign_tasks WHERE id=?", (row["task_id"],)
        ).fetchone()
        return row, task

    def _transition(
        self,
        row: sqlite3.Row,
        task: sqlite3.Row,
        state: str,
        actor: str,
        reason: str,
        *,
        result: dict | None = None,
    ) -> None:
        if state not in EDGES.get(row["state"], set()):
            raise CampaignError(
                f"Forbidden lifecycle transition: {row['state']} -> {state}"
            )
        serialized = canonical_json(result).decode() if result is not None else None
        self.connection.execute(
            "UPDATE campaign_attempts SET state=?,revision=revision+1,result_json=?,"
            "lease_hash=CASE WHEN ? THEN NULL ELSE lease_hash END "
            "WHERE id=? AND revision=?",
            (state, serialized, int(state in TERMINAL), row["id"], row["revision"]),
        )
        self._event(
            task["campaign_id"],
            row["id"],
            row["revision"] + 1,
            actor,
            "transition",
            reason,
            {
                "from": row["state"],
                "to": state,
                "result_sha256": digest(result) if result is not None else None,
            },
        )

    def transition(
        self,
        attempt_id: str,
        expected_revision: int,
        state: str,
        *,
        authority: str,
        actor: str,
        reason: str,
    ) -> dict:
        if state in {"queued", "running", "succeeded", "partial"}:
            raise CampaignError(
                "Use reservation, lease or validated publication for this transition."
            )
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            self._operator(task["campaign_id"], authority)
            self._transition(row, task, state, actor, reason)
        return self.attempt(attempt_id)

    def reserve(
        self,
        attempt_id: str,
        expected_revision: int,
        allocation: Allocation,
        *,
        authority: str,
        actor: str,
    ) -> dict:
        allocation = Allocation.model_validate(allocation)
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            self._operator(task["campaign_id"], authority)
            _, budget = self._approved(task["campaign_id"])
            if row["state"] != "validated":
                raise CampaignError(
                    "Only a validated attempt can reserve and queue work."
                )
            for parent in json.loads(task["dependencies_json"]):
                dependency = self.connection.execute(
                    (
                        "SELECT state FROM campaign_attempts WHERE task_id=? "
                        "ORDER BY number DESC LIMIT 1"
                    ),
                    (parent,),
                ).fetchone()
                eligible = (
                    {"succeeded", "partial"}
                    if task["allow_partial_dependencies"]
                    else {"succeeded"}
                )
                if dependency is None or dependency["state"] not in eligible:
                    raise CampaignError(
                        "An unsuccessful/unrun dependency cannot silently supply"
                        " a child task."
                    )
            records = self.connection.execute(
                "SELECT * FROM campaign_reservations WHERE campaign_id=?",
                (task["campaign_id"],),
            ).fetchall()
            active = [
                Allocation.model_validate_json(item["allocation_json"])
                for item in records
                if item["state"] != "settled"
            ]
            spent = [item for item in records if item["state"] == "settled"]
            exceeded = (
                len(active) + 1 > budget.max_concurrent_workers
                or sum(item.cores for item in active) + allocation.cores
                > budget.max_cpu_cores
                or sum(item.gpu_count for item in active) + allocation.gpu_count
                > budget.max_gpus
                or sum(item.memory_mb for item in active) + allocation.memory_mb
                > budget.max_memory_mb
                or sum(item.scratch_mb for item in active) + allocation.scratch_mb
                > budget.max_scratch_mb
                or sum(item.wall_seconds for item in active)
                + sum(item["charged_wall"] for item in spent)
                + allocation.wall_seconds
                > budget.max_wall_seconds
                or sum(item.cores * item.wall_seconds for item in active)
                + sum(item["charged_cpu"] for item in spent)
                + allocation.cores * allocation.wall_seconds
                > budget.max_cpu_core_seconds
                or sum(item.gpu_count * item.wall_seconds for item in active)
                + sum(item["charged_gpu"] for item in spent)
                + allocation.gpu_count * allocation.wall_seconds
                > budget.max_gpu_seconds
                or len(active) + sum(item["dispatched"] for item in spent) + 1
                > budget.max_tasks
            )
            if exceeded:
                raise BudgetExceeded(
                    "Atomic campaign reservation exceeds an approved "
                    "consumption or concurrent allocation ceiling."
                )
            self.connection.execute(
                (
                    "INSERT INTO "
                    "campaign_reservations(attempt_id,campaign_id,allocation_json,state)"
                    " VALUES(?,?,?,?)"
                ),
                (
                    attempt_id,
                    task["campaign_id"],
                    allocation.model_dump_json(),
                    "reserved",
                ),
            )
            self._transition(
                row,
                task,
                "queued",
                actor,
                "Reserved campaign resources atomically before dispatch.",
            )
            self._event(
                task["campaign_id"],
                attempt_id,
                row["revision"] + 1,
                actor,
                "reservation",
                "Granted bounded resource tokens.",
                allocation.model_dump(),
            )
        return self.attempt(attempt_id)

    def record_dispatch(
        self,
        attempt_id: str,
        expected_revision: int,
        *,
        pid: int,
        authority: str,
        actor: str,
    ) -> dict:
        """Record an actual Popen launch even if it exits before lease acquisition.

        The trusted operator supplies the PID observed from its owned Popen
        handle. This is an operational observation, not launch authorization or
        evidence that a scientific engine calculation succeeded. Missing birth
        metadata remain null; reconciliation then errs toward retaining resources
        if that PID is currently alive.
        """
        if type(pid) is not int or pid <= 0:
            raise CampaignError(
                "Record the positive PID from the actual owned process handle."
            )
        try:
            process_start = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            process_start = None
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            campaign = self._operator(task["campaign_id"], authority)
            reservation = self.connection.execute(
                "SELECT * FROM campaign_reservations WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if (
                row["state"] != "queued"
                or reservation is None
                or reservation["state"] != "reserved"
            ):
                raise CampaignError(
                    "Record actual launch only for its queued reserved attempt."
                )
            if reservation["dispatched"]:
                if row["pid"] != pid:
                    raise CampaignError(
                        "An observed dispatch cannot be reassigned to another process."
                    )
                return self.attempt(attempt_id)
            budget = CampaignBudget.model_validate_json(campaign["budget_json"])
            self.connection.execute(
                "UPDATE campaign_reservations SET dispatched=1 WHERE attempt_id=?",
                (attempt_id,),
            )
            self.connection.execute(
                (
                    "UPDATE campaign_attempts SET "
                    "pid=?,process_start=?,host=?,revision=revision+1 WHERE "
                    "id=? AND revision=?"
                ),
                (
                    pid,
                    process_start,
                    socket.gethostname(),
                    attempt_id,
                    expected_revision,
                ),
            )
            self._event(
                task["campaign_id"],
                attempt_id,
                expected_revision + 1,
                actor,
                "dispatch_observed",
                "Recorded actual owned process launch; no inferred scientific success.",
                {
                    "observed_pid": pid,
                    "observed_process_start": process_start,
                    "host": socket.gethostname(),
                    "approval_active_at_observation": not campaign["revoked"]
                    and time.time() < budget.expires_at_unix,
                },
            )
        return self.attempt(attempt_id)

    def lease(
        self,
        attempt_id: str,
        expected_revision: int,
        *,
        pid: int,
        actor: str,
        authority: str,
        heartbeat_seconds: float = 10,
        lease_seconds: float = 60,
    ) -> dict:
        if (
            not math.isfinite(heartbeat_seconds)
            or not math.isfinite(lease_seconds)
            or heartbeat_seconds <= 0
            or lease_seconds < 3 * heartbeat_seconds
        ):
            raise CampaignError(
                "Lease must cover at least three positive heartbeat intervals."
            )
        process = psutil.Process(pid)
        start = process.create_time()
        token = secrets.token_urlsafe(32)
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            self._operator(task["campaign_id"], authority)
            self._approved(task["campaign_id"])
            reservation = self.connection.execute(
                "SELECT state FROM campaign_reservations WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if (
                row["state"] != "queued"
                or reservation is None
                or reservation["state"] != "reserved"
            ):
                raise CampaignError("Only the queued reserved attempt may be leased.")
            if row["pid"] is not None and (
                row["pid"] != pid
                or row["process_start"] is not None
                and row["process_start"] != start
            ):
                raise AuthorityError(
                    "Lease process identity differs from the observed owned dispatch."
                )
            generation = task["next_generation"] + 1
            self.connection.execute(
                "UPDATE campaign_tasks SET next_generation=? WHERE id=?",
                (generation, task["id"]),
            )
            self.connection.execute(
                (
                    "UPDATE campaign_attempts SET "
                    "lease_hash=?,generation=?,lease_expires=?,heartbeat_seconds=?,lease_seconds=?,host=?,pid=?,process_start=?"
                    " WHERE id=?"
                ),
                (
                    _token_hash(token),
                    generation,
                    time.time() + lease_seconds,
                    heartbeat_seconds,
                    lease_seconds,
                    socket.gethostname(),
                    pid,
                    start,
                    attempt_id,
                ),
            )
            self.connection.execute(
                "UPDATE campaign_reservations SET dispatched=1 WHERE attempt_id=?",
                (attempt_id,),
            )
            self._transition(
                row, task, "running", actor, "Leased an observed local owned process."
            )
        return {
            **self.attempt(attempt_id),
            "lease_token": token,
            "lease_generation": generation,
        }

    def _lease(self, row: sqlite3.Row, token: str, generation: int) -> None:
        if (
            row["state"] in TERMINAL
            or not isinstance(token, str)
            or type(generation) is not int
            or generation != row["generation"]
            or not row["lease_hash"]
            or not secrets.compare_digest(row["lease_hash"], _token_hash(token))
            or time.time() >= row["lease_expires"]
        ):
            raise AuthorityError(
                "Worker mutation rejected for an expired/stale attempt lease."
            )

    def heartbeat(
        self, attempt_id: str, *, lease_token: str, lease_generation: int
    ) -> None:
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM campaign_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise CampaignError("Attempt is absent.")
            task = self.connection.execute(
                "SELECT * FROM campaign_tasks WHERE id=?", (row["task_id"],)
            ).fetchone()
            self._approved(task["campaign_id"])
            self._lease(row, lease_token, lease_generation)
            self.connection.execute(
                "UPDATE campaign_attempts SET lease_expires=? WHERE id=?",
                (time.time() + row["lease_seconds"], attempt_id),
            )

    def worker_transition(
        self,
        attempt_id: str,
        expected_revision: int,
        state: str,
        *,
        lease_token: str,
        lease_generation: int,
        actor: str,
        reason: str,
    ) -> dict:
        if state not in {"collecting", "validating", "failed"}:
            raise CampaignError(
                "Worker transitions cannot bypass validated publication "
                "or operator cancellation."
            )
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            self._approved(task["campaign_id"])
            self._lease(row, lease_token, lease_generation)
            self._transition(row, task, state, actor, reason)
        return self.attempt(attempt_id)

    def cancel(
        self,
        attempt_id: str,
        expected_revision: int,
        *,
        authority: str,
        actor: str,
        reason: str,
    ) -> dict:
        # Cancellation invalidates the lease, but never assumes its process stopped
        # or frees possibly consumed resources before owned-process reconciliation.
        return self.transition(
            attempt_id,
            expected_revision,
            "cancelled",
            authority=authority,
            actor=actor,
            reason=reason,
        )

    def _settle(
        self, row: sqlite3.Row, task: sqlite3.Row, usage: MeasuredUsage, actor: str
    ) -> bool:
        record = self.connection.execute(
            "SELECT * FROM campaign_reservations WHERE attempt_id=?", (row["id"],)
        ).fetchone()
        if record is None:
            raise CampaignError(
                "Only an actual reserved attempt has resource consumption to settle."
            )
        if record["state"] == "settled":
            if record["usage_json"] == usage.model_dump_json():
                return False
            raise CampaignError("Settled actual consumption cannot be rewritten.")
        allocation = Allocation.model_validate_json(record["allocation_json"])
        cpu = usage.cpu_core_seconds
        gpu = usage.gpu_seconds
        if not record["dispatched"] and any(
            value not in (None, 0) for value in (usage.wall_seconds, cpu, gpu)
        ):
            raise CampaignError(
                "An undispatched task cannot claim observed engine consumption."
            )
        # Unknown measurements remain null. Conservative budget charges are an
        # explicitly separate accounting policy, never fabricated telemetry.
        charged_cpu = (
            cpu
            if cpu is not None
            else allocation.cores * allocation.wall_seconds
            if record["dispatched"]
            else 0
        )
        charged_gpu = (
            gpu
            if gpu is not None
            else allocation.gpu_count * allocation.wall_seconds
            if record["dispatched"]
            else 0
        )
        self.connection.execute(
            "UPDATE campaign_reservations SET state='settled',usage_json=?,"
            "charged_wall=?,charged_cpu=?,charged_gpu=? WHERE attempt_id=?",
            (
                usage.model_dump_json(),
                usage.wall_seconds,
                charged_cpu,
                charged_gpu,
                row["id"],
            ),
        )
        overrun = (
            usage.wall_seconds > allocation.wall_seconds
            or charged_cpu > allocation.cores * allocation.wall_seconds
            or charged_gpu > allocation.gpu_count * allocation.wall_seconds
            or usage.peak_memory_mb is not None
            and usage.peak_memory_mb > allocation.memory_mb
            or usage.peak_scratch_mb is not None
            and usage.peak_scratch_mb > allocation.scratch_mb
        )
        self._event(
            task["campaign_id"],
            row["id"],
            row["revision"],
            actor,
            "settlement",
            "Settled actual usage and released resources.",
            {
                "measured_usage": usage.model_dump(),
                "charged_cpu_core_seconds": charged_cpu,
                "charged_gpu_seconds": charged_gpu,
                "allocation_exceeded": overrun,
                "unknown_measurement_policy": (
                    "charge_reserved_ceiling_without_inventing_measurements"
                ),
            },
        )
        return overrun

    def publish(
        self,
        attempt_id: str,
        expected_revision: int,
        result: dict,
        usage: MeasuredUsage,
        publication: Callable[[], Any],
        *,
        state: Literal["succeeded", "partial", "failed"],
        lease_token: str,
        lease_generation: int,
        actor: str,
    ) -> dict:
        """Publish under current lease/CAS ownership; rollback if publication fails.

        Filesystem and SQLite are not one distributed transaction: a crash after
        publication before COMMIT leaves an authentic artifact and unfinished
        attempt. Recovery must verify/reconcile it; no false success is committed.
        """
        usage = MeasuredUsage.model_validate(usage)
        if not isinstance(result, dict) or not result:
            raise CampaignError(
                "Publication requires a validated, identified result manifest."
            )
        with self._transaction():
            row, task = self._current(attempt_id, expected_revision)
            self._approved(task["campaign_id"])
            self._lease(row, lease_token, lease_generation)
            if row["state"] != "validating":
                raise CampaignError(
                    "Collect and validate genuine artifacts before publication."
                )
            publication()
            overrun = self._settle(row, task, usage, actor)
            if overrun and state == "succeeded":
                state = "partial"
            self._transition(
                row,
                task,
                state,
                actor,
                (
                    "Published validated immutable result under current "
                    "coordinator ownership."
                ),
                result=result,
            )
        return self.attempt(attempt_id)

    @staticmethod
    def _owned_process_alive(row: sqlite3.Row) -> bool:
        if row["pid"] is None:
            return False
        if row["host"] != socket.gethostname():
            raise CampaignError(
                "Remote process ownership requires a qualified scheduler"
                " reconciliation adapter."
            )
        try:
            process = psutil.Process(row["pid"])
            return (
                (
                    row["process_start"] is None
                    or process.create_time() == row["process_start"]
                )
                and process.status() != psutil.STATUS_ZOMBIE
                and process.is_running()
            )
        except psutil.NoSuchProcess:
            return False

    def reconcile(
        self,
        attempt_id: str,
        usage: MeasuredUsage,
        *,
        authority: str,
        actor: str,
        reason: str,
    ) -> dict:
        usage = MeasuredUsage.model_validate(usage)
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM campaign_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise CampaignError("Attempt is absent.")
            task = self.connection.execute(
                "SELECT * FROM campaign_tasks WHERE id=?", (row["task_id"],)
            ).fetchone()
            self._operator(task["campaign_id"], authority)
            if self._owned_process_alive(row):
                raise CampaignError(
                    "Reconciliation cannot release resources while the "
                    "original owned process is alive."
                )
            if row["state"] not in TERMINAL:
                if row["state"] == "running":
                    self._transition(row, task, "failed", actor, reason)
                    row = self.connection.execute(
                        "SELECT * FROM campaign_attempts WHERE id=?", (attempt_id,)
                    ).fetchone()
                else:
                    raise CampaignError(
                        "Reconcile a terminal attempt or a stopped running orphan."
                    )
            self._settle(row, task, usage, actor)
        return self.attempt(attempt_id)

    def revoke(
        self, campaign_id: str, *, authority: str, actor: str, reason: str
    ) -> None:
        with self._transaction():
            campaign = self._operator(campaign_id, authority)
            if not campaign["revoked"]:
                self.connection.execute(
                    "UPDATE campaigns SET revoked=1,revision=revision+1 WHERE id=?",
                    (campaign_id,),
                )
                self._event(
                    campaign_id,
                    None,
                    campaign["revision"] + 1,
                    actor,
                    "revocation",
                    reason,
                    {
                        "future_work_authorized": False,
                        "completed_lineage_unchanged": True,
                    },
                )

    def attempt(self, attempt_id: str) -> dict:
        row = self.connection.execute(
            "SELECT * FROM campaign_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise CampaignError("Attempt is absent.")
        return {key: row[key] for key in row.keys() if key != "lease_hash"}

    def events(self, campaign_id: str) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM campaign_events WHERE campaign_id=? ORDER BY sequence",
                (campaign_id,),
            )
        ]

    def accounting(self, campaign_id: str) -> list[dict]:
        self._campaign(campaign_id)
        return [
            dict(row)
            for row in self.connection.execute(
                (
                    "SELECT * FROM campaign_reservations WHERE campaign_id=?"
                    " ORDER BY attempt_id"
                ),
                (campaign_id,),
            )
        ]
