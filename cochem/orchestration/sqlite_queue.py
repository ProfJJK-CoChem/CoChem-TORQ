"""SQLite WAL task queue with fenced, unguessable attempt leases.

Use a local filesystem and one coordinator for remote workers. A task identifier
or PID alone never authorizes worker mutations. Migration invalidates legacy
RUNNING attempts because they have no verifiable ownership credentials.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field


class LeaseRequiredError(ValueError):
    """The caller omitted the credentials returned by lease_task()."""


class LeaseLostError(RuntimeError):
    """The attempt no longer owns this task, or its result conflicts."""


class TaskRecord(BaseModel):
    """Immutable snapshot; preserve lease credentials for this attempt only."""
    model_config = ConfigDict(frozen=True)
    task_id: str
    task_type: str
    state: str
    priority: int = 0
    payload_json: str
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    locked_by_pid: Optional[int] = None
    locked_by_host: Optional[str] = None
    created_at: float
    heartbeat_ts: Optional[float] = None
    completed_at: Optional[float] = None
    lease_token: Optional[str] = Field(default=None, repr=False)
    lease_generation: int = 0

    @property
    def payload(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(self.payload_json))

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return cast(Dict[str, Any], json.loads(self.result_json)) if self.result_json else None


class TaskCreate(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    max_retries: int = 3


class SQLiteTaskQueue:
    """Process-local connection; use a separate instance in each worker/thread."""

    def __init__(self, db_path: Union[str, Path], timeout_sec: float = 5.0) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec
        self._conn = sqlite3.connect(str(self.db_path), timeout=timeout_sec, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_pragmas_and_schema()

    def _init_pragmas_and_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {max(0, int(self.timeout_sec * 1000))}")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, task_type TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED')),
                    priority INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL,
                    result_json TEXT, error_message TEXT, retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3, locked_by_pid INTEGER,
                    locked_by_host TEXT, created_at REAL NOT NULL, heartbeat_ts REAL,
                    completed_at REAL, lease_token TEXT, lease_generation INTEGER NOT NULL DEFAULT 0
                )
            """)
            columns = {row['name'] for row in self._conn.execute("PRAGMA table_info(tasks)")}
            if 'lease_token' not in columns:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN lease_token TEXT")
            if 'lease_generation' not in columns:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN lease_generation INTEGER NOT NULL DEFAULT 0")
            # Old workers must be stopped before upgrading. Their credential-free
            # completions are rejected; never grant them a newly generated token.
            self._conn.execute("""
                UPDATE tasks SET
                    state = CASE WHEN retry_count + 1 < max_retries THEN 'PENDING' ELSE 'FAILED' END,
                    retry_count = retry_count + 1,
                    error_message = 'Lease migration: legacy attempt invalidated; ownership unavailable',
                    locked_by_pid = NULL, locked_by_host = NULL, heartbeat_ts = NULL,
                    completed_at = CASE WHEN retry_count + 1 < max_retries THEN NULL ELSE ? END
                WHERE state = 'RUNNING' AND lease_token IS NULL
            """, (time.time(),))
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_fetch ON tasks (state, priority DESC, created_at ASC)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_heartbeat ON tasks (state, heartbeat_ts)")
            self._conn.execute("COMMIT")
        except Exception:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            self._conn.close()
            raise

    @staticmethod
    def _credentials(lease_token: Optional[str], lease_generation: Optional[int]) -> None:
        if (not isinstance(lease_token, str) or not lease_token or
                type(lease_generation) is not int or lease_generation < 1):
            raise LeaseRequiredError("Pass lease_token and lease_generation from the original leased TaskRecord; task ID/PID alone cannot authorize a mutation")

    def enqueue_task(self, task_type: str, payload: Dict[str, Any], priority: int = 0,
                     max_retries: int = 3) -> str:
        task_id = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO tasks (task_id, task_type, state, priority, payload_json,
                               retry_count, max_retries, created_at)
            VALUES (?, ?, 'PENDING', ?, ?, 0, ?, ?)
        """, (task_id, task_type, priority, json.dumps(payload, allow_nan=False), max_retries, time.time()))
        return task_id

    def lease_task(self, worker_pid: int, worker_host: str, *, task_id: Optional[str] = None) -> Optional[TaskRecord]:
        """Atomically claim a task and advance its monotonic fencing generation."""
        for attempt in range(5):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute("""
                    SELECT task_id FROM tasks WHERE state = 'PENDING' AND (? IS NULL OR task_id = ?)
                    ORDER BY priority DESC, created_at ASC LIMIT 1
                """, (task_id, task_id)).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute("""
                    UPDATE tasks SET state = 'RUNNING', locked_by_pid = ?, locked_by_host = ?,
                        heartbeat_ts = ?, lease_token = ?, lease_generation = lease_generation + 1,
                        completed_at = NULL
                    WHERE task_id = ? AND state = 'PENDING'
                """, (worker_pid, worker_host, time.time(), secrets.token_urlsafe(32), row['task_id']))
                task = self.get_task(row['task_id'])
                self._conn.execute("COMMIT")
                return task
            except sqlite3.OperationalError as err:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                if 'locked' not in str(err) and 'busy' not in str(err):
                    raise
                if attempt == 4:
                    raise
                time.sleep(0.05 * 2**attempt)
        return None

    def heartbeat(self, task_id: str, worker_pid: Optional[int] = None, *,
                  lease_token: Optional[str] = None, lease_generation: Optional[int] = None) -> bool:
        self._credentials(lease_token, lease_generation)
        cursor = self._conn.execute("""
            UPDATE tasks SET heartbeat_ts = ? WHERE task_id = ? AND state = 'RUNNING'
                AND lease_token = ? AND lease_generation = ?
                AND (? IS NULL OR locked_by_pid = ?)
        """, (time.time(), task_id, lease_token, lease_generation, worker_pid, worker_pid))
        return cursor.rowcount == 1

    def complete_task(self, task_id: str, result: Dict[str, Any], *,
                      lease_token: Optional[str] = None, lease_generation: Optional[int] = None) -> None:
        """Commit once; identical repeat completions for the same attempt are safe."""
        self._credentials(lease_token, lease_generation)
        result_str = json.dumps(result, sort_keys=True, allow_nan=False)
        cursor = self._conn.execute("""
            UPDATE tasks SET state = 'COMPLETED', result_json = ?, completed_at = ?
            WHERE task_id = ? AND state = 'RUNNING' AND lease_token = ? AND lease_generation = ?
        """, (result_str, time.time(), task_id, lease_token, lease_generation))
        if cursor.rowcount == 0:
            task = self.get_task(task_id)
            if (task is not None and task.state == 'COMPLETED' and task.lease_token == lease_token
                    and task.lease_generation == lease_generation and task.result_json == result_str):
                return
            raise LeaseLostError(f"Task {task_id}: completion rejected for stale lease or conflicting result")

    def fail_task(self, task_id: str, error_message: str, can_retry: bool = True, *,
                  lease_token: Optional[str] = None, lease_generation: Optional[int] = None) -> None:
        self._credentials(lease_token, lease_generation)
        cursor = self._conn.execute("""
            UPDATE tasks SET
                state = CASE WHEN ? AND retry_count + 1 < max_retries THEN 'PENDING' ELSE 'FAILED' END,
                retry_count = retry_count + 1, error_message = ?, locked_by_pid = NULL,
                locked_by_host = NULL, heartbeat_ts = NULL, lease_token = NULL,
                completed_at = CASE WHEN ? AND retry_count + 1 < max_retries THEN NULL ELSE ? END
            WHERE task_id = ? AND state = 'RUNNING' AND lease_token = ? AND lease_generation = ?
        """, (can_retry, error_message, can_retry, time.time(), task_id, lease_token, lease_generation))
        if cursor.rowcount == 0:
            raise LeaseLostError(f"Task {task_id}: failure rejected for stale lease")

    def complete_task_with_publication(self, task_id: str, result: Dict[str, Any],
                                       publish: Callable[[], Any], *, lease_token: Optional[str] = None,
                                       lease_generation: Optional[int] = None) -> None:
        """Fence one atomic filesystem publication while holding coordinator ownership.

        SQLite and the filesystem cannot share a distributed transaction. A crash
        after rename and before COMMIT leaves an authentic sealed bundle and an
        unfinished task, never a completed task whose bundle was not published.
        Recovery must verify that bundle and reconcile the recorded attempt.
        """
        self._credentials(lease_token, lease_generation)
        serialized = json.dumps(result, sort_keys=True, allow_nan=False)
        try:
            self._conn.execute('BEGIN IMMEDIATE')
            task = self.get_task(task_id)
            if (task is not None and task.state == 'COMPLETED' and task.lease_token == lease_token
                    and task.lease_generation == lease_generation and task.result_json == serialized):
                self._conn.execute('COMMIT')
                return
            if (task is None or task.state != 'RUNNING' or task.lease_token != lease_token
                    or task.lease_generation != lease_generation):
                raise LeaseLostError(f'Task {task_id}: artifact publication rejected for stale lease')
            publish()
            self._conn.execute("""
                UPDATE tasks SET state = 'COMPLETED', result_json = ?, completed_at = ?
                WHERE task_id = ? AND state = 'RUNNING' AND lease_token = ? AND lease_generation = ?
            """, (serialized, time.time(), task_id, lease_token, lease_generation))
            self._conn.execute('COMMIT')
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute('ROLLBACK')
            raise

    def reclaim_orphaned_tasks(self, timeout_grace_sec: float = 30.0) -> List[str]:
        """Coordinator operation: transactionally invalidate expired attempts."""
        if timeout_grace_sec < 0:
            raise ValueError('timeout_grace_sec must be nonnegative')
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute("""
                SELECT task_id FROM tasks WHERE state = 'RUNNING' AND (? - heartbeat_ts) > ?
            """, (time.time(), timeout_grace_sec)).fetchall()
            for row in rows:
                self._conn.execute("""
                    UPDATE tasks SET
                        state = CASE WHEN retry_count + 1 < max_retries THEN 'PENDING' ELSE 'FAILED' END,
                        retry_count = retry_count + 1, error_message = 'Heartbeat lease expired',
                        locked_by_pid = NULL, locked_by_host = NULL, heartbeat_ts = NULL, lease_token = NULL,
                        completed_at = CASE WHEN retry_count + 1 < max_retries THEN NULL ELSE ? END
                    WHERE task_id = ? AND state = 'RUNNING'
                """, (time.time(), row['task_id']))
            self._conn.execute("COMMIT")
            return [row['task_id'] for row in rows]
        except Exception:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return TaskRecord(**dict(row)) if row is not None else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteTaskQueue:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
