"""ACID-compliant centralized SQLite task and job queue engine.

Provenance & Specifications:
- Method Matrix [M]: Centralized ACID state machine eliminating multi-process race conditions.
- SQLite Pragmas [D]: WAL journal mode, busy_timeout=5000, synchronous=NORMAL, foreign_keys=ON.
- Concurrency [E]: Immediate write transactions with exponential backoff on lock contention.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class TaskRecord(BaseModel):
    """Immutable representation of a task state record."""

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

    @property
    def payload(self) -> Dict[str, Any]:
        """Parsed payload dictionary."""
        return cast(Dict[str, Any], json.loads(self.payload_json))

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """Parsed result dictionary if completed."""
        return cast(Dict[str, Any], json.loads(self.result_json)) if self.result_json else None


class TaskCreate(BaseModel):
    """Specification for submitting a new task."""

    model_config = ConfigDict(frozen=True)

    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    max_retries: int = 3


class SQLiteTaskQueue:
    """ACID task queue engine backed by SQLite in WAL mode."""

    def __init__(self, db_path: Union[str, Path], timeout_sec: float = 5.0) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_sec = timeout_sec

        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout_sec,
            isolation_level=None,  # Autocommit mode for explicit transaction boundaries
        )
        self._conn.row_factory = sqlite3.Row
        self._init_pragmas_and_schema()

    def _init_pragmas_and_schema(self) -> None:
        """Apply performance and durability pragmas and initialize task schema."""
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.execute("PRAGMA foreign_keys = ON;")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
                priority INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                locked_by_pid INTEGER,
                locked_by_host TEXT,
                created_at REAL NOT NULL,
                heartbeat_ts REAL,
                completed_at REAL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_fetch ON tasks (state, priority DESC, created_at ASC);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_heartbeat ON tasks (state, heartbeat_ts);"
        )

    def enqueue_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        max_retries: int = 3,
    ) -> str:
        """Enqueue a new task and return its unique identifier."""
        task_id = str(uuid.uuid4())
        payload_str = json.dumps(payload)
        now = time.time()

        self._conn.execute(
            """
            INSERT INTO tasks (
                task_id, task_type, state, priority, payload_json,
                retry_count, max_retries, created_at
            ) VALUES (?, ?, 'PENDING', ?, ?, 0, ?, ?);
            """,
            (task_id, task_type, priority, payload_str, max_retries, now),
        )
        return task_id

    def lease_task(self, worker_pid: int, worker_host: str) -> Optional[TaskRecord]:
        """Atomically claim the highest-priority pending task lease."""
        now = time.time()
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                self._conn.execute("BEGIN IMMEDIATE;")
                cursor = self._conn.execute(
                    """
                    SELECT task_id, task_type, state, priority, payload_json,
                           result_json, error_message, retry_count, max_retries,
                           locked_by_pid, locked_by_host, created_at, heartbeat_ts, completed_at
                    FROM tasks
                    WHERE state = 'PENDING'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1;
                    """
                )
                row = cursor.fetchone()

                if row is None:
                    self._conn.execute("COMMIT;")
                    return None

                selected_task_id = row["task_id"]

                self._conn.execute(
                    """
                    UPDATE tasks
                    SET state = 'RUNNING',
                        locked_by_pid = ?,
                        locked_by_host = ?,
                        heartbeat_ts = ?
                    WHERE task_id = ?;
                    """,
                    (worker_pid, worker_host, now, selected_task_id),
                )
                self._conn.execute("COMMIT;")

                task_dict = dict(row)
                task_dict["state"] = "RUNNING"
                task_dict["locked_by_pid"] = worker_pid
                task_dict["locked_by_host"] = worker_host
                task_dict["heartbeat_ts"] = now

                return TaskRecord(**task_dict)

            except sqlite3.OperationalError as err:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK;")
                if "locked" in str(err) or "busy" in str(err):
                    backoff = 0.05 * (2**attempt)
                    time.sleep(backoff)
                else:
                    raise err

        return None

    def heartbeat(self, task_id: str, worker_pid: int) -> bool:
        """Update lease heartbeat timestamp for an active task."""
        now = time.time()
        cursor = self._conn.execute(
            """
            UPDATE tasks
            SET heartbeat_ts = ?
            WHERE task_id = ? AND locked_by_pid = ? AND state = 'RUNNING';
            """,
            (now, task_id, worker_pid),
        )
        return cursor.rowcount > 0

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """Transition task state to COMPLETED and persist physics result payload."""
        now = time.time()
        result_str = json.dumps(result)
        cursor = self._conn.execute(
            """
            UPDATE tasks
            SET state = 'COMPLETED',
                result_json = ?,
                completed_at = ?
            WHERE task_id = ? AND state = 'RUNNING';
            """,
            (result_str, now, task_id),
        )
        if cursor.rowcount == 0:
            raise RuntimeError(f"Task {task_id} cannot be completed because it is not in RUNNING state")

    def fail_task(self, task_id: str, error_message: str, can_retry: bool = True) -> None:
        """Record task execution failure with retry escalation logic."""
        now = time.time()
        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            cursor = self._conn.execute(
                "SELECT retry_count, max_retries FROM tasks WHERE task_id = ?;",
                (task_id,),
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.execute("COMMIT;")
                return

            retry_count = row["retry_count"]
            max_retries = row["max_retries"]
            new_retry_count = retry_count + 1

            if can_retry and new_retry_count < max_retries:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET state = 'PENDING',
                        retry_count = ?,
                        error_message = ?,
                        locked_by_pid = NULL,
                        locked_by_host = NULL,
                        heartbeat_ts = NULL
                    WHERE task_id = ?;
                    """,
                    (new_retry_count, error_message, task_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET state = 'FAILED',
                        retry_count = ?,
                        error_message = ?,
                        completed_at = ?
                    WHERE task_id = ?;
                    """,
                    (new_retry_count, error_message, now, task_id),
                )
            self._conn.execute("COMMIT;")
        except sqlite3.OperationalError:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK;")
            raise

    def reclaim_orphaned_tasks(self, timeout_grace_sec: float = 30.0) -> List[str]:
        """Detect tasks with expired worker heartbeats and reclaim or fail them."""
        now = time.time()
        reclaimed_ids: List[str] = []

        try:
            self._conn.execute("BEGIN IMMEDIATE;")
            cursor = self._conn.execute(
                """
                SELECT task_id, retry_count, max_retries, heartbeat_ts
                FROM tasks
                WHERE state = 'RUNNING' AND (? - heartbeat_ts) > ?;
                """,
                (now, timeout_grace_sec),
            )
            rows = cursor.fetchall()

            for row in rows:
                task_id = row["task_id"]
                retry_count = row["retry_count"]
                max_retries = row["max_retries"]
                new_retry_count = retry_count + 1
                reclaimed_ids.append(task_id)

                if new_retry_count < max_retries:
                    self._conn.execute(
                        """
                        UPDATE tasks
                        SET state = 'PENDING',
                            retry_count = ?,
                            error_message = 'Reclaimed: Heartbeat lease expired',
                            locked_by_pid = NULL,
                            locked_by_host = NULL,
                            heartbeat_ts = NULL
                        WHERE task_id = ?;
                        """,
                        (new_retry_count, task_id),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE tasks
                        SET state = 'FAILED',
                            retry_count = ?,
                            error_message = 'Failed: Heartbeat lease expired and retries exhausted',
                            completed_at = ?
                        WHERE task_id = ?;
                        """,
                        (new_retry_count, now, task_id),
                    )

            self._conn.execute("COMMIT;")
            return reclaimed_ids
        except sqlite3.OperationalError:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK;")
            raise


    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve task record by ID."""
        cursor = self._conn.execute(
            """
            SELECT task_id, task_type, state, priority, payload_json,
                   result_json, error_message, retry_count, max_retries,
                   locked_by_pid, locked_by_host, created_at, heartbeat_ts, completed_at
            FROM tasks
            WHERE task_id = ?;
            """,
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return TaskRecord(**dict(row))

    def close(self) -> None:
        """Close SQLite database connection."""
        self._conn.close()

    def __enter__(self) -> SQLiteTaskQueue:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
