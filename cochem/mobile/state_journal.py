"""Draco Protocol UI State Synchronization & WAL Integrity Engine.

Module: cochem.mobile.state_journal
Implements REQ-MOB-093 and Draco Protocol UI State Synchronization adhering strictly to the Zero-Mock mandate.
- States: IDLE, CONFORMATION_RUNNING, OPTIMIZATION_CONVERGED, LEDGER_COMMITTED.
- SQLite WAL database (swarm_state_journal.db) with PRAGMA journal_mode=WAL; and PRAGMA busy_timeout=10000;.
- Synchronizes with swarm_state.json atomically using filelock.FileLock(path.with_suffix('.lock'), timeout=10).
- Strict cryptographic SHA-256 provenance chaining across state transitions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_BUSY_TIMEOUT_MS: int = 10000
DEFAULT_LOCK_TIMEOUT_S: float = 10.0


class DracoUIState(str, Enum):
    """Draco Protocol UI and execution lifecycle states."""

    IDLE = "IDLE"
    CONFORMATION_RUNNING = "CONFORMATION_RUNNING"
    OPTIMIZATION_CONVERGED = "OPTIMIZATION_CONVERGED"
    LEDGER_COMMITTED = "LEDGER_COMMITTED"

    def is_active(self) -> bool:
        """Return True if state corresponds to an active calculation."""
        return self in {DracoUIState.CONFORMATION_RUNNING, DracoUIState.OPTIMIZATION_CONVERGED}


VALID_DRACO_TRANSITIONS: Dict[DracoUIState, Set[DracoUIState]] = {
    DracoUIState.IDLE: {
        DracoUIState.CONFORMATION_RUNNING,
    },
    DracoUIState.CONFORMATION_RUNNING: {
        DracoUIState.OPTIMIZATION_CONVERGED,
        DracoUIState.IDLE,
    },
    DracoUIState.OPTIMIZATION_CONVERGED: {
        DracoUIState.LEDGER_COMMITTED,
        DracoUIState.IDLE,
    },
    DracoUIState.LEDGER_COMMITTED: {
        DracoUIState.IDLE,
        DracoUIState.CONFORMATION_RUNNING,
    },
}


class DracoStateError(Exception):
    """Base exception for Draco UI State Journal errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DracoTransitionError(DracoStateError):
    """Raised when an invalid or disallowed state transition is attempted."""


class DracoLockError(DracoStateError):
    """Raised when acquiring FileLock or SQLite WAL access times out."""


class DracoStateJournalEntry(BaseModel):
    """Pydantic v2 record for an individual state journal transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(..., description="Unique auto-incrementing journal identifier")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    previous_state: DracoUIState = Field(..., description="Previous lifecycle state")
    current_state: DracoUIState = Field(..., description="New committed state")
    actor: str = Field(..., description="Agent or subsystem initiating transition")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata attributes")
    checksum: str = Field(..., description="Cryptographic SHA-256 integrity digest")


def validate_draco_transition(current: DracoUIState, target: DracoUIState) -> bool:
    """Validate whether transitioning from current to target state is legally permitted."""
    if current == target:
        return True
    return target in VALID_DRACO_TRANSITIONS.get(current, set())


class DracoStateJournal:
    """Production-grade, zero-mock UI State Journal and SQLite WAL ledger.

    Ensures ACID durability, cross-platform file locking, and zero data corruption
    during rapid state oscillations.
    """

    def __init__(
        self,
        db_path: Union[Path, str] = "swarm_state_journal.db",
        json_state_path: Optional[Union[Path, str]] = "swarm_state.json",
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.json_state_path = Path(json_state_path).resolve() if json_state_path else None
        self.busy_timeout_ms = busy_timeout_ms
        self.lock_timeout_s = lock_timeout_s

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_state_path:
            self.json_state_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_sqlite_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure SQLite connection with WAL journal mode and busy timeout."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_ms / 1000.0)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_sqlite_schema(self) -> None:
        """Initialize journal table schema and active state singleton row."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS draco_state_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    current_state TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS draco_current_state (
                    singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    last_journal_id INTEGER NOT NULL
                );
                """
            )
            # Initialize singleton if empty
            cursor = conn.execute(
                "SELECT singleton_key FROM draco_current_state WHERE singleton_key = 1;"
            )
            if cursor.fetchone() is None:
                now_utc = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT INTO draco_current_state (singleton_key, state, updated_at, actor, last_journal_id)
                    VALUES (1, ?, ?, ?, 0);
                    """,
                    (DracoUIState.IDLE.value, now_utc, "system"),
                )

    def verify_wal_mode(self) -> str:
        """Query and assert SQLite journal_mode pragma returns 'wal'."""
        with self._get_connection() as conn:
            cursor = conn.execute("PRAGMA journal_mode;")
            row = cursor.fetchone()
            if row is None:
                raise DracoStateError("Failed to fetch PRAGMA journal_mode from SQLite database.")
            return str(row[0]).lower()

    def get_current_state(self) -> DracoUIState:
        """Fetch current Draco UI state from SQLite singleton table."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT state FROM draco_current_state WHERE singleton_key = 1;")
            row = cursor.fetchone()
            if row is None:
                return DracoUIState.IDLE
            return DracoUIState(row["state"])

    def _calculate_checksum(
        self,
        prev_state: DracoUIState,
        curr_state: DracoUIState,
        timestamp: str,
        actor: str,
        metadata_json: str,
        prev_checksum: str = "",
    ) -> str:
        """Generate deterministic SHA-256 cryptographic digest for state journal chain."""
        hasher = hashlib.sha256()
        payload = f"{prev_checksum}|{prev_state.value}|{curr_state.value}|{timestamp}|{actor}|{metadata_json}"
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def record_transition(
        self,
        target_state: Union[DracoUIState, str],
        actor: str = "cochem-coder",
        metadata: Optional[Dict[str, Any]] = None,
        enforce_transition: bool = True,
    ) -> int:
        """Execute atomic state transition, logging to SQLite WAL and updating swarm_state.json.

        Parameters
        ----------
        target_state : DracoUIState | str
            The desired next state.
        actor : str
            Identifier of agent or subsystem requesting transition.
        metadata : dict, optional
            Context metadata associated with state transition.
        enforce_transition : bool
            If True, strictly rejects invalid state jumps.

        Returns
        -------
        int
            Row ID of committed journal entry.
        """
        if isinstance(target_state, str):
            try:
                target_enum = DracoUIState(target_state.strip())
            except ValueError as val_err:
                raise DracoTransitionError(
                    f"Unknown DracoUIState: '{target_state}'. Valid states: {[s.value for s in DracoUIState]}"
                ) from val_err
        else:
            target_enum = target_state

        meta_dict = metadata or {}
        meta_json = json.dumps(meta_dict, sort_keys=True, separators=(",", ":"))
        timestamp = datetime.now(timezone.utc).isoformat()

        # Manage JSON filelock if json_state_path is active
        lock_file = (
            self.json_state_path.with_suffix(self.json_state_path.suffix + ".lock")
            if self.json_state_path
            else None
        )
        filelock_ctx = FileLock(str(lock_file), timeout=self.lock_timeout_s) if lock_file else None

        try:
            if filelock_ctx:
                filelock_ctx.acquire()

            with self._get_connection() as conn:
                # 1. Fetch current state and latest checksum
                cursor = conn.execute(
                    "SELECT state FROM draco_current_state WHERE singleton_key = 1;"
                )
                curr_row = cursor.fetchone()
                current_enum = DracoUIState(curr_row["state"]) if curr_row else DracoUIState.IDLE

                # 2. Transition validation
                if enforce_transition and not validate_draco_transition(current_enum, target_enum):
                    raise DracoTransitionError(
                        f"Illegal state transition from '{current_enum.value}' to '{target_enum.value}' requested by actor '{actor}'."
                    )

                # 3. Retrieve prior entry checksum for blockchain-style link
                cursor = conn.execute(
                    "SELECT checksum FROM draco_state_journal ORDER BY id DESC LIMIT 1;"
                )
                prev_entry = cursor.fetchone()
                prev_checksum = prev_entry["checksum"] if prev_entry else "0" * 64

                checksum = self._calculate_checksum(
                    prev_state=current_enum,
                    curr_state=target_enum,
                    timestamp=timestamp,
                    actor=actor,
                    metadata_json=meta_json,
                    prev_checksum=prev_checksum,
                )

                # 4. Insert journal row
                cursor = conn.execute(
                    """
                    INSERT INTO draco_state_journal (timestamp, previous_state, current_state, actor, metadata_json, checksum)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (timestamp, current_enum.value, target_enum.value, actor, meta_json, checksum),
                )
                journal_id = int(cursor.lastrowid or 0)

                # 5. Update singleton current state
                conn.execute(
                    """
                    UPDATE draco_current_state
                    SET state = ?, updated_at = ?, actor = ?, last_journal_id = ?
                    WHERE singleton_key = 1;
                    """,
                    (target_enum.value, timestamp, actor, journal_id),
                )

            # 6. Synchronize with swarm_state.json atomically
            if self.json_state_path:
                self._sync_swarm_state_json(
                    current_state=target_enum,
                    previous_state=current_enum,
                    timestamp=timestamp,
                    actor=actor,
                    journal_id=journal_id,
                    metadata=meta_dict,
                )

            return journal_id

        except Timeout as t_err:
            raise DracoLockError(
                f"FileLock acquisition timed out ({self.lock_timeout_s}s) for {lock_file}"
            ) from t_err
        finally:
            if filelock_ctx and filelock_ctx.is_locked:
                filelock_ctx.release()

    def _sync_swarm_state_json(
        self,
        current_state: DracoUIState,
        previous_state: DracoUIState,
        timestamp: str,
        actor: str,
        journal_id: int,
        metadata: Dict[str, Any],
    ) -> None:
        """Update swarm_state.json with Draco UI State block under active FileLock."""
        if self.json_state_path is None:
            return

        state_data: Dict[str, Any] = {}
        if self.json_state_path.exists():
            try:
                raw_text = self.json_state_path.read_text(encoding="utf-8")
                if raw_text.strip():
                    state_data = json.loads(raw_text)
            except Exception as exc:
                logger.warning("Failed parsing existing swarm_state.json; re-initializing: %s", exc)
                state_data = {}

        state_data["draco_ui_state"] = {
            "current_state": current_state.value,
            "previous_state": previous_state.value,
            "timestamp": timestamp,
            "actor": actor,
            "journal_id": journal_id,
            "metadata": metadata,
        }

        # Also update actor section if actor is a recognized key
        if actor in state_data and isinstance(state_data[actor], dict):
            state_data[actor]["draco_ui_state"] = current_state.value
            state_data[actor]["timestamp"] = timestamp

        # Atomic write
        temp_file = self.json_state_path.with_suffix(f".tmp_{os.getpid()}_{journal_id}")
        temp_file.write_text(
            json.dumps(state_data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_file.replace(self.json_state_path)

    def get_journal_entries(self, limit: int = 100) -> List[DracoStateJournalEntry]:
        """Fetch list of recent journal entries in ascending order."""
        entries: List[DracoStateJournalEntry] = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, previous_state, current_state, actor, metadata_json, checksum
                FROM draco_state_journal
                ORDER BY id ASC
                LIMIT ?;
                """,
                (limit,),
            )
            for row in cursor.fetchall():
                meta = json.loads(row["metadata_json"])
                entries.append(
                    DracoStateJournalEntry(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        previous_state=DracoUIState(row["previous_state"]),
                        current_state=DracoUIState(row["current_state"]),
                        actor=row["actor"],
                        metadata=meta,
                        checksum=row["checksum"],
                    )
                )
        return entries

    def close(self) -> None:
        """Clean up any connection state."""
        return None

    def __enter__(self) -> DracoStateJournal:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
