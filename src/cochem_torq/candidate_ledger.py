"""Reversible candidate selections over immutable supplied scientific evidence.

This local ledger neither discovers conformers nor establishes search recall,
scientific accuracy, or an authenticated signature. Input records have no computed
properties. Result references require an actually verified TORQ artifact shard.
Selections never mutate requests, result files, or completed calculations.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import stat
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal
from uuid import UUID, uuid4

from .domain import CalculationRequest, Molecule, canonical_json, digest, read_json

SelectionState = Literal["retained", "excluded", "quarantined"]
DecisionMode = Literal["manual", "blind_prediction", "assignment_assisted"]


class CandidateLedgerError(RuntimeError):
    """The local ledger violates its declared integrity or operation contract."""


class CandidateRevisionConflictError(CandidateLedgerError):
    """A selection changed after the caller inspected its revision."""


def _text(value: str, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{label} must be nonempty text of at most {maximum} characters."
        )
    return value.strip()


def _uuid(value: str | UUID) -> str:
    if not isinstance(value, str | UUID):
        raise ValueError("Candidate identities must be UUIDs.")
    return str(UUID(str(value)))


def _content(definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "molecule": definition["molecule"],
        "recipe": definition["recipe"],
        "origin": definition["origin"],
        "geometry_status": definition["geometry_status"],
        "native_artifact_hashes": definition["native_artifact_hashes"],
        "result_manifest_sha256": (
            definition["result_reference"]["manifest_sha256"]
            if definition["result_reference"]
            else None
        ),
    }


class CandidateLedger:
    """Append-only SQLite definitions and exact-revision selection events.

    Files use private permissions, FULL synchronous transactions and WAL. SQL
    triggers reject ordinary updates/deletes; a global hash chain detects changed
    bytes unless an administrator replaces both evidence and chain. Caller actor
    labels are explicit provenance, not a substitute for remote authentication.
    """

    def __init__(self, path: str | Path) -> None:
        self._closed = False
        self.path = Path(path).expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        created = False
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            os.close(descriptor)
            created = True
        except FileExistsError:
            self._check_file()
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if created or version == 0:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if tables:
                    raise CandidateLedgerError(
                        "Existing file is not a candidate ledger."
                    )
                connection.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE candidate_definitions (
                        candidate_id TEXT PRIMARY KEY,
                        definition_json TEXT NOT NULL,
                        definition_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE candidate_events (
                        sequence INTEGER PRIMARY KEY,
                        candidate_id TEXT NOT NULL REFERENCES candidate_definitions,
                        revision INTEGER NOT NULL CHECK(revision > 0),
                        event_json TEXT NOT NULL,
                        event_sha256 TEXT NOT NULL,
                        UNIQUE(candidate_id, revision)
                    );
                    CREATE TRIGGER definitions_no_update
                    BEFORE UPDATE ON candidate_definitions BEGIN
                        SELECT RAISE(ABORT, 'candidate definitions are immutable');
                    END;
                    CREATE TRIGGER definitions_no_delete
                    BEFORE DELETE ON candidate_definitions BEGIN
                        SELECT RAISE(ABORT, 'candidate definitions cannot be deleted');
                    END;
                    CREATE TRIGGER events_no_update
                    BEFORE UPDATE ON candidate_events BEGIN
                        SELECT RAISE(ABORT, 'candidate events are immutable');
                    END;
                    CREATE TRIGGER events_no_delete
                    BEFORE DELETE ON candidate_events BEGIN
                        SELECT RAISE(ABORT, 'candidate events cannot be deleted');
                    END;
                    PRAGMA user_version=1;
                    COMMIT;
                """)
            elif version != 1:
                raise CandidateLedgerError("Unsupported candidate ledger schema.")
            connection.execute("BEGIN")
            self._verify(connection)

    def __enter__(self) -> CandidateLedger:
        self.verify_integrity()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def close(self) -> None:
        """Invalidate this handle; each transaction already closes its connection."""
        self._closed = True

    def _check_file(self) -> None:
        metadata = self.path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise CandidateLedgerError(
                "Candidate ledger must be a private regular file."
            )

    @contextlib.contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise CandidateLedgerError("Candidate ledger handle is closed.")
        self._check_file()
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _decode(text: str) -> dict[str, Any]:
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value = {}
            for key, item in pairs:
                if key in value:
                    raise CandidateLedgerError("Duplicate ledger JSON key.")
                value[key] = item
            return value

        value = json.loads(text, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise CandidateLedgerError("Ledger records must be JSON objects.")
        canonical_json(value)
        return value

    def _verify(self, connection: sqlite3.Connection) -> None:
        definitions = {}
        for row in connection.execute("SELECT * FROM candidate_definitions"):
            definition = self._decode(row["definition_json"])
            if (
                digest(definition) != row["definition_sha256"]
                or definition["candidate_id"] != row["candidate_id"]
                or _uuid(definition["candidate_id"]) != definition["candidate_id"]
                or definition["schema_version"] != "cochem.torq.candidate/1"
                or digest(_content(definition)) != definition["content_sha256"]
            ):
                raise CandidateLedgerError("Candidate definition integrity failed.")
            CalculationRequest.model_validate(definition["request"])
            Molecule.model_validate(definition["molecule"])
            if definition["request"]["recipe"] != definition["recipe"]["id"]:
                raise CandidateLedgerError(
                    "Candidate recipe identity differs from request."
                )
            definitions[row["candidate_id"]] = (definition, row["definition_sha256"])
        previous_hash = None
        revisions: dict[str, int] = {}
        states: dict[str, str] = {}
        last_candidate_events: dict[str, dict[str, Any]] = {}
        for sequence, row in enumerate(
            connection.execute("SELECT * FROM candidate_events ORDER BY sequence"), 1
        ):
            event = self._decode(row["event_json"])
            candidate_id = row["candidate_id"]
            if candidate_id not in definitions:
                raise CandidateLedgerError(
                    "An event has no immutable candidate definition."
                )
            definition, definition_hash = definitions[candidate_id]
            revision = revisions.get(candidate_id, 0) + 1
            if (
                row["sequence"] != sequence
                or event["sequence"] != sequence
                or digest(event) != row["event_sha256"]
                or event["previous_event_sha256"] != previous_hash
                or event["candidate_id"] != candidate_id
                or row["revision"] != revision
                or event["revision"] != revision
                or event["definition_sha256"] != definition_hash
                or event["content_sha256"] != definition["content_sha256"]
                or event["origin"] != definition["origin"]
                or event["recipe_sha256"] != definition["recipe"]["recipe_sha256"]
                or event["source_identity"] != definition["source_identity"]
                or event["native_artifact_hashes"]
                != definition["native_artifact_hashes"]
                or event["previous_selection_state"] != states.get(candidate_id)
                or event["review_status"] != "needs_review"
            ):
                raise CandidateLedgerError(
                    "Candidate event chain/revision integrity failed."
                )
            _text(event["actor"], "Actor", 256)
            _text(event["reason"], "Reason", 4096)
            timestamp = datetime.fromisoformat(event["time_utc"])
            offset = timestamp.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                raise CandidateLedgerError(
                    "Candidate event requires an actual UTC timestamp."
                )
            _uuid(event["event_id"])
            action, state = event["action"], event["selection_state"]
            prior = states.get(candidate_id)
            valid = (
                (
                    action == "register"
                    and revision == 1
                    and state in {"retained", "quarantined"}
                )
                or (
                    action == "retain"
                    and prior in {"retained", "quarantined"}
                    and state == "retained"
                )
                or (
                    action == "exclude"
                    and prior in {"retained", "quarantined"}
                    and state == "excluded"
                )
                or (
                    action == "quarantine"
                    and prior in {"retained", "quarantined"}
                    and state == "quarantined"
                )
                or (
                    action == "restore"
                    and prior == "excluded"
                    and state in {"retained", "quarantined"}
                )
            )
            if not valid or event["decision_mode"] not in {
                "manual",
                "blind_prediction",
                "assignment_assisted",
            }:
                raise CandidateLedgerError("Candidate event lifecycle is invalid.")
            if (
                action == "restore"
                and state
                != last_candidate_events[candidate_id]["previous_selection_state"]
            ):
                raise CandidateLedgerError(
                    "Restore does not preserve the excluded state."
                )
            states[candidate_id] = state
            revisions[candidate_id] = revision
            previous_hash = row["event_sha256"]
            last_candidate_events[candidate_id] = event
        if set(revisions) != set(definitions):
            raise CandidateLedgerError(
                "Every candidate requires its registration event."
            )

    def verify_integrity(self) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._verify(connection)
            return {
                "schema_version": "cochem.torq.candidate-ledger-integrity/1",
                "status": "verified_local_byte_integrity",
                "authenticated_signature": False,
                "scientific_accuracy_established": False,
                "candidates": connection.execute(
                    "SELECT count(*) FROM candidate_definitions"
                ).fetchone()[0],
                "events": connection.execute(
                    "SELECT count(*) FROM candidate_events"
                ).fetchone()[0],
            }

    def _append(
        self,
        connection: sqlite3.Connection,
        definition: dict[str, Any],
        action: str,
        state: SelectionState,
        prior: SelectionState | None,
        revision: int,
        actor: str,
        reason: str,
        decision_mode: DecisionMode,
    ) -> dict[str, Any]:
        previous = connection.execute(
            "SELECT sequence, event_sha256 FROM candidate_events "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        event = {
            "schema_version": "cochem.torq.candidate-event/1",
            "sequence": previous["sequence"] + 1 if previous else 1,
            "event_id": str(uuid4()),
            "candidate_id": definition["candidate_id"],
            "revision": revision,
            "action": action,
            "selection_state": state,
            "previous_selection_state": prior,
            "actor": actor,
            "reason": reason,
            "decision_mode": decision_mode,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "definition_sha256": digest(definition),
            "content_sha256": definition["content_sha256"],
            "origin": definition["origin"],
            "recipe_sha256": definition["recipe"]["recipe_sha256"],
            "source_identity": definition["source_identity"],
            "native_artifact_hashes": definition["native_artifact_hashes"],
            "review_status": "needs_review",
            "previous_event_sha256": previous["event_sha256"] if previous else None,
        }
        connection.execute(
            "INSERT INTO candidate_events VALUES (?, ?, ?, ?, ?)",
            (
                event["sequence"],
                definition["candidate_id"],
                revision,
                canonical_json(event).decode(),
                digest(event),
            ),
        )
        return {
            **definition,
            "revision": revision,
            "selection_state": state,
            "review_status": "needs_review",
            "last_event": event,
        }

    def _register(
        self,
        definition: dict[str, Any],
        state: SelectionState,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        actor, reason = _text(actor, "Actor", 256), _text(reason, "Reason", 4096)
        definition["content_sha256"] = digest(_content(definition))
        with self._connection(write=True) as connection:
            self._verify(connection)
            if connection.execute(
                "SELECT 1 FROM candidate_definitions WHERE candidate_id=?",
                (definition["candidate_id"],),
            ).fetchone():
                raise CandidateRevisionConflictError("Candidate UUID already exists.")
            connection.execute(
                "INSERT INTO candidate_definitions VALUES (?, ?, ?)",
                (
                    definition["candidate_id"],
                    canonical_json(definition).decode(),
                    digest(definition),
                ),
            )
            return self._append(
                connection,
                definition,
                "register",
                state,
                None,
                1,
                actor,
                reason,
                "manual",
            )

    def register_request(
        self,
        request: dict[str, Any],
        *,
        actor: str,
        reason: str,
        candidate_id: str | UUID | None = None,
    ) -> dict[str, Any]:
        from .application import source_identity
        from .registry import get_profile

        model = CalculationRequest.model_validate(request)
        definition = {
            "schema_version": "cochem.torq.candidate/1",
            "candidate_id": _uuid(
                candidate_id if candidate_id is not None else uuid4()
            ),
            "request": model.model_dump(mode="json"),
            "molecule": model.molecule.model_dump(mode="json"),
            "origin": "supplied_input",
            "geometry_status": "input",
            "recipe": get_profile(model.recipe),
            "source_identity": source_identity(),
            "native_artifact_hashes": [],
            "result_reference": None,
            "quality": {
                "evidence": "input_only",
                "stage_statuses": {},
                "uncertainty": None,
                "uncertainty_status": "not_computed",
                "search_completeness": "not_established",
            },
        }
        return self._register(definition, "retained", actor=actor, reason=reason)

    def register_shard(
        self,
        directory: str | Path,
        *,
        actor: str,
        reason: str,
        candidate_id: str | UUID | None = None,
    ) -> dict[str, Any]:
        from .artifacts import file_digest, verify_shard

        root = Path(directory).resolve()
        manifest = verify_shard(root)
        request = CalculationRequest.model_validate(read_json(root / "request.json"))
        result = read_json(root / "result.json")
        geometry = result["stages"]["equilibrium_geometry"]
        optimized = geometry["status"] == "available"
        molecule = request.molecule.model_dump(mode="json")
        if optimized:
            molecule["geometry_bohr"] = geometry["value"]["geometry_bohr"]
            Molecule.model_validate(molecule)
        definition = {
            "schema_version": "cochem.torq.candidate/1",
            "candidate_id": _uuid(
                candidate_id if candidate_id is not None else uuid4()
            ),
            "request": request.model_dump(mode="json"),
            "molecule": molecule,
            "origin": "verified_result",
            "geometry_status": "optimized" if optimized else "input",
            "recipe": result["recipe"],
            "source_identity": result["source_identity"],
            "native_artifact_hashes": [
                item
                for item in manifest["files"]
                if item["path"].startswith(("engine/", "anharmonic-research/"))
            ],
            "result_reference": {
                "directory": str(root),
                "manifest_sha256": file_digest(root / "manifest.json"),
                "request_id": str(request.request_id),
                "inventory": manifest["files"],
            },
            "quality": {
                "evidence": "verified_result_bytes",
                "scientific_status": result["status"],
                "stage_statuses": {
                    name: {
                        key: stage.get(key)
                        for key in (
                            "status",
                            "reason",
                            "absence_kind",
                            "quality_flags",
                            "uncertainty",
                        )
                    }
                    for name, stage in result["stages"].items()
                },
                "experimental_accuracy_established": result[
                    "experimental_accuracy_established"
                ],
                "identification_ready": result["identification_ready"],
                "search_completeness": "not_established",
            },
        }
        return self._register(
            definition,
            "retained" if optimized else "quarantined",
            actor=actor,
            reason=reason,
        )

    def _change(
        self,
        candidate_id: str | UUID,
        action: Literal["retain", "exclude", "restore", "quarantine"],
        *,
        expected_revision: int,
        actor: str,
        reason: str,
        decision_mode: DecisionMode = "manual",
    ) -> dict[str, Any]:
        identity = _uuid(candidate_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("Expected revision must be a positive integer.")
        actor, reason = _text(actor, "Actor", 256), _text(reason, "Reason", 4096)
        if decision_mode not in {"manual", "blind_prediction", "assignment_assisted"}:
            raise ValueError("Unknown candidate selection decision mode.")
        with self._connection(write=True) as connection:
            self._verify(connection)
            row = connection.execute(
                "SELECT * FROM candidate_definitions WHERE candidate_id=?", (identity,)
            ).fetchone()
            if row is None:
                raise KeyError("Candidate UUID was not registered.")
            definition = self._decode(row["definition_json"])
            last = connection.execute(
                "SELECT * FROM candidate_events WHERE candidate_id=? "
                "ORDER BY revision DESC LIMIT 1",
                (identity,),
            ).fetchone()
            if last["revision"] != expected_revision:
                raise CandidateRevisionConflictError(
                    "Candidate revision changed; inspect anew."
                )
            previous = self._decode(last["event_json"])
            prior = previous["selection_state"]
            state: SelectionState
            if action == "restore":
                if prior != "excluded":
                    raise ValueError("Restore requires an excluded candidate.")
                state = previous["previous_selection_state"]
            elif prior == "excluded":
                raise ValueError("Restore an excluded candidate before other changes.")
            else:
                target_states: dict[str, SelectionState] = {
                    "retain": "retained",
                    "exclude": "excluded",
                    "quarantine": "quarantined",
                }
                state = target_states[action]
            return self._append(
                connection,
                definition,
                action,
                state,
                prior,
                expected_revision + 1,
                actor,
                reason,
                decision_mode,
            )

    def retain(
        self,
        candidate_id: str | UUID,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
        decision_mode: DecisionMode = "manual",
    ) -> dict[str, Any]:
        return self._change(
            candidate_id,
            "retain",
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
            decision_mode=decision_mode,
        )

    def exclude(
        self,
        candidate_id: str | UUID,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
        decision_mode: DecisionMode = "manual",
    ) -> dict[str, Any]:
        return self._change(
            candidate_id,
            "exclude",
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
            decision_mode=decision_mode,
        )

    def restore(
        self,
        candidate_id: str | UUID,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        return self._change(
            candidate_id,
            "restore",
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
        )

    def quarantine(
        self,
        candidate_id: str | UUID,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        return self._change(
            candidate_id,
            "quarantine",
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
        )

    def _records(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = connection.execute("""
            SELECT d.definition_json, e.revision, e.event_json
            FROM candidate_definitions d JOIN candidate_events e
            ON d.candidate_id=e.candidate_id
            WHERE e.revision=(SELECT max(revision) FROM candidate_events
                              WHERE candidate_id=d.candidate_id)
            ORDER BY d.candidate_id
        """).fetchall()
        return [
            {
                **self._decode(row["definition_json"]),
                "revision": row["revision"],
                "selection_state": self._decode(row["event_json"])["selection_state"],
                "review_status": "needs_review",
                "last_event": self._decode(row["event_json"]),
            }
            for row in rows
        ]

    def candidates(self, *, include_excluded: bool = True) -> list[dict[str, Any]]:
        if type(include_excluded) is not bool:
            raise ValueError("include_excluded must be a boolean.")
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._verify(connection)
            return [
                record
                for record in self._records(connection)
                if include_excluded or record["selection_state"] != "excluded"
            ]

    def inspect(self, candidate_id: str | UUID) -> dict[str, Any]:
        identity = _uuid(candidate_id)
        for record in self.candidates():
            if record["candidate_id"] == identity:
                return record
        raise KeyError("Candidate UUID was not registered.")

    def history(self, candidate_id: str | UUID) -> list[dict[str, Any]]:
        identity = _uuid(candidate_id)
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._verify(connection)
            if (
                connection.execute(
                    "SELECT 1 FROM candidate_definitions WHERE candidate_id=?",
                    (identity,),
                ).fetchone()
                is None
            ):
                raise KeyError("Candidate UUID was not registered.")
            return [
                self._decode(row["event_json"])
                for row in connection.execute(
                    "SELECT event_json FROM candidate_events WHERE candidate_id=? "
                    "ORDER BY revision",
                    (identity,),
                )
            ]

    def selection_snapshot(self) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._verify(connection)
            records = self._records(connection)
            last = connection.execute(
                "SELECT sequence, event_sha256 FROM candidate_events "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            snapshot = {
                "schema_version": "cochem.torq.candidate-selection/1",
                "revision": last["sequence"] if last else 0,
                "event_chain_sha256": last["event_sha256"] if last else None,
                "candidate_ids": [
                    item["candidate_id"]
                    for item in records
                    if item["selection_state"] == "retained"
                ],
                "excluded_candidate_ids": [
                    item["candidate_id"]
                    for item in records
                    if item["selection_state"] == "excluded"
                ],
                "quarantined_candidate_ids": [
                    item["candidate_id"]
                    for item in records
                    if item["selection_state"] == "quarantined"
                ],
                "candidates": [
                    {
                        key: item[key]
                        for key in (
                            "candidate_id",
                            "revision",
                            "content_sha256",
                            "selection_state",
                        )
                    }
                    for item in records
                ],
                "needs_review": True,
                "search_completeness": "not_established",
                "automatic_pruning_enabled": False,
            }
            return {**snapshot, "snapshot_sha256": digest(snapshot)}
