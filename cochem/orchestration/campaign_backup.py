"""Private, verified SQLite campaign snapshots and explicitly fenced recovery.

This trusted local service requires ownership of the files and a separate opaque
recovery capability. It is not a remote authentication adapter. A restore never
resumes work: approvals are revoked, unfinished attempts cancelled, and original
process/resource observations retained for measured reconciliation.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import secrets
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from cochem.orchestration import campaign as campaign_module
from cochem.orchestration.campaign import (
    TERMINAL,
    Allocation,
    CampaignBudget,
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
)
from cochem_torq.domain import canonical_json, digest

FORMAT = "cochem-campaign-sqlite-backup-v1"
RESTORE_POLICY = "revoke-approvals-cancel-unfinished-preserve-accounting-v1"
TABLES = (
    "campaigns",
    "campaign_tasks",
    "campaign_attempts",
    "campaign_reservations",
    "campaign_events",
)
METADATA_SQL = (
    "CREATE TABLE campaign_recovery_metadata "
    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
)
Progress = Callable[[str], None]
METADATA_KEYS = {
    "schema_version",
    "adopted_at_unix",
    "pre_migration_manifest_sha256",
    "pre_migration_logical_sha256",
    "actor_sha256",
    "reason_sha256",
}


class CampaignBackupError(CampaignError):
    """A snapshot, capability, schema or destination failed verification."""


@dataclass(frozen=True)
class CreatedCampaignBackup:
    """Only ``receipt`` is public; retain the recovery capability privately."""

    receipt: dict[str, Any]
    recovery_authority: str = field(repr=False)


def _hash_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _explicit(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignBackupError(f"An explicit {label} is required.")
    return value


def _components(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise CampaignBackupError(
                "Backup and recovery paths cannot contain symlinks."
            )
    return path


def _private_file(path: Path) -> Path:
    path = _components(path)
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise CampaignBackupError("A required private file is absent.") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise CampaignBackupError(
            "Private files must be regular files owned by this user."
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise CampaignBackupError(
            "A campaign database or manifest must be private (0600)."
        )
    return path


def _parent(path: Path, *, create: bool = False) -> Path:
    path = _components(path)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise CampaignBackupError(
            "Publication requires an owned directory without other writers."
        )
    return path


def _exclusive_file(path: Path, contents: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(source: Path, destination: Path) -> None:
    if sys.platform != "linux":
        raise CampaignBackupError(
            "Atomic no-replace publication requires Linux renameat2."
        )
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        raise CampaignBackupError("Atomic no-replace renameat2 is unavailable.")
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise CampaignBackupError(f"Immutable publication failed: {os.strerror(code)}.")
    _fsync_directory(destination.parent)


@contextlib.contextmanager
def _connection(path: Path, *, readonly: bool) -> Iterator[sqlite3.Connection]:
    _private_file(path)
    uri = path.as_uri() + ("?mode=ro" if readonly else "?mode=rw")
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    if not readonly:
        connection.execute("PRAGMA synchronous=FULL")
    try:
        yield connection
    finally:
        connection.close()


def _schema(connection: sqlite3.Connection) -> list[list[Any]]:
    return [
        list(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    ]


def _expected_schema(version: int) -> list[list[Any]]:
    with tempfile.TemporaryDirectory(prefix="cochem-campaign-schema-") as temporary:
        store = CampaignCoordinator(Path(temporary) / "schema.sqlite")
        try:
            if version == 1:
                store.connection.execute(METADATA_SQL)
            return _schema(store.connection)
        finally:
            store.close()


def _inventory(connection: sqlite3.Connection) -> dict[str, Any]:
    integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
    if integrity != ["ok"] or list(connection.execute("PRAGMA foreign_key_check")):
        raise CampaignBackupError(
            "SQLite integrity or foreign-key verification failed."
        )
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1) or _schema(connection) != _expected_schema(version):
        raise CampaignBackupError(
            "Unknown coordinator schema/version; no migration is authorized."
        )
    if version == 1:
        metadata = dict(
            connection.execute("SELECT key,value FROM campaign_recovery_metadata")
        )
        if set(metadata) != METADATA_KEYS or metadata["schema_version"] != "1":
            raise CampaignBackupError("Unknown schema-adoption metadata.")
    tables = TABLES + (("campaign_recovery_metadata",) if version == 1 else ())
    inventory = {}
    for table in tables:
        columns = [
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        ]
        primary = [
            row[1]
            for row in connection.execute(f'PRAGMA table_info("{table}")')
            if row[5]
        ]
        order = ",".join(f'"{key}"' for key in primary)
        records = [
            list(row)
            for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}')
        ]
        inventory[table] = {
            "rows": len(records),
            "columns": columns,
            "sha256": digest(records),
        }
    # The AUTOINCREMENT high-water mark matters after append-only recovery events.
    sequences = [
        list(row)
        for row in connection.execute(
            "SELECT name,seq FROM sqlite_sequence ORDER BY name"
        )
    ]
    campaigns = {
        row["id"]: row for row in connection.execute("SELECT * FROM campaigns")
    }
    for row in campaigns.values():
        try:
            plan = json.loads(row["plan_json"])
            if (
                not isinstance(plan, dict)
                or not plan
                or digest(plan) != row["plan_sha256"]
            ):
                raise CampaignBackupError(
                    "A campaign plan does not match its immutable digest."
                )
            CampaignBudget.model_validate_json(row["budget_json"])
        except (ValueError, TypeError) as exc:
            raise CampaignBackupError(
                "A campaign contains malformed plan or budget JSON."
            ) from exc
    tasks = {
        row["id"]: row for row in connection.execute("SELECT * FROM campaign_tasks")
    }
    for row in tasks.values():
        try:
            dependencies = json.loads(row["dependencies_json"])
            identity = {
                "payload": json.loads(row["payload_json"]),
                "engine": row["engine"],
                "recipe": row["recipe"],
                "dependencies": dependencies,
                "allow_partial_dependencies": bool(row["allow_partial_dependencies"]),
                "plan_sha256": campaigns[row["campaign_id"]]["plan_sha256"],
            }
            if digest(identity) != row["task_key"]:
                raise CampaignBackupError("A task differs from its immutable identity.")
            if (
                not isinstance(dependencies, list)
                or len(set(dependencies)) != len(dependencies)
                or any(
                    parent not in tasks
                    or tasks[parent]["campaign_id"] != row["campaign_id"]
                    for parent in dependencies
                )
            ):
                raise CampaignBackupError("A task dependency is absent or conflicts.")
        except (ValueError, TypeError, KeyError) as exc:
            raise CampaignBackupError(
                "Malformed task identity or dependencies."
            ) from exc
    if connection.execute(
        "SELECT count(*) FROM campaign_attempts AS child "
        "JOIN campaign_attempts AS parent ON child.parent_attempt_id=parent.id "
        "WHERE child.task_id!=parent.task_id OR child.number!=parent.number+1"
    ).fetchone()[0]:
        raise CampaignBackupError("An attempt lineage conflicts with its parent.")
    for row in connection.execute("SELECT * FROM campaign_reservations"):
        try:
            Allocation.model_validate_json(row["allocation_json"])
            if row["usage_json"] is not None:
                MeasuredUsage.model_validate_json(row["usage_json"])
            attempt = connection.execute(
                "SELECT task_id FROM campaign_attempts WHERE id=?",
                (row["attempt_id"],),
            ).fetchone()
            if (
                row["state"] not in {"reserved", "settled"}
                or tasks[attempt["task_id"]]["campaign_id"] != row["campaign_id"]
                or (row["state"] == "settled" and row["usage_json"] is None)
            ):
                raise CampaignBackupError("Reservation accounting lacks valid lineage.")
        except (ValueError, TypeError, KeyError) as exc:
            raise CampaignBackupError("Malformed reservation accounting.") from exc
    if connection.execute(
        "SELECT count(*) FROM campaign_events AS event "
        "LEFT JOIN campaigns ON event.campaign_id=campaigns.id "
        "LEFT JOIN campaign_attempts AS attempt ON event.attempt_id=attempt.id "
        "LEFT JOIN campaign_tasks AS task ON attempt.task_id=task.id "
        "WHERE campaigns.id IS NULL OR (event.attempt_id IS NOT NULL AND "
        "(attempt.id IS NULL OR task.campaign_id!=event.campaign_id))"
    ).fetchone()[0]:
        raise CampaignBackupError(
            "An append-only event lacks matching campaign lineage."
        )
    for row in connection.execute("SELECT state FROM campaign_attempts"):
        if row["state"] not in campaign_module.STATES:
            raise CampaignBackupError("A campaign contains an unknown lifecycle state.")
    result = {
        "user_version": version,
        "schema_sha256": digest(_schema(connection)),
        "tables": inventory,
        "sequence_sha256": digest(sequences),
        "integrity_check": "ok",
        "foreign_key_violations": 0,
    }
    return {**result, "logical_sha256": digest(result)}


def inspect_campaign_database(path: str | Path) -> dict[str, Any]:
    """Return a consistent, bounded inventory without actors, tokens or plan data."""
    with _connection(_components(Path(path)), readonly=True) as connection:
        connection.execute("BEGIN")
        try:
            return _inventory(connection)
        finally:
            connection.execute("ROLLBACK")


def create_campaign_backup(
    source: str | Path,
    destination: str | Path,
    *,
    source_session: str,
    actor: str,
    reason: str,
    progress: Progress | None = None,
) -> CreatedCampaignBackup:
    """Online-backup committed WAL state and publish a new private immutable bundle.

    The manifest and snapshot are 0600; the directory is 0700. The returned
    recovery capability must be stored outside public receipts and source control.
    ``source_session`` is a caller-supplied trusted session identity, hashed in the
    manifest. It is provenance, not authentication or a scientific observation.
    """
    source = _private_file(Path(source))
    destination = _components(Path(destination))
    parent = _parent(destination.parent, create=True)
    if destination.exists():
        raise CampaignBackupError("An immutable backup destination already exists.")
    for value, label in (
        (source_session, "source session"),
        (actor, "actor"),
        (reason, "reason"),
    ):
        _explicit(value, label)
    stage = Path(tempfile.mkdtemp(prefix=".campaign-backup-", dir=parent))
    capability = secrets.token_urlsafe(32)
    try:
        snapshot = stage / "campaign.sqlite"
        _exclusive_file(snapshot, b"")
        before = source.stat()
        with (
            _connection(source, readonly=True) as original,
            _connection(snapshot, readonly=False) as copied,
        ):
            original.backup(copied, pages=256)
            copied.execute("PRAGMA journal_mode=DELETE")
            inventory = _inventory(copied)
        after = source.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise CampaignBackupError(
                "The source database identity changed during backup."
            )
        with snapshot.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "format": FORMAT,
            "backup_id": str(uuid4()),
            "created_at_unix": time.time(),
            "snapshot_sha256": _hash_file(snapshot),
            "inventory": inventory,
            "source_session_sha256": _sha(source_session),
            "source_location_sha256": _sha(str(source)),
            "source_file_identity_sha256": digest([before.st_dev, before.st_ino]),
            "implementation_sha256": _hash_file(Path(campaign_module.__file__)),
            "actor_sha256": _sha(actor),
            "reason_sha256": _sha(reason),
            "recovery_authority_sha256": _sha(capability),
            "restore_policy": RESTORE_POLICY,
        }
        _exclusive_file(stage / "manifest.json", canonical_json(manifest))
        _fsync_directory(stage)
        if progress:
            progress("backup_verified_before_publication")
        _publish(stage, destination)
        receipt = {
            "format": FORMAT,
            "backup_id": manifest["backup_id"],
            "manifest_sha256": _hash_file(destination / "manifest.json"),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "inventory": inventory,
            "source_session_sha256": manifest["source_session_sha256"],
            "scientific_results_invented": False,
            "active_authority_restored": False,
        }
        return CreatedCampaignBackup(receipt, capability)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def verify_campaign_backup(
    bundle: str | Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Verify a trusted pinned manifest, snapshot bytes and exact logical inventory."""
    bundle = _components(Path(bundle))
    info = bundle.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise CampaignBackupError("A backup bundle must be owned and private (0700).")
    if {entry.name for entry in bundle.iterdir()} != {
        "manifest.json",
        "campaign.sqlite",
    }:
        raise CampaignBackupError(
            "A backup bundle contains unexpected or missing files."
        )
    manifest_path = _private_file(bundle / "manifest.json")
    if not secrets.compare_digest(_hash_file(manifest_path), expected_manifest_sha256):
        raise CampaignBackupError("The backup manifest differs from its trusted pin.")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (ValueError, UnicodeError) as exc:
        raise CampaignBackupError("Malformed backup manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise CampaignBackupError("Unknown backup format.")
    required = {
        "format",
        "backup_id",
        "created_at_unix",
        "snapshot_sha256",
        "inventory",
        "source_session_sha256",
        "source_location_sha256",
        "source_file_identity_sha256",
        "implementation_sha256",
        "actor_sha256",
        "reason_sha256",
        "recovery_authority_sha256",
        "restore_policy",
    }
    hashes = [key for key in required if key.endswith("_sha256")]
    if (
        set(manifest) != required
        or any(
            not isinstance(manifest.get(key), str)
            or len(manifest[key]) != 64
            or any(character not in "0123456789abcdef" for character in manifest[key])
            for key in hashes
        )
        or not isinstance(manifest.get("backup_id"), str)
        or not isinstance(manifest.get("created_at_unix"), (int, float))
        or not math.isfinite(manifest["created_at_unix"])
        or manifest["created_at_unix"] <= 0
    ):
        raise CampaignBackupError("Malformed backup manifest fields.")
    if canonical_json(manifest) != manifest_bytes:
        raise CampaignBackupError("A backup manifest must use canonical JSON.")
    snapshot = _private_file(bundle / "campaign.sqlite")
    if _hash_file(snapshot) != manifest.get("snapshot_sha256"):
        raise CampaignBackupError("Backup snapshot bytes changed.")
    try:
        inventory = inspect_campaign_database(snapshot)
    except sqlite3.DatabaseError as exc:
        raise CampaignBackupError(
            "Backup snapshot is not a valid coordinator database."
        ) from exc
    if inventory != manifest.get("inventory"):
        raise CampaignBackupError(
            "Backup logical inventory does not match its manifest."
        )
    if manifest.get("restore_policy") != RESTORE_POLICY:
        raise CampaignBackupError("Unknown recovery authority policy.")
    return manifest


def _authorize(manifest: dict[str, Any], capability: str, source_session: str) -> None:
    if not isinstance(capability, str) or not secrets.compare_digest(
        _sha(capability), manifest.get("recovery_authority_sha256", "")
    ):
        raise CampaignBackupError("A separate valid recovery capability is required.")
    if not isinstance(source_session, str) or not secrets.compare_digest(
        _sha(source_session), manifest.get("source_session_sha256", "")
    ):
        raise CampaignBackupError(
            "The trusted source session does not match this snapshot."
        )


def restore_campaign_backup(
    bundle: str | Path,
    target: str | Path,
    *,
    expected_manifest_sha256: str,
    recovery_authority: str,
    source_session: str,
    actor: str,
    reason: str,
    expected_source_logical_sha256: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Restore into a new target with every prior approval and worker lease fenced.

    No resource reservation is released and no worker is presumed stopped. Old
    operator capabilities are retained solely for existing measured reconciliation
    operations. New work requires an independently created/approved campaign.
    ``expected_source_logical_sha256`` rejects a stale snapshot when the caller
    requires a particular source revision. Never replace a live coordinator.
    """
    bundle = _components(Path(bundle))
    manifest = verify_campaign_backup(
        bundle, expected_manifest_sha256=expected_manifest_sha256
    )
    _authorize(manifest, recovery_authority, source_session)
    _explicit(actor, "recovery actor")
    _explicit(reason, "recovery reason")
    if (
        expected_source_logical_sha256 is not None
        and expected_source_logical_sha256 != manifest["inventory"]["logical_sha256"]
    ):
        raise CampaignBackupError(
            "The snapshot is stale for the required source revision."
        )
    target = _components(Path(target))
    parent = _parent(target.parent, create=True)
    if target.exists() or any(
        Path(str(target) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")
    ):
        raise CampaignBackupError(
            "Recovery requires an absent target and absent sidecars."
        )
    stage = Path(tempfile.mkdtemp(prefix=".campaign-restore-", dir=parent))
    staged = stage / "campaign.sqlite"
    store = None
    try:
        # Copy exactly the verified immutable snapshot, then verify the copy again.
        _exclusive_file(staged, (bundle / "campaign.sqlite").read_bytes())
        if _hash_file(staged) != manifest["snapshot_sha256"]:
            raise CampaignBackupError("The snapshot changed while staging recovery.")
        if progress:
            progress("restore_staged_before_fencing")
        store = CampaignCoordinator(staged)
        with store._transaction():
            if _inventory(store.connection) != manifest["inventory"]:
                raise CampaignBackupError("The staged database inventory changed.")
            campaigns = store.connection.execute(
                "SELECT * FROM campaigns ORDER BY id"
            ).fetchall()
            cancelled = 0
            for campaign in campaigns:
                store.connection.execute(
                    "UPDATE campaigns SET revoked=1,revision=revision+1 WHERE id=?",
                    (campaign["id"],),
                )
                store._event(
                    campaign["id"],
                    None,
                    campaign["revision"] + 1,
                    actor,
                    "offline_recovery_revocation",
                    reason,
                    {
                        "backup_id": manifest["backup_id"],
                        "future_work_authorized": False,
                        "worker_liveness_observed": False,
                        "accounting_released": False,
                    },
                )
            attempts = store.connection.execute(
                "SELECT * FROM campaign_attempts ORDER BY id"
            ).fetchall()
            for attempt in attempts:
                if attempt["state"] in TERMINAL:
                    continue
                task = store.connection.execute(
                    "SELECT * FROM campaign_tasks WHERE id=?", (attempt["task_id"],)
                ).fetchone()
                store._transition(attempt, task, "cancelled", actor, reason)
                cancelled += 1
            store.connection.execute(
                "UPDATE campaign_tasks SET next_generation=next_generation+1"
            )
            restored_inventory = _inventory(store.connection)
            if store.connection.execute(
                "SELECT count(*) FROM campaigns WHERE revoked=0"
            ).fetchone()[0]:
                raise CampaignBackupError("Recovery left an active campaign approval.")
            if store.connection.execute(
                "SELECT count(*) FROM campaign_attempts WHERE state NOT IN "
                "('succeeded','partial','failed','cancelled') "
                "OR lease_hash IS NOT NULL"
            ).fetchone()[0]:
                raise CampaignBackupError(
                    "Recovery left an unfinished attempt or worker capability."
                )
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        store.connection.execute("PRAGMA journal_mode=DELETE")
        store.close()
        store = None
        if inspect_campaign_database(staged) != restored_inventory:
            raise CampaignBackupError("Final restored inventory changed after commit.")
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        if progress:
            progress("restore_fenced_before_publication")
        _publish(staged, target)
        return {
            "format": "cochem-campaign-offline-restore-v1",
            "backup_id": manifest["backup_id"],
            "source_inventory": manifest["inventory"],
            "restored_inventory": restored_inventory,
            "restored_database_sha256": _hash_file(target),
            "cancelled_unfinished_attempts": cancelled,
            "campaign_approvals_revoked": len(campaigns),
            "worker_liveness_observed": False,
            "accounting_released": False,
            "scientific_results_invented": False,
            "fresh_campaign_approval_required": True,
        }
    finally:
        if store is not None:
            store.close()
        if stage.exists():
            shutil.rmtree(stage)


def adopt_campaign_schema_version(
    source: str | Path,
    bundle: str | Path,
    *,
    expected_manifest_sha256: str,
    recovery_authority: str,
    source_session: str,
    actor: str,
    reason: str,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Atomically introduce v1 metadata to the exact known unversioned schema.

    This is a single explicit migration, not a generic migration engine. A
    verified pre-migration backup and unchanged logical source revision are
    mandatory. Rollback uses ``restore_campaign_backup`` to a new fenced target.
    """
    source = _private_file(Path(source))
    manifest = verify_campaign_backup(
        bundle, expected_manifest_sha256=expected_manifest_sha256
    )
    _authorize(manifest, recovery_authority, source_session)
    _explicit(actor, "migration actor")
    _explicit(reason, "migration reason")
    if _sha(str(source)) != manifest["source_location_sha256"]:
        raise CampaignBackupError(
            "The migration backup belongs to another source location."
        )
    source_info = source.stat()
    if (
        digest([source_info.st_dev, source_info.st_ino])
        != manifest["source_file_identity_sha256"]
    ):
        raise CampaignBackupError("The migration source file identity changed.")
    with _connection(source, readonly=False) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            before = _inventory(connection)
            if before != manifest["inventory"]:
                raise CampaignBackupError(
                    "The live source changed after its pre-migration backup."
                )
            if before["user_version"] != 0:
                raise CampaignBackupError(
                    "Only the known unversioned-to-v1 adoption is supported."
                )
            connection.execute(METADATA_SQL)
            values = {
                "schema_version": "1",
                "adopted_at_unix": str(time.time()),
                "pre_migration_manifest_sha256": expected_manifest_sha256,
                "pre_migration_logical_sha256": before["logical_sha256"],
                "actor_sha256": _sha(actor),
                "reason_sha256": _sha(reason),
            }
            connection.executemany(
                "INSERT INTO campaign_recovery_metadata(key,value) VALUES(?,?)",
                values.items(),
            )
            connection.execute("PRAGMA user_version=1")
            after = _inventory(connection)
            if progress:
                progress("schema_adoption_before_commit")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    return {
        "format": "cochem-campaign-schema-adoption-v1",
        "from_user_version": 0,
        "to_user_version": 1,
        "pre_migration_manifest_sha256": expected_manifest_sha256,
        "before_inventory": before,
        "after_inventory": after,
        "campaign_authority_changed": False,
        "scientific_results_invented": False,
        "rollback_requires_new_fenced_target": True,
    }
