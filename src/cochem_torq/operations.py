"""Local artifact admission, verified backup/restore and explicit schema rollback.

These are operational filesystem controls, not scientific qualification. Quota
reservations coordinate local processes; callers must also monitor actual output
growth. They are not a replacement for a filesystem-enforced project quota.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from .artifacts import file_digest, publish_shard, verify_shard
from .domain import canonical_json, read_json


class ArtifactBackpressureError(RuntimeError):
    """The actual local artifact store cannot admit the requested work."""


class ArtifactObservationError(ArtifactBackpressureError):
    """Mutable scratch could not produce a complete bounded observation."""

    def __init__(self, scans: list[dict[str, int]]) -> None:
        super().__init__(
            "Mutable scratch inventory did not stabilize within three scans; "
            "unobserved file sizes and exact peak usage remain unknown."
        )
        self.observed_scans = tuple(dict(scan) for scan in scans)


@dataclass(frozen=True)
class ArtifactQuotaPolicy:
    max_owned_bytes: int = 1024 * 1024 * 1024
    minimum_free_bytes: int = 256 * 1024 * 1024
    max_files: int = 100_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < (
                1 if name != "minimum_free_bytes" else 0
            ):
                raise ValueError(
                    f"{name} must be an explicit nonnegative byte/count integer."
                )


def _root(path: str | Path, *, create: bool = False) -> Path:
    target = Path(path).absolute()
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError("Operational artifact paths cannot traverse symlinks.")
    if create:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.is_dir():
        raise NotADirectoryError(target)
    return target


def artifact_usage(
    root: str | Path,
    *,
    max_files: int = 100_000,
    mutable_scratch: bool = False,
    max_owned_bytes: int | None = None,
) -> dict[str, Any]:
    """Observe regular-file bytes without claiming an atomic snapshot or peak.

    Static inventories stay strict. Live scratch may lose a checkpoint between
    directory enumeration and stat; only that explicit disappearance triggers
    up to three rescans. Permission errors, links and special files still reject.
    A persistently incomplete inventory fails closed instead of assigning zero
    bytes to files that were never measured.
    """
    target = _root(root)
    if type(max_files) is not int or max_files < 1:
        raise ValueError("max_files must be a positive integer.")
    if type(mutable_scratch) is not bool:
        raise ValueError("mutable_scratch must be an explicit boolean.")
    if max_owned_bytes is not None and (
        type(max_owned_bytes) is not int or max_owned_bytes < 0
    ):
        raise ValueError("max_owned_bytes must be a nonnegative integer or None.")
    scans: list[dict[str, int]] = []
    for attempt in range(3 if mutable_scratch else 1):
        size = count = disappeared = 0

        def directory_error(error: OSError) -> None:
            nonlocal disappeared
            if mutable_scratch and isinstance(error, FileNotFoundError):
                disappeared += 1
                return
            raise error

        # Directory descriptors anchor each stat to the directory actually
        # opened by fwalk, including when path components are renamed.
        for _, directories, files, descriptor in os.fwalk(
            target, follow_symlinks=False, onerror=directory_error
        ):
            for name in directories + files:
                try:
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    if not mutable_scratch:
                        raise
                    disappeared += 1
                    continue
                if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
                    raise ValueError(
                        "Artifact accounting forbids symlinks and special files."
                    )
                if stat.S_ISREG(info.st_mode):
                    size += info.st_size
                    count += 1
                    if count > max_files:
                        raise ArtifactBackpressureError(
                            "Artifact file-count quota is exhausted."
                        )
                    if max_owned_bytes is not None and size > max_owned_bytes:
                        raise ArtifactBackpressureError(
                            "The worker exceeded its actual artifact output budget."
                        )
        scans.append(
            {
                "owned_bytes": size,
                "files": count,
                "disappeared_entries": disappeared,
            }
        )
        if not disappeared:
            return {
                "owned_bytes": size,
                "files": count,
                "free_bytes": shutil.disk_usage(target).free,
                "measurement_kind": "non_atomic_directory_observation",
                "exact_peak_usage_available": False,
                "scan_attempts": attempt + 1,
                "disappeared_entries_observed": sum(
                    scan["disappeared_entries"] for scan in scans
                ),
                "maximum_observed_scan_bytes": max(
                    scan["owned_bytes"] for scan in scans
                ),
            }
    raise ArtifactObservationError(scans)


@contextmanager
def _quota_lock(target: Path) -> Iterator[None]:
    lock = target / ".torq-artifact-quota.lock"
    if lock.is_symlink():
        raise ValueError("Quota locks cannot be symlinks.")
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _reservations(target: Path) -> list[dict[str, Any]]:
    directory = target / ".torq-reservations"
    if not directory.exists():
        return []
    _root(directory)
    records = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Malformed artifact reservation store.")
        record = read_json(path)
        if (
            record.get("schema_version") != "cochem.torq.artifact-reservation/1"
            or type(record.get("reserved_bytes")) is not int
            or record["reserved_bytes"] < 0
            or type(record.get("pid")) is not int
            or type(record.get("process_created_at")) not in (int, float)
            or not isinstance(record.get("host"), str)
        ):
            raise ValueError("Unsupported artifact reservation record.")
        records.append(record)
    return records


def _assess(
    target: Path, policy: ArtifactQuotaPolicy, incoming_bytes: int
) -> dict[str, Any]:
    if type(incoming_bytes) is not int or incoming_bytes < 0:
        raise ValueError("incoming_bytes must be an explicit nonnegative integer.")
    usage = artifact_usage(target, max_files=policy.max_files)
    reserved = sum(record["reserved_bytes"] for record in _reservations(target))
    report = {
        "schema_version": "cochem.torq.artifact-quota/1",
        "root": str(target),
        **usage,
        "active_reserved_bytes": reserved,
        "incoming_bytes": incoming_bytes,
        "policy": asdict(policy),
        "scientific_qualification": False,
    }
    if usage["owned_bytes"] + reserved + incoming_bytes > policy.max_owned_bytes:
        raise ArtifactBackpressureError(
            "Artifact byte quota is exhausted; export retained results "
            "before submitting more work."
        )
    if usage["free_bytes"] - reserved - incoming_bytes < policy.minimum_free_bytes:
        raise ArtifactBackpressureError(
            "Actual filesystem free space is below the required reserve."
        )
    return report


def check_artifact_quota(
    root: str | Path,
    *,
    policy: ArtifactQuotaPolicy = ArtifactQuotaPolicy(),
    incoming_bytes: int = 0,
) -> dict[str, Any]:
    target = _root(root, create=True)
    with _quota_lock(target):
        return _assess(target, policy, incoming_bytes)


@contextmanager
def admit_artifact_work(
    root: str | Path,
    *,
    incoming_bytes: int,
    policy: ArtifactQuotaPolicy = ArtifactQuotaPolicy(),
) -> Iterator[dict[str, Any]]:
    """Reserve a bounded output budget under a real cross-process file lock.

    A killed owner leaves its reservation in place and causes backpressure until
    explicit reconciliation proves that the local process has ended.
    """
    if type(incoming_bytes) is not int or incoming_bytes < 0:
        raise ValueError("incoming_bytes must be an explicit nonnegative integer.")
    target = _root(root, create=True)
    reservation_path = None
    record = {
        "schema_version": "cochem.torq.artifact-reservation/1",
        "reserved_bytes": incoming_bytes,
        "pid": os.getpid(),
        "process_created_at": psutil.Process().create_time(),
        "host": socket.gethostname(),
    }
    raw = canonical_json(record) + b"\n"
    with _quota_lock(target):
        report = _assess(target, policy, incoming_bytes + len(raw))
        if report["files"] + 1 > policy.max_files:
            raise ArtifactBackpressureError("Artifact file-count quota is exhausted.")
        report["incoming_bytes"] = incoming_bytes
        report["reservation_metadata_bytes"] = len(raw)
        directory = target / ".torq-reservations"
        directory.mkdir(exist_ok=True, mode=0o700)
        reservation_path = directory / f"{uuid4()}.json"
        with reservation_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    try:
        yield report
    finally:
        with _quota_lock(target):
            reservation_path.unlink(missing_ok=True)


def reconcile_artifact_reservations(root: str | Path) -> dict[str, Any]:
    """Remove only reservations whose exact local process identity is gone."""
    target = _root(root)
    removed, retained = [], []
    with _quota_lock(target):
        _reservations(target)  # Validate the complete store before removing anything.
        directory = target / ".torq-reservations"
        for path in sorted(directory.iterdir()) if directory.exists() else []:
            record = read_json(path)
            if record["host"] != socket.gethostname():
                retained.append(path.name)
                continue
            try:
                alive = (
                    psutil.Process(record["pid"]).create_time()
                    == record["process_created_at"]
                )
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied:
                alive = True
            if alive:
                retained.append(path.name)
            else:
                path.unlink()
                removed.append(path.name)
    return {
        "removed_dead_local_reservations": removed,
        "retained_reservations": retained,
    }


def enforce_artifact_budget(
    root: str | Path, *, baseline_bytes: int, max_growth_bytes: int
) -> dict[str, Any]:
    """Check actual growth; the caller owns stopping its worker on rejection."""
    if any(
        type(value) is not int or value < 0
        for value in (baseline_bytes, max_growth_bytes)
    ):
        raise ValueError("Artifact growth limits must be nonnegative integers.")
    return artifact_usage(
        root,
        mutable_scratch=True,
        max_owned_bytes=baseline_bytes + max_growth_bytes,
    )


def _expected_digest(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Supply the independently retained manifest SHA-256.")


def _sync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    directories = [root, *(p for p in root.rglob("*") if p.is_dir())]
    for path in reversed(directories):
        descriptor = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_backup(
    backup: str | Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    _expected_digest(expected_manifest_sha256)
    root = _root(backup)
    if {p.name for p in root.iterdir()} != {"backup.json", "shard"}:
        raise ValueError("Backup inventory contains missing or unrecorded entries.")
    if (root / "backup.json").is_symlink():
        raise ValueError("Backup metadata cannot be a symlink.")
    receipt: dict[str, Any] = read_json(root / "backup.json")
    if (
        receipt.get("schema_version") != "cochem.torq.backup/1"
        or receipt.get("shard_schema_version") != "cochem.torq.shard/1"
    ):
        raise ValueError(
            "Unsupported backup/shard schema; explicit versioned migration is required."
        )
    if (
        receipt.get("source_manifest_sha256") != expected_manifest_sha256
        or file_digest(root / "shard/manifest.json") != expected_manifest_sha256
    ):
        raise ValueError(
            "Backup manifest differs from the independently retained digest."
        )
    manifest = verify_shard(root / "shard")
    if (
        receipt.get("request_id") != manifest["request_id"]
        or receipt.get("file_count") != len(manifest["files"]) + 1
    ):
        raise ValueError("Backup receipt and sealed shard identities differ.")
    return receipt


def backup_shard(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy a genuine sealed shard to a fresh immutable backup directory."""
    source = _root(source)
    manifest = verify_shard(source)
    expected = file_digest(source / "manifest.json")
    target = Path(destination).absolute()
    _root(target.parent, create=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Backups are immutable; choose a new destination.")
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Backup and source directories cannot contain each other.")
    staging = Path(tempfile.mkdtemp(prefix=".torq-backup-", dir=target.parent))
    try:
        shutil.copytree(source, staging / "shard")
        receipt = {
            "schema_version": "cochem.torq.backup/1",
            "shard_schema_version": "cochem.torq.shard/1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": expected,
            "request_id": manifest["request_id"],
            "file_count": len(manifest["files"]) + 1,
            "scientific_qualification": False,
        }
        with (staging / "backup.json").open("xb") as stream:
            stream.write(canonical_json(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_backup(staging, expected_manifest_sha256=expected)
        _sync_tree(staging)
        lock = target.parent / f".{target.name}.backup.lock"
        if lock.is_symlink():
            raise ValueError("Backup publication locks cannot be symlinks.")
        with lock.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    "Backups are immutable; choose a new destination."
                )
            os.rename(staging, target)
            descriptor = os.open(target.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return {**receipt, "backup_directory": str(target)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_shard(
    backup: str | Path,
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
) -> Path:
    """Restore verified bytes under a new name, without replacing any result."""
    verify_backup(backup, expected_manifest_sha256=expected_manifest_sha256)
    target = Path(destination).absolute()
    _root(target.parent, create=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Restore cannot overwrite an existing artifact.")
    source = _root(backup) / "shard"
    if target.is_relative_to(_root(backup)) or source.is_relative_to(target):
        raise ValueError("Restore destination cannot overlap the backup.")
    staging = Path(tempfile.mkdtemp(prefix=".torq-restore-", dir=target.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        if file_digest(staging / "manifest.json") != expected_manifest_sha256:
            raise ValueError("Backup changed while it was being restored.")
        verify_shard(staging)
        _sync_tree(staging)
        return publish_shard(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def rollback_shard(
    previous_backup: str | Path,
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
) -> Path:
    """Restore a prior supported-schema snapshot, never rewrite its schema."""
    return restore_shard(
        previous_backup, destination, expected_manifest_sha256=expected_manifest_sha256
    )
