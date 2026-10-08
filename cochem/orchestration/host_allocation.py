"""One private local deployment ledger for host-wide resource reservations.

This is additional authority to a scientific campaign budget, never a replacement.
Only callers sharing the same database/operator capability share its limits.
Expired leases retain resources until explicit completion or actual process-death
reconciliation. Unbound dispatch after a crash remains unknown and reserved.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cochem_torq.domain import canonical_json, digest

MIB = 1024 * 1024
ACTIVE = ("reserved", "dispatching", "running")
SCHEMA = "cochem-host-allocation-v1"


class HostAllocationError(RuntimeError):
    """Host ownership, capacity, revision, lease or process evidence is invalid."""


class HostCapacityError(HostAllocationError):
    pass


class HostRevisionConflictError(HostAllocationError):
    pass


class HostAuthorityError(HostAllocationError):
    pass


HostRevisionConflict = HostRevisionConflictError


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class HostAllocationPolicy(_StrictModel):
    max_cpu_cores: StrictInt = Field(ge=1)
    max_memory_mb: StrictInt = Field(ge=1)
    max_scratch_mb: StrictInt = Field(ge=0)
    max_workers: StrictInt = Field(ge=1)
    gpu_ids: list[str] = Field(default_factory=list)
    scratch_probe_root: str = Field(min_length=1)


class HostAllocationRequest(_StrictModel):
    cores: StrictInt = Field(ge=1)
    memory_mb: StrictInt = Field(ge=1)
    scratch_mb: StrictInt = Field(ge=0)
    worker_slots: StrictInt = Field(default=1, ge=1)
    gpu_ids: list[str] = Field(default_factory=list)
    child_inventory_scope: Literal[
        "single_owned_worker", "constrained-evaluation-v1"
    ] = "single_owned_worker"


class ObservedHostUsage(_StrictModel):
    peak_memory_mb: float | None = Field(default=None, ge=0)
    peak_scratch_mb: float | None = Field(default=None, ge=0)
    measurement_source: str = Field(min_length=1)


@dataclass(frozen=True)
class AcquiredHostAllocation:
    receipt: dict[str, Any]
    lease_token: str = field(repr=False)


@dataclass(frozen=True)
class CreatedHostAllocationLedger:
    receipt: dict[str, Any]
    authority: str = field(repr=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _explicit(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise HostAllocationError(f"An explicit {label} is required.")


def _path(value: str | Path) -> Path:
    path = Path(os.path.abspath(value))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise HostAuthorityError("Host allocation paths cannot contain symlinks.")
    return path


def _private_directory(path: Path, *, create: bool = False) -> Path:
    path = _path(path)
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise HostAuthorityError(
            "A host ledger directory must be owned and private (0700)."
        )
    return path


def _private_file(path: Path) -> Path:
    path = _path(path)
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise HostAuthorityError("Host ledger files must be owned and private (0600).")
    return path


def _exclusive(path: Path, contents: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(source: Path, destination: Path) -> None:
    if sys.platform != "linux":
        raise HostAuthorityError("Atomic private ledger publication requires Linux.")
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        raise HostAuthorityError(
            "Atomic private ledger publication requires renameat2."
        )
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        raise HostAuthorityError(
            f"Immutable ledger publication failed: {os.strerror(ctypes.get_errno())}."
        )
    _sync_directory(destination.parent)


def _host_identity() -> dict[str, Any]:
    if sys.platform != "linux":
        raise HostAuthorityError(
            "Host/boot/namespace qualification currently requires Linux."
        )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    machine_path = Path("/etc/machine-id")
    machine = (
        machine_path.read_text().strip() if machine_path.is_file() else "unavailable"
    )
    identity = {
        "hostname_sha256": _sha(socket.gethostname()),
        "boot_id": boot,
        "machine_id_sha256": _sha(machine),
        "pid_namespace_inode": Path("/proc/self/ns/pid").stat().st_ino,
        "mount_namespace_inode": Path("/proc/self/ns/mnt").stat().st_ino,
        "uid": os.getuid(),
    }
    return {**identity, "host_sha256": digest(identity)}


def _cgroup_directories() -> list[Path]:
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    unified = next(
        (line.split(":", 2)[2] for line in lines if line.startswith("0::")), None
    )
    if unified is None:
        raise HostCapacityError(
            "Actual cgroup-v1 limits are not qualified; cgroup-v2 evidence is required."
        )
    # A container may expose a delegated cgroup as /sys/fs/cgroup even when
    # /proc/self/cgroup retains a path above that mount. Prefer the actual leaf.
    root = Path("/sys/fs/cgroup")
    relative = unified.lstrip("/")
    leaf = root
    candidate = root / relative
    if ".." not in Path(relative).parts and candidate.is_dir():
        leaf = candidate
    directories = [leaf]
    for parent in leaf.parents:
        if parent == root:
            directories.append(parent)
            break
        if root not in parent.parents:
            break
        directories.append(parent)
    return directories


def _gpu_probe() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "status": "unavailable",
            "gpu_ids": [],
            "reason": "nvidia-smi is not installed; no GPU capacity is asserted.",
        }
    try:
        process = subprocess.run(
            [executable, "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "status": "unavailable",
            "gpu_ids": [],
            "reason": "The genuine NVIDIA inventory probe did not complete.",
        }
    if process.returncode != 0:
        return {
            "status": "unavailable",
            "gpu_ids": [],
            "reason": "The genuine NVIDIA inventory probe failed.",
        }
    identifiers = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(not value.startswith("GPU-") for value in identifiers)
    ):
        return {
            "status": "unavailable",
            "gpu_ids": [],
            "reason": (
                "The genuine probe did not provide a valid unique "
                "physical GPU inventory."
            ),
        }
    return {
        "status": "available",
        "gpu_ids": identifiers,
        "probe": "nvidia-smi --query-gpu=uuid --format=csv,noheader",
        "stdout_sha256": _sha(process.stdout),
    }


def probe_host_resources(scratch_root: str | Path) -> dict[str, Any]:
    """Read real Linux/psutil/filesystem limits; no benchmark or GPU promise."""
    root = _path(scratch_root)
    if not root.is_dir():
        raise HostCapacityError(
            "The scratch capacity probe requires an existing directory."
        )
    affinity = os.sched_getaffinity(0)
    cores = len(affinity)
    memory = psutil.virtual_memory()
    memory_limit = memory.total
    available = memory.available
    cpu_observed = False
    memory_observed = False
    for directory in _cgroup_directories():
        cpu_file = directory / "cpu.max"
        if cpu_file.is_file():
            cpu_observed = True
            fields = cpu_file.read_text().split()
            if len(fields) != 2 or int(fields[1]) <= 0:
                raise HostCapacityError("The actual cgroup CPU limit is malformed.")
            if fields[0] != "max":
                cores = min(cores, int(fields[0]) // int(fields[1]))
        memory_file = directory / "memory.max"
        if memory_file.is_file():
            memory_observed = True
            maximum = memory_file.read_text().strip()
            if maximum != "max":
                current_file = directory / "memory.current"
                if not current_file.is_file():
                    raise HostCapacityError(
                        "A finite cgroup memory limit lacks actual usage."
                    )
                maximum_bytes = int(maximum)
                current_bytes = int(current_file.read_text())
                memory_limit = min(memory_limit, maximum_bytes)
                available = min(available, max(0, maximum_bytes - current_bytes))
    if not cpu_observed or not memory_observed:
        raise HostCapacityError(
            "Actual cgroup-v2 CPU/RAM limits are unavailable; capacity is unqualified."
        )
    info = root.stat()
    filesystem = os.statvfs(root)
    return {
        "host": _host_identity(),
        "cpu_cores": cores,
        "cpu_affinity_sha256": digest(sorted(affinity)),
        "memory_total_mb": memory_limit // MIB,
        "memory_available_mb": available // MIB,
        "scratch_free_mb": filesystem.f_bavail * filesystem.f_frsize // MIB,
        "scratch_device": info.st_dev,
        "gpu": _gpu_probe(),
        "measured_at_unix": time.time(),
        "memory_and_scratch_unit": "MiB (1024**2 bytes)",
    }


def _validate_policy(policy: HostAllocationPolicy, probe: dict[str, Any]) -> None:
    if (
        policy.max_cpu_cores > probe["cpu_cores"]
        or policy.max_memory_mb > probe["memory_total_mb"]
        or policy.max_scratch_mb > probe["scratch_free_mb"]
        or len(set(policy.gpu_ids)) != len(policy.gpu_ids)
        or not set(policy.gpu_ids).issubset(probe["gpu"]["gpu_ids"])
    ):
        raise HostCapacityError("Requested host policy exceeds actual probed capacity.")


def _process(pid: int, *, require_descendant: bool = True) -> dict[str, Any]:
    if type(pid) is not int or pid < 1:
        raise HostAuthorityError("A positive actual process PID is required.")
    try:
        process = psutil.Process(pid)
        start = process.create_time()
        if (
            process.uids().effective != os.getuid()
            or process.status() == psutil.STATUS_ZOMBIE
            or not process.is_running()
        ):
            raise HostAuthorityError(
                "The observed process is not a live owned process."
            )
        if (
            require_descendant
            and pid != os.getpid()
            and os.getpid() not in {parent.pid for parent in process.parents()}
        ):
            raise HostAuthorityError(
                "Bind only the current process or its actual descendant."
            )
        return {"pid": pid, "process_start": start}
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        raise HostAuthorityError(
            "Actual owned process identity is unavailable."
        ) from exc


def _death_observation(pid: int, process_start: float) -> dict[str, Any]:
    try:
        process = psutil.Process(pid)
        current_start = process.create_time()
        if current_start != process_start:
            return {
                "status": "original_process_dead",
                "proof": "actual_pid_reuse",
                "observed_create_time": current_start,
            }
        if process.status() == psutil.STATUS_ZOMBIE:
            return {
                "status": "original_process_dead",
                "proof": "actual_zombie_state",
                "observed_create_time": current_start,
            }
    except psutil.NoSuchProcess:
        return {"status": "original_process_dead", "proof": "actual_pid_absent"}
    except psutil.AccessDenied as exc:
        raise HostAuthorityError(
            "Process liveness is unknown; resources remain reserved."
        ) from exc
    raise HostAllocationError(
        "The original bound process is still alive; resources remain reserved."
    )


def _valid_process_start_time(value: Any, *, nullable: bool = False) -> bool:
    """Accept either finite JSON number representation of an observed timestamp.

    RFC8785 encodes integral observed floats as integer tokens. Numeric identity
    equality remains exact; booleans and nonfinite/overflowing values are invalid.
    """
    if nullable and value is None:
        return True
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _nested_death_observations(
    workspace: Path, owner_pid: int, owner_start: float, *, required: bool = False
) -> list[dict[str, Any]]:
    """Fence the known constrained adapter's independently sessioned children.

    The adapter durably creates each evaluation/input before Popen. An empty
    inventory therefore has no dispatched child in this supported adapter. A
    partial launch with no actual ownership/wait evidence stays unknown.
    Parent-death signals and the outer process's death are never child proofs.
    """
    evaluations = _path(workspace / "constrained-engine" / "evaluations")
    if not evaluations.exists():
        if required:
            raise HostAuthorityError(
                "Required constrained launch inventory is missing; resources "
                "remain reserved with unknown child ownership."
            )
        return []
    _private_directory(evaluations)
    observations: list[dict[str, Any]] = []

    def read(path: Path, schema: str, fields: set[str]) -> tuple[dict[str, Any], str]:
        try:
            raw = _private_file(path).read_bytes()
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or set(value) != fields | {"schema_version"}
                or value["schema_version"] != schema
            ):
                raise ValueError("Unexpected process receipt schema.")
            return value, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError) as exc:
            raise HostAuthorityError(
                "Nested process evidence is unavailable/invalid; resources remain "
                "reserved."
            ) from exc

    def identity(value: Any, *, nullable: bool = False) -> bool:
        return _valid_process_start_time(value, nullable=nullable)

    for root in sorted(evaluations.iterdir()):
        if (
            not root.name.startswith("evaluation-")
            or not root.name.removeprefix("evaluation-").isdigit()
        ):
            raise HostAuthorityError("Unexpected nested process inventory entry.")
        _private_directory(root)
        try:
            _private_file(root / "backend-request.json")
        except OSError as exc:
            raise HostAuthorityError(
                "Nested launch inventory is incomplete; resources remain reserved."
            ) from exc
        binding_path = _path(root / "owner-binding.json")
        parent_path = _path(root / "parent-process-observation.json")
        wait_path = _path(root / "worker-process.json")
        binding = parent = waited = None
        binding_hash = parent_hash = wait_hash = None
        if binding_path.exists():
            binding, binding_hash = read(
                binding_path,
                "cochem.torq.native-owner-binding/1",
                {
                    "owner_pid",
                    "owner_create_time",
                    "worker_pid",
                    "worker_create_time",
                    "parent_death_signal",
                    "platform",
                },
            )
            if (
                binding["owner_pid"] != owner_pid
                or binding["owner_create_time"] != owner_start
                or type(binding["owner_pid"]) is not int
                or not identity(binding["owner_create_time"])
                or type(binding["worker_pid"]) is not int
                or binding["worker_pid"] <= 0
                or not identity(binding["worker_create_time"])
                or type(binding["parent_death_signal"]) is not int
                or binding["parent_death_signal"] != signal.SIGKILL
                or binding["platform"] != "linux"
            ):
                raise HostAuthorityError("Nested child has a foreign owner binding.")
        if parent_path.exists():
            parent, parent_hash = read(
                parent_path,
                "cochem.torq.native-parent-observation/1",
                {
                    "owner_pid",
                    "owner_create_time",
                    "worker_pid",
                    "worker_create_time",
                    "identity_observation",
                },
            )
            if (
                parent["owner_pid"] != owner_pid
                or parent["owner_create_time"] != owner_start
                or type(parent["owner_pid"]) is not int
                or not identity(parent["owner_create_time"])
                or type(parent["worker_pid"]) is not int
                or parent["worker_pid"] <= 0
                or not identity(parent["worker_create_time"], nullable=True)
                or parent["identity_observation"]
                != (
                    "observed"
                    if parent["worker_create_time"] is not None
                    else "unavailable"
                )
                or (
                    binding is not None
                    and (
                        parent["worker_pid"] != binding["worker_pid"]
                        or (
                            parent["worker_create_time"] is not None
                            and parent["worker_create_time"]
                            != binding["worker_create_time"]
                        )
                    )
                )
            ):
                raise HostAuthorityError("Nested parent process identity conflicts.")
        if wait_path.exists():
            waited, wait_hash = read(
                wait_path,
                "cochem.torq.native-worker-wait/1",
                {
                    "worker_pid",
                    "worker_create_time",
                    "wait_completed",
                    "returncode",
                    "owner_binding_sha256",
                    "parent_observed_limit_failure",
                },
            )
            if (
                parent is None
                or type(waited["worker_pid"]) is not int
                or waited["worker_pid"] != parent["worker_pid"]
                or not identity(waited["worker_create_time"], nullable=True)
                or waited["worker_create_time"] != parent["worker_create_time"]
                or waited["wait_completed"] is not True
                or type(waited["returncode"]) is not int
                or waited["owner_binding_sha256"] != binding_hash
                or not (
                    waited["parent_observed_limit_failure"] is None
                    or isinstance(waited["parent_observed_limit_failure"], str)
                )
            ):
                raise HostAuthorityError("Nested actual-wait evidence conflicts.")
        worker = binding if binding is not None else parent
        if worker is None or (worker["worker_create_time"] is None and waited is None):
            raise HostAuthorityError(
                "Nested process ownership/death is unknown; resources remain reserved."
            )
        proof = (
            _death_observation(worker["worker_pid"], worker["worker_create_time"])[
                "proof"
            ]
            if worker["worker_create_time"] is not None
            else "actual_parent_wait"
        )
        observations.append(
            {
                "evaluation_id": root.name,
                "worker_pid": worker["worker_pid"],
                "worker_create_time": worker["worker_create_time"],
                "process_death_proof": proof,
                "owner_binding_sha256": binding_hash,
                "parent_observation_sha256": parent_hash,
                "worker_wait_sha256": wait_hash,
            }
        )
    if required and not observations:
        raise HostAuthorityError(
            "Required constrained launch inventory is empty; resources remain "
            "reserved with unknown child ownership."
        )
    return observations


class HostAllocationLedger:
    """Trusted local API, requiring the same private ledger/capability for all jobs."""

    @classmethod
    def initialize(
        cls, path: str | Path, policy: HostAllocationPolicy, *, actor: str
    ) -> CreatedHostAllocationLedger:
        path = _path(path)
        _private_directory(path.parent, create=True)
        _explicit(actor, "deployment actor")
        policy = HostAllocationPolicy.model_validate(policy)
        probe = probe_host_resources(policy.scratch_probe_root)
        _validate_policy(policy, probe)
        authority = secrets.token_urlsafe(32)
        _exclusive(path, b"")
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE deployment (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_id TEXT NOT NULL, authority_hash TEXT NOT NULL,
                    actor_hash TEXT NOT NULL, host_json TEXT NOT NULL,
                    policy_json TEXT NOT NULL, probe_json TEXT NOT NULL,
                    scratch_device INTEGER NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE allocations (
                    id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL, actor_hash TEXT NOT NULL,
                    workspace TEXT NOT NULL, workspace_device INTEGER NOT NULL,
                    workspace_inode INTEGER NOT NULL, request_json TEXT NOT NULL,
                    state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    lease_hash TEXT, lease_expires REAL NOT NULL,
                    lease_seconds REAL NOT NULL, owner_pid INTEGER NOT NULL,
                    owner_start REAL NOT NULL, pid INTEGER, process_start REAL,
                    created_at REAL NOT NULL, finished_at REAL, usage_json TEXT,
                    final_observation_json TEXT);
                CREATE UNIQUE INDEX allocations_active_attempt
                    ON allocations(campaign_id,attempt_id)
                    WHERE state IN ('reserved','dispatching','running');
                CREATE TABLE events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    allocation_id TEXT REFERENCES allocations(id),
                    revision INTEGER NOT NULL, actor_hash TEXT NOT NULL,
                    kind TEXT NOT NULL, timestamp REAL NOT NULL,
                    reason TEXT NOT NULL, details_json TEXT NOT NULL);
                CREATE TRIGGER events_no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
                CREATE TRIGGER events_no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
                CREATE TRIGGER allocations_terminal_immutable
                    BEFORE UPDATE ON allocations
                    WHEN OLD.state IN ('finished','reconciled','undispatched_released')
                    BEGIN SELECT RAISE(ABORT,'Terminal allocations are immutable'); END;
            """)
            connection.execute(
                "INSERT INTO deployment VALUES(1,?,?,?,?,?,?,?,?)",
                (
                    SCHEMA,
                    _sha(authority),
                    _sha(actor),
                    canonical_json(probe["host"]).decode(),
                    policy.model_dump_json(),
                    canonical_json(probe).decode(),
                    probe["scratch_device"],
                    time.time(),
                ),
            )
            connection.execute("PRAGMA user_version=1")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise HostAllocationError("Initial host ledger integrity failed.")
        except BaseException:
            connection.close()
            path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
        return CreatedHostAllocationLedger(
            {
                "schema": SCHEMA,
                "host_sha256": probe["host"]["host_sha256"],
                "policy": policy.model_dump(),
                "policy_sha256": digest(policy.model_dump()),
                "authority_private": True,
            },
            authority,
        )

    def __init__(self, path: str | Path, *, authority: str, actor: str):
        self.path = _private_file(Path(path))
        _private_directory(self.path.parent)
        _explicit(actor, "deployment actor")
        self._actor = actor
        self._authority = authority
        self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=15)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._deployment()
            if self.connection.execute("PRAGMA integrity_check").fetchone()[
                0
            ] != "ok" or list(self.connection.execute("PRAGMA foreign_key_check")):
                raise HostAllocationError("Host ledger integrity verification failed.")
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def _deployment(self) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM deployment WHERE singleton=1"
        ).fetchone()
        if (
            not isinstance(row, sqlite3.Row)
            or row["schema_id"] != SCHEMA
            or self.connection.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise HostAllocationError("Unknown host allocation schema.")
        if (
            not isinstance(self._authority, str)
            or not secrets.compare_digest(row["authority_hash"], _sha(self._authority))
            or not secrets.compare_digest(row["actor_hash"], _sha(self._actor))
        ):
            raise HostAuthorityError(
                "The private deployment capability and bound actor are required."
            )
        if json.loads(row["host_json"]) != _host_identity():
            raise HostAuthorityError(
                "The exact host, boot or namespace identity changed; "
                "recovery must be reviewed."
            )
        return row

    @property
    def policy(self) -> HostAllocationPolicy:
        return HostAllocationPolicy.model_validate_json(
            self._deployment()["policy_json"]
        )

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._deployment()
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _event(
        self,
        allocation: str,
        revision: int,
        kind: str,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        _explicit(reason, "operational reason")
        self.connection.execute(
            "INSERT INTO events(allocation_id,revision,actor_hash,kind,timestamp,"
            "reason,details_json) VALUES(?,?,?,?,?,?,?)",
            (
                allocation,
                revision,
                _sha(self._actor),
                kind,
                time.time(),
                reason,
                canonical_json(details).decode(),
            ),
        )

    def _row(self, allocation: str, revision: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM allocations WHERE id=?", (allocation,)
        ).fetchone()
        if not isinstance(row, sqlite3.Row):
            raise HostAllocationError("The host allocation is absent.")
        if type(revision) is not int or row["revision"] != revision:
            raise HostRevisionConflict("The durable host allocation revision changed.")
        if row["actor_hash"] != _sha(self._actor):
            raise HostAuthorityError(
                "The allocation belongs to another deployment actor."
            )
        return row

    def _lease(self, row: sqlite3.Row, token: str) -> None:
        if (
            row["state"] not in ACTIVE
            or not isinstance(token, str)
            or not row["lease_hash"]
            or not secrets.compare_digest(row["lease_hash"], _sha(token))
            or time.time() >= row["lease_expires"]
        ):
            raise HostAuthorityError(
                "The host allocation lease is expired, stale or terminal; "
                "resources remain reserved."
            )
        self._workspace(row)

    def _workspace(self, row: sqlite3.Row) -> None:
        path = _path(row["workspace"])
        info = path.stat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise HostAuthorityError("The current workspace ownership is unsafe.")
        if (info.st_dev, info.st_ino) != (
            row["workspace_device"],
            row["workspace_inode"],
        ):
            raise HostAuthorityError("The reserved workspace identity changed.")

    def _capacity(
        self,
        request: HostAllocationRequest,
        policy: HostAllocationPolicy,
        probe: dict[str, Any],
    ) -> None:
        records = [
            HostAllocationRequest.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT request_json FROM allocations "
                "WHERE state IN ('reserved','dispatching','running')"
            )
        ]
        checks = (
            (
                sum(item.cores for item in records) + request.cores,
                min(policy.max_cpu_cores, probe["cpu_cores"]),
                "CPU cores",
            ),
            (
                sum(item.memory_mb for item in records) + request.memory_mb,
                min(policy.max_memory_mb, probe["memory_available_mb"]),
                "RAM",
            ),
            (
                sum(item.scratch_mb for item in records) + request.scratch_mb,
                min(policy.max_scratch_mb, probe["scratch_free_mb"]),
                "scratch",
            ),
            (
                sum(item.worker_slots for item in records) + request.worker_slots,
                policy.max_workers,
                "worker slots",
            ),
        )
        for demand, capacity, label in checks:
            if demand > capacity:
                raise HostCapacityError(
                    f"Shared host {label} capacity is exhausted; "
                    "existing reservations remain held."
                )
        used_gpus = {gpu for item in records for gpu in item.gpu_ids}
        if (
            len(set(request.gpu_ids)) != len(request.gpu_ids)
            or not set(request.gpu_ids).issubset(policy.gpu_ids)
            or not set(request.gpu_ids).issubset(probe["gpu"]["gpu_ids"])
            or used_gpus.intersection(request.gpu_ids)
        ):
            raise HostCapacityError(
                "The requested physical GPU slots are unavailable or already reserved."
            )

    def acquire(
        self,
        *,
        campaign_id: str,
        attempt_id: str,
        workspace: str | Path,
        request: HostAllocationRequest,
        lease_seconds: float = 60,
        pid: int | None = None,
    ) -> AcquiredHostAllocation:
        for value, label in (
            (campaign_id, "campaign identity"),
            (attempt_id, "attempt identity"),
        ):
            _explicit(value, label)
        if (
            not isinstance(lease_seconds, (float, int))
            or isinstance(lease_seconds, bool)
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
            or lease_seconds > 3600
        ):
            raise HostAllocationError(
                "A finite positive host lease duration "
                "no longer than 3600s is required."
            )
        request = HostAllocationRequest.model_validate(request)
        path = _path(workspace)
        if not path.is_dir():
            raise HostAuthorityError(
                "The actual workspace must exist before host allocation."
            )
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise HostAuthorityError(
                "The workspace must be owned with no other writers."
            )
        owner = _process(os.getpid())
        dispatch = _process(pid) if pid is not None else None
        token = secrets.token_urlsafe(32)
        allocation = str(uuid4())
        with self._transaction():
            policy = self.policy
            probe = probe_host_resources(policy.scratch_probe_root)
            deployment = self._deployment()
            if (
                info.st_dev != deployment["scratch_device"]
                or probe["scratch_device"] != deployment["scratch_device"]
            ):
                raise HostCapacityError(
                    "The workspace is outside the pinned scratch filesystem."
                )
            self._capacity(request, policy, probe)
            # Lock contention and genuine probing may outlive a child. Bind only
            # an identity observed again inside this same admission transaction.
            current_owner = _process(owner["pid"])
            if current_owner != owner:
                raise HostAuthorityError("The reserving process identity changed.")
            if dispatch is not None and _process(dispatch["pid"]) != dispatch:
                raise HostAuthorityError("The dispatch process identity changed.")
            try:
                self.connection.execute(
                    "INSERT INTO allocations(id,campaign_id,attempt_id,actor_hash,"
                    "workspace,workspace_device,workspace_inode,request_json,state,"
                    "lease_hash,lease_expires,lease_seconds,owner_pid,owner_start,"
                    "pid,process_start,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        allocation,
                        campaign_id,
                        attempt_id,
                        _sha(self._actor),
                        str(path),
                        info.st_dev,
                        info.st_ino,
                        request.model_dump_json(),
                        "running" if dispatch else "reserved",
                        _sha(token),
                        time.time() + lease_seconds,
                        lease_seconds,
                        owner["pid"],
                        owner["process_start"],
                        dispatch["pid"] if dispatch else None,
                        dispatch["process_start"] if dispatch else None,
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise HostAllocationError(
                    "This campaign attempt already holds an active host allocation."
                ) from exc
            self._event(
                allocation,
                0,
                "acquired",
                "Reserved explicit resources across the shared deployment ledger.",
                {
                    "request": request.model_dump(),
                    "process_bound": dispatch is not None,
                    "probe_sha256": digest(probe),
                },
            )
        return AcquiredHostAllocation(self.allocation(allocation), token)

    def begin_dispatch(
        self, allocation_id: str, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            if row["state"] != "reserved":
                raise HostAllocationError(
                    "Only an undispatched reservation may begin dispatch."
                )
            self.connection.execute(
                "UPDATE allocations SET state='dispatching',revision=revision+1 "
                "WHERE id=?",
                (allocation_id,),
            )
            self._event(
                allocation_id,
                expected_revision + 1,
                "dispatch_started",
                "Dispatch may launch a process; an unbound crash remains unknown.",
                {"process_bound": False, "resources_released": False},
            )
        return self.allocation(allocation_id)

    def bind_dispatch(
        self, allocation_id: str, expected_revision: int, lease_token: str, *, pid: int
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            if (
                row["state"] not in ("reserved", "dispatching")
                or row["pid"] is not None
            ):
                raise HostAllocationError(
                    "This reservation is already bound or cannot dispatch."
                )
            actual = _process(pid)
            self.connection.execute(
                "UPDATE allocations SET state='running',revision=revision+1,pid=?,"
                "process_start=? WHERE id=?",
                (actual["pid"], actual["process_start"], allocation_id),
            )
            self._event(
                allocation_id,
                expected_revision + 1,
                "dispatch_bound",
                "Observed actual live owned process identity.",
                {"pid": actual["pid"], "process_start": actual["process_start"]},
            )
        return self.allocation(allocation_id)

    def heartbeat(
        self, allocation_id: str, expected_revision: int, lease_token: str
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            actual = _process(
                row["pid"] if row["pid"] is not None else row["owner_pid"],
                require_descendant=False,
            )
            expected = (
                row["process_start"] if row["pid"] is not None else row["owner_start"]
            )
            if actual["process_start"] != expected:
                raise HostAuthorityError("The host lease process identity changed.")
            self.connection.execute(
                "UPDATE allocations SET lease_expires=?,revision=revision+1 WHERE id=?",
                (time.time() + row["lease_seconds"], allocation_id),
            )
            self._event(
                allocation_id,
                expected_revision + 1,
                "heartbeat",
                "Observed current owned process under the exact host lease.",
                {"process_live": True},
            )
        return self.allocation(allocation_id)

    def observe_worker_exit(
        self,
        allocation_id: str,
        expected_revision: int,
        lease_token: str,
    ) -> dict[str, Any]:
        """Record actual exit racing a heartbeat without forgiving lost authority.

        The exact current unexpired lease, revision, workspace, actor and host
        must still validate. This does not release resources or claim science
        succeeded; explicit actual-death reconciliation remains mandatory.
        """
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            if row["state"] != "running" or row["pid"] is None:
                raise HostAllocationError("An actual bound worker is required.")
            observation = _death_observation(row["pid"], row["process_start"])
            self.connection.execute(
                "UPDATE allocations SET revision=revision+1 WHERE id=?",
                (allocation_id,),
            )
            self._event(
                allocation_id,
                expected_revision + 1,
                "worker_exit_observed",
                "Actual worker exited while current host authority remained valid.",
                {"observation": observation, "resources_released": False},
            )
        return self.allocation(allocation_id)

    def _release(
        self,
        row: sqlite3.Row,
        *,
        state: str,
        reason: str,
        usage: ObservedHostUsage | None,
        observation: dict[str, Any],
    ) -> None:
        if usage is not None:
            usage = ObservedHostUsage.model_validate(usage)
            _explicit(usage.measurement_source, "resource measurement source")
        self.connection.execute(
            "UPDATE allocations SET state=?,revision=revision+1,lease_hash=NULL,"
            "finished_at=?,usage_json=?,final_observation_json=? WHERE id=?",
            (
                state,
                time.time(),
                usage.model_dump_json() if usage else None,
                canonical_json(observation).decode(),
                row["id"],
            ),
        )
        self._event(
            row["id"],
            row["revision"] + 1,
            state,
            reason,
            {
                "process_observation": observation,
                "resource_peaks": usage.model_dump() if usage else None,
                "missing_peaks_invented": False,
            },
        )

    def finish(
        self,
        allocation_id: str,
        expected_revision: int,
        lease_token: str,
        *,
        reason: str,
        usage: ObservedHostUsage | None = None,
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            if row["state"] != "running" or row["pid"] is None:
                raise HostAllocationError(
                    "Finish requires an actually bound dispatched process."
                )
            current = _process(os.getpid())
            if (
                row["pid"] == current["pid"]
                and row["process_start"] == current["process_start"]
            ):
                observation = {
                    "status": "trusted_current_worker_reported_work_finished",
                    "process_stopped": False,
                }
            else:
                observation = _death_observation(row["pid"], row["process_start"])
            observation["nested_process_observations"] = _nested_death_observations(
                Path(row["workspace"]),
                row["pid"],
                row["process_start"],
                required=HostAllocationRequest.model_validate_json(
                    row["request_json"]
                ).child_inventory_scope
                == "constrained-evaluation-v1",
            )
            self._release(
                row,
                state="finished",
                reason=reason,
                usage=usage,
                observation=observation,
            )
        return self.allocation(allocation_id)

    def release_undispatched(
        self,
        allocation_id: str,
        expected_revision: int,
        lease_token: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            self._lease(row, lease_token)
            current = _process(os.getpid())
            if (
                row["state"] != "reserved"
                or row["pid"] is not None
                or row["owner_pid"] != current["pid"]
                or row["owner_start"] != current["process_start"]
            ):
                raise HostAllocationError(
                    "Only the live reserving owner may release "
                    "before any dispatch begins."
                )
            self._release(
                row,
                state="undispatched_released",
                reason=reason,
                usage=None,
                observation={
                    "status": "trusted_live_owner_reported_no_dispatch_started",
                    "process_stopped": False,
                },
            )
        return self.allocation(allocation_id)

    def reconcile(
        self,
        allocation_id: str,
        expected_revision: int,
        *,
        reason: str,
        usage: ObservedHostUsage | None = None,
    ) -> dict[str, Any]:
        with self._transaction():
            row = self._row(allocation_id, expected_revision)
            if row["state"] not in ACTIVE:
                raise HostAllocationError(
                    "Only an active allocation may be reconciled."
                )
            self._workspace(row)
            if row["pid"] is None or row["process_start"] is None:
                raise HostAllocationError(
                    "Dispatch/process ownership is unknown; "
                    "resources remain reserved for reviewed recovery."
                )
            observation = _death_observation(row["pid"], row["process_start"])
            observation["nested_process_observations"] = _nested_death_observations(
                Path(row["workspace"]),
                row["pid"],
                row["process_start"],
                required=HostAllocationRequest.model_validate_json(
                    row["request_json"]
                ).child_inventory_scope
                == "constrained-evaluation-v1",
            )
            self._release(
                row,
                state="reconciled",
                reason=reason,
                usage=usage,
                observation=observation,
            )
        return self.allocation(allocation_id)

    def allocation(self, allocation_id: str) -> dict[str, Any]:
        self._deployment()
        row = self.connection.execute(
            "SELECT * FROM allocations WHERE id=?", (allocation_id,)
        ).fetchone()
        if row is None:
            raise HostAllocationError("The host allocation is absent.")
        return {
            "allocation_id": row["id"],
            "campaign_id": row["campaign_id"],
            "attempt_id": row["attempt_id"],
            "state": row["state"],
            "revision": row["revision"],
            "request": json.loads(row["request_json"]),
            "lease_expires": row["lease_expires"],
            "process_bound": row["pid"] is not None,
            "resource_peaks": json.loads(row["usage_json"])
            if row["usage_json"]
            else None,
            "final_observation": json.loads(row["final_observation_json"])
            if row["final_observation_json"]
            else None,
        }

    def accounting(self) -> dict[str, Any]:
        self._deployment()
        requests = [
            HostAllocationRequest.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT request_json FROM allocations "
                "WHERE state IN ('reserved','dispatching','running')"
            )
        ]
        return {
            "policy": self.policy.model_dump(),
            "active_allocations": len(requests),
            "reserved_cpu_cores": sum(item.cores for item in requests),
            "reserved_memory_mb": sum(item.memory_mb for item in requests),
            "reserved_scratch_mb": sum(item.scratch_mb for item in requests),
            "reserved_worker_slots": sum(item.worker_slots for item in requests),
            "reserved_gpu_ids": sorted(
                gpu for item in requests for gpu in item.gpu_ids
            ),
            "expired_leases_release_capacity": False,
        }


def bootstrap_host_allocation(
    *,
    actor: str,
    directory: str | Path | None = None,
    policy: HostAllocationPolicy | None = None,
    scratch_root: str | Path | None = None,
) -> HostAllocationLedger:
    """Atomically open/bootstrap the common private ledger, preserving its policy.

    The default root is the user's private state directory, independent of a
    checkout or campaign. A configured root must be shared by every deployment
    caller. DB and capability are admitted together as an immutable private bundle.
    """
    _explicit(actor, "deployment actor")
    if directory is None:
        state = os.environ.get("XDG_STATE_HOME")
        root = (
            (Path(state) if state else Path.home() / ".local" / "state")
            / "cochem-torq"
            / "host-allocation"
        )
    else:
        root = Path(directory)
    root = _private_directory(root, create=True)
    host = _host_identity()
    bundle = root / host["host_sha256"]
    lock = root / "bootstrap.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    _private_file(lock)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if bundle.exists():
            _private_directory(bundle)
            authority_file = _private_file(bundle / "operator-capability.txt")
            ledger = HostAllocationLedger(
                bundle / "ledger.sqlite",
                authority=authority_file.read_text(),
                actor=actor,
            )
            if policy is not None and ledger.policy != policy:
                ledger.close()
                raise HostCapacityError(
                    "The reviewed pinned host policy cannot be silently overwritten."
                )
            return ledger
        probe_root = str(_path(scratch_root or tempfile.gettempdir()))
        if policy is None:
            probe = probe_host_resources(probe_root)
            if probe["cpu_cores"] < 1 or probe["memory_available_mb"] < 2:
                raise HostCapacityError(
                    "Actual host capacity cannot support one configured worker."
                )
            policy = HostAllocationPolicy(
                max_cpu_cores=probe["cpu_cores"],
                max_memory_mb=max(1, int(probe["memory_available_mb"] * 0.8)),
                max_scratch_mb=int(probe["scratch_free_mb"] * 0.8),
                max_workers=min(16, probe["cpu_cores"]),
                gpu_ids=[],
                scratch_probe_root=probe_root,
            )
        stage = Path(tempfile.mkdtemp(prefix=".bootstrap-", dir=root))
        try:
            created = HostAllocationLedger.initialize(
                stage / "ledger.sqlite", policy, actor=actor
            )
            _exclusive(stage / "operator-capability.txt", created.authority.encode())
            _sync_directory(stage)
            _publish(stage, bundle)
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        return HostAllocationLedger(
            bundle / "ledger.sqlite", authority=created.authority, actor=actor
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
