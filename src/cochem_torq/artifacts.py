"""Immutable per-worker shards and a verifying, single-writer artifact merger.

Scientific workers never share a writable HDF5 handle. Publication locks are
local to the merger; the existing fenced SQLite coordinator owns task leases.
Hashes establish byte integrity, not signatures or scientific accuracy.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .domain import (
    PRODUCT_TO_STAGE,
    SPECTROSCOPY_STAGES,
    CalculationRequest,
    StageResult,
    canonical_json,
    digest,
    read_json,
)


def file_digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory(directory: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact bundles forbid symlinks.")
        if path.is_file() and path.relative_to(directory).as_posix() != "manifest.json":
            entries.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                }
            )
        elif not path.is_file() and not path.is_dir():
            raise ValueError(
                "Artifact bundles permit only regular files and directories."
            )
    return entries


def _sync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def seal_shard(
    directory: str | Path,
    *,
    request_sha256: str,
    request_id: str,
    recipe_sha256: str,
    source_identity: dict[str, Any],
    worker_id: str,
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        raise FileExistsError("An immutable shard cannot be resealed.")
    if (
        not (directory / "result.json").is_file()
        or not (directory / "request.json").is_file()
    ):
        raise ValueError("A shard requires its authentic request and typed result.")
    manifest = {
        "schema_version": "cochem.torq.shard/1",
        "request_id": request_id,
        "request_sha256": request_sha256,
        "recipe_sha256": recipe_sha256,
        "source_identity": source_identity,
        "worker_id": worker_id,
        "files": inventory(directory),
    }
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(canonical_json(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    for item in manifest["files"]:
        _sync(directory / item["path"])
    return manifest


def verify_shard(
    directory: str | Path,
    *,
    expected_request_sha256: str | None = None,
    expected_recipe_sha256: str | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    if directory.is_symlink() or (directory / "manifest.json").is_symlink():
        raise ValueError("Artifact roots and manifests cannot be symlinks.")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "cochem.torq.shard/1":
        raise ValueError("Unsupported shard schema.")
    for field in ("request_sha256", "recipe_sha256"):
        value = manifest.get(field, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Invalid {field}.")
    if (
        expected_request_sha256
        and manifest["request_sha256"] != expected_request_sha256
    ):
        raise ValueError("The downloaded shard belongs to a different request.")
    if expected_recipe_sha256 and manifest["recipe_sha256"] != expected_recipe_sha256:
        raise ValueError("The shard belongs to a different scientific recipe.")
    listed = manifest.get("files", [])
    if not listed or len({item["path"] for item in listed}) != len(listed):
        raise ValueError("Empty or duplicate artifact inventory.")
    for item in listed:
        path = PurePosixPath(item["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item["path"]
            or path.as_posix() == "manifest.json"
        ):
            raise ValueError("Unsafe artifact path.")
    if inventory(directory) != listed:
        raise ValueError(
            "Artifact inventory/hash mismatch; missing, changed or unrecorded bytes."
        )
    request = read_json(directory / "request.json")
    result = read_json(directory / "result.json")
    CalculationRequest.model_validate(request)
    if sha256(canonical_json(request)).hexdigest() != manifest["request_sha256"]:
        raise ValueError("Request bytes do not match the canonical request identity.")
    if (
        str(request.get("request_id")) != manifest["request_id"]
        or str(result.get("request_id")) != manifest["request_id"]
    ):
        raise ValueError("Request/result identity mismatch.")
    if result.get("recipe_sha256") != manifest["recipe_sha256"]:
        raise ValueError("Result/recipe identity mismatch.")
    if result.get("request_sha256") != manifest["request_sha256"]:
        raise ValueError("Scientific result belongs to a different molecular request.")
    if result.get("source_identity") != manifest["source_identity"]:
        raise ValueError("Result/source identity mismatch.")
    recipe = dict(result.get("recipe", {}))
    recipe.pop("recipe_sha256", None)
    if (
        digest(recipe) != manifest["recipe_sha256"]
        or recipe.get("id") != request["recipe"]
    ):
        raise ValueError("Scientific recipe bytes/identity mismatch.")
    if result.get("schema_version") != "cochem.torq.result/1" or result.get(
        "status"
    ) not in {"complete", "partial", "failed"}:
        raise ValueError("Unsupported scientific result contract.")
    if result.get("molecule") != request["molecule"]:
        raise ValueError("Molecular identity/state mismatch.")
    if not isinstance(result.get("stages"), dict) or set(SPECTROSCOPY_STAGES) - set(
        result["stages"]
    ):
        raise ValueError("The complete typed spectroscopy-stage ledger is missing.")
    for stage in result["stages"].values():
        StageResult.model_validate(stage)
    if not isinstance(result.get("errors"), list):
        raise ValueError("Scientific result errors must be an explicit list.")
    if result["status"] == "complete" and (
        result["errors"]
        or recipe.get("runnable") is not True
        or any(
            result["stages"].get(PRODUCT_TO_STAGE[product], {}).get("status")
            != "available"
            for product in request["products"]
        )
    ):
        raise ValueError(
            "Completion contradicts requested-product availability or execution errors."
        )
    # Version 1 has no qualified identification/accuracy profile. A sealed hash
    # cannot upgrade experimental calculations into benchmark-qualified claims.
    if (
        result.get("experimental_accuracy_established") is not False
        or result.get("identification_ready") is not False
    ):
        raise ValueError(
            "Unsupported scientific accuracy/identification qualification."
        )
    return manifest


def publish_shard(staging: str | Path, destination: str | Path) -> Path:
    """Publish once by atomic rename, serialized across actual OS processes."""
    staging, destination = Path(staging).resolve(), Path(destination).absolute()
    verify_shard(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.publication.lock"
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("A published artifact bundle cannot be overwritten.")
        # Cross-device publication first copies to a private sibling directory.
        if staging.stat().st_dev != destination.parent.stat().st_dev:
            copied = Path(tempfile.mkdtemp(prefix=".shard-", dir=destination.parent))
            shutil.copytree(staging, copied, dirs_exist_ok=True)
            verify_shard(copied)
            staging = copied
        os.rename(staging, destination)
        descriptor = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return destination


def merge_shards(shards: list[str | Path], destination: str | Path) -> Path:
    """One merger records independently sealed, identity-consistent shards."""
    if not shards:
        raise ValueError("Supply at least one genuine worker shard.")
    manifests = [verify_shard(path) for path in shards]
    identities = {
        (
            m["request_id"],
            m["request_sha256"],
            m["recipe_sha256"],
            canonical_json(m["source_identity"]),
        )
        for m in manifests
    }
    if len(identities) != 1 or len({m["worker_id"] for m in manifests}) != len(
        manifests
    ):
        raise ValueError(
            "Worker shards have conflicting request, recipe, source "
            "or worker identities."
        )
    target = Path(destination).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".merge-", dir=target.parent))
    try:
        records = []
        for index, (source, manifest) in enumerate(zip(shards, manifests)):
            name = f"shard-{index:04d}"
            shutil.copytree(source, staging / name)
            records.append(
                {
                    "directory": name,
                    "manifest_sha256": file_digest(staging / name / "manifest.json"),
                    "worker_id": manifest["worker_id"],
                }
            )
        bundle = {
            "schema_version": "cochem.torq.merged-shards/1",
            "shards": records,
            "request_sha256": manifests[0]["request_sha256"],
            "recipe_sha256": manifests[0]["recipe_sha256"],
        }
        (staging / "bundle.json").write_bytes(canonical_json(bundle) + b"\n")
        _sync(staging / "bundle.json")
        lock = target.parent / f".{target.name}.publication.lock"
        with lock.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if target.exists() or target.is_symlink():
                raise FileExistsError("A merged bundle cannot be overwritten.")
            os.rename(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)
