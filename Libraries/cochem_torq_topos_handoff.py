"""Durable import of a reviewed TOPOS ensemble; this does not run a TORQ solver.

All three CoChem packages are mandatory for this interface. TOPOS owns the
versioned producer contract and validates its scientific provenance. TORQ owns
this importer and writes acknowledgment only after its own import is durable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

IMPORT_SCHEMA = "torq-reviewed-ensemble/0.1.0"
COMMIT_SCHEMA = "torq-reviewed-ensemble-commit/0.1.0"
FILES = {"handoff.json", "ensemble.json", "receipt.json"}


def _api():
    try:
        from topos.review import CONSUMER_RECEIPT_SCHEMA, verify_torq_handoff
        from topos.storage import digest_json, fsync_directory, json_bytes
    except ImportError as exc:
        raise RuntimeError(
            "TOPOS is required by the mandatory CoChem handoff interface"
        ) from exc
    return (
        CONSUMER_RECEIPT_SCHEMA,
        verify_torq_handoff,
        digest_json,
        fsync_directory,
        json_bytes,
    )


def _import_body(handoff: dict, consumer: dict, importer_sha256: str) -> dict:
    return {
        "schema_version": IMPORT_SCHEMA,
        "status": "imported-awaiting-calculation",
        "computation_performed": False,
        "consumer": consumer,
        "importer_sha256": importer_sha256,
        "run_id": handoff["run_id"],
        "handoff_sha256": handoff["handoff_sha256"],
        "ensemble_schema": handoff["ensemble_schema"],
        "ensemble_sha256": handoff["ensemble_sha256"],
        "members": handoff["members"],
        "request": handoff["request"],
        "review": handoff["review"],
        "limitations": handoff["limitations"]
        + [
            "Receipt acknowledges import only, "
            "not TORQ calculation or kinetic validation",
            "A requested TORQ calculation must separately validate "
            "its physical capability and required raw files",
        ],
    }


def load_imported_ensemble(destination: str | Path) -> dict:
    """Read a committed import and revalidate its exact geometry/schema/membership."""
    receipt_schema, verify, digest, _, _ = _api()
    target = Path(destination)
    if target.is_symlink() or not target.is_dir():
        raise ValueError("TORQ import must be an existing regular directory")
    if any(p.is_symlink() or not p.is_file() for p in target.iterdir()):
        raise ValueError("TORQ import contains non-file or symlink membership")
    if {p.name for p in target.iterdir()} != FILES | {"commit.json"}:
        raise ValueError("TORQ import membership differs from its transaction")
    commit = json.loads((target / "commit.json").read_text(encoding="utf-8"))
    if (
        commit.get("schema_version") != COMMIT_SCHEMA
        or set(commit.get("files", {})) != FILES
    ):
        raise ValueError("TORQ import commit schema/membership mismatch")
    for name in FILES:
        content = (target / name).read_bytes()
        if commit["files"][name] != {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }:
            raise ValueError("TORQ imported file checksum mismatch")
    handoff = verify(target / "handoff.json")
    ensemble = json.loads((target / "ensemble.json").read_text(encoding="utf-8"))
    receipt = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
    consumer = receipt.get("consumer", {})
    if (
        consumer.get("name") != "CoChem-TORQ"
        or set(consumer) != {"name", "version"}
        or not isinstance(consumer["version"], str)
        or not consumer["version"].strip()
    ):
        raise ValueError("TORQ import consumer identity mismatch")
    importer_hash = ensemble.get("importer_sha256", "")
    if len(importer_hash) != 64 or any(
        c not in "0123456789abcdef" for c in importer_hash
    ):
        raise ValueError("TORQ import does not identify the importer source")
    if ensemble != _import_body(handoff, consumer, importer_hash):
        raise ValueError("TORQ imported ensemble differs from the accepted source")
    timestamp = datetime.fromisoformat(receipt["consumed_at"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("TORQ receipt timestamp needs a timezone")
    expected = {
        "schema_version": receipt_schema,
        "run_id": handoff["run_id"],
        "ensemble_schema": handoff["ensemble_schema"],
        "ensemble_sha256": handoff["ensemble_sha256"],
        "handoff_sha256": handoff["handoff_sha256"],
        "members": [
            {"member_id": m["member_id"], "geometry_sha256": m["geometry_sha256"]}
            for m in handoff["members"]
        ],
        "consumer": consumer,
        "consumed_at": receipt["consumed_at"],
        "status": "consumed",
    }
    if receipt != {**expected, "receipt_sha256": digest(expected)}:
        raise ValueError("TORQ consumption receipt differs from the durable import")
    return {"ensemble": ensemble, "receipt": receipt}


def consume_topos_handoff(handoff_path: str | Path, destination: str | Path) -> dict:
    """Atomically import exact reviewed members, then return TORQ's real receipt.

    Existing identical committed imports are idempotent. A changed source requires
    a new destination; the importer never overwrites an earlier ensemble version.
    """
    from filelock import FileLock

    receipt_schema, verify, digest, fsync_directory, json_bytes = _api()
    source, target = Path(handoff_path), Path(destination).absolute()
    if target.is_symlink() or source.is_symlink():
        raise ValueError("TORQ handoff paths cannot be symlinks")
    handoff = verify(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(
        str(target.parent / ("." + target.name + ".import.lock")), timeout=10
    ):
        if target.exists():
            imported = load_imported_ensemble(target)
            if imported["receipt"]["handoff_sha256"] != handoff["handoff_sha256"]:
                raise FileExistsError(
                    "An immutable different TORQ ensemble already occupies "
                    "this destination"
                )
            return imported
        consumer = {"name": "CoChem-TORQ", "version": version("CoChem-TORQ")}
        ensemble = _import_body(
            handoff, consumer, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        body = {
            "schema_version": receipt_schema,
            "run_id": handoff["run_id"],
            "ensemble_schema": handoff["ensemble_schema"],
            "ensemble_sha256": handoff["ensemble_sha256"],
            "handoff_sha256": handoff["handoff_sha256"],
            "members": [
                {"member_id": m["member_id"], "geometry_sha256": m["geometry_sha256"]}
                for m in handoff["members"]
            ],
            "consumer": consumer,
            "consumed_at": datetime.now(timezone.utc).isoformat(),
            "status": "consumed",
        }
        receipt = {**body, "receipt_sha256": digest(body)}
        staging = Path(
            tempfile.mkdtemp(prefix="." + target.name + ".pending-", dir=target.parent)
        )
        try:
            inventory = {}
            for name, value in (
                ("handoff.json", handoff),
                ("ensemble.json", ensemble),
                ("receipt.json", receipt),
            ):
                content = json_bytes(value) + b"\n"
                with (staging / name).open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                inventory[name] = {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            with (staging / "commit.json").open("xb") as stream:
                stream.write(
                    json_bytes({"schema_version": COMMIT_SCHEMA, "files": inventory})
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            load_imported_ensemble(staging)
            fsync_directory(staging)
            os.rename(staging, target)
            fsync_directory(target.parent)
            return load_imported_ensemble(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a reviewed TOPOS ensemble; no TORQ calculation is performed"
    )
    parser.add_argument("handoff", type=Path)
    parser.add_argument("destination", type=Path)
    options = parser.parse_args(argv)
    try:
        imported = consume_topos_handoff(options.handoff, options.destination)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(3, f"TORQ import failed: {exc}\n")
    print(json.dumps(imported["receipt"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
