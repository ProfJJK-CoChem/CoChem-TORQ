"""Immutable attempt-specific PES samples and coordinator-fenced HDF5 merges.

Array integrity and ownership do not establish molecular accuracy. Caller-supplied
samples remain explicitly unqualified; mathematical samples are never labelled
engine observations. Existing local SWMR stores are independent numerical APIs.
Only a current committed coordinator receipt makes an output authoritative.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cochem.orchestration.campaign import (
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
    RevisionConflict,
)
from cochem_torq.domain import Molecule, canonical_json, read_json


class PESIdentity(BaseModel):
    """Explicit atom order, electronic state, method, source and sample units."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    molecule: Molecule
    recipe: dict[str, Any]
    source_identity: dict[str, Any]
    evidence_class: Literal[
        "supplied_data", "mathematical_validation", "native_engine_observation"
    ]
    coordinate_unit: Literal["bohr", "dimensionless"]
    energy_unit: Literal["hartree", "dimensionless"]
    engine: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def identity_contract(self) -> PESIdentity:
        if self.molecule.atom_ids is None:
            raise ValueError("PES samples require explicit ordered atom identifiers.")
        if not isinstance(self.recipe.get("id"), str) or not self.recipe["id"]:
            raise ValueError("PES identity requires an explicit recipe identifier.")
        canonical_json(self.recipe)
        canonical_json(self.source_identity)
        if not self.source_identity:
            raise ValueError("PES identity requires actual source provenance.")
        if self.evidence_class == "mathematical_validation" and (
            self.coordinate_unit != "dimensionless"
            or self.energy_unit != "dimensionless"
        ):
            raise ValueError("Mathematical samples do not establish quantum units.")
        if self.evidence_class == "native_engine_observation" and (
            self.coordinate_unit != "bohr"
            or self.energy_unit != "hartree"
            or self.engine != "PySCF"
            or self.recipe["id"]
            not in {
                "hf-sto-3g-pes-validation",
                "hf-sto-3g-internal-pes-validation",
            }
            or self.source_identity.get("schema_version")
            != "cochem.torq.native-pes-source/1"
        ):
            raise ValueError(
                "Native PES observations require the exact supported bridge identity."
            )
        return self

    def record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _hash(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _require_locking() -> None:
    # HDF5's environment policy can override h5py's explicit locking=True.
    # Best-effort or disabled locks cannot establish this ownership protocol.
    policy = os.environ.get("HDF5_USE_FILE_LOCKING")
    if policy is not None and policy.upper() not in {"TRUE", "1"}:
        raise CampaignError("PES publication requires mandatory HDF5 file locking.")


def _rename_exclusive(source: Path, target: Path) -> None:
    """Atomically publish without replacement on the canonical Linux runtime."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise CampaignError(
            "This PES protocol requires Linux atomic no-replace rename."
        )
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. Unsupported filesystems fail closed.
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _molecule_identity(identity: PESIdentity) -> str:
    molecule = identity.molecule.model_dump(mode="json")
    # PES coordinates differ by frame; immutable chemical identity excludes them.
    molecule.pop("geometry_bohr")
    return _digest(molecule)


def _frame_ids(values: list[str], count: int) -> list[str]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Exactly one physical sample UUID is required per frame.")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Sample identifiers must be UUID strings.")
    normalized = [str(UUID(value)) for value in values]
    if normalized != values or len(set(normalized)) != count:
        raise ValueError("Sample UUIDs must be canonical, unique and ordered.")
    return normalized


def _arrays(
    identity: PESIdentity,
    coordinates: Any,
    energies: Any,
    frame_ids: list[str],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    if np.iscomplexobj(coordinates) or np.iscomplexobj(energies):
        raise ValueError("Complex samples cannot be silently cast to real PES data.")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    expected = (len(identity.molecule.symbols), 3)
    if (
        coordinates.ndim != 3
        or coordinates.shape[1:] != expected
        or not len(coordinates)
    ):
        raise ValueError(
            "PES coordinates need a nonempty exact [samples,atoms,3] shape."
        )
    if energies.shape != (len(coordinates),):
        raise ValueError("Every coordinate frame requires its supplied scalar energy.")
    if not np.isfinite(coordinates).all() or not np.isfinite(energies).all():
        raise ValueError("Unavailable/nonfinite samples cannot be sealed as PES data.")
    return coordinates, energies, _frame_ids(frame_ids, len(coordinates))


def shard_task_payload(identity: PESIdentity, frame_ids: list[str]) -> dict[str, Any]:
    return {
        "kind": "pes_shard",
        "identity": identity.record(),
        "frame_ids": _frame_ids(frame_ids, len(frame_ids)),
    }


def merge_task_payload(
    identity: PESIdentity, sources: list[str | Path]
) -> dict[str, Any]:
    manifests = [verify_pes_artifact(path) for path in sources]
    return {
        "kind": "pes_merge",
        "identity": identity.record(),
        "source_refs": [
            _source_ref(path, manifest) for path, manifest in zip(sources, manifests)
        ],
    }


def _source_ref(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_manifest_sha256": _hash(Path(path) / "manifest.json"),
        "ownership": manifest["ownership"],
        "identity_sha256": manifest["identity_sha256"],
    }


def _owner(
    coordinator: CampaignCoordinator,
    attempt_id: str,
    expected_revision: int,
    token: str,
    generation: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _require_locking()
    if type(expected_revision) is not int or type(generation) is not int:
        raise ValueError("Attempt revision and fencing generation must be integers.")
    attempt = coordinator.attempt(attempt_id)
    if attempt["revision"] != expected_revision:
        raise RevisionConflict("PES writer requires the current attempt revision.")
    if attempt["state"] != "validating" or attempt["generation"] != generation:
        raise CampaignError("PES publication requires its current validating attempt.")
    coordinator.heartbeat(attempt_id, lease_token=token, lease_generation=generation)
    task = coordinator.connection.execute(
        "SELECT * FROM campaign_tasks WHERE id=?", (attempt["task_id"],)
    ).fetchone()
    if (
        task is None
        or task["engine"] != payload["identity"]["engine"]
        or task["recipe"] != payload["identity"]["recipe"]["id"]
    ):
        raise CampaignError("PES engine/recipe differs from the authorized task.")
    if read_json_text(task["payload_json"]) != payload:
        raise CampaignError("PES data/source identities differ from the approved task.")
    return {
        "campaign_id": task["campaign_id"],
        "task_id": task["id"],
        "attempt_id": attempt_id,
        "lease_generation": generation,
        "validation_revision": expected_revision,
        "committed_revision": expected_revision + 1,
    }


def read_json_text(value: str) -> dict[str, Any]:
    import json

    decoded = json.loads(value)
    canonical_json(decoded)
    if not isinstance(decoded, dict):
        raise ValueError("PES task identity must be a JSON object.")
    return decoded


def _write(
    directory: Path,
    identity: PESIdentity,
    coordinates: np.ndarray[Any, Any],
    energies: np.ndarray[Any, Any],
    frame_ids: list[str],
    ownership: dict[str, Any],
    kind: Literal["shard", "merge"],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    _require_locking()
    path = directory / "samples.h5"
    with h5py.File(path, "x", libver="latest", locking=True) as store:
        store.attrs["schema_version"] = "cochem.pes.samples/1"
        store.attrs["identity_json"] = canonical_json(identity.record()).decode()
        store.attrs["identity_sha256"] = _digest(identity.record())
        store.attrs["coordinate_unit"] = identity.coordinate_unit
        store.attrs["energy_unit"] = identity.energy_unit
        store.attrs["scientific_accuracy_established"] = False
        store.create_dataset(
            "coordinates",
            data=coordinates,
            chunks=True,
            compression="gzip",
            fletcher32=True,
        )
        store.create_dataset(
            "energies", data=energies, chunks=True, compression="gzip", fletcher32=True
        )
        store.create_dataset(
            "frame_ids", data=frame_ids, dtype=h5py.string_dtype("utf-8")
        )
        # No readers see staging. The marker still distinguishes complete arrays
        # from allocated tails and is validated again after HDF5 closes.
        store.create_dataset(
            "committed_frames", data=np.array([len(frame_ids)], dtype="uint64")
        )
        store.flush()
    path.chmod(0o600)
    _sync_file(path)
    manifest = {
        "schema_version": "cochem.pes.artifact/1",
        "kind": kind,
        "identity": identity.record(),
        "identity_sha256": _digest(identity.record()),
        "molecule_sha256": _molecule_identity(identity),
        "recipe_sha256": _digest(identity.recipe),
        "source_identity_sha256": _digest(identity.source_identity),
        "scientific_accuracy_established": False,
        "ownership": ownership,
        "sources": sources,
        "sample_count": len(frame_ids),
        "frame_ids": frame_ids,
        "files": [
            {
                "path": "samples.h5",
                "sha256": _hash(path),
                "size_bytes": path.stat().st_size,
            }
        ],
    }
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    (directory / "manifest.json").chmod(0o600)
    _sync_file(directory / "manifest.json")
    _sync_directory(directory)
    verify_pes_artifact(directory)
    return manifest


def verify_pes_artifact(directory: str | Path) -> dict[str, Any]:
    _require_locking()
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("PES artifacts must be regular sealed directories.")
    if {path.name for path in root.iterdir()} != {"samples.h5", "manifest.json"}:
        raise ValueError("PES artifact has incomplete or unsealed extra files.")
    if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise ValueError("PES artifacts forbid symlinks and special files.")
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("PES manifest must be an identity object.")
    identity = PESIdentity.model_validate(manifest["identity"])
    if (
        manifest["schema_version"] != "cochem.pes.artifact/1"
        or manifest["kind"] not in {"shard", "merge"}
        or manifest["identity_sha256"] != _digest(identity.record())
        or manifest["molecule_sha256"] != _molecule_identity(identity)
        or manifest["recipe_sha256"] != _digest(identity.recipe)
        or manifest["source_identity_sha256"] != _digest(identity.source_identity)
        or manifest["scientific_accuracy_established"] is not False
    ):
        raise ValueError("PES identity/schema integrity is invalid.")
    path = root / "samples.h5"
    expected_files = [
        {"path": "samples.h5", "sha256": _hash(path), "size_bytes": path.stat().st_size}
    ]
    if manifest["files"] != expected_files:
        raise ValueError("PES file inventory/hash differs from sealed bytes.")
    with h5py.File(path, "r", libver="latest", locking=True) as store:
        if (
            set(store) != {"coordinates", "energies", "frame_ids", "committed_frames"}
            or store.attrs.get("schema_version") != "cochem.pes.samples/1"
            or store.attrs.get("identity_sha256") != manifest["identity_sha256"]
            or read_json_text(store.attrs["identity_json"]) != identity.record()
            or store.attrs.get("coordinate_unit") != identity.coordinate_unit
            or store.attrs.get("energy_unit") != identity.energy_unit
            or bool(store.attrs.get("scientific_accuracy_established", True))
        ):
            raise ValueError("PES HDF5 schema/identity differs from its manifest.")
        if (
            store["committed_frames"].shape != (1,)
            or store["committed_frames"].dtype != np.dtype("uint64")
            or store["coordinates"].dtype != np.dtype("float64")
            or store["energies"].dtype != np.dtype("float64")
        ):
            raise ValueError("PES HDF5 dataset types/commit marker are invalid.")
        coordinates, energies, ids = _arrays(
            identity,
            store["coordinates"][:],
            store["energies"][:],
            store["frame_ids"].asstr()[:].tolist(),
        )
        if (
            int(store["committed_frames"][0]) != len(ids)
            or manifest["sample_count"] != len(ids)
            or manifest["frame_ids"] != ids
        ):
            raise ValueError("PES artifact contains an incomplete committed prefix.")
    return manifest


def _receipt(target: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "cochem.pes.publication-receipt/1",
        "artifact_manifest_sha256": _hash(target / "manifest.json"),
        "identity_sha256": manifest["identity_sha256"],
        "molecule_sha256": manifest["molecule_sha256"],
        "recipe_sha256": manifest["recipe_sha256"],
        "source_identity_sha256": manifest["source_identity_sha256"],
        "ownership": manifest["ownership"],
        "artifact_path": str(target),
        "kind": manifest["kind"],
        "sample_count": manifest["sample_count"],
        "scientific_accuracy_established": False,
    }


def _admit(
    coordinator: CampaignCoordinator,
    sources: list[str | Path],
    manifests: list[dict[str, Any]],
) -> None:
    seen = set()
    for source, manifest in zip(sources, manifests):
        owner = manifest["ownership"]
        if owner["attempt_id"] in seen:
            raise ValueError("Duplicate source attempts cannot enter a PES merge.")
        seen.add(owner["attempt_id"])
        current = coordinator.committed_attempt_for_admission(
            owner["attempt_id"], expected_revision=owner["committed_revision"]
        )
        if (
            current["task_id"] != owner["task_id"]
            or current["campaign_id"] != owner["campaign_id"]
            or current["lease_generation"] != owner["lease_generation"]
            or current["result"] != _receipt(Path(source).absolute(), manifest)
        ):
            raise CampaignError(
                "Source shard is not the exact current committed artifact."
            )
        if verify_pes_artifact(source) != manifest:
            raise ValueError("Source changed after PES merge preparation.")


def _publish(
    coordinator: CampaignCoordinator,
    staging: Path,
    target: Path,
    manifest: dict[str, Any],
    attempt_id: str,
    expected_revision: int,
    lease_token: str,
    lease_generation: int,
    usage: MeasuredUsage,
    actor: str,
    sources: list[str | Path],
    source_manifests: list[dict[str, Any]],
) -> None:
    lock = target.parent / f".{target.name}.publication.lock"
    with lock.open("a+b") as stream:
        os.fchmod(stream.fileno(), 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        receipt = _receipt(staging, manifest)
        receipt["artifact_path"] = str(target)

        def commit() -> None:
            if source_manifests:
                _admit(coordinator, sources, source_manifests)
            verify_pes_artifact(staging)
            if manifest["identity"]["evidence_class"] == "native_engine_observation":
                # A label or a caller-supplied no-op callback cannot qualify data.
                # The bridge checks source authority and genuine native quantities
                # inside this same coordinator publication transaction.
                from cochem_torq.pes_storage import admit_native_scan_points

                admit_native_scan_points(coordinator, staging)
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    "An immutable PES artifact cannot be overwritten."
                )
            _rename_exclusive(staging, target)
            _sync_directory(target.parent)

        coordinator.publish(
            attempt_id,
            expected_revision,
            receipt,
            usage,
            commit,
            state="succeeded",
            lease_token=lease_token,
            lease_generation=lease_generation,
            actor=actor,
        )


def publish_pes_shard(
    coordinator: CampaignCoordinator,
    attempt_id: str,
    expected_revision: int,
    *,
    identity: PESIdentity,
    coordinates: Any,
    energies: Any,
    frame_ids: list[str],
    destination: str | Path,
    lease_token: str,
    lease_generation: int,
    usage: MeasuredUsage,
    actor: str,
) -> Path:
    """Seal supplied samples and publish only under their current approved lease."""
    coordinates, energies, frame_ids = _arrays(
        identity, coordinates, energies, frame_ids
    )
    ownership = _owner(
        coordinator,
        attempt_id,
        expected_revision,
        lease_token,
        lease_generation,
        shard_task_payload(identity, frame_ids),
    )
    target = Path(destination).absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".pes-{attempt_id}-", dir=target.parent))
    try:
        manifest = _write(
            staging, identity, coordinates, energies, frame_ids, ownership, "shard", []
        )
        _publish(
            coordinator,
            staging,
            target,
            manifest,
            attempt_id,
            expected_revision,
            lease_token,
            lease_generation,
            usage,
            actor,
            [],
            [],
        )
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def merge_pes_shards(
    coordinator: CampaignCoordinator,
    attempt_id: str,
    expected_revision: int,
    *,
    identity: PESIdentity,
    sources: list[str | Path],
    destination: str | Path,
    lease_token: str,
    lease_generation: int,
    usage: MeasuredUsage,
    actor: str,
) -> Path:
    """A sole writer creates a new artifact; authoritative admission occurs last."""
    if not sources:
        raise ValueError("Supply genuine, committed independent PES shards.")
    manifests = [verify_pes_artifact(path) for path in sources]
    if any(
        manifest["identity_sha256"] != _digest(identity.record())
        for manifest in manifests
    ):
        raise ValueError("PES atom, state, method, source or unit identities differ.")
    refs = [_source_ref(path, manifest) for path, manifest in zip(sources, manifests)]
    ownership = _owner(
        coordinator,
        attempt_id,
        expected_revision,
        lease_token,
        lease_generation,
        {"kind": "pes_merge", "identity": identity.record(), "source_refs": refs},
    )
    target = Path(destination).absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    coordinate_blocks: list[np.ndarray[Any, Any]] = []
    energy_blocks: list[np.ndarray[Any, Any]] = []
    frame_ids: list[str] = []
    # Consume private, checksum-verified copies. A source changed during a read
    # and restored before final admission cannot contaminate merged samples.
    # Source authority is independently checked again inside publication.
    with tempfile.TemporaryDirectory(
        prefix=f".pes-inputs-{attempt_id}-", dir=target.parent
    ) as snapshots:
        for index, (source, manifest) in enumerate(zip(sources, manifests)):
            snapshot = Path(snapshots) / str(index)
            snapshot.mkdir(mode=0o700)
            for name in ("samples.h5", "manifest.json"):
                shutil.copyfile(Path(source) / name, snapshot / name)
            if verify_pes_artifact(snapshot) != manifest:
                raise ValueError("PES source snapshot differs from sealed bytes.")
            with h5py.File(snapshot / "samples.h5", "r", locking=True) as store:
                coordinate_blocks.append(store["coordinates"][:])
                energy_blocks.append(store["energies"][:])
                frame_ids.extend(manifest["frame_ids"])
    coordinates, energies, frame_ids = _arrays(
        identity,
        np.concatenate(coordinate_blocks),
        np.concatenate(energy_blocks),
        frame_ids,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".pes-merge-{attempt_id}-", dir=target.parent)
    )
    try:
        manifest = _write(
            staging,
            identity,
            coordinates,
            energies,
            frame_ids,
            ownership,
            "merge",
            refs,
        )
        _publish(
            coordinator,
            staging,
            target,
            manifest,
            attempt_id,
            expected_revision,
            lease_token,
            lease_generation,
            usage,
            actor,
            sources,
            manifests,
        )
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def recover_pes_publication(
    coordinator: CampaignCoordinator,
    attempt_id: str,
    expected_revision: int,
    *,
    destination: str | Path,
    lease_token: str,
    lease_generation: int,
    usage: MeasuredUsage,
    actor: str,
    sources: list[str | Path] | None = None,
) -> Path:
    """Commit an intact orphan under its same current lease; never edit its bytes.

    Expired/revoked/superseded ownership rejects recovery. An authorized new
    attempt must create its own new version; old orphan bytes are retained as
    evidence and are never silently relabelled or accepted as success.
    """
    target = Path(destination).absolute()
    manifest = verify_pes_artifact(target)
    identity = PESIdentity.model_validate(manifest["identity"])
    sources = sources or []
    source_manifests = [verify_pes_artifact(path) for path in sources]
    payload = (
        shard_task_payload(identity, manifest["frame_ids"])
        if manifest["kind"] == "shard"
        else merge_task_payload(identity, sources)
    )
    owner = _owner(
        coordinator,
        attempt_id,
        expected_revision,
        lease_token,
        lease_generation,
        payload,
    )
    if manifest["ownership"] != owner or (
        manifest["kind"] == "merge" and manifest["sources"] != payload["source_refs"]
    ):
        raise CampaignError("Orphan identity/ownership is incompatible with recovery.")

    def recover() -> None:
        if manifest["kind"] == "merge":
            _admit(coordinator, sources, source_manifests)
        if verify_pes_artifact(target) != manifest:
            raise ValueError("Orphaned artifact changed during recovery.")
        if identity.evidence_class == "native_engine_observation":
            from cochem_torq.pes_storage import admit_native_scan_points

            admit_native_scan_points(coordinator, target)
        _sync_directory(target.parent)

    coordinator.publish(
        attempt_id,
        expected_revision,
        _receipt(target, manifest),
        usage,
        recover,
        state="succeeded",
        lease_token=lease_token,
        lease_generation=lease_generation,
        actor=actor,
    )
    return target


def verify_committed_pes(
    coordinator: CampaignCoordinator, destination: str | Path
) -> dict[str, Any]:
    """Read-only authoritative check; a bare path is never committed evidence."""
    target = Path(destination).absolute()
    manifest = verify_pes_artifact(target)
    owner = manifest["ownership"]
    attempt = coordinator.attempt(owner["attempt_id"])
    # A standalone reader cannot invoke transaction-only admission. It verifies
    # the durable receipt; later merger admission checks current task generation.
    if (
        attempt["state"] != "succeeded"
        or attempt["revision"] != owner["committed_revision"]
        or attempt["generation"] != owner["lease_generation"]
        or read_json_text(attempt["result_json"]) != _receipt(target, manifest)
    ):
        raise CampaignError(
            "PES artifact lacks a succeeded coordinator commit receipt."
        )
    return manifest
