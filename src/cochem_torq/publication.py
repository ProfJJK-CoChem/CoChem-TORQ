"""Verified reproducibility exports; packaging never establishes scientific accuracy."""

from __future__ import annotations

import fcntl
import importlib.util
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import file_digest, verify_shard
from .domain import canonical_json, digest, read_json
from .operations import (
    ArtifactBackpressureError,
    ArtifactQuotaPolicy,
    admit_artifact_work,
    artifact_usage,
)

_SCHEMA = "cochem.torq.publication-bundle/1"
_README = """# TORQ reproducibility evidence

This bundle contains unchanged, verified calculation/failure evidence and the
matching numerical-analysis source. It is not an independently validated paper,
an experimental identification, a DOI deposit, or a deployment authorization.

1. Retain the independently recorded SHA-256 of publication-manifest.json. Use
   cochem_torq.publication.verify_publication_bundle with that digest before reuse.
2. Read publication.json, including every unavailable stage, error, quality flag,
   and the explicit unknown calibration/uncertainty fields. Native inputs, logs,
   checkpoints, arrays, basis definitions and engine fingerprints remain in shard/.
3. Install a legitimate TORQ distribution/source checkout matching the recorded
   scientific code_sha256, with the actual engine and dependency versions recorded
   by the calculation. analysis/source/ is a source identity snapshot, not a
   fabricated standalone installation or a claim that its dependencies are bundled.
4. Prepare a NEW request compatible with your installed request schema, preserving
   the intended molecule, state, isotopes, exact method and numerical settings.
   Choose a new request ID; keep the original archived request unchanged.
5. Run `cochem-torq doctor --execution github_actions`, then `cochem-torq plan
   --request NEW_REQUEST.json --output NEW_PLAN.json`. Review actual costs,
   prerequisites, unavailable products and approximation/accuracy limitations.
   Set COCHEM_TORQ_GITHUB_REPOSITORY to the actual owner/repository and
   COCHEM_TORQ_CALCULATION_REF to the reviewed branch/commit. Authenticate the
   existing GitHub CLI outside this bundle. Use full paths and a new idempotency
   key, for example:

       torq_plan=/absolute/path/NEW_PLAN.json
       torq_approved=/absolute/path/APPROVED_PLAN.json
       torq_results=/absolute/path/NEW_RESULTS
       cochem-torq approve-plan --plan "$torq_plan" --actor "Your name" \\
           --output "$torq_approved"
       cochem-torq run --approved-plan "$torq_approved" \\
           --idempotency-key "YOUR_NEW_KEY" --destination "$torq_results"

   The canonical calculation workflow must already be activated on the repository
   default branch; the bundle never activates it or grants credentials.
6. Retain the new run's own native files and sealed artifact. A numerical replay
   and independent reference/benchmark qualification are separate requirements.

Citations are copied only from recorded recipes/engine references. Missing
references remain missing; this exporter does not invent or verify bibliography.
Hash inventories establish byte consistency, not author signatures or peer review.
"""


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Unsafe publication inventory path.")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.as_posix() != relative
    ):
        raise ValueError("Unsafe publication inventory path.")
    return root.joinpath(*pure.parts)


def _safe_directory(path: str | Path, *, create: bool = False) -> Path:
    root = Path(path).absolute()
    for component in (root, *root.parents):
        if component.is_symlink():
            raise ValueError("Publication paths cannot traverse symlinks.")
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def _all_files(root: Path) -> list[dict[str, Any]]:
    """Include every file, including names reserved by other artifact schemas."""
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Publication bundles forbid symlinks.")
        if path.is_file():
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                }
            )
        elif not path.is_dir():
            raise ValueError("Publication bundles require regular files/directories.")
    return entries


def _source_snapshot(
    expected_sha256: str,
) -> tuple[dict[str, bytes], list[dict[str, str]]]:
    from .application import RUNTIME_SOURCE_MODULES, source_identity

    if source_identity()["code_sha256"] != expected_sha256:
        raise ValueError(
            "Publication export requires the legitimate installed source matching "
            "the sealed scientific code SHA-256; "
            "no current-code substitution is permitted."
        )
    package = Path(__file__).resolve().parent
    sources: dict[str, bytes] = {}
    records = []
    for path in sorted(package.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("Scientific source snapshots forbid symlinks.")
        name = f"cochem_torq/{path.relative_to(package).as_posix()}"
        raw = path.read_bytes()
        sources[name] = raw
        records.append({"path": name, "sha256": sha256(raw).hexdigest()})
    # Preserve the actual source_identity order and path labels. These are real
    # imported modules bound by the application's scientific-source identity.
    for module in RUNTIME_SOURCE_MODULES:
        specification = importlib.util.find_spec(module)
        if specification is None or not specification.origin:
            raise ValueError(f"Required runtime source is unavailable: {module}")
        path = Path(specification.origin)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Require genuine regular scientific source files.")
        raw = path.read_bytes()
        sources[f"{module.replace('.', '/')}.py"] = raw
        records.append({"path": module, "sha256": sha256(raw).hexdigest()})
    if digest(records) != expected_sha256:
        raise ValueError(
            "Scientific source changed during publication snapshot capture."
        )
    return sources, records


def _native_records(
    shard: Path, manifest: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for key in ("native_result", "harmonic_native_result"):
        native = result.get(key)
        if native is None:
            continue
        if not isinstance(native, dict):
            raise ValueError("Native engine evidence must be an actual result object.")
        matching = [
            item
            for item in manifest["files"]
            if item["sha256"] == native.get("manifest_sha256")
            and PurePosixPath(item["path"]).name == "manifest.json"
        ]
        if not matching:
            raise ValueError("Native result has no retained matching engine manifest.")
        for item in matching:
            native_manifest_path = _path(shard, item["path"])
            native_manifest = read_json(native_manifest_path)
            if (
                native_manifest.get("schema_version")
                != "cochem-torq.engine-artifacts.v1"
                or native_manifest != native.get("artifacts")
                or not isinstance(native_manifest.get("artifacts"), list)
            ):
                raise ValueError(
                    "Native engine manifest differs from its retained result."
                )
            names = set()
            for artifact in native_manifest["artifacts"]:
                relative = artifact["path"]
                path = _path(native_manifest_path.parent, relative)
                if relative in names or not path.is_file() or path.is_symlink():
                    raise ValueError(
                        "Native engine inventory is incomplete or duplicated."
                    )
                names.add(relative)
                if (
                    path.stat().st_size != artifact["size_bytes"]
                    or file_digest(path) != artifact["sha256"]
                ):
                    raise ValueError("Native engine inventory bytes do not verify.")
        records.append(
            {
                "result_field": key,
                "manifest_sha256": native["manifest_sha256"],
                "retained_manifests": [item["path"] for item in matching],
                "engine": native.get("engine"),
                "engine_version": native.get("engine_version"),
                "engine_installation_sha256": native.get("engine_installation_sha256"),
                "adapter_source_sha256": native.get("adapter_source_sha256"),
                "python_version": native.get("python_version"),
                "libxc_version": native.get("libxc_version"),
                "platform": native.get("platform"),
                "basis_definition_sha256": native.get("basis_definition_sha256"),
                "method": native.get("method"),
                "settings": native.get("settings"),
                "status": native.get("status"),
                "scf": native.get("scf"),
                "stability": native.get("stability"),
                "optimization": native.get("optimization"),
                "correlation": native.get("correlation"),
                "dispersion": native.get("dispersion"),
                "density_functional": native.get("density_functional"),
                "quality_flags": native.get("quality_flags"),
                "errors": native.get("errors"),
            }
        )
    return records


def _summary(
    shard: Path, manifest: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    request = read_json(shard / "request.json")
    ledger = {
        name: {
            "status": stage["status"],
            "reason": stage.get("reason"),
            "absence_kind": stage.get("absence_kind"),
            "quality_flags": stage.get("quality_flags", []),
            "payload_reference": f"shard/result.json#/stages/{name}/value"
            if stage["status"] == "available"
            else None,
        }
        for name, stage in result["stages"].items()
    }
    native = _native_records(shard, manifest, result)
    citations = []
    recipe_citation = result["recipe"].get("citation")
    if recipe_citation is not None:
        citations.append(
            {"source": "shard/result.json#/recipe/citation", "value": recipe_citation}
        )
    for record in native:
        references = (record.get("density_functional") or {}).get("references")
        if references:
            citations.append(
                {
                    "source": (
                        f"shard/result.json#/{record['result_field']}"
                        "/density_functional/references"
                    ),
                    "value": references,
                }
            )
    geometry = result["stages"]["equilibrium_geometry"]
    return {
        "schema_version": "cochem.torq.publication-evidence/1",
        "serialization_profile": "RFC8785",
        "request_id": manifest["request_id"],
        "request_sha256": manifest["request_sha256"],
        "shard_manifest_sha256": file_digest(shard / "manifest.json"),
        "recipe_sha256": manifest["recipe_sha256"],
        "source_identity": manifest["source_identity"],
        "scientific_cache_key": manifest.get("scientific_cache_key"),
        "molecule": request["molecule"],
        "initial_geometry_bohr": request["molecule"]["geometry_bohr"],
        "equilibrium_geometry_bohr": (geometry.get("value") or {}).get("geometry_bohr")
        if geometry["status"] == "available"
        else None,
        "resolved_isotopes": result.get("resolved_isotopes"),
        "constants": result.get("constants"),
        "scientific_goal": request.get("scientific_goal"),
        "recipe": result["recipe"],
        "native_engine_evidence": native,
        "result_status": result["status"],
        "stage_ledger": ledger,
        "errors": result["errors"],
        "eligibility": "failed_evidence_only"
        if result["status"] == "failed"
        else "partial"
        if result["status"] == "partial"
        else "exploratory",
        "publication_validated": False,
        "experimental_accuracy_established": False,
        "identification_ready": False,
        "doi": None,
        "calibration": {
            "independent_benchmark_evidence": None,
            "calibrated_uncertainty": None,
            "reason": (
                "No independently qualified calibration evidence "
                "is established by this export."
            ),
        },
        "citations": citations,
        "citation_verification": (
            "Copied verbatim from retained recipe/engine records; "
            "not independently verified by the exporter."
        ),
        "limitations": [
            "Byte integrity and matching analysis code "
            "do not establish scientific accuracy.",
            result["recipe"].get("reason"),
            result["recipe"].get("accuracy"),
            "Unavailable stages remain unavailable; "
            "calibration and uncertainty are not inferred.",
            "No DOI deposit, default-branch activation "
            "or automatic calculation is performed.",
        ],
    }


def _copy_file(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as original, destination.open("xb") as copied:
        shutil.copyfileobj(original, copied)
        copied.flush()
        os.fsync(copied.fileno())
    os.chmod(destination, 0o600)
    if (
        destination.stat().st_size != expected["size_bytes"]
        or file_digest(destination) != expected["sha256"]
    ):
        raise ValueError("Immutable shard changed during publication export.")


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def verify_publication_bundle(
    bundle: str | Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Verify bytes against an independently retained publication manifest digest."""
    root = _safe_directory(bundle)
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_manifest_sha256)
        or file_digest(root / "publication-manifest.json") != expected_manifest_sha256
    ):
        raise ValueError(
            "Publication manifest differs from the independently retained SHA-256."
        )
    manifest = read_json(root / "publication-manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("Publication manifest must be a JSON object.")
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("Unsupported publication bundle schema.")
    listed = manifest.get("files")
    if not isinstance(listed, list) or not listed:
        raise ValueError("Publication bundle has no byte inventory.")
    for item in listed:
        _path(root, item["path"])
    actual = [
        item for item in _all_files(root) if item["path"] != "publication-manifest.json"
    ]
    if actual != listed:
        raise ValueError("Publication bundle inventory/hash mismatch.")
    shard_manifest = verify_shard(root / "shard")
    if file_digest(root / "shard/manifest.json") != manifest["shard_manifest_sha256"]:
        raise ValueError("Publication bundle changed the retained shard manifest.")
    result = read_json(root / "shard/result.json")
    expected_summary = _summary(root / "shard", shard_manifest, result)
    if read_json(root / "publication.json") != expected_summary:
        raise ValueError(
            "Publication summary differs from its actual calculation evidence."
        )
    if (
        manifest.get("serialization_profile") != "RFC8785"
        or manifest.get("publication_validated") is not False
        or manifest.get("request_id") != shard_manifest["request_id"]
        or manifest.get("eligibility") != expected_summary["eligibility"]
        or manifest.get("scientific_code_sha256")
        != shard_manifest["source_identity"]["code_sha256"]
    ):
        raise ValueError(
            "Publication manifest contradicts its actual evidence or eligibility."
        )
    sources = read_json(root / "analysis/source-inventory.json")
    if (
        sources.get("schema_version") != "cochem.torq.analysis-source/1"
        or sources.get("scientific_code_sha256")
        != shard_manifest["source_identity"]["code_sha256"]
        or digest(sources["scientific_code_records"])
        != shard_manifest["source_identity"]["code_sha256"]
    ):
        raise ValueError(
            "Analysis source identity differs from the sealed calculation."
        )
    source_files = sources["source_files"]
    observed = _all_files(root / "analysis/source")
    if observed != source_files:
        raise ValueError("Packaged numerical analysis sources do not verify.")
    expected_records = []
    for item in source_files:
        name = item["path"]
        label = name if name.startswith("cochem_torq/") else name[:-3].replace("/", ".")
        expected_records.append({"path": label, "sha256": item["sha256"]})
    # Package files are lexicographically sorted; application identity places its
    # explicit supplemental modules after the package source records.
    expected_records.sort(
        key=lambda item: (
            not item["path"].startswith("cochem_torq/"),
            PurePosixPath(item["path"]).parts,
        )
    )
    if expected_records != sources["scientific_code_records"]:
        raise ValueError(
            "Analysis source inventory omits or substitutes identity-bearing code."
        )
    return manifest


def export_publication_bundle(
    shard: str | Path,
    destination: str | Path,
    *,
    policy: ArtifactQuotaPolicy = ArtifactQuotaPolicy(),
) -> dict[str, Any]:
    """Publish once: authentic shard + matching analysis code + explicit limitations."""
    from .application import source_identity

    source = _safe_directory(shard)
    manifest = verify_shard(source)
    source_digest = manifest["source_identity"].get("code_sha256")
    if not isinstance(source_digest, str):
        raise ValueError(
            "The sealed calculation has no legitimate analysis source digest."
        )
    target = Path(destination).absolute()
    _safe_directory(target.parent, create=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "Publication bundles are immutable; select a fresh destination."
        )
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Publication destination and original shard cannot overlap.")
    source_files, code_records = _source_snapshot(source_digest)
    result = read_json(source / "result.json")
    summary = _summary(source, manifest, result)
    source_entries = [
        {"path": name, "size_bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name, raw in sorted(
            source_files.items(), key=lambda item: PurePosixPath(item[0]).parts
        )
    ]
    generated = {
        "README.md": _README.encode("utf-8"),
        "publication.json": canonical_json(summary) + b"\n",
        "analysis/source-inventory.json": canonical_json(
            {
                "schema_version": "cochem.torq.analysis-source/1",
                "scientific_code_sha256": source_digest,
                "scientific_code_records": code_records,
                "source_files": source_entries,
            }
        )
        + b"\n",
        **{f"analysis/source/{name}": raw for name, raw in source_files.items()},
    }
    shard_entries = [
        *manifest["files"],
        {
            "path": "manifest.json",
            "size_bytes": (source / "manifest.json").stat().st_size,
            "sha256": file_digest(source / "manifest.json"),
        },
    ]
    files = sorted(
        [{**item, "path": f"shard/{item['path']}"} for item in shard_entries]
        + [
            {"path": name, "size_bytes": len(raw), "sha256": sha256(raw).hexdigest()}
            for name, raw in generated.items()
        ],
        key=lambda item: PurePosixPath(item["path"]).parts,
    )
    publication_manifest = {
        "schema_version": _SCHEMA,
        "serialization_profile": "RFC8785",
        "request_id": manifest["request_id"],
        "shard_manifest_sha256": file_digest(source / "manifest.json"),
        "scientific_code_sha256": source_digest,
        "eligibility": summary["eligibility"],
        "publication_validated": False,
        "files": files,
    }
    manifest_bytes = canonical_json(publication_manifest) + b"\n"
    expected_manifest = sha256(manifest_bytes).hexdigest()
    budget = sum(item["size_bytes"] for item in files) + len(manifest_bytes)
    staging = None
    with admit_artifact_work(target.parent, incoming_bytes=budget, policy=policy):
        try:
            staging = Path(
                tempfile.mkdtemp(prefix=".torq-publication-", dir=target.parent)
            )
            for item in shard_entries:
                _copy_file(
                    _path(source, item["path"]),
                    _path(staging / "shard", item["path"]),
                    item,
                )
            for name, raw in generated.items():
                _write(_path(staging, name), raw)
            _write(staging / "publication-manifest.json", manifest_bytes)
            verify_publication_bundle(
                staging, expected_manifest_sha256=expected_manifest
            )
            if source_identity()["code_sha256"] != source_digest:
                raise ValueError(
                    "Installed scientific source changed during publication export."
                )
            verify_shard(source)
            if (
                file_digest(source / "manifest.json")
                != publication_manifest["shard_manifest_sha256"]
            ):
                raise ValueError(
                    "Original scientific shard changed during publication export."
                )
            if (
                artifact_usage(staging, max_files=policy.max_files)["owned_bytes"]
                != budget
            ):
                raise ValueError(
                    "Publication staging exceeded its declared byte inventory."
                )
            store_usage = artifact_usage(target.parent, max_files=policy.max_files)
            if (
                store_usage["owned_bytes"] > policy.max_owned_bytes
                or store_usage["free_bytes"] < policy.minimum_free_bytes
            ):
                raise ArtifactBackpressureError(
                    "Actual publication store exhausted its artifact quota."
                )
            lock = target.parent / f".{target.name}.publication.lock"
            descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "a+b") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(
                        "Publication bundles cannot overwrite prior evidence."
                    )
                directories = [
                    staging,
                    *(path for path in staging.rglob("*") if path.is_dir()),
                ]
                for path in reversed(directories):
                    directory_fd = os.open(path, os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                os.rename(staging, target)
                parent_fd = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
    return {
        "schema_version": _SCHEMA,
        "request_id": manifest["request_id"],
        "directory": str(target),
        "manifest_sha256": expected_manifest,
        "shard_manifest_sha256": publication_manifest["shard_manifest_sha256"],
        "scientific_code_sha256": source_digest,
        "files": len(files),
        "size_bytes": budget,
        "eligibility": summary["eligibility"],
        "publication_validated": False,
    }
