"""Separate immutable Actions controller and scientific source bindings.

GitHub reads establish repository/commit/tree identities. Local receipts and
sealed result inventories bind the reviewed request, recipe and implementation;
these integrity checks do not certify scientific accuracy.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .domain import digest, read_json

SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate source catalog field.")
        value[key] = item
    return value


def _finite(value: str) -> Any:
    raise ValueError(f"Nonfinite source catalog number: {value}")


def scientific_source_binding(
    api: Callable[..., Any], repository: str, controller_commit: str
) -> dict[str, Any]:
    """Derive source pins from actual GitHub objects at an immutable commit."""
    if not SHA.fullmatch(controller_commit):
        raise ValueError("Actions controller requires an immutable Git commit.")

    def commit_tree(owner: str, revision: str) -> tuple[str, dict[str, Any]]:
        commit = api(f"repos/{owner}/git/commits/{revision}")
        tree_oid = (commit.get("tree") or {}).get("sha", "")
        if commit.get("sha") != revision or not SHA.fullmatch(tree_oid):
            raise ValueError("GitHub scientific source commit/tree identity mismatch.")
        tree = api(f"repos/{owner}/git/trees/{tree_oid}?recursive=1")
        if tree.get("sha") != tree_oid or tree.get("truncated") is not False:
            raise ValueError(
                "GitHub source tree inventory is incomplete or mismatched."
            )
        entries = tree.get("tree")
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("path"), str)
            for item in entries
        ):
            raise ValueError("GitHub source tree inventory is malformed.")
        records = {item["path"]: item for item in entries}
        if len(records) != len(entries):
            raise ValueError("GitHub source tree inventory contains duplicate paths.")
        return tree_oid, records

    controller_tree, files = commit_tree(repository, controller_commit)
    binding: dict[str, Any] = {
        "schema_version": "cochem.torq.scientific-source-binding/1",
        "route": "native-torq",
        "controller_repository": repository,
        "controller_commit": controller_commit,
        "scientific_repository": repository,
        "scientific_commit": controller_commit,
        "scientific_tree_oid": controller_tree,
    }
    catalog_entry = files.get("scripts/module-distribution.json")
    helper_entry = files.get("scripts/native_torq_actions.py")
    if catalog_entry is None and helper_entry is None:
        member = files.get("src/cochem_torq/application.py", {})
        if member.get("type") != "blob" or member.get("mode") not in {
            "100644",
            "100755",
        }:
            raise ValueError("The controller is not an owning native TORQ source.")
        return binding
    if (
        not isinstance(catalog_entry, dict)
        or not isinstance(helper_entry, dict)
        or any(
            item.get("type") != "blob"
            or item.get("mode") not in {"100644", "100755"}
            or not SHA.fullmatch(item.get("sha", ""))
            for item in (catalog_entry, helper_entry)
        )
    ):
        raise ValueError(
            "A separate native TORQ controller requires its real catalog/helper."
        )
    blob = api(f"repos/{repository}/git/blobs/{catalog_entry['sha']}")
    if blob.get("encoding") != "base64" or blob.get("sha") != catalog_entry["sha"]:
        raise ValueError("GitHub catalog blob identity mismatch.")
    encoded = blob.get("content")
    if not isinstance(encoded, str) or len(encoded) > 3 * 1024 * 1024:
        raise ValueError("GitHub catalog blob exceeds the intake bound.")
    raw = base64.b64decode("".join(encoded.split()), validate=True)
    oid = hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False
    ).hexdigest()
    if len(raw) > 2 * 1024 * 1024 or len(raw) != blob.get("size") or oid != blob["sha"]:
        raise ValueError("GitHub catalog bytes differ from their Git object.")
    catalog = json.loads(raw, object_pairs_hook=_unique, parse_constant=_finite)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != (
        "cochem.module-distribution/1"
    ):
        raise ValueError("Unsupported native TORQ controller source catalog.")
    modules = catalog.get("modules")
    spec = modules.get("torq") if isinstance(modules, dict) else None
    if (
        not isinstance(spec, dict)
        or spec.get("distribution") != "CoChem-TORQ"
        or not isinstance(spec.get("repository"), str)
        or not REPOSITORY.fullmatch(spec["repository"])
        or not isinstance(spec.get("revision"), str)
        or not SHA.fullmatch(spec["revision"])
    ):
        raise ValueError(
            "The catalog has no explicit immutable native TORQ source pin."
        )
    scientific_tree, scientific_files = commit_tree(
        spec["repository"], spec["revision"]
    )
    if scientific_files.get("src/cochem_torq/application.py", {}).get("type") != "blob":
        raise ValueError("The pinned scientific repository lacks native TORQ source.")
    binding.update(
        route="native-base-controller",
        scientific_repository=spec["repository"],
        scientific_commit=spec["revision"],
        scientific_tree_oid=scientific_tree,
        catalog_sha256=hashlib.sha256(raw).hexdigest(),
        catalog_spec_sha256=digest(spec),
    )
    return binding


def _retained_file(root: Path, name: str, maximum: int) -> Path:
    path = root / name
    if (
        any(part.is_symlink() for part in (path, *path.parents) if part != root.parent)
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
        or not 0 < path.stat().st_size <= maximum
    ):
        raise ValueError("Native controller evidence is missing, unsafe or oversized.")
    return path


def verify_controller_artifact(
    artifact: Path,
    binding: dict[str, Any],
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Verify original sidecar bytes alongside the already verified native shard."""
    evidence = read_json(
        _retained_file(
            artifact, "torq-evidence/controller-and-source.json", 1024 * 1024
        )
    )
    if not isinstance(evidence, dict):
        raise ValueError("Native controller/source evidence must be an object.")
    expected = {
        "schema_version": "cochem.native-torq-controller/1",
        "controller_repository": binding["controller_repository"],
        "controller_commit": binding["controller_commit"],
        "torq_repository": binding["scientific_repository"],
        "torq_commit": binding["scientific_commit"],
        "source_tree_oid": binding["scientific_tree_oid"],
        "catalog_sha256": binding["catalog_sha256"],
        "catalog_spec_sha256": binding["catalog_spec_sha256"],
        "transport_schema": "cochem.torq.request/1",
        "scientific_calculation_performed": False,
        "scientific_release_certified": False,
    }
    if any(evidence.get(key) != value for key, value in expected.items()) or any(
        type(evidence.get(key)) is not bool
        for key in ("scientific_calculation_performed", "scientific_release_certified")
    ):
        raise ValueError(
            "Native controller evidence differs from the actual source binding."
        )
    origin = f"https://github.com/{binding['scientific_repository']}"
    if evidence.get("observed_torq_origin") not in {origin, origin + ".git"} or (
        not isinstance(evidence.get("source_sha256"), str)
        or not SHA256.fullmatch(evidence["source_sha256"])
    ):
        raise ValueError(
            "Native controller source inventory/origin evidence is malformed."
        )
    raw_request = _retained_file(
        artifact, "torq-input/request.json", 16384
    ).read_bytes()
    approval_path = _retained_file(artifact, "torq-input/approved-plan.json", 32768)
    raw_approval = approval_path.read_bytes()
    if (
        hashlib.sha256(raw_request).hexdigest() != receipt["request_sha256"]
        or hashlib.sha256(raw_approval).hexdigest() != receipt["approval_sha256"]
    ):
        raise ValueError(
            "Original hosted request/approval bytes differ from their receipt."
        )
    from .service import ApprovedPlan

    approval = ApprovedPlan.model_validate(read_json(approval_path))
    request = read_json(artifact / "torq-input/request.json")
    if (
        approval.plan["request"] != request
        or approval.plan["plan_sha256"] != receipt["approved_plan_sha256"]
        or approval.source_identity.get("code_sha256")
        != receipt["expected_code_sha256"]
        or approval.plan.get("scientific_recipe", {}).get("recipe_sha256")
        != manifest["recipe_sha256"]
        or approval.source_identity.get("git_commit")
        not in {None, binding["scientific_commit"]}
    ):
        raise ValueError(
            "Original hosted approval does not bind the scientific result."
        )
    transport = read_json(
        _retained_file(artifact, "torq-input/request.transport.json", 16384)
    )
    if (
        not isinstance(transport, dict)
        or any(
            transport.get(key) != value
            for key, value in {
                "schema_version": "cochem.actions-request/1",
                "request_id": manifest["request_id"],
                "request_sha256": receipt["request_sha256"],
                "source_commit": binding["controller_commit"],
                "request_bytes": len(raw_request),
                "engine_profile": "pyscf-linux-cpu",
                "validated_transport": True,
                "scientific_calculation_performed": False,
            }.items()
        )
        or any(
            type(transport.get(key)) is not bool
            for key in ("validated_transport", "scientific_calculation_performed")
        )
    ):
        raise ValueError(
            "Original request transport differs from its controller receipt."
        )
    revision = (
        _retained_file(artifact, "torq-evidence/torq-source-commit.txt", 128)
        .read_text()
        .strip()
    )
    if revision != binding["scientific_commit"]:
        raise ValueError(
            "Retained scientific source revision differs from its catalog pin."
        )
