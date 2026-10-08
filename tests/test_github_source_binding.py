"""Actual local Git and native artifacts exercise source-binding validators.

The callable below translates genuine local Git objects into the validator's
input format. It observes no GitHub API, workflow run or hosted download. The
controller-shaped catalog contains the actual committed request-validator bytes
solely for source-object classification; no BASE controller execution is claimed.
Live GitHub submission/retrieval remains unrun under the student deployment hold.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from cochem_torq.application import execute_request
from cochem_torq.artifacts import verify_shard
from cochem_torq.domain import canonical_json, digest, read_json
from cochem_torq.github_provenance import (
    _finite,
    _retained_file,
    _unique,
    scientific_source_binding,
    verify_controller_artifact,
)
from cochem_torq.publication import (
    export_publication_bundle,
    verify_publication_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = "ProfJJK-CoChem/CoChem-TORQ"
LOCAL_CATALOG = "local/source-object-check"
pytestmark = [pytest.mark.real_engine, pytest.mark.student_profile]


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


class LocalGitObjects:
    """Pure source-format adapter backed only by real local Git commands."""

    def __init__(self, repositories: dict[str, Path]):
        self.repositories = repositories

    def __call__(self, endpoint: str) -> dict:
        _, owner, name, kind, object_type, remainder = endpoint.split("/", 5)
        assert kind == "git"
        root = self.repositories[f"{owner}/{name}"]
        revision = remainder.split("?", 1)[0]
        if object_type == "commits":
            assert git(root, "cat-file", "-t", revision).strip() == b"commit"
            return {
                "sha": git(root, "rev-parse", revision).decode().strip(),
                "tree": {
                    "sha": git(root, "rev-parse", f"{revision}^{{tree}}")
                    .decode()
                    .strip()
                },
            }
        if object_type == "trees":
            assert git(root, "cat-file", "-t", revision).strip() == b"tree"
            entries = []
            for line in git(root, "ls-tree", "-rz", revision).split(b"\0"):
                if line:
                    metadata, name_bytes = line.split(b"\t")
                    mode, item_type, oid = metadata.decode().split()
                    entries.append(
                        {
                            "path": name_bytes.decode(),
                            "mode": mode,
                            "type": item_type,
                            "sha": oid,
                        }
                    )
            return {
                "sha": git(root, "rev-parse", revision).decode().strip(),
                "truncated": False,  # Native ls-tree completed its whole inventory.
                "tree": entries,
            }
        assert object_type == "blobs"
        raw = git(root, "cat-file", "blob", revision)
        return {
            "sha": git(root, "rev-parse", revision).decode().strip(),
            "size": len(raw),
            "encoding": "base64",
            "content": base64.b64encode(raw).decode(),
        }


@pytest.fixture
def local_catalog(tmp_path):
    """Commit actual source bytes; no Actions event/run/receipt is constructed."""
    source = tmp_path / "local-catalog"
    source.mkdir()
    git(source, "init", "-q")
    revision = git(ROOT, "rev-parse", "HEAD").decode().strip()
    specification = {
        "distribution": "CoChem-TORQ",
        "repository": SCIENTIFIC,
        "revision": revision,
    }
    catalog = {
        "schema_version": "cochem.module-distribution/1",
        "modules": {"torq": specification},
    }
    scripts = source / "scripts"
    scripts.mkdir()
    (scripts / "module-distribution.json").write_bytes(canonical_json(catalog))
    # This is a local source-object input, not a claimed BASE native controller.
    (scripts / "native_torq_actions.py").write_bytes(
        git(ROOT, "show", f"{revision}:ci_tools/actions_request.py")
    )
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Local source-object test",
        "-c",
        "user.email=source-object@example.invalid",
        "commit",
        "-qm",
        "Actual local Git source-object input",
    )
    local_revision = git(source, "rev-parse", "HEAD").decode().strip()
    adapter = LocalGitObjects({SCIENTIFIC: ROOT, LOCAL_CATALOG: source})
    return source, local_revision, specification, adapter


def test_direct_source_binding_uses_the_actual_owning_git_commit(local_catalog):
    _, _, specification, adapter = local_catalog
    revision = specification["revision"]
    binding = scientific_source_binding(adapter, SCIENTIFIC, revision)
    assert binding["route"] == "native-torq"
    assert binding["controller_commit"] == binding["scientific_commit"] == revision
    assert binding["scientific_tree_oid"] == (
        git(ROOT, "rev-parse", f"{revision}^{{tree}}").decode().strip()
    )


def test_local_catalog_binding_derives_both_original_git_identities(local_catalog):
    source, revision, specification, adapter = local_catalog
    binding = scientific_source_binding(adapter, LOCAL_CATALOG, revision)
    assert binding["route"] == "native-base-controller"
    assert binding["controller_commit"] == revision
    assert binding["scientific_commit"] == specification["revision"]
    assert binding["controller_commit"] != binding["scientific_commit"]
    assert (
        binding["catalog_sha256"]
        == sha256(
            (source / "scripts/module-distribution.json").read_bytes()
        ).hexdigest()
    )
    assert binding["catalog_spec_sha256"] == digest(specification)


@pytest.mark.parametrize("revision", ["", "invalid", "a" * 39, "a" * 41, "A" * 40])
def test_invalid_revision_is_rejected_without_retrieving_any_object(revision):
    def no_object_access(endpoint):
        raise AssertionError("Malformed revision must fail before source access")

    with pytest.raises(ValueError, match="immutable"):
        scientific_source_binding(no_object_access, SCIENTIFIC, revision)


def test_duplicate_and_nonfinite_catalog_literals_are_explicit_invalid_inputs():
    with pytest.raises(ValueError, match="Duplicate"):
        _unique([("revision", "first"), ("revision", "second")])
    for literal in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="Nonfinite"):
            _finite(literal)
    assert _unique([("revision", "one"), ("distribution", "two")]) == {
        "revision": "one",
        "distribution": "two",
    }


@pytest.fixture(scope="module")
def actual_hf(tmp_path_factory):
    source = tmp_path_factory.mktemp("actual-hf-source-boundary") / "result"
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0, 0, 0], [0, 0, 1.4]],
            "charge": 0,
            "multiplicity": 1,
        },
        "recipe": "hf-sto-3g-education",
        "products": ["geometry"],
        "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 60},
    }
    result = execute_request(request, source)
    assert result["status"] == "complete", result["errors"]
    verify_shard(source)
    return source


def test_genuine_native_publication_retains_original_request_and_source(
    actual_hf, tmp_path
):
    manifest = verify_shard(actual_hf)
    output = tmp_path / "publication"
    publication = export_publication_bundle(actual_hf, output)
    verified = verify_publication_bundle(
        output, expected_manifest_sha256=publication["manifest_sha256"]
    )
    assert (
        verified["scientific_code_sha256"] == manifest["source_identity"]["code_sha256"]
    )
    assert (
        read_json(output / "publication.json")["source_identity"]
        == manifest["source_identity"]
    )
    assert (output / "shard/request.json").read_bytes() == (
        actual_hf / "request.json"
    ).read_bytes()


def test_modified_native_request_fails_the_actual_inventory(actual_hf, tmp_path):
    changed = tmp_path / "changed-shard"
    shutil.copytree(actual_hf, changed)
    (changed / "request.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="inventory/hash"):
        verify_shard(changed)
    verify_shard(actual_hf)


def test_local_native_result_cannot_invent_a_hosted_controller_receipt(actual_hf):
    with pytest.raises(ValueError, match="missing"):
        verify_controller_artifact(actual_hf, {}, {}, verify_shard(actual_hf))


@pytest.mark.parametrize("corruption", ["symlink", "oversized", "escape"])
def test_original_sidecar_intake_rejects_unsafe_actual_files(tmp_path, corruption):
    root = tmp_path / "retained"
    root.mkdir()
    original = tmp_path / "original.json"
    original.write_bytes(b'{"actual_local_input":true}\n')
    path = root / "input.json"
    if corruption == "symlink":
        path.symlink_to(original)
        name, maximum = "input.json", 1024
    elif corruption == "oversized":
        path.write_bytes(original.read_bytes())
        name, maximum = "input.json", 1
    else:
        name, maximum = "../original.json", 1024
    with pytest.raises(ValueError, match="unsafe|oversized"):
        _retained_file(root, name, maximum)
    assert original.read_bytes() == b'{"actual_local_input":true}\n'
