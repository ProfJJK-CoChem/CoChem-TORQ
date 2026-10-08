"""Real Git objects and genuine local HF shards test hosted transport guards.

The GitHub API and artifact transfer are controlled boundary fixtures; these
tests do not claim an actual hosted dispatch or hosted calculation occurred.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from ci_tools.actions_request import write_request
from cochem_torq.application import execute_request
from cochem_torq.artifacts import inventory, verify_shard
from cochem_torq.domain import canonical_json, read_json
from cochem_torq.github import GitHubAccessError, GitHubActions
from cochem_torq.github_provenance import scientific_source_binding
from cochem_torq.service import approve_plan, plan_request

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = "ProfJJK-CoChem/CoChem-TORQ"
CONTROLLER = "student/private-project"
pytestmark = [pytest.mark.real_engine, pytest.mark.student_profile]


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


@lru_cache
def committed_source_inventory(revision):
    """Hash actual immutable Git blob bytes using the native source inventory rule."""
    records = []
    for entry in git(ROOT, "ls-tree", "-rz", revision).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t")
            mode, kind, oid = metadata.split()
            assert kind == b"blob"
            records.append((name, mode, oid))
    records.sort()
    data = subprocess.check_output(
        ["git", "-C", str(ROOT), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for _, _, oid in records),
    )
    hasher, cursor = sha256(), 0
    for name, mode, oid in records:
        end = data.index(b"\n", cursor)
        observed, kind, size = data[cursor:end].split()
        assert observed == oid and kind == b"blob"
        cursor = end + 1
        content = data[cursor : cursor + int(size)]
        cursor += int(size) + 1
        file_kind = (
            b"symlink"
            if mode == b"120000"
            else (b"executable" if mode == b"100755" else b"file")
        )
        hasher.update(name + b"\0" + file_kind + b"\0" + sha256(content).digest())
    assert cursor == len(data)
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def actual_hf(tmp_path_factory):
    parent = tmp_path_factory.mktemp("actual-hf-transport")
    request = {
        "request_id": str(uuid4()),
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
    scientific_commit = git(ROOT, "rev-parse", "HEAD").decode().strip()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("COCHEM_SOURCE_COMMIT", scientific_commit)
        approved = approve_plan(plan_request(request), actor="local transport test")
        request = approved["plan"]["request"]
        shard = parent / "actual-result"
        result = execute_request(request, shard)
    assert result["status"] == "complete", result["errors"]
    verify_shard(shard)
    return shard, approved, scientific_commit


@pytest.fixture
def transport(tmp_path, actual_hf, monkeypatch):
    shard, approved, scientific_commit = actual_hf
    source = tmp_path / "controller"
    source.mkdir()
    git(source, "init", "-q")
    catalog = {
        "schema_version": "cochem.module-distribution/1",
        "modules": {
            "torq": {
                "distribution": "CoChem-TORQ",
                "repository": SCIENTIFIC,
                "revision": scientific_commit,
            }
        },
    }
    (source / "scripts").mkdir()
    (source / "scripts/module-distribution.json").write_bytes(canonical_json(catalog))
    (source / "scripts/native_torq_actions.py").write_bytes(
        (ROOT.parent / "CoChem-BASE/scripts/native_torq_actions.py").read_bytes()
    )
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Local test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Actual source-binding fixture",
    )
    controller_commit = git(source, "rev-parse", "HEAD").decode().strip()
    request = approved["plan"]["request"]
    run = {
        "id": 123,
        "head_sha": controller_commit,
        "workflow_id": 456,
        "html_url": "https://github.com/student/private-project/actions/runs/123",
        "status": "completed",
        "conclusion": "success",
        "event": "workflow_dispatch",
        "display_title": f"TORQ {request['request_id']}",
    }

    def api(endpoint, *, data=None):
        if endpoint.endswith("/actions/runs/123"):
            return run
        if endpoint.endswith("/actions/workflows/calculation.yml"):
            return {"id": 456, "state": "active"}
        if endpoint.endswith("/actions/workflows/calculation.yml/dispatches"):
            return None
        if "/actions/workflows/calculation.yml/runs?" in endpoint:
            return {"workflow_runs": [run]}
        if endpoint.startswith(f"repos/{CONTROLLER}/"):
            repository, suffix = source, endpoint.removeprefix(f"repos/{CONTROLLER}/")
        elif endpoint.startswith(f"repos/{SCIENTIFIC}/"):
            repository, suffix = ROOT, endpoint.removeprefix(f"repos/{SCIENTIFIC}/")
        else:
            raise AssertionError(endpoint)
        if suffix.startswith("commits/"):
            return {"sha": controller_commit}
        revision = suffix.split("/")[-1].split("?")[0]
        if suffix.startswith("git/commits/"):
            return {
                "sha": revision,
                "tree": {
                    "sha": git(repository, "rev-parse", f"{revision}^{{tree}}")
                    .decode()
                    .strip()
                },
            }
        if suffix.startswith("git/trees/"):
            entries = []
            for line in git(repository, "ls-tree", "-rz", revision).split(b"\0"):
                if not line:
                    continue
                metadata, name = line.split(b"\t")
                mode, kind, oid = metadata.decode().split()
                entries.append(
                    {"path": name.decode(), "mode": mode, "type": kind, "sha": oid}
                )
            return {"sha": revision, "truncated": False, "tree": entries}
        if suffix.startswith("git/blobs/"):
            raw = git(repository, "cat-file", "blob", revision)
            return {
                "sha": revision,
                "size": len(raw),
                "encoding": "base64",
                "content": base64.b64encode(raw).decode(),
            }
        raise AssertionError(endpoint)

    state = tmp_path / "receipts"
    state.mkdir()
    client = GitHubActions(CONTROLLER, state_directory=state)
    monkeypatch.setattr(client, "_api", api)
    binding = scientific_source_binding(api, CONTROLLER, controller_commit)
    artifact = tmp_path / "retained" / f"torq-result-{request['request_id']}-123-1"
    shutil.copytree(shard, artifact / "torq-results/result")
    raw_request, raw_approval = canonical_json(request), canonical_json(approved)
    write_request(
        artifact / "torq-input/request.json",
        raw_request,
        request,
        source_commit=controller_commit,
    )
    (artifact / "torq-input/approved-plan.json").write_bytes(raw_approval)
    evidence = {
        "schema_version": "cochem.native-torq-controller/1",
        "controller_repository": CONTROLLER,
        "controller_commit": controller_commit,
        "torq_repository": SCIENTIFIC,
        "torq_commit": scientific_commit,
        "observed_torq_origin": f"https://github.com/{SCIENTIFIC}.git",
        "source_tree_oid": binding["scientific_tree_oid"],
        "source_sha256": committed_source_inventory(scientific_commit),
        "catalog_sha256": binding["catalog_sha256"],
        "catalog_spec_sha256": binding["catalog_spec_sha256"],
        "transport_schema": "cochem.torq.request/1",
        "scientific_calculation_performed": False,
        "scientific_release_certified": False,
    }
    (artifact / "torq-evidence").mkdir()
    (artifact / "torq-evidence/controller-and-source.json").write_bytes(
        canonical_json(evidence)
    )
    (artifact / "torq-evidence/torq-source-commit.txt").write_text(
        scientific_commit + "\n"
    )
    receipt = {
        "repository": CONTROLLER,
        "source_commit": controller_commit,
        "scientific_source_binding": binding,
        "run_id": "123",
        "workflow_id": 456,
        "request_id": request["request_id"],
        "request_sha256": sha256(raw_request).hexdigest(),
        "approval_sha256": sha256(raw_approval).hexdigest(),
        "approved_plan_sha256": approved["plan"]["plan_sha256"],
        "expected_recipe_sha256": approved["plan"]["scientific_recipe"][
            "recipe_sha256"
        ],
        "expected_code_sha256": approved["source_identity"]["code_sha256"],
    }
    receipt_path = state / "123.json"
    receipt_path.write_bytes(canonical_json(receipt))
    observed = []
    original_run = subprocess.run

    def download(command, **options):
        if command[0] != "gh":
            return original_run(command, **options)
        assert command[:3] == ["gh", "run", "download"]
        observed.append(options["env"])
        destination = Path(command[command.index("--dir") + 1])
        shutil.copytree(artifact, destination / artifact.name, symlinks=True)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", download)
    return client, artifact, receipt_path, receipt, api, approved, observed, run


def test_separate_revisions_retrieve_actual_hf_and_keep_original_bytes(
    transport, tmp_path
):
    client, artifact, _, receipt, _, _, _, _ = transport
    assert (
        receipt["source_commit"]
        != receipt["scientific_source_binding"]["scientific_commit"]
    )
    before = inventory(artifact)
    roots = client.download("123", tmp_path / "download")
    assert (
        verify_shard(roots[0])["source_identity"]["code_sha256"]
        == receipt["expected_code_sha256"]
    )
    assert inventory(roots[0].parents[1]) == before == inventory(artifact)


@pytest.mark.parametrize(
    "mode,actions,kept",
    [
        ("auto", False, False),
        ("environment", False, True),
        ("stored-cli", True, True),
    ],
)
def test_download_selects_same_private_auth_as_api(
    transport, tmp_path, monkeypatch, mode, actions, kept
):
    client, _, _, _, _, _, observed, _ = transport
    monkeypatch.setenv("CODESPACES", "true")
    monkeypatch.setenv("COCHEM_PRIVATE_GH_AUTH", mode)
    monkeypatch.setenv("GH_TOKEN", "boundary-fixture-only")
    monkeypatch.setenv("GITHUB_ACTIONS", str(actions).lower())
    client.download("123", tmp_path / "download")
    assert ("GH_TOKEN" in observed[0]) is kept


@pytest.mark.parametrize(
    "field",
    [
        "controller_commit",
        "torq_commit",
        "source_tree_oid",
        "catalog_sha256",
        "catalog_spec_sha256",
        "observed_torq_origin",
    ],
)
def test_tampered_controller_evidence_cannot_change_source(transport, tmp_path, field):
    client, artifact, *_ = transport
    path = artifact / "torq-evidence/controller-and-source.json"
    evidence = read_json(path)
    evidence[field] = "0" * len(evidence[field])
    path.write_bytes(canonical_json(evidence))
    with pytest.raises(ValueError, match="evidence|origin"):
        client.download("123", tmp_path / "download")
    assert not (tmp_path / "download/run-123").exists()


@pytest.mark.parametrize(
    "path",
    [
        "torq-input/request.json",
        "torq-input/approved-plan.json",
        "torq-evidence/torq-source-commit.txt",
    ],
)
def test_original_sidecar_bytes_are_immutable(transport, tmp_path, path):
    client, artifact, *_ = transport
    (artifact / path).write_bytes((artifact / path).read_bytes() + b" ")
    if path.endswith(".txt"):
        (artifact / path).write_text("0" * 40)
    with pytest.raises(ValueError, match="bytes|revision"):
        client.download("123", tmp_path / "download")


def test_native_controller_requires_owned_receipt(transport, tmp_path):
    client, _, receipt_path, *_ = transport
    receipt_path.unlink()
    with pytest.raises(GitHubAccessError, match="owned request"):
        client.download("123", tmp_path / "download")


@pytest.mark.parametrize(
    "field", ["source_commit", "workflow_id", "request_id", "run_id"]
)
def test_receipt_cannot_alias_another_controller_run(transport, tmp_path, field):
    client, _, path, receipt, *_ = transport
    receipt[field] = "0" * 40
    path.write_bytes(canonical_json(receipt))
    with pytest.raises(GitHubAccessError, match="identities"):
        client.download("123", tmp_path / "download")


def test_receipt_scientific_pin_is_rederived_from_real_git_catalog(transport, tmp_path):
    client, _, path, receipt, *_ = transport
    receipt["scientific_source_binding"]["scientific_commit"] = "0" * 40
    path.write_bytes(canonical_json(receipt))
    with pytest.raises(GitHubAccessError, match="binding"):
        client.download("123", tmp_path / "download")


def test_direct_torq_binding_preserves_owning_revision(transport, actual_hf):
    _, _, _, _, api, *_ = transport
    revision = actual_hf[2]
    binding = scientific_source_binding(api, SCIENTIFIC, revision)
    assert binding["route"] == "native-torq"
    assert binding["controller_commit"] == binding["scientific_commit"] == revision


def test_submission_keeps_separate_controller_and_scientific_approval(
    transport, tmp_path
):
    client, _, _, _, _, approved, *_ = transport
    client.state_directory = tmp_path / "new-submission"
    receipt = client.submit(
        approved["plan"]["request"],
        idempotency_key="actual-git",
        approved_plan_sha256=approved["plan"]["plan_sha256"],
        approved_plan=approved,
    )
    assert receipt["run_id"] == "123"
    assert (
        receipt["scientific_source_binding"]["scientific_commit"]
        == approved["source_identity"]["git_commit"]
    )


def test_changed_native_shard_bytes_still_fail_inventory(transport, tmp_path):
    client, artifact, *_ = transport
    (artifact / "torq-results/result/request.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="inventory/hash"):
        client.download("123", tmp_path / "download")


def test_direct_torq_retrieval_keeps_same_commit_guard(transport, tmp_path):
    client, _, receipt_path, receipt, original_api, _, _, run = transport
    client.repository = SCIENTIFIC
    binding = scientific_source_binding(
        original_api,
        SCIENTIFIC,
        receipt["scientific_source_binding"]["scientific_commit"],
    )
    run["head_sha"] = binding["scientific_commit"]
    receipt.update(
        repository=SCIENTIFIC,
        source_commit=run["head_sha"],
        scientific_source_binding=binding,
    )
    receipt_path.write_bytes(canonical_json(receipt))

    def api(endpoint, *, data=None):
        if endpoint.endswith("/actions/runs/123"):
            return run
        return original_api(endpoint, data=data)

    client._api = api
    roots = client.download("123", tmp_path / "download")
    identity = verify_shard(roots[0])["source_identity"]
    assert identity["declared_workflow_commit"] == receipt["source_commit"]


@pytest.mark.parametrize("field", ["declared_workflow_commit", "code_sha256"])
def test_consistently_resealed_wrong_scientific_source_is_rejected(
    transport, tmp_path, field
):
    client, artifact, *_ = transport
    shard = artifact / "torq-results/result"
    result = read_json(shard / "result.json")
    manifest = read_json(shard / "manifest.json")
    result["source_identity"][field] = "0" * len(result["source_identity"][field])
    (shard / "result.json").write_bytes(canonical_json(result))
    manifest["source_identity"] = result["source_identity"]
    manifest["files"] = inventory(shard)
    (shard / "manifest.json").write_bytes(canonical_json(manifest))
    verify_shard(shard)  # Internal integrity cannot replace the submission binding.
    with pytest.raises(GitHubAccessError, match="source commit|implementation"):
        client.download("123", tmp_path / "download")


@pytest.mark.parametrize("corruption", ["truncated", "wrong-tree", "wrong-blob"])
def test_git_source_binding_rejects_incomplete_or_changed_actual_objects(
    transport, corruption
):
    _, _, _, receipt, original_api, *_ = transport

    def api(endpoint):
        value = original_api(endpoint)
        if "/git/trees/" in endpoint:
            if corruption == "truncated":
                value["truncated"] = True
            elif corruption == "wrong-tree":
                value["sha"] = "0" * 40
        elif "/git/blobs/" in endpoint and corruption == "wrong-blob":
            value["content"] = base64.b64encode(b"{}").decode()
        return value

    with pytest.raises(ValueError, match="inventory|bytes"):
        scientific_source_binding(api, CONTROLLER, receipt["source_commit"])


def test_original_controller_sidecars_cannot_be_symlinks(transport, tmp_path):
    client, artifact, *_ = transport
    path = artifact / "torq-input/approved-plan.json"
    original = path.read_bytes()
    outside = tmp_path / "outside-approval.json"
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        client.download("123", tmp_path / "download")
