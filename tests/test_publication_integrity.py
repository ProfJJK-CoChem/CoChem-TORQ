"""Actual rejected requests, real files/processes and genuine HF publication exports."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256

import pytest

from cochem_torq.artifacts import file_digest, inventory, verify_shard
from cochem_torq.domain import CalculationRequest, canonical_json, read_json
from cochem_torq.operations import ArtifactBackpressureError, ArtifactQuotaPolicy
from cochem_torq.publication import export_publication_bundle, verify_publication_bundle
from tests.test_application_contracts import actual_failure_shard


def _hashes(root):
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _reseal_export(root):
    path = root / "publication-manifest.json"
    manifest = read_json(path)
    manifest["files"] = [
        item for item in inventory(root) if item["path"] != "publication-manifest.json"
    ]
    path.write_bytes(canonical_json(manifest) + b"\n")
    return file_digest(path)


def test_genuine_rejected_request_exports_only_failure_evidence(tmp_path):
    source = actual_failure_shard(tmp_path / "actual-rejected-request")
    original = _hashes(source)
    target = tmp_path / "exports" / "failure"
    receipt = export_publication_bundle(source, target)
    verify_publication_bundle(
        target, expected_manifest_sha256=receipt["manifest_sha256"]
    )
    evidence = read_json(target / "publication.json")
    assert evidence["eligibility"] == "failed_evidence_only"
    assert evidence["result_status"] == "failed"
    assert not evidence["publication_validated"]
    assert not evidence["experimental_accuracy_established"]
    assert not evidence["identification_ready"]
    assert evidence["doi"] is None
    assert evidence["native_engine_evidence"] == []
    assert evidence["equilibrium_geometry_bohr"] is None
    assert evidence["calibration"]["calibrated_uncertainty"] is None
    assert evidence["calibration"]["independent_benchmark_evidence"] is None
    assert evidence["citations"] == []  # The actual blocked recipe has no citation.
    assert all(
        stage["payload_reference"] is None
        for stage in evidence["stage_ledger"].values()
    )
    assert _hashes(source) == original == _hashes(target / "shard")
    assert "NEW request" in (target / "README.md").read_text()


def test_real_failure_bundle_is_reproducible_and_preserves_original_files(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    first = export_publication_bundle(source, tmp_path / "exports" / "first")
    second = export_publication_bundle(source, tmp_path / "exports" / "second")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert _hashes(tmp_path / "exports" / "first") == _hashes(
        tmp_path / "exports" / "second"
    )


def test_real_cli_publication_export_uses_the_same_verified_contract(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "cli-publication"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.cli",
            "--json",
            "export",
            str(source),
            "--format",
            "publication",
            "--destination",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    response = json.loads(process.stdout)
    receipt = response["data"]
    assert response["schema_version"] == "cochem.torq.cli-response/1"
    assert response["request_id"] == receipt["request_id"]
    assert receipt["eligibility"] == "failed_evidence_only"
    assert not receipt["publication_validated"]
    verify_publication_bundle(
        target, expected_manifest_sha256=receipt["manifest_sha256"]
    )


def test_export_cannot_overwrite_original_or_prior_publication(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "first"
    export_publication_bundle(source, target)
    original = _hashes(target)
    with pytest.raises(FileExistsError, match="immutable"):
        export_publication_bundle(source, target)
    with pytest.raises(ValueError, match="overlap"):
        export_publication_bundle(source, source / "nested-export")
    assert _hashes(target) == original
    verify_shard(source)


def test_wrong_failure_source_identity_cannot_substitute_current_code(
    tmp_path,
):
    source = actual_failure_shard(tmp_path / "rejected")
    result_path = source / "result.json"
    result = read_json(result_path)
    result["source_identity"]["code_sha256"] = "0" * 64
    result_path.write_bytes(canonical_json(result))
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["source_identity"] = result["source_identity"]
    manifest["files"] = inventory(source)
    manifest_path.write_bytes(canonical_json(manifest))
    # This internally consistent deliberately corrupted failure tests the extra
    # matching-source requirement; it contains no invented engine observations.
    verify_shard(source)
    with pytest.raises(ValueError, match="legitimate installed source matching"):
        export_publication_bundle(source, tmp_path / "exports" / "bad-source")
    assert not (tmp_path / "exports" / "bad-source").exists()


def test_quota_backpressure_rejects_before_publication_and_cleans_owned_staging(
    tmp_path,
):
    source = actual_failure_shard(tmp_path / "rejected")
    store = tmp_path / "exports"
    with pytest.raises(ArtifactBackpressureError, match="byte quota"):
        export_publication_bundle(
            source,
            store / "too-large",
            policy=ArtifactQuotaPolicy(max_owned_bytes=1024, minimum_free_bytes=0),
        )
    assert not (store / "too-large").exists()
    assert not list(store.glob(".torq-publication-*"))
    assert not list(store.glob(".torq-reservations/*"))


def test_actual_file_count_quota_is_enforced_for_complete_bundle(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    store = tmp_path / "exports"
    with pytest.raises(ArtifactBackpressureError, match="file-count"):
        export_publication_bundle(
            source,
            store / "too-many",
            policy=ArtifactQuotaPolicy(max_files=3, minimum_free_bytes=0),
        )
    assert not (store / "too-many").exists()
    assert not list(store.glob(".torq-publication-*"))


def test_symlinks_and_untrusted_manifest_digest_are_rejected(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        export_publication_bundle(link, tmp_path / "exports" / "bad-link")
    target = tmp_path / "exports" / "valid"
    receipt = export_publication_bundle(source, target)
    assert receipt["manifest_sha256"] != "0" * 64
    with pytest.raises(ValueError, match="independently retained"):
        verify_publication_bundle(target, expected_manifest_sha256="0" * 64)


def test_resealed_export_cannot_promote_failure_into_publication_qualification(
    tmp_path,
):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "valid"
    export_publication_bundle(source, target)
    path = target / "publication.json"
    evidence = read_json(path)
    evidence["publication_validated"] = True
    path.write_bytes(canonical_json(evidence))
    with pytest.raises(ValueError, match="summary differs"):
        verify_publication_bundle(
            target, expected_manifest_sha256=_reseal_export(target)
        )


def test_resealed_manifest_cannot_promote_publication_eligibility(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "valid"
    export_publication_bundle(source, target)
    path = target / "publication-manifest.json"
    manifest = read_json(path)
    manifest["publication_validated"] = True
    path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="eligibility"):
        verify_publication_bundle(target, expected_manifest_sha256=file_digest(path))


def test_changed_packaged_analysis_source_is_detected(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "valid"
    export_publication_bundle(source, target)
    path = target / "analysis/source/cochem_torq/spectroscopy/harmonic.py"
    with path.open("ab") as stream:
        stream.write(b"\n# Deliberate changed-source rejection case.\n")
    with pytest.raises(ValueError, match="sources do not verify"):
        verify_publication_bundle(
            target, expected_manifest_sha256=_reseal_export(target)
        )


def test_unrecorded_foreign_schema_manifest_cannot_hide_from_inventory(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "valid"
    receipt = export_publication_bundle(source, target)
    (target / "manifest.json").write_bytes(
        b'{"purpose":"deliberate unrecorded-file rejection case"}'
    )
    with pytest.raises(ValueError, match="inventory/hash mismatch"):
        verify_publication_bundle(
            target, expected_manifest_sha256=receipt["manifest_sha256"]
        )


@pytest.mark.parametrize("process_count", [2, 4])
def test_real_concurrent_export_processes_publish_only_one_immutable_target(
    tmp_path, process_count
):
    source = actual_failure_shard(tmp_path / "rejected")
    target = tmp_path / "exports" / "same-destination"
    code = """
import sys
from cochem_torq.publication import export_publication_bundle
try:
    export_publication_bundle(sys.argv[1], sys.argv[2])
except FileExistsError:
    raise SystemExit(17)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(source), str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(process_count)
    ]
    try:
        outputs = [process.communicate(timeout=30) for process in processes]
        assert sorted(process.returncode for process in processes) == [
            0, *([17] * (process_count - 1))
        ], outputs
        verify_publication_bundle(
            target,
            expected_manifest_sha256=file_digest(target / "publication-manifest.json"),
        )
        assert not list(target.parent.glob(".torq-publication-*"))
        assert not list(target.parent.glob(".torq-reservations/*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_real_concurrent_exports_observe_complete_shared_store_quota(tmp_path):
    source = actual_failure_shard(tmp_path / "rejected")
    seed = export_publication_bundle(source, tmp_path / "measurement" / "bundle")
    store = tmp_path / "bounded-exports"
    bound = seed["size_bytes"] + 4096  # One real bundle plus bounded control files.
    code = """
import sys
from cochem_torq.operations import ArtifactBackpressureError, ArtifactQuotaPolicy
from cochem_torq.publication import export_publication_bundle
try:
    export_publication_bundle(sys.argv[1], sys.argv[2],
        policy=ArtifactQuotaPolicy(
            max_owned_bytes=int(sys.argv[3]), minimum_free_bytes=0))
except ArtifactBackpressureError:
    raise SystemExit(23)
"""
    targets = [store / f"publication-{number}" for number in range(4)]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(source), str(target), str(bound)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for target in targets
    ]
    try:
        outputs = [process.communicate(timeout=30) for process in processes]
        assert sorted(process.returncode for process in processes) == [
            0, 23, 23, 23
        ], outputs
        published = [target for target in targets if target.exists()]
        assert len(published) == 1
        verify_publication_bundle(
            published[0], expected_manifest_sha256=seed["manifest_sha256"]
        )
        observed_bytes = sum(
            path.stat().st_size for path in store.rglob("*") if path.is_file()
        )
        assert observed_bytes <= bound
        assert not list(store.glob(".torq-publication-*"))
        assert not list(store.glob(".torq-reservations/*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(scope="module")
def genuine_hf_shard(tmp_path_factory):
    import pyscf

    from cochem_torq.application import execute_request

    assert pyscf.__version__ == "2.14.0"
    root = tmp_path_factory.mktemp("genuine-publication-hf")
    request = CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["hydrogen-one", "hydrogen-two"],
                "isotopes": [1, 1],
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry", "harmonic", "equilibrium_constants"],
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
        }
    )
    result = execute_request(request, root / "shard")
    assert result["status"] == "complete", result["errors"]
    verify_shard(root / "shard")
    return root / "shard"


@pytest.mark.real_engine
def test_genuine_hf_native_sources_and_verbatim_citations_are_retained(
    genuine_hf_shard, tmp_path
):
    target = tmp_path / "exports" / "actual-hf"
    receipt = export_publication_bundle(genuine_hf_shard, target)
    verify_publication_bundle(
        target, expected_manifest_sha256=receipt["manifest_sha256"]
    )
    result = read_json(genuine_hf_shard / "result.json")
    evidence = read_json(target / "publication.json")
    assert evidence["eligibility"] == "exploratory"
    assert (
        not evidence["publication_validated"] and not evidence["identification_ready"]
    )
    assert evidence["resolved_isotopes"] == result["resolved_isotopes"]
    assert (
        evidence["molecule"]["charge"] == 0
        and evidence["molecule"]["multiplicity"] == 1
    )
    assert evidence["molecule"]["atom_ids"] == ["hydrogen-one", "hydrogen-two"]
    assert evidence["citations"][0]["value"] == result["recipe"]["citation"]
    assert len(evidence["native_engine_evidence"]) == 2
    assert all(
        record["engine_version"] == "2.14.0"
        for record in evidence["native_engine_evidence"]
    )
    assert all(
        record["scf"]["converged"] for record in evidence["native_engine_evidence"]
    )
    assert all(
        record["engine_installation_sha256"]
        for record in evidence["native_engine_evidence"]
    )
    assert list((target / "shard").rglob("*.chk"))
    assert _hashes(genuine_hf_shard) == _hashes(target / "shard")


@pytest.mark.real_engine
def test_native_checkpoint_corruption_cannot_be_exported(genuine_hf_shard, tmp_path):
    damaged = tmp_path / "deliberately-damaged-native"
    shutil.copytree(genuine_hf_shard, damaged)
    checkpoint = next(damaged.rglob("*.chk"))
    original_sha256 = file_digest(checkpoint)
    with checkpoint.open("ab") as stream:
        stream.write(b"deliberate damaged-byte integrity check")
    assert file_digest(checkpoint) != original_sha256
    with pytest.raises(ValueError, match="inventory/hash mismatch"):
        export_publication_bundle(damaged, tmp_path / "exports" / "bad-native")


@pytest.mark.real_engine
def test_genuine_calculation_export_is_deterministic(genuine_hf_shard, tmp_path):
    first = export_publication_bundle(genuine_hf_shard, tmp_path / "exports" / "first")
    second = export_publication_bundle(
        genuine_hf_shard, tmp_path / "exports" / "second"
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (
        sha256((tmp_path / "exports/first/publication.json").read_bytes()).digest()
        == sha256((tmp_path / "exports/second/publication.json").read_bytes()).digest()
    )
