"""Genuine artifact/process admission and real-engine shard backup checks."""

from __future__ import annotations

import multiprocessing as mp
from hashlib import sha256

import pytest

from cochem_torq.artifacts import file_digest, verify_shard
from cochem_torq.domain import CalculationRequest, canonical_json, read_json
from cochem_torq.operations import (
    ArtifactBackpressureError,
    ArtifactQuotaPolicy,
    admit_artifact_work,
    artifact_usage,
    backup_shard,
    check_artifact_quota,
    enforce_artifact_budget,
    reconcile_artifact_reservations,
    restore_shard,
    rollback_shard,
    verify_backup,
)


def _reservation_owner(root, connection):
    with admit_artifact_work(
        root,
        incoming_bytes=6000,
        policy=ArtifactQuotaPolicy(max_owned_bytes=10_000, minimum_free_bytes=0),
    ):
        connection.send("real local reservation active")
        connection.recv()


def test_actual_file_bytes_and_budget_exhaustion(tmp_path):
    root = tmp_path / "owned-artifacts"
    root.mkdir()
    (root / "arithmetic.bin").write_bytes(bytes(range(256)))
    assert artifact_usage(root)["owned_bytes"] == 256
    with pytest.raises(ArtifactBackpressureError, match="byte quota"):
        check_artifact_quota(
            root,
            incoming_bytes=257,
            policy=ArtifactQuotaPolicy(max_owned_bytes=512, minimum_free_bytes=0),
        )
    with pytest.raises(ArtifactBackpressureError, match="actual artifact"):
        enforce_artifact_budget(root, baseline_bytes=128, max_growth_bytes=127)
    assert (
        enforce_artifact_budget(root, baseline_bytes=128, max_growth_bytes=128)[
            "owned_bytes"
        ]
        == 256
    )


def test_actual_free_space_reserve_rejects_before_work(tmp_path):
    actual_free = artifact_usage(tmp_path)["free_bytes"]
    with pytest.raises(ArtifactBackpressureError, match="filesystem free"):
        check_artifact_quota(
            tmp_path,
            policy=ArtifactQuotaPolicy(
                minimum_free_bytes=actual_free + 1024 * 1024 * 1024
            ),
        )
    assert not list(tmp_path.glob(".torq-reservations/*"))


def test_real_process_reservation_prevents_over_admission_and_recovers_dead_owner(
    tmp_path,
):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_reservation_owner, args=(tmp_path, child))
    process.start()
    try:
        assert parent.poll(15) and parent.recv() == "real local reservation active"
        with pytest.raises(ArtifactBackpressureError, match="byte quota"):
            with admit_artifact_work(
                tmp_path,
                incoming_bytes=6000,
                policy=ArtifactQuotaPolicy(
                    max_owned_bytes=10_000, minimum_free_bytes=0
                ),
            ):
                pytest.fail(
                    "Actual concurrent output budgets must not overbook this store."
                )
        assert (
            len(reconcile_artifact_reservations(tmp_path)["retained_reservations"]) == 1
        )
        process.kill()
        process.join(10)
        assert process.exitcode != 0
        # The killed owner has not deleted its reservation or claimed completion.
        assert len(list((tmp_path / ".torq-reservations").iterdir())) == 1
        recovered = reconcile_artifact_reservations(tmp_path)
        assert len(recovered["removed_dead_local_reservations"]) == 1
        assert not recovered["retained_reservations"]
        with admit_artifact_work(
            tmp_path,
            incoming_bytes=6000,
            policy=ArtifactQuotaPolicy(max_owned_bytes=10_000, minimum_free_bytes=0),
        ):
            assert (
                check_artifact_quota(
                    tmp_path,
                    policy=ArtifactQuotaPolicy(
                        max_owned_bytes=10_000, minimum_free_bytes=0
                    ),
                )["active_reserved_bytes"]
                == 6000
            )
        assert not list((tmp_path / ".torq-reservations").iterdir())
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
        child.close()


def test_aborted_owner_releases_reservation_without_work_claim(tmp_path):
    with pytest.raises(RuntimeError, match="actual owner interruption"):
        with admit_artifact_work(tmp_path, incoming_bytes=128):
            raise RuntimeError("actual owner interruption")
    assert not list((tmp_path / ".torq-reservations").iterdir())


def test_file_count_and_symlinks_fail_closed(tmp_path):
    (tmp_path / "one").write_bytes(b"actual accounting data")
    (tmp_path / "two").write_bytes(b"actual accounting data")
    with pytest.raises(ArtifactBackpressureError, match="file-count"):
        artifact_usage(tmp_path, max_files=1)
    (tmp_path / "link").symlink_to(tmp_path / "one")
    with pytest.raises(ValueError, match="symlinks"):
        artifact_usage(tmp_path)


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_noninteger_or_negative_reservation_cannot_enter_store(tmp_path, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        with admit_artifact_work(tmp_path, incoming_bytes=value):
            pytest.fail("Invalid accounting values cannot reserve storage.")
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def genuine_h2_shard(tmp_path_factory):
    from cochem_torq.application import execute_request

    root = tmp_path_factory.mktemp("genuine-h2-backup")
    request = CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
                "charge": 0,
                "multiplicity": 1,
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry", "harmonic", "equilibrium_constants"],
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
        }
    )
    result = execute_request(request, root / "shard")
    assert result["status"] == "complete"
    verify_shard(root / "shard")
    return root / "shard"


@pytest.mark.real_engine
def test_actual_engine_shard_backup_restore_and_prior_schema_rollback(
    genuine_h2_shard, tmp_path
):
    source = genuine_h2_shard
    original = {
        p.relative_to(source).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    receipt = backup_shard(source, tmp_path / "backup")
    expected = file_digest(source / "manifest.json")
    assert receipt["source_manifest_sha256"] == expected
    assert (
        verify_backup(tmp_path / "backup", expected_manifest_sha256=expected)[
            "scientific_qualification"
        ]
        is False
    )
    restored = restore_shard(
        tmp_path / "backup", tmp_path / "restored", expected_manifest_sha256=expected
    )
    previous = rollback_shard(
        tmp_path / "backup", tmp_path / "prior-v1", expected_manifest_sha256=expected
    )
    for root in (source, restored, previous):
        assert {
            p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        } == original
        assert verify_shard(root)["schema_version"] == "cochem.torq.shard/1"


@pytest.mark.real_engine
def test_existing_results_and_backups_are_never_overwritten(genuine_h2_shard, tmp_path):
    receipt = backup_shard(genuine_h2_shard, tmp_path / "backup")
    expected = receipt["source_manifest_sha256"]
    with pytest.raises(FileExistsError):
        backup_shard(genuine_h2_shard, tmp_path / "backup")
    original = file_digest(genuine_h2_shard / "manifest.json")
    with pytest.raises(FileExistsError):
        restore_shard(
            tmp_path / "backup", genuine_h2_shard, expected_manifest_sha256=expected
        )
    assert file_digest(genuine_h2_shard / "manifest.json") == original


@pytest.mark.real_engine
def test_changed_genuine_native_bytes_rejected_without_publication(
    genuine_h2_shard, tmp_path
):
    receipt = backup_shard(genuine_h2_shard, tmp_path / "backup")
    native = next((tmp_path / "backup/shard").rglob("*.chk"))
    with native.open("ab") as stream:
        stream.write(b"explicit integrity-corruption check")
    with pytest.raises(ValueError, match="inventory/hash"):
        restore_shard(
            tmp_path / "backup",
            tmp_path / "cannot-publish",
            expected_manifest_sha256=receipt["source_manifest_sha256"],
        )
    assert not (tmp_path / "cannot-publish").exists()
    verify_shard(genuine_h2_shard)


@pytest.mark.real_engine
def test_unsupported_schema_never_silently_migrates_original(
    genuine_h2_shard, tmp_path
):
    receipt = backup_shard(genuine_h2_shard, tmp_path / "backup")
    metadata = read_json(tmp_path / "backup/backup.json")
    metadata["shard_schema_version"] = "cochem.torq.shard/2"
    (tmp_path / "backup/backup.json").write_bytes(canonical_json(metadata))
    with pytest.raises(ValueError, match="explicit versioned migration"):
        rollback_shard(
            tmp_path / "backup",
            tmp_path / "cannot-migrate",
            expected_manifest_sha256=receipt["source_manifest_sha256"],
        )
    assert not (tmp_path / "cannot-migrate").exists()
    assert verify_shard(genuine_h2_shard)["schema_version"] == "cochem.torq.shard/1"


@pytest.mark.real_engine
def test_retained_manifest_identity_is_required(genuine_h2_shard, tmp_path):
    backup_shard(genuine_h2_shard, tmp_path / "backup")
    with pytest.raises(ValueError, match="independently retained digest"):
        restore_shard(
            tmp_path / "backup",
            tmp_path / "wrong-request",
            expected_manifest_sha256="0" * 64,
        )
    assert not (tmp_path / "wrong-request").exists()
