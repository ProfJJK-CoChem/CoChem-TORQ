"""Genuine HDF5, SQLite ownership and abrupt-process-crash validation.

Samples are actual evaluations of a declared dimensionless quadratic function.
They are never used as quantum observations or molecular-accuracy evidence.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pytest

from cochem.orchestration.campaign import (
    Allocation,
    AuthorityError,
    CampaignBudget,
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
    RevisionConflict,
)
from cochem.storage.fenced_pes import (
    PESIdentity,
    merge_pes_shards,
    merge_task_payload,
    publish_pes_shard,
    recover_pes_publication,
    shard_task_payload,
    verify_committed_pes,
    verify_pes_artifact,
)

ACTOR = f"observed-local-user:{os.getuid()}"


def identity():
    return PESIdentity.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "charge": 0,
                "multiplicity": 1,
                "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                "atom_ids": ["a", "b"],
            },
            "recipe": {
                "id": "declared_quadratic_mathcheck",
                "definition": "sum(x**2)",
                "qualification": "mathematical validation only",
            },
            "source_identity": {
                "source_file_sha256": sha256(Path(__file__).read_bytes()).hexdigest()
            },
            "evidence_class": "mathematical_validation",
            "coordinate_unit": "dimensionless",
            "energy_unit": "dimensionless",
            "engine": "numpy_quadratic_evaluation",
        }
    )


def campaign(tmp_path):
    coordinator = CampaignCoordinator(tmp_path / "campaign.sqlite")
    receipt = coordinator.create_campaign(
        {"purpose": "genuine HDF5 ownership and mathematical sample checks"},
        CampaignBudget(
            max_wall_seconds=400.0,
            max_cpu_core_seconds=400.0,
            max_tasks=8,
            max_concurrent_workers=1,
            max_cpu_cores=1,
            max_memory_mb=256,
            max_scratch_mb=32,
            expires_at_unix=time.time() + 180,
            allowed_engines=[identity().engine],
            allowed_recipes=[identity().recipe["id"]],
        ),
        actor=ACTOR,
    )
    return coordinator, receipt


def validating(coordinator, receipt, payload, *, wall_seconds=20.0, lease_seconds=90.0):
    sample_identity = identity()
    task = coordinator.register_task(
        receipt["campaign_id"],
        payload,
        engine=sample_identity.engine,
        recipe=sample_identity.recipe["id"],
        plan_sha256=receipt["plan_sha256"],
        authority=receipt["authority"],
        actor=ACTOR,
    )
    attempt = coordinator.new_attempt(
        task["task_id"], authority=receipt["authority"], actor=ACTOR
    )
    attempt = coordinator.transition(
        attempt["id"],
        attempt["revision"],
        "validated",
        authority=receipt["authority"],
        actor=ACTOR,
        reason="Validated actual array/schema operation",
    )
    attempt = coordinator.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(wall_seconds=wall_seconds, cores=1, memory_mb=128, scratch_mb=8),
        authority=receipt["authority"],
        actor=ACTOR,
    )
    attempt = coordinator.lease(
        attempt["id"],
        attempt["revision"],
        pid=os.getpid(),
        authority=receipt["authority"],
        actor=ACTOR,
        heartbeat_seconds=lease_seconds / 4,
        lease_seconds=lease_seconds,
    )
    token, generation = attempt["lease_token"], attempt["lease_generation"]
    for state in ("collecting", "validating"):
        attempt = coordinator.worker_transition(
            attempt["id"],
            attempt["revision"],
            state,
            lease_token=token,
            lease_generation=generation,
            actor=ACTOR,
            reason=f"Actual operation entered {state}",
        )
    return {**attempt, "lease_token": token, "lease_generation": generation}


def measurements(start):
    wall, cpu = start
    return MeasuredUsage(
        wall_seconds=time.monotonic() - wall,
        cpu_core_seconds=time.process_time() - cpu,
        measurement_source="actual monotonic/process CPU differences",
    )


def credentials(attempt):
    return {
        "lease_token": attempt["lease_token"],
        "lease_generation": attempt["lease_generation"],
    }


def source(
    coordinator,
    receipt,
    root,
    *,
    offset=0.0,
    short_allocation=False,
    lease_seconds=90.0,
):
    sample_identity = identity()
    coordinates = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 + offset]]], dtype=np.float64
    )
    energy = np.sum(coordinates**2, axis=(1, 2))
    ids = [str(uuid4())]
    attempt = validating(
        coordinator,
        receipt,
        shard_task_payload(sample_identity, ids),
        wall_seconds=0.05 if short_allocation else 20.0,
        lease_seconds=lease_seconds,
    )
    started = (time.monotonic(), time.process_time())
    if short_allocation:
        time.sleep(0.08)
    publish_pes_shard(
        coordinator,
        attempt["id"],
        attempt["revision"],
        identity=sample_identity,
        coordinates=coordinates,
        energies=energy,
        frame_ids=ids,
        destination=root,
        usage=measurements(started),
        actor=ACTOR,
        **credentials(attempt),
    )
    return attempt, coordinates, energy


def merger(coordinator, receipt, sources):
    return validating(coordinator, receipt, merge_task_payload(identity(), sources))


def test_real_independent_shards_merge_under_one_current_coordinator(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        _, first_coordinates, first_energies = source(
            coordinator, approved, tmp_path / "first"
        )
        _, second_coordinates, second_energies = source(
            coordinator, approved, tmp_path / "second", offset=0.2
        )
        originals = [
            verify_committed_pes(coordinator, tmp_path / name)
            for name in ("first", "second")
        ]
        sources = [tmp_path / "first", tmp_path / "second"]
        attempt = merger(coordinator, approved, sources)
        started = (time.monotonic(), time.process_time())
        destination = merge_pes_shards(
            coordinator,
            attempt["id"],
            attempt["revision"],
            identity=identity(),
            sources=sources,
            destination=tmp_path / "merged-v1",
            usage=measurements(started),
            actor=ACTOR,
            **credentials(attempt),
        )
        manifest = verify_committed_pes(coordinator, destination)
        assert manifest["kind"] == "merge" and manifest["sample_count"] == 2
        assert len(manifest["sources"]) == 2
        assert not manifest["scientific_accuracy_established"]
        with h5py.File(destination / "samples.h5", "r", locking=True) as store:
            np.testing.assert_array_equal(
                store["coordinates"][:],
                np.concatenate([first_coordinates, second_coordinates]),
            )
            np.testing.assert_array_equal(
                store["energies"][:], np.concatenate([first_energies, second_energies])
            )
            assert store.attrs["energy_unit"] == "dimensionless"
            assert int(store["committed_frames"][0]) == 2
        assert [
            verify_committed_pes(coordinator, path) for path in sources
        ] == originals
    finally:
        coordinator.close()


@pytest.mark.parametrize("fault", ["token", "generation", "revision", "revoked"])
def test_invalid_owner_blocks_real_hdf5_publication_before_any_target(tmp_path, fault):
    coordinator, approved = campaign(tmp_path)
    ids = [str(uuid4())]
    attempt = validating(coordinator, approved, shard_task_payload(identity(), ids))
    arguments = credentials(attempt)
    revision = attempt["revision"]
    if fault == "token":
        arguments["lease_token"] = "invalid-not-issued-authority"
    elif fault == "generation":
        arguments["lease_generation"] += 1
    elif fault == "revision":
        revision -= 1
    else:
        coordinator.revoke(
            approved["campaign_id"],
            authority=approved["authority"],
            actor=ACTOR,
            reason="Actual explicit withdrawal",
        )
    destination = tmp_path / "blocked"
    try:
        with pytest.raises((AuthorityError, CampaignError, RevisionConflict)):
            publish_pes_shard(
                coordinator,
                attempt["id"],
                revision,
                identity=identity(),
                coordinates=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]),
                energies=np.array([1.0]),
                frame_ids=ids,
                destination=destination,
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **arguments,
            )
        assert not destination.exists()
        assert not list(tmp_path.glob(".pes-*"))
    finally:
        coordinator.close()


def test_expired_completed_source_lease_is_not_required_for_merger_admission(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        attempt, _, _ = source(
            coordinator, approved, tmp_path / "source", lease_seconds=0.35
        )
        expiry = coordinator.attempt(attempt["id"])["lease_expires"]
        # The historical lease is already invalidated by successful publication.
        with pytest.raises(AuthorityError):
            coordinator.heartbeat(attempt["id"], **credentials(attempt))
        assert isinstance(expiry, float)
        # Actually cross the recorded historical deadline; no patched clock.
        time.sleep(max(0.0, expiry - time.time()) + 0.02)
        assert time.time() >= expiry
        merge_attempt = merger(coordinator, approved, [tmp_path / "source"])
        target = merge_pes_shards(
            coordinator,
            merge_attempt["id"],
            merge_attempt["revision"],
            identity=identity(),
            sources=[tmp_path / "source"],
            destination=tmp_path / "new-version",
            usage=measurements((time.monotonic(), time.process_time())),
            actor=ACTOR,
            **credentials(merge_attempt),
        )
        assert verify_committed_pes(coordinator, target)["sample_count"] == 1
    finally:
        coordinator.close()


def test_partial_source_is_rejected_without_fabricated_repair(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        attempt, _, _ = source(
            coordinator, approved, tmp_path / "partial", short_allocation=True
        )
        assert coordinator.attempt(attempt["id"])["state"] == "partial"
        actual_source = verify_pes_artifact(tmp_path / "partial")
        merged = merger(coordinator, approved, [tmp_path / "partial"])
        with pytest.raises(AuthorityError, match="committed result"):
            merge_pes_shards(
                coordinator,
                merged["id"],
                merged["revision"],
                identity=identity(),
                sources=[tmp_path / "partial"],
                destination=tmp_path / "invalid-merge",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(merged),
            )
        assert not (tmp_path / "invalid-merge").exists()
        assert verify_pes_artifact(tmp_path / "partial") == actual_source
    finally:
        coordinator.close()


def test_duplicate_samples_and_source_attempts_are_rejected(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        duplicated = [tmp_path / "source", tmp_path / "source"]
        attempt = merger(coordinator, approved, duplicated)
        with pytest.raises(ValueError, match="unique"):
            merge_pes_shards(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=identity(),
                sources=duplicated,
                destination=tmp_path / "duplicate",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(attempt),
            )
        assert not (tmp_path / "duplicate").exists()
    finally:
        coordinator.close()


@pytest.mark.parametrize("change", ["shape", "nonfinite", "complex", "missing_energy"])
def test_incomplete_or_unavailable_sample_arrays_never_create_a_shard(tmp_path, change):
    coordinator, approved = campaign(tmp_path)
    ids = [str(uuid4())]
    attempt = validating(coordinator, approved, shard_task_payload(identity(), ids))
    coordinates = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    energies = np.sum(coordinates**2, axis=(1, 2))
    if change == "shape":
        coordinates = coordinates[:, :1]
    elif change == "nonfinite":
        energies[0] = np.nan
    elif change == "complex":
        coordinates = coordinates.astype(complex) + 1j
    else:
        energies = np.array([])
    try:
        with pytest.raises(ValueError):
            publish_pes_shard(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=identity(),
                coordinates=coordinates,
                energies=energies,
                frame_ids=ids,
                destination=tmp_path / "incomplete",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(attempt),
            )
        assert not (tmp_path / "incomplete").exists()
    finally:
        coordinator.close()


def test_real_hdf5_corruption_is_rejected_by_sealed_hash(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        with h5py.File(tmp_path / "source" / "samples.h5", "r+", locking=True) as store:
            store["energies"][0] += 1.0
        with pytest.raises(ValueError, match="hash differs"):
            verify_pes_artifact(tmp_path / "source")
    finally:
        coordinator.close()


@pytest.mark.parametrize("field", ["atom_order", "spin", "recipe", "source", "units"])
def test_incompatible_atom_method_source_and_units_cannot_merge(tmp_path, field):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        actual_identity = identity().record()
        if field == "atom_order":
            actual_identity["molecule"]["atom_ids"] = ["b", "a"]
        elif field == "spin":
            actual_identity["molecule"]["multiplicity"] = 3
        elif field == "recipe":
            actual_identity["recipe"] = {"id": "distinct_unqualified_recipe"}
        elif field == "source":
            actual_identity["source_identity"] = {
                "source_file_sha256": sha256(
                    Path(sys.modules[PESIdentity.__module__].__file__).read_bytes()
                ).hexdigest()
            }
        else:
            actual_identity.update(
                evidence_class="supplied_data",
                coordinate_unit="bohr",
                energy_unit="hartree",
            )
        requested = PESIdentity.model_validate(actual_identity)
        attempt = merger(coordinator, approved, [tmp_path / "source"])
        with pytest.raises(ValueError, match="identities differ"):
            merge_pes_shards(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=requested,
                sources=[tmp_path / "source"],
                destination=tmp_path / "incompatible",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(attempt),
            )
        assert not (tmp_path / "incompatible").exists()
    finally:
        coordinator.close()


def test_incomplete_commit_marker_rejected_even_after_actual_checksum_reseal(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        root = tmp_path / "source"
        with h5py.File(root / "samples.h5", "r+", locking=True) as store:
            store["committed_frames"][0] = 2
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["files"][0].update(
            sha256=sha256((root / "samples.h5").read_bytes()).hexdigest(),
            size_bytes=(root / "samples.h5").stat().st_size,
        )
        # Actual intentionally corrupted files test schema validation, never
        # become accepted observations or evidence of molecular correctness.
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="incomplete committed prefix"):
            verify_pes_artifact(root)
    finally:
        coordinator.close()


def test_identical_copied_bytes_without_the_committed_path_are_not_authoritative(
    tmp_path,
):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        shutil.copytree(tmp_path / "source", tmp_path / "copy")
        assert verify_pes_artifact(tmp_path / "copy")
        with pytest.raises(CampaignError, match="commit receipt"):
            verify_committed_pes(coordinator, tmp_path / "copy")
        attempt = merger(coordinator, approved, [tmp_path / "copy"])
        with pytest.raises(CampaignError, match="exact current committed artifact"):
            merge_pes_shards(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=identity(),
                sources=[tmp_path / "copy"],
                destination=tmp_path / "blocked",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(attempt),
            )
        assert not (tmp_path / "blocked").exists()
    finally:
        coordinator.close()


@pytest.mark.parametrize("policy", ["FALSE", "0", "BEST_EFFORT"])
def test_real_process_rejects_hdf5_policy_that_disables_mandatory_locking(
    tmp_path, policy
):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        script = """
import sys
from cochem.orchestration.campaign import CampaignError
from cochem.storage.fenced_pes import verify_pes_artifact
try:
    verify_pes_artifact(sys.argv[1])
except CampaignError as error:
    print(str(error))
    sys.exit(3)
sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "source")],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "HDF5_USE_FILE_LOCKING": policy},
        )
        assert result.returncode == 3, result.stderr
        assert "mandatory HDF5 file locking" in result.stdout
    finally:
        coordinator.close()


def test_new_artifacts_never_overwrite_existing_immutable_paths(tmp_path):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        manifest = verify_committed_pes(coordinator, tmp_path / "source")
        attempt = merger(coordinator, approved, [tmp_path / "source"])
        with pytest.raises(FileExistsError):
            merge_pes_shards(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=identity(),
                sources=[tmp_path / "source"],
                destination=tmp_path / "source",
                usage=measurements((time.monotonic(), time.process_time())),
                actor=ACTOR,
                **credentials(attempt),
            )
        assert verify_committed_pes(coordinator, tmp_path / "source") == manifest
        assert coordinator.attempt(attempt["id"])["state"] == "validating"
    finally:
        coordinator.close()


def test_two_real_processes_cannot_publish_two_results_for_one_attempt(tmp_path):
    coordinator, approved = campaign(tmp_path)
    processes = []
    try:
        source(coordinator, approved, tmp_path / "source")
        attempt = merger(coordinator, approved, [tmp_path / "source"])
        target = tmp_path / "one-publication"
        barrier = tmp_path / "actual-start-barrier"
        script = r"""
import contextlib,json,sys,time
from pathlib import Path
from cochem.orchestration.campaign import (
    CampaignCoordinator,CampaignError,MeasuredUsage)
from cochem.storage.fenced_pes import PESIdentity,verify_pes_artifact,merge_pes_shards
arguments=json.load(sys.stdin)
print('ready',flush=True)
deadline=time.monotonic()+10
while not Path(arguments['barrier']).exists():
    if time.monotonic()>=deadline:
        sys.exit(8)
    time.sleep(0.01)
started_wall=time.monotonic(); started_cpu=time.process_time()
with contextlib.closing(CampaignCoordinator(arguments['database'])) as coordinator:
    identity=PESIdentity.model_validate(verify_pes_artifact(arguments['source'])['identity'])
    try:
        merge_pes_shards(coordinator,arguments['attempt_id'],arguments['revision'],
            identity=identity,sources=[arguments['source']],destination=arguments['target'],
            lease_token=arguments['token'],lease_generation=arguments['generation'],
            usage=MeasuredUsage(wall_seconds=time.monotonic()-started_wall,
                cpu_core_seconds=time.process_time()-started_cpu,
                measurement_source='actual child prepublication measurements'),
            actor='actual concurrent I/O process')
    except CampaignError:
        print('rejected-current-attempt',flush=True)
        sys.exit(3)
    print('published',flush=True)
"""
        payload = json.dumps(
            {
                "database": str(tmp_path / "campaign.sqlite"),
                "source": str(tmp_path / "source"),
                "target": str(target),
                "barrier": str(barrier),
                "attempt_id": attempt["id"],
                "revision": attempt["revision"],
                "token": attempt["lease_token"],
                "generation": attempt["lease_generation"],
            }
        )
        for _ in range(2):
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            process.stdin.write(payload)
            process.stdin.close()
            process.stdin = None
        assert [process.stdout.readline().strip() for process in processes] == [
            "ready",
            "ready",
        ]
        barrier.write_text("Actual parent released both child I/O processes")
        output = [process.communicate(timeout=20) for process in processes]
        assert sorted(process.returncode for process in processes) == [0, 3], output
        assert sorted(stdout.strip() for stdout, _ in output) == [
            "published",
            "rejected-current-attempt",
        ]
        assert verify_committed_pes(coordinator, target)["sample_count"] == 1
        assert coordinator.attempt(attempt["id"])["revision"] == attempt["revision"] + 1
        assert not list(tmp_path.glob(".pes-merge-*"))
        assert not list(tmp_path.glob(".pes-inputs-*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        coordinator.close()


@pytest.mark.parametrize("ownership", ["current", "expired", "revoked"])
def test_abrupt_real_process_exit_after_rename_recovers_without_editing_hdf5(
    tmp_path, ownership
):
    coordinator, approved = campaign(tmp_path)
    try:
        source(coordinator, approved, tmp_path / "source")
        sources = [tmp_path / "source"]
        attempt = validating(
            coordinator,
            approved,
            merge_task_payload(identity(), sources),
            lease_seconds=2.0 if ownership == "expired" else 90.0,
        )
        target = tmp_path / "orphaned-merge"
        script = r"""
import json,os,sys,tempfile,time,contextlib
from pathlib import Path
import h5py,numpy as np
from cochem.orchestration.campaign import CampaignCoordinator,MeasuredUsage
from cochem.storage.fenced_pes import (PESIdentity,verify_pes_artifact,
    merge_task_payload,_owner,_write,_source_ref,_admit,_receipt,_sync_directory)
arguments=json.load(sys.stdin)
started_wall=time.monotonic(); started_cpu=time.process_time()
with contextlib.closing(CampaignCoordinator(arguments['database'])) as coordinator:
    source=Path(arguments['source']); target=Path(arguments['target'])
    manifest=verify_pes_artifact(source)
    identity=PESIdentity.model_validate(manifest['identity'])
    owner=_owner(coordinator,arguments['attempt_id'],arguments['revision'],
        arguments['token'],arguments['generation'],merge_task_payload(identity,[source]))
    with h5py.File(source/'samples.h5','r',locking=True) as store:
        coordinates=store['coordinates'][:]; energies=store['energies'][:]
    staging=Path(tempfile.mkdtemp(prefix='.actual-interrupted-merge-',dir=target.parent))
    merged=_write(staging,identity,coordinates,energies,manifest['frame_ids'],owner,'merge',
        [_source_ref(source,manifest)])
    receipt=_receipt(staging,merged); receipt['artifact_path']=str(target)
    def actual_crash():
        _admit(coordinator,[source],[manifest])
        os.rename(staging,target); _sync_directory(target.parent)
        os._exit(71)
    coordinator.publish(arguments['attempt_id'],arguments['revision'],receipt,
        MeasuredUsage(wall_seconds=time.monotonic()-started_wall,
                      cpu_core_seconds=time.process_time()-started_cpu,
                      measurement_source='actual child prepublication measurements'),
        actual_crash,state='succeeded',lease_token=arguments['token'],
        lease_generation=arguments['generation'],actor='actual child I/O process')
"""
        payload = {
            "database": str(tmp_path / "campaign.sqlite"),
            "source": str(sources[0]),
            "target": str(target),
            "attempt_id": attempt["id"],
            "revision": attempt["revision"],
            "token": attempt["lease_token"],
            "generation": attempt["lease_generation"],
        }
        started = (time.monotonic(), time.process_time())
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert result.returncode == 71, result.stderr
        assert coordinator.attempt(attempt["id"])["state"] == "validating"
        original = {path.name: path.read_bytes() for path in target.iterdir()}
        with pytest.raises(CampaignError, match="commit receipt"):
            verify_committed_pes(coordinator, target)
        with pytest.raises(AuthorityError):
            recover_pes_publication(
                coordinator,
                attempt["id"],
                attempt["revision"],
                destination=target,
                lease_token="unissued-token",
                lease_generation=attempt["lease_generation"],
                usage=measurements(started),
                actor=ACTOR,
                sources=sources,
            )
        if ownership == "expired":
            expiry = coordinator.attempt(attempt["id"])["lease_expires"]
            time.sleep(max(0.0, expiry - time.time()) + 0.02)
        elif ownership == "revoked":
            coordinator.revoke(
                approved["campaign_id"],
                authority=approved["authority"],
                actor=ACTOR,
                reason="Actual approval withdrawal after interrupted publication",
            )
        arguments = dict(
            destination=target,
            usage=MeasuredUsage(
                wall_seconds=time.monotonic() - started[0],
                cpu_core_seconds=None,
                measurement_source=(
                    "actual parent wall time; total CPU unavailable after child exit"
                ),
            ),
            actor=ACTOR,
            sources=sources,
            **credentials(attempt),
        )
        if ownership != "current":
            with pytest.raises(AuthorityError):
                recover_pes_publication(
                    coordinator, attempt["id"], attempt["revision"], **arguments
                )
            assert coordinator.attempt(attempt["id"])["state"] == "validating"
            assert {
                path.name: path.read_bytes() for path in target.iterdir()
            } == original
            return
        recover_pes_publication(
            coordinator,
            attempt["id"],
            attempt["revision"],
            **arguments,
        )
        assert verify_committed_pes(coordinator, target)["kind"] == "merge"
        assert {path.name: path.read_bytes() for path in target.iterdir()} == original
    finally:
        coordinator.close()
