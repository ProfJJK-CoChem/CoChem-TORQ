"""Actual approved RHF observations, native snapshots and fenced HDF5 admission.

All physical samples come from genuine PySCF workers. Invalid inputs are explicit
mutations of those real observations; no engine, filesystem or coordinator APIs
are patched and no substituted scientific result is admitted.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from cochem.orchestration.campaign import (
    Allocation,
    CampaignBudget,
    CampaignCoordinator,
    CampaignError,
    MeasuredUsage,
)
from cochem.storage.fenced_pes import (
    PESIdentity,
    merge_pes_shards,
    merge_task_payload,
    publish_pes_shard,
    shard_task_payload,
)
from cochem_torq.domain import CalculationRequest, canonical_json, read_json
from cochem_torq.pes_storage import (
    NativePESCollection,
    _validate_initial_guess,
    native_shard_task_payload,
    prepare_native_scan_collection,
    publish_native_scan_shard,
    verify_committed_native_pes,
    verify_native_collection,
)
from cochem_torq.scan import ApprovedScanExecutor, ScanPlan, ScanPointResult
from cochem_torq.service import ApprovedPlan, approve_plan, plan_request

ACTOR = f"genuine-native-storage-check:{os.getuid()}"


def request():
    scan = ScanPlan.model_validate(
        {
            "coordinates": [
                {
                    "coordinate_id": "H-H",
                    "kind": "bond",
                    "atom_indices": [0, 1],
                    "unit": "bohr",
                    "domain": {"minimum": 1.2, "maximum": 1.6, "periodic": False},
                }
            ],
            "grid": [[1.4]],
            "sampling_strategy": "full_grid",
            "passes": [
                {"purpose": "forward", "sample_indices": [0]},
                {"purpose": "reverse", "sample_indices": [0]},
                {"purpose": "challenge", "sample_indices": [0]},
            ],
            "coordinate_treatment": "fixed",
            "unscanned_coordinates": "frozen_cartesian",
            "additional_constraints": [],
            "initial_guess_policy": "independent_pyscf_minao",
            "budget": {"max_physical_calls": 3, "per_point_wall_seconds": 30},
            "energy_recheck_tolerance_hartree": 1e-9,
            "density_recheck_tolerance": 1e-8,
        }
    )
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.1, 0.2, 0.0], [0.1, 0.2, 1.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["left-H", "right-H"],
            },
            "recipe": "hf-sto-3g-pes-validation",
            "products": ["pes_scan"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 90},
            "source_provenance": {"pes_scan": scan.model_dump(mode="json")},
        }
    )


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    root = tmp_path_factory.mktemp("genuine-native-pes")
    workspace = root / "scan"
    approved = approve_plan(
        plan_request(request(), execution="local_validation"), actor=ACTOR
    )
    with ApprovedScanExecutor(approved, workspace) as executor:
        surface = executor.run()
        points = executor.points
    assert surface.status == "complete"
    assert len(points) == 3 and all(point.status == "available" for point in points)
    coordinator = CampaignCoordinator(root / ".scan.scan-campaign.sqlite")
    collection = prepare_native_scan_collection(
        coordinator,
        workspace=workspace,
        points=points,
        destination=root / "native-bundle",
    )
    yield root, workspace, points, coordinator, collection
    coordinator.close()


def storage_campaign(coordinator, collection):
    return coordinator.create_campaign(
        {
            "kind": "genuine_native_pes_storage",
            "identity": collection.identity.record(),
        },
        CampaignBudget(
            max_wall_seconds=300.0,
            max_cpu_core_seconds=300.0,
            max_tasks=8,
            max_concurrent_workers=1,
            max_cpu_cores=1,
            max_memory_mb=2048,
            max_scratch_mb=64,
            expires_at_unix=time.time() + 300,
            allowed_engines=[collection.identity.engine],
            allowed_recipes=[collection.identity.recipe["id"]],
        ),
        actor=ACTOR,
    )


@contextmanager
def writer(coordinator, campaign, identity, payload):
    task = coordinator.register_task(
        campaign["campaign_id"],
        payload,
        engine=identity.engine,
        recipe=identity.recipe["id"],
        plan_sha256=campaign["plan_sha256"],
        authority=campaign["authority"],
        actor=ACTOR,
    )
    attempt = coordinator.new_attempt(
        task["task_id"], authority=campaign["authority"], actor=ACTOR
    )
    attempt = coordinator.transition(
        attempt["id"],
        attempt["revision"],
        "validated",
        authority=campaign["authority"],
        actor=ACTOR,
        reason="Validated genuine native storage task",
    )
    attempt = coordinator.reserve(
        attempt["id"],
        attempt["revision"],
        Allocation(wall_seconds=30.0, cores=1, memory_mb=2048, scratch_mb=32),
        authority=campaign["authority"],
        actor=ACTOR,
    )
    attempt = coordinator.lease(
        attempt["id"],
        attempt["revision"],
        pid=os.getpid(),
        authority=campaign["authority"],
        actor=ACTOR,
        heartbeat_seconds=10.0,
        lease_seconds=120.0,
    )
    credentials = {
        "lease_token": attempt["lease_token"],
        "lease_generation": attempt["lease_generation"],
    }
    for state in ("collecting", "validating"):
        attempt = coordinator.worker_transition(
            attempt["id"],
            attempt["revision"],
            state,
            actor=ACTOR,
            reason=f"Native storage writer entered {state}",
            **credentials,
        )
    started = time.monotonic()
    try:
        yield attempt, credentials
    finally:
        current = coordinator.attempt(attempt["id"])
        if current["state"] == "validating":
            coordinator.publish(
                current["id"],
                current["revision"],
                {
                    "status": "rejected_storage_test_operation",
                    "no_artifact_admitted": True,
                },
                MeasuredUsage(
                    wall_seconds=time.monotonic() - started,
                    measurement_source="actual rejected storage-operation interval",
                ),
                lambda: None,
                state="failed",
                actor=ACTOR,
                **credentials,
            )


def measurement(start):
    return MeasuredUsage(
        wall_seconds=time.monotonic() - start,
        measurement_source="actual parent monotonic storage interval",
    )


@pytest.mark.real_engine
def test_genuine_scan_bytes_and_commits_publish_then_merge(observations, tmp_path):
    _, workspace, points, coordinator, collection = observations
    campaign = storage_campaign(coordinator, collection)
    original_hashes = {
        point.point_id: sha256(
            (workspace / point.native_manifest_path).read_bytes()
        ).hexdigest()
        for point in points
    }
    destinations = []
    for index, point in enumerate(points):
        ids = [point.point_id]
        payload = native_shard_task_payload(collection, ids)
        with writer(coordinator, campaign, collection.identity, payload) as (
            attempt,
            credentials,
        ):
            started = time.monotonic()
            destination = publish_native_scan_shard(
                coordinator,
                attempt["id"],
                attempt["revision"],
                collection=collection,
                frame_ids=ids,
                destination=tmp_path / f"shard-{index}",
                usage=measurement(started),
                actor=ACTOR,
                **credentials,
            )
        manifest = verify_committed_native_pes(coordinator, destination)
        assert manifest["identity"]["evidence_class"] == "native_engine_observation"
        assert manifest["scientific_accuracy_established"] is False
        destinations.append(destination)
    payload = merge_task_payload(collection.identity, destinations)
    with writer(coordinator, campaign, collection.identity, payload) as (
        attempt,
        credentials,
    ):
        started = time.monotonic()
        merged = merge_pes_shards(
            coordinator,
            attempt["id"],
            attempt["revision"],
            identity=collection.identity,
            sources=destinations,
            destination=tmp_path / "merged",
            usage=measurement(started),
            actor=ACTOR,
            **credentials,
        )
    manifest = verify_committed_native_pes(coordinator, merged)
    assert manifest["sample_count"] == 3
    assert manifest["frame_ids"] == list(collection.frame_ids)
    with h5py.File(merged / "samples.h5", "r", locking=True) as store:
        np.testing.assert_array_equal(
            store["coordinates"][:], [point.molecule.geometry_bohr for point in points]
        )
        np.testing.assert_array_equal(
            store["energies"][:], [point.energy_hartree for point in points]
        )
        assert store.attrs["coordinate_unit"] == "bohr"
        assert store.attrs["energy_unit"] == "hartree"
    assert original_hashes == {
        point.point_id: sha256(
            (workspace / point.native_manifest_path).read_bytes()
        ).hexdigest()
        for point in points
    }


@pytest.mark.parametrize(
    "change", ["energy", "geometry", "state", "source", "atom_order"]
)
def test_altered_point_cannot_replace_a_genuine_commit(observations, tmp_path, change):
    _, workspace, points, coordinator, _ = observations
    raw = points[0].model_dump(mode="json")
    if change == "energy":
        raw["energy_hartree"] += 1.0
    elif change == "geometry":
        raw["molecule"]["geometry_bohr"][1][2] += 0.2
    elif change == "state":
        raw["molecule"]["multiplicity"] = 3
    elif change == "source":
        raw["source_identity_sha256"] = "0" * 64
    else:
        raw["molecule"]["atom_ids"].reverse()
    changed = ScanPointResult.model_validate(raw)
    with pytest.raises((ValueError, CampaignError)):
        prepare_native_scan_collection(
            coordinator,
            workspace=workspace,
            points=(changed,),
            destination=tmp_path / "altered",
        )
    assert not (tmp_path / "altered").exists()


def test_missing_or_failed_observation_is_never_stored(observations, tmp_path):
    _, workspace, points, coordinator, _ = observations
    raw = points[0].model_dump(mode="json")
    raw.update(
        status="unavailable",
        energy_hartree=None,
        gradient_hartree_bohr=None,
        density=None,
        reason="explicit unavailable-input rejection check",
    )
    missing = ScanPointResult.model_validate(raw)
    with pytest.raises((ValueError, CampaignError)):
        prepare_native_scan_collection(
            coordinator,
            workspace=workspace,
            points=(missing,),
            destination=tmp_path / "missing",
        )


@pytest.mark.parametrize(
    "change", ["coordinate_unit", "energy_unit", "engine", "recipe"]
)
def test_native_units_and_method_scope_are_strict(observations, change):
    _, _, _, _, collection = observations
    raw = collection.identity.record()
    if change in {"coordinate_unit", "energy_unit"}:
        raw[change] = "dimensionless"
    elif change == "engine":
        raw["engine"] = "undeclared_engine"
    else:
        raw["recipe"]["id"] = "revDSD-PBEP86-D4"
    with pytest.raises(ValidationError):
        PESIdentity.model_validate(raw)


@pytest.mark.parametrize("selection", ["reversed", "duplicate", "unknown", "empty"])
def test_native_frame_identity_and_order_are_strict(observations, selection):
    _, _, _, _, collection = observations
    ids = list(collection.frame_ids)
    if selection == "reversed":
        ids.reverse()
    elif selection == "duplicate":
        ids.append(ids[0])
    elif selection == "unknown":
        ids[0] = str(uuid4())
    else:
        ids.clear()
    with pytest.raises(ValueError):
        native_shard_task_payload(collection, ids)


def test_arbitrary_arrays_cannot_be_published_under_a_native_label(
    observations, tmp_path
):
    _, _, points, coordinator, collection = observations
    campaign = storage_campaign(coordinator, collection)
    point = points[0]
    ids = [point.point_id]
    with writer(
        coordinator,
        campaign,
        collection.identity,
        shard_task_payload(collection.identity, ids),
    ) as (attempt, credentials):
        started = time.monotonic()
        with pytest.raises(ValueError, match="HDF5 quantities"):
            publish_pes_shard(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=collection.identity,
                coordinates=[point.molecule.geometry_bohr],
                energies=[point.energy_hartree + 1.0],
                frame_ids=ids,
                destination=tmp_path / "arbitrary",
                usage=measurement(started),
                actor=ACTOR,
                **credentials,
            )
    assert not (tmp_path / "arbitrary").exists()


def test_native_label_without_real_bundle_fails_closed(observations, tmp_path):
    _, _, points, coordinator, collection = observations
    raw = deepcopy(collection.identity.record())
    raw["source_identity"]["native_bundle_directory"] = str(tmp_path / "not-observed")
    identity = PESIdentity.model_validate(raw)
    altered = collection.model_copy(update={"identity": identity})
    campaign = storage_campaign(coordinator, altered)
    point = points[0]
    ids = [point.point_id]
    with writer(coordinator, campaign, identity, shard_task_payload(identity, ids)) as (
        attempt,
        credentials,
    ):
        started = time.monotonic()
        with pytest.raises(ValueError, match="actual regular"):
            publish_pes_shard(
                coordinator,
                attempt["id"],
                attempt["revision"],
                identity=identity,
                coordinates=[point.molecule.geometry_bohr],
                energies=[point.energy_hartree],
                frame_ids=ids,
                destination=tmp_path / "label-only",
                usage=measurement(started),
                actor=ACTOR,
                **credentials,
            )
    assert not (tmp_path / "label-only").exists()


def test_changed_native_snapshot_is_rejected(observations):
    _, _, points, _, collection = observations
    from pathlib import Path

    path = (
        Path(collection.bundle_directory)
        / "points"
        / points[0].point_id
        / "native"
        / "pyscf.log"
    )
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"explicit corrupted-evidence test\n")
        with pytest.raises(ValueError, match="inventory/bytes"):
            verify_native_collection(collection)
    finally:
        path.write_bytes(original)
    verify_native_collection(collection)


def test_collection_metadata_mutation_cannot_substitute_a_recipe(observations):
    _, _, _, _, collection = observations
    raw = collection.model_dump(mode="json")
    raw["identity"]["recipe"]["basis"] = "cc-pvtz"
    altered = NativePESCollection.model_validate(raw)
    with pytest.raises(ValueError, match="approval/recipe/source/units"):
        verify_native_collection(altered)


@pytest.mark.parametrize("field", ["preserved", "qualification", "numerical_stability"])
def test_collection_identity_rejects_numeric_boolean_substitutes(observations, field):
    _, _, _, _, collection = observations
    raw = collection.model_dump(mode="json")
    if field == "preserved":
        raw["identity"]["source_identity"]["original_native_bytes_preserved"] = 1
    elif field == "qualification":
        raw["identity"]["source_identity"]["independent_scientific_qualification"] = 0
    else:
        raw["identity"]["recipe"]["numerical"]["check_stability"] = 1
    altered = NativePESCollection.model_validate(raw)
    with pytest.raises(ValueError, match="approval/recipe/source/units"):
        verify_native_collection(altered)


def test_missing_source_authority_rejects_genuine_bytes(observations, tmp_path):
    _, _, points, _, collection = observations
    # A genuinely empty independent coordinator has no original engine receipts.
    coordinator = CampaignCoordinator(tmp_path / "independent.sqlite")
    try:
        campaign = storage_campaign(coordinator, collection)
        point = points[0]
        ids = [point.point_id]
        with writer(
            coordinator,
            campaign,
            collection.identity,
            native_shard_task_payload(collection, ids),
        ) as (attempt, credentials):
            started = time.monotonic()
            with pytest.raises(CampaignError, match="absent"):
                publish_native_scan_shard(
                    coordinator,
                    attempt["id"],
                    attempt["revision"],
                    collection=collection,
                    frame_ids=ids,
                    destination=tmp_path / "unauthorized",
                    usage=measurement(started),
                    actor=ACTOR,
                    **credentials,
                )
        assert not (tmp_path / "unauthorized").exists()
    finally:
        coordinator.close()


def test_original_native_log_remains_required_at_publication(observations, tmp_path):
    _, workspace, points, coordinator, collection = observations
    point = points[0]
    path = workspace / point.native_manifest_path
    log = path.parent / "pyscf.log"
    original = log.read_bytes()
    campaign = storage_campaign(coordinator, collection)
    ids = [point.point_id]
    try:
        log.write_bytes(original + b"explicit corrupted original-native test\n")
        with writer(
            coordinator,
            campaign,
            collection.identity,
            native_shard_task_payload(collection, ids),
        ) as (attempt, credentials):
            started = time.monotonic()
            with pytest.raises(
                ValueError, match="native artifact bytes|Native artifact bytes"
            ):
                publish_native_scan_shard(
                    coordinator,
                    attempt["id"],
                    attempt["revision"],
                    collection=collection,
                    frame_ids=ids,
                    destination=tmp_path / "changed-original",
                    usage=measurement(started),
                    actor=ACTOR,
                    **credentials,
                )
    finally:
        log.write_bytes(original)
    assert not (tmp_path / "changed-original").exists()


@pytest.mark.parametrize(
    "change", ["revision", "lease_token", "generation", "cancelled"]
)
def test_native_writer_requires_exact_current_lease(observations, tmp_path, change):
    _, _, points, coordinator, collection = observations
    point = points[0]
    ids = [point.point_id]
    campaign = storage_campaign(coordinator, collection)
    with writer(
        coordinator,
        campaign,
        collection.identity,
        native_shard_task_payload(collection, ids),
    ) as (attempt, credentials):
        revision = attempt["revision"]
        altered_credentials = dict(credentials)
        if change == "revision":
            revision -= 1
        elif change == "lease_token":
            altered_credentials["lease_token"] = "explicit invalid-credential input"
        elif change == "generation":
            altered_credentials["lease_generation"] += 1
        else:
            coordinator.cancel(
                attempt["id"],
                attempt["revision"],
                authority=campaign["authority"],
                actor=ACTOR,
                reason="Actual operator cancellation before native publication",
            )
        started = time.monotonic()
        with pytest.raises((ValueError, CampaignError)):
            publish_native_scan_shard(
                coordinator,
                attempt["id"],
                revision,
                collection=collection,
                frame_ids=ids,
                destination=tmp_path / "stale-writer",
                usage=measurement(started),
                actor=ACTOR,
                **altered_credentials,
            )
    assert not (tmp_path / "stale-writer").exists()


def test_legitimate_completed_native_lease_may_expire(observations, tmp_path):
    _, _, points, coordinator, collection = observations
    point = points[0]
    original = coordinator.attempt(point.attempt_id)
    assert original["state"] == "succeeded"
    delay = max(0.0, original["lease_expires"] - time.time() + 0.02)
    assert delay < 12.0  # The real native executor's lease is ten seconds.
    if delay:
        time.sleep(delay)
    assert original["lease_expires"] < time.time()
    campaign = storage_campaign(coordinator, collection)
    ids = [point.point_id]
    with writer(
        coordinator,
        campaign,
        collection.identity,
        native_shard_task_payload(collection, ids),
    ) as (attempt, credentials):
        started = time.monotonic()
        destination = publish_native_scan_shard(
            coordinator,
            attempt["id"],
            attempt["revision"],
            collection=collection,
            frame_ids=ids,
            destination=tmp_path / "historical-source-lease",
            usage=measurement(started),
            actor=ACTOR,
            **credentials,
        )
    result = verify_committed_native_pes(coordinator, destination)
    assert result["frame_ids"] == ids


def continuation_request():
    raw = request().model_dump(mode="json")
    declaration = raw["source_provenance"]["pes_scan"]
    declaration.update(
        grid=[[1.3], [1.5]],
        initial_guess_policy="previous_point_density",
        unscanned_coordinates="minimum_displacement_embedding",
        cartesian_movable_atom_indices=[1],
        passes=[
            {"purpose": "forward", "sample_indices": [0, 1]},
            {"purpose": "reverse", "sample_indices": [1, 0]},
            {"purpose": "challenge", "sample_indices": [0, 1]},
        ],
        budget={"max_physical_calls": 6, "per_point_wall_seconds": 30},
    )
    raw["recipe"] = "hf-sto-3g-internal-pes-validation"
    raw["resources"]["wall_seconds"] = 180
    raw["source_provenance"]["pes_scan"] = ScanPlan.model_validate(
        declaration
    ).model_dump(mode="json")
    return CalculationRequest.model_validate(raw)


@pytest.fixture(scope="module")
def continued_observations(tmp_path_factory):
    root = tmp_path_factory.mktemp("genuine-continued-native-pes")
    workspace = root / "scan"
    approved = approve_plan(
        plan_request(continuation_request(), execution="local_validation"), actor=ACTOR
    )
    with ApprovedScanExecutor(approved, workspace) as executor:
        surface = executor.run()
        points = executor.points
    assert surface.status == "complete"
    assert len(points) == 6 and all(point.status == "available" for point in points)
    coordinator = CampaignCoordinator(root / ".scan.scan-campaign.sqlite")
    collection = prepare_native_scan_collection(
        coordinator,
        workspace=workspace,
        points=points,
        destination=root / "continued-native-bundle",
    )
    yield root, workspace, points, coordinator, collection
    coordinator.close()


@pytest.mark.real_engine
def test_actual_forward_reverse_continuation_preserves_and_admits_ancestors(
    continued_observations, tmp_path
):
    _, workspace, points, coordinator, collection = continued_observations
    for index, point in enumerate(points):
        native = read_json(workspace / point.native_manifest_path)
        assert native["schema_version"] == "cochem-torq.engine-artifacts.v1"
        result = read_json(
            (workspace / point.native_manifest_path).parent / "result.json"
        )
        if index in {1, 3}:
            assert point.parent_point_ids == (points[index - 1].point_id,)
            consumption = result["checkpoint_consumption"]
            assert consumption["engine_checkpoint_reused"] is True
            assert consumption["fresh_scf_kernel_called"] is True
            assert consumption["final_result_reused"] is False
            assert (
                consumption["source_manifest_sha256"]
                == points[index - 1].native_manifest_sha256
            )
        else:
            assert point.parent_point_ids == ()
            assert result.get("checkpoint_consumption") is None
    # A child-only shard still requires the original committed parent lineage.
    campaign = storage_campaign(coordinator, collection)
    ids = [points[1].point_id]
    with writer(
        coordinator,
        campaign,
        collection.identity,
        native_shard_task_payload(collection, ids),
    ) as (attempt, credentials):
        started = time.monotonic()
        destination = publish_native_scan_shard(
            coordinator,
            attempt["id"],
            attempt["revision"],
            collection=collection,
            frame_ids=ids,
            destination=tmp_path / "continued-child",
            usage=measurement(started),
            actor=ACTOR,
            **credentials,
        )
    result = verify_committed_native_pes(coordinator, destination)
    assert result["frame_ids"] == ids
    _, preserved = verify_native_collection(collection)
    assert preserved == points


def test_continued_child_collection_requires_complete_parent_lineage(
    continued_observations, tmp_path
):
    _, workspace, points, coordinator, _ = continued_observations
    with pytest.raises(ValueError, match="continued ancestor"):
        prepare_native_scan_collection(
            coordinator,
            workspace=workspace,
            points=(points[1],),
            destination=tmp_path / "child-without-parent",
        )


@pytest.mark.parametrize(
    "change", ["parent", "manifest", "recipe", "policy", "missing"]
)
def test_original_continuation_reference_cannot_be_substituted(
    continued_observations, tmp_path, change
):
    _, workspace, points, coordinator, _ = continued_observations
    path = (
        workspace / points[1].native_manifest_path
    ).parent.parent / "backend-request.json"
    original = path.read_bytes()
    raw = read_json(path)
    if change == "parent":
        raw["initial_guess"]["parent_point_id"] = points[2].point_id
    elif change == "manifest":
        raw["initial_guess"]["source_manifest_sha256"] = "0" * 64
    elif change == "recipe":
        raw["initial_guess"]["recipe_sha256"] = "0" * 64
    elif change == "policy":
        raw["initial_guess"]["policy"] = "independent_pyscf_minao"
    else:
        raw.pop("initial_guess")
    try:
        path.write_bytes(canonical_json(raw))
        with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
            prepare_native_scan_collection(
                coordinator,
                workspace=workspace,
                points=points,
                destination=tmp_path / "substituted-continuation",
            )
    finally:
        path.write_bytes(original)
    assert not (tmp_path / "substituted-continuation").exists()


@pytest.mark.parametrize(
    "field",
    [
        "engine_checkpoint_reused",
        "new_physical_calculation_required",
        "fresh_scf_kernel_called",
        "scf_iteration_state_resumed",
        "final_result_reused",
        "chemical_accuracy_established",
        "project_orbitals",
        "immutable_inputs_verified",
    ],
)
def test_native_consumption_flags_reject_numeric_boolean_substitutes(
    continued_observations, field
):
    _, workspace, points, _, _ = continued_observations
    point = points[1]
    directory = (workspace / point.native_manifest_path).parent.parent
    approved = ApprovedPlan.model_validate(read_json(workspace / "approved-plan.json"))
    backend = read_json(directory / "backend-request.json")
    native = read_json(directory / "native" / "result.json")
    native["checkpoint_consumption"][field] = int(
        native["checkpoint_consumption"][field]
    )
    # Explicit malformed native input to the readback validator, without writing
    # replacement provider bytes or changing any engine/coordinator behavior.
    with pytest.raises(ValueError, match="checkpoint consumption"):
        _validate_initial_guess(approved, point, directory, backend, native)


@pytest.mark.real_engine
def test_general_water_observation_enters_native_storage(tmp_path):
    raw = request().model_dump(mode="json")
    declaration = raw["source_provenance"]["pes_scan"]
    declaration.update(
        coordinates=[
            {
                "coordinate_id": "water-OH",
                "kind": "bond",
                "atom_indices": [0, 1],
                "unit": "bohr",
                "domain": {"minimum": 1.7, "maximum": 2.0, "periodic": False},
            }
        ],
        grid=[[1.8]],
        unscanned_coordinates="minimum_displacement_embedding",
        cartesian_movable_atom_indices=[1],
    )
    raw["source_provenance"]["pes_scan"] = ScanPlan.model_validate(
        declaration
    ).model_dump(mode="json")
    raw["recipe"] = "hf-sto-3g-internal-pes-validation"
    raw["molecule"] = {
        "symbols": ["O", "H", "H"],
        "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.7, 0.0, -0.4]],
        "charge": 0,
        "multiplicity": 1,
        "atom_ids": ["water-O", "water-H1", "water-H2"],
    }
    approved = approve_plan(
        plan_request(
            CalculationRequest.model_validate(raw), execution="local_validation"
        ),
        actor=ACTOR,
    )
    workspace = tmp_path / "water"
    with ApprovedScanExecutor(approved, workspace) as executor:
        surface = executor.run()
        points = executor.points
    assert surface.status == "complete"
    coordinator = CampaignCoordinator(tmp_path / ".water.scan-campaign.sqlite")
    try:
        collection = prepare_native_scan_collection(
            coordinator,
            workspace=workspace,
            points=points,
            destination=tmp_path / "water-native-bundle",
        )
        campaign = storage_campaign(coordinator, collection)
        ids = list(collection.frame_ids)
        with writer(
            coordinator,
            campaign,
            collection.identity,
            native_shard_task_payload(collection, ids),
        ) as (attempt, credentials):
            started = time.monotonic()
            destination = publish_native_scan_shard(
                coordinator,
                attempt["id"],
                attempt["revision"],
                collection=collection,
                frame_ids=ids,
                destination=tmp_path / "water-shard",
                usage=measurement(started),
                actor=ACTOR,
                **credentials,
            )
        manifest = verify_committed_native_pes(coordinator, destination)
        assert manifest["identity"]["molecule"]["symbols"] == ["O", "H", "H"]
        assert manifest["sample_count"] == 3
        with h5py.File(destination / "samples.h5", "r", locking=True) as store:
            assert store["coordinates"].shape == (3, 3, 3)
            np.testing.assert_array_equal(
                store["energies"][:], [point.energy_hartree for point in points]
            )
    finally:
        coordinator.close()
