"""Actual fixed H2 engine scans and declared coordinate-contract rejection.

Constructed inputs are mathematical coordinates or invalid metadata, never
substitute physical energies, provider records, engine outputs or executables.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.application import validate_request, worker_execute
from cochem_torq.domain import CalculationRequest, PrerequisiteError, digest, read_json
from cochem_torq.operations import ArtifactBackpressureError
from cochem_torq.registry import get_profile
from cochem_torq.scan import (
    ApprovedScanExecutor,
    CoordinateDomain,
    InternalCoordinate,
    ScanPlan,
    ScanSurface,
    _native_scan_evidence,
    _rename_no_replace,
    _verify_native,
    _write_immutable,
    geometry_for_sample,
    scan_blocking_reasons,
    scan_definition,
    scan_plan_for_request,
)
from cochem_torq.service import approve_plan, plan_request, validate_approved_plan


def scan_request(
    *, scf_max_cycle: int = 100, one_sample: bool = False
) -> CalculationRequest:
    grid = [[1.4]] if one_sample else [[1.2], [1.4], [1.6]]
    indices = list(range(len(grid)))
    declaration = {
        "coordinates": [
            {
                "coordinate_id": "H-H",
                "kind": "bond",
                "atom_indices": [0, 1],
                "unit": "bohr",
                "domain": {"minimum": 1.2, "maximum": 1.6, "periodic": False},
            }
        ],
        "grid": grid,
        "sampling_strategy": "full_grid",
        "passes": [
            {"purpose": "forward", "sample_indices": indices},
            {"purpose": "reverse", "sample_indices": list(reversed(indices))},
            {"purpose": "challenge", "sample_indices": indices},
        ],
        "coordinate_treatment": "fixed",
        "unscanned_coordinates": "frozen_cartesian",
        "additional_constraints": [],
        "initial_guess_policy": "independent_pyscf_minao",
        "budget": {"max_physical_calls": 3 * len(grid), "per_point_wall_seconds": 20},
        "energy_recheck_tolerance_hartree": 1e-9,
        "density_recheck_tolerance": 1e-8,
        "scf_max_cycle": scf_max_cycle,
    }
    # Normalizing the declaration records every explicit algorithm default in
    # the reviewed content rather than treating omitted fields as observations.
    scan = ScanPlan.model_validate(declaration)
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.3, 0.4, 0.0], [0.3, 0.4, 1.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["H-left", "H-right"],
            },
            "recipe": "hf-sto-3g-pes-validation",
            "products": ["pes_scan"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 180},
            "source_provenance": {"pes_scan": scan.model_dump(mode="json")},
        }
    )


def test_plan_counts_actual_unique_samples_and_independent_calls():
    request = scan_request()
    scan = scan_definition(request)
    plan = scan_plan_for_request(request, get_profile(request.recipe))
    assert scan.unique_sample_count == 3
    assert scan.planned_physical_calls == 9
    assert plan["unique_sample_count"] == 3
    assert len(plan["tasks"]) == 9
    assert all(
        task["operation"] == "fixed_coordinate_energy_gradient"
        for task in plan["tasks"]
    )
    assert plan["stationary_points_claimed"] is False
    body = dict(plan)
    declared_digest = body.pop("plan_sha256")
    assert declared_digest == digest(body)
    with pytest.raises(ValidationError):
        scan.grid = ((1.5,),)
    with pytest.raises(ValidationError):
        scan.coordinates[0].domain.minimum = 0.5


@pytest.mark.parametrize("index,distance", [(0, 1.2), (1, 1.4), (2, 1.6)])
def test_fixed_geometry_moves_only_the_declared_second_atom(index, distance):
    request = scan_request()
    scan = scan_definition(request)
    point = geometry_for_sample(request, scan, index)
    np.testing.assert_array_equal(
        point.geometry_bohr[0], request.molecule.geometry_bohr[0]
    )
    assert np.linalg.norm(
        np.asarray(point.geometry_bohr[1]) - point.geometry_bohr[0]
    ) == pytest.approx(distance, rel=1e-15)
    assert point.atom_ids == ("H-left", "H-right")
    assert point.charge == 0 and point.multiplicity == 1
    assert request.molecule.geometry_bohr[1][2] == 1.4


@pytest.mark.parametrize("index", [True, -1, 3, 1.0])
def test_non_grid_indices_are_rejected(index):
    request = scan_request()
    with pytest.raises(ValueError, match="grid index"):
        geometry_for_sample(request, scan_definition(request), index)


@pytest.mark.parametrize("change", ["duplicate", "outside", "count", "boolean"])
def test_invalid_declared_grid_is_rejected(change):
    raw = scan_definition(scan_request()).model_dump(mode="json")
    if change == "duplicate":
        raw["grid"][1] = raw["grid"][0]
    elif change == "outside":
        raw["grid"][0] = [1.1]
    elif change == "count":
        raw["budget"]["max_physical_calls"] = 8
    else:
        raw["coordinates"][0]["domain"]["periodic"] = "false"
    with pytest.raises(ValidationError):
        ScanPlan.model_validate(raw)


def test_periodicity_uses_full_torsional_domain_not_rotational_symmetry():
    torsion = InternalCoordinate(
        coordinate_id="mathematical-torsion",
        kind="dihedral",
        atom_indices=(0, 1, 2, 3),
        unit="degree",
        domain=CoordinateDomain(
            minimum=0.0, maximum=360.0, periodic=True, period=360.0
        ),
    )
    raw = scan_definition(scan_request()).model_dump(mode="json")
    raw["coordinates"] = [torsion.model_dump(mode="json")]
    raw["grid"] = [[0.0], [120.0], [240.0]]
    scan = ScanPlan.model_validate(raw)
    assert scan.unique_sample_count == 3
    raw["grid"][-1] = [360.0]
    with pytest.raises(ValidationError, match="half-open"):
        ScanPlan.model_validate(raw)
    with pytest.raises(ValidationError, match="2pi"):
        InternalCoordinate.model_validate(
            {
                **torsion.model_dump(mode="json"),
                "domain": {
                    "minimum": 0.0,
                    "maximum": 120.0,
                    "periodic": True,
                    "period": 120.0,
                },
            }
        )


@pytest.mark.parametrize("change", ["relaxed", "continuation", "constraint", "mapping"])
def test_unimplemented_scientific_semantics_block_before_engine_work(change):
    request = scan_request()
    raw = request.model_dump(mode="json")
    if change == "relaxed":
        raw["source_provenance"]["pes_scan"]["coordinate_treatment"] = "relaxed"
        raw["source_provenance"]["pes_scan"]["unscanned_coordinates"] = (
            "constrained_relaxation"
        )
    elif change == "continuation":
        raw["source_provenance"]["pes_scan"]["initial_guess_policy"] = (
            "previous_point_density"
        )
    elif change == "constraint":
        raw["source_provenance"]["pes_scan"]["additional_constraints"] = [
            "declared-extra-constraint"
        ]
    else:
        raw["molecule"]["atom_ids"] = None
    changed = CalculationRequest.model_validate(raw)
    assert scan_blocking_reasons(changed, scan_definition(changed))
    checked = validate_request(changed, execution="local_validation")
    assert not checked["executable"]


def test_scan_uses_standard_approval_and_blocks_hosted_workflow(tmp_path):
    request = scan_request()
    review = plan_request(request, execution="local_validation")
    assert review["executable"], review["blocking_reasons"]
    approved = approve_plan(review, actor=f"actual-release-test-os:{os.getuid()}")
    validated = validate_approved_plan(approved)
    assert validated.approval.max_tasks == 9
    assert validated.plan["scan_sha256"] == digest(
        scan_definition(request).model_dump(mode="json")
    )
    assert not plan_request(request, execution="github_actions")["executable"]
    corrupted = deepcopy(approved)
    corrupted["plan"]["request"]["source_provenance"]["pes_scan"]["grid"][0][0] = 1.3
    with pytest.raises(ValueError, match="digest"):
        validate_approved_plan(corrupted)
    with pytest.raises(PrerequisiteError, match="approved scan executor"):
        worker_execute(request, tmp_path / "wrong-worker")
    assert not (tmp_path / "wrong-worker" / "engine").exists()


@pytest.fixture(scope="module")
def actual_scan(tmp_path_factory):
    request = scan_request()
    approved = approve_plan(
        plan_request(request, execution="local_validation"),
        actor=f"actual-release-test-os:{os.getuid()}",
    )
    directory = tmp_path_factory.mktemp("actual-pes") / "scan"
    with ApprovedScanExecutor(approved, directory) as executor:
        with pytest.raises(AttributeError):
            executor.scan = scan_definition(request)
        surface = executor.run()
    return surface, directory


@pytest.mark.real_engine
def test_genuine_h2_scan_preserves_absolute_energy_and_fixed_geometry(actual_scan):
    surface, directory = actual_scan
    assert surface.status == "complete"
    assert surface.physical_call_count == 9
    assert (
        surface.unique_planned_samples
        == surface.unique_physically_attempted_samples
        == 3
    )
    assert not surface.stationary_points_verified
    assert not surface.experimental_accuracy_established
    assert not surface.identification_ready
    energies = [node.energy_hartree for node in surface.nodes]
    assert energies[1] < energies[0] and energies[1] < energies[2]
    assert energies[1] == pytest.approx(-1.11671432506255, abs=1e-9)
    for point in surface.points:
        assert point.status == "available"
        assert point.geometry_status == "fixed_nonstationary_sample"
        assert point.cpu_core_seconds is None
        native_path = directory / point.native_manifest_path
        assert (
            sha256(native_path.read_bytes()).hexdigest() == point.native_manifest_sha256
        )
        native = read_json(native_path.parent / "result.json")
        assert native["energy_hartree"] == point.energy_hartree
        np.testing.assert_array_equal(
            native["gradient_hartree_bohr"], point.gradient_hartree_bohr
        )
        assert native["scf"]["converged"] is True
        assert point.density.electron_count == 2
    assert ScanSurface.model_validate(read_json(directory / "result.json")) == surface


@pytest.mark.real_engine
def test_independent_reverse_and_challenge_checks_retain_actual_density(actual_scan):
    surface, directory = actual_scan
    assert len(surface.comparisons) == 6
    assert all(not comparison.inconsistent for comparison in surface.comparisons)
    assert all(
        abs(comparison.energy_difference_hartree) < 1e-9
        for comparison in surface.comparisons
    )
    assert all(
        comparison.density_difference_frobenius < 1e-8
        for comparison in surface.comparisons
    )
    assert surface.hysteresis_scope.endswith("no_continuation_branch_completeness")
    accounting = read_json(directory / "campaign-accounting.json")
    assert len(accounting) == 9
    assert all(
        row["dispatched"] == 1 and row["state"] == "settled" for row in accounting
    )
    assert len({point.task_id for point in surface.points}) == 9
    assert len({point.attempt_id for point in surface.points}) == 9


@pytest.mark.real_engine
def test_actual_scf_nonconvergence_stays_failed_without_replacement(tmp_path):
    request = scan_request(scf_max_cycle=1, one_sample=True)
    approved = approve_plan(
        plan_request(request, execution="local_validation"),
        actor=f"actual-release-test-os:{os.getuid()}",
    )
    directory = tmp_path / "nonconverged"
    with ApprovedScanExecutor(approved, directory) as executor:
        surface = executor.run()
    assert surface.status == "failed"
    assert surface.physical_call_count == 3
    assert all(
        point.status == "failed" and point.energy_hartree is None and point.reason
        for point in surface.points
    )
    assert surface.nodes[0].status == "failed"
    assert surface.nodes[0].energy_hartree is None
    assert len(surface.nodes[0].failed_observation_ids) == 3
    for point in surface.points:
        native = read_json(
            (directory / point.native_manifest_path).parent / "result.json"
        )
        assert native["scf"]["converged"] is False
        assert native["status"] == "failed"
        assert native["energy_hartree"] is None
        assert any("converge" in item["message"] for item in native["errors"])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinate_domains_are_rejected(value):
    with pytest.raises(ValidationError):
        CoordinateDomain(minimum=1.0, maximum=value, periodic=False)


def test_atomic_publication_preserves_existing_directory_and_source(tmp_path):
    source = tmp_path / "actual-unpublished-directory"
    source.mkdir()
    (source / "actual-content.txt").write_text(
        "actual filesystem content", encoding="utf-8"
    )
    destination = tmp_path / "existing-empty-directory"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _rename_no_replace(source, destination)
    assert (source / "actual-content.txt").read_text() == "actual filesystem content"
    assert destination.is_dir() and not list(destination.iterdir())


def test_competing_actual_process_publications_have_one_immutable_winner(tmp_path):
    sources = [tmp_path / "publisher-one", tmp_path / "publisher-two"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "publisher.txt").write_text(str(index), encoding="utf-8")
    destination = tmp_path / "shared-publication"
    program = """
from pathlib import Path
import sys
from cochem_torq.scan import _rename_no_replace
try:
    _rename_no_replace(Path(sys.argv[1]), Path(sys.argv[2]))
except FileExistsError:
    raise SystemExit(3)
"""
    processes = [
        subprocess.Popen([sys.executable, "-c", program, str(source), str(destination)])
        for source in sources
    ]
    codes = [process.wait(timeout=15) for process in processes]
    assert sorted(codes) == [0, 3]
    winner = int((destination / "publisher.txt").read_text())
    assert not sources[winner].exists()
    assert (sources[1 - winner] / "publisher.txt").read_text() == str(1 - winner)


@pytest.mark.real_engine
def test_corrupted_copy_of_actual_native_evidence_cannot_be_accepted(
    actual_scan, tmp_path
):
    surface, root = actual_scan
    actual_native = (root / surface.points[0].native_manifest_path).parent
    copy = tmp_path / "corrupted-actual-copy"
    shutil.copytree(actual_native, copy)
    with (copy / "pyscf.log").open("a", encoding="utf-8") as stream:
        stream.write("Intentional byte corruption for an integrity rejection test.\n")
    with pytest.raises(ValueError, match="immutable inventory"):
        _verify_native(copy)
    rejected = _native_scan_evidence(copy, surface.points[0].molecule)
    assert rejected.status == "failed"
    assert rejected.electronic is rejected.gradient is rejected.density is None
    assert rejected.reason and "immutable inventory" in rejected.reason
    assert (
        rejected.native_manifest_sha256
        == sha256((copy / "manifest.json").read_bytes()).hexdigest()
    )
    assert (copy / "pyscf.log").is_file()
    assert (
        _native_scan_evidence(actual_native, surface.points[0].molecule).status
        == "available"
    )


def test_final_surface_metadata_cannot_bypass_actual_scratch_ceiling(tmp_path):
    request = scan_request(one_sample=True)
    approved = approve_plan(
        plan_request(request, execution="local_validation"),
        actor=f"actual-release-test-os:{os.getuid()}",
        max_scratch_bytes=1024**2,
    )
    with ApprovedScanExecutor(approved, tmp_path / "quota") as executor:
        payload = executor.workspace / "actual-filesystem-allocation.txt"
        payload.write_bytes(b"x" * 1024**2)
        with pytest.raises(ArtifactBackpressureError):
            executor.finish()
        assert payload.stat().st_size == 1024**2
        assert executor.points == ()


def test_standalone_metadata_publication_is_immutable(tmp_path):
    destination = tmp_path / "actual-declared-record.json"
    _write_immutable(destination, {"declaration": "actual first record"})
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        _write_immutable(destination, {"declaration": "conflicting second record"})
    assert destination.read_bytes() == original
    assert len(list(tmp_path.glob("*.pending"))) == 1
