"""Actual ledgers, approved missing nodes, genuine failures and real calculations."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from cochem_torq.application import execute_request
from cochem_torq.artifacts import file_digest
from cochem_torq.candidate_ledger import CandidateLedger
from cochem_torq.domain import CalculationRequest, canonical_json, digest, read_json
from cochem_torq.landscape_inspection import (
    LandscapeFilter,
    LandscapeInspection,
    filter_landscape,
    inspect_candidate_ledger,
    inspect_scan_landscape,
    render_landscape_html,
)
from cochem_torq.scan import ApprovedScanExecutor, ScanPlan
from cochem_torq.service import approve_plan, plan_request
from cochem_torq.student_app import StudentSession, launch_student_app
from tests.test_application_contracts import actual_failure_shard
from tests.test_pes_scan_integrity import scan_request


def supplied_request(recipe="hf-sto-3g-education"):
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["hydrogen-1", "hydrogen-2"],
            },
            "recipe": recipe,
            "products": ["geometry", "harmonic"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 120},
            "source_provenance": {"producer": "explicit_supplied_test_geometry"},
        }
    ).model_dump(mode="json")


def test_absent_ledger_view_creates_no_database_or_invented_energies(tmp_path):
    path = tmp_path / "absent.sqlite"
    view = inspect_candidate_ledger(path)
    assert not path.exists()
    assert view.observations == view.comparison_groups == ()
    assert view.source_reference["status"] == "absent"
    assert view.coverage["candidate_search_coverage"] is None
    assert view.populations is view.conformational_entropy is None
    assert not view.automatic_pruning_enabled


def test_supplied_input_has_explicit_missing_energy_uncertainty_and_history(tmp_path):
    path = tmp_path / "actual.sqlite"
    with CandidateLedger(path) as ledger:
        original = ledger.register_request(
            supplied_request(),
            actor="local test user",
            reason="Actual supplied geometry",
        )
        snapshot = ledger.selection_snapshot()
    view = inspect_candidate_ledger(path)
    row = view.observations[0]
    assert row.observation_id == original["candidate_id"]
    assert row.energy_status == "not_computed"
    assert row.energy_hartree is row.relative_energy_hartree is None
    assert row.comparability_sha256 is None
    assert row.uncertainty["status"] == "not_computed"
    assert [event["action"] for event in row.history] == ["register"]
    assert view.selection_snapshot == snapshot
    assert view.coverage["available_energy_candidates"] == 0
    with CandidateLedger(path) as ledger:
        assert ledger.selection_snapshot() == snapshot


def test_reversible_filters_do_not_change_selection_history_or_definitions(tmp_path):
    path = tmp_path / "selection.sqlite"
    with CandidateLedger(path) as ledger:
        original = ledger.register_request(
            supplied_request(), actor="student", reason="Record supplied input"
        )
        ledger.exclude(
            original["candidate_id"],
            expected_revision=1,
            actor="student",
            reason="Temporarily exclude actual input",
        )
        excluded_snapshot = ledger.selection_snapshot()
    view = inspect_candidate_ledger(path)
    original_view = view.model_dump(mode="json")
    assert filter_landscape(view, LandscapeFilter(selection_states=("retained",))) == ()
    assert filter_landscape(view, LandscapeFilter()) == view.observations
    assert filter_landscape(view, LandscapeFilter(available_energies_only=True)) == ()
    assert view.model_dump(mode="json") == original_view
    with CandidateLedger(path) as ledger:
        assert ledger.selection_snapshot() == excluded_snapshot
        restored = ledger.restore(
            original["candidate_id"],
            expected_revision=2,
            actor="student",
            reason="Restore same recorded input",
        )
    restored_view = inspect_candidate_ledger(path)
    assert restored_view.observations[0].selection_state == "retained"
    assert restored["content_sha256"] == original["content_sha256"]
    assert [event["action"] for event in restored_view.observations[0].history] == [
        "register",
        "exclude",
        "restore",
    ]


def test_uncalculated_quarantine_can_be_manually_retained_without_fake_energy(tmp_path):
    session = StudentSession(tmp_path / "downloads")
    session.load_request(supplied_request())
    original = session.register_input_candidate(actor="student", reason="Record input")
    with CandidateLedger(session.candidate_ledger_path) as ledger:
        ledger.quarantine(
            original["candidate_id"],
            expected_revision=1,
            actor="student",
            reason="Requires scientific review",
        )
    session.change_candidate_selection(
        original["candidate_id"],
        revision=2,
        action="retain",
        actor="student",
        reason="Retain input for a future calculation",
    )
    view = LandscapeInspection.model_validate(session.inspect_ensemble())
    assert view.observations[0].selection_state == "retained"
    assert view.observations[0].energy_status == "not_computed"
    assert view.observations[0].energy_hartree is None
    assert [event["action"] for event in view.observations[0].history] == [
        "register",
        "quarantine",
        "retain",
    ]


def test_html_escapes_actual_user_history_and_has_no_energy_markers(tmp_path):
    path = tmp_path / "unsafe-text.sqlite"
    with CandidateLedger(path) as ledger:
        ledger.register_request(
            supplied_request(),
            actor="<script>test-input</script>",
            reason="<img src=x onerror=alert(1)>",
        )
    displayed = render_landscape_html(inspect_candidate_ledger(path))
    assert "<script>" not in displayed and "<img src=x" not in displayed
    assert "&lt;script&gt;" in displayed
    assert "<circle " not in displayed
    assert "not_computed" in displayed


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_relative_display_limits_are_rejected(value):
    with pytest.raises(ValidationError):
        LandscapeFilter(maximum_relative_energy_hartree=value)


def test_unknown_comparison_group_cannot_supply_a_rank_or_reference(tmp_path):
    view = inspect_candidate_ledger(tmp_path / "absent.sqlite")
    with pytest.raises(ValueError, match="actual observed comparison group"):
        filter_landscape(view, LandscapeFilter(comparison_group_sha256="f" * 64))


def test_read_only_inspector_does_not_initialize_an_unrelated_sqlite_file(tmp_path):
    import sqlite3

    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE actual_user_data(value TEXT)")
        connection.execute("INSERT INTO actual_user_data VALUES ('preserve')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="existing candidate ledger"):
        inspect_candidate_ledger(path)
    assert path.read_bytes() == before


def test_genuine_rejected_application_result_keeps_energy_unavailable(tmp_path):
    source = actual_failure_shard(tmp_path / "actual-rejection")
    path = tmp_path / "rejected.sqlite"
    with CandidateLedger(path) as ledger:
        record = ledger.register_shard(
            source, actor="student", reason="Keep real rejected-method evidence"
        )
    view = inspect_candidate_ledger(path)
    assert view.observations[0].observation_id == record["candidate_id"]
    assert view.observations[0].energy_status == "unavailable"
    assert view.observations[0].energy_hartree is None
    assert view.observations[0].reason
    assert view.observations[0].selection_state == "quarantined"
    assert view.comparison_groups == ()


def test_genuinely_unexecuted_approved_scan_retains_every_missing_node(tmp_path):
    request = scan_request()
    approved = approve_plan(
        plan_request(request, execution="local_validation"),
        actor="actual local approval user",
    )
    root = tmp_path / "unexecuted"
    with ApprovedScanExecutor(approved, root) as executor:
        result = executor.finish()
    assert result.physical_call_count == 0
    view = inspect_scan_landscape(
        root, expected_result_sha256=file_digest(root / "result.json")
    )
    assert len(view.observations) == 3
    assert all(row.energy_status == "not_computed" for row in view.observations)
    assert all(row.energy_hartree is None for row in view.observations)
    assert view.coverage["uncomputed_grid_nodes"] == 3
    assert view.coverage["observed_grid_fraction"] == 0.0
    assert "nonperiodic" in render_landscape_html(view)
    assert "<circle " not in render_landscape_html(view)
    assert not view.search_completeness_established


@pytest.fixture(scope="module")
def genuine_scan(tmp_path_factory):
    request = scan_request()
    approved = approve_plan(
        plan_request(request, execution="local_validation"),
        actor="real scan qualification user",
    )
    root = tmp_path_factory.mktemp("cochem_exec_landscape_scan") / "partial"
    with ApprovedScanExecutor(approved, root) as executor:
        first = executor.evaluate(0, purpose="forward")
        second = executor.evaluate(1, purpose="forward")
        result = executor.finish()
    assert first.status == second.status == "available"
    assert result.status == "partial"
    return root, result


@pytest.mark.real_engine
def test_genuine_scan_shows_actual_energies_and_explicit_uncomputed_nodes(genuine_scan):
    root, surface = genuine_scan
    view = inspect_scan_landscape(
        root, expected_result_sha256=file_digest(root / "result.json")
    )
    assert view.coverage["available_grid_nodes"] == 2
    assert view.coverage["uncomputed_grid_nodes"] == 1
    assert view.coverage["observed_grid_fraction"] == pytest.approx(2 / 3)
    assert len(view.comparison_groups) == 1
    minimum = min(point.energy_hartree for point in surface.points)
    for row, node in zip(view.observations, surface.nodes):
        assert row.energy_hartree == node.energy_hartree
        if node.energy_hartree is not None:
            assert row.relative_energy_hartree == node.energy_hartree - minimum
            assert row.references["native_manifest_sha256"]
        else:
            assert row.relative_energy_hartree is row.relative_reference_id is None
        assert row.stationary_status == "not_established_by_this_view"
        assert row.uncertainty["status"] == "uncalibrated"
        assert row.repeat_energy_spread_hartree is None
    displayed = render_landscape_html(view)
    assert displayed.count("<circle ") == 2
    assert "not_computed" in displayed
    assert "actual markers only" in displayed
    assert "No populations" in displayed
    assert view.populations is view.conformational_entropy is None


@pytest.mark.real_engine
def test_filtered_actual_scan_preserves_reference_and_never_zero_fills(genuine_scan):
    root, _ = genuine_scan
    view = inspect_scan_landscape(
        root, expected_result_sha256=file_digest(root / "result.json")
    )
    before = view.model_dump(mode="json")
    energy_rows = filter_landscape(view, LandscapeFilter(available_energies_only=True))
    assert len(energy_rows) == 2
    limited = filter_landscape(
        view, LandscapeFilter(maximum_relative_energy_hartree=0.0)
    )
    assert len(limited) == 2  # Actual zero reference plus explicitly missing node.
    assert any(row.energy_hartree is None for row in limited)
    assert all(
        row.relative_reference_id == view.comparison_groups[0].reference_observation_id
        for row in energy_rows
    )
    assert filter_landscape(view) == view.observations
    assert view.model_dump(mode="json") == before


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "damage", ["native_result", "checkpoint", "point_identity", "invented_node_energy"]
)
def test_corrupted_actual_scan_evidence_is_rejected(genuine_scan, tmp_path, damage):
    root, surface = genuine_scan
    copied = tmp_path / "deliberately-damaged-scan"
    shutil.copytree(root, copied)
    first = surface.points[0]
    native = copied / first.native_manifest_path
    if damage == "native_result":
        (native.parent / "result.json").write_text("{}\n")
    elif damage == "checkpoint":
        path = native.parent / "wavefunction.chk"
        path.write_bytes(path.read_bytes()[:128])
    elif damage == "point_identity":
        path = native.parent.parent / "point-result.json"
        point = read_json(path)
        point["coordinate_values"][0] += 0.01
        path.write_bytes(canonical_json(point))
    else:
        body = read_json(copied / "result.json")
        body["nodes"][-1]["energy_hartree"] = surface.points[0].energy_hartree
        (copied / "result.json").write_bytes(canonical_json(body))
    with pytest.raises(ValueError):
        inspect_scan_landscape(
            copied, expected_result_sha256=file_digest(copied / "result.json")
        )


@pytest.fixture(scope="module")
def genuine_refinements(tmp_path_factory):
    root = tmp_path_factory.mktemp("cochem_exec_landscape_refinements")
    output = []
    for recipe in ("hf-sto-3g-education", "hf-cc-pvdz-research"):
        directory = root / recipe
        result = execute_request(supplied_request(recipe), directory)
        assert result["status"] == "complete", result["errors"]
        output.append((directory, result))
    return output


@pytest.mark.real_engine
def test_real_different_recipes_have_separate_energy_zero_groups(
    genuine_refinements, tmp_path
):
    path = tmp_path / "real-results.sqlite"
    with CandidateLedger(path) as ledger:
        for directory, _ in genuine_refinements:
            ledger.register_shard(
                directory, actor="real student", reason="Retain actual calculated model"
            )
    view = inspect_candidate_ledger(path)
    assert len(view.comparison_groups) == 2
    assert all(row.energy_status == "available" for row in view.observations)
    assert len({row.comparability_sha256 for row in view.observations}) == 2
    assert all(row.relative_energy_hartree == 0.0 for row in view.observations)
    assert all(
        not group.cross_group_ranking_authorized for group in view.comparison_groups
    )
    assert {row.energy_recipe for row in view.observations} == {
        "hf-sto-3g-education",
        "hf-cc-pvdz-research",
    }
    assert render_landscape_html(view).count("<svg ") == 2


@pytest.mark.real_engine
def test_unavailable_actual_candidate_files_do_not_keep_rankable_energy(
    genuine_refinements, tmp_path
):
    source, _ = genuine_refinements[0]
    copied = tmp_path / "actual-result-copy"
    shutil.copytree(source, copied)
    path = tmp_path / "missing-raw.sqlite"
    with CandidateLedger(path) as ledger:
        ledger.register_shard(
            copied, actor="student", reason="Keep actual raw result copy"
        )
    (copied / "engine" / "final" / "wavefunction.chk").unlink()
    view = inspect_candidate_ledger(path)
    assert view.observations[0].energy_status == "changed_or_invalid"
    assert view.observations[0].energy_hartree is None
    assert view.observations[0].relative_energy_hartree is None
    assert view.comparison_groups == ()
    assert view.observations[0].references["result_reference"]


def test_student_ui_excludes_local_pes_profiles_and_exposes_read_only_views():
    app = launch_student_app()
    try:
        recipe = next(
            child
            for child in app.children
            if getattr(child, "description", "") == "Recipe"
        )
        assert all(
            "pes-validation" not in identifier for _, identifier in recipe.options
        )
        nested = [
            item
            for child in app.children
            for item in (getattr(child, "children", ()) or (child,))
        ]
        names = {getattr(child, "description", "") for child in nested}
        assert "Retain reviewed candidate" in names
        assert "Inspect retained scan" in names
        assert "Only authenticated energies" in names
        assert app.student_session.submission is None
    finally:
        app.close()


def test_periodic_topology_is_declared_without_inventing_an_observed_surface(tmp_path):
    # Mathematical supplied Cartesian coordinates; no electronic calculation.
    raw = scan_request().model_dump(mode="json")
    raw["recipe"] = "hf-sto-3g-internal-pes-validation"
    raw["molecule"] = {
        "symbols": ["C", "C", "H", "H"],
        "geometry_bohr": [
            [-0.75, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [-1.25, 1.0, 0.0],
            [1.25, 0.5, 0.9],
        ],
        "charge": 0,
        "multiplicity": 1,
        "atom_ids": ["carbon-1", "carbon-2", "hydrogen-1", "hydrogen-2"],
    }
    declaration = raw["source_provenance"]["pes_scan"]
    declaration["coordinates"] = [
        {
            "coordinate_id": "declared-mathematical-dihedral",
            "kind": "dihedral",
            "atom_indices": [2, 0, 1, 3],
            "unit": "degree",
            "domain": {
                "minimum": -180.0,
                "maximum": 180.0,
                "periodic": True,
                "period": 360.0,
            },
        }
    ]
    declaration["grid"] = [[-170.0], [0.0], [170.0]]
    declaration["unscanned_coordinates"] = "minimum_displacement_embedding"
    declaration["cartesian_movable_atom_indices"] = [2, 3]
    raw["source_provenance"]["producer"] = "supplied_mathematical_coordinate_array"
    raw["source_provenance"]["pes_scan"] = ScanPlan.model_validate(
        declaration
    ).model_dump(mode="json")
    request = CalculationRequest.model_validate(raw)
    approved = approve_plan(
        plan_request(request, execution="local_validation"), actor="local student"
    )
    root = tmp_path / "unexecuted-periodic-design"
    with ApprovedScanExecutor(approved, root) as executor:
        executor.finish()
    view = inspect_scan_landscape(
        root, expected_result_sha256=file_digest(root / "result.json")
    )
    assert "periodic, period=360.0" in render_landscape_html(view)
    assert all(row.energy_hartree is None for row in view.observations)
    assert [row.coordinate_values for row in view.observations] == [
        (-170.0,),
        (0.0,),
        (170.0,),
    ]


@pytest.mark.real_engine
def test_genuine_failed_scf_remains_missing_with_failed_and_uncomputed_nodes(tmp_path):
    request = scan_request(scf_max_cycle=1)
    approved = approve_plan(
        plan_request(request, execution="local_validation"), actor="real local user"
    )
    root = tmp_path / "actual-failed-scf"
    with ApprovedScanExecutor(approved, root) as executor:
        point = executor.evaluate(0, purpose="forward")
        executor.finish()
    assert point.status == "failed" and point.energy_hartree is None
    view = inspect_scan_landscape(
        root, expected_result_sha256=file_digest(root / "result.json")
    )
    assert view.observations[0].energy_status == "failed"
    assert all(row.energy_hartree is None for row in view.observations)
    assert view.coverage["failed_grid_nodes"] == 1
    assert view.coverage["uncomputed_grid_nodes"] == 2
    assert "<circle " not in render_landscape_html(view)


@pytest.mark.real_engine
def test_sampled_candidate_uses_observed_source_recipe_and_no_stationary_claim(
    genuine_scan, tmp_path
):
    root, surface = genuine_scan
    point = min(
        surface.points,
        key=lambda item: item.energy_hartree,
    )
    native_path = root / point.native_manifest_path
    point_path = native_path.parent.parent / "point-result.json"
    request = supplied_request()
    request["molecule"] = point.molecule.model_dump(mode="json")
    request["source_provenance"]["adaptive_candidate"] = {
        "point_id": point.point_id,
        "scan_sha256": point.scan_sha256,
        "constraints_sha256": point.scan_sha256,
        "point_result_path": str(point_path),
        "point_result_sha256": file_digest(point_path),
        "native_manifest_path": str(native_path),
        "native_manifest_sha256": point.native_manifest_sha256,
        "candidate_kind": "lowest_observed_point",
        "verified_stationary": False,
    }
    path = tmp_path / "actual-sampled-candidate.sqlite"
    with CandidateLedger(path) as ledger:
        candidate = ledger.register_request(
            request, actor="student", reason="Record observed finite-grid candidate"
        )
        ledger.quarantine(
            candidate["candidate_id"],
            expected_revision=1,
            actor="student",
            reason="Stationarity and constraint release uncomputed",
        )
    view = inspect_candidate_ledger(path)
    row = view.observations[0]
    assert row.energy_status == "available", row.reason
    assert row.energy_hartree == point.energy_hartree
    assert row.energy_recipe == "hf-sto-3g-pes-validation"
    assert row.requested_recipe == "hf-sto-3g-education"
    assert row.geometry_status == "fixed_nonstationary_sample"
    assert row.selection_state == "quarantined"
    assert row.stationary_status == "not_established_by_this_view"
    assert view.comparison_groups[0].definition["recipe_sha256"] == point.recipe_sha256
    assert (
        view.comparison_groups[0].definition["constraint_identity"] == point.scan_sha256
    )
    assert row.references["geometry_sha256"] == digest(
        point.molecule.model_dump(mode="json")
    )


@pytest.mark.real_engine
def test_retained_scan_inspection_runs_without_importing_a_quantum_engine(genuine_scan):
    root, _ = genuine_scan
    code = (
        "import sys\n"
        "from cochem_torq.landscape_inspection import inspect_scan_landscape\n"
        "result=inspect_scan_landscape(sys.argv[1],"
        "expected_result_sha256=sys.argv[2])\n"
        "assert len(result.observations)==3\n"
        "assert not any(name=='pyscf' or name.startswith('pyscf.') "
        "for name in sys.modules)\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(root), file_digest(root / "result.json")],
        check=True,
        capture_output=True,
        text=True,
    )
