"""Real file, numerical-frame and application boundary tests for the student UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.student_app import StudentSession, input_mass_frame, molecule_from_xyz

XYZ = (
    "3\nStarting geometry, not a calculated result\n"
    "O 0 0 0\nH 0.75 0 0.55\nH -0.75 0 0.55"
)


def test_xyz_frame_is_mass_centered_right_handed_and_rigid_motion_invariant() -> None:
    molecule = molecule_from_xyz(XYZ, charge=0, multiplicity=1)
    frame = input_mass_frame(molecule)
    np.testing.assert_allclose(
        np.average(frame.coordinates_angstrom, axis=0, weights=frame.masses_u),
        0,
        atol=1e-12,
    )
    np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame.rotation) > 0
    angle = 0.53
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    moved = dict(
        molecule,
        geometry_bohr=(
            np.asarray(molecule["geometry_bohr"]) @ rotation + [3, -1, 7]
        ).tolist(),
    )
    np.testing.assert_allclose(
        input_mass_frame(moved).moments_u_angstrom2,
        frame.moments_u_angstrom2,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2\nmissing atoms\nH 0 0 0",
        "1\ninvalid\nH nan 0 0",
        "1\ninvalid\nH 0 0 0 extra",
        "2\nfirst\nH 0 0 0\nH 0 0 1\n2\nsecond\nH 0 0 0\nH 0 0 1",
    ],
)
def test_malformed_or_multiframe_xyz_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        molecule_from_xyz(text, charge=0, multiplicity=1)


def test_explicit_isotope_mass_changes_inertia_without_coordinate_changes() -> None:
    ordinary = input_mass_frame(molecule_from_xyz(XYZ, charge=0, multiplicity=1))
    labelled = input_mass_frame(
        molecule_from_xyz(XYZ.replace("H ", "D "), charge=0, multiplicity=1)
    )
    assert labelled.masses_u[1] > ordinary.masses_u[1]
    assert labelled.moments_u_angstrom2[-1] > ordinary.moments_u_angstrom2[-1]


def test_session_roundtrips_actual_request_file_without_submitting(
    tmp_path: Path,
) -> None:
    session = StudentSession(tmp_path / "results")
    request = session.prepare_xyz(
        XYZ,
        charge=0,
        multiplicity=1,
        recipe="hf-sto-3g-education",
        products=["geometry"],
        cores=2,
        memory_mb=2048,
        wall_seconds=600,
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    other = StudentSession(tmp_path / "other")
    assert other.load_request(path) == request
    assert other.submission is None
    assert not other.results_directory.exists()


def test_session_validates_with_real_application_and_reports_unqualified_product() -> (
    None
):
    session = StudentSession()
    session.prepare_xyz(
        XYZ,
        charge=0,
        multiplicity=1,
        recipe="hf-sto-3g-education",
        products=["identification_catalog"],
        cores=1,
        memory_mb=2048,
        wall_seconds=600,
    )
    report = session.validate()
    assert report["blocked_products"] == ["identification_catalog"]
    assert session.request["request_id"] == report["request"]["request_id"]
    assert session.validate()["request_sha256"] == report["request_sha256"]
    assert session.submission is None


def test_session_missing_request_and_run_id_are_explicit_errors() -> None:
    session = StudentSession()
    with pytest.raises(ValueError, match="Import"):
        session._request()
    with pytest.raises(ValueError, match="run ID"):
        session._run_id(None)


@pytest.mark.parametrize(
    "content",
    [
        '{"molecule": {}, "molecule": {}}',
        '{"molecule": NaN}',
        '{"molecule": {}, "source_provenance": {"invalid": 1e999}}',
    ],
)
def test_ambiguous_or_nonfinite_json_file_is_rejected(
    tmp_path: Path, content: str
) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        StudentSession().load_request(source)


def test_widget_launch_and_real_preview_click_use_actual_request() -> None:
    from cochem_torq.student_app import launch_student_app

    app = launch_student_app()
    textarea = next(
        widget for widget in app.children if widget.__class__.__name__ == "Textarea"
    )
    textarea.value = XYZ
    buttons = [
        widget
        for child in app.children
        for widget in getattr(child, "children", ())
        if getattr(widget, "description", "") == "Import and preview"
    ]
    assert len(buttons) == 1
    buttons[0].click()
    assert app.student_session.request["molecule"]["charge"] == 0
    assert app.student_session.request["molecule"]["symbols"] == ["O", "H", "H"]
    assert app.student_session.submission is None
    recipe = next(
        widget
        for widget in app.children
        if getattr(widget, "description", "") == "Recipe"
    )
    recipe.value = "hf-cc-pvdz-research"
    assert app.student_session.request is None
    app.close()


def prepared_session():
    session = StudentSession()
    session.prepare_xyz(
        XYZ,
        charge=0,
        multiplicity=1,
        recipe="hf-sto-3g-education",
        products=["geometry"],
        cores=1,
        memory_mb=512,
        wall_seconds=60,
    )
    return session


def test_review_and_approval_are_explicit_and_preserve_request_retry_identity():
    session = prepared_session()
    key = session.idempotency_key
    with pytest.raises(ValueError, match="explicitly approve"):
        session.submit()
    with pytest.raises(ValueError, match="Review"):
        session.approve(actor="course-student")
    review = session.review_plan()
    request_id = session.request["request_id"]
    assert session.approved_plan is None
    assert review["approval"] is None
    assert review["plan"]["resources"]["wall_seconds"] == 60
    approved = session.approve(actor="course-student")
    assert approved["approval"]["plan_sha256"] == review["plan"]["plan_sha256"]
    assert approved["approval"]["actor"] == "course-student"
    assert approved["approval"]["max_cpu_core_seconds"] == 60
    assert session.idempotency_key == key
    assert session.validate()["request"]["request_id"] == request_id
    assert session.idempotency_key == key
    assert session.submission is None
    session.review_plan()
    assert session.approved_plan is None
    assert session.request["request_id"] == request_id
    assert session.idempotency_key == key


def test_changed_request_cannot_submit_a_previous_approval():
    session = prepared_session()
    session.review_plan()
    session.approve(actor="course-student")
    session.request["products"] = ["harmonic"]
    with pytest.raises(ValueError, match="differs"):
        session.submit()
    assert session.approved_plan is None
    assert session.plan_review is None
    assert session.submission is None


def test_loading_or_clearing_input_invalidates_review_approval_and_retry_key():
    session = prepared_session()
    session.review_plan()
    session.approve(actor="course-student")
    old_key = session.idempotency_key
    request = json.loads(json.dumps(session.request))
    session.load_request(request)
    assert session.plan_review is None and session.approved_plan is None
    assert session.idempotency_key == old_key
    session.clear_request()
    assert session.request is None
    assert session.idempotency_key is None


def test_widget_requires_plan_review_and_explicit_actor_approval():
    from cochem_torq.student_app import launch_student_app

    app = launch_student_app()

    def descendants(widget):
        yield widget
        for child in getattr(widget, "children", ()):
            yield from descendants(child)

    controls = list(descendants(app))
    by_name = {getattr(control, "description", ""): control for control in controls}
    source = next(
        control for control in controls if control.__class__.__name__ == "Textarea"
    )
    source.value = XYZ
    assert by_name["Submit to Actions"].disabled
    by_name["Import and preview"].click()
    by_name["Review plan"].click()
    assert app.student_session.plan_review is not None
    assert app.student_session.approved_plan is None
    assert by_name["Approve this plan"].disabled
    by_name["Approved by"].value = "course-student"
    assert not by_name["Approve this plan"].disabled
    assert by_name["Submit to Actions"].disabled
    by_name["Approve this plan"].click()
    assert app.student_session.approved_plan is not None
    assert not by_name["Submit to Actions"].disabled
    by_name["Approved by"].value = "another-student"
    assert app.student_session.approved_plan is None
    assert by_name["Submit to Actions"].disabled
    by_name["Approve this plan"].click()
    by_name["Wall sec"].value = 120
    assert app.student_session.request is None
    assert app.student_session.approved_plan is None
    assert by_name["Submit to Actions"].disabled
    app.close()


def test_resume_uses_a_verified_failed_shard_and_requires_new_approval(tmp_path):
    from cochem_torq.application import validate_request, worker_execute
    from cochem_torq.artifacts import seal_shard
    from cochem_torq.domain import CalculationRequest, canonical_json

    original = CalculationRequest.model_validate(
        {
            "molecule": molecule_from_xyz(XYZ, charge=0, multiplicity=1),
            "recipe": "revdsd-pbep86-d4-experimental",
            "products": ["geometry"],
        }
    )
    directory = tmp_path / "actual-rejection"
    directory.mkdir()
    checked = validate_request(original)
    result = worker_execute(original, directory)
    assert result["status"] == "failed"
    (directory / "request.json").write_bytes(canonical_json(checked["request"]))
    (directory / "result.json").write_bytes(canonical_json(result))
    seal_shard(
        directory,
        request_sha256=checked["request_sha256"],
        request_id=str(original.request_id),
        recipe_sha256=checked["recipe"]["recipe_sha256"],
        source_identity=result["source_identity"],
        worker_id="actual-unqualified-profile",
    )
    session = StudentSession()
    recovered = session.resume(directory)
    assert recovered["request_id"] != str(original.request_id)
    lineage = recovered["source_provenance"]["attempt_lineage"]
    assert lineage["previous_request_id"] == str(original.request_id)
    assert lineage["engine_checkpoint_reused"] is False
    assert session.approved_plan is None
    assert not session.review_plan()["executable"]
    with pytest.raises(ValueError, match="explicitly approve"):
        session.submit()


def test_cancel_requires_an_explicit_reason_before_transport():
    session = StudentSession()
    with pytest.raises(ValueError, match="bounded reason"):
        session.cancel(reason="", run_id="0")
