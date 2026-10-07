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
    "content", ['{"molecule": {}, "molecule": {}}', '{"molecule": NaN}']
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
