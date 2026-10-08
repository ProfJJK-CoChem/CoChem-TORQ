"""Actual separate-producer verification and student import, without calculation.

The retained fixture was produced by a genuine BASE-authorized xTB calculation;
these tests preserve its evidence and do not rerun or replace that calculation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.domain import CalculationRequest, canonical_json
from cochem_torq.reviewed_topos import (
    import_reviewed_topos_member,
    inspect_reviewed_topos,
)
from cochem_torq.student_app import StudentSession, launch_student_app
from cochem_torq.units import ANGSTROM_BOHR

FIXTURE = Path(__file__).parent / "fixtures" / "topos_native_xtb_reviewed_water.json"
FIXTURE_SHA256 = "d76512c2c41cc796842db7ae0d9458078f74f8bd158f01bf6216e1075f40795c"


@pytest.fixture
def producer_python() -> Path:
    explicit = os.environ.get("COCHEM_REVIEWED_TOPOS_PYTHON")
    if explicit:
        return Path(explicit).absolute()
    from ci_tools.setup_reviewed_topos import verify_profile

    return verify_profile(Path(__file__).resolve().parents[1])


def _member() -> dict:
    return json.loads(FIXTURE.read_bytes())["members"][0]


def _import(producer_python: Path, destination: Path, **changes: object) -> dict:
    arguments = dict(
        producer_python=producer_python,
        member_id=_member()["member_id"],
        recipe="hf-sto-3g-education",
        products=["geometry"],
        resources={"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
    )
    arguments.update(changes)
    return import_reviewed_topos_member(FIXTURE, destination, **arguments)


def test_actual_producer_verifies_the_original_native_fixture(producer_python):
    original = FIXTURE.read_bytes()
    assert sha256(original).hexdigest() == FIXTURE_SHA256
    observed = inspect_reviewed_topos(FIXTURE, producer_python=producer_python)
    assert observed["source_sha256"] == FIXTURE_SHA256
    assert observed["producer"]["python"] == str(producer_python)
    assert (
        observed["handoff"]["members"][0]["candidate"]["energy_hartree"]
        == -5.070544054679
    )
    assert FIXTURE.read_bytes() == original


def test_geometry_import_preserves_source_and_uses_an_independent_new_recipe(
    producer_python,
    tmp_path,
):
    destination = tmp_path / "import"
    original = FIXTURE.read_bytes()
    request = _import(producer_python, destination)
    model = CalculationRequest.model_validate(request)
    member = _member()
    np.testing.assert_array_equal(
        model.molecule.geometry_bohr,
        np.asarray(member["molecule"]["coordinates"]) * ANGSTROM_BOHR,
    )
    assert model.molecule.atom_ids == member["molecule"]["atom_ids"]
    assert model.molecule.isotopes == member["molecule"]["isotopes"]
    assert model.molecule.charge == member["molecule"]["charge"]
    assert model.molecule.multiplicity == member["molecule"]["multiplicity"]
    assert model.recipe == "hf-sto-3g-education"
    imported = model.source_provenance["reviewed_topos"]
    assert imported["source_recipe"]["method"] == "GFN2-xTB"
    assert imported["observed_attempt"]["engine_version"] == "6.7.1"
    assert imported["source_result_reused"] is False
    assert imported["engine_checkpoint_reused"] is False
    assert imported["computation_performed"] is False
    assert len(canonical_json(request)) < 16 * 1024
    assert (destination / "source-handoff.json").read_bytes() == original
    assert json.loads((destination / "request.json").read_bytes()) == request
    inventory = json.loads((destination / "manifest.json").read_bytes())["files"]
    assert set(inventory) == {
        "source-handoff.json",
        "request.json",
        "verification.json",
    }
    for name, expected in inventory.items():
        actual = destination / name
        assert sha256(actual.read_bytes()).hexdigest() == expected["sha256"]
        assert actual.stat().st_size == expected["bytes"]
        assert actual.stat().st_mode & 0o222 == 0
    assert FIXTURE.read_bytes() == original


@pytest.mark.parametrize(
    "field",
    [
        "coordinates",
        "isotopes",
        "charge",
        "multiplicity",
        "atom_ids",
        "coordinate_units",
    ],
)
def test_altered_geometry_identity_rejects_before_import(
    producer_python, tmp_path, field
):
    handoff = json.loads(FIXTURE.read_bytes())
    molecule = handoff["members"][0]["molecule"]
    if field == "coordinates":
        molecule[field][0][0] += 0.1
    elif field == "isotopes":
        molecule[field][0] = 18
    elif field in {"charge", "multiplicity"}:
        molecule[field] += 1
    elif field == "atom_ids":
        molecule[field].reverse()
    else:
        molecule[field] = "bohr"
    source = tmp_path / "changed.json"
    source.write_text(json.dumps(handoff))
    destination = tmp_path / "import"
    with pytest.raises(ValueError, match="producer rejected"):
        import_reviewed_topos_member(
            source,
            destination,
            producer_python=producer_python,
            member_id=_member()["member_id"],
            recipe="hf-sto-3g-education",
            products=["geometry"],
            resources={"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "change", ["unknown_member", "stale_handoff", "unknown_recipe"]
)
def test_selection_and_recipe_mismatches_never_publish(
    producer_python, tmp_path, change
):
    arguments = {
        "unknown_member": {"member_id": "absent-member"},
        "stale_handoff": {"expected_handoff_sha256": "0" * 64},
        "unknown_recipe": {"recipe": "invented-target-recipe"},
    }[change]
    with pytest.raises(ValueError):
        _import(producer_python, tmp_path / "import", **arguments)
    assert not (tmp_path / "import").exists()


def test_existing_import_is_preserved_and_requires_a_new_destination(
    producer_python, tmp_path
):
    destination = tmp_path / "import"
    _import(producer_python, destination)
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(ValueError, match="new destination"):
        _import(producer_python, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_symlink_input_is_rejected_before_the_producer_runs(producer_python, tmp_path):
    link = tmp_path / "handoff.json"
    link.symlink_to(FIXTURE)
    with pytest.raises(ValueError, match="nonsymlink"):
        inspect_reviewed_topos(link, producer_python=producer_python)


def test_nonproducer_interpreter_is_not_implicitly_replaced(tmp_path):
    environment = tmp_path / "no-topos"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    with pytest.raises(ValueError, match="producer rejected"):
        inspect_reviewed_topos(FIXTURE, producer_python=environment / "bin" / "python")


def test_student_session_creates_new_request_and_clears_previous_approval(
    producer_python, tmp_path
):
    session = StudentSession(tmp_path / "results")
    session.approved_plan = {"previous": True}
    request = session.prepare_reviewed_topos(
        FIXTURE,
        producer_python=producer_python,
        member_id=_member()["member_id"],
        recipe="hf-cc-pvdz-research",
        products=["geometry"],
        cores=1,
        memory_mb=1024,
        wall_seconds=120,
    )
    assert request["recipe"] == "hf-cc-pvdz-research"
    assert session.approved_plan is None
    assert session.submission is None
    assert session.idempotency_key == request["request_id"]
    imports = list((tmp_path / "topos-imports").iterdir())
    assert len(imports) == 1
    assert (imports[0] / "source-handoff.json").read_bytes() == FIXTURE.read_bytes()


def _widgets(widget):
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _widgets(child)


def test_real_notebook_upload_inspects_selects_and_previews_without_calculation(
    producer_python, tmp_path
):
    app = launch_student_app(results_directory=tmp_path / "results")
    controls = list(_widgets(app))
    next(
        w for w in controls if getattr(w, "description", "") == "Input"
    ).value = "reviewed_topos"
    next(
        w for w in controls if getattr(w, "description", "") == "TOPOS Python"
    ).value = str(producer_python)
    upload = next(w for w in controls if w.__class__.__name__ == "FileUpload")
    original = FIXTURE.read_bytes()
    upload.value = (
        {
            "name": "outside-reviewed-ensemble.json",
            "type": "application/json",
            "size": len(original),
            "content": memoryview(original),
            "last_modified": datetime.now(timezone.utc),
        },
    )
    next(
        w
        for w in controls
        if getattr(w, "description", "") == "Inspect reviewed TOPOS ensemble"
    ).click()
    member = next(
        w for w in controls if getattr(w, "description", "") == "TOPOS member"
    )
    member.value = _member()["member_id"]
    assert member.value == _member()["member_id"]
    assert app.student_session.request is None
    next(
        w for w in controls if getattr(w, "description", "") == "Import and preview"
    ).click()
    request = app.student_session.request
    assert request is not None
    assert request["source_provenance"]["reviewed_topos"]["member_id"] == member.value
    assert app.student_session.submission is None
    assert app.student_session.approved_plan is None
    assert (
        list((tmp_path / "topos-imports").glob("*/source-handoff.json"))[0].read_bytes()
        == original
    )
    next(
        w for w in controls if getattr(w, "description", "") == "TOPOS Python"
    ).value = "/changed/interpreter"
    assert app.student_session.request is None
    assert not member.options
    app.close()


def test_real_cli_exports_the_verified_new_request(producer_python, tmp_path):
    destination = tmp_path / "import"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.cli",
            "import-topos-reviewed",
            "--handoff",
            str(FIXTURE),
            "--producer-python",
            str(producer_python),
            "--member",
            _member()["member_id"],
            "--recipe",
            "hf-sto-3g-education",
            "--product",
            "geometry",
            "--output-dir",
            str(destination),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    reply = json.loads(process.stdout)
    assert reply["status"] == "needs_review"
    assert reply["data"]["computation_performed"] is False
    assert reply["data"]["requires_plan_review"] is True
    assert reply["data"]["request"] == json.loads(
        (destination / "request.json").read_bytes()
    )
