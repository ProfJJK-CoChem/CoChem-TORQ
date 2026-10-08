"""Student tests use genuine PySCF, OS processes and actual artifact consumers."""

import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cochem_torq.application import execute_request
from cochem_torq.artifacts import merge_shards, seal_shard, verify_shard
from cochem_torq.domain import StageResult, canonical_json, digest, read_json
from cochem_torq.ecosystem import export_base_calculation_result
from cochem_torq.student_app import inspect_downloaded_results

pytestmark = [pytest.mark.real_engine, pytest.mark.student_profile]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_water(tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    request = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request["resources"]["cores"] = 1
    request["products"].append("rigid_rotor_catalog")
    directory = tmp_path_factory.mktemp("actual-student") / "water"
    result = execute_request(request, directory)
    return directory, result


def test_actual_student_workflow_and_consumers(actual_water, tmp_path):
    directory, result = actual_water
    assert result["status"] == "complete", result["errors"]
    assert result["experimental_accuracy_established"] is False
    assert result["identification_ready"] is False
    for name in (
        "electronic_structure",
        "equilibrium_geometry",
        "equilibrium_constants",
        "harmonic_analysis",
        "rigid_rotor_catalog",
    ):
        stage = StageResult.model_validate(result["stages"][name])
        assert stage.status == "available"
    harmonic = result["stages"]["harmonic_analysis"]["value"]
    assert len(harmonic["frequencies_cm1"]) == 3
    assert min(harmonic["frequencies_cm1"]) > 0
    assert harmonic["external_rank"] == 6
    assert result["native_result"]["optimization"]["final_gradient_verified"]
    assert not np.allclose(
        result["molecule"]["geometry_bohr"], result["native_result"]["geometry_bohr"]
    )
    assert result["stages"]["equilibrium_constants"]["observable"] == "Be"
    assert (
        result["stages"]["rigid_rotor_catalog"]["value"]["identification_qualified"]
        is False
    )
    assert len(result["stages"]["rigid_rotor_catalog"]["value"]["lines"]) > 0
    manifest = verify_shard(directory)
    assert manifest["source_identity"]["code_sha256"] != sha256(b"[]").hexdigest()
    from qcelemental.models import AtomicResult

    atomic = AtomicResult(**read_json(directory / "atomic-result.json"))
    assert atomic.success and atomic.driver.value == "hessian"
    summary = inspect_downloaded_results([directory])
    assert summary[0]["scientific_status"] == "complete"
    exported = export_base_calculation_result(directory, tmp_path / "base-result.json")
    data = read_json(exported)
    assert data["energy_hartree"] == result["native_result"]["energy_hartree"]
    assert data["converged"] is True
    assert data["quantity"] == "total_electronic_energy"


def test_actual_bundle_integrity_and_stale_result_rejection(actual_water, tmp_path):
    directory, _ = actual_water
    changed = tmp_path / "changed"
    shutil.copytree(directory, changed)
    (changed / "worker.log").write_text("changed actual artifact bytes")
    with pytest.raises(ValueError, match="inventory/hash"):
        verify_shard(changed)
    unsafe = tmp_path / "unsafe"
    shutil.copytree(directory, unsafe)
    (unsafe / "escape").symlink_to(directory / "request.json")
    with pytest.raises(ValueError, match="symlink"):
        verify_shard(unsafe)
    duplicated = tmp_path / "duplicate-merge"
    with pytest.raises(ValueError, match="conflicting"):
        merge_shards([directory, directory], duplicated)


@pytest.mark.parametrize(
    "corruption",
    ["identification_claim", "accuracy_claim", "missing_product", "hidden_error"],
)
def test_resealed_semantically_corrupted_real_result_is_rejected(
    actual_water, tmp_path, corruption
):
    directory, _ = actual_water
    altered = tmp_path / corruption
    shutil.copytree(directory, altered)
    manifest = read_json(altered / "manifest.json")
    result = read_json(altered / "result.json")
    # Deliberate adversarial metadata corruption of a genuine calculation bundle.
    if corruption == "identification_claim":
        result["identification_ready"] = True
    elif corruption == "accuracy_claim":
        result["experimental_accuracy_established"] = True
    elif corruption == "missing_product":
        stage = result["stages"]["rigid_rotor_catalog"]
        stage.update(
            status="unavailable",
            value=None,
            reason="Deliberately removed.",
            absence_kind="not_computed",
        )
    else:
        result["errors"].append({"code": "DELIBERATELY_ADDED_ERROR"})
    (altered / "result.json").write_bytes(canonical_json(result) + b"\n")
    (altered / "manifest.json").unlink()
    seal_shard(
        altered,
        request_sha256=manifest["request_sha256"],
        request_id=manifest["request_id"],
        recipe_sha256=manifest["recipe_sha256"],
        source_identity=manifest["source_identity"],
        worker_id=manifest["worker_id"],
    )
    with pytest.raises(ValueError, match="Completion contradicts|qualification"):
        verify_shard(altered)


@pytest.mark.parametrize(
    "corruption",
    [
        "isotope_number",
        "isotope_mass",
        "isotope_policy",
        "isotope_reference",
        "mode_count",
        "mode_order",
        "mode_basis",
        "principal_axes",
    ],
)
def test_resealed_genuine_catalog_context_must_match_retained_physical_data(
    actual_water, tmp_path, corruption
):
    directory, _ = actual_water
    altered = tmp_path / corruption
    shutil.copytree(directory, altered)
    manifest = read_json(altered / "manifest.json")
    result = read_json(altered / "result.json")
    catalog = result["stages"]["rigid_rotor_catalog"]
    context = catalog["value"]["scientific_context"]
    # The adversary changes a copy of a genuine engine calculation and reseals
    # it. Schema validity/byte hashes alone must not certify physical identity.
    if corruption == "isotope_number":
        context["isotope_numbers"][1] = 2
    elif corruption == "isotope_mass":
        context["isotope_masses_u"][1] += 0.00001
    elif corruption == "isotope_policy":
        context["isotope_selection_policies"][1] = "explicit_mass_number"
    elif corruption == "isotope_reference":
        context["isotope_reference_sha256"] = "0" * 64
    elif corruption == "mode_count":
        context["mode_count"] -= 1
        context["mode_order"].pop()
    elif corruption == "mode_order":
        context["mode_order"].reverse()
    elif corruption == "mode_basis":
        context["mode_basis_sha256"] = "0" * 64
    else:
        # Sign-flipping two axes is still an orthonormal right-handed frame, so
        # the strict schema passes, while the retained inertia frame disagrees.
        axes = np.asarray(context["frame_axes_columns"])
        axes[:, :2] *= -1
        context["frame_axes_columns"] = axes.tolist()
        context["frame_sha256"] = digest(
            {"frame_type": context["frame_type"], "axes_columns": axes.tolist()}
        )
    StageResult.model_validate(catalog)
    (altered / "result.json").write_bytes(canonical_json(result) + b"\n")
    (altered / "manifest.json").unlink()
    seal_shard(
        altered,
        request_sha256=manifest["request_sha256"],
        request_id=manifest["request_id"],
        recipe_sha256=manifest["recipe_sha256"],
        source_identity=manifest["source_identity"],
        worker_id=manifest["worker_id"],
    )
    with pytest.raises(
        ValueError, match="isotope provenance|mode identities|principal frame"
    ):
        verify_shard(altered)


def test_cli_exact_request_sha_rejection_before_calculation(tmp_path):
    request = read_json(ROOT / "examples/student/water-hf-teaching.json")
    path = tmp_path / "request.json"
    path.write_bytes(canonical_json(request) + b"\n")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.cli",
            "execute",
            "--request",
            str(path),
            "--output-dir",
            str(output),
            "--expected-sha256",
            sha256(canonical_json(request)).hexdigest(),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "transport digest mismatch" in completed.stdout
    assert not output.exists()


def test_unqualified_requested_stages_preserve_real_earlier_results(tmp_path):
    request = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request["resources"]["cores"] = 1
    request["products"] = [
        "geometry",
        "harmonic",
        "ground_state_constants",
        "identification_catalog",
    ]
    result = execute_request(request, tmp_path / "partial")
    assert result["status"] == "partial"
    assert result["stages"]["equilibrium_geometry"]["status"] == "available"
    assert result["stages"]["harmonic_analysis"]["status"] == "available"
    for stage in ("ground_state_constants", "identification_catalog"):
        assert result["stages"][stage]["status"] == "blocked"
        assert result["stages"][stage]["value"] is None
    verify_shard(tmp_path / "partial")


def test_actual_owned_worker_timeout_has_no_scientific_fallback(tmp_path):
    request = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request["resources"]["wall_seconds"] = 1
    result = execute_request(request, tmp_path / "timed-out")
    assert result["status"] in {"failed", "partial"}
    assert any(
        error.get("code") == "WALL_TIMEOUT"
        for error in result["errors"]
        if isinstance(error, dict)
    )
    assert result["stages"]["identification_catalog"]["value"] is None
    verify_shard(tmp_path / "timed-out")


@pytest.mark.parametrize(
    "products,expected_status",
    [
        (["geometry", "harmonic", "anharmonic_force_field"], "complete"),
        (["vpt2"], "partial"),
    ],
)
def test_actual_application_anharmonic_validation_keeps_rotational_gates(
    tmp_path, products, expected_status
):
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[-0.8, 0, 0], [0.8, 0, 0]],
            "charge": 0,
            "multiplicity": 1,
        },
        "recipe": "hf-sto-3g-anharmonic-validation",
        "products": products,
        "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 120},
    }
    result = execute_request(request, tmp_path / "experimental-h2")
    assert result["status"] == expected_status, result["errors"]
    field = result["stages"]["anharmonic_force_field"]
    assert field["status"] == "available" and field["value"]["derivative_converged"]
    assert "experimental_unqualified" in field["quality_flags"]
    vpt2 = result["stages"]["vpt2"]
    assert vpt2["observable"] == "vibrational_only_vpt2"
    actual_vpt2 = result["anharmonic_validation"]["stages"]["vibrational_vpt2"]
    assert vpt2["status"] == actual_vpt2["status"]
    assert vpt2["value"] == actual_vpt2["value"]
    if vpt2["status"] == "available":
        assert vpt2["value"]["rotation_vibration_available"] is False
        assert vpt2["value"]["independent_scientific_qualification"] is False
    else:
        assert vpt2["reason"] == actual_vpt2["reason"]
        assert "perturbative applicability gate" in vpt2["reason"]
        diagnostics = result["stages"]["resonance_analysis"]["value"]["resonances"]
        assert diagnostics
        assert all(item["kind"] == "strong_anharmonic_coupling" for item in diagnostics)
    assert result["stages"]["ground_state_constants"]["value"] is None
    assert result["resolved_isotopes"][0]["label"] == "1H"
    assert result["resolved_isotopes"][0]["source"]["database_sha256"]
    verify_shard(tmp_path / "experimental-h2")
