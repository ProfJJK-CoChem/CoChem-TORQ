"""Actual local research runner and bounded preflight; no substituted engines."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from cochem_torq.domain import CalculationRequest
from cochem_torq.engines.pyscf_backend import PySCFBackend
from cochem_torq.research_pipeline import (
    execute_anharmonic_validation,
    planned_energy_evaluations,
)
from cochem_torq.spectroscopy import analyze_hessian


@pytest.fixture(scope="module")
def actual_h2_reference(tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    from Libraries.cochem_isotopes import isotope_mass

    backend_request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[-0.8, 0, 0], [0.8, 0, 0]],
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
            "dispersion": None,
        },
        "properties": ["energy", "gradient", "hessian"],
        "settings": {
            "threads": 1,
            "memory_mb": 2048,
            "scf_energy_tolerance": 1e-12,
            "scf_gradient_tolerance": 1e-7,
            "check_stability": True,
        },
    }
    root = tmp_path_factory.mktemp("genuine-h2-research-reference")
    native = PySCFBackend().optimize(backend_request, root / "optimized")
    assert native["status"] == "complete", native.get("errors")
    harmonic = analyze_hessian(
        native["geometry_bohr"],
        [isotope_mass("H")] * 2,
        native["hessian_hartree_bohr2"],
    )
    request = CalculationRequest.model_validate(
        {
            "molecule": backend_request["molecule"],
            "recipe": "hf-sto-3g-anharmonic-validation",
            "products": ["anharmonic_force_field"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 120},
        }
    )
    return request, harmonic, backend_request


def test_exact_preflight_energy_counts():
    assert planned_energy_evaluations(1, (0.08, 0.04)) == 7
    assert planned_energy_evaluations(3, (0.08, 0.04)) == 107


def test_genuine_research_runner_preserves_field_and_blocked_correction(
    actual_h2_reference, tmp_path
):
    request, harmonic, backend_request = actual_h2_reference
    directory = tmp_path / "genuine-field"
    result = execute_anharmonic_validation(
        request, harmonic, backend_request, directory
    )
    assert result["status"] == "experimental_unqualified"
    assert result["outcome"] == "partial"
    assert result["identification_ready"] is False
    assert result["rotation_vibration_available"] is False
    assert result["stages"]["anharmonic_force_field"]["status"] == "available"
    field = result["stages"]["anharmonic_force_field"]["value"]
    assert field["derivative_converged"]
    assert len(result["displaced_calculations"]) == field["evaluation_count"] == 7
    assert result["stages"]["resonance_analysis"]["status"] == "available"
    assert result["stages"]["vibrational_vpt2"]["status"] == "blocked"
    assert result["stages"]["vibrational_vpt2"]["value"] is None
    manifest_path = directory / "research-manifest.json"
    assert (
        sha256(manifest_path.read_bytes()).hexdigest()
        == result["artifact_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text())
    for file in manifest["files"]:
        path = directory / file["path"]
        assert sha256(path.read_bytes()).hexdigest() == file["sha256"]
    for observation in result["displaced_calculations"]:
        path = directory / observation["workspace"] / "manifest.json"
        assert sha256(path.read_bytes()).hexdigest() == observation["manifest_sha256"]
    assert not any("alpha" in key or key == "B0" for key in result)


@pytest.mark.parametrize(
    "failure",
    ["mode_count", "unrequested", "method", "state", "resource", "invariance"],
)
def test_research_preflight_rejects_invalid_scope_without_files(
    actual_h2_reference, tmp_path, failure
):
    request, harmonic, backend_request = actual_h2_reference
    backend_request = deepcopy(backend_request)
    data = request.model_dump(mode="json")
    if failure == "mode_count":
        # Deliberately corrupt the authenticated reference copy. This cannot
        # become physical evidence; the bound rejects it before any work.
        harmonic = replace(harmonic, frequencies_cm1=np.ones(4))
    if failure == "unrequested":
        data["products"] = ["geometry"]
    if failure == "method":
        backend_request["method"]["name"] = "pbe"
    if failure == "state":
        backend_request["molecule"]["charge"] = 2
    if failure == "resource":
        data["resources"]["memory_mb"] = 256
    if failure == "invariance":
        harmonic = replace(harmonic, external_residual_relative=1.0)
    directory = tmp_path / failure
    with pytest.raises(ValueError):
        execute_anharmonic_validation(data, harmonic, backend_request, directory)
    assert not directory.exists()


def test_actual_failed_displaced_scf_retains_raw_results_without_field(
    actual_h2_reference, tmp_path
):
    request, harmonic, backend_request = actual_h2_reference
    backend_request = deepcopy(backend_request)
    backend_request["settings"]["scf_max_cycle"] = 1
    directory = tmp_path / "real-scf-failure"
    result = execute_anharmonic_validation(
        request, harmonic, backend_request, directory
    )
    assert result["outcome"] == "failed"
    assert result["stages"]["anharmonic_force_field"]["status"] == "failed"
    assert result["stages"]["anharmonic_force_field"]["value"] is None
    assert len(result["displaced_calculations"]) == 1
    observed = result["displaced_calculations"][0]
    assert observed["status"] == "failed"
    assert observed["energy_hartree"] is None
    assert (directory / observed["workspace"] / "pyscf.log").stat().st_size > 0
    assert (directory / "research-manifest.json").is_file()


def test_actual_reference_gradient_rejects_displaced_nonstationary_geometry(
    actual_h2_reference, tmp_path
):
    request, harmonic, backend_request = actual_h2_reference
    perturbed = harmonic.coordinates_bohr.copy()
    perturbed[0, 0] -= 0.1
    perturbed[1, 0] += 0.1
    # Deliberately reuse a reference basis at an invalid displaced origin;
    # the independently executed quantum gradient must reject stationarity.
    changed_reference = replace(harmonic, coordinates_bohr=perturbed)
    result = execute_anharmonic_validation(
        request, changed_reference, backend_request, tmp_path / "nonstationary"
    )
    assert result["outcome"] == "failed"
    assert result["stages"]["anharmonic_force_field"]["value"] is None
    observed = result["displaced_calculations"][0]
    assert observed["status"] == "complete"
    assert np.isfinite(observed["energy_hartree"])
    assert (
        observed["gradient_max_hartree_bohr"]
        > result["protocol"]["reference_gradient_max_hartree_bohr"]
    )
    assert "stationarity" in result["errors"][0]["message"]
