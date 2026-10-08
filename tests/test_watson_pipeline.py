"""Actual worker-to-Watson workflow and native-parent integrity acceptance.

The requested HF/STO-3G water ground constants remain absent when the measured
default applicability bound fails. These tests execute actual quantum engines
and retain their artifacts; they create no substitute scientific observations.
"""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest

from cochem_torq.application import source_identity, worker_execute
from cochem_torq.domain import CalculationRequest, StageResult, canonical_json, digest
from cochem_torq.scan import _verify_native
from cochem_torq.spectroscopy.advanced_products import UnreducedHarmonicDistortion
from cochem_torq.spectroscopy.results import ForceFieldData, ResonanceAnalysisData
from cochem_torq.spectroscopy.rovibrational import RovibrationalPrecursors
from cochem_torq.spectroscopy.rovibrational_perturbation import (
    WatsonVibrationRotationResult,
)
from tests.test_rovibrational_perturbation import (
    _reference_alpha,
    _reference_semirigid_energies,
)


@pytest.fixture(scope="module")
def actual_water_watson_worker(tmp_path_factory):
    directory = tmp_path_factory.mktemp("genuine-water-watson-worker")
    request = CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["O", "H", "H"],
                "geometry_bohr": [
                    [0.0, 0.0, 0.0],
                    [0.0, -1.45, 1.12],
                    [0.0, 1.45, 1.12],
                ],
                "atom_ids": ["water-oxygen", "water-hydrogen-a", "water-hydrogen-b"],
                "isotopes": [16, 1, 1],
                "charge": 0,
                "multiplicity": 1,
            },
            "recipe": "hf-sto-3g-anharmonic-validation",
            "products": ["ground_state_constants"],
            "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 300},
        }
    )
    before = source_identity()
    result = worker_execute(request, directory)
    after = source_identity()
    (directory / "request.json").write_bytes(
        canonical_json(request.model_dump(mode="json"))
    )
    (directory / "worker-returned.json").write_bytes(canonical_json(result))
    (directory / "source-stability.json").write_bytes(
        canonical_json(
            {
                "before": before,
                "after": after,
                "source_stable": before == after,
                "independent_scientific_qualification": False,
            }
        )
    )
    assert before == after, "Scientific source changed during actual worker execution."
    assert result["source_identity"] == before
    return request, result, directory


@pytest.mark.real_engine
def test_genuine_worker_retains_typed_watson_route_and_blocked_b0(
    actual_water_watson_worker,
):
    request, result, directory = actual_water_watson_worker
    assert result["request_sha256"] == digest(request.model_dump(mode="json"))
    assert result["status"] == "partial", result.get("errors")
    assert result["identification_ready"] is False
    assert result["experimental_accuracy_established"] is False
    stages = result["stages"]
    for name in (
        "electronic_structure",
        "equilibrium_geometry",
        "harmonic_analysis",
        "equilibrium_constants",
        "anharmonic_force_field",
        "resonance_analysis",
        "rovibrational_precursors",
        "vibration_rotation_corrections",
        "centrifugal_distortion",
    ):
        assert stages[name]["status"] == "available", (
            name,
            stages[name],
            result["errors"],
        )
        StageResult.model_validate(stages[name])
    field = ForceFieldData.model_validate(stages["anharmonic_force_field"]["value"])
    ResonanceAnalysisData.model_validate(stages["resonance_analysis"]["value"])
    precursor = RovibrationalPrecursors.model_validate_json(
        canonical_json(stages["rovibrational_precursors"]["value"])
    )
    watson = WatsonVibrationRotationResult.model_validate_json(
        canonical_json(stages["vibration_rotation_corrections"]["value"])
    )
    distortion = UnreducedHarmonicDistortion.model_validate_json(
        canonical_json(stages["centrifugal_distortion"]["value"])
    )
    assert field.derivative_converged
    assert field.evaluation_count == 107
    assert watson.force_field_sha256 == field.source_digest
    assert watson.scientific_context == field.scientific_context
    assert watson.precursor_sha256 == digest(precursor.model_dump(mode="json"))
    assert distortion.distortion == watson.harmonic_distortion
    assert precursor.identity.atom_ids == tuple(request.molecule.atom_ids)
    assert field.scientific_context.isotope_numbers == [16, 1, 1]
    assert field.scientific_context.evidence_class == "engine_calculation"
    assert field.scientific_context.frame_type == "principal_inertia"
    assert watson.protocol.maximum_anharmonic_coupling_ratio == 0.1
    assert watson.states[0].maximum_observed_anharmonic_coupling_ratio > 0.1
    assert watson.ground_state_constants_mhz is None
    assert watson.model_alpha_tensors_mhz is None
    assert watson.semirigid_vpt2 is None
    assert stages["ground_state_constants"]["status"] == "blocked"
    assert stages["ground_state_constants"]["value"] is None
    assert stages["ground_state_constants"]["reason"]
    assert stages["semirigid_vpt2"]["status"] == "blocked"
    assert stages["semirigid_vpt2"]["value"] is None
    assert not watson.independent_scientific_qualification
    assert not watson.identification_ready
    assert watson.algebraic_alpha_tensors_hartree is not None
    assert np.allclose(
        np.diagonal(watson.algebraic_alpha_tensors_hartree, axis1=1, axis2=2),
        _reference_alpha(precursor, field),
        rtol=1e-11,
        atol=1e-20,
    )
    assert np.allclose(
        [state.algebraic_semirigid_energy_hartree for state in watson.states],
        _reference_semirigid_energies(precursor, field),
        rtol=1e-12,
        atol=1e-15,
    )
    assert any(
        error["code"] == "MODEL_GROUND_STATE_CONSTANTS_UNAVAILABLE"
        for error in result["errors"]
        if isinstance(error, dict)
    )
    research = result["anharmonic_validation"]
    assert research["rotation_vibration_available"] is False
    assert research["independent_scientific_qualification"] is False
    assert research["identification_ready"] is False
    for name in (
        "anharmonic_force_field",
        "resonance_analysis",
        "rovibrational_precursors",
        "vibration_rotation_corrections",
        "centrifugal_distortion",
        "semirigid_vpt2",
        "ground_state_constants",
    ):
        assert stages[name]["value"] == research["stages"][name]["value"]
    assert stages["vibration_rotation_corrections"]["parents"] == [
        "rovibrational_precursors",
        "anharmonic_force_field",
    ]
    assert stages["centrifugal_distortion"]["parents"] == [
        "vibration_rotation_corrections"
    ]
    assert stages["ground_state_constants"]["parents"] == [
        "vibration_rotation_corrections",
        "centrifugal_distortion",
    ]
    saved = json.loads((directory / "result.json").read_text())
    assert saved["stages"] == stages


@pytest.mark.real_engine
def test_genuine_worker_verifies_every_native_parent_and_reference_gradient(
    actual_water_watson_worker,
):
    _, result, directory = actual_water_watson_worker
    research = result["anharmonic_validation"]
    research_directory = directory / "anharmonic-research"
    watson = WatsonVibrationRotationResult.model_validate_json(
        canonical_json(result["stages"]["vibration_rotation_corrections"]["value"])
    )
    field = ForceFieldData.model_validate(
        result["stages"]["anharmonic_force_field"]["value"]
    )
    observations = research["displaced_calculations"]
    assert len(observations) == field.evaluation_count == 107
    genuine_parents = set()
    for observation in observations:
        native, actual_manifest = _verify_native(
            research_directory / observation["workspace"]
        )
        assert native["status"] == observation["status"] == "complete"
        assert actual_manifest == observation["manifest_sha256"]
        assert observation["geometry_sha256"] == digest(native["geometry_bohr"])
        assert observation["energy_hartree"] == native["energy_hartree"]
        genuine_parents.add(actual_manifest)
    genuine_parents.update(
        sha256((research_directory / name).read_bytes()).hexdigest()
        for name in ("protocol.json", "harmonic-input.json")
    )
    assert set(field.scientific_context.parent_artifact_sha256) == genuine_parents
    reference_row = next(row for row in observations if row["id"] == 0)
    native_reference, reference_manifest = _verify_native(
        research_directory / reference_row["workspace"]
    )
    assert watson.stationary_reference.source_artifact_sha256 == reference_manifest
    assert reference_manifest in field.scientific_context.parent_artifact_sha256
    assert np.array_equal(
        watson.stationary_reference.gradient_hartree_bohr,
        native_reference["gradient_hartree_bohr"],
    )
    assert watson.stationary_reference.geometry_sha256 == digest(
        native_reference["geometry_bohr"]
    )
    assert (
        np.max(np.abs(watson.stationary_reference.gradient_hartree_bohr))
        <= (research["protocol"]["reference_gradient_max_hartree_bohr"])
    )
    correction_path = research_directory / "watson-corrections.json"
    assert (
        WatsonVibrationRotationResult.model_validate_json(correction_path.read_text())
        == watson
    )
    correction_sha = sha256(correction_path.read_bytes()).hexdigest()
    distortion = UnreducedHarmonicDistortion.model_validate_json(
        canonical_json(result["stages"]["centrifugal_distortion"]["value"])
    )
    assert distortion.watson_result_artifact_sha256 == correction_sha
    assert distortion.scientific_context.parent_artifact_sha256 == [correction_sha]
    for name in (
        "recipe_sha256",
        "protocol_sha256",
        "geometry_sha256",
        "mode_basis_sha256",
        "frame_sha256",
    ):
        assert getattr(distortion.scientific_context, name) == getattr(
            field.scientific_context, name
        )
    manifest_path = research_directory / "research-manifest.json"
    assert (
        sha256(manifest_path.read_bytes()).hexdigest()
        == research["artifact_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        actual = research_directory / record["path"]
        assert actual.stat().st_size == record["size_bytes"]
        assert sha256(actual.read_bytes()).hexdigest() == record["sha256"]
    assert not (research_directory / "model-ground-state-constants.json").exists()
    assert not (research_directory / "semirigid-vpt2.json").exists()
