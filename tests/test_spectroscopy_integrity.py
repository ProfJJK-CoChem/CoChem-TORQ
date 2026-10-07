"""Analytic rotor, strict contract and actual failing CLI execution checks.

Analytic geometries are mathematical examples, not asserted engine results.
The only electronic values below come from archived repository ORCA artifacts.
No engine, quantum output, or unavailable backend is simulated.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pydantic import ValidationError
import pytest
from scipy.constants import atomic_mass, h

from Libraries.cochem_torq_pipeline import (
    TorqPipeline, normalize_and_validate_payload, pipeline_exit_code,
)
from Libraries.cochem_torq_spectroscopy import (
    EquilibriumConstants, HarmonicAnalysis, Provenance, StageResult,
    UncertaintyAssessment, build_spectroscopy_report,
    equilibrium_constants_from_geometry, resolve_isotopic_masses,
)
from Libraries.torq_config import TorqRunParams


ROOT = Path(__file__).resolve().parents[1]


def config(**updates):
    data = dict(tier="T1", wall_time_tier="normal", engine="ORCA", method="HF",
                basis_set="STO-3G", keywords=["Opt"])
    data.update(updates)
    return TorqRunParams(**data)


def test_diatomic_rigid_rotor_matches_reduced_mass_formula():
    separation = 0.75
    result = equilibrium_constants_from_geometry(["1H", "2H"], [[0, 0, 0], [0, 0, separation]])
    m1, m2 = result.isotope_masses_u
    reduced_mass = m1 * m2 / (m1 + m2)
    inertia_si = reduced_mass * atomic_mass * (separation * 1e-10)**2
    expected_mhz = h / (8 * np.pi**2 * inertia_si * 1e6)
    assert result.rotor_type == "linear"
    assert result.A_mhz is None
    assert result.B_mhz == pytest.approx(expected_mhz, rel=1e-13)
    assert result.C_mhz == pytest.approx(expected_mhz, rel=1e-13)
    assert result.label == "Be"
    assert "null" in result.model_dump_json()


def test_rotor_is_invariant_to_rotation_and_translation():
    coordinates = np.array([[0, 0, 0], [0.9, 0.7, 0], [-0.9, 0.7, 0]])
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    original = equilibrium_constants_from_geometry(["16O", "1H", "1H"], coordinates)
    transformed = equilibrium_constants_from_geometry(["16O", "1H", "1H"], coordinates @ rotation + [2, -7, 11])
    assert [original.A_mhz, original.B_mhz, original.C_mhz] == pytest.approx(
        [transformed.A_mhz, transformed.B_mhz, transformed.C_mhz], rel=1e-13)
    assert original.rotor_type == "nonlinear"


def test_isotope_selection_is_explicit_and_not_average_atomic_weight():
    labels, masses, selected = resolve_isotopic_masses(["C", "H", "D"])
    assert labels == ("12C", "1H", "2H")
    assert selected is True
    assert masses[0] == pytest.approx(12.0, abs=1e-10)
    assert masses[2] > masses[1]
    with pytest.raises(ValueError):
        resolve_isotopic_masses(["Tc"])
    with pytest.raises(ValueError):
        resolve_isotopic_masses(["999C"])


def test_monatomic_rotational_constants_are_undefined():
    result = equilibrium_constants_from_geometry(["4He"], [[3, 4, 5]])
    assert result.rotor_type == "atom"
    assert result.A_mhz is result.B_mhz is result.C_mhz is None


@pytest.mark.parametrize("coordinates", [[], [[0, 0, float("nan")]], [[0, float("inf"), 0]], [[0, 0]]])
def test_malformed_geometry_cannot_produce_constants(coordinates):
    with pytest.raises(ValueError):
        equilibrium_constants_from_geometry(["1H"], coordinates)


def test_coincident_atoms_are_not_a_rotor():
    with pytest.raises(ValueError, match="Coincident"):
        equilibrium_constants_from_geometry(["1H", "1H"], [[0, 0, 0], [0, 0, 0]])


def test_missing_stage_cannot_contain_a_numeric_result():
    mathematical_result = equilibrium_constants_from_geometry(["1H", "1H"], [[0, 0, 0], [0, 0, 1]])
    evidence = Provenance(engine="analytic_rigid_rotor", method="h/(8*pi^2*I)", basis_set="not_applicable",
        parser="analytic", source_sha256=sha256(b"two isotope masses separated by one Angstrom").hexdigest())
    with pytest.raises(ValidationError):
        StageResult[EquilibriumConstants](status="unavailable", value=mathematical_result,
            reason="No calculation exists", provenance=evidence)
    with pytest.raises(ValidationError):
        StageResult[EquilibriumConstants](status="available", provenance=evidence)
    stage = StageResult[EquilibriumConstants](status="unavailable", reason="No calculation exists", provenance=evidence)
    assert stage.value is None
    assert stage.uncertainty.status == "uncalibrated"


def test_uncertainty_cannot_be_calibrated_without_evidence():
    with pytest.raises(ValidationError):
        UncertaintyAssessment(status="calibrated")


def test_harmonic_contract_requires_complete_mode_set():
    with pytest.raises(ValidationError, match="complete vibrational mode set"):
        HarmonicAnalysis(frequencies_cm1=(), expected_vibrational_modes=1, stationary_point="minimum")
    with pytest.raises(ValidationError):
        HarmonicAnalysis(frequencies_cm1=(float("nan"),), expected_vibrational_modes=1, stationary_point="minimum")


def test_archived_electronic_result_survives_missing_optimization_and_hessian():
    from Libraries.cochem_torq_engine import ORCAStepResult, _orca_scf_converged, _read_orca_engrad
    energy, gradient, atomic_numbers, coords = _read_orca_engrad(ROOT / "test.engrad", 10)
    property_path = ROOT / "test.property.txt"
    property_content = property_path.read_text()
    # This archived artifact explicitly reports normal termination of LeanSCF;
    # it carries no evidence of optimization or harmonic characterization.
    result = ORCAStepResult(
        energy=energy, coordinates=coords, gradient=gradient,
        normally_terminated='&Status [&Type "String"] "NORMAL TERMINATION"' in property_content,
        scf_converged=_orca_scf_converged("", property_path),
        coordinates_source=str(ROOT / "test.engrad"), raw_output=property_content,
        metadata={"artifact_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "test.engrad", property_path)}},
    )
    symbols = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]
    report = build_spectroscopy_report(result, symbols, engine="ORCA", method="archived; unverified", basis_set="archived; unverified")
    assert report.electronic_structure.status == "available"
    provenance = report.electronic_structure.provenance
    assert provenance.source_artifacts[str(ROOT / "test.engrad")] == sha256((ROOT / "test.engrad").read_bytes()).hexdigest()
    assert provenance.source_sha256 != sha256(property_content.encode()).hexdigest()
    assert provenance.digest_scope == "raw_artifact_manifest_and_consumed_parsed_result"
    assert report.electronic_structure.value.energy_hartree == pytest.approx(energy)
    assert report.equilibrium_geometry.status == "blocked"
    assert report.equilibrium_constants.value is None
    assert report.harmonic_analysis.value is None
    assert report.dipole_moment.value is None
    assert report.vpt2.status == report.ground_state_constants.status == report.catalog.status == "blocked"
    assert not report.product_available("catalog")
    payload = json.loads(report.model_dump_json())
    assert payload["harmonic_analysis"]["value"] is None
    assert payload["dipole_moment"]["value"] is None


def test_unsupported_engine_is_rejected_before_execution():
    pipeline = TorqPipeline(config(engine="CFOUR", method="CCSD(T)"))
    with pytest.raises(NotImplementedError, match="not implemented"):
        pipeline.run({"symbols": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.75]]})
    assert pipeline.state_history == ["S_0", "S_FAILED"]


def test_unrequested_discovery_is_not_silently_substituted():
    with pytest.raises(NotImplementedError, match="profile"):
        TorqPipeline(config()).run({"symbols": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.75]],
                                   "workflow_profile": "discovery"})


def test_pipeline_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        normalize_and_validate_payload({"symbols": ["H"], "coordinates": [[0, 0, float("nan")]]})


@pytest.mark.parametrize("charge,multiplicity", [(0.5, 1), (0, 0), (0, 2), (True, 1)])
def test_pipeline_rejects_inconsistent_charge_or_multiplicity(charge, multiplicity):
    with pytest.raises(ValueError):
        TorqPipeline(config()).run({"symbols": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.75]],
                                   "charge": charge, "multiplicity": multiplicity})


def test_exit_codes_never_report_partial_or_unknown_as_success():
    assert pipeline_exit_code("success") == 0
    for status in ("partial", "failed", "blocked", "rejected", "cancelled", "unexpected"):
        assert pipeline_exit_code(status) != 0


def test_actual_cli_missing_engine_preserves_input_and_nonzero_exit(tmp_path):
    input_path = tmp_path / "hydrogen.xyz"
    # A blank XYZ comment is valid and must not discard the first atom.
    input_path.write_text("2\n\nH 0 0 0\nH 0 0 0.75\n")
    output = tmp_path / "artifacts"
    env = os.environ.copy()
    env.update(ORCA_PATH=str(tmp_path / "absent-orca"), COCHEM_ROOT=str(ROOT))
    completed = subprocess.run([
        sys.executable, "-m", "Libraries.cochem_torq_pipeline", "--input", str(input_path),
        "--output", str(output), "--scratch", str(tmp_path / "scratch"),
        "--theory", "HF/STO-3G", "--mode", "equilibrium",
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    results = json.loads((output / "pipeline_results.json").read_text())
    assert results["status"] in ("blocked", "failed")
    assert results["structure_artifact"]["geometry_role"] == "input"
    structure = Path(results["structure_artifact"]["path"]).read_text()
    assert "Input geometry; no converged optimization" in structure
    assert "Optimized by" not in structure
    assert not (output / "final_structure.xyz").exists()
    assert "absent-orca" in results["error"]


@pytest.mark.parametrize("field", ["units", "coordinate_units", "coordinates_units"])
def test_input_bohr_cannot_be_silently_relabeled_angstrom(field):
    with pytest.raises(ValueError, match="unsupported units"):
        normalize_and_validate_payload({"symbols": ["H"], "coordinates": [[0, 0, 1]], field: "bohr"})


def test_counterpoise_is_blocked_until_actually_implemented():
    pipeline = TorqPipeline(config(basis_set="def2-TZVP", bsse_correction="CP"))
    with pytest.raises(NotImplementedError, match="BSSE correction"):
        pipeline.run({"symbols": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.75]]})
    assert pipeline.state_history == ["S_0", "S_FAILED"]
