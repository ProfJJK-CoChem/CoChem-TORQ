"""Integrity checks using archived ORCA artifacts and explicit damaged copies.

The repository's test.engrad/test.property.txt are parser evidence, not a new
electronic-structure calculation or an accuracy benchmark. Corrupted copies
below exercise software rejection paths; they are never scientific results.
No engine executable, calculation output, or wavefunction is simulated.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from pydantic import ValidationError
import pytest

from Libraries.cochem_torq_engine import (
    DispatchPayload,
    ExecutionContext,
    ORCAParseError,
    ORCAStepResult,
    _finite_number,
    _orca_scf_converged,
    _parse_orca_engrad_or_output,
    _read_orca_engrad,
    _read_orca_final_coordinates,
    _source_artifact_metadata,
    dynamic_wavefunction_propagation,
    get_isotopic_mass,
    gpu4pyscf_dynamic_batching,
    opi_persistent_threading,
    route_cascade_rules,
    route_method_matrix,
    stateful_scf_checkpointing,
    validate_spin_contamination,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ENGRAD = REPOSITORY / "test.engrad"
PROPERTY = REPOSITORY / "test.property.txt"
SYMBOLS = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]


@pytest.fixture
def archived_values():
    return _read_orca_engrad(ENGRAD, len(SYMBOLS))


def test_archived_engrad_parses_observed_energy_gradient_and_geometry(archived_values):
    energy, gradient, numbers, coordinates = archived_values
    assert energy == pytest.approx(-229.626787795591, abs=1e-12)
    assert gradient.shape == (10, 3)
    assert gradient[0].tolist() == pytest.approx([-0.027031066927, 0.100391996481, 0.035466569352])
    assert np.linalg.norm(gradient) == pytest.approx(1.6406344557043946, abs=1e-10)
    assert numbers == [6, 6, 8, 8, 1, 1, 1, 1, 1, 1]
    assert coordinates.shape == (10, 3)
    assert coordinates[0].tolist() == pytest.approx([-0.732, 0.385, 0], abs=5e-8)


def test_archived_convergence_requires_explicit_property_evidence():
    assert _orca_scf_converged("", PROPERTY) is True
    assert _orca_scf_converged("ORCA TERMINATED NORMALLY") is False
    energy, gradient, converged = _parse_orca_engrad_or_output(ENGRAD, "", 10)
    assert energy < 0
    assert np.linalg.norm(gradient) > 1
    assert converged is True


def test_engrad_does_not_imply_scf_or_geometry_convergence(tmp_path):
    path = tmp_path / "isolated.engrad"
    shutil.copyfile(ENGRAD, path)
    _, _, converged = _parse_orca_engrad_or_output(path, "ORCA TERMINATED NORMALLY", 10)
    assert converged is False


@pytest.mark.parametrize("replacement", ["NaN", "Infinity", "-Inf", "unavailable"])
def test_corrupt_energy_is_rejected(tmp_path, replacement):
    path = tmp_path / "damaged.engrad"
    path.write_text(ENGRAD.read_text().replace("-229.626787795591", replacement))
    with pytest.raises(ORCAParseError):
        _parse_orca_engrad_or_output(path, "", 10)


@pytest.mark.parametrize("damage", ["missing_energy", "missing_gradient", "truncated_gradient", "nonfinite_gradient", "truncated_geometry", "nonfinite_geometry"])
def test_missing_and_damaged_sections_are_rejected(tmp_path, damage):
    content = ENGRAD.read_text()
    if damage == "missing_energy":
        content = content.replace("# The current total energy in Eh", "# unavailable energy section")
    elif damage == "missing_gradient":
        content = content.replace("# The current gradient in Eh/bohr", "# unavailable gradient section")
    elif damage == "truncated_gradient":
        content = content.replace("      -0.027031066927\n", "")
    elif damage == "nonfinite_gradient":
        content = content.replace("-0.027031066927", "NaN")
    elif damage == "truncated_geometry":
        content = "\n".join(content.splitlines()[:-1])
    elif damage == "nonfinite_geometry":
        content = content.replace("-1.3832795", "NaN")
    path = tmp_path / "damaged.engrad"
    path.write_text(content)
    with pytest.raises(ORCAParseError):
        _parse_orca_engrad_or_output(path, "", 10)


def test_wrong_atom_count_and_empty_output_are_rejected(tmp_path):
    with pytest.raises(ORCAParseError, match="atom count"):
        _read_orca_engrad(ENGRAD, 9)
    with pytest.raises(ORCAParseError, match="Missing final ORCA energy"):
        _parse_orca_engrad_or_output(tmp_path / "absent.engrad", "ORCA TERMINATED NORMALLY", 10)


def test_observed_geometry_is_read_and_atom_mapping_checked(archived_values):
    coordinates, source = _read_orca_final_coordinates(REPOSITORY / "test", "", SYMBOLS)
    np.testing.assert_array_equal(coordinates, archived_values[3])
    assert source == str(ENGRAD)
    with pytest.raises(ORCAParseError, match="atom mapping"):
        _read_orca_final_coordinates(REPOSITORY / "test", "", ["N", *SYMBOLS[1:]])


def test_missing_final_geometry_is_not_replaced(tmp_path):
    with pytest.raises(ORCAParseError, match="input geometry was not substituted"):
        _read_orca_final_coordinates(tmp_path / "absent", "", SYMBOLS)


def test_result_has_no_energy_or_convergence_defaults(archived_values):
    energy, gradient, _, coordinates = archived_values
    with pytest.raises(ValidationError):
        ORCAStepResult(coordinates=coordinates)
    result = ORCAStepResult(energy=energy, coordinates=coordinates, gradient=gradient)
    assert result.converged is False
    assert result.scf_converged is False
    assert result.optimization_converged is None
    assert result.frequencies is None
    assert result.dipole_moment is None
    with pytest.raises(ValidationError, match="explicit electronic convergence"):
        ORCAStepResult(energy=energy, coordinates=coordinates, converged=True)


def test_nonfinite_result_and_geometry_are_rejected(archived_values):
    energy, gradient, _, coordinates = archived_values
    with pytest.raises(ValidationError):
        ORCAStepResult(energy=float("nan"), coordinates=coordinates)
    damaged_gradient = gradient.copy()
    damaged_gradient[0, 0] = np.nan
    with pytest.raises(ValidationError):
        ORCAStepResult(energy=energy, coordinates=coordinates, gradient=damaged_gradient)
    damaged_coordinates = coordinates.copy()
    damaged_coordinates[0, 0] = np.inf
    with pytest.raises(ValidationError):
        DispatchPayload(symbols=SYMBOLS, coordinates=damaged_coordinates)
    with pytest.raises(ValidationError, match="atom symbol"):
        DispatchPayload(symbols=SYMBOLS[:-1], coordinates=coordinates)


def test_missing_wavefunction_never_generates_engine_checkpoint(tmp_path, archived_values):
    energy, _, _, coordinates = archived_values
    context = ExecutionContext(custom_shm_dir=tmp_path / "shm")
    result = ORCAStepResult(energy=energy, coordinates=coordinates, converged=True,
                            scf_converged=True, normally_terminated=True)
    payload = DispatchPayload(symbols=SYMBOLS, coordinates=coordinates)
    with pytest.raises(ValueError, match="authentic ORCA GBW"):
        dynamic_wavefunction_propagation(result, payload, context)
    assert not (tmp_path / "shm").exists()


def test_missing_element_or_unknown_method_never_substitutes(tmp_path, archived_values):
    context = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
    coordinates = archived_values[3]
    with pytest.raises(ValueError, match="Atom symbols are required"):
        route_cascade_rules(coordinates, context)
    with pytest.raises(ValueError, match="Unsupported method tier"):
        route_method_matrix(SYMBOLS, coordinates, target_tier="unregistered", context=context)
    with pytest.raises(ValueError, match="isotopic mass"):
        get_isotopic_mass("C", 999)


def test_wrong_engine_rejected_before_execution(archived_values):
    payload = DispatchPayload(symbols=SYMBOLS, coordinates=archived_values[3], executor="TorqCfourExecutor")
    with pytest.raises(ValueError, match="cannot execute requested executor"):
        next(opi_persistent_threading(payload))


def test_array_archive_is_hdf5_and_not_a_native_engine_file(tmp_path, archived_values):
    context = ExecutionContext(custom_scratch_dir=tmp_path / "scratch")
    path = stateful_scf_checkpointing(1, archived_values[1], context, checkpoint_type="gbw")
    assert path.suffix == ".h5"
    assert not path.with_suffix(".gbw").exists()
    with pytest.raises(TypeError):
        stateful_scf_checkpointing(2, object(), context)


def test_missing_memory_and_spin_observables_are_not_filled(archived_values):
    context = ExecutionContext()
    context.gpu_available = False
    context.vram_mb = 0
    with pytest.raises(ValueError, match="observed GPU memory"):
        gpu4pyscf_dynamic_batching([archived_values[3]], context)
    with pytest.raises(Exception, match="No observed"):
        validate_spin_contamination("", 1, is_unrestricted=False)
    with pytest.raises(ValueError, match="finite"):
        validate_spin_contamination(2, float("nan"))


def test_numeric_parser_preserves_zero_and_scientific_notation():
    assert _finite_number("0", "numeric software input") == 0
    assert _finite_number("-1.5D-3", "numeric software input") == -0.0015


def test_actual_artifact_manifest_binds_engrad_and_property_bytes():
    metadata = _source_artifact_metadata([ENGRAD, PROPERTY])
    digests = metadata["artifact_sha256"]
    assert digests == {
        str(ENGRAD): hashlib.sha256(ENGRAD.read_bytes()).hexdigest(),
        str(PROPERTY): hashlib.sha256(PROPERTY.read_bytes()).hexdigest(),
    }
    canonical_map = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert metadata["source_manifest_sha256"] == hashlib.sha256(canonical_map.encode("utf-8")).hexdigest()
    assert _source_artifact_metadata([PROPERTY, ENGRAD]) == metadata


def test_artifact_provenance_rejects_missing_files_and_detects_changes(tmp_path):
    damaged = tmp_path / "damaged.engrad"
    shutil.copyfile(ENGRAD, damaged)
    before = _source_artifact_metadata([damaged])
    damaged.write_bytes(damaged.read_bytes()[:-1])
    after = _source_artifact_metadata([damaged])
    assert before["source_manifest_sha256"] != after["source_manifest_sha256"]
    with pytest.raises(FileNotFoundError):
        _source_artifact_metadata([tmp_path / "absent.engrad"])
    with pytest.raises(ORCAParseError, match="actual source artifacts"):
        _source_artifact_metadata([])
