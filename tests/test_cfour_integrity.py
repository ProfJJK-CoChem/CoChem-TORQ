"""Fail-closed contracts using the real bridge, without simulated engine output.

These are schema, absence-handling and storage tests, not CFOUR physics tests.
Live positive-result qualification requires archived genuine engine fixtures.
"""

import json

import h5py
import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_cfour_bridge import (
    CFOUREnergies,
    CFOURNuclearQuadrupole,
    CFOUROutputParser,
    CFOUROutputPayload,
    CFOURUnavailableError,
    CFOURZmatBuilder,
    TorqCfourExecutor,
    export_cfour_to_spcat_dict,
    save_cfour_to_hdf5,
)


@pytest.mark.parametrize("missing_output", ["", " ", "\n\n"])
def test_absent_output_never_becomes_physical_data_or_success(missing_output):
    payload = CFOUROutputParser.parse_full_output(missing_output)
    assert payload.calculation_converged is None
    assert payload.calc_method is None
    assert payload.basis_set is None
    assert payload.optimized_coordinates is None
    for section in (
        payload.energies,
        payload.rotational_constants,
        payload.quartic_distortion,
        payload.sextic_distortion,
        payload.vibrational_data,
        payload.dipole_moment,
    ):
        assert all(value is None for value in section.model_dump().values())
    assert payload.validation_issues


def test_unknown_convergence_cannot_export_ground_state_catalog():
    payload = CFOUROutputParser.parse_full_output("")
    with pytest.raises(ValueError, match="positively verified"):
        export_cfour_to_spcat_dict(payload)


def test_unimplemented_gradient_adapter_rejects_before_starting_process():
    with pytest.raises(CFOURUnavailableError, match="frequency vector is not a Hessian"):
        TorqCfourExecutor().execute(None)


def test_missing_explicit_binary_is_not_replaced_by_any_engine(tmp_path):
    with pytest.raises(CFOURUnavailableError, match="not configured or executable"):
        TorqCfourExecutor(str(tmp_path / "not-installed-xcfour")).resolve_binary()


def test_absent_isotope_quadrupole_data_remain_unavailable():
    record = CFOURNuclearQuadrupole(atom_index=1, element="N")
    assert record.isotope_mass_number is None
    assert record.Q_mbarn is None
    assert record.efg_tensor_au is None
    assert record.chi_tensor_kHz is None
    assert record.chi_aa_kHz is None
    assert record.asymmetry_eta is None


def test_unqualified_quadrupole_metadata_rejected_without_parsing_results():
    with pytest.raises(ValueError, match="isotope_by_atom"):
        CFOUROutputParser.parse_quadrupole_coupling("", nuclear_q_mbarn={"N": 1})


def test_nonfinite_energy_cannot_enter_result_schema():
    with pytest.raises(ValidationError):
        CFOUREnergies(final_energy_hartree=float("nan"))
    energies = CFOUREnergies()
    with pytest.raises(ValidationError):
        energies.scf_energy_hartree = float("inf")


def test_no_method_or_basis_is_assumed_for_input_deck():
    with pytest.raises(TypeError):
        CFOURZmatBuilder.generate_full_zmat_input(symbols=[], coordinates=[])


def test_hdf5_round_trip_keeps_nulls_and_immutable_attempts(tmp_path):
    payload = CFOUROutputParser.parse_full_output("")
    path = tmp_path / "missing-cfour-output.h5"
    save_cfour_to_hdf5(payload, path)
    with h5py.File(path, "r") as handle:
        group = handle["ab_initio/cfour"]
        assert "final_energy_hartree" not in group.attrs
        assert "A0_MHz" not in group.attrs
        assert "harmonic_frequencies_cm1" not in group
        saved = json.loads(group["payload_json"].asstr()[()])
        recovered = CFOUROutputPayload.model_validate(saved)
        assert recovered == payload
    with pytest.raises(ValueError, match="immutable"):
        save_cfour_to_hdf5(payload, path)
