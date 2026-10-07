"""Explicit absence and mathematical invariants; no simulated engine outputs."""

from pathlib import Path
import math

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    FortranOverflowError,
    SPCATBridgeError,
    TorqSpcatBridge,
    _fallback_point_group_solver,
    _pg_to_sigma,
    _resolve_nuclear_spin_ratio,
    apply_symmetry_divisors,
    calculate_rotational_constants_from_geometry,
    calculate_rotational_partition_function,
    calculate_vibrational_partition_function,
    compute_coupled_partition_functions,
    format_fortran_double,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_var,
    get_atomic_mass,
    low_frequency_lam_trap,
    route_3tier_abinitio_payload,
    vibrational_partition_coupling,
)
from Libraries.cochem_torq_engine import _read_orca_engrad


ARCHIVE = Path(__file__).resolve().parents[1] / "test.engrad"
SYMBOLS = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]


def test_isotope_masses_are_not_natural_average_substitutes():
    assert get_atomic_mass("C") == get_atomic_mass("12C") == 12
    assert get_atomic_mass("D") == get_atomic_mass("2H")
    assert 13 < get_atomic_mass("13C") < 13.01
    with pytest.raises(ValueError, match="No tabulated mass"):
        get_atomic_mass("999C")
    with pytest.raises(ValueError, match="Specify an isotope"):
        get_atomic_mass("Tc")


def test_rotational_constants_are_invariant_under_translation_and_rotation():
    _, _, _, coordinates = _read_orca_engrad(ARCHIVE, 10)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    original = calculate_rotational_constants_from_geometry(coordinates, SYMBOLS)
    transformed = calculate_rotational_constants_from_geometry(coordinates @ rotation + 7, SYMBOLS)
    for quantity, value in original.items():
        assert transformed[quantity] == pytest.approx(value, rel=1e-12)


def test_linear_axis_has_absent_A_not_finite_sentinel():
    # Mathematical two-point rotor, not a claimed optimized molecular fixture.
    constants = calculate_rotational_constants_from_geometry([[0, 0, -1], [0, 0, 1]], ["12C", "12C"])
    assert constants["A"] is None
    assert constants["B"] == pytest.approx(constants["C"])
    assert constants["Ia"] == pytest.approx(0)
    with pytest.raises(ValueError, match="singular"):
        calculate_rotational_constants_from_geometry([[0, 0, 0], [0, 0, 0]], ["12C", "12C"])


def test_point_group_and_spin_weights_are_never_guessed():
    _, _, _, coordinates = _read_orca_engrad(ARCHIVE, 10)
    with pytest.raises(SPCATBridgeError, match="solver"):
        _fallback_point_group_solver(coordinates, SYMBOLS)
    with pytest.raises(SPCATBridgeError, match="cannot be inferred"):
        _resolve_nuclear_spin_ratio("C2v", SYMBOLS, {})
    with pytest.raises(SPCATBridgeError, match="Unsupported point group"):
        _pg_to_sigma("undetermined")
    with pytest.raises(SPCATBridgeError, match="state-resolved"):
        apply_symmetry_divisors(coordinates, SYMBOLS, use_nuclear_spin=True)


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_temperature_cannot_create_partition_one(temperature):
    with pytest.raises(ValueError):
        calculate_rotational_partition_function(100, 50, 40, temperature)
    with pytest.raises(ValueError):
        calculate_vibrational_partition_function([100], temperature)


def test_invalid_rotational_constants_are_not_clamped():
    with pytest.raises(ValueError, match="A/B/C"):
        calculate_rotational_partition_function(None, 50, 40, 300)
    with pytest.raises(ValueError, match="A/B/C"):
        calculate_rotational_partition_function(0, 50, 40, 300)
    with pytest.raises(ValueError, match="symmetry"):
        calculate_rotational_partition_function(100, 50, 40, 300, sigma=0)


def test_unavailable_partitions_do_not_become_one():
    with pytest.raises(ValueError, match="Missing rotational"):
        vibrational_partition_coupling({}, {300: 1}, [300])
    with pytest.raises(ValueError, match="Missing vibrational"):
        vibrational_partition_coupling({300: 1}, {}, [300])
    with pytest.raises(ValueError, match="one value"):
        vibrational_partition_coupling([], [1], [300])
    with pytest.raises(ValueError, match="unavailable"):
        calculate_vibrational_partition_function(None, 300)


def test_partition_multiplication_is_explicit_mathematics():
    assert vibrational_partition_coupling({300: 2}, {300: 3}, [300]) == {300: 6}
    assert vibrational_partition_coupling([20], [30], [300]) == {300: 600}
    # Values are not guessed to be frequencies from their magnitude.
    with pytest.raises(ValueError, match="explicit all_frequencies"):
        vibrational_partition_coupling([20], [30], [300], lam_frequency=30)


@pytest.mark.parametrize("frequency", [0, -100, np.nan, np.inf])
def test_invalid_vibrational_modes_cannot_be_silently_dropped(frequency):
    with pytest.raises(ValueError):
        calculate_vibrational_partition_function([frequency], 300)
    with pytest.raises(ValueError):
        low_frequency_lam_trap([frequency])


def test_a_dvr_label_requires_actual_dvr_input():
    with pytest.raises(ValueError, match="no DVR spectrum input"):
        compute_coupled_partition_functions(100, 50, 40, [100], [300], is_dvr=True)
    with pytest.raises(ValueError, match="not present"):
        calculate_vibrational_partition_function([100], 300, [200])


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_fortran_invalid_values_are_never_clamped(value):
    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard(value)
    with pytest.raises(ValueError, match="Clamping"):
        fortran_overflow_guard(value, clamp_on_overflow=True)
    with pytest.raises(FortranOverflowError):
        format_fortran_double(value)


def test_unknown_uncertainty_is_not_filled_in():
    energy = _read_orca_engrad(ARCHIVE, 10)[0]
    assert math.isclose(float(format_fortran_double(energy).replace("D", "E")), energy)
    with pytest.raises(ValueError, match="explicitly supplied"):
        fortran_double_precision_formatter(10000, energy)


def test_unqualified_native_writers_do_not_create_files(tmp_path):
    path = tmp_path / "catalog.var"
    with pytest.raises(NotImplementedError, match="unqualified"):
        generate_spcat_var("integrity_negative_case", {}, filepath=path)
    with pytest.raises(NotImplementedError, match="unqualified"):
        generate_spcat_int("integrity_negative_case", {}, filepath_template=path)
    assert not path.exists()


@pytest.mark.parametrize("invalid_json", ["", "{}", "{", "[]", '{"energy": NaN}'])
def test_invalid_tensor_json_is_not_replaced_by_defaults(tmp_path, invalid_json):
    path = tmp_path / "malformed_tensor.json"
    path.write_text(invalid_json)
    with pytest.raises(ValueError):
        TorqSpcatBridge(path, tmp_path / "missing_engine_output")


def test_archived_energy_does_not_invent_dipoles_frequencies_or_vpt2():
    energy = _read_orca_engrad(ARCHIVE, 10)[0]
    record = {"energy_hartree": energy, "source": str(ARCHIVE)}
    result = route_3tier_abinitio_payload(orca_data=record)
    assert result.electronic_energy_hartree == energy
    assert result.harmonic_frequencies is None
    assert result.dipole_moments_debye is None
    assert result.vpt2_x_matrix is None
    assert result.is_analytic_vpt2_active is False
    assert result.to_dict()["harmonic_frequencies"] is None
    with pytest.raises(ValueError, match="No supplied record meets"):
        route_3tier_abinitio_payload(orca_data=record, require_analytic_vpt2=True)
