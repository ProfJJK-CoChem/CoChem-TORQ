"""Actual database and declared scalar checks of assembly radius definitions.

These checks validate lookup/measurement contracts, not optimized coordination
structures, intermolecular energies, or a homogeneous empirical radius dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from mendeleev import element

from cochem.mobile.assembly.alignment import calculate_metal_donor_distance
from cochem.mobile.assembly.constants import (
    _validate_atomic_number,
    _validate_positive_property,
    _validate_positive_radius_pm,
    calculate_d_electron_count,
    formal_transition_metal_electron_record,
    get_atomic_number,
    get_covalent_radius_angstrom,
    get_standard_atomic_weight,
    get_vdw_radius_angstrom,
    validate_quantum_parity,
)
from cochem.mobile.assembly.exceptions import MendeleevLookupError, QuantumParityError


@pytest.mark.parametrize("symbol", ["H", "C", "N", "O", "Fe", "Pt"])
def test_radius_values_use_the_exact_named_database_fields(symbol):
    actual = element(symbol)
    assert get_covalent_radius_angstrom(symbol) == actual.covalent_radius_pyykko / 100
    assert get_vdw_radius_angstrom(symbol) == actual.vdw_radius / 100
    assert math.isfinite(get_covalent_radius_angstrom(symbol))
    assert math.isfinite(get_vdw_radius_angstrom(symbol))


def test_missing_named_radius_raises_for_actual_database_entry():
    actual = element("Rf")
    assert actual.vdw_radius is None
    with pytest.raises(MendeleevLookupError, match="vdw_radius is unavailable"):
        get_vdw_radius_angstrom("Rf")


@pytest.mark.parametrize(
    "value", [None, 0, -1.0, math.nan, math.inf, -math.inf, True, "145", 1 << 4096]
)
def test_declared_invalid_radius_scalar_is_rejected(value):
    # This exercises a mathematical/type validator directly; it does not create
    # fake provider records or claim these inputs occur in the actual database.
    with pytest.raises(MendeleevLookupError):
        _validate_positive_radius_pm(value, "declared-input", "covalent_radius_pyykko")


@pytest.mark.parametrize("value", [1, 145.5, np.float64(150.0)])
def test_declared_finite_positive_scalar_validation_preserves_its_value(value):
    assert _validate_positive_radius_pm(value, "declared-input", "vdw_radius") == value


def test_metal_donor_distance_consumer_uses_the_named_pyykko_radius_sum():
    iron = element("Fe")
    nitrogen = element("N")
    assert calculate_metal_donor_distance("Fe", "N") == pytest.approx(
        (iron.covalent_radius_pyykko + nitrogen.covalent_radius_pyykko) / 100,
        rel=1e-15,
        abs=0,
    )


@pytest.mark.parametrize("symbol", ["H", "Fe", "Zn", "Og"])
def test_mass_and_atomic_number_match_actual_database(symbol):
    actual = element(symbol)
    assert get_atomic_number(symbol) == actual.atomic_number
    assert get_standard_atomic_weight(symbol) == actual.atomic_weight


@pytest.mark.parametrize("value", [None, True, 0, -1, 119, 26.0, math.nan])
def test_declared_invalid_atomic_number_scalar_is_rejected(value):
    with pytest.raises(MendeleevLookupError):
        _validate_atomic_number(value, "declared-input")


@pytest.mark.parametrize("value", [None, True, 0, -1, math.nan, math.inf, 1 << 4096])
def test_declared_invalid_atomic_weight_scalar_is_rejected(value):
    with pytest.raises(MendeleevLookupError):
        _validate_positive_property(value, "declared-input", "atomic_weight")


@pytest.mark.parametrize(
    "symbol,oxidation,d_count,formal_count",
    [("Fe", 2, 6, 6), ("Zn", 2, 10, 10), ("Zn", 0, None, 12), ("Sc", 7, None, -4)],
)
def test_formal_model_preserves_raw_count_without_inventing_d_occupancy(
    symbol, oxidation, d_count, formal_count
):
    actual = element(symbol)
    record = formal_transition_metal_electron_record(symbol, oxidation)
    assert record["model"] == "formal_group_minus_oxidation_bookkeeping"
    assert record["formal_valence_electron_count"] == formal_count
    assert formal_count == actual.group_id - oxidation
    assert record["d_electron_count"] == d_count
    assert record["actual_electronic_population_available"] is False
    assert calculate_d_electron_count(symbol, oxidation) == (
        d_count,
        actual.atomic_number - oxidation,
    )
    assert record["d_electron_count_status"] == (
        "formal_bookkeeping" if d_count is not None else "outside_d_shell_range"
    )


@pytest.mark.parametrize("oxidation", [True, False, -1, 8, 2.0, math.nan, math.inf])
def test_invalid_declared_oxidation_does_not_enter_formal_model(oxidation):
    with pytest.raises(ValueError, match="integer"):
        formal_transition_metal_electron_record("Fe", oxidation)
    with pytest.raises(ValueError, match="integer"):
        calculate_d_electron_count("Fe", oxidation)


@pytest.mark.parametrize(
    "electrons,multiplicity",
    [(0, 1), (1, 2), (2, 1), (2, 3), (26, 1), (26, 27), (25, 2)],
)
def test_valid_declared_electron_spin_states(electrons, multiplicity):
    validate_quantum_parity(electrons, multiplicity)


@pytest.mark.parametrize(
    "electrons,multiplicity",
    [
        (True, 2),
        (False, 1),
        (-1, 2),
        (2.0, 1),
        (math.nan, 1),
        (math.inf, 1),
        (2, True),
        (2, 0),
        (2, -1),
        (2, 1.0),
        (2, math.nan),
        (2, math.inf),
        (2, 2),
        (0, 3),
        (2, 5),
    ],
)
def test_invalid_declared_electron_spin_states(electrons, multiplicity):
    with pytest.raises(QuantumParityError):
        validate_quantum_parity(electrons, multiplicity)
