"""IPC mass APIs use actual table values and never substitute isotope identity."""

import pytest
from mendeleev import element

from Libraries.cochem_torq_ipc import get_atomic_mass, get_isotopic_mass


@pytest.mark.parametrize("symbol", ["H", "C", "O", "Og"])
def test_ipc_atomic_mass_preserves_actual_standard_weight_semantics(symbol):
    actual_table = element(symbol)
    assert actual_table.atomic_weight is not None
    assert get_atomic_mass(symbol) == float(actual_table.atomic_weight)


@pytest.mark.parametrize(("symbol", "mass_number"), [("H", 2), ("C", 13), ("O", 18)])
def test_ipc_isotopic_mass_selects_the_actual_requested_table_row(symbol, mass_number):
    actual_table = element(symbol)
    actual_isotope = next(
        isotope
        for isotope in actual_table.isotopes
        if isotope.mass_number == mass_number
    )
    assert actual_isotope.mass is not None
    mass = get_isotopic_mass(symbol, mass_number)
    assert mass == float(actual_isotope.mass)
    assert mass != float(actual_table.atomic_weight)
    assert mass != float(mass_number)


@pytest.mark.parametrize("symbol", ["H", "C", "Og"])
def test_ipc_missing_isotope_is_never_replaced_by_weight_or_integer(symbol):
    assert all(isotope.mass_number != 999 for isotope in element(symbol).isotopes)
    with pytest.raises(ValueError, match="No tabulated mass for requested isotope"):
        get_isotopic_mass(symbol, 999)


@pytest.mark.parametrize("mass_number", [True, False, 0, -1, 1.0, "1", None])
def test_ipc_requested_isotope_requires_a_positive_integer_identity(mass_number):
    with pytest.raises(ValueError, match="positive integer mass number"):
        get_isotopic_mass("H", mass_number)


@pytest.mark.parametrize("symbol", [None, True, 1, "", "   "])
def test_ipc_physical_mass_lookup_rejects_missing_or_nontext_element(symbol):
    with pytest.raises(ValueError, match="requires an element symbol"):
        get_atomic_mass(symbol)
    with pytest.raises(ValueError, match="requires an element symbol"):
        get_isotopic_mass(symbol, 1)


def test_ipc_element_case_normalization_keeps_the_same_requested_isotope():
    assert get_atomic_mass(" c ") == get_atomic_mass("C")
    assert get_isotopic_mass(" c ", 13) == get_isotopic_mass("C", 13)


def test_ipc_unknown_element_cannot_produce_a_mass():
    with pytest.raises(ValueError):
        get_atomic_mass("UnknownElement")
    with pytest.raises(ValueError):
        get_isotopic_mass("UnknownElement", 1)
