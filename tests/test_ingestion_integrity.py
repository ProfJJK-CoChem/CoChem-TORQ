"""Actual database isotope identity and declared legacy structure input checks.

The Cartesian coordinates below are declared input examples, not engine results.
Every accepted reference mass is read from the installed Mendeleev database.
"""

import json

import pytest
from mendeleev import element

from cochem.core.ingestors.protocols import MolecularStructureData
from Libraries.cochem_path_manager import (
    enforce_ciaaw_masses,
    get_atomic_mass,
    get_isotopic_mass,
)


def atomic_structure(symbol="H", **changes):
    return MolecularStructureData(
        symbols=[symbol], coordinates=[(0.0, 0.0, 0.0)], **changes
    )


def tabulated_mass(symbol, mass_number):
    isotope = next(
        isotope
        for isotope in element(symbol).isotopes
        if isotope.mass_number == mass_number
    )
    assert isotope.mass is not None
    return float(isotope.mass)


@pytest.mark.parametrize("symbol", ["H", "C", "Og"])
def test_missing_requested_isotope_never_becomes_standard_weight_or_mass_number(symbol):
    table = element(symbol)
    assert all(isotope.mass_number != 999 for isotope in table.isotopes)
    with pytest.raises(ValueError, match="No tabulated mass for requested isotope"):
        atomic_structure(symbol, isotopes=[999])


def test_supplied_standard_weight_cannot_bypass_missing_isotope_identity():
    weight = float(element("H").atomic_weight)
    with pytest.raises(ValueError, match="No tabulated mass for requested isotope"):
        atomic_structure("H", isotopes=[999], masses=[weight])


@pytest.mark.parametrize(("symbol", "mass_number"), [("H", 2), ("C", 13)])
def test_supported_isotope_uses_its_actual_tabulated_mass(symbol, mass_number):
    expected = tabulated_mass(symbol, mass_number)
    structure = atomic_structure(symbol, isotopes=[mass_number])
    assert structure.isotopes == [mass_number]
    assert structure.masses == [expected]
    assert structure.mass_selection == "tabulated_requested_isotope_masses"
    assert expected != float(element(symbol).atomic_weight)


def test_supplied_mass_cannot_contradict_a_supported_requested_isotope():
    with pytest.raises(ValueError, match="Supplied mass contradicts"):
        atomic_structure("C", isotopes=[13], masses=[tabulated_mass("C", 12)])


def test_matching_supplied_isotope_mass_is_independently_revalidated_on_readback():
    structure = atomic_structure("C", isotopes=[13], masses=[tabulated_mass("C", 13)])
    payload = json.loads(structure.model_dump_json())
    payload["mass_selection"] = "database_standard_atomic_weights"
    restored = MolecularStructureData.model_validate(payload)
    assert restored.isotopes == [13]
    assert restored.masses == [tabulated_mass("C", 13)]
    assert restored.mass_selection == "tabulated_requested_isotope_masses"


def test_requested_isotope_readback_rejects_a_mass_changed_to_standard_weight():
    structure = atomic_structure("C", isotopes=[13])
    payload = json.loads(structure.model_dump_json())
    payload["masses"] = [float(element("C").atomic_weight)]
    with pytest.raises(ValueError, match="Supplied mass contradicts"):
        MolecularStructureData.model_validate(payload)


def test_unspecified_isotope_can_use_actual_database_standard_weight():
    structure = atomic_structure("C")
    assert structure.isotopes is None
    assert structure.masses == [float(element("C").atomic_weight)]
    assert structure.mass_selection == "database_standard_atomic_weights"


def test_supplied_bare_element_mass_remains_an_explicit_input_not_a_db_claim():
    declared_mass = tabulated_mass("C", 13)
    structure = atomic_structure(
        "C", masses=[declared_mass], mass_selection="database_standard_atomic_weights"
    )
    assert structure.isotopes is None
    assert structure.masses == [declared_mass]
    assert structure.mass_selection == "explicitly_supplied_masses"


def test_bare_element_readback_cannot_authenticate_database_origin_by_a_label():
    structure = atomic_structure("C")
    restored = MolecularStructureData.model_validate_json(structure.model_dump_json())
    assert restored.masses == structure.masses
    assert restored.isotopes is None
    assert restored.mass_selection == "explicitly_supplied_masses"


def test_claimed_explicit_mass_policy_without_mass_input_is_recomputed():
    structure = atomic_structure("C", mass_selection="explicitly_supplied_masses")
    assert structure.mass_selection == "database_standard_atomic_weights"


def test_supplied_mass_cannot_bypass_unknown_element_identity():
    with pytest.raises(ValueError, match="Dynamic element resolution failed"):
        atomic_structure("UnknownElement", masses=[tabulated_mass("H", 1)])


@pytest.mark.parametrize("masses", [[], [1.0, 2.0], (1.0,), [True], [0.0], [-1.0]])
def test_malformed_supplied_mass_vector_fails_without_a_lookup_substitution(masses):
    with pytest.raises(ValueError):
        atomic_structure("H", masses=masses)


@pytest.mark.parametrize("mass", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_declared_mass_is_rejected(mass):
    with pytest.raises(ValueError, match="Invalid positive atomic mass"):
        atomic_structure("H", masses=[mass])


@pytest.mark.parametrize("isotopes", [[], [1, 2], (1,), [True], [0], [-1], [1.0]])
def test_malformed_isotope_identity_is_rejected(isotopes):
    with pytest.raises(ValueError):
        atomic_structure("H", isotopes=isotopes)


def test_empty_structure_is_not_an_untyped_zero_mass_result():
    with pytest.raises(ValueError, match="at least one physical atom"):
        MolecularStructureData(symbols=[], coordinates=[])


def test_existing_coordinate_and_mass_units_remain_angstrom_and_atomic_mass_units():
    requested = [tabulated_mass("H", 1), tabulated_mass("H", 2)]
    structure = MolecularStructureData(
        symbols=["H", "H"],
        coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.75)],
        isotopes=[1, 2],
    )
    assert structure.coordinates == [(0.0, 0.0, 0.0), (0.0, 0.0, 0.75)]
    assert structure.masses == requested
    assert structure.mass_selection == "tabulated_requested_isotope_masses"


@pytest.mark.parametrize("element_identity", ["H", 1])
def test_public_path_manager_rejects_an_isotope_absent_from_actual_table(
    element_identity,
):
    assert all(isotope.mass_number != 999 for isotope in element("H").isotopes)
    with pytest.raises(ValueError, match="No tabulated mass for requested isotope"):
        get_isotopic_mass(element_identity, 999)


def test_public_path_mass_wrapper_cannot_turn_nucleon_number_into_atomic_mass():
    with pytest.raises(ValueError, match="No tabulated mass for requested isotope"):
        enforce_ciaaw_masses(["999H"])


@pytest.mark.parametrize("mass_number", [True, 0, -1, 1.0])
def test_public_path_manager_rejects_invalid_requested_isotope_identity(mass_number):
    with pytest.raises(ValueError, match="positive integer"):
        get_isotopic_mass("H", mass_number)


def test_public_path_manager_no_abundance_requires_explicit_tabulated_isotope():
    actual_table = element("Og")
    assert all(
        isotope.abundance is None or isotope.abundance <= 0
        for isotope in actual_table.isotopes
    )
    with pytest.raises(ValueError, match="no natural-abundance default exists"):
        get_isotopic_mass("Og")
    with pytest.raises(ValueError, match="no natural-abundance default exists"):
        enforce_ciaaw_masses(["Og"])
    selected = next(
        isotope for isotope in actual_table.isotopes if isotope.mass is not None
    )
    assert get_isotopic_mass("Og", selected.mass_number) == float(selected.mass)


def test_public_path_manager_separates_standard_weight_and_isotope_mass():
    assert get_atomic_mass("C") == float(element("C").atomic_weight)
    assert get_isotopic_mass("C", 13) == tabulated_mass("C", 13)
    assert enforce_ciaaw_masses(["D", "T"]).tolist() == [
        tabulated_mass("H", 2),
        tabulated_mass("H", 3),
    ]
