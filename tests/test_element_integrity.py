"""Real database checks of local element contracts; no electronic-engine evidence."""
import pytest

from Libraries.cochem_isotopes import isotope_mass
from cochem.core.exceptions import MissingDataError, CoChemError
from cochem.core.mendeleev_invariants import (
    get_element_mass, get_isotope_mass, get_element_cache, mendeleev_resolver,
)


def test_standalone_element_contract_uses_isotope_masses():
    assert get_element_mass("H") == isotope_mass("1H")
    assert get_element_mass(1) == isotope_mass("1H")
    assert get_isotope_mass("H", 2) == isotope_mass("D")
    with pytest.raises(MissingDataError):
        get_isotope_mass("H", 1000)
    with pytest.raises(ValueError, match="Specify an isotope"):
        get_element_mass("Tc")


def test_absent_isotope_abundance_not_zero():
    with pytest.raises(MissingDataError):
        mendeleev_resolver.get_isotope_abundance("H", 1000)
    assert mendeleev_resolver.get_isotope_abundance("H", 3) is None


def test_cache_snapshot_is_independent_and_radii_units_explicit():
    mendeleev_resolver.get_element("H")
    original = get_element_cache()
    assert "H" in original
    original.clear()
    assert "H" in get_element_cache()
    assert mendeleev_resolver.get_vdw_radius_angstrom("H") == mendeleev_resolver.get_vdw_radius("H") / 100


def test_local_error_preserves_diagnostics():
    error = CoChemError("Missing engine", error_code="UNAVAILABLE", engine="unprovisioned")
    assert str(error) == "Missing engine"
    assert error.details["engine"] == "unprovisioned"
