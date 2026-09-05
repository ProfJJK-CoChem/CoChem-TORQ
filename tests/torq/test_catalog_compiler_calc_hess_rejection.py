"""
Test Catalog Compiler Calc_Hess Rejection
SRS Chunk 09, Suggestion #86 (Method Matrix v4 §8B.3)
Zero-Mock compliant: Real input deck validation, unconditional exception checks.
"""
import pytest

from cochem_base.exceptions import MethodMatrixViolationError
from Libraries.cochem_catalog_compiler import validate_orca_deck


def test_rejection_of_calc_hess_true_standalone():
    """Pass input deck with standalone Calc_Hess true and assert MethodMatrixViolationError."""
    deck = """! B3LYP def2-TZVP Opt
%geom
  Calc_Hess true
end
* xyz 0 1
O 0.000 0.000 0.000
H 0.000 0.757 0.586
H 0.000 -0.757 0.586
*
"""
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        validate_orca_deck(deck)
    assert "Calc_Hess true" in str(exc_info.value)
    assert "strictly prohibited" in str(exc_info.value)


def test_rejection_of_calc_hess_true_with_inhess():
    """Pass input deck with both InHess XTB2 and Calc_Hess true and assert unconditional rejection."""
    deck = """! B3LYP def2-TZVP Opt
%geom
  InHess XTB2
  Calc_Hess true
end
* xyz 0 1
O 0.000 0.000 0.000
H 0.000 0.757 0.586
H 0.000 -0.757 0.586
*
"""
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        validate_orca_deck(deck)
    assert "Calc_Hess true" in str(exc_info.value)
    assert "strictly prohibited" in str(exc_info.value)


def test_allow_inhess_without_calc_hess():
    """Verify that legitimate model Hessian decks (InHess XTB2 or Lindh) pass cleanly."""
    deck = """! B3LYP def2-TZVP defgrid3 Opt
%geom
  InHess XTB2
end
* xyz 0 1
O 0.000 0.000 0.000
H 0.000 0.757 0.586
H 0.000 -0.757 0.586
*
"""
    result = validate_orca_deck(deck, preliminary_opt=False, production_opt=True)
    assert result is True
