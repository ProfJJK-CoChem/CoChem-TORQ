"""
Test Strict Open-Shell Spin Contamination Halting Gate
SRS Chunk 09, Suggestion #87 (Method Matrix v4 §8B.3)
Zero-Mock compliant: Authentic ORCA output parsing, typed domain exceptions.
"""
import pytest

from cochem_base.exceptions import (
    MissingTelemetryError,
    ProvenanceErrorCode,
    SpinContaminationError,
)
from Libraries.cochem_torq_engine import validate_spin_contamination


def test_spin_contamination_aborts_on_high_contamination():
    """Feed ORCA output with <S^2> = 1.15 for a doublet; assert SpinContaminationError."""
    orca_output_contaminated = """
-------------------------------------------------------------------------------
                            ORCA SCF GRADIENT
-------------------------------------------------------------------------------
Expectation value <S**2> : 1.150000
Ideal value S(S+1)       : 0.750000
-------------------------------------------------------------------------------
"""
    with pytest.raises(SpinContaminationError) as exc_info:
        validate_spin_contamination(orca_output_contaminated, multiplicity=2)

    err = exc_info.value
    assert "Spin contamination" in str(err)
    assert err.error_code in (ProvenanceErrorCode.SPIN_CONTAMINATION_EXCEEDED, ProvenanceErrorCode.ERR_SPIN_CONTAMINATION, "ERR_SPIN_CONTAMINATION")


def test_spin_contamination_passes_on_clean_output():
    """Feed clean output (<S^2> = 0.76; contamination = 1.33%) and assert successful completion."""
    orca_output_clean = """
-------------------------------------------------------------------------------
Expectation value <S**2> : 0.760000
-------------------------------------------------------------------------------
"""
    s2_ideal, s2_obs, rel_dev = validate_spin_contamination(orca_output_clean, multiplicity=2)
    assert abs(s2_obs - 0.76) < 1e-6
    assert abs(s2_ideal - 0.75) < 1e-6
    assert rel_dev < 10.0


def test_missing_spin_telemetry_in_unrestricted_calculation():
    """Assert MissingTelemetryError if <S^2> is absent in unrestricted open-shell calculation."""
    orca_output_no_spin = """
-------------------------------------------------------------------------------
TOTAL RUN TIME: 0 days 0 hours 1 minutes 23 seconds
-------------------------------------------------------------------------------
"""
    with pytest.raises(MissingTelemetryError):
        validate_spin_contamination(orca_output_no_spin, multiplicity=2, is_unrestricted=True)
