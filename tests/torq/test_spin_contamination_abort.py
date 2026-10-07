"""Spin-diagnostic scalar mathematics and real archived missing telemetry.

Scalar inputs here test the acceptance formula only. They are not invented ORCA
outputs, native spin observations, or evidence of a qualified open-shell method.
"""

from pathlib import Path

import pytest

from Libraries.cochem_torq_engine import (
    EngineExecutionError,
    SpinContaminationError,
    validate_spin_contamination,
)


def test_spin_contamination_aborts_on_high_contamination():
    with pytest.raises(SpinContaminationError, match="spin contamination"):
        validate_spin_contamination(multiplicity=2, s_squared=1.15)


def test_spin_contamination_passes_on_clean_output():
    ideal, scalar, deviation = validate_spin_contamination(
        multiplicity=2, s_squared=0.76
    )
    assert ideal == 0.75  # S=1/2: S(S+1), exact spin algebra.
    assert scalar == 0.76
    assert deviation == pytest.approx(100 * abs(0.76 - 0.75) / 0.75)


def test_missing_spin_telemetry_in_unrestricted_calculation():
    archive = Path(__file__).resolve().parents[2] / "test.engrad"
    # The tracked native gradient archive contains no spin diagnostic. Its
    # existence cannot supply an unobserved expectation value for a doublet.
    with pytest.raises(EngineExecutionError, match="No observed"):
        validate_spin_contamination(
            archive.read_text(), multiplicity=2, is_unrestricted=True
        )
