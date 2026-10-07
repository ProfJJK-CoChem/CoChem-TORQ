"""Exact mathematical WKB action checks and rejection of invented rotor inputs."""
import math

import pytest

from Libraries.cochem_torq_slicer import (
    PLANCK_HBAR_SI, WKBTunnelingResult, evaluate_wkb_action_integral,
    get_dynamic_reduced_inertia, wkb_tunneling_estimator,
)


@pytest.mark.parametrize("name", ["CH3", "OH", "arbitrary"])
def test_rotor_name_does_not_supply_inertia(name):
    with pytest.raises(ValueError, match="Cannot derive"):
        get_dynamic_reduced_inertia(name)


def test_inertia_periodicity_and_potential_shape_must_be_explicit():
    with pytest.raises(ValueError, match="potential"):
        wkb_tunneling_estimator("CH3", 1000)
    with pytest.raises(ValueError, match="inertia"):
        wkb_tunneling_estimator("CH3", 1000, periodicity=3, potential_model="cosine")
    with pytest.raises(ValueError, match="periodicity"):
        wkb_tunneling_estimator("CH3", 1000, 3, potential_model="cosine")


def test_constant_barrier_action_matches_exact_integral():
    height, energy, inertia, width = 2e-20, 3e-21, 5e-47, 0.7
    action = evaluate_wkb_action_integral(lambda theta: height, energy, 0, width, inertia)
    expected = width * math.sqrt(2 * inertia * (height - energy)) / PLANCK_HBAR_SI
    assert action == pytest.approx(expected, rel=1e-12)


def test_classically_allowed_or_invalid_intervals_are_rejected():
    with pytest.raises(ValueError, match="allowed"):
        evaluate_wkb_action_integral(lambda theta: 1e-21, 2e-21, 0, 1, 5e-47)
    with pytest.raises(ValueError, match="positive inertia"):
        evaluate_wkb_action_integral(lambda theta: 1e-21, 0, 0, 1, 0)


def test_cosine_model_estimate_is_explicit_and_never_final_splitting():
    result = wkb_tunneling_estimator("label_only", 1000, 3, 3, potential_model="cosine")
    validated = WKBTunnelingResult(**result)
    assert validated.model == "cosine_potential_symmetric_two_well_wkb_estimate"
    assert "not_periodic_rotor_A_E_splitting" in validated.quality_flags
    assert validated.quantum_treatment_required is True
    assert 0 < validated.tunneling_probability < 1
    assert math.log(validated.tunneling_probability) == pytest.approx(validated.log_tunneling_probability)


def test_no_fabricated_below_barrier_energy_for_free_rotor():
    with pytest.raises(ValueError, match="positive barrier"):
        wkb_tunneling_estimator("CH3", 0, 3, 3, potential_model="cosine")
    with pytest.raises(ValueError, match="above the barrier"):
        wkb_tunneling_estimator("CH3", 0.0001, 3, 3, potential_model="cosine")
