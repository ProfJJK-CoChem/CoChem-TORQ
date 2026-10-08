"""Canonical Watson operator checks and genuine, fail-closed native integration.

Independent reference expressions follow the actually retrieved NITROGEN source
at commit 85ab1d0c965b2f613fc810a394de0c2006bd6f6e. The implementation uses
operator sums, not these reference expressions. Mathematical potentials are
explicit analytic models, never fabricated electronic-structure observations.
"""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest

from cochem_torq.spectroscopy.forcefield import build_force_field
from cochem_torq.spectroscopy.harmonic import (
    HarmonicResult,
    analyze_hessian,
    artifact_digest,
    equilibrium_rotor,
)
from cochem_torq.spectroscopy.rovibrational import build_rovibrational_precursors
from cochem_torq.spectroscopy.rovibrational_perturbation import (
    WatsonCorrectionProtocol,
    WatsonVibrationRotationResult,
    calculate_watson_vibration_rotation,
    canonical_watson_mass_scaling_probe,
    harmonic_distortion_operator,
    watson_volume_identity,
)
from cochem_torq.spectroscopy.rovibrational_solver import (
    StationaryReference,
    body_fixed_angular_momentum,
)
from cochem_torq.units import ATOMIC_MASS_ELECTRON, HARTREE_CM1, HARTREE_JOULE, h
from tests.test_rovibrational_solver import (
    _context,
    _field_data,
    _mathematical_case,
)
from tests.test_rovibrational_solver import (
    genuine_water_field as _genuine_water_field,
)

genuine_water_field = _genuine_water_field


def _reference_alpha(precursors, field):
    """Closed expressions independent of production oscillator-state sums."""
    inertia = np.asarray(precursors.inertia_electron_mass_bohr2).diagonal()
    derivative = np.asarray(precursors.inertia_first_derivative)
    zeta = np.asarray(precursors.coriolis_zeta)
    frequency = np.asarray(field.frequencies_hartree)
    cubic = np.asarray(field.cubic_hartree)
    equilibrium = 1 / (2 * inertia)
    count = len(frequency)
    harmonic, coriolis, anharmonic = (np.zeros((count, 3)) for _ in range(3))
    for mode in range(count):
        for axis in range(3):
            harmonic[mode, axis] = (
                -2
                * equilibrium[axis] ** 2
                / frequency[mode]
                * sum(
                    3
                    * derivative[axis, other_axis, mode] ** 2
                    / (4 * inertia[other_axis])
                    for other_axis in range(3)
                )
            )
            coriolis[mode, axis] = (
                -2
                * equilibrium[axis] ** 2
                / frequency[mode]
                * sum(
                    zeta[axis, mode, other] ** 2
                    * (3 * frequency[mode] ** 2 + frequency[other] ** 2)
                    / (frequency[mode] ** 2 - frequency[other] ** 2)
                    for other in range(count)
                    if other != mode
                )
            )
            anharmonic[mode, axis] = -(equilibrium[axis] ** 2) * sum(
                derivative[axis, axis, other]
                * cubic[mode, mode, other]
                / frequency[other] ** 1.5
                for other in range(count)
            )
    return harmonic + coriolis + anharmonic


def _reference_semirigid_energies(precursors, field):
    """Independent g0/xij formulas from the retained open reference source."""
    w = np.asarray(field.frequencies_hartree)
    phi3 = np.asarray(field.cubic_hartree)
    phi4 = np.asarray(field.quartic_hartree)
    zeta = np.asarray(precursors.coriolis_zeta)
    rotational = 1 / (2 * np.diag(precursors.inertia_electron_mass_bohr2))
    count = len(w)

    def delta(i, j, k):
        return (
            (w[i] + w[j] + w[k])
            * (w[i] - w[j] - w[k])
            * (-w[i] + w[j] - w[k])
            * (-w[i] - w[j] + w[k])
        )

    g0 = sum(
        phi4[i, i, i, i] / 64 - 7 * phi3[i, i, i] ** 2 / (576 * w[i])
        for i in range(count)
    )
    for i in range(count):
        for j in range(count):
            if i != j:
                g0 += 3 / 64 * w[j] * phi3[i, i, j] ** 2 / (4 * w[i] ** 2 - w[j] ** 2)
        for j in range(i):
            for k in range(j):
                g0 -= w[i] * w[j] * w[k] * phi3[i, j, k] ** 2 / (4 * delta(i, j, k))
    for axis in range(3):
        g0 -= rotational[axis] / 4
        g0 -= (
            rotational[axis]
            / 2
            * sum(zeta[axis, i, j] ** 2 for i in range(count) for j in range(i))
        )
    x = np.zeros((count, count))
    for i in range(count):
        x[i, i] = phi4[i, i, i, i] / 16 - 5 * phi3[i, i, i] ** 2 / (48 * w[i])
        x[i, i] -= sum(
            phi3[i, i, k] ** 2
            * (8 * w[i] ** 2 - 3 * w[k] ** 2)
            / (16 * w[k] * (4 * w[i] ** 2 - w[k] ** 2))
            for k in range(count)
            if k != i
        )
        for j in range(i):
            value = phi4[i, i, j, j] / 4
            value -= phi3[i, i, j] ** 2 * w[i] / (2 * (4 * w[i] ** 2 - w[j] ** 2))
            value -= phi3[i, j, j] ** 2 * w[j] / (2 * (4 * w[j] ** 2 - w[i] ** 2))
            value -= phi3[i, i, i] * phi3[i, j, j] / (4 * w[i])
            value -= phi3[j, j, j] * phi3[i, i, j] / (4 * w[j])
            for k in range(count):
                if k in (i, j):
                    continue
                value += (
                    phi3[i, j, k] ** 2
                    * w[k]
                    * (w[i] ** 2 + w[j] ** 2 - w[k] ** 2)
                    / (2 * delta(i, j, k))
                )
                value -= phi3[i, i, k] * phi3[j, j, k] / (4 * w[k])
            value += sum(
                rotational[a] * zeta[a, i, j] ** 2 * (w[i] / w[j] + w[j] / w[i])
                for a in range(3)
            )
            x[i, j] = x[j, i] = value
    states = (tuple([0] * count),) + tuple(
        tuple(int(i == j) for i in range(count)) for j in range(count)
    )
    energies = []
    for state in states:
        occupation = np.asarray(state) + 0.5
        energies.append(
            g0
            + np.dot(occupation, w)
            + sum(
                x[i, j] * occupation[i] * occupation[j]
                for i in range(count)
                for j in range(i + 1)
            )
        )
    return np.asarray(energies)


def _mathematical_anharmonic_case(*, frequencies=None, phase=None, cubic_scale=1.0):
    precursors, old_field = _mathematical_case()
    coordinates = np.asarray(precursors.input_geometry_bohr)
    masses = np.asarray(precursors.identity.isotope_masses_u)
    frequency = np.asarray(old_field.frequencies_hartree)
    weighted = np.asarray(precursors.input_mass_weighted_modes)
    cartesian = weighted / np.sqrt(np.repeat(masses * ATOMIC_MASS_ELECTRON, 3))[:, None]
    transform = cartesian / np.sqrt(frequency)[None, :]
    harmonic = HarmonicResult(
        coordinates,
        masses,
        frequency * HARTREE_CM1,
        frequency,
        weighted,
        cartesian,
        transform,
        frequency**2,
        6,
        precursors.harmonic_external_residual,
        0.0,
        "positive_definite_vibrational_hessian",
        float(frequency.sum() / 2),
        precursors.identity.harmonic_source_sha256,
    )
    physical_transform = transform.copy()
    if frequencies is not None:
        frequency = np.asarray(frequencies)
        mass_vector = np.repeat(masses * ATOMIC_MASS_ELECTRON, 3)
        hessian = (weighted * frequency**2) @ weighted.T
        hessian *= np.sqrt(mass_vector[:, None] * mass_vector[None, :])
        harmonic = analyze_hessian(coordinates, masses, hessian)
        frequency = harmonic.angular_frequencies_au
        transform = np.asarray(harmonic.dimensionless_to_cartesian)
        physical_transform = transform.copy()
    if phase is not None:
        from dataclasses import replace

        phase = np.asarray(phase)
        harmonic = replace(
            harmonic,
            mass_weighted_modes=harmonic.mass_weighted_modes * phase,
            cartesian_modes=harmonic.cartesian_modes * phase,
            dimensionless_to_cartesian=harmonic.dimensionless_to_cartesian * phase,
        )
    rotor = equilibrium_rotor(coordinates, masses)
    context = _context(
        ["O", "H", "H"],
        harmonic,
        rotor,
        evidence_class="mathematical_model",
        parents=[artifact_digest({"analytic_harmonic_source": harmonic.source_digest})],
        recipe={
            "analytic": "explicit coupled cubic/quartic polynomial",
            "cubic_scale": cubic_scale,
        },
    )

    def potential(geometry):
        q = np.linalg.lstsq(
            physical_transform, (geometry - coordinates).ravel(), rcond=None
        )[0]
        value = 0.5 * np.dot(frequency, q * q)
        value += cubic_scale * (
            (8e-6 * q[0] ** 3 - 1e-5 * q[1] ** 3 + 5e-6 * q[2] ** 3) / 6
            + (3e-6 * q[0] ** 2 * q[1] - 2e-6 * q[0] * q[2] ** 2) / 2
            + 4e-6 * q[0] * q[1] * q[2]
        )
        value += 2e-5 * np.sum(q**4) / 24 + 3e-6 * q[0] ** 2 * q[1] ** 2 / 4
        return float(value)

    field = build_force_field(
        harmonic,
        potential,
        evaluator_identity="declared analytic weak coupled polynomial",
        absolute_tolerance_hartree=1e-7,
    )
    typed = _field_data(field, context)
    precursors = build_rovibrational_precursors(harmonic, rotor, context)
    reference = StationaryReference(
        source_artifact_sha256=precursors.identity.parent_artifact_sha256[0],
        geometry_sha256=context.geometry_sha256,
        evidence_class="mathematical_model",
        gradient_hartree_bohr=((0.0, 0.0, 0.0),) * len(masses),
        maximum_allowed_gradient_hartree_bohr=1e-12,
    )
    return precursors, typed, reference


@pytest.fixture(scope="module")
def mathematical_anharmonic_case():
    return _mathematical_anharmonic_case()


def _calculate(case, **kwargs):
    precursors, field, reference = case
    return calculate_watson_vibration_rotation(
        precursors, field, stationary_reference=reference, **kwargs
    )


def test_ladder_alpha_matches_independent_closed_watson_formulas(
    mathematical_anharmonic_case,
):
    precursors, field, _ = mathematical_anharmonic_case
    result = _calculate(mathematical_anharmonic_case)
    assert result.model_alpha_tensors_mhz is not None
    alpha = np.asarray(result.algebraic_alpha_tensors_hartree)
    assert np.allclose(
        np.diagonal(alpha, axis1=1, axis2=2),
        _reference_alpha(precursors, field),
        rtol=1e-11,
        atol=1e-20,
    )
    ground = np.asarray(result.states[0].algebraic_coefficient_tensor_hartree)
    equilibrium = np.asarray(result.equilibrium_coefficient_tensor_hartree)
    assert np.allclose(
        ground - equilibrium, -0.5 * alpha.sum(axis=0), rtol=1e-11, atol=1e-20
    )
    assert np.allclose(
        result.ground_state_constants_mhz,
        np.diag(ground) * HARTREE_JOULE / h / 1e6,
        rtol=1e-12,
    )
    assert not result.identification_ready


def test_semirigid_state_energies_match_independent_g0_xij_expressions(
    mathematical_anharmonic_case,
):
    precursors, field, _ = mathematical_anharmonic_case
    result = _calculate(mathematical_anharmonic_case)
    assert result.semirigid_vpt2 is not None
    expected = _reference_semirigid_energies(precursors, field)
    assert np.allclose(
        result.semirigid_vpt2.state_energies_hartree, expected, rtol=1e-12, atol=1e-15
    )
    assert np.allclose(
        result.semirigid_vpt2.fundamental_frequencies_cm1,
        (expected[1:] - expected[0]) * HARTREE_CM1,
        atol=1e-8,
    )
    assert all(
        value >= 0
        for value in result.semirigid_vpt2.coriolis_kinetic_first_order_hartree
    )
    assert result.semirigid_vpt2.watson_volume_scalar_hartree < 0
    assert not result.semirigid_vpt2.identification_ready
    assert not result.full_resonant_GVPT2


def test_normal_mode_phase_changes_preserve_physical_model_predictions(
    mathematical_anharmonic_case,
):
    original = _calculate(mathematical_anharmonic_case)
    changed_case = _mathematical_anharmonic_case(phase=(-1.0, 1.0, -1.0))
    changed = _calculate(changed_case)
    assert changed.precursor_sha256 != original.precursor_sha256
    assert changed.scientific_context.mode_basis_sha256 != (
        original.scientific_context.mode_basis_sha256
    )
    assert np.allclose(
        changed.algebraic_alpha_tensors_hartree,
        original.algebraic_alpha_tensors_hartree,
        rtol=1e-10,
        atol=1e-18,
    )
    assert np.allclose(
        changed.ground_state_constants_mhz,
        original.ground_state_constants_mhz,
        rtol=1e-11,
    )
    assert changed.semirigid_vpt2 is not None
    assert original.semirigid_vpt2 is not None
    assert np.allclose(
        changed.semirigid_vpt2.state_energies_hartree,
        original.semirigid_vpt2.state_energies_hartree,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.allclose(
        changed.harmonic_distortion.unreduced_tau_hartree,
        original.harmonic_distortion.unreduced_tau_hartree,
        rtol=1e-11,
        atol=1e-23,
    )


def test_exact_fermi_resonance_keeps_missing_energy_and_explicit_block_reason():
    result = _calculate(
        _mathematical_anharmonic_case(frequencies=(0.009, 0.018, 0.029))
    )
    affected = result.states[2]
    assert affected.vibrational_state == (0, 1, 0)
    assert affected.cubic_second_order_hartree is None
    assert affected.algebraic_semirigid_energy_hartree is None
    assert affected.semirigid_energy_status == "blocked"
    assert affected.equilibrium_axis_model_constants_mhz is None
    assert any("Fermi" in reason for reason in affected.blocking_reasons)
    assert any(
        "denominator" in reason for reason in affected.semirigid_energy_blocking_reasons
    )
    assert result.semirigid_vpt2 is None
    assert result.model_alpha_tensors_mhz is None
    assert result.algebraic_alpha_tensors_hartree is not None
    assert result.harmonic_distortion.status == "available_unreduced_harmonic_model"


def test_tiny_cubic_resonance_below_analysis_floor_still_blocks_isolated_constants():
    result = _calculate(
        _mathematical_anharmonic_case(
            frequencies=(0.009, 0.018, 0.029), cubic_scale=1e-8
        )
    )
    affected = result.states[2]
    assert affected.maximum_observed_anharmonic_coupling_ratio > 0.1
    assert affected.cubic_second_order_hartree is None
    assert affected.model_constants_status == "blocked"
    assert affected.equilibrium_axis_model_constants_mhz is None
    assert any(
        "cubic vibrational denominator" in value for value in affected.blocking_reasons
    )
    assert result.model_alpha_tensors_mhz is None
    assert result.semirigid_vpt2 is None
    assert not result.independent_scientific_qualification


def test_degenerate_harmonic_modes_need_block_and_do_not_repair_coriolis_denominator():
    result = _calculate(_mathematical_anharmonic_case(frequencies=(0.01, 0.01, 0.019)))
    affected = result.states[1:3]
    assert all(state.model_constants_status == "blocked" for state in affected)
    assert all(
        any("near-degenerate" in reason for reason in state.blocking_reasons)
        for state in affected
    )
    blocked = [
        contribution
        for state in affected
        for contribution in state.virtual_state_contributions
        if contribution.operator == "coriolis_second_order"
        and contribution.contribution_tensor_hartree is None
    ]
    assert blocked
    assert all(abs(value.denominator_hartree) < 1e-14 for value in blocked)
    assert all("Coriolis denominator" in value.blocking_reason for value in blocked)
    assert all(state.coriolis_response_tensor_hartree is None for state in affected)
    assert result.algebraic_alpha_tensors_hartree is None
    assert result.model_alpha_tensors_mhz is None
    assert result.semirigid_vpt2 is None
    assert not result.full_resonant_GVPT2


def test_strong_cubic_model_fails_default_applicability_but_retains_diagnostics():
    result = _calculate(_mathematical_anharmonic_case(cubic_scale=1000.0))
    assert result.protocol.maximum_anharmonic_coupling_ratio == 0.1
    assert result.protocol.maximum_coriolis_coupling_ratio == 0.1
    assert result.protocol.maximum_relative_rotational_correction == 0.1
    assert result.model_alpha_tensors_mhz is None
    assert result.ground_state_constants_mhz is None
    assert result.semirigid_vpt2 is None
    assert result.algebraic_alpha_tensors_hartree is not None
    assert all(state.model_constants_status == "blocked" for state in result.states)
    assert all(
        state.algebraic_semirigid_energy_hartree is not None for state in result.states
    )
    assert (
        max(state.maximum_observed_anharmonic_coupling_ratio for state in result.states)
        > 0.1
    )


@pytest.mark.parametrize("identity", ["source", "geometry", "evidence"])
def test_actual_stationary_gradient_must_bind_to_the_same_source_and_geometry(
    mathematical_anharmonic_case, identity
):
    precursors, field, reference = mathematical_anharmonic_case
    changed = {
        "source": {"source_artifact_sha256": "e" * 64},
        "geometry": {"geometry_sha256": "e" * 64},
        "evidence": {"evidence_class": "engine_calculation"},
    }[identity]
    with pytest.raises(ValueError):
        calculate_watson_vibration_rotation(
            precursors,
            field,
            stationary_reference=reference.model_copy(update=changed),
        )


@pytest.mark.parametrize(
    "coordinate", [(0.0, 0.0, 0.0), (1.0, -2.0, 0.5), (3.0, 0.2, -1.0)]
)
def test_fixed_measure_volume_scalar_independently_equals_watson_trace(
    mathematical_anharmonic_case, coordinate
):
    derived, watson = watson_volume_identity(
        mathematical_anharmonic_case[0], coordinate
    )
    assert derived == pytest.approx(watson, rel=1e-12, abs=1e-20)


@pytest.mark.parametrize("angular_momentum", [0, 1, 2, 3, 4, 5])
def test_unreduced_distortion_tensor_and_response_square_have_same_operator(
    mathematical_anharmonic_case, angular_momentum
):
    result = _calculate(mathematical_anharmonic_case)
    operators = body_fixed_angular_momentum(angular_momentum)
    tau = np.asarray(result.harmonic_distortion.unreduced_tau_hartree)
    direct = np.zeros((2 * angular_momentum + 1,) * 2, dtype=complex)
    for a, b, c, d in product(range(3), repeat=4):
        direct += (
            tau[a, b, c, d]
            * operators[a]
            @ operators[b]
            @ operators[c]
            @ operators[d]
            / 4
        )
    actual = harmonic_distortion_operator(result.harmonic_distortion, angular_momentum)
    assert np.allclose(actual, direct, rtol=1e-12, atol=1e-20)
    assert np.allclose(actual, actual.conj().T, atol=1e-20)
    assert np.max(np.linalg.eigvalsh(actual)) <= 1e-19
    assert result.harmonic_distortion.A_reduced_constants is None
    assert result.harmonic_distortion.S_reduced_constants is None


def test_harmonic_response_matches_independent_tau_reference(
    mathematical_anharmonic_case,
):
    precursors, field, _ = mathematical_anharmonic_case
    result = _calculate(mathematical_anharmonic_case)
    inertia = np.diag(precursors.inertia_electron_mass_bohr2)
    first = np.asarray(precursors.inertia_first_derivative)
    frequency = np.asarray(field.frequencies_hartree)
    expected = np.zeros((3, 3, 3, 3))
    for a, b, c, d in product(range(3), repeat=4):
        expected[a, b, c, d] = (
            -0.5
            * sum(
                first[a, b, k] * first[c, d, k] / frequency[k] ** 2
                for k in range(len(frequency))
            )
            / (inertia[a] * inertia[b] * inertia[c] * inertia[d])
        )
    assert np.allclose(
        result.harmonic_distortion.unreduced_tau_hartree,
        expected,
        rtol=1e-12,
        atol=1e-25,
    )


def _oscillator_basis(count, maximum_quanta):
    return tuple(
        state
        for state in product(range(maximum_quanta + 1), repeat=count)
        if sum(state) <= maximum_quanta
    )


def test_canonical_mass_probe_converges_to_leading_rotational_operator(
    mathematical_anharmonic_case,
    tmp_path,
):
    precursors, field, _ = mathematical_anharmonic_case
    result = _calculate(mathematical_anharmonic_case)
    basis = _oscillator_basis(3, 5)
    operators = body_fixed_angular_momentum(1)
    equilibrium = np.asarray(result.equilibrium_coefficient_tensor_hartree)
    correction = (
        np.asarray(result.states[0].algebraic_coefficient_tensor_hartree) - equilibrium
    )
    reference_operator = sum(
        equilibrium[a, b] * operators[a] @ operators[b]
        for a, b in product(range(3), repeat=2)
    )
    correction_operator = sum(
        correction[a, b] * operators[a] @ operators[b]
        for a, b in product(range(3), repeat=2)
    )
    levels, vectors = np.linalg.eigh(reference_operator)
    expected = np.real(np.diag(vectors.conj().T @ correction_operator @ vectors))
    errors = []
    comparisons = []
    for epsilon in (0.3, 0.2, 0.1):
        vibration = canonical_watson_mass_scaling_probe(
            precursors, field, basis_states=basis, J=0, epsilon=epsilon
        )
        rotation = canonical_watson_mass_scaling_probe(
            precursors, field, basis_states=basis, J=1, epsilon=epsilon
        )
        vibration_levels, vibration_vectors = np.linalg.eigh(vibration)
        rotation_levels, rotation_vectors = np.linalg.eigh(rotation)
        lowest_vibration = vibration_levels[0]
        lowest_rotation = rotation_levels[:3]
        observed = (
            (lowest_rotation - lowest_vibration) / epsilon**4 - levels
        ) / epsilon**2
        errors.append(float(np.max(np.abs(observed - expected))))
        np.savez_compressed(
            tmp_path / f"mass-probe-epsilon-{epsilon:.1f}.npz",
            vibration_hamiltonian=vibration,
            vibration_eigenvalues=vibration_levels,
            vibration_eigenvectors=vibration_vectors,
            rotation_hamiltonian=rotation,
            rotation_eigenvalues=rotation_levels,
            rotation_eigenvectors=rotation_vectors,
        )
        comparisons.append(
            {
                "epsilon": epsilon,
                "ground_vibrational_energy_hartree": float(lowest_vibration),
                "ground_J1_energies_hartree": lowest_rotation.tolist(),
                "observed_leading_rotational_response_hartree": observed.tolist(),
                "maximum_absolute_response_error_hartree": errors[-1],
            }
        )
    (tmp_path / "mass-probe-comparison.json").write_text(
        json.dumps(
            {
                "evidence_class": "mathematical_model",
                "model": "canonical finite Galerkin Watson H2-H6 polynomial",
                "force_field_sha256": result.force_field_sha256,
                "precursor_sha256": result.precursor_sha256,
                "algorithm_source_sha256": result.algorithm_source_sha256,
                "basis_states": basis,
                "expected_rotational_response_hartree": expected.tolist(),
                "comparisons": comparisons,
                "independent_scientific_qualification": False,
            },
            indent=2,
        )
    )
    (tmp_path / "mathematical-force-field.json").write_text(field.model_dump_json())
    (tmp_path / "mathematical-precursors.json").write_text(precursors.model_dump_json())
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 2e-8


@pytest.mark.parametrize("parameter", ["maximum_modes", "maximum_intermediate_states"])
def test_operator_resource_caps_stop_before_unbounded_work(
    mathematical_anharmonic_case, parameter
):
    with pytest.raises(ValueError, match="budget"):
        _calculate(
            mathematical_anharmonic_case,
            protocol=WatsonCorrectionProtocol(**{parameter: 1}),
        )


@pytest.mark.parametrize(
    "corruption",
    ["tensor", "constants", "alpha", "availability", "energy", "qualification"],
)
def test_result_readback_cannot_upgrade_or_replace_source_bound_products(
    mathematical_anharmonic_case, corruption
):
    result = _calculate(mathematical_anharmonic_case)
    data = json.loads(result.model_dump_json())
    if corruption == "tensor":
        data["states"][0]["algebraic_coefficient_tensor_hartree"][0][0] += 0.1
    elif corruption == "constants":
        data["ground_state_constants_mhz"][0] += 1.0
    elif corruption == "alpha":
        data["algebraic_alpha_tensors_hartree"][0][0][0] += 0.01
    elif corruption == "availability":
        data["states"][0]["blocking_reasons"] = ["actual applicability failed"]
    elif corruption == "energy":
        data["semirigid_vpt2"]["state_energies_hartree"][0] += 0.1
    else:
        data["identification_ready"] = True
    with pytest.raises(ValueError):
        WatsonVibrationRotationResult.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "substitution",
    [
        "force_field",
        "precursor",
        "context",
        "compensated_child_terms",
        "child_energy",
        "rotation_availability",
        "state_energy_terms",
        "state_energy_status",
        "virtual_response",
        "virtual_occupation",
        "stricter_protocol",
    ],
)
def test_resealed_records_still_cross_bind_children_terms_and_actual_protocol(
    mathematical_anharmonic_case, substitution
):
    result = _calculate(mathematical_anharmonic_case)
    data = json.loads(result.model_dump_json())
    child = data["semirigid_vpt2"]
    if substitution == "force_field":
        child["force_field_sha256"] = "e" * 64
    elif substitution == "precursor":
        child["precursor_sha256"] = "e" * 64
    elif substitution == "context":
        child["scientific_context"]["recipe_sha256"] = "e" * 64
    elif substitution == "compensated_child_terms":
        child["quartic_first_order_hartree"][0] += 0.001
        child["cubic_second_order_hartree"][0] -= 0.001
    elif substitution == "child_energy":
        child["state_energies_hartree"] = [
            energy + 0.001 for energy in child["state_energies_hartree"]
        ]
        child["ground_energy_hartree"] += 0.001
        child["quartic_first_order_hartree"] = [
            energy + 0.001 for energy in child["quartic_first_order_hartree"]
        ]
    elif substitution == "rotation_availability":
        child["rotation_vibration_model_available"] = False
    elif substitution == "state_energy_terms":
        data["states"][0]["quartic_first_order_hartree"] += 0.001
    elif substitution == "state_energy_status":
        data["states"][0]["semirigid_energy_status"] = "blocked"
    elif substitution == "virtual_response":
        record = data["states"][0]["virtual_state_contributions"][0]
        record["contribution_tensor_hartree"][0][0] += 1e-8
        record["numerator_tensor_hartree2"][0][0] = (
            record["contribution_tensor_hartree"][0][0] * record["denominator_hartree"]
        )
    elif substitution == "virtual_occupation":
        data["states"][0]["virtual_state_contributions"][0]["virtual_state"][0] += 1
    else:
        data["protocol"]["maximum_anharmonic_coupling_ratio"] = (
            data["states"][0]["maximum_observed_anharmonic_coupling_ratio"] / 2
        )
    data["content_sha256"] = artifact_digest(
        {name: value for name, value in data.items() if name != "content_sha256"}
    )
    with pytest.raises(ValueError):
        WatsonVibrationRotationResult.model_validate_json(json.dumps(data))


@pytest.mark.real_engine
def test_genuine_field_matches_independent_formulas_and_keeps_failed_applicability(
    genuine_water_field,
):
    precursors, field, directory, reference = genuine_water_field
    result = calculate_watson_vibration_rotation(
        precursors, field, stationary_reference=reference
    )
    assert np.allclose(
        np.diagonal(result.algebraic_alpha_tensors_hartree, axis1=1, axis2=2),
        _reference_alpha(precursors, field),
        rtol=1e-11,
        atol=1e-20,
    )
    expected_energy = _reference_semirigid_energies(precursors, field)
    assert np.allclose(
        [state.algebraic_semirigid_energy_hartree for state in result.states],
        expected_energy,
        rtol=1e-12,
        atol=1e-15,
    )
    assert result.ground_state_constants_mhz is None
    assert result.model_alpha_tensors_mhz is None
    assert result.semirigid_vpt2 is None
    assert all(state.blocking_reasons for state in result.states)
    assert result.states[0].maximum_observed_anharmonic_coupling_ratio > 0.1
    assert any(
        "Fermi" in reason
        for state in result.states
        for reason in state.blocking_reasons
    )
    assert result.harmonic_distortion.status == "available_unreduced_harmonic_model"
    assert not result.identification_ready
    (directory / "canonical-watson-leading.json").write_text(
        result.model_dump_json(indent=2)
    )


@pytest.mark.real_engine
def test_genuine_complete_mode_volume_identity_at_displacements(genuine_water_field):
    precursors, _, _, _ = genuine_water_field
    for coordinate in [(0.0, 0.0, 0.0), (1.0, -2.0, 0.5), (3.0, 0.2, -1.0)]:
        derived, watson = watson_volume_identity(precursors, coordinate)
        assert derived == pytest.approx(watson, rel=1e-12, abs=1e-20)
