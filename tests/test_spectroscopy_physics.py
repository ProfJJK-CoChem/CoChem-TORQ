"""Numerical physics tests; analytic model potentials are not engine evidence.

No external engine is substituted. These tests compare independent mathematical
limits, reject missing/invalid physical data, and exercise actual displacement
evaluation. Live electronic-structure validation is a separate release gate.
"""

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, h, k, physical_constants, pi

from cochem_torq.spectroscopy import (
    EnergyEvaluation,
    analyze_hessian,
    analyze_isotopologues,
    analyze_resonances,
    apply_rotation_vibration_correction,
    build_force_field,
    equilibrium_rotor,
    rigid_rotor_catalog,
    rigid_rotor_levels,
    solve_vibrational_polyad,
    vibrational_vpt2,
    wigner_3j,
)
from cochem_torq.spectroscopy.harmonic import HARTREE_CM1, artifact_digest

H_MASS = 1.00782503223
D_MASS = 2.01410177812


def diatomic_hessian(spring=0.4):
    geometry = np.array([[-0.7, 0, 0], [0.7, 0, 0]])
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = spring
    hessian[0, 3] = hessian[3, 0] = -spring
    return geometry, hessian


def test_diatomic_frequency_independent_reduced_mass_and_external_projection():
    geometry, hessian = diatomic_hessian()
    result = analyze_hessian(geometry, [H_MASS, H_MASS], hessian)
    mass_kg = H_MASS * physical_constants["atomic mass constant"][0]
    spring_si = (
        0.4
        * physical_constants["Hartree energy"][0]
        / physical_constants["Bohr radius"][0] ** 2
    )
    expected_cm = sqrt(spring_si / (mass_kg / 2)) / (2 * pi * c * 100)
    assert result.external_rank == 5
    assert len(result.frequencies_cm1) == 1
    # Independently tabulated CODATA constants have finite published precision.
    assert result.frequencies_cm1[0] == pytest.approx(expected_cm, rel=1e-12)
    assert result.harmonic_zpe_hartree == pytest.approx(
        result.angular_frequencies_au[0] / 2
    )
    assert result.external_residual_relative < 1e-14
    electron_masses = np.repeat(
        np.array([H_MASS, H_MASS])
        * physical_constants["atomic mass constant"][0]
        / physical_constants["electron mass"][0],
        3,
    )
    assert np.allclose(
        result.cartesian_modes.T @ np.diag(electron_masses) @ result.cartesian_modes,
        np.eye(1),
    )


def test_isotope_shift_rotor_translation_rotation_and_linear_absence():
    geometry, hessian = diatomic_hessian()
    isotopes = analyze_isotopologues(
        geometry, hessian, {"H2": [H_MASS, H_MASS], "D2": [D_MASS, D_MASS]}
    )
    assert isotopes["D2"].frequencies_cm1[0] / isotopes["H2"].frequencies_cm1[
        0
    ] == pytest.approx(sqrt(H_MASS / D_MASS))
    rotor = equilibrium_rotor(geometry, [H_MASS, H_MASS])
    rotated = geometry @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + np.array(
        [4.7, -2.1, 3.5]
    )
    other = equilibrium_rotor(rotated, [H_MASS, H_MASS])
    assert rotor.rotor_type == "linear"
    assert rotor.constants_mhz[0] is None
    assert other.constants_mhz == pytest.approx(rotor.constants_mhz)
    assert np.linalg.det(rotor.principal_axes_columns) > 0


def test_nonlinear_three_internal_modes_and_covariant_hessian():
    geometry = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 1.1], [-1.4, 0.0, 1.1]])
    hessian = np.zeros((9, 9))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        direction = geometry[a] - geometry[b]
        direction /= np.linalg.norm(direction)
        pair = 0.3 * np.outer(direction, direction)
        hessian[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] += pair
        hessian[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] += pair
        hessian[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] -= pair
        hessian[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] -= pair
    masses = [15.99491461957, H_MASS, H_MASS]
    result = analyze_hessian(geometry, masses, hessian)
    assert result.external_rank == 6
    assert len(result.frequencies_cm1) == 3
    assert result.stationary_character == "positive_definite_vibrational_hessian"
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    transform = np.kron(np.eye(3), rotation)
    rotated = analyze_hessian(
        geometry @ rotation.T + 4, masses, transform @ hessian @ transform.T
    )
    assert rotated.frequencies_cm1 == pytest.approx(result.frequencies_cm1, rel=1e-12)


def test_mixed_force_constants_have_correct_multiplicity():
    geometry = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 1.1], [-1.4, 0.0, 1.1]])
    hessian = np.zeros((9, 9))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        direction = geometry[a] - geometry[b]
        direction /= np.linalg.norm(direction)
        pair = 0.3 * np.outer(direction, direction)
        hessian[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] += pair
        hessian[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] += pair
        hessian[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] -= pair
        hessian[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] -= pair
    harmonic = analyze_hessian(geometry, [15.99491461957, H_MASS, H_MASS], hessian)
    inverse = np.linalg.pinv(harmonic.dimensionless_to_cartesian)
    a, b, cc, d = 1e-5, 2e-5, 3e-5, 4e-5

    def exact_polynomial(coordinates):
        q = inverse @ (coordinates - geometry).ravel()
        return float(
            np.dot(harmonic.angular_frequencies_au, q * q) / 2
            + a * q[0] ** 2 * q[1]
            + b * q[0] * q[1] * q[2]
            + cc * q[0] ** 2 * q[1] ** 2
            + d * q[1] ** 3 * q[2]
        )

    field = build_force_field(
        harmonic,
        exact_polynomial,
        evaluator_identity="exact three-mode polynomial",
        absolute_tolerance_hartree=1e-9,
        relative_tolerance=1e-6,
    )
    assert field.derivative_converged
    assert field.cubic_hartree[0, 0, 1] == pytest.approx(2 * a, rel=1e-8)
    assert field.cubic_hartree[0, 1, 2] == pytest.approx(b, rel=1e-8)
    assert field.quartic_hartree[0, 0, 1, 1] == pytest.approx(4 * cc, rel=1e-8)
    assert field.quartic_hartree[1, 1, 1, 2] == pytest.approx(6 * d, rel=1e-8)


@pytest.mark.parametrize("bad", ["asymmetric", "nonfinite", "negative_mass"])
def test_hessian_rejects_invalid_data(bad):
    geometry, hessian = diatomic_hessian()
    masses = [H_MASS, H_MASS]
    if bad == "asymmetric":
        hessian[1, 0] = 0.1
    if bad == "nonfinite":
        hessian[0, 0] = np.nan
    if bad == "negative_mass":
        masses[0] = -1
    with pytest.raises(ValueError):
        analyze_hessian(geometry, masses, hessian)


def test_imaginary_mode_is_retained_and_blocks_anharmonic_field():
    geometry, hessian = diatomic_hessian(-0.4)
    result = analyze_hessian(geometry, [H_MASS, H_MASS], hessian)
    assert result.frequencies_cm1[0] < 0
    assert result.harmonic_zpe_hartree is None
    assert result.dimensionless_to_cartesian is None
    with pytest.raises(ValueError):
        build_force_field(
            result,
            lambda coordinates: 0.0,
            evaluator_identity="unused; invalid Hessian rejects before evaluation",
        )


def oscillator_field(cubic_fraction=0.001, quartic_fraction=0.0001):
    geometry, hessian = diatomic_hessian()
    harmonic = analyze_hessian(geometry, [H_MASS, H_MASS], hessian)
    omega = harmonic.angular_frequencies_au[0]
    a, b = cubic_fraction * omega, quartic_fraction * omega
    inverse = np.linalg.pinv(harmonic.dimensionless_to_cartesian)

    def potential(coordinates):
        q = float((inverse @ (coordinates - geometry).ravel())[0])
        return float(omega * q * q / 2 + a * q**3 + b * q**4)

    field = build_force_field(
        harmonic,
        potential,
        evaluator_identity=(
            "analytic anharmonic oscillator; numerical-method verification only"
        ),
        absolute_tolerance_hartree=1e-10,
        relative_tolerance=1e-6,
    )
    return field, a, b


def test_actual_displacement_derivatives_and_closed_form_vpt2():
    field, a, b = oscillator_field()
    omega = field.frequencies_hartree[0]
    assert field.cubic_hartree[0, 0, 0] == pytest.approx(6 * a, rel=1e-8)
    assert field.quartic_hartree[0, 0, 0, 0] == pytest.approx(24 * b, rel=1e-7)
    assert field.derivative_converged
    assert (
        field.evaluation_count == 7
    )  # overlapping central stencils share only identical geometries
    assert len(field.displacement_records) == field.evaluation_count
    assert all(
        record.source_artifact_sha256 is None for record in field.displacement_records
    )
    result = vibrational_vpt2(field)
    expected_ground = omega / 2 + 3 * b / 4 - 11 * a * a / (8 * omega)
    expected_fundamental = omega + 3 * b - 15 * a * a / (2 * omega)
    assert result.ground_energy_hartree == pytest.approx(expected_ground, rel=1e-9)
    assert result.fundamental_frequencies_cm1[0] == pytest.approx(
        expected_fundamental * HARTREE_CM1, rel=1e-9
    )
    assert result.rotation_vibration_available is False
    # Independently diagonalized finite oscillator basis verifies perturbative
    # asymptotics without reusing the sparse perturbation implementation.
    dimension = 70
    annihilation = np.diag(np.sqrt(np.arange(1, dimension)), 1)
    q = (annihilation + annihilation.T) / sqrt(2)
    matrix = (
        np.diag(omega * (np.arange(dimension) + 0.5))
        + a * np.linalg.matrix_power(q, 3)
        + b * np.linalg.matrix_power(q, 4)
    )
    exact = np.linalg.eigvalsh(matrix)
    assert (
        abs((exact[1] - exact[0]) * HARTREE_CM1 - result.fundamental_frequencies_cm1[0])
        < 0.002
    )


def test_strong_one_mode_coupling_is_an_applicability_diagnostic():
    field, cubic_coefficient, _ = oscillator_field(
        cubic_fraction=0.15, quartic_fraction=0.05
    )
    diagnostics = analyze_resonances(field)
    assert diagnostics
    assert all(item.kind == "strong_anharmonic_coupling" for item in diagnostics)
    assert all(item.harmonic_detuning_cm1 > 10 for item in diagnostics)
    ground_to_fundamental = next(
        item for item in diagnostics if (item.state_a, item.state_b) == ((0,), (1,))
    )
    # Independently evaluate <0|a q^3|1>=3a/(2 sqrt(2)), using harmonic
    # oscillator ladder algebra rather than the sparse operator implementation.
    assert ground_to_fundamental.coupling_cm1 == pytest.approx(
        abs(cubic_coefficient) * 3 / (2 * sqrt(2)) * HARTREE_CM1, rel=1e-8
    )
    with pytest.raises(ValueError, match="perturbative applicability gate") as error:
        vibrational_vpt2(field)
    assert "explicit polyad treatment required" not in str(error.value)


def test_force_field_energy_failure_budget_and_convergence_gate():
    geometry, hessian = diatomic_hessian()
    harmonic = analyze_hessian(geometry, [H_MASS, H_MASS], hessian)

    def invalid(coordinates):
        return np.nan

    with pytest.raises(ValueError):
        build_force_field(
            harmonic, invalid, evaluator_identity="rejected invalid scalar"
        )
    with pytest.raises(RuntimeError):
        build_force_field(
            harmonic,
            lambda coordinates: float(np.sum(coordinates**2)),
            evaluator_identity="analytic polynomial",
            max_evaluations=1,
        )
    field, _, _ = oscillator_field()
    with pytest.raises(ValueError):
        vibrational_vpt2(replace(field, derivative_converged=False))


def test_fermi_resonance_detected_and_explicit_polyad_is_hermitian():
    field, _, _ = oscillator_field()
    frequencies = np.array([0.01, 0.02])
    cubic = np.zeros((2, 2, 2))
    cubic[0, 0, 1] = cubic[0, 1, 0] = cubic[1, 0, 0] = 0.0002
    quartic = np.zeros((2, 2, 2, 2))
    field = replace(
        field,
        frequencies_hartree=frequencies,
        cubic_hartree=cubic,
        quartic_hartree=quartic,
        source_digest=artifact_digest(
            {
                "model": "exact two-mode Fermi Hamiltonian",
                "omega": frequencies,
                "cubic": cubic,
                "quartic": quartic,
            }
        ),
    )
    resonances = analyze_resonances(field)
    assert any(
        item.kind == "Fermi" and {item.state_a, item.state_b} == {(2, 0), (0, 1)}
        for item in resonances
    )
    with pytest.raises(ValueError, match="resonance"):
        vibrational_vpt2(field)
    polyad = solve_vibrational_polyad(field, [(0, 1), (2, 0)])
    assert np.allclose(
        polyad.effective_hamiltonian_hartree, polyad.effective_hamiltonian_hartree.T
    )
    assert np.allclose(polyad.mixing_columns.T @ polyad.mixing_columns, np.eye(2))
    assert polyad.energies_hartree[1] > polyad.energies_hartree[0]
    assert polyad.effective_hamiltonian_hartree[0, 1] == pytest.approx(0.0002 / 4)


def test_vibration_rotation_formula_requires_real_external_correction_metadata(
    tmp_path,
):
    from hashlib import sha256

    geometry, _ = diatomic_hessian()
    rotor = equilibrium_rotor(geometry, [H_MASS, H_MASS])
    correction_artifact = tmp_path / "formula-input.txt"
    correction_artifact.write_text(
        "Exact mathematical formula input: alpha_B=alpha_C=12MHz; one mode\n"
    )
    corrected = apply_rotation_vibration_correction(
        rotor,
        {"B": [12.0], "C": [12.0]},
        correction_method="explicit mathematical alpha input for formula test",
        correction_artifact_sha256=sha256(correction_artifact.read_bytes()).hexdigest(),
        independent_validation_reference=(
            "formula test only; no molecular validation claim"
        ),
        equilibrium_geometry_digest=rotor.geometry_digest,
        isotope_masses_u=[H_MASS, H_MASS],
    )
    assert corrected.constants_mhz[0] is None
    assert corrected.constants_mhz[1] == pytest.approx(rotor.constants_mhz[1] - 6)
    with pytest.raises(ValueError):
        apply_rotation_vibration_correction(
            rotor,
            {"B": [12], "C": [12]},
            correction_method="",
            correction_artifact_sha256="",
            independent_validation_reference="",
            equilibrium_geometry_digest=rotor.geometry_digest,
            isotope_masses_u=[H_MASS, H_MASS],
        )


def test_wigner_orthogonality_and_known_limit():
    assert wigner_3j(1, 1, 0, 0, 0, 0) == pytest.approx(-1 / sqrt(3))
    assert wigner_3j(1, 1, 1, 0, 0, 0) == pytest.approx(0)
    for j1, j2, j3 in ((2, 1, 3), (3, 1, 3), (1, 1, 0)):
        summed = sum(
            wigner_3j(j1, j2, j3, m1, m2, -m1 - m2) ** 2
            for m1 in range(-j1, j1 + 1)
            for m2 in range(-j2, j2 + 1)
        )
        assert summed == pytest.approx(1, abs=2e-14)


def test_rigid_rotor_linear_spacing_honl_london_and_einstein_rate():
    catalog = rigid_rotor_catalog(
        (None, 50000.0, 50000.0),
        [1.0, 0, 0],
        temperature_kelvin=5,
        J_max=15,
        constant_observable="Be",
    )
    assert len(catalog.lines) == 15
    for line in catalog.lines:
        assert line.frequency_mhz == pytest.approx(100000 * line.upper_J)
        assert line.summed_dipole_strength_debye2 == pytest.approx(
            line.upper_J, rel=1e-12
        )
        assert line.einstein_A_s1 > 0
    line = catalog.lines[0]
    expected = (
        16 * pi**3 * (1e11) ** 3 * (3.33564e-30) ** 2 / (3 * epsilon_0 * h * c**3 * 3)
    )
    assert line.einstein_A_s1 == pytest.approx(expected)
    direct = sum(
        (2 * J + 1) * np.exp(-h * 1e6 * 50000 * J * (J + 1) / (k * 5))
        for J in range(100)
    )
    assert catalog.partition_function == pytest.approx(direct, rel=1e-12)
    assert catalog.partition_converged_at_requested_tolerance
    assert catalog.identification_qualified is False
    assert "SPCAT" not in catalog.model_identity


def test_symmetric_top_eigenvalues_and_asymmetric_j1_independent_formula():
    symmetric = rigid_rotor_levels((10000.0, 5000.0, 5000.0), 3)
    actual = sorted(item.energy_mhz for item in symmetric if item.J == 2)
    expected = sorted(5000 * 6 + (10000 - 5000) * K * K for K in range(-2, 3))
    assert actual == pytest.approx(expected)
    asymmetric = rigid_rotor_levels((10000.0, 5000.0, 3000.0), 1)
    actual = sorted(item.energy_mhz for item in asymmetric if item.J == 1)
    assert actual == pytest.approx([5000 + 3000, 10000 + 3000, 10000 + 5000])


def test_asymmetric_j0_to_j1_axis_dipole_sum_rule():
    catalog = rigid_rotor_catalog(
        (10000.0, 5000.0, 3000.0),
        [0.3, 0.4, 0.5],
        temperature_kelvin=5,
        J_max=4,
        constant_observable="Be",
    )
    ground_lines = [line for line in catalog.lines if line.lower_J == 0]
    assert len(ground_lines) == 3
    assert [line.frequency_mhz for line in ground_lines] == pytest.approx(
        [8000, 13000, 15000]
    )
    assert [
        line.summed_dipole_strength_debye2 for line in ground_lines
    ] == pytest.approx([0.09, 0.16, 0.25])


def test_catalog_rejects_missing_dipole_and_reports_unconverged_partition():
    with pytest.raises(ValueError):
        rigid_rotor_catalog(
            (None, 50000.0, 50000.0),
            [None, 0, 0],
            temperature_kelvin=5,
            J_max=2,
            constant_observable="Be",
        )
    with pytest.raises(ValueError):
        rigid_rotor_catalog(
            (None, 50000.0, 50000.0),
            [1, 2, 0],
            temperature_kelvin=5,
            J_max=2,
            constant_observable="Be",
        )
    catalog = rigid_rotor_catalog(
        (None, 500.0, 500.0),
        [1, 0, 0],
        temperature_kelvin=300,
        J_max=2,
        constant_observable="Be",
    )
    assert not catalog.partition_converged_at_requested_tolerance


def test_live_pyscf_h2_optimization_hessian_and_anharmonic_displacements(tmp_path):
    """A genuine small calculation validates wiring, not spectroscopy accuracy.

    HF/STO-3G is deliberately identified and is unsuitable for high-accuracy
    identification. Optional engine absence is an explicit skipped evidence lane.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    from copy import deepcopy

    from cochem_torq.engines.pyscf_backend import PySCFBackend

    backend = PySCFBackend()
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[-0.8, 0, 0], [0.8, 0, 0]],
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
        },
        "properties": ["energy", "gradient", "hessian"],
        "settings": {
            "threads": 1,
            "scf_energy_tolerance": 1e-13,
            "scf_gradient_tolerance": 1e-9,
        },
    }
    optimized = backend.optimize(request, tmp_path / "optimized")
    assert optimized["status"] == "complete", optimized.get("errors")
    assert optimized["optimization"]["converged"]
    harmonic = analyze_hessian(
        optimized["geometry_bohr"], [H_MASS, H_MASS], optimized["hessian_hartree_bohr2"]
    )
    assert harmonic.external_rank == 5
    assert harmonic.stationary_character == "positive_definite_vibrational_hessian"
    assert harmonic.external_residual_relative < 1e-4
    calls = []

    def displaced_energy(coordinates):
        calculation = deepcopy(request)
        calculation["molecule"]["geometry_bohr"] = coordinates.tolist()
        calculation["properties"] = ["energy"]
        result = backend.evaluate(
            calculation, tmp_path / f"displacement-{len(calls):03d}"
        )
        assert result["status"] == "complete", result.get("errors")
        calls.append(result)
        return EnergyEvaluation(result["energy_hartree"], result["manifest_sha256"])

    field = build_force_field(
        harmonic,
        displaced_energy,
        evaluator_identity="PySCF 2.14.0 RHF/STO-3G; scientific accuracy unqualified",
        steps=(0.08, 0.04),
        absolute_tolerance_hartree=1e-6,
        relative_tolerance=0.03,
    )
    assert field.evaluation_count == len(calls) == 7
    assert all(
        record.source_artifact_sha256 is not None
        for record in field.displacement_records
    )
    assert field.derivative_converged
    assert abs(field.cubic_hartree[0, 0, 0]) > 1e-6
    assert abs(field.quartic_hartree[0, 0, 0, 0]) > 1e-6
    # This minimal-basis H2 force field exceeds the configured conservative
    # coupling/detuning gate. Do not relax that threshold merely to make this
    # calculation pass: retain the field and report the blocked next stage.
    assert analyze_resonances(field)
    assert not any(item.kind == "Fermi" for item in analyze_resonances(field))
    with pytest.raises(ValueError, match="perturbative applicability gate"):
        vibrational_vpt2(field)
    observed = {
        record.q_dimensionless[0]: record.energy_hartree
        for record in field.displacement_records
    }
    fine_step = field.steps_dimensionless[1]
    curvature = (
        observed[fine_step] + observed[-fine_step] - 2 * observed[0.0]
    ) / fine_step**2
    assert curvature == pytest.approx(harmonic.angular_frequencies_au[0], rel=0.001)
