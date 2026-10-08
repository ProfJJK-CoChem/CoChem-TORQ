"""Independent metric/weak-form limits and genuine HF quartic-field integration.

Explicit mathematical models verify nuclear-motion algebra; they are never
labelled engine calculations or molecular spectroscopic accuracy benchmarks.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from itertools import product

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.linalg import eigh

from cochem_torq.engines.pyscf_backend import PySCFBackend
from cochem_torq.spectroscopy.forcefield import EnergyEvaluation, build_force_field
from cochem_torq.spectroscopy.harmonic import (
    analyze_hessian,
    artifact_digest,
    equilibrium_rotor,
)
from cochem_torq.spectroscopy.results import (
    ForceFieldData,
    ScientificContext,
    make_scientific_context,
)
from cochem_torq.spectroscopy.rovibrational import build_rovibrational_precursors
from cochem_torq.spectroscopy.rovibrational_solver import (
    RovibrationalVariationalResult,
    StationaryReference,
    VariationalProtocol,
    body_fixed_angular_momentum,
    compare_local_variational_results,
    local_kinetic_metric,
    rovibrational_energy_difference,
)
from cochem_torq.spectroscopy.rovibrational_solver import (
    solve_local_rovibrational_model as _solve_local_rovibrational_model,
)
from cochem_torq.units import ATOMIC_MASS_ELECTRON, HARTREE_CM1


def solve_local_rovibrational_model(precursors, force_field, protocol, **kwargs):
    """Supply an exact declared mathematical gradient, never an engine fallback."""
    if "stationary_reference" not in kwargs:
        if precursors.identity.input_evidence_class != "mathematical_model":
            raise ValueError(
                "Engine calculations require their actual native gradient."
            )
        kwargs["stationary_reference"] = StationaryReference(
            source_artifact_sha256=precursors.identity.parent_artifact_sha256[0],
            geometry_sha256=precursors.identity.geometry_sha256,
            evidence_class="mathematical_model",
            gradient_hartree_bohr=tuple(
                (0.0, 0.0, 0.0) for _ in precursors.identity.symbols
            ),
            maximum_allowed_gradient_hartree_bohr=1e-12,
        )
    return _solve_local_rovibrational_model(precursors, force_field, protocol, **kwargs)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _context(symbols, harmonic, rotor, *, evidence_class, parents, recipe):
    from Libraries.cochem_isotopes import isotope_record

    return ScientificContext.model_validate(
        make_scientific_context(
            molecule={
                "symbols": symbols,
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": [f"nucleus-{index}" for index in range(len(symbols))],
            },
            geometry_bohr=harmonic.coordinates_bohr,
            isotope_provenance=[isotope_record(symbol) for symbol in symbols],
            recipe_sha256=artifact_digest(recipe),
            protocol_sha256=artifact_digest(
                {"purpose": "local variational model test"}
            ),
            parent_artifact_sha256=parents,
            evidence_class=evidence_class,
            harmonic=harmonic,
            principal_axes_columns=rotor.principal_axes_columns,
        )
    )


def _field_data(field, context):
    payload = _plain(asdict(field))
    payload["scientific_context"] = context.model_dump(mode="json")
    return ForceFieldData.model_validate_json(json.dumps(payload))


def _mathematical_case(scale=1.0):
    from Libraries.cochem_isotopes import isotope_mass

    coordinates = scale * np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.8, 0.0, -0.5]]
    )
    masses = np.asarray([isotope_mass(symbol) for symbol in ("O", "H", "H")])
    electron_masses = masses * ATOMIC_MASS_ELECTRON
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    external = []
    for axis in np.eye(3):
        external.extend(
            (
                (np.tile(axis, (3, 1)) * np.sqrt(electron_masses)[:, None]).ravel(),
                (np.cross(axis, centered) * np.sqrt(electron_masses)[:, None]).ravel(),
            )
        )
    vectors, _, _ = np.linalg.svd(np.asarray(external).T, full_matrices=True)
    internal = vectors[:, 6:]
    frequencies = np.asarray([0.009, 0.013, 0.019])
    weighted_hessian = (internal * frequencies**2) @ internal.T
    mass_vector = np.repeat(electron_masses, 3)
    hessian = weighted_hessian * np.sqrt(mass_vector[:, None] * mass_vector[None, :])
    harmonic = analyze_hessian(coordinates, masses, hessian)
    rotor = equilibrium_rotor(coordinates, masses)
    context = _context(
        ["O", "H", "H"],
        harmonic,
        rotor,
        evidence_class="mathematical_model",
        parents=[artifact_digest({"explicit_projector_Hessian": hessian})],
        recipe={"analytic": "three independent harmonic coordinates", "scale": scale},
    )
    transformation = np.asarray(harmonic.dimensionless_to_cartesian)

    def potential(displaced):
        q = np.linalg.lstsq(
            transformation, (displaced - coordinates).ravel(), rcond=None
        )[0]
        return float(0.5 * np.dot(harmonic.angular_frequencies_au, q**2))

    field = build_force_field(
        harmonic,
        potential,
        evaluator_identity="declared analytic quadratic model",
        absolute_tolerance_hartree=1e-7,
    )
    assert field.derivative_converged
    return build_rovibrational_precursors(harmonic, rotor, context), _field_data(
        field, context
    )


@pytest.fixture(scope="module")
def mathematical_case():
    return _mathematical_case()


def _protocol(**changes):
    parameters = {
        "half_widths_dimensionless": (0.25, 0.25, 0.25),
        "sine_basis_counts": (2, 2, 1),
        "quadrature_orders": (9, 9, 7),
        "angular_momenta": (0, 1),
        "retained_states_per_j": 6,
        "maximum_matrix_dimension": 100,
        "maximum_quadrature_nodes": 10000,
    }
    return VariationalProtocol(**{**parameters, **changes})


@pytest.mark.parametrize("angular_momentum", range(6))
def test_body_generators_have_anomalous_commutator_and_integer_casimir(
    angular_momentum,
):
    operators = body_fixed_angular_momentum(angular_momentum)
    for index, (first, second) in enumerate(((1, 2), (2, 0), (0, 1))):
        assert np.allclose(operators[index], operators[index].conj().T, atol=1e-14)
        assert np.allclose(
            operators[first] @ operators[second] - operators[second] @ operators[first],
            -1j * operators[index],
            atol=1e-14,
        )
    assert np.allclose(
        sum(operator @ operator for operator in operators),
        angular_momentum * (angular_momentum + 1) * np.eye(2 * angular_momentum + 1),
        atol=1e-14,
    )


def _direct_metric(precursors, q):
    masses = np.asarray(precursors.identity.isotope_masses_u) * ATOMIC_MASS_ELECTRON
    modes = np.asarray(precursors.mass_weighted_modes_principal)
    frequencies = np.asarray(precursors.harmonic_frequencies_cm1) / HARTREE_CM1
    displacement = (
        modes / np.sqrt(masses)[:, None, None] / np.sqrt(frequencies)[None, None, :]
    )
    coordinates = np.asarray(precursors.centered_geometry_principal_bohr) + np.einsum(
        "aik,k->ai", displacement, q
    )
    rotation_velocities = [np.cross(axis, coordinates) for axis in np.eye(3)]
    coordinate_velocities = [displacement[:, :, index] for index in range(len(q))]
    velocities = rotation_velocities + coordinate_velocities
    return np.asarray(
        [
            [
                sum(mass * np.dot(a, b) for mass, a, b in zip(masses, first, second))
                for second in velocities
            ]
            for first in velocities
        ]
    )


@pytest.mark.parametrize(
    "displacement", [(0.1, -0.2, 0.05), (0.0, 0.0, 0.0), (-0.1, 0.2, -0.05)]
)
def test_metric_matches_independent_atomic_velocity_gram(
    mathematical_case, displacement
):
    precursors, _ = mathematical_case
    observed = local_kinetic_metric(precursors, displacement)
    direct = _direct_metric(precursors, displacement)
    assert np.allclose(observed.inertia_electron_mass_bohr2, direct[:3, :3], atol=1e-11)
    assert np.allclose(
        observed.rotational_vibrational_cross_metric, direct[:3, 3:], atol=1e-11
    )
    assert np.allclose(
        np.diag(observed.vibrational_metric_diagonal), direct[3:, 3:], atol=1e-11
    )
    schur = (
        direct[:3, :3] - direct[:3, 3:] @ np.linalg.inv(direct[3:, 3:]) @ direct[3:, :3]
    )
    assert np.allclose(observed.rotational_schur_complement, schur, atol=1e-11)
    reference = _direct_metric(precursors, np.zeros(3))
    assert observed.relative_volume_density == pytest.approx(
        np.sqrt(np.linalg.det(direct) / np.linalg.det(reference)), rel=1e-12
    )


def test_schur_weak_form_matches_independent_full_metric_operator(mathematical_case):
    precursors, field = mathematical_case
    protocol = _protocol()
    actual = solve_local_rovibrational_model(precursors, field, protocol)
    basis_indices = tuple(
        product(*(range(1, count + 1) for count in protocol.sine_basis_counts))
    )
    operators = body_fixed_angular_momentum(1)
    dimension = len(basis_indices) * 3
    matrix = np.zeros((dimension, dimension), dtype=complex)
    overlap = np.zeros((dimension, dimension))
    grids = [
        np.polynomial.legendre.leggauss(order) for order in protocol.quadrature_orders
    ]
    reference_det = np.linalg.det(_direct_metric(precursors, np.zeros(3)))
    for indices in product(*(range(order) for order in protocol.quadrature_orders)):
        coordinates = [
            protocol.half_widths_dimensionless[i] * grids[i][0][node]
            for i, node in enumerate(indices)
        ]
        weight = np.prod(
            [
                protocol.half_widths_dimensionless[i] * grids[i][1][node]
                for i, node in enumerate(indices)
            ]
        )
        metric = _direct_metric(precursors, coordinates)
        weight *= np.sqrt(np.linalg.det(metric) / reference_det)
        values, derivatives = [], []
        for occupation in basis_indices:
            arguments = [
                (coordinates[i] + width) * occupation[i] * np.pi / (2 * width)
                for i, width in enumerate(protocol.half_widths_dimensionless)
            ]
            factors = [
                np.sin(argument) / np.sqrt(width)
                for argument, width in zip(
                    arguments, protocol.half_widths_dimensionless
                )
            ]
            values.append(np.prod(factors))
            derivatives.append(
                [
                    np.cos(arguments[i])
                    * occupation[i]
                    * np.pi
                    / (2 * protocol.half_widths_dimensionless[i] ** 1.5)
                    * np.prod([value for j, value in enumerate(factors) if j != i])
                    for i in range(3)
                ]
            )
        values = np.asarray(values)
        derivatives = np.asarray(derivatives)
        inverse = np.linalg.inv(metric)
        # Each spin row is one actual wavefunction component in the weak form.
        for spin in range(3):
            gradient = np.vstack(
                (
                    *[
                        np.concatenate(
                            [1j * value * operator[spin] for value in values]
                        )
                        for operator in operators
                    ],
                    *[
                        np.concatenate(
                            [
                                derivative[i] * np.eye(3)[spin]
                                for derivative in derivatives
                            ]
                        )
                        for i in range(3)
                    ],
                )
            )
            matrix += 0.5 * weight * gradient.conj().T @ inverse @ gradient
        sampled_basis = np.kron(np.outer(values, values), np.eye(3))
        overlap += weight * sampled_basis
        q = np.asarray(coordinates)
        potential = 0.5 * np.dot(field.frequencies_hartree, q**2)
        potential += np.einsum("ijk,i,j,k", field.cubic_hartree, q, q, q) / 6
        potential += np.einsum("ijkl,i,j,k,l", field.quartic_hartree, q, q, q, q) / 24
        matrix += weight * potential * sampled_basis
    expected = eigh(matrix, overlap, eigvals_only=True)[:6]
    assert np.allclose(
        actual.sectors[1].energies_hartree_above_electronic_reference,
        expected,
        rtol=1e-11,
        atol=1e-12,
    )
    assert np.min(np.linalg.eigvalsh(matrix)) > 0
    assert actual.uniform_schur_eigenvalue_lower_bound > 0
    assert not actual.identification_ready
    assert actual.full_semirigid_vpt2 is False


def test_constant_metric_limit_matches_separable_dirichlet_hamiltonians():
    # The large-inertia limit is approached without magnifying Cartesian
    # subtraction noise until the actual derivative-convergence gate fails.
    precursors, field = _mathematical_case(scale=1e3)
    protocol = _protocol(
        half_widths_dimensionless=(1.0, 1.0, 1.0),
        sine_basis_counts=(3, 2, 1),
        quadrature_orders=(17, 17, 17),
        angular_momenta=(0,),
    )
    actual = solve_local_rovibrational_model(precursors, field, protocol)
    one_mode_levels = []
    for count, width, frequency in zip(
        protocol.sine_basis_counts,
        protocol.half_widths_dimensionless,
        field.frequencies_hartree,
    ):
        matrix = np.zeros((count, count))
        for row in range(count):
            for column in range(count):
                potential = quad(
                    lambda q: (
                        0.5
                        * frequency
                        * q
                        * q
                        * np.sin((row + 1) * np.pi * (q + width) / (2 * width))
                        * np.sin((column + 1) * np.pi * (q + width) / (2 * width))
                        / width
                    ),
                    -width,
                    width,
                    epsabs=1e-13,
                )[0]
                matrix[row, column] = potential + (
                    0.5 * frequency * ((row + 1) * np.pi / (2 * width)) ** 2
                    if row == column
                    else 0.0
                )
        one_mode_levels.append(np.linalg.eigvalsh(matrix))
    expected = sorted(sum(levels) for levels in product(*one_mode_levels))
    assert np.allclose(
        actual.sectors[0].energies_hartree_above_electronic_reference,
        expected,
        rtol=1e-8,
        atol=1e-9,
    )


def test_expanding_dirichlet_domain_approaches_independent_harmonic_ground_limit():
    precursors, field = _mathematical_case(scale=1e3)
    errors = []
    constant_other_mode_energies = 0.0
    for frequency in field.frequencies_hartree[1:]:
        constant_other_mode_energies += 0.5 * frequency * (np.pi / 2) ** 2
        constant_other_mode_energies += quad(
            lambda q: 0.5 * frequency * q**2 * np.sin(np.pi * (q + 1) / 2) ** 2,
            -1,
            1,
            epsabs=1e-13,
        )[0]
    unbounded_first_mode_ground = field.frequencies_hartree[0] / 2
    for width in (1.0, 2.0, 3.0):
        result = solve_local_rovibrational_model(
            precursors,
            field,
            _protocol(
                half_widths_dimensionless=(width, 1.0, 1.0),
                sine_basis_counts=(7, 1, 1),
                quadrature_orders=(25, 9, 9),
                angular_momenta=(0,),
            ),
        )
        first_mode_ground = (
            result.sectors[0].energies_hartree_above_electronic_reference[0]
            - constant_other_mode_energies
        )
        errors.append(abs(first_mode_ground - unbounded_first_mode_ground))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-5


def test_valid_linear_precursors_are_rejected_by_nonlinear_solver():
    from Libraries.cochem_isotopes import isotope_mass

    mass = isotope_mass("H")
    electron_mass = mass * ATOMIC_MASS_ELECTRON
    frequency = 0.01
    coordinates = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = electron_mass * frequency**2 / 2
    hessian[0, 3] = hessian[3, 0] = -electron_mass * frequency**2 / 2
    harmonic = analyze_hessian(coordinates, [mass, mass], hessian)
    rotor = equilibrium_rotor(coordinates, [mass, mass])
    context = _context(
        ["H", "H"],
        harmonic,
        rotor,
        evidence_class="mathematical_model",
        parents=[artifact_digest(hessian)],
        recipe={"analytic": "one-dimensional quadratic bond potential"},
    )
    precursors = build_rovibrational_precursors(harmonic, rotor, context)
    field = build_force_field(
        harmonic,
        lambda geometry: float(
            electron_mass
            * frequency**2
            / 4
            * (geometry[1, 0] - geometry[0, 0] - 2) ** 2
        ),
        evaluator_identity="declared analytic bond potential",
    )
    typed = _field_data(field, context)
    with pytest.raises(ValueError, match="nonlinear reference"):
        solve_local_rovibrational_model(
            precursors,
            typed,
            _protocol(
                half_widths_dimensionless=(0.25,),
                sine_basis_counts=(2,),
                quadrature_orders=(9,),
            ),
        )
    with pytest.raises(ValueError, match="nonlinear nonsingular"):
        local_kinetic_metric(precursors, (0.0,))


def test_source_bound_unassigned_energy_difference_has_no_invented_line(
    mathematical_case,
):
    actual = solve_local_rovibrational_model(*mathematical_case, _protocol())
    difference = rovibrational_energy_difference(
        actual, upper_state=(1, 0), lower_state=(0, 0)
    )
    assert difference.energy_difference_hartree == pytest.approx(
        actual.sectors[1].energies_hartree_above_electronic_reference[0]
        - actual.sectors[0].energies_hartree_above_electronic_reference[0]
    )
    assert difference.frequency_cm1 == pytest.approx(
        difference.energy_difference_hartree * HARTREE_CM1
    )
    assert difference.result_sha256 == actual.content_sha256
    assert difference.transition_allowed is None
    assert difference.dipole_strength is None
    assert difference.identification_ready is False
    assert (
        RovibrationalVariationalResult.model_validate_json(actual.model_dump_json())
        == actual
    )
    with pytest.raises(ValueError):
        rovibrational_energy_difference(actual, upper_state=(2, 0), lower_state=(0, 0))


@pytest.mark.parametrize("comparison", ["basis", "quadrature", "domain"])
def test_independent_convergence_axes_record_actual_passes_or_failures(
    mathematical_case, comparison
):
    first = solve_local_rovibrational_model(*mathematical_case, _protocol())
    changes = {
        "basis": {"sine_basis_counts": (1, 2, 1)},
        "quadrature": {"quadrature_orders": (11, 11, 9)},
        "domain": {"half_widths_dimensionless": (0.3, 0.3, 0.3)},
    }
    second = solve_local_rovibrational_model(
        *mathematical_case, _protocol(**changes[comparison])
    )
    observed = compare_local_variational_results(
        first, second, comparison=comparison, threshold_cm1=1e-5
    )
    assert observed.maximum_compared_energy_difference_cm1 >= 0
    assert observed.passed_for_compared_sorted_levels == (
        observed.maximum_compared_energy_difference_cm1 <= observed.threshold_cm1
    )
    assert observed.first_result_sha256 == first.content_sha256
    assert observed.independent_scientific_qualification is False
    with pytest.raises(ValueError, match="Change exactly"):
        compare_local_variational_results(
            first,
            second,
            comparison={"basis": "domain", "domain": "basis", "quadrature": "domain"}[
                comparison
            ],
            threshold_cm1=1.0,
        )


def test_chart_bound_checks_entire_domain_instead_of_only_grid_samples(
    mathematical_case,
):
    protocol = _protocol(half_widths_dimensionless=(100.0, 100.0, 100.0))
    with pytest.raises(ValueError, match="uniform local-chart certificate"):
        solve_local_rovibrational_model(*mathematical_case, protocol)


@pytest.mark.parametrize(
    "changes",
    [
        {"half_widths_dimensionless": (0.0, 0.25, 0.25)},
        {"quadrature_orders": (3, 3, 3)},
        {"sine_basis_counts": (0, 2, 1)},
        {"angular_momenta": (1,)},
        {"angular_momenta": (0, 6)},
        {"maximum_matrix_dimension": 1},
        {"maximum_quadrature_nodes": 1},
        {"maximum_workspace_bytes": 1},
        {"half_widths_dimensionless": (True, 0.25, 0.25)},
    ],
)
def test_invalid_or_unbudgeted_protocol_rejected(changes):
    with pytest.raises((ValueError, ValidationError)):
        _protocol(**changes)


@pytest.mark.parametrize(
    "corruption", ["energy", "qualification", "source", "protocol"]
)
def test_typed_readback_rejects_corrupt_result_binding(mathematical_case, corruption):
    result = solve_local_rovibrational_model(*mathematical_case, _protocol())
    payload = json.loads(result.model_dump_json())
    if corruption == "energy":
        payload["sectors"][0]["energies_hartree_above_electronic_reference"][0] += 0.1
    elif corruption == "qualification":
        payload["identification_ready"] = True
    elif corruption == "source":
        payload["force_field_sha256"] = artifact_digest({"wrong": "force field"})
    else:
        payload["protocol"]["half_widths_dimensionless"][0] *= 2
    with pytest.raises(ValueError):
        RovibrationalVariationalResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("corruption", ["source", "geometry", "evidence", "atom_count"])
def test_reference_gradient_requires_actual_matching_source_and_identity(
    mathematical_case, corruption
):
    precursors, field = mathematical_case
    data = {
        "source_artifact_sha256": precursors.identity.parent_artifact_sha256[0],
        "geometry_sha256": precursors.identity.geometry_sha256,
        "evidence_class": "mathematical_model",
        "gradient_hartree_bohr": tuple(
            (0.0, 0.0, 0.0) for _ in precursors.identity.symbols
        ),
        "maximum_allowed_gradient_hartree_bohr": 1e-12,
    }
    if corruption == "source":
        data["source_artifact_sha256"] = artifact_digest({"unrelated": "reference"})
    elif corruption == "geometry":
        data["geometry_sha256"] = artifact_digest({"unrelated": "geometry"})
    elif corruption == "evidence":
        data["evidence_class"] = "engine_calculation"
    else:
        data["gradient_hartree_bohr"] = ((0.0, 0.0, 0.0),)
    reference = StationaryReference(**data)
    with pytest.raises(ValueError, match="Reference gradient/source/geometry"):
        solve_local_rovibrational_model(
            precursors,
            field,
            _protocol(),
            stationary_reference=reference,
        )


def test_absent_or_failed_stationarity_never_uses_zero_gradient(mathematical_case):
    precursors, field = mathematical_case
    with pytest.raises(TypeError, match="stationary_reference"):
        _solve_local_rovibrational_model(precursors, field, _protocol())
    with pytest.raises(ValueError, match="stationarity gate"):
        StationaryReference(
            source_artifact_sha256=precursors.identity.parent_artifact_sha256[0],
            geometry_sha256=precursors.identity.geometry_sha256,
            evidence_class="mathematical_model",
            gradient_hartree_bohr=((0.1, 0.0, 0.0),) * len(precursors.identity.symbols),
            maximum_allowed_gradient_hartree_bohr=1e-4,
        )


def test_matrix_artifact_directory_is_exclusive_and_archive_claims_are_bound(
    mathematical_case, tmp_path
):
    directory = tmp_path / "already-owned"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        solve_local_rovibrational_model(
            *mathematical_case,
            _protocol(),
            artifact_directory=directory,
        )
    result = solve_local_rovibrational_model(*mathematical_case, _protocol())
    assert result.matrix_artifact_policy == "matrix archives unavailable"
    assert all(sector.matrix_artifact_sha256 is None for sector in result.sectors)
    payload = json.loads(result.model_dump_json())
    payload["matrix_artifact_policy"] = "matrix archives retained"
    with pytest.raises(ValueError):
        RovibrationalVariationalResult.model_validate_json(json.dumps(payload))


@pytest.fixture(scope="module")
def genuine_water_field(tmp_path_factory):
    from Libraries.cochem_isotopes import isotope_record

    root = tmp_path_factory.mktemp("genuine-water-local-rovibrational")
    request = {
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.15], [1.9, 0.0, -0.5]],
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {"name": "hf", "basis": "sto-3g", "reference": "restricted"},
        "properties": ["energy", "gradient", "hessian"],
        "settings": {
            "threads": 1,
            "scf_energy_tolerance": 1e-12,
            "check_stability": True,
        },
    }
    backend = PySCFBackend()
    native = backend.optimize(request, root / "optimized")
    assert native["status"] == "complete", native["errors"]
    assert native["stability"]["status"] == "stable"
    assert np.max(np.abs(native["gradient_hartree_bohr"])) < 2e-5
    harmonic = analyze_hessian(
        native["geometry_bohr"],
        [isotope_record(symbol)["mass_u"] for symbol in request["molecule"]["symbols"]],
        native["hessian_hartree_bohr2"],
    )
    rotor = equilibrium_rotor(harmonic.coordinates_bohr, harmonic.isotope_masses_u)
    context = _context(
        request["molecule"]["symbols"],
        harmonic,
        rotor,
        evidence_class="engine_calculation",
        parents=[native["manifest_sha256"]],
        recipe={
            "method": native["method"],
            "engine": native["engine_installation_sha256"],
        },
    )
    retained = []

    def evaluate(geometry):
        calculation = {
            **request,
            "properties": ["energy"],
            "molecule": {**request["molecule"], "geometry_bohr": geometry.tolist()},
        }
        directory = root / f"displacement-{len(retained):03d}"
        raw = backend.evaluate(calculation, directory)
        assert raw["status"] == "complete", raw["errors"]
        assert raw["stability"]["status"] == "stable"
        manifest_digest = sha256((directory / "manifest.json").read_bytes()).hexdigest()
        assert manifest_digest == raw["manifest_sha256"]
        retained.append((directory, raw))
        return EnergyEvaluation(raw["energy_hartree"], manifest_digest)

    field = build_force_field(
        harmonic,
        evaluate,
        evaluator_identity="actual PySCF restricted HF/STO-3G quartic field",
        absolute_tolerance_hartree=1e-6,
        relative_tolerance=0.02,
    )
    assert field.evaluation_count == len(retained) == 107
    assert field.derivative_converged
    stationarity = StationaryReference(
        source_artifact_sha256=native["manifest_sha256"],
        geometry_sha256=context.geometry_sha256,
        evidence_class="engine_calculation",
        gradient_hartree_bohr=tuple(
            tuple(float(component) for component in row)
            for row in native["gradient_hartree_bohr"]
        ),
        maximum_allowed_gradient_hartree_bohr=2e-5,
    )
    precursors = build_rovibrational_precursors(harmonic, rotor, context)
    typed = _field_data(field, context)
    (root / "force-field.json").write_text(typed.model_dump_json(indent=2))
    (root / "geometric-precursors.json").write_text(
        precursors.model_dump_json(indent=2)
    )
    (root / "stationary-reference.json").write_text(
        stationarity.model_dump_json(indent=2)
    )
    return (
        precursors,
        typed,
        root,
        stationarity,
    )


@pytest.mark.real_engine
def test_genuine_quartic_field_runs_j_resolved_model_and_preserves_native_artifacts(
    genuine_water_field,
):
    precursors, field, directory, stationarity = genuine_water_field
    result = solve_local_rovibrational_model(
        precursors,
        field,
        _protocol(),
        stationary_reference=stationarity,
        artifact_directory=directory / "matrix-replay",
    )
    assert result.force_field_sha256 == field.source_digest
    assert result.electronic_reference_energy_hartree == field.reference_energy_hartree
    assert len(result.native_parent_artifact_sha256) == 108
    assert result.precursor_context_sha256 == precursors.identity.input_context_sha256
    assert all(
        sector.maximum_eigenpair_relative_residual < 1e-10 for sector in result.sectors
    )
    assert result.effective_watson_constants is None
    assert result.rotation_vibration_alpha is None
    assert not result.full_semirigid_vpt2
    assert not result.independent_scientific_qualification
    assert not result.identification_ready
    assert (
        result.stationary_reference.gradient_hartree_bohr
        == stationarity.gradient_hartree_bohr
    )
    assert result.matrix_artifact_policy == "matrix archives retained"
    for sector in result.sectors:
        path = directory / "matrix-replay" / sector.matrix_artifact_name
        assert sha256(path.read_bytes()).hexdigest() == sector.matrix_artifact_sha256
        with np.load(path, allow_pickle=False) as saved:
            energies, vectors = (
                saved["eigenenergies_hartree"],
                saved["eigenvector_columns"],
            )
            matrix, overlap = saved["hamiltonian_hartree"], saved["overlap"]
            assert np.allclose(
                matrix @ vectors, (overlap @ vectors) * energies, atol=1e-11
            )
            assert np.array_equal(
                energies, sector.energies_hartree_above_electronic_reference
            )
    (directory / "local-variational-result.json").write_text(
        result.model_dump_json(indent=2)
    )
    assert (
        RovibrationalVariationalResult.model_validate_json(
            (directory / "local-variational-result.json").read_text()
        )
        == result
    )


@pytest.mark.real_engine
def test_genuine_field_quadrature_comparison_preserves_unqualified_scope(
    genuine_water_field,
):
    precursors, field, directory, stationarity = genuine_water_field
    first = solve_local_rovibrational_model(
        precursors, field, _protocol(), stationary_reference=stationarity
    )
    second = solve_local_rovibrational_model(
        precursors,
        field,
        _protocol(quadrature_orders=(11, 11, 9)),
        stationary_reference=stationarity,
    )
    comparison = compare_local_variational_results(
        first, second, comparison="quadrature", threshold_cm1=1e-3
    )
    # This observed low-order comparison fails. Tighter integration, rather
    # than a looser threshold, must establish accuracy at this box and basis.
    assert not comparison.passed_for_compared_sorted_levels
    assert not comparison.independent_scientific_qualification
    third = solve_local_rovibrational_model(
        precursors,
        field,
        _protocol(quadrature_orders=(13, 13, 11)),
        stationary_reference=stationarity,
    )
    fourth = solve_local_rovibrational_model(
        precursors,
        field,
        _protocol(quadrature_orders=(15, 15, 13)),
        stationary_reference=stationarity,
    )
    tightened = compare_local_variational_results(
        third, fourth, comparison="quadrature", threshold_cm1=1e-3
    )
    assert tightened.passed_for_compared_sorted_levels
    assert not tightened.independent_scientific_qualification
    (directory / "quadrature-comparisons.json").write_text(
        json.dumps(
            {
                "minimal_comparison": comparison.model_dump(mode="json"),
                "tightened_comparison": tightened.model_dump(mode="json"),
                "scope": (
                    "fixed small Dirichlet box/basis; no basis/domain qualification"
                ),
            },
            indent=2,
        )
    )
