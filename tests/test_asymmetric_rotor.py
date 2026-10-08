"""Mathematical rotor limits and missing-property rejection; no engine claims."""

import numpy as np
import pytest

from Libraries.cochem_torq_asymmetric_rotor import (
    AsymmetricTopDiagonalizer,
    RotationalConstants,
)


def test_asymmetric_rigid_j_one_matches_independent_analytic_eigenvalues():
    constants = RotationalConstants(A=10.0, B=6.0, C=3.0)
    levels = AsymmetricTopDiagonalizer(constants, j_max=1).solve_energy_levels()
    assert levels[(0, 0)] == 0.0
    np.testing.assert_allclose(
        [levels[(1, index)] for index in range(3)], [9.0, 13.0, 16.0], rtol=1e-14
    )
    assert all(len(identity) == 2 for identity in levels)


@pytest.mark.parametrize("angular_momentum", [0, 1, 2, 3])
def test_prolate_symmetric_top_matches_analytic_angular_momentum_limit(
    angular_momentum,
):
    constants = RotationalConstants(A=10.0, B=3.0, C=3.0)
    solver = AsymmetricTopDiagonalizer(constants, j_max=angular_momentum)
    levels = solver.solve_energy_levels()
    expected = sorted(
        3.0 * angular_momentum * (angular_momentum + 1) + 7.0 * projection**2
        for projection in range(-angular_momentum, angular_momentum + 1)
    )
    actual = [levels[(angular_momentum, index)] for index in range(len(expected))]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    assert len(levels) == (angular_momentum + 1) ** 2
    assert not solver.identification_qualified
    assert "Ka/Kc unassigned" in solver.quantum_number_convention


def test_spherical_top_and_declared_quartic_diagonal_limit():
    solver = AsymmetricTopDiagonalizer(RotationalConstants(3.0, 3.0, 3.0), j_max=2)
    matrix, _ = solver._build_watson_a_matrix(2)
    np.testing.assert_array_equal(matrix, np.eye(5) * 18.0)
    constants = RotationalConstants(10.0, 3.0, 3.0, D_J=0.01, D_JK=0.02, D_K=0.03)
    matrix, projections = AsymmetricTopDiagonalizer(constants)._build_watson_a_matrix(2)
    expected = [
        18.0 + 7.0 * p**2 - 0.36 - 0.12 * p**2 - 0.03 * p**4 for p in projections
    ]
    np.testing.assert_allclose(np.diag(matrix), expected, rtol=1e-14)
    np.testing.assert_array_equal(matrix, matrix.T)
    assert "quartic" in constants.hamiltonian_model


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1 + 0j, "10"])
def test_invalid_scientific_constants_are_rejected(value):
    with pytest.raises(ValueError, match="finite real"):
        RotationalConstants(value, 3.0, 2.0)


def test_absent_dipoles_do_not_become_numeric_properties():
    constants = RotationalConstants(10.0, 3.0, 2.0)
    assert constants.mu_a is None and constants.mu_b is None and constants.mu_c is None
    assert "distortion omitted" in constants.hamiltonian_model
    with pytest.raises(ValueError, match="actual dipole"):
        AsymmetricTopDiagonalizer(constants).compute_transitions()


def test_unimplemented_reduction_and_unqualified_intensity_do_not_substitute():
    constants = RotationalConstants(10.0, 3.0, 2.0, mu_a=1e-8, mu_b=0.0, mu_c=0.0)
    with pytest.raises(NotImplementedError, match="Watson S"):
        AsymmetricTopDiagonalizer(constants, reduction="S")
    with pytest.raises(
        NotImplementedError, match="intensity conventions are unqualified"
    ):
        AsymmetricTopDiagonalizer(constants).compute_transitions()


def test_blocked_catalog_export_preserves_output_path(tmp_path):
    constants = RotationalConstants(10.0, 3.0, 2.0, mu_a=0.0, mu_b=0.0, mu_c=0.0)
    destination = tmp_path / "new-output" / "catalog.parquet"
    with pytest.raises(NotImplementedError, match="unqualified"):
        AsymmetricTopDiagonalizer(constants).export_line_catalog_parquet(destination)
    assert not destination.exists() and not destination.parent.exists()


@pytest.mark.parametrize("angular_momentum", [-1, 1.5, True, 31])
def test_angular_momentum_bounds_are_exact(angular_momentum):
    with pytest.raises(ValueError, match="integer"):
        AsymmetricTopDiagonalizer(
            RotationalConstants(10.0, 3.0, 2.0), j_max=angular_momentum
        )
