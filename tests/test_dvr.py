"""Mathematical qualification of the explicit constant-F periodic DVR model.

These declared potentials are algorithm inputs, not electronic-engine outputs
or experimental molecular references. No molecule-specific accuracy is claimed.
"""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_dvr import KCAL_MOL_TO_CM1, RelaxedPESTorsionalDVR


def declared_scan():
    angles = np.linspace(0.0, 2.0 * math.pi, 65)
    return angles, np.zeros_like(angles)


@pytest.mark.parametrize("grid_size", [31, 32, 63, 64])
def test_free_rotor_has_exact_fourier_eigenvalues_and_residuals(grid_size):
    angles, potential = declared_scan()
    prefactor = 2.75
    solver = RelaxedPESTorsionalDVR(
        angles, potential, n_points=grid_size, f_rotational_constant_cm1=prefactor
    )
    energies, vectors = solver.diagonalize()
    modes = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    expected = np.sort(prefactor * modes**2)
    np.testing.assert_allclose(energies, expected, atol=2e-10, rtol=1e-12)
    np.testing.assert_allclose(
        solver.h_matrix @ vectors, vectors * energies, atol=3e-10
    )
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(grid_size), atol=1e-12)
    assert abs(energies[0]) < 2e-10


def test_potential_zero_shift_only_shifts_the_spectrum():
    angles, _ = declared_scan()
    potential = 0.5 * (1.0 - np.cos(2.0 * angles))
    offset = 2.0
    first = RelaxedPESTorsionalDVR(angles, potential, f_rotational_constant_cm1=5.0)
    shifted = RelaxedPESTorsionalDVR(
        angles, potential + offset, f_rotational_constant_cm1=5.0
    )
    first_values, _ = first.diagonalize()
    second_values, _ = shifted.diagonalize()
    np.testing.assert_allclose(
        second_values - first_values, offset * KCAL_MOL_TO_CM1, atol=1e-10
    )
    assert shifted.tunneling_splitting_cm1 == pytest.approx(
        first.tunneling_splitting_cm1, abs=1e-10
    )
    result = first.solve()
    assert result["state_assignment"] == "unavailable"
    assert "constant_kinetic_approximation" in result["quality_flags"]


@pytest.mark.parametrize(
    "metadata", [{}, {"symbols": ["H", "O", "O", "H"], "coords": np.zeros((4, 3))}]
)
def test_no_peroxide_or_whole_molecule_inertia_is_substituted(metadata):
    angles, potential = declared_scan()
    with pytest.raises(ValueError, match="explicit independently justified positive F"):
        RelaxedPESTorsionalDVR(angles, potential, **metadata)


@pytest.mark.parametrize("prefactor", [0.0, -1.0, np.nan, np.inf, True, 1j])
def test_invalid_kinetic_coefficient_fails(prefactor):
    angles, potential = declared_scan()
    with pytest.raises(ValueError, match="rotational constant F"):
        RelaxedPESTorsionalDVR(angles, potential, f_rotational_constant_cm1=prefactor)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1j])
def test_nonfinite_or_complex_potential_is_rejected(bad):
    angles, potential = declared_scan()
    corrupted = potential.astype(complex if isinstance(bad, complex) else float)
    corrupted[3] = bad
    with pytest.raises(ValueError, match="finite scan|real, not complex"):
        RelaxedPESTorsionalDVR(angles, corrupted, f_rotational_constant_cm1=1.0)


@pytest.mark.parametrize("grid", [0, 2, 4.5, True])
def test_invalid_grid_size_is_rejected(grid):
    angles, potential = declared_scan()
    with pytest.raises(ValueError, match="integer of at least four"):
        RelaxedPESTorsionalDVR(
            angles, potential, n_points=grid, f_rotational_constant_cm1=1.0
        )


def test_hdf5_scan_still_requires_an_explicit_kinetic_model(tmp_path):
    angles, potential = declared_scan()
    path = tmp_path / "declared-mathematical-scan.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("torsion_scan")
        group["dihedral_deg"] = np.degrees(angles)
        group["energy_kcal_mol"] = potential
    with pytest.raises(ValueError, match="explicit independently justified positive F"):
        RelaxedPESTorsionalDVR.from_hdf5(path)
    solver = RelaxedPESTorsionalDVR.from_hdf5(path, f_rotational_constant_cm1=3.0)
    values, _ = solver.diagonalize()
    assert abs(values[0]) < 1e-10
    assert values[1] == pytest.approx(3.0, abs=1e-10)
