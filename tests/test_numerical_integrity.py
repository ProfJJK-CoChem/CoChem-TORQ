"""Exact mathematical/file checks; these are not electronic-engine benchmarks."""
from __future__ import annotations

import math

import numpy as np
import pyarrow.ipc as ipc
import pytest

from Libraries.cochem_catalog_compiler import generate_methods_latex, MethodMatrixViolationError
from Libraries.cochem_isotopes import isotope_mass
from Libraries.cochem_jax_builder import (
    nan_tensor_watchdog, DVRConvergenceError, localized_vpt2_coupling,
    PLANCK_CONSTANT_JS, SPEED_OF_LIGHT_CMS, BOLTZMANN_CONSTANT_JK,
)
from Libraries.cochem_tensor_extractor import (
    allocate_pyarrow_ipc_buffer, TorqTensorExtractor,
    calculate_rays_asymmetry, diagonalize_inertia_tensor,
)


def test_unavailable_tensor_observables_are_arrow_nulls():
    buffer, digest = allocate_pyarrow_ipc_buffer({"point_id": "missing-observables"})
    row = ipc.open_stream(buffer).read_all().to_pylist()[0]
    assert digest
    assert row["point_id"] == "missing-observables"
    assert all(value is None for name, value in row.items() if name != "point_id")


def test_linear_geometry_retains_undefined_a():
    # Two unit point masses at +/- 0.5 Å: I=(0, 0.5, 0.5) u Å² exactly.
    extractor = TorqTensorExtractor(["H", "H"], [[-.5, 0, 0], [.5, 0, 0]], masses=[1, 1])
    result = extractor.extract_tensors()
    assert result["rotational_constants"]["A"] is None
    np.testing.assert_allclose(result["principal_moments_u_A2"], [0, .5, .5])
    buffer, _ = allocate_pyarrow_ipc_buffer(result)
    assert ipc.open_stream(buffer).read_all().to_pylist()[0]["A_MHz"] is None


@pytest.mark.parametrize("method", ["extract_thermal_nmr", "extract_raman_polarizability"])
def test_geometry_never_supplies_electronic_response(method):
    extractor = TorqTensorExtractor(["H", "H"], [[-.5, 0, 0], [.5, 0, 0]], masses=[1, 1])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(extractor, method)()


def test_mass_lookup_never_substitutes_requested_isotope():
    with pytest.raises(ValueError, match="requested isotope"):
        isotope_mass("1000H")
    with pytest.raises(ValueError, match="Specify an isotope"):
        isotope_mass("Tc")
    assert isotope_mass("D") == isotope_mass("2H")


def test_rotational_math_rejects_undefined_or_impossible_inputs():
    with pytest.raises(ValueError):
        diagonalize_inertia_tensor([[0, 0, 0]], [1], ["H"])
    with pytest.raises(ValueError):
        calculate_rays_asymmetry()
    with pytest.raises(ValueError):
        calculate_rays_asymmetry(1, 3, 2)
    assert calculate_rays_asymmetry(2, 2, 2).kappa is None


def test_finite_hamiltonian_recomputed_without_shift():
    hamiltonian = np.diag([1., 4.])
    eigenvalues, vectors = nan_tensor_watchdog(np.array([math.nan, 4.]), hamiltonian)
    np.testing.assert_array_equal(eigenvalues, [1., 4.])
    np.testing.assert_array_equal(hamiltonian @ vectors, vectors * eigenvalues)


def test_nonfinite_hamiltonian_is_never_zero_repaired():
    with pytest.raises(DVRConvergenceError, match="Nonfinite"):
        nan_tensor_watchdog(hamiltonian=np.array([[math.nan, 0], [0, 1]]))
    with pytest.raises(DVRConvergenceError, match="Eigenpairs"):
        nan_tensor_watchdog(np.array([1., 5.]), np.diag([1., 4.]), np.eye(2))


def test_separable_partition_preserves_low_positive_frequency():
    result = localized_vpt2_coupling([0., 20.], [[0., 0.], [0., 0.]], [1., 2.], [0], 100.)
    x = PLANCK_CONSTANT_JS * SPEED_OF_LIGHT_CMS * 2 / (BOLTZMANN_CONSTANT_JK * 100)
    assert result["q_stiff_vib"] == pytest.approx(1 / -math.expm1(-x))
    assert result["stiff_zpe_cm1"] == 1.
    assert result["couplings_between_DVR_and_stiff_modes"] == "neglected"


@pytest.mark.parametrize("energies,matrix,freqs,indices,temp", [
    ([], [[0.]], [1.], [], 100.),
    ([0.], [], [1.], [], 100.),
    ([0.], [[0.]], [1.], None, 100.),
    ([0.], [[0.]], [-1.], [], 100.),
    ([0.], [[0.]], [1.], [], 0.),
])
def test_incomplete_nuclear_motion_never_fills_values(energies, matrix, freqs, indices, temp):
    with pytest.raises(ValueError):
        localized_vpt2_coupling(energies, matrix, freqs, indices, temp)


def test_methods_writer_requires_evidence_and_never_infers_procedures(tmp_path):
    with pytest.raises(MethodMatrixViolationError, match="Missing method provenance"):
        generate_methods_latex({})
    # This is a serialization record, explicitly not a performed quantum calculation.
    metadata = dict(theory_level="serialization_test", basis_set="not_applicable",
                    software_version="test_record_1", provenance_hash="record_digest",
                    is_non_covalent=True)
    path = tmp_path / "methods.tex"
    content = generate_methods_latex(metadata, path, method_matrix_v4_check=False)
    assert path.read_text() == content
    for invented in ("ORCA 6.1", "DEFGRID3", "Frozen-Monomer", "Boys-Bernardi", "Pickett", "full nuclear spin"):
        assert invented not in content
    with pytest.raises(ValueError, match="units"):
        generate_methods_latex(metadata | {"rotational_constants": {"A": 1}}, method_matrix_v4_check=False)


def test_watchdog_rejects_zero_eigenvectors_and_unverified_finite_spectrum():
    with pytest.raises(DVRConvergenceError, match='orthonormal'):
        nan_tensor_watchdog([1., 9.], np.diag([1., 4.]), np.zeros((2, 2)))
    with pytest.raises(DVRConvergenceError, match='Eigenvalues'):
        nan_tensor_watchdog([1., 9.], np.diag([1., 4.]))
    with pytest.raises(DVRConvergenceError, match='dimensions'):
        nan_tensor_watchdog([1., 9.], np.diag([1., 4.]), np.empty((0, 0)))
    values, vectors = nan_tensor_watchdog([1., 4.], np.diag([1., 4.]))
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2))
    np.testing.assert_allclose(np.diag([1., 4.]) @ vectors, vectors * values)


def test_hermitian_solver_preserves_complex_coupling_and_rejects_invalid_matrix():
    from Libraries.cochem_jax_builder import jit_eigen_solver
    h = np.array([[0., 1j], [-1j, 0.]])
    values, vectors = jit_eigen_solver(h)
    np.testing.assert_allclose(values, [-1., 1.])
    validated, _ = nan_tensor_watchdog(values, h, vectors)
    np.testing.assert_allclose(validated, [-1., 1.])
    with pytest.raises(DVRConvergenceError, match='Hermitian'):
        jit_eigen_solver([[1., 100.], [0., 1.]])


@pytest.mark.parametrize('operator,grid', [
    (None, [0., 1., 2.]), (0., [0., 1., 2.]), (-1., [0., 1., 2.]),
    (1., None), (1., [0., 1., 3.]), (1., [0., 0., 0.]),
])
def test_dvr_requires_actual_mass_and_valid_explicit_grid(operator, grid):
    from Libraries.cochem_jax_builder import build_dvr_hamiltonian
    with pytest.raises(ValueError):
        build_dvr_hamiltonian([0., 0., 0.], kinetic_operator=operator, grid_points=grid)


def test_cartesian_mass_convention_is_consistent_in_one_and_two_dimensions():
    from Libraries.cochem_jax_builder import build_dvr_hamiltonian
    grid = np.array([0., .5, 1.])
    one = np.asarray(build_dvr_hamiltonian(np.zeros(3), 2., grid))
    two = np.asarray(build_dvr_hamiltonian(np.zeros((3, 3)), (2., 2.), (grid, grid), dimensions=2))
    np.testing.assert_allclose(two, np.kron(one, np.eye(3)) + np.kron(np.eye(3), one))


def test_periodic_free_rotor_spectrum_is_labelled_as_level_gap_not_tunnelling():
    from Libraries.cochem_jax_builder import JaxDVRBuilder, build_dvr_hamiltonian
    grid = np.arange(5) * 2 * np.pi / 5
    result = JaxDVRBuilder().solve_1d_rotor(grid, np.zeros(5), rotational_constant_cm1=2.)
    # Five Fourier states m=-2,-1,0,1,2 give B*m^2, a mathematical reduced model.
    np.testing.assert_allclose(result['eigenvalues'], [0., 2., 2., 8., 8.], atol=1e-12)
    assert result['tunneling_splitting_cm1'] is None
    assert result['lowest_level_gap_cm1'] == pytest.approx(2.)
    with pytest.raises(ValueError, match='2\\*pi'):
        build_dvr_hamiltonian(np.zeros(5), 2., np.linspace(0, 2*np.pi, 5), periodic=True)
    with pytest.raises(ValueError, match='angular inertia'):
        build_dvr_hamiltonian(np.zeros(5), 'H', grid, periodic=True)


def test_nonpositive_nuclear_mass_is_not_silently_removed_as_ghost():
    from Libraries.cochem_tensor_extractor import filter_ghost_atoms
    with pytest.raises(ValueError, match='positive mass'):
        filter_ghost_atoms([[0, 0, 0], [1, 0, 0]], ['H', 'H'], [1., -1.])
    with pytest.raises(ValueError, match='shape'):
        diagonalize_inertia_tensor([[0, 0, 0], [1, 0, 0]], [[1.], [1.]], filter_ghosts=False)
    with pytest.raises(ValueError, match='linear limit'):
        calculate_rays_asymmetry(None, 3., 2.)


def test_catalog_partition_never_invents_constants_or_repairs_invalid_values(tmp_path):
    from Libraries.cochem_catalog_compiler import TorqCatalogCompiler
    compiler = TorqCatalogCompiler(tmp_path / 'absent.cat', output_dir=tmp_path)
    with pytest.raises(ValueError, match='Explicit positive A'):
        compiler.compute_temperature_dependent_partition_function(100.)
    with pytest.raises(ValueError, match='ordered'):
        compiler.compute_temperature_dependent_partition_function(100., -10., 2., 1., 1)
    with pytest.raises(ValueError, match='symmetry number'):
        compiler.compute_temperature_dependent_partition_function(100., 10., 2., 1.)
    assert compiler.compute_temperature_dependent_partition_function(100., 10., 2., 1., 1) > 0


def test_catalog_nonfinite_and_overflow_rows_are_rejected(tmp_path):
    from Libraries.cochem_catalog_compiler import TorqCatalogCompiler, parse_spcat_cat_line, SPCATBridgeError
    with pytest.raises(SPCATBridgeError):
        parse_spcat_cat_line('nan 0.1 -1 3 0 1 1 101 1 0')
    source = tmp_path / 'overflow.cat'
    source.write_text('************* 0.1 -1 3 0 1 1 101 1 0\n')
    compiler = TorqCatalogCompiler(source, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match='overflow'):
        compiler.compile_to_parquet()
    assert not compiler.parquet_outpath.exists()



def test_genuinely_bent_near_linear_geometry_retains_finite_a_and_three_rotations():
    from Libraries.cochem_tensor_extractor import apply_cartesian_protections
    coordinates = [[-1., 0., 0.], [0., 0.0001, 0.], [1., 0., 0.]]
    inertia = diagonalize_inertia_tensor(coordinates, masses=[1., 1., 1.])
    assert 0 < inertia.principal_moments_u_A2[0] < 1e-6
    assert inertia.rotational_constants.A_MHz is not None
    protection = apply_cartesian_protections(coordinates, masses=[1., 1., 1.])
    assert protection.is_quasi_linear and not protection.is_linear
    assert protection.rotational_dof == 3
    assert protection.protected_rotational_constants.A_MHz == inertia.rotational_constants.A_MHz
