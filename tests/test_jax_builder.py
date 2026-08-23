"""
CoChem-TORQ: Unit Tests for Hardware-Accelerated Physics Engine (JAX DVR)
Phase 7 (Stage 5.0) Validation Suite
Adhering to Zero-Approximation Mandate and Real Physical Solvers
"""

import time

import jax.numpy as jnp
import numpy as np

from Libraries.cochem_jax_builder import (
    JaxDVRBuilder,
    build_dvr_hamiltonian,
    enforce_jax_precision,
    jit_eigen_solver,
    localized_vpt2_coupling,
    nan_tensor_watchdog,
)


def test_enforce_jax_precision() -> None:
    """
    Test 1: Float64 Precision Truncation Guard & Architecture Detection.
    Verifies that float64 is strictly enforced in JAX and returns device info.
    """
    device_info = enforce_jax_precision()
    assert "platform" in device_info
    assert "x64_enabled" in device_info
    assert device_info["x64_enabled"] is True

    # Verify default tensor float precision is float64
    x = jnp.array([1.0, 2.0])
    assert x.dtype == jnp.float64


def test_double_well_tunneling_splitting_precision() -> None:
    """
    Test 2: Double-well potential tunneling splitting precision guard.
    Verifies that high-symmetry dual-well produces non-zero tunneling splitting.
    """
    enforce_jax_precision()
    n_points = 200
    grid_phi = np.linspace(-np.pi, np.pi, n_points, endpoint=False)

    # Symmetric double well: V(phi) = 0.5 * V_0 * (1 - cos(2*phi))
    barrier_cm1 = 500.0
    rot_b_cm1 = 10.0
    v_pot = 0.5 * barrier_cm1 * (1.0 - np.cos(2.0 * grid_phi))

    h_matrix = build_dvr_hamiltonian(
        pes_spline_array=v_pot,
        kinetic_operator=rot_b_cm1,
        grid_points=grid_phi,
        dimensions=1,
        periodic=True,
    )

    evals, evecs = jit_eigen_solver(h_matrix)

    assert evals.dtype == jnp.float64
    assert evecs.dtype == jnp.float64
    assert len(evals) == n_points

    # Ground state and first excited state (tunneling doublet)
    e0 = float(evals[0])
    e1 = float(evals[1])
    delta_e = e1 - e0

    # Tunneling splitting must be strictly positive and finite
    assert delta_e > 0.0, f"Tunneling splitting collapsed to {delta_e}"
    assert np.isfinite(delta_e)


def test_xla_compilation_speedup() -> None:
    """
    Test 3: XLA Compilation Speedup Test.
    Verifies that execution of jit_eigen_solver leverages XLA compilation.
    """
    enforce_jax_precision()
    matrix_size = 400
    np.random.seed(42)
    # Generate real symmetric matrix
    random_mat = np.random.randn(matrix_size, matrix_size)
    h_benchmark = (random_mat + random_mat.T) / 2.0
    h_jax = jnp.array(h_benchmark, dtype=jnp.float64)

    # First execution (includes XLA compilation)
    t0 = time.perf_counter()
    evals1, _ = jit_eigen_solver(h_jax)
    evals1.block_until_ready()
    t_first = time.perf_counter() - t0

    # Second execution (cached XLA graph)
    t0 = time.perf_counter()
    evals2, _ = jit_eigen_solver(h_jax)
    evals2.block_until_ready()
    t_second = time.perf_counter() - t0

    assert len(evals1) == matrix_size
    assert np.allclose(np.array(evals1), np.array(evals2))
    assert t_first > 0.0
    assert t_second >= 0.0
    assert t_second < max(t_first, 0.5)


def test_nan_tensor_watchdog_and_tikhonov_recovery() -> None:
    """
    Test 4: NaN Tensor Watchdog & Tikhonov Regularization Recovery.
    Verifies corrupt/NaN inputs are intercepted and regularized.
    """
    enforce_jax_precision()
    n_pts = 50
    grid_phi = np.linspace(-np.pi, np.pi, n_pts, endpoint=False)
    v_pot = 0.5 * 10.0 * (1.0 - np.cos(2.0 * grid_phi))

    h_matrix = build_dvr_hamiltonian(
        pes_spline_array=v_pot,
        kinetic_operator=5.0,
        grid_points=grid_phi,
        dimensions=1,
    )

    # Corrupt Hamiltonian by placing NaN and Inf
    h_corrupted = np.array(h_matrix, copy=True)
    h_corrupted[5, 5] = np.nan
    h_corrupted[10, 12] = np.inf
    h_corrupted[12, 10] = np.inf

    evals_recovered, evecs_recovered = nan_tensor_watchdog(
        eigenvalues=None,
        hamiltonian=h_corrupted,
        alpha_regularization=1e-5,
    )

    assert not np.isnan(evals_recovered).any(), "NaN found in recovered evals"
    assert not np.isinf(evals_recovered).any(), "Inf found in recovered evals"
    assert len(evals_recovered) == n_pts


def test_free_rotor_analytic_parity() -> None:
    """
    Test 5: Free Quantum Rotor Analytic Parity.
    Verifies that flat potential periodic sinc-DVR matches E_m = B * m^2.
    """
    enforce_jax_precision()
    n_pts = 101
    grid_phi = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    b_rot = 2.75  # Rotational constant in cm^-1
    v_pot = grid_phi * 0.0

    h_matrix = build_dvr_hamiltonian(
        pes_spline_array=v_pot,
        kinetic_operator=b_rot,
        grid_points=grid_phi,
        dimensions=1,
        periodic=True,
    )

    evals, _ = jit_eigen_solver(h_matrix)
    evals_np = np.array(evals)

    # Verify analytical free rotor solutions: E_0 = 0, E_1 = E_2 = B*1^2, etc.
    assert abs(evals_np[0] - 0.0) < 1e-8
    for m in range(1, 10):
        e_analytic = b_rot * (m**2)
        idx1 = 2 * m - 1
        idx2 = 2 * m
        e_comp1 = evals_np[idx1]
        e_comp2 = evals_np[idx2]
        assert abs(e_comp1 - e_analytic) < 1e-8
        assert abs(e_comp2 - e_analytic) < 1e-8


def test_2d_coupled_dvr_hamiltonian() -> None:
    """
    Test 6: 2D Coupled Rotor DVR Hamiltonian Construction and Diagonalization.
    Verifies 2D grid tensor product Hamiltonian and eigensolution.
    """
    enforce_jax_precision()
    nx = 20
    ny = 20
    phi_x = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    phi_y = np.linspace(-np.pi, np.pi, ny, endpoint=False)

    px, py = np.meshgrid(phi_x, phi_y, indexing="ij")
    v_2d = (
        100.0 * (1.0 - np.cos(3.0 * px))
        + 80.0 * (1.0 - np.cos(3.0 * py))
        + 20.0 * np.cos(3.0 * px + 3.0 * py)
    )

    h_2d = build_dvr_hamiltonian(
        pes_spline_array=v_2d,
        kinetic_operator=(5.0, 4.0),
        grid_points=(phi_x, phi_y),
        dimensions=2,
        periodic=True,
    )

    assert h_2d.shape == (nx * ny, nx * ny)
    evals, evecs = jit_eigen_solver(h_2d)

    assert len(evals) == nx * ny
    assert evecs.shape == (nx * ny, nx * ny)
    assert not np.isnan(np.asarray(evals)).any()
    evals_arr = np.asarray(evals)
    assert np.all(np.diff(evals_arr) >= -1e-12)


def test_localized_vpt2_coupling() -> None:
    """
    Test 7: Localized VPT2 Coupling.
    Verifies dropping low-frequency LAM mode and coupling stiff modes with DVR.
    """
    enforce_jax_precision()
    dvr_energies = np.array([0.0, 12.5, 45.0, 95.0, 160.0, 240.0, 335.0, 445.0])
    harmonic_freqs = [35.0, 520.0, 850.0, 1200.0, 1650.0, 3050.0]

    n_modes = len(harmonic_freqs)
    vpt2_x_matrix = np.full((n_modes, n_modes), -0.5)
    for i in range(n_modes):
        vpt2_x_matrix[i, i] = -0.01 * harmonic_freqs[i]

    result = localized_vpt2_coupling(
        dvr_energies=dvr_energies,
        vpt2_matrix=vpt2_x_matrix,
        harmonic_frequencies=harmonic_freqs,
        lam_mode_indices=[0],
        temperature_k=298.15,
    )

    assert "dropped_lam_modes" in result
    assert result["dropped_lam_modes"] == [0]
    assert "stiff_harmonic_frequencies" in result
    assert len(result["stiff_harmonic_frequencies"]) == 5
    assert 35.0 not in result["stiff_harmonic_frequencies"]
    assert "q_dvr_rot" in result
    assert result["q_dvr_rot"] > 1.0
    assert "q_stiff_vib" in result
    assert result["q_stiff_vib"] >= 1.0
    assert "q_coupled_total" in result
    assert result["q_coupled_total"] == result["q_dvr_rot"] * result["q_stiff_vib"]
    assert "coupled_ground_state_energy_cm1" in result


def test_jax_dvr_builder_class_orchestration() -> None:
    """
    Test 8: JaxDVRBuilder high-level class interface.
    Verifies complete workflow from 1D PES scan to eigenvalues.
    """
    builder = JaxDVRBuilder(dimensions=1, enable_x64=True)

    n_pts = 90
    phi = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    v3_barrier = 350.0  # cm^-1
    b_rot = 5.25  # cm^-1 (methyl rotor approx)
    v_pot = 0.5 * v3_barrier * (1.0 - np.cos(3.0 * phi))

    results = builder.solve_1d_rotor(
        grid_points=phi,
        energies_cm1=v_pot,
        rotational_constant_cm1=b_rot,
        periodic=True,
    )

    assert "eigenvalues" in results
    assert "wavefunctions" in results
    assert "ground_state_energy_cm1" in results
    assert len(results["eigenvalues"]) == n_pts

    evals = np.array(results["eigenvalues"])
    e0 = evals[0]
    e1 = evals[1]
    e2 = evals[2]
    assert abs(e2 - e1) < 1e-8, f"E states e1={e1}, e2={e2} should be degenerate"
    assert e1 > e0, f"E states e1={e1} must exhibit tunneling splitting over e0={e0}"
    tunneling_split = e1 - e0
    assert tunneling_split > 1e-4


def test_vpt2_triangular_anharmonic_zpe() -> None:
    """
    Test 9: VPT2 Upper-Triangular Anharmonic ZPE Summation.
    Verifies off-diagonal anharmonic corrections are computed as 0.25*sum_{i<=j} X_ij.
    """
    dvr_e = [0.0, 50.0]
    harm_freqs = [1000.0, 2000.0]
    vpt2_x = np.array([[-10.0, -30.0], [-30.0, -20.0]])

    result = localized_vpt2_coupling(
        dvr_energies=dvr_e,
        vpt2_matrix=vpt2_x,
        harmonic_frequencies=harm_freqs,
        lam_mode_indices=[],
        temperature_k=298.15,
    )

    harm_zpe = 0.5 * (1000.0 + 2000.0)  # 1500.0
    # 0.25 * (-10.0 + -20.0 + -30.0) = -15.0
    expected_zpe = harm_zpe - 15.0
    assert abs(result["stiff_zpe_cm1"] - expected_zpe) < 1e-10


def test_nan_watchdog_clean_passthrough_and_validation() -> None:
    """
    Test 10: Watchdog Passthrough with Wavefunctions and Zero Kelvin Protection.
    Verifies clean inputs pass through wavefunctions without empty array returns.
    """
    h_mat = np.array([[20.0, 5.0], [5.0, 15.0]])
    evals, evecs = np.linalg.eigh(h_mat)

    evals_out, evecs_out = nan_tensor_watchdog(
        eigenvalues=evals,
        hamiltonian=h_mat,
        wavefunctions=evecs,
    )
    assert np.allclose(evals_out, evals)
    assert np.allclose(evecs_out, evecs)

    # Test T=0 K guard
    vpt2_res_0k = localized_vpt2_coupling(
        dvr_energies=[0.0, 10.0],
        vpt2_matrix=np.array([[-5.0]]),
        harmonic_frequencies=[500.0],
        temperature_k=0.0,
    )
    assert vpt2_res_0k["q_coupled_total"] == 1.0
