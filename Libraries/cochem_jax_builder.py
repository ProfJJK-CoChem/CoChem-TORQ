"""
CoChem-TORQ 0.0.11
Stage 5.0: The Hardware-Accelerated Physics Engine (JAX DVR)
-----------------------------------------------------------
Implements exact 1D and 2D Discrete Variable Representation (DVR) solvers
using Google JAX with XLA Just-In-Time (JIT) compilation. Replaces inaccurate
Rigid-Rotor Harmonic-Oscillator (RRHO) approximations for Large Amplitude
Motions (LAMs) with exact quantum mechanical nuclear Schrödinger solutions.

Compliant with Method Matrix v4 (§4.4, §8C, Table 2) and Phase 7 specifications.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-JAX-DVR] %(message)s"
)
logger = logging.getLogger("TorqJaxDVR")

# CODATA 2022 Physical Constants (Exact)
PLANCK_CONSTANT_JS = 6.62607015e-34  # Exact h (J s)
BOLTZMANN_CONSTANT_JK = 1.380649e-23  # Exact kB (J/K)
SPEED_OF_LIGHT_CMS = 29979245800.0  # Exact c (cm/s)
HBAR_JS = PLANCK_CONSTANT_JS / (2.0 * math.pi)
# Conversion factor hbar^2 / (2 * m_u) in cm^-1 * Angstrom^2 * amu
HBAR_SQ_OVER_2M_U_CM1_A2 = 16.857629206

# Try importing JAX and configuring float64 precision
try:
    import jax
    import jax.numpy as jnp

    # Force 64-bit precision immediately upon module import
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.warning(
        "JAX not installed in runtime environment; fallback mode will be limited."
    )


class CoChemPrecisionError(RuntimeError):
    """Raised when JAX float64 precision cannot be enforced."""

    pass


class DVRConvergenceError(RuntimeError):
    """Raised when the DVR eigenvalue solver fails to converge."""

    pass


def enforce_jax_precision() -> dict[str, Any]:
    """
    Enforces JAX 64-bit floating point precision (float64) and identifies hardware.

    Spectroscopic tunneling splittings can exist on the order of 10^-6 cm^-1.
    Standard 32-bit floating point precision causes numerical underflow.

    :return: Dictionary containing detected hardware platform, devices, and x64 status.
    :raises CoChemPrecisionError: If 64-bit precision cannot be activated.
    """
    if not JAX_AVAILABLE:
        raise CoChemPrecisionError(
            "JAX is not installed. Cannot enforce precision or allocate GPU memory."
        )

    # Update and verify x64 configuration
    jax.config.update("jax_enable_x64", True)

    test_tensor = jnp.zeros((2,), dtype=jnp.float64)
    if test_tensor.dtype != jnp.float64:
        raise CoChemPrecisionError(
            f"Failed to enforce JAX 64-bit precision. Dtype: {test_tensor.dtype}."
        )

    devices = jax.devices()
    default_backend = jax.default_backend()

    device_info = {
        "platform": default_backend,
        "devices": [str(d) for d in devices],
        "device_count": len(devices),
        "x64_enabled": True,
        "dtype": "float64",
    }

    logger.info(
        f"JAX float64 precision enforced on {default_backend.upper()} "
        f"({len(devices)} device(s): {devices[0]})"
    )
    return device_info


def _construct_1d_kinetic_matrix(
    n_pts: int, delta_x: float, kinetic_factor: float, periodic: bool = False
) -> np.ndarray:
    """
    Constructs a 1D Colbert-Miller sinc-DVR or Periodic sinc-DVR kinetic matrix.

    :param n_pts: Number of grid points.
    :param delta_x: Grid spacing.
    :param kinetic_factor: Kinetic prefactor (e.g. B in cm^-1).
    :param periodic: If True, applies Meyer/Colbert-Miller periodic boundaries.
    :return: (n_pts x n_pts) Kinetic energy matrix as numpy array.
    """
    t_matrix = np.zeros((n_pts, n_pts), dtype=np.float64)

    if periodic:
        # Periodic Sinc-DVR (Meyer-Colbert-Miller formalism for [0, 2pi))
        is_odd = n_pts % 2 == 1
        for i in range(n_pts):
            for j in range(n_pts):
                diff = i - j
                if diff == 0:
                    if is_odd:
                        t_matrix[i, i] = kinetic_factor * ((n_pts**2 - 1.0) / 12.0)
                    else:
                        t_matrix[i, i] = kinetic_factor * ((n_pts**2 + 2.0) / 12.0)
                else:
                    arg = np.pi * diff / float(n_pts)
                    sin_sq = np.sin(arg) ** 2
                    if sin_sq < 1e-16:
                        sin_sq = 1e-16
                    if is_odd:
                        t_matrix[i, j] = kinetic_factor * (
                            (((-1.0) ** diff) * np.cos(arg)) / (2.0 * sin_sq)
                        )
                    else:
                        t_matrix[i, j] = kinetic_factor * (
                            ((-1.0) ** diff) / (2.0 * sin_sq)
                        )
    else:
        # Standard Colbert-Miller Sinc-DVR (infinite / Dirichlet domain)
        factor = kinetic_factor / (delta_x**2)
        for i in range(n_pts):
            for j in range(n_pts):
                diff = i - j
                if diff == 0:
                    t_matrix[i, i] = factor * (np.pi**2 / 3.0)
                else:
                    t_matrix[i, j] = factor * (2.0 * ((-1.0) ** diff) / (diff**2))

    return t_matrix


def build_dvr_hamiltonian(
    pes_spline_array: list[float] | np.ndarray | Any,
    kinetic_operator: float | tuple[float, float] | np.ndarray | Any = 1.0,
    grid_points: list[float] | np.ndarray | tuple[Any, ...] | None = None,
    dimensions: int = 1,
    periodic: bool = False,
) -> Any:
    """
    Constructs the discretized quantum mechanical Hamiltonian matrix (H = T + V).

    :param pes_spline_array: 1D or 2D potential energy values (cm^-1 or hartree).
    :param kinetic_operator: Rotational constant B, reduced mass, or explicit matrix.
    :param grid_points: 1D coordinate array or tuple of (grid_x, grid_y) for 2D.
    :param dimensions: Coordinate dimensionality (1 or 2).
    :param periodic: If True, uses periodic sinc-DVR for angular torsions.
    :return: Discretized Hamiltonian matrix as a JAX float64 array.
    """
    if not JAX_AVAILABLE:
        raise CoChemPrecisionError("JAX is required for build_dvr_hamiltonian.")

    enforce_jax_precision()
    pes = np.asarray(pes_spline_array, dtype=np.float64)

    if dimensions == 1:
        n_pts = len(pes)
        if n_pts == 0:
            raise ValueError("Potential energy array cannot be empty.")

        if grid_points is not None:
            pts = np.asarray(grid_points, dtype=np.float64)
            if len(pts) != n_pts:
                raise ValueError(
                    f"Grid points length ({len(pts)}) does not match PES ({n_pts})."
                )
            delta_x = float(pts[1] - pts[0]) if n_pts > 1 else 1.0
            if n_pts > 2 and not np.allclose(
                np.diff(pts), delta_x, rtol=1e-5, atol=1e-8
            ):
                logger.warning("Non-uniform grid spacing detected for sinc-DVR.")
        else:
            delta_x = 2.0 * np.pi / n_pts if periodic else 1.0

        if isinstance(kinetic_operator, int | float):
            if periodic:
                # Rotational constant B (cm^-1) for periodic rotor
                b_const = float(kinetic_operator)
                t_mat = _construct_1d_kinetic_matrix(
                    n_pts, delta_x, b_const, periodic=True
                )
            else:
                # Mass-based kinetic factor in cm^-1: hbar^2 / (2 * m)
                mass = float(kinetic_operator)
                kinetic_factor = HBAR_SQ_OVER_2M_U_CM1_A2 / mass if mass > 0 else 1.0
                t_mat = _construct_1d_kinetic_matrix(
                    n_pts, delta_x, kinetic_factor, periodic=False
                )
        else:
            t_mat = np.asarray(kinetic_operator, dtype=np.float64)
            if t_mat.shape != (n_pts, n_pts):
                raise ValueError(
                    f"Kinetic matrix shape {t_mat.shape} must match ({n_pts}, {n_pts})."
                )

        v_mat = np.diag(pes)
        h_mat = t_mat + v_mat
        return jnp.array(h_mat, dtype=jnp.float64)

    elif dimensions == 2:
        if pes.ndim == 2:
            nx, ny = pes.shape
            v_flat = pes.flatten()
        elif pes.ndim == 1:
            if (
                grid_points is None
                or not isinstance(grid_points, tuple)
                or len(grid_points) != 2
            ):
                raise ValueError(
                    "2D DVR with 1D PES array requires grid_points=(grid_x, grid_y)."
                )
            nx = len(grid_points[0])
            ny = len(grid_points[1])
            if len(pes) != nx * ny:
                raise ValueError(
                    f"1D PES length ({len(pes)}) does not match 2D grid ({nx}x{ny})."
                )
            v_flat = pes
        else:
            raise ValueError(f"Invalid PES array shape for 2D DVR: {pes.shape}")

        if (
            grid_points is not None
            and isinstance(grid_points, tuple)
            and len(grid_points) == 2
        ):
            gx, gy = np.asarray(grid_points[0]), np.asarray(grid_points[1])
            dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
            dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
        else:
            dx = 2.0 * np.pi / nx if periodic else 1.0
            dy = 2.0 * np.pi / ny if periodic else 1.0

        if isinstance(kinetic_operator, tuple) and len(kinetic_operator) == 2:
            bx, by = float(kinetic_operator[0]), float(kinetic_operator[1])
            tx = _construct_1d_kinetic_matrix(nx, dx, bx, periodic=periodic)
            ty = _construct_1d_kinetic_matrix(ny, dy, by, periodic=periodic)
        elif isinstance(kinetic_operator, int | float):
            b = float(kinetic_operator)
            tx = _construct_1d_kinetic_matrix(nx, dx, b, periodic=periodic)
            ty = _construct_1d_kinetic_matrix(ny, dy, b, periodic=periodic)
        else:
            raise ValueError(
                "Kinetic operator for 2D DVR must be a tuple (Bx, By) or scalar."
            )

        # 2D Kinetic operator via Kronecker product: T_2D = Tx (x) I_y + I_x (x) Ty
        ix = np.eye(nx, dtype=np.float64)
        iy = np.eye(ny, dtype=np.float64)
        t_2d = np.kron(tx, iy) + np.kron(ix, ty)

        v_2d = np.diag(v_flat)
        h_2d = t_2d + v_2d
        return jnp.array(h_2d, dtype=jnp.float64)

    else:
        raise ValueError(
            f"Unsupported dimensionality {dimensions}. Supported dimensions: 1 or 2."
        )


if JAX_AVAILABLE:

    @jax.jit
    def _jit_eigh_core(h: Any) -> tuple[Any, Any]:
        """Internal JIT-compiled XLA eigenvalue solver."""
        return jnp.linalg.eigh(h)


def jit_eigen_solver(
    hamiltonian_matrix: np.ndarray | Any,
) -> tuple[Any, Any]:
    """
    Solves the eigenvalue problem for the discretized DVR Hamiltonian via JAX JIT.

    Guarantees float64 precision and real eigenvalues via XLA-compiled eigh.

    :param hamiltonian_matrix: Real symmetric or complex Hermitian Hamiltonian.
    :return: Tuple of (eigenvalues, eigenvectors) as JAX float64 arrays.
    """
    if not JAX_AVAILABLE:
        raise CoChemPrecisionError("JAX is required for jit_eigen_solver.")

    enforce_jax_precision()
    h_jax = jnp.asarray(hamiltonian_matrix, dtype=jnp.float64)

    # Symmetrize matrix to prevent tiny numerical asymmetry artifacts
    h_sym = 0.5 * (h_jax + h_jax.T)

    evals, evecs = _jit_eigh_core(h_sym)
    return evals, evecs


def nan_tensor_watchdog(
    eigenvalues: np.ndarray | Any | None = None,
    hamiltonian: np.ndarray | Any | None = None,
    wavefunctions: np.ndarray | Any | None = None,
    alpha_regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intercepts divergent eigenvalues, NaNs, Infs, or ill-conditioned DVR matrices.

    Applies Tikhonov Regularization (H_reg = H + alpha * I) to stabilize ill-conditioned
    Hamiltonian matrices and restore positive-definite stability.

    :param eigenvalues: Array of eigenvalues to check for NaNs/Infs (optional).
    :param hamiltonian: Input Hamiltonian matrix to regularize if corrupted.
    :param wavefunctions: Wavefunctions corresponding to eigenvalues (optional).
    :param alpha_regularization: Damping coefficient for Tikhonov regularization.
    :return: Tuple of validated, finite (eigenvalues, wavefunctions).
    :raises ValueError: If neither eigenvalues nor hamiltonian are provided.
    :raises DVRConvergenceError: If regularization fails to resolve NaNs.
    """
    if eigenvalues is None and hamiltonian is None:
        raise ValueError(
            "At least one of eigenvalues or hamiltonian must be provided "
            "to nan_tensor_watchdog."
        )

    has_nan_evals = eigenvalues is not None and (
        np.isnan(np.asarray(eigenvalues)).any()
        or np.isinf(np.asarray(eigenvalues)).any()
    )

    has_nan_h = hamiltonian is not None and (
        np.isnan(np.asarray(hamiltonian)).any()
        or np.isinf(np.asarray(hamiltonian)).any()
    )

    if not has_nan_evals and not has_nan_h:
        if eigenvalues is not None:
            evals_np = np.asarray(eigenvalues, dtype=np.float64)
            if len(evals_np) > 1 and (evals_np[1] - evals_np[0]) < -1e-12:
                logger.warning(
                    f"Non-physical inverted eigenvalue spectrum detected "
                    f"(E1={evals_np[1]:.4f} < E0={evals_np[0]:.4f})."
                )
            evecs_np = (
                np.asarray(wavefunctions, dtype=np.float64)
                if wavefunctions is not None
                else np.empty((0, 0))
            )
            return evals_np, evecs_np
        elif hamiltonian is not None:
            h_clean = np.asarray(hamiltonian, dtype=np.float64)
            h_sym = 0.5 * (h_clean + h_clean.T)
            evals_np, evecs_np = np.linalg.eigh(h_sym)
            return evals_np, evecs_np

    logger.warning(
        f"NaN/Inf tensor divergence detected! Intercepting crash and applying "
        f"Tikhonov Regularization (lambda = {alpha_regularization:.2e})."
    )

    if hamiltonian is None:
        raise DVRConvergenceError(
            "NaN/Inf detected in eigenvalues, but Hamiltonian matrix was "
            "not provided for regularization."
        )

    h_cleaned = np.asarray(hamiltonian, copy=True, dtype=np.float64)
    nan_mask = np.isnan(h_cleaned) | np.isinf(h_cleaned)
    h_cleaned[nan_mask] = 0.0

    # Symmetrize
    h_cleaned = 0.5 * (h_cleaned + h_cleaned.T)

    # Apply Tikhonov Regularization: H_reg = H + alpha * I
    n = h_cleaned.shape[0]
    h_reg = h_cleaned + alpha_regularization * np.eye(n, dtype=np.float64)

    # Diagonalize regularized matrix
    if JAX_AVAILABLE:
        try:
            evals_jax, evecs_jax = jit_eigen_solver(h_reg)
            evals_np = np.asarray(evals_jax, dtype=np.float64)
            evecs_np = np.asarray(evecs_jax, dtype=np.float64)
        except Exception as e:
            logger.warning(
                f"JAX diagonalization failed during recovery ({e}); "
                "using NumPy fallback."
            )
            evals_np, evecs_np = np.linalg.eigh(h_reg)
    else:
        evals_np, evecs_np = np.linalg.eigh(h_reg)

    if np.isnan(evals_np).any() or np.isinf(evals_np).any():
        raise DVRConvergenceError(
            "DVR matrix remains unsolvable and divergent after Tikhonov Regularization."
        )

    logger.info(
        f"Successfully recovered finite DVR eigenvalue spectrum via "
        f"Tikhonov Regularization. Ground state: {evals_np[0]:.6f} cm^-1."
    )
    return evals_np, evecs_np


def localized_vpt2_coupling(
    dvr_energies: list[float] | np.ndarray,
    vpt2_matrix: list[list[float]] | np.ndarray,
    harmonic_frequencies: list[float] | np.ndarray,
    lam_mode_indices: list[int] | None = None,
    temperature_k: float = 298.15,
) -> dict[str, Any]:
    """
    Merges exact internal rotor energies with VPT2 outputs for orthogonal stiff modes.

    Drops harmonic modes corresponding to Large Amplitude Motions (LAM) to avoid
    thermodynamic double-counting, coupling stiff modes with the exact DVR manifold.

    :param dvr_energies: Array of exact DVR eigenvalues in cm^-1.
    :param vpt2_matrix: Anharmonic X_ij matrix in cm^-1.
    :param harmonic_frequencies: List of all harmonic frequencies in cm^-1.
    :param lam_mode_indices: Indices of normal modes to drop (omega < 50 cm^-1 default).
    :param temperature_k: Temperature in Kelvin for partition function evaluation.
    :return: Dictionary with decoupled stiff frequencies and partition functions.
    """
    dvr_e = np.sort(np.asarray(dvr_energies, dtype=np.float64))
    vpt2_x = np.asarray(vpt2_matrix, dtype=np.float64)
    harm_freqs = np.asarray(harmonic_frequencies, dtype=np.float64)
    n_modes = len(harm_freqs)

    # Determine which modes to drop as LAM
    if lam_mode_indices is None:
        dropped_modes = [i for i, freq in enumerate(harm_freqs) if freq < 50.0]
    else:
        dropped_modes = list(lam_mode_indices)

    stiff_mode_indices = [i for i in range(n_modes) if i not in dropped_modes]
    stiff_frequencies = harm_freqs[stiff_mode_indices]

    logger.info(
        f"Localized VPT2 Coupling: Dropping {len(dropped_modes)} LAM mode(s) "
        f"{dropped_modes} from harmonic set. Retaining {len(stiff_frequencies)} "
        "stiff orthogonal modes."
    )

    if temperature_k <= 0.0:
        return {
            "dropped_lam_modes": dropped_modes,
            "stiff_harmonic_frequencies": stiff_frequencies.tolist(),
            "q_dvr_rot": 1.0,
            "q_stiff_vib": 1.0,
            "q_coupled_total": 1.0,
            "dvr_ground_state_energy_cm1": float(dvr_e[0]) if len(dvr_e) > 0 else 0.0,
            "stiff_zpe_cm1": 0.0,
            "coupled_ground_state_energy_cm1": float(dvr_e[0])
            if len(dvr_e) > 0
            else 0.0,
            "temperature_k": float(temperature_k),
        }

    # Exact DVR torsional partition function Q_dvr
    # Q_dvr = sum_n exp(- (E_n - E_0) / (kB * T))
    hc_cm = PLANCK_CONSTANT_JS * SPEED_OF_LIGHT_CMS  # Joules per cm^-1
    kt_j = BOLTZMANN_CONSTANT_JK * temperature_k

    e0 = dvr_e[0] if len(dvr_e) > 0 else 0.0
    relative_dvr_e = dvr_e - e0

    q_dvr = 0.0
    for energy_cm1 in relative_dvr_e:
        e_j = energy_cm1 * hc_cm
        arg = -e_j / kt_j
        if arg > -700.0:
            q_dvr += math.exp(arg)

    # Stiff vibrational partition function Q_stiff
    # Q_stiff = prod_i [ 1 / (1 - exp(- h c nu_i / (kB T))) ]
    q_stiff = 1.0
    for nu in stiff_frequencies:
        nu_clamped = max(float(nu), 10.0)
        e_vib_j = nu_clamped * hc_cm
        exp_arg = -e_vib_j / kt_j
        if exp_arg < -700.0:
            mode_q = 1.0
        else:
            denom = 1.0 - math.exp(exp_arg)
            mode_q = 1.0 / denom if abs(denom) > 1e-12 else 1.0
        q_stiff *= mode_q

    q_coupled_total = q_dvr * q_stiff

    # Compute anharmonic zero-point energy of stiff modes using upper triangular sum:
    # E_anh_ZPE = 0.25 * sum_{i <= j} X_{ij}
    stiff_x = (
        vpt2_x[np.ix_(stiff_mode_indices, stiff_mode_indices)]
        if vpt2_x.ndim == 2 and vpt2_x.shape[0] == n_modes
        else np.zeros((len(stiff_frequencies), len(stiff_frequencies)))
    )
    harmonic_zpe = 0.5 * float(np.sum(stiff_frequencies))
    anharmonic_zpe_correction = 0.25 * float(np.sum(np.triu(stiff_x)))
    total_stiff_zpe = harmonic_zpe + anharmonic_zpe_correction

    coupled_ground_state_energy = float(e0 + total_stiff_zpe)

    return {
        "dropped_lam_modes": dropped_modes,
        "stiff_harmonic_frequencies": stiff_frequencies.tolist(),
        "q_dvr_rot": float(q_dvr),
        "q_stiff_vib": float(q_stiff),
        "q_coupled_total": float(q_coupled_total),
        "dvr_ground_state_energy_cm1": float(e0),
        "stiff_zpe_cm1": float(total_stiff_zpe),
        "coupled_ground_state_energy_cm1": coupled_ground_state_energy,
        "temperature_k": float(temperature_k),
    }


class JaxDVRBuilder:
    """
    High-level orchestration class for hardware-accelerated DVR Schrödinger solvers.
    """

    def __init__(self, dimensions: int = 1, enable_x64: bool = True) -> None:
        """
        Initializes the JAX DVR Builder engine.

        :param dimensions: Coordinate dimensionality (1 or 2).
        :param enable_x64: Strictly enforce 64-bit precision.
        """
        self.dimensions = dimensions
        if enable_x64:
            self.device_info = enforce_jax_precision()
        else:
            self.device_info = {}

    def solve_1d_rotor(
        self,
        grid_points: list[float] | np.ndarray,
        energies_cm1: list[float] | np.ndarray,
        rotational_constant_cm1: float = 1.0,
        periodic: bool = True,
    ) -> dict[str, Any]:
        """
        Builds and solves the 1D DVR Hamiltonian for a hindered internal rotor.

        :param grid_points: Torsional angles in radians.
        :param energies_cm1: Potential energy values at each grid point in cm^-1.
        :param rotational_constant_cm1: Rotational constant B (in cm^-1).
        :param periodic: Use periodic boundary conditions (Meyer-Colbert-Miller).
        :return: Dictionary with eigenvalues, wavefunctions, and ground state.
        """
        h_matrix = build_dvr_hamiltonian(
            pes_spline_array=energies_cm1,
            kinetic_operator=rotational_constant_cm1,
            grid_points=grid_points,
            dimensions=1,
            periodic=periodic,
        )

        evals, evecs = jit_eigen_solver(h_matrix)

        # Pass through NaN watchdog for validation with wavefunctions
        evals_clean, evecs_clean = nan_tensor_watchdog(
            eigenvalues=evals,
            hamiltonian=h_matrix,
            wavefunctions=evecs,
        )

        tunneling = (
            float(evals_clean[1] - evals_clean[0]) if len(evals_clean) > 1 else 0.0
        )

        return {
            "hamiltonian": h_matrix,
            "eigenvalues": evals_clean,
            "wavefunctions": evecs_clean,
            "ground_state_energy_cm1": float(evals_clean[0]),
            "tunneling_splitting_cm1": tunneling,
            "num_points": len(evals_clean),
        }

    def solve_2d_coupled_rotors(
        self,
        grid_points_x: list[float] | np.ndarray,
        grid_points_y: list[float] | np.ndarray,
        pes_2d_cm1: np.ndarray,
        rotational_constants_cm1: tuple[float, float] = (1.0, 1.0),
        periodic: bool = True,
    ) -> dict[str, Any]:
        """
        Builds and solves the 2D DVR Hamiltonian for coupled internal rotors.

        :param grid_points_x: Grid points along dihedral coordinate 1.
        :param grid_points_y: Grid points along dihedral coordinate 2.
        :param pes_2d_cm1: 2D potential energy surface matrix (Nx x Ny).
        :param rotational_constants_cm1: Tuple of (Bx, By) in cm^-1.
        :param periodic: Use periodic boundary conditions.
        :return: Dictionary with 2D eigenvalues, wavefunctions, and ground state.
        """
        h_2d = build_dvr_hamiltonian(
            pes_spline_array=pes_2d_cm1,
            kinetic_operator=rotational_constants_cm1,
            grid_points=(grid_points_x, grid_points_y),
            dimensions=2,
            periodic=periodic,
        )

        evals, evecs = jit_eigen_solver(h_2d)
        evals_clean, evecs_clean = nan_tensor_watchdog(
            eigenvalues=evals,
            hamiltonian=h_2d,
            wavefunctions=evecs,
        )

        return {
            "hamiltonian_2d": h_2d,
            "eigenvalues": evals_clean,
            "wavefunctions": evecs_clean,
            "ground_state_energy_cm1": float(evals_clean[0]),
            "num_states": len(evals_clean),
        }
