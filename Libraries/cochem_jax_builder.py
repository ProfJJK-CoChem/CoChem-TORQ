"""Finite-grid 1D/2D reduced nuclear-motion models using JAX eigensolvers.

These solve the supplied discretized model. Grid convergence, molecular kinetic
operators, couplings and physical applicability require separate validation;
finite-grid results are not exact molecular nuclear-motion solutions.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from numbers import Real
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

# CODATA 2022 Physical Constants (Exact SI definitions)
PLANCK_CONSTANT_JS = 6.62607015e-34  # Exact h (J s)
BOLTZMANN_CONSTANT_JK = 1.380649e-23  # Exact kB (J/K)
SPEED_OF_LIGHT_CMS = 29979245800.0  # Exact c (cm/s)
HBAR_JS = PLANCK_CONSTANT_JS / (2.0 * math.pi)
ATOMIC_MASS_CONSTANT_KG = 1.66053906892e-27  # Measured kg/u (CODATA 2022)

# Conversion factor hbar^2 / (2 * m_u) in cm^-1 * Angstrom^2 * amu:
# Calculated dynamically from exact CODATA 2022 definitions
HBAR_SQ_OVER_2M_U_CM1_A2 = PLANCK_CONSTANT_JS / (
    8.0 * (math.pi**2) * SPEED_OF_LIGHT_CMS * ATOMIC_MASS_CONSTANT_KG * 1e-20
)


def get_atomic_mass(symbol: str) -> float:
    """Resolve the requested tabulated isotope without substituting another mass."""
    from Libraries.cochem_isotopes import isotope_mass
    return isotope_mass(symbol)


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

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class DVRConvergenceError(RuntimeError):
    """Raised when the DVR eigenvalue solver fails to converge."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


def enforce_jax_precision() -> dict[str, Any]:
    """
    Enforces JAX 64-bit floating point precision (float64) and identifies hardware.

    Spectroscopic tunneling splittings can exist on the order of 10^-6 cm^-1.
    # Maps execution paths: JIT/GPU (HPC), MPS (Apple Silicon), CPU fallback.
    # Standard 32-bit floating point precision causes numerical underflow.

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

    # Map execution paths
    backend_lower = default_backend.lower()
    if backend_lower in ("gpu", "cuda", "rocm"):
        execution_path = "JIT/GPU (HPC)"
    elif backend_lower in ("mps", "metal"):
        execution_path = "MPS (Apple Silicon)"
    else:
        import platform

        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            execution_path = "MPS/Apple Silicon CPU Fallback"
        else:
            execution_path = "CPU Vectorization Fallback (XLA AVX)"

    device_info = {
        "platform": default_backend,
        "execution_path": execution_path,
        "devices": [str(d) for d in devices],
        "device_count": len(devices),
        "x64_enabled": True,
        "dtype": "float64",
    }

    logger.info(
        f"JAX float64 precision enforced on {default_backend.upper()} "
        f"[{execution_path}] ({len(devices)} device(s): {devices[0]})"
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


def _real_finite_array(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be real; imaginary components cannot be discarded")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _validated_grid(grid: Any, count: int, periodic: bool) -> float:
    if grid is None:
        raise ValueError("Explicit coordinate grid required for a scalar kinetic operator")
    points = _real_finite_array(grid, "Coordinate grid")
    if points.shape != (count,) or count < 2:
        raise ValueError("Each coordinate grid must match the PES axis and contain at least two points")
    spacing = float(points[1] - points[0])
    if spacing <= 0 or not np.allclose(np.diff(points), spacing, rtol=1e-10, atol=1e-12):
        raise ValueError("Sinc DVR requires a strictly increasing uniform grid")
    if periodic and not math.isclose(count * spacing, 2 * math.pi, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Periodic angular grid must span 2*pi radians without a duplicated endpoint")
    return spacing


def _kinetic_prefactor(operator: Any, periodic: bool) -> float:
    if isinstance(operator, str):
        if periodic:
            raise ValueError("An isotope mass does not define angular inertia; provide B in cm^-1")
        value = get_atomic_mass(operator)
    elif isinstance(operator, Real) and not isinstance(operator, (bool, np.bool_)):
        value = float(operator)
    else:
        raise ValueError("Provide an explicit positive finite mass (u) or angular B (cm^-1)")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Mass or angular B must be finite and strictly positive")
    return value if periodic else HBAR_SQ_OVER_2M_U_CM1_A2 / value


def build_dvr_hamiltonian(
    pes_spline_array: list[float] | np.ndarray | Any,
    kinetic_operator: Any = None,
    grid_points: list[float] | np.ndarray | tuple[Any, ...] | None = None,
    dimensions: int = 1,
    periodic: bool = False,
) -> Any:
    """Build H=T+V in cm^-1 for a declared reduced-coordinate model.

    Potential energies must be supplied in cm^-1. For nonperiodic Cartesian
    sinc grids, coordinates are Angstrom and scalar operators are masses in u.
    For periodic angular grids, coordinates are radians on [q0,q0+2*pi), and
    scalar operators are positive B constants in cm^-1. Both dimensions use
    the same unit convention. A full finite real symmetric kinetic matrix is
    also accepted directly in cm^-1. No mass, spacing or B is inferred.
    """
    if not JAX_AVAILABLE:
        raise CoChemPrecisionError("JAX is required for build_dvr_hamiltonian")
    if dimensions not in (1, 2):
        raise ValueError("Supported dimensions are 1 and 2")
    if kinetic_operator is None:
        raise ValueError("An explicit kinetic operator is required")
    pes = _real_finite_array(pes_spline_array, "Potential energies")
    if dimensions == 1:
        if pes.ndim != 1 or pes.size < 2:
            raise ValueError("1D potential must be a vector of at least two points")
        shape = (len(pes),)
    elif pes.ndim == 2 and min(pes.shape) >= 2:
        shape = pes.shape
    elif pes.ndim == 1 and isinstance(grid_points, tuple) and len(grid_points) == 2:
        shape = (len(grid_points[0]), len(grid_points[1]))
        if min(shape) < 2 or pes.size != shape[0] * shape[1]:
            raise ValueError("Flattened 2D PES size must match both coordinate grids")
    else:
        raise ValueError("2D potential requires a two-dimensional array or explicit matching grid pair")
    flat = pes.ravel()
    scalar = isinstance(kinetic_operator, (str, Real))
    pair = isinstance(kinetic_operator, tuple) and len(kinetic_operator) == 2
    if not scalar and not pair:
        kinetic = _real_finite_array(kinetic_operator, "Kinetic matrix")
        if kinetic.shape != (len(flat), len(flat)) or not np.allclose(kinetic, kinetic.T, rtol=1e-12, atol=1e-12):
            raise ValueError("Kinetic matrix must be symmetric and match the full PES dimension")
    elif dimensions == 1:
        spacing = _validated_grid(grid_points, shape[0], periodic)
        kinetic = _construct_1d_kinetic_matrix(shape[0], spacing, _kinetic_prefactor(kinetic_operator, periodic), periodic)
    else:
        if not isinstance(grid_points, tuple) or len(grid_points) != 2:
            raise ValueError("2D scalar kinetics require an explicit (grid_x, grid_y) tuple")
        spacings = [_validated_grid(grid, count, periodic) for grid, count in zip(grid_points, shape)]
        operators = kinetic_operator if pair else (kinetic_operator, kinetic_operator)
        factors = [_kinetic_prefactor(operator, periodic) for operator in operators]
        tx = _construct_1d_kinetic_matrix(shape[0], spacings[0], factors[0], periodic)
        ty = _construct_1d_kinetic_matrix(shape[1], spacings[1], factors[1], periodic)
        kinetic = np.kron(tx, np.eye(shape[1])) + np.kron(np.eye(shape[0]), ty)
    enforce_jax_precision()
    return jnp.asarray(kinetic + np.diag(flat), dtype=jnp.float64)


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

    raw = np.asarray(hamiltonian_matrix)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    h = np.asarray(raw, dtype=dtype)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0
            or not np.isfinite(h).all() or not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12)):
        raise DVRConvergenceError("Hamiltonian must be finite and Hermitian; no symmetrization repair is performed")
    enforce_jax_precision()
    # Eigensolvers may treat sub-tolerance roundoff through their standard
    # Hermitian algorithms; no changed physical matrix is constructed here.
    h_jax = jnp.asarray(h, dtype=jnp.complex128 if np.iscomplexobj(h) else jnp.float64)
    evals, evecs = _jit_eigh_core(h_jax)
    return evals, evecs


def nan_tensor_watchdog(
    eigenvalues: np.ndarray | Any | None = None,
    hamiltonian: np.ndarray | Any | None = None,
    wavefunctions: np.ndarray | Any | None = None,
    alpha_regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a spectrum; recompute only from the unchanged finite Hamiltonian.

    ``alpha_regularization`` is retained for call compatibility and never applied.
    Replacing matrix entries or shifting a physical Hamiltonian is not recovery.
    """
    if eigenvalues is None and hamiltonian is None:
        raise ValueError("Provide eigenvalues or a Hamiltonian")
    h = None if hamiltonian is None else np.asarray(hamiltonian, dtype=np.complex128 if np.iscomplexobj(hamiltonian) else np.float64)
    if h is not None:
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0:
            raise DVRConvergenceError("Hamiltonian must be a nonempty square matrix")
        if not np.isfinite(h).all() or not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12):
            raise DVRConvergenceError("Nonfinite or non-Hermitian Hamiltonian; no values repaired")
    if eigenvalues is not None and np.iscomplexobj(eigenvalues):
        raise DVRConvergenceError("Hermitian eigenvalues must be real")
    e = None if eigenvalues is None else np.asarray(eigenvalues, dtype=float)
    if e is None or not np.isfinite(e).all():
        if h is None:
            raise DVRConvergenceError("Invalid eigenvalues without a finite Hamiltonian")
        e, vectors = np.linalg.eigh(h)
    else:
        vectors = np.asarray(wavefunctions, dtype=np.complex128 if np.iscomplexobj(wavefunctions) else np.float64) if wavefunctions is not None else np.empty((0, 0))
        if h is not None and wavefunctions is None:
            reference_values, vectors = np.linalg.eigh(h)
            if e.shape != reference_values.shape or not np.allclose(e, reference_values, rtol=1e-9, atol=1e-9):
                raise DVRConvergenceError("Eigenvalues do not solve the supplied Hamiltonian")
    if e.ndim != 1 or not e.size or not np.isfinite(e).all() or np.any(np.diff(e) < -1e-12):
        raise DVRConvergenceError("Spectrum must be finite, nonempty, and ordered")
    if vectors.size and not np.isfinite(vectors).all():
        raise DVRConvergenceError("Nonfinite wavefunctions")
    if vectors.size and (vectors.ndim != 2 or vectors.shape[1] != len(e)
            or not np.allclose(vectors.conj().T @ vectors, np.eye(len(e)), rtol=1e-9, atol=1e-9)):
        raise DVRConvergenceError("Eigenvectors must be an orthonormal basis")
    if wavefunctions is not None and (vectors.ndim != 2 or vectors.shape[1] != len(e)
                                     or vectors.shape[0] < len(e)):
        raise DVRConvergenceError("Eigenvector/eigenvalue dimensions disagree")
    if h is not None:
        if vectors.shape != h.shape or e.shape != (h.shape[0],):
            raise DVRConvergenceError("Eigenvector/eigenvalue dimensions disagree")
        if not np.allclose(h @ vectors, vectors * e, rtol=1e-9, atol=1e-9):
            raise DVRConvergenceError("Eigenpairs do not solve the supplied Hamiltonian")
    return e, vectors


def localized_vpt2_coupling(
    dvr_energies: list[float] | np.ndarray,
    vpt2_matrix: list[list[float]] | np.ndarray,
    harmonic_frequencies: list[float] | np.ndarray,
    lam_mode_indices: list[int] | None = None,
    temperature_k: float = 298.15,
) -> dict[str, Any]:
    """
    Combines a supplied reduced DVR spectrum with stiff-mode VPT2 zero-point corrections.

    Drops harmonic modes corresponding to Large Amplitude Motions (LAM) to avoid
    thermodynamic double-counting. DVR/stiff-mode coupling is neglected, and thermal stiff-mode sums remain harmonic.

    :param dvr_energies: Supplied finite-grid DVR eigenvalues in cm^-1.
    :param vpt2_matrix: Anharmonic X_ij matrix in cm^-1.
    :param harmonic_frequencies: List of all harmonic frequencies in cm^-1.
    :param lam_mode_indices: Explicit indices of normal modes replaced by the DVR model.
    :param temperature_k: Temperature in Kelvin for partition function evaluation.
    :return: Dictionary with decoupled stiff frequencies and partition functions.
    """
    dvr_e = np.sort(np.asarray(dvr_energies, dtype=np.float64))
    vpt2_x = np.asarray(vpt2_matrix, dtype=np.float64)
    harm_freqs = np.asarray(harmonic_frequencies, dtype=np.float64)
    n_modes = len(harm_freqs)

    if dvr_e.ndim != 1 or not dvr_e.size or not np.isfinite(dvr_e).all():
        raise ValueError("A nonempty finite DVR spectrum is required")
    if harm_freqs.ndim != 1 or not np.isfinite(harm_freqs).all():
        raise ValueError("Harmonic frequencies must be a finite vector")
    if vpt2_x.shape != (n_modes, n_modes) or not np.isfinite(vpt2_x).all():
        raise ValueError("The full finite anharmonic matrix is required")
    if lam_mode_indices is None:
        raise ValueError("LAM mode selection must be explicit; frequency alone is insufficient")
    dropped_modes = list(lam_mode_indices)
    if len(set(dropped_modes)) != len(dropped_modes) or any(type(i) is not int or i < 0 or i >= n_modes for i in dropped_modes):
        raise ValueError("Invalid or duplicate LAM mode indices")
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Thermal partition evaluation requires finite T > 0")

    stiff_mode_indices = [i for i in range(n_modes) if i not in dropped_modes]
    stiff_frequencies = harm_freqs[stiff_mode_indices]

    logger.info(
        f"Localized VPT2 Coupling: Dropping {len(dropped_modes)} LAM mode(s) "
        f"{dropped_modes} from harmonic set. Retaining {len(stiff_frequencies)} "
        "stiff orthogonal modes."
    )

    if np.any(stiff_frequencies <= 0):
        raise ValueError("Stiff-mode frequencies must be positive; no frequency floor applied")

    # Truncated supplied DVR torsional partition sum Q_dvr
    # Q_dvr = sum_n exp(- (E_n - E_0) / (kB * T))
    hc_cm = PLANCK_CONSTANT_JS * SPEED_OF_LIGHT_CMS  # Joules per cm^-1
    kt_j = BOLTZMANN_CONSTANT_JK * temperature_k

    e0 = dvr_e[0]
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
        e_vib_j = float(nu) * hc_cm
        exp_arg = -e_vib_j / kt_j
        if exp_arg < -700.0:
            mode_q = 1.0
        else:
            denom = -math.expm1(exp_arg)
            mode_q = 1.0 / denom
        q_stiff *= mode_q

    q_coupled_total = q_dvr * q_stiff

    # Compute anharmonic zero-point energy of stiff modes using upper triangular sum:
    # E_anh_ZPE = 0.25 * sum_{i <= j} X_{ij}
    stiff_x = vpt2_x[np.ix_(stiff_mode_indices, stiff_mode_indices)]
    harmonic_zpe = 0.5 * float(np.sum(stiff_frequencies))
    anharmonic_zpe_correction = 0.25 * float(np.sum(np.triu(stiff_x)))
    total_stiff_zpe = harmonic_zpe + anharmonic_zpe_correction

    coupled_ground_state_energy = float(e0 + total_stiff_zpe)

    return {
        "model": "separable_DVR_plus_stiff_VPT2_ZPE_harmonic_thermal_sum",
        "couplings_between_DVR_and_stiff_modes": "neglected",
        "dvr_partition_scope": "supplied finite spectrum; tail convergence unverified",
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
        rotational_constant_cm1: float | None = None,
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
        if not periodic:
            raise ValueError("This rotor API requires periodic angles; use an explicit kinetic matrix for other boundary conditions")
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
            float(evals_clean[1] - evals_clean[0]) if len(evals_clean) > 1 else None
        )

        return {
            "hamiltonian": h_matrix,
            "eigenvalues": evals_clean,
            "wavefunctions": evecs_clean,
            "ground_state_energy_cm1": float(evals_clean[0]),
            "lowest_level_gap_cm1": tunneling,
            "tunneling_splitting_cm1": None,
            "tunneling_assignment_status": "unavailable; symmetry/localization assignment required",
            "model": "finite_grid_reduced_1D_rotor",
            "num_points": len(evals_clean),
        }

    def solve_2d_coupled_rotors(
        self,
        grid_points_x: list[float] | np.ndarray,
        grid_points_y: list[float] | np.ndarray,
        pes_2d_cm1: np.ndarray,
        rotational_constants_cm1: tuple[float, float] | None = None,
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
        if not periodic:
            raise ValueError("This rotor API requires periodic angles; use an explicit kinetic matrix for other boundary conditions")
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
            "model": "finite_grid_reduced_2D_rotor",
        }
