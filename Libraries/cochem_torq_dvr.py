"""Periodic constant-kinetic torsional DVR for explicitly supplied potential scans.

The caller supplies and justifies the reduced rotational constant F and potential
model. This mathematical solver cannot establish isotope masses, relaxed-scan
provenance, rotation-vibration coupling or molecule-specific state assignments.
"""

from __future__ import annotations

import os

# Mandated by Method Matrix §QS-3 line 167:
# Strict FP64 double precision and bounded CUDA on startup
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline

from cochem_torq.units import (
    ATOMIC_MASS_KG,
    AVOGADRO_PER_MOL,
    PLANCK_JOULE_SECOND,
    SPEED_OF_LIGHT_METRE_SECOND,
)

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

# Physical conversion constants
KCAL_MOL_TO_CM1: float = 4184.0 / (
    AVOGADRO_PER_MOL * PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_METRE_SECOND * 100.0
)
CM1_TO_MHZ: float = SPEED_OF_LIGHT_METRE_SECOND * 100.0 / 1e6
HBAR2_2I_COEFF: float = PLANCK_JOULE_SECOND / (
    8.0 * math.pi**2 * SPEED_OF_LIGHT_METRE_SECOND * 100.0 * ATOMIC_MASS_KG * 1e-20
)


class RelaxedPESTorsionalDVR:
    """Finite Fourier DVR for an explicitly declared constant-F periodic model."""

    def __init__(
        self,
        theta_scan_rad: np.ndarray | Sequence[float],
        energies_kcal: np.ndarray | Sequence[float],
        n_points: int = 100,
        f_rotational_constant_cm1: float | None = None,
        symbols: Sequence[str] | None = None,
        coords: np.ndarray | Sequence[Sequence[float]] | None = None,
        n_grid: int | None = None,
    ) -> None:
        """Initialize Sinc-DVR solver with authentic torsional scan data.

        Parameters
        ----------
        theta_scan_rad : array-like
            Sampled dihedral angles in radians over [0, 2*pi].
        energies_kcal : array-like
            Relaxed electronic energies in kcal/mol relative to minimum.
        n_points : int, default=100
            Number of Colbert-Miller spatial grid points N.
        f_rotational_constant_cm1 : float, optional
            Explicit independently justified internal-rotation constant F in cm^-1.
            Missing values fail; neither an element name nor whole-molecule inertia
            determines a reduced internal-rotation kinetic coefficient.
        symbols : sequence of str, optional
            Legacy metadata argument; does not derive a kinetic coefficient.
        coords : array-like, optional
            Legacy metadata argument; does not derive a kinetic coefficient.
        n_grid : int, optional
            Alias for n_points.
        """
        for label, values in (("angles", theta_scan_rad), ("energies", energies_kcal)):
            raw = np.asarray(values)
            if np.iscomplexobj(raw):
                raise ValueError(f"Torsional {label} must be real, not complex")
        self.theta_scan_rad = np.asarray(theta_scan_rad, dtype=np.float64)
        self.energies_kcal = np.asarray(energies_kcal, dtype=np.float64)
        requested_points = n_grid if n_grid is not None else n_points
        if isinstance(requested_points, bool) or not isinstance(
            requested_points, (int, np.integer)
        ):
            raise ValueError("The DVR grid size must be an integer of at least four")
        self.n_points = int(requested_points)
        if self.n_points < 4:
            raise ValueError("The DVR grid size must be an integer of at least four")
        if (
            self.theta_scan_rad.ndim != 1
            or self.energies_kcal.ndim != 1
            or self.theta_scan_rad.shape != self.energies_kcal.shape
            or len(self.theta_scan_rad) < 4
            or not np.isfinite(self.theta_scan_rad).all()
            or not np.isfinite(self.energies_kcal).all()
            or not np.all(np.diff(self.theta_scan_rad) > 0)
        ):
            raise ValueError(
                "A matching finite scan on increasing angles "
                "with at least four samples is required"
            )
        if not (
            np.isclose(self.theta_scan_rad[0], 0.0, atol=1e-12, rtol=0.0)
            and np.isclose(self.theta_scan_rad[-1], 2.0 * math.pi, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("The periodic scan must explicitly span [0, 2*pi]")
        if f_rotational_constant_cm1 is None:
            raise ValueError(
                "An explicit independently justified positive F in cm^-1 "
                "is required; no inertia is inferred"
            )
        if isinstance(f_rotational_constant_cm1, (bool, complex, np.complexfloating)):
            raise ValueError(
                "The supplied reduced rotational constant F must be real and positive"
            )
        self.f_rotational_constant_cm1 = float(f_rotational_constant_cm1)
        if (
            not math.isfinite(self.f_rotational_constant_cm1)
            or self.f_rotational_constant_cm1 <= 0
        ):
            raise ValueError(
                "The supplied reduced rotational constant F must be finite and positive"
            )

        # Construct C^2 periodic cubic B-spline interpolation [M]
        # Ensure endpoints wrap periodically for smooth boundary conditions
        self.spline = make_interp_spline(
            self.theta_scan_rad,
            self.energies_kcal,
            bc_type="periodic",
            k=3,
        )

        # Grid discretization theta_i = 2 * pi * i / N on [0, 2*pi)
        self.grid_rad = np.array(
            [2.0 * math.pi * i / self.n_points for i in range(self.n_points)],
            dtype=np.float64,
        )
        self.v_grid_cm1 = self.spline(self.grid_rad) * KCAL_MOL_TO_CM1

        # Build Colbert-Miller kinetic energy matrix and Hamiltonian
        self.h_matrix = self._build_hamiltonian()
        self.eigenvalues_cm1: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None

    @staticmethod
    def _compute_reduced_f(symbols, coords) -> float:
        """Reject the historical whole-molecule/peroxide inertia substitution."""
        raise ValueError(
            "Provide explicit F from a declared rotor/frame kinetic model; "
            "geometry alone is insufficient"
        )

    def _build_hamiltonian(self) -> np.ndarray:
        """Represent -F d²/dtheta² on the finite periodic Fourier basis.

        Fourier modes have E_m=F*m². Transforming that diagonal operator yields
        the real symmetric circulant DVR kinetic matrix for either grid parity.
        """
        modes = np.fft.fftfreq(self.n_points, d=1.0 / self.n_points)
        kernel = np.fft.ifft(self.f_rotational_constant_cm1 * modes**2).real
        indices = np.arange(self.n_points)
        kinetic = kernel[(indices[:, None] - indices[None, :]) % self.n_points]
        return kinetic + np.diag(self.v_grid_cm1)

    def diagonalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalizes Hamiltonian using JAX 64-bit or NumPy eigh.

        Returns
        -------
        eigenvalues_cm1 : np.ndarray
            Eigenvalues sorted in ascending order (cm^-1).
        eigenvectors : np.ndarray
            Orthonormal torsional wavefunctions.
        """
        if HAS_JAX:
            h_jnp = jnp.asarray(self.h_matrix, dtype=jnp.float64)
            w, v = jnp.linalg.eigh(h_jnp)
            self.eigenvalues_cm1 = np.asarray(w, dtype=np.float64)
            self.eigenvectors = np.asarray(v, dtype=np.float64)
        else:
            w, v = np.linalg.eigh(self.h_matrix)
            self.eigenvalues_cm1 = np.asarray(w, dtype=np.float64)
            self.eigenvectors = np.asarray(v, dtype=np.float64)

        return self.eigenvalues_cm1, self.eigenvectors

    @property
    def tunneling_splitting_cm1(self) -> float:
        """Lowest adjacent-level gap; a tunneling-state assignment is not inferred."""
        if self.eigenvalues_cm1 is None:
            self.diagonalize()
        assert self.eigenvalues_cm1 is not None
        return float(self.eigenvalues_cm1[1] - self.eigenvalues_cm1[0])

    @property
    def tunneling_splitting_mhz(self) -> float:
        """Authentic ground-state tunneling splitting delta E_01 in MHz."""
        return self.tunneling_splitting_cm1 * CM1_TO_MHZ

    @property
    def barrier_height_kcal(self) -> float:
        """Torsional barrier height V_n in kcal/mol."""
        return float(np.max(self.energies_kcal) - np.min(self.energies_kcal))

    @property
    def barrier_height_cm1(self) -> float:
        """Torsional barrier height V_n in cm^-1."""
        return self.barrier_height_kcal * KCAL_MOL_TO_CM1

    def solve(self) -> dict[str, Any]:
        """Diagonalizes Hamiltonian and returns eigensystem and tunneling metrics."""
        eigenvalues, _ = self.diagonalize()
        ground = float(eigenvalues[0])
        return {
            "ground_energy_cm1": ground,
            "eigenvalues_cm1": eigenvalues.tolist(),
            "tunneling_split_cm1": self.tunneling_splitting_cm1,
            "tunneling_split_mhz": self.tunneling_splitting_mhz,
            "barrier_height_kcal": self.barrier_height_kcal,
            "barrier_height_cm1": self.barrier_height_cm1,
            "f_rot_cm1": self.f_rotational_constant_cm1,
            "kinetic_model": "explicit_constant_F_periodic_fourier_dvr",
            "state_assignment": "unavailable",
            "solver_backend": "jax" if HAS_JAX else "numpy",
            "quality_flags": [
                "constant_kinetic_approximation",
                "uncalibrated_level_gap",
            ],
        }

    @classmethod
    def from_hdf5(
        cls,
        h5_path: Path | str,
        group: str = "torsion_scan",
        n_points: int = 100,
        symbols: Sequence[str] | None = None,
        coords: np.ndarray | Sequence[Sequence[float]] | None = None,
        f_rotational_constant_cm1: float | None = None,
    ) -> RelaxedPESTorsionalDVR:
        """Load supplied scan bytes; file presence does not qualify the potential."""
        import filelock
        import h5py

        p = Path(h5_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Torsional HDF5 file not found: {p}")

        lock_path = p.with_suffix(".h5.lock")
        with filelock.FileLock(lock_path, timeout=15.0):
            with h5py.File(p, "r", libver="latest", swmr=True) as h5f:
                if group not in h5f:
                    raise KeyError(f"Group '{group}' not found in HDF5 file: {p}")
                grp = h5f[group]
                angles_deg = np.array(grp["dihedral_deg"])
                energies_kcal = np.array(grp["energy_kcal_mol"])

        if np.iscomplexobj(angles_deg) or np.iscomplexobj(energies_kcal):
            raise ValueError("Torsional HDF5 scan values must be real, not complex")
        angles_rad = np.radians(angles_deg)
        return cls(
            theta_scan_rad=angles_rad,
            energies_kcal=energies_kcal,
            n_points=n_points,
            symbols=symbols,
            coords=coords,
            f_rotational_constant_cm1=f_rotational_constant_cm1,
        )
