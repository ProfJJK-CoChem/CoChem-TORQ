"""CoChem-TORQ: Phase 4 Multi-Fidelity Spline Router & WKB Tunneling Estimator.

===========================================================================
Phase 4 (Stage 3.0 / 6.0) Implementation
----------------------------------------
Evaluates quantum and ML potential energy surface (PES) topography to fit
continuous 1D and 2D SciPy splines across discrete angular/torsional grid points,
analytically isolates critical topographic nodes (minima, transition state
saddles, maxima), and computes semiclassical Wentzel-Kramers-Brillouin (WKB)
quantum tunneling probabilities and splitting estimators for light rotors (-CH3, -OH).

Authoritative Standards:
- Method Matrix: Stage 3.0 / 6.0 Spline Fitting & Quantum Tunneling Routing
- Semiclassical Wentzel-Kramers-Brillouin (WKB) Tunneling Formulation
- CODATA 2022 Fundamental Physical Constants & Mendeleev Dynamic Masses
- Filesystem Air-Gap Policy (Domain A Static, Domain B Artifacts)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import quad  # type: ignore[import-untyped]
from scipy.interpolate import (  # type: ignore[import-untyped]
    CubicSpline,
    RectBivariateSpline,
)
from scipy.optimize import brentq, root  # type: ignore[import-untyped]

from cochem_torq.units import (
    ATOMIC_MASS_KG,
    AVOGADRO_PER_MOL,
    HARTREE_CM1,
    HARTREE_JOULE,
    PLANCK_JOULE_SECOND,
    SPEED_OF_LIGHT_METRE_SECOND,
)

# Configure module-level logger
logger: Final[logging.Logger] = logging.getLogger("CoChem-TORQ.Slicer")

# =============================================================================
# Fundamental Physical & Unit Conversion Constants (CODATA 2022)
# =============================================================================

HARTREE_TO_KCAL_MOL: Final[float] = HARTREE_JOULE * AVOGADRO_PER_MOL / 4184.0
HARTREE_TO_CM1: Final[float] = HARTREE_CM1
KCAL_MOL_TO_CM1: Final[float] = 4184.0 / (
    AVOGADRO_PER_MOL * PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_METRE_SECOND * 100.0
)
PLANCK_HBAR_SI: Final[float] = PLANCK_JOULE_SECOND / (2 * math.pi)
AMU_TO_KG: Final[float] = ATOMIC_MASS_KG
ANGSTROM_TO_M: Final[float] = 1.0e-10  # m / Angstrom
JOULE_TO_CM1: Final[float] = 1.0 / (
    PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_METRE_SECOND * 100.0
)
SPEED_OF_LIGHT_CM_S: Final[float] = SPEED_OF_LIGHT_METRE_SECOND * 100.0

# Standard light rotor groups subject to potential tunneling splitting
LIGHT_ROTOR_PATTERNS: Final[tuple[str, ...]] = (
    "CH3",
    "-CH3",
    "OH",
    "-OH",
    "NH2",
    "-NH2",
    "SH",
    "-SH",
    "CH2D",
    "CHD2",
    "CD3",
    "OD",
    "ND2",
)


# =============================================================================
# Pydantic Output Data Models
# =============================================================================


class StationaryPoint(BaseModel):
    """Pydantic model representing an analytically isolated 1D stationary point."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    angle_deg: float = Field(..., description="Torsional angle in degrees [0, 360)")
    angle_rad: float = Field(..., description="Torsional angle in radians [0, 2*pi)")
    energy_hartree: float = Field(..., description="Absolute energy in Hartree")
    energy_kcal_mol: float = Field(..., description="Absolute energy in kcal/mol")
    energy_cm1: float = Field(..., description="Absolute energy in cm^-1")
    curvature: float = Field(
        ..., description="Second derivative d2V/dtheta2 (curvature)"
    )
    type: str = Field(
        ..., description="Node classification: MINIMUM, MAXIMUM, or INFLECTION"
    )
    rel_energy_hartree: float = Field(
        default=0.0, description="Relative energy in Hartree"
    )
    rel_energy_kcal_mol: float = Field(
        default=0.0, description="Relative energy in kcal/mol"
    )
    rel_energy_cm1: float = Field(default=0.0, description="Relative energy in cm^-1")


class StationaryPoint2D(BaseModel):
    """Pydantic model representing an analytically isolated 2D stationary point."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    angles_deg: tuple[float, float] = Field(
        ..., description="Torsional angle pair in degrees"
    )
    angles_rad: tuple[float, float] = Field(
        ..., description="Torsional angle pair in radians"
    )
    energy_hartree: float = Field(..., description="Absolute energy in Hartree")
    energy_kcal_mol: float = Field(..., description="Absolute energy in kcal/mol")
    energy_cm1: float = Field(..., description="Absolute energy in cm^-1")
    hessian_eigenvalues: tuple[float, float] = Field(
        ..., description="Eigenvalues of 2D Hessian"
    )
    type: str = Field(
        ..., description="Classification: MINIMUM, SADDLE, MAXIMUM, or INFLECTION"
    )
    rel_energy_hartree: float = Field(
        default=0.0, description="Relative energy in Hartree"
    )
    rel_energy_kcal_mol: float = Field(
        default=0.0, description="Relative energy in kcal/mol"
    )
    rel_energy_cm1: float = Field(default=0.0, description="Relative energy in cm^-1")


class WKBTunnelingResult(BaseModel):
    """Pydantic model encapsulating semiclassical WKB tunneling predictions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    rotor_type: str = Field(
        ..., description="Chemical identifier of the rotating group"
    )
    is_light_rotor: bool = Field(
        ..., description="Whether group is classified as a light rotor"
    )
    barrier_height_cm1: float = Field(
        ..., description="Effective barrier height in cm^-1"
    )
    reduced_moment_inertia_amu_ang2: float = Field(
        ..., description="Reduced moment of inertia in amu * Angstrom^2"
    )
    tunneling_probability: float = Field(
        ..., description="Semiclassical WKB barrier transmission probability"
    )
    tunneling_splitting_mhz: float = Field(
        ..., description="Estimated ground-state tunneling splitting in MHz"
    )
    model: str
    quality_flags: list[str]
    log_tunneling_probability: float
    quantum_treatment_required: bool = Field(
        ...,
        description="Flag indicating if full quantum Hamiltonian is mandatory",
    )


# =============================================================================
# Helper Utilities: Dynamic Mass and Reduced Inertia
# =============================================================================


def get_dynamic_reduced_inertia(rotor_type: str) -> float:
    """Reject name-only inertia estimates; inertia requires geometry and masses.

    A group name supplies neither a rotation axis, its geometry nor the coupled
    frame inertia. Callers must compute and document their effective inertia
    for a declared kinetic-energy model and pass that value explicitly.
    """
    raise ValueError(
        f"Cannot derive a reduced inertia from rotor name {rotor_type!r}; "
        "provide geometry, isotope masses, rotor/frame partition "
        "and kinetic-energy model."
    )


# =============================================================================
# 1D Continuous Spline Fitting & Topological Node Isolation
# =============================================================================


def fit_continuous_splines(
    angles_deg: Sequence[float] | np.ndarray,
    energies_hartree: Sequence[float] | np.ndarray,
    periodic: bool = True,
    *,
    periodic_endpoint_tolerance_hartree: float | None = None,
) -> dict[str, Any]:
    """Fits continuous 1D periodic cubic splines across discrete torsional PES points.

    Analytically extracts all stationary points (minima, transition state maxima)
    via root-finding on the first derivative V'(theta) = 0 and classifies curvature
    via the second derivative V''(theta).

    :param angles_deg: Sequence of torsional angles in degrees.
    :param energies_hartree: Sequence of electronic energies in Hartree.
    :param periodic: Whether to enforce periodic boundary conditions across [0, 360).
    :return: Dictionary containing the fitted spline, stationary points, minima,
             maxima, global minimum, and barrier heights.
    :raises ValueError: If fewer than 4 points are supplied or lengths mismatch.
    """
    if np.iscomplexobj(angles_deg) or np.iscomplexobj(energies_hartree):
        raise ValueError("Spline angles and energies must be real, not complex")
    raw_angles = np.asarray(angles_deg, dtype=np.float64)
    raw_energies = np.asarray(energies_hartree, dtype=np.float64)
    if (
        raw_angles.ndim != 1
        or raw_energies.ndim != 1
        or not np.isfinite(raw_angles).all()
        or not np.isfinite(raw_energies).all()
    ):
        raise ValueError("Spline angles and energies must be finite vectors")

    if len(raw_angles) != len(raw_energies):
        raise ValueError(
            f"Angles count ({len(raw_angles)}) must equal "
            f"energies count ({len(raw_energies)})"
        )

    if len(raw_angles) < 4:
        raise ValueError(
            f"At least 4 points required for cubic spline fitting, "
            f"got {len(raw_angles)}"
        )

    # Sort angles into ascending order
    order = np.argsort(raw_angles)
    sorted_deg = raw_angles[order]
    sorted_e = raw_energies[order]

    # Convert to radians
    angles_rad = np.radians(sorted_deg)
    endpoint_closure = None

    if periodic:
        # Wrap endpoints for smooth periodic spline: append 2*pi point if needed
        span_covers_circle = (
            abs(sorted_deg[-1] - 360.0) > 1e-3 and abs(sorted_deg[0] - 0.0) < 1e-3
        )
        if span_covers_circle:
            angles_rad = np.append(angles_rad, 2.0 * math.pi)
            sorted_e = np.append(sorted_e, sorted_e[0])
            sorted_deg = np.append(sorted_deg, 360.0)
            endpoint_closure = {
                "source": "explicit_periodic_boundary_condition",
                "sampled_endpoint": False,
            }
        elif abs(sorted_deg[-1] - sorted_deg[0] - 360.0) < 1e-3:
            difference = float(sorted_e[-1] - sorted_e[0])
            if difference != 0.0:
                if (
                    periodic_endpoint_tolerance_hartree is None
                    or not math.isfinite(periodic_endpoint_tolerance_hartree)
                    or periodic_endpoint_tolerance_hartree < 0
                    or abs(difference) > periodic_endpoint_tolerance_hartree
                ):
                    raise ValueError(
                        "Unequal periodic endpoint energies require an explicit "
                        "finite closure tolerance covering the observed difference"
                    )
            endpoint_closure = {
                "source": "explicit_periodic_boundary_condition",
                "sampled_endpoint": True,
                "input_difference_hartree": difference,
                "allowed_tolerance_hartree": periodic_endpoint_tolerance_hartree,
            }
            sorted_e[-1] = sorted_e[0]

        spline = CubicSpline(angles_rad, sorted_e, bc_type="periodic")
    else:
        spline = CubicSpline(angles_rad, sorted_e, bc_type="natural")

    # Analytical first and second derivatives
    d_spline = spline.derivative(nu=1)
    d2_spline = spline.derivative(nu=2)

    # Dense sampling to locate sign changes of the first derivative
    dense_rad = np.linspace(float(angles_rad[0]), float(angles_rad[-1]), 1000)
    d_vals = d_spline(dense_rad)

    critical_rads: list[float] = []
    for i in range(len(dense_rad) - 1):
        if d_vals[i] * d_vals[i + 1] <= 0.0:
            try:
                root = float(brentq(d_spline, dense_rad[i], dense_rad[i + 1]))
                # Check uniqueness (within 1e-3 rad)
                if not any(abs(root - cr) < 1e-3 for cr in critical_rads):
                    critical_rads.append(root)
            except (ValueError, RuntimeError) as exc:
                logger.debug("Root isolation bracket skipped: %s", exc)

    critical_rads.sort()
    stationary_points: list[dict[str, Any]] = []

    for rad in critical_rads:
        deg = float(math.degrees(rad) % 360.0)
        e_hartree = float(spline(rad))
        curvature = float(d2_spline(rad))

        if curvature > 1e-7:
            node_type = "MINIMUM"
        elif curvature < -1e-7:
            node_type = "MAXIMUM"
        else:
            node_type = "INFLECTION"

        stationary_points.append(
            {
                "angle_deg": round(deg, 3),
                "angle_rad": round(rad, 5),
                "energy_hartree": e_hartree,
                "energy_kcal_mol": e_hartree * HARTREE_TO_KCAL_MOL,
                "energy_cm1": e_hartree * HARTREE_TO_CM1,
                "curvature": curvature,
                "type": node_type,
            }
        )

    # Separate minima and maxima
    minima = [p for p in stationary_points if p["type"] == "MINIMUM"]
    maxima = [p for p in stationary_points if p["type"] == "MAXIMUM"]

    # Retain sampled evidence separately; it is not a characterized stationary minimum.
    global_min = (
        min(minima, key=lambda p: float(p["energy_hartree"])) if minima else None
    )
    min_idx = int(np.argmin(sorted_e))
    sampled_min = {
        "angle_deg": float(sorted_deg[min_idx]),
        "angle_rad": float(angles_rad[min_idx]),
        "energy_hartree": float(sorted_e[min_idx]),
        "type": "SAMPLED_POINT",
    }

    # Relative energies relative to global minimum
    e_ref = float((global_min or sampled_min)["energy_hartree"])
    for p in stationary_points:
        p["rel_energy_hartree"] = p["energy_hartree"] - e_ref
        p["rel_energy_kcal_mol"] = p["rel_energy_hartree"] * HARTREE_TO_KCAL_MOL
        p["rel_energy_cm1"] = p["rel_energy_hartree"] * HARTREE_TO_CM1

    if maxima and global_min is not None:
        max_barrier_kcal = float(max(p["rel_energy_kcal_mol"] for p in maxima))
        max_barrier_cm1 = float(max(p["rel_energy_cm1"] for p in maxima))
    else:
        max_barrier_kcal = None
        max_barrier_cm1 = None
    sampled_span = float(np.max(sorted_e) - np.min(sorted_e))

    logger.info(
        "1D Spline fitted: %d stationary points (%d minima, %d maxima, "
        "characterized stationary energy span = %s kcal/mol)",
        len(stationary_points),
        len(minima),
        len(maxima),
        max_barrier_kcal,
    )

    return {
        "spline": spline,
        "stationary_points": stationary_points,
        "global_minimum": global_min,
        "sampled_minimum": sampled_min,
        "sampled_energy_span_hartree": sampled_span,
        "relative_energy_reference": "characterized_minimum"
        if global_min
        else "sampled_minimum",
        "quality_flags": [
            "interpolated_surface",
            "stationary_search_completeness_unverified",
            "transition_connectivity_unverified",
        ],
        "minima": minima,
        "maxima": maxima,
        "max_barrier_kcal_mol": max_barrier_kcal,
        "max_barrier_cm1": max_barrier_cm1,
        "angles_deg": sorted_deg,
        "energies_hartree": sorted_e,
        "input_angles_deg": raw_angles.copy(),
        "input_energies_hartree": raw_energies.copy(),
        "endpoint_closure": endpoint_closure,
    }


# =============================================================================
# 2D Bivariate Continuous Spline Fitting & Coupled Rotor Analysis
# =============================================================================


def fit_continuous_2d_splines(
    angles_deg_1: Sequence[float] | np.ndarray,
    angles_deg_2: Sequence[float] | np.ndarray,
    energies_hartree: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Fits continuous 2D bivariate splines across 2D torsional potential grids.

    Analytically extracts 2D critical points, computes the 2D Hessian matrix,
    classifies stationary points (minima, saddle points TS1, maxima), and
    computes inter-basin barrier thresholds.

    :param angles_deg_1: 1D sequence of angles along coordinate 1 (in degrees).
    :param angles_deg_2: 1D sequence of angles along coordinate 2 (in degrees).
    :param energies_hartree: 2D array of electronic energies (shape: (N1, N2)).
    :return: Dictionary containing the 2D spline, classified stationary points,
             minima, saddles, maxima, and global minimum.
    :raises ValueError: If dimensions are mismatched or fewer than 4x4 points.
    """
    if any(
        np.iscomplexobj(values)
        for values in (angles_deg_1, angles_deg_2, energies_hartree)
    ):
        raise ValueError("Spline angles and energies must be real, not complex")
    deg1 = np.asarray(angles_deg_1, dtype=np.float64)
    deg2 = np.asarray(angles_deg_2, dtype=np.float64)
    energies = np.asarray(energies_hartree, dtype=np.float64)
    if (
        deg1.ndim != 1
        or deg2.ndim != 1
        or not all(np.isfinite(values).all() for values in (deg1, deg2, energies))
    ):
        raise ValueError("Spline angles and energies must be finite arrays")

    if energies.shape != (len(deg1), len(deg2)):
        raise ValueError(
            f"Energies shape {energies.shape} does not match grid "
            f"({len(deg1)}, {len(deg2)})"
        )

    if len(deg1) < 4 or len(deg2) < 4:
        raise ValueError(
            "At least 4 points along each axis required for 2D spline fitting."
        )

    # Ensure coordinates are strictly increasing
    order1 = np.argsort(deg1)
    deg1 = deg1[order1]
    order2 = np.argsort(deg2)
    deg2 = deg2[order2]
    energies = energies[order1][:, order2]

    # Convert to radians
    rad1 = np.radians(deg1)
    rad2 = np.radians(deg2)

    # Fit 2D RectBivariateSpline (kx=3, ky=3 for bicubic)
    spline_2d = RectBivariateSpline(rad1, rad2, energies, kx=3, ky=3)

    # Dense search for 2D stationary points where gradient is zero
    n_sample = 20
    test_r1 = np.linspace(rad1[0], rad1[-1], n_sample)
    test_r2 = np.linspace(rad2[0], rad2[-1], n_sample)

    found_roots: list[tuple[float, float]] = []

    energy_scale = float(np.ptp(energies))

    def gradient(coords):
        r1, r2 = coords
        return (
            np.array(
                [
                    float(spline_2d(r1, r2, dx=1, dy=0)[0, 0]),
                    float(spline_2d(r1, r2, dx=0, dy=1)[0, 0]),
                ]
            )
            / energy_scale
        )

    def gradient_jacobian(coords):
        r1, r2 = coords
        h11 = float(spline_2d(r1, r2, dx=2, dy=0)[0, 0])
        h12 = float(spline_2d(r1, r2, dx=1, dy=1)[0, 0])
        h22 = float(spline_2d(r1, r2, dx=0, dy=2)[0, 0])
        return np.array([[h11, h12], [h12, h22]]) / energy_scale

    # A flat sampled surface has no isolated stationary points to characterize.
    if energy_scale > 0:
        for r1_init in test_r1:
            for r2_init in test_r2:
                solved = root(gradient, [r1_init, r2_init], jac=gradient_jacobian)
                if not solved.success or not np.isfinite(solved.x).all():
                    continue
                r1_opt, r2_opt = map(float, solved.x)
                if not (
                    rad1[0] <= r1_opt <= rad1[-1] and rad2[0] <= r2_opt <= rad2[-1]
                ):
                    continue
                if np.linalg.norm(gradient(solved.x)) > 1e-8:
                    continue
                if not any(
                    math.hypot(r1_opt - old1, r2_opt - old2) < 0.05
                    for old1, old2 in found_roots
                ):
                    found_roots.append((r1_opt, r2_opt))

    stationary_points_2d: list[dict[str, Any]] = []

    for r1_pt, r2_pt in found_roots:
        e_val = float(spline_2d(r1_pt, r2_pt)[0, 0])
        h11 = float(spline_2d(r1_pt, r2_pt, dx=2, dy=0)[0, 0])
        h12 = float(spline_2d(r1_pt, r2_pt, dx=1, dy=1)[0, 0])
        h22 = float(spline_2d(r1_pt, r2_pt, dx=0, dy=2)[0, 0])

        # Hessian matrix eigenvalues
        hessian = np.array([[h11, h12], [h12, h22]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(hessian)
        l1, l2 = float(eigvals[0]), float(eigvals[1])

        if l1 > 1e-7 and l2 > 1e-7:
            point_type = "MINIMUM"
        elif l1 < -1e-7 and l2 < -1e-7:
            point_type = "MAXIMUM"
        elif (l1 * l2) < -1e-7:
            point_type = "SADDLE"
        else:
            point_type = "INFLECTION"

        deg_pair = (
            round(float(math.degrees(r1_pt) % 360.0), 3),
            round(float(math.degrees(r2_pt) % 360.0), 3),
        )

        stationary_points_2d.append(
            {
                "angles_deg": deg_pair,
                "angles_rad": (round(r1_pt, 5), round(r2_pt, 5)),
                "energy_hartree": e_val,
                "energy_kcal_mol": e_val * HARTREE_TO_KCAL_MOL,
                "energy_cm1": e_val * HARTREE_TO_CM1,
                "hessian_eigenvalues": (l1, l2),
                "type": point_type,
            }
        )

    minima_2d = [p for p in stationary_points_2d if p["type"] == "MINIMUM"]
    saddles_2d = [p for p in stationary_points_2d if p["type"] == "SADDLE"]
    maxima_2d = [p for p in stationary_points_2d if p["type"] == "MAXIMUM"]

    global_min_2d = (
        min(minima_2d, key=lambda p: float(p["energy_hartree"])) if minima_2d else None
    )
    min_idx = np.unravel_index(np.argmin(energies), energies.shape)
    sampled_min_2d = {
        "angles_deg": (float(deg1[min_idx[0]]), float(deg2[min_idx[1]])),
        "angles_rad": (float(rad1[min_idx[0]]), float(rad2[min_idx[1]])),
        "energy_hartree": float(energies[min_idx]),
        "type": "SAMPLED_POINT",
    }
    e_ref_2d = float((global_min_2d or sampled_min_2d)["energy_hartree"])
    for p in stationary_points_2d:
        p["rel_energy_hartree"] = p["energy_hartree"] - e_ref_2d
        p["rel_energy_kcal_mol"] = p["rel_energy_hartree"] * HARTREE_TO_KCAL_MOL
        p["rel_energy_cm1"] = p["rel_energy_hartree"] * HARTREE_TO_CM1

    if saddles_2d and global_min_2d is not None:
        barrier_2d_kcal = float(max(p["rel_energy_kcal_mol"] for p in saddles_2d))
        barrier_2d_cm1 = float(max(p["rel_energy_cm1"] for p in saddles_2d))
    elif maxima_2d and global_min_2d is not None:
        barrier_2d_kcal = float(max(p["rel_energy_kcal_mol"] for p in maxima_2d))
        barrier_2d_cm1 = float(max(p["rel_energy_cm1"] for p in maxima_2d))
    else:
        barrier_2d_kcal = None
        barrier_2d_cm1 = None

    logger.info(
        "2D Spline fitted: %d stationary points (%d minima, %d saddles, "
        "%d maxima, characterized stationary energy span = %s kcal/mol)",
        len(stationary_points_2d),
        len(minima_2d),
        len(saddles_2d),
        len(maxima_2d),
        barrier_2d_kcal,
    )

    return {
        "spline": spline_2d,
        "stationary_points": stationary_points_2d,
        "global_minimum": global_min_2d,
        "sampled_minimum": sampled_min_2d,
        "sampled_energy_span_hartree": float(np.ptp(energies)),
        "relative_energy_reference": "characterized_minimum"
        if global_min_2d
        else "sampled_minimum",
        "quality_flags": [
            "interpolated_surface",
            "stationary_search_completeness_unverified",
            "transition_connectivity_unverified",
        ],
        "minima": minima_2d,
        "saddles": saddles_2d,
        "maxima": maxima_2d,
        "max_barrier_kcal_mol": barrier_2d_kcal,
        "max_barrier_cm1": barrier_2d_cm1,
    }


# =============================================================================
# Wentzel-Kramers-Brillouin (WKB) Tunneling Estimator
# =============================================================================


def wkb_tunneling_estimator(
    rotor_type: str,
    barrier_height_cm1: float,
    reduced_moment_inertia_amu_ang2: float | None = None,
    periodicity: int | None = None,
    *,
    potential_model: str | None = None,
) -> dict[str, Any]:
    """Applies semiclassical Wentzel-Kramers-Brillouin (WKB) estimation.

    Evaluates the barrier transmission probability and ground-state torsional
    tunneling splitting for light rotors (-CH3, -OH, -NH2, -SH).

    :param rotor_type: Chemical identifier of the rotor group (e.g. '-CH3', 'OH').
    :param barrier_height_cm1: Torsional barrier height in cm^-1.
    :param reduced_moment_inertia_amu_ang2: Reduced inertia (amu * Angstrom^2).
    :param periodicity: Rotational barrier periodicity (e.g. 3 for C3v, 2 for C2v).
    :return: Dictionary containing tunneling probability, splitting, and quantum flag.
    """
    clean_rotor = rotor_type.strip().upper()
    is_light_rotor = any(group in clean_rotor for group in LIGHT_ROTOR_PATTERNS)

    if potential_model != "cosine":
        raise ValueError(
            "WKB requires the explicitly selected cosine-potential approximation; "
            "no potential shape is inferred."
        )
    if (
        reduced_moment_inertia_amu_ang2 is None
        or not math.isfinite(reduced_moment_inertia_amu_ang2)
        or reduced_moment_inertia_amu_ang2 <= 0
    ):
        raise ValueError(
            "A finite positive, independently justified reduced inertia is required."
        )
    if type(periodicity) is not int or periodicity < 1:
        raise ValueError(
            "A positive integer potential periodicity must be specified explicitly."
        )
    if not math.isfinite(barrier_height_cm1) or barrier_height_cm1 <= 0:
        raise ValueError(
            "Below-barrier WKB requires a finite positive barrier; "
            "free rotation needs another model."
        )
    eff_inertia = float(reduced_moment_inertia_amu_ang2)

    # Moment of inertia in SI units: kg * m^2
    i_red_si = eff_inertia * AMU_TO_KG * (ANGSTROM_TO_M**2)

    # Barrier height V0 in Joules
    v0_joules = float(barrier_height_cm1) / JOULE_TO_CM1

    # Declared cosine model; E0 is a harmonic local-well approximation.
    omega_0 = float(periodicity) * math.sqrt(v0_joules / (2.0 * i_red_si))
    e0_joules = 0.5 * PLANCK_HBAR_SI * omega_0
    if e0_joules >= v0_joules:
        raise ValueError(
            "Estimated ground-state energy is above the barrier; "
            "below-barrier WKB is inapplicable."
        )
    turning_1 = math.acos(1.0 - 2.0 * e0_joules / v0_joules) / periodicity
    turning_2 = 2.0 * math.pi / periodicity - turning_1
    action = evaluate_wkb_action_integral(
        lambda theta: 0.5 * v0_joules * (1.0 - math.cos(periodicity * theta)),
        e0_joules,
        turning_1,
        turning_2,
        i_red_si,
    )
    tunneling_probability = math.exp(-2.0 * action)
    # Isolated symmetric two-well estimate DeltaE=(hbar*omega/pi)*exp(-S).
    # It is not a periodic-rotor A/E splitting or a spin-statistical assignment.
    splitting_energy = PLANCK_HBAR_SI * omega_0 / math.pi * math.exp(-action)
    tunneling_splitting_mhz = splitting_energy / (2 * math.pi * PLANCK_HBAR_SI) / 1e6
    quantum_required = (
        True  # A qualified nuclear-motion solver must validate this estimate.
    )

    logger.info(
        "WKB tunneling estimate for %s: barrier=%.1f cm^-1, P_tunnel=%.2e, "
        "Splitting=%.4f MHz, QuantumRequired=%s",
        rotor_type,
        barrier_height_cm1,
        tunneling_probability,
        tunneling_splitting_mhz,
        quantum_required,
    )

    return {
        "rotor_type": rotor_type,
        "is_light_rotor": is_light_rotor,
        "barrier_height_cm1": barrier_height_cm1,
        "reduced_moment_inertia_amu_ang2": eff_inertia,
        "tunneling_probability": tunneling_probability,
        "tunneling_splitting_mhz": tunneling_splitting_mhz,
        "quantum_treatment_required": quantum_required,
        "model": "cosine_potential_symmetric_two_well_wkb_estimate",
        "quality_flags": [
            "harmonic_local_ground_energy",
            "not_periodic_rotor_A_E_splitting",
            "uncalibrated",
        ],
        "log_tunneling_probability": -2.0 * action,
    }


def evaluate_wkb_action_integral(
    potential_func: Callable[[float], float],
    energy_joules: float,
    theta_turning_1_rad: float,
    theta_turning_2_rad: float,
    reduced_moment_inertia_kg_m2: float,
) -> float:
    """Evaluates the general 1D numerical WKB phase/action integral.

    Calculates:
        S = (1 / hbar) * int_{theta1}^{theta2} sqrt(2*I_red*max(0, V(th)-E)) dth

    :param potential_func: Continuous potential function V(theta) in Joules.
    :param energy_joules: Total energy level E in Joules.
    :param theta_turning_1_rad: Inner classical turning point in radians.
    :param theta_turning_2_rad: Outer classical turning point in radians.
    :param reduced_moment_inertia_kg_m2: Moment of inertia in kg * m^2.
    :return: Dimensionless WKB action integral S.
    """

    if not all(
        math.isfinite(value)
        for value in (
            energy_joules,
            theta_turning_1_rad,
            theta_turning_2_rad,
            reduced_moment_inertia_kg_m2,
        )
    ):
        raise ValueError("WKB action inputs must be finite.")
    if reduced_moment_inertia_kg_m2 <= 0 or theta_turning_2_rad <= theta_turning_1_rad:
        raise ValueError(
            "WKB action requires positive inertia and an ordered forbidden interval."
        )

    def integrand(theta: float) -> float:
        v_val = potential_func(theta)
        if not math.isfinite(v_val):
            raise ValueError("The potential returned a nonfinite energy.")
        delta = v_val - energy_joules
        if delta < 0:
            raise ValueError(
                "The supplied interval includes a classically allowed region."
            )
        return math.sqrt(2.0 * reduced_moment_inertia_kg_m2 * delta)

    integral_val, _ = quad(
        integrand, theta_turning_1_rad, theta_turning_2_rad, limit=100
    )
    action = float(integral_val / PLANCK_HBAR_SI)
    return action


__all__ = [
    "HARTREE_TO_KCAL_MOL",
    "HARTREE_TO_CM1",
    "KCAL_MOL_TO_CM1",
    "PLANCK_HBAR_SI",
    "AMU_TO_KG",
    "ANGSTROM_TO_M",
    "JOULE_TO_CM1",
    "SPEED_OF_LIGHT_CM_S",
    "StationaryPoint",
    "StationaryPoint2D",
    "WKBTunnelingResult",
    "fit_continuous_splines",
    "fit_continuous_2d_splines",
    "wkb_tunneling_estimator",
    "evaluate_wkb_action_integral",
    "get_dynamic_reduced_inertia",
]
