"""Legacy frozen-monomer geometry arithmetic and explicit constraint formatting.

These helpers analyze supplied geometry/gradients and format Cartesian ORCA
constraints. They do not execute optimizations, qualify electronic methods,
establish spectroscopic accuracy, or generate a CFOUR geometry without a reviewed
ZMAT contract. Missing reference values and undefined observables remain absent.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
from mendeleev import element as mendeleev_element
from pydantic import BaseModel, ConfigDict, Field

from cochem_torq.units import (
    ATOMIC_MASS_KG,
    AVOGADRO_PER_MOL,
    BOHR_ANGSTROM,
    BOHR_METRE,
    CONSTANTS_PROFILE,
    ELEMENTARY_CHARGE_COULOMB,
    HARTREE_JOULE,
    PI,
    PLANCK_JOULE_SECOND,
    SPEED_OF_LIGHT_METRE_SECOND,
)

logger = logging.getLogger("TorqConstraints")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: [CoChem-TORQ-Constraints] %(message)s",
    )


# =============================================================================
# 1. Fundamental Physical Constants & Conversion Factors (CODATA / IUPAC)
# =============================================================================

BOHR_TO_ANGSTROM: Final[float] = BOHR_ANGSTROM
ANGSTROM_TO_BOHR: Final[float] = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV: Final[float] = HARTREE_JOULE / ELEMENTARY_CHARGE_COULOMB
HARTREE_TO_KCAL_PER_MOL: Final[float] = HARTREE_JOULE * AVOGADRO_PER_MOL / 4184
HARTREE_TO_KJ_PER_MOL: Final[float] = HARTREE_JOULE * AVOGADRO_PER_MOL / 1000

# 1 Eh/a0 = E_h / a_0 N. The previous 8.2387225e-13 value was wrong by 1e5.
HARTREE_PER_BOHR_TO_NEWTON: Final[float] = HARTREE_JOULE / BOHR_METRE
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: Final[float] = HARTREE_TO_EV / BOHR_TO_ANGSTROM

# Rotational constant conversion factor: h / (8 * pi^2) in MHz * amu * Å^2
AMU_ANGSTROM2_TO_MHZ: Final[float] = PLANCK_JOULE_SECOND / (
    8 * PI**2 * ATOMIC_MASS_KG * 1e-20 * 1e6
)
MHZ_TO_WAVENUMBERS_CM1: Final[float] = 1e6 / (SPEED_OF_LIGHT_METRE_SECOND * 100)

# Strict Method Matrix v4 Convergence Thresholds [M]
TOL_MAX_G_DEFAULT_EH_BOHR: Final[float] = 1.0e-5  # ~8.238722e-13 N
TOL_RMS_G_DEFAULT_EH_BOHR: Final[float] = 3.0e-6  # 3e-6 Eh/a0
TOL_E_DEFAULT_EH: Final[float] = 1.0e-7  # 1e-7 Eh
TOL_RMS_D_DEFAULT_BOHR: Final[float] = 5.0e-5  # 5e-5 a0
TOL_MAX_D_DEFAULT_BOHR: Final[float] = 1.0e-4  # 1e-4 a0
TOL_RESIDUAL_G_FROZEN_EH_BOHR: Final[float] = (
    1.0e-5  # Threshold for deformation escalation
)


# =============================================================================
# 2. Exception Hierarchy
# =============================================================================


class MethodMatrixConstraintError(Exception):
    """Base exception for all CoChem-TORQ constraint and convergence violations."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConstraintConvergenceError(MethodMatrixConstraintError):
    """Raised when gradient or displacement thresholds (TolMaxG 1e-5) are violated."""

    pass


class SpatialCollisionError(MethodMatrixConstraintError):
    """Raised when atoms in a complex approach unphysically close (nuclear overlap)."""

    pass


class FragmentDissociationError(MethodMatrixConstraintError):
    """Raised when monomer fragments drift beyond non-covalent interaction range."""

    pass


class DeformationEscalationError(MethodMatrixConstraintError):
    """Raised when residual gradients on frozen monomer coordinates exceed TolMaxG."""

    pass


# =============================================================================
# 3. Mendeleev Dynamic Integration (Mendeleev Mandate)
# =============================================================================


def get_dynamic_atomic_mass(symbol: str) -> float:
    """
    Retrieve standard atomic mass (in amu / Da) dynamically via the mendeleev library.
    Strictly forbids hardcoded mass lookups under the Mendeleev Mandate.
    """
    clean = str(symbol).strip().capitalize()
    el = mendeleev_element(clean)
    if el.mass is None:
        raise ValueError(f"No atomic mass found in mendeleev for element '{symbol}'.")
    return float(el.mass)


def get_dynamic_isotopic_mass(symbol: str, mass_number: int | None = None) -> float:
    """Read the selected tabulated isotope; never substitute an atomic weight."""
    from Libraries.cochem_isotopes import isotope_mass

    clean = str(symbol).strip().capitalize()
    if mass_number is not None and (
        isinstance(mass_number, bool)
        or not isinstance(mass_number, (int, np.integer))
        or mass_number <= 0
    ):
        raise ValueError("An explicit isotope requires a positive integer mass number.")
    return isotope_mass(f"{mass_number}{clean}" if mass_number is not None else clean)


def get_dynamic_pyykko_radius(symbol: str) -> float:
    """
    Retrieve Pyykkö single-bond covalent radius in Angstroms dynamically from mendeleev.
    (mendeleev provides covalent_radius_pyykko in picometers, converted to Å via / 100.0).
    """
    clean = str(symbol).strip().capitalize()
    el = mendeleev_element(clean)
    if hasattr(el, "covalent_radius_pyykko") and el.covalent_radius_pyykko is not None:
        return float(el.covalent_radius_pyykko) / 100.0
    raise ValueError(f"No tabulated Pyykkö radius is available for '{symbol}'.")


def get_dynamic_vdw_radius(symbol: str) -> float:
    """
    Retrieve van der Waals radius in Angstroms dynamically from mendeleev.
    (mendeleev provides vdw_radius in picometers, converted to Å via / 100.0).
    """
    clean = str(symbol).strip().capitalize()
    el = mendeleev_element(clean)
    if hasattr(el, "vdw_radius") and el.vdw_radius is not None:
        return float(el.vdw_radius) / 100.0
    raise ValueError(f"No tabulated van der Waals radius is available for '{symbol}'.")


def get_dynamic_atomic_number(symbol: str) -> int:
    """Retrieve atomic number (Z) dynamically from mendeleev."""
    clean = str(symbol).strip().capitalize()
    el = mendeleev_element(clean)
    return int(el.atomic_number)


# =============================================================================
# 4. Enums & Pydantic Data Structures
# =============================================================================


class MonomerConstraintMode(str, enum.Enum):
    """
    Method Matrix v4 Section 9A.2 Monomer Flag Taxonomy:
    - RELAXED: All coordinates optimized at the current tier level.
    - FROZEN_ISO: Monomers frozen at isolated-monomer geometries (r_e^SE or high-level ab initio).
                 Deformation is omitted and must be assessed for the given system.
    - FROZEN_INC: Monomers frozen at in-complex geometries from higher level (retains deformation; S66).
    """

    RELAXED = "relaxed"
    FROZEN_ISO = "frozen-iso"
    FROZEN_INC = "frozen-inc"


class ConstraintCoordinateType(str, enum.Enum):
    """Types of geometric constraints applied during optimization."""

    CARTESIAN = "cartesian"
    INTERNAL_BONDS = "internal_bonds"
    INTERNAL_ALL = "internal_all"
    RIGID_BODY = "rigid_body"


class MonomerFragment(BaseModel):
    """Specification of an isolated or in-complex monomer fragment within a van der Waals system."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fragment_id: int
    atom_indices: list[int]
    symbols: list[str]
    coordinates: list[list[float]]
    center_of_mass: list[float]
    total_mass_amu: float
    molecular_formula: str
    constraint_mode: MonomerConstraintMode = MonomerConstraintMode.FROZEN_ISO
    is_hydrogen_bonded: bool | None = None
    has_potential_hydrogen_bond_donor: bool = False
    deformation_warning: bool = False


class IntermolecularMetrics(BaseModel):
    """Quantitative physical metrics governing the intermolecular degrees of freedom."""

    center_of_mass_distance_angstrom: float | None
    min_interatomic_distance_angstrom: float | None
    max_interatomic_distance_angstrom: float | None
    contact_atom_pairs: list[tuple[int, int, float]]
    is_collision: bool = False
    is_dissociated: bool = False


class RotationalConstants(BaseModel):
    """
    Principal moments of inertia and rotational constants (A, B, C) in MHz and cm^-1,
    with Ray's asymmetry parameter and inertial defect.
    """

    I_a_amu_angstrom2: float
    I_b_amu_angstrom2: float
    I_c_amu_angstrom2: float
    A_MHz: float | None
    B_MHz: float | None
    C_MHz: float | None
    A_cm1: float | None
    B_cm1: float | None
    C_cm1: float | None
    kappa: float | None
    inertial_defect_amu_angstrom2: float
    planar_moments_amu_angstrom2: dict[str, float]
    rotor_type: str
    mass_policy: str = "standard_atomic_weights; not isotopologue-specific"
    constants_profile: str = CONSTANTS_PROFILE


class RotationalErrorBudget(BaseModel):
    """
    Declared first-order effective-distance sensitivities, not molecular accuracy.
    """

    R_intermolecular_angstrom: float
    delta_R_angstrom: float
    delta_B_over_B_percent: float
    delta_B_MHz: float | None
    delta_A_over_A_percent: float | None
    delta_A_MHz: float | None
    uniform_monomer_error_angstrom: float
    equivalent_monomer_error_for_B_angstrom: float | None
    reference_bond_length_angstrom: float | None = None
    monomer_B_sensitivity_percent_per_angstrom: float | None = None
    evidence_class: str = "declared_parameter_sensitivity"
    accuracy_qualified: bool = False
    headline_verdict: str


class ConvergenceAudit(BaseModel):
    """
    Detailed audit of optimization gradient components against strict TolMaxG 1e-5 Eh/a0.
    Monitors residual gradients on frozen coordinates to detect active deformation channels.
    """

    tol_max_g_eh_bohr: float
    tol_rms_g_eh_bohr: float
    max_g_observed_eh_bohr: float
    rms_g_observed_eh_bohr: float
    converged: bool
    frozen_atom_indices: list[int]
    residual_max_g_frozen_eh_bohr: float
    residual_rms_g_frozen_eh_bohr: float
    deformation_channel_active: bool
    escalation_required: bool
    audit_notes: list[str] = Field(default_factory=list)


class ConstraintEnforcementResult(BaseModel):
    """Master output result for Phase 4 (Stage 3.0-3.5) constraint enforcement."""

    success: bool
    constraint_mode: MonomerConstraintMode
    num_atoms: int
    num_monomers: int
    monomer_fragments: list[MonomerFragment]
    intermolecular_metrics: IntermolecularMetrics
    rotational_constants: RotationalConstants
    error_budget: RotationalErrorBudget
    convergence_audit: ConvergenceAudit | None = None
    convergence_status: Literal["unassessed", "converged", "unconverged"]
    orca_geom_constraints_block: str
    cfour_zmat_constraints_block: str | None
    cfour_zmat_status: Literal["unavailable"] = "unavailable"
    cfour_zmat_unavailable_reason: str = (
        "No reviewed molecule-specific ZMAT contract was supplied; "
        "no geometry variables or values were invented."
    )
    provenance_tags: dict[str, str] = Field(default_factory=dict)


# =============================================================================
# 5. Core Mathematical & Physical Algorithms
# =============================================================================


def partition_monomer_fragments(
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]] | np.ndarray,
    tolerance_multiplier: float = 1.20,
    constraint_mode: MonomerConstraintMode = MonomerConstraintMode.FROZEN_ISO,
    collision_threshold_angstrom: float = 1.20,
) -> tuple[list[MonomerFragment], IntermolecularMetrics]:
    """
    Partitions a molecular complex into distinct monomer fragments using dynamic
    Pyykkö covalent radii and connected component graph analysis.
    Evaluates intermolecular metrics, collision safety, and hydrogen bond donors.
    """
    raw_coordinates = np.asarray(coordinates)
    if not np.isrealobj(raw_coordinates):
        raise ValueError("Fragment geometry must contain real coordinates.")
    coords = np.asarray(raw_coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms == 0:
        raise ValueError("Cannot partition empty coordinate array.")
    if coords.shape != (n_atoms, 3) or not np.isfinite(coords).all():
        raise ValueError(
            f"Coordinates shape {coords.shape} does not match (N={n_atoms}, 3)."
        )

    # Dynamic Pyykkö radii & distance matrix
    radii = np.array([get_dynamic_pyykko_radius(s) for s in symbols], dtype=np.float64)
    vdw_radii = np.array([get_dynamic_vdw_radius(s) for s in symbols], dtype=np.float64)

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Covalent cutoff matrix
    covalent_cutoff = (
        radii[:, np.newaxis] + radii[np.newaxis, :]
    ) * tolerance_multiplier
    adj = (dist_matrix < covalent_cutoff) & (dist_matrix > 1e-4)

    # Connected components (BFS)
    visited = [False] * n_atoms
    component_indices: list[list[int]] = []

    for i in range(n_atoms):
        if not visited[i]:
            comp: list[int] = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                neighbors = np.where(adj[curr])[0]
                for nbr in neighbors:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(int(nbr))
            component_indices.append(sorted(comp))

    # Build MonomerFragment models
    monomer_fragments: list[MonomerFragment] = []
    for f_idx, idxs in enumerate(component_indices):
        frag_syms = [symbols[i] for i in idxs]
        frag_coords = coords[idxs]
        frag_masses = [get_dynamic_atomic_mass(s) for s in frag_syms]
        total_frag_mass = sum(frag_masses)

        # Center of mass
        com = (
            np.sum(frag_coords * np.array(frag_masses)[:, np.newaxis], axis=0)
            / total_frag_mass
        )

        # Chemical formula string
        counts: dict[str, int] = {}
        for s in frag_syms:
            counts[s] = counts.get(s, 0) + 1
        formula = "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items()))

        # Check for potential strong hydrogen bond donors (O-H, N-H, F-H)
        has_h_donor = False
        for i_local, sym in enumerate(frag_syms):
            if sym in ("O", "N", "F"):
                for j_local, sym2 in enumerate(frag_syms):
                    if sym2 == "H":
                        d_bond = np.linalg.norm(
                            frag_coords[i_local] - frag_coords[j_local]
                        )
                        if d_bond < 1.30:
                            has_h_donor = True
                            break

        monomer_fragments.append(
            MonomerFragment(
                fragment_id=f_idx,
                atom_indices=idxs,
                symbols=frag_syms,
                coordinates=frag_coords.tolist(),
                center_of_mass=com.tolist(),
                total_mass_amu=float(total_frag_mass),
                molecular_formula=formula,
                constraint_mode=constraint_mode,
                is_hydrogen_bonded=None,
                has_potential_hydrogen_bond_donor=has_h_donor,
                deformation_warning=has_h_donor
                and constraint_mode == MonomerConstraintMode.FROZEN_ISO,
            )
        )

    # Check overall non-bonded minimum nuclear distance across distinct atoms
    min_overall_dist = (
        float(np.min(dist_matrix + np.diag(np.full(n_atoms, 1e9))))
        if n_atoms > 1
        else float("inf")
    )

    # Intermolecular metrics across fragments
    if len(component_indices) >= 2:
        com0 = np.array(monomer_fragments[0].center_of_mass)
        com1 = np.array(monomer_fragments[1].center_of_mass)
        com_dist = float(np.linalg.norm(com0 - com1))

        # Interatomic contact pairs
        contacts: list[tuple[int, int, float]] = []
        min_d = float("inf")
        max_d = 0.0

        for idx1 in component_indices[0]:
            for idx2 in component_indices[1]:
                d = float(dist_matrix[idx1, idx2])
                min_d = min(min_d, d)
                max_d = max(max_d, d)
                cutoff_vdw = vdw_radii[idx1] + vdw_radii[idx2]
                if d <= cutoff_vdw:
                    contacts.append((idx1, idx2, d))

        is_collision = min_d < collision_threshold_angstrom or min_overall_dist < 0.60
        is_dissociated = com_dist > 12.0

        metrics = IntermolecularMetrics(
            center_of_mass_distance_angstrom=com_dist,
            min_interatomic_distance_angstrom=min_d if np.isfinite(min_d) else None,
            max_interatomic_distance_angstrom=max_d,
            contact_atom_pairs=contacts,
            is_collision=is_collision,
            is_dissociated=is_dissociated,
        )
    else:
        is_collision = min_overall_dist < 0.60
        metrics = IntermolecularMetrics(
            center_of_mass_distance_angstrom=None,
            min_interatomic_distance_angstrom=(
                min_overall_dist if np.isfinite(min_overall_dist) else None
            ),
            max_interatomic_distance_angstrom=None,
            contact_atom_pairs=[],
            is_collision=is_collision,
            is_dissociated=False,
        )

    return monomer_fragments, metrics


def compute_rotational_constants(
    symbols: Sequence[str], coordinates: Sequence[Sequence[float]] | np.ndarray
) -> RotationalConstants:
    """
    Computes principal moments of inertia (I_a <= I_b <= I_c) and rotational constants
    (A >= B >= C) using dynamic atomic masses from mendeleev.
    Evaluates Ray's asymmetry parameter kappa, inertial defect Delta, and planar moments.
    """
    raw_coordinates = np.asarray(coordinates)
    if not np.isrealobj(raw_coordinates):
        raise ValueError("Rotational geometry must contain real coordinates.")
    coords = np.asarray(raw_coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms == 0 or coords.shape != (n_atoms, 3) or not np.isfinite(coords).all():
        raise ValueError("Invalid coordinates for rotational constants calculation.")

    masses = np.array([get_dynamic_atomic_mass(s) for s in symbols], dtype=np.float64)
    total_mass = np.sum(masses)

    # Shift to Center of Mass
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    r = coords - com

    # Inertia tensor construction
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    I_xx = np.sum(masses * (y**2 + z**2))
    I_yy = np.sum(masses * (x**2 + z**2))
    I_zz = np.sum(masses * (x**2 + y**2))
    I_xy = -np.sum(masses * x * y)
    I_xz = -np.sum(masses * x * z)
    I_yz = -np.sum(masses * y * z)

    inertia_tensor = np.array(
        [[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]], dtype=np.float64
    )

    eigenvals = np.linalg.eigvalsh(inertia_tensor)
    # Roundoff may produce a tiny negative eigenvalue for a rank-deficient
    # inertia tensor. Retain its physical zero; never invent a positive moment.
    zero_tolerance = 64 * np.finfo(float).eps * max(float(np.max(eigenvals)), 1.0)
    if np.min(eigenvals) < -zero_tolerance:
        raise ValueError("Inertia tensor has a nonphysical negative principal moment.")
    eigenvals[np.abs(eigenvals) <= zero_tolerance] = 0.0
    I_a, I_b, I_c = float(eigenvals[0]), float(eigenvals[1]), float(eigenvals[2])

    # Rotational constants in MHz: A >= B >= C
    # Zero moments have undefined rotational constants: a linear rotor's A
    # and all constants of an atom must remain absent.
    A_MHz = AMU_ANGSTROM2_TO_MHZ / I_a if I_a > 0 else None
    B_MHz = AMU_ANGSTROM2_TO_MHZ / I_b if I_b > 0 else None
    C_MHz = AMU_ANGSTROM2_TO_MHZ / I_c if I_c > 0 else None

    A_cm1 = A_MHz * MHZ_TO_WAVENUMBERS_CM1 if A_MHz is not None else None
    B_cm1 = B_MHz * MHZ_TO_WAVENUMBERS_CM1 if B_MHz is not None else None
    C_cm1 = C_MHz * MHZ_TO_WAVENUMBERS_CM1 if C_MHz is not None else None

    # Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)
    denom = A_MHz - C_MHz if A_MHz is not None and C_MHz is not None else None
    if denom is not None and B_MHz is not None and abs(denom) > 1e-4:
        kappa = float((2.0 * B_MHz - A_MHz - C_MHz) / denom)
    else:
        kappa = None

    # Inertial defect Delta = I_c - I_a - I_b
    inertial_defect = float(I_c - I_a - I_b)

    # Planar moments: P_aa = (I_b + I_c - I_a) / 2, etc.
    p_aa = float((I_b + I_c - I_a) / 2.0)
    p_bb = float((I_a + I_c - I_b) / 2.0)
    p_cc = float((I_a + I_b - I_c) / 2.0)

    # Rotor classification
    if I_c == 0:
        rotor_type = "atom"
    elif I_a == 0:
        rotor_type = "linear"
    elif abs(I_a - I_b) < 1e-3 and abs(I_b - I_c) < 1e-3:
        rotor_type = "spherical"
    elif abs(I_a - I_b) < 1e-3:
        rotor_type = "oblate_symmetric"
    elif abs(I_b - I_c) < 1e-3:
        rotor_type = "prolate_symmetric"
    else:
        rotor_type = "asymmetric"

    return RotationalConstants(
        I_a_amu_angstrom2=I_a,
        I_b_amu_angstrom2=I_b,
        I_c_amu_angstrom2=I_c,
        A_MHz=A_MHz,
        B_MHz=B_MHz,
        C_MHz=C_MHz,
        A_cm1=A_cm1,
        B_cm1=B_cm1,
        C_cm1=C_cm1,
        kappa=kappa,
        inertial_defect_amu_angstrom2=inertial_defect,
        planar_moments_amu_angstrom2={"P_aa": p_aa, "P_bb": p_bb, "P_cc": p_cc},
        rotor_type=rotor_type,
    )


def propagate_rotational_error(
    R_angstrom: float,
    delta_R_angstrom: float = 0.002,
    monomer_bond_error_angstrom: float = 0.010,
    base_constants: RotationalConstants | None = None,
    *,
    reference_bond_length_angstrom: float | None = None,
    monomer_B_sensitivity_percent_per_angstrom: float | None = None,
) -> RotationalErrorBudget:
    """Evaluate declared perturbations under a rigid effective-distance model.

    The relation delta B/B = -2 delta R/R is a first-order sensitivity for an
    inertia contribution proportional to R squared. It is not a calibrated
    molecular error model. Absolute changes require actual base constants;
    monomer-length and monomer-B effects require explicit reference inputs.
    """
    for name, value in (
        ("intermolecular distance", R_angstrom),
        ("distance perturbation", delta_R_angstrom),
        ("monomer perturbation", monomer_bond_error_angstrom),
        ("reference bond length", reference_bond_length_angstrom),
        ("monomer-B sensitivity", monomer_B_sensitivity_percent_per_angstrom),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"Explicit {name} must be a finite real number.")
    if not np.isfinite(R_angstrom) or R_angstrom <= 0:
        raise ValueError(
            f"Intermolecular distance R must be positive, got {R_angstrom}."
        )
    if not np.isfinite(delta_R_angstrom) or not np.isfinite(
        monomer_bond_error_angstrom
    ):
        raise ValueError("Declared parameter perturbations must be finite.")
    for name, value in (
        ("reference bond length", reference_bond_length_angstrom),
        ("monomer-B sensitivity", monomer_B_sensitivity_percent_per_angstrom),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"Explicit {name} must be positive and finite.")

    # Delta B / B = -2 * Delta R / R
    delta_B_over_B = -2.0 * (delta_R_angstrom / R_angstrom) * 100.0  # percentage
    b_val = base_constants.B_MHz if base_constants is not None else None
    delta_B_MHz = b_val * (delta_B_over_B / 100.0) if b_val is not None else None

    delta_A_over_A = (
        -2.0 * (monomer_bond_error_angstrom / reference_bond_length_angstrom) * 100
        if reference_bond_length_angstrom is not None
        else None
    )
    a_val = base_constants.A_MHz if base_constants is not None else None
    delta_A_MHz = (
        a_val * (delta_A_over_A / 100)
        if a_val is not None and delta_A_over_A is not None
        else None
    )

    equiv_monomer_error = (
        abs(delta_B_over_B) / monomer_B_sensitivity_percent_per_angstrom
        if monomer_B_sensitivity_percent_per_angstrom is not None
        else None
    )

    verdict = (
        f"A declared ΔR of {delta_R_angstrom:.4f} Å gives a first-order "
        f"effective-distance sensitivity ΔB/B of {delta_B_over_B:.3f}%. "
        "This approximation establishes no accuracy target or calibrated uncertainty."
    )

    return RotationalErrorBudget(
        R_intermolecular_angstrom=R_angstrom,
        delta_R_angstrom=delta_R_angstrom,
        delta_B_over_B_percent=delta_B_over_B,
        delta_B_MHz=delta_B_MHz,
        delta_A_over_A_percent=delta_A_over_A,
        delta_A_MHz=delta_A_MHz,
        uniform_monomer_error_angstrom=monomer_bond_error_angstrom,
        equivalent_monomer_error_for_B_angstrom=equiv_monomer_error,
        reference_bond_length_angstrom=reference_bond_length_angstrom,
        monomer_B_sensitivity_percent_per_angstrom=(
            monomer_B_sensitivity_percent_per_angstrom
        ),
        headline_verdict=verdict,
    )


def audit_gradient_convergence(
    gradient_eh_bohr: Sequence[Sequence[float]] | np.ndarray,
    frozen_atom_indices: Sequence[int] | None = None,
    tol_max_g: float = TOL_MAX_G_DEFAULT_EH_BOHR,
    tol_rms_g: float = TOL_RMS_G_DEFAULT_EH_BOHR,
    tol_residual_frozen: float = TOL_RESIDUAL_G_FROZEN_EH_BOHR,
) -> ConvergenceAudit:
    """
    Audits Cartesian nuclear gradient against Method Matrix v4 strict convergence criteria:
    - TolMaxG <= 1.0e-5 Eh/a0 on free (unconstrained) coordinates.
    - TolRMSG <= 3.0e-6 Eh/a0 on free (unconstrained) coordinates.
    - Residual gradient on frozen coordinates monitored against 1.0e-5 Eh/a0.
    """
    raw_gradient = np.asarray(gradient_eh_bohr)
    if not np.isrealobj(raw_gradient):
        raise ValueError("A convergence audit requires a real Cartesian gradient.")
    g_arr = np.asarray(raw_gradient, dtype=np.float64)
    if g_arr.ndim == 1:
        if g_arr.size % 3 != 0:
            raise ValueError("1D gradient array size must be a multiple of 3.")
        g_arr = g_arr.reshape(-1, 3)

    if (
        g_arr.ndim != 2
        or g_arr.shape[1] != 3
        or not g_arr.size
        or not np.isfinite(g_arr).all()
    ):
        raise ValueError(
            "A convergence audit requires a finite nonempty (N, 3) gradient."
        )
    if any(
        not np.isfinite(value) or value <= 0
        for value in (tol_max_g, tol_rms_g, tol_residual_frozen)
    ):
        raise ValueError("Convergence tolerances must be positive and finite.")

    n_atoms = g_arr.shape[0]
    frozen_set = set(frozen_atom_indices or [])
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < n_atoms
        for index in frozen_set
    ):
        raise ValueError("Frozen gradient rows must be in-range integer atom indices.")
    free_indices = [i for i in range(n_atoms) if i not in frozen_set]

    notes: list[str] = []

    # Free coordinates audit
    if free_indices:
        g_free = g_arr[free_indices]
        max_g_observed = float(np.max(np.abs(g_free)))
        rms_g_observed = float(np.sqrt(np.mean(g_free**2)))
    else:
        # Fully frozen system
        max_g_observed = 0.0
        rms_g_observed = 0.0

    converged_max = max_g_observed <= tol_max_g
    converged_rms = rms_g_observed <= tol_rms_g
    converged = converged_max and converged_rms

    if not converged:
        notes.append(
            f"Optimization unconverged: max|g_free|={max_g_observed:.3e} (tol={tol_max_g:.1e}), "
            f"rms|g_free|={rms_g_observed:.3e} (tol={tol_rms_g:.1e}) Eh/a0."
        )

    # Frozen coordinates residual gradient audit
    if frozen_set:
        frozen_list = sorted(list(frozen_set))
        g_frozen = g_arr[frozen_list]
        res_max_g = float(np.max(np.abs(g_frozen)))
        res_rms_g = float(np.sqrt(np.mean(g_frozen**2)))
    else:
        res_max_g = 0.0
        res_rms_g = 0.0

    deformation_active = res_max_g > tol_residual_frozen
    escalation_required = False

    if deformation_active:
        escalation_required = True
        notes.append(
            f"Deformation Warning (§9A.1): Residual gradient on frozen coordinates "
            f"(max={res_max_g:.3e} Eh/a0) exceeds TolMaxG ({tol_residual_frozen:.1e} Eh/a0). "
            f"Constraint is doing real work; complexation deformation is non-negligible. "
            f"Escalate to relaxed or frozen-inc optimization."
        )

    return ConvergenceAudit(
        tol_max_g_eh_bohr=tol_max_g,
        tol_rms_g_eh_bohr=tol_rms_g,
        max_g_observed_eh_bohr=max_g_observed,
        rms_g_observed_eh_bohr=rms_g_observed,
        converged=converged,
        frozen_atom_indices=sorted(list(frozen_set)),
        residual_max_g_frozen_eh_bohr=res_max_g,
        residual_rms_g_frozen_eh_bohr=res_rms_g,
        deformation_channel_active=deformation_active,
        escalation_required=escalation_required,
        audit_notes=notes,
    )


def generate_orca_constraint_block(
    frozen_atom_indices: Sequence[int],
    symbols: Sequence[str] | None = None,
    coordinates: Sequence[Sequence[float]] | np.ndarray | None = None,
    mode: ConstraintCoordinateType = ConstraintCoordinateType.CARTESIAN,
) -> str:
    """
    Generates an authoritative ORCA %geom block with strict Method Matrix v4 convergence
    thresholds and Cartesian or internal coordinate constraints.
    """
    mode = ConstraintCoordinateType(mode)
    if mode not in (
        ConstraintCoordinateType.CARTESIAN,
        ConstraintCoordinateType.INTERNAL_BONDS,
    ):
        raise RuntimeError(
            "This constraint mode requires a reviewed complete coordinate adapter; no substitute was generated."
        )
    if mode == ConstraintCoordinateType.INTERNAL_BONDS and (
        symbols is None or coordinates is None
    ):
        raise ValueError(
            "Internal bond constraints require explicit symbols and geometry."
        )
    indices = list(frozen_atom_indices)
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or symbols is not None
        and index >= len(symbols)
        for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError(
            "Frozen atom indices must be distinct nonnegative integer rows."
        )
    lines: list[str] = [
        "%geom",
        "  TolE 1e-7",
        "  TolRMSG 3e-6",
        "  TolMaxG 1e-5",
        "  TolRMSD 5e-5",
        "  TolMaxD 1e-4",
    ]

    frozen_list = sorted(list(set(frozen_atom_indices)))
    if frozen_list:
        lines.append("  Constraints")
        if mode == ConstraintCoordinateType.CARTESIAN:
            for idx in frozen_list:
                lines.append(f"    {{ C {idx} C }}")
        elif mode == ConstraintCoordinateType.INTERNAL_BONDS:
            coords = np.asarray(coordinates)
            if (
                not np.isrealobj(coords)
                or coords.shape != (len(symbols), 3)
                or not np.isfinite(coords).all()
            ):
                raise ValueError(
                    "Internal bond constraints require finite real (N, 3) geometry."
                )
            radii = np.array(
                [get_dynamic_pyykko_radius(symbols[i]) for i in range(len(symbols))]
            )
            for i_idx, i in enumerate(frozen_list):
                for j in frozen_list[i_idx + 1 :]:
                    d = np.linalg.norm(coords[i] - coords[j])
                    if d < (radii[i] + radii[j]) * 1.25:
                        lines.append(f"    {{ B {i} {j} C }}")
        lines.append("  end")

    lines.append("end")
    return "\n".join(lines)


def generate_orca_frozen_monomer_constraints_block(
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]] | np.ndarray,
    frozen_monomer: str = "both",
    mode: str = "internal",
) -> str:
    """Render explicit Cartesian constraints without importing a sibling engine.

    Internal-coordinate generation belongs to the separately installed BASE
    provider and requires a reviewed process adapter. A TORQ import must never
    load BASE into the same Python environment or substitute Cartesian freezing
    for the requested internal-coordinate physics.
    """
    if mode not in ("cartesian", "internal"):
        raise ValueError("Constraint mode must be 'cartesian' or 'internal'.")
    if frozen_monomer not in (
        "A",
        "a",
        "monomer_a",
        "B",
        "b",
        "monomer_b",
        "both",
        "ALL",
    ):
        raise ValueError("Select frozen monomer A, B, or both explicitly.")
    geometry = np.asarray(coordinates)
    if (
        geometry.shape != (len(symbols), 3)
        or not np.isrealobj(geometry)
        or not np.all(np.isfinite(geometry))
    ):
        raise ValueError("Supply finite real Cartesian coordinates with shape (N, 3).")
    indices_a, indices_b = list(atoms_a), list(atoms_b)
    indices = indices_a + indices_b
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < len(symbols)
        for index in indices
    ) or len(set(indices)) != len(indices):
        raise ValueError("Monomer indices must be distinct in-range integer atom rows.")
    sub_a = atoms_a if frozen_monomer in ("A", "a", "monomer_a", "both", "ALL") else []
    sub_b = atoms_b if frozen_monomer in ("B", "b", "monomer_b", "both", "ALL") else []

    if mode == "cartesian":
        frozen_atoms = sorted(list(set(list(sub_a) + list(sub_b))))
        lines = [
            "%geom",
            "  TolE 1e-7",
            "  TolRMSG 3e-6",
            "  TolMaxG 1e-5",
            "  TolRMSD 5e-5",
            "  TolMaxD 1e-4",
        ]
        if frozen_atoms:
            lines.append("  Constraints")
            for idx in frozen_atoms:
                lines.append(f"    {{ C {idx} C }}")
            lines.append("  end")
        lines.append("end")
        return "\n".join(lines)

    raise RuntimeError(
        "Internal frozen-monomer constraints require a reviewed isolated BASE "
        "process adapter. Install BASE separately with scripts/student_setup.py; "
        "this legacy in-process operation is unavailable and no substitute was used."
    )


def generate_cfour_constraint_block(
    zmat_var_names: Sequence[str],
    frozen_var_names: Sequence[str],
    geo_conv: int = 5,
    geo_maxcyc: int = 50,
) -> str:
    """Reject the obsolete placeholder API without real ZMAT geometry values."""
    raise RuntimeError(
        "CFOUR constraint generation requires an actual reviewed molecule-specific "
        "ZMAT with geometry values and variable mapping. Variable names alone "
        "cannot establish an executable geometry; no placeholder was emitted."
    )


# =============================================================================
# 6. Master Constraint Manager Engine (Phase 4 / Stage 3.0-3.5)
# =============================================================================


class TorqConstraintManager:
    """
    Master constraint manager for CoChem-TORQ Phase 4 (Stage 3.0-3.5).
    Enforces frozen-monomer spatial protections, dynamic Mendeleev mass and radii retrieval,
    rotational error budgeting (fixing A vs B, C), and strict TolMaxG 1e-5 Eh/a0 convergence.
    """

    def __init__(
        self,
        tol_max_g: float = TOL_MAX_G_DEFAULT_EH_BOHR,
        tol_rms_g: float = TOL_RMS_G_DEFAULT_EH_BOHR,
        collision_threshold_angstrom: float = 1.20,
        dissociation_threshold_angstrom: float = 12.0,
    ) -> None:
        self.tol_max_g = tol_max_g
        self.tol_rms_g = tol_rms_g
        self.collision_threshold = collision_threshold_angstrom
        self.dissociation_threshold = dissociation_threshold_angstrom

    def analyze_and_enforce(
        self,
        symbols: Sequence[str],
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        constraint_mode: MonomerConstraintMode = MonomerConstraintMode.FROZEN_ISO,
        gradient: Sequence[Sequence[float]] | np.ndarray | None = None,
        freeze_monomer_index: int = 0,
    ) -> ConstraintEnforcementResult:
        """
        Executes full Phase 4 (Stage 3.0-3.5) analysis:
        1. Dynamic Mendeleev fragment partitioning and spatial collision guard.
        2. Principal rotational constants and asymmetry analysis.
        3. Error propagation budget (A vs B, C sensitivity).
        4. TolMaxG 1e-5 gradient convergence audit and deformation monitoring.
        5. ORCA / CFOUR constraint block formatting.
        """
        raw_coordinates = np.asarray(coordinates)
        if not np.isrealobj(raw_coordinates):
            raise ValueError("Constraint management requires real Cartesian geometry.")
        coords = np.asarray(raw_coordinates, dtype=np.float64)
        n_atoms = len(symbols)

        # 1. Monomer Partitioning
        monomers, inter_metrics = partition_monomer_fragments(
            symbols=symbols,
            coordinates=coords,
            constraint_mode=constraint_mode,
            collision_threshold_angstrom=self.collision_threshold,
        )

        # Guard against spatial collapse or runaway dissociation
        if inter_metrics.is_collision:
            raise SpatialCollisionError(
                f"[ERR_SPATIAL_COLLISION] Minimum interatomic distance "
                f"{inter_metrics.min_interatomic_distance_angstrom:.3f} Å is below "
                f"collision threshold {self.collision_threshold:.2f} Å."
            )
        if inter_metrics.is_dissociated:
            raise FragmentDissociationError(
                f"[ERR_FRAGMENT_DISSOCIATION] Monomer center-of-mass distance "
                f"{inter_metrics.center_of_mass_distance_angstrom:.3f} Å exceeds "
                f"dissociation limit {self.dissociation_threshold:.2f} Å."
            )

        r_eff = inter_metrics.center_of_mass_distance_angstrom
        if len(monomers) != 2 or r_eff is None or not np.isfinite(r_eff) or r_eff <= 0:
            raise ValueError(
                "Intermolecular error propagation requires exactly two fragments "
                "with a measured positive finite center-of-mass distance; "
                "no reference distance was substituted."
            )

        # 2. Rotational Constants
        rot_constants = compute_rotational_constants(symbols, coords)

        # 3. Error Budget & Allocation
        error_budget = propagate_rotational_error(
            R_angstrom=r_eff,
            delta_R_angstrom=0.002,
            monomer_bond_error_angstrom=0.010,
            base_constants=rot_constants,
        )

        # Determine frozen atoms
        frozen_indices: list[int] = []
        if constraint_mode in (
            MonomerConstraintMode.FROZEN_ISO,
            MonomerConstraintMode.FROZEN_INC,
        ):
            if len(monomers) > freeze_monomer_index:
                frozen_indices = monomers[freeze_monomer_index].atom_indices

        # 4. Convergence Audit
        audit: ConvergenceAudit | None = None
        if gradient is not None:
            audit = audit_gradient_convergence(
                gradient_eh_bohr=gradient,
                frozen_atom_indices=frozen_indices,
                tol_max_g=self.tol_max_g,
                tol_rms_g=self.tol_rms_g,
            )

        # 5. Constraint Code Generation
        orca_block = generate_orca_constraint_block(
            frozen_atom_indices=frozen_indices, symbols=symbols, coordinates=coords
        )

        provenance = {
            "TolMaxG": f"{self.tol_max_g:.1e} Eh/a0 [M]",
            "TolRMSG": f"{self.tol_rms_g:.1e} Eh/a0 [M]",
            "Monomer_A_accuracy": "unqualified; no independent accuracy evidence",
            "Intermolecular_R_rule": "first-order effective-distance sensitivity only",
            "Method_Matrix_Section": "9A.1-9A.2, 9A.7",
        }

        return ConstraintEnforcementResult(
            success=audit is not None and audit.converged,
            constraint_mode=constraint_mode,
            num_atoms=n_atoms,
            num_monomers=len(monomers),
            monomer_fragments=monomers,
            intermolecular_metrics=inter_metrics,
            rotational_constants=rot_constants,
            error_budget=error_budget,
            convergence_audit=audit,
            convergence_status=(
                "unassessed"
                if audit is None
                else "converged"
                if audit.converged
                else "unconverged"
            ),
            orca_geom_constraints_block=orca_block,
            cfour_zmat_constraints_block=None,
            provenance_tags=provenance,
        )
