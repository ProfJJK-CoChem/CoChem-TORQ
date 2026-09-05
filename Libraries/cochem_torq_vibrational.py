"""Double-Autograd Cartesian Hessian, CIAAW Masses, Eckart Projection & Vibrational Analyzer.

Method Matrix v4 Provenance Tags:
- [M] Mandated: Dynamic Mendeleev CIAAW monoisotopic mass retrieval, float64 precision.
- [D] Derived: Double-autograd Cartesian Hessian, Gram-Schmidt Eckart projector, CODATA 2022 dimensional scaling.
- [E] Empirical: Orthonormalization threshold 1e-7, signed frequency non-NaN reporting.

Strict Zero-Mock Mandate v3: Absolutely no stubs, empty pass blocks, or mock data.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple
import mendeleev
import torch

from Libraries.cochem_torq_inference_errors import (
    NumericalParityError,
    PhysicsDivergenceError,
)
from Libraries.cochem_torq_inference_schemas import VibrationalModes

# CODATA 2022 Fundamental Constants [M]/[D]
# 1 eV = 1.602176634e-19 J
# 1 A  = 1.0e-10 m
# 1 u  = 1.66053906660e-27 kg
# c    = 2.99792458e10 cm/s
# h    = 4.135667696e-15 eV*s
# kappa = 1.602176634e-19 / ( (1e-10)^2 * 1.66053906660e-27 ) = 9.648533212331e27 s^-2 / (eV * A^-2 * u^-1)
CODATA_2022_KAPPA = 9.648533212331e27
CODATA_2022_C_CMS = 2.99792458e10
CODATA_2022_H_EVS = 4.135667696e-15
CODATA_2022_HC_EV_CM = CODATA_2022_H_EVS * CODATA_2022_C_CMS  # 1.239841984e-4 eV*cm
# Conversion factor from sqrt(eV / (A^2 * u)) to cm^-1: sqrt(kappa) / (2 * pi * c)
CODATA_2022_FREQ_FACTOR = math.sqrt(CODATA_2022_KAPPA) / (2.0 * math.pi * CODATA_2022_C_CMS)  # ~521.4708316 cm^-1


def resolve_ciaaw_monoisotopic_mass(atomic_number: int) -> float:
    """Retrieve pure CIAAW monoisotopic mass for the most abundant isotope of element Z [M].

    Queries mendeleev.element(Z).isotopes and filters by maximum terrestrial
    isotopic abundance. Standard terrestrial average weights are strictly forbidden.

    Parameters
    ----------
    atomic_number : int
        Nuclear charge Z (1 <= Z <= 118).

    Returns
    -------
    float
        Pure monoisotopic mass in unified atomic mass units (u).

    Raises
    ------
    PhysicsDivergenceError
        If element is invalid or no isotopic record is available.
    """
    if atomic_number < 1 or atomic_number > 118:
        raise PhysicsDivergenceError(
            f"Invalid atomic number Z={atomic_number}. Must be in chemical domain [1, 118].",
            error_code="TORQ_INVALID_ATOMIC_NUMBER",
            component="monoisotopic_resolver",
            diagnostics={"atomic_number": atomic_number},
        )

    try:
        elem = mendeleev.element(int(atomic_number))
    except Exception as exc:
        raise PhysicsDivergenceError(
            f"Failed to retrieve element record for Z={atomic_number}: {exc}",
            error_code="TORQ_MENDELEEV_QUERY_FAIL",
            component="monoisotopic_resolver",
            diagnostics={"atomic_number": atomic_number, "exception": str(exc)},
        ) from exc

    isotopes = [
        iso
        for iso in elem.isotopes
        if iso.abundance is not None and iso.abundance > 0.0
    ]
    if not isotopes:
        isotopes = elem.isotopes
    if not isotopes:
        raise PhysicsDivergenceError(
            f"No isotopic mass records available for atomic number Z={atomic_number}.",
            error_code="TORQ_NO_ISOTOPES",
            component="monoisotopic_resolver",
            diagnostics={"atomic_number": atomic_number, "symbol": elem.symbol},
        )

    most_abundant = max(isotopes, key=lambda iso: iso.abundance or 0.0)
    if most_abundant.mass is None:
        raise PhysicsDivergenceError(
            f"CIAAW mass undefined for element {elem.symbol} (Z={atomic_number}).",
            error_code="TORQ_UNDEFINED_ISOTOPE_MASS",
            component="monoisotopic_resolver",
            diagnostics={"atomic_number": atomic_number, "symbol": elem.symbol},
        )

    return float(most_abundant.mass)


def compute_cartesian_hessian(
    coords: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Evaluate Cartesian second-derivative Hessian matrix via double autograd in torch.float64 [D].

    Parameters
    ----------
    coords : torch.Tensor
        Cartesian coordinates of shape [N, 3], strictly torch.float64.
    energy_fn : Callable[[torch.Tensor], torch.Tensor]
        Total potential energy function in eV.

    Returns
    -------
    torch.Tensor
        Symmetrized Cartesian Hessian matrix of shape [3N, 3N] in eV / A^2.

    Raises
    ------
    NumericalParityError
        If coords.dtype != torch.float64.
    """
    if coords.dtype != torch.float64:
        raise NumericalParityError(
            f"Double-autograd Hessian requires torch.float64, got {coords.dtype}.",
            error_code="TORQ_PRECISION_MISMATCH",
            component="cartesian_hessian",
            diagnostics={"dtype": str(coords.dtype)},
        )

    coords_eval = coords.detach().clone().requires_grad_(True)
    energy = energy_fn(coords_eval)
    grad = torch.autograd.grad(energy, coords_eval, create_graph=True)[0]
    grad_flat = grad.reshape(-1)
    num_dofs = grad_flat.shape[0]

    hessian_rows = []
    for k in range(num_dofs):
        retain = k < (num_dofs - 1)
        row = torch.autograd.grad(
            grad_flat[k],
            coords_eval,
            retain_graph=retain,
            create_graph=False,
        )[0]
        hessian_rows.append(row.reshape(-1))

    H = torch.stack(hessian_rows, dim=0)
    # Active Hermitian symmetrization to eliminate numerical asymmetry [D]
    H_sym = 0.5 * (H + H.T)
    return H_sym


def compute_eckart_projector(
    coords: torch.Tensor,
    masses: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Construct Gram-Schmidt Eckart projection operator removing translational and rotational modes [D].

    Parameters
    ----------
    coords : torch.Tensor
        Cartesian coordinates of shape [N, 3], strictly torch.float64.
    masses : torch.Tensor
        Atomic masses of shape [N], strictly torch.float64.

    Returns
    -------
    Tuple[torch.Tensor, int]
        P: Eckart projection operator [3N, 3N] satisfying P^2 = P, P^T = P, Tr(P) = 3N - D.
        D: Number of projected degrees of freedom (6 for non-linear, 5 for linear, 3 for single atom).
    """
    if coords.dtype != torch.float64 or masses.dtype != torch.float64:
        raise NumericalParityError(
            "Eckart projector requires torch.float64 precision for both coordinates and masses.",
            error_code="TORQ_PRECISION_MISMATCH",
            component="eckart_projector",
            diagnostics={"coords_dtype": str(coords.dtype), "masses_dtype": str(masses.dtype)},
        )

    num_atoms = coords.shape[0]
    device = coords.device

    # 1. Center of mass translation [D]
    total_mass = torch.sum(masses)
    if total_mass <= 0.0:
        raise PhysicsDivergenceError(
            f"Total molecular mass is non-positive: {total_mass.item():.6e} u.",
            component="eckart_projector",
        )
    r_com = torch.sum(coords * masses.unsqueeze(1), dim=0) / total_mass
    rel_coords = coords - r_com

    # 2. Mass-weighted translation basis vectors (T_alpha, alpha in {x, y, z}) [D]
    sqrt_masses = torch.sqrt(masses)
    basis_candidates = []
    for alpha in range(3):
        t = torch.zeros((num_atoms, 3), dtype=torch.float64, device=device)
        t[:, alpha] = sqrt_masses
        basis_candidates.append(t.reshape(-1))

    # 3. Mass-weighted rotation basis vectors (R_alpha, alpha in {x, y, z}) [D]
    unit_axes = torch.eye(3, dtype=torch.float64, device=device)
    for alpha in range(3):
        axis = unit_axes[alpha]
        rot = torch.zeros((num_atoms, 3), dtype=torch.float64, device=device)
        for i in range(num_atoms):
            rot[i] = sqrt_masses[i] * torch.linalg.cross(axis, rel_coords[i])
        basis_candidates.append(rot.reshape(-1))

    # 4. Sequential Gram-Schmidt orthonormalization, discarding null vectors (< 1e-7) [D]
    ortho_basis = []
    for candidate in basis_candidates:
        w = candidate.clone()
        for u in ortho_basis:
            w = w - torch.dot(u, w) * u
        norm = torch.linalg.norm(w)
        if norm > 1e-7:
            ortho_basis.append(w / norm)

    d_proj = len(ortho_basis)
    if d_proj == 0:
        p_matrix = torch.eye(3 * num_atoms, dtype=torch.float64, device=device)
        return p_matrix, 0

    u_matrix = torch.stack(ortho_basis, dim=1)  # [3N, D]
    identity = torch.eye(3 * num_atoms, dtype=torch.float64, device=device)
    p_matrix = identity - u_matrix @ u_matrix.T
    p_matrix = 0.5 * (p_matrix + p_matrix.T)

    return p_matrix, d_proj


def analyze_vibrational_frequencies(
    coords: torch.Tensor,
    atomic_numbers: Sequence[int],
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    filter_projected: bool = True,
) -> VibrationalModes:
    """Compute mass-weighted projected normal modes, CODATA 2022 frequencies, and ZPVE [M]/[D].

    Parameters
    ----------
    coords : torch.Tensor
        Equilibrium Cartesian coordinates [N, 3] in Angstroms, strictly torch.float64.
    atomic_numbers : Sequence[int]
        Nuclear charges Z for each atom.
    energy_fn : Callable[[torch.Tensor], torch.Tensor]
        Total potential energy function in eV.
    filter_projected : bool
        If True (default), project and remove the D translational/rotational zero modes,
        reporting strictly the 3N - D genuine vibrational normal modes.

    Returns
    -------
    VibrationalModes
        Certified normal mode report containing signed wavenumbers (cm^-1), eigenvalues,
        imaginary mode count, and ZPVE (eV).
    """
    if coords.dtype != torch.float64:
        raise NumericalParityError(
            f"Vibrational analysis requires torch.float64 coordinates, got {coords.dtype}.",
            error_code="TORQ_PRECISION_MISMATCH",
            component="vibrational_analyzer",
            diagnostics={"dtype": str(coords.dtype)},
        )

    num_atoms = coords.shape[0]
    if len(atomic_numbers) != num_atoms:
        raise ValueError(
            f"Atomic numbers length ({len(atomic_numbers)}) does not match coordinates atom count ({num_atoms})."
        )

    # 1. Resolve pure CIAAW monoisotopic masses [M]
    mass_list = [resolve_ciaaw_monoisotopic_mass(int(z)) for z in atomic_numbers]
    masses = torch.tensor(mass_list, dtype=torch.float64, device=coords.device)

    # 2. Evaluate Cartesian Hessian [D]
    H_cart = compute_cartesian_hessian(coords, energy_fn)

    # 3. Mass-weight the Hessian: H_tilde_{ia, jb} = H_{ia, jb} / sqrt(m_i * m_j) [D]
    mass_repeat = torch.repeat_interleave(masses, 3)
    inv_sqrt_mass = 1.0 / torch.sqrt(mass_repeat)
    H_tilde = H_cart * torch.outer(inv_sqrt_mass, inv_sqrt_mass)

    # 4. Gram-Schmidt Eckart projection [D]
    P_eckart, d_proj = compute_eckart_projector(coords, masses)
    H_proj = P_eckart @ H_tilde @ P_eckart
    H_proj = 0.5 * (H_proj + H_proj.T)

    # 5. Diagonalization via torch.linalg.eigh [D]
    raw_eigenvalues, _ = torch.linalg.eigh(H_proj)

    # 6. Filter out the D translational and rotational zero modes (|lambda| < 1e-7) [D]
    if filter_projected and d_proj > 0:
        # Sort indices by absolute eigenvalue magnitude; the D smallest are the projected zero modes
        abs_order = sorted(range(len(raw_eigenvalues)), key=lambda i: abs(raw_eigenvalues[i].item()))
        vib_indices = sorted(abs_order[d_proj:])
        selected_eigenvalues = [raw_eigenvalues[i].item() for i in vib_indices]
    else:
        selected_eigenvalues = [ev.item() for ev in raw_eigenvalues]

    # Sort genuine normal modes by signed eigenvalue
    selected_eigenvalues = sorted(selected_eigenvalues)

    # 7. CODATA 2022 Signed Frequencies: nu_k = sgn(lambda_k) * C_freq * sqrt(|lambda_k|) [D]
    frequencies_cm1: List[float] = []
    imaginary_count = 0
    for lam in selected_eigenvalues:
        if abs(lam) < 1e-12:
            freq = 0.0
        else:
            sgn = 1.0 if lam >= 0.0 else -1.0
            freq = sgn * CODATA_2022_FREQ_FACTOR * math.sqrt(abs(lam))

        frequencies_cm1.append(float(freq))
        if freq < 0.0:
            imaginary_count += 1

    # 8. Harmonic Zero-Point Vibrational Energy (ZPVE) summed over real modes (lambda_k > 0) [D]
    # E_ZPE = 0.5 * sum_{k: nu_k > 0} (h * c * nu_k) in eV
    zpe_ev = 0.5 * sum(
        CODATA_2022_HC_EV_CM * f
        for f in frequencies_cm1
        if f > 0.0
    )

    return VibrationalModes(
        frequencies_cm1=frequencies_cm1,
        zero_point_energy_ev=float(zpe_ev),
        imaginary_mode_count=imaginary_count,
        eigenvalues=selected_eigenvalues,
        projected_degrees_of_freedom=d_proj,
        mass_weighting_standard="CIAAW_MONOISOTOPIC",
    )
