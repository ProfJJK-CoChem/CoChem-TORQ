"""Machine-Precision Finite-Difference Gradient Verification Suite for CoChem-TORQ.

Method Matrix v4 Provenance Tags:
- [M] Mandated: Strict torch.float64 precision enforcement and finite-difference acceptance tolerances.
- [D] Derived: Central finite-difference mathematical formulation and relative Frobenius error metric.
- [E] Empirical: Displacement step size h = 1e-4 Angstrom and tolerance = 1e-4 eV/Angstrom.

Strict Zero-Mock Mandate v3: Absolutely no stubs, empty pass blocks, or mock data.
"""

from __future__ import annotations

from typing import Callable
import torch

from Libraries.cochem_torq_inference_errors import NumericalParityError
from Libraries.cochem_torq_inference_schemas import FiniteDiffVerificationResult


def verify_finite_difference_forces(
    coords: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float = 1e-4,
    tolerance: float = 1e-4,
) -> FiniteDiffVerificationResult:
    """Verify conservative analytical forces against central finite-difference gradients [M].

    Parameters
    ----------
    coords : torch.Tensor
        Cartesian coordinates of shape [N, 3]. Must strictly have dtype=torch.float64.
    energy_fn : Callable[[torch.Tensor], torch.Tensor]
        Function computing total molecular potential energy in eV.
    step_size : float
        Displacement step size h in Angstroms (default 1e-4 A) [E].
    tolerance : float
        Acceptance threshold for L_infinity and relative Frobenius error (default 1e-4 eV/A) [M].

    Returns
    -------
    FiniteDiffVerificationResult
        Validation report certifying parity and precision metrics.

    Raises
    ------
    NumericalParityError
        If coords.dtype != torch.float64, or if force errors exceed tolerance.
    """
    if coords.dtype != torch.float64:
        raise NumericalParityError(
            f"Mandatory float64 precision violated: received tensor of dtype {coords.dtype}. "
            "Finite-difference force verification requires torch.float64 to eliminate subtractive cancellation noise.",
            error_code="TORQ_PRECISION_MISMATCH",
            component="finite_difference",
            diagnostics={"received_dtype": str(coords.dtype), "required_dtype": "torch.float64"},
        )

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Cartesian coordinates must have shape [N, 3], got shape {list(coords.shape)}.")

    num_atoms = coords.shape[0]
    h = float(step_size)

    # 1. Analytical forces via automatic differentiation [D]
    coords_eval = coords.detach().clone().requires_grad_(True)
    energy = energy_fn(coords_eval)
    grad = torch.autograd.grad(energy, coords_eval, create_graph=False)[0]
    forces_analytic = -grad

    # 2. Central finite-difference numerical forces across 3N degrees of freedom [D]
    forces_num = torch.zeros_like(coords)
    for i in range(num_atoms):
        for alpha in range(3):
            coords_plus = coords.detach().clone()
            coords_minus = coords.detach().clone()
            coords_plus[i, alpha] += h
            coords_minus[i, alpha] -= h

            e_plus = energy_fn(coords_plus)
            e_minus = energy_fn(coords_minus)

            forces_num[i, alpha] = -(e_plus - e_minus) / (2.0 * h)

    # 3. Acceptance metrics [M]
    abs_errors = torch.abs(forces_analytic - forces_num)
    max_absolute_error = float(torch.max(abs_errors).item())

    frob_diff = torch.linalg.norm(forces_analytic - forces_num).item()
    frob_analytic = torch.linalg.norm(forces_analytic).item()
    relative_frobenius_error = float(frob_diff / (frob_analytic + 1e-12))

    passed = bool(max_absolute_error < tolerance and relative_frobenius_error < tolerance)

    if not passed:
        raise NumericalParityError(
            f"Finite-difference force parity check failed: "
            f"L_infinity = {max_absolute_error:.6e} eV/A (tol = {tolerance:.6e}), "
            f"RelErr_F = {relative_frobenius_error:.6e} (tol = {tolerance:.6e}).",
            error_code="TORQ_NUMERICAL_PARITY_ERROR",
            component="finite_difference",
            diagnostics={
                "max_absolute_error": max_absolute_error,
                "relative_frobenius_error": relative_frobenius_error,
                "step_size": h,
                "tolerance": float(tolerance),
                "passed": False,
            },
        )

    return FiniteDiffVerificationResult(
        max_absolute_error=max_absolute_error,
        relative_frobenius_error=relative_frobenius_error,
        step_size=h,
        passed=passed,
        dtype=str(coords.dtype),
        atom_count=num_atoms,
    )
