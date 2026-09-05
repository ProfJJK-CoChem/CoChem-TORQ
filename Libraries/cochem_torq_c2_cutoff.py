"""C^2-smooth quintic switching cutoff function for continuous forces and Hessians.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic analytical calculus and autograd parity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from Libraries.cochem_torq_inference_errors import CutoffContinuityError
from Libraries.cochem_torq_inference_schemas import C2SmoothCutoffConfig


def quintic_c2_envelope(distances: torch.Tensor, rc: float = 5.0) -> torch.Tensor:
    """Evaluate quintic C^2 switching envelope f_c(r) on interatomic distances. [M]/[D]

    f_c(r) = 1 - 10*u^3 + 15*u^4 - 6*u^5  for r <= rc (where u = r / rc)
    f_c(r) = 0                             for r > rc
    """
    if rc <= 0.0:
        raise CutoffContinuityError(
            f"Cutoff radius rc={rc} must be strictly positive.",
            diagnostics={"rc": rc},
        )
    u = torch.clamp(distances / rc, min=0.0)
    poly = 1.0 - 10.0 * (u ** 3) + 15.0 * (u ** 4) - 6.0 * (u ** 5)
    return torch.where(distances <= rc, poly, torch.zeros_like(distances))


def quintic_c2_first_derivative(distances: torch.Tensor, rc: float = 5.0) -> torch.Tensor:
    """Evaluate exact analytical first radial derivative df_c/dr. [D]

    df_c/dr = -(30/rc) * u^2 * (1 - u)^2  for r <= rc (where u = r / rc)
    df_c/dr = 0                            for r > rc
    """
    if rc <= 0.0:
        raise CutoffContinuityError(
            f"Cutoff radius rc={rc} must be strictly positive.",
            diagnostics={"rc": rc},
        )
    u = torch.clamp(distances / rc, min=0.0)
    dpoly = -(30.0 / rc) * (u ** 2) * ((1.0 - u) ** 2)
    return torch.where(distances <= rc, dpoly, torch.zeros_like(distances))


def quintic_c2_second_derivative(distances: torch.Tensor, rc: float = 5.0) -> torch.Tensor:
    """Evaluate exact analytical second radial derivative d^2f_c/dr^2. [D]

    d^2f_c/dr^2 = -(60/rc^2) * u * (1 - u) * (1 - 2*u)  for r <= rc
    d^2f_c/dr^2 = 0                                       for r > rc
    """
    if rc <= 0.0:
        raise CutoffContinuityError(
            f"Cutoff radius rc={rc} must be strictly positive.",
            diagnostics={"rc": rc},
        )
    u = torch.clamp(distances / rc, min=0.0)
    d2poly = -(60.0 / (rc ** 2)) * u * (1.0 - u) * (1.0 - 2.0 * u)
    return torch.where(distances <= rc, d2poly, torch.zeros_like(distances))


def quintic_c2_spatial_gradient(
    r_i: torch.Tensor,
    r_j: torch.Tensor,
    rc: float = 5.0,
) -> torch.Tensor:
    """Compute exact analytical spatial gradient with respect to Cartesian coordinates r_i. [D]

    r_ij = r_j - r_i
    grad_{r_i} f_c(r_ij) = (df_c/dr) * (r_i - r_j) / r_ij = - (df_c/dr) * (r_ij / r_ij)
    """
    r_ij = r_j - r_i  # (..., 3)
    dist = torch.norm(r_ij, dim=-1, keepdim=True)  # (..., 1)
    df_dr = quintic_c2_first_derivative(dist, rc=rc)  # (..., 1)

    # Handle singular r_ij = 0
    safe_dist = torch.clamp(dist, min=1e-12)
    grad = df_dr * (-r_ij / safe_dist)
    # Mask where distance is zero or greater than rc
    mask = (dist > 1e-12) & (dist <= rc)
    return torch.where(mask, grad, torch.zeros_like(grad))


def quintic_c2_spatial_hessian(
    r_i: torch.Tensor,
    r_j: torch.Tensor,
    rc: float = 5.0,
) -> torch.Tensor:
    """Compute exact analytical 3x3 Cartesian spatial Hessian with respect to r_i. [D]

    H_{alpha, beta} = (d^2f_c/dr^2) * (r_{ij, alpha} * r_{ij, beta} / r^2)
                    + (df_c/dr) * (delta_{alpha, beta} / r - r_{ij, alpha} * r_{ij, beta} / r^3)
    """
    r_ij = r_j - r_i  # (..., 3)
    dist = torch.norm(r_ij, dim=-1, keepdim=True)  # (..., 1)

    d2f = quintic_c2_second_derivative(dist, rc=rc)  # (..., 1)
    df = quintic_c2_first_derivative(dist, rc=rc)    # (..., 1)

    safe_dist = torch.clamp(dist, min=1e-12)
    safe_r2 = safe_dist ** 2
    safe_r3 = safe_dist ** 3

    # Outer product r_ij * r_ij^T -> (..., 3, 3)
    outer = torch.matmul(r_ij.unsqueeze(-1), r_ij.unsqueeze(-2))  # (..., 3, 3)
    eye = torch.eye(3, dtype=r_i.dtype, device=r_i.device)
    while eye.ndim < outer.ndim:
        eye = eye.unsqueeze(0)

    term1 = d2f.unsqueeze(-1) * (outer / safe_r2.unsqueeze(-1))
    term2 = df.unsqueeze(-1) * (eye / safe_dist.unsqueeze(-1) - outer / safe_r3.unsqueeze(-1))
    hessian = term1 + term2

    mask = (dist > 1e-12) & (dist <= rc)
    return torch.where(mask.unsqueeze(-1), hessian, torch.zeros_like(hessian))


def verify_cutoff_continuity(rc: float = 5.0, tolerance: float = 1e-7) -> bool:
    """Verify strict C^2 boundary continuity at r = rc within numerical tolerance. [D]"""
    if rc <= 0.0:
        raise CutoffContinuityError(
            f"Cutoff boundary rc={rc} must be strictly positive.",
            diagnostics={"rc": rc},
        )
    rc_tensor = torch.tensor([rc], dtype=torch.float64)
    val = float(quintic_c2_envelope(rc_tensor, rc=rc)[0])
    d1 = float(quintic_c2_first_derivative(rc_tensor, rc=rc)[0])
    d2 = float(quintic_c2_second_derivative(rc_tensor, rc=rc)[0])

    if abs(val) > tolerance:
        raise CutoffContinuityError(
            f"Cutoff value at boundary f_c(rc)={val} exceeds tolerance {tolerance}",
            diagnostics={"f_c": val, "rc": rc, "tolerance": tolerance},
        )
    if abs(d1) > tolerance:
        raise CutoffContinuityError(
            f"Cutoff first derivative at boundary df_c/dr(rc)={d1} exceeds tolerance {tolerance}",
            diagnostics={"df_c": d1, "rc": rc, "tolerance": tolerance},
        )
    if abs(d2) > tolerance:
        raise CutoffContinuityError(
            f"Cutoff second derivative at boundary d2f_c/dr2(rc)={d2} exceeds tolerance {tolerance}",
            diagnostics={"d2f_c": d2, "rc": rc, "tolerance": tolerance},
        )
    return True


class C2SmoothCutoff(nn.Module):
    """PyTorch Module encapsulating C^2-smooth quintic radial cutoff envelope. [M]/[D]"""

    def __init__(
        self,
        config: Optional[C2SmoothCutoffConfig] = None,
        cutoff_radius_rc: Optional[float] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self.rc = float(config.cutoff_radius_rc)
        elif cutoff_radius_rc is not None:
            self.rc = float(cutoff_radius_rc)
        else:
            self.rc = 5.0

        if self.rc <= 0.0:
            raise CutoffContinuityError(
                f"Cutoff radius rc={self.rc} must be strictly positive.",
                diagnostics={"rc": self.rc},
            )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Forward pass evaluating f_c(r). [M]"""
        return quintic_c2_envelope(distances, rc=self.rc)

    def compute_radial_derivatives(
        self, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (f_c, df_c/dr, d^2f_c/dr^2). [D]"""
        fc = quintic_c2_envelope(distances, rc=self.rc)
        d1 = quintic_c2_first_derivative(distances, rc=self.rc)
        d2 = quintic_c2_second_derivative(distances, rc=self.rc)
        return fc, d1, d2

    def compute_spatial_gradient(
        self, r_i: torch.Tensor, r_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial gradient with respect to r_i. [D]"""
        return quintic_c2_spatial_gradient(r_i, r_j, rc=self.rc)

    def compute_spatial_hessian(
        self, r_i: torch.Tensor, r_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute Cartesian 3x3 spatial Hessian with respect to r_i. [D]"""
        return quintic_c2_spatial_hessian(r_i, r_j, rc=self.rc)

    def verify_continuity(self, tolerance: float = 1e-7) -> bool:
        """Verify boundary continuity for configured rc. [D]"""
        return verify_cutoff_continuity(rc=self.rc, tolerance=tolerance)
