"""Force-Matching Loss Function with Second-Order Autograd (REQ-TORQ-TRAIN-098 [D]).

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic autograd mechanics and unit consistency.
Features:
- Conservative atomic forces: F_hat = -dE/dR via autograd.
- Autograd double-backward second-order derivative retention (create_graph=True).
- Robust Huber loss on force vectors resisting repulsive-wall core clashes.
- Size-extensive per-atom normalized multi-task loss (eV and Angstroms).
- Continuous angular force cosine similarity tracking metric.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

try:
    from cochem_base.schemas import ForceMatchingLossConfig
except ImportError:
    try:
        from Libraries.cochem_torq_training_schemas import ForceMatchingLossConfig
    except ImportError:
        from cochem_torq_training_schemas import ForceMatchingLossConfig


def compute_conservative_forces(
    energy: torch.Tensor,
    coordinates: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute analytical conservative atomic forces as the negative spatial gradient -dE/dR. [D]"""
    if not coordinates.requires_grad:
        raise ValueError("Coordinates must have requires_grad=True to compute conservative forces.")

    grad = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=coordinates,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]

    if grad is None:
        raise RuntimeError("Autograd returned None for spatial energy gradient.")

    return -grad


def huber_force_loss(
    force_error: torch.Tensor,
    delta_f: float = 0.01,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute vector Smooth-L1 Huber loss on 3D force error vectors. [D]

    L_Huber(x, delta) = 0.5 * ||x||^2 / delta  if ||x|| <= delta
                      = ||x|| - 0.5 * delta   if ||x|| > delta
    """
    norm = torch.sqrt(torch.sum(torch.square(force_error), dim=-1) + eps)
    loss = torch.where(
        norm <= delta_f,
        0.5 * torch.square(norm) / delta_f,
        norm - 0.5 * delta_f,
    )
    return loss


def huber_scalar_loss(
    error: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    """Compute standard 1D scalar Huber loss component-wise. [D]"""
    abs_err = torch.abs(error)
    return torch.where(
        abs_err <= delta,
        0.5 * torch.square(abs_err) / delta,
        abs_err - 0.5 * delta,
    )


def compute_angular_cosine_similarity(
    pred_forces: torch.Tensor,
    target_forces: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean angular force cosine similarity metric: (F . F_hat) / (||F|| * ||F_hat|| + eps). [D]"""
    dot_product = torch.sum(pred_forces * target_forces, dim=-1)
    norm_pred = torch.norm(pred_forces, p=2, dim=-1)
    norm_target = torch.norm(target_forces, p=2, dim=-1)
    cosine_sim = dot_product / (norm_pred * norm_target + eps)
    return torch.clamp(torch.mean(cosine_sim), min=-1.0, max=1.0)


class ForceMatchingLoss(nn.Module):
    """Multi-task force-matching loss engine with second-order autograd (REQ-TORQ-TRAIN-098 [D])."""

    def __init__(self, config: Optional[ForceMatchingLossConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else ForceMatchingLossConfig()

    def forward(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        pred_forces: torch.Tensor,
        target_forces: torch.Tensor,
        num_atoms: Sequence[int] | torch.Tensor,
        pred_virial: Optional[torch.Tensor] = None,
        target_virial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite energy, force Huber, and virial loss. [D]"""
        if isinstance(num_atoms, torch.Tensor):
            num_atoms_tensor = num_atoms.to(
                dtype=pred_energy.dtype, device=pred_energy.device
            )
        else:
            num_atoms_tensor = torch.tensor(
                list(num_atoms), dtype=pred_energy.dtype, device=pred_energy.device
            )

        total_atoms = torch.sum(num_atoms_tensor)
        _batch_size = pred_energy.numel()

        # 1. Energy loss: per-atom squared error
        # L_E = (1 / B) * sum(|E_b - E_hat_b|^2 / N_b)
        energy_diff = torch.flatten(pred_energy) - torch.flatten(target_energy)
        per_atom_energy_sq_error = torch.square(energy_diff) / num_atoms_tensor
        unweighted_energy_loss = torch.mean(per_atom_energy_sq_error)
        weighted_energy_loss = (
            self.config.energy_weight * unweighted_energy_loss
        )

        # 2. Force loss: Huber loss normalized by mode (atom_norm vs coordinate_component) [D]
        force_error = pred_forces - target_forces
        norm_mode = getattr(self.config, "normalization_mode", "atom_norm")
        delta_f = getattr(self.config, "huber_delta_force", 0.05)

        if norm_mode == "atom_norm":
            # Evaluates Huber loss over Euclidean vector norms ||F_i - F_hat_i||_2
            huber_per_atom = huber_force_loss(force_error, delta_f=delta_f)
            total_huber_sum = torch.sum(huber_per_atom)
            # Normalization by total atoms (REQ-TORQ-TRAIN-098 / Suggestion #55):
            # Eliminates the invalid 3x force gradient attenuation
            unweighted_force_loss = total_huber_sum / total_atoms.clamp(min=1.0)
        else:
            # Mode 'coordinate_component': evaluate Huber on each Cartesian component explicitly
            huber_coords = huber_scalar_loss(force_error, delta=delta_f)
            total_huber_sum = torch.sum(huber_coords)
            unweighted_force_loss = total_huber_sum / (
                3.0 * total_atoms.clamp(min=1.0)
            )

        weighted_force_loss = self.config.force_weight * unweighted_force_loss

        # 3. Virial loss (for periodic boundary conditions)
        if (
            self.config.virial_weight > 0.0
            and pred_virial is not None
            and target_virial is not None
        ):
            virial_diff = pred_virial - target_virial
            unweighted_virial_loss = torch.mean(torch.square(virial_diff))
            weighted_virial_loss = (
                self.config.virial_weight * unweighted_virial_loss
            )
        else:
            weighted_virial_loss = torch.tensor(
                0.0, dtype=pred_energy.dtype, device=pred_energy.device
            )

        # Total multi-task loss
        total_loss = (
            weighted_energy_loss + weighted_force_loss + weighted_virial_loss
        )

        # Metrics
        energy_mae = torch.mean(
            torch.abs(energy_diff) / num_atoms_tensor
        )  # eV/atom
        energy_mae_mev = energy_mae * 1000.0  # meV/atom

        # Force MAE in eV/Angstrom
        force_abs_diff = torch.abs(force_error)
        force_mae = torch.sum(force_abs_diff) / (
            3.0 * total_atoms.clamp(min=1.0)
        )

        angular_sim = compute_angular_cosine_similarity(
            pred_forces, target_forces
        )

        return {
            "loss": total_loss,
            "energy_loss": weighted_energy_loss,
            "force_loss": weighted_force_loss,
            "weighted_force_loss": weighted_force_loss,
            "unweighted_force_loss": unweighted_force_loss,
            "virial_loss": weighted_virial_loss,
            "energy_mae": energy_mae,
            "energy_mae_mev": energy_mae_mev,
            "force_mae": force_mae,
            "angular_similarity": angular_sim,
        }
