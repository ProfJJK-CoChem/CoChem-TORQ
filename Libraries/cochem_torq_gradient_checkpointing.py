"""Gradient Checkpointing Engine for High-Batch MLFF Training (REQ-TORQ-TRAIN-091 [D]).

Enforces autograd second-derivative mechanics (use_reentrant=False, preserve_rng_state=True)
for conservative atomic forces F = -dE/dR and joint composite loss backpropagation.
"""

from __future__ import annotations

import tracemalloc
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from Libraries.cochem_torq_training_errors import PrecisionDivergenceError


def compute_composite_loss(
    predicted_energy: torch.Tensor,
    reference_energy: torch.Tensor,
    predicted_forces: torch.Tensor,
    reference_forces: torch.Tensor,
    energy_loss_weight: float = 1.0,
    force_loss_weight: float = 100.0,
    num_atoms_per_molecule: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate joint composite MLFF loss normalized by atomic degrees of freedom. [D]
    
    L(theta) = w_E * (1/B) sum |E_b - E_b^ref|^2 + w_F * (1 / (3 * sum N_b)) sum ||F_i - F_i^ref||^2
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (total_loss, energy_loss_component, force_loss_component)
    """
    batch_size = predicted_energy.shape[0] if predicted_energy.dim() > 0 else 1
    energy_diff = predicted_energy.view(-1) - reference_energy.view(-1)
    e_loss = torch.mean(torch.square(energy_diff))

    force_diff = predicted_forces.reshape(-1, 3) - reference_forces.reshape(-1, 3)
    squared_force_errors = torch.sum(torch.square(force_diff), dim=-1)  # shape (N_atoms,)

    if num_atoms_per_molecule is not None:
        total_atoms = torch.sum(num_atoms_per_molecule).item()
    else:
        total_atoms = force_diff.shape[0]

    # Total degrees of freedom = 3 * sum(N_b)
    total_dof = 3.0 * float(total_atoms) if total_atoms > 0 else 3.0
    f_loss = torch.sum(squared_force_errors) / total_dof

    total_loss = (energy_loss_weight * e_loss) + (force_loss_weight * f_loss)

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        raise PrecisionDivergenceError(
            f"Composite loss diverged: NaN or Inf detected (e_loss={e_loss.item()}, f_loss={f_loss.item()}).",
            diagnostics={
                "energy_loss": float(e_loss.item()),
                "force_loss": float(f_loss.item()),
            },
        )

    return total_loss, e_loss, f_loss


class CheckpointedMLFF(nn.Module):
    """PyTorch wrapper integrating activation recomputation with autograd force derivation. [D]
    
    Uses use_reentrant=False in torch.utils.checkpoint to cleanly support second-derivative
    autograd graphs required for force loss backpropagation:
        F_i = -dE / dR_i
        dL / dTheta = dL/dE * dE/dTheta + dL/dF * d^2E / (dTheta dR)
    """

    def __init__(
        self,
        energy_model: nn.Module,
        gradient_checkpointing: bool = True,
        preserve_rng_state: bool = True,
    ) -> None:
        super().__init__()
        self.energy_model = energy_model
        self.gradient_checkpointing = gradient_checkpointing
        self.preserve_rng_state = preserve_rng_state

    def forward(
        self,
        coordinates: torch.Tensor,
        species: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both total energy and analytical atomic forces. [D]
        
        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic Cartesian coordinates (N_atoms, 3), must require grad.
        species : torch.Tensor
            Atomic numbers Z (N_atoms,).
        edge_index : Optional[torch.Tensor]
            Graph topology edge indices (2, N_edges), excluded from recomputation graph.
        batch : Optional[torch.Tensor]
            Batch indices mapping atoms to molecules.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (energy, forces) where forces = -dE/dcoordinates
        """
        if not coordinates.requires_grad:
            coordinates.requires_grad_(True)

        def forward_energy(coords: torch.Tensor) -> torch.Tensor:
            # Species and edge_index are treated as immutable graph topology fixtures
            return self.energy_model(coords, species, edge_index=edge_index, batch=batch)

        if self.gradient_checkpointing and self.training:
            energy = checkpoint(
                forward_energy,
                coordinates,
                use_reentrant=False,
                preserve_rng_state=self.preserve_rng_state,
            )
        else:
            energy = forward_energy(coordinates)

        # Analytical spatial autograd forces: F = -dE/dR
        # create_graph=True and retain_graph=True allow higher-order differentiation
        # into model parameters theta during loss backpropagation
        grad_outputs = torch.ones_like(energy)
        grad_coords = torch.autograd.grad(
            outputs=energy,
            inputs=coordinates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        forces = -grad_coords
        return energy, forces


def evaluate_forces_parity(
    model: nn.Module,
    coordinates: torch.Tensor,
    species: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Evaluate bit-for-bit force parity between eager execution and checkpointed execution. [M]
    
    Returns
    -------
    Tuple[float, torch.Tensor, torch.Tensor]
        (max_delta_f, eager_forces, checkpointed_forces)
    """
    coords_eager = coordinates.clone().detach().requires_grad_(True)
    coords_chkpt = coordinates.clone().detach().requires_grad_(True)
    was_training = model.training

    # Capture initial RNG states to ensure identical stochastic masks across both passes
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    # 1. Eager execution
    wrapper_eager = CheckpointedMLFF(model, gradient_checkpointing=False)
    if was_training:
        wrapper_eager.train()
    else:
        wrapper_eager.eval()
    _, forces_eager = wrapper_eager(coords_eager, species, edge_index=edge_index)

    # Restore RNG state for checkpointed pass
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng)

    # 2. Checkpointed execution
    wrapper_chkpt = CheckpointedMLFF(model, gradient_checkpointing=True)
    if was_training:
        wrapper_chkpt.train()
    else:
        wrapper_chkpt.eval()
    _, forces_chkpt = wrapper_chkpt(coords_chkpt, species, edge_index=edge_index)

    # Restore original training state
    if was_training:
        model.train()
    else:
        model.eval()

    max_delta = float(torch.max(torch.abs(forces_eager - forces_chkpt)).item())
    return max_delta, forces_eager, forces_chkpt


def profile_checkpointing_memory(
    model: nn.Module,
    coordinates: torch.Tensor,
    species: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Profile memory consumption with and without gradient checkpointing. [M]
    
    Tracks CUDA VRAM if available, otherwise measures CPU peak memory via tracemalloc.
    """
    results: Dict[str, float] = {}

    if torch.cuda.is_available():
        device = coordinates.device
        torch.cuda.reset_peak_memory_stats(device)
        wrapper_eager = CheckpointedMLFF(model, gradient_checkpointing=False).to(device)
        coords_eager = coordinates.clone().detach().requires_grad_(True).to(device)
        e_e, f_e = wrapper_eager(coords_eager, species.to(device), edge_index)
        loss_e = (e_e.sum() + f_e.sum())
        loss_e.backward()
        vram_eager = float(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        wrapper_chk = CheckpointedMLFF(model, gradient_checkpointing=True).to(device)
        coords_chk = coordinates.clone().detach().requires_grad_(True).to(device)
        e_c, f_c = wrapper_chk(coords_chk, species.to(device), edge_index)
        loss_c = (e_c.sum() + f_c.sum())
        loss_c.backward()
        vram_chk = float(torch.cuda.max_memory_allocated(device))

        results["eager_bytes"] = vram_eager
        results["checkpoint_bytes"] = vram_chk
        results["ratio"] = vram_chk / max(vram_eager, 1.0)
    else:
        # Tier 1 Windows CPU memory tracking via tracemalloc
        tracemalloc.start()
        wrapper_e = CheckpointedMLFF(model, gradient_checkpointing=False)
        coords_e = coordinates.clone().detach().requires_grad_(True)
        e_e, f_e = wrapper_e(coords_e, species, edge_index)
        loss_e = (e_e.sum() + f_e.sum())
        loss_e.backward()
        current_e, peak_e = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        wrapper_c = CheckpointedMLFF(model, gradient_checkpointing=True)
        coords_c = coordinates.clone().detach().requires_grad_(True)
        e_c, f_c = wrapper_c(coords_c, species, edge_index)
        loss_c = (e_c.sum() + f_c.sum())
        loss_c.backward()
        current_c, peak_c = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["eager_bytes"] = float(peak_e)
        results["checkpoint_bytes"] = float(peak_c)
        results["ratio"] = float(peak_c) / max(float(peak_e), 1.0)

    return results
