"""Dedicated TorchScript Compilation Engine for C++ Inference (REQ-TORQ-TRAIN-093 [D]).

Features:
- Standalone torch.jit.script compilation supporting dynamic molecular shapes (N_atoms x 3).
- Double-precision (float64) numerical parity validation for energy and forces.
- Parity tolerances: relative energy <= 1e-6, force L_inf <= 1e-6 Hartree/Å.
- ParityVerificationError raised upon parity failure or corruption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from Libraries.cochem_torq_training_errors import ParityVerificationError
from Libraries.cochem_torq_training_schemas import TorchScriptExportConfig


class TorchScriptableMLFF(nn.Module):
    """Reference TorchScript-compatible MLFF backbone supporting ragged molecular tensors. [D]"""

    def __init__(self, hidden_dim: int = 32, num_radial: int = 16) -> None:
        super().__init__()
        self.species_emb = nn.Embedding(119, hidden_dim)
        self.num_radial = num_radial
        self.r_centers = nn.Parameter(
            torch.linspace(0.6, 6.0, num_radial, dtype=torch.float64),
            requires_grad=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_radial, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=torch.float64),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        species: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch_ptr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate potential energy in Hartree from Cartesian coordinates in Angstroms. [D]"""
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (N, N, 3)
        dist = torch.norm(diff + 1e-12, dim=-1)  # (N, N)

        # Smooth Gaussian radial basis smearing
        r_exp = torch.exp(-0.5 * torch.square((dist.unsqueeze(-1) - self.r_centers) / 0.4))  # (N, N, R)
        # Exclude self-interaction diagonal
        eye_mask = torch.eye(coordinates.size(0), dtype=torch.bool, device=coordinates.device).unsqueeze(-1)
        r_exp = torch.where(eye_mask, torch.zeros_like(r_exp), r_exp)
        radial_features = torch.sum(r_exp, dim=1)  # (N, R)

        z_emb = self.species_emb(species).to(torch.float64)  # (N, H)
        node_features = torch.cat([z_emb, radial_features], dim=-1)  # (N, H + R)
        atomic_energies = self.mlp(node_features).squeeze(-1)  # (N,)

        return torch.sum(atomic_energies)


def compute_energy_and_forces_eager(
    model: nn.Module,
    coordinates: torch.Tensor,
    species: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    batch_ptr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute potential energy and analytical spatial autograd forces in float64. [D]"""
    coords = coordinates.clone().detach().to(torch.float64).requires_grad_(True)
    energy = model(coords, species, edge_index=edge_index, batch_ptr=batch_ptr)
    grad_outputs = torch.ones_like(energy)
    grad_coords = torch.autograd.grad(
        outputs=energy,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
    )[0]
    forces = -grad_coords
    return energy.detach(), forces.detach()


def export_model_to_torchscript(
    model: nn.Module,
    export_path: Path,
) -> torch.jit.ScriptModule:
    """Compile model via torch.jit.script and save to disk. [M]"""
    model.eval()
    scripted = torch.jit.script(model)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(export_path))
    return scripted


def verify_torchscript_parity(
    eager_model: nn.Module,
    scripted_model: torch.jit.ScriptModule,
    test_fixtures: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    config: TorchScriptExportConfig,
) -> Dict[str, Any]:
    """Validate numerical parity in float64 between eager and TorchScript models. [M]
    
    Raises ParityVerificationError if parity error exceeds configured thresholds:
    - Relative energy error <= config.energy_relative_tolerance
    - Force L_inf error <= config.force_parity_tolerance_hartree_angstrom
    """
    eager_model.eval()
    scripted_model.eval()

    verification_results: List[Dict[str, float]] = []

    for idx, (coords, species) in enumerate(test_fixtures):
        # 1. Eager evaluation
        e_eager, f_eager = compute_energy_and_forces_eager(eager_model, coords, species)

        # 2. Scripted evaluation
        c_script = coords.clone().detach().to(torch.float64).requires_grad_(True)
        e_script = scripted_model(c_script, species)
        grad_out = torch.ones_like(e_script)
        grad_coords = torch.autograd.grad(
            outputs=e_script,
            inputs=c_script,
            grad_outputs=grad_out,
            create_graph=False,
            retain_graph=False,
        )[0]
        f_script = -grad_coords

        # Compute errors
        abs_energy_err = float(torch.abs(e_eager - e_script.detach()).item())
        denom = max(float(torch.abs(e_eager).item()), 1e-12)
        rel_energy_err = abs_energy_err / denom
        max_force_err = float(torch.max(torch.abs(f_eager - f_script.detach())).item())

        fixture_metrics = {
            "fixture_index": idx,
            "num_atoms": coords.shape[0],
            "abs_energy_error": abs_energy_err,
            "rel_energy_error": rel_energy_err,
            "max_force_error": max_force_err,
        }
        verification_results.append(fixture_metrics)

        # Check tolerances
        if rel_energy_err > config.energy_relative_tolerance:
            raise ParityVerificationError(
                f"TorchScript parity failure on fixture {idx} (N={coords.shape[0]}): "
                f"Relative energy error {rel_energy_err:.2e} exceeds tolerance {config.energy_relative_tolerance:.2e}.",
                diagnostics=fixture_metrics,
            )

        if max_force_err > config.force_parity_tolerance_hartree_angstrom:
            raise ParityVerificationError(
                f"TorchScript parity failure on fixture {idx} (N={coords.shape[0]}): "
                f"Max force error {max_force_err:.2e} Hartree/Å exceeds tolerance {config.force_parity_tolerance_hartree_angstrom:.2e}.",
                diagnostics=fixture_metrics,
            )

    return {"status": "PASSED", "fixtures_tested": len(test_fixtures), "metrics": verification_results}


def compile_and_validate_torchscript(
    model: nn.Module,
    config: TorchScriptExportConfig,
    test_fixtures: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.jit.ScriptModule:
    """Full compilation and verification pipeline exporting verified TorchScript binary. [D]"""
    scripted = export_model_to_torchscript(model, config.export_path)

    if config.validate_against_fixtures and test_fixtures:
        verify_torchscript_parity(model, scripted, test_fixtures, config)

    return scripted
