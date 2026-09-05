"""Multi-Task Learning Head with Log-Variance Homoscedastic Loss for CoChem-TORQ.

Method Matrix v4 Provenance Tags:
- [M] Mandated: Multi-task auxiliary head for total potential energy and HOMO-LUMO gap.
- [D] Derived: Log-variance homoscedastic uncertainty loss formulation and analytical gradients.
- [E] Empirical: Huber loss threshold limits (delta_E = 0.01 eV, delta_G = 0.05 eV).

Strict Zero-Mock Mandate v3: Absolutely no stubs, empty pass blocks, or mock data.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from Libraries.cochem_torq_inference_errors import PhysicsDivergenceError


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Smooth Huber loss envelope for robust regression against outliers [D].

    L_Huber(y, y*; delta) = 0.5 * (y - y*)^2        if |y - y*| <= delta
                          = delta * (|y - y*| - 0.5 * delta)  otherwise
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff <= delta,
        0.5 * (diff**2),
        delta * (diff - 0.5 * delta),
    )
    return torch.mean(loss)


class MultiTaskHead(nn.Module):
    """Auxiliary equivariant regression head predicting energy and HOMO-LUMO gap [M].

    Enforces strictly positive HOMO-LUMO frontier orbital gap via smooth Softplus
    activation and defensive physical divergence checks.
    """

    def __init__(
        self,
        in_features: int = 128,
        hidden_dim: int = 64,
        epsilon_gap: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.epsilon_gap = float(epsilon_gap)

        self.energy_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )

        self.gap_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting potential energy and HOMO-LUMO gap [M].

        Parameters
        ----------
        x : torch.Tensor
            Latent representation tensor of shape [batch_size, in_features].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            energy: [batch_size, 1] in eV.
            homo_lumo_gap: [batch_size, 1] in eV.
        """
        weight_dtype = self.energy_head[0].weight.dtype
        if x.dtype != weight_dtype:
            if x.dtype == torch.float64:
                self.to(torch.float64)
            else:
                x = x.to(weight_dtype)

        energy = self.energy_head(x)
        raw_gap = self.gap_head(x)
        gap = F.softplus(raw_gap) + self.epsilon_gap

        if not torch.jit.is_tracing():
            if (gap <= 0.0).any():
                min_val = float(torch.min(gap).item())
                raise PhysicsDivergenceError(
                    f"HOMO-LUMO gap predicted non-positive value: {min_val:.6e} eV.",
                    error_code="TORQ_GAP_DIVERGENCE",
                    component="multi_task_head",
                    diagnostics={"min_gap_ev": min_val, "epsilon_gap": self.epsilon_gap},
                )

        return energy, gap


class HomoscedasticMultiTaskLoss(nn.Module):
    """Joint homoscedastic multi-task objective with learnable log-variances [D].

    L_multi = 0.5 * exp(-s_E) * L_E(delta_E) + 0.5 * exp(-s_G) * L_G(delta_G) + 0.5 * (s_E + s_G)
    """

    def __init__(
        self,
        delta_energy: float = 0.01,
        delta_gap: float = 0.05,
        init_s_energy: float = 0.0,
        init_s_gap: float = 0.0,
    ) -> None:
        super().__init__()
        self.delta_energy = float(delta_energy)
        self.delta_gap = float(delta_gap)

        self.s_E = nn.Parameter(torch.tensor(float(init_s_energy), dtype=torch.float64))
        self.s_G = nn.Parameter(torch.tensor(float(init_s_gap), dtype=torch.float64))

    def forward(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        pred_gap: torch.Tensor,
        target_gap: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate joint homoscedastic loss [D].

        Parameters
        ----------
        pred_energy : torch.Tensor
            Predicted energy in eV.
        target_energy : torch.Tensor
            Reference energy in eV.
        pred_gap : torch.Tensor
            Predicted HOMO-LUMO gap in eV.
        target_gap : torch.Tensor
            Reference HOMO-LUMO gap in eV.

        Returns
        -------
        torch.Tensor
            Joint scalar loss.
        """
        loss_E = huber_loss(pred_energy, target_energy, self.delta_energy)
        loss_G = huber_loss(pred_gap, target_gap, self.delta_gap)

        s_E_cast = self.s_E.to(loss_E.dtype)
        s_G_cast = self.s_G.to(loss_G.dtype)

        term_E = 0.5 * torch.exp(-s_E_cast) * loss_E
        term_G = 0.5 * torch.exp(-s_G_cast) * loss_G
        regularization = 0.5 * (s_E_cast + s_G_cast)

        return term_E + term_G + regularization

    def compute_analytical_log_variance_gradients(
        self,
        pred_energy: torch.Tensor,
        target_energy: torch.Tensor,
        pred_gap: torch.Tensor,
        target_gap: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact analytical gradients dL/ds_E and dL/ds_G [D].

        dL/ds_E = -0.5 * exp(-s_E) * L_E + 0.5
        dL/ds_G = -0.5 * exp(-s_G) * L_G + 0.5
        """
        loss_E = huber_loss(pred_energy, target_energy, self.delta_energy)
        loss_G = huber_loss(pred_gap, target_gap, self.delta_gap)

        s_E_cast = self.s_E.to(loss_E.dtype)
        s_G_cast = self.s_G.to(loss_G.dtype)

        grad_s_E = -0.5 * torch.exp(-s_E_cast) * loss_E + 0.5
        grad_s_G = -0.5 * torch.exp(-s_G_cast) * loss_G + 0.5

        return grad_s_E, grad_s_G
