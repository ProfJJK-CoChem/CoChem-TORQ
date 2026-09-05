"""Dynamic Mixed-Precision (FP16/BF16) Training Engine (REQ-TORQ-TRAIN-096 [M]).

Features:
- Hardware-aware precision dispatching via zero-mock dependency injection:
    * Compute Capability >= 8.0 (Ampere/Hopper/Ada) -> BF16 (no loss scaling required)
    * Compute Capability < 8.0 (Volta/Turing) -> FP16 with GradScaler(init_scale=65536)
    * CPU / Apple Silicon (MPS) -> FP32 execution (scaler = None)
- Coordinate autograd graphs (forces) retained strictly in FP32/FP64 to prevent underflow.
- Explicit gradient unscaling prior to torch.nn.utils.clip_grad_norm_.
- PrecisionDivergenceError raised upon consecutive unrecoverable NaN/Inf gradients.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from Libraries.cochem_torq_gradient_checkpointing import compute_composite_loss
from Libraries.cochem_torq_training_errors import PrecisionDivergenceError
from Libraries.cochem_torq_training_schemas import TrainingDynamicsConfig


def resolve_amp_precision(
    device: torch.device,
    compute_capability: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.dtype, Optional[torch.amp.GradScaler]]:
    """Determine optimal mixed-precision configuration without mocking hardware. [D]
    
    Parameters
    ----------
    device : torch.device
        Target execution device (CPU, CUDA, MPS).
    compute_capability : Optional[Tuple[int, int]]
        CUDA compute capability tuple (major, minor). If None and device is CUDA,
        queried dynamically from hardware runtime.
        
    Returns
    -------
    Tuple[torch.dtype, Optional[torch.amp.GradScaler]]
        (precision_dtype, grad_scaler_instance_or_none)
    """
    dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    if dev_type == "cuda":
        cap = compute_capability
        if cap is None and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(device)

        if cap is not None and cap >= (8, 0):
            # Ampere, Hopper, Ada Lovelace: Native BF16 with full 8-bit dynamic exponent
            return torch.bfloat16, None
        else:
            # Volta, Turing, or legacy: FP16 requires dynamic loss scaling
            scaler = torch.amp.GradScaler("cuda", init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5)
            return torch.float16, scaler

    # Tier 1 Windows CPU, macOS MPS, or CPU environments
    return torch.float32, None


class AMPTrainer:
    """Production MLFF mixed-precision trainer managing force autograd graphs and scaling. [M]"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingDynamicsConfig,
        device: Optional[torch.device] = None,
        compute_capability: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype, self.scaler = resolve_amp_precision(self.device, compute_capability)
        self.consecutive_divergence_count = 0
        self.max_divergence_ceiling = 3

    def train_step(
        self,
        coordinates: torch.Tensor,
        species: torch.Tensor,
        ref_energy: torch.Tensor,
        ref_forces: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """Execute a single physical training step with force derivation and gradient clipping. [M]
        
        Returns
        -------
        Tuple[float, float, float]
            (total_loss, energy_loss, force_loss)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Invariant: coordinates must be floating-point (FP32 or FP64) with autograd enabled
        coords = coordinates.clone().detach().to(self.device).requires_grad_(True)
        spec = species.to(self.device)
        ref_e = ref_energy.to(self.device)
        ref_f = ref_forces.to(self.device)

        dev_type = self.device.type if hasattr(self.device, "type") else "cpu"

        # 1. Forward energy evaluation under autocast
        if dev_type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                pred_energy = self.model(coords, spec, edge_index=edge_index, batch=batch)
        else:
            pred_energy = self.model(coords, spec, edge_index=edge_index, batch=batch)

        # 2. Force derivation: spatial negative gradient in full precision
        grad_out = torch.ones_like(pred_energy)
        grad_coords = torch.autograd.grad(
            outputs=pred_energy,
            inputs=coords,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        pred_forces = -grad_coords

        # 3. Composite loss
        total_loss, e_loss, f_loss = compute_composite_loss(
            predicted_energy=pred_energy,
            reference_energy=ref_e,
            predicted_forces=pred_forces,
            reference_forces=ref_f,
            energy_loss_weight=self.config.energy_loss_weight,
            force_loss_weight=self.config.force_loss_weight,
        )

        # 4. Backward pass & scaling
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            # Explicit unscale before clipping norm [M]
            self.scaler.unscale_(self.optimizer)

            has_nan_or_inf = False
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_or_inf = True
                        break

            if has_nan_or_inf:
                self.consecutive_divergence_count += 1
                if self.consecutive_divergence_count >= self.max_divergence_ceiling:
                    raise PrecisionDivergenceError(
                        f"Unrecoverable gradient divergence detected: NaN/Inf gradients "
                        f"persisted across {self.consecutive_divergence_count} consecutive scale attempts.",
                        diagnostics={"consecutive_divergences": self.consecutive_divergence_count},
                    )
                self.scaler.step(self.optimizer)  # scaler will skip optimizer step
                self.scaler.update()
                self.optimizer.zero_grad()
                return float(total_loss.item()), float(e_loss.item()), float(f_loss.item())

            self.consecutive_divergence_count = 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_gradient_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()

            has_nan_or_inf = False
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_or_inf = True
                        break

            if has_nan_or_inf:
                self.consecutive_divergence_count += 1
                if self.consecutive_divergence_count >= self.max_divergence_ceiling:
                    raise PrecisionDivergenceError(
                        f"Unrecoverable gradient divergence detected: NaN/Inf gradients "
                        f"persisted across {self.consecutive_divergence_count} consecutive steps.",
                        diagnostics={"consecutive_divergences": self.consecutive_divergence_count},
                    )
                self.optimizer.zero_grad()
                return float(total_loss.item()), float(e_loss.item()), float(f_loss.item())

            self.consecutive_divergence_count = 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_gradient_norm)
            self.optimizer.step()

        return float(total_loss.item()), float(e_loss.item()), float(f_loss.item())
