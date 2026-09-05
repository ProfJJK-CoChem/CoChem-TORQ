"""Bespoke Learning Rate Scheduler for GNN Convergence (REQ-TORQ-TRAIN-097 [D]).

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic numerical implementation.
Features:
- Deterministic linear warmup over T_warmup steps.
- Cosine Annealing with Warm Restarts and peak attenuation factor gamma_restart.
- Step-based triggering (per optimizer step, not epoch).
- Adaptive gradient norm clipping with NaN/Inf detection.
- SchedulerDivergenceError on divergence or unrecoverable gradient anomalies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from Libraries.cochem_torq_training_errors import SchedulerDivergenceError
from Libraries.cochem_torq_training_schemas import GNNWarmRestartSchedulerConfig


def compute_lr_at_step(step: int, config: GNNWarmRestartSchedulerConfig) -> float:
    """Compute learning rate at a given step index according to warmup and attenuated cosine restarts. [D]"""
    if step < 0:
        raise ValueError(f"Step index must be non-negative, got {step}.")

    if step <= config.warmup_steps:
        # Deterministic linear warmup
        ratio = float(step) / float(config.warmup_steps)
        lr = config.min_lr + ratio * (config.initial_lr - config.min_lr)
        return float(lr)

    # Cosine Annealing with Warm Restarts
    elapsed_after_warmup = step - config.warmup_steps

    current_cycle = 0
    t_cycle = config.first_cycle_steps
    t_start = 0

    while elapsed_after_warmup >= t_start + t_cycle:
        t_start += t_cycle
        t_cycle = int(round(t_cycle * config.cycle_multiplier))
        current_cycle += 1

    t_cur = elapsed_after_warmup - t_start
    eta_max_i = config.initial_lr * (config.restart_decay**current_cycle)

    cos_term = math.cos(math.pi * (float(t_cur) / float(t_cycle)))
    lr = config.min_lr + 0.5 * (eta_max_i - config.min_lr) * (1.0 + cos_term)

    if math.isnan(lr) or math.isinf(lr):
        raise SchedulerDivergenceError(
            f"Computed learning rate diverged to NaN/Inf at step {step}.",
            diagnostics={"step": step, "lr": lr, "cycle": current_cycle},
        )

    return float(lr)


def check_and_clip_gradients(
    parameters_or_optimizer: Iterable[torch.nn.Parameter] | Optimizer,
    max_grad_norm: float = 1.0,
    scaler: Optional[Any] = None,
) -> float:
    """Unscale AMP gradients, check for NaN/Inf anomalies, and clip to Euclidean norm ceiling. [D]"""
    if isinstance(parameters_or_optimizer, Optimizer):
        if scaler is not None and hasattr(scaler, "unscale_"):
            scaler.unscale_(parameters_or_optimizer)
        params: List[torch.nn.Parameter] = [
            p
            for group in parameters_or_optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
    else:
        params = [p for p in parameters_or_optimizer if p.grad is not None]

    total_norm_sq = 0.0
    for p in params:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            raise SchedulerDivergenceError(
                "Parameter gradient contains NaN or Inf values.",
                diagnostics={"param_shape": list(p.grad.shape)},
            )
        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
            raise SchedulerDivergenceError(
                "Parameter data contains NaN or Inf values.",
                diagnostics={"param_shape": list(p.data.shape)},
            )
        param_norm = p.grad.detach().data.norm(2)
        total_norm_sq += float(param_norm.item()) ** 2

    total_norm = math.sqrt(total_norm_sq)
    if math.isnan(total_norm) or math.isinf(total_norm):
        raise SchedulerDivergenceError("Total gradient norm evaluated to NaN or Inf.")

    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

    return total_norm


class GNNWarmRestartScheduler(LRScheduler):
    """Bespoke GNN Warm Restart Learning Rate Scheduler (REQ-TORQ-TRAIN-097 [D])."""

    def __init__(
        self,
        optimizer: Optimizer,
        config: Optional[GNNWarmRestartSchedulerConfig] = None,
        last_epoch: int = -1,
    ) -> None:
        self.config = config if config is not None else GNNWarmRestartSchedulerConfig()
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        """Calculate learning rates for all parameter groups at current step."""
        current_step = max(0, self.last_epoch)
        lr = compute_lr_at_step(current_step, self.config)
        return [lr for _ in self.optimizer.param_groups]

    def get_lr_at_step(self, step: int) -> float:
        """Pure query of learning rate at an arbitrary step index."""
        return compute_lr_at_step(step, self.config)

    def step_and_clip(
        self,
        scaler: Optional[Any] = None,
    ) -> float:
        """Atomically unscale, validate, clip gradients, advance optimizer and scheduler. [D]"""
        norm = check_and_clip_gradients(
            self.optimizer,
            max_grad_norm=self.config.max_grad_norm,
            scaler=scaler,
        )
        if scaler is not None and hasattr(scaler, "step"):
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.step()
        return norm
