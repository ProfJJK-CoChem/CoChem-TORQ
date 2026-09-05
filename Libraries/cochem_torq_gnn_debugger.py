"""Message-passing GNN gradient health debugger and numerical anomaly detector.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic hook mechanics, gradient norm tracking, and traps.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
import warnings
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from Libraries.cochem_torq_inference_errors import (
    GradientExplosionError,
    VanishingGradientWarning,
)
from Libraries.cochem_torq_inference_schemas import GNNGradientDebuggerConfig


class GNNGradientDebugger:
    """Non-intrusive diagnostic probe monitoring activations and gradients across GNN layers. [D]"""

    def __init__(self, config: Optional[GNNGradientDebuggerConfig] = None) -> None:
        self.config = config or GNNGradientDebuggerConfig()
        self.handles: List[RemovableHandle] = []
        self.activation_norms: Dict[str, float] = {}
        self.gradient_norms: Dict[str, float] = {}
        self.gradient_history: Dict[str, List[float]] = {}
        self.consecutive_vanishing_count: int = 0
        self.attached_model: Optional[nn.Module] = None

    def reset(self) -> None:
        """Reset internal telemetry and gradient tracking states. [M]"""
        self.activation_norms.clear()
        self.gradient_norms.clear()
        self.gradient_history.clear()
        self.consecutive_vanishing_count = 0

    def detach(self) -> None:
        """Remove all active forward and backward hooks from monitored module. [M]"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.attached_model = None

    def _forward_hook(self, name: str, module: nn.Module, inputs: Any, output: Any) -> None:
        """Forward hook tracking node activation norms and numerical stability. [D]"""
        if not self.config.enabled:
            return

        tensors = []
        if isinstance(output, torch.Tensor):
            tensors.append(output)
        elif isinstance(output, (tuple, list)):
            tensors.extend([t for t in output if isinstance(t, torch.Tensor)])

        for idx, tensor in enumerate(tensors):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise GradientExplosionError(
                    f"NaN or Inf detected in forward activations of layer '{name}' (subtensor {idx}).",
                    diagnostics={"layer": name, "subtensor_idx": idx},
                )
            norm = float(torch.norm(tensor.detach()).item())
            self.activation_norms[f"{name}.out_{idx}"] = norm

    def _backward_hook(
        self, name: str, module: nn.Module, grad_input: Any, grad_output: Any
    ) -> None:
        """Backward hook tracking layer gradient norms, vanishing trends, and explosion traps. [D]"""
        if not self.config.enabled:
            return

        tensors = []
        if isinstance(grad_output, torch.Tensor):
            tensors.append(grad_output)
        elif isinstance(grad_output, (tuple, list)):
            tensors.extend([t for t in grad_output if isinstance(t, torch.Tensor)])

        for idx, grad in enumerate(tensors):
            if grad is None:
                continue
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                raise GradientExplosionError(
                    f"NaN or Inf detected in backward gradients of layer '{name}' (grad {idx}).",
                    diagnostics={"layer": name, "grad_idx": idx},
                )
            norm = float(torch.norm(grad.detach()).item())
            key = f"{name}.grad_{idx}"
            self.gradient_norms[key] = norm
            if key not in self.gradient_history:
                self.gradient_history[key] = []
            self.gradient_history[key].append(norm)

            # Exploding gradient trap
            if norm > self.config.exploding_grad_threshold:
                raise GradientExplosionError(
                    f"Gradient norm {norm:.3e} exceeds explosion threshold "
                    f"{self.config.exploding_grad_threshold:.3e} in layer '{name}'.",
                    diagnostics={
                        "layer": name,
                        "grad_norm": norm,
                        "threshold": self.config.exploding_grad_threshold,
                    },
                )

            # Vanishing gradient warning tracker
            if norm < self.config.vanishing_grad_threshold:
                self.consecutive_vanishing_count += 1
                if self.consecutive_vanishing_count >= self.config.consecutive_vanishing_blocks:
                    warnings.warn(
                        f"GNN gradient vanishing alert: norm {norm:.3e} < {self.config.vanishing_grad_threshold:.3e} "
                        f"sustained across {self.consecutive_vanishing_count} layers in '{name}'.",
                        category=VanishingGradientWarning,
                        stacklevel=2,
                    )
            else:
                self.consecutive_vanishing_count = 0

    def _param_hook(self, param_name: str, grad: torch.Tensor) -> None:
        """Parameter tensor gradient hook for direct parameter gradient tracking. [D]"""
        if not self.config.enabled:
            return
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            raise GradientExplosionError(
                f"NaN or Inf detected in gradient of parameter '{param_name}'.",
                diagnostics={"param": param_name},
            )
        norm = float(torch.norm(grad.detach()).item())
        self.gradient_norms[f"param.{param_name}"] = norm
        if norm > self.config.exploding_grad_threshold:
            raise GradientExplosionError(
                f"Parameter gradient norm {norm:.3e} exceeds explosion threshold "
                f"{self.config.exploding_grad_threshold:.3e} in parameter '{param_name}'.",
                diagnostics={
                    "param": param_name,
                    "grad_norm": norm,
                    "threshold": self.config.exploding_grad_threshold,
                },
            )

    def attach(self, model: nn.Module) -> None:
        """Attach forward and full-backward hooks to all submodules and parameter tensors. [M]"""
        self.detach()
        self.reset()
        self.attached_model = model

        if not self.config.enabled:
            return

        for name, submodule in model.named_modules():
            # Skip the root container if it has children
            if list(submodule.children()):
                continue

            f_handle = submodule.register_forward_hook(
                lambda m, inp, out, n=name: self._forward_hook(n, m, inp, out)
            )
            self.handles.append(f_handle)

            b_handle = submodule.register_full_backward_hook(
                lambda m, gi, go, n=name: self._backward_hook(n, m, gi, go)
            )
            self.handles.append(b_handle)

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                p_handle = param.register_hook(
                    lambda grad, pn=param_name: self._param_hook(pn, grad)
                )
                self.handles.append(p_handle)

    @contextmanager
    def probe(self, model: nn.Module) -> Generator["GNNGradientDebugger", None, None]:
        """Context manager providing scoped monitoring with guaranteed cleanup upon exit. [M]"""
        self.attach(model)
        try:
            yield self
        finally:
            self.detach()

    def get_activation_norms(self) -> Dict[str, float]:
        """Return recorded forward activation norms. [M]"""
        return dict(self.activation_norms)

    def get_gradient_norms(self) -> Dict[str, float]:
        """Return recorded backward gradient norms. [M]"""
        return dict(self.gradient_norms)
