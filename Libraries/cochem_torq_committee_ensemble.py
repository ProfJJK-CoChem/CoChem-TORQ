"""Committee inference and model disagreement, without error calibration claims.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
M-1 sample dispersion is descriptive ensemble disagreement. Correlated model
predictions do not establish error probabilities, calibrated uncertainty, or
coverage guarantees. Memory queries retain their actual device and provider.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

try:
    from cochem_base.schemas import CommitteeEnsembleConfig
except ImportError:
    from pydantic import BaseModel, ConfigDict, Field

    class CommitteeEnsembleConfig(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        vectorized: bool = True
        vram_headroom_threshold_mb: float = Field(default=2048.0, ge=512.0)
        concurrency_mode: Literal["vmap", "cuda_streams", "serial"] = "vmap"
        max_batch_size: int = Field(default=128, ge=1)


try:
    from Libraries.cochem_torq_inference_errors import EnsembleConsensusError
except ImportError:

    class EnsembleConsensusError(RuntimeError):
        """Raised when ensemble prediction violates consensus or dimensions mismatch."""

        def __init__(self, msg: str, diagnostics: dict[str, Any] | None = None):
            super().__init__(msg)
            self.diagnostics = diagnostics or {}


logger = logging.getLogger("cochem.torq.committee_ensemble")


@dataclass(frozen=True)
class CommitteePrediction:
    """Frozen result bindings with uncalibrated model-disagreement metadata.

    Stored Torch tensors retain their normal mutability/autograd behavior.
    """

    mean: torch.Tensor
    std: torch.Tensor
    variance: torch.Tensor
    predictions: torch.Tensor
    mean_energy: torch.Tensor
    mean_forces: torch.Tensor | None = None
    energy_variance: torch.Tensor | None = None
    per_atom_force_variance: torch.Tensor | None = None
    max_force_std: float | None = None
    model_energies: torch.Tensor | None = None
    model_forces: torch.Tensor | None = None
    uncertainty_kind: Literal["ensemble_disagreement"] = field(
        default="ensemble_disagreement", init=False
    )
    calibration_status: Literal["not_established"] = field(
        default="not_established", init=False
    )

    def __iter__(self):
        """Support standard unpacking: mean, std = ensemble(x)."""
        return iter((self.mean, self.std))


@dataclass(frozen=True)
class DeviceMemoryTelemetry:
    """Actual query result scoped to its device and memory definition."""

    device: str
    memory_kind: str
    query_provider: str
    available_mb: float

    def __post_init__(self):
        expected = {
            "cpu": ("system_ram", "psutil.virtual_memory"),
            "cuda": ("gpu_vram", "torch.cuda.mem_get_info"),
        }.get(torch.device(self.device).type)
        if expected is None or expected != (self.memory_kind, self.query_provider):
            raise EnsembleConsensusError(
                "Unsupported or mismatched memory telemetry profile."
            )
        if (
            not isinstance(self.available_mb, Real)
            or isinstance(self.available_mb, bool)
            or not math.isfinite(self.available_mb)
            or self.available_mb < 0
        ):
            raise EnsembleConsensusError(
                "Available memory must be finite and nonnegative."
            )


def get_device_memory_telemetry(device: str | torch.device) -> DeviceMemoryTelemetry:
    """Query CPU system RAM or the specified CUDA device; never substitute one."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise EnsembleConsensusError(
                "CUDA memory telemetry is unavailable: no CUDA device."
            )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception as err:
            raise EnsembleConsensusError(
                "CUDA memory query failed; GPU capacity is unavailable."
            ) from err
        return DeviceMemoryTelemetry(
            str(device),
            "gpu_vram",
            "torch.cuda.mem_get_info",
            float(free_bytes) / (1024.0 * 1024.0),
        )
    if device.type == "cpu":
        try:
            import psutil

            free_bytes = psutil.virtual_memory().available
        except Exception as err:
            raise EnsembleConsensusError(
                "CPU system-memory query failed; capacity is unavailable."
            ) from err
        return DeviceMemoryTelemetry(
            "cpu",
            "system_ram",
            "psutil.virtual_memory",
            float(free_bytes) / (1024.0 * 1024.0),
        )
    raise EnsembleConsensusError(
        f"Memory telemetry is unsupported for device profile {device.type!r}."
    )


def get_available_vram_mb() -> float:
    """Query actual CUDA VRAM; unavailable CUDA/query results raise explicitly."""
    return get_device_memory_telemetry("cuda").available_mb


def compute_committee_moments(
    energies: torch.Tensor,
    forces: torch.Tensor | None = None,
) -> CommitteePrediction:
    """Compute M-1 sample dispersion of finite real committee predictions. [D]

    energies: (M, ...) e.g. (M, B) or (M,)
    forces: (M, ..., N, 3) or None

    Energy variance is sum_m (E_m - mean_E)^2 / (M-1). Per-atom
    force variance averages squared vector disagreement over three components;
    max_force_std instead uses its vector norm, without the component average.
    None of these statistics establishes calibrated prediction error or coverage.
    """

    def require_finite_real(value: torch.Tensor, name: str) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or not value.numel()
            or not torch.isfinite(value).all().item()
        ):
            raise EnsembleConsensusError(
                f"{name} must be a nonempty finite real floating-point tensor."
            )

    require_finite_real(energies, "Committee energies")
    if energies.ndim == 0:
        raise EnsembleConsensusError("Committee energies require a model dimension.")
    m = energies.shape[0]
    if m < 2:
        raise EnsembleConsensusError(
            f"Committee ensemble requires at least 2 models for variance estimation, got M={m}.",
            diagnostics={"num_models": m},
        )

    # Descriptive M-1 sample dispersion; models need not be independent draws.
    mean_e = torch.mean(energies, dim=0)
    e_diff = energies - mean_e.unsqueeze(0)
    e_var = torch.sum(e_diff**2, dim=0) / float(m - 1)
    require_finite_real(mean_e, "Computed energy mean")
    require_finite_real(e_var, "Computed energy dispersion")
    if torch.any(e_var < 0).item():
        raise EnsembleConsensusError("Computed energy dispersion cannot be negative.")
    e_std = torch.sqrt(e_var)
    require_finite_real(e_std, "Computed energy standard deviation")

    if forces is not None:
        require_finite_real(forces, "Committee forces")
        if forces.ndim < 3 or forces.shape[-1] != 3:
            raise EnsembleConsensusError(
                "Committee forces require shape (M, ..., N, 3)."
            )
        if forces.shape[0] != m:
            raise EnsembleConsensusError(
                f"Mismatched ensemble model count: energies has M={m}, forces has M={forces.shape[0]}.",
                diagnostics={"energies_M": m, "forces_M": forces.shape[0]},
            )
        mean_f = torch.mean(forces, dim=0)
        f_diff = forces - mean_f.unsqueeze(0)
        f_sq_norm = torch.sum(f_diff**2, dim=-1)
        atom_f_var = torch.sum(f_sq_norm, dim=0) / float(3.0 * (m - 1))
        atom_f_sum = torch.sum(f_sq_norm, dim=0) / float(m - 1)
        require_finite_real(mean_f, "Computed force mean")
        require_finite_real(atom_f_var, "Computed component-averaged force dispersion")
        require_finite_real(atom_f_sum, "Computed vector force dispersion")
        if torch.any(atom_f_sum < 0).item() or torch.any(atom_f_var < 0).item():
            raise EnsembleConsensusError(
                "Computed force dispersion cannot be negative."
            )
        atom_f_std = torch.sqrt(atom_f_sum)
        require_finite_real(atom_f_std, "Computed vector force standard deviation")
        max_f_std = float(torch.max(atom_f_std).item())

        return CommitteePrediction(
            mean=mean_e,
            std=e_std,
            variance=e_var,
            predictions=energies,
            mean_energy=mean_e,
            mean_forces=mean_f,
            energy_variance=e_var,
            per_atom_force_variance=atom_f_var,
            max_force_std=max_f_std,
            model_energies=energies,
            model_forces=forces,
        )

    return CommitteePrediction(
        mean=mean_e,
        std=e_std,
        variance=e_var,
        predictions=energies,
        mean_energy=mean_e,
        mean_forces=None,
        energy_variance=e_var,
        per_atom_force_variance=None,
        max_force_std=None,
        model_energies=energies,
        model_forces=None,
    )


class CommitteeEnsemble(nn.Module):
    """Vectorized ensemble coordinator for M neural network models with VRAM safeguards. [M]/[D]"""

    def __init__(
        self,
        models: Sequence[nn.Module],
        config: CommitteeEnsembleConfig | None = None,
    ) -> None:
        super().__init__()
        if len(models) < 2:
            raise EnsembleConsensusError(
                f"Committee requires at least 2 models, got {len(models)}.",
                diagnostics={"num_models": len(models)},
            )
        self.models = nn.ModuleList(models)
        self.config = config or CommitteeEnsembleConfig()
        self.num_models = len(models)
        self.last_execution_mode: str = "serial"
        self.last_memory_telemetry: DeviceMemoryTelemetry | None = None

    def _execution_device(self, args: Any, kwargs: Any) -> torch.device:
        """Determine the actual common device of model and input tensors."""
        devices = set()

        def observe(value):
            if isinstance(value, torch.Tensor):
                devices.add(
                    value.device if value.device.type != "cpu" else torch.device("cpu")
                )
            elif isinstance(value, dict):
                for item in value.values():
                    observe(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    observe(item)

        for model in self.models:
            observe(tuple(model.parameters()))
            observe(tuple(model.buffers()))
        observe(args)
        observe(kwargs)
        if len(devices) > 1:
            raise EnsembleConsensusError(
                "Mixed device profiles cannot share one memory-capacity query."
            )
        return next(iter(devices), torch.device("cpu"))

    def _check_homogeneous_architectures(self) -> bool:
        """Check if all models share identical parameter names and tensor shapes."""
        base_params = dict(self.models[0].named_parameters())
        base_keys = list(base_params.keys())

        for m in self.models[1:]:
            m_params = dict(m.named_parameters())
            if list(m_params.keys()) != base_keys:
                return False
            for k in base_keys:
                if m_params[k].shape != base_params[k].shape:
                    return False
        return True

    def forward(self, *args: Any, **kwargs: Any) -> CommitteePrediction:
        """Evaluate ensemble forward pass using vectorized vmap, parallel CUDA streams, or serial fallback. [M]"""
        requested_mode = (
            self.config.concurrency_mode if self.config.vectorized else "serial"
        )

        # The query definition follows the actual execution device.
        self.last_memory_telemetry = None
        device = self._execution_device(args, kwargs)
        if requested_mode == "cuda_streams" and device.type != "cuda":
            raise EnsembleConsensusError(
                "CUDA streams require a CUDA execution profile."
            )
        self.last_memory_telemetry = get_device_memory_telemetry(device)
        free_mb = self.last_memory_telemetry.available_mb
        if free_mb < self.config.vram_headroom_threshold_mb:
            logger.warning(
                "RESOURCE_GUARD: Available memory (%.2f MB) is below threshold (%.2f MB). "
                "Throttling inference to sequential fallback to prevent OOM.",
                free_mb,
                self.config.vram_headroom_threshold_mb,
            )
            mode = "serial"
        else:
            mode = requested_mode

        if mode == "vmap":
            if not self._check_homogeneous_architectures():
                logger.info(
                    "Ensemble models have heterogeneous architectures; falling back from vmap."
                )
                mode = "cuda_streams" if device.type == "cuda" else "serial"

        self.last_execution_mode = mode

        if mode == "vmap":
            return self._forward_vmap(*args, **kwargs)
        elif mode == "cuda_streams" and device.type == "cuda":
            return self._forward_cuda_streams(*args, **kwargs)
        else:
            return self._forward_serial(*args, **kwargs)

    def _forward_vmap(self, *args: Any, **kwargs: Any) -> CommitteePrediction:
        """Vectorized forward execution across committee models via torch.vmap. [D]"""
        param_keys = list(dict(self.models[0].named_parameters()).keys())
        stacked_params = {
            k: torch.stack([dict(m.named_parameters())[k] for m in self.models], dim=0)
            for k in param_keys
        }
        buffer_keys = list(dict(self.models[0].named_buffers()).keys())
        stacked_buffers = {
            k: torch.stack([dict(m.named_buffers())[k] for m in self.models], dim=0)
            for k in buffer_keys
        }

        def _single_forward(
            p_dict: dict[str, torch.Tensor], b_dict: dict[str, torch.Tensor], *a: Any
        ) -> Any:
            return functional_call(self.models[0], (p_dict, b_dict), a, kwargs)

        in_dims = (0, 0, *(None for _ in args))
        vmapped_fn = vmap(_single_forward, in_dims=in_dims)
        raw_outputs = vmapped_fn(stacked_params, stacked_buffers, *args)

        if isinstance(raw_outputs, tuple):
            if len(raw_outputs) >= 2:
                return compute_committee_moments(raw_outputs[0], raw_outputs[1])
            return compute_committee_moments(raw_outputs[0])
        elif isinstance(raw_outputs, dict):
            return compute_committee_moments(
                raw_outputs["energy"], raw_outputs.get("forces")
            )
        else:
            return compute_committee_moments(raw_outputs)

    def _forward_cuda_streams(self, *args: Any, **kwargs: Any) -> CommitteePrediction:
        """Parallel forward execution across asynchronous CUDA streams. [D]"""
        streams = [torch.cuda.Stream() for _ in range(self.num_models)]
        raw_results: list[Any] = [None] * self.num_models

        for idx, (m, s) in enumerate(zip(self.models, streams)):
            with torch.cuda.stream(s):
                raw_results[idx] = m(*args, **kwargs)

        torch.cuda.synchronize()
        return self._aggregate_raw_results(raw_results)

    def _forward_serial(self, *args: Any, **kwargs: Any) -> CommitteePrediction:
        """Sequential single-threaded forward loop fallback. [M]"""
        raw_results = [m(*args, **kwargs) for m in self.models]
        return self._aggregate_raw_results(raw_results)

    def _aggregate_raw_results(self, raw_results: Sequence[Any]) -> CommitteePrediction:
        """Aggregate raw list of outputs from independent model invocations into moments."""
        first = raw_results[0]
        if isinstance(first, tuple):
            all_e = torch.stack([r[0] for r in raw_results], dim=0)
            all_f = (
                torch.stack([r[1] for r in raw_results], dim=0)
                if len(first) > 1
                else None
            )
            return compute_committee_moments(all_e, all_f)
        elif isinstance(first, dict):
            all_e = torch.stack([r["energy"] for r in raw_results], dim=0)
            all_f = (
                torch.stack([r["forces"] for r in raw_results], dim=0)
                if "forces" in first
                else None
            )
            return compute_committee_moments(all_e, all_f)
        else:
            all_e = torch.stack(list(raw_results), dim=0)
            return compute_committee_moments(all_e)

    def predict(self, *args: Any, **kwargs: Any) -> CommitteePrediction:
        """Alias for forward evaluation. [M]"""
        return self.forward(*args, **kwargs)
