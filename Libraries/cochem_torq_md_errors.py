"""Domain exception hierarchy for TORQ Molecular Dynamics (Chunk 21).

Method Matrix v4 Provenance Tags:
- [M] Mandated: Authentic exception hierarchy, zero pass blocks, strict error codes.
- [D] Derived: Rigorous diagnostic payloads for symplectic and replica exchange state telemetry.
- [E] Empirical: Telemetry threshold bounds for numerical instability and drift tracking.

Strict Zero-Mock Mandate v3: Completely authentic physics and mathematical rigor.
No pass blocks or dead-end stubs permitted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TorqMDError(Exception):
    """Root base exception for all TORQ molecular dynamics failures [M]."""

    def __init__(
        self,
        message: str,
        error_code: str = "TORQ_MD_GENERIC_ERROR",
        component: str = "molecular_dynamics",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.diagnostics = diagnostics or {}


class EnergyDriftExceededError(TorqMDError):
    """Raised when NVE relative total energy drift |Delta E / E_0| exceeds tolerance (1e-4) [M]."""

    def __init__(
        self,
        drift: float,
        tolerance: float,
        step: int,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"NVE relative energy drift {drift:.6e} exceeded tolerance {tolerance:.6e} at step {step}.",
            error_code="ENERGY_DRIFT_EXCEEDED",
            component="symplectic_integrator",
            diagnostics=diagnostics,
        )
        self.drift = drift
        self.tolerance = tolerance
        self.step = step


class SymplecticIntegratorError(TorqMDError):
    """Raised when non-finite (NaN or Inf) values are encountered in coordinates, velocities, or forces [M]."""

    def __init__(
        self,
        tensor_name: str,
        step: int,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Non-finite values encountered in {tensor_name} at step {step}.",
            error_code="SYMPLECTIC_NON_FINITE_TENSOR",
            component="symplectic_integrator",
            diagnostics=diagnostics,
        )
        self.tensor_name = tensor_name
        self.step = step


class ReplicaExchangeDivergenceError(TorqMDError):
    """Raised when rolling swap acceptance drops below 5% or temperatures deviate by > 50 K [M]."""

    def __init__(
        self,
        reason: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Replica Exchange divergent state: {reason}",
            error_code="REMD_DIVERGENCE",
            component="remd_engine",
            diagnostics=diagnostics,
        )
        self.reason = reason


class HardwareDispatchError(TorqMDError):
    """Raised when requested hardware devices are unavailable or fail double-precision compliance [M]."""

    def __init__(
        self,
        device_requested: str,
        reason: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Hardware dispatch failed for '{device_requested}': {reason}",
            error_code="HARDWARE_DISPATCH_FAILED",
            component="hardware_dispatcher",
            diagnostics=diagnostics,
        )
        self.device_requested = device_requested
        self.reason = reason
