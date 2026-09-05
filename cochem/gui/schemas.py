"""
CoChem GUI Schemas and Configuration Data Contracts (REQ-MOB-006).

Strict Zero-Mock Mandate:
- Pydantic v2 immutable schemas with strict type annotations.
- Hardware-aware execution device resolution.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class QuantumAdvancedConfig(BaseModel):
    """
    Pydantic v2 Data Contract for Advanced Quantum Chemistry Execution Parameters.
    
    Enforces frozen immutability and forbids unknown fields.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    grid_level: Annotated[
        Literal[1, 2, 3, 4, 5, 6, 7],
        Field(default=3, description="ORCA DFT integration grid level"),
    ] = 3
    max_scf_cycles: Annotated[
        int,
        Field(default=100, ge=10, le=500, description="Maximum SCF convergence cycles"),
    ] = 100
    soscf_fallback: Annotated[
        bool,
        Field(default=True, description="Enable Second-Order SCF fallback upon failure"),
    ] = True
    conv_tol: Annotated[
        float,
        Field(default=1e-6, gt=0.0, le=1e-3, description="SCF energy convergence tolerance (Hartree)"),
    ] = 1e-6
    cuda_acceleration: Annotated[
        bool,
        Field(default=False, description="Request GPU/CUDA acceleration if available"),
    ] = False


class CameraState(BaseModel):
    """Immutable camera orientation and frustum state using scalar-last quaternion."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    quaternion: Annotated[Tuple[float, float, float, float], Field(default=(0.0, 0.0, 0.0, 1.0), description="[x, y, z, w] unit quaternion")]
    position: Annotated[Tuple[float, float, float], Field(default=(0.0, 0.0, 10.0))]
    target: Annotated[Tuple[float, float, float], Field(default=(0.0, 0.0, 0.0))]
    fov_degrees: Annotated[float, Field(default=45.0, gt=10.0, lt=120.0)]
    focal_distance: Annotated[float, Field(default=10.0, gt=0.0)]


class ViewportConfig(BaseModel):
    """Immutable viewport resolution and DPI configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    width_px: Annotated[int, Field(default=800, ge=200)]
    height_px: Annotated[int, Field(default=600, ge=200)]
    device_pixel_ratio: Annotated[float, Field(default=1.0, ge=0.5, le=5.0)]
    background_color: Annotated[str, Field(default="#ffffff")]


class SelectionState(BaseModel):
    """Immutable atom and dihedral selection state."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_atom_indices: Annotated[Tuple[int, ...], Field(default=())]
    active_dihedral: Annotated[Optional[Tuple[int, int, int, int]], Field(default=None)]


class VibrationTensor(BaseModel):
    """Immutable normal mode vibrational coordinate displacements."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode_index: Annotated[int, Field(ge=1)]
    frequency_cm1: Annotated[float, Field(description="Vibrational frequency in wavenumbers")]
    displacement_vectors: Annotated[Tuple[Tuple[float, float, float], ...], Field(description="Cartesian displacements per atom [N x 3]")]


def resolve_execution_device(config: QuantumAdvancedConfig) -> str:
    """
    Inspect hardware and resolve target execution device ('cuda' or 'cpu').

    Gracefully falls back to 'cpu' when CUDA hardware is absent or when
    cuda_acceleration is disabled.

    Args:
        config: Immutable quantum advanced configuration.

    Returns:
        Target device string ('cuda' or 'cpu').
    """
    if not config.cuda_acceleration:
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except (ImportError, RuntimeError, AttributeError, OSError):
        return "cpu"

    return "cpu"

