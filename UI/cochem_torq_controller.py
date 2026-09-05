"""CoChem-TORQ Reactive Pipeline Controller.

Manages pipeline state, geometry provenance tracking (SHA-256), downstream cache
invalidation, and stage dependency gating to prevent stale coordinate execution.
Method Matrix Reference: State Persistence, Provenance Tracking, and Reproducibility Directives [M].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class PipelineState:
    molecule_name: str = ""
    geometry_hash: str = ""
    coords: Optional[np.ndarray] = None
    symbols: list[str] = field(default_factory=list)
    rotational_constants: Optional[dict[str, float]] = None
    pes_scan_completed: bool = False
    dvr_completed: bool = False
    spcat_completed: bool = False
    results_cache: dict[str, Any] = field(default_factory=dict)


class TORQPipelineController:
    """Reactive controller governing TORQ pipeline state and dependency cache invalidation."""

    def __init__(self) -> None:
        self.state = PipelineState()

    def load_preset(self, name: str, symbols: list[str], coords: np.ndarray) -> str:
        """Invalidates downstream caches and establishes new active geometry digest.
        
        Args:
            name: Molecule preset identifier.
            symbols: List of element symbols.
            coords: Array of Cartesian coordinates shape (N, 3).
            
        Returns:
            SHA-256 hex digest of the new active geometry.
        """
        arr = np.ascontiguousarray(coords, dtype=np.float64)
        geom_bytes = arr.tobytes() + "".join(symbols).encode("utf-8")
        new_hash = hashlib.sha256(geom_bytes).hexdigest()

        self.state = PipelineState(
            molecule_name=name,
            geometry_hash=new_hash,
            coords=np.copy(arr),
            symbols=list(symbols),
            rotational_constants=None,
            pes_scan_completed=False,
            dvr_completed=False,
            spcat_completed=False,
            results_cache={},
        )
        return new_hash

    def invalidate_cache(self) -> None:
        """Purges downstream calculation results and resets execution flags."""
        self.state.results_cache.clear()
        self.state.pes_scan_completed = False
        self.state.dvr_completed = False
        self.state.spcat_completed = False

    def set_rotational_constants(self, constants: dict[str, float]) -> None:
        """Records rotational constants and registers in cache."""
        self.state.rotational_constants = dict(constants)
        self.state.results_cache["rotational_constants"] = dict(constants)

    def record_pes_scan(self, pes_data: Any) -> None:
        """Records completed PES scan and marks stage completed."""
        self.state.results_cache["pes_scan"] = pes_data
        self.state.pes_scan_completed = True

    def record_dvr(self, dvr_data: Any) -> None:
        """Records completed DVR eigenvalues and wavefunctions."""
        self.state.results_cache["dvr"] = dvr_data
        self.state.dvr_completed = True

    def record_spcat(self, spcat_data: Any) -> None:
        """Records completed SPCAT line catalog."""
        self.state.results_cache["spcat"] = spcat_data
        self.state.spcat_completed = True

    def get_active_target_banner(self) -> str:
        """Generates formatted active target confirmation banner."""
        if not self.state.molecule_name:
            return "No active molecule loaded."
        hash_prefix = self.state.geometry_hash[:16] if self.state.geometry_hash else "UNHASHED"
        return f"Active Target: {self.state.molecule_name} (SHA-256: {hash_prefix}...)"
