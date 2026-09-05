"""Reactive State Synchronization & Molecule Re-Initialization Controller.

Method Matrix Reference: State Persistence & Concurrency Directives §8A, §8C.
Provides Pydantic v2 backed reactive controller, atomic cache purging, HDF5 SWMR
persistence, and downstream execution gating via StateDesynchronizationError.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import filelock
import h5py
import numpy as np
from pydantic import BaseModel, Field


class StateDesynchronizationError(RuntimeError):
    """Raised when downstream cells execute with stale or uninitialized state."""


class TORQStateModel(BaseModel):
    """Pydantic v2 reactive pipeline state model."""

    model_config = {"extra": "allow"}

    molecule_name: str = Field(default="", description="Active molecule preset name")
    geometry_hash: str = Field(default="", description="SHA-256 digest")
    symbols: list[str] = Field(default_factory=list, description="Atomic symbols")
    coords: list[list[float]] | None = Field(
        default=None, description="Cartesian coordinates"
    )
    rotational_constants: dict[str, float] | None = Field(
        default=None, description="Calculated rotational constants in MHz"
    )
    pes_scan_completed: bool = Field(default=False, description="PES scan status")
    dvr_completed: bool = Field(default=False, description="DVR status")
    spcat_completed: bool = Field(default=False, description="SPCAT status")
    is_initialized: bool = Field(default=False, description="Initialization status")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="State snapshot UTC timestamp",
    )


class TORQPipelineController:
    """Reactive controller governing TORQ pipeline state and execution gating."""

    def __init__(self, scratch_dir: Path | str | None = None) -> None:
        self.state = TORQStateModel()
        self.results_cache: dict[str, Any] = {}
        scr_env = os.environ.get("COCH_SCRATCH", "scratch")
        self.scratch_dir = Path(scratch_dir or scr_env).resolve()

    def reinitialize_molecule(
        self,
        name: str,
        symbols: Sequence[str],
        coords: np.ndarray | Sequence[Sequence[float]],
        h5_store_path: Path | str | None = None,
    ) -> str:
        """Executes a clean reactive state reset and persists to HDF5 PESStore."""
        # 1. Purge downstream memory cache and calculation flags
        self.results_cache.clear()

        # 2. Clear stale temporary files in scratch T_scr
        if self.scratch_dir.exists():
            for child in self.scratch_dir.iterdir():
                if child.is_dir() and child.name.startswith("torq_job_"):
                    shutil.rmtree(child, ignore_errors=True)

        # 3. Compute SHA-256 digest of new active geometry
        coords_arr = np.ascontiguousarray(coords, dtype=np.float64)
        sym_list = [str(s).strip().capitalize() for s in symbols]
        geom_bytes = coords_arr.tobytes() + "".join(sym_list).encode("utf-8")
        new_hash = hashlib.sha256(geom_bytes).hexdigest()

        # 4. Construct initialized Pydantic state model
        self.state = TORQStateModel(
            molecule_name=name,
            geometry_hash=new_hash,
            symbols=sym_list,
            coords=coords_arr.tolist(),
            rotational_constants=None,
            pes_scan_completed=False,
            dvr_completed=False,
            spcat_completed=False,
            is_initialized=True,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        # 5. Atomically persist active state to Thread-Safe HDF5 PESStore
        if h5_store_path:
            p = Path(h5_store_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            lock_file = p.with_suffix(".h5.lock")
            with filelock.FileLock(lock_file, timeout=15.0):
                mode = "r+" if p.exists() else "w"
                with h5py.File(p, mode, libver="latest") as h5f:
                    grp_name = "active_state"
                    if grp_name in h5f:
                        del h5f[grp_name]
                    grp = h5f.create_group(grp_name)
                    grp.attrs["molecule_name"] = name
                    grp.attrs["geometry_hash"] = new_hash
                    grp.attrs["state_json"] = self.state.model_dump_json()
                    grp.create_dataset("coordinates", data=coords_arr)

        return new_hash

    def verify_stage_prerequisites(self, stage: str) -> None:
        """Enforces downstream execution gating to prevent running with stale state."""
        if not self.state.is_initialized:
            raise StateDesynchronizationError(
                "Molecule state is uninitialized or desynchronized. Please click "
                "'Load & Re-Initialize Molecule' before executing downstream cells."
            )

        stage_lower = stage.lower()
        if stage_lower in ("dvr", "phase5", "tunneling"):
            if not self.state.pes_scan_completed:
                raise StateDesynchronizationError(
                    "Torsional scan has not completed. Execute Phase 4 PES Scan "
                    "before running DVR."
                )

        if stage_lower in ("spcat", "phase10", "catalog"):
            if not self.state.rotational_constants:
                raise StateDesynchronizationError(
                    "Rotational constants have not been computed. Execute Phase 2 "
                    "Geometry/VPT2 before compiling spectroscopic line catalog."
                )

    def set_rotational_constants(self, constants: dict[str, float]) -> None:
        """Records rotational constants and registers in cache."""
        self.state.rotational_constants = dict(constants)
        self.results_cache["rotational_constants"] = dict(constants)

    def record_pes_scan(self, pes_data: Any) -> None:
        """Records completed PES scan and marks stage completed."""
        self.state.pes_scan_completed = True
        self.results_cache["pes_scan"] = pes_data

    def record_dvr(self, dvr_data: Any) -> None:
        """Records completed DVR eigenvalues and wavefunctions."""
        self.state.dvr_completed = True
        self.results_cache["dvr"] = dvr_data

    def record_spcat(self, spcat_data: Any) -> None:
        """Records completed SPCAT line catalog."""
        self.state.spcat_completed = True
        self.results_cache["spcat"] = spcat_data

    def get_active_target_banner(self) -> str:
        """Generates formatted active target confirmation banner."""
        if not self.state.is_initialized or not self.state.molecule_name:
            return "No active molecule loaded. Click 'Load & Re-Initialize Molecule'."
        hash_prefix = self.state.geometry_hash[:16]
        return f"Active Target: {self.state.molecule_name} (SHA-256: {hash_prefix}...)"
