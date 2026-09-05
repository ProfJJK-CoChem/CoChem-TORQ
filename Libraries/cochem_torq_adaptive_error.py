"""
CoChem-TORQ: Adaptive Delta-Learning Escalator & Error-Driven Multi-Tier Refinement
==================================================================================
Phase 5 (Stage 4.0) Implementation
----------------------------------
Governs the error-driven multi-fidelity escalation core for torsional potential
energy surfaces (PES). Compares sparse low-tier and mid-tier electronic energies,
isolates energetic residual spikes (Delta E > 0.5 kcal/mol chemical accuracy),
demarcates surrounding topological basins for high-tier escalation (DLPNO-CCSD(T)),
and reconstructs the continuous composite PES using boundary-anchored cubic
splines and 2D tensor splines.

Strict Zero-Mock Mandate & Authentic Physics:
- Exact residual error calculus: Delta E(q) = E_mid(q) - E_low(q)
- Additive delta-learning surface synthesis: E_comp(q) = E_low(q) + S(Delta E(q))
- Dynamic atomic property retrieval via the Mendeleev library.
- Tripartite Filesystem Air-Gap compliance (Domain A Static, B Scratch, C Artifacts).
- PyArrow zero-copy tabular serialization for inter-module IPC.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import psutil  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from mendeleev import element as mendeleev_element  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from scipy.interpolate import (  # type: ignore[import-untyped]
    CubicSpline,
    RBFInterpolator,
)

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: [CoChem-TORQ-AdaptiveError] %(message)s",
)
logger = logging.getLogger("CoChem-TORQ.AdaptiveError")

# Physical & Conversion Constants
HARTREE_TO_KCAL_PER_MOL: Final[float] = 627.5094740631
EV_TO_KCAL_PER_MOL: Final[float] = 23.06054801
KJ_TO_KCAL_PER_MOL: Final[float] = 1.0 / 4.184
DEFAULT_CHEMICAL_ACCURACY_THRESHOLD_KCAL: Final[float] = 0.5


# =============================================================================
# 1. Air-Gap Architecture & Path Resolution
# =============================================================================


class AirGapViolationError(PermissionError):
    """Raised when an operation attempts to write to Ring 1 static repository space."""

    pass


def get_repo_root() -> Path:
    """Locates the Domain A / Ring 1 immutable Git repository root."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent.resolve()

    env_val = os.environ.get("COCHEM_REPO_DIR")
    if env_val:
        repo_path = Path(env_val).resolve()
        if repo_path.is_dir():
            return repo_path

    return Path.cwd().resolve()


def get_artifacts_dir() -> Path:
    """Returns the dynamically resolved artifacts output directory (Ring 3)."""
    env_dir = os.environ.get("COCHEM_ARTIFACTS_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
    else:
        p = Path.home() / "cochem_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_scratch_dir() -> Path:
    """Returns the dynamically resolved ephemeral scratch directory (Ring 2)."""
    env_dir = os.environ.get("COCHEM_SCRATCH_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
    else:
        temp_base = os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp"))
        p = Path(temp_base) / "cochem_scratch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_air_gap_path(target_path: Path | str) -> Path:
    """Validates that target_path does not reside inside the repository root."""
    path_resolved = Path(target_path).resolve()
    repo_root = get_repo_root()

    try:
        path_resolved.relative_to(repo_root)
        # If relative does not raise, target_path is inside repo_root
        raise AirGapViolationError(
            f"Air-Gap Violation: Attempted write operation to repo '{path_resolved}'."
        )
    except ValueError:
        # Not relative to repo_root -> path is valid
        pass

    return path_resolved


# =============================================================================
# 2. Enums, Models & Data Contracts
# =============================================================================


class EnergyUnit(str, enum.Enum):
    """Energy units supported for quantum potential energy surfaces."""

    KCAL_PER_MOL = "kcal/mol"
    HARTREE = "Hartree"
    EV = "eV"
    KJ_PER_MOL = "kJ/mol"


class AnchorPointType(str, enum.Enum):
    """Topographic classification of stationary and anchor nodes."""

    MINIMUM = "MINIMUM"
    TRANSITION_STATE = "TRANSITION_STATE"
    INFLECTION = "INFLECTION"
    GRID_NODE = "GRID_NODE"


class AnchorPoint(BaseModel):
    """Represents a discrete anchor point along torsional coordinates."""

    point_id: str
    coordinates: list[float] = Field(
        ..., description="Torsional coordinate(s) in degrees"
    )
    point_type: AnchorPointType = Field(default=AnchorPointType.GRID_NODE)
    energy_low: float = Field(..., description="Low-tier energy (kcal/mol)")
    energy_mid: float | None = Field(
        default=None, description="Mid-tier energy (kcal/mol)"
    )
    energy_high: float | None = Field(
        default=None, description="High-tier energy (kcal/mol)"
    )
    residual_delta: float | None = Field(
        default=None, description="Energetic residual Delta E (kcal/mol)"
    )
    requires_escalation: bool = Field(default=False)
    escalation_tier: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopographicBasin(BaseModel):
    """Demarcates a spatial basin surrounding a high-residual anchor point."""

    basin_id: str
    anchor_point: AnchorPoint
    basin_center: list[float]
    coordinate_bounds: list[tuple[float, float]] = Field(
        ..., description="Lower and upper bounds for each coordinate dimension"
    )
    max_residual_delta: float
    points_in_basin: list[list[float]] = Field(default_factory=list)
    target_theory_level: str = Field(default="DLPNO-CCSD(T)")
    is_escalated: bool = Field(default=True)


class ResidualDeltaResult(BaseModel):
    """Summary of residual delta evaluation across coordinates."""

    num_points: int
    max_delta_kcal_mol: float
    mean_delta_kcal_mol: float
    rmse_delta_kcal_mol: float
    exceeds_chemical_accuracy: bool
    threshold_kcal_mol: float = Field(default=DEFAULT_CHEMICAL_ACCURACY_THRESHOLD_KCAL)
    residuals: list[float]
    coordinates: list[list[float]]
    provenance_hash: str


class EscalationManifest(BaseModel):
    """Execution manifest routing regions to high-tier quantum engines."""

    manifest_id: str
    total_anchors_evaluated: int
    escalated_anchors_count: int
    chemical_accuracy_threshold_kcal_mol: float = Field(
        default=DEFAULT_CHEMICAL_ACCURACY_THRESHOLD_KCAL
    )
    escalated_points: list[AnchorPoint] = Field(default_factory=list)
    topographic_basins: list[TopographicBasin] = Field(default_factory=list)
    target_quantum_tier: str = Field(default="DLPNO-CCSD(T)")
    provenance_hash: str
    created_at_utc: str


class DeltaSplineSurface(BaseModel):
    """Continuous additive delta-learning composite potential energy surface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimensions: int
    is_periodic: bool
    period: float = Field(default=360.0)
    knot_coordinates: list[list[float]]
    knot_deltas: list[float]
    dense_grid_points: list[list[float]]
    dense_low_tier_energies: list[float]
    dense_composite_energies: list[float]
    spline_correction_energies: list[float]
    spline_object: Any | None = Field(default=None)
    provenance_hash: str


# =============================================================================
# 3. Energy Conversion & Dynamic Mendeleev Mass Retrieval
# =============================================================================


def convert_energy(
    value: float | np.ndarray,
    from_unit: EnergyUnit | str,
    to_unit: EnergyUnit | str = EnergyUnit.KCAL_PER_MOL,
) -> float | np.ndarray:
    """Converts energy values between Hartree, eV, kJ/mol, and kcal/mol."""
    from_u = EnergyUnit(from_unit)
    to_u = EnergyUnit(to_unit)

    if from_u == to_u:
        return value

    # Convert from source to kcal/mol
    if from_u == EnergyUnit.KCAL_PER_MOL:
        val_kcal = value
    elif from_u == EnergyUnit.HARTREE:
        val_kcal = value * HARTREE_TO_KCAL_PER_MOL
    elif from_u == EnergyUnit.EV:
        val_kcal = value * EV_TO_KCAL_PER_MOL
    elif from_u == EnergyUnit.KJ_PER_MOL:
        val_kcal = value * KJ_TO_KCAL_PER_MOL
    else:
        raise ValueError(f"Unsupported source energy unit: {from_u}")

    # Convert from kcal/mol to target
    if to_u == EnergyUnit.KCAL_PER_MOL:
        return val_kcal
    elif to_u == EnergyUnit.HARTREE:
        return val_kcal / HARTREE_TO_KCAL_PER_MOL
    elif to_u == EnergyUnit.EV:
        return val_kcal / EV_TO_KCAL_PER_MOL
    elif to_u == EnergyUnit.KJ_PER_MOL:
        return val_kcal / KJ_TO_KCAL_PER_MOL
    else:
        raise ValueError(f"Unsupported target energy unit: {to_u}")


def get_mendeleev_element_data(
    symbol_or_atomic_number: str | int,
) -> dict[str, Any]:
    """Retrieves atomic and isotopic physical data via Mendeleev library."""
    el = mendeleev_element(symbol_or_atomic_number)
    cov_rad = float(el.covalent_radius_pyykko) if el.covalent_radius_pyykko else None
    vdw_rad = float(el.vdw_radius) if el.vdw_radius else None

    return {
        "symbol": el.symbol,
        "name": el.name,
        "atomic_number": el.atomic_number,
        "atomic_weight": float(el.atomic_weight),
        "mass_number": el.mass_number,
        "covalent_radius_pyykko": cov_rad,
        "vdw_radius": vdw_rad,
    }


def get_system_telemetry() -> dict[str, Any]:
    """Collects real-time host telemetry metrics using psutil."""
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=0.01)),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024**3), 3),
        "ram_used_gb": round(vm.used / (1024**3), 3),
        "ram_available_gb": round(vm.available / (1024**3), 3),
        "ram_percent": float(vm.percent),
    }


# =============================================================================
# 4. Core Functionality: Residual Delta Evaluation
# =============================================================================


def compute_residual_delta(
    low_tier_energies: Sequence[float] | np.ndarray,
    mid_or_high_tier_energies: Sequence[float] | np.ndarray,
    coordinates: Sequence[Sequence[float] | float] | np.ndarray | None = None,
    input_units: EnergyUnit | str = EnergyUnit.KCAL_PER_MOL,
    threshold_kcal_mol: float = DEFAULT_CHEMICAL_ACCURACY_THRESHOLD_KCAL,
) -> ResidualDeltaResult:
    """Evaluates energetic residuals Delta E = E_mid - E_low between grids.

    :param low_tier_energies: Array of low-tier electronic energies.
    :param mid_or_high_tier_energies: Array of mid/high-tier electronic energies.
    :param coordinates: Optional spatial coordinates for each point.
    :param input_units: Energy units of the supplied inputs.
    :param threshold_kcal_mol: Threshold to flag escalation (default: 0.5).
    :return: ResidualDeltaResult with residuals, statistics, and hash.
    """
    e_low = np.asarray(low_tier_energies, dtype=np.float64)
    e_mid = np.asarray(mid_or_high_tier_energies, dtype=np.float64)

    if e_low.size == 0 or e_mid.size == 0:
        raise ValueError("Energy arrays cannot be empty.")
    if e_low.shape != e_mid.shape:
        raise ValueError(
            f"Mismatch between low-tier ({e_low.shape}) and mid-tier ({e_mid.shape})."
        )

    # Convert to kcal/mol
    e_low_kcal = np.asarray(
        convert_energy(e_low, input_units, EnergyUnit.KCAL_PER_MOL),
        dtype=np.float64,
    )
    e_mid_kcal = np.asarray(
        convert_energy(e_mid, input_units, EnergyUnit.KCAL_PER_MOL),
        dtype=np.float64,
    )

    # Compute raw residuals: Delta E = E_mid - E_low
    residuals = e_mid_kcal - e_low_kcal
    abs_residuals = np.abs(residuals)

    max_delta = float(np.max(abs_residuals))
    mean_delta = float(np.mean(abs_residuals))
    rmse_delta = float(np.sqrt(np.mean(residuals**2)))
    exceeds_threshold = bool(max_delta > threshold_kcal_mol)

    # Format coordinates
    n_pts = len(residuals)
    if coordinates is not None:
        coords_raw = np.asarray(coordinates, dtype=np.float64)
        if len(coords_raw) != n_pts:
            raise ValueError(
                f"Mismatch between energies ({n_pts}) and coordinates ({len(coords_raw)})."
            )
        if coords_raw.ndim == 1:
            coords_list = [[float(c)] for c in coords_raw]
        else:
            coords_list = [c.tolist() for c in coords_raw]
    else:
        coords_list = [[float(i)] for i in range(n_pts)]

    # Compute SHA-256 cryptographic provenance hash
    payload_repr = {
        "num_points": n_pts,
        "max_delta": round(max_delta, 8),
        "mean_delta": round(mean_delta, 8),
        "rmse_delta": round(rmse_delta, 8),
        "residuals": [round(float(r), 8) for r in residuals],
    }
    provenance_hash = hashlib.sha256(
        json.dumps(payload_repr, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return ResidualDeltaResult(
        num_points=n_pts,
        max_delta_kcal_mol=max_delta,
        mean_delta_kcal_mol=mean_delta,
        rmse_delta_kcal_mol=rmse_delta,
        exceeds_chemical_accuracy=exceeds_threshold,
        threshold_kcal_mol=threshold_kcal_mol,
        residuals=[float(r) for r in residuals],
        coordinates=coords_list,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# 5. Core Functionality: Topographic Basin Escalator
# =============================================================================


def escalate_topographic_basins(
    anchor_points: Sequence[AnchorPoint | dict[str, Any]],
    threshold_kcal_mol: float = DEFAULT_CHEMICAL_ACCURACY_THRESHOLD_KCAL,
    basin_half_width_deg: float = 15.0,
    target_tier: str = "DLPNO-CCSD(T)",
    dense_grid_coords: Sequence[Sequence[float] | float] | np.ndarray | None = None,
) -> EscalationManifest:
    """Identifies anchor points exceeding threshold and routes them to higher tier.

    :param anchor_points: Sequence of AnchorPoint objects or dictionaries.
    :param threshold_kcal_mol: Threshold for escalation (default: 0.5).
    :param basin_half_width_deg: Coordinate width defining localized basin.
    :param target_tier: Target quantum chemistry level of theory.
    :param dense_grid_coords: Optional dense coordinate grid.
    :return: EscalationManifest containing escalated points and basins.
    """
    validated_anchors: list[AnchorPoint] = []
    for a in anchor_points:
        if isinstance(a, AnchorPoint):
            validated_anchors.append(a)
        elif isinstance(a, dict):
            validated_anchors.append(AnchorPoint(**a))
        else:
            raise TypeError(f"Invalid anchor point type: {type(a)}")

    escalated_points: list[AnchorPoint] = []
    basins: list[TopographicBasin] = []

    dense_arr: np.ndarray | None = None
    if dense_grid_coords is not None:
        dense_arr = np.asarray(dense_grid_coords, dtype=np.float64)
        if dense_arr.ndim == 1:
            dense_arr = dense_arr[:, np.newaxis]

    for idx, anchor in enumerate(validated_anchors):
        residual = anchor.residual_delta
        if (
            residual is None
            and anchor.energy_mid is not None
            and anchor.energy_low is not None
        ):
            residual = anchor.energy_mid - anchor.energy_low
            anchor.residual_delta = residual

        if residual is not None and abs(residual) > threshold_kcal_mol:
            anchor.requires_escalation = True
            anchor.escalation_tier = target_tier
            escalated_points.append(anchor)

            coord_bounds: list[tuple[float, float]] = []
            for c in anchor.coordinates:
                c_min = float(c - basin_half_width_deg)
                c_max = float(c + basin_half_width_deg)
                coord_bounds.append((c_min, c_max))

            points_in_basin: list[list[float]] = []
            if dense_arr is not None:
                in_basin_mask = np.ones(len(dense_arr), dtype=bool)
                for dim_idx, (b_min, b_max) in enumerate(coord_bounds):
                    dim_coords = dense_arr[:, dim_idx]
                    in_basin_mask &= (dim_coords >= b_min) & (dim_coords <= b_max)

                matching_coords = dense_arr[in_basin_mask]
                points_in_basin = [pt.tolist() for pt in matching_coords]

            basin = TopographicBasin(
                basin_id=f"basin_{idx:03d}_{anchor.point_id}",
                anchor_point=anchor,
                basin_center=[float(c) for c in anchor.coordinates],
                coordinate_bounds=coord_bounds,
                max_residual_delta=float(abs(residual)),
                points_in_basin=points_in_basin,
                target_theory_level=target_tier,
                is_escalated=True,
            )
            basins.append(basin)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_id = f"escalation_manifest_{ts_str}"
    created_at = datetime.now(timezone.utc).isoformat()

    manifest_data = {
        "manifest_id": manifest_id,
        "total_anchors": len(validated_anchors),
        "escalated_count": len(escalated_points),
        "threshold": threshold_kcal_mol,
        "target_tier": target_tier,
        "escalated_ids": [p.point_id for p in escalated_points],
    }
    provenance_hash = hashlib.sha256(
        json.dumps(manifest_data, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return EscalationManifest(
        manifest_id=manifest_id,
        total_anchors_evaluated=len(validated_anchors),
        escalated_anchors_count=len(escalated_points),
        chemical_accuracy_threshold_kcal_mol=threshold_kcal_mol,
        escalated_points=escalated_points,
        topographic_basins=basins,
        target_quantum_tier=target_tier,
        provenance_hash=provenance_hash,
        created_at_utc=created_at,
    )


# =============================================================================
# 6. Core Functionality: Boundary-Anchored Residual Surface Interpolation
# =============================================================================


def interpolate_residual_surface(
    dense_grid_coords: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    dense_low_tier_energies: Sequence[float] | np.ndarray,
    anchor_coords: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    anchor_deltas: Sequence[float] | np.ndarray,
    is_periodic: bool = True,
    period: float = 360.0,
    bc_type: str = "periodic",
) -> DeltaSplineSurface:
    """Constructs an additive delta-learning potential energy surface.

    Formula: E_comp(q) = E_low(q) + S(Delta E(q))

    :param dense_grid_coords: Coordinates of dense evaluation grid (1D or 2D).
    :param dense_low_tier_energies: Energies of dense low-tier grid.
    :param anchor_coords: Sparse anchor coordinates where Delta E was evaluated.
    :param anchor_deltas: Energetic residuals Delta E at anchor coordinates.
    :param is_periodic: Enforce periodic boundary continuity across domain.
    :param period: Fundamental period for cyclic coordinates (default: 360.0).
    :param bc_type: Boundary condition type for CubicSpline.
    :return: DeltaSplineSurface containing dense composite energies.
    """
    dense_x_raw = np.asarray(dense_grid_coords, dtype=np.float64)
    dense_e_low = np.asarray(dense_low_tier_energies, dtype=np.float64)
    knots_x_raw = np.asarray(anchor_coords, dtype=np.float64)
    knots_delta = np.asarray(anchor_deltas, dtype=np.float64)

    if knots_x_raw.size == 0 or knots_delta.size == 0:
        raise ValueError("Anchor coordinates and delta residuals cannot be empty.")
    if len(knots_x_raw) != len(knots_delta):
        raise ValueError(
            f"Mismatch between anchor coords ({len(knots_x_raw)}) "
            f"and deltas ({len(knots_delta)})."
        )
    if len(dense_x_raw) != len(dense_e_low):
        raise ValueError(
            f"Mismatch between dense coords ({len(dense_x_raw)}) "
            f"and energies ({len(dense_e_low)})."
        )

    # Detect dimensionality
    if dense_x_raw.ndim == 1 or (dense_x_raw.ndim == 2 and dense_x_raw.shape[1] == 1):
        dims = 1
        dense_x = dense_x_raw.flatten()
        knots_x = knots_x_raw.flatten()

        if is_periodic and bc_type == "periodic":
            sorted_indices = np.argsort(knots_x)
            kx_sorted = knots_x[sorted_indices]
            kd_sorted = knots_delta[sorted_indices]

            if not np.isclose(kx_sorted[0], 0.0) or not np.isclose(
                kx_sorted[-1], period
            ):
                if not np.isclose(kx_sorted[0], 0.0):
                    kx_sorted = np.insert(kx_sorted, 0, 0.0)
                    kd_sorted = np.insert(kd_sorted, 0, kd_sorted[-1])
                if not np.isclose(kx_sorted[-1], period):
                    kx_sorted = np.append(kx_sorted, period)
                    kd_sorted = np.append(kd_sorted, kd_sorted[0])

            if not np.isclose(kd_sorted[0], kd_sorted[-1]):
                mean_bound = 0.5 * (kd_sorted[0] + kd_sorted[-1])
                kd_sorted[0] = mean_bound
                kd_sorted[-1] = mean_bound

            spline = CubicSpline(kx_sorted, kd_sorted, bc_type="periodic")
            delta_dense = spline(dense_x)
        else:
            sorted_indices = np.argsort(knots_x)
            kx_sorted = knots_x[sorted_indices]
            kd_sorted = knots_delta[sorted_indices]
            spline = CubicSpline(kx_sorted, kd_sorted, bc_type="natural")
            delta_dense = spline(dense_x)

        dense_coords_list = [[float(x)] for x in dense_x]
        knots_coords_list = [[float(x)] for x in kx_sorted]
        knots_deltas_list = [float(d) for d in kd_sorted]

    elif dense_x_raw.ndim == 2 and dense_x_raw.shape[1] == 2:
        dims = 2
        dense_x = dense_x_raw
        knots_x = knots_x_raw

        rbf = RBFInterpolator(
            knots_x, knots_delta, kernel="thin_plate_spline", smoothing=0.0
        )
        delta_dense = rbf(dense_x)
        spline = rbf

        dense_coords_list = [
            [float(dense_x[i, 0]), float(dense_x[i, 1])]
            for i in range(dense_x.shape[0])
        ]
        knots_coords_list = [
            [float(knots_x[i, 0]), float(knots_x[i, 1])]
            for i in range(knots_x.shape[0])
        ]
        knots_deltas_list = [float(d) for d in knots_delta]

    else:
        raise ValueError(f"Unsupported coordinate dimensionality: {dense_x_raw.shape}")

    composite_energies = dense_e_low + delta_dense

    provenance_data = {
        "dims": dims,
        "is_periodic": is_periodic,
        "dense_points_count": len(dense_e_low),
        "knots_count": len(knots_deltas_list),
        "mean_composite_energy": round(float(np.mean(composite_energies)), 8),
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_data, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return DeltaSplineSurface(
        dimensions=dims,
        is_periodic=is_periodic,
        period=period,
        knot_coordinates=knots_coords_list,
        knot_deltas=knots_deltas_list,
        dense_grid_points=dense_coords_list,
        dense_low_tier_energies=[float(e) for e in dense_e_low],
        dense_composite_energies=[float(e) for e in composite_energies],
        spline_correction_energies=[float(e) for e in delta_dense],
        spline_object=spline,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# 7. PyArrow Tabular Serialization with Air-Gap Enforcement
# =============================================================================


def export_manifest_pyarrow(
    manifest: EscalationManifest, output_path: Path | str
) -> Path:
    """Serializes an EscalationManifest to PyArrow Parquet format."""
    out = validate_air_gap_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for b in manifest.topographic_basins:
        rows.append(
            {
                "manifest_id": manifest.manifest_id,
                "point_id": b.anchor_point.point_id,
                "coordinates": json.dumps(b.anchor_point.coordinates),
                "point_type": b.anchor_point.point_type.value,
                "energy_low": b.anchor_point.energy_low,
                "energy_mid": b.anchor_point.energy_mid,
                "residual_delta": b.anchor_point.residual_delta,
                "requires_escalation": b.anchor_point.requires_escalation,
                "target_theory_level": b.target_theory_level,
                "basin_id": b.basin_id,
                "coordinate_bounds": json.dumps(b.coordinate_bounds),
                "points_in_basin_count": len(b.points_in_basin),
                "provenance_hash": manifest.provenance_hash,
                "created_at_utc": manifest.created_at_utc,
            }
        )

    if not rows:
        rows.append(
            {
                "manifest_id": manifest.manifest_id,
                "point_id": "NONE_ESCALATED",
                "coordinates": "[]",
                "point_type": "NONE",
                "energy_low": 0.0,
                "energy_mid": None,
                "residual_delta": 0.0,
                "requires_escalation": False,
                "target_theory_level": manifest.target_quantum_tier,
                "basin_id": "NONE",
                "coordinate_bounds": "[]",
                "points_in_basin_count": 0,
                "provenance_hash": manifest.provenance_hash,
                "created_at_utc": manifest.created_at_utc,
            }
        )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)
    logger.info(
        f"Exported EscalationManifest to Parquet at '{out}' ({table.num_rows} rows)."
    )
    return out


def export_surface_pyarrow(
    surface: DeltaSplineSurface, output_path: Path | str
) -> Path:
    """Serializes a DeltaSplineSurface to PyArrow Parquet format."""
    out = validate_air_gap_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for coords, e_low, e_comp, e_delta in zip(
        surface.dense_grid_points,
        surface.dense_low_tier_energies,
        surface.dense_composite_energies,
        surface.spline_correction_energies,
    ):
        rows.append(
            {
                "coordinates": json.dumps(coords),
                "dense_low_tier_energies": e_low,
                "spline_correction_energies": e_delta,
                "dense_composite_energies": e_comp,
                "dimensions": surface.dimensions,
                "is_periodic": surface.is_periodic,
                "provenance_hash": surface.provenance_hash,
            }
        )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out)
    logger.info(
        f"Exported DeltaSplineSurface to Parquet at '{out}' ({table.num_rows} rows)."
    )
    return out
