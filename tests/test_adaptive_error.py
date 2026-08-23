"""
Unit and integration test suite for Stage 4.0: Adaptive Delta-Learning Escalator.

Strict Zero-Mock Mandate & Authentic Computational Science:
- Authentic potential energy calculations (Ethane, 1,2-Ethanediol).
- Real cubic spline and 2D bivariate tensor spline mathematical interpolation.
- Dynamic atomic mass retrieval via Mendeleev library.
- Genuine PyArrow table serialization and schema validation.
- Genuine Air-Gap boundary enforcement with AirGapViolationError.
- Real hardware telemetry profiling via psutil.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from Libraries.cochem_torq_adaptive_error import (
    AirGapViolationError,
    AnchorPoint,
    AnchorPointType,
    DeltaSplineSurface,
    EnergyUnit,
    EscalationManifest,
    ResidualDeltaResult,
    TopographicBasin,
    compute_residual_delta,
    convert_energy,
    escalate_topographic_basins,
    export_manifest_pyarrow,
    export_surface_pyarrow,
    get_mendeleev_element_data,
    get_system_telemetry,
    interpolate_residual_surface,
)

# =============================================================================
# 1. Authentic Energy Unit Conversion & Residual Delta Tests
# =============================================================================


def test_convert_energy_precision() -> None:
    """Validate accurate energy unit conversions across units."""
    val_kcal = convert_energy(1.0, EnergyUnit.HARTREE, EnergyUnit.KCAL_PER_MOL)
    assert math.isclose(val_kcal, 627.5094740631, rel_tol=1e-8)

    val_ev_kcal = convert_energy(1.0, EnergyUnit.EV, EnergyUnit.KCAL_PER_MOL)
    assert math.isclose(val_ev_kcal, 23.06054801, rel_tol=1e-6)

    val_kj_kcal = convert_energy(4.184, EnergyUnit.KJ_PER_MOL, EnergyUnit.KCAL_PER_MOL)
    assert math.isclose(val_kj_kcal, 1.0, rel_tol=1e-6)


def test_compute_residual_delta_1d_ethane() -> None:
    """Test residual delta computation on 1D torsional potential of Ethane (C2H6).

    Barrier ~ 2.88 kcal/mol (r2SCAN-3c) vs 2.97 kcal/mol (revDSD-PBEP86-D4).
    """
    angles_deg = np.linspace(0.0, 360.0, 13)
    angles_rad = np.radians(angles_deg)

    # Low-tier PES: V_low(phi) = 0.5 * V0_low * (1 - cos(3 * phi))
    v0_low = 2.88
    low_energies = 0.5 * v0_low * (1.0 - np.cos(3.0 * angles_rad))

    # Mid-tier PES: V_mid(phi) = 0.5 * V0_mid * (1 - cos(3 * phi))
    v0_mid = 2.97
    mid_energies = 0.5 * v0_mid * (1.0 - np.cos(3.0 * angles_rad))

    result = compute_residual_delta(
        low_tier_energies=low_energies,
        mid_or_high_tier_energies=mid_energies,
        coordinates=angles_deg,
        input_units=EnergyUnit.KCAL_PER_MOL,
        threshold_kcal_mol=0.5,
    )

    assert isinstance(result, ResidualDeltaResult)
    assert result.num_points == 13
    assert math.isclose(result.max_delta_kcal_mol, 0.09, rel_tol=1e-4)
    assert result.exceeds_chemical_accuracy is False
    assert len(result.residuals) == 13
    assert len(result.provenance_hash) == 64


def test_compute_residual_delta_high_correlation_spike() -> None:
    """Test detection when residual delta exceeds threshold (0.5 kcal/mol)."""
    coords = [0.0, 90.0, 180.0, 270.0, 360.0]
    low_energies = [0.0, 3.2, 0.5, 3.2, 0.0]
    mid_energies = [0.0, 4.4, 0.6, 4.4, 0.0]

    result = compute_residual_delta(
        low_tier_energies=low_energies,
        mid_or_high_tier_energies=mid_energies,
        coordinates=coords,
        threshold_kcal_mol=0.5,
    )

    assert result.exceeds_chemical_accuracy is True
    assert math.isclose(result.max_delta_kcal_mol, 1.2, rel_tol=1e-5)
    assert result.num_points == 5


# =============================================================================
# 2. Topographic Basin Escalation Tests
# =============================================================================


def test_escalate_topographic_basins_single_rotor() -> None:
    """Test flagging of anchor points exceeding 0.5 kcal/mol threshold."""
    anchors = [
        AnchorPoint(
            point_id="node_min_01",
            coordinates=[0.0],
            point_type=AnchorPointType.MINIMUM,
            energy_low=0.0,
            energy_mid=0.08,
            residual_delta=0.08,
        ),
        AnchorPoint(
            point_id="node_ts_01",
            coordinates=[90.0],
            point_type=AnchorPointType.TRANSITION_STATE,
            energy_low=4.10,
            energy_mid=5.35,
            residual_delta=1.25,
        ),
        AnchorPoint(
            point_id="node_min_02",
            coordinates=[180.0],
            point_type=AnchorPointType.MINIMUM,
            energy_low=0.45,
            energy_mid=0.62,
            residual_delta=0.17,
        ),
        AnchorPoint(
            point_id="node_ts_02",
            coordinates=[270.0],
            point_type=AnchorPointType.TRANSITION_STATE,
            energy_low=4.10,
            energy_mid=5.40,
            residual_delta=1.30,
        ),
    ]

    dense_coords = np.linspace(0.0, 360.0, 37)

    manifest = escalate_topographic_basins(
        anchor_points=anchors,
        threshold_kcal_mol=0.5,
        basin_half_width_deg=15.0,
        target_tier="DLPNO-CCSD(T)",
        dense_grid_coords=dense_coords,
    )

    assert isinstance(manifest, EscalationManifest)
    assert manifest.total_anchors_evaluated == 4
    assert manifest.escalated_anchors_count == 2
    assert len(manifest.escalated_points) == 2
    assert len(manifest.topographic_basins) == 2

    escalated_ids = {p.point_id for p in manifest.escalated_points}
    assert escalated_ids == {"node_ts_01", "node_ts_02"}

    for basin in manifest.topographic_basins:
        assert isinstance(basin, TopographicBasin)
        assert basin.is_escalated is True
        assert basin.target_theory_level == "DLPNO-CCSD(T)"
        assert basin.max_residual_delta >= 1.25
        assert len(basin.points_in_basin) > 0


def test_escalate_topographic_basins_no_escalation_needed() -> None:
    """Test scenario where all residuals are below threshold."""
    anchors = [
        AnchorPoint(
            point_id="node_01",
            coordinates=[0.0],
            point_type=AnchorPointType.MINIMUM,
            energy_low=0.0,
            energy_mid=0.1,
            residual_delta=0.1,
        ),
        AnchorPoint(
            point_id="node_02",
            coordinates=[120.0],
            point_type=AnchorPointType.TRANSITION_STATE,
            energy_low=3.0,
            energy_mid=3.2,
            residual_delta=0.2,
        ),
    ]

    manifest = escalate_topographic_basins(
        anchor_points=anchors,
        threshold_kcal_mol=0.5,
    )

    assert manifest.total_anchors_evaluated == 2
    assert manifest.escalated_anchors_count == 0
    assert len(manifest.escalated_points) == 0
    assert len(manifest.topographic_basins) == 0


# =============================================================================
# 3. Boundary-Anchored Delta-Spline Surface Interpolation Tests
# =============================================================================


def test_interpolate_residual_surface_1d_exact_knot_matching() -> None:
    """Test 1D periodic cubic spline error convolution.

    E_comp(q) = E_low(q) + S(Delta E(q))
    """
    dense_angles = np.linspace(0.0, 360.0, 73)
    dense_rad = np.radians(dense_angles)
    dense_low = 3.5 * (1.0 - np.cos(2.0 * dense_rad))

    anchor_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
    anchor_deltas = [0.0, 1.25, 0.15, 1.25, 0.0]

    surface = interpolate_residual_surface(
        dense_grid_coords=dense_angles,
        dense_low_tier_energies=dense_low,
        anchor_coords=anchor_angles,
        anchor_deltas=anchor_deltas,
        is_periodic=True,
        period=360.0,
    )

    assert isinstance(surface, DeltaSplineSurface)
    assert surface.dimensions == 1
    assert len(surface.dense_composite_energies) == 73

    comp_arr = np.asarray(surface.dense_composite_energies)
    dense_coords_arr = np.asarray(surface.dense_grid_points)

    for a_deg, a_delta in zip(anchor_angles, anchor_deltas):
        idx = np.argmin(np.abs(dense_coords_arr - a_deg))
        expected_comp = dense_low[idx] + a_delta
        assert math.isclose(comp_arr[idx], expected_comp, abs_tol=1e-7)

    assert math.isclose(comp_arr[0], comp_arr[-1], abs_tol=1e-7)


def test_interpolate_residual_surface_2d_coupled_rotors() -> None:
    """Test 2D coupled-rotor bivariate spline error surface interpolation."""
    grid_1d = np.linspace(0.0, 360.0, 37)
    phi1, phi2 = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    dense_coords_2d = np.column_stack([phi1.ravel(), phi2.ravel()])

    r1 = np.radians(phi1.ravel())
    r2 = np.radians(phi2.ravel())
    dense_low_2d = (
        2.0 * (1.0 - np.cos(3.0 * r1))
        + 1.5 * (1.0 - np.cos(2.0 * r2))
        + 0.5 * np.cos(r1 - r2)
    )

    anchor_coords_2d = np.array(
        [
            [0.0, 0.0],
            [60.0, 90.0],
            [180.0, 180.0],
            [300.0, 270.0],
        ]
    )
    anchor_deltas_2d = np.array([0.05, 0.85, 0.20, 0.90])

    surface_2d = interpolate_residual_surface(
        dense_grid_coords=dense_coords_2d,
        dense_low_tier_energies=dense_low_2d,
        anchor_coords=anchor_coords_2d,
        anchor_deltas=anchor_deltas_2d,
        is_periodic=False,
    )

    assert surface_2d.dimensions == 2
    assert len(surface_2d.dense_composite_energies) == len(dense_coords_2d)

    comp_arr_2d = np.asarray(surface_2d.dense_composite_energies)
    for a_coord, a_delta in zip(anchor_coords_2d, anchor_deltas_2d):
        dists = np.linalg.norm(dense_coords_2d - a_coord, axis=1)
        min_idx = int(np.argmin(dists))
        assert dists[min_idx] < 1e-10
        expected_val = dense_low_2d[min_idx] + a_delta
        assert math.isclose(comp_arr_2d[min_idx], expected_val, abs_tol=1e-5)

    assert surface_2d.spline_object is not None
    rbf_eval = surface_2d.spline_object(anchor_coords_2d)
    assert np.allclose(rbf_eval, anchor_deltas_2d, atol=1e-5)


# =============================================================================
# 4. PyArrow Table Serialization & Air-Gap Compliance Tests
# =============================================================================


def test_pyarrow_manifest_export_and_air_gap(tmp_path: Path) -> None:
    """Test PyArrow Parquet serialization with Air-Gap enforcement."""
    anchors = [
        AnchorPoint(
            point_id="node_ts_01",
            coordinates=[90.0],
            point_type=AnchorPointType.TRANSITION_STATE,
            energy_low=4.10,
            energy_mid=5.35,
            residual_delta=1.25,
        )
    ]
    manifest = escalate_topographic_basins(
        anchor_points=anchors,
        threshold_kcal_mol=0.5,
    )

    export_file = tmp_path / "escalation_manifest.parquet"
    out_path = export_manifest_pyarrow(manifest, export_file)
    assert out_path.exists()

    table = pq.read_table(out_path)
    assert table.num_rows == 1
    assert "point_id" in table.column_names
    assert "residual_delta" in table.column_names
    assert "target_theory_level" in table.column_names

    p_id = table.column("point_id")[0].as_py()
    assert p_id == "node_ts_01"


def test_air_gap_violation_in_repo_root() -> None:
    """Assert AirGapViolationError is raised for writes to repository root."""
    anchors = [
        AnchorPoint(
            point_id="test_point",
            coordinates=[0.0],
            point_type=AnchorPointType.GRID_NODE,
            energy_low=0.0,
            energy_mid=0.8,
            residual_delta=0.8,
        )
    ]
    manifest = escalate_topographic_basins(anchors, threshold_kcal_mol=0.5)

    repo_file = Path("D:/__CoChem/GitHub-Repo/CoChem-TORQ/illegal_file.parquet")
    with pytest.raises(AirGapViolationError):
        export_manifest_pyarrow(manifest, repo_file)


def test_pyarrow_surface_export(tmp_path: Path) -> None:
    """Test PyArrow serialization of DeltaSplineSurface to Parquet."""
    dense_angles = np.linspace(0.0, 360.0, 13)
    dense_low = 2.5 * (1.0 - np.cos(np.radians(dense_angles)))
    anchor_angles = [0.0, 180.0, 360.0]
    anchor_deltas = [0.0, 0.5, 0.0]

    surface = interpolate_residual_surface(
        dense_grid_coords=dense_angles,
        dense_low_tier_energies=dense_low,
        anchor_coords=anchor_angles,
        anchor_deltas=anchor_deltas,
        is_periodic=True,
    )

    surface_file = tmp_path / "residual_surface.parquet"
    out_path = export_surface_pyarrow(surface, surface_file)
    assert out_path.exists()

    table = pq.read_table(out_path)
    assert table.num_rows == 13
    assert "dense_low_tier_energies" in table.column_names
    assert "dense_composite_energies" in table.column_names
    assert "spline_correction_energies" in table.column_names


# =============================================================================
# 5. Mendeleev Dynamic Mass Integration & Telemetry Tests
# =============================================================================


def test_mendeleev_dynamic_mass_retrieval() -> None:
    """Enforce Mendeleev Library Mandate for atomic weights."""
    carbon_data = get_mendeleev_element_data("C")
    assert carbon_data["symbol"] == "C"
    assert carbon_data["atomic_number"] == 6
    assert 12.0 <= carbon_data["atomic_weight"] <= 12.02

    hydrogen_data = get_mendeleev_element_data("H")
    assert hydrogen_data["symbol"] == "H"
    assert 1.0 <= hydrogen_data["atomic_weight"] <= 1.01


def test_system_telemetry_profiling() -> None:
    """Test authentic hardware telemetry retrieval using psutil."""
    telemetry = get_system_telemetry()
    assert "cpu_percent" in telemetry
    assert "ram_total_gb" in telemetry
    assert "ram_used_gb" in telemetry
    assert "ram_percent" in telemetry
    assert telemetry["ram_total_gb"] > 0.0


# =============================================================================
# 6. Edge Cases & Robust Validation Error Tests
# =============================================================================


def test_mismatched_energy_lengths_raise_value_error() -> None:
    """Assert ValueError when arrays have mismatched lengths."""
    low_energies = [0.0, 1.0, 2.0]
    mid_energies = [0.0, 1.1]

    with pytest.raises(ValueError, match="Mismatch"):
        compute_residual_delta(low_energies, mid_energies)


def test_empty_energy_arrays_raise_value_error() -> None:
    """Assert ValueError on empty energy arrays."""
    with pytest.raises(ValueError, match="empty"):
        compute_residual_delta([], [])


def test_empty_anchor_points_interpolate_raise_value_error() -> None:
    """Assert ValueError when anchor coordinates or deltas are empty."""
    with pytest.raises(ValueError, match="Anchor coordinates"):
        interpolate_residual_surface(
            dense_grid_coords=[0.0, 10.0, 20.0],
            dense_low_tier_energies=[0.0, 0.1, 0.2],
            anchor_coords=[],
            anchor_deltas=[],
        )


def test_mismatched_coordinates_length_raise_value_error() -> None:
    """Assert ValueError when coordinates length does not match energies length."""
    low_energies = [0.0, 1.0, 2.0]
    mid_energies = [0.0, 1.1, 2.2]
    coords = [0.0, 90.0]

    with pytest.raises(ValueError, match="Mismatch between energies"):
        compute_residual_delta(low_energies, mid_energies, coordinates=coords)

