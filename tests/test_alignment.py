"""
CoChem-TORQ: Exact Eckart Frame Aligner & Rotational Constants Engine Test Suite
================================================================================
Phase 2 (Stage 1.0 - 2.0) Authentic Physical Test Matrix
--------------------------------------------------------
Tests mass-weighted Cartesian normalization, rigid-rotor frame stability,
inertia tensor diagonalization, right-handed coordinate enforcement (det(R) = +1),
rotational constants in multiple units, Ray's asymmetry parameter, inertial defect,
rotor classification across all 5 canonical types, Eckart coordinate inversion,
and Tripartite Filesystem Air-Gap security.

Authoritative Standards:
- CIAAW / IUPAC Exact Mono-Isotopic Masses
- CODATA 2018 / 2022 Fundamental Physical Constants
- Method Matrix: Stage 1.0 - 2.0 Gateway & Eckart Alignment Protocols
- Provenance tags: [M] Measured, [D] Derived, [E] Estimated
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import scipy.constants as const
from scipy.spatial.transform import Rotation

from Libraries.cochem_torq_alignment import (
    AirGapViolationError,
    CoChemAirGapRing,
    CoChemPathManager,
    EckartAligner,
    EckartAlignmentResult,
    classify_rotor_type,
    compute_inertia_tensor,
    compute_inertial_defect,
    compute_ray_asymmetry_parameter,
    compute_rotational_constants,
    diagonalize_principal_axes,
    enforce_ciaaw_masses,
    invert_eckart_coordinates,
    serialize_eckart_matrix,
    translate_com_to_origin,
)


# ============================================================================
# Authentic Physical Geometry Fixtures
# ============================================================================

@pytest.fixture
def water_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium C2v water (H2O) geometry:
    r(OH) = 0.9578 Angstrom, theta(HOH) = 104.5 degrees. [D]
    """
    r_oh = 0.9578
    theta_rad = np.radians(104.5)
    syms = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh * np.sin(theta_rad / 2.0), r_oh * np.cos(theta_rad / 2.0), 0.0],
            [-r_oh * np.sin(theta_rad / 2.0), r_oh * np.cos(theta_rad / 2.0), 0.0],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def co2_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium linear D_inf_h carbon dioxide (CO2) geometry:
    r(CO) = 1.1600 Angstrom. [D]
    """
    r_co = 1.1600
    syms = ["O", "C", "O"]
    coords = np.array(
        [
            [0.0, 0.0, -r_co],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r_co],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def methane_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium tetrahedral Td methane (CH4) geometry:
    r(CH) = 1.0870 Angstrom. [D]
    """
    r_ch = 1.0870
    a = r_ch / np.sqrt(3.0)
    syms = ["C", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def methyl_chloride_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic equilibrium prolate C3v methyl chloride (CH3Cl) geometry:
    r(CCl) = 1.7810 Angstrom, r(CH) = 1.0860 Angstrom, theta(HCH) = 110.5 degrees. [D]
    """
    r_ccl = 1.7810
    r_ch = 1.0860
    theta_hch = np.radians(110.5)
    # Distance from z-axis for H atoms: r_ch * sin(theta_tilt)
    # Using tetrahedral relation: cos(angle between C-H and C3 axis) = cos(theta_c3)
    cos_theta_c3 = np.sqrt((2.0 * np.cos(theta_hch) + 1.0) / 3.0) if (2.0 * np.cos(theta_hch) + 1.0) > 0 else 0.35
    sin_theta_c3 = np.sqrt(1.0 - cos_theta_c3**2)
    r_xy = r_ch * sin_theta_c3
    z_h = -r_ch * cos_theta_c3

    syms = ["C", "Cl", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r_ccl],
            [r_xy, 0.0, z_h],
            [-r_xy * 0.5, r_xy * np.sqrt(3.0) / 2.0, z_h],
            [-r_xy * 0.5, -r_xy * np.sqrt(3.0) / 2.0, z_h],
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def benzene_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic planar D6h benzene (C6H6) geometry:
    r(CC) = 1.3970 Angstrom, r(CH) = 1.0840 Angstrom. [D]
    """
    r_cc = 1.3970
    r_ch = 1.0840
    syms = ["C"] * 6 + ["H"] * 6
    coords_list: list[list[float]] = []
    for i in range(6):
        phi = i * np.pi / 3.0
        coords_list.append([r_cc * np.cos(phi), r_cc * np.sin(phi), 0.0])
    for i in range(6):
        phi = i * np.pi / 3.0
        coords_list.append([(r_cc + r_ch) * np.cos(phi), (r_cc + r_ch) * np.sin(phi), 0.0])

    return syms, np.array(coords_list, dtype=np.float64)


@pytest.fixture
def fluoropropane_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic asymmetric top 2-fluoropropane (C3H7F) geometry: [D]
    """
    syms = ["C", "F", "C", "C", "H", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],  # C2
            [1.4000, 0.0000, 0.0000],  # F
            [-0.5000, 1.2500, 0.0000],  # C1
            [-0.5000, -0.6250, 1.0825],  # C3
            [-0.3000, 0.0000, -1.0000],  # H(C2)
            [-1.5500, 1.2500, 0.0000],  # H1a
            [-0.1500, 1.7500, 0.8900],  # H1b
            [-0.1500, 1.7500, -0.8900],  # H1c
            [-1.5500, -0.6250, 1.0825],  # H3a
            [-0.1500, -0.1250, 1.9725],  # H3b
            [-0.1500, -1.6500, 1.0825],  # H3c
        ],
        dtype=np.float64,
    )
    return syms, coords


@pytest.fixture
def ethanol_geometry() -> tuple[list[str], np.ndarray]:
    """
    Authentic anti-conformer ethanol (CH3CH2OH) geometry: [D]
    """
    syms = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0000, 0.0000, 0.0000],  # C1
            [1.5200, 0.0000, 0.0000],  # C2
            [2.0800, 1.2800, 0.0000],  # O
            [3.0400, 1.2000, 0.0000],  # H(O)
            [-0.3800, 1.0200, 0.0000],  # H1a
            [-0.3800, -0.5100, 0.8900],  # H1b
            [-0.3800, -0.5100, -0.8900],  # H1c
            [1.9000, -0.5100, 0.8900],  # H2a
            [1.9000, -0.5100, -0.8900],  # H2b
        ],
        dtype=np.float64,
    )
    return syms, coords


# ============================================================================
# 1. CIAAW Mono-Isotopic Mass Enforcement Tests
# ============================================================================

def test_ciaaw_mass_enforcement_primary_elements() -> None:
    """Verifies that CIAAW mono-isotopic masses match IUPAC definitions exactly. [M]"""
    syms = ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "Br", "I"]
    masses = enforce_ciaaw_masses(syms)

    assert len(masses) == len(syms)
    assert masses.dtype == np.float64
    assert np.isclose(masses[0], 1.00782503223, atol=1e-10)  # 1H
    assert np.isclose(masses[1], 12.00000000000, atol=1e-10)  # 12C
    assert np.isclose(masses[2], 14.00307400443, atol=1e-10)  # 14N
    assert np.isclose(masses[3], 15.99491461957, atol=1e-10)  # 16O
    assert np.isclose(masses[4], 18.99840316273, atol=1e-10)  # 19F
    assert np.isclose(masses[7], 31.97207073, atol=1e-8)  # 32S
    assert np.isclose(masses[8], 34.96885271, atol=1e-8)  # 35Cl
    assert np.isclose(masses[9], 78.9183376, atol=1e-7)  # 79Br
    assert np.isclose(masses[10], 126.9044719, atol=1e-7)  # 127I


def test_ciaaw_mass_enforcement_isotopologues() -> None:
    """Verifies that specific isotopes (D, 13C, 15N, 18O, 37Cl) resolve accurately. [M]"""
    syms = ["D", "2H", "T", "13C", "15N", "18O", "37Cl", "81Br"]
    masses = enforce_ciaaw_masses(syms)

    assert np.isclose(masses[0], 2.01410177812, atol=1e-10)  # D
    assert np.isclose(masses[1], 2.01410177812, atol=1e-10)  # 2H
    assert np.isclose(masses[2], 3.0160492779, atol=1e-9)  # T
    assert np.isclose(masses[3], 13.00335483507, atol=1e-10)  # 13C
    assert np.isclose(masses[4], 15.00010889888, atol=1e-10)  # 15N
    assert np.isclose(masses[5], 17.99915961286, atol=1e-10)  # 18O
    assert np.isclose(masses[6], 36.96590262, atol=1e-8)  # 37Cl
    assert np.isclose(masses[7], 80.9162897, atol=1e-7)  # 81Br


def test_ciaaw_mass_enforcement_case_insensitivity() -> None:
    """Verifies that lowercase and mixed-case atomic symbols are resolved correctly. [D]"""
    syms = ["c", "h", "o", "cl", "BR", "na"]
    masses = enforce_ciaaw_masses(syms)

    assert np.isclose(masses[0], 12.00000000000, atol=1e-10)
    assert np.isclose(masses[1], 1.00782503223, atol=1e-10)
    assert np.isclose(masses[2], 15.99491461957, atol=1e-10)
    assert np.isclose(masses[3], 34.96885271, atol=1e-8)
    assert np.isclose(masses[4], 78.9183376, atol=1e-7)
    assert np.isclose(masses[5], 22.9897692820, atol=1e-8)


def test_ciaaw_mass_enforcement_errors() -> None:
    """Verifies error handling on empty lists and invalid symbols."""
    assert len(enforce_ciaaw_masses([])) == 0

    with pytest.raises(ValueError, match="Empty or blank atomic symbol"):
        enforce_ciaaw_masses(["C", "  ", "O"])

    with pytest.raises(ValueError, match="Unrecognized or invalid atomic symbol"):
        enforce_ciaaw_masses(["C", "NonExistentElementXYZ", "O"])


# ============================================================================
# 2. Center of Mass Translation & Precision Tests
# ============================================================================

def test_translate_com_to_origin_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """Verifies mass-weighted COM centering for water with residual < 1e-12 Angstrom. [D]"""
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)

    centered_geo, com_vector = translate_com_to_origin(coords, masses)

    # Check centered geometry shape
    assert centered_geo.shape == coords.shape
    assert com_vector.shape == (3,)

    # Total mass and mass-weighted residual
    total_mass = np.sum(masses)
    residual_com = np.sum(masses[:, None] * centered_geo, axis=0) / total_mass
    residual_norm = float(np.linalg.norm(residual_com))

    assert residual_norm < 1e-12, f"COM residual {residual_norm:.3e} exceeds tolerance 1e-12"
    assert np.allclose(coords - com_vector, centered_geo, atol=1e-14)


def test_translate_com_arbitrary_shift(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Tests COM translation with a deliberate large spatial translation (+100, -250, +500 A). [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    shift = np.array([100.0, -250.0, 500.0])
    shifted_coords = coords + shift

    centered_geo, com_vector = translate_com_to_origin(shifted_coords, masses)

    total_mass = np.sum(masses)
    residual_com = np.sum(masses[:, None] * centered_geo, axis=0) / total_mass
    residual_norm = float(np.linalg.norm(residual_com))

    assert residual_norm < 1e-12
    # Verify that subtracting the initial COM and recentering matches
    orig_centered, orig_com = translate_com_to_origin(coords, masses)
    assert np.allclose(centered_geo, orig_centered, atol=1e-12)
    assert np.allclose(com_vector, orig_com + shift, atol=1e-12)


def test_translate_com_validation_errors() -> None:
    """Verifies input validation on translate_com_to_origin."""
    # Mismatched dimensions
    with pytest.raises(ValueError, match="Geometry must be a 2D array of shape"):
        bad_geom_2d = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        translate_com_to_origin(bad_geom_2d, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Masses must be a 1D array"):
        valid_geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        translate_com_to_origin(valid_geom, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Geometry array cannot be empty"):
        translate_com_to_origin(np.empty((0, 3)), np.empty(0))

    with pytest.raises(ValueError, match="Total molecular mass must be strictly positive"):
        zero_mass_geom = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        translate_com_to_origin(zero_mass_geom, np.array([0.0, 0.0]))


# ============================================================================
# 3. Moment of Inertia Tensor Calculation Tests
# ============================================================================

def test_compute_inertia_tensor_symmetry_and_values(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies that the calculated 3x3 inertia tensor is strictly symmetric
    and adheres to parallel axis theorem invariants. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)
    centered_geo, com_vector = translate_com_to_origin(coords, masses)

    tensor = compute_inertia_tensor(centered_geo, masses)

    assert tensor.shape == (3, 3)
    # Check strict symmetry
    assert np.allclose(tensor, tensor.T, atol=1e-14)

    # Manual analytical calculation
    x = centered_geo[:, 0]
    y = centered_geo[:, 1]
    z = centered_geo[:, 2]

    expected_ixx = np.sum(masses * (y**2 + z**2))
    expected_iyy = np.sum(masses * (x**2 + z**2))
    expected_izz = np.sum(masses * (x**2 + y**2))
    expected_ixy = -np.sum(masses * x * y)

    assert np.isclose(tensor[0, 0], expected_ixx, atol=1e-14)
    assert np.isclose(tensor[1, 1], expected_iyy, atol=1e-14)
    assert np.isclose(tensor[2, 2], expected_izz, atol=1e-14)
    assert np.isclose(tensor[0, 1], expected_ixy, atol=1e-14)


# ============================================================================
# 4. Principal Axis Diagonalization & Right-Handed Coordinate Frame Tests
# ============================================================================

def test_diagonalize_principal_axes_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies principal moment ordering (Ia <= Ib <= Ic), right-handed rotation (det(R) = +1),
    and vanishing off-diagonal elements (< 1e-11) for water. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)

    aligned_geo, principal_moments, rot_mat = diagonalize_principal_axes(coords, masses)

    # 1. Principal moment ordering
    ia, ib, ic = principal_moments
    assert ia <= ib <= ic
    assert ia > 0.0

    # 2. Right-handed coordinate frame: det(R) == +1.0
    det_r = float(np.linalg.det(rot_mat))
    assert np.isclose(det_r, 1.0, atol=1e-12)

    # 3. Orthonormality of rotation matrix: R @ R.T == I
    assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-12)
    assert np.allclose(rot_mat.T @ rot_mat, np.eye(3), atol=1e-12)

    # 4. Vanishing off-diagonal elements in transformed frame
    transformed_tensor = compute_inertia_tensor(aligned_geo, masses)
    off_diag = np.array(
        [
            transformed_tensor[0, 1],
            transformed_tensor[0, 2],
            transformed_tensor[1, 2],
        ]
    )
    assert np.max(np.abs(off_diag)) < 1e-11
    assert np.allclose(np.diag(transformed_tensor), principal_moments, atol=1e-12)


def test_diagonalize_principal_axes_invariance(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Rigorous invariance test: shifts and rotates 2-fluoropropane by arbitrary 3D angles
    and asserts that extracted principal moments are invariant to within 1e-9 Da*A^2. [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    # Base alignment
    _, base_moments, _ = diagonalize_principal_axes(coords, masses)

    # Apply random 3D rotation and large shift
    euler_angles = [34.5, 56.7, 78.9]
    random_rot = Rotation.from_euler("zyx", euler_angles, degrees=True).as_matrix()
    shifted_coords = (coords @ random_rot.T) + np.array([50.0, -25.0, 100.0])

    # Re-align transformed molecule
    aligned_geo_new, new_moments, rot_mat_new = diagonalize_principal_axes(shifted_coords, masses)

    # Assert invariant eigenvalues
    assert np.allclose(base_moments, new_moments, atol=1e-9)
    assert np.isclose(np.linalg.det(rot_mat_new), 1.0, atol=1e-12)

    # Check vanishing off-diagonals
    new_tensor = compute_inertia_tensor(aligned_geo_new, masses)
    off_diag = [new_tensor[0, 1], new_tensor[0, 2], new_tensor[1, 2]]
    assert np.max(np.abs(off_diag)) < 1e-11


# ============================================================================
# 5. Exact Coordinate Inversion Round-Trip Tests
# ============================================================================

def test_coordinate_inversion_roundtrip(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies that invert_eckart_coordinates accurately reconstructs original
    laboratory frame coordinates to within 1e-12 Angstroms. [D]
    """
    syms, coords = fluoropropane_geometry
    masses = enforce_ciaaw_masses(syms)

    # Shift and rotate
    rot = Rotation.from_euler("xyz", [15.0, 45.0, 60.0], degrees=True).as_matrix()
    lab_coords = (coords @ rot.T) + np.array([-12.34, 56.78, 90.12])

    centered_geo, com_vector = translate_com_to_origin(lab_coords, masses)
    aligned_geo, _, rot_mat = diagonalize_principal_axes(lab_coords, masses)

    reconstructed_coords = invert_eckart_coordinates(aligned_geo, rot_mat, com_vector)

    assert reconstructed_coords.shape == lab_coords.shape
    max_inversion_error = float(np.max(np.abs(reconstructed_coords - lab_coords)))
    assert (
        max_inversion_error < 1e-12
    ), f"Inversion error {max_inversion_error:.3e} exceeds 1e-12 Angstrom tolerance"


def test_coordinate_inversion_validation_errors() -> None:
    """Verifies dimension validation on invert_eckart_coordinates."""
    with pytest.raises(ValueError, match="aligned_geometry must have shape"):
        bad_geom_2d = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        invert_eckart_coordinates(bad_geom_2d, np.eye(3), np.array([0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="rotation_matrix must have shape"):
        valid_geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        invert_eckart_coordinates(valid_geom, np.eye(2), np.array([0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="com_vector must have shape"):
        valid_geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        invert_eckart_coordinates(valid_geom, np.eye(3), np.array([0.0, 0.0, 0.0, 0.0]))


# ============================================================================
# 6. Rotational Constants & Unit Conversion Tests
# ============================================================================

def test_rotational_constants_water_units(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies rotational constants for water across all supported units
    (MHz, GHz, cm-1, Hz, J) against CODATA analytical relations. [D]
    """
    syms, coords = water_geometry
    masses = enforce_ciaaw_masses(syms)
    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    a_ghz, b_ghz, c_ghz = compute_rotational_constants(principal_moments, unit="GHz")
    a_cm1, b_cm1, c_cm1 = compute_rotational_constants(principal_moments, unit="cm-1")
    a_hz, b_hz, c_hz = compute_rotational_constants(principal_moments, unit="Hz")
    a_j, b_j, c_j = compute_rotational_constants(principal_moments, unit="J")

    # Analytical unit ratios
    assert np.isclose(a_ghz, a_mhz / 1.0e3, rtol=1e-12)
    assert np.isclose(b_ghz, b_mhz / 1.0e3, rtol=1e-12)
    assert np.isclose(c_ghz, c_mhz / 1.0e3, rtol=1e-12)

    assert np.isclose(a_hz, a_mhz * 1.0e6, rtol=1e-12)
    assert np.isclose(b_hz, b_mhz * 1.0e6, rtol=1e-12)
    assert np.isclose(c_hz, c_mhz * 1.0e6, rtol=1e-12)

    # Conversion between MHz and cm^-1: 1 cm^-1 = c (cm/s) * 1e-6 MHz = 29979.2458 MHz
    speed_of_light_cm_s = const.c * 100.0
    assert np.isclose(a_mhz * 1.0e6 / speed_of_light_cm_s, a_cm1, rtol=1e-12)
    assert np.isclose(b_mhz * 1.0e6 / speed_of_light_cm_s, b_cm1, rtol=1e-12)
    assert np.isclose(c_mhz * 1.0e6 / speed_of_light_cm_s, c_cm1, rtol=1e-12)

    # Wavenumber values for H2O equilibrium: A ~ 27.4 cm^-1, B ~ 14.6 cm^-1, C ~ 9.5 cm^-1 [M]
    assert 25.0 < a_cm1 < 30.0
    assert 12.0 < b_cm1 < 16.0
    assert 8.0 < c_cm1 < 11.0


def test_rotational_constants_unsupported_unit() -> None:
    """Verifies that requesting an invalid unit raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported rotational constant unit"):
        compute_rotational_constants(np.array([1.0, 2.0, 3.0]), unit="eV")


# ============================================================================
# 7. Canonical Rotor Classifications & Ray's Kappa Tests
# ============================================================================

def test_linear_rotor_co2(co2_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies linear rotor properties for carbon dioxide (CO2):
    Ia ~ 0, Ib == Ic, rotor_type == 'linear', Ray's kappa == -1.0, Delta == 0.0. [D]
    """
    syms, coords = co2_geometry
    masses = enforce_ciaaw_masses(syms)

    aligned_geo, principal_moments, rot_mat = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert ia < 1e-12  # Zero moment along linear molecular axis
    assert np.isclose(ib, ic, rtol=1e-10)

    # Classification
    rotor = classify_rotor_type(principal_moments)
    assert rotor == "linear"

    # Rotational constants
    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert math.isinf(a_mhz)
    assert np.isclose(b_mhz, c_mhz, rtol=1e-10)
    # CO2 B ~ 11698 MHz (0.3902 cm^-1) [M]
    assert 11000.0 < b_mhz < 12500.0

    # Ray's kappa for linear molecule
    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert kappa == -1.0

    # Inertial defect for linear molecule: Ic - Ia - Ib == 0
    delta = compute_inertial_defect(principal_moments)
    assert np.isclose(delta, 0.0, atol=1e-12)


def test_spherical_top_methane(methane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies spherical top properties for methane (CH4):
    Ia == Ib == Ic, rotor_type == 'spherical_top', Ray's kappa == 0.0. [D]
    """
    syms, coords = methane_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert np.isclose(ia, ib, rtol=1e-6)
    assert np.isclose(ib, ic, rtol=1e-6)

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "spherical_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert np.isclose(a_mhz, b_mhz, rtol=1e-6)
    assert np.isclose(b_mhz, c_mhz, rtol=1e-6)

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert kappa == 0.0


def test_prolate_symmetric_top_methyl_chloride(methyl_chloride_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies prolate symmetric top properties for CH3Cl:
    Ia < Ib == Ic (A > B == C), rotor_type == 'prolate_symmetric_top', Ray's kappa == -1.0. [D]
    """
    syms, coords = methyl_chloride_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert ia < ib
    assert np.isclose(ib, ic, rtol=1e-4)

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "prolate_symmetric_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert a_mhz > b_mhz
    assert np.isclose(b_mhz, c_mhz, rtol=1e-4)

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert np.isclose(kappa, -1.0, atol=1e-3)


def test_oblate_symmetric_top_benzene(benzene_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies oblate symmetric top properties for planar benzene (C6H6):
    Ia == Ib < Ic (A == B > C), rotor_type == 'oblate_symmetric_top', Ray's kappa == +1.0, Delta == 0.0. [D]
    """
    syms, coords = benzene_geometry
    masses = enforce_ciaaw_masses(syms)

    _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

    ia, ib, ic = principal_moments
    assert np.isclose(ia, ib, rtol=1e-6)
    assert ib < ic
    assert np.isclose(ic, ia + ib, rtol=1e-6)  # Planar condition Ic = Ia + Ib

    rotor = classify_rotor_type(principal_moments)
    assert rotor == "oblate_symmetric_top"

    a_mhz, b_mhz, c_mhz = compute_rotational_constants(principal_moments, unit="MHz")
    assert np.isclose(a_mhz, b_mhz, rtol=1e-6)
    assert b_mhz > c_mhz

    kappa = compute_ray_asymmetry_parameter((a_mhz, b_mhz, c_mhz))
    assert np.isclose(kappa, 1.0, atol=1e-5)

    delta = compute_inertial_defect(principal_moments)
    assert np.isclose(delta, 0.0, atol=1e-10)


def test_asymmetric_tops_water_and_ethanol(
    water_geometry: tuple[list[str], np.ndarray],
    ethanol_geometry: tuple[list[str], np.ndarray],
) -> None:
    """
    Verifies asymmetric top properties for water and ethanol:
    Ia < Ib < Ic (A > B > C), rotor_type == 'asymmetric_top', -1.0 < kappa < 1.0. [D]
    """
    for syms, coords in [water_geometry, ethanol_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, principal_moments, _ = diagonalize_principal_axes(coords, masses)

        rotor = classify_rotor_type(principal_moments)
        assert rotor == "asymmetric_top"

        rc = compute_rotational_constants(principal_moments, unit="MHz")
        assert rc[0] > rc[1] > rc[2]

        kappa = compute_ray_asymmetry_parameter(rc)
        assert -1.0 < kappa < 1.0


# ============================================================================
# 8. Inertial Defect & Planarity Tests
# ============================================================================

def test_inertial_defect_planarity(
    water_geometry: tuple[list[str], np.ndarray],
    benzene_geometry: tuple[list[str], np.ndarray],
    co2_geometry: tuple[list[str], np.ndarray],
    methane_geometry: tuple[list[str], np.ndarray],
    ethanol_geometry: tuple[list[str], np.ndarray],
) -> None:
    """
    Verifies that strictly planar equilibrium molecules have Delta == 0.0
    while non-planar molecules have Delta < 0.0. [D]
    """
    # Planar molecules
    for syms, coords in [water_geometry, benzene_geometry, co2_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, moments, _ = diagonalize_principal_axes(coords, masses)
        delta = compute_inertial_defect(moments)
        assert np.isclose(delta, 0.0, atol=1e-10), f"Planar molecule {syms} defect {delta} != 0"

    # Non-planar molecules
    for syms, coords in [methane_geometry, ethanol_geometry]:
        masses = enforce_ciaaw_masses(syms)
        _, moments, _ = diagonalize_principal_axes(coords, masses)
        delta = compute_inertial_defect(moments)
        assert delta < -0.1, f"Non-planar molecule {syms} defect {delta} should be negative"


# ============================================================================
# 9. Air-Gap Architecture & Serialization Tests
# ============================================================================

def test_serialize_eckart_matrix_air_gap_violation() -> None:
    """
    Adversarially asserts that attempting to serialize Eckart matrices into
    Ring 1 Domain A static repository space raises AirGapViolationError. [D]
    """
    repo_root = CoChemPathManager.get_repo_root()
    forbidden_target = repo_root / "Libraries" / "forbidden_provenance.json"

    identity_rot = np.eye(3)
    origin_com = np.zeros(3)

    with pytest.raises(AirGapViolationError) as exc_info:
        serialize_eckart_matrix(identity_rot, origin_com, forbidden_target)

    assert "CRITICAL AIR-GAP BREACH" in str(exc_info.value)
    assert not forbidden_target.exists()


def test_serialize_eckart_matrix_success() -> None:
    """
    Verifies atomic serialization to dynamic scratch / artifacts tier (Ring 2 / Ring 3)
    and validates SHA-256 and JSON contents. [D]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "fit_provenance.json"

        rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        com = np.array([1.23, 4.56, 7.89])
        meta = {"molecule": "test_system", "method": "CCSD(T)-F12"}

        written_path = serialize_eckart_matrix(
            rot, com, target_path, metadata=meta, provenance_tag="[M]"
        )

        assert written_path.exists()
        with open(written_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["provenance"] == "[M]"
        assert np.isclose(data["det_r"], 1.0, atol=1e-12)
        assert data["com_vector"] == [1.23, 4.56, 7.89]
        assert data["rotation_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert data["metadata"]["molecule"] == "test_system"
        assert "sha256" in data


# ============================================================================
# 10. High-Level EckartAligner & EckartAlignmentResult Model Tests
# ============================================================================

def test_eckart_aligner_pipeline_water(water_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies the high-level EckartAligner and EckartAlignmentResult Pydantic model
    for water alignment and coordinate inversion. [D]
    """
    syms, coords = water_geometry
    aligner = EckartAligner(symbols=syms, provenance_tag="[D]")

    result = aligner.align(coords, metadata={"author": "CoChem-TORQ"})

    assert isinstance(result, EckartAlignmentResult)
    assert result.rotor_type == "asymmetric_top"
    assert result.det_r == 1.0
    assert result.provenance == "[D]"
    assert result.metadata["author"] == "CoChem-TORQ"

    # Test numpy export
    np_aligned = result.to_numpy()
    assert isinstance(np_aligned, np.ndarray)
    assert np_aligned.shape == (3, 3)

    # Test dictionary export
    dict_repr = result.to_dict()
    assert dict_repr["rotor_type"] == "asymmetric_top"
    assert len(dict_repr["aligned_geometry"]) == 3

    # Test inversion method on result object
    reconstructed = result.invert_coordinates()
    assert np.allclose(reconstructed, coords, atol=1e-12)


def test_eckart_aligner_align_molecule_classmethod(fluoropropane_geometry: tuple[list[str], np.ndarray]) -> None:
    """
    Verifies the convenience EckartAligner.align_molecule classmethod on 2-fluoropropane. [D]
    """
    syms, coords = fluoropropane_geometry
    result = EckartAligner.align_molecule(syms, coords, provenance_tag="[E]")

    assert result.rotor_type == "asymmetric_top"
    assert result.provenance == "[E]"
    assert len(result.exact_masses) == 11
    assert np.isclose(result.det_r, 1.0, atol=1e-12)

    # Invert and verify precision
    reconstructed = result.invert_coordinates()
    assert np.allclose(reconstructed, coords, atol=1e-12)


def test_eckart_result_serialization() -> None:
    """Verifies that EckartAlignmentResult.serialize correctly exports to disk."""
    syms = ["O", "H", "H"]
    r_oh = 0.9578
    th = np.radians(104.5)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh * np.sin(th / 2.0), r_oh * np.cos(th / 2.0), 0.0],
            [-r_oh * np.sin(th / 2.0), r_oh * np.cos(th / 2.0), 0.0],
        ]
    )
    result = EckartAligner.align_molecule(syms, coords)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "water_eckart.json"
        saved_path = result.serialize(out_file)

        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["rotor_type"] == "asymmetric_top"
        assert len(data["metadata"]["symbols"]) == 3
