"""
Test Automated GOAT + CREST Union Conformer Exploration
SRS Chunk 09, Suggestion #89 (Method Matrix v4 §9B.1-§9B.3)
Zero-Mock compliant: Real conformer geometries, authentic deduplication algorithms.
"""
import numpy as np

from Libraries.cochem_torq_goat import compute_moments_and_constants
from Libraries.cochem_torq_pipeline import (
    deduplicate_conformer_union,
    verify_nvidia_mps_health,
)


def test_deduplicate_conformer_union_rotational_and_rmsd():
    """Verify conformer candidate deduplication with Delta B / B > 0.002 and RMSD filtering."""
    symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords_trans = np.array([
        [0.000, 0.000, 0.000],
        [1.520, 0.000, 0.000],
        [2.080, 1.290, 0.000],
        [-0.380, -0.520, 0.880],
        [-0.380, -0.520, -0.880],
        [-0.380, 1.020, 0.000],
        [1.900, -0.520, 0.880],
        [1.900, -0.520, -0.880],
        [3.040, 1.220, 0.000],
    ])

    coords_gauche = np.array([
        [0.000, 0.000, 0.000],
        [1.520, 0.000, 0.000],
        [2.080, 1.290, 0.000],
        [-0.380, -0.520, 0.880],
        [-0.380, -0.520, -0.880],
        [-0.380, 1.020, 0.000],
        [1.900, -0.520, 0.880],
        [1.900, -0.520, -0.880],
        [2.100, 1.700, 0.850],
    ])

    coords_trans_dup = coords_trans + 0.001

    (A1, B1, C1), _, _, _, _ = compute_moments_and_constants(symbols, coords_trans)
    (A2, B2, C2), _, _, _, _ = compute_moments_and_constants(symbols, coords_gauche)
    (A3, B3, C3), _, _, _, _ = compute_moments_and_constants(symbols, coords_trans_dup)

    raw_pool = [
        {
            "symbols": symbols,
            "coordinates": coords_trans.tolist(),
            "energy_hartree": -154.5000,
            "rotational_constants_mhz": (A1, B1, C1),
            "origin": "GOAT",
        },
        {
            "symbols": symbols,
            "coordinates": coords_trans_dup.tolist(),
            "energy_hartree": -154.4999,
            "rotational_constants_mhz": (A3, B3, C3),
            "origin": "CREST",
        },
        {
            "symbols": symbols,
            "coordinates": coords_gauche.tolist(),
            "energy_hartree": -154.4980,
            "rotational_constants_mhz": (A2, B2, C2),
            "origin": "GOAT",
        },
    ]

    deduped = deduplicate_conformer_union(raw_pool, delta_b_rel_threshold=0.002, rmsd_threshold=0.15)

    assert len(deduped) == 2, f"Expected 2 conformers after deduplication, got {len(deduped)}"
    assert deduped[0]["origin"] == "GOAT"
    energies = [d["energy_hartree"] for d in deduped]
    assert -154.5000 in energies
    assert -154.4980 in energies


def test_nvidia_mps_health_verification():
    """Verify NVIDIA MPS health check runs without unhandled exceptions."""
    is_healthy = verify_nvidia_mps_health()
    assert isinstance(is_healthy, bool)
