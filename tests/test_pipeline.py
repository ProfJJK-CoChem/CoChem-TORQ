# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
Empirical pipeline tests for CoChem-TORQ.
Strictly adheres to Anti-Spoofing Directives: zero mock patches of unimplemented physics steps.
"""

import numpy as np
import pytest
from Libraries.torq_config import TorqRunParams
from Libraries.cochem_torq_pipeline import (
    TorqPipeline,
    normalize_and_validate_payload,
)


@pytest.fixture
def valid_config() -> TorqRunParams:
    return TorqRunParams(
        tier="t1",
        wall_time_tier="normal",
        engine="orca",
        method="B3LYP",
        basis_set="def2-SVP",
        keywords=["Opt", "Freq"],
    )


def test_pipeline_initialization(valid_config: TorqRunParams) -> None:
    pipeline = TorqPipeline(valid_config)
    assert pipeline.config.method == "B3LYP"
    assert pipeline.config.basis_set == "def2-SVP"
    assert pipeline.state == "S_0"
    assert pipeline.state_history == ["S_0"]


def test_normalize_and_validate_payload_aliases() -> None:
    # Test 'atoms' and 'coords'
    p1 = {"atoms": ["H", "H"], "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]}
    syms1, coords1 = normalize_and_validate_payload(p1)
    assert syms1 == ["H", "H"]
    assert coords1.shape == (2, 3)

    # Test 'symbols' and 'coordinates'
    p2 = {"symbols": ["C", "O"], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]}
    syms2, coords2 = normalize_and_validate_payload(p2)
    assert syms2 == ["C", "O"]
    assert coords2.shape == (2, 3)

    # Test 'elements' and 'geometry'
    p3 = {"elements": ["N", "N"], "geometry": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]}
    syms3, coords3 = normalize_and_validate_payload(p3)
    assert syms3 == ["N", "N"]
    assert coords3.shape == (2, 3)


def test_normalize_and_validate_payload_validation_errors() -> None:
    # Missing / empty dict
    with pytest.raises(ValueError, match=r"\[MISSING DATA\]"):
        normalize_and_validate_payload({})

    # Missing symbols/atoms
    with pytest.raises(ValueError, match=r"\[INVALID_PAYLOAD\]"):
        normalize_and_validate_payload({"coords": [[0.0, 0.0, 0.0]]})

    # Missing coordinates
    with pytest.raises(ValueError, match=r"\[INVALID_PAYLOAD\]"):
        normalize_and_validate_payload({"symbols": ["H"]})

    # Dimension mismatch: coords not shape (N, 3)
    with pytest.raises(ValueError, match=r"\[INVALID_PAYLOAD\]"):
        normalize_and_validate_payload({"symbols": ["H"], "coords": [0.0, 0.0, 0.0]})

    # Count mismatch between symbols and coords
    with pytest.raises(ValueError, match=r"\[INVALID_PAYLOAD\]"):
        normalize_and_validate_payload(
            {"symbols": ["H", "H"], "coords": [[0.0, 0.0, 0.0]]}
        )

    # Invalid atomic symbol lookup via Mendeleev
    with pytest.raises(ValueError, match=r"\[INVALID_SYMBOL\]"):
        normalize_and_validate_payload(
            {"symbols": ["FakeElement"], "coords": [[0.0, 0.0, 0.0]]}
        )


def test_pipeline_execution_state_transitions(valid_config: TorqRunParams) -> None:
    """
    Verifies that calling the pipeline with a physical geometry payload progresses
    through all required state transitions to completion.
    """
    pipeline = TorqPipeline(valid_config)

    # Attempting to run without payload should raise ValueError and set state to S_FAILED
    with pytest.raises(ValueError, match=r"\[MISSING DATA\]"):
        pipeline.run({})
    assert pipeline.state == "S_FAILED"

    fresh_pipeline = TorqPipeline(valid_config)
    payload = {
        "atoms": ["C", "H", "H", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
    }
    result = fresh_pipeline.run(payload)

    assert result["status"] == "success"
    assert result["processed_payload"] == payload
    assert fresh_pipeline.state == "S_COMPLETE"

    expected_history = [
        "S_0",
        "S_TOPOLOGY",
        "S_MLFF_FILTER",
        "S_CONFORMER_SEARCH",
        "S_QUANTUM_OPT",
        "S_SPECTRAL_SYNTHESIS",
        "S_EXPORT",
        "S_COMPLETE",
    ]
    assert fresh_pipeline.state_history == expected_history
    assert "mlff_results" in result
    assert "conformer_results" in result
    assert "spectral_results" in result
    assert "export_metadata" in result


def test_pipeline_config_overrides_forwarding() -> None:
    """Verifies that TorqRunParams user configuration overrides are forwarded without dropping."""
    custom_config = TorqRunParams(
        tier="t3",
        wall_time_tier="normal",
        engine="orca",
        method="wB97X-D4",
        basis_set="def2-TZVP",
        keywords=["Opt", "TightSCF"],
        dispersion="D4",
        anharmonicity="VPT2",
        cabs_mappings={"def2-TZVP": "def2-TZVP-CABS"},
    )
    pipeline = TorqPipeline(custom_config)
    payload = {
        "atoms": ["O", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        "charge": 0,
        "multiplicity": 1,
    }
    result = pipeline.run(payload)
    assert result["status"] == "success"
    assert pipeline.state == "S_COMPLETE"
    assert result["export_metadata"]["method"] == "wB97X-D4"
    assert result["export_metadata"]["tier"] == "t3"
    assert result["export_metadata"]["charge"] == 0
    assert result["export_metadata"]["multiplicity"] == 1


def test_normalize_and_validate_payload_isotopic_symbols() -> None:
    """Verifies that Mendeleev isotope and alias symbols (e.g. D, 13C, 18O) are supported."""
    payload = {
        "symbols": ["13C", "18O", "D", "T"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.13],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    }
    syms, coords = normalize_and_validate_payload(payload)
    assert syms == ["13C", "18O", "D", "T"]
    assert coords.shape == (4, 3)

    # Invalid isotope
    with pytest.raises(ValueError, match=r"\[INVALID_SYMBOL\]"):
        normalize_and_validate_payload(
            {"symbols": ["9999C"], "coords": [[0.0, 0.0, 0.0]]}
        )


