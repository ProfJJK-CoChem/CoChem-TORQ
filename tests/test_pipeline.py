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
    """Rejected/unsupported input cannot advance through invented science stages."""
    pipeline = TorqPipeline(valid_config)
    with pytest.raises(ValueError):
        pipeline.run({})
    assert pipeline.state == "S_FAILED"
    rejected = TorqPipeline(valid_config.model_copy(update={"engine": "CFOUR"}))
    payload = {
        "symbols": ["H", "H"],
        "coordinates": [[0, 0, 0], [0, 0, 0.74]],
        "charge": 0,
        "multiplicity": 1,
    }
    with pytest.raises(NotImplementedError, match="not implemented"):
        rejected.run(payload)
    assert rejected.state == "S_FAILED"
    assert "S_QUANTUM_OPT" not in rejected.state_history
    assert "S_SPECTRAL_SYNTHESIS" not in rejected.state_history


def test_pipeline_config_overrides_forwarding() -> None:
    """An explicitly requested unsupported engine is retained and rejected."""
    config = TorqRunParams(
        tier="T1",
        wall_time_tier="normal",
        engine="CFOUR",
        method="CCSD(T)",
        basis_set="cc-pVDZ",
        keywords=["Opt"],
    )
    pipeline = TorqPipeline(config)
    payload = {
        "symbols": ["H", "H"],
        "coordinates": [[0, 0, 0], [0, 0, 0.74]],
        "charge": 0,
        "multiplicity": 1,
    }
    with pytest.raises(NotImplementedError, match="CFOUR"):
        pipeline.run(payload)
    assert pipeline.config.engine == "CFOUR"
    assert pipeline.config.method == "CCSD(T)"
    assert pipeline.config.basis_set == "cc-pVDZ"
    assert pipeline.state == "S_FAILED"


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
