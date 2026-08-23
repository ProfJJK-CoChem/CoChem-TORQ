# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
Empirical pipeline tests for CoChem-TORQ.
Strictly adheres to Anti-Spoofing Directives: zero mock patches of unimplemented physics steps.
"""

import pytest
from Libraries.torq_config import TorqRunParams
from Libraries.cochem_torq_pipeline import TorqPipeline

@pytest.fixture
def valid_config():
    return TorqRunParams(
        tier="t1",
        wall_time_tier="normal",
        engine="orca",
        method="B3LYP",
        basis_set="def2-SVP",
        keywords=["Opt", "Freq"]
    )

def test_pipeline_initialization(valid_config):
    pipeline = TorqPipeline(valid_config)
    assert pipeline.config.method == "B3LYP"
    assert pipeline.config.basis_set == "def2-SVP"
    assert pipeline.state == "S_0"

def test_pipeline_execution_state_transitions(valid_config):
    """
    Verifies that calling the pipeline with a physical geometry payload progresses
    through all required state transitions to completion.
    """
    pipeline = TorqPipeline(valid_config)
    
    # Attempting to run without payload should raise ValueError
    with pytest.raises(ValueError, match=r"\[MISSING DATA\]"):
        pipeline.run({})
        
    payload = {"atoms": ["C", "H", "H", "H", "H"], "coords": [[0,0,0], [1,1,1], [-1,-1,1], [1,-1,-1], [-1,1,-1]]}
    result = pipeline.run(payload)
    
    assert result["status"] == "success"
    assert result["processed_payload"] == payload
    assert pipeline.state == "S_COMPLETE"
