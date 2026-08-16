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

def test_pipeline_unimplemented_step_raises_honest_error(valid_config):
    """
    Verifies that calling an unimplemented state transition raises NotImplementedError
    rather than being bypassed with mocks or synthetic state assertions.
    """
    pipeline = TorqPipeline(valid_config)
    with pytest.raises(NotImplementedError, match=r"\[MISSING DATA\]"):
        pipeline.run()
