"""Delta-ML Architecture and D3(BJ) Dispersion Layer (Suggestion #60 / Method Matrix v4 §9A.5 [M], [D])."""

from __future__ import annotations

import sys
from pathlib import Path

_torq_libs = (Path(__file__).resolve().parent.parent.parent.parent / "CoChem-TORQ" / "Libraries").resolve()
if _torq_libs.is_dir() and str(_torq_libs) not in sys.path:
    sys.path.insert(0, str(_torq_libs))

from Libraries.cochem_torq_delta_ml import (
    BaselinePhysicsEngine,
    DeltaMLEngine,
    EMTBaselineEngine,
    GFN2xTBEngine,
    LennardJonesBaselineEngine,
    PM6Engine,
    UnitHarmonizer,
)
from Libraries.cochem_torq_dispersion_d3 import (
    DispersionD3Layer,
    compute_coordination_numbers,
)
from Libraries.cochem_torq_inference_schemas import DispersionD3Config
from cochem_base.schemas import DeltaMLDispersionConfig

__all__ = [
    "BaselinePhysicsEngine",
    "DeltaMLDispersionConfig",
    "DeltaMLEngine",
    "DispersionD3Config",
    "DispersionD3Layer",
    "EMTBaselineEngine",
    "GFN2xTBEngine",
    "LennardJonesBaselineEngine",
    "PM6Engine",
    "UnitHarmonizer",
    "compute_coordination_numbers",
]
