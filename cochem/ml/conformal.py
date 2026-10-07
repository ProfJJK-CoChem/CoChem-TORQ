"""Inductive Split-Conformal Prediction and Force Calibration (Suggestion #54 / Method Matrix v4 §12.5, §19 [M], [D])."""

from __future__ import annotations

import sys
from pathlib import Path

_torq_libs = (Path(__file__).resolve().parent.parent.parent.parent / "CoChem-TORQ" / "Libraries").resolve()
if _torq_libs.is_dir() and str(_torq_libs) not in sys.path:
    sys.path.insert(0, str(_torq_libs))

from Libraries.cochem_torq_conformal import (
    CalibrationSample,
    ConformalPredictor,
    ConformalCalibrationError,
)
from Libraries.cochem_torq_inference_schemas import ConformalInterval, ConformalPredictorConfig

__all__ = [
    "CalibrationSample",
    "ConformalCalibrationError",
    "ConformalInterval",
    "ConformalPredictor",
    "ConformalPredictorConfig",
]
