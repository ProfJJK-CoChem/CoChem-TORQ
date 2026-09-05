"""Neural Network & Kernel Physics Models (Suggestions #53, #55, #58 / Method Matrix v4 §8C, §10.3 [M], [D])."""

from __future__ import annotations

import sys
from pathlib import Path

_torq_libs = (Path(__file__).resolve().parent.parent.parent.parent / "CoChem-TORQ" / "Libraries").resolve()
if _torq_libs.is_dir() and str(_torq_libs) not in sys.path:
    sys.path.insert(0, str(_torq_libs))

from Libraries.cochem_torq_ani2x_transfer import (
    ANI2xModel,
    cosine_cutoff_envelope,
    quintic_c2_envelope,
)
from Libraries.cochem_torq_force_matching import (
    ForceMatchingLoss,
    compute_angular_cosine_similarity,
    compute_conservative_forces,
    huber_force_loss,
    huber_scalar_loss,
)
from cochem_base.core_engine.cochem_core_auto_pes import (
    ExactKernelRidgeEstimator,
)
from cochem_base.schemas import (
    ANI2xCutoffConfig,
    ForceMatchingLossConfig,
    KrrRegularizationConfig,
)

__all__ = [
    "ANI2xCutoffConfig",
    "ANI2xModel",
    "ExactKernelRidgeEstimator",
    "ForceMatchingLoss",
    "ForceMatchingLossConfig",
    "KrrRegularizationConfig",
    "compute_angular_cosine_similarity",
    "compute_conservative_forces",
    "cosine_cutoff_envelope",
    "huber_force_loss",
    "huber_scalar_loss",
    "quintic_c2_envelope",
]
