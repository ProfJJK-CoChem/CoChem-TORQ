"""CoChem-TORQ OET Client Library Wrapper.

Exports OETClient, PhysicalOETFallbackCalculator, emit_fallback_alert, and utilities.
Method Matrix Reference: Method Matrix v4 §10.8 (Active Learning & Fallback Provenance Auditing) [M].
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts directory is in sys.path
_scripts_dir = (Path(__file__).resolve().parent.parent / "scripts").resolve()
if _scripts_dir.is_dir() and str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from oet_client import (
    AirGapViolationError,
    EngradResult,
    ExtInpData,
    OETClient,
    OETClientConfig,
    OETDaemonConnectionError,
    OETFallbackAlertManifest,
    PhysicalOETFallbackCalculator,
    convert_ase_forces_to_orca_gradient,
    convert_orca_gradient_to_ase_forces,
    emit_fallback_alert,
    read_extinp,
    read_xyz,
    run_oet_client,
    write_engrad,
    write_xyz,
)

__all__ = [
    "AirGapViolationError",
    "EngradResult",
    "ExtInpData",
    "OETClient",
    "OETClientConfig",
    "OETDaemonConnectionError",
    "OETFallbackAlertManifest",
    "PhysicalOETFallbackCalculator",
    "convert_ase_forces_to_orca_gradient",
    "convert_orca_gradient_to_ase_forces",
    "emit_fallback_alert",
    "read_extinp",
    "read_xyz",
    "run_oet_client",
    "write_engrad",
    "write_xyz",
]
