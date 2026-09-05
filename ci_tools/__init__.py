"""CoChem-BASE CI Tools Package.

Pre-flight air-gap enforcement, anti-spoofing, and integrity validation utilities
for continuous integration pipelines across Windows, Linux, and macOS platforms.
"""

from __future__ import annotations

from .ci_airgap_sweep import (
    AirgapSweepSummary,
    AirgapViolation,
    calculate_shannon_entropy,
    check_config_pollution,
    format_airgap_report,
    inspect_magic_number,
    is_restricted_extension,
    is_xyz_coordinate_payload,
    run_airgap_sweep,
)

__all__ = [
    "AirgapSweepSummary",
    "AirgapViolation",
    "calculate_shannon_entropy",
    "check_config_pollution",
    "format_airgap_report",
    "inspect_magic_number",
    "is_restricted_extension",
    "is_xyz_coordinate_payload",
    "run_airgap_sweep",
]
