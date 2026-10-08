"""Deprecated in-process chain APIs and an explicit isolated BASE handoff bridge.

BASE chain implementation belongs to its separately verified environment.
This facade never substitutes a model potential for an electronic calculation.
"""

from __future__ import annotations

from typing import Any

from cochem_torq.base_bridge import BaseHandoffBridge


class LegacyChainUnavailableError(ImportError):
    """Former in-process BASE/fictional-physics APIs have no TORQ authority."""


LEGACY_NAMES = frozenset(
    {
        "Chain",
        "ChainStage",
        "Stage",
        "StateChainingAuditor",
        "ExecutionArrow",
        "StateRecord",
        "CanonicalArrow",
        "CounterpoiseType",
        "CANONICAL_ARROWS",
        "get_atomic_mass",
        "get_isotopic_mass",
        "MissingBinaryError",
        "ConvergenceFailureError",
        "CorruptOutputError",
        "compute_rotational_properties",
        "analyze_hessian_and_normal_modes",
        "evaluate_vdw_potential_and_derivatives",
    }
)


def __getattr__(name: str) -> Any:
    if name in LEGACY_NAMES:
        raise LegacyChainUnavailableError(
            f"Libraries.chain.{name} is unavailable in TORQ. BASE chain code "
            "belongs to its isolated environment. Use BaseHandoffBridge for "
            "verified geometry transfer or cochem_torq.application for explicit "
            "qualified TORQ calculations; fabricated/force_fallback execution "
            "has no supported replacement."
        )
    raise AttributeError(name)


__all__ = ["BaseHandoffBridge", "LegacyChainUnavailableError"]

if __name__ == "__main__":
    raise SystemExit(
        "The former chain-script interface is unavailable. Use TORQ's explicit "
        "calculation CLI or BaseHandoffBridge with the separately verified BASE "
        "environment. --force-fallback is unsupported."
    )
