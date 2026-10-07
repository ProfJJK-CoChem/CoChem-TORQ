"""Unqualified CFOUR capability must reject without a stand-in executable.

No CFOUR binary or GENBAS archive is synthesized. Positive site qualification
requires the actual executable, genuine basis library, and native derivative data.
"""

from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_torq_cfour_bridge import CFOURUnavailableError, TorqCfourExecutor
from Libraries.cochem_torq_engine import DispatchPayload, route_method_matrix


def test_cfour_missing_dependency_raises_tool_unavailable(tmp_path: Path):
    missing = tmp_path / "xcfour-not-provisioned"
    executor = TorqCfourExecutor(cfour_path=str(missing))
    with pytest.raises(CFOURUnavailableError, match="not configured or executable"):
        executor.resolve_binary()
    assert not missing.exists()


def test_cfour_valid_environment_routes_to_cfour_executor(tmp_path: Path):
    """A requested coupled-cluster method cannot invent its unsupported derivatives."""
    payload = DispatchPayload(
        symbols=["C", "O"],
        coordinates=[[0, 0, 0], [0, 0, 1.13]],
        charge=0,
        multiplicity=1,
        method="CCSD(T)",
        basis_set="ANO0",
    )
    executor = TorqCfourExecutor(cfour_path=str(tmp_path / "not-provisioned-xcfour"))
    with pytest.raises(
        CFOURUnavailableError, match="qualified.*adapter|separately qualified"
    ):
        executor.execute(payload)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tier", ["T3C-3d", "T4C-1mo", "CFOUR"])
def test_unqualified_cfour_routing_never_imports_base_or_substitutes_a_method(tier):
    with pytest.raises(NotImplementedError, match="no qualified.*derivative adapter"):
        route_method_matrix(["C", "O"], [[0, 0, 0], [0, 0, 1.13]], target_tier=tier)


def test_cfour_tier_requires_a_qualified_adapter_before_environment_import():
    with pytest.raises(NotImplementedError, match="qualified"):
        route_method_matrix(
            ["C", "O"], np.array([[0, 0, 0], [0, 0, 1.13]]), target_tier="T3C-3d"
        )
