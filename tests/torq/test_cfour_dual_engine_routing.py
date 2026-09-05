"""
Test Dual-Engine CFOUR Execution Routing & CCSD(T) Integration
SRS Chunk 09, Suggestion #88 (Method Matrix v4 §9 & §13-§14)
Zero-Mock compliant: Real environment inspection, anti-demotion mandate verification.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

from cochem_base.exceptions import ToolUnavailableError
from Libraries.cochem_torq_cfour_bridge import TorqCfourExecutor
from Libraries.cochem_torq_engine import route_method_matrix


def test_cfour_missing_dependency_raises_tool_unavailable():
    """In an environment without GENBAS or CFOUR binaries, assert ToolUnavailableError is raised."""
    old_genbas = os.environ.pop("GENBAS", None)
    symbols = ["C", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])

    try:
        with pytest.raises(ToolUnavailableError) as exc_info:
            route_method_matrix(symbols, coords, target_tier="T3C-3d")

        err = exc_info.value
        assert "ERR_TOOL_UNAVAILABLE" in str(err)
        assert "CFOUR binary or GENBAS library is not configured" in str(err)
        assert "Silent demotion to DFT is strictly prohibited" in str(err)
    finally:
        if old_genbas:
            os.environ["GENBAS"] = old_genbas


def test_cfour_valid_environment_routes_to_cfour_executor():
    """Request tier T3C-3d with configured environment; verify routing to TorqCfourExecutor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genbas_path = os.path.join(tmpdir, "GENBAS")
        with open(genbas_path, "w", encoding="utf-8") as f:
            f.write("C:ANO1\nO:ANO1\n")

        xcfour_name = "xcfour.exe" if sys.platform == "win32" else "xcfour"
        xcfour_path = os.path.join(tmpdir, xcfour_name)
        with open(xcfour_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho CFOUR\n")
        os.chmod(xcfour_path, 0o755)

        old_genbas = os.environ.get("GENBAS")
        old_path = os.environ.get("PATH", "")
        try:
            os.environ["GENBAS"] = genbas_path
            os.environ["PATH"] = tmpdir + os.pathsep + old_path

            symbols = ["C", "O"]
            coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])

            payload = route_method_matrix(symbols, coords, target_tier="T3C-3d")

            assert payload.executor == "TorqCfourExecutor"
            assert payload.metadata.get("executor") == "TorqCfourExecutor"
            assert payload.method == "CCSD(T)"

            executor = TorqCfourExecutor()
            resolved_bin = executor.resolve_binary()
            assert resolved_bin.exists()
        finally:
            if old_genbas:
                os.environ["GENBAS"] = old_genbas
            else:
                os.environ.pop("GENBAS", None)
            os.environ["PATH"] = old_path
