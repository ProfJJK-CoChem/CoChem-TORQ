"""Comprehensive Zero-Mock Physical Test Suite for UI/Start_TORQ.ipynb.

Validates the primary autocratic Jupyter UI entry point required by:
- SRS Task 1 (Section 2) & Task 3 (Jupyter Backend & Master DOM Orchestrator)
- Readme.md (Section 2 & 7)
- Method Matrix v4 (§4.4, §8A, §8B, §8C, §9B, §10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final
import pytest
import numpy as np

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH: Final[Path] = REPO_ROOT / "UI" / "Start_TORQ.ipynb"


def test_start_torq_file_exists_and_valid_json() -> None:
    """Verifies Start_TORQ.ipynb exists, is non-empty, and parses as valid JSON."""
    assert NOTEBOOK_PATH.exists(), f"Notebook file not found at {NOTEBOOK_PATH}"
    assert NOTEBOOK_PATH.is_file(), f"Notebook path is not a file: {NOTEBOOK_PATH}"
    assert NOTEBOOK_PATH.stat().st_size > 0, "Notebook file is empty"

    content = NOTEBOOK_PATH.read_text(encoding="utf-8")
    data = json.loads(content)

    assert "cells" in data, "Notebook missing 'cells' key"
    assert "metadata" in data, "Notebook missing 'metadata' key"
    assert "nbformat" in data, "Notebook missing 'nbformat' key"
    assert data["nbformat"] == 4, f"Expected nbformat 4, got {data['nbformat']}"


def test_start_torq_structure_and_phases() -> None:
    """Verifies that all 10 phases and required code cells are present."""
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = data["cells"]
    
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    
    assert len(code_cells) >= 10, f"Expected at least 10 code cells, found {len(code_cells)}"
    assert len(markdown_cells) >= 8, f"Expected at least 8 markdown cells, found {len(markdown_cells)}"
    
    # Required tokens and phase identifiers across cells
    all_source = " ".join("".join(c.get("source", [])) for c in cells)
    
    assert "JAX_ENABLE_X64" in all_source
    assert "normalize_and_validate_payload" in all_source
    assert "mendeleev" in all_source
    assert "EckartAligner" in all_source
    assert "TorqTopology" in all_source
    assert "deduplicate_stage_b_spectroscopic" in all_source
    assert "route_cascade_rules" in all_source
    assert "fit_continuous_splines" in all_source
    assert "build_dvr_hamiltonian" in all_source
    assert "build_complete_spcat_payload" in all_source
    assert "pyarrow.parquet" in all_source or "pq.write_table" in all_source
    assert "plotly" in all_source


def test_start_torq_headless_execution() -> None:
    """Physically executes Start_TORQ.ipynb headless and verifies zero-error execution."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT / "UI")}})
    
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Cell {idx+1} ({cell.get('id')}) encountered error: {out.get('ename')}: {out.get('evalue')}"
            )
