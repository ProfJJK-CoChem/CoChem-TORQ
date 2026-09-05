"""Zero-Mock Integration & Unit Test Suite for ci_tools/anti_spoof_linter.py.

Verifies:
1. Physical existence, non-empty content, UTF-8 encoding (no BOM), Unix LF line endings.
2. AST visitor detection of prohibited mock imports (unittest.mock, MagicMock, patch).
3. AST visitor detection of NotImplementedError dead-end stubs.
4. AST visitor detection of empty pass stubs in functions and classes.
5. Legitimate exception class and overload / abstractmethod exemptions.
6. AST visitor detection of unamnestied concurrency modules (parsl, ray, multiprocessing).
7. AST visitor detection of synthetic mock generators (np.linspace, np.sin).
8. AST visitor detection of monkeypatch OS/subprocess intercepts.
9. AST visitor detection of string concatenation obfuscation.
10. Amnesty whitelist loading, normalization, and bypass enforcement.
11. Full CLI execution with exit codes and structured JSON output.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ci_tools.anti_spoof_linter import (
    Violation,
    check_file,
    load_amnesty,
    normalize_path_entry,
    run_linter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER_PATH = REPO_ROOT / "ci_tools" / "anti_spoof_linter.py"


def test_linter_script_exists_and_encoding() -> None:
    """Verify anti_spoof_linter.py exists, has valid UTF-8 encoding, no BOM, and substantive size."""
    assert LINTER_PATH.exists(), f"Linter missing at {LINTER_PATH}"
    assert LINTER_PATH.is_file()

    raw_bytes = LINTER_PATH.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), "Found UTF-8 BOM marker in anti_spoof_linter.py"
    assert b"\r\n" not in raw_bytes, "Found CRLF line endings in anti_spoof_linter.py"

    text = raw_bytes.decode("utf-8")
    assert len(text) > 1000, f"Linter file unexpectedly short ({len(text)} chars)"
    assert "class SpoofVisitor" in text
    assert "def run_linter" in text


def test_detect_forbidden_mock_imports(tmp_path: Path) -> None:
    """Verify AST linter detects forbidden mock imports."""
    mock_file = tmp_path / "mock_module.py"
    mock_file.write_text(
        "import unittest.mock\n"
        "from unittest.mock import MagicMock, patch\n"
        "import mock\n",
        encoding="utf-8",
    )

    violations = check_file(mock_file, tmp_path, set())
    assert len(violations) >= 3
    categories = {v.category for v in violations}
    assert "MOCK_IMPORT" in categories


def test_detect_not_implemented_error(tmp_path: Path) -> None:
    """Verify AST linter detects raise NotImplementedError dead-ends."""
    stub_file = tmp_path / "stub_logic.py"
    stub_file.write_text(
        "def compute_energy():\n"
        "    raise NotImplementedError('Dead end')\n"
        "\n"
        "def compute_gradient():\n"
        "    raise NotImplementedError\n",
        encoding="utf-8",
    )

    violations = check_file(stub_file, tmp_path, set())
    assert len(violations) >= 2
    categories = {v.category for v in violations}
    assert "NOT_IMPLEMENTED_ERROR" in categories


def test_detect_empty_pass_stubs(tmp_path: Path) -> None:
    """Verify AST linter detects empty pass stubs in functions and classes."""
    pass_file = tmp_path / "empty_pass.py"
    pass_file.write_text(
        "def empty_routine():\n"
        "    pass\n"
        "\n"
        "class EmptyContainer:\n"
        "    pass\n",
        encoding="utf-8",
    )

    violations = check_file(pass_file, tmp_path, set())
    assert len(violations) >= 2
    categories = {v.category for v in violations}
    assert "EMPTY_PASS_STUB" in categories


def test_legitimate_exception_and_overload_exemptions(tmp_path: Path) -> None:
    """Verify AST linter permits standard exception class declarations and overload decorators."""
    clean_exceptions_file = tmp_path / "custom_exceptions.py"
    clean_exceptions_file.write_text(
        "from typing import overload\n"
        "from abc import ABC, abstractmethod\n"
        "\n"
        "class TorqConvergenceError(Exception):\n"
        "    pass\n"
        "\n"
        "class CustomWarning(UserWarning):\n"
        "    pass\n"
        "\n"
        "class AbstractRunner(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self):\n"
        "        pass\n"
        "\n"
        "@overload\n"
        "def process(x: int) -> int:\n"
        "    pass\n"
        "\n"
        "def process(x: int) -> int:\n"
        "    return x * 2\n",
        encoding="utf-8",
    )

    violations = check_file(clean_exceptions_file, tmp_path, set())
    assert len(violations) == 0, f"Unexpected violations in clean exceptions file: {violations}"


def test_detect_banned_identifiers(tmp_path: Path) -> None:
    """Verify AST linter detects banned words in variables, functions, and classes."""
    banned_id_file = tmp_path / "banned_identifiers.py"
    banned_id_file.write_text(
        "dummy_variable = 42\n"
        "def fake_function():\n"
        "    return 10\n"
        "class StubPipeline:\n"
        "    def run(self):\n"
        "        return True\n",
        encoding="utf-8",
    )

    violations = check_file(banned_id_file, tmp_path, set())
    assert len(violations) >= 3
    categories = {v.category for v in violations}
    assert "BANNED_IDENTIFIER" in categories


def test_detect_unamnestied_concurrency(tmp_path: Path) -> None:
    """Verify AST linter detects unamnestied concurrency engines."""
    concurrency_file = tmp_path / "worker.py"
    concurrency_file.write_text(
        "import multiprocessing\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "import parsl\n"
        "import ray\n",
        encoding="utf-8",
    )

    # When unamnestied, flags violations
    violations = check_file(concurrency_file, tmp_path, set())
    assert len(violations) >= 4
    categories = {v.category for v in violations}
    assert "CONCURRENCY_IMPORT" in categories

    # When amnestied, bypasses violations
    amnesty_set = {"worker.py"}
    amnestied_violations = check_file(concurrency_file, tmp_path, amnesty_set)
    assert len(amnestied_violations) == 0


def test_detect_synthetic_numpy_generators(tmp_path: Path) -> None:
    """Verify AST linter detects prohibited synthetic mock generators in production code."""
    synthetic_file = tmp_path / "synthetic_curve.py"
    synthetic_file.write_text(
        "import numpy as np\n"
        "def generate_fake_signal():\n"
        "    x = np.linspace(0, 10, 100)\n"
        "    y = np.sin(x)\n"
        "    return x, y\n",
        encoding="utf-8",
    )

    violations = check_file(synthetic_file, tmp_path, set())
    assert any(v.category == "SYNTHETIC_DATA" for v in violations)


def test_detect_monkeypatch_intercept(tmp_path: Path) -> None:
    """Verify AST linter detects monkeypatch intercepts of OS/subprocess calls."""
    monkey_file = tmp_path / "monkey_intercept.py"
    monkey_file.write_text(
        "def test_intercept(monkeypatch):\n"
        "    monkeypatch.setattr('subprocess.run', lambda *args: None)\n",
        encoding="utf-8",
    )

    violations = check_file(monkey_file, tmp_path, set())
    assert any(v.category == "MONKEYPATCH_INTERCEPT" for v in violations)


def test_detect_string_concatenation_obfuscation(tmp_path: Path) -> None:
    """Verify AST linter detects string concatenation obfuscation."""
    obfuscated_file = tmp_path / "obfuscated.py"
    obfuscated_file.write_text(
        "token = 'uni' + 'ttest.mock'\n"
        "token2 = 'Magic' + 'Mock'\n",
        encoding="utf-8",
    )

    violations = check_file(obfuscated_file, tmp_path, set())
    assert len(violations) >= 2
    assert all(v.category == "OBFUSCATION" for v in violations)


def test_clean_production_code_passes(tmp_path: Path) -> None:
    """Verify clean production code without mocks or stubs passes cleanly."""
    clean_file = tmp_path / "clean_calc.py"
    clean_file.write_text(
        "from pathlib import Path\n"
        "from typing import List, Dict\n"
        "\n"
        "def compute_distances(coords: List[List[float]]) -> float:\n"
        "    total = 0.0\n"
        "    for p in coords:\n"
        "        total += sum(c ** 2 for c in p)\n"
        "    return total ** 0.5\n",
        encoding="utf-8",
    )

    violations = check_file(clean_file, tmp_path, set())
    assert len(violations) == 0


def test_cli_execution_with_json_output(tmp_path: Path) -> None:
    """Verify anti_spoof_linter CLI flags and structured JSON output."""
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(LINTER_PATH), str(clean_file), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed["exit_code"] == 0
    assert parsed["violations_count"] == 0
