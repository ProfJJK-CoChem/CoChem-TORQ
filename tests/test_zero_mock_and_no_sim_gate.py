"""Hard CI AST & regex gate enforcing zero test doubles, zero simulation modes, and empty amnesty."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_PATH = Path(__file__).resolve()

EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".conda",
        ".mypy_cache",
        ".ruff_cache",
        ".eggs",
        "node_modules",
    }
)

EXCLUDED_REL_PATHS = frozenset(
    {
        "ci_tools/anti_spoof_linter.py",
        "src/cochem_base/core_engine/anti_spoof_linter.py",
        "tests/test_zero_mock_and_no_sim_gate.py",
    }
)

PROHIBITED_MODULES = frozenset(
    {
        "unittest.mock",
        "mock",
        "pytest_mock",
    }
)

PROHIBITED_SYMBOLS = frozenset(
    {
        "MagicMock",
        "Mock",
        "AsyncMock",
        "PropertyMock",
        "create_autospec",
        "NonCallableMock",
        "mock_open",
        "monkeypatch",
    }
)

PROHIBITED_PREFIXES = (
    "fake_",
    "FAKE_",
)

PROHIBITED_EXACT_NAMES = frozenset(
    {
        "generate_deterministic_solver_output",
        "poll_or_simulate_execution",
        "COLAB_WORKER_FAKE_GPU_NAME",
        "simulated_temp",
        "dummy_force",
    }
)

SIMULATION_MODE_PATTERN = re.compile(r"""\bmode\s*=\s*(['"])simulation\1""")


def _is_prohibited_module(module_name: str) -> bool:
    return any(
        module_name == banned or module_name.startswith(banned + ".")
        for banned in PROHIBITED_MODULES
    )


def _is_prohibited_identifier(name: str) -> bool:
    return name in PROHIBITED_EXACT_NAMES or name.startswith(PROHIBITED_PREFIXES)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _iter_repo_python_files(repo_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")
        )
        root_path = Path(dirpath)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            candidate = root_path / filename
            if candidate.resolve() == SELF_PATH:
                continue
            rel = candidate.relative_to(repo_root).as_posix()
            if rel in EXCLUDED_REL_PATHS:
                continue
            discovered.append(candidate)
    return discovered


def scan_python_source(source: str, filename: str = "<memory>") -> list[str]:
    """Scan Python source with AST and regex rules for prohibited constructs."""
    tree = ast.parse(source, filename=filename)
    findings: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_prohibited_module(alias.name):
                    findings.append(
                        f"{filename}:{node.lineno}: prohibited import '{alias.name}'"
                    )
                if alias.asname and (
                    alias.asname in PROHIBITED_SYMBOLS
                    or _is_prohibited_identifier(alias.asname)
                ):
                    findings.append(
                        f"{filename}:{node.lineno}: prohibited alias '{alias.asname}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_prohibited_module(mod):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited import-from '{mod}'"
                )
            elif mod == "unittest":
                for alias in node.names:
                    if alias.name == "mock":
                        findings.append(
                            f"{filename}:{node.lineno}: prohibited import 'from {mod} import {alias.name}'"
                        )
            for alias in node.names:
                if alias.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(alias.name):
                    findings.append(
                        f"{filename}:{node.lineno}: prohibited imported symbol '{alias.name}'"
                    )
                if alias.asname and (
                    alias.asname in PROHIBITED_SYMBOLS
                    or _is_prohibited_identifier(alias.asname)
                ):
                    findings.append(
                        f"{filename}:{node.lineno}: prohibited alias '{alias.asname}'"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.name):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited definition '{node.name}'"
                )
        elif isinstance(node, ast.arg):
            if node.arg in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.arg):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited parameter '{node.arg}'"
                )
        elif isinstance(node, ast.Name):
            if node.id in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.id):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited reference '{node.id}'"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.attr):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited attribute '.{node.attr}'"
                )
        elif isinstance(node, ast.keyword):
            if node.arg and (
                node.arg in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.arg)
            ):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited keyword '{node.arg}'"
                )
            if (
                node.arg == "mode"
                and isinstance(node.value, ast.Constant)
                and node.value.value == "simulation"
            ):
                findings.append(
                    f"{filename}:{node.lineno}: prohibited simulation mode argument"
                )
        elif isinstance(node, ast.Call):
            callee = _dotted_name(node.func)
            if callee in {"importlib.import_module", "import_module", "__import__"}:
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and _is_prohibited_module(node.args[0].value)
                ):
                    findings.append(
                        f"{filename}:{node.lineno}: prohibited dynamic import '{node.args[0].value}'"
                    )

    for match in SIMULATION_MODE_PATTERN.finditer(source):
        lineno = source.count("\n", 0, match.start()) + 1
        findings.append(
            f"{filename}:{lineno}: prohibited simulation mode literal '{match.group(0)}'"
        )

    return findings


def validate_amnesty_files(repo_root: Path) -> list[str]:
    """Ensure every .anti_spoof_amnesty.json in the repository parses strictly to []."""
    findings: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")
        )
        if ".anti_spoof_amnesty.json" not in filenames:
            continue
        amnesty_path = Path(dirpath) / ".anti_spoof_amnesty.json"
        rel = amnesty_path.relative_to(repo_root).as_posix()
        try:
            parsed = json.loads(amnesty_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            findings.append(f"{rel}: invalid JSON ({exc})")
            continue
        if parsed != []:
            findings.append(f"{rel}: expected empty list [], got {parsed!r}")
    return findings


def test_repository_zero_test_doubles_and_no_simulation_gate() -> None:
    """Hard CI gate: fail on any prohibited test-double, simulation mode, or non-empty amnesty."""
    py_files = _iter_repo_python_files(REPO_ROOT)
    assert len(py_files) > 0, f"No Python files discovered under {REPO_ROOT}"

    all_findings: list[str] = []
    for py_path in py_files:
        rel = py_path.relative_to(REPO_ROOT).as_posix()
        source = py_path.read_text(encoding="utf-8-sig")
        all_findings.extend(scan_python_source(source, filename=rel))

    all_findings.extend(validate_amnesty_files(REPO_ROOT))
    assert not all_findings, (
        "Zero-double / zero-simulation CI gate failed:\n" + "\n".join(all_findings)
    )


def test_amnesty_file_is_empty_if_present(tmp_path: Path) -> None:
    """Verify repository amnesty is empty and non-empty amnesty files are rejected."""
    assert validate_amnesty_files(REPO_ROOT) == []

    bad_amnesty = tmp_path / ".anti_spoof_amnesty.json"
    bad_amnesty.write_text('["Libraries/example.py"]\n', encoding="utf-8")
    assert len(validate_amnesty_files(tmp_path)) == 1

    bad_amnesty.write_text("[]\n", encoding="utf-8")
    assert validate_amnesty_files(tmp_path) == []


def test_gate_scanner_detects_prohibited_constructs() -> None:
    """Self-verify that every banned import, symbol, identifier, and simulation mode is caught."""
    snippets = [
        "import unittest.mock\n",
        "from unittest import mock\n",
        "import pytest_mock\n",
        "import importlib\nmod = importlib.import_module('unittest.mock')\n",
        "obj = MagicMock()\n",
        "obj = Mock()\n",
        "obj = AsyncMock()\n",
        "obj = PropertyMock()\n",
        "obj = create_autospec(int)\n",
        "obj = NonCallableMock()\n",
        "h = mock_open()\n",
        "def f(monkeypatch):\n    return monkeypatch\n",
        "def fake_solver():\n    return 0\n",
        "FAKE_ENERGY = -76.0\n",
        "def generate_deterministic_solver_output():\n    return ''\n",
        "def poll_or_simulate_execution():\n    return None\n",
        "COLAB_WORKER_FAKE_GPU_NAME = 'A100'\n",
        "simulated_temp = 298.15\n",
        "dummy_force = 0.0\n",
        'run_calc(mode="simulation")\n',
        "run_calc(mode='simulation')\n",
    ]
    for idx, snippet in enumerate(snippets):
        detected = scan_python_source(snippet, filename=f"check_{idx}.py")
        assert detected, f"Gate missed prohibited snippet #{idx}: {snippet!r}"

    clean_snippet = (
        "from pathlib import Path\n"
        "def compute_distance(a: float, b: float) -> float:\n"
        "    return abs(a - b)\n"
    )
    assert scan_python_source(clean_snippet, filename="clean.py") == []
