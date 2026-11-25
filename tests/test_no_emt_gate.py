"""CI AST gate prohibiting ASE Effective Medium Theory (EMT) across CoChem-TORQ.

Effective Medium Theory (EMT) is a metal-focused empirical potential that is
physically invalid for organic, main-group, and rotational spectroscopy
workflows in CoChem-TORQ. This gate parses the Python AST of every tracked
source and test file in the repository and fails if any file imports the
ASE EMT calculator module or invokes ``EMT(...)``.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_EMT_MODULE = "ase.calculators.emt"
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".conda",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def _collect_repository_python_files(repo_root: Path) -> list[Path]:
    """Return all tracked Python files in the repository (fallback to rglob outside git)."""
    if (repo_root / ".git").exists():
        proc = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            paths = []
            for rel in proc.stdout.splitlines():
                candidate = repo_root / rel.strip()
                if candidate.is_file():
                    paths.append(candidate)
            return sorted(paths)

    discovered: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(repo_root).parts):
            continue
        if path.is_file():
            discovered.append(path)
    return sorted(discovered)


def _dotted_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def scan_python_source(source: str, filename: str = "<memory>") -> list[str]:
    """Inspect a Python AST and return all ASE EMT import/call violations."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN_EMT_MODULE or alias.name.startswith(
                    FORBIDDEN_EMT_MODULE + "."
                ):
                    violations.append(
                        f"{filename}:{node.lineno}: forbidden import '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == FORBIDDEN_EMT_MODULE or module.startswith(
                FORBIDDEN_EMT_MODULE + "."
            ):
                violations.append(
                    f"{filename}:{node.lineno}: forbidden import-from '{module}'"
                )
            elif module == "ase.calculators":
                for alias in node.names:
                    if alias.name.lower() == "emt":
                        violations.append(
                            f"{filename}:{node.lineno}: forbidden import 'from ase.calculators import {alias.name}'"
                        )
        elif isinstance(node, ast.Call):
            call_name = _dotted_call_name(node.func)
            if call_name == "EMT" or call_name.endswith(".EMT"):
                violations.append(
                    f"{filename}:{node.lineno}: forbidden call '{call_name}(...)'"
                )
            if call_name in {"importlib.import_module", "import_module", "__import__"}:
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    target = node.args[0].value
                    if target == FORBIDDEN_EMT_MODULE or target.startswith(
                        FORBIDDEN_EMT_MODULE + "."
                    ):
                        violations.append(
                            f"{filename}:{node.lineno}: forbidden dynamic import '{target}'"
                        )

    return violations


def test_repository_contains_zero_ase_emt_imports_or_calls() -> None:
    """Hard CI gate: zero ASE EMT imports or EMT() calls anywhere in tracked Python files."""
    py_files = _collect_repository_python_files(REPO_ROOT)
    assert len(py_files) >= 50, (
        f"Expected at least 50 tracked Python files in CoChem-TORQ, found {len(py_files)}"
    )

    all_violations: list[str] = []
    for path in py_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        all_violations.extend(scan_python_source(source, filename=rel))

    assert not all_violations, (
        "ASE EMT metal potential is strictly prohibited in CoChem-TORQ.\n"
        + "\n".join(all_violations)
    )


def test_no_emt_ast_gate_detects_direct_and_dynamic_violations() -> None:
    """Self-verify that the AST gate catches all prohibited EMT import and call forms."""
    direct_import = f"from {FORBIDDEN_EMT_MODULE} import EMT\ncalc = EMT()\n"
    direct_violations = scan_python_source(direct_import, "sample_direct.py")
    assert len(direct_violations) == 2

    module_import = (
        f"import {FORBIDDEN_EMT_MODULE} as calc_mod\ncalc = calc_mod.EMT()\n"
    )
    module_violations = scan_python_source(module_import, "sample_module.py")
    assert len(module_violations) == 2

    parent_import = "from ase.calculators import emt\ncalc = emt.EMT()\n"
    parent_violations = scan_python_source(parent_import, "sample_parent.py")
    assert len(parent_violations) == 2

    dynamic_import = (
        f"import importlib\nmod = importlib.import_module({FORBIDDEN_EMT_MODULE!r})\n"
    )
    dynamic_violations = scan_python_source(dynamic_import, "sample_dynamic.py")
    assert len(dynamic_violations) == 1

    clean_source = (
        "from ase.calculators.morse import MorsePotential\n"
        "calc = MorsePotential(epsilon=4.747, r0=0.7414, rho0=1.4398)\n"
    )
    assert scan_python_source(clean_source, "sample_clean.py") == []
