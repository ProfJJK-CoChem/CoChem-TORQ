"""# zero-stub anti-spoofing engine
CoChem Anti-Spoof Linter (ci_tools/anti_spoof_linter.py)

Authoritative AST and static analyzer enforcing Zero-Mock & Anti-Spoofing Protocol v2
across the CoChem repository, Council modules, and execution pipelines. Verifies zero
stub logic, mocks, synthetic bypasses, and unamnestied parallel libraries.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger("anti_spoof_linter")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

AMNESTY_FILENAME: str = ".anti_spoof_amnesty.json"

BANNED_MOCK_MODULES: Set[str] = {
    "unittest.mock",
    "mock",
    "pytest_mock",
}

BANNED_MOCK_ATTRIBUTES: Set[str] = {
    "MagicMock",
    "Mock",
    "patch",
    "PropertyMock",
    "AsyncMock",
    "create_autospec",
    "NonCallableMock",
    "ANY",
    "sentinel",
    "call",
    "call_args",
}

BANNED_CONCURRENCY_MODULES: Set[str] = {
    "multiprocessing",
    "concurrent.futures",
    "parsl",
    "dask",
    "ray",
    "mpi4py",
    "threading",
    "celery",
}

BANNED_NUMPY_GENERATORS: Set[str] = {
    "linspace",
    "zeros",
    "ones",
    "eye",
    "sin",
    "rand",
    "randn",
    "normal",
    "uniform",
    "choice",
    "randint",
}

BANNED_IDENTIFIER_WORDS: Set[str] = {
    "dummy",
    "fake",
    "placeholder",
    "synthetic",
    "stub",
    "mock",
}

BANNED_OBFUSCATION_TOKENS: Set[str] = {
    "exec",
    "eval",
    "__import__",
}

EXCLUDED_DIRS: Set[str] = {
    "build",
    "dist",
    ".venv",
    ".conda",
    "venv",
    "site-packages",
    "artifacts",
    "datasets",
    "data",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".vscode",
    ".idea",
    ".trash",
    "Report_Archive",
    "scratch",
    "node_modules",
}

EXEMPTION_PHRASES: Set[str] = {
    "zero-stub",
    "mocking forbidden",
    "without mock",
    "anti-spoof",
    "anti_spoof",
    "anti-hallucination",
    "amnesty",
}


@dataclass(frozen=True)
class Violation:
    """Immutable record of an anti-spoof or zero-mock compliance violation."""
    file_path: str
    line: int
    col: int
    category: str
    symbol: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line,
            "col": self.col,
            "category": self.category,
            "symbol": self.symbol,
            "message": self.message,
        }


def normalize_path_entry(path_str: str) -> Tuple[str, ...]:
    """Normalize a path entry into unified posix format with sub-path aliases."""
    clean = path_str.replace("\\", "/").strip("/")
    parts = clean.split("/")
    variants = [clean]
    if len(parts) > 1:
        variants.append("/".join(parts[1:]))
    return tuple(dict.fromkeys(variants))


def find_repository_root(start_path: Union[str, Path]) -> Path:
    """Locate the root directory of the repository via indicators, env var, or traversal."""
    if "COCHEM_ROOT" in os.environ and os.environ["COCHEM_ROOT"]:
        return Path(os.environ["COCHEM_ROOT"]).resolve()

    p = Path(start_path).resolve()
    if p.is_file():
        p = p.parent

    current = p
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / AMNESTY_FILENAME).exists() or (current / ".git").exists():
            return current
        current = current.parent

    return p


def load_amnesty(root_dir: Union[str, Path]) -> Set[str]:
    """Load authorized zero-mock amnesty whitelist from .anti_spoof_amnesty.json."""
    p = Path(root_dir).resolve()
    target_file = p if p.is_file() and p.name == AMNESTY_FILENAME else None

    if not target_file:
        candidates = [
            p / AMNESTY_FILENAME,
            p.parent / AMNESTY_FILENAME,
            p.parent.parent / AMNESTY_FILENAME,
        ]
        if "COCHEM_ROOT" in os.environ:
            candidates.append(Path(os.environ["COCHEM_ROOT"]) / AMNESTY_FILENAME)
        for c in candidates:
            if c.exists() and c.is_file():
                target_file = c
                break

    if not target_file or not target_file.exists():
        return set()

    amnesty_set: Set[str] = set()
    try:
        content = target_file.read_text(encoding="utf-8-sig")
        data = json.loads(content)
        raw_entries: List[str] = []
        if isinstance(data, list):
            raw_entries = [str(x) for x in data]
        elif isinstance(data, dict):
            files_field = data.get("files", [])
            if isinstance(files_field, list):
                raw_entries = [str(x) for x in files_field]
            elif isinstance(files_field, dict):
                raw_entries = list(files_field.keys())

        for entry in raw_entries:
            for variant in normalize_path_entry(entry):
                amnesty_set.add(variant)
    except Exception as e:
        logger.warning(f"Could not parse amnesty file {target_file}: {e}")

    return amnesty_set


def save_amnesty(root_dir: Path, violations_dict: Dict[str, List[Violation]]) -> Path:
    """Generate or update .anti_spoof_amnesty.json with current authorized baseline."""
    target_file = root_dir / AMNESTY_FILENAME
    entries = sorted(violations_dict.keys())
    data = {
        "description": "Authenticated physical HPC dispatchers, zero-copy shared memory IPC, and local process executors verified under Anti-Spoof Protocol v2.",
        "files": entries,
    }
    target_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target_file


class SpoofVisitor(ast.NodeVisitor):
    """AST Visitor detecting prohibited mock patterns, stubs, synthetic generators, and intercepts."""

    def __init__(
        self,
        filepath: Path,
        rel_path: str,
        is_exempt: bool,
        amnesty_set: Set[str],
    ):
        self.filepath = filepath
        self.rel_path = rel_path
        self.is_exempt = is_exempt
        self.amnesty_set = amnesty_set
        self.violations: List[Violation] = []
        self.is_test_file = "test" in filepath.stem.lower() or "tests" in filepath.parts

    def _is_amnestied_concurrency(self) -> bool:
        norm_variants = normalize_path_entry(self.rel_path)
        return any(v in self.amnesty_set for v in norm_variants)

    def _check_ident(self, name: str, node: ast.AST, context: str) -> None:
        if self.is_exempt:
            return
        if self.is_test_file and name.startswith(("test_", "Test")):
            return

        name_lower = name.lower()
        tokens = set(re.findall(r"[a-z]+", re.sub(r"([A-Z])", r" \1", name).lower()))
        for kw in BANNED_IDENTIFIER_WORDS:
            if kw in tokens or kw in name_lower:
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=lineno,
                        col=col,
                        category="BANNED_IDENTIFIER",
                        symbol=name,
                        message=f"Banned identifier word '{kw}' detected in {context} '{name}'",
                    )
                )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if alias.name in BANNED_MOCK_MODULES or base_mod in BANNED_MOCK_MODULES:
                if not self.is_exempt:
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="MOCK_IMPORT",
                            symbol=alias.name,
                            message=f"Prohibited mock module import '{alias.name}'",
                        )
                    )
            elif alias.name in BANNED_CONCURRENCY_MODULES or base_mod in BANNED_CONCURRENCY_MODULES:
                if not self.is_exempt and not self._is_amnestied_concurrency():
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="CONCURRENCY_IMPORT",
                            symbol=alias.name,
                            message=f"Unamnestied concurrency import '{alias.name}'",
                        )
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        base_mod = mod.split(".")[0]

        if mod in BANNED_MOCK_MODULES or base_mod in BANNED_MOCK_MODULES or (mod == "unittest" and any(a.name == "mock" for a in node.names)):
            if not self.is_exempt:
                for alias in node.names:
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="MOCK_IMPORT",
                            symbol=f"{mod}.{alias.name}" if mod else alias.name,
                            message=f"Prohibited mock symbol import '{alias.name}' from '{mod}'",
                        )
                    )
        elif mod in BANNED_CONCURRENCY_MODULES or base_mod in BANNED_CONCURRENCY_MODULES or (mod == "concurrent" and any(a.name == "futures" for a in node.names)):
            if not self.is_exempt and not self._is_amnestied_concurrency():
                for alias in node.names:
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="CONCURRENCY_IMPORT",
                            symbol=f"{mod}.{alias.name}" if mod else alias.name,
                            message=f"Unamnestied concurrency symbol import '{alias.name}' from '{mod}'",
                        )
                    )
        else:
            if mod == "numpy":
                for alias in node.names:
                    if alias.name in BANNED_NUMPY_GENERATORS:
                        if not self.is_exempt:
                            self.violations.append(
                                Violation(
                                    file_path=self.rel_path,
                                    line=node.lineno,
                                    col=node.col_offset,
                                    category="SYNTHETIC_DATA",
                                    symbol=alias.name,
                                    message=f"Prohibited synthetic numpy generator '{alias.name}'",
                                )
                            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_ident(node.name, node, "function")
        self._check_function_stubs(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_ident(node.name, node, "async function")
        self._check_function_stubs(node)
        self.generic_visit(node)

    def _check_function_stubs(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if self.is_exempt:
            return
        decorators = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        decorators += [d.attr for d in node.decorator_list if isinstance(d, ast.Attribute)]
        if "abstractmethod" in decorators or "overload" in decorators:
            return

        body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        if not body or all(isinstance(n, (ast.Pass, ast.Expr)) and (isinstance(n, ast.Pass) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is ...)) for n in body):
            self.violations.append(
                Violation(
                    file_path=self.rel_path,
                    line=node.lineno,
                    col=node.col_offset,
                    category="EMPTY_PASS_STUB",
                    symbol=node.name,
                    message=f"Empty pass/ellipsis stub in function '{node.name}'",
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_ident(node.name, node, "class")
        if not self.is_exempt:
            base_names = set()
            for b in node.bases:
                if isinstance(b, ast.Name):
                    base_names.add(b.id)
                elif isinstance(b, ast.Attribute):
                    base_names.add(b.attr)
            is_allowed_empty = any(b in {"Exception", "BaseException", "UserWarning", "Warning", "Protocol", "ABC"} or "Error" in b for b in base_names)

            body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
            if not is_allowed_empty and (not body or all(isinstance(n, (ast.Pass, ast.Expr)) and (isinstance(n, ast.Pass) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is ...)) for n in body)):
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="EMPTY_PASS_STUB",
                        symbol=node.name,
                        message=f"Empty pass/ellipsis stub in class '{node.name}'",
                    )
                )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if not self.is_exempt and node.exc:
            exc_name = ""
            if isinstance(node.exc, ast.Name):
                exc_name = node.exc.id
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                exc_name = node.exc.func.id
            if exc_name == "NotImplementedError":
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="NOT_IMPLEMENTED_ERROR",
                        symbol=exc_name,
                        message="Forbidden NotImplementedError raise dead-end",
                    )
                )
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_with_items(node.items, node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._check_with_items(node.items, node.lineno, node.col_offset)
        self.generic_visit(node)

    def _check_with_items(self, items: List[ast.withitem], lineno: int, col: int) -> None:
        if self.is_exempt:
            return
        for item in items:
            expr = item.context_expr
            if isinstance(expr, ast.Call):
                func = expr.func
                func_name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                if func_name == "patch":
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=lineno,
                            col=col,
                            category="MOCK_IMPORT",
                            symbol="patch",
                            message="Mock patch context manager detected",
                        )
                    )
                elif func_name == "raises" and expr.args:
                    arg0 = expr.args[0]
                    if isinstance(arg0, ast.Name) and arg0.id == "NotImplementedError":
                        self.violations.append(
                            Violation(
                                file_path=self.rel_path,
                                line=lineno,
                                col=col,
                                category="NOT_IMPLEMENTED_ERROR",
                                symbol="pytest.raises(NotImplementedError)",
                                message="Tautological test assertion on NotImplementedError",
                            )
                        )

    def visit_Call(self, node: ast.Call) -> None:
        if self.is_exempt:
            self.generic_visit(node)
            return

        if isinstance(node.func, ast.Attribute) and node.func.attr == "assertRaises":
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "NotImplementedError":
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="NOT_IMPLEMENTED_ERROR",
                        symbol="assertRaises(NotImplementedError)",
                        message="Tautological test assertion on NotImplementedError",
                    )
                )

        func_name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if func_name in {"__import__", "import_module"} and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            target_mod = node.args[0].value
            base_target = target_mod.split(".")[0]
            if target_mod in BANNED_MOCK_MODULES or base_target in BANNED_MOCK_MODULES:
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="MOCK_IMPORT",
                        symbol=target_mod,
                        message=f"Dynamic mock module import '{target_mod}'",
                    )
                )
            elif target_mod in BANNED_CONCURRENCY_MODULES or base_target in BANNED_CONCURRENCY_MODULES:
                if not self._is_amnestied_concurrency():
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="CONCURRENCY_IMPORT",
                            symbol=target_mod,
                            message=f"Dynamic unamnestied concurrency import '{target_mod}'",
                        )
                    )

        if isinstance(node.func, ast.Attribute) and node.func.attr == "setattr":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "monkeypatch":
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="MONKEYPATCH_INTERCEPT",
                        symbol="monkeypatch.setattr",
                        message="Prohibited monkeypatch intercept of core system interfaces",
                    )
                )

        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in BANNED_NUMPY_GENERATORS:
                val = node.func.value
                val_name = val.id if isinstance(val, ast.Name) else (val.attr if isinstance(val, ast.Attribute) else "")
                if val_name in {"np", "numpy", "random"}:
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="SYNTHETIC_DATA",
                            symbol=f"{val_name}.{attr_name}",
                            message=f"Prohibited synthetic generator '{val_name}.{attr_name}'",
                        )
                    )
        elif isinstance(node.func, ast.Name) and node.func.id in BANNED_NUMPY_GENERATORS:
            self.violations.append(
                Violation(
                    file_path=self.rel_path,
                    line=node.lineno,
                    col=node.col_offset,
                    category="SYNTHETIC_DATA",
                    symbol=node.func.id,
                    message=f"Prohibited synthetic generator function '{node.func.id}'",
                )
            )

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not self.is_exempt and isinstance(node.op, ast.Add):
            reconstructed = self._extract_concat_str(node)
            if reconstructed:
                lower = reconstructed.lower()
                if "mock" in lower or "unittest.mock" in lower or "magicmock" in lower:
                    self.violations.append(
                        Violation(
                            file_path=self.rel_path,
                            line=node.lineno,
                            col=node.col_offset,
                            category="OBFUSCATION",
                            symbol=reconstructed,
                            message=f"Obfuscated mock token via string concatenation '{reconstructed}'",
                        )
                    )
        self.generic_visit(node)

    def _extract_concat_str(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._extract_concat_str(node.left)
            right = self._extract_concat_str(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        if not self.is_exempt:
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
            combined = "".join(parts)
            if "mock" in combined.lower() or "unittest.mock" in combined.lower():
                self.violations.append(
                    Violation(
                        file_path=self.rel_path,
                        line=node.lineno,
                        col=node.col_offset,
                        category="OBFUSCATION",
                        symbol=combined,
                        message=f"Obfuscated mock token via f-string '{combined}'",
                    )
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not self.is_exempt and isinstance(node.value, str):
            val = node.value.strip()
            if len(val) >= 4 and len(val) % 4 == 0 and re.match(r"^[A-Za-z0-9+/]+={0,2}$", val):
                try:
                    decoded = base64.b64decode(val).decode("utf-8", errors="ignore").lower()
                    if "mock" in decoded or "fake" in decoded:
                        self.violations.append(
                            Violation(
                                file_path=self.rel_path,
                                line=node.lineno,
                                col=node.col_offset,
                                category="OBFUSCATION",
                                symbol=val,
                                message=f"Obfuscated base64 mock payload '{val}'",
                            )
                        )
                except Exception:
                    pass

            if len(val) >= 8 and len(val) % 2 == 0 and re.match(r"^[0-9a-fA-F]+$", val):
                try:
                    decoded_hex = bytes.fromhex(val).decode("utf-8", errors="ignore").lower()
                    if "mock" in decoded_hex or "unittest.mock" in decoded_hex:
                        self.violations.append(
                            Violation(
                                file_path=self.rel_path,
                                line=node.lineno,
                                col=node.col_offset,
                                category="OBFUSCATION",
                                symbol=val,
                                message=f"Obfuscated hex mock payload '{val}'",
                            )
                        )
                except Exception:
                    pass

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self._check_ident(node.id, node, "variable")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self._check_ident(node.attr, node, "attribute")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_ident(node.arg, node, "argument")
        self.generic_visit(node)


def check_file(
    file_path: Path,
    repo_root: Path,
    amnesty_set: Set[str],
) -> List[Violation]:
    """Analyze a single Python file for AST anti-spoof violations."""
    rel_path = file_path.relative_to(repo_root).as_posix() if repo_root in file_path.parents or file_path == repo_root else file_path.name

    is_tool_exemption = file_path.name == "anti_spoof_linter.py" or "test_anti_spoof" in file_path.name
    is_api_mock_exemption = "api_mocks" in file_path.parts

    is_exempt = is_tool_exemption or is_api_mock_exemption

    try:
        content = file_path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        return [
            Violation(
                file_path=rel_path,
                line=e.lineno or 1,
                col=e.offset or 0,
                category="SYNTAX_ERROR",
                symbol="ast.parse",
                message=f"Syntax error: {e.msg}",
            )
        ]
    except Exception as e:
        return [
            Violation(
                file_path=rel_path,
                line=1,
                col=0,
                category="IO_ERROR",
                symbol="file_read",
                message=f"Could not read file: {e}",
            )
        ]

    visitor = SpoofVisitor(
        filepath=file_path,
        rel_path=rel_path,
        is_exempt=is_exempt,
        amnesty_set=amnesty_set,
    )
    visitor.visit(tree)
    return visitor.violations


def run_linter(
    targets: Optional[Sequence[Union[str, Path]]] = None,
    repo_root: Optional[Path] = None,
    strict_mode: bool = True,
    generate_amnesty: bool = False,
) -> Tuple[int, Dict[str, List[Violation]]]:
    """Execute anti-spoof static analysis across specified targets or whole repository."""
    if not repo_root:
        repo_root = find_repository_root(targets[0] if targets else Path.cwd())

    amnesty_set = load_amnesty(repo_root)
    all_violations: Dict[str, List[Violation]] = {}

    target_paths: List[Path] = []
    if targets:
        for t in targets:
            tp = Path(t).resolve()
            if tp.exists():
                target_paths.append(tp)
    else:
        target_paths = [repo_root]

    for tp in target_paths:
        if tp.is_file() and tp.suffix == ".py":
            v = check_file(tp, repo_root, amnesty_set)
            if v:
                all_violations[str(tp.relative_to(repo_root) if repo_root in tp.parents else tp.name)] = v
        elif tp.is_dir():
            for root, dirs, files in os.walk(tp):
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
                for f in files:
                    if f.endswith(".py"):
                        p = Path(root) / f
                        v = check_file(p, repo_root, amnesty_set)
                        if v:
                            all_violations[str(p.relative_to(repo_root) if repo_root in p.parents else p.name)] = v

    if generate_amnesty:
        save_amnesty(repo_root, all_violations)
        return 0, all_violations

    has_violations = len(all_violations) > 0
    exit_code = 1 if (has_violations and strict_mode) else 0
    return exit_code, all_violations


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CoChem Anti-Spoof Linter & Zero-Mock Enforcement Engine")
    parser.add_argument("targets", nargs="*", default=[], help="File(s) or directory paths to audit")
    parser.add_argument("--json", action="store_true", help="Output results in structured JSON format")
    parser.add_argument("--strict", action="store_true", default=False, help="Fail with non-zero exit code if violations found")
    parser.add_argument("--generate-amnesty", action="store_true", help="Generate or update .anti_spoof_amnesty.json baseline")
    args = parser.parse_args(argv)

    targets = [Path(t) for t in args.targets] if args.targets else [Path.cwd()]
    repo_root = find_repository_root(targets[0])

    exit_code, violations = run_linter(
        targets=targets,
        repo_root=repo_root,
        strict_mode=args.strict,
        generate_amnesty=args.generate_amnesty,
    )

    total_violations = sum(len(v_list) for v_list in violations.values())

    if args.json:
        output_payload = {
            "exit_code": exit_code,
            "violations_count": total_violations,
            "files_count": len(violations),
            "violations": {f: [v.to_dict() for v in v_list] for f, v_list in violations.items()},
        }
        print(json.dumps(output_payload, indent=2))
        return exit_code

    if total_violations == 0:
        print("[LINT SUCCESS] Zero-mock compliance verified. Zero stubs, mocks, or spoofing detected.")
        return 0

    print(f"[SPOOFING DETECTED] Found {total_violations} violation(s) across {len(violations)} file(s):")
    for f, v_list in violations.items():
        print(f"\nFile: {f}")
        for v in v_list:
            print(f"  - Line {v.line}:{v.col} [{v.category}] ({v.symbol}): {v.message}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
