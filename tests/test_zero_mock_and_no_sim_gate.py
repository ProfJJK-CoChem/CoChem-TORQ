"""Hard CI AST, concurrency determinism & mockup-leak firewall gate.

Enforces signed decisions own-1126-84f4 (M2 hardening) and swarm-1126-9fc4 (mockup firewall):
- Zero test doubles, indirect mock aliases, dynamic mock imports, sys.modules mock injection,
  HTTP/network mock packages, or module-level / engine-module setattr interception.
- Stdlib concurrency (threading, multiprocessing, concurrent.futures) allowed while
  .anti_spoof_amnesty.json is strictly [].
- Serial-vs-parallel concurrency determinism (workers=1 vs workers=2 hash-identical output,
  no swallowed exceptions) on authentic CoChem-TORQ rotational physics.
- Mockup-leak firewall blocking cochem:mockup=true PNG tEXt/zTXt/iTXt or PDF XMP metadata
  and 'NOT SCIENTIFIC DATA' watermark strings in tracked images/PDFs.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

import pytest

from ci_tools.anti_spoof_linter import check_file as anti_spoof_check_file
from Libraries.cochem_torq_asymmetric_rotor import (
    AsymmetricTopDiagonalizer,
    RotationalConstants,
)

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
        "responses",
        "vcr",
        "vcrpy",
        "requests_mock",
        "mockito",
        "flexmock",
        "respx",
        "httpretty",
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
        "mocker",
        "MockerFixture",
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

ENGINE_SETATTR_TARGET_TOKENS = frozenset(
    {
        "engine",
        "runner",
        "solver",
        "bridge",
        "subprocess",
        "pyscf",
        "orca",
        "xtb",
        "crest",
        "cfour",
        "spcat",
        "spfit",
        "cochem",
        "torq",
        "topos",
        "mace",
        "ase",
        "calculator",
    }
)

SIMULATION_MODE_PATTERN = re.compile(r"""\bmode\s*=\s*(['"])simulation\1""")

MEDIA_EXTENSIONS = frozenset({".png", ".pdf", ".jpg", ".jpeg", ".svg", ".webp", ".gif"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MOCKUP_METADATA_PATTERNS = (
    re.compile(r"""cochem:mockup\s*[:=]\s*['"]?true\b""", re.IGNORECASE),
    re.compile(r"""<cochem:mockup[^>]*>\s*true\s*</cochem:mockup>""", re.IGNORECASE),
    re.compile(r"""not\s+scientific\s+data""", re.IGNORECASE),
)
PDF_STREAM_PATTERN = re.compile(rb"stream\r?\n(.*?)\r?\n?endstream", re.DOTALL)


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


def _extract_constant_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _extract_constant_str(node.left)
        right = _extract_constant_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


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


class _GateASTScanner(ast.NodeVisitor):
    """Scope- and alias-aware AST scanner for prohibited test doubles and injections."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: list[str] = []
        self._imported_modules: dict[str, str] = {}
        self._importlib_aliases: set[str] = {"importlib"}
        self._import_module_aliases: set[str] = {"import_module", "__import__"}
        self._sys_aliases: set[str] = {"sys"}
        self._sys_modules_aliases: set[str] = set()
        self._unittest_aliases: set[str] = {"unittest"}
        self._scope_depth: int = 0

    def _is_sys_modules_expr(self, node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id in self._sys_aliases
        ):
            return True
        if isinstance(node, ast.Name) and node.id in self._sys_modules_aliases:
            return True
        return False

    def _is_engine_setattr_target(self, target: ast.AST) -> bool:
        if isinstance(target, ast.Subscript) and self._is_sys_modules_expr(target.value):
            return True
        if isinstance(target, ast.Name) and target.id in self._imported_modules:
            return True
        dotted = _dotted_name(target)
        if not dotted:
            return False
        if dotted.split(".")[0] in self._imported_modules:
            return True
        tokens = set(re.findall(r"[a-z0-9]+", dotted.lower()))
        return bool(tokens & ENGINE_SETATTR_TARGET_TOKENS)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            local_name = alias.asname or base_mod
            self._imported_modules[local_name] = alias.name
            if alias.name == "importlib":
                self._importlib_aliases.add(local_name)
            elif alias.name == "sys":
                self._sys_aliases.add(local_name)
            elif alias.name == "unittest":
                self._unittest_aliases.add(local_name)

            if _is_prohibited_module(alias.name):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited import '{alias.name}'"
                )
            if alias.asname and (
                alias.asname in PROHIBITED_SYMBOLS
                or _is_prohibited_identifier(alias.asname)
            ):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited alias '{alias.asname}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            local_name = alias.asname or alias.name
            full_target = f"{mod}.{alias.name}" if mod else alias.name
            self._imported_modules[local_name] = full_target
            if mod == "importlib" and alias.name == "import_module":
                self._import_module_aliases.add(local_name)
            elif mod == "builtins" and alias.name == "__import__":
                self._import_module_aliases.add(local_name)
            elif mod == "sys" and alias.name == "modules":
                self._sys_modules_aliases.add(local_name)

        if _is_prohibited_module(mod):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited import-from '{mod}'"
            )
        elif mod == "unittest":
            for alias in node.names:
                if alias.name == "mock":
                    as_suffix = f" as {alias.asname}" if alias.asname else ""
                    self.findings.append(
                        f"{self.filename}:{node.lineno}: prohibited import 'from {mod} import {alias.name}{as_suffix}'"
                    )
        for alias in node.names:
            if alias.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(alias.name):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited imported symbol '{alias.name}'"
                )
            if alias.asname and (
                alias.asname in PROHIBITED_SYMBOLS
                or _is_prohibited_identifier(alias.asname)
            ):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited alias '{alias.asname}'"
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.name):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited definition '{node.name}'"
            )
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.name):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited definition '{node.name}'"
            )
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.name):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited definition '{node.name}'"
            )
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.arg):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited parameter '{node.arg}'"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.id):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited reference '{node.id}'"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr == "mock"
            and isinstance(node.value, ast.Name)
            and node.value.id in self._unittest_aliases
        ):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited indirect mock attribute '{node.value.id}.mock'"
            )
        if node.attr in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.attr):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited attribute '.{node.attr}'"
            )
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg and (
            node.arg in PROHIBITED_SYMBOLS or _is_prohibited_identifier(node.arg)
        ):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited keyword '{node.arg}'"
            )
        if (
            node.arg == "mode"
            and isinstance(node.value, ast.Constant)
            and node.value.value == "simulation"
        ):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited simulation mode argument"
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_sys_modules_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._sys_modules_aliases.add(target.id)
        for target in node.targets:
            if isinstance(target, ast.Subscript) and self._is_sys_modules_expr(target.value):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited sys.modules mock injection"
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            node.value is not None
            and self._is_sys_modules_expr(node.value)
            and isinstance(node.target, ast.Name)
        ):
            self._sys_modules_aliases.add(node.target.id)
        if isinstance(node.target, ast.Subscript) and self._is_sys_modules_expr(node.target.value):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited sys.modules mock injection"
            )
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and self._is_sys_modules_expr(target.value):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited sys.modules mutation via del"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_dynamic_import = False
        if isinstance(node.func, ast.Name) and node.func.id in self._import_module_aliases:
            is_dynamic_import = True
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in {"import_module", "__import__"}:
                is_dynamic_import = True
            elif (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in self._importlib_aliases
                and node.func.attr == "import_module"
            ):
                is_dynamic_import = True

        if is_dynamic_import:
            target_arg: ast.AST | None = node.args[0] if node.args else None
            if target_arg is None:
                for kw in node.keywords:
                    if kw.arg == "name":
                        target_arg = kw.value
                        break
            target_mod = _extract_constant_str(target_arg)
            if target_mod and _is_prohibited_module(target_mod):
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited dynamic import '{target_mod}'"
                )

        if (
            isinstance(node.func, ast.Attribute)
            and self._is_sys_modules_expr(node.func.value)
            and node.func.attr
            in {"setdefault", "update", "__setitem__", "__delitem__", "pop", "clear"}
        ):
            self.findings.append(
                f"{self.filename}:{node.lineno}: prohibited sys.modules.{node.func.attr} injection"
            )

        callee = _dotted_name(node.func)
        if callee in {"setattr", "builtins.setattr"}:
            if self._scope_depth == 0:
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited module-level setattr interception"
                )
            elif node.args and self._is_engine_setattr_target(node.args[0]):
                target_label = _dotted_name(node.args[0]) or "sys.modules[...]"
                self.findings.append(
                    f"{self.filename}:{node.lineno}: prohibited setattr on engine/imported module '{target_label}'"
                )

        self.generic_visit(node)


def scan_python_source(source: str, filename: str = "<memory>") -> list[str]:
    """Scan Python source with AST and regex rules for prohibited constructs."""
    tree = ast.parse(source, filename=filename)
    scanner = _GateASTScanner(filename=filename)
    scanner.visit(tree)
    findings = list(scanner.findings)

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


def _check_text_for_mockup_markers(text: str, label: str) -> list[str]:
    hits: list[str] = []
    for pattern in MOCKUP_METADATA_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(f"{label}: matched mockup marker {match.group(0)!r}")
    return hits


def scan_png_for_mockup_leaks(path: Path, rel_label: str | None = None) -> list[str]:
    """Inspect PNG tEXt/zTXt/iTXt chunks and raw bytes for mockup metadata or watermarks."""
    label = rel_label or path.as_posix()
    raw = path.read_bytes()
    findings: list[str] = []

    if raw.startswith(PNG_SIGNATURE):
        offset = len(PNG_SIGNATURE)
        total = len(raw)
        while offset + 12 <= total:
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            chunk_type = raw[offset + 4 : offset + 8]
            data_start = offset + 8
            data_end = data_start + length
            if data_end + 4 > total:
                break
            chunk_data = raw[data_start:data_end]
            offset = data_end + 4

            if chunk_type == b"tEXt":
                if b"\x00" in chunk_data:
                    key_b, val_b = chunk_data.split(b"\x00", 1)
                    key = key_b.decode("latin-1", errors="ignore").strip()
                    val = val_b.decode("latin-1", errors="ignore").strip()
                    if key.lower() == "cochem:mockup" and val.lower() == "true":
                        findings.append(
                            f"{label}: PNG tEXt chunk declares cochem:mockup=true"
                        )
                    findings.extend(
                        _check_text_for_mockup_markers(f"{key}={val}", f"{label} [PNG tEXt]")
                    )
            elif chunk_type == b"zTXt":
                if b"\x00" in chunk_data:
                    key_b, rest = chunk_data.split(b"\x00", 1)
                    if len(rest) >= 2 and rest[0] == 0:
                        try:
                            val_b = zlib.decompress(rest[1:])
                        except zlib.error:
                            val_b = b""
                        key = key_b.decode("latin-1", errors="ignore").strip()
                        val = val_b.decode("latin-1", errors="ignore").strip()
                        if key.lower() == "cochem:mockup" and val.lower() == "true":
                            findings.append(
                                f"{label}: PNG zTXt chunk declares cochem:mockup=true"
                            )
                        findings.extend(
                            _check_text_for_mockup_markers(
                                f"{key}={val}", f"{label} [PNG zTXt]"
                            )
                        )
            elif chunk_type == b"iTXt":
                parts = chunk_data.split(b"\x00", 5)
                if len(parts) >= 3:
                    key = parts[0].decode("utf-8", errors="ignore").strip()
                    rest = chunk_data[len(parts[0]) + 1 :]
                    if len(rest) >= 2:
                        comp_flag = rest[0]
                        tail = rest[2:]
                        # Skip language_tag\0translated_keyword\0
                        tail_parts = tail.split(b"\x00", 2)
                        text_bytes = tail_parts[2] if len(tail_parts) == 3 else tail
                        if comp_flag == 1:
                            try:
                                text_bytes = zlib.decompress(text_bytes)
                            except zlib.error:
                                text_bytes = b""
                        val = text_bytes.decode("utf-8", errors="ignore").strip()
                        if key.lower() == "cochem:mockup" and val.lower() == "true":
                            findings.append(
                                f"{label}: PNG iTXt chunk declares cochem:mockup=true"
                            )
                        findings.extend(
                            _check_text_for_mockup_markers(
                                f"{key}={val}", f"{label} [PNG iTXt]"
                            )
                        )

    findings.extend(
        _check_text_for_mockup_markers(
            raw.decode("latin-1", errors="ignore"), f"{label} [raw]"
        )
    )
    return list(dict.fromkeys(findings))


def scan_pdf_for_mockup_leaks(path: Path, rel_label: str | None = None) -> list[str]:
    """Inspect PDF XMP metadata, raw text, and FlateDecode streams for mockup markers."""
    label = rel_label or path.as_posix()
    raw = path.read_bytes()
    findings: list[str] = []

    findings.extend(
        _check_text_for_mockup_markers(
            raw.decode("latin-1", errors="ignore"), f"{label} [PDF raw/XMP]"
        )
    )

    for idx, match in enumerate(PDF_STREAM_PATTERN.finditer(raw)):
        stream_bytes = match.group(1).strip(b"\r\n")
        if not stream_bytes:
            continue
        try:
            decompressed = zlib.decompress(stream_bytes)
        except zlib.error:
            continue
        decoded = decompressed.decode("latin-1", errors="ignore")
        findings.extend(
            _check_text_for_mockup_markers(decoded, f"{label} [PDF stream #{idx}]")
        )

    return list(dict.fromkeys(findings))


def _iter_tracked_media_files(repo_root: Path) -> list[Path]:
    """List tracked image/PDF assets via git ls-files, falling back to directory walk."""
    if (repo_root / ".git").exists():
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            files: list[Path] = []
            for entry in proc.stdout.split(b"\x00"):
                if not entry:
                    continue
                rel = entry.decode("utf-8", errors="surrogateescape")
                candidate = repo_root / rel
                if candidate.suffix.lower() in MEDIA_EXTENSIONS and candidate.is_file():
                    files.append(candidate)
            return sorted(files)

    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        root_path = Path(dirpath)
        for filename in sorted(filenames):
            candidate = root_path / filename
            if candidate.suffix.lower() in MEDIA_EXTENSIONS:
                discovered.append(candidate)
    return discovered


def scan_tracked_media_for_mockup_leaks(repo_root: Path) -> tuple[int, list[str]]:
    """Scan all tracked images and PDFs for mockup metadata or watermark strings."""
    media_files = _iter_tracked_media_files(repo_root)
    findings: list[str] = []
    for media_path in media_files:
        rel = media_path.relative_to(repo_root).as_posix()
        suffix = media_path.suffix.lower()
        if suffix == ".png":
            findings.extend(scan_png_for_mockup_leaks(media_path, rel_label=rel))
        elif suffix == ".pdf":
            findings.extend(scan_pdf_for_mockup_leaks(media_path, rel_label=rel))
        else:
            raw_text = media_path.read_bytes().decode("latin-1", errors="ignore")
            findings.extend(_check_text_for_mockup_markers(raw_text, rel))
    return len(media_files), findings


def _build_minimal_png(extra_chunks: list[tuple[bytes, bytes]] | None = None) -> bytes:
    """Build a valid 1x1 RGB PNG in pure stdlib with optional extra chunks."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    parts = [PNG_SIGNATURE, _chunk(b"IHDR", ihdr)]
    for ctype, cdata in extra_chunks or []:
        parts.append(_chunk(ctype, cdata))
    parts.append(_chunk(b"IDAT", idat))
    parts.append(_chunk(b"IEND", b""))
    return b"".join(parts)


def _build_minimal_pdf(
    xmp_metadata: str = "",
    flate_text: str = "",
) -> bytes:
    """Build a compact synthetic PDF byte stream for firewall canary testing."""
    body_parts = [b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"]
    if xmp_metadata:
        body_parts.append(
            b"2 0 obj\n<< /Type /Metadata /Subtype /XML >>\nstream\n"
            + xmp_metadata.encode("utf-8")
            + b"\nendstream\nendobj\n"
        )
    if flate_text:
        compressed = zlib.compress(flate_text.encode("utf-8"))
        body_parts.append(
            b"3 0 obj\n<< /Filter /FlateDecode /Length "
            + str(len(compressed)).encode("ascii")
            + b" >>\nstream\n"
            + compressed
            + b"\nendstream\nendobj\n"
        )
    body_parts.append(b"%%EOF\n")
    return b"".join(body_parts)


def _solve_rotational_spec(spec: dict[str, Any]) -> tuple[str, dict[str, float]]:
    constants = RotationalConstants(**spec["constants"])
    solver = AsymmetricTopDiagonalizer(constants=constants, reduction="A", j_max=spec["j_max"])
    levels = solver.solve_energy_levels()
    canonical_levels = {f"{j}:{idx}": float(energy) for (j, idx), energy in sorted(levels.items())}
    return str(spec["label"]), canonical_levels


def _run_rotational_batch(
    specs: list[dict[str, Any]],
    workers: int,
) -> tuple[dict[str, dict[str, float]], str]:
    """Execute authentic Watson-A diagonalizations across a ThreadPoolExecutor."""
    ordered: dict[str, dict[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_solve_rotational_spec, spec) for spec in specs]
        for fut in futures:
            label, levels = fut.result()
            ordered[label] = levels
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return ordered, digest


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


def test_amnesty_file_is_strictly_empty_list(tmp_path: Path) -> None:
    """Strictly assert repository .anti_spoof_amnesty.json exists and equals []."""
    root_amnesty = REPO_ROOT / ".anti_spoof_amnesty.json"
    assert root_amnesty.is_file(), "Root .anti_spoof_amnesty.json must exist and equal []"
    assert json.loads(root_amnesty.read_text(encoding="utf-8-sig")) == []
    assert validate_amnesty_files(REPO_ROOT) == []

    bad_amnesty = tmp_path / ".anti_spoof_amnesty.json"
    bad_amnesty.write_text('["Libraries/example.py"]\n', encoding="utf-8")
    assert len(validate_amnesty_files(tmp_path)) == 1

    bad_amnesty.write_text("[]\n", encoding="utf-8")
    assert validate_amnesty_files(tmp_path) == []


def test_gate_scanner_detects_prohibited_constructs() -> None:
    """Seeded-canary verification covering direct mocks, indirect aliases, sys.modules, HTTP mocks, and setattr."""
    snippets = [
        "import unittest.mock\n",
        "import unittest.mock as um\n",
        "from unittest import mock\n",
        "from unittest import mock as hidden_m\n",
        "import unittest as ut\nx = ut.mock\n",
        "import pytest_mock\n",
        "import responses\n",
        "from responses import activate\n",
        "import vcr\n",
        "import vcrpy\n",
        "import requests_mock\n",
        "import importlib\nmod = importlib.import_module('unittest.mock')\n",
        "import importlib\nmod = importlib.import_module('mock')\n",
        "import importlib\nmod = importlib.import_module('pytest_mock')\n",
        "import importlib as il\nmod = il.import_module('unittest.mock')\n",
        "from importlib import import_module as im\nmod = im('unittest.' + 'mock')\n",
        "import sys\nsys.modules['pyscf'] = object()\n",
        "import sys as s\ns.modules.update({'orca': object()})\n",
        "from sys import modules as mods\nmods.setdefault('xtb', object())\n",
        "import sys\nmods = sys.modules\nmods['crest'] = object()\n",
        "def check_mocker(mocker):\n    return mocker.patch('os.path.exists')\n",
        "import Libraries.cochem_torq_engine as eng\nsetattr(eng, 'run_orca', lambda *a: None)\n",
        "import subprocess\ndef f():\n    setattr(subprocess, 'run', lambda *a: None)\n",
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
        assert detected, f"Gate missed prohibited canary #{idx}: {snippet!r}"

    clean_snippet = (
        "import concurrent.futures\n"
        "from concurrent import futures\n"
        "import multiprocessing\n"
        "import sys\n"
        "import threading\n"
        "from pathlib import Path\n"
        "class Record:\n"
        "    def set_field(self, key: str, value: float) -> None:\n"
        "        setattr(self, key, value)\n"
        "def compute_distance(a: float, b: float) -> float:\n"
        "    _ = 'pyscf' in sys.modules\n"
        "    return abs(a - b)\n"
    )
    assert scan_python_source(clean_snippet, filename="clean.py") == []


def test_anti_spoof_linter_catches_indirect_mock_aliases_and_injections(tmp_path: Path) -> None:
    """Seeded-canary cross-check proving ci_tools/anti_spoof_linter.py catches every widened bypass."""
    canaries = {
        "alias_unittest_mock.py": "from unittest import mock as secret_m\n",
        "attr_unittest_mock.py": "import unittest as ut\nx = ut.mock\n",
        "importlib_unittest_mock.py": "import importlib\nm = importlib.import_module('unittest.mock')\n",
        "importlib_mock_alias.py": "from importlib import import_module as im\nm = im('pytest_mock')\n",
        "sys_modules_inject.py": "import sys\nsys.modules['pyscf'] = object()\n",
        "sys_modules_update.py": "import sys as s\ns.modules.update({'orca': object()})\n",
        "pytest_mock_mocker.py": "def run_check(mocker):\n    return mocker.patch('os.getcwd')\n",
        "http_mock_libs.py": "import responses\nimport vcrpy\nimport requests_mock\n",
        "engine_setattr.py": "import Libraries.cochem_torq_engine as eng\nsetattr(eng, 'run_orca', lambda *a: None)\n",
    }
    for fname, code in canaries.items():
        path = tmp_path / fname
        path.write_text(code, encoding="utf-8")
        violations = anti_spoof_check_file(path, tmp_path, set())
        assert violations, f"anti_spoof_linter missed seeded canary {fname}: {code!r}"

    stdlib_clean = tmp_path / "stdlib_concurrency_clean.py"
    stdlib_clean.write_text(
        "import threading\n"
        "import multiprocessing\n"
        "import concurrent.futures\n"
        "from concurrent import futures\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    assert anti_spoof_check_file(stdlib_clean, tmp_path, set()) == []


def test_serial_vs_parallel_concurrency_determinism_and_error_propagation() -> None:
    """Verify workers=1 vs workers=2 produces SHA-256 identical outputs and never swallows exceptions."""
    specs = [
        {"label": "water_like", "constants": {"A": 835840.0, "B": 435351.0, "C": 278138.0, "D_J": 37.5}, "j_max": 6},
        {"label": "formaldehyde_like", "constants": {"A": 281970.5, "B": 38833.9, "C": 34004.2, "D_J": 0.075}, "j_max": 6},
        {"label": "so2_like", "constants": {"A": 60778.5, "B": 10318.1, "C": 8799.7, "D_J": 0.0066}, "j_max": 6},
        {"label": "formic_like", "constants": {"A": 77512.2, "B": 12055.1, "C": 10416.1, "D_J": 0.0099}, "j_max": 6},
        {"label": "hof_like", "constants": {"A": 585630.0, "B": 26760.0, "C": 25510.0, "D_J": 0.095}, "j_max": 6},
        {"label": "h2s_like", "constants": {"A": 310182.0, "B": 270884.0, "C": 141705.0, "D_J": 19.6}, "j_max": 6},
    ]

    out_serial, hash_serial = _run_rotational_batch(specs, workers=1)
    out_parallel, hash_parallel = _run_rotational_batch(specs, workers=2)

    assert hash_serial == hash_parallel, (
        f"Serial (workers=1) vs parallel (workers=2) hash mismatch: {hash_serial} != {hash_parallel}"
    )
    assert out_serial == out_parallel
    assert len(out_serial) == len(specs)

    # Verify exceptions raised in worker threads are propagated and never swallowed.
    bad_specs = [
        specs[0],
        {"label": "invalid_ordering", "constants": {"A": 1000.0, "B": 2000.0, "C": 500.0}, "j_max": 3},
        specs[1],
    ]
    with pytest.raises(ValueError, match="A >= B >= C > 0"):
        _run_rotational_batch(bad_specs, workers=2)


def test_repository_tracked_media_mockup_firewall() -> None:
    """Firewall: reject cochem:mockup=true PNG/PDF metadata or 'NOT SCIENTIFIC DATA' watermarks in tracked media."""
    scanned_count, findings = scan_tracked_media_for_mockup_leaks(REPO_ROOT)
    assert scanned_count >= 1, "Expected at least one tracked image/PDF asset to scan"
    assert not findings, "Mockup-leak media firewall failed:\n" + "\n".join(findings)


def test_mockup_firewall_rejects_seeded_png_and_pdf_canaries(tmp_path: Path) -> None:
    """Self-verify PNG tEXt/zTXt and PDF XMP/stream mockup canaries are rejected while clean assets pass."""
    clean_png = tmp_path / "clean.png"
    clean_png.write_bytes(_build_minimal_png([(b"tEXt", b"Software\x00CoChem-TORQ")]))
    assert scan_png_for_mockup_leaks(clean_png) == []

    mockup_png = tmp_path / "mockup_meta.png"
    mockup_png.write_bytes(_build_minimal_png([(b"tEXt", b"cochem:mockup\x00true")]))
    assert scan_png_for_mockup_leaks(mockup_png), "Firewall missed PNG tEXt cochem:mockup=true"

    watermark_png = tmp_path / "watermark_ztxt.png"
    watermark_png.write_bytes(
        _build_minimal_png(
            [(b"zTXt", b"Comment\x00\x00" + zlib.compress(b"NOT SCIENTIFIC DATA"))]
        )
    )
    assert scan_png_for_mockup_leaks(watermark_png), "Firewall missed PNG zTXt watermark"

    clean_pdf = tmp_path / "clean.pdf"
    clean_pdf.write_bytes(
        _build_minimal_pdf(
            xmp_metadata="<x:xmpmeta><dc:title>Authentic Spectrum</dc:title></x:xmpmeta>",
            flate_text="BT (Rotational Transition 38833.9 MHz) Tj ET",
        )
    )
    assert scan_pdf_for_mockup_leaks(clean_pdf) == []

    mockup_pdf = tmp_path / "mockup_xmp.pdf"
    mockup_pdf.write_bytes(
        _build_minimal_pdf(
            xmp_metadata="<x:xmpmeta><cochem:mockup>true</cochem:mockup></x:xmpmeta>"
        )
    )
    assert scan_pdf_for_mockup_leaks(mockup_pdf), "Firewall missed PDF XMP cochem:mockup=true"

    watermark_pdf = tmp_path / "watermark_stream.pdf"
    watermark_pdf.write_bytes(
        _build_minimal_pdf(flate_text="BT (NOT SCIENTIFIC DATA) Tj ET")
    )
    assert scan_pdf_for_mockup_leaks(watermark_pdf), "Firewall missed PDF FlateDecode watermark"
