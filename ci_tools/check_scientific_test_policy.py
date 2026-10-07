"""Inventory explicit runtime replacement APIs in Python test source.

This narrow AST check catches imports of mocking packages and common runtime
patch operations. It does not prove that data are genuine, that an executable is
the expected scientific engine, or that a method has been independently validated.
Those require source review, native artifacts, and mandatory real-engine gates.
String literals used as data in linter tests are not executed Python and are not
reported as runtime replacement. Environment-only setenv/delenv are inventoried
separately; configuring a process is not replacement of its calculation output.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

MOCK_PACKAGES = ("unittest.mock", "pytest_mock", "mock", "mockito")
REPLACEMENT_METHODS = {
    "setattr",
    "delattr",
    "setitem",
    "delitem",
    "patch",
    "patch.object",
    "patch.dict",
    "Mock",
    "MagicMock",
    "AsyncMock",
    "stub",
    "spy",
}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def inventory(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: dict[str, str] = {}
    findings = []
    configuration_calls = []
    for node in ast.walk(tree):
        imports = []
        if isinstance(node, ast.Import):
            imports = [
                (entry.name, entry.asname or entry.name.split(".")[0])
                for entry in node.names
            ]
        elif isinstance(node, ast.ImportFrom):
            imports = [
                (f"{node.module}.{entry.name}", entry.asname or entry.name)
                for entry in node.names
            ]
        for canonical, local in imports:
            aliases[local] = canonical
            if any(
                canonical == package or canonical.startswith(package + ".")
                for package in MOCK_PACKAGES
            ):
                findings.append(
                    {"line": node.lineno, "kind": "mocking_import", "name": canonical}
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        parts = name.split(".")
        canonical = aliases.get(parts[0], parts[0]) + (
            "." + ".".join(parts[1:]) if len(parts) > 1 else ""
        )
        if parts[0] == "monkeypatch" and parts[-1] in {"setenv", "delenv"}:
            configuration_calls.append({"line": node.lineno, "name": name})
        elif any(
            canonical == package or canonical.startswith(package + ".")
            for package in MOCK_PACKAGES
        ) or (
            parts[0] in {"monkeypatch", "mocker", "mock"}
            and ".".join(parts[1:]) in REPLACEMENT_METHODS
        ):
            findings.append(
                {
                    "line": node.lineno,
                    "kind": "runtime_replacement_call",
                    "name": canonical,
                }
            )
    return {
        "path": str(path),
        "findings": findings,
        "environment_configuration_calls": configuration_calls,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", default=[Path("tests")])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    files = sorted(
        {
            entry
            for path in arguments.paths
            for entry in (path.rglob("*.py") if path.is_dir() else [path])
        }
    )
    results = []
    for path in files:
        try:
            results.append(inventory(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            results.append(
                {
                    "path": str(path),
                    "findings": [{"kind": "unreadable_source", "message": str(error)}],
                    "environment_configuration_calls": [],
                }
            )
    finding_count = sum(len(entry["findings"]) for entry in results)
    report = {
        "schema": "cochem.scientific-test-api-inventory/1",
        "files_examined": len(files),
        "explicit_runtime_replacement_findings": finding_count,
        "environment_configuration_calls": sum(
            len(entry["environment_configuration_calls"]) for entry in results
        ),
        "scope": (
            "AST imports and common replacement APIs only; no assertion of "
            "scientific accuracy, artifact authenticity, or universal "
            "no-fabrication safety."
        ),
        "files_with_findings_or_configuration": [
            entry
            for entry in results
            if entry["findings"] or entry["environment_configuration_calls"]
        ],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 1 if finding_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
