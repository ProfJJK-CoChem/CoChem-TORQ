"""Hardened Allowlist-Based AST Pre-Import Security Scanner.
Enforces static verification of plugins before dynamic import or execution.
Blocks unauthorized modules, builtins, reflection hooks, and dunder traversals.
"""

from __future__ import annotations

import ast
from typing import Set


class PluginSecurityViolationError(PermissionError):
    """Raised when plugin code violates AST security constraints or execution policies."""


class PluginASTSecurityScanner(ast.NodeVisitor):
    """Authoritative AST inspector enforcing strict module allowlisting and token blacklisting."""

    ALLOWED_MODULES: Set[str] = {
        "math",
        "numpy",
        "scipy",
        "mendeleev",
        "dataclasses",
        "typing",
        "collections",
        "itertools",
        "functools",
        "json",
        "re",
        "abc",
        "cochem",
        "src",
    }

    PROHIBITED_NAMES: Set[str] = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "__builtins__",
        "__subclasses__",
        "__bases__",
        "__class__",
    }

    def visit_Import(self, node: ast.Import) -> None:
        """Inspect and restrict all direct import statements."""
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod not in self.ALLOWED_MODULES:
                raise PluginSecurityViolationError(
                    f"Prohibited module import: {alias.name}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Inspect and restrict all from-import statements and imported members."""
        if not node.module:
            raise PluginSecurityViolationError(
                "Relative imports without module name are prohibited."
            )

        root_mod = node.module.split(".")[0]
        if root_mod not in self.ALLOWED_MODULES:
            raise PluginSecurityViolationError(
                f"Prohibited from-import module: {node.module}"
            )

        for alias in node.names:
            if alias.name == "*":
                raise PluginSecurityViolationError(
                    "Wildcard imports ('from ... import *') are prohibited."
                )
            if alias.name in self.PROHIBITED_NAMES or alias.name.startswith("_"):
                raise PluginSecurityViolationError(
                    f"Prohibited member import: {alias.name}"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Inspect attribute access and block private/dunder traversal."""
        if node.attr.startswith("_") or node.attr in self.PROHIBITED_NAMES:
            raise PluginSecurityViolationError(
                f"Prohibited attribute or dunder access: {node.attr}"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Inspect identifier references to block prohibited global/builtin identifiers."""
        if node.id in self.PROHIBITED_NAMES:
            raise PluginSecurityViolationError(
                f"Prohibited identifier reference: {node.id}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect function and method invocations for prohibited execution targets."""
        if isinstance(node.func, ast.Name) and node.func.id in self.PROHIBITED_NAMES:
            raise PluginSecurityViolationError(f"Prohibited call: {node.func.id}")
        self.generic_visit(node)


def scan_plugin_source(source_code: str, filename: str = "<plugin>") -> None:
    """Parse and statically verify plugin source code prior to dynamic import."""
    tree = ast.parse(source_code, filename=filename)
    scanner = PluginASTSecurityScanner()
    scanner.visit(tree)
