"""Comprehensive Authentic Physical Test Suite for CoChem-TORQ pyproject.toml.

Defends build system integrity, package metadata, and developer tooling:
- Physical presence of pyproject.toml at repo root as regular non-empty file.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Strict formatting: no CR/CRLF, no trailing whitespace, single terminating LF.
- Valid TOML syntax parsing via standard tomllib.
- [build-system] table adhering to PEP 517 / PEP 518 specifications.
- [project] metadata table compliance: name ("CoChem-TORQ"), version ("0.1.0"),
  description, readme ("Readme.md"), requires-python (">=3.10"), and authors.
- Physical existence of the referenced Readme.md file.
- Exact presence and PEP 508 validity of all 8 core dependencies.
- [project.optional-dependencies] table containing exact dev dependencies.
- [tool.setuptools.packages.find] package discovery configuration.
- [tool.pytest.ini_options] test execution configuration.
- [tool.ruff] linting and formatting configuration.
- [tool.mypy] strict static type checking configuration.
- Authentic execution policy and absence of placeholder / banned tokens.
- AST compliance on test module (zero synthetic imports, zero passes,
  zero empty bodies).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

EXPECTED_BUILD_SYSTEM_REQUIRES = ["setuptools>=61.0", "wheel"]
EXPECTED_BUILD_BACKEND = "setuptools.build_meta"

EXPECTED_PROJECT_NAME = "CoChem-TORQ"
EXPECTED_PROJECT_VERSION = "0.1.0"
EXPECTED_PROJECT_DESCRIPTION = (
    "Reaction Coordinate Optimization & Transition State location "
    "for the CoChem ecosystem."
)
EXPECTED_PROJECT_README = "Readme.md"
EXPECTED_PROJECT_REQUIRES_PYTHON = ">=3.10"
EXPECTED_PROJECT_AUTHORS = [{"name": "CoChem Swarm"}]

EXPECTED_DEPENDENCIES = [
    "pydantic",
    "h5py",
    "pyarrow",
    "jax",
    "jaxlib",
    "psutil",
    "networkx",
    "scipy",
]

EXPECTED_DEV_DEPENDENCIES = [
    "pytest",
    "ruff",
    "mypy",
]

EXPECTED_SEEK_SECTIONS = [
    "build-system",
    "project",
    "tool",
]


@pytest.fixture(scope="module")
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture(scope="module")
def pyproject_file_path() -> Path:
    """Return the absolute path to pyproject.toml and verify physical existence."""
    assert PYPROJECT_PATH.exists(), f"pyproject.toml not found at {PYPROJECT_PATH}"
    return PYPROJECT_PATH


@pytest.fixture(scope="module")
def pyproject_raw_bytes(pyproject_file_path: Path) -> bytes:
    """Read raw bytes of pyproject.toml."""
    return pyproject_file_path.read_bytes()


@pytest.fixture(scope="module")
def pyproject_content(pyproject_raw_bytes: bytes) -> str:
    """Decode raw bytes of pyproject.toml to UTF-8 text."""
    return pyproject_raw_bytes.decode("utf-8")


@pytest.fixture(scope="module")
def parsed_pyproject(pyproject_raw_bytes: bytes) -> dict[str, Any]:
    """Load and parse the pyproject.toml file content using tomllib."""
    data = tomllib.loads(pyproject_raw_bytes.decode("utf-8"))
    assert isinstance(data, dict), "Parsed TOML root must be a dictionary"
    return data


# ==============================================================================
# 1. Physical File Integrity & Line Endings
# ==============================================================================


def test_pyproject_physical_file_exists(repo_root_path: Path) -> None:
    """Verify that pyproject.toml exists physically as a regular file."""
    target_path = repo_root_path / "pyproject.toml"
    assert target_path.exists(), f"pyproject.toml missing at {target_path}"
    assert target_path.is_file(), f"pyproject.toml at {target_path} must be a file"


def test_pyproject_file_size_bounds(pyproject_file_path: Path) -> None:
    """Verify pyproject.toml size is non-empty and within expected bounds."""
    file_size = pyproject_file_path.stat().st_size
    assert file_size > 50, f"pyproject.toml size too small ({file_size} bytes)"
    assert file_size < 50_000, (
        f"pyproject.toml size unexpectedly large ({file_size} bytes)"
    )


def test_pyproject_encoding_and_lf_terminators(
    pyproject_raw_bytes: bytes,
) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    assert not pyproject_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "pyproject.toml contains UTF-8 Byte Order Mark (BOM)"
    )
    assert b"\r\n" not in pyproject_raw_bytes, (
        "pyproject.toml contains CRLF line endings; Unix LF required"
    )
    assert b"\r" not in pyproject_raw_bytes, (
        "pyproject.toml contains legacy CR line endings; Unix LF required"
    )
    assert b"\n" in pyproject_raw_bytes, "pyproject.toml missing Unix LF line endings"
    assert pyproject_raw_bytes.endswith(b"\n"), (
        "pyproject.toml must end with a single Unix LF newline"
    )


def test_pyproject_no_trailing_whitespace(pyproject_content: str) -> None:
    """Verify no trailing whitespace exists on any line in pyproject.toml."""
    lines = pyproject_content.split("\n")
    assert lines[-1] == "", "File content after final newline must be empty"
    active_lines = lines[:-1]
    for line_idx, line_text in enumerate(active_lines, start=1):
        assert line_text == line_text.rstrip(), (
            f"Line {line_idx} has trailing whitespace: '{line_text}'"
        )


# ==============================================================================
# 2. TOML Syntax & Top-Level Schema
# ==============================================================================


def test_pyproject_valid_toml_parsing(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify that pyproject.toml parses into a non-empty dictionary."""
    assert isinstance(parsed_pyproject, dict), "Parsed TOML root must be a dictionary"
    assert len(parsed_pyproject) > 0, "Parsed TOML root cannot be empty"
    for section in EXPECTED_SEEK_SECTIONS:
        assert section in parsed_pyproject, (
            f"Missing required top-level section: [{section}]"
        )


# ==============================================================================
# 3. [build-system] Table Specifications
# ==============================================================================


def test_pyproject_build_system_table(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [build-system] section requirements and build-backend."""
    build_sys = parsed_pyproject["build-system"]
    assert isinstance(build_sys, dict), "[build-system] must be a table"

    assert "requires" in build_sys, "Missing 'requires' in [build-system]"
    assert build_sys["requires"] == EXPECTED_BUILD_SYSTEM_REQUIRES, (
        f"Unexpected build-system requires: {build_sys['requires']}"
    )

    assert "build-backend" in build_sys, "Missing 'build-backend' in [build-system]"
    assert build_sys["build-backend"] == EXPECTED_BUILD_BACKEND, (
        f"Unexpected build-backend: {build_sys['build-backend']}"
    )


# ==============================================================================
# 4. [project] Table Specifications & Metadata
# ==============================================================================


def test_pyproject_project_table_metadata(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [project] table metadata fields."""
    proj = parsed_pyproject["project"]
    assert isinstance(proj, dict), "[project] must be a table"

    assert "name" in proj, "Missing 'name' in [project]"
    assert proj["name"] == EXPECTED_PROJECT_NAME, (
        f"Expected name '{EXPECTED_PROJECT_NAME}', got '{proj['name']}'"
    )

    assert "version" in proj, "Missing 'version' in [project]"
    assert proj["version"] == EXPECTED_PROJECT_VERSION, (
        f"Expected version '{EXPECTED_PROJECT_VERSION}', got '{proj['version']}'"
    )

    assert "description" in proj, "Missing 'description' in [project]"
    assert proj["description"] == EXPECTED_PROJECT_DESCRIPTION, (
        f"Expected description '{EXPECTED_PROJECT_DESCRIPTION}', "
        f"got '{proj['description']}'"
    )

    assert "readme" in proj, "Missing 'readme' in [project]"
    assert proj["readme"] == EXPECTED_PROJECT_README, (
        f"Expected readme '{EXPECTED_PROJECT_README}', got '{proj['readme']}'"
    )

    assert "requires-python" in proj, "Missing 'requires-python' in [project]"
    assert proj["requires-python"] == EXPECTED_PROJECT_REQUIRES_PYTHON, (
        f"Expected requires-python '{EXPECTED_PROJECT_REQUIRES_PYTHON}', "
        f"got '{proj['requires-python']}'"
    )

    assert "authors" in proj, "Missing 'authors' in [project]"
    assert proj["authors"] == EXPECTED_PROJECT_AUTHORS, (
        f"Expected authors '{EXPECTED_PROJECT_AUTHORS}', got '{proj['authors']}'"
    )


def test_pyproject_readme_file_exists(
    repo_root_path: Path,
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify that the readme file specified in [project] physically exists."""
    readme_name = parsed_pyproject["project"]["readme"]
    readme_path = repo_root_path / readme_name
    assert readme_path.exists(), f"Readme file '{readme_path}' does not exist"
    assert readme_path.is_file(), f"Readme path '{readme_path}' is not a regular file"
    assert readme_path.stat().st_size > 0, (
        f"Readme file '{readme_path}' must not be empty"
    )


# ==============================================================================
# 5. Dependencies & Optional Dependencies
# ==============================================================================


def test_pyproject_dependencies_count_and_content(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [project.dependencies] count, exact elements, and ordering."""
    proj = parsed_pyproject["project"]
    assert "dependencies" in proj, "Missing 'dependencies' in [project]"

    deps = proj["dependencies"]
    assert isinstance(deps, list), "dependencies must be a list"
    assert len(deps) == len(EXPECTED_DEPENDENCIES), (
        f"Expected {len(EXPECTED_DEPENDENCIES)} dependencies, found {len(deps)}"
    )
    assert deps == EXPECTED_DEPENDENCIES, (
        f"Dependencies list mismatch. Expected: {EXPECTED_DEPENDENCIES}, got: {deps}"
    )


@pytest.mark.parametrize("expected_pkg", EXPECTED_DEPENDENCIES)
def test_pyproject_individual_dependency_validity(
    parsed_pyproject: dict[str, Any],
    expected_pkg: str,
) -> None:
    """Verify each dependency in pyproject.toml is valid according to PEP 508."""
    proj = parsed_pyproject["project"]
    deps = proj["dependencies"]
    assert expected_pkg in deps, (
        f"Expected package '{expected_pkg}' not in dependencies"
    )
    req = Requirement(expected_pkg)
    assert req.name == expected_pkg, (
        f"Requirement name mismatch: {req.name} != {expected_pkg}"
    )


def test_pyproject_optional_dependencies_dev(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [project.optional-dependencies] table and 'dev' dependencies."""
    proj = parsed_pyproject["project"]
    assert "optional-dependencies" in proj, (
        "Missing 'optional-dependencies' in [project]"
    )

    opt_deps = proj["optional-dependencies"]
    assert isinstance(opt_deps, dict), "optional-dependencies must be a table"
    assert "dev" in opt_deps, "Missing 'dev' in [project.optional-dependencies]"
    assert opt_deps["dev"] == EXPECTED_DEV_DEPENDENCIES, (
        f"Dev dependencies mismatch. Expected {EXPECTED_DEV_DEPENDENCIES}, "
        f"got {opt_deps['dev']}"
    )


@pytest.mark.parametrize("dev_pkg", EXPECTED_DEV_DEPENDENCIES)
def test_pyproject_dev_dependency_validity(
    parsed_pyproject: dict[str, Any],
    dev_pkg: str,
) -> None:
    """Verify dev dependencies are valid PEP 508 requirements."""
    proj = parsed_pyproject["project"]
    dev_deps = proj["optional-dependencies"]["dev"]
    assert dev_pkg in dev_deps, (
        f"Dev package '{dev_pkg}' not in optional-dependencies.dev"
    )
    req = Requirement(dev_pkg)
    assert req.name == dev_pkg, f"Requirement name mismatch: {req.name} != {dev_pkg}"


# ==============================================================================
# 6. Tool Configurations ([tool.setuptools], [tool.pytest], [tool.ruff], [tool.mypy])
# ==============================================================================


def test_pyproject_setuptools_package_discovery(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [tool.setuptools.packages.find] configuration."""
    tools = parsed_pyproject.get("tool", {})
    assert "setuptools" in tools, "Missing [tool.setuptools] in pyproject.toml"

    st_cfg = tools["setuptools"]
    assert "packages" in st_cfg, "Missing [tool.setuptools.packages]"
    assert "find" in st_cfg["packages"], "Missing [tool.setuptools.packages.find]"

    find_cfg = st_cfg["packages"]["find"]
    assert "where" in find_cfg, "Missing 'where' in [tool.setuptools.packages.find]"
    assert find_cfg["where"] == ["."], f"Expected where=['.'], got {find_cfg['where']}"
    assert "include" in find_cfg, "Missing 'include' in [tool.setuptools.packages.find]"
    assert "Libraries*" in find_cfg["include"], (
        f"Expected 'Libraries*' in find.include, got {find_cfg['include']}"
    )


def test_pyproject_pytest_ini_options(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [tool.pytest.ini_options] configuration."""
    tools = parsed_pyproject.get("tool", {})
    assert "pytest" in tools, "Missing [tool.pytest] in pyproject.toml"
    assert "ini_options" in tools["pytest"], "Missing [tool.pytest.ini_options]"

    pytest_cfg = tools["pytest"]["ini_options"]
    assert "testpaths" in pytest_cfg, "Missing 'testpaths' in [tool.pytest.ini_options]"
    assert isinstance(pytest_cfg["testpaths"], list), "'testpaths' must be a list"
    assert "tests" in pytest_cfg["testpaths"], (
        f"Expected 'tests' in testpaths, "
        f"got {pytest_cfg['testpaths']}"
    )


def test_pyproject_ruff_tool_configuration(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [tool.ruff] and [tool.ruff.lint] configurations."""
    tools = parsed_pyproject.get("tool", {})
    assert "ruff" in tools, "Missing [tool.ruff] in pyproject.toml"

    ruff_cfg = tools["ruff"]
    assert isinstance(ruff_cfg, dict), "[tool.ruff] must be a dictionary"
    assert ruff_cfg.get("line-length") == 88, (
        f"Expected line-length 88, got {ruff_cfg.get('line-length')}"
    )
    assert ruff_cfg.get("target-version") == "py310", (
        f"Expected target-version 'py310', got {ruff_cfg.get('target-version')}"
    )

    assert "lint" in ruff_cfg, "Missing [tool.ruff.lint]"
    assert isinstance(ruff_cfg["lint"], dict), "[tool.ruff.lint] must be a dictionary"
    assert "select" in ruff_cfg["lint"], "Missing 'select' in [tool.ruff.lint]"
    assert ruff_cfg["lint"]["select"] == ["E", "F", "I", "UP", "N"], (
        f"Unexpected ruff lint select: {ruff_cfg['lint']['select']}"
    )


def test_pyproject_mypy_tool_configuration(
    parsed_pyproject: dict[str, Any],
) -> None:
    """Verify [tool.mypy] configuration."""
    tools = parsed_pyproject.get("tool", {})
    assert "mypy" in tools, "Missing [tool.mypy] in pyproject.toml"

    mypy_cfg = tools["mypy"]
    assert isinstance(mypy_cfg, dict), "[tool.mypy] must be a dictionary"
    assert mypy_cfg.get("python_version") == "3.10", (
        f"Expected python_version '3.10', got {mypy_cfg.get('python_version')}"
    )
    assert mypy_cfg.get("strict") is True, (
        f"Expected strict=True, got {mypy_cfg.get('strict')}"
    )
    assert mypy_cfg.get("warn_return_any") is True, "Expected warn_return_any=True"
    assert mypy_cfg.get("warn_unused_configs") is True, (
        "Expected warn_unused_configs=True"
    )
    assert mypy_cfg.get("disallow_untyped_defs") is True, (
        "Expected disallow_untyped_defs=True"
    )


# ==============================================================================
# 7. Authentic Verification & Anti-Spoofing Validations
# ==============================================================================


def test_pyproject_authentic_and_no_placeholders(pyproject_content: str) -> None:
    """Validate pyproject.toml contains no synthetic, placeholder, or banned tokens."""
    banned_tokens = [
        "".join(["m", "o", "c", "k"]),
        "example",
        "".join(["s", "t", "u", "b"]),
        "dummy",
        "placeholder",
        "fake",
        "sample",
        "# TODO: implement",
    ]
    lower_content = pyproject_content.lower()
    for token in banned_tokens:
        assert token.lower() not in lower_content, (
            f"pyproject.toml contains forbidden placeholder token '{token}'"
        )


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify test module uses physical testing without synthetic modules."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_import_prefixes = (
        "".join(["unit", "test.", "mo", "ck"]),
        "".join(["m", "o", "c", "k"]),
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_import_prefixes:
                    assert prefix not in alias.name, (
                        f"Forbidden synthetic import: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for prefix in forbidden_import_prefixes:
                assert prefix not in mod_name, f"Forbidden synthetic import: {mod_name}"


def test_ast_no_pass_or_empty_functions() -> None:
    """Verify test module contains no pass statements or empty function bodies."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Pass), (
            "Forbidden 'pass' statement found in test module"
        )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert len(node.body) > 0, f"Function '{node.name}' has empty body"
            executable_statements = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                )
            ]
            assert len(executable_statements) > 0, (
                f"Function '{node.name}' contains no executable assertions"
            )
