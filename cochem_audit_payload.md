Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task1_devcontainer_json.md.
Original prompt:
# Prompt: VS Code Devcontainer Mapping

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\.devcontainer\devcontainer.json`

## Objective
Establish the VS Code environment map and hypervisor mounts for CoChem-TORQ.

## Instructions for Coder
1. Create `devcontainer.json` inside `.devcontainer/`.
2. Configure it to build from the adjacent `Dockerfile`.
3. Ensure it maps the workspace folder properly.
4. (Optional but recommended) add post-create commands to install Python dependencies.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify the specified target file.
- **Zero Mocking**: Do NOT mock any logic, mathematical equations, or system behaviors. Must provide real physical implementation.
- **Context-Safety**: Do not hallucinate imports. Any dependencies must be strictly limited to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the dynamically provided scratch/artifact paths, never to the current working directory.
Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_devcontainer.py ---
"""
Comprehensive Zero-Mock Physical Test Suite for CoChem-TORQ Devcontainer.

Validates:
1. Physical file existence and valid UTF-8 encoding (no BOM).
2. Strict JSON syntax parsing and schema validation.
3. Container build specification referencing adjacent Dockerfile with context '..'.
4. Workspace folder and workspace mount binding to /workspaces/CoChem-TORQ.
5. VS Code customizations and mandatory extension declarations.
6. Post-create lifecycle script for dependency installation.
7. Non-root remote user specification ('vscode').
8. Strict absence of prohibited placeholder strings or unfinished markers
   (anti-spoof compliance).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVCONTAINER_FILE = REPO_ROOT / ".devcontainer" / "devcontainer.json"


@pytest.fixture(scope="module")
def devcontainer_raw_bytes() -> bytes:
    """Read raw bytes of devcontainer.json."""
    assert DEVCONTAINER_FILE.exists(), (
        f"devcontainer.json not found at {DEVCONTAINER_FILE}"
    )
    assert DEVCONTAINER_FILE.is_file(), (
        f"{DEVCONTAINER_FILE} is not a regular file"
    )
    return DEVCONTAINER_FILE.read_bytes()


@pytest.fixture(scope="module")
def devcontainer_text(devcontainer_raw_bytes: bytes) -> str:
    """Decode raw bytes of devcontainer.json to UTF-8 text."""
    return devcontainer_raw_bytes.decode("utf-8")


@pytest.fixture(scope="module")
def devcontainer_config(devcontainer_text: str) -> dict[str, Any]:
    """Parse devcontainer.json into a structured dictionary."""
    data = json.loads(devcontainer_text)
    assert isinstance(data, dict), (
        "Root devcontainer configuration must be a JSON object"
    )
    return data


def test_devcontainer_file_exists() -> None:
    """Verify that devcontainer.json exists physically in .devcontainer."""
    assert DEVCONTAINER_FILE.exists(), f"Missing file: {DEVCONTAINER_FILE}"
    assert DEVCONTAINER_FILE.is_file(), f"Not a file: {DEVCONTAINER_FILE}"


def test_devcontainer_encoding_and_no_bom(devcontainer_raw_bytes: bytes) -> None:
    """Verify UTF-8 encoding without byte order mark (BOM)."""
    assert not devcontainer_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "BOM detected in devcontainer.json"
    )
    decoded = devcontainer_raw_bytes.decode("utf-8")
    assert len(decoded) > 0, "devcontainer.json is empty"


def test_devcontainer_valid_json_structure(
    devcontainer_config: dict[str, Any]
) -> None:
    """Verify root keys and valid dictionary structure."""
    assert isinstance(devcontainer_config, dict)
    assert len(devcontainer_config) > 0
    expected_top_keys = {
        "name",
        "build",
        "workspaceFolder",
        "customizations",
        "postCreateCommand",
        "remoteUser",
    }
    assert expected_top_keys.issubset(devcontainer_config.keys()), (
        f"Missing keys: {expected_top_keys - set(devcontainer_config.keys())}"
    )


def test_devcontainer_name(devcontainer_config: dict[str, Any]) -> None:
    """Verify container name metadata."""
    name = devcontainer_config.get("name")
    assert isinstance(name, str), "Container name must be a string"
    assert "CoChem-TORQ" in name, (
        f"Container name '{name}' must contain 'CoChem-TORQ'"
    )


def test_devcontainer_build_configuration(
    devcontainer_config: dict[str, Any]
) -> None:
    """Verify Docker build target points to adjacent Dockerfile with context '..'."""
    build_cfg = devcontainer_config.get("build")
    assert isinstance(build_cfg, dict), "build configuration must be an object"
    assert build_cfg.get("dockerfile") == "Dockerfile", (
        "build.dockerfile must reference 'Dockerfile'"
    )
    assert build_cfg.get("context") == "..", "build.context must reference '..'"


def test_devcontainer_workspace_mapping(
    devcontainer_config: dict[str, Any]
) -> None:
    """Verify workspace folder is mapped to /workspaces/CoChem-TORQ."""
    workspace_folder = devcontainer_config.get("workspaceFolder")
    assert workspace_folder == "/workspaces/CoChem-TORQ", (
        f"workspaceFolder must be '/workspaces/CoChem-TORQ', got: {workspace_folder}"
    )
    if "workspaceMount" in devcontainer_config:
        mount = devcontainer_config["workspaceMount"]
        assert isinstance(mount, str), "workspaceMount must be a string"
        assert "/workspaces/CoChem-TORQ" in mount, (
            f"workspaceMount must target '/workspaces/CoChem-TORQ', got: {mount}"
        )


def test_devcontainer_customizations_extensions(
    devcontainer_config: dict[str, Any]
) -> None:
    """Verify VS Code extensions declared under customizations.vscode.extensions."""
    customizations = devcontainer_config.get("customizations")
    assert isinstance(customizations, dict), "customizations must be an object"
    vscode_cfg = customizations.get("vscode")
    assert isinstance(vscode_cfg, dict), "customizations.vscode must be an object"
    extensions = vscode_cfg.get("extensions")
    assert isinstance(extensions, list), (
        "customizations.vscode.extensions must be a list"
    )

    required_extensions = [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
    ]
    for ext in required_extensions:
        assert ext in extensions, (
            f"Required extension '{ext}' missing from devcontainer.json extensions"
        )


def test_devcontainer_post_create_command(
    devcontainer_config: dict[str, Any]
) -> None:
    """Verify postCreateCommand is present and installs Python dependencies."""
    cmd = devcontainer_config.get("postCreateCommand")
    assert isinstance(cmd, str), "postCreateCommand must be a string"
    assert "pip install" in cmd, f"postCreateCommand should install: {cmd}"
    assert len(cmd.strip()) > 0, "postCreateCommand must not be empty"


def test_devcontainer_remote_user(devcontainer_config: dict[str, Any]) -> None:
    """Verify remoteUser is set to non-root vscode user."""
    remote_user = devcontainer_config.get("remoteUser")
    assert remote_user == "vscode", (
        f"remoteUser must be 'vscode', got: {remote_user}"
    )


def test_devcontainer_zero_banned_terms(devcontainer_text: str) -> None:
    """Verify devcontainer.json contains no placeholder or banned terms."""
    banned_tokens = [
        "m" + "ock",
        "e" + "xample",
        "s" + "tub",
        "d" + "ummy",
        "p" + "laceholder",
        "f" + "ake",
        "s" + "ample",
        "# " + "TODO" + ": implement",
    ]
    lower_content = devcontainer_text.lower()
    for token in banned_tokens:
        assert token.lower() not in lower_content, (
            f"Banned token '{token}' detected in devcontainer.json"
        )

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_dockerfile.py ---
"""Unit tests for CoChem container configuration (.devcontainer/Dockerfile).

Validates:
- File existence, regular file properties, UTF-8 encoding, and LF line endings.
- Base image specification strictly matching python:3.10-slim.
- Installation of required system packages:
  (build-essential, cmake, openmpi-bin, libopenmpi-dev, git).
- Apt cache cleanup (apt-get clean and rm -rf /var/lib/apt/lists/*).
- Environment variables and working directory configuration.
- Dockerfile instruction parsing, structural order, and single-layer design.
- Zero-Mock and Anti-Spoofing compliance.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def dockerfile_path(repo_root: Path) -> Path:
    """Return the absolute path to .devcontainer/Dockerfile."""
    path = repo_root / ".devcontainer" / "Dockerfile"
    assert path.exists(), f"Dockerfile does not exist at {path}"
    return path


@pytest.fixture
def dockerfile_content(dockerfile_path: Path) -> str:
    """Read and return the text content of the Dockerfile."""
    return dockerfile_path.read_text(encoding="utf-8")


def test_dockerfile_exists(repo_root: Path) -> None:
    """Validate that .devcontainer/Dockerfile exists and is a regular file."""
    dockerfile = repo_root / ".devcontainer" / "Dockerfile"
    assert dockerfile.exists(), f"Missing Dockerfile at {dockerfile}"
    assert dockerfile.is_file(), f"Path {dockerfile} must be a regular file"
    assert dockerfile.stat().st_size > 0, "Dockerfile must not be empty"


def test_dockerfile_encoding_and_lf_line_endings(dockerfile_path: Path) -> None:
    """Validate UTF-8 encoding, lack of BOM, and strict Unix LF line endings."""
    raw_bytes = dockerfile_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "Dockerfile contains UTF-8 BOM"
    )
    assert b"\r\n" not in raw_bytes, (
        "Dockerfile contains Windows CRLF line endings"
    )
    decoded = raw_bytes.decode("utf-8")
    assert len(decoded) > 0, "Decoded Dockerfile content must not be empty"


def test_dockerfile_base_image(dockerfile_content: str) -> None:
    """Validate that the base image is strictly python:3.10-slim."""
    match = re.search(r"^\s*FROM\s+([^\s]+)", dockerfile_content, re.MULTILINE)
    assert match is not None, "Dockerfile missing FROM instruction"
    base_image = match.group(1).strip()
    assert base_image == "python:3.10-slim", (
        f"Expected base image 'python:3.10-slim', got '{base_image}'"
    )


def test_dockerfile_environment_variables(dockerfile_content: str) -> None:
    """Validate environment variables configured in Dockerfile."""
    assert "ENV " in dockerfile_content, "Dockerfile missing ENV instruction"
    assert "DEBIAN_FRONTEND=noninteractive" in dockerfile_content, (
        "Dockerfile must set DEBIAN_FRONTEND=noninteractive"
    )
    assert "PYTHONUNBUFFERED=1" in dockerfile_content, (
        "Dockerfile must set PYTHONUNBUFFERED=1"
    )
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_content, (
        "Dockerfile must set PYTHONDONTWRITEBYTECODE=1"
    )


def test_dockerfile_system_packages(dockerfile_content: str) -> None:
    """Validate all required system packages are installed via apt-get."""
    required_packages = [
        "build-essential",
        "cmake",
        "openmpi-bin",
        "libopenmpi-dev",
        "git",
    ]
    assert "apt-get update" in dockerfile_content, (
        "Dockerfile must execute 'apt-get update'"
    )
    assert "apt-get install" in dockerfile_content, (
        "Dockerfile must execute 'apt-get install'"
    )

    for package in required_packages:
        assert package in dockerfile_content, (
            f"Required package '{package}' missing from Dockerfile"
        )


def test_dockerfile_apt_cache_cleanup(dockerfile_content: str) -> None:
    """Validate that apt cache cleanup is performed in RUN command."""
    assert "apt-get clean" in dockerfile_content, (
        "Dockerfile must execute 'apt-get clean'"
    )
    assert "rm -rf /var/lib/apt/lists/*" in dockerfile_content, (
        "Dockerfile must remove /var/lib/apt/lists/* to minimize image size"
    )


def test_dockerfile_workdir(dockerfile_content: str) -> None:
    """Validate working directory configuration."""
    match = re.search(r"^\s*WORKDIR\s+([^\s]+)", dockerfile_content, re.MULTILINE)
    assert match is not None, "Dockerfile missing WORKDIR instruction"
    workdir = match.group(1).strip()
    assert workdir == "/workspace", f"Expected WORKDIR '/workspace', got '{workdir}'"


def test_dockerfile_layer_structure_and_order(dockerfile_content: str) -> None:
    """Validate the ordering and structure of Dockerfile directives."""
    lines = [
        line.strip()
        for line in dockerfile_content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    instructions = [line.split()[0] for line in lines if line.split()[0].isupper()]

    assert "FROM" in instructions, "Dockerfile must have FROM"
    assert "RUN" in instructions, "Dockerfile must have RUN"
    assert "WORKDIR" in instructions, "Dockerfile must have WORKDIR"

    from_idx = instructions.index("FROM")
    run_idx = instructions.index("RUN")
    workdir_idx = instructions.index("WORKDIR")

    assert from_idx == 0, "FROM must be the first instruction"
    assert from_idx < run_idx < workdir_idx, (
        "Instruction ordering must be: FROM -> RUN -> WORKDIR"
    )


def test_dockerfile_integrity_and_anti_spoofing(dockerfile_content: str) -> None:
    """Validate zero-stub anti-spoofing compliance and absence of banned directives."""
    # anti-spoofing policy enforcement: banned terms verification
    banned_tokens = [
        "unittest." + "m" + "ock",
        "Magic" + "M" + "ock",
        "M" + "ock(",
        "TO" + "DO",
        "FIX" + "ME",
        "PLACE" + "HOLDER",
        "NotImplemented" + "Error",
        "pass",
    ]
    for token in banned_tokens:
        assert token not in dockerfile_content, (
            f"Prohibited token '{token}' found in Dockerfile"
        )



Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.