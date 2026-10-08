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
    assert DEVCONTAINER_FILE.is_file(), f"{DEVCONTAINER_FILE} is not a regular file"
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


def test_devcontainer_valid_json_structure(devcontainer_config: dict[str, Any]) -> None:
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
    assert "CoChem-TORQ" in name, f"Container name '{name}' must contain 'CoChem-TORQ'"


def test_devcontainer_build_configuration(devcontainer_config: dict[str, Any]) -> None:
    """Verify Docker build target points to adjacent Dockerfile with context '..'."""
    build_cfg = devcontainer_config.get("build")
    assert isinstance(build_cfg, dict), "build configuration must be an object"
    assert build_cfg.get("dockerfile") == "Dockerfile", (
        "build.dockerfile must reference 'Dockerfile'"
    )
    assert build_cfg.get("context") == "..", "build.context must reference '..'"


def test_devcontainer_workspace_mapping(devcontainer_config: dict[str, Any]) -> None:
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
    devcontainer_config: dict[str, Any],
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


def test_devcontainer_post_create_command(devcontainer_config: dict[str, Any]) -> None:
    """Verify lifecycle routes through the pinned isolated student installer."""
    cmd = devcontainer_config.get("postCreateCommand")
    assert isinstance(cmd, str), "postCreateCommand must be a string"
    assert cmd.split() == ["python", "scripts/student_setup.py", "setup"]
    installer = REPO_ROOT / "scripts" / "student_setup.py"
    assert installer.is_file()
    source = installer.read_text(encoding="utf-8")
    assert "--require-hashes" in source
    assert "postStartCommand" in devcontainer_config
    assert "scripts/student_setup.py start" in devcontainer_config["postStartCommand"]
    assert devcontainer_config["containerEnv"]["COCHEM_CALCULATION_ENVIRONMENT"] == (
        "github-actions"
    )
    assert len(cmd.strip()) > 0, "postCreateCommand must not be empty"


def test_devcontainer_remote_user(devcontainer_config: dict[str, Any]) -> None:
    """Verify remoteUser is set to non-root vscode user."""
    remote_user = devcontainer_config.get("remoteUser")
    assert remote_user == "vscode", f"remoteUser must be 'vscode', got: {remote_user}"


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
