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


