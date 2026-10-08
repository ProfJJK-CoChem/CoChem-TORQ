"""Actual process checks for external native-build publication boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPOSITORY = Path(__file__).absolute().parents[1]
HELPER = REPOSITORY / "ci_tools" / "provision_pickett_validation.py"


def invoke(root: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY / "src"), str(REPOSITORY))
    )
    return subprocess.run(
        [sys.executable, str(HELPER), "--root", str(root)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_provisioning_cannot_create_or_bundle_upstream_inside_checkout(tmp_path):
    root = REPOSITORY / f"prohibited-pickett-build-{uuid4().hex}"
    result = invoke(root, tmp_path)
    assert result.returncode == 1
    assert "outside repository checkouts" in result.stderr
    assert result.stdout == ""
    assert not root.exists()


def test_existing_external_root_is_never_overwritten(tmp_path):
    root = tmp_path / "retained-installation"
    root.mkdir()
    sentinel = root / "retained-evidence.txt"
    sentinel.write_text("Retain this actual preexisting evidence.\n")
    result = invoke(root, tmp_path)
    assert result.returncode == 1
    assert "fresh root; no overwrite" in result.stderr
    assert sentinel.read_text() == "Retain this actual preexisting evidence.\n"
    assert sorted(path.name for path in root.iterdir()) == [sentinel.name]


def test_another_actual_git_checkout_cannot_receive_upstream_files(tmp_path):
    checkout = tmp_path / "another-repository"
    subprocess.run(
        ["git", "init", str(checkout)], check=True, capture_output=True, timeout=30
    )
    root = checkout / "upstream-build"
    result = invoke(root, tmp_path)
    assert result.returncode == 1
    assert "outside repository checkouts" in result.stderr
    assert not root.exists()


def test_symlink_ancestor_cannot_redirect_the_external_build(tmp_path):
    destination = tmp_path / "actual-destination"
    destination.mkdir()
    alias = tmp_path / "redirect"
    alias.symlink_to(destination, target_is_directory=True)
    result = invoke(alias / "new-installation", tmp_path)
    assert result.returncode == 1
    assert "cannot traverse symlinks" in result.stderr
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "root", [Path("relative-installation"), Path("/tmp/../tmp/new")]
)
def test_relative_and_traversal_roots_are_rejected_before_network(root, tmp_path):
    result = invoke(root, tmp_path)
    assert result.returncode == 1
    assert "absolute without traversal" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_external_parent_is_not_silently_created(tmp_path):
    root = tmp_path / "missing-parent" / "new-installation"
    result = invoke(root, tmp_path)
    assert result.returncode == 1
    assert "parent must already exist" in result.stderr
    assert list(tmp_path.iterdir()) == []
