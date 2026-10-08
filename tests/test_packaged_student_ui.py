"""Packaged notebook delivery and actual UI CLI, without runtime replacement."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.resources import files

import pytest

from cochem_torq.ui import prepare_notebook


def test_packaged_notebook_is_copied_and_existing_student_outputs_are_preserved(
    tmp_path,
):
    notebook = prepare_notebook(tmp_path / "interface")
    original = files("UI").joinpath("Start_TORQ.ipynb").read_bytes()
    assert notebook.read_bytes() == original
    updated = json.loads(original)
    updated["metadata"]["student_note"] = "Actual local student workspace"
    changed = json.dumps(updated).encode()
    notebook.write_bytes(changed)
    assert prepare_notebook(tmp_path / "interface") == notebook
    assert notebook.read_bytes() == changed
    assert files("UI").joinpath("Start_TORQ.ipynb").read_bytes() == original


def test_packaged_notebook_rejects_a_symlink_without_touching_the_original(tmp_path):
    workspace = tmp_path / "interface"
    notebook = prepare_notebook(workspace)
    original = tmp_path / "original.ipynb"
    original.write_bytes(notebook.read_bytes())
    notebook.unlink()
    notebook.symlink_to(original)
    before = original.read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        prepare_notebook(workspace)
    assert original.read_bytes() == before


def test_packaged_ui_cli_help_never_starts_jupyter_or_creates_a_workspace(tmp_path):
    workspace = tmp_path / "uncreated"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.ui",
            "--help",
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "cochem-torq-ui" in process.stdout
    assert "--workspace" in process.stdout
    assert not workspace.exists()


def test_packaged_ui_rejects_an_invalid_port_before_creating_notebook(tmp_path):
    workspace = tmp_path / "uncreated"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cochem_torq.ui",
            "--port",
            "0",
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "nonprivileged port" in process.stderr
    assert not workspace.exists()
