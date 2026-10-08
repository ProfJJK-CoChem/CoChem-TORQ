"""Genuine wheel and Git ownership checks; no credential or science substitutes.

The fixture builds and installs a real wheel in a dedicated environment, using
the entry environment's third-party dependencies. Editable/source CI is supported.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cochem_torq.application import _source_git_repository


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def installed_artifact(tmp_path_factory):
    """Build actual source; never execute the parent's editable installation hooks."""
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("cochem_exec_source_identity_artifact")
    source = root / "source"
    shutil.copytree(
        repository,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "build",
            "dist",
            "*.egg-info",
            ".venv",
            ".venvs",
            "venv",
        ),
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "PYTHON", "LD_", "DYLD_"))
        and not any(
            word in key.upper()
            for word in (
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "AUTHORIZATION",
            )
        )
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    wheels = root / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheels),
            str(source),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    built = list(wheels.glob("*.whl"))
    assert len(built) == 1
    prefix = root / "installed-env"
    subprocess.run(
        [sys.executable, "-I", "-B", "-m", "venv", str(prefix)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    program = "import sysconfig;print(sysconfig.get_path('purelib'))"
    purelib = Path(
        subprocess.check_output(
            [str(python), "-I", "-B", "-c", program],
            env=environment,
            text=True,
        ).strip()
    )
    dependency_paths = sorted(
        {
            str(Path(str(dist.locate_file(""))).resolve())
            for dist in importlib.metadata.distributions()
            if not dist.metadata.get("Name", "").lower().startswith("cochem")
        }
    )
    assert dependency_paths
    # Adding a path does not execute that directory's editable .pth hooks. Own
    # wheel modules take precedence; all science origins are checked below.
    (purelib / "third-party-dependency-profile.pth").write_text(
        "\n".join(dependency_paths) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-installed",
            str(built[0]),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return python


def observed_installed_identity(python: Path) -> dict:
    program = """import json,sys,pathlib
import cochem_torq.application as app
from importlib.metadata import distribution
prefix=pathlib.Path(sys.prefix).resolve()
origin=pathlib.Path(app.__file__).resolve()
assert origin.is_relative_to(prefix), 'A noneditable installed TORQ wheel is required'
dist=distribution('CoChem-TORQ')
assert pathlib.Path(dist.locate_file('cochem_torq/application.py')).resolve()==origin
import importlib.util
for name in app.RUNTIME_SOURCE_MODULES:
 origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
 assert origin.is_relative_to(prefix)
print(json.dumps({'identity':app.source_identity(),'prefix':str(prefix),'origin':str(origin)}))
"""
    process = subprocess.run(
        [str(python), "-I", "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(process.stdout)


@pytest.mark.parametrize(
    "ancestor", ["placeholder", "unrelated_git", "installed_file_in_git"]
)
def test_actual_installed_wheel_does_not_inherit_ancestor_git_authority(
    ancestor,
    installed_artifact,
):
    before = observed_installed_identity(installed_artifact)
    prefix = Path(before["prefix"])
    metadata = prefix / ".git"
    note = prefix / "source-identity-test-unrelated.txt"
    assert not metadata.exists() and not note.exists(), (
        "Use a dedicated clean installed-wheel test prefix."
    )
    try:
        if ancestor == "placeholder":
            metadata.mkdir()
            (metadata / "placeholder").write_text("Not a Git source checkout.\n")
        else:
            git(prefix, "init", "--quiet")
            git(prefix, "config", "user.name", "Source identity test")
            git(prefix, "config", "user.email", "identity@example.invalid")
            if ancestor == "unrelated_git":
                note.write_text("Unrelated repository content.\n")
                git(prefix, "add", note.name)
            else:
                git(
                    prefix, "add", Path(before["origin"]).relative_to(prefix).as_posix()
                )
            git(prefix, "commit", "--quiet", "-m", "Actual Git identity boundary test")
        after = observed_installed_identity(installed_artifact)
        assert after["identity"]["git_commit"] is None
        assert after["identity"]["working_tree_dirty"] is None
        assert after["identity"]["code_sha256"] == before["identity"]["code_sha256"]
        # A genuine plan and approval must still work, not just a metadata probe.
        program = """import json
from cochem_torq.service import plan_request,approve_plan,validate_approved_plan
request={'schema_version':'cochem.torq.request/1',
'molecule':{'symbols':['H','H'],'geometry_bohr':[[0.,0.,0.],[0.,0.,1.4]],'charge':0,'multiplicity':1},
'recipe':'hf-sto-3g-education','products':['geometry'],
'resources':{'cores':1,'memory_mb':1024,'wall_seconds':60}}
approved=approve_plan(plan_request(request),actor='Actual installed Git boundary test')
assert approved['source_identity']['git_commit'] is None
assert validate_approved_plan(approved).source_identity['code_sha256']
print(json.dumps({'approval_validated':True}))
"""
        process = subprocess.run(
            [str(installed_artifact), "-I", "-B", "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert json.loads(process.stdout) == {"approval_validated": True}
    finally:
        shutil.rmtree(metadata, ignore_errors=False)
        note.unlink(missing_ok=True)
    assert (
        observed_installed_identity(installed_artifact)["identity"]
        == before["identity"]
    )


@pytest.fixture
def actual_checkout(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "real-torq-checkout"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(repository), str(checkout)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return checkout


def test_genuine_source_checkout_owns_tracked_implementation(actual_checkout):
    package = actual_checkout / "src/cochem_torq"
    assert _source_git_repository(package) == actual_checkout
    assert (
        git(actual_checkout, "ls-files", "src/cochem_torq/application.py")
        == "src/cochem_torq/application.py"
    )


def test_public_source_identity_preserves_real_checkout_commit_and_dirty_state(
    actual_checkout,
):
    from cochem_torq import application

    implementation = actual_checkout / "src/cochem_torq/application.py"
    implementation.write_bytes(Path(application.__file__).read_bytes())
    with implementation.open("a") as output:
        output.write("\n# Explicit source-identity dirty-state fixture.\n")
    program = """import json,pathlib,sys
from cochem_torq.application import source_identity
import cochem_torq.application as app
assert pathlib.Path(app.__file__).resolve()==pathlib.Path(sys.argv[1]).resolve()
print(json.dumps(source_identity()))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(actual_checkout / "src")
    observed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(implementation)],
        cwd=actual_checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    identity = json.loads(observed.stdout)
    assert identity["git_commit"] == git(actual_checkout, "rev-parse", "HEAD")
    assert identity["working_tree_dirty"] is True
    assert identity["code_sha256"]


def test_unversioned_source_never_inherits_unrelated_parent_git(actual_checkout):
    parent = actual_checkout.parent
    git(parent, "init", "--quiet")
    shutil.rmtree(actual_checkout / ".git")
    assert _source_git_repository(actual_checkout / "src/cochem_torq") is None


def test_invalid_actual_source_git_metadata_fails_without_fabricated_provenance(
    actual_checkout,
):
    shutil.rmtree(actual_checkout / ".git")
    (actual_checkout / ".git").mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        _source_git_repository(actual_checkout / "src/cochem_torq")


def test_declared_workflow_commit_is_not_promoted_to_installed_git_authority(
    installed_artifact,
):
    environment = dict(os.environ)
    environment["COCHEM_SOURCE_COMMIT"] = "a" * 40
    program = """import json
from cochem_torq.application import source_identity
identity=source_identity()
assert identity['git_commit'] is None
assert identity['declared_workflow_commit']=='a'*40
assert identity['code_sha256']
print(json.dumps({'declared_only':True}))
"""
    process = subprocess.run(
        [str(installed_artifact), "-I", "-B", "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(process.stdout) == {"declared_only": True}
