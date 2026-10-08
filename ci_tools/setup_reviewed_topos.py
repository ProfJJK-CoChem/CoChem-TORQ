"""Provision a separate immutable TOPOS reviewed-ensemble consumer test profile.

This bounded profile installs the real TOPOS wheel without its unrelated engine
and UI dependencies. It qualifies only retained reviewed-ensemble consumption;
it does not qualify TOPOS calculations, its full SRS, or the legacy SDK profile.
Authentication uses the ordinary configured Git/GitHub CLI credential flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _execution_environment(*, source_auth: bool = False) -> dict[str, str]:
    """Restrict configured credentials to the explicit source retrieval phase."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    if not source_auth:
        for name in tuple(environment):
            if re.search(
                r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|API_KEY|"
                r"PRIVATE_KEY|ACCESS_KEY|SSH_AUTH)",
                name,
                flags=re.IGNORECASE,
            ):
                environment.pop(name)
    return environment


def _run(
    arguments: list[str], *, cwd: Path | None = None, source_auth: bool = False
) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=_execution_environment(source_auth=source_auth),
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_root(repository: Path) -> Path:
    return repository.parent / ".environment" / "cochem-torq" / "reviewed-topos"


def profile_paths(repository: Path, root: Path | None = None) -> tuple[Path, Path]:
    manifest = json.loads(
        (repository / "ci_tools" / "reviewed-topos-modules.json").read_text()
    )
    selected = root or Path(
        os.environ.get("COCHEM_REVIEWED_TOPOS_ROOT", str(default_root(repository)))
    )
    profile = selected.resolve() / manifest["revision"]
    return profile / "env" / "bin" / "python", profile / "receipt.json"


def verify_profile(repository: Path, root: Path | None = None) -> Path:
    """Fail closed unless the reviewed installed wheel and clean source agree."""
    manifest_path = repository / "ci_tools" / "reviewed-topos-modules.json"
    manifest = json.loads(manifest_path.read_text())
    python, receipt_path = profile_paths(repository, root)
    if not python.is_file() or not receipt_path.is_file():
        raise RuntimeError(
            "Required isolated reviewed TOPOS profile is absent; run "
            "python ci_tools/setup_reviewed_topos.py before the mandatory tests"
        )
    receipt = json.loads(receipt_path.read_text())
    source = receipt_path.parent / "source"
    expected_remote = f"https://github.com/{manifest['repository']}.git"
    if (
        receipt["manifest_sha256"] != _sha256(manifest_path)
        or receipt["core_requirements_sha256"]
        != _sha256(repository / manifest["core_requirements"])
        or receipt["version"] != manifest["version"]
        or receipt["revision"] != manifest["revision"]
        or receipt["source"] != str(source)
        or receipt["python"] != str(python)
        or _run(["git", "-C", str(source), "rev-parse", "HEAD"]) != manifest["revision"]
        or (source / ".git" / "HEAD").read_text().strip() != manifest["revision"]
        or _run(["git", "-C", str(source), "remote", "get-url", "origin"])
        != expected_remote
        or _run(["git", "-C", str(source), "status", "--porcelain"])
    ):
        raise RuntimeError("Reviewed TOPOS source/profile identity changed")
    if set(receipt["installed_modules"]) != set(manifest["required_modules"]):
        raise RuntimeError("Reviewed TOPOS installed module membership changed")
    for module, identity in receipt["installed_modules"].items():
        original = source.joinpath(*module.split(".")).with_suffix(".py")
        if identity["sha256"] != _sha256(original):
            raise RuntimeError(
                "Reviewed TOPOS installed/source module identity changed"
            )
    program = """
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

receipt = json.loads(sys.argv[1])
assert metadata.version('cochem-topos') == receipt['version']
assert metadata.version('CoChem-TORQ') == receipt['torq_version']
for name, expected in receipt['installed_modules'].items():
    path = Path(importlib.import_module(name).__file__)
    assert str(path) == expected['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
"""
    subprocess.run(
        [str(python), "-I", "-c", program, json.dumps(receipt)],
        env=_execution_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return python


def setup(repository: Path, root: Path) -> dict[str, object]:
    manifest_path = repository / "ci_tools" / "reviewed-topos-modules.json"
    manifest = json.loads(manifest_path.read_text())
    profile = root.resolve() / manifest["revision"]
    source = profile / "source"
    profile.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        _run(
            [
                "gh",
                "repo",
                "clone",
                manifest["repository"],
                str(source),
                "--",
                "--no-checkout",
                "--filter=blob:none",
            ],
            source_auth=True,
        )
        _run(
            ["git", "-C", str(source), "checkout", "--detach", manifest["revision"]],
            source_auth=True,
        )
    if (
        _run(["git", "-C", str(source), "rev-parse", "HEAD"]) != manifest["revision"]
        or _run(["git", "-C", str(source), "status", "--porcelain"])
        or (source / ".git" / "HEAD").read_text().strip() != manifest["revision"]
        or _run(["git", "-C", str(source), "remote", "get-url", "origin"])
        != f"https://github.com/{manifest['repository']}.git"
    ):
        raise RuntimeError(
            "Existing reviewed TOPOS source differs; refusing replacement"
        )
    python = profile / "env" / "bin" / "python"
    if not python.exists():
        _run([sys.executable, "-m", "venv", str(profile / "env")])
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(repository / manifest["core_requirements"]),
        ]
    )
    _run([str(python), "-m", "pip", "install", *manifest["test_requirements"]])
    # Install genuine wheels, without resolving unrelated sibling engine stacks.
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(source),
            str(repository),
        ]
    )
    program = """
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

modules = {}
for name in json.loads(sys.argv[1]):
    path = Path(importlib.import_module(name).__file__)
    modules[name] = {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
    }
print(json.dumps({
    'installed_modules': modules,
    'version': metadata.version('cochem-topos'),
    'torq_version': metadata.version('CoChem-TORQ'),
    'installed_distributions': {
        d.metadata['Name']: d.version for d in metadata.distributions()
    }
}, sort_keys=True))
"""
    installed = json.loads(
        _run(
            [str(python), "-I", "-c", program, json.dumps(manifest["required_modules"])]
        )
    )
    for module, identity in installed["installed_modules"].items():
        original = source.joinpath(*module.split(".")).with_suffix(".py")
        if identity["sha256"] != _sha256(original):
            raise RuntimeError("Installed TOPOS wheel differs from its reviewed source")
    receipt: dict[str, object] = {
        "schema_version": "cochem.reviewed-topos-installation/1",
        "scope": manifest["scope"],
        "manifest_sha256": _sha256(manifest_path),
        "core_requirements_sha256": _sha256(repository / manifest["core_requirements"]),
        "revision": manifest["revision"],
        "source": str(source),
        "python": str(python),
        **installed,
    }
    (profile / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verify_profile(repository, root)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    root = options.root or default_root(repository)
    if options.check:
        print(verify_profile(repository, root))
    else:
        receipt = setup(repository, root)
        print(json.dumps({"python": receipt["python"], "source": receipt["source"]}))


if __name__ == "__main__":
    main()
