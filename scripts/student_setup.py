"""Codespaces lifecycle for the TORQ interface and isolated BASE/TOPOS modules.

Calculations are submitted to the canonical Actions workflow. Opening the
interface never performs a quantum calculation or claims an engine qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ci_tools" / "ecosystem-modules.json"


def artifact_root(value: Path | None) -> Path:
    target = (
        (
            value
            or Path(
                os.environ.get("COCHEM_ARTIFACT_DIR", Path.home() / "CoChem_Artifacts")
            )
        )
        .expanduser()
        .resolve()
    )
    if target == ROOT or ROOT in target.parents:
        raise ValueError(
            "Student runtime artifacts must be outside the source checkout"
        )
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    return target


def build_environment() -> dict[str, str]:
    excluded = {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "SSH_ASKPASS",
        "GIT_ASKPASS",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key not in excluded
        and not key.startswith("GIT_")
        and not any(
            word in key.upper()
            for word in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
        )
    }


def setup(target: Path, *, install_interface: bool = True) -> dict:
    target = artifact_root(target)
    env = build_environment()
    if install_interface:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(ROOT / "ci_tools" / "requirements-interface-py312.txt"),
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)],
            env=env,
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "check"], env=env, check=True)
    manifest = json.loads(MANIFEST.read_text())
    spec = manifest["modules"]["base"]
    bootstrap = target / "bootstrap" / spec["revision"]
    if not bootstrap.exists():
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                f"https://github.com/{spec['repository']}.git",
                str(bootstrap),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(bootstrap), "checkout", "--detach", spec["revision"]],
            check=True,
        )
    observed = subprocess.check_output(
        ["git", "-C", str(bootstrap), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(bootstrap), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    remote = subprocess.check_output(
        ["git", "-C", str(bootstrap), "remote", "get-url", "origin"], text=True
    ).strip()
    detached = (bootstrap / ".git" / "HEAD").read_text().strip()
    if (
        observed != spec["revision"]
        or detached != observed
        or dirty
        or remote != f"https://github.com/{spec['repository']}.git"
    ):
        raise RuntimeError(
            "BASE installer checkout differs from the reviewed immutable source"
        )
    installer = bootstrap / "scripts" / "manage_modules.py"
    modules = target / "Modules"
    # A reviewed adapter-requirement change invalidates an old receipt even when
    # the upstream source commit stays pinned. Preserve that evidence explicitly.
    for module, reviewed in manifest["modules"].items():
        old_receipt = modules / module / "installation.json"
        if old_receipt.exists():
            recorded = json.loads(old_receipt.read_text())
            reviewed_hash = hashlib.sha256(
                json.dumps(reviewed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if recorded.get("manifest_spec_sha256") != reviewed_hash:
                raise RuntimeError(
                    f"The reviewed {module} installation specification changed. "
                    f"Preserve {target} and rerun setup with "
                    "--artifact-dir NEW_DIRECTORY; the existing verified "
                    "environment is not overwritten."
                )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(installer),
            "install",
            "--manifest",
            str(MANIFEST),
            "--root",
            str(modules),
            "--modules",
            "base",
            "topos",
            "--json",
        ],
        cwd=target,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    receipts = json.loads(completed.stdout)
    github_cli = shutil.which("gh")
    cli_observation = {"available": github_cli is not None, "path": github_cli}
    if github_cli is not None:
        cli_version = subprocess.run(
            [github_cli, "--version"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout.splitlines()[0]
        cli_observation["version_text"] = cli_version
    record = {
        "schema_version": "cochem.student-environment/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "installed",
        "interface_python": sys.executable,
        "module_root": str(modules),
        "module_manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "base_installer_revision": spec["revision"],
        "base_installer_sha256": hashlib.sha256(installer.read_bytes()).hexdigest(),
        "github_cli": cli_observation,
        "modules": receipts["modules"],
        "scientific_accuracy_established": False,
        "canonical_calculation_environment": "GitHub Actions",
        "canonical_interface_environment": "GitHub Codespaces",
    }
    state = target / "interface"
    state.mkdir(exist_ok=True, mode=0o700)
    (state / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
    print("TORQ interface and pinned BASE/TOPOS isolated installations verified.")
    return record


def runtime_environment(target: Path, token: str) -> dict[str, str]:
    env = build_environment()
    # Codespaces/explicit user credentials authorize the interface's requested
    # Actions dispatch. They never enter pip/build hooks or engine containers.
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        if variable in os.environ:
            env[variable] = os.environ[variable]
    env.update(
        {
            "COCHEM_ARTIFACT_DIR": str(target),
            "COCHEM_MODULE_ROOT": str(target / "Modules"),
            "COCHEM_MODULE_MANIFEST": str(MANIFEST),
            "JUPYTER_TOKEN": token,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def ready(port: int, token: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/contents/UI/Start_TORQ.ipynb?content=0",
        headers={"Authorization": "token " + token},
    )
    try:
        with opener.open(request, timeout=2) as response:
            body = json.loads(response.read(16 * 1024))
            return (
                response.status == 200
                and body.get("type") == "notebook"
                and body.get("path") == "UI/Start_TORQ.ipynb"
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def owned_process(record: dict):
    import psutil

    try:
        process = psutil.Process(record["pid"])
        if (
            process.create_time() == record["create_time"]
            and process.cmdline() == record["command"]
        ):
            return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass
    return None


def start(target: Path, port: int, timeout: float) -> dict:
    if not 1024 <= port <= 65535 or not 0 < timeout <= 60:
        raise ValueError(
            "Use a nonprivileged port and a startup timeout of at most 60 seconds"
        )
    target = artifact_root(target)
    state = target / "interface"
    state.mkdir(exist_ok=True, mode=0o700)
    server_file, token_file = state / "server.json", state / "token"
    if server_file.exists() and token_file.exists():
        previous = json.loads(server_file.read_text())
        process = owned_process(previous)
        if process is not None:
            if previous["port"] == port and ready(port, token_file.read_text().strip()):
                print(f"TORQ Jupyter interface is ready on port {port}.")
                return previous
            raise RuntimeError(
                "An owned interface exists but failed readiness; "
                "inspect its log or use stop"
            )
    token = secrets.token_urlsafe(32)
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(token)
    command = [
        str(Path(sys.executable).parent / "cochem-torq"),
        "interface",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--notebook",
        str(ROOT / "UI" / "Start_TORQ.ipynb"),
    ]
    with (state / "jupyter.log").open("ab") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=runtime_environment(target, token),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        import psutil

        process = psutil.Process(child.pid)
        deadline = time.monotonic() + timeout
        while child.poll() is None and time.monotonic() < deadline:
            if ready(port, token):
                record = {
                    "pid": child.pid,
                    "create_time": process.create_time(),
                    "command": process.cmdline(),
                    "port": port,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "health_endpoint": "/api/contents/UI/Start_TORQ.ipynb?content=0",
                    "authenticated_notebook_retrieved": True,
                }
                server_file.write_text(json.dumps(record) + "\n")
                (state / "lifecycle.json").write_text(
                    json.dumps({**record, "status": "ready"}) + "\n"
                )
                print(
                    f"TORQ Jupyter interface is ready on private port {port}; "
                    f"token is stored in {token_file}."
                )
                return record
            time.sleep(0.25)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)
        raise
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    raise RuntimeError(f"Jupyter startup failed; inspect {state / 'jupyter.log'}")


def stop(target: Path) -> None:
    server = target / "interface" / "server.json"
    if not server.exists():
        return
    process = owned_process(json.loads(server.read_text()))
    if process is not None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except Exception:
            if owned_process(json.loads(server.read_text())) is not None:
                os.killpg(process.pid, signal.SIGKILL)
    (server.parent / "lifecycle.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "previous_server": json.loads(server.read_text()),
                "owned_process_found": process is not None,
            }
        )
        + "\n"
    )
    server.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "start", "check", "stop"])
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--no-interface-install",
        action="store_true",
        help="Use an already installed interface during validation",
    )
    args = parser.parse_args(argv)
    target = artifact_root(args.artifact_dir)
    if args.action == "setup":
        setup(target, install_interface=not args.no_interface_install)
    elif args.action == "start":
        start(target, args.port, args.timeout)
    elif args.action == "stop":
        stop(target)
    else:
        token = (target / "interface" / "token").read_text().strip()
        if not ready(args.port, token):
            raise RuntimeError("TORQ Jupyter interface is not ready")
        print("TORQ authenticated Jupyter notebook health passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
