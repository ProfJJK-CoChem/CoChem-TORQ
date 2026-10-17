"""Build pinned external SPCAT/SPFIT for local numerical validation only.

Upstream copyright does not establish redistribution permission. Keep the fresh
installation root outside the checkout AND every uploaded artifact directory.
This helper never bundles upstream source or binaries, never installs systemwide,
and never qualifies a scientific method from compilation alone.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any

from cochem_torq.domain import canonical_json, digest
from cochem_torq.spectroscopy.pickett_backend import (
    SUPPORTED_SOURCE_INVENTORY,
    SUPPORTED_SOURCE_SHA256,
    SUPPORTED_UPSTREAM,
    load_pickett_installation,
)

UPSTREAM = SUPPORTED_UPSTREAM
_TOOLS = ("gcc", "make", "ar", "ranlib")
_REPOSITORY = Path(__file__).absolute().parents[1]
_FETCH_ATTEMPTS = 4
_FETCH_TIMEOUT_SECONDS = 120
_FETCH_BACKOFF_SECONDS = 2.0


def _safe(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Provisioning paths must be absolute without traversal.")
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("Provisioning paths cannot traverse symlinks.")
    if any(character in str(path) for character in "\r\n\x00"):
        raise ValueError("Provisioning paths cannot contain control delimiters.")
    return path


def _new_external_root(root: Path) -> Path:
    root = _safe(root)
    checkouts = [_REPOSITORY]
    if workspace := os.environ.get("GITHUB_WORKSPACE"):
        checkouts.append(Path(workspace).absolute())
    if any(
        root == checkout or checkout in root.parents for checkout in checkouts
    ) or any((parent / ".git").exists() for parent in root.parents):
        raise ValueError("Pickett must be provisioned outside repository checkouts.")
    if root.exists():
        raise FileExistsError("Provisioning requires a fresh root; no overwrite.")
    parent = _safe(root.parent)
    if not parent.is_dir():
        raise NotADirectoryError("Provisioning root parent must already exist.")
    root.mkdir(mode=0o700)
    return root


def _sha(path: Path) -> str:
    return sha256(_safe(path).read_bytes()).hexdigest()


def _write_new(path: Path, content: bytes) -> None:
    descriptor = os.open(
        _safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _run(
    arguments: list[str], *, cwd: Path, log: Path, input_bytes: bytes | None = None
) -> bytes:
    environment = dict(os.environ)
    environment["MAKEFLAGS"] = ""
    environment["MFLAGS"] = ""
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            input=input_bytes,
            capture_output=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Provisioning command exceeded its real time limit."
        ) from error
    # Logs stay in the external private root; no environment values are recorded.
    with log.open("ab") as stream:
        stream.write(json.dumps(arguments).encode() + b"\n")
        stream.write(result.stdout)
        stream.write(result.stderr)
        stream.write(f"\nexit_code={result.returncode}\n".encode())
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        raise RuntimeError(f"Provisioning command failed; inspect external log {log}.")
    return result.stdout


def _log_fetch(log: Path, record: dict[str, Any]) -> None:
    with log.open("ab") as stream:
        stream.write(json.dumps(record, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fetch(name: str, expected_sha256: str, size: int, log: Path) -> bytes:
    """Retrieve one pinned upstream file; only transient failures are retried."""
    url = UPSTREAM + name
    request = urllib.request.Request(
        url, headers={"User-Agent": "cochem-torq-pickett-provisioning"}
    )
    for attempt in range(1, _FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=_FETCH_TIMEOUT_SECONDS
            ) as response:
                if response.geturl() != url:
                    raise ValueError(
                        f"Pinned upstream {name} redirected; no other source is used."
                    )
                content = response.read(size + 1)
        except urllib.error.HTTPError as error:
            _log_fetch(log, {"url": url, "attempt": attempt, "http": error.code})
            if error.code < 500:
                raise RuntimeError(
                    f"Pinned upstream {name} is unavailable (HTTP {error.code})."
                ) from error
        except (OSError, http.client.HTTPException) as error:
            _log_fetch(
                log, {"url": url, "attempt": attempt, "error": type(error).__name__}
            )
        else:
            actual = sha256(content).hexdigest()
            _log_fetch(
                log,
                {
                    "url": url,
                    "attempt": attempt,
                    "bytes": len(content),
                    "sha256": actual,
                },
            )
            if len(content) != size or actual != expected_sha256:
                raise ValueError(
                    f"Retrieved {name} differs from the pinned upstream bytes."
                )
            return content
        if attempt < _FETCH_ATTEMPTS:
            time.sleep(_FETCH_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise RuntimeError(
        f"Pinned upstream {name} could not be retrieved; inspect external log {log}."
    )


def _retrieve_sources(source: Path, log: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name, expected, size in SUPPORTED_SOURCE_INVENTORY:
        if Path(name).name != name or any(c in name for c in "\\\r\n\x00"):
            raise ValueError("Pinned upstream contains an unsupported source entry.")
        path = _safe(source / name)
        _write_new(path, _fetch(name, expected, size, log))
        inventory.append(
            {"path": name, "sha256": _sha(path), "size_bytes": path.stat().st_size}
        )
    inventory.sort(key=lambda record: record["path"])
    if digest(inventory) != SUPPORTED_SOURCE_SHA256:
        raise ValueError(
            "Actual source inventory is outside the pinned adapter contract."
        )
    return inventory


def _append_github_env(path: Path, receipt: dict[str, Any]) -> None:
    path = _safe(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        observed = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.getuid()
            or observed.st_nlink != 1
        ):
            raise ValueError("GitHub environment output must be an owned regular file.")
        stream.write(
            "COCHEM_TORQ_PICKETT_PROVISIONING_JSON="
            f"{receipt['provisioning_manifest_path']}\n"
            "COCHEM_TORQ_PICKETT_PROVISIONING_SHA256="
            f"{receipt['provisioning_manifest_sha256']}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def provision(root: Path) -> dict[str, Any]:
    """Retrieve, independently hash, compile, and verify a new native installation."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The current Pickett adapter requires native Linux ELF.")
    missing = [tool for tool in _TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Required actual build tools are absent: {', '.join(missing)}"
        )
    root = _new_external_root(root)
    previous_umask = os.umask(0o077)
    try:
        source = root / "source"
        source.mkdir(mode=0o700)
        log = root / "provision.log"
        _write_new(log, b"")
        inventory = _retrieve_sources(source, log)
        for name in ("spcat", "spfit", "splib.a"):
            if (source / name).exists():
                raise ValueError(
                    "Fresh source unexpectedly contains a prebuilt executable."
                )
        build_log = root / "build.log"
        _write_new(build_log, b"")
        build_command = ["make", "-j1", "spcat", "spfit"]
        _run(build_command, cwd=source, log=build_log)
        compiler = (
            _run(["gcc", "--version"], cwd=source, log=log).decode().splitlines()[0]
        )
        # A native ELF check and source revalidation happen in the real consumer.
        manifest: dict[str, Any] = {
            "schema_version": "cochem.torq.pickett-provisioning/2",
            "upstream_url": UPSTREAM,
            "source_directory": str(source),
            "source_inventory": inventory,
            "source_inventory_sha256": digest(inventory),
            "source_inventory_matches_pinned_bytes": True,
            "upstream_prebuilt_files_retrieved": False,
            "rebuilt_splib_sha256": _sha(source / "splib.a"),
            "build_command": build_command,
            "build_log_sha256": _sha(build_log),
            "compiler": compiler,
            "manual_sha256": _sha(source / "calpgm.pdf"),
            "copyright_source_path": "calcat.c",
            "copyright_source_sha256": _sha(source / "calcat.c"),
            "copyright_status": (
                "Copyright (C) 1989, California Institute of Technology; all rights "
                "reserved. NASA sponsorship acknowledgement is not a "
                "distribution license."
            ),
            "distribution_permission": "unverified_not_bundled",
            "binary_redistribution_authorized": False,
            "local_usage_record": (
                "Task-authorized local native numerical validation of actually "
                "retrieved public source; no assertion of redistribution license."
            ),
            "scientific_qualification": False,
            "spcat": {"path": str(source / "spcat"), "sha256": _sha(source / "spcat")},
            "spfit": {"path": str(source / "spfit"), "sha256": _sha(source / "spfit")},
        }
        manifest_path = root / "provisioning.json"
        _write_new(manifest_path, canonical_json(manifest) + b"\n")
        manifest_sha = _sha(manifest_path)
        load_pickett_installation(
            manifest_path, expected_manifest_sha256=manifest_sha
        ).verify()
        return {
            "schema_version": "cochem.torq.pickett-provisioning-receipt/2",
            "provisioning_manifest_path": str(manifest_path),
            "provisioning_manifest_sha256": manifest_sha,
            "upstream_url": UPSTREAM,
            "source_inventory_sha256": SUPPORTED_SOURCE_SHA256,
            "spcat_sha256": manifest["spcat"]["sha256"],
            "spfit_sha256": manifest["spfit"]["sha256"],
            "binary_redistribution_authorized": False,
            "scientific_qualification": False,
            "installation_upload_permitted": False,
        }
    finally:
        os.umask(previous_umask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--github-env",
        type=Path,
        help="Append the two validated pins to existing GITHUB_ENV.",
    )
    args = parser.parse_args()
    try:
        receipt = provision(args.root)
        if args.github_env is not None:
            _append_github_env(args.github_env, receipt)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Pickett provisioning failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
