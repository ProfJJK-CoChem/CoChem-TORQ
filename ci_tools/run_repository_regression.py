"""Run every repository test and retain bounded, source-bound public evidence.

Raw pytest output stays in ``private/`` because failure traces can contain test
credentials. The public JUnit preserves actual names, outcomes and durations,
but omits trace/output bodies. Native evidence is a bounded subset of complete
PySCF inventories; it does not establish independent scientific qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

try:
    from ci_tools.validate_release_evidence import source_tree_sha256
except ModuleNotFoundError:
    from validate_release_evidence import source_tree_sha256

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_BUNDLES = 64
CONTROL_PATH = re.compile(
    r"(?:corrupt|damaged|changed|mutat|forg|tamper|invalid|fake|synthetic)",
    re.IGNORECASE,
)
PRIVATE_NAME = re.compile(
    r"(?:secret|token|private|authority|lease|idempotency|hmac|password|"
    r"ledger|sqlite|server|approval|receipt|\.env|\.pid|(?:^|[-_.])key(?:$|[-_.]))",
    re.IGNORECASE,
)
PRIVATE_FIELDS = {
    "access_token",
    "api_key",
    "authorization",
    "authority_token",
    "idempotency_key",
    "lease_token",
    "password",
    "private_key",
    "secret",
    "token",
    "github_token",
    "jupyter_token",
    "refresh_token",
}
PRIVATE_CONTENT = re.compile(
    rb"(?:authorization\s*:\s*(?:bearer|basic)\s+|"
    rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|"
    rb"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}|"
    rb"https?://[^\s/@]+:[^\s/@]+@)",
    re.IGNORECASE,
)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path: Path) -> dict:
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError("Unbounded or linked JSON")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON")

    result = json.loads(
        path.read_text(), object_pairs_hook=unique, parse_constant=invalid_constant
    )
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def _has_private_field(value) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in PRIVATE_FIELDS or _has_private_field(item)
            for key, item in value.items()
        )
    if isinstance(value, str):
        return PRIVATE_CONTENT.search(value.encode()) is not None
    return isinstance(value, list) and any(_has_private_field(item) for item in value)


def _has_private_bytes(path: Path) -> bool:
    with path.open("rb") as stream:
        previous = b""
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if PRIVATE_CONTENT.search(previous + chunk):
                return True
            previous = chunk[-1024:]
    return False


def _inventory(manifest: Path, adapter_digest: str) -> list[Path]:
    data = _json(manifest)
    if (
        data.get("schema_version") != "cochem-torq.engine-artifacts.v1"
        or set(data) != {"schema_version", "artifacts"}
        or _has_private_field(data)
    ):
        raise ValueError("Unsupported inventory")
    root = manifest.parent
    result = _json(root / "result.json")
    if (
        result.get("schema_version") != "cochem-torq.pyscf-result.v1"
        or result.get("status") != "complete"
        or result.get("engine") != "PySCF"
        or result.get("adapter_source_sha256") != adapter_digest
    ):
        raise ValueError("Incomplete or differently sourced native result")
    entries = data.get("artifacts")
    if not isinstance(entries, list) or not 0 < len(entries) <= 4096:
        raise ValueError("Unbounded inventory")
    names = set()
    files = []
    size = manifest.stat().st_size
    for entry in entries:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or set(entry) != {"path", "size_bytes", "sha256"}
        ):
            raise ValueError("Invalid inventory path")
        parts = PurePosixPath(relative)
        if (
            parts.is_absolute()
            or str(parts) != relative
            or any(
                part in {".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9_.+-]+", part)
                or PRIVATE_NAME.search(part)
                for part in parts.parts
            )
            or parts.suffix not in {".json", ".log", ".chk", ".npy", ".npz", ".xyz"}
            or relative in names
        ):
            raise ValueError("Private or unsafe inventory path")
        names.add(relative)
        path = root.joinpath(*parts.parts)
        if any(
            parent.is_symlink() for parent in [path, *path.parents] if parent != root
        ):
            raise ValueError("Linked inventory path")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != entry.get(
            "size_bytes"
        ):
            raise ValueError("Native file inventory mismatch")
        size += metadata.st_size
        if size > MAX_BUNDLE_BYTES or _digest(path) != entry.get("sha256"):
            raise ValueError("Native inventory hash or size mismatch")
        if path.suffix == ".json" and _has_private_field(_json(path)):
            raise ValueError("Private JSON content")
        if _has_private_bytes(path):
            raise ValueError("Private native content")
        files.append(path)
    if not {"result.json", "request.json", "pyscf.log", "wavefunction.chk"} <= names:
        raise ValueError("Native calculation inventory incomplete")
    return [manifest, *files]


def curate_engine_evidence(source: Path, destination: Path, repository: Path) -> dict:
    """Copy only verified declared native bytes, never the entire test directory."""
    destination.mkdir(parents=True, exist_ok=False)
    adapter_digest = _digest(repository / "src/cochem_torq/engines/pyscf_backend.py")
    report = {
        "schema_version": "cochem.torq.curated-native-evidence/1",
        "scope": "complete current-adapter PySCF native inventories only",
        "independent_scientific_qualification": False,
        "bundles": [],
        "rejected_or_unsupported": 0,
        "excluded_control_directories": 0,
        "bytes_copied": 0,
        "truncated_by_bound": False,
    }
    if not source.is_dir() or source.is_symlink():
        return report
    examined = 0
    for directory, directories, names in os.walk(source, followlinks=False):
        report["excluded_control_directories"] += sum(
            bool(CONTROL_PATH.search(name)) for name in directories
        )
        directories[:] = sorted(
            name
            for name in directories
            if not (Path(directory) / name).is_symlink()
            and not CONTROL_PATH.search(name)
        )
        if "manifest.json" not in names:
            continue
        examined += 1
        if examined > 4096 or len(report["bundles"]) >= MAX_BUNDLES:
            report["truncated_by_bound"] = True
            break
        manifest = Path(directory) / "manifest.json"
        try:
            files = _inventory(manifest, adapter_digest)
            size = sum(path.stat().st_size for path in files)
            if report["bytes_copied"] + size > MAX_TOTAL_BYTES:
                report["truncated_by_bound"] = True
                continue
            identity = _digest(manifest)
            target = destination / identity
            if target.exists():
                continue
            target.mkdir()
            try:
                for path in files:
                    output = target / path.relative_to(manifest.parent)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, output)
                    if _digest(output) != _digest(path):
                        raise ValueError("Native file changed during evidence copy")
                _inventory(target / "manifest.json", adapter_digest)
            except (OSError, ValueError, TypeError, KeyError):
                shutil.rmtree(target)
                raise
            report["bundles"].append({"manifest_sha256": identity, "bytes": size})
            report["bytes_copied"] += size
        except (OSError, ValueError, TypeError, KeyError):
            report["rejected_or_unsupported"] += 1
    return report


def sanitize_junit(source: Path, destination: Path) -> dict:
    """Preserve genuine outcomes while excluding trace and captured-output bodies."""
    if source.is_symlink() or not 0 < source.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError("A bounded actual JUnit report is required")
    tree = ElementTree.parse(source)
    cases = list(tree.iter("testcase"))
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for tag, count in (
            ("failure", "failures"),
            ("error", "errors"),
            ("skipped", "skipped"),
        ):
            counts[count] += int(case.find(tag) is not None)
    for item in tree.iter():
        for child in list(item):
            if child.tag in {"system-out", "system-err", "properties"}:
                item.remove(child)
        if item.tag in {"failure", "error", "skipped"}:
            item.attrib.clear()
            item.text = "Body retained only in the private original report."
        elif item.tag in {"testcase", "testsuite", "testsuites"}:
            allowed = {
                "name",
                "classname",
                "time",
                "tests",
                "failures",
                "errors",
                "skipped",
            }
            item.attrib = {
                key: value for key, value in item.attrib.items() if key in allowed
            }
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {
        **counts,
        "raw_report_sha256": _digest(source),
        "public_report_sha256": _digest(destination),
        "public_report_policy": (
            "actual outcomes and durations; trace/output bodies excluded"
        ),
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(repository: Path, output: Path, basetemp: Path, timeout: int) -> int:
    repository = repository.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    public = output / "public"
    private.mkdir(mode=0o700)
    public.mkdir()
    before = source_tree_sha256(repository)
    versions = {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
    }
    _write(
        public / "environment.json",
        {"python": platform.python_version(), "versions": versions},
    )
    audit = subprocess.run(
        [sys.executable, str(repository / "ci_tools/validate_release_evidence.py")],
        cwd=repository,
        capture_output=True,
        timeout=60,
        check=False,
    )
    (public / "full-srs-audit.json").write_bytes(audit.stdout)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository))
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    environment.update({"JAX_ENABLE_X64": "true", "PYTHONDONTWRITEBYTECODE": "1"})
    for name in ("COCHEM_SOURCE_READ_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        environment.pop(name, None)
    environment.pop("PYTEST_ADDOPTS", None)
    raw_report = private / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--basetemp",
        str(basetemp),
        "--junitxml",
        str(raw_report),
        str(repository / "tests"),
    ]
    started = time.monotonic()
    timed_out = False
    with (private / "pytest.log").open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=repository,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status = 124
    after = source_tree_sha256(repository)
    summary = {
        "schema_version": "cochem.torq.repository-regression/1",
        "scope": "all repository tests without marker or name exclusions",
        "source_tree_sha256_before": before,
        "source_tree_sha256_after": after,
        "source_unchanged": before == after,
        "pytest_exit_code": status,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "audit_exit_code": audit.returncode,
        "full_srs_qualification": False,
        "selected_command": command,
        "junit": None,
    }
    try:
        summary["junit"] = sanitize_junit(raw_report, public / "junit.xml")
    except (OSError, ValueError, ElementTree.ParseError):
        summary["junit_report_error"] = "No bounded parseable actual JUnit report"
    summary["native_evidence"] = curate_engine_evidence(
        basetemp, public / "native-evidence", repository
    )
    passed = (
        status == 0
        and not timed_out
        and before == after
        and audit.returncode == 0
        and summary["junit"] is not None
        and summary["junit"]["tests"] > 0
        and all(
            summary["junit"][name] == 0 for name in ("failures", "errors", "skipped")
        )
    )
    summary["passed"] = passed
    _write(public / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    arguments = parser.parse_args()
    if not 1 <= arguments.timeout <= 3600:
        parser.error("Timeout must be between 1 and 3600 seconds")
    return run(
        arguments.repository, arguments.output, arguments.basetemp, arguments.timeout
    )


if __name__ == "__main__":
    raise SystemExit(main())
