"""Actual process checks for external native-build publication boundaries."""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from ci_tools import provision_pickett_validation as helper

REPOSITORY = Path(__file__).absolute().parents[1]
HELPER = REPOSITORY / "ci_tools" / "provision_pickett_validation.py"
PINNED = b"Pinned upstream bytes.\n"


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


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    """An actual local HTTP server standing in for the upstream directory URL."""
    routes: dict[str, tuple[int, bytes, dict[str, str]]] = {}
    requests: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            status, body, headers = routes.get(self.path, (404, b"", {}))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(helper, "UPSTREAM", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setattr(helper, "_FETCH_BACKOFF_SECONDS", 0.0)
    log = tmp_path / "provision.log"
    log.write_bytes(b"")
    try:
        yield routes, requests, log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def fetch(log: Path) -> bytes:
    return helper._fetch("calcat.c", sha256(PINNED).hexdigest(), len(PINNED), log)


def test_pinned_bytes_are_retrieved_once_and_logged(upstream):
    routes, requests, log = upstream
    routes["/calcat.c"] = (200, PINNED, {})
    assert fetch(log) == PINNED
    assert requests == ["/calcat.c"]
    record = json.loads(log.read_text())
    assert record["sha256"] == sha256(PINNED).hexdigest()


def test_changed_upstream_bytes_are_rejected_without_retry(upstream):
    routes, requests, log = upstream
    routes["/calcat.c"] = (200, b"Pinned upstream bytes!\n", {})
    with pytest.raises(ValueError, match="differs from the pinned upstream bytes"):
        fetch(log)
    assert requests == ["/calcat.c"]


def test_missing_upstream_file_is_not_retried_or_substituted(upstream):
    routes, requests, log = upstream
    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch(log)
    assert requests == ["/calcat.c"]


def test_redirect_to_another_location_is_refused(upstream):
    routes, requests, log = upstream
    routes["/calcat.c"] = (302, b"", {"Location": "/mirror/calcat.c"})
    routes["/mirror/calcat.c"] = (200, PINNED, {})
    with pytest.raises(ValueError, match="redirected"):
        fetch(log)


def test_server_errors_are_retried_then_reported_as_unavailable(upstream):
    routes, requests, log = upstream
    routes["/calcat.c"] = (500, b"Internal Server Error", {})
    with pytest.raises(RuntimeError, match="could not be retrieved"):
        fetch(log)
    assert requests == ["/calcat.c"] * helper._FETCH_ATTEMPTS
    assert [json.loads(line)["http"] for line in log.read_text().splitlines()] == [
        500
    ] * helper._FETCH_ATTEMPTS
