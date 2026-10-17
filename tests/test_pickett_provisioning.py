"""Actual process checks for external native-build publication boundaries."""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple
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


class Reply(NamedTuple):
    status: int
    body: bytes = b""
    headers: tuple[tuple[str, str], ...] = ()
    truncate: bool = False
    drip_seconds: float = 0.0


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    """An actual local HTTP server; each path serves its queued replies in order."""
    routes: dict[str, list[Reply]] = {}
    requests: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            replies = routes.get(self.path, [Reply(404)])
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
            self.send_response(reply.status)
            for key, value in reply.headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            body = reply.body[: len(reply.body) // 2] if reply.truncate else reply.body
            step = 1 if reply.drip_seconds else max(len(body), 1)
            try:
                for start in range(0, len(body), step):
                    self.wfile.write(body[start : start + step])
                    self.wfile.flush()
                    time.sleep(reply.drip_seconds)
            except OSError:
                pass  # The client abandoned a deliberately slow transfer.

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    log = tmp_path / "provision.log"
    log.write_bytes(b"")
    try:
        yield routes, requests, log, f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def fetch(log: Path, base: str, **options: float) -> bytes:
    return helper._fetch(
        "calcat.c",
        sha256(PINNED).hexdigest(),
        len(PINNED),
        log,
        upstream=base,
        backoff_seconds=0.0,
        **options,
    )


def log_records(log: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_pinned_bytes_are_retrieved_once_and_logged(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(200, PINNED)]
    assert fetch(log, base) == PINNED
    assert requests == ["/calcat.c"]
    assert log_records(log)[0]["sha256"] == sha256(PINNED).hexdigest()


def test_changed_upstream_bytes_are_rejected_without_retry(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(200, b"Pinned upstream bytes!\n")]
    with pytest.raises(ValueError, match="differs from the pinned upstream bytes"):
        fetch(log, base)
    assert requests == ["/calcat.c"]


def test_announced_size_change_is_rejected_without_retry(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(200, PINNED + b"x")]
    with pytest.raises(ValueError, match="differs from the pinned upstream bytes"):
        fetch(log, base)
    assert requests == ["/calcat.c"]


def test_transfer_cut_short_is_retried_not_reported_as_changed(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(200, PINNED, truncate=True), Reply(200, PINNED)]
    assert fetch(log, base) == PINNED
    assert requests == ["/calcat.c"] * 2
    assert log_records(log)[0]["error"] == "IncompleteRead"


def test_slow_transfer_is_bounded_by_the_attempt_deadline(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(200, PINNED, drip_seconds=0.2)]
    with pytest.raises(RuntimeError, match="could not be retrieved"):
        fetch(log, base, deadline_seconds=0.5)
    assert requests == ["/calcat.c"] * helper._FETCH_ATTEMPTS
    assert {record["error"] for record in log_records(log)} == {"TimeoutError"}


def test_missing_upstream_file_is_not_retried_or_substituted(upstream):
    routes, requests, log, base = upstream
    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch(log, base)
    assert requests == ["/calcat.c"]


def test_redirect_is_refused_without_contacting_its_target(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(302, headers=(("Location", "/mirror/calcat.c"),))]
    routes["/mirror/calcat.c"] = [Reply(200, PINNED)]
    with pytest.raises(RuntimeError, match="redirected"):
        fetch(log, base)
    assert requests == ["/calcat.c"]
    assert log_records(log) == [
        {
            "url": base + "calcat.c",
            "attempt": 1,
            "http": 302,
            "location": "/mirror/calcat.c",
        }
    ]


def test_server_errors_are_retried_then_reported_as_unavailable(upstream):
    routes, requests, log, base = upstream
    routes["/calcat.c"] = [Reply(500, b"Internal Server Error")]
    with pytest.raises(RuntimeError, match="could not be retrieved"):
        fetch(log, base)
    assert requests == ["/calcat.c"] * helper._FETCH_ATTEMPTS
    assert [record["http"] for record in log_records(log)] == [
        500
    ] * helper._FETCH_ATTEMPTS


def test_every_attempt_through_a_proxy_keeps_tls(monkeypatch, tmp_path):
    """Retries must tunnel to port 443 each time, never fall back to plain HTTP."""
    seen: list[str] = []

    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            seen.append(self.rfile.readline().decode("latin-1").strip())
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Proxy)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy = f"http://127.0.0.1:{server.server_address[1]}"
    for name in ("no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, proxy)
    log = tmp_path / "provision.log"
    log.write_bytes(b"")
    try:
        with pytest.raises(RuntimeError, match="could not be retrieved"):
            fetch(log, "https://pinned-upstream.invalid/calpgm/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    assert (
        seen
        == ["CONNECT pinned-upstream.invalid:443 HTTP/1.1"] * helper._FETCH_ATTEMPTS
    )
    assert all("502" in str(record["reason"]) for record in log_records(log))
