"""Bounded HTTPS retrieval of actual public reference bytes and provenance.

The provider's availability, reference identity and scientific review are separate
from a successful download. This module preserves inherited proxy and TLS trust.
It never supplies missing values or follows a failed request with substitute data.
"""

from __future__ import annotations

import hashlib
import os
import re
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .domain import canonical_json


def _public_https_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port not in {None, 443}
    ):
        raise ValueError("Use a public HTTPS locator without user credentials.")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if re.search(
            r"token|secret|credential|signature|apikey|password", normalized
        ) or normalized in {
            "sig",
            "auth",
            "authentication",
            "authorization",
            "bearer",
            "key",
            "accesskey",
            "awsaccesskeyid",
        }:
            raise ValueError("Reference locators cannot contain credential parameters.")
    return url


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        # Signed public provider redirects may be used by urllib, but never
        # persisted. Authentication values are never constructed or extracted.
        parsed = urlsplit(newurl)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Reference redirects must preserve HTTPS trust.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _location_without_query(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))


def _write_new(path: Path, value: Any) -> None:
    payload = canonical_json(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def fetch_published_source(
    *,
    url: str,
    citation: str,
    destination: str | Path,
    reuse_permission: str,
    expected_sha256: str | None = None,
    max_bytes: int = 16 * 1024 * 1024,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Retrieve a public source into a new private directory, retaining failures.

    ``expected_sha256`` binds a repeat retrieval to previously retained bytes;
    absence records a first observation, not authenticated publisher authorship.
    A caller must inspect ``status`` before extraction. No denied-host bypass,
    TLS override, automatic retries or alternate source fallback is attempted.
    """
    _public_https_url(url)
    if not citation.strip() or not reuse_permission.strip():
        raise ValueError("Record an actual citation and explicit reuse status.")
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError(
            "An expected SHA-256 requires 64 lowercase hexadecimal digits."
        )
    if type(max_bytes) is not int or not 1 <= max_bytes <= 256 * 1024 * 1024:
        raise ValueError("Use an explicit finite source-byte limit.")
    if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 120:
        raise ValueError("Use an explicit bounded HTTP timeout.")
    root = Path(destination).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Source destinations cannot traverse symbolic links.")
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    record: dict[str, Any] = {
        "schema_version": "cochem.torq.reference-retrieval/1",
        "requested_url": url,
        "citation": citation,
        "reuse_permission": reuse_permission,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_bytes": max_bytes,
        "expected_sha256": expected_sha256,
        "independent_reference_qualification": False,
        "tls_verification": True,
        "inherited_proxy_preserved": True,
    }
    pending: Path | None = None
    try:
        opener = build_opener(
            _HTTPSRedirects(), HTTPSHandler(context=ssl.create_default_context())
        )
        request = Request(
            url, headers={"User-Agent": "CoChem-TORQ-reference-retrieval/1"}
        )
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ValueError(
                    "A reference source requires an actual HTTP 200 response."
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError("Source exceeds the approved byte limit.")
            descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=root)
            pending = Path(temporary)
            count = 0
            sha = hashlib.sha256()
            with os.fdopen(descriptor, "wb") as stream:
                while block := response.read(min(1024 * 1024, max_bytes - count + 1)):
                    count += len(block)
                    if count > max_bytes:
                        raise ValueError("Source exceeds the approved byte limit.")
                    stream.write(block)
                    sha.update(block)
                stream.flush()
                os.fsync(stream.fileno())
            observed_sha = sha.hexdigest()
            record.update(
                bytes=count,
                observed_sha256=observed_sha,
                response_url_without_query=_location_without_query(response.url),
                media_type=response.headers.get_content_type(),
            )
            if not count:
                raise ValueError("An empty response is not a reference source.")
            if expected_sha256 is not None and observed_sha != expected_sha256:
                record["status"] = "checksum_mismatch"
                os.link(pending, root / "unverified-source.bin")
            else:
                record["status"] = "retrieved"
                record["source_path"] = "source.bin"
                os.link(pending, root / "source.bin")
    except HTTPError as error:
        record.update(status="http_error", http_status=error.code)
    except URLError as error:
        # Proxy CONNECT failures occur before an origin response. Keep that
        # distinction and avoid serializing query-bearing URLs from diagnostics.
        record.update(status="transport_error", error_type=type(error.reason).__name__)
        match = re.search(r"Tunnel connection failed: (\d{3})", str(error.reason))
        if match:
            record["proxy_connect_status"] = int(match.group(1))
    except (OSError, ValueError) as error:
        record.update(status="retrieval_error", error_type=type(error).__name__)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_new(root / "retrieval.json", record)
    return record


def execute_reference_fetch(arguments: Any) -> dict[str, Any]:
    """CLI adapter; errors remain actual failed retrieval outcomes."""
    return fetch_published_source(
        url=arguments.url,
        citation=arguments.citation,
        destination=arguments.output_dir,
        reuse_permission=arguments.reuse_permission,
        expected_sha256=arguments.expected_sha256,
        max_bytes=arguments.max_bytes,
        timeout_seconds=arguments.timeout_seconds,
    )
