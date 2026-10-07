"""Tripartite Air-Gap Webhook Receiver & Boundary Enforcement Daemon.

Module: cochem.mobile.airgap_receiver
Implements REQ-MOB-092 and Tripartite Air-Gap boundary enforcement adhering strictly to the Zero-Mock mandate.
- Tier 1: Authenticated HTTP POST payload validation via HMAC-SHA256 (X-CoChem-Signature).
- Tier 2: Rejects write attempts to source directory ($COCH_SRC), isolates writes into designated scratch directory.
- Tier 3: Writes transaction log to SQLite WAL database (airgap_transactions.db) with PRAGMA journal_mode=WAL.
- Dynamic port allocation on 127.0.0.1:0 with stdout handshake: PORT=<port> PID=<pid>.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import socketserver
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cochem.mobile.payload_serializer import (
    get_hmac_secret,
    strict_json_loads,
    validate_job_id,
    validate_xyz_structure_dynamic,
)

logger = logging.getLogger(__name__)


def get_resolved_coch_src() -> Path:
    """Resolve and return canonical $COCH_SRC directory."""
    raw = os.environ.get("COCH_SRC")
    if raw and raw.strip():
        return Path(raw.strip()).resolve()
    cwd = Path.cwd()
    candidate = cwd / "src"
    if candidate.exists():
        return candidate.resolve()
    return cwd.resolve()


def is_path_in_source_dir(
    target_path: Union[Path, str], src_dir: Optional[Path] = None
) -> bool:
    """Check whether a target path resolves inside the protected source directory."""
    source_root = (src_dir or get_resolved_coch_src()).resolve()
    try:
        resolved = Path(target_path).resolve()
        return resolved == source_root or source_root in resolved.parents
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Cannot resolve the destination path safely.") from exc


def verify_hmac_signature(body_bytes: bytes, signature: str, secret_key: str) -> bool:
    """Verify HMAC-SHA256 signature using constant-time comparison."""
    secret = get_hmac_secret(secret_key)
    if not isinstance(signature, str) or not signature.strip():
        return False
    expected = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    # Support both raw hex and sha256=prefix
    clean_sig = signature.strip()
    if clean_sig.startswith("sha256="):
        clean_sig = clean_sig[7:]
    if re.fullmatch(r"[0-9a-fA-F]{64}", clean_sig) is None:
        return False
    return hmac.compare_digest(expected.lower(), clean_sig.lower())


class AirGapReceiverHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler enforcing Tripartite Air-Gap security model."""

    server_scratch_dir: Path
    server_src_dir: Path
    server_db_path: Path
    server_secret_key: str

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging for clean test stdout/stderr capture."""
        logger.debug(
            "%s - - [%s] %s",
            self.address_string(),
            self.log_date_time_string(),
            format % args,
        )

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        """Send JSON HTTP response with UTF-8 encoding and content-length."""
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handle health-check requests."""
        if self.path in ("/health", "/status", "/"):
            self._send_json_response(
                200,
                {
                    "status": "HEALTHY",
                    "pid": os.getpid(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            self._send_json_response(404, {"status": "ERROR", "error": "Not Found"})

    def do_POST(self) -> None:
        """Process incoming webhook payload under Tripartite Air-Gap verification."""
        content_length_header = self.headers.get("Content-Length")
        if not content_length_header:
            self._send_json_response(
                411,
                {
                    "status": "ERROR",
                    "error": "Length Required",
                    "tier": "TIER_1_REJECTED",
                },
            )
            return

        try:
            content_length = int(content_length_header)
        except ValueError:
            self._send_json_response(
                400,
                {
                    "status": "ERROR",
                    "error": "Invalid Content-Length header",
                    "tier": "TIER_1_REJECTED",
                },
            )
            return

        if content_length < 0:
            self._send_json_response(
                400,
                {"status": "ERROR", "error": "Invalid Content-Length header", "tier": "TIER_1_REJECTED"},
            )
            return
        body_bytes = self.rfile.read(content_length)
        if len(body_bytes) != content_length:
            self._send_json_response(
                400,
                {"status": "ERROR", "error": "Incomplete request body", "tier": "TIER_1_REJECTED"},
            )
            return

        # -------------------------------------------------------------
        # Air-Gap Tier 1: Authenticated POST & HMAC-SHA256 Verification
        # -------------------------------------------------------------
        sig_header = (
            self.headers.get("X-CoChem-Signature")
            or self.headers.get("X-Hub-Signature-256")
            or self.headers.get("X-Signature")
            or ""
        )

        try:
            configured_key = getattr(self, "server_secret_key", None)
            if configured_key is None:
                raise ValueError("Receiver authentication is not configured.")
            secret = get_hmac_secret(configured_key)
        except ValueError:
            self._send_json_response(
                503,
                {
                    "status": "ERROR",
                    "error": "Receiver authentication is not configured.",
                    "tier": "TIER_1_REJECTED",
                },
            )
            return
        if not verify_hmac_signature(body_bytes, sig_header, secret):
            self._send_json_response(
                401,
                {
                    "status": "ERROR",
                    "error": "Air-gap Tier 1 violation: Invalid or missing X-CoChem-Signature HMAC-SHA256 header",
                    "tier": "TIER_1_REJECTED",
                },
            )
            return

        try:
            payload_dict = strict_json_loads(body_bytes.decode("utf-8"))
            if not isinstance(payload_dict, dict):
                raise ValueError("JSON payload must be a root object.")
            job_id = validate_job_id(payload_dict.get("job_id", uuid.uuid4().hex))
            if "molecule_xyz" in payload_dict:
                validate_xyz_structure_dynamic(payload_dict["molecule_xyz"])
        except Exception as exc:
            self._send_json_response(
                422,
                {
                    "status": "ERROR",
                    "error": f"Air-gap Tier 1 violation: Invalid JSON schema: {exc}",
                    "tier": "TIER_1_REJECTED",
                },
            )
            return

        # -------------------------------------------------------------
        # Air-Gap Tier 2: Destination Isolation (Reject writes to $COCH_SRC)
        # -------------------------------------------------------------
        src_dir = self.server_src_dir
        scratch_dir = self.server_scratch_dir
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Inspect payload for any explicit attempt to direct writes to $COCH_SRC
        suspicious_keys = (
            "target_file",
            "target_path",
            "destination",
            "output_dir",
            "write_path",
            "file_path",
            "output_artifact_dir",
        )
        for key in suspicious_keys:
            if key in payload_dict and isinstance(payload_dict[key], str):
                dest_candidate = payload_dict[key].strip()
                if dest_candidate:
                    try:
                        protected = is_path_in_source_dir(dest_candidate, src_dir)
                    except ValueError:
                        protected = True
                    if protected:
                        self._send_json_response(
                            403,
                            {
                                "status": "ERROR",
                                "error": f"Air-gap Tier 2 violation: Write attempt to protected source directory ($COCH_SRC: {src_dir}) rejected.",
                                "tier": "TIER_2_REJECTED",
                                "rejected_path": dest_candidate,
                            },
                        )
                        return

        # Files use fresh receiver-generated names. Persist the exact authenticated
        # bytes so the ledger signature continues to verify the stored artifact.
        scratch_file = scratch_dir / f"payload_{uuid.uuid4().hex}.json"
        with scratch_file.open("xb") as output:
            output.write(body_bytes)

        # -------------------------------------------------------------
        # Air-Gap Tier 3: SQLite WAL Ledger Transaction
        # -------------------------------------------------------------
        db_path = self.server_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        client_ip = self.client_address[0] if self.client_address else "127.0.0.1"

        try:
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS airgap_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        signature TEXT NOT NULL,
                        client_ip TEXT NOT NULL,
                        payload_size INTEGER NOT NULL,
                        job_id TEXT,
                        scratch_file TEXT NOT NULL,
                        status TEXT NOT NULL,
                        tier_passed TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    INSERT INTO airgap_transactions (timestamp, signature, client_ip, payload_size, job_id, scratch_file, status, tier_passed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        timestamp_utc,
                        sig_header,
                        client_ip,
                        len(body_bytes),
                        job_id,
                        str(scratch_file),
                        "COMMITTED",
                        "TIER_3_WAL",
                    ),
                )
            conn.close()
        except Exception as db_exc:
            logger.error("Failed to commit Tier 3 SQLite WAL transaction: %s", db_exc)
            self._send_json_response(
                500,
                {
                    "status": "ERROR",
                    "error": f"Air-gap Tier 3 failure: SQLite WAL write error: {db_exc}",
                    "tier": "TIER_3_FAILED",
                },
            )
            return

        self._send_json_response(
            200,
            {
                "status": "SUCCESS",
                "message": "Payload ingested under Tripartite Air-Gap",
                "job_id": job_id,
                "scratch_file": str(scratch_file),
                "tier": "TIER_3_COMMITTED",
                "timestamp": timestamp_utc,
            },
        )


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for non-blocking concurrency."""

    daemon_threads = True
    allow_reuse_address = True


def make_airgap_receiver_server(
    host: str = "127.0.0.1",
    port: int = 0,
    scratch_dir: Optional[Union[Path, str]] = None,
    src_dir: Optional[Union[Path, str]] = None,
    db_path: Optional[Union[Path, str]] = None,
    secret_key: Optional[str] = None,
) -> Tuple[ThreadedHTTPServer, int]:
    """Factory creating configured AirGapReceiver server bound to ephemeral/static port."""
    # Reject missing/empty keys before creating directories or binding a socket.
    resolved_secret = get_hmac_secret(secret_key)
    resolved_scratch = Path(scratch_dir or (Path.cwd() / "scratch")).resolve()
    resolved_src = Path(src_dir or get_resolved_coch_src()).resolve()
    resolved_db = Path(db_path or (Path.cwd() / "airgap_transactions.db")).resolve()
    if is_path_in_source_dir(resolved_scratch, resolved_src) or is_path_in_source_dir(
        resolved_db, resolved_src
    ):
        raise ValueError("Receiver scratch and ledger paths must be outside the protected source directory.")
    resolved_scratch.mkdir(parents=True, exist_ok=True)
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    # Each actual server gets its own handler configuration. Mutating global
    # handler attributes would silently replace an existing receiver's key and
    # storage roots when a second receiver starts in the same process.
    class ConfiguredReceiverHandler(AirGapReceiverHTTPRequestHandler):
        server_scratch_dir = resolved_scratch
        server_src_dir = resolved_src
        server_db_path = resolved_db
        server_secret_key = resolved_secret

    server = ThreadedHTTPServer((host, port), ConfiguredReceiverHandler)
    assigned_port = server.server_address[1]
    return server, assigned_port


def main() -> None:
    """CLI entrypoint for running standalone Air-Gap Webhook Receiver process."""
    parser = argparse.ArgumentParser(
        description="CoChem Mobile Tripartite Air-Gap Webhook Receiver Daemon"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Binding host interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Binding port (0 for dynamic/ephemeral)"
    )
    parser.add_argument(
        "--scratch-dir", default=None, help="Designated isolated scratch directory"
    )
    parser.add_argument(
        "--src-dir", default=None, help="Protected source directory ($COCH_SRC)"
    )
    parser.add_argument(
        "--db-path", default=None, help="Path to SQLite WAL transactions database"
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="Explicit authentication key; otherwise require COCHEM_HMAC_SECRET",
    )

    args = parser.parse_args()

    server, port = make_airgap_receiver_server(
        host=args.host,
        port=args.port,
        scratch_dir=args.scratch_dir,
        src_dir=args.src_dir,
        db_path=args.db_path,
        secret_key=args.secret,
    )

    # Critical handshake contract: write PORT=<port> PID=<pid> to stdout and flush immediately
    handshake_msg = f"PORT={port} PID={os.getpid()}\n"
    sys.stdout.write(handshake_msg)
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt as _e:
        logger.debug(f"Ignored exception: {_e}")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
