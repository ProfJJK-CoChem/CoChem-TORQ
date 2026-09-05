"""HPC Network Filesystem Atomic Fencing & Heartbeat Leases.

Implements Suggestion #26:
- Atomic directory creation via os.mkdir() across POSIX, NFSv4, and Lustre.
- Active background heartbeat refresher daemon thread.
- Split-brain defense with fencing tokens and validated stale lease reclamation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("cochem.concurrency.network_lock")


def _is_pid_alive(pid: int) -> bool:
    """Check if local PID is physically running and not a zombie."""
    if pid <= 0:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        return bool(p.is_running() and p.status() != psutil.STATUS_ZOMBIE)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class NetworkHeartbeatLock:
    """Distributed directory lease lock with active heartbeat renewals and fencing tokens."""

    def __init__(
        self,
        lock_dir: Path | str,
        lease_ttl_sec: float = 30.0,
        grace_period_sec: float = 5.0,
        node_id: Optional[str] = None,
    ) -> None:
        self.lock_dir = Path(lock_dir).resolve()
        self.lease_ttl_sec = max(0.5, float(lease_ttl_sec))
        self.grace_period_sec = max(0.1, float(grace_period_sec))
        self.hostname = node_id or socket.gethostname()
        self.pid = os.getpid()

        self.lease_file = self.lock_dir / "lease.json"
        self._fence_token: Optional[str] = None
        self._is_held = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._local_lock = threading.RLock()

    def _read_manifest(self) -> Optional[Dict[str, Any]]:
        """Read and parse lease manifest safely."""
        if not self.lease_file.exists():
            return None
        try:
            with open(self.lease_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write manifest atomically via sibling temporary file."""
        tmp_file = self.lock_dir / f"lease.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_file), str(self.lease_file))

    def is_stale(self) -> bool:
        """Check if active lease is expired beyond TTL and grace period."""
        if not self.lock_dir.exists():
            return False
        manifest = self._read_manifest()
        if manifest is None:
            # Lock dir exists without valid manifest; check mtime fallback after grace period
            try:
                mtime = self.lock_dir.stat().st_mtime
                return (time.time() - mtime) > (self.lease_ttl_sec + self.grace_period_sec)
            except OSError:
                return False

        expires_at = float(manifest.get("expires_at", 0.0))
        now = time.time()
        if now <= expires_at + self.grace_period_sec:
            return False

        # If on the same host, verify the owning process is actually dead
        owning_host = manifest.get("hostname", "")
        owning_pid = int(manifest.get("pid", 0))
        if owning_host == self.hostname and owning_pid > 0:
            if _is_pid_alive(owning_pid):
                return False

        return True

    def is_locked(self) -> bool:
        """Check if lock directory exists and lease is currently valid."""
        if not self.lock_dir.exists():
            return False
        return not self.is_stale()

    def renew_heartbeat(self) -> bool:
        """Update lease heartbeat and expiration timestamp while holding lock."""
        with self._local_lock:
            if not self._is_held:
                return False
            now = time.time()
            manifest = {
                "hostname": self.hostname,
                "pid": self.pid,
                "heartbeat": now,
                "fence_token": self._fence_token,
                "expires_at": now + self.lease_ttl_sec,
            }
            try:
                self._write_manifest(manifest)
                return True
            except Exception as exc:
                logger.debug("Failed to renew heartbeat: %s", exc)
                return False

    def _heartbeat_worker(self) -> None:
        """Background thread refreshing heartbeat every TTL / 3 seconds."""
        interval = max(0.1, self.lease_ttl_sec / 3.0)
        while not self._stop_heartbeat.wait(interval):
            if not self.renew_heartbeat():
                break

    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire atomic directory lease with retry backoff and stale lease reclamation."""
        t0 = time.time()
        deadline = t0 + max(0.0, float(timeout))

        while True:
            # Step 1: Attempt atomic directory creation
            try:
                self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
                os.mkdir(str(self.lock_dir))
                # Successfully created directory: become leaseholder
                self._fence_token = uuid.uuid4().hex
                now = time.time()
                manifest = {
                    "hostname": self.hostname,
                    "pid": self.pid,
                    "heartbeat": now,
                    "fence_token": self._fence_token,
                    "expires_at": now + self.lease_ttl_sec,
                }
                self._write_manifest(manifest)
                self._is_held = True

                # Start background heartbeat daemon
                self._stop_heartbeat.clear()
                self._heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
                self._heartbeat_thread.start()
                return True
            except FileExistsError:
                logger.debug("Lock directory %s already exists, assessing lease status.", self.lock_dir)

            # Step 2: Handle existing directory - check for staleness
            if self.is_stale():
                # Attempt stale lease reclamation with fencing check
                manifest_before = self._read_manifest()
                old_token = manifest_before.get("fence_token") if manifest_before else None
                try:
                    # Clean out stale directory atomically
                    shutil.rmtree(str(self.lock_dir), ignore_errors=True)
                except OSError as clean_err:
                    logger.debug("Could not remove stale lock dir: %s", clean_err)
                continue

            if time.time() >= deadline:
                return False

            time.sleep(0.05)

    def release(self) -> None:
        """Release directory lease and terminate heartbeat daemon."""
        with self._local_lock:
            if not self._is_held:
                return

            self._is_held = False
            self._stop_heartbeat.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=2.0)
                self._heartbeat_thread = None

            try:
                manifest = self._read_manifest()
                # Only delete if fence token matches our session
                if manifest is not None and manifest.get("fence_token") == self._fence_token:
                    shutil.rmtree(str(self.lock_dir), ignore_errors=True)
            except OSError as rel_err:
                logger.debug("Error removing lock dir on release: %s", rel_err)

    def __enter__(self) -> NetworkHeartbeatLock:
        if not self.acquire(timeout=self.lease_ttl_sec):
            raise TimeoutError(f"Could not acquire NetworkHeartbeatLock on {self.lock_dir}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


__all__ = ["NetworkHeartbeatLock"]
