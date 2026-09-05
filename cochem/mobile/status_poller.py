"""Decoupled Status Polling Engine, Atomic File Locking, and GitHub API Poller.

Strictly adhering to SRS Chunk 08 (REQ-MOB-073) and the Zero-Mock Mandate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, Union

import filelock
import requests

from cochem.mobile.job_state import (
    ExecutionTier,
    JobStatus,
    JobStatusRecord,
    validate_status_transition,
)
from cochem.mobile.payload_serializer import (
    canonical_json_dumps,
    get_cochem_state_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT_SECONDS: float = 5.0
DEFAULT_MAX_BACKOFF_SECONDS: float = 60.0
DEFAULT_BASE_DELAY_SECONDS: float = 2.0
DEFAULT_BACKOFF_FACTOR: float = 1.5
DEFAULT_JITTER_MAX_SECONDS: float = 1.0


def compute_backoff_delay(
    attempt: int,
    base: float = DEFAULT_BASE_DELAY_SECONDS,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay: float = DEFAULT_MAX_BACKOFF_SECONDS,
    jitter_max: float = DEFAULT_JITTER_MAX_SECONDS,
) -> float:
    """Calculate bounded jittered exponential backoff delay.

    Formula: T_poll = min(max_delay, base * (factor ^ attempt) + random.uniform(0.0, jitter_max))
    """
    safe_attempt = max(0, attempt)
    exponential_part = base * (factor**safe_attempt)
    jitter = random.uniform(0.0, jitter_max)
    total_delay = exponential_part + jitter
    return float(min(max_delay, total_delay))


def write_status_atomic(
    record: JobStatusRecord,
    state_dir: Optional[Path] = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """Atomically write JobStatusRecord to $COCHEM_STATE_DIR/{job_id}.status.json using FileLock and os.replace."""
    resolved_state_dir = (state_dir or get_cochem_state_dir()).resolve()
    resolved_state_dir.mkdir(parents=True, exist_ok=True)

    status_path = resolved_state_dir / f"{record.job_id}.status.json"
    lock_path = resolved_state_dir / f"{record.job_id}.lock"
    tmp_path = resolved_state_dir / f"{record.job_id}.status.json.tmp"

    lock = filelock.FileLock(str(lock_path), timeout=lock_timeout)
    with lock:
        canonical_str = canonical_json_dumps(record.model_dump())
        raw_bytes = canonical_str.encode("utf-8")

        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, status_path)

    return status_path


def read_status_atomic(
    job_id: str,
    state_dir: Optional[Path] = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Optional[JobStatusRecord]:
    """Atomically read JobStatusRecord from $COCHEM_STATE_DIR/{job_id}.status.json using FileLock."""
    resolved_state_dir = (state_dir or get_cochem_state_dir()).resolve()
    status_path = resolved_state_dir / f"{job_id}.status.json"
    lock_path = resolved_state_dir / f"{job_id}.lock"

    if not status_path.exists():
        return None

    lock = filelock.FileLock(str(lock_path), timeout=lock_timeout)
    with lock:
        if not status_path.exists():
            return None
        raw_bytes = status_path.read_bytes()

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
        return JobStatusRecord.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse status file %s: %s", status_path, exc)
        return None


def update_status_progress(
    job_id: str,
    status: JobStatus,
    progress_percent: float = 0.0,
    current_step: str = "",
    error_message: Optional[str] = None,
    output_hdf5_path: Optional[str] = None,
    tier: Optional[ExecutionTier] = None,
    state_dir: Optional[Path] = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> JobStatusRecord:
    """Atomically update and persist JobStatusRecord with state transition validation."""
    existing_record = read_status_atomic(job_id, state_dir=state_dir, lock_timeout=lock_timeout)

    resolved_tier = tier or (
        existing_record.tier if existing_record else ExecutionTier.LOCAL_TIER_1
    )

    if existing_record is not None:
        if not validate_status_transition(existing_record.status, status):
            logger.warning(
                "Invalid status transition for job %s: from %s to %s",
                job_id,
                existing_record.status,
                status,
            )

    updated_record = JobStatusRecord(
        job_id=job_id,
        status=status,
        tier=resolved_tier,
        progress_percent=min(100.0, max(0.0, progress_percent)),
        current_step=current_step,
        error_message=error_message,
        output_hdf5_path=output_hdf5_path,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    write_status_atomic(updated_record, state_dir=state_dir, lock_timeout=lock_timeout)
    return updated_record


def poll_github_workflow_run(
    owner: str,
    repo: str,
    run_id: int,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Query GitHub Actions Workflow Runs REST API endpoint."""
    resolved_token = (
        token or os.environ.get("GITHUB_TOKEN") or os.environ.get("COCHEM_DISPATCH_SECRET")
    )
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CoChem-Mobile-Poller/1.0",
    }
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token.strip()}"

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    result: Dict[str, Any] = resp.json()
    return result


def map_github_run_to_job_status(
    run_data: Dict[str, Any],
) -> Tuple[JobStatus, float, str, Optional[str]]:
    """Map GitHub Actions Workflow Run payload to CoChem JobStatus, progress %, step, and error msg."""
    gh_status = str(run_data.get("status", "")).lower()
    gh_conclusion = str(run_data.get("conclusion", "") or "").lower()

    if gh_status in {"queued", "waiting", "requested", "pending"}:
        return JobStatus.QUEUED, 5.0, "GitHub Action workflow queued in runner queue", None
    elif gh_status == "in_progress":
        return (
            JobStatus.RUNNING,
            50.0,
            "GitHub Action workflow executing in cloud environment",
            None,
        )
    elif gh_status == "completed":
        if gh_conclusion == "success":
            return JobStatus.COMPLETED, 100.0, "GitHub Action workflow completed successfully", None
        elif gh_conclusion == "timed_out":
            return (
                JobStatus.TIMED_OUT,
                99.0,
                "GitHub Action workflow timed out",
                "Run exceeded max time",
            )
        elif gh_conclusion == "cancelled":
            return JobStatus.CANCELLED, 0.0, "GitHub Action workflow cancelled", "Job was cancelled"
        else:
            return (
                JobStatus.FAILED,
                100.0,
                f"GitHub Action workflow failed (conclusion: {gh_conclusion})",
                f"Execution failed: {gh_conclusion}",
            )

    return JobStatus.RUNNING, 20.0, f"Workflow state: {gh_status}", None


async def poll_job_status_async(
    job_id: str,
    state_dir: Optional[Path] = None,
    on_update: Optional[Callable[[JobStatusRecord], Union[Coroutine[Any, Any, None], None]]] = None,
    max_attempts: int = 120,
    base_delay: float = 0.5,
    factor: float = 1.3,
    max_delay: float = 5.0,
    timeout_seconds: float = 300.0,
) -> JobStatusRecord:
    """Asynchronously poll status record until terminal state is achieved or timeout occurs."""
    start_time = time.monotonic()
    last_status: Optional[JobStatus] = None

    for attempt in range(max_attempts):
        if (time.monotonic() - start_time) > timeout_seconds:
            logger.warning("Polling timed out for job %s after %.1fs", job_id, timeout_seconds)
            record = update_status_progress(
                job_id=job_id,
                status=JobStatus.TIMED_OUT,
                current_step="Polling timeout reached",
                error_message="Operation timed out while waiting for job completion",
                state_dir=state_dir,
            )
            return record

        record = read_status_atomic(job_id, state_dir=state_dir)
        if record is not None:
            if on_update is not None and record.status != last_status:
                last_status = record.status
                res = on_update(record)
                if asyncio.iscoroutine(res):
                    await res

            if record.is_terminal():
                return record

        delay = compute_backoff_delay(
            attempt=attempt,
            base=base_delay,
            factor=factor,
            max_delay=max_delay,
            jitter_max=0.2,
        )
        await asyncio.sleep(delay)

    # Max attempts reached
    record = update_status_progress(
        job_id=job_id,
        status=JobStatus.TIMED_OUT,
        current_step="Polling maximum attempts exceeded",
        error_message="Maximum status polling attempts exceeded",
        state_dir=state_dir,
    )
    return record
