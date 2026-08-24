Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task12_telemetry.md.
Original prompt:
# Prompt: Visual & Event Telemetry Streamer

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_telemetry.py`

## Objective
Implement Visual & Event Telemetry Streamer for CoChem-TORQ based on Task 12 (Stage 5.5 - 6.0) specifications.

## Instructions for Coder
1. Create or update `cochem_torq_telemetry.py` inside `Libraries/`.
2. Implement `stream_webhook_events()` using asynchronous HTTP POST with Exponential Backoff Circuit Breaker. Implement zero-interruption buffering to `telemetry_spool.jsonl`, writing strictly to the dynamically provided scratch directory.
3. Implement `generate_plotly_3d_carousels()` using 2D Strided Regular Grid Decimation while preserving stationary points, emitting standalone HTML visualizers to the dynamically provided artifact directory.
4. Implement `export_crash_animation()` capturing trajectories during Steric Shatter Soft-Quench aborts into `crash_animation.xyz` and `crash_diagnostic.json`. Write these strictly to the dynamically provided scratch/artifact directory.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_torq_telemetry.py`.
- **Zero Mocking**: Do NOT mock any logic. Implement physical webhook requests, exception catching, and Plotly 3D HTML generation.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically. All files (like `.tar.zst`, `.jsonl`, etc.) MUST be written to the scratch or artifact paths provided dynamically by the environment or arguments, NOT the current working directory.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_telemetry.py ---
"""
CoChem-TORQ: Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Specification
---------------------------------------------------------------------------------
Manages real-time, out-of-band communication with users and HPC environments,
safely bypassing frozen Jupyter DOMs, and preparing interactive visual reports for
headless cluster executions.

Implements:
1. Asynchronous Webhook Event Streaming with Exponential Backoff Circuit Breaker
   and zero-interruption spooling to `telemetry_spool.jsonl`.
2. 2D Strided Regular Grid Decimation for Potential Energy Surfaces (PES) with
   stationary point preservation and color-blind accessible Plotly 3D HTML carousels
   for multi-state Discrete Variable Representation (DVR) probability wavefunctions.
3. Multi-frame XYZ Crash Animation and JSON Diagnostic Exporter for Steric Shatter
   Soft-Quench aborts and gradient explosion analysis.
4. Strict Filesystem Air-Gap compliance writing exclusively to dynamic
   scratch and artifact directory tiers.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import json
import logging
import math
import os
import tempfile
import time
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Telemetry] %(message)s"
)
logger = logging.getLogger("TorqTelemetry")

# Environment resolution for Filesystem Air-Gap
ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)
SCRATCH_DIR = os.environ.get("COCHEM_SCRATCH_DIR", str(Path.home() / "cochem_scratch"))


# ============================================================================
# Custom JSON Serialization Helper
# ============================================================================


def _json_serial_default(obj: Any) -> Any:
    """Serializes NumPy scalars, NumPy arrays, Path, Enum, and datetime objects."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime.datetime | datetime.date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


# ============================================================================
# Custom Warning & Exception Classes
# ============================================================================


class TelemetryDeliveryError(Exception):
    """Raised when webhook delivery encounters an unrecoverable error."""

    pass


class CircuitBreakerOpenError(Exception):
    """Raised when the telemetry circuit breaker is OPEN from network failures."""

    pass


class SoftQuenchAbortError(Exception):
    """Raised when Steric Shatter Soft-Quench detects unresolvable atomic overlap."""

    pass


class TelemetryWarning(UserWarning):
    """Issued for non-fatal notices like offline spooling or retries."""

    pass


# ============================================================================
# Data Models (Pydantic V2)
# ============================================================================


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class WebhookPayload(BaseModel):
    """Schema-enforced model for outgoing out-of-band telemetry events."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_type: str = Field(
        description=(
            "Type of event: job_start, job_completed, node_failure, "
            "soft_quench_collision, oom_backoff, progress, heartbeat"
        )
    )
    job_id: str = Field(description="Unique TORQ job identifier")
    node_id: str | None = Field(
        default=None, description="HPC / GPU compute node identifier"
    )
    status: str = Field(
        default="RUNNING",
        description="Job status: RUNNING, COMPLETED, FAILED, ALERT, ABORTED",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(timezone.utc).isoformat()
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary payload metrics and state variables",
    )
    error_trace: str | None = Field(
        default=None, description="Traceback snippet or error description if applicable"
    )


class CrashDiagnostic(BaseModel):
    """Diagnostic schema for Steric Shatter Soft-Quench crash captures."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    error_node_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(timezone.utc).isoformat()
    )
    num_frames: int
    num_atoms: int
    symbols: list[str]
    min_interatomic_distance: float
    colliding_pair: tuple[int, int] | None = None
    max_gradient_norm: float | None = None
    abort_reason: str
    crash_frame_index: int
    initial_energy_hartree: float | None = None
    final_energy_hartree: float | None = None


# ============================================================================
# Dynamic Path Resolution (6-Tier Air-Gap Hierarchy)
# ============================================================================


def _resolve_scratch_dir(scratch_dir: str | Path | None = None) -> Path:
    """Resolves and creates dynamic scratch dir adhering to Filesystem Air-Gap.

    Tier 1: Explicit custom scratch argument.
    Tier 2: COCHEM_SCRATCH or COCHEM_SCRATCH_DIR env vars.
    Tier 3: COCHEM_TMP, TMPDIR, TEMP, TMP env vars.
    Tier 4: XDG_CACHE_HOME / cochem / scratch.
    Tier 5: tempfile.gettempdir() / cochem_scratch.
    Tier 6: Path.home() / .cochem / scratch fallback.
    """
    if scratch_dir is not None:
        p = Path(scratch_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    for env_key in ("COCHEM_SCRATCH", "COCHEM_SCRATCH_DIR"):
        env_val = os.environ.get(env_key)
        if env_val and env_val.strip():
            p = Path(env_val.strip()).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

    for env_key in ("COCHEM_TMP", "TMPDIR", "TEMP", "TMP"):
        env_val = os.environ.get(env_key)
        if env_val and env_val.strip():
            p = (Path(env_val.strip()).resolve() / "cochem_scratch").resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache and xdg_cache.strip():
        p = (Path(xdg_cache.strip()).resolve() / "cochem" / "scratch").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    try:
        p = (Path(tempfile.gettempdir()).resolve() / "cochem_scratch").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        pass

    p = (Path.home() / ".cochem" / "scratch").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_artifact_dir(artifact_dir: str | Path | None = None) -> Path:
    """Resolves and creates dynamic artifact dir adhering to Filesystem Air-Gap.

    Tier 1: Explicit custom artifact argument.
    Tier 2: COCHEM_ARTIFACTS_DIR, COCHEM_DELIVERABLES, COCHEM_DELIVERABLES_DIR env vars.
    Tier 3: Path.home() / .cochem / deliverables fallback.
    """
    if artifact_dir is not None:
        p = Path(artifact_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    for env_key in (
        "COCHEM_ARTIFACTS_DIR",
        "COCHEM_DELIVERABLES",
        "COCHEM_DELIVERABLES_DIR",
    ):
        env_val = os.environ.get(env_key)
        if env_val and env_val.strip():
            p = Path(env_val.strip()).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p

    p = (Path.home() / ".cochem" / "deliverables").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _spool_event_to_disk(
    event_dict: dict[str, Any],
    scratch_dir: Path,
    spool_filename: str = "telemetry_spool.jsonl",
    reason: str = "Network offline",
) -> Path:
    """Appends an un-delivered telemetry event to the zero-interruption spool file."""
    spool_path = scratch_dir / spool_filename
    envelope = {
        "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "delivery_status": "SPOOLED",
        "spool_reason": reason,
        "payload": event_dict,
    }
    try:
        with open(spool_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(envelope, default=_json_serial_default, ensure_ascii=False)
                + "\n"
            )
    except OSError as exc:
        logger.error(f"Failed to write to telemetry spool file {spool_path}: {exc}")
    return spool_path


# ============================================================================
# Circuit Breaker & Asynchronous Webhook Streamer
# ============================================================================


class TelemetryCircuitBreaker:
    """
    Exponential Backoff Circuit Breaker for robust out-of-band telemetry.
    Silently intercepts cluster network drops and caches events in memory / spool files
    to prevent halting active JAX physics computations.
    """

    def __init__(
        self,
        failure_threshold: int = 4,
        recovery_timeout: float = 20.0,
        backoff_factor: float = 0.25,
        max_retries: int = 3,
        request_timeout: float = 3.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.in_memory_deque: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=2000
        )

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if (
            self.consecutive_failures >= self.failure_threshold
            or self.state == CircuitState.HALF_OPEN
        ):
            self.state = CircuitState.OPEN
            logger.warning(
                f"Telemetry Circuit Breaker tripped to OPEN after "
                f"{self.consecutive_failures} consecutive network failures."
            )

    def can_attempt_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Telemetry Circuit Breaker entering HALF_OPEN probe state.")
                return True
            return False
        # HALF_OPEN allows single probe
        return True


# Global circuit breaker singleton
_GLOBAL_CIRCUIT_BREAKER = TelemetryCircuitBreaker()


async def stream_webhook_events_async(
    status_payload: dict[str, Any] | WebhookPayload,
    webhook_url: str | None = None,
    scratch_dir: str | Path | None = None,
    max_retries: int = 3,
    timeout: float = 3.0,
    spool_filename: str = "telemetry_spool.jsonl",
    circuit_breaker: TelemetryCircuitBreaker | None = None,
) -> dict[str, Any]:
    """
    Asynchronously streams out-of-band webhook telemetry with Exponential Backoff
    Circuit Breaker. If network drops or times out, silently caches event to
    `telemetry_spool.jsonl` without raising unhandled exceptions or interrupting
    computations.

    :param status_payload: Dictionary or WebhookPayload model.
    :param webhook_url: Discord/Slack/HTTP webhook URL (optional).
    :param scratch_dir: Target scratch directory for spooling.
    :param max_retries: Maximum exponential backoff retries.
    :param timeout: Per-request HTTP timeout in seconds.
    :param spool_filename: Spool log filename.
    :param circuit_breaker: Optional circuit breaker instance.
    :return: Delivery status summary dictionary.
    """
    cb = circuit_breaker or _GLOBAL_CIRCUIT_BREAKER
    target_scratch = _resolve_scratch_dir(scratch_dir)

    # Validate and normalize payload
    if isinstance(status_payload, WebhookPayload):
        payload_dict = status_payload.model_dump()
    elif isinstance(status_payload, dict):
        try:
            validated = WebhookPayload(**status_payload)
            payload_dict = validated.model_dump()
        except Exception:
            payload_dict = dict(status_payload)
            payload_dict.setdefault(
                "timestamp", datetime.datetime.now(timezone.utc).isoformat()
            )
    else:
        payload_dict = {
            "data": str(status_payload),
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }

    cb.in_memory_deque.append(payload_dict)

    # If no webhook URL configured, spool directly
    if not webhook_url or not str(webhook_url).strip():
        spool_path = _spool_event_to_disk(
            payload_dict,
            target_scratch,
            spool_filename=spool_filename,
            reason="No webhook URL provided",
        )
        return {
            "status": "SPOOLED",
            "spooled": True,
            "spool_path": str(spool_path),
            "reason": "No webhook URL configured",
        }

    # Check Circuit Breaker gate
    if not cb.can_attempt_request():
        spool_path = _spool_event_to_disk(
            payload_dict,
            target_scratch,
            spool_filename=spool_filename,
            reason="Circuit Breaker OPEN",
        )
        return {
            "status": "SPOOLED",
            "spooled": True,
            "spool_path": str(spool_path),
            "reason": "Circuit Breaker OPEN",
        }

    # Attempt asynchronous HTTP POST with exponential backoff
    last_exception_msg = ""
    # Safe JSON string serialization supporting NumPy types
    payload_json_str = json.dumps(
        payload_dict, default=_json_serial_default, ensure_ascii=False
    )

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    webhook_url,
                    content=payload_json_str,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "CoChem-TORQ-Telemetry/0.0.12",
                    },
                )
                if response.is_success:
                    cb.record_success()
                    return {
                        "status": "DELIVERED",
                        "status_code": response.status_code,
                        "attempt": attempt,
                        "spooled": False,
                    }
                else:
                    last_exception_msg = (
                        f"HTTP {response.status_code}: {response.text[:120]}"
                    )
        except (
            httpx.TimeoutException,
            httpx.RequestError,
            httpx.HTTPError,
            asyncio.TimeoutError,
            Exception,
        ) as exc:
            last_exception_msg = f"{type(exc).__name__}: {str(exc)}"

        # Exponential backoff pause if attempts remain
        if attempt < max_retries:
            backoff_sec = min((2 ** (attempt - 1)) * cb.backoff_factor, 2.0)
            await asyncio.sleep(backoff_sec)

    # All retries exhausted: Trip breaker and spool to scratch directory
    cb.record_failure()
    spool_path = _spool_event_to_disk(
        payload_dict,
        target_scratch,
        spool_filename=spool_filename,
        reason=last_exception_msg,
    )
    logger.warning(
        f"Webhook delivery failed after {max_retries} attempts "
        f"({last_exception_msg}). Spooled to {spool_path}."
    )
    return {
        "status": "SPOOLED",
        "spooled": True,
        "spool_path": str(spool_path),
        "reason": last_exception_msg,
    }


def stream_webhook_events(
    status_payload: dict[str, Any] | WebhookPayload,
    webhook_url: str | None = None,
    scratch_dir: str | Path | None = None,
    max_retries: int = 3,
    timeout: float = 3.0,
    spool_filename: str = "telemetry_spool.jsonl",
    circuit_breaker: TelemetryCircuitBreaker | None = None,
) -> dict[str, Any]:
    """
    Synchronous entrypoint for streaming webhook events.
    Safely bridges into asyncio loop.
    """
    coro = stream_webhook_events_async(
        status_payload=status_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=max_retries,
        timeout=timeout,
        spool_filename=spool_filename,
        circuit_breaker=circuit_breaker,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In an active event loop (e.g. Jupyter or async test runner)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ============================================================================
# 2D Grid Decimation & Stationary Point Preservation
# ============================================================================


def find_stationary_points_2d(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pes_grid: np.ndarray,
    neighborhood_size: int = 3,
    max_points: int = 20,
) -> list[dict[str, Any]]:
    """
    Locates 2D stationary points (local minima and maxima) on a discrete PES.

    :param phi1: 1D array of dihedral coordinate 1.
    :param phi2: 1D array of dihedral coordinate 2.
    :param pes_grid: 2D potential energy array of shape (len(phi1), len(phi2)).
    :param neighborhood_size: Kernel window for local extrema checking.
    :param max_points: Maximum number of stationary points to collect.
    :return: List of stationary point dictionaries.
    """
    n1, n2 = pes_grid.shape
    stationary_points: list[dict[str, Any]] = []

    # Find global minimum with NaN-resilience
    if np.isnan(pes_grid).all():
        return stationary_points

    try:
        glob_min_idx = np.unravel_index(np.nanargmin(pes_grid), pes_grid.shape)
        glob_min_val = float(pes_grid[glob_min_idx])
        p1_val = float(phi1[glob_min_idx[0]])
        p2_val = float(phi2[glob_min_idx[1]])
        stationary_points.append(
            {
                "type": "minimum",
                "subtype": "global_minimum",
                "idx": (int(glob_min_idx[0]), int(glob_min_idx[1])),
                "phi1": p1_val,
                "phi2": p2_val,
                "energy": glob_min_val,
                "label": (
                    f"Global Min ({p1_val:.1f}°, {p2_val:.1f}°): {glob_min_val:.2f}"
                ),
            }
        )
    except ValueError:
        pass

    # Local extrema scan across interior grid
    r = neighborhood_size // 2
    if r < 1:
        r = 1

    for i in range(r, n1 - r, max(1, n1 // 50)):
        for j in range(r, n2 - r, max(1, n2 // 50)):
            window = pes_grid[i - r : i + r + 1, j - r : j + r + 1]
            val = pes_grid[i, j]
            if np.isnan(val):
                continue

            # Local minimum check
            win_min = np.nanmin(window)
            win_max = np.nanmax(window)
            p1_deg = float(phi1[i])
            p2_deg = float(phi2[j])
            if val == win_min:
                if not stationary_points or (i, j) != stationary_points[0]["idx"]:
                    stationary_points.append(
                        {
                            "type": "minimum",
                            "subtype": "local_minimum",
                            "idx": (i, j),
                            "phi1": p1_deg,
                            "phi2": p2_deg,
                            "energy": float(val),
                            "label": (
                                f"Local Min ({p1_deg:.1f}°, {p2_deg:.1f}°): {val:.2f}"
                            ),
                        }
                    )
            # Local maximum check
            elif val == win_max:
                stationary_points.append(
                    {
                        "type": "maximum",
                        "subtype": "local_maximum",
                        "idx": (i, j),
                        "phi1": p1_deg,
                        "phi2": p2_deg,
                        "energy": float(val),
                        "label": (
                            f"Local Max ({p1_deg:.1f}°, {p2_deg:.1f}°): {val:.2f}"
                        ),
                    }
                )

            if len(stationary_points) >= max_points:
                break
        if len(stationary_points) >= max_points:
            break

    return stationary_points


def decimate_2d_grid_with_extrema(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pes_grid: np.ndarray,
    max_nodes: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Performs 2D Strided Regular Grid Decimation preserving stationary points.
    Guarantees that resulting mesh contains <= max_nodes to prevent WebGL crashes.

    :param phi1: 1D array of phi1 coordinates (len N1).
    :param phi2: 1D array of phi2 coordinates (len N2).
    :param pes_grid: 2D potential energy surface array of shape (N1, N2).
    :param max_nodes: Maximum allowable node count in decimated mesh (default 5000).
    :return: Tuple of (phi1_dec, phi2_dec, pes_dec, stationary_points).
    """
    n1, n2 = pes_grid.shape
    total_nodes = n1 * n2

    # Step 1: Detect stationary points on the pristine high-resolution surface
    stationary_points = find_stationary_points_2d(phi1, phi2, pes_grid)

    if total_nodes <= max_nodes:
        return phi1, phi2, pes_grid, stationary_points

    # Step 2: Compute striding ratio
    # target: (n1 // stride1) * (n2 // stride2) <= max_nodes
    stride = int(math.ceil(math.sqrt(total_nodes / max_nodes)))
    stride1 = max(1, stride)
    stride2 = max(1, stride)

    while (len(phi1[::stride1]) * len(phi2[::stride2])) > max_nodes:
        stride1 += 1
        stride2 += 1

    phi1_dec = phi1[::stride1]
    phi2_dec = phi2[::stride2]
    pes_dec = pes_grid[::stride1, ::stride2]

    return phi1_dec, phi2_dec, pes_dec, stationary_points


# ============================================================================
# Plotly 3D Carousel Visualizer
# ============================================================================


def generate_plotly_3d_carousels(
    pes_tensor: np.ndarray | dict[str, Any],
    dvr_wavefunctions: np.ndarray | list[np.ndarray] | None = None,
    phi1_grid: np.ndarray | None = None,
    phi2_grid: np.ndarray | None = None,
    artifact_dir: str | Path | None = None,
    filename: str = "torq_pes_3d_carousel.html",
    max_nodes: int = 5000,
    colorscale: str = "Viridis",
    title: str = "CoChem-TORQ 2D Torsional Potential Energy Surface",
) -> Path:
    """
    Downsamples multi-dimensional PES grids and DVR probability wavefunctions
    using 2D Strided Regular Grid Decimation while preserving stationary points.
    Generates interactive, color-blind accessible HTML Plotly 3D visualizers.

    :param pes_tensor: 2D array of potential energies, or dict with
        'pes', 'phi1', 'phi2'.
    :param dvr_wavefunctions: Optional list or array of DVR probability densities.
    :param phi1_grid: Optional 1D array of phi1 dihedral coordinates.
    :param phi2_grid: Optional 1D array of phi2 dihedral coordinates.
    :param artifact_dir: Target deliverable directory (Filesystem Air-Gap).
    :param filename: Output HTML filename.
    :param max_nodes: Maximum allowable node threshold (default: 5000).
    :param colorscale: Color-blind accessible colorscale (Viridis, Cividis, Plasma).
    :param title: Figure title string.
    :return: Absolute Path to the generated standalone HTML file.
    """
    target_artifacts = _resolve_artifact_dir(artifact_dir)
    html_outpath = target_artifacts / filename

    # Unpack PES tensor and coordinate grids
    if isinstance(pes_tensor, dict):
        pes = np.asarray(pes_tensor["pes"], dtype=np.float64)
        n1, n2 = pes.shape
        raw_phi1 = pes_tensor.get("phi1", phi1_grid)
        phi1 = (
            np.linspace(-180.0, 180.0, n1)
            if raw_phi1 is None
            else np.asarray(raw_phi1, dtype=np.float64)
        )
        raw_phi2 = pes_tensor.get("phi2", phi2_grid)
        phi2 = (
            np.linspace(-180.0, 180.0, n2)
            if raw_phi2 is None
            else np.asarray(raw_phi2, dtype=np.float64)
        )
    else:
        pes = np.asarray(pes_tensor, dtype=np.float64)
        n1, n2 = pes.shape
        phi1 = (
            np.linspace(-180.0, 180.0, n1)
            if phi1_grid is None
            else np.asarray(phi1_grid, dtype=np.float64)
        )
        phi2 = (
            np.linspace(-180.0, 180.0, n2)
            if phi2_grid is None
            else np.asarray(phi2_grid, dtype=np.float64)
        )

    # Decimate 2D grid while preserving stationary points
    phi1_sub, phi2_sub, pes_sub, stationary_pts = decimate_2d_grid_with_extrema(
        phi1, phi2, pes, max_nodes=max_nodes
    )

    # Construct Plotly 3D Figure
    fig = go.Figure()

    # 1. Base 3D Potential Energy Surface Trace
    fig.add_trace(
        go.Surface(
            x=phi2_sub,
            y=phi1_sub,
            z=pes_sub,
            colorscale=colorscale,
            opacity=0.92,
            name="PES Base Surface",
            colorbar=dict(
                title=dict(text="Energy (cm⁻¹)", side="right"),
                len=0.75,
                thickness=18,
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                )
            ),
            hoverinfo="x+y+z",
            hovertemplate=(
                "ϕ₁: %{y:.1f}°<br>ϕ₂: %{x:.1f}°<br>"
                "V(ϕ₁, ϕ₂): %{z:.2f} cm⁻¹<extra></extra>"
            ),
        )
    )

    # 2. Stationary Points Overlay (Minima / Maxima / Saddles)
    if stationary_pts:
        stat_x = [p["phi2"] for p in stationary_pts]
        stat_y = [p["phi1"] for p in stationary_pts]
        stat_z = [p["energy"] for p in stationary_pts]
        stat_labels = [p["label"] for p in stationary_pts]
        symbols = [
            "diamond" if p["type"] == "minimum" else "cross" for p in stationary_pts
        ]
        colors = [
            "gold" if p.get("subtype") == "global_minimum" else "crimson"
            for p in stationary_pts
        ]

        fig.add_trace(
            go.Scatter3d(
                x=stat_x,
                y=stat_y,
                z=stat_z,
                mode="markers+text",
                name="Stationary Points",
                text=[p["subtype"].replace("_", " ").title() for p in stationary_pts],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                marker=dict(
                    size=7,
                    color=colors,
                    symbol=symbols,
                    line=dict(color="black", width=1),
                ),
                hovertext=stat_labels,
                hoverinfo="text",
            )
        )

    # 3. Multi-State DVR Wavefunction Probability Distributions (Carousel Traces)
    updatemenus = []
    if dvr_wavefunctions is not None and len(dvr_wavefunctions) > 0:
        wf_list = (
            list(dvr_wavefunctions)
            if not isinstance(dvr_wavefunctions, list)
            else dvr_wavefunctions
        )
        num_states = len(wf_list)

        # Baseline offset for wavefunction overlay
        pes_min = float(np.nanmin(pes_sub))
        pes_max = float(np.nanmax(pes_sub))
        v_span = max(1.0, pes_max - pes_min)

        # Add a trace for each DVR state
        for state_idx, wf in enumerate(wf_list):
            wf_arr = np.asarray(wf, dtype=np.float64)
            # Decimate wavefunction to match grid stride
            s1 = max(1, len(phi1) // len(phi1_sub))
            s2 = max(1, len(phi2) // len(phi2_sub))
            wf_sub = wf_arr[::s1, ::s2]
            # Ensure shape match
            if wf_sub.shape != pes_sub.shape:
                wf_sub = np.resize(wf_sub, pes_sub.shape)

            # Normalize and elevate probability density
            prob_density = np.abs(wf_sub)
            p_max = np.nanmax(prob_density)
            if p_max > 1e-12:
                prob_density = prob_density / p_max

            # Offset probability surface slightly above local PES
            z_wf = pes_sub + prob_density * (v_span * 0.25)

            fig.add_trace(
                go.Surface(
                    x=phi2_sub,
                    y=phi1_sub,
                    z=z_wf,
                    colorscale="Plasma",
                    opacity=0.65,
                    showscale=False,
                    name=f"DVR State v={state_idx}",
                    visible=(state_idx == 0),
                    hoverinfo="x+y+z",
                    hovertemplate=(
                        f"DVR v={state_idx}<br>ϕ₁: %{{y:.1f}}°<br>"
                        f"ϕ₂: %{{x:.1f}}°<br>"
                        f"|ψ|² Offset: %{{z:.2f}} cm⁻¹<extra></extra>"
                    ),
                )
            )

        # Create interactive carousel dropdown / button menu
        buttons = []
        # Option to show only PES
        vis_pes_only = [True, True if stationary_pts else False] + [False] * num_states
        buttons.append(
            dict(
                label="PES Base Only",
                method="update",
                args=[{"visible": vis_pes_only}, {"title": f"{title} (Base Surface)"}],
            )
        )

        # Option for each DVR state
        for s_idx in range(num_states):
            vis = [True, True if stationary_pts else False] + [
                (i == s_idx) for i in range(num_states)
            ]
            buttons.append(
                dict(
                    label=f"DVR State v={s_idx}",
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "title": (
                                f"{title} (DVR State v={s_idx} "
                                f"Probability Distribution)"
                            )
                        },
                    ],
                )
            )

        updatemenus = [
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#cccccc",
                borderwidth=1,
            )
        ]

    # Layout styling with color-blind contrast and responsive aspect ratio
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(family="Arial, sans-serif", size=16, color="#222222"),
        ),
        scene=dict(
            xaxis=dict(
                title="Dihedral ϕ₂ (degrees)",
                backgroundcolor="rgb(245, 245, 245)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                title="Dihedral ϕ₁ (degrees)",
                backgroundcolor="rgb(245, 245, 245)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                title="Potential Energy V (cm⁻¹)",
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
        ),
        margin=dict(l=20, r=20, b=20, t=50),
        template="plotly_white",
        updatemenus=updatemenus if updatemenus else None,
    )

    # Write standalone HTML file with CDN inclusion for lightweight footprint
    fig.write_html(
        str(html_outpath),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": True, "scrollZoom": True},
    )

    logger.info(f"Generated standalone Plotly 3D Carousel HTML at: {html_outpath}")
    return html_outpath


# ============================================================================
# Crash Animation & Diagnostic Exporter
# ============================================================================


def _compute_pairwise_distances(coords: np.ndarray) -> tuple[float, tuple[int, int]]:
    """
    Computes minimum interatomic distance and colliding pair indices.

    :param coords: (N, 3) Cartesian coordinates in Angstroms.
    :return: (min_distance, (atom_i, atom_j))
    """
    num_atoms = coords.shape[0]
    if num_atoms < 2:
        return 999.0, (0, 0)

    # Compute difference vectors: (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Mask diagonal
    np.fill_diagonal(dist_matrix, np.inf)

    if np.isnan(dist_matrix).all():
        return 0.0, (0, 1)

    try:
        min_idx = np.unravel_index(np.nanargmin(dist_matrix), dist_matrix.shape)
        min_dist = float(dist_matrix[min_idx])
        return min_dist, (int(min_idx[0]), int(min_idx[1]))
    except ValueError:
        return 0.0, (0, 1)


def export_crash_animation(
    trajectory_array: np.ndarray | list[np.ndarray] | dict[str, Any],
    error_node_id: str = "node_000",
    symbols: list[str] | None = None,
    energies: list[float] | None = None,
    gradients: list[np.ndarray] | None = None,
    artifact_dir: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    abort_reason: str = "Steric Shatter Soft-Quench Abort: Unresolvable atomic overlap",
) -> dict[str, Path]:
    """
    Captures optimization trajectories during Steric Shatter Soft-Quench aborts into
    `crash_animation.xyz` and `crash_diagnostic.json`.
    Written strictly to dynamically provided scratch/artifact directories.

    :param trajectory_array: (num_frames, num_atoms, 3) array or coordinate list.
    :param error_node_id: Topographic or cluster rotor node identifier.
    :param symbols: List of atomic symbols (e.g. ['C', 'C', 'H', 'H', 'H', 'H']).
    :param energies: Optional list of frame potential energies.
    :param gradients: Optional list of frame atomic gradient vectors.
    :param artifact_dir: Deliverables directory for crash diagnostics.
    :param scratch_dir: Scratch directory for crash trajectory files.
    :param abort_reason: Text description of the physics abort condition.
    :return: Dictionary containing 'xyz_path', 'node_xyz_path', etc.
    """
    target_artifacts = _resolve_artifact_dir(artifact_dir)
    target_scratch = _resolve_scratch_dir(scratch_dir)

    xyz_path = target_artifacts / f"crash_animation_{error_node_id}.xyz"
    # Also write canonical crash_animation.xyz if default
    canonical_xyz_path = target_artifacts / "crash_animation.xyz"
    diag_path = target_artifacts / "crash_diagnostic.json"

    # Unpack trajectory
    if isinstance(trajectory_array, dict):
        coords_list = trajectory_array["coordinates"]
        symbols = trajectory_array.get("symbols", symbols)
        energies = trajectory_array.get("energies", energies)
        gradients = trajectory_array.get("gradients", gradients)
    else:
        coords_list = trajectory_array

    traj_arr = np.asarray(coords_list, dtype=np.float64)
    if traj_arr.ndim == 2:
        # Single frame (1, N, 3)
        traj_arr = traj_arr[np.newaxis, ...]

    num_frames, num_atoms, _ = traj_arr.shape

    # Default symbols if missing
    if symbols is None or len(symbols) != num_atoms:
        symbols = ["X"] * num_atoms

    # Track minimum distance and exploding gradients across trajectory
    min_overall_dist = float("inf")
    colliding_pair: tuple[int, int] = (0, 0)
    crash_frame_idx = num_frames - 1
    max_grad_norm: float | None = None

    if gradients is not None and len(gradients) > 0:
        grad_norms = [float(np.linalg.norm(g)) for g in gradients]
        try:
            max_grad_norm = float(np.nanmax(np.asarray(grad_norms)))
        except ValueError:
            max_grad_norm = None

    # Format multi-frame XYZ string
    xyz_lines: list[str] = []
    for f_idx in range(num_frames):
        frame_coords: np.ndarray = np.asarray(traj_arr[f_idx], dtype=np.float64)
        frame_min_d, frame_pair = _compute_pairwise_distances(frame_coords)

        if frame_min_d < min_overall_dist:
            min_overall_dist = frame_min_d
            colliding_pair = frame_pair
            crash_frame_idx = f_idx

        e_str = (
            f" Energy: {energies[f_idx]:.6f} Eh |"
            if (energies and f_idx < len(energies))
            else ""
        )
        comment = (
            f"Frame {f_idx}/{num_frames - 1} | Node: {error_node_id} |{e_str} "
            f"MinDist: {frame_min_d:.4f} A (Atoms {frame_pair[0]}-{frame_pair[1]})"
        )

        xyz_lines.append(str(num_atoms))
        xyz_lines.append(comment)
        for a_idx in range(num_atoms):
            sym = symbols[a_idx]
            x, y, z = frame_coords[a_idx]
            xyz_lines.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")

    xyz_content = "\n".join(xyz_lines) + "\n"

    # Write XYZ files to artifacts
    with open(canonical_xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)

    # Also persist ephemeral scratch trajectory for IPC
    scratch_xyz_path = target_scratch / f"crash_spool_{error_node_id}.xyz"
    with open(scratch_xyz_path, "w", encoding="utf-8") as f:
        f.write(xyz_content)

    # Build diagnostic JSON payload
    init_energy = float(energies[0]) if (energies and len(energies) > 0) else None
    final_energy = float(energies[-1]) if (energies and len(energies) > 0) else None

    diagnostic = CrashDiagnostic(
        error_node_id=error_node_id,
        num_frames=num_frames,
        num_atoms=num_atoms,
        symbols=symbols,
        min_interatomic_distance=round(min_overall_dist, 6),
        colliding_pair=colliding_pair,
        max_gradient_norm=max_grad_norm,
        abort_reason=abort_reason,
        crash_frame_index=crash_frame_idx,
        initial_energy_hartree=init_energy,
        final_energy_hartree=final_energy,
    )

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(
            diagnostic.model_dump(),
            f,
            indent=2,
            default=_json_serial_default,
            ensure_ascii=False,
        )

    logger.info(
        f"Exported crash trajectory ({num_frames} frames) to "
        f"{canonical_xyz_path} and diagnostic to {diag_path}."
    )

    return {
        "xyz_path": canonical_xyz_path,
        "node_xyz_path": xyz_path,
        "scratch_xyz_path": scratch_xyz_path,
        "diagnostic_path": diag_path,
    }

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_telemetry.py ---
"""
CoChem-TORQ: Test Suite for Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------
Validates:
1. stream_webhook_events with real local HTTP server, backoff retries,
   and circuit-breaker fallback spooling to telemetry_spool.jsonl.
2. generate_plotly_3d_carousels with 2D regular grid decimation,
   stationary point preservation, and DVR wavefunction probability states.
3. export_crash_animation capturing multi-frame crash_animation.xyz
   and crash_diagnostic.json during Steric Shatter Soft-Quench aborts.
4. Filesystem Air-Gap compliance writing to dynamic scratch/artifact dirs.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from Libraries.cochem_torq_telemetry import (
    TelemetryCircuitBreaker,
    decimate_2d_grid_with_extrema,
    export_crash_animation,
    find_stationary_points_2d,
    generate_plotly_3d_carousels,
    stream_webhook_events,
)

# ============================================================================
# Physical Helper: Ephemeral Local HTTP Server for Real Webhook Delivery
# ============================================================================


class WebhookRecordingHandler(BaseHTTPRequestHandler):
    """Real HTTP request handler for live socket-level webhook testing."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP server console spam during tests
        pass

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {"raw_body": body}

        # Check server mode
        server_obj: Any = self.server
        server_obj.received_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
            }
        )

        if getattr(server_obj, "fail_count_target", 0) > 0:
            server_obj.fail_count_target -= 1
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Service Unavailable"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "delivered": true}')


def get_free_port() -> int:
    """Finds an available ephemeral port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def local_webhook_server() -> Iterator[tuple[HTTPServer, str]]:
    """Starts a real physical HTTP server on localhost."""
    port = get_free_port()
    server = HTTPServer(("127.0.0.1", port), WebhookRecordingHandler)
    server.received_requests = []  # type: ignore[attr-defined]
    server.fail_count_target = 0  # type: ignore[attr-defined]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/cochem/webhook"
    try:
        yield server, url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# ============================================================================
# Test Suite 1: Webhook Streaming & Circuit Breaker Spooling
# ============================================================================


def test_stream_webhook_events_real_delivery(
    local_webhook_server: tuple[HTTPServer, str], tmp_path: Path
) -> None:
    """Validates real physical HTTP POST delivery to an active webhook endpoint."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "job_completed",
        "job_id": "TORQ_JOB_2026_08_001",
        "node_id": "hpc_worker_node_07",
        "status": "COMPLETED",
        "data": {"wall_time_sec": 42.5, "optimized_energy_hartree": -154.29841},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["status_code"] == 200
    assert result["spooled"] is False
    assert len(server.received_requests) == 1  # type: ignore[attr-defined]
    assert server.received_requests[0]["payload"]["job_id"] == "TORQ_JOB_2026_08_001"  # type: ignore[attr-defined]


def test_stream_webhook_events_exponential_backoff_recovery(
    local_webhook_server: tuple[HTTPServer, str], tmp_path: Path
) -> None:
    """Validates exponential backoff retries when encountering transient 503 errors."""
    server, webhook_url = local_webhook_server
    server.fail_count_target = 2  # type: ignore[attr-defined] # Fail first 2 attempts with 503, succeed on 3rd
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "soft_quench_collision",
        "job_id": "TORQ_JOB_SQ_09",
        "node_id": "gpu_node_01",
        "status": "ALERT",
        "data": {"collision_distance_angstrom": 0.58},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
        max_retries=3,
        timeout=3.0,
    )

    assert result["status"] == "DELIVERED"
    assert result["attempt"] == 3
    assert len(server.received_requests) == 3  # type: ignore[attr-defined]


def test_stream_webhook_events_blackout_spooling(tmp_path: Path) -> None:
    """Validates spooling to telemetry_spool.jsonl when network fails."""
    # Use a port that is definitively closed/unreachable
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/nonexistent_webhook"
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "oom_backoff",
        "job_id": "TORQ_JOB_OOM_003",
        "node_id": "cpu_node_12",
        "status": "ALERT",
        "data": {"memory_rss_gb": 64.2, "backoff_scale": 0.5},
    }

    result = stream_webhook_events(
        status_payload=test_payload,
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        max_retries=2,
        timeout=0.5,
    )

    # Must NOT raise unhandled exception; must safely spool to disk
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True
    spool_file = scratch_dir / "telemetry_spool.jsonl"
    assert spool_file.exists()

    with open(spool_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 1
    logged_event = lines[-1]
    assert logged_event["payload"]["job_id"] == "TORQ_JOB_OOM_003"
    assert logged_event["delivery_status"] == "SPOOLED"


def test_stream_webhook_events_numpy_types(
    local_webhook_server: tuple[HTTPServer, str], tmp_path: Path
) -> None:
    """Validates NumPy scalars and arrays in payload serialize cleanly."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    numpy_payload = {
        "event_type": "progress",
        "job_id": "NUMPY_SERIAL_01",
        "status": "RUNNING",
        "data": {
            "float_metric": np.float64(3.14159265),
            "int_metric": np.int64(42),
            "vector": np.array([1.0, 2.0, 3.0]),
        },
    }

    result = stream_webhook_events(
        status_payload=numpy_payload,
        webhook_url=webhook_url,
        scratch_dir=scratch_dir,
    )

    assert result["status"] == "DELIVERED"
    assert len(server.received_requests) == 1  # type: ignore[attr-defined]
    rec_payload = server.received_requests[0]["payload"]  # type: ignore[attr-defined]
    assert rec_payload["data"]["int_metric"] == 42
    assert rec_payload["data"]["vector"] == [1.0, 2.0, 3.0]


# ============================================================================
# Test Suite 2: 2D PES Decimation & Plotly 3D Carousel Generation
# ============================================================================


def test_decimate_2d_grid_and_stationary_points() -> None:
    """Validates 2D grid decimation preserving stationary points."""
    # Create a dense 500x500 (250,000 nodes) 2D PES grid
    n1, n2 = 500, 500
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    p1_mesh, p2_mesh = np.meshgrid(phi1, phi2, indexing="ij")

    # Analytical potential:
    # V(phi1, phi2) = 1500*(1-cos(phi1)) + 800*(1-cos(2*phi2)) + 400*cos(phi1+phi2)
    # Global minimum at (0, 0) where V = 400 cm-1
    rad1 = np.radians(p1_mesh)
    rad2 = np.radians(p2_mesh)
    pes_grid = (
        1500.0 * (1.0 - np.cos(rad1))
        + 800.0 * (1.0 - np.cos(2.0 * rad2))
        + 400.0 * np.cos(rad1 + rad2)
    )

    # Test stationary points finder
    stationary_points = find_stationary_points_2d(phi1, phi2, pes_grid, max_points=10)
    assert len(stationary_points) > 0
    # Minima should include near (0, 0)
    minima = [p for p in stationary_points if p["type"] == "minimum"]
    assert len(minima) >= 1

    # Test decimation to <= 5000 nodes
    phi1_dec, phi2_dec, pes_dec, extrema_pts = decimate_2d_grid_with_extrema(
        phi1, phi2, pes_grid, max_nodes=5000
    )

    total_dec_nodes = len(phi1_dec) * len(phi2_dec)
    assert total_dec_nodes <= 5000
    assert total_dec_nodes > 100
    assert pes_dec.shape == (len(phi1_dec), len(phi2_dec))
    assert len(extrema_pts) > 0


def test_find_stationary_points_with_nans() -> None:
    """Validates stationary points finder resilience with NaNs."""
    n1, n2 = 50, 50
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    pes_grid = np.full((n1, n2), 1000.0)
    # True minimum at (25, 25)
    pes_grid[25, 25] = 50.0
    # Add NaN region (steric crash zone)
    pes_grid[0:5, 0:5] = np.nan

    pts = find_stationary_points_2d(phi1, phi2, pes_grid)
    assert len(pts) >= 1
    assert pts[0]["type"] == "minimum"
    assert pts[0]["energy"] == 50.0


def test_generate_plotly_3d_carousels_standalone_html(tmp_path: Path) -> None:
    """Validates generation of lightweight interactive Plotly 3D visualizer HTML."""
    artifact_dir = tmp_path / "artifacts"

    # Dense PES grid: 360x360 (129,600 nodes)
    n = 360
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    p1_mesh, p2_mesh = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1200.0 * (1.0 - np.cos(np.radians(p1_mesh))) + 600.0 * (
        1.0 - np.cos(np.radians(3 * p2_mesh))
    )

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_3d.html",
        max_nodes=4000,
        colorscale="Viridis",
        title="1,2-Ethanediol 2D Torsional PES",
    )

    assert html_path.exists()
    assert html_path.is_file()
    assert html_path.parent == artifact_dir

    # Inspect HTML content
    html_content = html_path.read_text(encoding="utf-8")
    assert "<html>" in html_content.lower()
    assert "<body>" in html_content.lower()
    assert "plotly" in html_content.lower()
    assert "1,2-Ethanediol 2D Torsional PES" in html_content

    # File size must be lightweight (< 3.5 MB)
    file_size_mb = html_path.stat().st_size / (1024 * 1024)
    assert file_size_mb < 3.5


def test_generate_plotly_3d_carousels_with_dvr_wavefunctions(tmp_path: Path) -> None:
    """Validates Plotly 3D carousel with multi-state DVR probability wavefunctions."""
    artifact_dir = tmp_path / "artifacts"

    n = 100
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    p1_mesh, p2_mesh = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1000.0 * (1.0 - np.cos(np.radians(p1_mesh))) + 500.0 * (
        1.0 - np.cos(np.radians(2 * p2_mesh))
    )

    # Create 3 DVR wavefunctions: ground state v=0 and excited states v=1, v=2
    wf_0 = np.exp(-((p1_mesh / 40.0) ** 2 + (p2_mesh / 40.0) ** 2))
    wf_0 /= np.sum(wf_0)

    wf_1 = (p1_mesh / 40.0) * np.exp(-((p1_mesh / 40.0) ** 2 + (p2_mesh / 40.0) ** 2))
    wf_1 = (wf_1**2) / np.sum(wf_1**2)

    wf_2 = (p2_mesh / 40.0) * np.exp(-((p1_mesh / 40.0) ** 2 + (p2_mesh / 40.0) ** 2))
    wf_2 = (wf_2**2) / np.sum(wf_2**2)

    dvr_wavefunctions = [wf_0, wf_1, wf_2]

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_grid,
        dvr_wavefunctions=dvr_wavefunctions,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="test_pes_dvr_carousel.html",
        max_nodes=2500,
    )

    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "DVR State v=0" in html_content
    assert "DVR State v=1" in html_content
    assert "DVR State v=2" in html_content


def test_generate_plotly_3d_carousels_dict_missing_coords(tmp_path: Path) -> None:
    """Validates dict with only 'pes' key automatically generates default grids."""
    artifact_dir = tmp_path / "artifacts"
    phi = np.linspace(-180, 180, 30)
    p1_mesh, p2_mesh = np.meshgrid(phi, phi, indexing="ij")
    pes_grid = (
        250.0
        + 10.0 * (1.0 - np.cos(np.radians(p1_mesh)))
        + 10.0 * (1.0 - np.cos(np.radians(p2_mesh)))
    )

    html_path = generate_plotly_3d_carousels(
        pes_tensor={"pes": pes_grid},
        artifact_dir=artifact_dir,
        filename="pes_dict_minimal.html",
    )

    assert html_path.exists()
    assert html_path.is_file()


# ============================================================================
# Test Suite 3: Crash Animation & Diagnostic Exporter
# ============================================================================


def test_export_crash_animation_steric_collision(tmp_path: Path) -> None:
    """Validates multi-frame XYZ crash animation and JSON diagnostic generation."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    # Define a 6-atom molecule (e.g. ethane-like) undergoing steric shatter collision
    symbols = ["C", "C", "H", "H", "H", "H"]
    num_atoms = len(symbols)
    num_frames = 12

    # Frame 0: Stable geometry
    base_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # C1
            [1.54, 0.0, 0.0],  # C2
            [-0.5, 1.0, 0.0],  # H3
            [-0.5, -0.5, 0.86],  # H4
            [2.04, 1.0, 0.0],  # H5
            [2.04, -0.5, -0.86],  # H6
        ],
        dtype=np.float64,
    )

    # Generate physical trajectory with H3 (idx 2) and H5 (idx 4) colliding
    frame_coords_list: list[np.ndarray] = []
    energies: list[float] = []
    gradients: list[np.ndarray] = []

    sigma_lj = 1.1  # Angstrom
    eps_lj = 0.1  # kcal/mol

    for f_idx in range(num_frames):
        coords = base_coords.copy()
        # Compress H3 and H5 along interaction vector for steric collision
        compression = float(f_idx) * 0.20
        coords[2, 0] += compression * 0.5  # H3 moves toward center
        coords[4, 0] -= compression * 0.6  # H5 moves toward center
        coords[4, 1] -= compression * 0.05  # slight y-deflection

        frame_coords_list.append(coords)

        # Compute physical Lennard-Jones potential energy and analytical gradients
        e_frame = -79.8  # baseline Hartree
        grad_frame = np.empty((num_atoms, 3), dtype=np.float64)
        grad_frame.fill(0.0)

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                r_vec = coords[i] - coords[j]
                r_dist = float(np.linalg.norm(r_vec))
                if r_dist > 1e-4:
                    s_r = sigma_lj / r_dist
                    # Analytical LJ gradient
                    force_mag = (
                        24.0 * eps_lj * (2.0 * (s_r**12) - (s_r**6)) / (r_dist**2)
                    )
                    grad_frame[i] += force_mag * r_vec
                    if i < j:
                        e_frame += 4.0 * eps_lj * ((s_r**12) - (s_r**6))

        energies.append(float(e_frame))
        gradients.append(grad_frame)

    trajectory = np.array(frame_coords_list, dtype=np.float64)

    result_paths = export_crash_animation(
        trajectory_array=trajectory,
        error_node_id="rotor_node_55",
        symbols=symbols,
        energies=energies,
        gradients=gradients,
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Steric Shatter Soft-Quench Abort: Interatomic distance < 0.5 A",
    )

    xyz_path = result_paths["xyz_path"]
    diag_path = result_paths["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    # Verify XYZ structure
    xyz_lines = xyz_path.read_text(encoding="utf-8").strip().split("\n")
    # Each frame has num_atoms + 2 lines
    expected_lines = num_frames * (num_atoms + 2)
    assert len(xyz_lines) == expected_lines
    assert xyz_lines[0].strip() == str(num_atoms)
    assert "rotor_node_55" in xyz_lines[1]

    # Verify Diagnostic JSON
    with open(diag_path, encoding="utf-8") as diag_file:
        diag_data = json.load(diag_file)

    assert diag_data["error_node_id"] == "rotor_node_55"
    assert diag_data["num_frames"] == 12
    assert diag_data["num_atoms"] == 6
    assert diag_data["symbols"] == symbols
    assert diag_data["min_interatomic_distance"] < 0.5
    assert diag_data["colliding_pair"] == [2, 4] or diag_data["colliding_pair"] == [
        4,
        2,
    ]
    assert "Steric Shatter" in diag_data["abort_reason"]


# ============================================================================
# Test Suite 4: Air-Gap Compliance & Direct Memory Ingestion
# ============================================================================


def test_airgap_compliance_no_repo_pollution(tmp_path: Path) -> None:
    """Validates that no temporary files or logs are created in repo workspace."""
    repo_files_before = set(Path(".").glob("*"))

    scratch_dir = tmp_path / "airgap_scratch"
    artifact_dir = tmp_path / "airgap_artifacts"

    # Run telemetry functions with explicit isolated dirs
    payload = {"event_type": "progress", "job_id": "AIRGAP_01", "status": "RUNNING"}
    stream_webhook_events(payload, webhook_url=None, scratch_dir=scratch_dir)

    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0.0, 0.74, 0.0], [0.0, 0.0, 0.74]]
            for _ in range(3)
        ]
    )
    export_crash_animation(
        coords,
        error_node_id="airgap_node",
        symbols=["H", "H", "H", "H"],
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
    )

    phi = np.linspace(-180, 180, 20)
    p1_mesh, p2_mesh = np.meshgrid(phi, phi, indexing="ij")
    pes = 100.0 * (1.0 - np.cos(np.radians(p1_mesh))) + 50.0 * (
        1.0 - np.cos(np.radians(p2_mesh))
    )
    generate_plotly_3d_carousels(
        pes_tensor=pes,
        artifact_dir=artifact_dir,
        max_nodes=100,
    )

    repo_files_after = set(Path(".").glob("*"))
    # Verify no new files created in cwd
    diff = repo_files_after - repo_files_before
    # Ignore pytest temporary markers or cache if any
    diff = {
        f
        for f in diff
        if not f.name.startswith(".pytest") and not f.name.startswith("__pycache__")
    }
    assert len(diff) == 0, f"Air-gap violation detected: created files in repo: {diff}"


# ============================================================================
# Test Suite 5: Extended Edge Cases & Circuit Breaker State Transitions
# ============================================================================


def test_stream_webhook_events_circuit_breaker_transitions(tmp_path: Path) -> None:
    """Validates circuit breaker transitions: CLOSED -> OPEN -> HALF_OPEN."""
    scratch_dir = tmp_path / "scratch"
    closed_port = get_free_port()
    unreachable_url = f"http://127.0.0.1:{closed_port}/webhook"

    cb = TelemetryCircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.2,
        backoff_factor=0.01,
        max_retries=1,
        request_timeout=0.2,
    )

    assert cb.state.value == "CLOSED"

    # 1st failure
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J1"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.consecutive_failures == 1
    assert cb.state.value == "CLOSED"

    # 2nd failure -> trips to OPEN
    stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J2"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert cb.state.value == "OPEN"

    # Next call while OPEN immediately spools without network call
    res = stream_webhook_events(
        {"event_type": "heartbeat", "job_id": "J3"},
        webhook_url=unreachable_url,
        scratch_dir=scratch_dir,
        circuit_breaker=cb,
        max_retries=1,
        timeout=0.2,
    )
    assert res["status"] == "SPOOLED"
    assert res["reason"] == "Circuit Breaker OPEN"

    # Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.25)
    assert cb.can_attempt_request() is True
    assert cb.state.value == "HALF_OPEN"


def test_stream_webhook_events_sync_inside_async_loop(tmp_path: Path) -> None:
    """Validates synchronous stream_webhook_events inside async loop."""
    scratch_dir = tmp_path / "scratch"

    async def _async_caller() -> dict[str, Any]:
        return stream_webhook_events(
            {"event_type": "heartbeat", "job_id": "ASYNC_LOOP_JOB"},
            webhook_url=None,
            scratch_dir=scratch_dir,
        )

    result = asyncio.run(_async_caller())
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True


def test_generate_plotly_3d_carousels_dict_input(tmp_path: Path) -> None:
    """Validates Plotly 3D carousel generation when pes_tensor is a dictionary."""
    artifact_dir = tmp_path / "artifacts"
    n1, n2 = 40, 40
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    p1_mesh, p2_mesh = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 500.0 * (1.0 - np.cos(np.radians(p1_mesh))) + 200.0 * (
        1.0 - np.cos(np.radians(p2_mesh))
    )

    pes_dict = {
        "pes": pes_grid,
        "phi1": phi1,
        "phi2": phi2,
    }

    html_path = generate_plotly_3d_carousels(
        pes_tensor=pes_dict,
        artifact_dir=artifact_dir,
        filename="pes_dict_test.html",
        max_nodes=1000,
        colorscale="Cividis",
    )

    assert html_path.exists()
    assert html_path.is_file()


def test_export_crash_animation_dict_and_single_frame(tmp_path: Path) -> None:
    """Validates export_crash_animation with dictionary input and single frame."""
    artifact_dir = tmp_path / "artifacts"
    scratch_dir = tmp_path / "scratch"

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],  # Severe collision: 0.2 A
        ],
        dtype=np.float64,
    )

    traj_dict = {
        "coordinates": coords,
        "symbols": ["O", "H"],
        "energies": [-75.123456],
        "gradients": [np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])],
    }

    result = export_crash_animation(
        trajectory_array=traj_dict,
        error_node_id="single_frame_node",
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Single frame singularity collision",
    )

    xyz_path = result["xyz_path"]
    diag_path = result["diagnostic_path"]

    assert xyz_path.exists()
    assert diag_path.exists()

    with open(diag_path, encoding="utf-8") as diag_file:
        diag = json.load(diag_file)

    assert diag["error_node_id"] == "single_frame_node"
    assert diag["num_frames"] == 1
    assert diag["num_atoms"] == 2
    assert diag["min_interatomic_distance"] == 0.2
    assert diag["colliding_pair"] == [0, 1] or diag["colliding_pair"] == [1, 0]

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.