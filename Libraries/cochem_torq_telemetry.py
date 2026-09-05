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
        import threading

        res: list[Any] = [None]
        err: list[Optional[Exception]] = [None]

        def _runner() -> None:
            try:
                res[0] = asyncio.run(coro)
            except Exception as exc:
                err[0] = exc

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        if err[0] is not None:
            raise err[0]
        return res[0]
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
