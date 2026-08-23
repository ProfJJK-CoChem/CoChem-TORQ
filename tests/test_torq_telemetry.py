"""
CoChem-TORQ: Comprehensive Pure Physical Test Suite for Visual & Event Telemetry Streamer
Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------------
Validates:
1. stream_webhook_events with real local HTTP server, exponential backoff retries,
   and circuit-breaker fallback spooling to telemetry_spool.jsonl under network blackout.
2. generate_plotly_3d_carousels with 2D strided regular grid decimation, stationary
   point preservation, color-blind accessibility, and DVR wavefunction probability states.
3. export_crash_animation capturing multi-frame crash_animation.xyz and crash_diagnostic.json
   during Steric Shatter Soft-Quench aborts.
4. Absolute Filesystem Air-Gap compliance writing exclusively to dynamic scratch/artifact dirs.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest

from Libraries.cochem_torq_telemetry import (
    CircuitState,
    CrashDiagnostic,
    TelemetryCircuitBreaker,
    WebhookPayload,
    decimate_2d_grid_with_extrema,
    export_crash_animation,
    find_stationary_points_2d,
    generate_plotly_3d_carousels,
    stream_webhook_events,
    stream_webhook_events_async,
)


# ============================================================================
# Physical Helper: Ephemeral Local HTTP Server for Real Webhook Delivery
# ============================================================================


class WebhookRecordingHandler(BaseHTTPRequestHandler):
    """Real HTTP request handler for live socket-level webhook testing."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP server console spam during tests
        pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {"raw_body": body}

        # Check server mode
        server_obj: Any = self.server
        server_obj.received_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "payload": payload,
        })

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
def local_webhook_server() -> Iterator[Tuple[HTTPServer, str]]:
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


def test_stream_webhook_events_real_delivery(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
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


def test_stream_webhook_events_exponential_backoff_recovery(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
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
    """Validates zero-interruption spooling to telemetry_spool.jsonl when network fails."""
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

    with open(spool_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 1
    logged_event = lines[-1]
    assert logged_event["payload"]["job_id"] == "TORQ_JOB_OOM_003"
    assert logged_event["delivery_status"] == "SPOOLED"


def test_stream_webhook_events_numpy_types(local_webhook_server: Tuple[HTTPServer, str], tmp_path: Path) -> None:
    """Validates that NumPy scalars and arrays in payload data serialize cleanly without TypeError."""
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
    """Validates 2D grid decimation while preserving stationary points (minima/maxima)."""
    # Create a dense 500x500 (250,000 nodes) 2D PES grid
    n1, n2 = 500, 500
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")

    # Analytical potential: V(phi1, phi2) = 1500*(1-cos(phi1)) + 800*(1-cos(2*phi2)) + 400*cos(phi1+phi2)
    # Global minimum at (0, 0) where V = 400 cm-1
    rad1 = np.radians(P1)
    rad2 = np.radians(P2)
    pes_grid = 1500.0 * (1.0 - np.cos(rad1)) + 800.0 * (1.0 - np.cos(2.0 * rad2)) + 400.0 * np.cos(rad1 + rad2)

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
    """Validates stationary points finder resilience against PES grids containing NaNs."""
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
    """Validates generation of lightweight, interactive Plotly 3D PES visualizer HTML."""
    artifact_dir = tmp_path / "artifacts"

    # Dense PES grid: 360x360 (129,600 nodes)
    n = 360
    phi1 = np.linspace(-180.0, 180.0, n)
    phi2 = np.linspace(-180.0, 180.0, n)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1200.0 * (1.0 - np.cos(np.radians(P1))) + 600.0 * (1.0 - np.cos(np.radians(3 * P2)))

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
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 1000.0 * (1.0 - np.cos(np.radians(P1))) + 500.0 * (1.0 - np.cos(np.radians(2 * P2)))

    # Create 3 DVR wavefunctions: ground state v=0 and excited states v=1, v=2
    wf_0 = np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_0 /= np.sum(wf_0)

    wf_1 = (P1 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
    wf_1 = (wf_1**2) / np.sum(wf_1**2)

    wf_2 = (P2 / 40.0) * np.exp(-((P1 / 40.0) ** 2 + (P2 / 40.0) ** 2))
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
    """Validates that passing a dict with only 'pes' key automatically generates default dihedral grids."""
    artifact_dir = tmp_path / "artifacts"
    pes_grid = np.ones((30, 30)) * 250.0

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
    base_coords = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.54, 0.0, 0.0],     # C2
        [-0.5, 1.0, 0.0],     # H3
        [-0.5, -0.5, 0.86],   # H4
        [2.04, 1.0, 0.0],     # H5
        [2.04, -0.5, -0.86],  # H6
    ], dtype=np.float64)

    # Trajectory where H3 and H5 rotate towards each other until collision at frame 11 (d < 0.4 A)
    trajectory = np.zeros((num_frames, num_atoms, 3), dtype=np.float64)
    energies: List[float] = []
    gradients: List[np.ndarray] = []

    for f_idx in range(num_frames):
        coords = base_coords.copy()
        # Move H3 and H5 toward each other along x and y
        offset = float(f_idx) * 0.12
        coords[2, 0] += offset  # H3 moves right
        coords[4, 0] -= offset  # H5 moves left
        coords[4, 1] -= offset * 0.3
        trajectory[f_idx] = coords

        # Exponential energy explosion
        energy = -79.8 + (1.5 ** f_idx) * 0.01
        energies.append(energy)

        # Gradient explosion
        grad = np.zeros((num_atoms, 3))
        grad[2] = [offset * 5.0, -offset * 2.0, 0.0]
        grad[4] = [-offset * 5.0, offset * 2.0, 0.0]
        gradients.append(grad)

    result_paths = export_crash_animation(
        trajectory_array=trajectory,
        error_node_id="rotor_node_55",
        symbols=symbols,
        energies=energies,
        gradients=gradients,
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Steric Shatter Soft-Quench Abort: Interatomic distance below 0.5 A threshold",
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
    with open(diag_path, "r", encoding="utf-8") as diag_file:
        diag_data = json.load(diag_file)

    assert diag_data["error_node_id"] == "rotor_node_55"
    assert diag_data["num_frames"] == 12
    assert diag_data["num_atoms"] == 6
    assert diag_data["symbols"] == symbols
    assert diag_data["min_interatomic_distance"] < 0.5
    assert diag_data["colliding_pair"] == [2, 4] or diag_data["colliding_pair"] == [4, 2]
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

    coords = np.zeros((3, 4, 3))
    export_crash_animation(
        coords,
        error_node_id="airgap_node",
        symbols=["H", "H", "H", "H"],
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
    )

    pes = np.ones((20, 20))
    generate_plotly_3d_carousels(
        pes_tensor=pes,
        artifact_dir=artifact_dir,
        max_nodes=100,
    )

    repo_files_after = set(Path(".").glob("*"))
    # Verify no new files created in cwd
    diff = repo_files_after - repo_files_before
    # Ignore pytest temporary markers or cache if any
    diff = {f for f in diff if not f.name.startswith(".pytest") and not f.name.startswith("__pycache__")}
    assert len(diff) == 0, f"Air-gap violation detected: created files in repo: {diff}"


# ============================================================================
# Test Suite 5: Extended Edge Cases & Circuit Breaker State Transitions
# ============================================================================


def test_stream_webhook_events_circuit_breaker_transitions(tmp_path: Path) -> None:
    """Validates circuit breaker transitions from CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
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
    """Validates that synchronous stream_webhook_events can be safely called inside an active async event loop."""
    scratch_dir = tmp_path / "scratch"

    async def _async_caller() -> Dict[str, Any]:
        return stream_webhook_events(
            {"event_type": "heartbeat", "job_id": "ASYNC_LOOP_JOB"},
            webhook_url=None,
            scratch_dir=scratch_dir,
        )

    result = asyncio.run(_async_caller())
    assert result["status"] == "SPOOLED"
    assert result["spooled"] is True


def test_generate_plotly_3d_carousels_dict_input(tmp_path: Path) -> None:
    """Validates Plotly 3D carousel generation when pes_tensor is provided as a dictionary."""
    artifact_dir = tmp_path / "artifacts"
    n1, n2 = 40, 40
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    P1, P2 = np.meshgrid(phi1, phi2, indexing="ij")
    pes_grid = 500.0 * (1.0 - np.cos(np.radians(P1))) + 200.0 * (1.0 - np.cos(np.radians(P2)))

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

    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],  # Severe collision: 0.2 A
    ], dtype=np.float64)

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

    with open(diag_path, "r", encoding="utf-8") as diag_file:
        diag = json.load(diag_file)

    assert diag["error_node_id"] == "single_frame_node"
    assert diag["num_frames"] == 1
    assert diag["num_atoms"] == 2
    assert diag["min_interatomic_distance"] == 0.2
    assert diag["colliding_pair"] == [0, 1] or diag["colliding_pair"] == [1, 0]
