"""Real transport/file/render checks and explicitly mathematical analytic models.

The periodic and Lennard-Jones examples are defined mathematical test potentials,
not quantum-chemistry outputs, experimental observations, optimized molecules or
method-qualification evidence. LJ energies/gradients are converted explicitly
from kcal/mol and Å to hartree and hartree/bohr for the diagnostic API. Local HTTP
checks execute actual servers, requests, failure responses and filesystem writes.
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
from scipy.constants import Avogadro, physical_constants

from Libraries.cochem_torq_telemetry import (
    TelemetryCircuitBreaker,
    decimate_2d_grid_with_extrema,
    export_crash_animation,
    find_stationary_points_2d,
    generate_plotly_3d_carousels,
    stream_webhook_events,
)

# ============================================================================
# Transport helper: ephemeral local HTTP server
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
    """Start an actual HTTP server on localhost; no calculation is executed."""
    server = HTTPServer(("127.0.0.1", 0), WebhookRecordingHandler)
    port = server.server_address[1]
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
    """Validate actual HTTP POST delivery without a scientific calculation claim."""
    server, webhook_url = local_webhook_server
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "transport_check",
        "job_id": "LOCAL_HTTP_TRANSPORT_CHECK",
        "node_id": "localhost",
        "status": "TRANSPORT_ONLY",
        "data": {"calculation_executed": False},
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
    received = server.received_requests[0]["payload"]  # type: ignore[attr-defined]
    assert {key: received[key] for key in test_payload} == test_payload
    assert received["error_trace"] is None
    from datetime import datetime

    assert datetime.fromisoformat(received["timestamp"]).tzinfo is not None


def test_stream_webhook_events_exponential_backoff_recovery(
    local_webhook_server: tuple[HTTPServer, str], tmp_path: Path
) -> None:
    """Validates exponential backoff retries when encountering transient 503 errors."""
    server, webhook_url = local_webhook_server
    server.fail_count_target = 2  # type: ignore[attr-defined] # Fail first 2 attempts with 503, succeed on 3rd
    scratch_dir = tmp_path / "scratch"

    test_payload = {
        "event_type": "transport_retry_check",
        "job_id": "LOCAL_HTTP_RETRY_CHECK",
        "node_id": "localhost",
        "status": "TRANSPORT_ONLY",
        "data": {"calculation_executed": False, "requested_http_failures": 2},
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
        "event_type": "transport_connection_failure_check",
        "job_id": "LOCAL_HTTP_CONNECTION_FAILURE_CHECK",
        "node_id": "localhost",
        "status": "TRANSPORT_ONLY",
        "data": {"calculation_executed": False},
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
    assert logged_event["payload"]["job_id"] == test_payload["job_id"]
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
    # Sample a declared analytic periodic potential; no molecular engine data.
    n1, n2 = 500, 500
    phi1 = np.linspace(-180.0, 180.0, n1)
    phi2 = np.linspace(-180.0, 180.0, n2)
    p1_mesh, p2_mesh = np.meshgrid(phi1, phi2, indexing="ij")

    # Analytical potential:
    # V(phi1, phi2) = 1500*(1-cos(phi1)) + 800*(1-cos(2*phi2)) + 400*cos(phi1+phi2)
    # Global lower bound -400 occurs at (0, ±pi); (0,0) is another local minimum.
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
    # Deliberately constructed array minimum for masked-data algorithm checking.
    pes_grid[25, 25] = 50.0
    # NaNs explicitly mark unavailable array entries; no collision is inferred.
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
        title="Analytic periodic test potential (no molecular calculation)",
    )

    assert html_path.exists()
    assert html_path.is_file()
    assert html_path.parent == artifact_dir

    # Inspect HTML content
    html_content = html_path.read_text(encoding="utf-8")
    assert "<html>" in html_content.lower()
    assert "<body>" in html_content.lower()
    assert "plotly" in html_content.lower()
    assert "Analytic periodic test potential (no molecular calculation)" in html_content

    # File size must be lightweight (< 3.5 MB)
    file_size_mb = html_path.stat().st_size / (1024 * 1024)
    assert file_size_mb < 3.5


def test_generate_plotly_3d_carousels_with_analytic_free_rotor_densities(
    tmp_path: Path,
) -> None:
    """Render exact mathematical periodic free-rotor densities, not DVR output."""
    artifact_dir = tmp_path / "artifacts"
    n = 100
    phi1 = np.linspace(-180.0, 180.0, n, endpoint=False)
    phi2 = phi1.copy()
    angle1, angle2 = np.meshgrid(np.radians(phi1), np.radians(phi2), indexing="ij")
    # H = -d²/dphi1²-d²/dphi2² on a periodic torus; potential identically zero.
    # The constant, cos(phi1), and cos(phi2) are exact eigenfunctions, including
    # legitimate real linear combinations within the degenerate excited space.
    cell_area = (2 * np.pi / n) ** 2
    probabilities = []
    for eigenfunction in (np.ones_like(angle1), np.cos(angle1), np.cos(angle2)):
        density = eigenfunction**2
        density /= density.sum() * cell_area
        assert density.sum() * cell_area == pytest.approx(1.0, abs=1e-12)
        probabilities.append(density)
    html_path = generate_plotly_3d_carousels(
        pes_tensor=np.zeros((n, n)),
        dvr_wavefunctions=probabilities,
        phi1_grid=phi1,
        phi2_grid=phi2,
        artifact_dir=artifact_dir,
        filename="analytic_free_rotor_densities.html",
        max_nodes=2500,
        title="Analytic free-rotor densities: mathematical rendering check",
    )
    html_content = html_path.read_text()
    for index in range(3):
        assert f"Supplied density {index}" in html_content
    assert "DVR State" not in html_content


def test_generate_plotly_3d_carousels_rejects_missing_coords(tmp_path: Path) -> None:
    """An array does not establish unprovided physical angular coordinates."""
    phi = np.linspace(-180.0, 180.0, 30)
    a, b = np.meshgrid(phi, phi, indexing="ij")
    potential = 10 * (1 - np.cos(np.radians(a))) + 10 * (1 - np.cos(np.radians(b)))
    with pytest.raises(ValueError, match="Actual angle grids"):
        generate_plotly_3d_carousels(
            pes_tensor={"pes": potential}, artifact_dir=tmp_path
        )
    assert not list(tmp_path.glob("*.html"))


# ============================================================================
# Test Suite 3: Crash Animation & Diagnostic Exporter
# ============================================================================


KCAL_MOL_TO_HARTREE = 4184.0 / (Avogadro * physical_constants["Hartree energy"][0])
ANGSTROM_PER_BOHR = physical_constants["Bohr radius"][0] / 1e-10


def _analytic_lj_pair(coordinates_angstrom: np.ndarray) -> tuple[float, np.ndarray]:
    """Defined mathematical LJ pair model; no fitted atomistic parameter claim.

    sigma=1.1 Å and epsilon=0.1 kcal/mol are explicit model-definition inputs.
    The energy origin is the standard V(r→infinity)=0; no quantum energy offset.
    """
    sigma_angstrom, epsilon_kcal_mol = 1.1, 0.1
    delta = coordinates_angstrom[0] - coordinates_angstrom[1]
    radius = float(np.linalg.norm(delta))
    if radius <= 0:
        raise ValueError("The mathematical LJ potential is singular at r=0.")
    ratio6 = (sigma_angstrom / radius) ** 6
    energy_kcal_mol = 4 * epsilon_kcal_mol * (ratio6**2 - ratio6)
    derivative_kcal_mol_angstrom = (
        24 * epsilon_kcal_mol * (ratio6 - 2 * ratio6**2) / radius
    )
    first = derivative_kcal_mol_angstrom * delta / radius
    gradient = np.asarray([first, -first]) * KCAL_MOL_TO_HARTREE * ANGSTROM_PER_BOHR
    return energy_kcal_mol * KCAL_MOL_TO_HARTREE, gradient


def test_analytic_lj_units_and_gradient_sign():
    """Independent finite differences check the analytic mathematical derivative."""
    geometry = np.asarray([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0]])
    _, gradient = _analytic_lj_pair(geometry)
    step_angstrom = 1e-6
    for atom in range(2):
        for axis in range(3):
            plus, minus = geometry.copy(), geometry.copy()
            plus[atom, axis] += step_angstrom
            minus[atom, axis] -= step_angstrom
            numerical = (_analytic_lj_pair(plus)[0] - _analytic_lj_pair(minus)[0]) / (
                2 * step_angstrom
            )
            assert gradient[atom, axis] == pytest.approx(
                numerical * ANGSTROM_PER_BOHR, abs=1e-10
            )
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-15)


def test_export_analytic_lj_collision_trajectory(tmp_path: Path) -> None:
    """Real file export of a defined mathematical model, never engine evidence."""
    frames = np.asarray(
        [[[0.0, 0.0, 0.0], [radius, 0.0, 0.0]] for radius in np.linspace(2.0, 0.3, 12)]
    )
    pairs = [_analytic_lj_pair(frame) for frame in frames]
    energies = [item[0] for item in pairs]
    gradients = [item[1] for item in pairs]
    result_paths = export_crash_animation(
        trajectory_array=frames,
        error_node_id="analytic_lj_pair",
        symbols=["H", "H"],
        energies=energies,
        gradients=gradients,
        artifact_dir=tmp_path / "artifacts",
        scratch_dir=tmp_path / "scratch",
        abort_reason="Analytic mathematical LJ pair diagnostic; no molecular/quantum calculation or optimization",
    )
    xyz = result_paths["xyz_path"].read_text().splitlines()
    assert len(xyz) == 12 * (2 + 2)
    assert "analytic_lj_pair" in xyz[1]
    diagnostic = json.loads(result_paths["diagnostic_path"].read_text())
    assert diagnostic["num_frames"] == 12 and diagnostic["num_atoms"] == 2
    assert diagnostic["symbols"] == ["H", "H"]
    assert diagnostic["min_interatomic_distance"] == pytest.approx(0.3)
    assert diagnostic["colliding_pair"] == [0, 1]
    assert diagnostic["initial_energy_hartree"] == pytest.approx(energies[0])
    assert diagnostic["final_energy_hartree"] == pytest.approx(energies[-1])
    assert "mathematical LJ" in diagnostic["abort_reason"]
    # Full coordinate precision survives the actual exported XYZ artifact.
    assert float(xyz[-1].split()[1]) == frames[-1, 1, 0]


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
        phi1_grid=phi,
        phi2_grid=phi,
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
        # No energy/gradient was evaluated for this coordinate-format test.
    }

    result = export_crash_animation(
        trajectory_array=traj_dict,
        error_node_id="single_frame_node",
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        abort_reason="Coordinate-format mathematical check; no physical calculation",
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


@pytest.mark.parametrize(
    "bad_density",
    [
        np.ones((4, 4)),
        -np.ones((10, 10)),
        np.full((10, 10), np.nan),
        np.zeros((10, 10)),
    ],
)
def test_render_rejects_invalid_density_without_resize_or_sign_repair(
    tmp_path: Path, bad_density
):
    phi = np.linspace(-180.0, 180.0, 10)
    with pytest.raises(ValueError, match="Supplied probability density"):
        generate_plotly_3d_carousels(
            np.zeros((10, 10)),
            dvr_wavefunctions=[bad_density],
            phi1_grid=phi,
            phi2_grid=phi,
            artifact_dir=tmp_path,
        )
    assert not list(tmp_path.glob("*.html"))
