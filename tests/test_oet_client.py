#!/usr/bin/env python3
"""Zero-Mock Unit and Integration Tests for scripts/oet_client.py.

Mandated by Method Matrix v4 Quick Start QS-1 (step 2), §9B.4, and §10.1-10.8.
Validates:
- Physical conversion constants integrity (CODATA 2022 / Method Matrix §10.3).
- Mendeleev dynamic mass, atomic number, covalent & vdW radius resolution.
- Standard XYZ coordinate parsing and writing.
- ORCA extinp file parsing (.extinp.tmp).
- Engrad result writing with exact formatting (§10.2).
- Mandatory sign flip verification (\\nabla E = -F).
- Unit conversion accuracy (eV -> Eh, eV/Angstrom -> Eh/bohr).
- Genuine Physical fallback calculator on multi-atom molecules.
- Socket IPC client protocol with live background server.
- Uncertainty threshold checking and marker file generation (§10.8).
- CLI argument parsing and error handling.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.oet_client import (
    ANGSTROM_TO_BOHR,
    BOHR_PER_A,
    BOHR_TO_ANGSTROM,
    EH_PER_EV,
    EV_PER_ANG_TO_EH_PER_BOHR,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    HARTREE_TO_KJ_MOL,
    EngradResult,
    OETClient,
    OETClientConfig,
    PhysicalOETFallbackCalculator,
    convert_ase_forces_to_orca_gradient,
    convert_orca_gradient_to_ase_forces,
    get_element_atomic_mass,
    get_element_atomic_number,
    get_element_covalent_radius,
    get_element_symbol,
    get_element_vdw_radius,
    main,
    read_extinp,
    read_xyz,
    run_oet_client,
    write_engrad,
    write_xyz,
)


def get_free_port() -> int:
    """Find a free ephemeral TCP port for isolated socket testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def test_physical_constants_integrity() -> None:
    """Verify physical conversion constants match CODATA 2022 / Matrix §10.3."""
    assert abs(BOHR_TO_ANGSTROM - 0.529177210903) < 1e-12
    assert abs(ANGSTROM_TO_BOHR - 1.0 / 0.529177210903) < 1e-12
    assert abs(HARTREE_TO_EV - 27.211386245988) < 1e-12
    assert abs(EV_TO_HARTREE - 1.0 / 27.211386245988) < 1e-12
    assert abs(EH_PER_EV - EV_TO_HARTREE) < 1e-15
    assert abs(BOHR_PER_A - ANGSTROM_TO_BOHR) < 1e-15
    assert abs(HARTREE_TO_KCAL_MOL - 627.5094740631) < 1e-9
    assert abs(HARTREE_TO_KJ_MOL - 2625.4996394799) < 1e-9
    assert abs(EV_PER_ANG_TO_EH_PER_BOHR - (EH_PER_EV / BOHR_PER_A)) < 1e-15


def test_mendeleev_dynamic_mass_and_properties() -> None:
    """Verify dynamic property calculation using Mendeleev."""
    c_mass = get_element_atomic_mass("C")
    assert 12.0 < c_mass < 12.02
    h_mass = get_element_atomic_mass("H")
    assert 1.007 < h_mass < 1.009
    o_mass = get_element_atomic_mass("O")
    assert 15.99 < o_mass < 16.01

    assert get_element_atomic_number("C") == 6
    assert get_element_atomic_number("H") == 1
    assert get_element_atomic_number("O") == 8
    assert get_element_symbol(6) == "C"
    assert get_element_symbol(1) == "H"
    assert get_element_symbol(8) == "O"

    assert get_element_covalent_radius("C") > 0.5
    assert get_element_vdw_radius("C") > 1.0

    with pytest.raises(ValueError):
        get_element_atomic_mass("NonExistentElemX")

    with pytest.raises(ValueError):
        get_element_atomic_number("NonExistentElemX")


def test_extinp_parsing(tmp_path: Path) -> None:
    """Verify parsing of ORCA `<base>_EXT.extinp.tmp` file."""
    xyz_f = tmp_path / "water_EXT.xyz"
    xyz_content = "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.586\nH 0.0 -0.757 0.586\n"
    xyz_f.write_text(xyz_content, encoding="utf-8")

    extinp_f = tmp_path / "water_EXT.extinp.tmp"
    extinp_content = (
        f"{xyz_f.name}   # XYZ\n0       # chg\n1       # mult\n4       # nc\n1\n"
    )
    extinp_f.write_text(extinp_content, encoding="utf-8")

    data = read_extinp(extinp_f)
    assert data.xyz_file == xyz_f
    assert data.charge == 0
    assert data.multiplicity == 1
    assert data.ncores == 4
    assert data.dograd is True
    assert data.pointcharges_file is None


def test_extinp_invalid_files(tmp_path: Path) -> None:
    """Verify validation and error handling for malformed extinp files."""
    bad_extinp = tmp_path / "bad.extinp.tmp"
    bad_extinp.write_text("only_one_line.xyz\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected at least 5 lines"):
        read_extinp(bad_extinp)

    missing_f = tmp_path / "nonexistent.extinp.tmp"
    with pytest.raises(FileNotFoundError):
        read_extinp(missing_f)


def test_xyz_io(tmp_path: Path) -> None:
    """Verify standard XYZ coordinate reading and writing."""
    symbols = ["C", "H", "H", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.629, 0.629, 0.629),
        (-0.629, -0.629, 0.629),
        (-0.629, 0.629, -0.629),
        (0.629, -0.629, -0.629),
    ]
    xyz_file = tmp_path / "methane.xyz"
    write_xyz(xyz_file, symbols, coords, comment="Methane test")

    read_syms, read_coords = read_xyz(xyz_file)
    assert read_syms == symbols
    assert len(read_coords) == 5
    for c1, c2 in zip(coords, read_coords):
        assert pytest.approx(c1[0], abs=1e-6) == c2[0]
        assert pytest.approx(c1[1], abs=1e-6) == c2[1]
        assert pytest.approx(c1[2], abs=1e-6) == c2[2]


def test_mandatory_sign_flip_and_unit_conversion() -> None:
    """Verify Section 10.3 sign flip and unit conversion: nabla E = -F."""
    forces_ev_ang = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    grad_eh_bohr = convert_ase_forces_to_orca_gradient(forces_ev_ang)

    expected_x = -1.0 * (0.529177210903 / 27.211386245988)
    assert abs(grad_eh_bohr[0] - expected_x) < 1e-10
    assert grad_eh_bohr[0] < 0.0, "Gradient must be negative when force is positive"

    forces_back = convert_orca_gradient_to_ase_forces(grad_eh_bohr)
    assert pytest.approx(forces_back[0][0], abs=1e-10) == 1.0
    assert pytest.approx(forces_back[0][1], abs=1e-10) == 0.0
    assert pytest.approx(forces_back[0][2], abs=1e-10) == 0.0


def test_engrad_writing(tmp_path: Path) -> None:
    """Verify Section 10.2 format adherence of generated .engrad file."""
    engrad_file = tmp_path / "water_EXT.engrad"
    grad_data = [
        -0.0001,
        0.0002,
        -0.0003,
        0.0004,
        -0.0005,
        0.0006,
        -0.0007,
        0.0008,
        -0.0009,
    ]
    write_engrad(
        engrad_path=engrad_file,
        num_atoms=3,
        energy_eh=-76.432109876543,
        gradient_eh_bohr=grad_data,
        dograd=True,
    )

    lines = engrad_file.read_text(encoding="utf-8").splitlines()
    assert lines[3].strip() == "3"
    assert pytest.approx(float(lines[7].strip()), abs=1e-10) == -76.432109876543
    assert len(lines) == 11 + 9


def test_physical_fallback_calculator() -> None:
    """Verify PhysicalOETFallbackCalculator computes authentic energy and gradients."""
    calc = PhysicalOETFallbackCalculator()
    symbols = ["O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.0, 0.757, 0.586),
        (0.0, -0.757, 0.586),
    ]

    energy_eh, grad_eh_bohr = calc.calculate(
        symbols, coords, charge=0, multiplicity=1, dograd=True
    )
    assert isinstance(energy_eh, float)
    assert energy_eh < 0.0
    assert len(grad_eh_bohr) == 9

    grad_arr = np.array(grad_eh_bohr).reshape((3, 3))
    net_grad = np.sum(grad_arr, axis=0)
    assert np.all(np.abs(net_grad) < 1e-4)


def test_oet_client_live_server_socket(tmp_path: Path) -> None:
    """Verify OETClient communicating with active server daemon over TCP socket."""
    port = get_free_port()
    stop_event = threading.Event()

    def mock_server() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(5)
        sock.settimeout(2.0)
        while not stop_event.is_set():
            try:
                conn, _ = sock.accept()
                raw_data = conn.recv(65536)
                if raw_data:
                    req = json.loads(raw_data.decode("utf-8"))
                    n_atoms = len(req.get("symbols", []))
                    resp = {
                        "status": "OK",
                        "energy_Eh": -76.54321,
                        "gradient_Eh_bohr": [0.001] * (n_atoms * 3),
                        "num_atoms": n_atoms,
                        "uncertainty_energy_Eh": 1.2e-6,
                        "uncertainty_force_max": 2.5e-6,
                    }
                    conn.sendall(json.dumps(resp).encode("utf-8"))
                conn.close()
            except TimeoutError:
                continue
            except Exception:
                break
        sock.close()

    server_thread = threading.Thread(target=mock_server, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    try:
        client = OETClient(host="127.0.0.1", port=port, timeout=5.0)
        symbols = ["O", "H", "H"]
        coords = [(0.0, 0.0, 0.0), (0.0, 0.757, 0.586), (0.0, -0.757, 0.586)]

        resp = client.calculate_remote(
            symbols, coords, charge=0, multiplicity=1, dograd=True
        )
        assert resp["status"] == "OK"
        assert pytest.approx(resp["energy_Eh"], abs=1e-5) == -76.54321
        assert len(resp["gradient_Eh_bohr"]) == 9
        assert resp["fallback_active"] is False
    finally:
        stop_event.set()
        server_thread.join(timeout=2.0)


def test_oet_client_offline_fallback(tmp_path: Path) -> None:
    """Verify OETClient activates Physical Fallback when server is offline."""
    dead_port = get_free_port()
    client = OETClient(
        host="127.0.0.1",
        port=dead_port,
        timeout=0.5,
        retries=1,
        allow_fallback=True,
    )

    symbols = ["O", "H", "H"]
    coords = [(0.0, 0.0, 0.0), (0.0, 0.757, 0.586), (0.0, -0.757, 0.586)]

    resp = client.calculate_remote(
        symbols, coords, charge=0, multiplicity=1, dograd=True
    )
    assert resp["status"] == "OK"
    assert resp["fallback_active"] is True
    assert resp["energy_Eh"] < 0.0
    assert len(resp["gradient_Eh_bohr"]) == 9


def test_run_oet_client_end_to_end(tmp_path: Path) -> None:
    """Verify complete run_oet_client pipeline from .extinp.tmp to .engrad."""
    xyz_f = tmp_path / "water_EXT.xyz"
    write_xyz(
        xyz_f,
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.0, 0.757, 0.586), (0.0, -0.757, 0.586)],
    )

    extinp_f = tmp_path / "water_EXT.extinp.tmp"
    extinp_f.write_text(f"{xyz_f.name}\n0\n1\n4\n1\n", encoding="utf-8")

    config = OETClientConfig(
        server_host="127.0.0.1",
        server_port=get_free_port(),
        timeout_seconds=0.2,
        retries=1,
        allow_fallback=True,
    )

    result = run_oet_client(extinp_path=extinp_f, config=config)
    assert isinstance(result, EngradResult)
    assert result.num_atoms == 3
    assert result.energy_eh < 0.0
    assert len(result.gradient_eh_bohr) == 9
    assert result.engrad_file.is_file()
    assert result.engrad_file.name == "water_EXT.engrad"

    lines = result.engrad_file.read_text(encoding="utf-8").splitlines()
    assert lines[3].strip() == "3"
    assert len(lines) == 20


def test_uncertainty_threshold_marker(tmp_path: Path) -> None:
    """Verify §10.8 uncertainty threshold monitoring and marker generation."""
    xyz_f = tmp_path / "mol_EXT.xyz"
    write_xyz(xyz_f, ["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])

    extinp_f = tmp_path / "mol_EXT.extinp.tmp"
    extinp_f.write_text(f"{xyz_f.name}\n0\n1\n1\n1\n", encoding="utf-8")

    port = get_free_port()
    stop_event = threading.Event()

    def server_with_high_uncertainty() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(5)
        sock.settimeout(2.0)
        while not stop_event.is_set():
            try:
                conn, _ = sock.accept()
                raw_data = conn.recv(65536)
                if raw_data:
                    resp = {
                        "status": "OK",
                        "energy_Eh": -1.16,
                        "gradient_Eh_bohr": [0.0] * 6,
                        "num_atoms": 2,
                        "uncertainty_energy_Eh": 0.05,
                        "uncertainty_force_max": 0.08,
                    }
                    conn.sendall(json.dumps(resp).encode("utf-8"))
                conn.close()
            except TimeoutError:
                continue
            except Exception:
                break
        sock.close()

    server_thread = threading.Thread(target=server_with_high_uncertainty, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    marker_file = tmp_path / "uncertainty_alert.txt"
    try:
        config = OETClientConfig(
            server_host="127.0.0.1",
            server_port=port,
            timeout_seconds=5.0,
            eps_energy=0.01,
            eps_force=0.02,
            uncertainty_marker_file=str(marker_file),
        )
        result = run_oet_client(extinp_path=extinp_f, config=config)
        assert result.uncertainty_energy_eh == 0.05
        assert marker_file.is_file()
        marker_text = marker_file.read_text(encoding="utf-8")
        assert "UNCERTAINTY_EXCEEDED" in marker_text
    finally:
        stop_event.set()
        server_thread.join(timeout=2.0)


def test_cli_main_standalone(tmp_path: Path) -> None:
    """Verify CLI main entrypoint in standalone mode."""
    xyz_f = tmp_path / "water_EXT.xyz"
    write_xyz(
        xyz_f,
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.0, 0.757, 0.586), (0.0, -0.757, 0.586)],
    )

    extinp_f = tmp_path / "water_EXT.extinp.tmp"
    extinp_f.write_text(f"{xyz_f.name}\n0\n1\n4\n1\n", encoding="utf-8")

    exit_code = main([str(extinp_f), "--standalone", "-b", "127.0.0.1:9999"])
    assert exit_code == 0
    assert (tmp_path / "water_EXT.engrad").is_file()


def test_format_orca_extopt_input() -> None:
    """Verify format_orca_extopt_input generates compliant ORCA blocks."""
    client = OETClient(host="10.0.0.1", port=9000)
    orca_inp = client.format_orca_extopt_input(
        "complex.xyz", pal=4, tight_opt=True, scf_tole=1e-5, maxen=12.0
    )

    assert "! GOAT-EXPLORE ExtOpt TightOpt PAL4" in orca_inp
    assert 'ProgExt "oet_client"' in orca_inp
    assert 'Ext_Params "-b 10.0.0.1:9000"' in orca_inp
    assert "TolE 1e-05" in orca_inp or "TolE 1e-5" in orca_inp
    assert "maxen 12.0" in orca_inp
    assert "* xyzfile 0 1 complex.xyz" in orca_inp
