"""Actual ORCA input contracts, tabulated properties and unavailable-engine CLI.

Live positive OET engine qualification requires a genuine separately provisioned
server; these checks do not create one or invent electronic observations.
"""

import socket
from pathlib import Path

import numpy as np
import pytest
from mendeleev import element

from Libraries.cochem_isotopes import isotope_mass
from Libraries.cochem_torq_engine import _read_orca_engrad
from scripts.oet_client import (
    OETClientConfig,
    OETDaemonUnavailableError,
    convert_ase_forces_to_orca_gradient,
    convert_orca_gradient_to_ase_forces,
    get_element_atomic_mass,
    get_element_covalent_radius,
    get_element_vdw_radius,
    main,
    read_extinp,
    read_xyz,
    run_oet_client,
    write_xyz,
)

REPO = Path(__file__).resolve().parents[1]


def actual_input_files(directory):
    xyz = directory / "water_EXT.xyz"
    xyz.write_text(
        "3\nStarting input\nO 0 0 0\nH 0.9572 0 0\nH -0.2399872 0.927297 0\n"
    )
    contract = directory / "water_EXT.extinp.tmp"
    contract.write_text(f"{xyz.name}\n0\n1\n1\n1\n")
    return xyz, contract


def test_input_xyz_and_explicit_external_state_round_trip(tmp_path):
    xyz, contract = actual_input_files(tmp_path)
    parsed = read_extinp(contract)
    assert parsed.xyz_file == xyz
    assert (parsed.charge, parsed.multiplicity, parsed.ncores, parsed.dograd) == (
        0,
        1,
        1,
        True,
    )
    symbols, positions = read_xyz(xyz)
    destination = write_xyz(tmp_path / "copy.xyz", symbols, positions)
    assert read_xyz(destination) == (symbols, positions)


@pytest.mark.parametrize("state", ["0\n1\n0\n1", "0\n1\n1\n2", "0\n0\n1\n1"])
def test_invalid_external_settings_are_not_changed(tmp_path, state):
    xyz, contract = actual_input_files(tmp_path)
    contract.write_text(f"{xyz.name}\n{state}\n")
    with pytest.raises(ValueError):
        read_extinp(contract)


def test_requested_point_charges_cannot_silently_disappear(tmp_path):
    _, contract = actual_input_files(tmp_path)
    with contract.open("a") as output:
        output.write("missing.pointcharges\n")
    with pytest.raises(FileNotFoundError, match="point-charge"):
        read_extinp(contract)


@pytest.mark.parametrize("mutation", ["nonfinite", "extra-frame", "empty"])
def test_invalid_xyz_is_rejected_without_substitution(tmp_path, mutation):
    xyz, _ = actual_input_files(tmp_path)
    if mutation == "nonfinite":
        xyz.write_text(xyz.read_text().replace("0.9572", "nan"))
    elif mutation == "extra-frame":
        xyz.write_text(xyz.read_text() + xyz.read_text())
    else:
        xyz.write_text("0\nNo atoms\n")
    with pytest.raises(ValueError):
        read_xyz(xyz)


def test_properties_match_actual_named_database_values():
    assert get_element_atomic_mass("C") == isotope_mass("C") == 12.0
    assert get_element_atomic_mass("13C") == isotope_mass("13C")
    assert (
        get_element_covalent_radius("C")
        == float(element("C").covalent_radius_pyykko) / 100
    )
    assert get_element_vdw_radius("C") == float(element("C").vdw_radius) / 100


def test_native_file_derivative_sign_and_units_round_trip():
    _, gradient, _, _ = _read_orca_engrad(REPO / "test.engrad", 10)
    forces = convert_orca_gradient_to_ase_forces(gradient.reshape(-1))
    recovered = convert_ase_forces_to_orca_gradient(forces)
    np.testing.assert_allclose(recovered, gradient.reshape(-1), rtol=2e-16, atol=1e-16)
    assert np.sign(forces[0][0]) == -np.sign(gradient[0, 0])


def test_actual_cli_connection_failure_preserves_input_and_writes_no_energy(tmp_path):
    xyz, contract = actual_input_files(tmp_path)
    input_bytes = (xyz.read_bytes(), contract.read_bytes())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
        cfg = OETClientConfig(server_port=port, timeout_seconds=0.2, retries=1)
        with pytest.raises(OETDaemonUnavailableError):
            run_oet_client(contract, config=cfg)
        status = main(
            [str(contract), "--port", str(port), "--retries", "1", "--timeout", "0.2"]
        )
    assert status != 0
    assert (xyz.read_bytes(), contract.read_bytes()) == input_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [xyz.name, contract.name]
    )
