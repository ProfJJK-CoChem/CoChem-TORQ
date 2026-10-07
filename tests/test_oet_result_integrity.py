"""Real connection failure, native-file normalization and fail-closed routes.

No server or electronic-structure output is generated here. An existing native
ORCA file exercises parsing/serialization only; it does not qualify a live OET
daemon, an ML potential, or the chemistry of that historical calculation.
"""

import socket
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_torq_engine import _read_orca_engrad
from Libraries.cochem_torq_goat import (
    GoatConfig,
    GoatExecutionError,
    GoatExtOptDriver,
    GoatRunner,
)
from scripts.oet_client import (
    OETClient,
    OETDaemonUnavailableError,
    PhysicalOETFallbackCalculator,
    write_engrad,
)

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_derivative_record():
    energy, gradient, numbers, coordinates = _read_orca_engrad(
        REPOSITORY / "test.engrad", 10
    )
    return {
        "status": "OK",
        "energy_Eh": energy,
        "gradient_Eh_bohr": gradient.reshape(-1).tolist(),
        "num_atoms": len(numbers),
        "provenance": {"source": "repository-native-file", "path": "test.engrad"},
    }


def test_closed_real_socket_never_selects_physical_model(tmp_path):
    # Binding without listening keeps the exact loopback port unavailable.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bound:
        bound.bind(("127.0.0.1", 0))
        client = OETClient(
            host="127.0.0.1",
            port=bound.getsockname()[1],
            timeout=0.2,
            retries=1,
            allow_fallback=True,
            scratch_dir=tmp_path / "scratch",
            artifacts_dir=tmp_path / "artifacts",
        )
        with pytest.raises(OETDaemonUnavailableError, match="No replacement engine"):
            client.calculate_remote(["H", "H"], [(0, 0, 0), (0, 0, 0.74)])
    assert list(tmp_path.iterdir()) == []


def test_historical_standalone_and_calculator_routes_are_unavailable():
    with pytest.raises(OETDaemonUnavailableError, match="Standalone"):
        OETClient(standalone=True, allow_fallback=True).calculate_remote(
            ["H", "H"], [(0, 0, 0), (0, 0, 0.74)]
        )
    with pytest.raises(OETDaemonUnavailableError):
        PhysicalOETFallbackCalculator().calculate(["H", "H"], [(0, 0, 0), (0, 0, 0.74)])


def test_native_file_values_are_preserved_without_invented_uncertainty(
    native_derivative_record,
):
    normalized = OETClient()._normalize_server_response(
        native_derivative_record, n_atoms=10
    )
    assert normalized["energy_Eh"] == native_derivative_record["energy_Eh"]
    assert (
        normalized["gradient_Eh_bohr"] == native_derivative_record["gradient_Eh_bohr"]
    )
    assert normalized["uncertainty_energy_Eh"] is None
    assert normalized["uncertainty_force_max"] is None
    assert normalized["uncertainty_status"] == "unavailable"
    assert normalized["response_provenance"] == native_derivative_record["provenance"]


def test_missing_derivative_remains_unavailable(native_derivative_record, tmp_path):
    record = dict(native_derivative_record)
    record.pop("gradient_Eh_bohr")
    client = OETClient()
    with pytest.raises(ValueError, match="zero is not a substitute"):
        client._normalize_server_response(record, dograd=True, n_atoms=10)
    normalized = client._normalize_server_response(record, dograd=False, n_atoms=10)
    assert normalized["gradient_Eh_bohr"] is None
    with pytest.raises(ValueError, match="actual derivative"):
        write_engrad(
            tmp_path / "missing.engrad", 10, record["energy_Eh"], [], dograd=False
        )
    assert not (tmp_path / "missing.engrad").exists()


@pytest.mark.parametrize("mutation", ["units", "status", "nonfinite", "atom-count"])
def test_incompatible_native_record_variant_is_rejected(
    native_derivative_record, mutation
):
    record = dict(native_derivative_record)
    if mutation == "units":
        record["energy"] = record.pop("energy_Eh")
    elif mutation == "status":
        record.pop("status")
    elif mutation == "nonfinite":
        record["energy_Eh"] = float("nan")
    else:
        record["num_atoms"] = 9
    with pytest.raises((ValueError, RuntimeError)):
        OETClient()._normalize_server_response(record, n_atoms=10)


def test_native_derivatives_serialize_without_changed_values(
    native_derivative_record, tmp_path
):
    destination = write_engrad(
        tmp_path / "observed.engrad",
        10,
        native_derivative_record["energy_Eh"],
        native_derivative_record["gradient_Eh_bohr"],
    )
    numbers = [
        float(line)
        for line in destination.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert numbers[0] == 10
    assert numbers[1] == native_derivative_record["energy_Eh"]
    np.testing.assert_array_equal(
        numbers[2:], native_derivative_record["gradient_Eh_bohr"]
    )


def test_goat_cannot_activate_unqualified_physical_seed_route(tmp_path):
    runner = GoatRunner(GoatConfig(driver=GoatExtOptDriver.PHYSICAL))
    with pytest.raises(GoatExecutionError, match="scientifically unqualified"):
        runner._run_physical_seed_conformer_search([], tmp_path, "actual-input")
    assert list(tmp_path.iterdir()) == []
