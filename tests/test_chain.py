"""Actual isolated BASE handoff and genuine spectroscopy migration.

No ORCA execution or eleven-arrow method-matrix qualification is asserted.
Static geometry arithmetic and genuine engine Hessians have separate scopes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import atomic_mass, h, physical_constants, pi

from cochem_torq.domain import PrerequisiteError, read_json
from cochem_torq.spectroscopy.harmonic import analyze_hessian, equilibrium_rotor
from Libraries.chain import BaseHandoffBridge, LegacyChainUnavailableError
from Libraries.cochem_isotopes import isotope_mass, isotope_record

ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_load_sibling_namespace():
    program = (
        "import sys; import Libraries.chain; "
        "assert not any(name=='cochem_base' or name.startswith('cochem_base.') "
        "for name in sys.modules)"
    )
    process = subprocess.run(
        [sys.executable, "-B", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize(
    "name",
    [
        "Chain",
        "evaluate_vdw_potential_and_derivatives",
        "analyze_hessian_and_normal_modes",
    ],
)
def test_old_in_process_and_invented_physics_apis_are_explicitly_unavailable(name):
    import Libraries.chain as chain

    with pytest.raises(LegacyChainUnavailableError, match="isolated environment"):
        getattr(chain, name)


def test_legacy_script_cannot_emit_force_fallback_success(tmp_path):
    process = subprocess.run(
        [sys.executable, "-B", str(ROOT / "Libraries/chain.py"), "--force-fallback"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=30,
    )
    assert process.returncode != 0
    assert "unsupported" in process.stderr
    assert "SUCCESS" not in process.stdout
    assert not list(tmp_path.iterdir())


def test_isotope_mass_policy_is_explicit_and_reproducible():
    hydrogen, deuterium = isotope_record("1H"), isotope_record("2H")
    assert hydrogen["mass_number"] == 1 and deuterium["mass_number"] == 2
    assert hydrogen["selection_policy"] == "explicit_mass_number"
    assert (
        hydrogen["source"]["database_sha256"] == deuterium["source"]["database_sha256"]
    )
    assert isotope_mass("2H") > isotope_mass("1H")
    assert isotope_mass("13C") > isotope_mass("12C")


def test_explicit_diatomic_inertia_matches_independent_formula_and_isotope_shift():
    length_bohr = 1.4
    coordinates = np.array([[0.0, 0.0, -length_bohr / 2], [0.0, 0.0, length_bohr / 2]])
    mass = isotope_mass("1H")
    rotor = equilibrium_rotor(coordinates, [mass, mass])
    expected_moment = mass * length_bohr**2 / 2
    bohr = physical_constants["Bohr radius"][0]
    expected_mhz = h / (8 * pi**2 * atomic_mass * bohr**2 * expected_moment * 1e6)
    assert rotor.rotor_type == "linear" and rotor.constants_mhz[0] is None
    np.testing.assert_allclose(rotor.constants_mhz[1:], expected_mhz, rtol=1e-12)
    heavier = equilibrium_rotor(coordinates, [isotope_mass("2H")] * 2)
    assert heavier.constants_mhz[1] < rotor.constants_mhz[1]
    assert rotor.observable == "Be"


def test_missing_provider_fails_before_any_execution(tmp_path):
    bridge = BaseHandoffBridge(
        tmp_path / "missing-modules", ROOT / "ci_tools/ecosystem-modules.json"
    )
    with pytest.raises(PrerequisiteError, match="missing or changed"):
        bridge.verify(tmp_path / "new-package")
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def isolated_base_bridge():
    configured = os.environ.get("COCHEM_MODULE_ROOT")
    if not configured:
        pytest.fail(
            "Set COCHEM_MODULE_ROOT to a real verified isolated BASE installation; "
            "this integration cannot be qualified without it."
        )
    return BaseHandoffBridge(configured)


@pytest.mark.ecosystem
def test_actual_base_producer_runs_isolated_and_preserves_pending_geometry(
    isolated_base_bridge, tmp_path
):
    seed = tmp_path / "input.xyz"
    seed.write_text(
        "2\nExplicit geometry only; no calculated energy\nH 0 0 -0.37\nH 0 0 0.37\n"
    )
    package = tmp_path / "actual-base-package"
    record = isolated_base_bridge.prepare_geometry(
        seed,
        package,
        molecule_id="geometry-h2",
        charge=0,
        multiplicity=1,
        atom_ids=["h-a", "h-b"],
    )
    observed = read_json(package / "handoff.json")
    assert observed["scientific_execution_performed"] is False
    assert observed["status"] == "pending_integration"
    assert record.energy is None and record.source_convergence == "unknown"
    assert record.molecule.charge == 0 and record.molecule.multiplicity == 1
    assert [atom.atom_id for atom in record.molecule.atoms] == ["h-a", "h-b"]
    assert observed["options"]["atoms"][0]["atom_id"] == "h-a"
    assert (
        record.source.repository_revision == "1a3c633f6cb0c6256223298ed95d71196ffc3689"
    )
    assert read_json(package / "torq-conformer.json")["energy"] is None
    assert (package / "artifact.xyz").read_bytes() == seed.read_bytes()
    with pytest.raises(FileExistsError):
        isolated_base_bridge.prepare_geometry(
            seed,
            package,
            molecule_id="geometry-h2",
            charge=0,
            multiplicity=1,
            atom_ids=["h-a", "h-b"],
        )


@pytest.mark.ecosystem
def test_invalid_explicit_state_never_starts_provider(isolated_base_bridge, tmp_path):
    seed = tmp_path / "input.xyz"
    seed.write_text("2\nGeometry only\nH 0 0 -0.37\nH 0 0 0.37\n")
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="parity"):
        isolated_base_bridge.prepare_geometry(
            seed,
            tmp_path / "bad-state",
            molecule_id="h2",
            charge=0,
            multiplicity=2,
            atom_ids=["a", "b"],
        )
    assert not (tmp_path / "bad-state").exists()


@pytest.fixture(scope="module")
def actual_water(tmp_path_factory):
    from cochem_torq.application import execute_request
    from cochem_torq.artifacts import verify_shard

    request = read_json(ROOT / "examples/student/water-hf-teaching.json")
    request["resources"]["cores"] = 1
    destination = tmp_path_factory.mktemp("actual-chain-migration") / "water"
    result = execute_request(request, destination)
    assert result["status"] == "complete", result["errors"]
    verify_shard(destination)
    return result


@pytest.mark.real_engine
def test_genuine_water_hessian_has_three_internal_modes_and_isotope_reanalysis(
    actual_water,
):
    native = actual_water["harmonic_native_result"]
    geometry = native["geometry_bohr"]
    masses = [isotope_mass("16O"), isotope_mass("1H"), isotope_mass("1H")]
    hessian = native["hessian_hartree_bohr2"]
    parent = analyze_hessian(geometry, masses, hessian)
    heavier = analyze_hessian(
        geometry, [masses[0], isotope_mass("2H"), isotope_mass("2H")], hessian
    )
    assert parent.external_rank == 6 and len(parent.frequencies_cm1) == 3
    assert parent.cartesian_modes.shape == (9, 3)
    assert parent.stationary_character == "positive_definite_vibrational_hessian"
    assert np.all(parent.frequencies_cm1 > 0)
    assert heavier.harmonic_zpe_hartree < parent.harmonic_zpe_hartree
    light_rotor = equilibrium_rotor(geometry, masses)
    heavy_rotor = equilibrium_rotor(
        geometry, [masses[0], isotope_mass("2H"), isotope_mass("2H")]
    )
    assert all(
        a < b
        for a, b in zip(
            heavy_rotor.constants_mhz, light_rotor.constants_mhz, strict=True
        )
    )
    assert actual_water["experimental_accuracy_established"] is False
    assert actual_water["identification_ready"] is False
