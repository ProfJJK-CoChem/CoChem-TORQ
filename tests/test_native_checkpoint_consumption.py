"""Actual PySCF checkpoint consumption; genuine H2/water, no engine replacements."""

import shutil
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from pyscf import lib, scf

from cochem_torq.capabilities import _native_bundle
from cochem_torq.domain import canonical_json, read_json
from cochem_torq.engines.checkpoint_restart import (
    evaluate_with_checkpoint,
    prepare_checkpoint_density,
)
from cochem_torq.engines.pyscf_backend import PySCFBackend, _environment
from cochem_torq.registry import get_profile

pytestmark = pytest.mark.real_engine


@pytest.fixture(scope="module", params=["h2", "water"])
def native_source(request, tmp_path_factory):
    molecule = {
        "h2": {
            "symbols": ["H", "H"],
            "atom_ids": ["hydrogen-1", "hydrogen-2"],
            "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        },
        "water": {
            "symbols": ["O", "H", "H"],
            "atom_ids": ["oxygen", "hydrogen-1", "hydrogen-2"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 1.43, 1.1], [0.0, -1.43, 1.1]],
        },
    }[request.param]
    molecule.update(charge=0, multiplicity=1)
    native_request = {
        "molecule": molecule,
        "method": {"name": "hf", "basis": "sto-3g", "reference": "restricted"},
        "properties": ["energy", "gradient", "dipole"],
        "settings": {"threads": 1},
    }
    directory = tmp_path_factory.mktemp(f"cochem_exec_checkpoint_{request.param}")
    result = PySCFBackend().evaluate(native_request, directory)
    assert result["status"] == "complete", result["errors"]
    assert result["stability"]["status"] == "stable"
    return directory, read_json(directory / "request.json"), result


def recipe():
    return get_profile("hf-sto-3g-education")["recipe_sha256"]


def inputs(directory):
    return {
        path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file() and "scratch" not in path.relative_to(directory).parts
    }


def test_preparation_alone_never_claims_consumption(native_source, tmp_path):
    source, request, original = native_source
    before = inputs(source)
    prepared = prepare_checkpoint_density(
        request,
        tmp_path / "prepared",
        source_native_directory=source,
        recipe_sha256=recipe(),
        expected_source_manifest_sha256=original["manifest_sha256"],
    )
    try:
        evidence = prepared.consumption_evidence()
        assert not evidence["engine_checkpoint_reused"]
        assert not evidence["fresh_scf_kernel_called"]
        assert not evidence["final_result_reused"]
        assert evidence["startup_observations"] == []
        assert prepared.mf.cycles == 0
        assert not (prepared.workspace / "wavefunction.chk").exists()
        assert prepared.initial_density.shape == (prepared.mol.nao_nr(),) * 2
        assert abs(evidence["initial_electron_count"] - prepared.mol.nelectron) < 1e-8
        original_mol = lib.chkfile.load_mol(str(source / "wavefunction.chk"))
        orbitals = lib.chkfile.load(str(source / "wavefunction.chk"), "scf")
        real_density = scf.hf.make_rdm1(orbitals["mo_coeff"], orbitals["mo_occ"])
        np.testing.assert_allclose(
            prepared.initial_density, real_density, atol=1e-10, rtol=0
        )
        assert original_mol.nelectron == prepared.mol.nelectron
        assert not (prepared.workspace / "sealed-input.chk").stat().st_mode & 0o222
        assert (prepared.workspace / "restart-input.chk").stat().st_ino != (
            source / "wavefunction.chk"
        ).stat().st_ino
        for entry in prepared.snapshot["inventory"]:
            copied = Path(prepared.snapshot["directory"]) / entry["path"]
            assert sha256(copied.read_bytes()).hexdigest() == entry["sha256"]
            assert not copied.stat().st_mode & 0o222
        assert inputs(source) == before
    finally:
        prepared.close()


def test_genuine_kernel_consumes_saved_density_and_publishes_new_native_result(
    native_source, tmp_path
):
    source, request, original = native_source
    before = inputs(source)
    result = evaluate_with_checkpoint(
        request,
        tmp_path / "fresh",
        source_native_directory=source,
        recipe_sha256=recipe(),
    )
    assert result["status"] == "complete", result["errors"]
    evidence = result["checkpoint_consumption"]
    assert evidence["engine_checkpoint_reused"]
    assert evidence["fresh_scf_kernel_called"]
    assert not evidence["scf_iteration_state_resumed"]
    assert not evidence["final_result_reused"]
    assert not evidence["chemical_accuracy_established"]
    assert evidence["startup_observations"][0]["first_cycle"] == 1
    assert (
        evidence["initial_density_sha256"]
        == evidence["startup_observations"][0]["first_cycle_input_density_sha256"]
    )
    assert evidence["changed_fields"] == []
    assert result["scf"]["cycles"] >= 1
    assert abs(result["energy_hartree"] - original["energy_hartree"]) < 1e-9
    np.testing.assert_allclose(
        result["gradient_hartree_bohr"],
        original["gradient_hartree_bohr"],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        result["dipole_debye"], original["dipole_debye"], atol=1e-7, rtol=0
    )
    assert inputs(source) == before
    directory = Path(result["manifest_path"]).parent
    stored_request, stored_result = _native_bundle(directory)
    assert stored_request["molecule"] == stored_result["molecule"]
    assert stored_result["checkpoint_consumption"] == evidence
    new_molecule = lib.chkfile.load_mol(str(directory / "wavefunction.chk"))
    new_orbitals = lib.chkfile.load(str(directory / "wavefunction.chk"), "scf")
    np.testing.assert_allclose(
        new_molecule.atom_coords(unit="Bohr"), result["geometry_bohr"], atol=1e-12
    )
    assert abs(new_orbitals["e_tot"] - result["scf"]["energy_hartree"]) < 1e-10
    assert "init E=" in (directory / "pyscf.log").read_text()


def test_explicit_geometry_continuation_recalculates_at_actual_target(
    native_source, tmp_path
):
    source, request, original = native_source
    target = deepcopy(request)
    target["molecule"]["geometry_bohr"][-1][2] += 0.04
    with pytest.raises(
        ValueError, match="Incompatible restart fingerprint fields.*geometry_bohr"
    ):
        prepare_checkpoint_density(
            target,
            tmp_path / "unauthorized",
            source_native_directory=source,
            recipe_sha256=recipe(),
        )
    result = evaluate_with_checkpoint(
        target,
        tmp_path / "continued",
        source_native_directory=source,
        recipe_sha256=recipe(),
        allow_geometry_change=True,
    )
    reference = PySCFBackend().evaluate(target, tmp_path / "independent-fresh-guess")
    assert result["status"] == reference["status"] == "complete"
    assert result["geometry_bohr"] == target["molecule"]["geometry_bohr"]
    assert result["checkpoint_consumption"]["changed_fields"] == ["geometry_bohr"]
    assert abs(result["energy_hartree"] - reference["energy_hartree"]) < 1e-9
    assert abs(result["energy_hartree"] - original["energy_hartree"]) > 1e-5
    np.testing.assert_allclose(
        result["gradient_hartree_bohr"],
        reference["gradient_hartree_bohr"],
        atol=1e-8,
        rtol=0,
    )


def test_low_level_runner_observes_density_distinct_from_default_guess(
    native_source, tmp_path
):
    source, request, _ = native_source
    prepared = prepare_checkpoint_density(
        request,
        tmp_path / "direct",
        source_native_directory=source,
        recipe_sha256=recipe(),
    )
    try:
        with _environment(request["settings"], prepared.workspace):
            default_guess = prepared.mf.get_init_guess(prepared.mol, "minao")
            assert np.linalg.norm(default_guess - prepared.initial_density) > 1e-3
            energy = prepared.run_scf()
            assert np.isfinite(energy) and prepared.mf.converged
            assert prepared.consumption_evidence()["engine_checkpoint_reused"]
            with pytest.raises(ValueError, match="exactly one fresh SCF"):
                prepared.run_scf()
    finally:
        prepared.close()


@pytest.mark.parametrize(
    "change",
    ["atom_order", "atom_ids", "isotope", "charge", "settings", "basis", "method"],
)
def test_incompatible_fingerprints_never_start_scf(native_source, tmp_path, change):
    source, request, _ = native_source
    target = deepcopy(request)
    if change == "atom_order":
        for name in ("symbols", "atom_ids", "geometry_bohr", "isotope_symbols"):
            target["molecule"][name] = list(reversed(target["molecule"][name]))
    elif change == "atom_ids":
        target["molecule"]["atom_ids"][0] = "other-atom"
    elif change == "isotope":
        target["molecule"]["symbols"][-1] = "2H"
    elif change == "charge":
        target["molecule"]["charge"] = 2
    elif change == "settings":
        target["settings"]["scf_max_cycle"] += 1
    else:
        target["method"][change if change == "basis" else "name"] = (
            "cc-pvdz" if change == "basis" else "pbe"
        )
    with pytest.raises(ValueError):
        prepare_checkpoint_density(
            target,
            tmp_path / "rejected",
            source_native_directory=source,
            recipe_sha256=recipe(),
            allow_geometry_change=True,
        )
    assert not (tmp_path / "rejected" / "wavefunction.chk").exists()


def test_arbitrary_recipe_hash_and_manifest_identity_are_rejected(
    native_source, tmp_path
):
    source, request, _ = native_source
    with pytest.raises(ValueError, match="registered recipe"):
        prepare_checkpoint_density(
            request,
            tmp_path / "invented-recipe",
            source_native_directory=source,
            recipe_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="source manifest identity"):
        prepare_checkpoint_density(
            request,
            tmp_path / "wrong-manifest",
            source_native_directory=source,
            recipe_sha256=recipe(),
            expected_source_manifest_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "checkpoint_bytes",
        "inventoried_occupations",
        "inventoried_orbital_energy",
        "missing_inventory_file",
        "extra_unlisted_file",
    ],
)
def test_corrupted_native_input_is_rejected_without_repairs(
    native_source, tmp_path, damage
):
    source, request, _ = native_source
    damaged = tmp_path / "damaged-native"
    shutil.copytree(source, damaged)
    if damage == "checkpoint_bytes":
        path = damaged / "wavefunction.chk"
        path.write_bytes(path.read_bytes()[:128])
    elif damage in {"inventoried_occupations", "inventoried_orbital_energy"}:
        path = damaged / "wavefunction.chk"
        key = "mo_occ" if damage == "inventoried_occupations" else "mo_energy"
        observed = lib.chkfile.load(str(path), f"scf/{key}")
        altered = np.array(observed, copy=True)
        altered[0] = 1.0 if key == "mo_occ" else np.nan
        lib.chkfile.save(str(path), f"scf/{key}", altered)
        manifest = read_json(damaged / "manifest.json")
        for entry in manifest["artifacts"]:
            if entry["path"] == "wavefunction.chk":
                entry.update(
                    sha256=sha256(path.read_bytes()).hexdigest(),
                    size_bytes=path.stat().st_size,
                )
        (damaged / "manifest.json").write_bytes(canonical_json(manifest))
    elif damage == "missing_inventory_file":
        (damaged / "pyscf.log").unlink()
    else:
        (damaged / "unlisted-data.txt").write_text("deliberately unlisted test input\n")
    before = inputs(damaged)
    with pytest.raises(ValueError):
        prepare_checkpoint_density(
            request,
            tmp_path / "never-calculated",
            source_native_directory=damaged,
            recipe_sha256=recipe(),
        )
    assert inputs(damaged) == before
    assert not (tmp_path / "never-calculated" / "wavefunction.chk").exists()


def test_original_and_nonempty_workspaces_cannot_be_used_as_output(
    native_source, tmp_path
):
    source, request, _ = native_source
    with pytest.raises(ValueError, match="must be separate"):
        prepare_checkpoint_density(
            request,
            source,
            source_native_directory=source,
            recipe_sha256=recipe(),
        )
    destination = tmp_path / "nonempty"
    destination.mkdir()
    (destination / "retained.txt").write_text("actual pre-existing user input\n")
    with pytest.raises(ValueError, match="must be empty"):
        prepare_checkpoint_density(
            request,
            destination,
            source_native_directory=source,
            recipe_sha256=recipe(),
        )


def test_mutated_prepared_density_cannot_be_consumed(native_source, tmp_path):
    source, request, _ = native_source
    prepared = prepare_checkpoint_density(
        request,
        tmp_path / "density-altered",
        source_native_directory=source,
        recipe_sha256=recipe(),
    )
    try:
        prepared.initial_density[0, 0] += 0.1
        with pytest.raises(RuntimeError, match="density changed before use"):
            prepared.run_scf()
        assert not prepared.kernel_called
        assert prepared.observations == []
    finally:
        prepared.close()


def test_mutated_separate_input_is_not_repaired_or_consumed(native_source, tmp_path):
    source, request, _ = native_source
    original = inputs(source)
    prepared = prepare_checkpoint_density(
        request,
        tmp_path / "checkpoint-altered",
        source_native_directory=source,
        recipe_sha256=recipe(),
    )
    try:
        path = prepared.workspace / "restart-input.chk"
        damaged = path.read_bytes()[:128]
        path.write_bytes(damaged)
        with pytest.raises(RuntimeError, match="separate restart input was modified"):
            prepared.run_scf()
        assert not prepared.kernel_called
        assert path.read_bytes() == damaged
        assert inputs(source) == original
    finally:
        prepared.close()


def test_explicit_isotope_array_changes_are_rejected(native_source, tmp_path):
    source, request, _ = native_source
    target = deepcopy(request)
    target["molecule"]["isotopes"] = [None] * len(target["molecule"]["symbols"])
    target["molecule"]["isotopes"][-1] = 2
    with pytest.raises(ValueError, match="isotope declarations differ"):
        prepare_checkpoint_density(
            target,
            tmp_path / "isotope-array",
            source_native_directory=source,
            recipe_sha256=recipe(),
        )


def test_real_checkpoint_restarted_hf_analytic_hessian(native_source, tmp_path):
    source, request, _ = native_source
    target = deepcopy(request)
    target["properties"] = ["energy", "gradient", "hessian"]
    result = evaluate_with_checkpoint(
        target,
        tmp_path / "hessian",
        source_native_directory=source,
        recipe_sha256=recipe(),
    )
    reference = PySCFBackend().evaluate(target, tmp_path / "independent-hessian")
    assert result["status"] == reference["status"] == "complete"
    assert result["hessian_evidence"]["derivative"] == "analytic"
    assert result["checkpoint_consumption"]["engine_checkpoint_reused"]
    np.testing.assert_allclose(
        result["hessian_hartree_bohr2"],
        reference["hessian_hartree_bohr2"],
        atol=1e-8,
        rtol=0,
    )
    _native_bundle(Path(result["manifest_path"]).parent)


def test_real_optimizer_final_checkpoint_can_supply_initial_density(
    native_source, tmp_path
):
    _, request, _ = native_source
    optimized = PySCFBackend().optimize(request, tmp_path / "real-optimization")
    assert optimized["status"] == "complete", optimized["errors"]
    final = read_json(tmp_path / "real-optimization" / "final" / "request.json")
    final["optimization"] = {}
    result = evaluate_with_checkpoint(
        final,
        tmp_path / "from-optimized-final",
        source_native_directory=tmp_path / "real-optimization",
        recipe_sha256=recipe(),
    )
    assert result["status"] == "complete", result["errors"]
    assert result["geometry_bohr"] == optimized["geometry_bohr"]
    assert abs(result["energy_hartree"] - optimized["energy_hartree"]) < 1e-9
    assert result["checkpoint_consumption"]["engine_checkpoint_reused"]
    _native_bundle(Path(result["manifest_path"]).parent)


@pytest.mark.parametrize(
    "profile_id",
    [
        "pbe-d4-def2-svp-validation",
        "b3lyp-d4-def2-svp-validation",
        "mp2-cc-pvdz-validation",
    ],
)
def test_named_dft_d4_and_mp2_recipes_recalculate_genuine_components(
    tmp_path, profile_id
):
    profile = get_profile(profile_id)
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {"name": profile["method"], "basis": profile["basis"]},
        "settings": {"threads": 1, **profile["numerical"]},
        "properties": ["energy", "gradient"],
    }
    original = PySCFBackend().evaluate(request, tmp_path / "original")
    assert original["status"] == "complete", original["errors"]
    result = evaluate_with_checkpoint(
        request,
        tmp_path / "continued",
        source_native_directory=tmp_path / "original",
        recipe_sha256=profile["recipe_sha256"],
        expected_source_manifest_sha256=original["manifest_sha256"],
    )
    assert result["status"] == "complete", result["errors"]
    assert result["checkpoint_consumption"]["engine_checkpoint_reused"]
    assert abs(result["energy_hartree"] - original["energy_hartree"]) < 1e-9
    np.testing.assert_allclose(
        result["gradient_hartree_bohr"],
        original["gradient_hartree_bohr"],
        atol=1e-8,
        rtol=0,
    )
    if profile["method"] == "mp2":
        assert result["mp2"]["correlation_energy_hartree"] != 0
        assert result["mp2"]["frozen_orbitals"] == 0
    else:
        assert result["dispersion"]["variant"] == "D4(BJ)-EEQ-ATM"
        assert result["density_functional"]["gradient_grid_response"]
    _native_bundle(Path(result["manifest_path"]).parent)
