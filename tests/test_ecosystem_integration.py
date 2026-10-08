"""Real schema/file interoperability; no simulated electronic-structure results.

TOPOS writes archived TORQ ORCA observations using its actual serializer. These
checks establish serialization compatibility only, never method accuracy. The
BASE producer runs its real handoff code in an isolated Python subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.constants import physical_constants

from cochem_torq.ecosystem import (
    AtomIdentity,
    ConformerHandoff,
    HandoffError,
    MethodProvenance,
    MoleculeHandoff,
    export_base_calculation_result,
    load_conformer_handoff,
    parse_xyz,
    read_base_handoff,
    read_topos_handoff,
    write_conformer_handoff,
)
from Libraries.cochem_torq_engine import _read_orca_engrad

REPOSITORY = Path(__file__).resolve().parents[1]
BASE_REVISION = "83462724849f1ef0be8c70ffcad6265d6af99388"
TOPOS_REVISION = "6a01b0f2adb7cff02edda6e339facf3d6f93904d"
SYMBOLS = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]
ATOM_IDS = [f"archive-atom-{index}" for index in range(10)]


def sibling_repository(name: str, revision: str) -> Path:
    override = os.environ.get(f"COCHEM_{name.upper()}_SOURCE")
    path = Path(override) if override else REPOSITORY.parent / f"CoChem-{name.upper()}"
    if not path.is_dir():
        if os.environ.get("COCHEM_REQUIRE_ECOSYSTEM") == "1":
            pytest.fail(
                f"Required real {name} checkout is absent; interoperability "
                "gate cannot pass"
            )
        pytest.skip(
            f"Real {name} checkout absent; ecosystem interoperability "
            "profile unverified"
        )
    observed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert observed == revision, (
        f"Review and repin the changed {name} consumer before qualification"
    )
    return path


def sibling_python(name: str) -> str:
    supplied = os.environ.get(f"COCHEM_{name.upper()}_PYTHON")
    return supplied if supplied else sys.executable


def run_provider(
    name: str, source: Path, program: Path, cwd: Path, *args: str
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(source)])
    return subprocess.run(
        [sibling_python(name), "-B", str(program), *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def archived_structure(tmp_path: Path) -> tuple[Path, float, list[list[float]]]:
    energy, gradient, numbers, geometry = _read_orca_engrad(
        REPOSITORY / "test.engrad", 10
    )
    assert numbers == [6, 6, 8, 8, 1, 1, 1, 1, 1, 1]
    xyz = tmp_path / "archived.xyz"
    rows = ["10", "Archived test.engrad geometry; this is not a new calculation"]
    rows.extend(
        f"{symbol} {row[0]:.15g} {row[1]:.15g} {row[2]:.15g}"
        for symbol, row in zip(SYMBOLS, geometry, strict=True)
    )
    xyz.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return xyz, energy, gradient.tolist()


@pytest.fixture
def archived_recipe() -> MethodProvenance:
    return MethodProvenance.from_recipe(
        recipe_id="archived-orca-input-unqualified",
        engine="ORCA",
        engine_version="6.1.1",
        method="r2SCAN-3c",
        basis="def2-TZVP",
        parameters={
            "input_sha256": hashlib.sha256(
                (REPOSITORY / "test.inp").read_bytes()
            ).hexdigest(),
            "auxiliary_basis": "def2/J",
            "grid": "DEFGRID3",
            "scope": "archived parser/serialization evidence; not method qualification",
        },
    )


@pytest.fixture
def real_topos_database(tmp_path: Path, archived_structure: tuple) -> Path:
    topos = sibling_repository("topos", TOPOS_REVISION)
    xyz, energy, gradient = archived_structure
    inputs = tmp_path / "observations.json"
    inputs.write_text(json.dumps({"energy": energy, "gradient": gradient}))
    script = tmp_path / "serialize_actual_topos.py"
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "from cascade_engine.cochem_cascade_hdf5 import CascadeHDF5Serializer\n"
        "r=json.loads(Path(sys.argv[2]).read_text())\n"
        "s=CascadeHDF5Serializer(sys.argv[1])\n"
        "s.write_tier_data('archive','T4',r['energy'],gradient=r['gradient'],geometry=Path(sys.argv[3]).read_text())\n"
        "s.close()\n",
        encoding="utf-8",
    )
    database = tmp_path / "landscape.h5"
    run_provider("topos", topos, script, tmp_path, str(database), str(inputs), str(xyz))
    return database


@pytest.fixture
def real_base_package(tmp_path: Path, archived_structure: tuple) -> Path:
    base = sibling_repository("base", BASE_REVISION)
    script = tmp_path / "prepare_actual_base.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from cochem_base.interfaces.artifact_handoff import (\n"
        "    prepare_module_handoff,load_module_handoff)\n"
        "r=prepare_module_handoff('torq',sys.argv[1],sys.argv[2],operation='geometry_analysis')\n"
        "assert load_module_handoff(Path(sys.argv[2])/'handoff.json')==r\n",
        encoding="utf-8",
    )
    package = tmp_path / "base-package"
    run_provider(
        "base", base, script, tmp_path, str(archived_structure[0]), str(package)
    )
    return package / "handoff.json"


def topos_arguments(recipe: MethodProvenance) -> dict:
    return {
        "geometry_id": "archive",
        "tier": "T4",
        "molecule_id": "archived-molecule",
        "charge": 0,
        "multiplicity": 1,
        "atom_ids": ATOM_IDS,
        "geometry_unit": "angstrom",
        "method_provenance": recipe,
        "energy_kind": "total_electronic",
        "reference_id": None,
        "repository_revision": TOPOS_REVISION,
    }


def base_arguments() -> dict:
    return {
        "molecule_id": "archived-molecule",
        "charge": 0,
        "multiplicity": 1,
        "atom_ids": ATOM_IDS,
        "repository_revision": BASE_REVISION,
    }


def test_real_topos_serializer_roundtrip_preserves_observed_data(
    real_topos_database: Path,
    archived_recipe: MethodProvenance,
    archived_structure: tuple,
    tmp_path: Path,
) -> None:
    record = read_topos_handoff(real_topos_database, **topos_arguments(archived_recipe))
    assert record.energy is not None
    assert record.energy.value == archived_structure[1]
    assert record.source_convergence == "unknown"
    assert "source_convergence_unavailable" in record.quality_flags
    assert (
        record.source.artifact_sha256
        == hashlib.sha256(real_topos_database.read_bytes()).hexdigest()
    )
    application = record.to_application_molecule()
    _, xyz_positions = parse_xyz(archived_structure[0].read_text())
    bohr_angstrom = physical_constants["Bohr radius"][0] * 1e10
    np.testing.assert_allclose(
        np.array(application["geometry_bohr"]) * bohr_angstrom,
        xyz_positions,
        rtol=0,
        atol=1e-12,
    )
    assert application["atom_ids"] == ATOM_IDS
    path = write_conformer_handoff(record, tmp_path / "torq-handoff.json")
    assert load_conformer_handoff(path) == record
    with pytest.raises(FileExistsError):
        write_conformer_handoff(record, path)


def test_real_base_producer_is_consumed_without_inventing_properties(
    real_base_package: Path,
) -> None:
    record = read_base_handoff(real_base_package, **base_arguments())
    assert record.energy is None and record.source_method is None
    assert record.source_convergence == "unknown"
    assert record.to_application_molecule()["symbols"] == SYMBOLS
    assert record.to_application_provenance()["atom_mapping"] == ATOM_IDS
    assert "input_geometry_is_not_an_optimized_result" in record.quality_flags


def test_real_base_candidate_enters_actual_application_request_with_lineage(
    real_base_package: Path,
) -> None:
    from cochem_torq.domain import CalculationRequest

    record = read_base_handoff(real_base_package, **base_arguments())
    request = CalculationRequest.model_validate(
        {
            "molecule": record.to_application_molecule(),
            "source_provenance": record.to_application_provenance(),
            "recipe": "hf-sto-3g-education",
            "products": ["equilibrium_constants"],
        }
    )
    assert request.molecule.atom_ids == ATOM_IDS
    assert request.molecule.isotopes == [None] * 10
    assert request.source_provenance["handoff"]["energy"] is None
    assert request.source_provenance["handoff"]["source"]["manifest_sha256"] == (
        hashlib.sha256(real_base_package.read_bytes()).hexdigest()
    )


@pytest.fixture(scope="module")
def genuine_torq_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip(
        "pyscf",
        reason="Genuine PySCF calculation required for reverse consumer qualification",
    )
    from cochem_torq.application import execute_request

    output = tmp_path_factory.mktemp("actual-torq-h2") / "result-bundle"
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            "charge": 0,
            "multiplicity": 1,
            "atom_ids": ["h-one", "h-two"],
        },
        "recipe": "hf-sto-3g-education",
        "products": ["equilibrium_constants"],
        "resources": {"cores": 1, "memory_mb": 512, "wall_seconds": 120},
    }
    response = execute_request(request, output)
    assert response["status"] == "complete"
    return output


@pytest.mark.real_engine
def test_genuine_torq_run_export_is_accepted_by_actual_base_consumer(
    genuine_torq_run: Path, tmp_path: Path
) -> None:
    base = sibling_repository("base", BASE_REVISION)
    exported = export_base_calculation_result(
        genuine_torq_run, tmp_path / "electronic-energy.json"
    )
    data = json.loads(exported.read_text())
    assert data["quantity"] == "total_electronic_energy"
    assert data["energy_hartree"] < -1.0
    assert data["scf"]["converged"] is True
    assert data["stability"]["status"] == "stable"
    assert data["molecule"]["atom_ids"] == ["h-one", "h-two"]
    script = tmp_path / "consume_actual_torq_result.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from cochem_base.interfaces.artifact_handoff import (\n"
        "    prepare_module_handoff,load_module_handoff)\n"
        "r=prepare_module_handoff('base',sys.argv[1],sys.argv[2],operation='ingest_result')\n"
        "assert r.artifact.kind=='calculation_result'\n"
        "assert r.artifact.metadata['engine']=='PySCF'\n"
        "assert load_module_handoff(Path(sys.argv[2])/'handoff.json')==r\n",
        encoding="utf-8",
    )
    run_provider(
        "base",
        base,
        script,
        tmp_path,
        str(exported),
        str(tmp_path / "base-result-package"),
    )
    with pytest.raises(FileExistsError):
        export_base_calculation_result(genuine_torq_run, exported)


@pytest.mark.real_engine
def test_reverse_export_rejects_damaged_actual_torq_shard(
    genuine_torq_run: Path, tmp_path: Path
) -> None:
    damaged = tmp_path / "damaged-run"
    shutil.copytree(genuine_torq_run, damaged)
    data = json.loads((damaged / "result.json").read_text())
    data["stages"]["electronic_structure"]["value"]["energy_hartree"] = 0.0
    (damaged / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inventory/hash mismatch"):
        export_base_calculation_result(damaged, tmp_path / "rejected-export.json")
    assert not (tmp_path / "rejected-export.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "checksum",
        "metadata",
        "outside",
        "size",
        "recipient",
        "scope",
        "duplicate",
        "timestamp",
        "available",
        "artifact_extra",
    ],
)
def test_real_base_damaged_manifest_is_rejected(
    real_base_package: Path, change: str
) -> None:
    data = json.loads(real_base_package.read_text())
    if change == "checksum":
        data["artifact"]["sha256"] = "0" * 64
    elif change == "metadata":
        data["artifact"]["metadata"]["symbols"][0] = "H"
    elif change == "outside":
        data["artifact"]["filename"] = "../archived.xyz"
    elif change == "size":
        data["artifact"]["size_bytes"] = True
    elif change == "recipient":
        data["module_id"] = "topos"
    elif change == "scope":
        data["scientific_execution_performed"] = True
    elif change == "timestamp":
        data["created_at"] = "2026-10-07T12:00:00"
    elif change == "available":
        data["capability"]["status"] = "available"
    elif change == "artifact_extra":
        data["artifact"]["invented_source"] = "unreviewed extension"
    elif change == "duplicate":
        real_base_package.write_text('{"module_id":"torq","module_id":"topos"}')
        with pytest.raises(HandoffError, match="Duplicate"):
            read_base_handoff(real_base_package, **base_arguments())
        return
    real_base_package.write_text(json.dumps(data))
    with pytest.raises((HandoffError, ValueError)):
        read_base_handoff(real_base_package, **base_arguments())


@pytest.mark.parametrize(
    "change",
    ["missing_energy", "nan_energy", "string_energy", "bad_geometry", "version"],
)
def test_real_topos_damaged_snapshot_is_rejected(
    real_topos_database: Path,
    archived_recipe: MethodProvenance,
    change: str,
    tmp_path: Path,
) -> None:
    damaged = tmp_path / "damaged.h5"
    shutil.copyfile(real_topos_database, damaged)
    with h5py.File(damaged, "a") as database:
        group = database["archive/T4"]
        if change == "missing_energy":
            del group.attrs["electronic_energy_hartree"]
        elif change == "nan_energy":
            group.attrs["electronic_energy_hartree"] = float("nan")
        elif change == "string_energy":
            group.attrs["electronic_energy_hartree"] = "not an observed number"
        elif change == "bad_geometry":
            del group["geometry_xyz"]
            group.create_dataset(
                "geometry_xyz", data=b"2\nTruncated archived geometry\nC 0 0 0\n"
            )
        elif change == "version":
            database.attrs["format_version"] = "999"
    with pytest.raises((HandoffError, ValidationError)):
        read_topos_handoff(damaged, **topos_arguments(archived_recipe))


def test_topos_does_not_infer_missing_state_mapping_or_convergence(
    real_topos_database: Path, archived_recipe: MethodProvenance
) -> None:
    args = topos_arguments(archived_recipe)
    del args["charge"]
    with pytest.raises(TypeError):
        read_topos_handoff(real_topos_database, **args)
    args = topos_arguments(archived_recipe)
    args["source_convergence"] = "converged"
    with pytest.raises(HandoffError, match="convergence_evidence"):
        read_topos_handoff(real_topos_database, **args)
    args = topos_arguments(archived_recipe)
    args["atom_ids"] = ATOM_IDS[:-1]
    with pytest.raises(HandoffError, match="every"):
        read_topos_handoff(real_topos_database, **args)


def test_geometry_bound_energy_cannot_move_to_another_structure(
    real_topos_database: Path, archived_recipe: MethodProvenance
) -> None:
    record = read_topos_handoff(real_topos_database, **topos_arguments(archived_recipe))
    payload = record.model_dump(mode="json")
    payload["molecule"]["geometry"][0][0] += 0.1
    with pytest.raises(ValidationError, match="different molecular geometry"):
        ConformerHandoff.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    ["spin", "charge", "symbol", "mapping", "boolean", "nonfinite", "rows", "overlap"],
)
def test_molecular_state_and_mapping_reject_invalid_inputs(
    archived_structure: tuple, change: str
) -> None:
    symbols, geometry = parse_xyz(archived_structure[0].read_text())
    payload = {
        "molecule_id": "archive",
        "atoms": [
            {"atom_id": atom_id, "symbol": symbol}
            for atom_id, symbol in zip(ATOM_IDS, symbols, strict=True)
        ],
        "geometry": [list(row) for row in geometry],
        "geometry_unit": "angstrom",
        "charge": 0,
        "multiplicity": 1,
    }
    if change == "spin":
        payload["multiplicity"] = 2
    elif change == "charge":
        payload["charge"] = True
    elif change == "symbol":
        payload["atoms"][0]["symbol"] = "Xx"
    elif change == "mapping":
        payload["atoms"][0]["atom_id"] = ATOM_IDS[1]
    elif change == "boolean":
        payload["geometry"][0][0] = True
    elif change == "nonfinite":
        payload["geometry"][0][0] = float("inf")
    elif change == "rows":
        payload["geometry"].pop()
    elif change == "overlap":
        payload["geometry"][0] = payload["geometry"][1]
    with pytest.raises(ValidationError):
        MoleculeHandoff.model_validate(payload)


def test_full_recipe_digest_detects_changed_scientific_settings(
    archived_recipe: MethodProvenance,
) -> None:
    payload = archived_recipe.model_dump(mode="json")
    payload["parameters"]["grid"] = "a changed grid"
    with pytest.raises(ValidationError, match="full source method"):
        MethodProvenance.model_validate(payload)


def test_isotope_notation_does_not_become_an_element_or_average_mass() -> None:
    with pytest.raises(ValidationError):
        AtomIdentity(atom_id="carbon", symbol="13C")
    with pytest.raises(ValidationError):
        AtomIdentity(atom_id="carbon", symbol="C", isotope_mass_number=5)
    assert AtomIdentity(atom_id="carbon", symbol="C").isotope_mass_number is None


def test_real_base_cannot_satisfy_fenced_worker_contract() -> None:
    base = sibling_repository("base", BASE_REVISION)
    script = (
        "import inspect\n"
        "from cochem.orchestration.sqlite_queue import SQLiteTaskQueue\n"
        "parameters=inspect.signature(SQLiteTaskQueue.complete_task).parameters\n"
        "assert 'lease_token' not in parameters\n"
        "assert 'lease_generation' not in parameters\n"
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(base / "src"), str(base)])
    subprocess.run(
        [sibling_python("base"), "-B", "-c", script],
        cwd=base,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
