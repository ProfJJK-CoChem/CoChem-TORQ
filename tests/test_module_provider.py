"""Canonical BASE boundary and genuine bounded restricted-PySCF evidence.

Constructed geometry cases test mathematical invariants; real_engine cases
execute native calculations and retain their authentic artifacts in basetemp.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from cochem_torq import module_provider as provider
from cochem_torq.ecosystem import module_provider


def handoff(tmp_path: Path, operation: str, xyz: str, options: dict) -> Path:
    directory = tmp_path / "input"
    directory.mkdir()
    artifact = directory / "molecule.xyz"
    artifact.write_text(xyz)
    record = {
        "schema_version": "cochem.module-handoff/1",
        "module_id": "torq",
        "operation": operation,
        "scientific_execution_performed": False,
        "options": options,
        "artifact": {
            "kind": "geometry_xyz",
            "filename": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
    }
    path = directory / "handoff.json"
    path.write_text(json.dumps(record, allow_nan=False))
    return path


def options() -> dict:
    return {
        "method": {"name": "hf", "basis": "sto-3g"},
        "charge": 0,
        "multiplicity": 1,
        "cores": 1,
        "memory_mb": 512,
    }


H2 = "2\nH2 input in angstrom; not optimized\nH 0 0 0\nH 0 0 0.74\n"


def test_entry_point_exposes_unique_owned_callable_and_truthful_nbo_probe() -> None:
    metadata = module_provider()
    assert metadata["scientific_api_version"] == "cochem.module-provider/1"
    assert metadata["integration_contract"] == "cochem.module-handoff/1"
    assert metadata["execute_handoff"] == "cochem_torq.module_provider:execute_handoff"
    assert len(set(metadata["operations"])) == len(metadata["operations"]) == 5
    entries = [
        entry
        for entry in importlib.metadata.distribution("CoChem-TORQ").entry_points
        if entry.group == "cochem.modules" and entry.name == "torq"
    ]
    assert len(entries) == 1
    assert (
        entries[0].load()()["scientific_api_version"]
        == metadata["scientific_api_version"]
    )
    nbo = metadata["scientific_apis"]["nbo_analysis"]
    assert type(nbo["available"]) is bool
    if nbo["available"]:
        assert "nbo7" in nbo["runtime_receipts"]
        assert len(nbo["runtime_receipts"]["nbo7"]["sha256"]) == 64
        assert nbo["modes_available"]
    else:
        assert nbo["reason"]
    absent_probe = provider.probe_nbo7(env={}, search_defaults=False)
    assert absent_probe["available"] is False
    assert absent_probe["reason"]
    nao = metadata["scientific_apis"]["wiberg_nao"]
    assert type(nao["available"]) is bool
    if nao["available"]:
        assert nao["algorithm"]["engine_version"]
    else:
        assert nao["reason"]
    assert nao["supported_engines"] == nao["required_engines"] == ["pyscf"]
    assert nao["default_method"] == {"name": "b3lyp", "basis": "def2-svp"}
    assert nao["input_elements"] == "all_periodic_table_all_electron"
    assert "npa" not in metadata["operations"]


def test_nbo7_fails_closed_when_binaries_absent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NBO 7"):
        provider.run_nbo7_analysis(
            symbols=["H", "H"],
            coordinates_angstrom=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            env={},
            search_defaults=False,
            work_dir=tmp_path / "absent_nbo",
        )


@pytest.mark.parametrize(
    "xyz",
    [
        "3\ntruncated\nO 0 0 0\nH 0 0 1\n",
        "2\nextra\nH 0 0 0\nH 0 0 1\nH 0 1 0\n",
        "2\nnonfinite\nH 0 0 0\nH 0 0 nan\n",
        "2\nnonfinite\nH 0 0 0\nH 0 0 inf\n",
        "2\ncoincident\nH 0 0 0\nH 0 0 0\n",
        "2\nbad symbol\nH 0 0 0\nH; 0 0 1\n",
        "2\nextra field\nH 0 0 0\nH 0 0 1 2\n",
        "1\nno molecular rotor\nHe 0 0 0\n",
    ],
)
def test_rejects_invalid_xyz_before_science(tmp_path, xyz) -> None:
    source = tmp_path / "invalid.xyz"
    source.write_text(xyz)
    with pytest.raises(ValueError):
        provider.read_xyz(source)


def test_isotope_aliases_and_measured_masses_preserved(tmp_path) -> None:
    from mendeleev import element

    source = tmp_path / "aliases.xyz"
    source.write_text(
        "4\nisotope aliases\nC-13 0 0 0\nC13 0 0 1.4\nD 0 1 0\nT 0 0 2.4\n"
    )
    assert provider.read_xyz(source)[0] == ["13C", "13C", "2H", "3H"]
    elements, masses = provider._scan_nuclear_masses(["13C", "37Cl", "18O", "D", "T"])
    assert elements == ["C", "Cl", "O", "H", "H"]
    for label, observed in zip(["13C", "37Cl", "18O", "2H", "3H"], masses, strict=True):
        symbol = label.lstrip("0123456789")
        number = int(label.removesuffix(symbol))
        assert observed == next(
            iso.mass for iso in element(symbol).isotopes if iso.mass_number == number
        )
    for label in ("Tc", "999C"):
        with pytest.raises(ValueError, match="measured"):
            provider._scan_nuclear_masses([label])


def test_atomic_json_and_matrix_paths_cannot_overwrite_existing_evidence(
    tmp_path,
) -> None:
    target = tmp_path / "result.json"
    original = b'{"observation":"retain"}\n'
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        provider._publish_json(target, {"replace": True})
    assert target.read_bytes() == original
    with pytest.raises(ValueError):
        provider._publish_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_handoff_checksum_duplicate_field_and_symlink_rejection(tmp_path) -> None:
    source = handoff(tmp_path, "geometry_analysis", H2, {})
    artifact = source.parent / "molecule.xyz"
    original = artifact.read_bytes()
    artifact.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed"):
        provider.execute_handoff(handoff_path=source, output_directory=tmp_path / "out")
    artifact.write_bytes(original)
    source.write_text(
        source.read_text().replace(
            '"module_id": "torq"', '"module_id": "torq", "module_id": "torq"'
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        provider.execute_handoff(handoff_path=source, output_directory=tmp_path / "out")
    alias = tmp_path / "alias.xyz"
    alias.symlink_to(artifact)
    with pytest.raises(ValueError, match="regular"):
        provider.read_xyz(alias)


def test_geometry_analysis_preserves_linear_axis_absence_and_isotope_identity(
    tmp_path,
) -> None:
    source = handoff(
        tmp_path,
        "geometry_analysis",
        "2\nexplicit isotopes\n13C 0 0 0\n18O 0 0 1.13\n",
        {},
    )
    result = provider.execute_handoff(
        handoff_path=source, output_directory=tmp_path / "out"
    )
    native = result["result"]
    assert result["status"] == "succeeded"
    assert native["symbols"] == ["13C", "18O"]
    assert native["rotational_constants_mhz"][0] is None
    assert native["rotational_constants_mhz"][1] == pytest.approx(
        native["rotational_constants_mhz"][2], rel=1e-12
    )
    assert len(result["undefined_observables"]) == 3
    assert result["validation"]["coordinate_roundtrip_max_error_angstrom"] <= 1e-9
    assert result["validation"]["independent_inertia_and_mhz_check"] == "passed"
    assert "supplied geometry" in result["scope"]


@pytest.mark.real_engine
def test_actual_rigid_h2_scan_retains_native_energy_and_artifact_hashes(
    tmp_path,
) -> None:
    source = handoff(
        tmp_path,
        "research_scan",
        H2,
        {**options(), "fragments": [[0], [1]], "distances_angstrom": [0.74, 0.9]},
    )
    directory = tmp_path / "out"
    result = provider.execute_handoff(handoff_path=source, output_directory=directory)
    assert result["status"] == "completed"
    points = result["result"]["points"]
    assert points[0]["energy_hartree"] == pytest.approx(-1.1167593074, abs=1e-8)
    assert points[1]["energy_hartree"] > points[0]["energy_hartree"]
    for point in points:
        assert point["status"] == "computed"
        native = directory / point["result_path"]
        manifest = directory / point["native_manifest_path"]
        assert hashlib.sha256(native.read_bytes()).hexdigest() == point["result_sha256"]
        assert (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            == point["native_manifest_sha256"]
        )
        assert json.loads(native.read_text())["scf"]["converged"] is True


@pytest.mark.real_engine
def test_mixed_fragment_mass_com_native_scan_matches_nuclear_convention(
    tmp_path,
) -> None:
    from mendeleev import element

    xyz = (
        "5\nHCl and water rigid fragments\nH 0 0 0\nCl 1.3 0 0\nO 4 0 0\nH"
        " 4.9572 0 0\nH 3.76 .9273 0\n"
    )
    fragments = [[0, 1], [2, 3, 4]]
    source = handoff(
        tmp_path,
        "research_scan",
        xyz,
        {**options(), "fragments": fragments, "distances_angstrom": [4.0]},
    )
    result = provider.execute_handoff(
        handoff_path=source, output_directory=tmp_path / "out"
    )
    assert result["status"] == "completed"
    data = result["result"]
    masses = np.asarray(data["masses_u"])
    assert (
        masses[1]
        == max(
            (iso for iso in element("Cl").isotopes if iso.abundance),
            key=lambda iso: iso.abundance,
        ).mass
    )
    assert masses[1] != element("Cl").atomic_weight
    point = np.asarray(data["points"][0]["geometry_angstrom"])
    centers = [
        np.average(point[part], axis=0, weights=masses[part]) for part in fragments
    ]
    assert np.linalg.norm(centers[1] - centers[0]) == pytest.approx(4.0, abs=1e-12)
    original = np.asarray(provider.read_xyz(source.parent / "molecule.xyz")[1])
    for part in fragments:
        before = np.linalg.norm(
            original[part][:, None] - original[part][None, :], axis=2
        )
        after = np.linalg.norm(point[part][:, None] - point[part][None, :], axis=2)
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)


@pytest.mark.real_engine
def test_actual_h2_lowdin_wiberg_density_charge_and_immutable_matrices(
    tmp_path,
) -> None:
    source = handoff(tmp_path, "wiberg_lowdin", H2, options())
    directory = tmp_path / "out"
    result = provider.execute_handoff(handoff_path=source, output_directory=directory)
    assert result["status"] == "completed"
    data = result["result"]
    assert data["analysis_kind"] == "wiberg_lowdin"
    assert data["bonds"][0]["value"] == pytest.approx(1.0, abs=1e-10)
    assert data["charges_e"] == pytest.approx([0.0, 0.0], abs=1e-10)
    assert data["validation"]["density_trace"] == pytest.approx(2.0, abs=1e-7)
    assert data["validation"]["minimum_overlap_eigenvalue"] > 1e-10
    matrix = directory / "lowdin-density-overlap.npz"
    original = matrix.read_bytes()
    assert hashlib.sha256(original).hexdigest() == data["matrix_sha256"]
    with np.load(matrix, allow_pickle=False) as arrays:
        assert np.trace(arrays["density_ao"] @ arrays["overlap_ao"]) == pytest.approx(
            2.0, abs=1e-7
        )
        np.testing.assert_allclose(
            arrays["density_lowdin"], arrays["density_lowdin"].T, rtol=0, atol=1e-10
        )
    with pytest.raises(ValueError, match="empty"):
        provider.execute_handoff(handoff_path=source, output_directory=directory)
    assert matrix.read_bytes() == original


def test_mp2_is_not_relabelled_as_a_correlated_checkpoint_density(tmp_path) -> None:
    source = handoff(
        tmp_path,
        "wiberg_lowdin",
        H2,
        {**options(), "method": {"name": "mp2", "basis": "sto-3g"}},
    )
    # The producer rejects a correlated-density claim before optional imports.
    with pytest.raises(ValueError, match="correlated MP2 density"):
        provider.wiberg_lowdin(source, tmp_path / "direct")
    assert not (tmp_path / "direct").exists()

    # Preserve the public boundary's availability-first contract in both real
    # installed-runtime and genuine no-PySCF core environments.
    if provider.scientific_metadata()["scientific_apis"]["wiberg_lowdin"]["available"]:
        with pytest.raises(ValueError, match="correlated MP2 density"):
            provider.execute_handoff(
                handoff_path=source, output_directory=tmp_path / "out"
            )
    else:
        result = provider.execute_handoff(
            handoff_path=source, output_directory=tmp_path / "out"
        )
        assert result["status"] == "unavailable"
        assert result["result"] is None
        assert result["reason"]
    assert not any((tmp_path / "out").iterdir())


@pytest.mark.parametrize(
    "changed",
    [
        {"cores": 3},
        {"memory_mb": 2049},
        {"multiplicity": 3},
        {"charge": True},
        {"distances_angstrom": [0.74, 0.74]},
        {"distances_angstrom": [float("inf")]},
        {"fragments": [[0], [0, 1]]},
        {"method": {"name": "revdsd", "basis": "sto-3g"}},
    ],
)
def test_invalid_or_unsupported_scan_tuples_rejected_without_calculation(
    changed,
) -> None:
    request = {
        **options(),
        "fragments": [[0], [1]],
        "distances_angstrom": [0.74],
        **changed,
    }
    with pytest.raises(ValueError):
        provider._scan_options(request, 2)


@pytest.mark.real_engine
def test_actual_failed_native_scan_keeps_failed_points_and_no_invented_energy(
    tmp_path,
) -> None:
    request = {
        **options(),
        "method": {"name": "hf", "basis": "not-a-real-basis"},
        "fragments": [[0], [1]],
        "distances_angstrom": [0.74],
    }
    source = handoff(tmp_path, "research_scan", H2, request)
    directory = tmp_path / "out"
    result = provider.execute_handoff(handoff_path=source, output_directory=directory)
    assert result["status"] == "partial"
    point = result["result"]["points"][0]
    assert point["status"] == "failed"
    assert point["energy_hartree"] is None
    native = json.loads((directory / "scan-point-000" / "result.json").read_text())
    assert native["status"] == "failed"
    assert native["energy_hartree"] is None
    assert native["errors"]
    assert (directory / "scan-progress-000.json").is_file()


# Direct producer calls remain explicit validation; the final test separately
# exercises the activated public boundary.
def nao_options():
    return {**options(), "engine": "pyscf"}


@pytest.mark.parametrize(
    "change",
    [
        {"engine": None},
        {"engine": "orca"},
        {"engine": "cfour"},
        {"method": {"name": "mp2", "basis": "sto-3g"}},
        {"method": {"name": "revdsd", "basis": "def2-svp"}},
        {"charge": True},
        {"charge": 1},
        {"multiplicity": 3},
        {"cores": True},
        {"memory_mb": 20000},
    ],
)
def test_nao_refuses_other_explicit_tuples_before_native(tmp_path, change):
    source = handoff(tmp_path, "wiberg_nao", H2, {**nao_options(), **change})
    with pytest.raises(ValueError):
        provider.wiberg_nao(source, tmp_path / "out")
    assert not (tmp_path / "out/native-electronic").exists()


def test_nao_requires_backend_and_accepts_general_elements(tmp_path):
    source = handoff(tmp_path, "wiberg_nao", H2, options())
    with pytest.raises(ValueError, match="explicit"):
        provider.wiberg_nao(source, tmp_path / "out")
    checked_ch4 = provider._nao_options(
        {"engine": "pyscf"}, ["C", "H", "H", "H", "H"]
    )
    assert checked_ch4["method"] == {"name": "b3lyp", "basis": "def2-svp"}
    checked_hf = provider._nao_options(
        {**nao_options(), "method": {"name": "b3lyp", "basis": "def2-svp"}},
        ["H", "F"],
    )
    assert checked_hf["method"] == {"name": "b3lyp", "basis": "def2-svp"}
    assert not (tmp_path / "out/native-electronic").exists()


@pytest.mark.real_engine
def test_nao_default_mutable_configuration_is_part_of_identity():
    """Requires the genuine installed engine, but does not execute another SCF."""
    from pyscf.lo import nao

    original = nao.AOSHELL[1]
    before = provider._nao_algorithm_identity()
    try:
        nao.AOSHELL[1] = (original[0], "2s0p0d0f")
        with pytest.raises(ValueError, match="default AOSHELL"):
            provider._nao_algorithm_identity()
    finally:
        nao.AOSHELL[1] = original
    assert provider._nao_algorithm_identity() == before


def _independent_pair_checks(result, directory):
    """Independently recontract matrices without invented chemistry references."""
    import math
    import os
    import sys

    import pyscf

    source_provider = (
        Path(__file__).resolve().parents[1] / "src/cochem_torq/module_provider.py"
    ).resolve()
    env_expected = os.environ.get("COCHEM_NAO_EXPECTED_PROVIDER")
    actual_provider = Path(provider.__file__).resolve()
    expected = Path(env_expected).resolve() if env_expected else actual_provider
    wrapper = directory / "actual-operation-wrapper.json"
    wrapper.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    assert actual_provider == expected
    assert Path(result["provider_file"]).resolve() == expected
    assert actual_provider.read_bytes() == source_provider.read_bytes()
    assert (
        result["provider_sha256"]
        == hashlib.sha256(source_provider.read_bytes()).hexdigest()
    )
    (directory / "invocation-origins.json").write_text(
        json.dumps(
            {
                "cwd": os.getcwd(),
                "pythonpath": os.environ.get("PYTHONPATH"),
                "sys_executable": sys.executable,
                "sys_prefix": sys.prefix,
                "provider_module_file": str(Path(provider.__file__).resolve()),
                "pyscf_module_file": str(Path(pyscf.__file__).resolve()),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )

    data = result["result"]
    assert data["schema_version"] == "cochem.wiberg-nao/1"
    assert data["analysis_kind"] == "wiberg_nao"
    assert data["engine"] == "pyscf" and data["engine_version"] == "2.14.0"
    assert not {"charges_e", "populations_e", "npa", "nbo", "e2"}.intersection(data)
    assert data["validation"]["diagnostic_cutoffs"] is None
    with np.load(directory / "nao-density-transform.npz", allow_pickle=False) as arrays:
        density = arrays["density_nao"]
        assert arrays["coefficients_nao"].shape == density.shape
        np.testing.assert_allclose(
            arrays["coefficients_nao"].T
            @ arrays["overlap_ao"]
            @ arrays["coefficients_nao"],
            np.eye(len(density)),
            atol=1e-7,
            rtol=0,
        )
        assert (
            abs(float(np.trace(density)) - data["validation"]["electron_count"]) <= 1e-7
        )
        columns = [entry["columns"] for entry in data["atom_column_indices"]]
        assert [i for group in columns for i in group] == list(range(len(density)))
        for pair in data["bonds"]:
            # Positive sum's floating summation bound, from dimension and machine
            # epsilon, not a fitted chemical tolerance or np.testing default.
            products = [
                float(density[i, j]) ** 2
                for i in columns[pair["atom_i"]]
                for j in columns[pair["atom_j"]]
            ]
            reference = math.fsum(products)
            rounding = len(products) * np.finfo(np.float64).eps
            bound = rounding / (1 - rounding) * reference + math.ulp(reference)
            assert abs(pair["value"] - reference) <= bound
            assert pair["value"] >= 0
    return data


@pytest.mark.real_engine
def test_actual_h2_nao_wiberg_analytic_limit(tmp_path):
    from pyscf import lib

    source = handoff(tmp_path, "wiberg_nao", H2, nao_options())
    directory = tmp_path / "out"
    directory.mkdir()
    previous = lib.num_threads()
    try:
        # Actual caller state differs from request1: producer must scope itself,
        # then restore this state, rather than merely relying on test envvars.
        lib.num_threads(2)
        result = provider.wiberg_nao(source, directory)
        assert lib.num_threads() == 2
    finally:
        lib.num_threads(previous)
    assert result["status"] == "completed"
    data = _independent_pair_checks(result, directory)
    # For equivalent normalized single1s AOs and the real RHF bonding state,
    # this NAO construction reduces to symmetric orthogonalization: W_HH=1.
    assert data["bonds"][0]["value"] == pytest.approx(1.0, abs=1e-10, rel=0)
    assert data["validation"]["density_trace"] == pytest.approx(2.0, abs=1e-7, rel=0)
    assert data["atom_column_indices"][0]["columns"] == [0]
    assert data["atom_column_indices"][1]["columns"] == [1]
    assert data["validation"]["pyscf_threads_during_nao"] == 1
    original = (directory / "nao-density-transform.npz").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        provider.wiberg_nao(source, directory)
    assert (directory / "nao-density-transform.npz").read_bytes() == original


WATER_NAO = np.asarray(
    [
        [0.0, 0.0, -0.09822449881565],
        [0.0, -0.75808041118326, 0.53757836098152],
        [0.0, 0.75808041118326, 0.53757836098152],
    ]
)


def _declared_xyz(symbols, coordinates):
    return (
        str(len(symbols))
        + "\nDeclared NAO engineering input, not optimized\n"
        + "\n".join(
            symbol + " " + " ".join(format(float(v), ".17g") for v in row)
            for symbol, row in zip(symbols, coordinates, strict=True)
        )
        + "\n"
    )


@pytest.fixture(scope="module")
def genuine_water_nao_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("genuine-water-nao-invariance")
    axis = np.asarray([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    angle = 0.37
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    permutation = [2, 0, 1]
    definitions = [
        ("original", ["O", "H", "H"], WATER_NAO, [0, 1, 2]),
        (
            "fixed-rotation-translation",
            ["O", "H", "H"],
            WATER_NAO @ rotation.T + np.asarray([0.25, -0.5, 0.75]),
            [0, 1, 2],
        ),
        (
            "fixed-atom-permutation",
            ["H", "O", "H"],
            WATER_NAO[permutation],
            permutation,
        ),
    ]
    cases = []
    for name, symbols, geometry, mapping in definitions:
        case = root / name
        case.mkdir()
        source = handoff(
            case, "wiberg_nao", _declared_xyz(symbols, geometry), nao_options()
        )
        directory = case / "out"
        directory.mkdir()
        actual = provider.wiberg_nao(source, directory)
        data = _independent_pair_checks(actual, directory)
        cases.append((directory, data, mapping))
    return root, cases


@pytest.mark.real_engine
def test_actual_water_nao_fixed_rotation_translation_and_permutation(
    genuine_water_nao_cases,
):
    root, cases = genuine_water_nao_cases
    original = {
        (pair["atom_i"], pair["atom_j"]): pair["value"] for pair in cases[0][1]["bonds"]
    }
    comparisons = []
    for directory, data, mapping in cases[1:]:
        for pair in data["bonds"]:
            key = tuple(sorted((mapping[pair["atom_i"]], mapping[pair["atom_j"]])))
            error = abs(pair["value"] - original[key])
            comparisons.append(
                {
                    "case": directory.parent.name,
                    "original_pair": key,
                    "absolute_error": error,
                    "absolute_target": 1e-10,
                    "relative_target": 0,
                }
            )
    # Preserve all actual mapped errors before applying the predeclared NEW
    # engineering invariance target; do not choose a favorable orientation.
    (root / "mapped-pair-errors.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    assert all(row["absolute_error"] <= 1e-10 for row in comparisons), comparisons


@pytest.mark.real_engine
def test_split_valence_nao_algorithm_executes_genuine_pyscf_nao(tmp_path):
    import sys

    from pyscf.lo import nao

    checked = {**nao_options(), "method": {"name": "hf", "basis": "6-31g"}}
    validated = provider._nao_options(checked, ["O", "H", "H"])
    assert validated["method"] == {"name": "hf", "basis": "6-31g"}
    directory = tmp_path / "split-valence-algorithm-only"
    directory.mkdir()
    native, checkpoint, molecule, coefficients, occupations, overlap = (
        provider._restricted_scf_density(
            ["O", "H", "H"],
            WATER_NAO.tolist(),
            checked,
            directory,
        )
    )
    orbital_energies, _ = provider._nao_checkpoint_binding(
        native,
        checkpoint,
        molecule,
        checked,
        ["O", "H", "H"],
    )
    entries = []
    previous_profile = sys.getprofile()

    def observe_actual_calls(frame, event, arg):
        if event == "call" and frame.f_code is nao.nao.__code__:
            entries.append(str(Path(frame.f_code.co_filename).resolve()))
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    try:
        sys.setprofile(observe_actual_calls)
        transform, density_ao, density_nao, mapping, bonds, validation, algorithm = (
            provider._nao_transform(
                molecule,
                coefficients,
                occupations,
                orbital_energies,
                overlap,
                basis="6-31g",
            )
        )
    finally:
        sys.setprofile(previous_profile)
    assert entries == [str(Path(nao.__file__).resolve())]
    values, vectors = np.linalg.eigh(overlap)
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    lowdin_density = square_root @ density_ao @ square_root
    lowdin_pairs, pair_differences = [], []
    for pair in bonds:
        left, right = (
            mapping[pair["atom_i"]]["columns"],
            mapping[pair["atom_j"]]["columns"],
        )
        value = float(np.sum(lowdin_density[np.ix_(left, right)] ** 2))
        lowdin_pairs.append(
            {"atom_i": pair["atom_i"], "atom_j": pair["atom_j"], "value": value}
        )
        pair_differences.append(
            {
                "atom_i": pair["atom_i"],
                "atom_j": pair["atom_j"],
                "nao_minus_lowdin": pair["value"] - value,
            }
        )
    observation = {
        "scope": (
            "Separate actual split-valence NAO algorithm challenge; "
            "no provider-tuple activation"
        ),
        "native": native,
        "algorithm": algorithm,
        "actual_nao_call_files": entries,
        "validation": validation,
        "atom_columns": mapping,
        "bonds": bonds,
        "lowdin_pairs_same_native_density": lowdin_pairs,
        "gauge_invariant_pair_differences": pair_differences,
        "nao_vs_lowdin_density_max_abs_difference": float(
            np.max(np.abs(density_nao - lowdin_density))
        ),
        "difference_has_no_posthoc_cutoff": True,
    }
    (directory / "algorithm-observation.json").write_text(
        json.dumps(observation, indent=2, allow_nan=False) + "\n"
    )
    with (directory / "actual-nao-and-lowdin.npz").open("xb") as stream:
        np.savez(
            stream,
            coefficients_nao=transform,
            density_nao=density_nao,
            density_lowdin=lowdin_density,
            density_ao=density_ao,
            overlap_ao=overlap,
        )


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "corruption",
    [
        "energy_nan",
        "energy_boolean",
        "checkpoint_basis",
        "retained_basis",
        "cropped_orbitals",
        "nonfinite_orbital_energy",
        "fractional_occupation",
        "cartesian_flag",
        "ecp_flag",
        "atom_shell_partition",
    ],
)
def test_nao_rejects_corrupted_copies_of_genuine_native_evidence(
    genuine_water_nao_cases,
    tmp_path,
    corruption,
):
    import shutil

    import h5py
    from pyscf import gto, lib

    _, cases = genuine_water_nao_cases
    source = cases[0][0] / "native-electronic"
    target = tmp_path / "corrupt-native"
    shutil.copytree(source, target)
    checkpoint = target / "wavefunction.chk"
    native = json.loads((target / "result.json").read_text())
    molecule = lib.chkfile.load_mol(str(checkpoint))
    checked = nao_options()
    if corruption.startswith("energy_"):
        with h5py.File(checkpoint, "a") as handle:
            del handle["scf/e_tot"]
            handle["scf/e_tot"] = float("nan") if corruption == "energy_nan" else True
    elif corruption == "checkpoint_basis":
        molecule._basis["O"] = gto.basis.load("6-31g", "O")
    elif corruption == "retained_basis":
        changed = json.loads((target / "basis-definition.json").read_text())
        changed["O"] = gto.basis.load("6-31g", "O")
        (target / "basis-definition.json").write_text(json.dumps(changed))
    if corruption in {
        "energy_nan",
        "energy_boolean",
        "checkpoint_basis",
        "retained_basis",
    }:
        with pytest.raises(ValueError, match="energy|basis"):
            provider._nao_checkpoint_binding(
                native, checkpoint, molecule, checked, ["O", "H", "H"]
            )
    else:
        data = lib.chkfile.load(str(checkpoint), "scf")
        coefficients, occupations, energies = (
            np.asarray(data[key]) for key in ("mo_coeff", "mo_occ", "mo_energy")
        )
        overlap = molecule.intor_symmetric("int1e_ovlp")
        if corruption == "cropped_orbitals":
            coefficients, occupations, energies = (
                coefficients[:, :-1],
                occupations[:-1],
                energies[:-1],
            )
        elif corruption == "nonfinite_orbital_energy":
            energies[0] = float("nan")
        elif corruption == "fractional_occupation":
            occupations[0] = 1.5
        elif corruption == "cartesian_flag":
            molecule.cart = True
        elif corruption == "ecp_flag":
            molecule.ecp = {"O": "unqualified"}
        elif corruption == "atom_shell_partition":
            molecule._bas[0, gto.ATOM_OF] = 1
        with pytest.raises(ValueError):
            provider._nao_transform(
                molecule, coefficients, occupations, energies, overlap
            )
    assert not (target / "nao-density-transform.npz").exists()
    assert not (target / "bond-analysis.json").exists()


@pytest.mark.parametrize("setting", ["alias", "directory"])
@pytest.mark.real_engine
def test_nao_rejects_a_loader_that_selects_a_different_basis_file(tmp_path, setting):
    """Engine-profile configuration control; not an additional native calculation."""
    from pyscf import gto

    original_alias, original_directory = gto.basis.ALIAS["sto3g"], gto.basis._BASIS_DIR
    try:
        if setting == "alias":
            gto.basis.ALIAS["sto3g"] = "6-31g.dat"
        else:
            gto.basis._BASIS_DIR = str(tmp_path)
        with pytest.raises(ValueError, match="pinned basis file"):
            provider._nao_algorithm_identity()
    finally:
        gto.basis.ALIAS["sto3g"] = original_alias
        gto.basis._BASIS_DIR = original_directory
    assert provider._nao_algorithm_identity()["basis_loader"]["alias"] == "sto-3g.dat"


@pytest.mark.real_engine
def test_actual_unstable_restricted_h2_does_not_publish_nao(tmp_path):
    from pyscf import lib

    stretched = (
        "2\nDeclared stretched RHF stability challenge, not an equilibrium\n"
        "H 0 0 0\nH 0 0 3.0\n"
    )
    source = handoff(tmp_path, "wiberg_nao", stretched, nao_options())
    directory = tmp_path / "out"
    directory.mkdir()
    previous = lib.num_threads()
    try:
        lib.num_threads(2)
        with pytest.raises(ValueError, match="converged and stable"):
            provider.wiberg_nao(source, directory)
        assert lib.num_threads() == 2
    finally:
        lib.num_threads(previous)
    native = json.loads((directory / "native-electronic/result.json").read_text())
    assert native["stability"]["status"] != "stable"
    assert not (directory / "bond-analysis.json").exists()
    assert not (directory / "nao-density-transform.npz").exists()


def test_executable_nao_public_boundary_refuses_legacy_ambiguous_backend(tmp_path):
    source = handoff(tmp_path, "wiberg_nao", H2, options())
    available = module_provider()["scientific_apis"]["wiberg_nao"]["available"]
    if available:
        with pytest.raises(ValueError, match="explicit"):
            provider.execute_handoff(
                handoff_path=source, output_directory=tmp_path / "out"
            )
    else:
        # The existing public boundary reports unavailable runtimes before
        # operation-specific input validation; do not fabricate an engine or
        # silently substitute a different analysis in a genuine core install.
        result = provider.execute_handoff(
            handoff_path=source, output_directory=tmp_path / "out"
        )
        assert result["status"] == "unavailable" and result["result"] is None
        assert result["reason"]
        valid = tmp_path / "explicit-valid-request"
        valid.mkdir()
        explicit = handoff(valid, "wiberg_nao", H2, nao_options())
        unavailable = provider.execute_handoff(
            handoff_path=explicit, output_directory=valid / "out"
        )
        assert unavailable["status"] == "unavailable"
        assert unavailable["result"] is None and unavailable["reason"]
        assert not (valid / "out/native-electronic").exists()
    assert not (tmp_path / "out/native-electronic").exists()


@pytest.mark.real_engine
def test_actual_public_nao_entrypoint_has_distinct_schema_and_source(tmp_path):
    from pyscf import lib

    assert module_provider()["scientific_apis"]["wiberg_nao"]["available"] is True
    source = handoff(tmp_path, "wiberg_nao", H2, nao_options())
    directory = tmp_path / "out"
    previous = lib.num_threads()
    result = provider.execute_handoff(handoff_path=source, output_directory=directory)
    data = _independent_pair_checks(result, directory)
    assert result["operation"] == "wiberg_nao" and result["status"] == "completed"
    assert result["provider_distribution"] == "CoChem-TORQ"
    assert data["bonds"][0]["value"] == pytest.approx(1.0, abs=1e-10, rel=0)
    assert result["handoff_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert data["validation"]["pyscf_threads_during_nao"] == 1
    assert lib.num_threads() == previous


@pytest.mark.real_engine
def test_actual_general_element_nao_wiberg_hf_and_n2(tmp_path):
    hf_res = provider.run_wiberg_nao(
        ["H", "F"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]],
        method={"name": "b3lyp", "basis": "def2-svp"},
        output_directory=tmp_path / "hf_nao",
    )
    assert hf_res["status"] == "completed"
    hf_data = hf_res["result"]
    assert hf_data["atoms"] == ["H", "F"]
    assert hf_data["method"] == {"name": "b3lyp", "basis": "def2-svp"}
    assert 0.5 < hf_data["bonds"][0]["value"] < 1.1
    assert hf_data["validation"]["electron_count"] == 10

    n2_res = provider.run_wiberg_nao(
        ["N", "N"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.098]],
        output_directory=tmp_path / "n2_nao",
    )
    assert n2_res["status"] == "completed"
    n2_data = n2_res["result"]
    assert n2_data["atoms"] == ["N", "N"]
    assert n2_data["method"] == {"name": "b3lyp", "basis": "def2-svp"}
    assert 2.7 < n2_data["bonds"][0]["value"] < 3.3
    assert n2_data["validation"]["electron_count"] == 14


@pytest.mark.real_engine
def test_actual_nbo7_orca_handoff_and_standalone_gennbo(tmp_path):
    if not provider.probe_nbo7()["available"]:
        pytest.skip("Native NBO 7 binaries are not installed on this host.")
    h2o_xyz = (
        "3\nWater for authentic NBO 7 verification\n"
        "O 0.0 0.0 0.0\n"
        "H 0.0 0.75716 0.58626\n"
        "H 0.0 -0.75716 0.58626\n"
    )
    source = handoff(
        tmp_path,
        "nbo_analysis",
        h2o_xyz,
        {
            "method": {"name": "hf", "basis": "sto-3g"},
            "charge": 0,
            "multiplicity": 1,
            "cores": 1,
            "memory_mb": 512,
        },
    )
    out_dir = tmp_path / "nbo_out"
    res = provider.execute_handoff(handoff_path=source, output_directory=out_dir)
    assert res["status"] == "completed"
    assert res["operation"] == "nbo_analysis"
    data = res["result"]
    assert data["nbo_banner_verified"] is True
    assert data["nbo_version"].startswith("7.")
    assert len(data["raw_output_sha256"]) == 64
    assert len(data["nbo_exe_sha256"]) == 64
    assert len(data["npa_charges"]) == 3
    o_chg = data["npa_charges"][0]["natural_charge"]
    h1_chg = data["npa_charges"][1]["natural_charge"]
    h2_chg = data["npa_charges"][2]["natural_charge"]
    assert o_chg < -0.2 and h1_chg > 0.1 and h2_chg > 0.1
    assert abs(o_chg + h1_chg + h2_chg) < 1e-4
    oh_bonds = [
        b["value"]
        for b in data["wiberg_bonds"]
        if {b["symbol_i"], b["symbol_j"]} == {"O", "H"}
    ]
    assert len(oh_bonds) == 2 and all(0.7 < val < 1.2 for val in oh_bonds)
    assert len(data["e2_interactions"]) >= 2
    assert data["archive_47_path"] and Path(data["archive_47_path"]).is_file()

    # Also verify standalone gennbo.i8.exe directly on the produced FILE.47
    gen_res = provider.run_nbo7_analysis(
        archive_47_path=data["archive_47_path"],
        work_dir=tmp_path / "gennbo_standalone",
    )
    assert gen_res["driver_mode"] == "gennbo_47"
    assert gen_res["nbo_banner_verified"] is True
    assert gen_res["nbo_version"].startswith("7.")
    assert gen_res["npa_charges"][0]["natural_charge"] == pytest.approx(
        o_chg, abs=1e-4
    )
    assert len(gen_res["wiberg_bonds"]) == 3
    assert len(gen_res["raw_output_sha256"]) == 64
