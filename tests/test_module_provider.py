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


def test_entry_point_exposes_unique_owned_callable_and_truthful_nbo_absence() -> None:
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
    for name in ("nbo_analysis", "wiberg_nao"):
        assert metadata["scientific_apis"][name]["available"] is False
        assert "producer" in metadata["scientific_apis"][name]["reason"]


@pytest.mark.parametrize("name", ["nbo_analysis", "wiberg_nao"])
def test_unavailable_named_analysis_never_returns_a_substituted_result(
    tmp_path, name
) -> None:
    source = handoff(tmp_path, name, H2, options())
    result = provider.execute_handoff(
        handoff_path=source, output_directory=tmp_path / "out"
    )
    assert result["operation"] == name
    assert result["status"] == "unavailable"
    assert result["result"] is None
    assert (
        result["provider_sha256"]
        == hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()
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
    with pytest.raises(ValueError, match="correlated MP2 density"):
        provider.execute_handoff(handoff_path=source, output_directory=tmp_path / "out")


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
