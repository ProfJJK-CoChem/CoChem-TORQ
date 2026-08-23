"""Comprehensive Test Suite for Cryptographic Payload Synthesizer.

Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------
Validates:
1. Kraitchman coordinates with real physical moments of inertia,
   singularity damping, ZPVE defect clamping, and piecewise Costain bounds.
2. OOM-proof PGOPHER XML skeleton generation inspecting Parquet metadata.
3. Provenance lock manifest generation under RFC 8785 Canonical JSON.
4. Deterministic .tar.zst payload bundling with normalized POSIX metadata.
5. Comprehensive payload integrity verification and tamper detection.
6. TorqExporter, PESStore, and export_qcschema integration.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import scipy.linalg as sla  # type: ignore[import-untyped]
import zstandard as zstd

from Libraries.cochem_torq_export import (
    CoChemIntegrityError,
    KraitchmanSingularityWarning,
    KraitchmanZPVEWarning,
    PESStore,
    TorqExporter,
    bundle_spycfit_payload,
    calculate_kraitchman_coords,
    canonical_json_dumps,
    compute_file_sha256,
    export_qcschema,
    generate_pgopher_skeleton,
    lock_provenance_payload,
    verify_payload_integrity,
)

# ============================================================================
# Physical Helper: Inertial Tensor & Moments for Rigid 3D Molecules
# ============================================================================


def compute_principal_moments(
    coordinates: npt.NDArray[np.float64], masses: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes center-of-mass shifted coordinates, principal moments of inertia.

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param masses: (N,) atomic masses in atomic mass units (u).
    :return: (principal_moments, principal_axes_matrix, aligned_coordinates)
    """
    total_mass = float(np.sum(masses))
    com = np.sum(coordinates * masses[:, None], axis=0) / total_mass
    coords_com = coordinates - com

    x = coords_com[:, 0]
    y = coords_com[:, 1]
    z = coords_com[:, 2]

    i_xx = float(np.sum(masses * (y**2 + z**2)))
    i_yy = float(np.sum(masses * (x**2 + z**2)))
    i_zz = float(np.sum(masses * (x**2 + y**2)))
    i_xy = -float(np.sum(masses * x * y))
    i_xz = -float(np.sum(masses * x * z))
    i_yz = -float(np.sum(masses * y * z))

    i_tensor = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]], dtype=np.float64
    )
    evals, evecs = sla.eigh(i_tensor)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    aligned = coords_com @ evecs
    return evals, evecs, aligned


# ============================================================================
# Test Suite 1: Kraitchman Coordinate Engine & Physical Invariants
# ============================================================================


def test_kraitchman_real_asymmetric_top() -> None:
    """Validates Kraitchman coordinates for 3D asymmetric top molecule."""
    coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],  # C
            [1.080000, 0.000000, 0.000000],  # H
            [-0.350000, 1.350000, 0.000000],  # F
            [-0.350000, -0.650000, 1.350000],  # Cl
            [-0.350000, -0.650000, -1.350000],  # Br
        ],
        dtype=np.float64,
    )
    masses_parent = np.array(
        [12.000000, 1.007825, 18.998403, 34.968853, 78.918337], dtype=np.float64
    )

    i_parent, _, aligned_parent = compute_principal_moments(coords, masses_parent)
    parent_mass = float(np.sum(masses_parent))

    masses_sub = masses_parent.copy()
    masses_sub[1] = 2.014102
    delta_m = 2.014102 - 1.007825

    i_sub, _, _ = compute_principal_moments(coords, masses_sub)
    true_h_coords = np.abs(aligned_parent[1])

    result = calculate_kraitchman_coords(
        parent_moments=i_parent,
        substituted_moments=i_sub,
        parent_mass=parent_mass,
        delta_m=delta_m,
    )

    calc_a = result["coords"]["a"]
    calc_b = result["coords"]["b"]
    calc_c = result["coords"]["c"]

    np.testing.assert_allclose(calc_a, true_h_coords[0], atol=1e-5)
    np.testing.assert_allclose(calc_b, true_h_coords[1], atol=1e-5)
    np.testing.assert_allclose(calc_c, true_h_coords[2], atol=1e-5)

    expected_mu = (parent_mass * delta_m) / (parent_mass + delta_m)
    assert math.isclose(result["reduced_mass"], expected_mu, rel_tol=1e-9)


def test_kraitchman_singularity_guard_damping() -> None:
    """Validates near-symmetric top damping guard (|Ia - Ib| < 1e-4)."""
    i_a = 15.00000
    i_b = 15.00005
    i_c = 30.00000

    parent_moments = {"Ia": i_a, "Ib": i_b, "Ic": i_c}
    sub_moments = {"Ia": i_a + 0.1, "Ib": i_b + 0.1, "Ic": i_c + 0.05}

    with pytest.warns(
        KraitchmanSingularityWarning, match="Singularity near-symmetric denominator"
    ):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=45.0,
            delta_m=1.003355,
            singularity_threshold=1e-4,
        )

    assert not math.isnan(result["coords"]["a"])
    assert not math.isnan(result["coords"]["b"])
    assert not math.isnan(result["coords"]["c"])
    assert result["coords"]["a"] >= 0.0


def test_kraitchman_zpve_defect_clamping() -> None:
    """Validates that negative radicands (R_g < 0) are clamped to 0.0000."""
    i_a, i_b, i_c = 10.0, 25.0, 30.0
    parent_moments = (i_a, i_b, i_c)
    sub_moments = (i_a + 1.5, i_b + 0.1, i_c + 0.1)

    with pytest.warns(
        KraitchmanZPVEWarning,
        match="ZPVE defect produced imaginary substitution coordinate",
    ):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=60.0,
            delta_m=1.003355,
        )

    assert result["coords"]["a"] == 0.0
    assert result["radicands"]["a"] < 0.0

    expected_error = math.sqrt(abs(result["radicands"]["a"]))
    assert math.isclose(result["costain_errors"]["a"], expected_error, rel_tol=1e-6)


def test_kraitchman_piecewise_costain_bounds() -> None:
    """Validates Piecewise Costain Bounds for large and small coordinates."""
    res = calculate_kraitchman_coords(
        parent_moments=(10.0, 20.0, 25.0),
        substituted_moments=(10.2, 20.4, 25.3),
        parent_mass=50.0,
        delta_m=1.00335,
    )

    for axis in ["a", "b", "c"]:
        coord = res["coords"][axis]
        error = res["costain_errors"][axis]
        radicand = res["radicands"][axis]

        if coord >= 0.15:
            assert math.isclose(error, 0.0015 / coord, rel_tol=1e-7)
        else:
            assert math.isclose(error, math.sqrt(abs(radicand)), rel_tol=1e-7)


# ============================================================================
# Test Suite 2: OOM-Proof PGOPHER XML Skeleton Generation
# ============================================================================


def test_generate_pgopher_skeleton_oom_proof(tmp_path: Path) -> None:
    """Validates PGOPHER XML generation inspecting Parquet metadata."""
    parquet_path = tmp_path / "spectral_catalog.parquet"
    json_path = tmp_path / "metadata.json"
    pgo_output = tmp_path / "deliverables" / "TargetMolecule.pgo"

    table = pa.Table.from_arrays(
        [
            pa.array([12345.67, 23456.78, 34567.89, 45678.90], type=pa.float64()),
            pa.array([-3.5, -4.2, -2.1, -5.8], type=pa.float64()),
            pa.array(
                ["1_0_1-0_0_0", "2_0_2-1_0_1", "2_1_1-1_1_0", "3_0_3-2_0_2"],
                type=pa.string(),
            ),
            pa.array([0.0, 0.41, 0.78, 1.15], type=pa.float64()),
        ],
        names=["frequency", "intensity", "quantum_numbers", "lower_state_energy"],
    )
    pq.write_table(table, str(parquet_path))

    meta_content = {
        "molecule_name": "TargetMolecule",
        "temperature_k": 150.0,
        "rotational_constants": {
            "A": 9876.54321,
            "B": 4321.09876,
            "C": 2109.87654,
        },
        "dipoles": {"mu_a": 1.45, "mu_b": 0.85, "mu_c": 0.12},
    }
    json_path.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")

    result_path = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        json_path=json_path,
        output_path=pgo_output,
    )

    assert Path(result_path).exists()
    assert Path(result_path) == pgo_output

    tree = ET.parse(str(pgo_output))
    root = tree.getroot()

    assert root.tag == "Document"
    assert root.attrib["Type"] == "PGopher"

    top = root.find(".//AsymmetricTop")
    assert top is not None

    params = {
        p.attrib["Name"]: float(p.attrib["Value"]) for p in top.findall("Parameter")
    }
    assert math.isclose(params["A"], 9876.54321, rel_tol=1e-5)
    assert math.isclose(params["B"], 4321.09876, rel_tol=1e-5)
    assert math.isclose(params["C"], 2109.87654, rel_tol=1e-5)
    assert math.isclose(params["mu_a"], 1.45, rel_tol=1e-5)
    assert math.isclose(params["mu_b"], 0.85, rel_tol=1e-5)
    assert math.isclose(params["mu_c"], 0.12, rel_tol=1e-5)

    meta_elem = root.find(".//Form/Metadata")
    assert meta_elem is not None
    assert meta_elem.attrib["NumTransitions"] == "4"
    assert meta_elem.attrib["NumColumns"] == "4"
    assert "frequency" in meta_elem.attrib["Columns"]


# ============================================================================
# Test Suite 3: Provenance Lock & RFC 8785 Canonical JSON
# ============================================================================


def test_lock_provenance_payload_canonical_json(tmp_path: Path) -> None:
    """Validates streaming SHA-256 and RFC 8785 Canonical JSON compliance."""
    payload_dir = tmp_path / "payload_workspace"
    payload_dir.mkdir()

    file_a = payload_dir / "molecule.var"
    file_a.write_text("VAR ROTATIONAL PARAMETERS A B C D\n" * 500, encoding="utf-8")

    file_b = payload_dir / "molecule.int"
    file_b.write_text("INT INTENSITY TRANSITIONS DIPOLE\n" * 300, encoding="utf-8")

    nested_dir = payload_dir / "tensors"
    nested_dir.mkdir()
    file_c = nested_dir / "large_tensor.bin"
    file_c.write_bytes(b"\xaa\xbb\xcc\xdd" * 4096)

    manifest_path = payload_dir / "spycfit_manifest.json"
    manifest = lock_provenance_payload(
        target_directory=payload_dir,
        output_manifest_path=manifest_path,
        metadata={"project": "CoChem-Unit-Test", "stage": "5.5"},
        rotational_constants={"A": 10000.0, "B": 5000.0, "C": 3000.0},
        dipoles={"mu_a": 1.2, "mu_b": 0.5, "mu_c": 0.1},
    )

    assert manifest_path.exists()
    assert manifest["format"] == "CoChem-SpycFit-Manifest"
    assert manifest["file_count"] == 3

    rel_paths = [f["relative_path"] for f in manifest["files"]]
    assert "spycfit_manifest.json" not in rel_paths
    assert "molecule.var" in rel_paths
    assert "molecule.int" in rel_paths
    assert "tensors/large_tensor.bin" in rel_paths

    large_entry = next(
        f for f in manifest["files"] if f["relative_path"] == "tensors/large_tensor.bin"
    )
    expected_large_sha = compute_file_sha256(file_c)[0]
    assert large_entry["sha256"] == expected_large_sha
    assert large_entry["size_bytes"] == 16384

    raw_manifest_text = manifest_path.read_text(encoding="utf-8")
    expected_canonical = canonical_json_dumps(manifest)
    assert raw_manifest_text == expected_canonical


# ============================================================================
# Test Suite 4: Deterministic .tar.zst Payload Bundling
# ============================================================================


def test_bundle_spycfit_payload_deterministic(tmp_path: Path) -> None:
    """Validates deterministic .tar.zst archive with normalized POSIX metadata."""
    payload_dir = tmp_path / "stage_deliverables"
    payload_dir.mkdir()

    (payload_dir / "spec.var").write_text("VAR FILE CONTENT\n", encoding="utf-8")
    (payload_dir / "spec.int").write_text("INT FILE CONTENT\n", encoding="utf-8")

    out_archive_dir = tmp_path / "exported_archives"
    archive_path_str = bundle_spycfit_payload(
        manifest_path_or_target_dir=payload_dir,
        output_dir=out_archive_dir,
        project_name="Water",
        compression_level=3,
    )

    archive_path = Path(archive_path_str)
    assert archive_path.exists()
    assert archive_path.name == "CoChem_Water_SpycFit_Payload.tar.zst"

    compressed_bytes = archive_path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_bytes), mode="r") as tar:
        members = tar.getmembers()
        assert len(members) >= 3

        for member in members:
            assert member.mtime == 0, f"mtime not normalized for {member.name}"
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            if member.isdir():
                assert member.mode == 0o755
            else:
                assert member.mode == 0o644


# ============================================================================
# Test Suite 5: Payload Verification & Tamper / Byte-Flip Error Injection
# ============================================================================


def test_verify_payload_integrity_pass_and_tamper(tmp_path: Path) -> None:
    """Validates cryptographic verification and error detection on tampered bytes."""
    payload_dir = tmp_path / "verify_workspace"
    payload_dir.mkdir()

    file_var = payload_dir / "spec.var"
    file_var.write_bytes(b"EXACT CANONICAL VAR PARAMETERS 1234567890")

    file_int = payload_dir / "spec.int"
    file_int.write_bytes(b"EXACT INTENSITIES 9876543210")

    lock_provenance_payload(payload_dir)
    archive_path = bundle_spycfit_payload(payload_dir, project_name="RigidRotor")

    assert verify_payload_integrity(payload_dir) is True
    assert verify_payload_integrity(payload_dir / "spycfit_manifest.json") is True
    assert verify_payload_integrity(archive_path) is True

    # Tamper Injection: Flip a single byte in spec.var
    original_bytes = file_var.read_bytes()
    tampered_bytes = original_bytes[:-1] + b"1"
    file_var.write_bytes(tampered_bytes)

    with pytest.raises(CoChemIntegrityError, match="SHA-256 hash mismatch"):
        verify_payload_integrity(payload_dir)

    # Missing File Injection: Delete spec.int
    file_var.write_bytes(original_bytes)
    file_int.unlink()

    with pytest.raises(CoChemIntegrityError, match="Missing file"):
        verify_payload_integrity(payload_dir)


# ============================================================================
# Test Suite 6: TorqExporter, PESStore & export_qcschema Integration
# ============================================================================


def test_torq_exporter_and_pes_store(tmp_path: Path) -> None:
    """Validates PESStore, TorqExporter, and export_qcschema."""
    h5_file = str(tmp_path / "pes_store.h5")
    store = PESStore(h5_file)

    store.append_data(step=1, coordinates=[0.0, 0.1, 0.2, 0.3], energy=-76.456)
    store.append_data(step=2, coordinates=[0.0, 0.15, 0.22, 0.35], energy=-76.458)

    with h5py.File(h5_file, "r") as f:
        assert "coordinates" in f
        assert "energies" in f
        assert f["coordinates"].shape == (2, 4)
        assert f["energies"].shape == (2,)
        assert f["coordinates"].scaleoffset is None
        assert f["energies"].scaleoffset is None

    export_dir = tmp_path / "zstd_exports"
    exporter = TorqExporter(export_dir=str(export_dir), zstd_compression_level=3)

    compressed_file = exporter.export_tensor_to_zstd(h5_file)
    assert Path(compressed_file).exists()

    success, metadata = exporter.verify_export(compressed_file)
    assert success is True
    assert metadata is not None
    assert metadata["compression_method"] == "Zstandard"

    qcschema_path = str(tmp_path / "qcschema.json")
    orca_result = {
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "symbols": ["O", "H"],
        "molecular_charge": 0,
        "molecular_multiplicity": 1,
        "driver": "energy",
        "method": "B3LYP",
        "basis": "def2-TZVP",
        "return_energy": -75.123456,
    }
    res_path = export_qcschema(orca_result, qcschema_path)
    assert Path(res_path).exists()

    with open(res_path, encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["schema_name"] == "qcschema_output"
    assert schema["properties"]["return_energy"] == -75.123456
    assert "hash" in schema["molecule"]["provenance"]


# ============================================================================
# Test Suite 7: Edge Cases, Archive Corruptions & Error Handling
# ============================================================================


def test_verify_payload_corrupted_archive_cases(tmp_path: Path) -> None:
    """Validates error raising on corrupt .tar.zst archives."""
    corrupt_zst = tmp_path / "corrupt.tar.zst"
    corrupt_zst.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00\x00_INVALID_ZSTD_GARBAGE")
    with pytest.raises(
        CoChemIntegrityError, match="Failed to decompress Zstandard archive"
    ):
        verify_payload_integrity(corrupt_zst)

    tar_no_manifest_buf = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_no_manifest_buf) as tar:
        ti = tarfile.TarInfo(name="isolated_data.txt")
        ti.size = 5
        tar.addfile(ti, io.BytesIO(b"HELLO"))

    no_manifest_zst = tmp_path / "no_manifest.tar.zst"
    no_manifest_zst.write_bytes(
        zstd.ZstdCompressor().compress(tar_no_manifest_buf.getvalue())
    )
    with pytest.raises(
        CoChemIntegrityError, match="Manifest 'spycfit_manifest.json' not found"
    ):
        verify_payload_integrity(no_manifest_zst)

    valid_dir = tmp_path / "valid_payload"
    valid_dir.mkdir()
    (valid_dir / "data.txt").write_text("VALID DATA 123", encoding="utf-8")
    lock_provenance_payload(valid_dir)
    valid_archive = bundle_spycfit_payload(valid_dir, project_name="CorruptTest")

    decompressed = zstd.ZstdDecompressor().decompress(Path(valid_archive).read_bytes())
    tar_tamper_buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r") as tar_in:
        with tarfile.open(fileobj=tar_tamper_buf, mode="w") as tar_out:
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member) if member.isreg() else None
                if member.name == "data.txt":
                    tampered_data = b"TAMPERED DATA!"
                    member.size = len(tampered_data)
                    tar_out.addfile(member, io.BytesIO(tampered_data))
                elif f:
                    tar_out.addfile(member, f)
                else:
                    tar_out.addfile(member)

    tampered_zst = tmp_path / "tampered_archive.tar.zst"
    tampered_zst.write_bytes(zstd.ZstdCompressor().compress(tar_tamper_buf.getvalue()))

    with pytest.raises(
        CoChemIntegrityError, match="(SHA-256 hash mismatch|File size mismatch)"
    ):
        verify_payload_integrity(tampered_zst)


def test_kraitchman_dictionary_and_planar_inputs() -> None:
    """Validates calculate_kraitchman_coords() with dictionary input structures."""
    parent_dict = {"a": 12.5, "b": 24.0, "c": 36.5}
    sub_dict = {"a": 12.8, "b": 24.4, "c": 36.9}

    res = calculate_kraitchman_coords(
        parent_moments=parent_dict,
        substituted_moments=sub_dict,
        parent_mass=78.0,
        delta_m=1.003355,
    )

    assert "coords" in res
    assert "costain_errors" in res
    assert "radicands" in res
    assert len(res["coords"]) == 3
    assert all(c >= 0.0 for c in res["coords"].values())


def test_file_not_found_guards(tmp_path: Path) -> None:
    """Validates FileNotFoundError guards across all export utilities."""
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        generate_pgopher_skeleton(parquet_path=non_existent / "catalog.parquet")

    with pytest.raises(FileNotFoundError):
        lock_provenance_payload(target_directory=non_existent)

    with pytest.raises(FileNotFoundError):
        bundle_spycfit_payload(manifest_path_or_target_dir=non_existent)
