"""
CoChem-TORQ: Comprehensive Pure Physical Execution Test Suite for Cryptographic Payload Synthesizer
Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------
Validates:
1. Kraitchman coordinate calculations with real physical moments of inertia,
   singularity damping, ZPVE defect clamping, and piecewise Costain bounds.
2. OOM-proof PGOPHER XML skeleton generation inspecting PyArrow Parquet metadata.
3. Provenance lock manifest generation under RFC 8785 Canonical JSON with streaming SHA-256.
4. Deterministic .tar.zst payload bundling with normalized POSIX metadata (mtime=0, 0644/0755).
5. Comprehensive payload integrity verification and tamper detection raising CoChemIntegrityError.
6. TorqExporter, PESStore, and export_qcschema integration.
"""

from __future__ import annotations

import io
import json
import math
import os
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
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
    coordinates: np.ndarray, masses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes center-of-mass shifted coordinates, principal moments of inertia (Ia, Ib, Ic),
    and aligned coordinates in the principal axis frame.
    
    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param masses: (N,) atomic masses in atomic mass units (u).
    :return: (principal_moments, principal_axes_matrix, aligned_coordinates)
    """
    # Shift to Center of Mass (COM)
    total_mass = float(np.sum(masses))
    com = np.sum(coordinates * masses[:, None], axis=0) / total_mass
    coords_com = coordinates - com

    # Inertia tensor components (u * A^2)
    x = coords_com[:, 0]
    y = coords_com[:, 1]
    z = coords_com[:, 2]

    Ixx = float(np.sum(masses * (y**2 + z**2)))
    Iyy = float(np.sum(masses * (x**2 + z**2)))
    Izz = float(np.sum(masses * (x**2 + y**2)))
    Ixy = float(-np.sum(masses * x * y))
    Ixz = float(-np.sum(masses * x * z))
    Iyz = float(-np.sum(masses * y * z))

    I_tensor = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(I_tensor)

    # Sort eigenvalues: Ia <= Ib <= Ic
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Align coordinates into principal axis frame
    aligned = coords_com @ evecs
    return evals, evecs, aligned


# ============================================================================
# Test Suite 1: Kraitchman Coordinate Engine & Physical Invariants
# ============================================================================


def test_kraitchman_real_asymmetric_top() -> None:
    """
    Validates Kraitchman coordinate derivation for a 3D asymmetric top molecule
    (Fluoroiodomethane derivative) against exact rigid-body coordinate invariants.
    """
    # Real physical Cartesian coordinates (Angstroms)
    # C, H, F, Cl, Br
    coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],   # C
            [1.080000, 0.000000, 0.000000],   # H
            [-0.350000, 1.350000, 0.000000],  # F
            [-0.350000, -0.650000, 1.350000], # Cl
            [-0.350000, -0.650000, -1.350000],# Br
        ],
        dtype=np.float64,
    )
    masses_parent = np.array([12.000000, 1.007825, 18.998403, 34.968853, 78.918337], dtype=np.float64)

    # Compute parent moments and aligned coordinates
    I_parent, _, aligned_parent = compute_principal_moments(coords, masses_parent)
    parent_mass = float(np.sum(masses_parent))

    # Substitute Hydrogen (index 1) with Deuterium (2.014102 u) -> delta_m = 1.006277 u
    masses_sub = masses_parent.copy()
    masses_sub[1] = 2.014102
    delta_m = 2.014102 - 1.007825

    I_sub, _, _ = compute_principal_moments(coords, masses_sub)

    # Target true coordinates of H in parent principal axis frame
    true_h_coords = np.abs(aligned_parent[1])

    # Run Kraitchman calculation
    result = calculate_kraitchman_coords(
        parent_moments=I_parent,
        substituted_moments=I_sub,
        parent_mass=parent_mass,
        delta_m=delta_m,
    )

    calc_a = result["coords"]["a"]
    calc_b = result["coords"]["b"]
    calc_c = result["coords"]["c"]

    # Assert exact physical agreement with principal frame Cartesian coordinates
    np.testing.assert_allclose(calc_a, true_h_coords[0], atol=1e-5)
    np.testing.assert_allclose(calc_b, true_h_coords[1], atol=1e-5)
    np.testing.assert_allclose(calc_c, true_h_coords[2], atol=1e-5)

    # Verify reduced mass
    expected_mu = (parent_mass * delta_m) / (parent_mass + delta_m)
    assert math.isclose(result["reduced_mass"], expected_mu, rel_tol=1e-9)


def test_kraitchman_singularity_guard_damping() -> None:
    """
    Validates that near-symmetric top denominators (|Ia - Ib| < 1e-4) trigger
    the KraitchmanSingularityWarning and apply damping guard to prevent divergence.
    """
    # Create near-symmetric moments where Ia and Ib differ by only 1e-5 (< 1e-4)
    Ia = 15.00000
    Ib = 15.00005
    Ic = 30.00000

    parent_moments = {"Ia": Ia, "Ib": Ib, "Ic": Ic}
    sub_moments = {"Ia": Ia + 0.1, "Ib": Ib + 0.1, "Ic": Ic + 0.05}

    with pytest.warns(KraitchmanSingularityWarning, match="Singularity near-symmetric denominator"):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=45.0,
            delta_m=1.003355,
            singularity_threshold=1e-4,
        )

    # Result should be finite and non-NaN
    assert not math.isnan(result["coords"]["a"])
    assert not math.isnan(result["coords"]["b"])
    assert not math.isnan(result["coords"]["c"])
    assert result["coords"]["a"] >= 0.0


def test_kraitchman_zpve_defect_clamping() -> None:
    """
    Validates that negative radicands (R_g < 0) arising from ZPVE defects or on-axis atoms
    are clamped to 0.0000 with a KraitchmanZPVEWarning.
    """
    # Momenta configured to yield an unphysical negative radicand on axis a:
    # dIa = 1.5, dIb = 0.1, dIc = 0.1 -> dPa = 0.5 * (0.2 - 1.5) = -0.65 < 0 -> Ra < 0
    Ia, Ib, Ic = 10.0, 25.0, 30.0
    parent_moments = (Ia, Ib, Ic)
    sub_moments = (Ia + 1.5, Ib + 0.1, Ic + 0.1)

    with pytest.warns(KraitchmanZPVEWarning, match="ZPVE defect produced imaginary substitution coordinate for axis a"):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=60.0,
            delta_m=1.003355,
        )

    # Coordinate for axis a must be clamped to exactly 0.0000
    assert result["coords"]["a"] == 0.0
    assert result["radicands"]["a"] < 0.0

    # For clamped coordinate (|a_s| = 0 < 0.15), Costain bound is sqrt(|R_a|)
    expected_error = math.sqrt(abs(result["radicands"]["a"]))
    assert math.isclose(result["costain_errors"]["a"], expected_error, rel_tol=1e-6)


def test_kraitchman_piecewise_costain_bounds() -> None:
    """
    Validates Piecewise Costain Bounds:
    - 0.0015 / |g_s| for |g_s| >= 0.15 A
    - sqrt(|R_g|) for |g_s| < 0.15 A
    """
    # Case 1: Large coordinate (|g_s| >= 0.15 A)
    # Using planar moments that yield |a_s| ~ 0.5 A
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
    """
    Validates PGOPHER XML generation inspecting real PyArrow Parquet metadata
    without loading the table into RAM, validating XML structure and rotational constants.
    """
    parquet_path = tmp_path / "spectral_catalog.parquet"
    json_path = tmp_path / "metadata.json"
    pgo_output = tmp_path / "deliverables" / "TargetMolecule.pgo"

    # Write a real PyArrow Parquet file
    table = pa.Table.from_arrays(
        [
            pa.array([12345.67, 23456.78, 34567.89, 45678.90], type=pa.float64()),
            pa.array([-3.5, -4.2, -2.1, -5.8], type=pa.float64()),
            pa.array(["1_0_1-0_0_0", "2_0_2-1_0_1", "2_1_1-1_1_0", "3_0_3-2_0_2"], type=pa.string()),
            pa.array([0.0, 0.41, 0.78, 1.15], type=pa.float64()),
        ],
        names=["frequency", "intensity", "quantum_numbers", "lower_state_energy"],
    )
    pq.write_table(table, str(parquet_path))

    # Write metadata JSON
    meta_content = {
        "molecule_name": "TargetMolecule",
        "temperature_k": 150.0,
        "rotational_constants": {"A": 9876.54321, "B": 4321.09876, "C": 2109.87654},
        "dipoles": {"mu_a": 1.45, "mu_b": 0.85, "mu_c": 0.12},
    }
    json_path.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")

    # Generate PGOPHER skeleton
    result_path = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        json_path=json_path,
        output_path=pgo_output,
    )

    assert Path(result_path).exists()
    assert Path(result_path) == pgo_output

    # Parse and validate XML structure
    tree = ET.parse(str(pgo_output))
    root = tree.getroot()

    assert root.tag == "Document"
    assert root.attrib["Type"] == "PGopher"

    # Validate Species and AsymmetricTop parameters
    top = root.find(".//AsymmetricTop")
    assert top is not None

    params = {p.attrib["Name"]: float(p.attrib["Value"]) for p in top.findall("Parameter")}
    assert math.isclose(params["A"], 9876.54321, rel_tol=1e-5)
    assert math.isclose(params["B"], 4321.09876, rel_tol=1e-5)
    assert math.isclose(params["C"], 2109.87654, rel_tol=1e-5)
    assert math.isclose(params["mu_a"], 1.45, rel_tol=1e-5)
    assert math.isclose(params["mu_b"], 0.85, rel_tol=1e-5)
    assert math.isclose(params["mu_c"], 0.12, rel_tol=1e-5)

    # Validate Form and Metadata attributes from Parquet
    meta_elem = root.find(".//Form/Metadata")
    assert meta_elem is not None
    assert meta_elem.attrib["NumTransitions"] == "4"
    assert meta_elem.attrib["NumColumns"] == "4"
    assert "frequency" in meta_elem.attrib["Columns"]


# ============================================================================
# Test Suite 3: Provenance Lock & RFC 8785 Canonical JSON
# ============================================================================


def test_lock_provenance_payload_canonical_json(tmp_path: Path) -> None:
    """
    Validates streaming SHA-256 hashing across 8192-byte binary chunks,
    exclusion of spycfit_manifest.json from the hashing loop, and
    RFC 8785 Canonical JSON compliance.
    """
    payload_dir = tmp_path / "payload_workspace"
    payload_dir.mkdir()

    # Create real files of various sizes (including >8192 bytes)
    file_a = payload_dir / "molecule.var"
    file_a.write_text("VAR ROTATIONAL PARAMETERS A B C D\n" * 500, encoding="utf-8")

    file_b = payload_dir / "molecule.int"
    file_b.write_text("INT INTENSITY TRANSITIONS DIPOLE\n" * 300, encoding="utf-8")

    nested_dir = payload_dir / "tensors"
    nested_dir.mkdir()
    file_c = nested_dir / "large_tensor.bin"
    file_c.write_bytes(b"\xAA\xBB\xCC\xDD" * 4096)  # 16,384 bytes (> 2 chunks)

    # Generate locked provenance manifest
    manifest_path = payload_dir / "spycfit_manifest.json"
    manifest = lock_provenance_payload(
        target_directory=payload_dir,
        output_manifest_path=manifest_path,
        metadata={"project": "CoChem-Unit-Test", "stage": "5.5"},
    )

    assert manifest_path.exists()
    assert manifest["format"] == "CoChem-SpycFit-Manifest"
    assert manifest["file_count"] == 3

    # Ensure manifest itself is not listed inside files
    rel_paths = [f["relative_path"] for f in manifest["files"]]
    assert "spycfit_manifest.json" not in rel_paths
    assert "molecule.var" in rel_paths
    assert "molecule.int" in rel_paths
    assert "tensors/large_tensor.bin" in rel_paths

    # Verify SHA-256 of large file matches independent hash
    large_entry = next(f for f in manifest["files"] if f["relative_path"] == "tensors/large_tensor.bin")
    expected_large_sha = compute_file_sha256(file_c)[0]
    assert large_entry["sha256"] == expected_large_sha
    assert large_entry["size_bytes"] == 16384

    # Verify RFC 8785 Canonical JSON formatting (no extraneous spaces, sorted keys)
    raw_manifest_text = manifest_path.read_text(encoding="utf-8")
    expected_canonical = canonical_json_dumps(manifest)
    assert raw_manifest_text == expected_canonical


# ============================================================================
# Test Suite 4: Deterministic .tar.zst Payload Bundling
# ============================================================================


def test_bundle_spycfit_payload_deterministic(tmp_path: Path) -> None:
    """
    Validates that bundle_spycfit_payload() generates a valid Zstandard-compressed
    tar archive with normalized POSIX metadata (mtime=0, uid=0, gid=0, 0644/0755).
    """
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

    # Decompress and inspect tar entries
    compressed_bytes = archive_path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_bytes), mode="r") as tar:
        members = tar.getmembers()
        assert len(members) >= 3

        for member in members:
            # Check POSIX normalization
            assert member.mtime == 0, f"mtime was not normalized to 0 for {member.name}"
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
    """
    Validates end-to-end cryptographic verification:
    - Passes cleanly on intact payload directory and .tar.zst archive.
    - Raises CoChemIntegrityError upon flipping a single byte in a deliverable.
    - Raises CoChemIntegrityError upon missing file in payload.
    """
    payload_dir = tmp_path / "verify_workspace"
    payload_dir.mkdir()

    file_var = payload_dir / "spec.var"
    file_var.write_bytes(b"EXACT CANONICAL VAR PARAMETERS 1234567890")

    file_int = payload_dir / "spec.int"
    file_int.write_bytes(b"EXACT INTENSITIES 9876543210")

    # Generate manifest and bundle archive
    manifest = lock_provenance_payload(payload_dir)
    archive_path = bundle_spycfit_payload(payload_dir, project_name="RigidRotor")

    # 1. Verification on intact directory
    assert verify_payload_integrity(payload_dir) is True

    # 2. Verification on intact manifest file path
    assert verify_payload_integrity(payload_dir / "spycfit_manifest.json") is True

    # 3. Verification on intact .tar.zst archive directly
    assert verify_payload_integrity(archive_path) is True

    # 4. Tamper Injection: Flip a single byte in spec.var (replace '0' with '1')
    original_bytes = file_var.read_bytes()
    tampered_bytes = original_bytes[:-1] + b"1"
    file_var.write_bytes(tampered_bytes)

    # Verification must catch flipped byte and raise CoChemIntegrityError
    with pytest.raises(CoChemIntegrityError, match="SHA-256 hash mismatch"):
        verify_payload_integrity(payload_dir)

    # 5. Missing File Injection: Delete spec.int
    file_var.write_bytes(original_bytes)  # Restore var file
    file_int.unlink()

    with pytest.raises(CoChemIntegrityError, match="Missing file"):
        verify_payload_integrity(payload_dir)


# ============================================================================
# Test Suite 6: TorqExporter, PESStore & export_qcschema Integration
# ============================================================================


def test_torq_exporter_and_pes_store(tmp_path: Path) -> None:
    """
    Validates legacy TorqExporter, PESStore scaleoffset-free appending,
    and export_qcschema FAIR JSON generation.
    """
    # Test PESStore
    h5_file = str(tmp_path / "pes_store.h5")
    store = PESStore(h5_file)

    # Append coordinate steps
    store.append_data(step=1, coordinates=[0.0, 0.1, 0.2, 0.3], energy=-76.456)
    store.append_data(step=2, coordinates=[0.0, 0.15, 0.22, 0.35], energy=-76.458)

    with h5py.File(h5_file, "r") as f:
        assert "coordinates" in f
        assert "energies" in f
        assert f["coordinates"].shape == (2, 4)
        assert f["energies"].shape == (2,)
        # Ensure scaleoffset is None per Section 6.4.3 rules
        assert f["coordinates"].scaleoffset is None
        assert f["energies"].scaleoffset is None

    # Test TorqExporter
    export_dir = tmp_path / "zstd_exports"
    exporter = TorqExporter(export_dir=str(export_dir), zstd_compression_level=3)

    compressed_file = exporter.export_tensor_to_zstd(h5_file)
    assert Path(compressed_file).exists()

    success, metadata = exporter.verify_export(compressed_file)
    assert success is True
    assert metadata is not None
    assert metadata["compression_method"] == "Zstandard"

    # Test export_qcschema
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

    with open(res_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["schema_name"] == "qcschema_output"
    assert schema["properties"]["return_energy"] == -75.123456
    assert "hash" in schema["molecule"]["provenance"]


# ============================================================================
# Test Suite 7: Edge Cases, Archive Corruptions & Error Handling
# ============================================================================


def test_verify_payload_corrupted_archive_cases(tmp_path: Path) -> None:
    """
    Validates that corrupt or invalid .tar.zst archives properly raise CoChemIntegrityError:
    1. Truncated / corrupt Zstandard bytes.
    2. Archive missing spycfit_manifest.json.
    3. Archive containing corrupted file bytes.
    """
    # 1. Invalid Zstandard bytes
    corrupt_zst = tmp_path / "corrupt.tar.zst"
    corrupt_zst.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00\x00_INVALID_ZSTD_GARBAGE")
    with pytest.raises(CoChemIntegrityError, match="Failed to decompress Zstandard archive"):
        verify_payload_integrity(corrupt_zst)

    # 2. Archive without spycfit_manifest.json
    tar_no_manifest_buf = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_no_manifest_buf) as tar:
        ti = tarfile.TarInfo(name="isolated_data.txt")
        ti.size = 5
        tar.addfile(ti, io.BytesIO(b"HELLO"))
    
    no_manifest_zst = tmp_path / "no_manifest.tar.zst"
    no_manifest_zst.write_bytes(zstd.ZstdCompressor().compress(tar_no_manifest_buf.getvalue()))
    with pytest.raises(CoChemIntegrityError, match="Manifest 'spycfit_manifest.json' not found"):
        verify_payload_integrity(no_manifest_zst)

    # 3. Archive with internal hash mismatch
    valid_dir = tmp_path / "valid_payload"
    valid_dir.mkdir()
    (valid_dir / "data.txt").write_text("VALID DATA 123", encoding="utf-8")
    lock_provenance_payload(valid_dir)
    valid_archive = bundle_spycfit_payload(valid_dir, project_name="CorruptTest")

    # Decompress tar, modify data.txt, recompress
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

    with pytest.raises(CoChemIntegrityError, match="(SHA-256 hash mismatch|File size mismatch)"):
        verify_payload_integrity(tampered_zst)


def test_kraitchman_dictionary_and_planar_inputs() -> None:
    """
    Validates calculate_kraitchman_coords() with dictionary input structures
    and verifies planar inertia relationships.
    """
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
    """
    Validates FileNotFoundError guards across all export utilities.
    """
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        generate_pgopher_skeleton(parquet_path=non_existent / "catalog.parquet")

    with pytest.raises(FileNotFoundError):
        lock_provenance_payload(target_directory=non_existent)

    with pytest.raises(FileNotFoundError):
        bundle_spycfit_payload(manifest_path_or_target_dir=non_existent)

