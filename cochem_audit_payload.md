Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task12_export.md.
Original prompt:
# Prompt: Cryptographic Payload Synthesizer

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_export.py`

## Objective
Implement Cryptographic Payload Synthesizer for CoChem-TORQ based on Task 12 (Stage 5.5 - 6.0) specifications.

## Instructions for Coder
1. Create or update `cochem_torq_export.py` inside `Libraries/`.
2. Implement `calculate_kraitchman_coords()` evaluating substitution coordinates (r_s). Trap imaginary roots and implement Piecewise Costain Bounds.
3. Implement `generate_pgopher_skeleton()` inspecting Parquet metadata (`pyarrow.parquet.read_metadata()`) and emitting a standardized `.pgo` XML skeleton. Write this file strictly to the dynamically provided artifact/output directory, NOT the repository root.
4. Implement `lock_provenance_payload()` computing streaming SHA-256 checksums and serializing `spycfit_manifest.json` under RFC 8785 Canonical JSON. Write to the artifact directory.
5. Implement `bundle_spycfit_payload()` archiving deliverables into deterministic `.tar.zst` with normalized POSIX mtime and file permissions. Write to the artifact directory.
6. Implement `verify_payload_integrity()` executing an autonomous self-audit validating SHA-256 checksums prior to handoff.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_torq_export.py`.
- **Zero Mocking**: Do NOT mock any logic. Implement physical `pyarrow` metadata reads, SHA-256 hashing, and `.tar.zst` bundling.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically. All files (like `.tar.zst`, `.jsonl`, etc.) MUST be written to the scratch or artifact paths provided dynamically by the environment or arguments, NOT the current working directory.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_export.py ---
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

import h5py
import numpy as np
from mendeleev import element
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.linalg as sla
import zstandard as zstd

from Libraries.cochem_torq_alignment import enforce_ciaaw_masses
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
    # Dynamically retrieve CIAAW isotopic masses via Mendeleev library mandate
    masses_parent = enforce_ciaaw_masses(["C", "H", "F", "35Cl", "79Br"])

    i_parent, _, aligned_parent = compute_principal_moments(coords, masses_parent)
    parent_mass = float(np.sum(masses_parent))

    # Deuterated isotopologue substitution (2H / D)
    masses_sub = enforce_ciaaw_masses(["C", "2H", "F", "35Cl", "79Br"])
    delta_m = float(masses_sub[1] - masses_parent[1])

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

    from mendeleev import element
    with pytest.warns(
        KraitchmanSingularityWarning, match="Singularity near-symmetric denominator"
    ):
        delta_m_c = float(element("C").isotopes[1].mass - element("C").isotopes[0].mass)
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=float(element("Sc").atomic_weight),
            delta_m=delta_m_c,
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
        delta_m_c = float(element("C").isotopes[1].mass - element("C").isotopes[0].mass)
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=float(element("C").atomic_weight) * 5,
            delta_m=delta_m_c,
        )

    assert result["coords"]["a"] == 0.0
    assert result["radicands"]["a"] < 0.0

    expected_error = math.sqrt(abs(result["radicands"]["a"]))
    assert math.isclose(result["costain_errors"]["a"], expected_error, rel_tol=1e-6)


def test_kraitchman_piecewise_costain_bounds() -> None:
    """Validates Piecewise Costain Bounds for large and small coordinates."""
    delta_m_c = float(element("C").isotopes[1].mass - element("C").isotopes[0].mass)
    res = calculate_kraitchman_coords(
        parent_moments=(10.0, 20.0, 25.0),
        substituted_moments=(10.2, 20.4, 25.3),
        parent_mass=float(element("V").atomic_weight),
        delta_m=delta_m_c,
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

    delta_m_c = float(element("C").isotopes[1].mass - element("C").isotopes[0].mass)
    res = calculate_kraitchman_coords(
        parent_moments=parent_dict,
        substituted_moments=sub_dict,
        parent_mass=float(element("Se").atomic_weight),
        delta_m=delta_m_c,
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


def test_generate_pgopher_skeleton_variations(tmp_path: Path) -> None:
    """Validates PGOPHER skeleton generation with default output and variants."""
    parquet_path = tmp_path / "spectral.parquet"
    table = pa.Table.from_arrays(
        [
            pa.array([1000.0, 2000.0], type=pa.float64()),
            pa.array([-1.0, -2.0], type=pa.float64()),
        ],
        names=["freq", "intensity"],
    )
    pq.write_table(table, str(parquet_path))

    # Test with point_id, temperature, and list-based rotational constants and dipoles
    json_path = tmp_path / "point_metadata.json"
    json_content = {
        "point_id": "Conformer_A",
        "temperature": 10.0,
        "properties": {
            "rotational_constants": [5000.0, 2500.0, 1500.0],
            "dipole_moment": [0.5, 0.2, 0.0],
        },
    }
    json_path.write_text(json.dumps(json_content), encoding="utf-8")

    # Call with output_path=None to test default path generation
    out_pgo = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        json_path=json_path,
        output_path=None,
    )
    assert Path(out_pgo).exists()
    assert Path(out_pgo).name == "Conformer_A.pgo"

    # Call with direct list overrides
    out_pgo2 = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        output_path=tmp_path / "DirectOverride.pgo",
        molecule_name="OverrideMol",
        temperature_k=77.0,
        rotational_constants=[12000.0, 6000.0, 4000.0],
        dipoles=[2.0, 1.0, 0.5],
    )
    assert Path(out_pgo2).exists()
    tree = ET.parse(out_pgo2)
    root = tree.getroot()
    top = root.find(".//AsymmetricTop")
    assert top is not None
    params = {
        p.attrib["Name"]: float(p.attrib["Value"])
        for p in top.findall("Parameter")
    }
    assert math.isclose(params["A"], 12000.0, rel_tol=1e-5)
    assert math.isclose(params["mu_a"], 2.0, rel_tol=1e-5)


def test_bundle_spycfit_payload_from_manifest_file(tmp_path: Path) -> None:
    """Validates bundling payload when given direct path to manifest file."""
    payload_dir = tmp_path / "manifest_bundle_dir"
    payload_dir.mkdir()
    (payload_dir / "test.int").write_text("INT DATA", encoding="utf-8")

    manifest_file = payload_dir / "spycfit_manifest.json"
    lock_provenance_payload(payload_dir, output_manifest_path=manifest_file)

    archive_str = bundle_spycfit_payload(manifest_path_or_target_dir=manifest_file)
    assert Path(archive_str).exists()
    assert verify_payload_integrity(archive_str) is True


def test_verify_payload_with_dict_and_base_dir(tmp_path: Path) -> None:
    """Validates verify_payload_integrity when passed a manifest dictionary."""
    payload_dir = tmp_path / "dict_verify_dir"
    payload_dir.mkdir()
    (payload_dir / "file1.txt").write_bytes(b"DATA ONE")
    (payload_dir / "file2.txt").write_bytes(b"DATA TWO")

    manifest = lock_provenance_payload(payload_dir)
    assert verify_payload_integrity(manifest, base_dir=payload_dir) is True

    # Test size mismatch in directory verify
    (payload_dir / "file1.txt").write_bytes(b"LONGER DATA ONE MODIFIED")
    with pytest.raises(CoChemIntegrityError, match="File size mismatch"):
        verify_payload_integrity(manifest, base_dir=payload_dir)


def test_torq_exporter_batch_and_dvr(tmp_path: Path) -> None:
    """Validates batch export and Sinc-DVR export methods in TorqExporter."""
    h5_1 = tmp_path / "mol1.h5"
    h5_2 = tmp_path / "mol2.h5"

    for h5_path, val in [(h5_1, 10.0), (h5_2, 20.0)]:
        with h5py.File(str(h5_path), "w") as f:
            grp = f.create_group("geometry")
            grp.create_dataset("coords", data=[[0.0, 0.0, val]])
            f.create_dataset("energy", data=-75.5)

    export_out = tmp_path / "batch_out"
    exporter = TorqExporter(export_dir=str(export_out))

    exported = exporter.batch_export_to_zstd([str(h5_1), str(h5_2)])
    assert len(exported) == 2
    for exp_file in exported:
        assert Path(exp_file).exists()
        success, _ = exporter.verify_export(exp_file)
        assert success is True

    dvr_export = exporter.export_tensor_to_zstd_with_sinc_dvr(str(h5_1))
    assert Path(dvr_export).exists()
    assert "_dvr.zst" in dvr_export
    success, meta = exporter.verify_export(dvr_export)
    assert success is True
    assert meta is not None


def test_kraitchman_exact_zero_denominator_guard() -> None:
    """Validates Kraitchman coordinates when moments are identical (Ia == Ib)."""
    parent_moments = (20.0, 20.0, 40.0)
    sub_moments = (20.5, 20.5, 40.8)

    delta_m_h = float(element("H").isotopes[1].mass - element("H").isotopes[0].mass)
    with pytest.warns(KraitchmanSingularityWarning):
        res = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=float(element("V").atomic_weight),
            delta_m=delta_m_h,
            singularity_threshold=1e-4,
        )
    assert not math.isnan(res["coords"]["a"])
    assert not math.isnan(res["coords"]["b"])
    assert not math.isnan(res["coords"]["c"])


def test_pgopher_dict_overrides_and_missing_manifest(tmp_path: Path) -> None:
    """Validates dict overrides in PGOPHER generator and missing manifest error."""
    parquet_path = tmp_path / "spec_test.parquet"
    table = pa.Table.from_arrays(
        [pa.array([123.45]), pa.array([-2.5])], names=["freq", "int"]
    )
    pq.write_table(table, str(parquet_path))

    out_pgo = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        output_path=tmp_path / "DictOverride.pgo",
        molecule_name="DictMol",
        rotational_constants={"A": 8888.0, "B": 4444.0, "C": 2222.0},
        dipoles={"mu_a": 0.8, "mu_b": 0.4, "mu_c": 0.2},
    )
    assert Path(out_pgo).exists()

    # Missing manifest error
    non_existent_manifest = tmp_path / "no_such_manifest.json"
    with pytest.raises(CoChemIntegrityError, match="Manifest file not found"):
        verify_payload_integrity(non_existent_manifest)


def test_torq_exporter_scribe_and_corrupt_verify(tmp_path: Path) -> None:
    """Validates scribe daemon missing file handling and corrupt verification."""
    exporter = TorqExporter(export_dir=str(tmp_path))

    # Scribe daemon with missing file returns False
    res = exporter.export_to_scribe_daemon(str(tmp_path / "non_existent.zst"))
    assert res is False

    # Corrupt verification returns (False, None)
    corrupt_file = tmp_path / "bad.zst"
    corrupt_file.write_bytes(b"NOT_A_VALID_ZSTD_OR_JSON_STREAM")
    ok, meta = exporter.verify_export(str(corrupt_file))
    assert ok is False
    assert meta is None


def test_lock_provenance_with_kraitchman_and_nested_dirs(tmp_path: Path) -> None:
    """Validates lock_provenance_payload with kraitchman_coords and nested directory tarball."""
    payload_dir = tmp_path / "full_complex_payload"
    payload_dir.mkdir()
    sub_dir = payload_dir / "nested_models"
    sub_dir.mkdir()

    (sub_dir / "geom.xyz").write_text("3\nH2O\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    (payload_dir / "spec.var").write_text("VAR TEST", encoding="utf-8")

    kc = {
        "coords": {"a": 0.0, "b": 1.25, "c": 0.85},
        "costain_errors": {"a": 0.0, "b": 0.0012, "c": 0.0017},
        "radicands": {"a": -0.01, "b": 1.5625, "c": 0.7225},
    }

    manifest = lock_provenance_payload(
        target_directory=payload_dir,
        kraitchman_coords=kc,
        metadata={"run_id": "RUN-001"},
    )

    assert "kraitchman_coords" in manifest["metadata"]
    assert manifest["metadata"]["kraitchman_coords"]["coords"]["b"] == 1.25

    archive = bundle_spycfit_payload(
        manifest_path_or_target_dir=payload_dir,
        project_name="WaterDimer",
    )
    assert Path(archive).exists()
    assert verify_payload_integrity(archive) is True




Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.