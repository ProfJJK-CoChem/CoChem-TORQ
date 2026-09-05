"""Unit and Integration Tests for CoChem-TORQ Vault & Dual-Intake Gateway
======================================================================
Validates Phase 2 (Stage 1.0-2.0) Dual-Intake Gateway, SHA-256 integrity hashing,
external .xyz/.mol valency & clash sanitization, Mendeleev dynamic mass tables,
and automated polling of isomer wavefunctions (.gbw) from landscape.h5.

Authoritative Standards:
- Method Matrix: Stage 1.0 - 2.0 Geometry Intake & Provenance Verification
- Method Matrix §8B / §8B.6 State Chaining & Wavefunction (.gbw) Polling
- Anti-Spoofing Protocol v2 (No mocks, authentic physics)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from Libraries.cochem_torq_vault import (
    ATOMIC_NUMBERS,
    CIAAW_ISOTOPIC_MASSES,
    TorqVaultGateway,
    compute_coordinate_hash,
    compute_sha256_hash,
    fetch_topos_matrices,
    get_atomic_mass,
    get_atomic_number,
    intake_external_geometry,
    parse_external_mol,
    parse_external_xyz,
    poll_isomer_wavefunctions,
    sanitize_geometry_valency_and_clashes,
    standardize_geometry_arrow,
    standardize_geometry_dataframe,
)

try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        MissingDataError,
        ProvenanceErrorCode,
    )
except ImportError:
    from Libraries.cochem_torq_vault import (  # type: ignore[assignment]
        CoChemIntegrityError,
        MissingDataError,
        ProvenanceErrorCode,
    )


# ==============================================================================
# 1. Dynamic Mendeleev Mass & Atomic Number Verification
# ==============================================================================

def test_mendeleev_exact_masses_and_isotopes() -> None:
    """Verifies dynamic Mendeleev property retrieval for elements and isotopes."""
    assert CIAAW_ISOTOPIC_MASSES["H"] == pytest.approx(1.00782503223, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["C"] == 12.00000000000
    assert CIAAW_ISOTOPIC_MASSES["O"] == pytest.approx(15.99491461957, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["N"] == pytest.approx(14.00307400443, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["F"] == pytest.approx(18.99840316273, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["Cl"] == pytest.approx(34.96885271, rel=1e-7)

    # Isotopes
    assert CIAAW_ISOTOPIC_MASSES["D"] == pytest.approx(2.01410177812, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["2H"] == pytest.approx(2.01410177812, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["T"] == pytest.approx(3.01604928132, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["3H"] == pytest.approx(3.01604928132, rel=1e-9)
    assert CIAAW_ISOTOPIC_MASSES["13C"] == pytest.approx(13.003354835, rel=1e-6)
    assert CIAAW_ISOTOPIC_MASSES["18O"] == pytest.approx(17.9991604, rel=1e-6)

    # Helper function
    assert get_atomic_mass("C") == 12.0
    assert get_atomic_mass("H") == pytest.approx(1.00782503223, rel=1e-9)

    # Mapping protocol
    assert "C" in CIAAW_ISOTOPIC_MASSES
    assert "InvalidElement" not in CIAAW_ISOTOPIC_MASSES
    assert len(CIAAW_ISOTOPIC_MASSES) > 0
    assert list(iter(CIAAW_ISOTOPIC_MASSES))


def test_dynamic_atomic_numbers() -> None:
    """Verifies atomic numbers mapping."""
    assert ATOMIC_NUMBERS["H"] == 1
    assert ATOMIC_NUMBERS["D"] == 1
    assert ATOMIC_NUMBERS["C"] == 6
    assert ATOMIC_NUMBERS["13C"] == 6
    assert ATOMIC_NUMBERS["N"] == 7
    assert ATOMIC_NUMBERS["O"] == 8
    assert ATOMIC_NUMBERS["18O"] == 8
    assert ATOMIC_NUMBERS["F"] == 9
    assert ATOMIC_NUMBERS["Cl"] == 17

    assert get_atomic_number("C") == 6
    assert get_atomic_number("O") == 8

    # Mapping protocol
    assert "C" in ATOMIC_NUMBERS
    assert "InvalidSym" not in ATOMIC_NUMBERS
    assert len(ATOMIC_NUMBERS) == 118


# ==============================================================================
# 2. Cryptographic Hashing & Coordinate Integrity
# ==============================================================================

def test_compute_sha256_and_coordinate_hash() -> None:
    """Verifies deterministic SHA-256 and coordinate hashes."""
    data_str = "CoChem-TORQ Test String"
    hash_str = compute_sha256_hash(data_str)
    assert len(hash_str) == 64
    assert hash_str == compute_sha256_hash(data_str.encode("utf-8"))

    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    c_hash1 = compute_coordinate_hash(coords, precision=6)
    c_hash2 = compute_coordinate_hash(coords + 1e-8, precision=6)
    assert len(c_hash1) == 64
    assert c_hash1 == c_hash2  # Due to precision rounding


# ==============================================================================
# 3. Geometry Standardization
# ==============================================================================

def test_standardize_geometry_dataframe_and_arrow() -> None:
    """Verifies DataFrame and PyArrow standardization with proper provenance tags."""
    symbols = ["C", "H", "H", "H", "O", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
        [0.0, 0.0, 1.5],
    ], dtype=np.float64)

    df = standardize_geometry_dataframe(symbols, coords, provenance_tag="[D]")
    assert len(df) == 6
    assert list(df.columns) == [
        "atom_index",
        "symbol",
        "atomic_number",
        "x",
        "y",
        "z",
        "mass_amu",
        "provenance",
    ]
    assert df["symbol"].iloc[0] == "C"
    assert df["atomic_number"].iloc[0] == 6
    assert df["provenance"].iloc[0] == "[D]"
    assert df["mass_amu"].iloc[0] == 12.0

    arrow_table = standardize_geometry_arrow(df)
    assert isinstance(arrow_table, pa.Table)
    assert arrow_table.num_rows == 6

    # Shape mismatch error
    with pytest.raises(ValueError, match="Coordinate shape mismatch"):
        standardize_geometry_dataframe(symbols, coords[:3])


# ==============================================================================
# 4. External Parsing & Valency/Clash Sanitization (.xyz & .mol)
# ==============================================================================

def test_parse_external_xyz_valid() -> None:
    """Verifies parsing of standard Cartesian XYZ format."""
    xyz_content = """3
Water molecule [D]
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""
    parsed = parse_external_xyz(xyz_content, sanitize=True)
    assert parsed["atom_count"] == 3
    assert parsed["symbols"] == ["O", "H", "H"]
    assert parsed["coordinates"].shape == (3, 3)
    assert parsed["masses"][0] == pytest.approx(15.99491461957, rel=1e-9)
    assert parsed["atomic_numbers"][0] == 8
    assert parsed["provenance"] == "[D]"
    assert len(parsed["sha256_hash"]) == 64
    assert isinstance(parsed["dataframe"], pd.DataFrame)
    assert isinstance(parsed["arrow_table"], pa.Table)


def test_parse_external_xyz_file_path(tmp_path: Path) -> None:
    """Verifies reading XYZ directly from file path."""
    xyz_file = tmp_path / "water.xyz"
    xyz_file.write_text("""3
Water
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""", encoding="utf-8")

    parsed = parse_external_xyz(xyz_file, sanitize=True)
    assert parsed["atom_count"] == 3
    assert parsed["symbols"] == ["O", "H", "H"]


def test_parse_external_xyz_clash_detection() -> None:
    """Verifies detection of severe steric clashes (< 0.4 Å)."""
    clash_xyz = """2
Severe clash
C  0.000000  0.000000  0.000000
C  0.000000  0.000000  0.100000
"""
    with pytest.raises(CoChemIntegrityError) as exc_info:
        parse_external_xyz(clash_xyz, sanitize=True)
    assert exc_info.value.error_code == ProvenanceErrorCode.PATHOLOGY_CLASH


def test_parse_external_xyz_corrupt_format() -> None:
    """Verifies error handling on corrupted XYZ files."""
    with pytest.raises(CoChemIntegrityError) as exc_info:
        parse_external_xyz("NOT A VALID XYZ")
    assert exc_info.value.error_code == ProvenanceErrorCode.INTEGRITY_VIOLATION

    with pytest.raises(MissingDataError) as exc_info2:
        parse_external_xyz("")
    assert exc_info2.value.error_code == ProvenanceErrorCode.MISSING_DATA


def test_parse_external_mol_v2000() -> None:
    """Verifies parsing of standard MDL Molfile V2000 format."""
    mol_v2000 = """Formaldehyde
  ChemDraw08272619002D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.2100    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9400   -0.5400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9400   -0.5400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  1  3  1  0
  1  4  1  0
M  END
"""
    parsed = parse_external_mol(mol_v2000, sanitize=True)
    assert parsed["atom_count"] == 4
    assert parsed["symbols"] == ["C", "O", "H", "H"]
    assert list(parsed["atomic_numbers"]) == [6, 8, 1, 1]
    assert len(parsed["bonds"]) == 3
    assert parsed["provenance"] == "[D]"
    assert len(parsed["sha256_hash"]) == 64


def test_parse_external_mol_v3000() -> None:
    """Verifies parsing of MDL Molfile V3000 format."""
    mol_v3000 = """Formaldehyde V3000
  OpenBabel

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 4 3 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 0.0000 0.0000 0.0000 0
M  V30 2 O 0.0000 1.2100 0.0000 0
M  V30 3 H -0.9400 -0.5400 0.0000 0
M  V30 4 H 0.9400 -0.5400 0.0000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 2 1 2
M  V30 2 1 1 3
M  V30 3 1 1 4
M  V30 END BOND
M  V30 END CTAB
M  END
"""
    parsed = parse_external_mol(mol_v3000, sanitize=True)
    assert parsed["atom_count"] == 4
    assert parsed["symbols"] == ["C", "O", "H", "H"]
    assert len(parsed["bonds"]) == 3


def test_intake_external_geometry_dispatcher() -> None:
    """Verifies universal external intake dispatcher."""
    xyz_content = """2
Diatomic [D]
C  0.0000  0.0000  0.0000
O  0.0000  0.0000  1.1300
"""
    parsed = intake_external_geometry(xyz_content)
    assert parsed["atom_count"] == 2
    assert parsed["symbols"] == ["C", "O"]


# ==============================================================================
# 5. Native Database Intake & Wavefunction Chaining (landscape.h5)
# ==============================================================================

def test_fetch_topos_matrices_and_polling(tmp_path: Path) -> None:
    """Verifies querying landscape.h5 for conformers and wavefunction pointers."""
    h5_path = tmp_path / "landscape.h5"
    gbw_file = tmp_path / "conf_001.gbw"
    gbw_file.write_bytes(b"GBW_WAVEFUNCTION_BINARY_DATA")

    with h5py.File(h5_path, "w") as fp:
        conf_grp = fp.create_group("conformers")
        c1 = conf_grp.create_group("conf_001")
        c1.create_dataset(
            "coordinates",
            data=np.array([[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]], dtype=np.float64),
        )
        c1.create_dataset("symbols", data=[b"C", b"O"])
        c1.attrs["energy_hartree"] = -113.82910
        c1.attrs["gbw_path"] = str(gbw_file)

        c2 = conf_grp.create_group("conf_002")
        c2.create_dataset(
            "coordinates",
            data=np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]], dtype=np.float64),
        )
        c2.create_dataset("symbols", data=[b"C", b"O"])
        c2.attrs["energy_hartree"] = -113.82500
        c2.attrs["gbw_path"] = "/nonexistent/conf_002.gbw"

    # Fetch specific conformer
    result = fetch_topos_matrices(h5_path, conformer_id="conf_001")
    assert result["conformer_id"] == "conf_001"
    assert result["symbols"] == ["C", "O"]
    assert result["energy_hartree"] == pytest.approx(-113.82910, rel=1e-6)
    assert result["gbw_path"] == str(gbw_file)
    assert result["provenance"] == "[M]"

    # Poll all isomers
    isomers = poll_isomer_wavefunctions(h5_path, require_gbw=False)
    assert len(isomers) == 2
    assert isomers[0]["conformer_id"] == "conf_001"
    assert isomers[0]["gbw_exists"] is True
    assert isomers[1]["conformer_id"] == "conf_002"
    assert isomers[1]["gbw_exists"] is False

    # Missing database
    with pytest.raises(MissingDataError):
        fetch_topos_matrices(tmp_path / "missing.h5")


# ==============================================================================
# 6. TorqVaultGateway Governor Class
# ==============================================================================

def test_torq_vault_gateway(tmp_path: Path) -> None:
    """Verifies TorqVaultGateway governor methods including parquet and xyz export."""
    gateway = TorqVaultGateway(artifacts_dir=tmp_path / "artifacts", sanitize=True)

    xyz_content = """3
Water [D]
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""
    geom_data = gateway.intake_external(xyz_content)
    assert geom_data["atom_count"] == 3

    # Export to Parquet
    pq_path = tmp_path / "water.parquet"
    out_pq = gateway.export_to_parquet(geom_data, pq_path)
    assert out_pq.exists()
    read_pq = pd.read_parquet(out_pq)
    assert len(read_pq) == 3
    assert "provenance" in read_pq.columns

    # Export to XYZ
    out_xyz_path = tmp_path / "exported_water.xyz"
    out_xyz = gateway.export_to_xyz(geom_data, out_xyz_path)
    assert out_xyz.exists()
    assert "3" in out_xyz.read_text(encoding="utf-8")


# ==============================================================================
# 7. Anti-Spoofing & AST Zero-Mock Verification
# ==============================================================================

def test_anti_spoofing_ast_audit() -> None:
    """AST audit asserting zero mocks, stubs, NotImplementedError, or empty pass blocks."""
    vault_file = (
        Path(__file__).resolve().parent.parent
        / "Libraries"
        / "cochem_torq_vault.py"
    )
    assert vault_file.exists(), f"Vault file not found at {vault_file}"
    tree = ast.parse(vault_file.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        # Disallow mock imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "mock" not in alias.name.lower(), (
                    f"Forbidden mock import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "mock" not in node.module.lower(), (
                    f"Forbidden mock import from {node.module}"
                )

        # Disallow NotImplementedError
        if isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Name) and node.exc.id == "NotImplementedError":
                pytest.fail("Found forbidden NotImplementedError raise in vault script")
            if isinstance(node.exc, ast.Call) and getattr(node.exc.func, "id", None) == "NotImplementedError":
                pytest.fail("Found forbidden NotImplementedError call in vault script")

        # Disallow empty pass blocks in functions
        if isinstance(node, ast.FunctionDef):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                pytest.fail(f"Found empty pass body in function '{node.name}'")
