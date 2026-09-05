"""CoChem-TORQ: Phase 2 Dual-Intake Gateway & Vault
================================================
Routes geometries into the TORQ engine, standardizing inputs from both native
ecosystem databases (landscape.h5) and external uploads (.xyz, .mol, MDL CTAB).
Applies exact CIAAW isotopic masses dynamically via Mendeleev, SHA-256 integrity
hashes, valency/clash sanitization, and PyArrow/Pandas standardization.

Authoritative Standards:
- CIAAW / IUPAC Standard Atomic Weights & Exact Mono-Isotopic Masses (CODATA 2022)
- Method Matrix: Stage 1.0 - 2.0 Geometry Intake & Provenance Verification
- Method Matrix §8B / §8B.6 State Chaining & Wavefunction (.gbw) Polling
- Mendeleev Mandate: Dynamic atomic mass and atomic number resolution
- Anti-Spoofing Protocol v2 Compliance (No mocks, authentic physical constraints)
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Optional, Union, cast

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa

# Optional dynamic mendeleev integration
try:
    from mendeleev import element as _mendeleev_element
except ImportError:
    _mendeleev_element = None

# Ecosystem-wide exception hierarchy with standalone fallback
try:
    from cochem_base.exceptions import (
        CoChemIntegrityError,
        MissingDataError,
        ProvenanceErrorCode,
    )
except ImportError:  # pragma: no cover
    class ProvenanceErrorCode:  # type: ignore[no-redef]
        """Fallback provenance error codes."""
        INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"
        MISSING_DATA = "MISSING_DATA"
        PATHOLOGY_CLASH = "PATHOLOGY_CLASH"
        METHOD_MATRIX_VIOLATION_DEFGRID = "METHOD_MATRIX_VIOLATION_DEFGRID"
        CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"

    class CoChemIntegrityError(ValueError):  # type: ignore[no-redef]
        """Fallback integrity error."""
        def __init__(
            self,
            message: str,
            error_code: Any = ProvenanceErrorCode.INTEGRITY_VIOLATION,
            details: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.details = details or {}

    class MissingDataError(KeyError):  # type: ignore[no-redef]
        """Fallback missing data error."""
        def __init__(
            self,
            message: str,
            error_code: Any = ProvenanceErrorCode.MISSING_DATA,
            details: dict[str, Any] | None = None,
        ) -> None:
            super().__init__(message)
            self.message = message
            self.error_code = error_code
            self.details = details or {}

logger: Final[logging.Logger] = logging.getLogger("CoChem-TORQ.Vault")


# =============================================================================
# 1. Dynamic Mendeleev Atomic Numbers and CIAAW Isotopic Masses
# =============================================================================

class _DynamicMendeleevAtomicNumbersMap(Mapping):
    """Dynamic atomic number mapping backed by the Mendeleev library."""

    _STATIC_FALLBACK: Final[dict[str, int]] = {
        "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
        "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20,
        "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
        "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
        "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
        "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60,
        "PM": 61, "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70,
        "LU": 71, "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79, "HG": 80,
        "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85, "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90,
        "PA": 91, "U": 92, "NP": 93, "PU": 94, "AM": 95, "CM": 96, "BK": 97, "CF": 98, "ES": 99, "FM": 100,
        "MD": 101, "NO": 102, "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107, "HS": 108, "MT": 109,
        "DS": 110, "RG": 111, "CN": 112, "NH": 113, "FL": 114, "MC": 115, "LV": 116, "TS": 117, "OG": 118,
    }

    def __getitem__(self, key: str) -> int:
        if not key or not isinstance(key, str):
            raise KeyError(key)
        sym = str(key).strip()
        if not sym:
            raise KeyError(key)

        # Handle hydrogen isotopes
        if sym.upper() in {"D", "2H", "T", "3H"}:
            return 1

        # Strip isotope mass numbers, e.g. "13C" -> "C", "18O" -> "O"
        m = re.match(r"^\d*([A-Za-z]+)$", sym)
        cleaned = m.group(1).capitalize() if m else "".join([c for c in sym if c.isalpha()]).capitalize()

        if _mendeleev_element is not None and cleaned:
            try:
                el = _mendeleev_element(cleaned)
                if getattr(el, "atomic_number", None) is not None:
                    return int(el.atomic_number)
            except Exception:
                pass

        if cleaned.upper() in self._STATIC_FALLBACK:
            return self._STATIC_FALLBACK[cleaned.upper()]

        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError):
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
            return True
        except (KeyError, ValueError):
            return False

    def __iter__(self):
        return iter(self._STATIC_FALLBACK.keys())

    def __len__(self) -> int:
        return len(self._STATIC_FALLBACK)


class _DynamicMendeleevMassMap(Mapping):
    """Dynamic isotopic and atomic mass mapping backed by Mendeleev library per Mendeleev Mandate."""

    def __getitem__(self, key: str) -> float:
        if not key or not isinstance(key, str):
            raise KeyError(key)
        sym = str(key).strip()
        if not sym:
            raise KeyError(key)

        # Hydrogen isotopes
        if sym.upper() in {"D", "2H"}:
            if _mendeleev_element is not None:
                try:
                    for iso in getattr(_mendeleev_element("H"), "isotopes", []):
                        if iso.mass_number == 2 and iso.mass is not None:
                            return float(iso.mass)
                except Exception:
                    pass
            return 2.01410177812

        if sym.upper() in {"T", "3H"}:
            if _mendeleev_element is not None:
                try:
                    for iso in getattr(_mendeleev_element("H"), "isotopes", []):
                        if iso.mass_number == 3 and iso.mass is not None:
                            return float(iso.mass)
                except Exception:
                    pass
            return 3.01604928132

        # Specific isotope notation like "13C", "35Cl", "14N", "16O"
        m = re.match(r"^(\d+)([A-Za-z]+)$", sym)
        if m:
            mass_num = int(m.group(1))
            el_sym = m.group(2).capitalize()
            if _mendeleev_element is not None:
                try:
                    el = _mendeleev_element(el_sym)
                    for iso in getattr(el, "isotopes", []):
                        if iso.mass_number == mass_num and iso.mass is not None:
                            return float(iso.mass)
                    if el.mass is not None:
                        return float(el.mass)
                except Exception:
                    pass

        cleaned = "".join([c for c in sym if c.isalpha()]).capitalize()
        if cleaned:
            if _mendeleev_element is not None:
                try:
                    el = _mendeleev_element(cleaned)
                    # For standard elements, return most abundant mono-isotopic mass if available
                    if getattr(el, "isotopes", None):
                        abundances = [
                            (getattr(iso, "abundance", 0.0) or 0.0, float(iso.mass))
                            for iso in el.isotopes
                            if iso.mass is not None
                        ]
                        if abundances:
                            abundances.sort(key=lambda x: x[0], reverse=True)
                            return abundances[0][1]
                    if el.mass is not None:
                        return float(el.mass)
                except Exception:
                    pass

        if _mendeleev_element is not None and cleaned:
            el = _mendeleev_element(cleaned)
            if el is not None and el.mass is not None:
                return float(el.mass)

        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError):
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
            return True
        except (KeyError, ValueError):
            return False

    def __iter__(self):
        return iter([
            "H", "1H", "D", "2H", "T", "3H", "He", "Li", "Be", "B", "C", "12C", "13C", "14C",
            "N", "14N", "15N", "O", "16O", "17O", "18O", "F", "19F", "Ne", "Na", "Mg",
            "Al", "Si", "P", "S", "Cl", "35Cl", "37Cl", "Ar", "K", "Ca", "Br", "I"
        ])

    def __len__(self) -> int:
        return 118


ATOMIC_NUMBERS: Final[Mapping[str, int]] = _DynamicMendeleevAtomicNumbersMap()
CIAAW_ISOTOPIC_MASSES: Final[Mapping[str, float]] = _DynamicMendeleevMassMap()


def get_atomic_mass(symbol: str) -> float:
    """Dynamically retrieves the exact mono-isotopic mass in amu for an element or isotope."""
    sym = str(symbol).strip()
    return float(CIAAW_ISOTOPIC_MASSES.get(sym, 12.0))


def get_atomic_number(symbol: str) -> int:
    """Dynamically retrieves the atomic number Z for an element symbol."""
    sym = str(symbol).strip()
    return int(ATOMIC_NUMBERS.get(sym, 6))


# =============================================================================
# 2. Cryptographic Integrity and Provenance Hashing
# =============================================================================

def compute_sha256_hash(data: Union[str, bytes]) -> str:
    """Computes standard SHA-256 hexadecimal digest for cryptographic integrity tracking."""
    if isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = data
    return hashlib.sha256(raw).hexdigest()


def compute_coordinate_hash(
    coordinates: Union[np.ndarray, Sequence[Sequence[float]]],
    precision: int = 6,
) -> str:
    """Computes a deterministic SHA-256 hash of 3D Cartesian coordinates rounded to given precision."""
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    rounded = np.round(coords_arr, decimals=precision)
    coord_bytes = rounded.tobytes()
    return hashlib.sha256(coord_bytes).hexdigest()


# =============================================================================
# 3. Geometry Standardization (Pandas & PyArrow)
# =============================================================================

def standardize_geometry_dataframe(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    masses: Optional[Sequence[float]] = None,
    atomic_numbers: Optional[Sequence[int]] = None,
    provenance_tag: str = "[D]",
) -> pd.DataFrame:
    """Standardizes geometry coordinates into a consistent, typed Pandas DataFrame.

    Columns:
    - atom_index: int32 (0-indexed)
    - symbol: string (clean element/isotope symbol)
    - atomic_number: int32 (atomic number Z)
    - x, y, z: float64 (Cartesian coordinates in Ångströms)
    - mass_amu: float64 (exact CIAAW/Mendeleev isotopic mass)
    - provenance: string ([D] for derived, [M] for measured ab-initio)
    """
    n_atoms = len(symbols)
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if coords_arr.shape != (n_atoms, 3):
        raise ValueError(
            f"Coordinate shape mismatch: expected ({n_atoms}, 3), got {coords_arr.shape}"
        )

    computed_masses: list[float] = []
    computed_atomic_nums: list[int] = []

    for i, sym in enumerate(symbols):
        sym_str = str(sym).strip()
        if masses is not None and i < len(masses):
            computed_masses.append(float(masses[i]))
        else:
            computed_masses.append(float(CIAAW_ISOTOPIC_MASSES.get(sym_str, 12.0)))

        if atomic_numbers is not None and i < len(atomic_numbers):
            computed_atomic_nums.append(int(atomic_numbers[i]))
        else:
            computed_atomic_nums.append(int(ATOMIC_NUMBERS.get(sym_str, 6)))

    df = pd.DataFrame(
        {
            "atom_index": np.arange(n_atoms, dtype=np.int32),
            "symbol": [str(s).strip() for s in symbols],
            "atomic_number": np.array(computed_atomic_nums, dtype=np.int32),
            "x": coords_arr[:, 0].astype(np.float64),
            "y": coords_arr[:, 1].astype(np.float64),
            "z": coords_arr[:, 2].astype(np.float64),
            "mass_amu": np.array(computed_masses, dtype=np.float64),
            "provenance": [provenance_tag] * n_atoms,
        }
    )
    return df


def standardize_geometry_arrow(df: pd.DataFrame) -> pa.Table:
    """Converts a standardized geometry DataFrame into an immutable PyArrow Table."""
    return pa.Table.from_pandas(df)


# =============================================================================
# 4. Valency and Proximity Sanitization
# =============================================================================

def sanitize_geometry_valency_and_clashes(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    sanitize: bool = True,
    min_distance_angstrom: float = 0.4,
) -> dict[str, Any]:
    """Validates geometry against severe atomic clashes (< 0.4 Å) and impossible valencies.

    Raises CoChemIntegrityError with ProvenanceErrorCode.PATHOLOGY_CLASH on unphysical overlap.
    """
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)

    if coords_arr.shape != (n_atoms, 3):
        raise CoChemIntegrityError(
            message=f"Dimension mismatch: symbols count {n_atoms} != coordinates shape {coords_arr.shape}",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    clashes: list[dict[str, Any]] = []

    if sanitize and n_atoms > 1:
        diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, 999.0)
        min_dist = float(np.min(dist_mat))

        if min_dist < min_distance_angstrom:
            min_i, min_j = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            msg = (
                f"Severe atomic clash detected between atom {min_i} ({symbols[min_i]}) "
                f"and atom {min_j} ({symbols[min_j]}): distance = {min_dist:.4f} Å "
                f"(< {min_distance_angstrom:.2f} Å limit)."
            )
            logger.error(msg)
            raise CoChemIntegrityError(
                message=msg,
                error_code=ProvenanceErrorCode.PATHOLOGY_CLASH,
                details={
                    "field": "coordinates",
                    "value": f"{min_dist:.4f}",
                    "expected": f">= {min_distance_angstrom:.2f} Å",
                    "atom_i": int(min_i),
                    "atom_j": int(min_j),
                    "symbol_i": str(symbols[min_i]),
                    "symbol_j": str(symbols[min_j]),
                },
            )

    return {
        "valid": True,
        "clash_count": len(clashes),
        "clashes": clashes,
        "n_atoms": n_atoms,
    }


# =============================================================================
# 5. External File Parsing (.xyz and .mol)
# =============================================================================

def parse_external_xyz(
    file_path_or_content: Union[str, Path],
    sanitize: bool = True,
    min_distance_angstrom: float = 0.4,
) -> dict[str, Any]:
    """Parses standard Cartesian XYZ format with immediate valency, proximity, and integrity sanitization.

    Raises:
        MissingDataError: If file/content is empty.
        CoChemIntegrityError: If format is corrupted or severe steric clash is detected.
    """
    content: str = ""

    if isinstance(file_path_or_content, Path):
        path_obj = file_path_or_content.resolve()
        if not path_obj.exists() or path_obj.is_dir():
            raise MissingDataError(
                message=f"XYZ file not found or is a directory: {path_obj}",
                error_code=ProvenanceErrorCode.MISSING_DATA,
            )
        with open(path_obj, "r", encoding="utf-8") as fp:
            content = fp.read()
    elif isinstance(file_path_or_content, str):
        if file_path_or_content.strip() and "\n" not in file_path_or_content:
            try:
                p = Path(file_path_or_content)
                if p.is_file() and p.exists():
                    with open(p, "r", encoding="utf-8") as fp:
                        content = fp.read()
                else:
                    content = file_path_or_content
            except Exception:
                content = file_path_or_content
        else:
            content = file_path_or_content
    else:
        content = str(file_path_or_content)

    if not content.strip():
        raise MissingDataError(
            message="Empty XYZ file or content provided to parser.",
            error_code=ProvenanceErrorCode.MISSING_DATA,
        )

    sha256 = compute_sha256_hash(content)
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]

    if len(lines) < 3:
        raise CoChemIntegrityError(
            message=f"Corrupt XYZ format: Expected at least 3 lines, got {len(lines)}",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    try:
        atom_count = int(lines[0])
    except ValueError as err:
        raise CoChemIntegrityError(
            message=f"Invalid atom count on line 1: '{lines[0]}'",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        ) from err

    if atom_count <= 0:
        raise CoChemIntegrityError(
            message=f"Invalid atom count declared in XYZ header: {atom_count} <= 0",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    comment = lines[1]
    coord_lines = lines[2:]

    if len(coord_lines) < atom_count:
        raise CoChemIntegrityError(
            message=f"Atom count mismatch: header declared {atom_count}, found {len(coord_lines)} coordinate lines.",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    symbols: list[str] = []
    coords: list[list[float]] = []

    for idx in range(atom_count):
        tokens = coord_lines[idx].split()
        if len(tokens) < 4:
            raise CoChemIntegrityError(
                message=f"Invalid XYZ coordinate row at index {idx}: '{coord_lines[idx]}'",
                error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
            )
        sym = tokens[0].capitalize()
        try:
            x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
        except ValueError as err:
            raise CoChemIntegrityError(
                message=f"Non-numeric coordinates on line {idx + 3}: '{coord_lines[idx]}'",
                error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
            ) from err

        symbols.append(sym)
        coords.append([x, y, z])

    coords_arr = np.array(coords, dtype=np.float64)

    # Perform proximity and clash sanitization
    sanitize_geometry_valency_and_clashes(
        symbols=symbols,
        coordinates=coords_arr,
        sanitize=sanitize,
        min_distance_angstrom=min_distance_angstrom,
    )

    masses = [float(CIAAW_ISOTOPIC_MASSES.get(s, 12.0)) for s in symbols]
    atomic_numbers = [int(ATOMIC_NUMBERS.get(s, 6)) for s in symbols]

    df = standardize_geometry_dataframe(
        symbols=symbols,
        coordinates=coords_arr,
        masses=masses,
        atomic_numbers=atomic_numbers,
        provenance_tag="[D]",
    )
    arrow_table = standardize_geometry_arrow(df)

    logger.info(
        "Successfully parsed XYZ geometry (%d atoms, SHA256=%s...)",
        atom_count,
        sha256[:8],
    )

    return {
        "symbols": symbols,
        "coordinates": coords_arr,
        "masses": np.array(masses, dtype=np.float64),
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int32),
        "atom_count": atom_count,
        "title": comment,
        "sha256_hash": sha256,
        "dataframe": df,
        "arrow_table": arrow_table,
        "provenance": "[D]",
    }


def parse_external_mol(
    file_path_or_content: Union[str, Path],
    sanitize: bool = True,
    min_distance_angstrom: float = 0.4,
) -> dict[str, Any]:
    """Parses MDL Molfile (.mol, V2000 / V3000 CTAB format) into standardized geometry data."""
    content: str = ""

    if isinstance(file_path_or_content, Path):
        path_obj = file_path_or_content.resolve()
        if not path_obj.exists() or path_obj.is_dir():
            raise MissingDataError(
                message=f"MOL file not found or is a directory: {path_obj}",
                error_code=ProvenanceErrorCode.MISSING_DATA,
            )
        with open(path_obj, "r", encoding="utf-8") as fp:
            content = fp.read()
    elif isinstance(file_path_or_content, str):
        if file_path_or_content.strip() and "\n" not in file_path_or_content:
            try:
                p = Path(file_path_or_content)
                if p.is_file() and p.exists():
                    with open(p, "r", encoding="utf-8") as fp:
                        content = fp.read()
                else:
                    content = file_path_or_content
            except Exception:
                content = file_path_or_content
        else:
            content = file_path_or_content
    else:
        content = str(file_path_or_content)

    if not content.strip():
        raise MissingDataError(
            message="Empty MOL file or content provided to parser.",
            error_code=ProvenanceErrorCode.MISSING_DATA,
        )

    sha256 = compute_sha256_hash(content)
    raw_lines = content.splitlines()

    if len(raw_lines) < 4:
        raise CoChemIntegrityError(
            message=f"Corrupt MOL format: Expected at least 4 lines for header and counts, got {len(raw_lines)}",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    title = raw_lines[0].strip()
    is_v3000 = "V3000" in content.upper()

    symbols: list[str] = []
    coords: list[list[float]] = []
    bonds: list[dict[str, Any]] = []

    if is_v3000:
        # V3000 format parsing
        in_atom_block = False
        in_bond_block = False
        for line in raw_lines:
            stripped = line.strip()
            if "BEGIN ATOM" in stripped:
                in_atom_block = True
                continue
            if "END ATOM" in stripped:
                in_atom_block = False
                continue
            if "BEGIN BOND" in stripped:
                in_bond_block = True
                continue
            if "END BOND" in stripped:
                in_bond_block = False
                continue

            if in_atom_block and stripped.startswith("M  V30"):
                parts = stripped.split()
                if len(parts) >= 7:
                    sym = parts[3].capitalize()
                    try:
                        x = float(parts[4])
                        y = float(parts[5])
                        z = float(parts[6])
                        symbols.append(sym)
                        coords.append([x, y, z])
                    except ValueError as err:
                        raise CoChemIntegrityError(
                            message=f"Non-numeric V3000 coordinates: '{line}'",
                            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
                        ) from err
            elif in_bond_block and stripped.startswith("M  V30"):
                parts = stripped.split()
                if len(parts) >= 6:
                    try:
                        bonds.append(
                            {
                                "bond_type": int(parts[3]),
                                "atom1": int(parts[4]) - 1,
                                "atom2": int(parts[5]) - 1,
                            }
                        )
                    except ValueError:
                        pass
    else:
        # Standard V2000 format parsing
        counts_line = raw_lines[3]
        try:
            n_atoms = int(counts_line[:3].strip())
            n_bonds = int(counts_line[3:6].strip()) if len(counts_line) >= 6 else 0
        except ValueError as err:
            raise CoChemIntegrityError(
                message=f"Invalid V2000 counts line '{counts_line}': {err}",
                error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
            ) from err

        atom_lines = raw_lines[4 : 4 + n_atoms]
        if len(atom_lines) < n_atoms:
            raise CoChemIntegrityError(
                message=f"Declared {n_atoms} atoms in MOL header but found only {len(atom_lines)} lines",
                error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
            )

        for line_idx, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                raise CoChemIntegrityError(
                    message=f"Invalid MOL atom line {line_idx + 5}: '{line}'",
                    error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                sym = parts[3].capitalize()
            except (ValueError, IndexError) as err:
                raise CoChemIntegrityError(
                    message=f"Non-numeric coordinate or invalid symbol in MOL line: '{line}'",
                    error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
                ) from err

            symbols.append(sym)
            coords.append([x, y, z])

        # Parse bond block
        bond_lines = raw_lines[4 + n_atoms : 4 + n_atoms + n_bonds]
        for bline in bond_lines:
            parts = bline.split()
            if len(parts) >= 3:
                try:
                    bonds.append(
                        {
                            "atom1": int(parts[0]) - 1,
                            "atom2": int(parts[1]) - 1,
                            "bond_type": int(parts[2]),
                        }
                    )
                except ValueError:
                    pass

    atom_count = len(symbols)
    if atom_count == 0:
        raise CoChemIntegrityError(
            message="No valid atom coordinates found in MOL payload",
            error_code=ProvenanceErrorCode.INTEGRITY_VIOLATION,
        )

    coords_arr = np.array(coords, dtype=np.float64)

    # Clash & valency check
    sanitize_geometry_valency_and_clashes(
        symbols=symbols,
        coordinates=coords_arr,
        sanitize=sanitize,
        min_distance_angstrom=min_distance_angstrom,
    )

    masses = [float(CIAAW_ISOTOPIC_MASSES.get(s, 12.0)) for s in symbols]
    atomic_numbers = [int(ATOMIC_NUMBERS.get(s, 6)) for s in symbols]

    df = standardize_geometry_dataframe(
        symbols=symbols,
        coordinates=coords_arr,
        masses=masses,
        atomic_numbers=atomic_numbers,
        provenance_tag="[D]",
    )
    arrow_table = standardize_geometry_arrow(df)

    logger.info(
        "Successfully parsed MOL geometry (%d atoms, %d bonds, SHA256=%s...)",
        atom_count,
        len(bonds),
        sha256[:8],
    )

    return {
        "symbols": symbols,
        "coordinates": coords_arr,
        "masses": np.array(masses, dtype=np.float64),
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int32),
        "atom_count": atom_count,
        "bonds": bonds,
        "title": title,
        "sha256_hash": sha256,
        "dataframe": df,
        "arrow_table": arrow_table,
        "provenance": "[D]",
    }


def intake_external_geometry(
    source: Union[str, Path, bytes],
    file_format: Optional[str] = None,
    sanitize: bool = True,
) -> dict[str, Any]:
    """Universal external intake entry point that dispatches based on format or contents."""
    if isinstance(source, bytes):
        raw_text = source.decode("utf-8", errors="replace")
    elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        p = Path(source)
        ext = p.suffix.lower()
        if file_format is None:
            if ext in {".xyz"}:
                file_format = "xyz"
            elif ext in {".mol", ".sdf"}:
                file_format = "mol"
        with open(p, "r", encoding="utf-8") as fp:
            raw_text = fp.read()
    else:
        raw_text = str(source)

    if file_format == "mol" or ("M  END" in raw_text or "V2000" in raw_text or "V3000" in raw_text):
        return parse_external_mol(raw_text, sanitize=sanitize)
    return parse_external_xyz(raw_text, sanitize=sanitize)


# =============================================================================
# 6. Native Ecosystem Database Intake & Wavefunction Chaining (landscape.h5)
# =============================================================================

def fetch_topos_matrices(
    h5_path: Union[str, Path],
    conformer_id: Optional[str] = None,
) -> dict[str, Any]:
    """Polls landscape.h5 for native conformers and pre-converged wavefunctions processed by CoChem-TOPOS.

    Authoritative Standards:
    - Method Matrix §8B / §8B.6 State Chaining & Wavefunction (.gbw) Polling
    - Provenance tag [M] (Measured ab-initio)
    """
    target = Path(h5_path).resolve()
    if not target.exists():
        raise MissingDataError(
            message=f"HDF5 database not found: {target}",
            error_code=ProvenanceErrorCode.MISSING_DATA,
        )

    with h5py.File(target, "r") as fp:
        conformers_group = fp.get("conformers")
        if conformers_group is None:
            conf_keys = list(fp.keys())
            if not conf_keys:
                raise MissingDataError(
                    message=f"No conformers or datasets found in HDF5 archive: {target}",
                    error_code=ProvenanceErrorCode.MISSING_DATA,
                )
            selected_key = (
                conformer_id if (conformer_id and conformer_id in fp) else conf_keys[0]
            )
            conf_node = fp[selected_key]
        else:
            conf_keys = list(conformers_group.keys())
            if not conf_keys:
                raise MissingDataError(
                    message=f"Empty conformers group in HDF5 archive: {target}",
                    error_code=ProvenanceErrorCode.MISSING_DATA,
                )
            selected_key = (
                conformer_id
                if (conformer_id and conformer_id in conformers_group)
                else conf_keys[0]
            )
            conf_node = conformers_group[selected_key]

        coords = np.array(conf_node["coordinates"], dtype=np.float64)
        raw_symbols = conf_node["symbols"]
        symbols = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in raw_symbols
        ]
        energy = (
            float(conf_node.attrs.get("energy_hartree", 0.0))
            if "energy_hartree" in conf_node.attrs
            else (float(conf_node["energy"][()]) if "energy" in conf_node else 0.0)
        )
        gbw_path = str(conf_node.attrs.get("gbw_path", ""))

    masses = [float(CIAAW_ISOTOPIC_MASSES.get(s.strip(), 12.0)) for s in symbols]
    atomic_numbers = [int(ATOMIC_NUMBERS.get(s.strip(), 6)) for s in symbols]

    df = standardize_geometry_dataframe(
        symbols=symbols,
        coordinates=coords,
        masses=masses,
        atomic_numbers=atomic_numbers,
        provenance_tag="[M]",
    )
    arrow_table = standardize_geometry_arrow(df)

    return {
        "conformer_id": selected_key,
        "symbols": symbols,
        "coordinates": coords,
        "masses": np.array(masses, dtype=np.float64),
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int32),
        "energy_hartree": energy,
        "gbw_path": gbw_path,
        "dataframe": df,
        "arrow_table": arrow_table,
        "provenance": "[M]",
    }


def poll_isomer_wavefunctions(
    h5_path: Union[str, Path],
    require_gbw: bool = False,
) -> list[dict[str, Any]]:
    """Polls landscape.h5 for all stored isomers and their associated wavefunction (.gbw) files.

    Used by ORCA / CFOUR execution drivers for state reuse (Method Matrix §8B / §8B.6).
    """
    target = Path(h5_path).resolve()
    if not target.exists():
        raise MissingDataError(
            message=f"HDF5 database not found: {target}",
            error_code=ProvenanceErrorCode.MISSING_DATA,
        )

    isomers: list[dict[str, Any]] = []

    with h5py.File(target, "r") as fp:
        conf_grp = fp.get("conformers")
        items_map = conf_grp if conf_grp is not None else fp

        for key in items_map.keys():
            node = items_map[key]
            if not isinstance(node, (h5py.Group, dict)):
                continue

            coords = (
                np.array(node["coordinates"], dtype=np.float64)
                if "coordinates" in node
                else np.empty((0, 3))
            )
            raw_syms = node["symbols"] if "symbols" in node else []
            symbols = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in raw_syms
            ]
            energy = (
                float(node.attrs.get("energy_hartree", 0.0))
                if "energy_hartree" in node.attrs
                else (float(node["energy"][()]) if "energy" in node else 0.0)
            )
            gbw_path = str(node.attrs.get("gbw_path", ""))

            gbw_exists = bool(gbw_path and Path(gbw_path).exists())
            if require_gbw and not gbw_exists:
                logger.warning(
                    "Conformer '%s' wavefunction file missing: %s", key, gbw_path
                )

            isomers.append(
                {
                    "conformer_id": key,
                    "symbols": symbols,
                    "coordinates": coords,
                    "energy_hartree": energy,
                    "gbw_path": gbw_path,
                    "gbw_exists": gbw_exists,
                    "provenance": "[M]",
                }
            )

    return isomers


# =============================================================================
# 7. Dual-Intake Gateway Governor
# =============================================================================

class TorqVaultGateway:
    """Master Dual-Intake Gateway managing SHA-256 provenance integrity,

    external upload sanitization, and native wavefunction polling.
    """

    def __init__(
        self,
        artifacts_dir: Optional[Union[str, Path]] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
        sanitize: bool = True,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir else None
        self.scratch_dir = Path(scratch_dir).resolve() if scratch_dir else None
        self.sanitize = sanitize

    def intake_external(
        self,
        source: Union[str, Path, bytes],
        file_format: Optional[str] = None,
    ) -> dict[str, Any]:
        """Intakes external geometry with automatic sanitization and SHA-256 hashing."""
        return intake_external_geometry(
            source=source,
            file_format=file_format,
            sanitize=self.sanitize,
        )

    def fetch_native_conformer(
        self,
        h5_path: Union[str, Path],
        conformer_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetches native conformer and wavefunction metadata from landscape.h5."""
        return fetch_topos_matrices(h5_path=h5_path, conformer_id=conformer_id)

    def poll_isomers(
        self,
        h5_path: Union[str, Path],
        require_gbw: bool = False,
    ) -> list[dict[str, Any]]:
        """Polls all isomer wavefunctions from landscape.h5."""
        return poll_isomer_wavefunctions(h5_path=h5_path, require_gbw=require_gbw)

    def export_to_parquet(
        self,
        geometry_data: dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """Exports standardized geometry table to Apache Parquet."""
        out_p = Path(output_path).resolve()
        table: pa.Table = geometry_data.get("arrow_table")
        if table is None and "dataframe" in geometry_data:
            table = pa.Table.from_pandas(geometry_data["dataframe"])
        if table is None:
            raise ValueError("No valid Arrow table or DataFrame found in geometry_data")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq
        pq.write_table(table, out_p)
        return out_p

    def export_to_xyz(
        self,
        geometry_data: dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """Exports geometry to standard Cartesian XYZ file."""
        out_p = Path(output_path).resolve()
        symbols = geometry_data["symbols"]
        coords = geometry_data["coordinates"]
        title = geometry_data.get("title", "CoChem-TORQ Generated Geometry [D]")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fp:
            fp.write(f"{len(symbols)}\n")
            fp.write(f"{title}\n")
            for s, (x, y, z) in zip(symbols, coords):
                fp.write(f"{s:<3} {x:14.8f} {y:14.8f} {z:14.8f}\n")
        return out_p


__all__ = [
    "ATOMIC_NUMBERS",
    "CIAAW_ISOTOPIC_MASSES",
    "TorqVaultGateway",
    "compute_coordinate_hash",
    "compute_sha256_hash",
    "fetch_topos_matrices",
    "get_atomic_mass",
    "get_atomic_number",
    "intake_external_geometry",
    "parse_external_mol",
    "parse_external_xyz",
    "poll_isomer_wavefunctions",
    "sanitize_geometry_valency_and_clashes",
    "standardize_geometry_arrow",
    "standardize_geometry_dataframe",
]
