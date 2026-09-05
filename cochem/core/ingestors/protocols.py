"""Structural Typing Protocols & Schema Validation for Ingestors.
Strictly adheres to Zero-Mock mandate and Mendeleev dynamic mass invariants.
"""

from __future__ import annotations

import math
import pathlib
from typing import Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cochem.core.mendeleev_invariants import MendeleevInvariantError


class QCValidationError(ValueError):
    """Base domain exception for physical validation failures in quantum chemistry results."""

    pass


class HessianSymmetryError(QCValidationError):
    """Raised when a Cartesian Hessian matrix violates symmetry |H_ij - H_ji| > 1e-5."""

    def __init__(self, message: str, max_asymmetry: float, indices: Tuple[int, int]) -> None:
        super().__init__(message)
        self.max_asymmetry = max_asymmetry
        self.indices = indices


class SpinContaminationError(QCValidationError):
    """Raised when calculated <S^2> deviates by > 10% from ideal S(S+1) reference value."""

    def __init__(self, message: str, s2_calc: float, s2_ref: float, deviation_percent: float) -> None:
        super().__init__(message)
        self.s2_calc = s2_calc
        self.s2_ref = s2_ref
        self.deviation_percent = deviation_percent


# ==============================================================================
# Frozen CODATA 2022 Physical Conversion Constants
# ==============================================================================
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / 0.529177210903
HARTREE_TO_KCAL_MOL: float = 627.5094740631
HARTREE_TO_WAVENUMBER: float = 219474.63136320


# ==============================================================================
# Structural Ingestion and Parser Protocols
# ==============================================================================
@runtime_checkable
class StructureIngestorProtocol(Protocol):
    """Structural protocol for reading molecular structures across XYZ, PDB, CIF, and MOL2."""

    def ingest(self, source: Union[pathlib.Path, str, bytes]) -> MolecularStructureData:
        """Parse source representation into validated MolecularStructureData."""
        raise TypeError("StructureIngestorProtocol.ingest must be implemented by conforming ingestors.")


@runtime_checkable
class QCLogParserProtocol(Protocol):
    """Structural protocol for parsing quantum chemistry engine log files."""

    def parse_log(self, log_path: pathlib.Path) -> QCResultsSchema:
        """Extract electronic energies, gradients, Hessians, and properties into QCResultsSchema."""
        raise TypeError("QCLogParserProtocol.parse_log must be implemented by conforming parsers.")


# ==============================================================================
# Validated Data Models
# ==============================================================================
class MolecularStructureData(BaseModel):
    """Immutable molecular structure with validated geometry and Mendeleev dynamic masses."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbols: List[str] = Field(..., description="IUPAC elemental symbols")
    coordinates: List[Tuple[float, float, float]] = Field(..., description="Cartesian coordinates in Angstroms")
    charge: int = Field(default=0, description="Net molecular charge")
    multiplicity: int = Field(default=1, description="Spin multiplicity (2S + 1 >= 1)")
    masses: List[float] = Field(default_factory=list, description="Atomic mass units resolved via Mendeleev")
    isotopes: Optional[List[int]] = Field(default=None, description="Optional mass numbers for isotopologue analysis")

    @model_validator(mode="before")
    @classmethod
    def validate_and_resolve_molecular_data(cls, data: Union[Dict[str, object], object]) -> Dict[str, object]:
        """Validate coordinate geometry sanity and dynamically resolve elemental/isotopic masses."""
        if not isinstance(data, dict):
            return data  # type: ignore[return-value]

        syms = data.get("symbols")
        coords = data.get("coordinates")
        mult = data.get("multiplicity", 1)
        user_masses = data.get("masses")
        isotopes = data.get("isotopes")

        if not isinstance(syms, list) or not isinstance(coords, list):
            raise ValueError("Both 'symbols' and 'coordinates' must be provided as lists.")

        n_atoms = len(syms)
        if len(coords) != n_atoms:
            raise ValueError(f"Mismatch: {n_atoms} symbols but {len(coords)} coordinate triplets provided.")

        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be an integer >= 1 (got {mult}).")

        # 1. Coordinate geometry sanity: non-finite check and distance check
        cleaned_coords: List[Tuple[float, float, float]] = []
        for idx, item in enumerate(coords):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(f"Coordinate at index {idx} must be a 3-tuple of floats.")
            x, y, z = float(item[0]), float(item[1]), float(item[2])
            if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
                raise ValueError(f"Non-finite coordinate detected at index {idx}: ({x}, {y}, {z})")
            cleaned_coords.append((x, y, z))

        # Check interatomic distance threshold: r_ij < 0.5 Angstrom is unphysical
        for i in range(n_atoms):
            xi, yi, zi = cleaned_coords[i]
            for j in range(i + 1, n_atoms):
                xj, yj, zj = cleaned_coords[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                if dist < 0.5:
                    raise ValueError(
                        f"Unphysical atomic distance {dist:.4f} Å between atom {i} ({syms[i]}) "
                        f"and atom {j} ({syms[j]}). Minimum physical threshold is 0.5 Å."
                    )

        # 2. Dynamic mass resolution strictly via Mendeleev
        resolved_masses: List[float] = []
        iso_list = isotopes if isinstance(isotopes, list) else None
        if iso_list is not None and len(iso_list) != n_atoms:
            raise ValueError(f"Mismatch: {n_atoms} symbols but {len(iso_list)} isotopic mass numbers provided.")

        if user_masses is not None and isinstance(user_masses, list) and len(user_masses) == n_atoms:
            for m in user_masses:
                val = float(m)
                if val <= 0.0 or math.isnan(val) or math.isinf(val):
                    raise MendeleevInvariantError(f"Invalid positive atomic mass: {val}")
                resolved_masses.append(val)
        else:
            for idx, sym in enumerate(syms):
                clean_sym = str(sym).strip().capitalize()
                try:
                    elem = element(clean_sym)
                except Exception as exc:
                    raise MendeleevInvariantError(
                        f"Dynamic element resolution failed for symbol '{clean_sym}': {exc}"
                    ) from exc
                target_iso = iso_list[idx] if iso_list is not None else None

                if target_iso is not None:
                    matched_mass: Optional[float] = None
                    for iso in elem.isotopes:
                        if iso.mass_number == target_iso and iso.mass is not None:
                            matched_mass = float(iso.mass)
                            break
                    if matched_mass is not None:
                        resolved_masses.append(matched_mass)
                    else:
                        weight = elem.atomic_weight or float(target_iso)
                        resolved_masses.append(float(weight))
                else:
                    if elem.atomic_weight is not None and float(elem.atomic_weight) > 0.0:
                        resolved_masses.append(float(elem.atomic_weight))
                    elif elem.mass_number is not None and float(elem.mass_number) > 0.0:
                        resolved_masses.append(float(elem.mass_number))
                    else:
                        raise MendeleevInvariantError(
                            f"Element '{clean_sym}' lacks a standard atomic weight and default mass number in dynamic "
                            "Mendeleev/IUPAC tables. A physical isotopic mass number must be explicitly specified in "
                            "'isotopes' (e.g., isotopes=[252, ...]) or 'masses'."
                        )

        for m in resolved_masses:
            if m <= 0.0 or math.isnan(m) or math.isinf(m):
                raise MendeleevInvariantError(f"Invalid positive atomic mass: {m}")

        data["coordinates"] = cleaned_coords
        data["masses"] = resolved_masses
        return data


class QCResultsSchema(BaseModel):
    """Immutable quantum chemistry calculation results enforcing physical consistency."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    total_energy: float = Field(..., description="Total electronic energy in Hartree")
    energy_breakdown: Optional[Dict[str, float]] = Field(default=None, description="Energy components in Hartree")
    gradient: Optional[List[float]] = Field(default=None, description="Nuclear Cartesian gradient (3N vector)")
    hessian: Optional[List[List[float]]] = Field(default=None, description="Nuclear Cartesian Hessian (3N x 3N matrix)")
    frequencies: Optional[List[float]] = Field(default=None, description="Harmonic vibrational frequencies in cm^-1")
    dipole_moment: Optional[Tuple[float, float, float]] = Field(default=None, description="Dipole moment in Debye")
    rotational_constants: Optional[Tuple[float, float, float]] = Field(default=None, description="Rotational constants (A, B, C) in GHz")
    s2_expectation: Optional[float] = Field(default=None, description="Calculated <S^2> expectation value")
    s2_ideal: Optional[float] = Field(default=None, description="Exact reference S(S+1) value")

    @model_validator(mode="before")
    @classmethod
    def validate_qc_results(cls, data: Union[Dict[str, object], object]) -> Dict[str, object]:
        """Validate energy finite-ness, gradient/Hessian dimensions, symmetry, and spin contamination."""
        if not isinstance(data, dict):
            return data  # type: ignore[return-value]

        tot_energy = data.get("total_energy")
        if tot_energy is not None:
            e_val = float(str(tot_energy))
            if math.isnan(e_val) or math.isinf(e_val):
                raise ValueError(f"Non-finite total energy: {e_val}")

        # Check gradient dimensionality
        grad = data.get("gradient")
        if grad is not None and isinstance(grad, list):
            if len(grad) % 3 != 0:
                raise ValueError(f"Nuclear Cartesian gradient length must be a multiple of 3 (3N), got {len(grad)}.")
            for val in grad:
                f_val = float(val)
                if math.isnan(f_val) or math.isinf(f_val):
                    raise ValueError(f"Non-finite gradient component: {f_val}")

        # Check Hessian dimensionality and symmetry
        hess = data.get("hessian")
        if hess is not None and isinstance(hess, list):
            dim = len(hess)
            if dim % 3 != 0:
                raise ValueError(f"Nuclear Cartesian Hessian dimension must be a multiple of 3 (3N), got {dim}.")
            for r_idx, row in enumerate(hess):
                if not isinstance(row, list) or len(row) != dim:
                    raise ValueError(f"Hessian row {r_idx} length {len(row) if isinstance(row, list) else 'invalid'} does not match dimension {dim}.")
                for c_idx, val in enumerate(row):
                    f_val = float(val)
                    if math.isnan(f_val) or math.isinf(f_val):
                        raise ValueError(f"Non-finite Hessian entry at ({r_idx}, {c_idx}): {f_val}")

            # Verify matrix symmetry: |H_ij - H_ji| < 1e-5
            max_asym = 0.0
            worst_indices = (0, 0)
            for i in range(dim):
                for j in range(i + 1, dim):
                    diff = abs(float(hess[i][j]) - float(hess[j][i]))
                    if diff > max_asym:
                        max_asym = diff
                        worst_indices = (i, j)

            if max_asym > 1e-5:
                i, j = worst_indices
                hij = float(hess[i][j])
                hji = float(hess[j][i])
                raise HessianSymmetryError(
                    f"Asymmetric Hessian matrix: |H[{i}][{j}] ({hij}) - H[{j}][{i}] ({hji})| = {max_asym:.6e} > 1e-5",
                    max_asymmetry=max_asym,
                    indices=(i, j),
                )

        # Spin contamination audit: deviation <= 10%
        s2_calc = data.get("s2_expectation")
        s2_ref = data.get("s2_ideal")
        if s2_calc is not None and s2_ref is not None:
            f_calc = float(str(s2_calc))
            f_ref = float(str(s2_ref))
            if f_ref > 1e-6:
                dev = abs(f_calc - f_ref) / f_ref
            else:
                dev = abs(f_calc - f_ref)
            if dev > 0.10:
                raise SpinContaminationError(
                    f"Spin contamination exceeded: deviation {dev:.2%} > 10% tolerance "
                    f"(s2_expectation={f_calc}, s2_ideal={f_ref}).",
                    s2_calc=f_calc,
                    s2_ref=f_ref,
                    deviation_percent=dev * 100.0,
                )

        return data
