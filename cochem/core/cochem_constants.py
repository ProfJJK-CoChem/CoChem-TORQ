"""Centralized Physical Constants & Dynamic Mendeleev Registry.
Strictly adheres to Mendeleev Mandate and CODATA 2018 Dynamic Lookup.
Zero hardcoding of atomic masses or periodic tables.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, Optional

import scipy.constants
from mendeleev import element


# Authoritative conversion constant for rotational constants: MHz * u * Angstrom^2
# Derived from CODATA 2022: C_rot = 10^-6 * h / (8 * pi^2 * u * Angstrom^2)
C_ROT_MHZ_U_ANG2: float = 505379.0084350172


@dataclass(frozen=True)
class PhysicalConstant:
    """Immutable representation of a physical constant with CODATA provenance."""

    name: str
    symbol: str
    value: float
    uncertainty: float
    unit: str
    provenance: str
    source: str = "CODATA 2018 / scipy.constants"


@dataclass(frozen=True)
class ElementProperties:
    """Immutable elemental structure dynamically populated from Mendeleev database."""

    atomic_number: int
    symbol: str
    name: str
    atomic_weight: float
    covalent_radius_pyykko: Optional[float]
    vdw_radius_bondi: Optional[float]
    provenance: str = "[M]"


class PhysicalConstantsRegistry:
    """Authoritative scientific registry for physical constants and elemental data."""

    STANDARD_TEMPERATURE_K: float = 298.15
    STANDARD_PRESSURE_PA: float = 101325.0

    _CONVENTIONAL_SYMBOLS: Dict[str, str] = {
        "Planck constant": "h",
        "Boltzmann constant": "k_B",
        "speed of light in vacuum": "c",
        "Avogadro constant": "N_A",
        "elementary charge": "e",
        "molar gas constant": "R",
        "atomic mass constant": "u",
    }

    _SYMBOL_ALIASES: Dict[str, str] = {
        "h": "Planck constant",
        "c": "speed of light in vacuum",
        "e": "elementary charge",
        "u": "atomic mass constant",
        "amu": "atomic mass constant",
        "k_B": "Boltzmann constant",
        "kB": "Boltzmann constant",
        "k": "Boltzmann constant",
        "N_A": "Avogadro constant",
        "NA": "Avogadro constant",
        "R": "molar gas constant",
    }

    _DERIVED_CONSTANTS: Dict[str, Tuple[float, str, str, str]] = {
        "C_ROT_MHZ_U_ANG2": (
            C_ROT_MHZ_U_ANG2,
            "MHz * u * Angstrom^2",
            "C_rot",
            "Authoritative rotational constant conversion factor (Method Matrix / CODATA 2022 derived)",
        ),
        "C_rot": (
            C_ROT_MHZ_U_ANG2,
            "MHz * u * Angstrom^2",
            "C_rot",
            "Authoritative rotational constant conversion factor (Method Matrix / CODATA 2022 derived)",
        ),
        "c_rot": (
            C_ROT_MHZ_U_ANG2,
            "MHz * u * Angstrom^2",
            "C_rot",
            "Authoritative rotational constant conversion factor (Method Matrix / CODATA 2022 derived)",
        ),
    }

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def get_constant(name: str) -> PhysicalConstant:
        """Query physical constant dynamically from registry or scipy CODATA database.

        Provenance tag rules (Method Matrix v4 & Task 10):
        - Exact CODATA SI standards (unc == 0.0) or measured CODATA standards: [M].
        - Derived analytical constants (e.g. C_rot = h / (8*pi^2)): [D].
        - Empirical / heuristic parameters: [E].
        """
        if name in PhysicalConstantsRegistry._DERIVED_CONSTANTS:
            val, unit, sym, src = PhysicalConstantsRegistry._DERIVED_CONSTANTS[name]
            return PhysicalConstant(
                name=name,
                symbol=sym,
                value=float(val),
                uncertainty=0.0,
                unit=unit,
                provenance="[D]",
                source=src,
            )

        resolved_name = PhysicalConstantsRegistry._SYMBOL_ALIASES.get(name, name)
        if resolved_name in PhysicalConstantsRegistry._DERIVED_CONSTANTS:
            val, unit, sym, src = PhysicalConstantsRegistry._DERIVED_CONSTANTS[resolved_name]
            return PhysicalConstant(
                name=resolved_name,
                symbol=name,
                value=float(val),
                uncertainty=0.0,
                unit=unit,
                provenance="[D]",
                source=src,
            )

        if resolved_name not in scipy.constants.physical_constants:
            raise KeyError(f"Constant '{name}' (resolved as '{resolved_name}') not found in CODATA registry.")

        val, unit, unc = scipy.constants.physical_constants[resolved_name]
        symbol = PhysicalConstantsRegistry._CONVENTIONAL_SYMBOLS.get(resolved_name, name)

        provenance = "[M]"

        return PhysicalConstant(
            name=resolved_name,
            symbol=symbol,
            value=float(val),
            uncertainty=float(unc),
            unit=str(unit),
            provenance=provenance,
            source="CODATA 2018 / scipy.constants",
        )

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def get_element(symbol_or_z: Any) -> ElementProperties:
        """Dynamically retrieve element properties from Mendeleev library."""
        elem = element(symbol_or_z)

        covalent_val = getattr(elem, "covalent_radius_pyykko", None)
        vdw_val = getattr(elem, "vdw_radius_bondi", None)

        covalent_radius: Optional[float] = (
            float(covalent_val) if covalent_val is not None else None
        )
        vdw_radius: Optional[float] = (
            float(vdw_val) if vdw_val is not None else None
        )

        return ElementProperties(
            atomic_number=int(elem.atomic_number),
            symbol=str(elem.symbol),
            name=str(elem.name),
            atomic_weight=float(elem.mass),
            covalent_radius_pyykko=covalent_radius,
            vdw_radius_bondi=vdw_radius,
            provenance="[M]",
        )

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def get_element_mass(symbol: str) -> float:
        """Dynamically retrieve IUPAC standard atomic mass from Mendeleev."""
        elem = element(symbol)
        return float(elem.mass)
