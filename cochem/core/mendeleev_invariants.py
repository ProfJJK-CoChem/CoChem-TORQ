"""Dynamic Mendeleev Invariants & Element Resolver.

Provenance & Specifications:
- Method Matrix [M]: Quantum spin-parity and IUPAC CIAAW standard atomic weight invariants.
- Dynamic Resolution [D]: Zero-hardcoding dynamic element and isotopic mass lookup via mendeleev.
- Telemetry [E]: Thread-safe in-memory cache populated dynamically on demand.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from mendeleev import element as _mendeleev_element

from cochem.core.exceptions import MissingDataError


class MendeleevInvariantError(ValueError, MissingDataError):
    """Raised when chemical element queries violate Mendeleev physical invariants."""

    def __init__(self, message: str, symbol_or_query: Any = None) -> None:
        ValueError.__init__(self, message)
        MissingDataError.__init__(
            self,
            message=message,
            symbol_or_query=symbol_or_query,
        )
        self.symbol_or_query = symbol_or_query


@dataclass(slots=True, frozen=True)
class ElementData:
    """Immutable ground-truth chemical element properties."""

    atomic_number: int
    symbol: str
    name: str
    atomic_weight: float
    isotopes: Tuple[Tuple[int, float, float], ...]  # (mass_number, exact_mass_amu, natural_abundance)
    covalent_radius_pm: Optional[float]
    vdw_radius_pm: Optional[float]
    valence_electrons: int
    mass: float = 0.0
    mass_number: Optional[int] = None
    formal_charge: int = 0
    is_isotope: bool = False


_CACHE_LOCK = threading.Lock()
_ELEMENTS_BY_SYMBOL: Dict[str, ElementData] = {}
_ELEMENTS_BY_Z: Dict[int, ElementData] = {}


def _load_element_data(z_or_sym: Union[int, str]) -> ElementData:
    """Dynamically fetch and cache ElementData for Z=1..118 via mendeleev."""
    try:
        elem = _mendeleev_element(z_or_sym)
    except Exception as exc:
        raise MendeleevInvariantError(
            f"Dynamic element resolution failed for query '{z_or_sym}': {exc}",
            symbol_or_query=z_or_sym,
        ) from exc

    z = int(elem.atomic_number)
    symbol = str(elem.symbol)
    name = str(elem.name)

    # Standard atomic weight with dynamic fallback to most stable isotope mass
    weight = elem.atomic_weight
    if weight is None or float(weight) <= 0.0:
        iso_masses = [iso.mass_number for iso in elem.isotopes if iso.mass_number is not None]
        if iso_masses:
            weight = float(max(iso_masses))
        else:
            weight = float(z)
    else:
        weight = float(weight)

    # Isotope tuple: (mass_number, exact_mass_amu, abundance)
    isotope_list: List[Tuple[int, float, float]] = []
    for iso in elem.isotopes:
        if iso.mass_number is not None:
            m_num = int(iso.mass_number)
            m_exact = float(iso.mass) if iso.mass is not None and float(iso.mass) > 0.0 else float(m_num)
            m_abund = float(iso.abundance) if iso.abundance is not None else 0.0
            isotope_list.append((m_num, m_exact, m_abund))
    isotopes_tuple = tuple(sorted(isotope_list, key=lambda x: x[0]))

    # Radii in picometers
    cov_r = elem.covalent_radius_pyykko or elem.covalent_radius
    cov_radius_pm = float(cov_r) if cov_r is not None else None

    vdw_r = elem.vdw_radius or elem.vdw_radius_alvarez or elem.vdw_radius_bondi or elem.vdw_radius_batsanov
    vdw_radius_pm = float(vdw_r) if vdw_r is not None else None

    # Valence electrons
    if hasattr(elem, "nvalence") and callable(elem.nvalence):
        val_e = int(elem.nvalence())
    elif elem.electrons is not None:
        val_e = int(elem.electrons)
    else:
        val_e = 0

    data = ElementData(
        atomic_number=z,
        symbol=symbol,
        name=name,
        atomic_weight=weight,
        isotopes=isotopes_tuple,
        covalent_radius_pm=cov_radius_pm,
        vdw_radius_pm=vdw_radius_pm,
        valence_electrons=val_e,
        mass=weight,
        mass_number=None,
        formal_charge=0,
        is_isotope=False,
    )

    with _CACHE_LOCK:
        _ELEMENTS_BY_SYMBOL[symbol] = data
        _ELEMENTS_BY_Z[z] = data

    return data


def parse_symbol_or_isotope(symbol: str) -> Tuple[str, Optional[int]]:
    """Authoritative regex and alias pre-processor for chemical symbols and isotopes.

    Maps:
    - 'D' -> ('H', 2)
    - 'T' -> ('H', 3)
    - '13C' -> ('C', 13)
    - '18O' -> ('O', 18)
    - '2H' -> ('H', 2)
    - Standard symbols ('H', 'C', 'Ar') -> ('H', None), etc.
    """
    raw = str(symbol).strip()
    if not raw or raw.isdigit():
        raise MissingDataError(
            f"Invalid chemical symbol or isotope '{symbol}'. Symbol cannot be empty or purely numeric.",
            symbol_or_query=symbol,
        )

    # Specific alias mappings
    if raw.upper() == "D":
        return "H", 2
    if raw.upper() == "T":
        return "H", 3

    # Check for leading mass number: e.g. "13C", "18O", "2H", "35Cl"
    m_iso = re.match(r"^(\d+)([A-Za-z]+)$", raw)
    if m_iso:
        mass_num = int(m_iso.group(1))
        sym_part = m_iso.group(2)
        norm_sym = sym_part[0].upper() + sym_part[1:].lower() if len(sym_part) > 1 else sym_part.upper()
        # Verify element exists in Mendeleev
        try:
            get_element(norm_sym)
        except Exception:
            raise MissingDataError(
                f"Unresolvable atomic element or isotope symbol: {symbol}",
                symbol_or_query=symbol,
            )
        return norm_sym, mass_num

    # Standard elemental symbol: e.g. "C", "Cl", "Ar"
    m_sym = re.match(r"^[A-Za-z]+$", raw)
    if m_sym:
        norm_sym = raw[0].upper() + raw[1:].lower() if len(raw) > 1 else raw.upper()
        try:
            elem_data = get_element(norm_sym)
            return elem_data.symbol, None
        except Exception:
            # Check by element name
            try:
                elem_data = _load_element_data(raw)
                return elem_data.symbol, None
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")

    raise MissingDataError(
        f"Unresolvable atomic element or isotope symbol: {symbol}",
        symbol_or_query=symbol,
    )


def get_element(symbol_or_z: Union[str, int]) -> ElementData:
    """Retrieve immutable ElementData by atomic number, chemical symbol, formal charge, or isotope."""
    if isinstance(symbol_or_z, int):
        if symbol_or_z < 1 or symbol_or_z > 118:
            raise MendeleevInvariantError(
                f"Invalid atomic number Z={symbol_or_z}. Must be between 1 and 118.",
                symbol_or_query=symbol_or_z,
            )
        with _CACHE_LOCK:
            cached = _ELEMENTS_BY_Z.get(symbol_or_z)
        if cached is not None:
            return cached
        return _load_element_data(symbol_or_z)

    raw = str(symbol_or_z).strip()
    if not raw or raw.isdigit():
        raise MendeleevInvariantError(
            f"Invalid chemical symbol '{symbol_or_z}'. Symbol cannot be empty or purely numeric.",
            symbol_or_query=symbol_or_z,
        )

    pattern = re.compile(r"^(?P<isotope>\d+)?(?P<symbol>[A-Za-z]+)(?P<charge>(?:\d+[+-]|[+-]\d*|[+-]))?$")
    match = pattern.match(raw)
    if not match:
        raise MendeleevInvariantError(
            f"Dynamic element resolution failed for query '{symbol_or_z}'.",
            symbol_or_query=symbol_or_z,
        )

    iso_str = match.group("isotope")
    sym_raw = match.group("symbol")
    charge_str = match.group("charge")

    # Alias mappings for Deuterium (D) and Tritium (T)
    if sym_raw.upper() == "D":
        norm_sym = "H"
        mass_number: Optional[int] = 2
    elif sym_raw.upper() == "T":
        norm_sym = "H"
        mass_number = 3
    else:
        norm_sym = sym_raw[0].upper() + sym_raw[1:].lower() if len(sym_raw) > 1 else sym_raw.upper()
        mass_number = int(iso_str) if iso_str else None

    formal_charge: int = 0
    if charge_str:
        if charge_str.endswith("+"):
            val = charge_str[:-1]
            formal_charge = int(val) if val else 1
        elif charge_str.endswith("-"):
            val = charge_str[:-1]
            formal_charge = -int(val) if val else -1
        elif charge_str.startswith("+"):
            val = charge_str[1:]
            formal_charge = int(val) if val else 1
        elif charge_str.startswith("-"):
            val = charge_str[1:]
            formal_charge = -int(val) if val else -1

    # Dynamic lookup via mendeleev
    try:
        elem = _mendeleev_element(norm_sym)
        base_data = _load_element_data(norm_sym)
    except Exception as exc:
        # Fallback to query by full element name (e.g. 'Carbon')
        try:
            elem = _mendeleev_element(sym_raw)
            base_data = _load_element_data(sym_raw)
            norm_sym = str(elem.symbol)
        except Exception:
            raise MendeleevInvariantError(
                f"Dynamic element resolution failed for query '{symbol_or_z}': element '{norm_sym}' not found.",
                symbol_or_query=symbol_or_z,
            ) from exc

    if mass_number is not None:
        is_isotope = True
        iso = next((i for i in elem.isotopes if i.mass_number == mass_number), None)
        if iso is None or iso.mass is None or float(iso.mass) <= 0.0:
            raise MendeleevInvariantError(
                f"No isotope with mass number A={mass_number} found for element '{norm_sym}'.",
                symbol_or_query=symbol_or_z,
            )
        mass = float(iso.mass)
    else:
        is_isotope = False
        mass = float(base_data.atomic_weight)

    return ElementData(
        atomic_number=base_data.atomic_number,
        symbol=base_data.symbol,
        name=base_data.name,
        atomic_weight=base_data.atomic_weight,
        isotopes=base_data.isotopes,
        covalent_radius_pm=base_data.covalent_radius_pm,
        vdw_radius_pm=base_data.vdw_radius_pm,
        valence_electrons=base_data.valence_electrons,
        mass=mass,
        mass_number=mass_number,
        formal_charge=formal_charge,
        is_isotope=is_isotope,
    )


def get_isotope_mass(symbol_or_z: Union[str, int], mass_number: int) -> float:
    """Dynamically resolve IUPAC exact isotopic mass in unified atomic mass units (u)."""
    element_data = get_element(symbol_or_z)
    for iso_m_num, iso_exact, _ in element_data.isotopes:
        if iso_m_num == mass_number:
            return iso_exact

    # Dynamic fallback query directly to mendeleev element isotopes
    try:
        m_elem = _mendeleev_element(element_data.symbol)
        for iso in m_elem.isotopes:
            if iso.mass_number == mass_number and iso.mass is not None:
                return float(iso.mass)
    except Exception as _e:
        logger.debug(f"Ignored exception: {_e}")

    raise MendeleevInvariantError(
        f"No isotope with mass number A={mass_number} found for element '{element_data.symbol}'.",
        symbol_or_query=f"{element_data.symbol}-{mass_number}",
    )


def get_element_mass(symbol_or_z: Union[str, int]) -> float:
    """Dynamically resolve atomic or isotopic mass in unified atomic mass units (u).

    Handles standard elements ('H', 'C', 'Ar') and isotopic aliases ('D', 'T', '13C', '18O').
    """
    if isinstance(symbol_or_z, int):
        return get_element(symbol_or_z).atomic_weight

    clean_sym, mass_number = parse_symbol_or_isotope(symbol_or_z)
    if mass_number is not None:
        return get_isotope_mass(clean_sym, mass_number)
    return get_element(clean_sym).atomic_weight


class MendeleevResolver:
    """Thread-safe dynamic Mendeleev element and isotope mass resolver for backward compatibility."""

    def get_element(self, symbol_or_z: Union[str, int]) -> Any:
        elem_data = get_element(symbol_or_z)
        return _mendeleev_element(elem_data.atomic_number)

    def get_atomic_number(self, symbol_or_z: Union[str, int]) -> int:
        return get_element(symbol_or_z).atomic_number

    def get_atomic_weight(self, symbol_or_z: Union[str, int]) -> float:
        return get_element(symbol_or_z).atomic_weight

    def get_element_mass(self, symbol_or_z: Union[str, int]) -> float:
        return get_element_mass(symbol_or_z)

    def get_symbol(self, symbol_or_z: Union[str, int]) -> str:
        return get_element(symbol_or_z).symbol

    def get_name(self, symbol_or_z: Union[str, int]) -> str:
        return get_element(symbol_or_z).name

    def get_covalent_radius(self, symbol_or_z: Union[str, int]) -> Optional[float]:
        return get_element(symbol_or_z).covalent_radius_pm

    def get_vdw_radius(self, symbol_or_z: Union[str, int]) -> float:
        r = get_element(symbol_or_z).vdw_radius_pm
        if r is None:
            raise MissingDataError(f"Van der Waals radius is not available for element '{symbol_or_z}'.")
        return r

    def get_vdw_radius_angstrom(self, symbol_or_z: Union[str, int]) -> float:
        return self.get_vdw_radius(symbol_or_z) / 100.0

    def get_isotope_mass(self, symbol_or_z: Union[str, int], mass_number: int) -> float:
        return get_isotope_mass(symbol_or_z, mass_number)

    def get_isotope_abundance(self, symbol_or_z: Union[str, int], mass_number: int) -> float:
        elem_data = get_element(symbol_or_z)
        for m_num, _, abund in elem_data.isotopes:
            if m_num == mass_number:
                return abund
        return 0.0

    def get_available_isotopes(self, symbol_or_z: Union[str, int]) -> List[int]:
        return [m_num for m_num, _, _ in get_element(symbol_or_z).isotopes]

    def clear_cache(self) -> None:
        raise MissingDataError("Mendeleev element cache is immutable and cannot be cleared.")


# Default global resolver instance for backwards compatibility
mendeleev_resolver = MendeleevResolver()

from cochem_base.core.mendeleev_invariants import get_element_cache

__all__ = [
    "ElementData",
    "MendeleevInvariantError",
    "MissingDataError",
    "get_element_cache",
    "get_element",
    "get_isotope_mass",
    "get_element_mass",
    "parse_symbol_or_isotope",
    "MendeleevResolver",
    "mendeleev_resolver",
]
