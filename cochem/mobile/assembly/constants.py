"""Dynamic physical constants and quantum parity verification backed by Mendeleev.

Strict adherence to the Zero-Mock and Mendeleev Library Mandates with thread-safe LRU caching.
"""

from __future__ import annotations

import functools
from typing import Set

from mendeleev import element

from cochem.mobile.assembly.exceptions import MendeleevLookupError, QuantumParityError

TRANSITION_METAL_ATOMIC_NUMBERS: Set[int] = (
    set(
        range(21, 31)  # 3d: Sc (21) - Zn (30)
    )
    .union(
        set(range(39, 49))  # 4d: Y (39) - Cd (48)
    )
    .union(
        set(range(71, 81))  # 5d: Lu (71) - Hg (80)
    )
)


@functools.lru_cache(maxsize=128)
def get_atomic_number(symbol: str) -> int:
    """Retrieve atomic number for a given element symbol.

    Args:
        symbol: IUPAC element symbol (e.g., 'Fe', 'Pt').

    Returns:
        Atomic number Z.

    Raises:
        MendeleevLookupError: If symbol cannot be resolved.
    """
    try:
        elem = element(symbol)
        z = elem.atomic_number
        if z is None or not isinstance(z, int):
            raise MendeleevLookupError(f"Invalid atomic number for element '{symbol}'.")
        return int(z)
    except Exception as exc:
        raise MendeleevLookupError(f"Failed to lookup element '{symbol}': {exc}") from exc


@functools.lru_cache(maxsize=128)
def validate_transition_metal(symbol: str) -> int:
    """Validate that the given symbol corresponds to a transition metal in Z in [21..30, 39..48, 71..80].

    Args:
        symbol: IUPAC element symbol.

    Returns:
        Atomic number Z.

    Raises:
        MendeleevLookupError: If element is unresolvable or not in transition metal range.
    """
    z = get_atomic_number(symbol)
    if z not in TRANSITION_METAL_ATOMIC_NUMBERS:
        raise MendeleevLookupError(
            f"Element '{symbol}' (Z={z}) is not a supported transition metal. "
            "Supported ranges: Z in [21..30] (3d), [39..48] (4d), [71..80] (5d)."
        )
    return z


@functools.lru_cache(maxsize=128)
def get_covalent_radius_angstrom(symbol: str) -> float:
    """Retrieve covalent radius in Angstroms using fallback cascade.

    Fallback cascade:
        covalent_radius_pyykko -> covalent_radius_cordero -> covalent_radius_slater ->
        covalent_radius_bragg -> covalent_radius -> atomic_radius.

    Args:
        symbol: IUPAC element symbol.

    Returns:
        Covalent radius in Angstroms (scaled from picometers).

    Raises:
        MendeleevLookupError: If no covalent radius is resolvable.
    """
    try:
        elem = element(symbol)
    except Exception as exc:
        raise MendeleevLookupError(f"Failed to lookup element '{symbol}': {exc}") from exc

    cascade_attrs = [
        "covalent_radius_pyykko",
        "covalent_radius_cordero",
        "covalent_radius_slater",
        "covalent_radius_bragg",
        "covalent_radius",
        "atomic_radius",
    ]

    for attr in cascade_attrs:
        val = getattr(elem, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return float(val) / 100.0

    raise MendeleevLookupError(f"No covalent radius resolvable for element '{symbol}'.")


@functools.lru_cache(maxsize=128)
def get_vdw_radius_angstrom(symbol: str) -> float:
    """Retrieve van der Waals radius in Angstroms using fallback cascade.

    Fallback cascade:
        vdw_radius -> vdw_radius_alvarez -> vdw_radius_baddi -> vdw_radius_bondi ->
        vdw_radius_truhlar -> vdw_radius_batsanov -> vdw_radius_uff -> vdw_radius_mm3.

    Args:
        symbol: IUPAC element symbol.

    Returns:
        van der Waals radius in Angstroms (scaled from picometers).

    Raises:
        MendeleevLookupError: If no vdW radius is resolvable.
    """
    try:
        elem = element(symbol)
    except Exception as exc:
        raise MendeleevLookupError(f"Failed to lookup element '{symbol}': {exc}") from exc

    cascade_attrs = [
        "vdw_radius",
        "vdw_radius_alvarez",
        "vdw_radius_baddi",
        "vdw_radius_bondi",
        "vdw_radius_truhlar",
        "vdw_radius_batsanov",
        "vdw_radius_uff",
        "vdw_radius_mm3",
    ]

    for attr in cascade_attrs:
        val = getattr(elem, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return float(val) / 100.0

    raise MendeleevLookupError(f"No van der Waals radius resolvable for element '{symbol}'.")


@functools.lru_cache(maxsize=128)
def get_standard_atomic_weight(symbol: str) -> float:
    """Retrieve standard atomic weight for a given element symbol.

    Args:
        symbol: IUPAC element symbol.

    Returns:
        Standard atomic weight in g/mol.

    Raises:
        MendeleevLookupError: If atomic weight cannot be retrieved.
    """
    try:
        elem = element(symbol)
        weight = elem.atomic_weight
        if weight is None or not isinstance(weight, (int, float)) or weight <= 0:
            raise MendeleevLookupError(f"Invalid atomic weight for element '{symbol}'.")
        return float(weight)
    except Exception as exc:
        raise MendeleevLookupError(f"Failed to lookup atomic weight for '{symbol}': {exc}") from exc


@functools.lru_cache(maxsize=128)
def calculate_d_electron_count(symbol: str, oxidation_state: int) -> tuple[int, int]:
    """Calculate d-electron count (d^n) and total electron count for transition metal ion.

    Args:
        symbol: Transition metal symbol.
        oxidation_state: Formal oxidation state (0 <= oxidation_state <= 7).

    Returns:
        Tuple of (d_electron_count, total_electron_count).

    Raises:
        MendeleevLookupError: If symbol is invalid or not a transition metal.
        ValueError: If oxidation_state is outside [0, 7].
    """
    if not (0 <= oxidation_state <= 7):
        raise ValueError(f"Oxidation state {oxidation_state} must be in range [0, 7].")

    z = validate_transition_metal(symbol)
    elem = element(symbol)
    group_id = elem.group_id
    if group_id is None:
        raise MendeleevLookupError(f"Could not resolve group_id for element '{symbol}'.")

    d_count = max(0, int(group_id) - int(oxidation_state))
    total_electrons = int(z) - int(oxidation_state)
    return d_count, total_electrons


def validate_quantum_parity(total_electrons: int, spin_multiplicity: int) -> None:
    """Validate quantum spin parity congruence: 2S congruent with total_electrons mod 2.

    Args:
        total_electrons: Total electron count of the chemical system.
        spin_multiplicity: Spin multiplicity (2S + 1).

    Raises:
        QuantumParityError: If 2S is not congruent with total_electrons mod 2.
    """
    if spin_multiplicity < 1:
        raise QuantumParityError(f"Spin multiplicity {spin_multiplicity} must be >= 1.")

    two_s = spin_multiplicity - 1
    if (two_s % 2) != (total_electrons % 2):
        raise QuantumParityError(
            f"Quantum parity violation: 2S={two_s} is not congruent with "
            f"Ne={total_electrons} mod 2."
        )
