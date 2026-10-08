"""Dynamic physical constants and quantum parity verification backed by Mendeleev.

Actual database values and explicit formal bookkeeping; no substitute values.
"""

from __future__ import annotations

import functools
import math
from numbers import Real
from typing import TypedDict

from mendeleev import element

from cochem.mobile.assembly.exceptions import MendeleevLookupError, QuantumParityError

TRANSITION_METAL_ATOMIC_NUMBERS: set[int] = (
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


class FormalTransitionMetalElectronRecord(TypedDict):
    """Formal counting model, never a measured or calculated d population."""

    model: str
    atomic_number: int
    group_id: int
    formal_oxidation_state: int
    formal_valence_electron_count: int
    d_electron_count: int | None
    d_electron_count_status: str
    actual_electronic_population_available: bool
    total_electron_count: int


def _validate_atomic_number(value: object, symbol: str) -> int:
    if type(value) is not int or not 1 <= value <= 118:
        raise MendeleevLookupError(f"Invalid atomic number for element '{symbol}'.")
    return value


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
        return _validate_atomic_number(elem.atomic_number, symbol)
    except Exception as exc:
        raise MendeleevLookupError(
            f"Failed to lookup element '{symbol}': {exc}"
        ) from exc


@functools.lru_cache(maxsize=128)
def validate_transition_metal(symbol: str) -> int:
    """Validate membership of the supported 3d, 4d, or 5d element ranges.

    Args:
        symbol: IUPAC element symbol.

    Returns:
        Atomic number Z.

    Raises:
        MendeleevLookupError: If the element is unknown or unsupported.
    """
    z = get_atomic_number(symbol)
    if z not in TRANSITION_METAL_ATOMIC_NUMBERS:
        raise MendeleevLookupError(
            f"Element '{symbol}' (Z={z}) is not a supported transition metal. "
            "Supported ranges: Z in [21..30] (3d), [39..48] (4d), [71..80] (5d)."
        )
    return z


def _validate_positive_property(
    value: object, symbol: str, property_name: str
) -> float:
    """Validate a named real scalar without changing its definition."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MendeleevLookupError(
            f"Required Mendeleev property {property_name} "
            f"is unavailable for '{symbol}'."
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise MendeleevLookupError(
            f"Mendeleev property {property_name} cannot be represented "
            f"as a finite value for '{symbol}'."
        ) from exc
    if not math.isfinite(result) or result <= 0:
        raise MendeleevLookupError(
            f"Mendeleev property {property_name} must be finite and positive "
            f"for '{symbol}'."
        )
    return result


def _validate_positive_radius_pm(
    value: object, symbol: str, property_name: str
) -> float:
    return _validate_positive_property(value, symbol, property_name)


def _named_radius_angstrom(symbol: str, property_name: str) -> float:
    try:
        elem = element(symbol)
    except Exception as exc:
        raise MendeleevLookupError(
            f"Failed to lookup element '{symbol}': {exc}"
        ) from exc
    radius_pm = _validate_positive_radius_pm(
        getattr(elem, property_name, None), symbol, property_name
    )
    radius_angstrom = radius_pm / 100.0
    if not math.isfinite(radius_angstrom) or radius_angstrom <= 0:
        raise MendeleevLookupError(
            f"Mendeleev property {property_name} cannot be represented in Angstroms "
            f"for '{symbol}'."
        )
    return radius_angstrom


@functools.lru_cache(maxsize=128)
def get_covalent_radius_angstrom(symbol: str) -> float:
    """Retrieve the named Pyykkö single-bond covalent radius, converted from pm.

    Only the installed Mendeleev ``covalent_radius_pyykko`` field is used.
    A missing or invalid entry raises MendeleevLookupError. This tabulated
    radius defines a geometric estimate; it does not establish a bond length
    from an optimized electronic structure.
    """
    return _named_radius_angstrom(symbol, "covalent_radius_pyykko")


@functools.lru_cache(maxsize=128)
def get_vdw_radius_angstrom(symbol: str) -> float:
    """Retrieve the named Mendeleev ``vdw_radius`` field, converted from pm.

    A missing or invalid entry raises MendeleevLookupError. This specific
    database field is not asserted to constitute a homogeneous Bondi radius
    dataset or a quantum calculation of a contact distance.
    """
    return _named_radius_angstrom(symbol, "vdw_radius")


@functools.lru_cache(maxsize=128)
def get_standard_atomic_weight(symbol: str) -> float:
    """Retrieve the actual Mendeleev ``atomic_weight`` field.

    This tabulated standard or representative value is not asserted to be
    an isotope-specific mass.

    Args:
        symbol: IUPAC element symbol.

    Returns:
        The named database value in g/mol.

    Raises:
        MendeleevLookupError: If atomic weight cannot be retrieved.
    """
    try:
        elem = element(symbol)
        return _validate_positive_property(elem.atomic_weight, symbol, "atomic_weight")
    except Exception as exc:
        raise MendeleevLookupError(
            f"Failed to lookup atomic weight for '{symbol}': {exc}"
        ) from exc


def formal_transition_metal_electron_record(
    symbol: str, oxidation_state: int
) -> FormalTransitionMetalElectronRecord:
    """Describe group-minus-formal-oxidation bookkeeping and its applicability.

    The unrestricted formal value is preserved. Values outside 0..10 cannot
    represent d-shell occupancy and therefore have no assigned d count.
    Neither value establishes actual electronic population or the physical
    validity of a caller-declared oxidation state.
    """
    if type(oxidation_state) is not int or not 0 <= oxidation_state <= 7:
        raise ValueError("Formal oxidation state must be an integer in [0, 7].")
    z = validate_transition_metal(symbol)
    elem = element(symbol)
    group_id = elem.group_id
    if type(group_id) is not int or not 1 <= group_id <= 18:
        raise MendeleevLookupError(
            f"Could not resolve a valid group_id for element '{symbol}'."
        )
    formal_count = group_id - oxidation_state
    d_count = formal_count if 0 <= formal_count <= 10 else None
    return {
        "model": "formal_group_minus_oxidation_bookkeeping",
        "atomic_number": z,
        "group_id": group_id,
        "formal_oxidation_state": oxidation_state,
        "formal_valence_electron_count": formal_count,
        "d_electron_count": d_count,
        "d_electron_count_status": (
            "formal_bookkeeping" if d_count is not None else "outside_d_shell_range"
        ),
        "actual_electronic_population_available": False,
        "total_electron_count": z - oxidation_state,
    }


@functools.lru_cache(maxsize=128, typed=True)
def calculate_d_electron_count(
    symbol: str, oxidation_state: int
) -> tuple[int | None, int]:
    """Return formal d bookkeeping and the caller-declared metal-ion count.

    The d count is null outside the simple model's 0..10 range. This function
    is not an electronic population analysis. For the complete model and
    unbounded formal value use ``formal_transition_metal_electron_record``.
    """
    record = formal_transition_metal_electron_record(symbol, oxidation_state)
    return record["d_electron_count"], record["total_electron_count"]


def validate_quantum_parity(total_electrons: int, spin_multiplicity: int) -> None:
    """Validate quantum spin parity congruence: 2S congruent with total_electrons mod 2.

    Args:
        total_electrons: Total electron count of the chemical system.
        spin_multiplicity: Spin multiplicity (2S + 1).

    Raises:
        QuantumParityError: If counts, maximum spin, or parity are invalid.
    """
    if type(total_electrons) is not int or total_electrons < 0:
        raise QuantumParityError("Total electron count must be a nonnegative integer.")
    if type(spin_multiplicity) is not int or spin_multiplicity < 1:
        raise QuantumParityError("Spin multiplicity must be a positive integer.")

    two_s = spin_multiplicity - 1
    if two_s > total_electrons:
        raise QuantumParityError("Declared 2S cannot exceed the total electron count.")
    if (two_s % 2) != (total_electrons % 2):
        raise QuantumParityError(
            f"Quantum parity violation: 2S={two_s} is not congruent with "
            f"Ne={total_electrons} mod 2."
        )
