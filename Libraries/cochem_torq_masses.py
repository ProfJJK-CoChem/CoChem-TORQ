"""Dynamic Mendeleev monoisotopic mass retrieval with explicit isotope selection.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Hardcoded mass dictionaries are strictly forbidden.
All masses are dynamically queried from the mendeleev library.
"""

from __future__ import annotations

import functools
from typing import List, Sequence
from numbers import Real
from mendeleev import element
from Libraries.cochem_isotopes import isotope_mass
import torch


@functools.lru_cache(maxsize=128)
def get_monoisotopic_mass(atomic_number: int, isotope_number: int | None = None) -> float:
    """Dynamically query monoisotopic mass using mendeleev with explicit isotope selection. [M]
    
    Parameters
    ----------
    atomic_number : int
        Atomic number Z (0 for ghost atoms, 1 <= Z <= 118).
        
    Returns
    -------
    float
        Monoisotopic mass in unified atomic mass units (u).
    """
    if not isinstance(atomic_number, Real) or int(atomic_number) != atomic_number:
        raise ValueError("Atomic number must be an integer.")
    atomic_number = int(atomic_number)
    if atomic_number == 0:
        return 0.0  # Ghost atom [M]
    if atomic_number < 0 or atomic_number > 118:
        raise ValueError(f"Atomic number Z={atomic_number} outside valid chemical range [0, 118].")

    el = element(atomic_number)
    symbol = f"{isotope_number}{el.symbol}" if isotope_number is not None else el.symbol
    return isotope_mass(symbol)


def get_monoisotopic_masses(atomic_numbers: Sequence[int]) -> List[float]:
    """Dynamically query monoisotopic masses for a sequence of atomic numbers. [M]"""
    return [get_monoisotopic_mass(z) for z in atomic_numbers]


def get_monoisotopic_masses_tensor(
    atomic_numbers: Sequence[int],
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return dynamic monoisotopic masses as a PyTorch tensor. [M]"""
    masses = get_monoisotopic_masses(atomic_numbers)
    return torch.tensor(masses, dtype=dtype, device=device)


@functools.lru_cache(maxsize=128)
def resolve_ciaaw_monoisotopic_mass(atomic_number: int) -> float:
    """Dynamically resolve the CIAAW monoisotopic mass for the most abundant isotope. [M]"""
    return get_monoisotopic_mass(atomic_number)


def get_atomic_masses(
    atomic_numbers: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Retrieve most-abundant-isotope masses in unified atomic mass units (u). [M]

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Tensor of atomic numbers Z of shape (N_atoms,).
    device : torch.device | str
        Target device for tensor allocation.

    Returns
    -------
    torch.Tensor
        Tensor of atomic masses in unified atomic mass units (u) of shape (N_atoms, 1)
        and dtype torch.float64.
    """
    masses = []
    for z in atomic_numbers.view(-1).tolist():
        masses.append(get_monoisotopic_mass(z))
    return torch.tensor(masses, dtype=torch.float64, device=device).unsqueeze(-1)


