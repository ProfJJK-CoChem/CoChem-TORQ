"""CoChem-TORQ Counterpoise Workflow & Single-Point Interaction Bracketing Module.

Compliant with Method Matrix v4 §1.2, §8B, and Anti-Spoofing Protocol v2.
Decouples single-point counterpoise corrections from active geometry optimization decks.
Enforces Boys-Bernardi interaction energy evaluation:
Delta E_CP = E_AB^{AB} - E_A^{AB} - E_B^{AB} [M].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from cochem_base.schemas import CounterpoiseResult, QuantumJobSpec


def calculate_discrete_counterpoise_energy(
    e_ab: float, e_a_ghost: float, e_b_ghost: float
) -> float:
    """Evaluate discrete 3-point counterpoise interaction energy on frozen-monomer relaxed structure.

    Delta E_CP = E_AB^{AB} - E_A^{AB} - E_B^{AB} [M].

    Parameters
    ----------
    e_ab : float
        Total energy of complex AB in the full dimer basis set (Hartree).
    e_a_ghost : float
        Single-point energy of Monomer A with Monomer B centers ghosted (Hartree).
    e_b_ghost : float
        Single-point energy of Monomer B with Monomer A centers ghosted (Hartree).

    Returns
    -------
    float
        Counterpoise-corrected interaction energy Delta E_CP in Hartrees [M].
    """
    return float(e_ab - e_a_ghost - e_b_ghost)


def calculate_counterpoise_correction(
    e_ab_ab: float,
    e_a_ab: float,
    e_b_ab: float,
    e_a_a: Optional[float] = None,
    e_b_b: Optional[float] = None,
) -> CounterpoiseResult:
    """Evaluate complete counterpoise interaction energy and BSSE correction [M].

    Parameters
    ----------
    e_ab_ab : float
        Total energy of complex AB in the full dimer basis set (Hartree).
    e_a_ab : float
        Single-point energy of Monomer A with Monomer B centers ghosted (Hartree).
    e_b_ab : float
        Single-point energy of Monomer B with Monomer A centers ghosted (Hartree).
    e_a_a : Optional[float]
        Single-point energy of Monomer A in its own monomer basis (Hartree).
    e_b_b : Optional[float]
        Single-point energy of Monomer B in its own monomer basis (Hartree).

    Returns
    -------
    CounterpoiseResult
        Structured container reporting Delta E_CP, BSSE, and raw interaction energy.
    """
    delta_e_cp = calculate_discrete_counterpoise_energy(e_ab_ab, e_a_ab, e_b_ab)
    e_bsse = None
    delta_e_raw = None

    if e_a_a is not None and e_b_b is not None:
        e_bsse = float((e_a_a - e_a_ab) + (e_b_b - e_b_ab))
        delta_e_raw = float(e_ab_ab - e_a_a - e_b_b)

    return CounterpoiseResult(
        delta_e_cp=delta_e_cp,
        e_ab_ab=e_ab_ab,
        e_a_ab=e_a_ab,
        e_b_ab=e_b_ab,
        e_a_a=e_a_a,
        e_b_b=e_b_b,
        e_bsse=e_bsse,
        delta_e_raw=delta_e_raw,
        provenance_tag="[M]",
    )


def generate_counterpoise_jobs(
    job_id: str,
    symbols: Sequence[str],
    coordinates: Union[Sequence[Sequence[float]], np.ndarray],
    atoms_a: Sequence[int],
    atoms_b: Sequence[int],
    method: str = "wB97M-V",
    basis_set: str = "def2-TZVP",
    aux_basis: Optional[str] = "def2/J",
    charge: int = 0,
    multiplicity: int = 1,
    extra_options: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, QuantumJobSpec]:
    """Generate discrete multi-job specifications for post-optimization counterpoise bracketing.

    Decouples counterpoise job generation from active geometry optimization.
    Dispatches 3 discrete single-point calculations on the converged unghosted geometry:
      1. CP_E_AB_AB: Complex in full dimer basis
      2. CP_E_A_AB: Monomer A in dimer basis (Monomer B ghosted)
      3. CP_E_B_AB: Monomer B in dimer basis (Monomer A ghosted)

    Parameters
    ----------
    job_id : str
        Base identifier for the calculation suite.
    symbols : Sequence[str]
        Atomic symbols of the dimer complex.
    coordinates : Union[Sequence[Sequence[float]], np.ndarray]
        Converged Cartesian coordinates in Angstroms.
    atoms_a : Sequence[int]
        0-based atom indices of Monomer A.
    atoms_b : Sequence[int]
        0-based atom indices of Monomer B.
    method : str
        Quantum functional or wave function method.
    basis_set : str
        Primary basis set name.
    aux_basis : Optional[str]
        Auxiliary fitting basis set name.
    charge : int
        Total charge of the complex.
    multiplicity : int
        Spin multiplicity of the complex.
    extra_options : str
        Additional engine flags.
    metadata : Optional[Dict[str, Any]]
        Contextual metadata.

    Returns
    -------
    Dict[str, QuantumJobSpec]
        Dictionary mapping job keys ('E_AB_AB', 'E_A_AB', 'E_B_AB') to job specifications.
    """
    coords_list = (
        coordinates.tolist() if hasattr(coordinates, "tolist") else list(coordinates)
    )
    meta = metadata or {}

    job_ab = QuantumJobSpec(
        job_id=f"{job_id}_cp_ab",
        symbols=list(symbols),
        coordinates=coords_list,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        ghost_atom_indices=None,
        job_type="CP_E_AB_AB",
        extra_options=extra_options,
        metadata={**meta, "counterpoise_component": "AB_in_AB_basis"},
    )

    job_a = QuantumJobSpec(
        job_id=f"{job_id}_cp_a",
        symbols=list(symbols),
        coordinates=coords_list,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        ghost_atom_indices=sorted(list(atoms_b)),
        job_type="CP_E_A_AB",
        extra_options=extra_options,
        metadata={**meta, "counterpoise_component": "A_in_AB_basis"},
    )

    job_b = QuantumJobSpec(
        job_id=f"{job_id}_cp_b",
        symbols=list(symbols),
        coordinates=coords_list,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis_set=basis_set,
        aux_basis=aux_basis,
        ghost_atom_indices=sorted(list(atoms_a)),
        job_type="CP_E_B_AB",
        extra_options=extra_options,
        metadata={**meta, "counterpoise_component": "B_in_AB_basis"},
    )

    return {
        "E_AB_AB": job_ab,
        "E_A_AB": job_a,
        "E_B_AB": job_b,
    }


__all__ = [
    "calculate_discrete_counterpoise_energy",
    "calculate_counterpoise_correction",
    "generate_counterpoise_jobs",
]
