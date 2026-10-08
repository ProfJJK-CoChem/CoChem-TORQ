"""Tripartite air-gapped async assembly dispatcher and core assembly engine.

Strict adherence to the Zero-Mock mandate, Method Matrix invariants, and quantum parity
checks.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from cochem.mobile.assembly.alignment import align_ligand_to_template
from cochem.mobile.assembly.clash import resolve_clashes_and_report
from cochem.mobile.assembly.constants import (
    formal_transition_metal_electron_record,
    get_atomic_number,
    validate_quantum_parity,
)
from cochem.mobile.assembly.exceptions import (
    CoordinationGeometryMismatchError,
    QuantumParityError,
)
from cochem.mobile.assembly.geometries import get_coordination_template_vectors
from cochem.mobile.assembly.models import (
    ComplexAssemblyRequest,
    ComplexAssemblyResult,
    LigandAttachment,
)
from cochem.mobile.assembly.storage import (
    calculate_provenance_hash,
    calculate_stoichiometry,
    format_xyz_string,
    save_complex_to_hdf5,
)

logger = logging.getLogger(__name__)


def infer_ligand_charge(ligand: LigandAttachment) -> int:
    """Return the explicitly declared charge; the legacy name does not imply inference.

    Composition and ligand labels cannot establish a formal electronic state.
    Missing charge is permitted for geometry inspection, but cannot be used
    to assemble or persist a chemical-state-bearing complex.
    """
    if ligand.formal_charge is None:
        raise ValueError(
            f"Ligand {ligand.ligand_id!r} requires an explicit "
            "formal_charge before assembly; "
            "its name and atomic composition do not establish charge."
        )
    if type(ligand.formal_charge) is not int:
        raise ValueError("Ligand formal_charge must be an explicitly declared integer.")
    return ligand.formal_charge


def assemble_complex_sync(
    request: ComplexAssemblyRequest,
    scratch_dir: Path | None = None,
    h5_path: Path | None = None,
) -> ComplexAssemblyResult:
    """Build and persist a heuristic geometric proposal with a declared state.

    1. Validates denticity matching and slot coverage.
    2. Requires explicit ligand charges and validates whole-complex electron parity.
    3. Generates ideal coordination template vectors.
    4. Rigidly aligns and transforms all ligand conformers.
    5. Applies the scaled database-radius contact heuristic and Rodrigues sweeps.
    6. Formats .xyz coordinates and commits to SWMR HDF5 store.
    7. Emits a typed, scientifically unqualified initial geometric proposal.

    Args:
        request: Validated ComplexAssemblyRequest payload.
        scratch_dir: Optional working directory for scratch artifacts.
        h5_path: Optional explicit path to HDF5 database.

    Returns:
        ComplexAssemblyResult containing xyz string, HDF5 key, hash, clash report, and
        energies.

    Raises:
        CoordinationGeometryMismatchError: If total denticity != CN or slots are
        invalid.
        QuantumParityError: If spin multiplicity violates quantum parity.
        MendeleevLookupError: If an element cannot be resolved.
        SingularRotationAxisError: If rotation axis vector is singular.
        KabschReflectionError: If proper SO(3) rotation fails.
        StericClashDetectedError: If steric overlap cannot be resolved.
        UnphysicalMonomerSeparationError: If COM distance is outside bounds.
        SWMRStorageLockError: If HDF5 lock acquisition fails.
    """
    target_cn = request.geometry.coordination_number

    # 1. Validate denticity sum matching
    total_denticity = sum(lig.denticity for lig in request.ligands)
    if total_denticity != target_cn:
        raise CoordinationGeometryMismatchError(
            f"Total ligand denticity ({total_denticity}) "
            "does not match coordination number "
            f"({target_cn}) for geometry '{request.geometry.value}'."
        )

    # 2. Validate template slot coverage
    all_assigned_slots: list[int] = []
    for lig in request.ligands:
        all_assigned_slots.extend(lig.target_vector_slots)

    if len(all_assigned_slots) != target_cn:
        raise CoordinationGeometryMismatchError(
            f"Total assigned slots ({len(all_assigned_slots)}) "
            f"does not match coordination number ({target_cn})."
        )

    for slot in all_assigned_slots:
        if not (0 <= slot < target_cn):
            raise CoordinationGeometryMismatchError(
                f"Assigned vector slot {slot} is out of valid range "
                f"[0, {target_cn - 1}]."
            )

    if len(set(all_assigned_slots)) != target_cn:
        raise CoordinationGeometryMismatchError(
            f"Duplicate vector slots detected in assignment: {all_assigned_slots}"
        )

    # 3. Whole-complex state counting. The metal oxidation state and ligand
    # charges are caller declarations, not graph/population-analysis results.
    total_formal_charge = request.oxidation_state + sum(
        infer_ligand_charge(lig) for lig in request.ligands
    )
    all_declared_symbols = [request.metal_symbol] + [
        symbol for ligand in request.ligands for symbol in ligand.atomic_symbols
    ]
    total_electrons = (
        sum(get_atomic_number(symbol) for symbol in all_declared_symbols)
        - total_formal_charge
    )
    if total_electrons <= 0:
        raise QuantumParityError(
            "Assembly requires a positive whole-complex electron count."
        )
    validate_quantum_parity(total_electrons, request.spin_multiplicity)
    metal_bookkeeping = formal_transition_metal_electron_record(
        request.metal_symbol, request.oxidation_state
    )
    d_count = metal_bookkeeping["d_electron_count"]

    # 4. Generate ideal coordination template vectors
    template_vectors = get_coordination_template_vectors(request.geometry)

    # 5. Align each ligand attachment
    all_symbols: list[str] = [request.metal_symbol]
    aligned_coord_blocks: list[np.ndarray] = [
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    ]

    ligand_slices: list[tuple[int, int]] = []
    donor_global_indices: set[int] = set()
    monodentate_info: list[tuple[int, int, int]] = []

    current_atom_idx = 1
    for lig_idx, lig in enumerate(request.ligands):
        lig_coords = align_ligand_to_template(
            lig, request.metal_symbol, template_vectors
        )
        num_lig_atoms = len(lig.atomic_symbols)
        start_idx = current_atom_idx
        end_idx = current_atom_idx + num_lig_atoms

        ligand_slices.append((start_idx, end_idx))
        all_symbols.extend(lig.atomic_symbols)
        aligned_coord_blocks.append(lig_coords)

        # Track donor global indices
        for donor_local in lig.donor_atom_indices:
            donor_global_indices.add(start_idx + donor_local)

        if lig.denticity == 1:
            monodentate_info.append(
                (lig_idx, start_idx, start_idx + lig.donor_atom_indices[0])
            )

        current_atom_idx = end_idx

    initial_coordinates = np.vstack(aligned_coord_blocks)

    # 6. Contact heuristic and rigid dihedral sweep. No molecular graph was
    # declared in this request, so UFF is explicitly unavailable.
    final_coordinates, clash_report, uff_energy = resolve_clashes_and_report(
        all_symbols,
        initial_coordinates,
        ligand_slices,
        monodentate_info,
        donor_global_indices,
        request.alpha_vdw,
    )

    # 7. Compute stoichiometry; formal charge was already explicitly validated.
    stoichiometry = calculate_stoichiometry(all_symbols)

    # 8. Compute cryptographic provenance hash and format .xyz
    req_dict = request.model_dump(mode="json")
    sha256_hash = calculate_provenance_hash(req_dict, final_coordinates)
    xyz_string = format_xyz_string(
        all_symbols,
        final_coordinates,
        stoichiometry,
        total_formal_charge,
        request.spin_multiplicity,
        sha256_hash,
    )

    # 9. Commit to SWMR HDF5 store
    record_key = f"{stoichiometry}_{sha256_hash[:12]}"
    if h5_path is None:
        target_dir = scratch_dir or Path.cwd()
        h5_path = target_dir / "complexes.h5"

    atomic_numbers = np.array(
        [get_atomic_number(sym) for sym in all_symbols],
        dtype=np.int32,
    )

    metadata: dict[str, Any] = {
        "stoichiometry": stoichiometry,
        "metal_symbol": request.metal_symbol,
        "oxidation_state": request.oxidation_state,
        "geometry": request.geometry.value,
        "total_formal_charge": total_formal_charge,
        "total_electron_count": total_electrons,
        "ligand_formal_charges": [lig.formal_charge for lig in request.ligands],
        "chemical_state_source": "caller_declared_not_independently_verified",
        "geometry_status": "initial_geometric_proposal",
        "geometry_model": "coordination_template_and_pyykko_radius_sum",
        "scientific_qualification": False,
        "quantum_calculation_performed": False,
        "d_electron_count": d_count,
        "d_electron_count_model": "formal_group_minus_oxidation_bookkeeping",
        "formal_valence_electron_count": metal_bookkeeping[
            "formal_valence_electron_count"
        ],
        "d_electron_count_status": metal_bookkeeping["d_electron_count_status"],
        "actual_d_population": None,
        "spin_multiplicity": request.spin_multiplicity,
        "sha256_provenance": sha256_hash,
        "uff_energy_kcal_mol": uff_energy,
        "uff_status": "unavailable_without_declared_molecular_graph",
        "clash_report": clash_report.model_dump(mode="json"),
        "alpha_vdw": request.alpha_vdw,
    }

    save_complex_to_hdf5(
        h5_path=h5_path,
        record_key=record_key,
        coordinates=final_coordinates,
        atomic_numbers=atomic_numbers,
        metadata=metadata,
    )

    return ComplexAssemblyResult(
        xyz_data=xyz_string,
        hdf5_record_key=record_key,
        sha256_provenance=sha256_hash,
        total_formal_charge=total_formal_charge,
        total_electron_count=total_electrons,
        d_electron_count=d_count,
        formal_valence_electron_count=metal_bookkeeping[
            "formal_valence_electron_count"
        ],
        d_electron_count_status=metal_bookkeeping["d_electron_count_status"],
        spin_multiplicity=request.spin_multiplicity,
        clash_report=clash_report,
        uff_energy_kcal_mol=uff_energy,
    )


async def assemble_complex_async(
    request: ComplexAssemblyRequest,
    scratch_dir: Path | None = None,
    h5_path: Path | None = None,
) -> ComplexAssemblyResult:
    """Asynchronously assemble a transition metal complex off the main event loop.

    Args:
        request: Validated ComplexAssemblyRequest.
        scratch_dir: Optional directory for scratch data.
        h5_path: Optional explicit HDF5 database path.

    Returns:
        ComplexAssemblyResult.
    """
    return await asyncio.to_thread(
        assemble_complex_sync,
        request,
        scratch_dir,
        h5_path,
    )
