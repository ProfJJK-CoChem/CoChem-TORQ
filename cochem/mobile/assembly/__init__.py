"""CoChem Mobile Algorithmic Complex Assembly Engine (SRS Chunk 07).

Production-grade, headless transition metal coordination complex builder adhering strictly
to the Zero-Mock mandate, Mendeleev Library Mandate, and Method Matrix quantum invariants.
"""

from __future__ import annotations

from cochem.mobile.assembly.alignment import (
    align_ligand_to_template,
    calculate_metal_donor_distance,
    kabsch_fit_proper,
    rotation_matrix_from_vectors,
)
from cochem.mobile.assembly.clash import (
    evaluate_steric_clashes,
    perform_rodrigues_dihedral_sweep,
    resolve_clashes_and_report,
    rodrigues_rotate_point,
    run_constrained_uff_relaxation,
)
from cochem.mobile.assembly.constants import (
    calculate_d_electron_count,
    get_atomic_number,
    get_covalent_radius_angstrom,
    get_standard_atomic_weight,
    get_vdw_radius_angstrom,
    validate_quantum_parity,
    validate_transition_metal,
)
from cochem.mobile.assembly.engine import (
    assemble_complex_async,
    assemble_complex_sync,
    infer_ligand_charge,
)
from cochem.mobile.assembly.exceptions import (
    CoChemMOBException,
    CoordinationGeometryMismatchError,
    KabschReflectionError,
    MendeleevLookupError,
    QuantumParityError,
    SingularRotationAxisError,
    StericClashDetectedError,
    SWMRStorageLockError,
    UnphysicalMonomerSeparationError,
)
from cochem.mobile.assembly.geometries import get_coordination_template_vectors
from cochem.mobile.assembly.models import (
    ComplexAssemblyRequest,
    ComplexAssemblyResult,
    CoordinationGeometryEnum,
    LigandAttachment,
    StericClashReport,
)
from cochem.mobile.assembly.storage import (
    calculate_provenance_hash,
    calculate_stoichiometry,
    format_xyz_string,
    read_complex_from_hdf5,
    save_complex_to_hdf5,
)

__all__ = [
    "CoChemMOBException",
    "ComplexAssemblyRequest",
    "ComplexAssemblyResult",
    "CoordinationGeometryEnum",
    "CoordinationGeometryMismatchError",
    "KabschReflectionError",
    "LigandAttachment",
    "MendeleevLookupError",
    "QuantumParityError",
    "SingularRotationAxisError",
    "StericClashDetectedError",
    "StericClashReport",
    "SWMRStorageLockError",
    "UnphysicalMonomerSeparationError",
    "align_ligand_to_template",
    "assemble_complex_async",
    "assemble_complex_sync",
    "calculate_d_electron_count",
    "calculate_metal_donor_distance",
    "calculate_provenance_hash",
    "calculate_stoichiometry",
    "evaluate_steric_clashes",
    "format_xyz_string",
    "get_atomic_number",
    "get_coordination_template_vectors",
    "get_covalent_radius_angstrom",
    "get_standard_atomic_weight",
    "get_vdw_radius_angstrom",
    "infer_ligand_charge",
    "kabsch_fit_proper",
    "perform_rodrigues_dihedral_sweep",
    "read_complex_from_hdf5",
    "resolve_clashes_and_report",
    "rodrigues_rotate_point",
    "rotation_matrix_from_vectors",
    "run_constrained_uff_relaxation",
    "save_complex_to_hdf5",
    "validate_quantum_parity",
    "validate_transition_metal",
]
