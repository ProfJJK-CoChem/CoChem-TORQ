"""Immutable Pydantic v2 data models and enums for CoChem Mobile complex assembly.

Strict adherence to the Zero-Mock mandate with frozen configuration and strict validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cochem.mobile.assembly.constants import validate_transition_metal


class CoordinationGeometryEnum(str, Enum):
    """Supported coordination geometries and polyhedra."""

    LINEAR = "LINEAR"  # CN = 2
    TRIGONAL_PLANAR = "TRIGONAL_PLANAR"  # CN = 3
    TETRAHEDRAL = "TETRAHEDRAL"  # CN = 4
    SQUARE_PLANAR = "SQUARE_PLANAR"  # CN = 4
    TRIGONAL_BIPYRAMIDAL = "TRIGONAL_BIPYRAMIDAL"  # CN = 5
    SQUARE_PYRAMIDAL = "SQUARE_PYRAMIDAL"  # CN = 5
    OCTAHEDRAL = "OCTAHEDRAL"  # CN = 6
    PENTAGONAL_BIPYRAMIDAL = "PENTAGONAL_BIPYRAMIDAL"  # CN = 7
    SQUARE_ANTIPRISMATIC = "SQUARE_ANTIPRISMATIC"  # CN = 8

    @property
    def coordination_number(self) -> int:
        """Return the ideal coordination number (CN) for this geometry."""
        cn_map = {
            CoordinationGeometryEnum.LINEAR: 2,
            CoordinationGeometryEnum.TRIGONAL_PLANAR: 3,
            CoordinationGeometryEnum.TETRAHEDRAL: 4,
            CoordinationGeometryEnum.SQUARE_PLANAR: 4,
            CoordinationGeometryEnum.TRIGONAL_BIPYRAMIDAL: 5,
            CoordinationGeometryEnum.SQUARE_PYRAMIDAL: 5,
            CoordinationGeometryEnum.OCTAHEDRAL: 6,
            CoordinationGeometryEnum.PENTAGONAL_BIPYRAMIDAL: 7,
            CoordinationGeometryEnum.SQUARE_ANTIPRISMATIC: 8,
        }
        return cn_map[self]


class LigandAttachment(BaseModel):
    """Specification of an individual ligand attachment with conformer coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ligand_id: Annotated[str, Field(min_length=1, description="Unique identifier for the ligand")]
    donor_atom_indices: Annotated[
        list[int],
        Field(description="0-indexed donor atom indices in ligand conformer"),
    ]
    denticity: Annotated[
        int,
        Field(ge=1, le=6, description="Denticity of the ligand attachment"),
    ]
    target_vector_slots: Annotated[
        list[int],
        Field(description="Assigned coordination template vector slot indices"),
    ]
    atomic_symbols: Annotated[
        list[str],
        Field(min_length=1, description="List of elemental symbols in conformer order"),
    ]
    coordinates: Annotated[
        list[tuple[float, float, float]],
        Field(
            min_length=1, description="Pre-calculated rigid 3D conformer coordinates in Angstroms"
        ),
    ]

    @model_validator(mode="after")
    def validate_attachment(self) -> LigandAttachment:
        """Validate geometric and structural integrity of the ligand attachment."""
        if len(self.donor_atom_indices) != self.denticity:
            raise ValueError(
                f"Number of donor indices ({len(self.donor_atom_indices)}) must match "
                f"denticity ({self.denticity})."
            )
        if len(self.target_vector_slots) != self.denticity:
            raise ValueError(
                f"Number of target vector slots ({len(self.target_vector_slots)}) must match "
                f"denticity ({self.denticity})."
            )
        if len(self.coordinates) != len(self.atomic_symbols):
            raise ValueError(
                f"Number of coordinates ({len(self.coordinates)}) must match "
                f"number of atomic symbols ({len(self.atomic_symbols)})."
            )
        if len(set(self.donor_atom_indices)) != len(self.donor_atom_indices):
            raise ValueError("Donor atom indices must be distinct.")
        if len(set(self.target_vector_slots)) != len(self.target_vector_slots):
            raise ValueError("Target vector slots must be distinct.")

        num_atoms = len(self.atomic_symbols)
        for idx in self.donor_atom_indices:
            if not (0 <= idx < num_atoms):
                raise ValueError(
                    f"Donor atom index {idx} out of range for ligand with {num_atoms} atoms."
                )

        return self


class ComplexAssemblyRequest(BaseModel):
    """Specification of a complete transition metal complex assembly request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metal_symbol: Annotated[
        str,
        Field(min_length=1, max_length=3, description="Transition metal elemental symbol"),
    ]
    oxidation_state: Annotated[
        int,
        Field(ge=0, le=7, description="Formal oxidation state of the central transition metal"),
    ]
    geometry: Annotated[
        CoordinationGeometryEnum,
        Field(description="Target coordination template geometry"),
    ]
    ligands: Annotated[
        list[LigandAttachment],
        Field(min_length=1, description="List of ligand attachments to assemble"),
    ]
    spin_multiplicity: Annotated[
        int,
        Field(ge=1, description="Spin multiplicity (2S + 1) of the complex"),
    ]
    alpha_vdw: Annotated[
        float,
        Field(default=0.60, ge=0.1, le=1.0, description="Steric clash tolerance factor"),
    ]

    @field_validator("metal_symbol")
    @classmethod
    def validate_metal(cls, v: str) -> str:
        """Validate metal symbol is a supported transition metal."""
        validate_transition_metal(v)
        return v


class StericClashReport(BaseModel):
    """Steric clash analysis report resulting from pairwise Bondi contact evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clash_detected: Annotated[
        bool,
        Field(description="True if non-bonded atom pair distance violates Bondi threshold"),
    ]
    clash_pairs: Annotated[
        list[tuple[int, int]],
        Field(default_factory=list, description="Pairs of global atom indices in clash"),
    ]
    min_observed_distance: Annotated[
        float,
        Field(description="Minimum non-bonded interatomic distance observed in Angstroms"),
    ]
    bondi_threshold: Annotated[
        float,
        Field(
            description="Calculated scaled Bondi cutoff distance for closest non-bonded pair in Angstroms"
        ),
    ]
    clash_resolved: Annotated[
        bool,
        Field(description="True if clashes were resolved via dihedral rotation / UFF relaxation"),
    ]


class ComplexAssemblyResult(BaseModel):
    """Result payload emitted upon completion of complex assembly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xyz_data: Annotated[
        str,
        Field(description="Valid POSIX/Windows UTF-8 .xyz formatted string"),
    ]
    hdf5_record_key: Annotated[
        str,
        Field(description="Unique record key in SWMR HDF5 persistence store"),
    ]
    sha256_provenance: Annotated[
        str,
        Field(description="Hexadecimal SHA-256 cryptographic provenance hash"),
    ]
    total_formal_charge: Annotated[
        int,
        Field(description="Total formal charge of the assembled coordination complex"),
    ]
    d_electron_count: Annotated[
        int,
        Field(ge=0, description="d-electron count (d^n) of the transition metal center"),
    ]
    spin_multiplicity: Annotated[
        int,
        Field(ge=1, description="Spin multiplicity (2S + 1) of the complex"),
    ]
    clash_report: Annotated[
        StericClashReport,
        Field(description="Steric clash and relaxation report"),
    ]
    uff_energy_kcal_mol: Annotated[
        float | None,
        Field(default=None, description="Constrained UFF force field energy in kcal/mol"),
    ]
