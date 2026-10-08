"""Immutable Pydantic v2 data models and enums for CoChem Mobile complex assembly.

Strict adherence to the Zero-Mock mandate with frozen configuration and strict
validation.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cochem.mobile.assembly.constants import (
    get_atomic_number,
    validate_transition_metal,
)


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

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    ligand_id: Annotated[
        str, Field(min_length=1, description="Unique identifier for the ligand")
    ]
    donor_atom_indices: Annotated[
        list[StrictInt],
        Field(description="0-indexed donor atom indices in ligand conformer"),
    ]
    denticity: Annotated[
        StrictInt,
        Field(ge=1, le=6, description="Denticity of the ligand attachment"),
    ]
    target_vector_slots: Annotated[
        list[StrictInt],
        Field(description="Assigned coordination template vector slot indices"),
    ]
    atomic_symbols: Annotated[
        list[str],
        Field(min_length=1, description="List of elemental symbols in conformer order"),
    ]
    coordinates: Annotated[
        list[
            tuple[
                Annotated[float, Field(strict=True)],
                Annotated[float, Field(strict=True)],
                Annotated[float, Field(strict=True)],
            ]
        ],
        Field(
            min_length=1,
            description="Pre-calculated rigid 3D conformer coordinates in Angstroms",
        ),
    ]
    formal_charge: Annotated[
        StrictInt | None,
        Field(
            default=None,
            description=(
                "Explicit caller-declared ligand formal charge. Missing "
                "charge permits inspection, but prohibits assembly; "
                "names and composition cannot determine it."
            ),
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
                f"Number of target vector slots ({len(self.target_vector_slots)}) "
                "must match "
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
        for symbol in self.atomic_symbols:
            if re.fullmatch(r"[A-Z][a-z]?", symbol) is None:
                raise ValueError(f"Expected an elemental symbol, received {symbol!r}.")
            get_atomic_number(symbol)

        num_atoms = len(self.atomic_symbols)
        for idx in self.donor_atom_indices:
            if not (0 <= idx < num_atoms):
                raise ValueError(
                    f"Donor atom index {idx} out of range "
                    f"for ligand with {num_atoms} atoms."
                )

        return self


class ComplexAssemblyRequest(BaseModel):
    """Specification of a complete transition metal complex assembly request."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metal_symbol: Annotated[
        str,
        Field(
            min_length=1, max_length=3, description="Transition metal elemental symbol"
        ),
    ]
    oxidation_state: Annotated[
        StrictInt,
        Field(
            ge=0,
            le=7,
            description="Formal oxidation state of the central transition metal",
        ),
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
        StrictInt,
        Field(ge=1, description="Spin multiplicity (2S + 1) of the complex"),
    ]
    alpha_vdw: Annotated[
        float,
        Field(
            default=0.60, ge=0.1, le=1.0, description="Steric clash tolerance factor"
        ),
    ]

    @field_validator("metal_symbol")
    @classmethod
    def validate_metal(cls, v: str) -> str:
        """Validate metal symbol is a supported transition metal."""
        validate_transition_metal(v)
        return v


class StericClashReport(BaseModel):
    """A scaled database-radius contact heuristic, without force or accuracy claims.

    Schema 2 replaces the incorrect historical ``bondi_threshold`` name.
    An absent eligible pair produces null distance/cutoff, never a synthetic zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["cochem.mobile.assembly.clash/2"] = (
        "cochem.mobile.assembly.clash/2"
    )
    radius_source: Literal["mendeleev.vdw_radius"] = "mendeleev.vdw_radius"
    method: Literal["scaled_database_radius_contact_heuristic"] = (
        "scaled_database_radius_contact_heuristic"
    )
    evaluated_pair_count: Annotated[StrictInt, Field(ge=0)]
    initial_clash_detected: bool
    initial_clash_pairs: list[tuple[int, int]] = Field(default_factory=list)

    clash_detected: Annotated[
        bool,
        Field(
            description=(
                "True if a final eligible atom pair violates the scaled "
                "database-radius threshold"
            )
        ),
    ]
    clash_pairs: Annotated[
        list[tuple[int, int]],
        Field(
            default_factory=list, description="Pairs of global atom indices in clash"
        ),
    ]
    min_observed_distance: Annotated[
        float | None,
        Field(
            ge=0,
            description=(
                "Minimum eligible interatomic distance in Angstroms, or "
                "null if no pair was evaluated"
            ),
        ),
    ]
    vdw_contact_threshold: Annotated[
        float | None,
        Field(
            gt=0,
            description=(
                "Scaled sum of named Mendeleev vdw_radius values for the "
                "closest eligible pair, in Angstroms; null if none"
            ),
        ),
    ]
    clash_resolved: Annotated[
        bool,
        Field(
            description=(
                "True only if an initially detected heuristic clash was resolved"
            )
        ),
    ]

    @model_validator(mode="after")
    def validate_observed_pairs(self) -> StericClashReport:
        missing = (
            self.min_observed_distance is None or self.vdw_contact_threshold is None
        )
        if missing != (self.evaluated_pair_count == 0):
            raise ValueError("Distance and cutoff require an actually evaluated pair.")
        if (self.min_observed_distance is None) != (self.vdw_contact_threshold is None):
            raise ValueError("Distance and cutoff availability must agree.")
        if self.clash_detected != bool(self.clash_pairs):
            raise ValueError("Final clash status must agree with actual final pairs.")
        if self.initial_clash_detected != bool(self.initial_clash_pairs):
            raise ValueError(
                "Initial clash status must agree with actual initial pairs."
            )
        if self.clash_resolved != (
            self.initial_clash_detected and not self.clash_detected
        ):
            raise ValueError(
                "Resolved status requires an initially observed clash "
                "and no final clash."
            )
        return self


class ComplexAssemblyResult(BaseModel):
    """Result payload emitted upon completion of complex assembly."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["cochem.mobile.assembly.result/2"] = (
        "cochem.mobile.assembly.result/2"
    )
    geometry_status: Literal["initial_geometric_proposal"] = (
        "initial_geometric_proposal"
    )
    geometry_model: Literal["coordination_template_and_pyykko_radius_sum"] = (
        "coordination_template_and_pyykko_radius_sum"
    )
    chemical_state_source: Literal["caller_declared_not_independently_verified"] = (
        "caller_declared_not_independently_verified"
    )
    scientific_qualification: Literal[False] = False
    quantum_calculation_performed: Literal[False] = False
    actual_d_population: None = None
    d_electron_count_model: Literal["formal_group_minus_oxidation_bookkeeping"] = (
        "formal_group_minus_oxidation_bookkeeping"
    )
    total_electron_count: Annotated[StrictInt, Field(gt=0)]
    formal_valence_electron_count: StrictInt
    d_electron_count_status: Literal["formal_bookkeeping", "outside_d_shell_range"]
    uff_status: Literal["unavailable_without_declared_molecular_graph"] = (
        "unavailable_without_declared_molecular_graph"
    )

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
        StrictInt | None,
        Field(
            ge=0,
            le=10,
            description=(
                "Formal group-minus-oxidation d-electron bookkeeping; "
                "null outside its 0..10 applicability range. This is not "
                "an electronic population analysis."
            ),
        ),
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
        Field(
            default=None, description="Constrained UFF force field energy in kcal/mol"
        ),
    ]

    @model_validator(mode="after")
    def validate_formal_bookkeeping(self) -> ComplexAssemblyResult:
        in_range = 0 <= self.formal_valence_electron_count <= 10
        expected_count = self.formal_valence_electron_count if in_range else None
        expected_status = "formal_bookkeeping" if in_range else "outside_d_shell_range"
        if (
            self.d_electron_count != expected_count
            or self.d_electron_count_status != expected_status
        ):
            raise ValueError(
                "Formal d-shell bookkeeping must preserve its raw value "
                "and applicability."
            )
        if self.uff_energy_kcal_mol is not None:
            raise ValueError(
                "This assembly request contains no molecular graph; UFF "
                "energy is unavailable."
            )
        return self
