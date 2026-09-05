"""CoChem Mobile 2D/3D Data Schemas.

Pydantic v2 schemas adhering strictly to the Zero-Mock mandate and immutable ConfigDict.
"""

from __future__ import annotations

from typing import Annotated

import mendeleev
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AtomCoordinate2D(BaseModel):
    """2D atom coordinate representation from the chemical sketcher."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_index: Annotated[int, Field(ge=0, description="0-indexed identifier of the atom")]
    symbol: Annotated[
        str,
        Field(min_length=1, max_length=3, description="IUPAC chemical element symbol"),
    ]
    x: Annotated[float, Field(description="2D Cartesian X coordinate in screen/canvas units")]
    y: Annotated[float, Field(description="2D Cartesian Y coordinate in screen/canvas units")]
    charge: Annotated[int, Field(default=0, ge=-7, le=7, description="Formal charge of the atom")]

    @property
    def atomic_weight(self) -> float:
        """Dynamic atomic weight derived from Mendeleev."""
        return float(mendeleev.element(self.symbol).atomic_weight)


class AtomCoordinate3D(BaseModel):
    """3D atom coordinate representation from synthesized conformer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_index: Annotated[
        int, Field(ge=0, description="0-indexed identifier of the atom in 3D conformer")
    ]
    symbol: Annotated[
        str,
        Field(min_length=1, max_length=3, description="IUPAC chemical element symbol"),
    ]
    x: Annotated[float, Field(description="3D Cartesian X coordinate in Angstroms")]
    y: Annotated[float, Field(description="3D Cartesian Y coordinate in Angstroms")]
    z: Annotated[float, Field(description="3D Cartesian Z coordinate in Angstroms")]

    @property
    def atomic_weight(self) -> float:
        """Dynamic atomic weight derived from Mendeleev."""
        return float(mendeleev.element(self.symbol).atomic_weight)


class ValenceValidationResultSchema(BaseModel):
    """Sanitization and valence validation result schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: Annotated[
        bool,
        Field(description="True if molecule passes valence and sanitization checks"),
    ]
    diagnostic_message: Annotated[
        str,
        Field(default="", description="Sanitization diagnostic message or error detail"),
    ]
    atom_error_indices: Annotated[
        list[int],
        Field(
            default_factory=list,
            description="0-indexed list of atom indices causing valence violations",
        ),
    ]


class SketcherPayloadSchema(BaseModel):
    """Payload emitted by the touch sketcher widget to the Python backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    smiles: Annotated[str, Field(min_length=1, description="SMILES string representation")]
    molfile_v2000: Annotated[
        str, Field(min_length=10, description="MDL Molfile V2000 format block")
    ]
    chiral_centers_count: Annotated[
        int,
        Field(default=0, ge=0, description="Number of chiral centers identified in sketch"),
    ]
    atoms_2d: Annotated[
        list[AtomCoordinate2D],
        Field(
            default_factory=list,
            description="List of 2D atom coordinates from the sketcher canvas",
        ),
    ]

    @field_validator("molfile_v2000")
    @classmethod
    def validate_v2000(cls, v: str) -> str:
        """Validate that molfile_v2000 contains required V2000 and M  END markers."""
        if "V2000" not in v:
            raise ValueError("molfile_v2000 must contain 'V2000' header indicator.")
        if "M  END" not in v and "M END" not in v:
            raise ValueError("molfile_v2000 must contain 'M  END' termination record.")
        return v


class Conformer3DResultSchema(BaseModel):
    """Synthesized 3D conformer result containing 3D coordinates, energy, and validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: Annotated[
        bool,
        Field(description="True if 3D conformer generation and optimization succeeded"),
    ]
    smiles: Annotated[str, Field(description="Canonical SMILES representation")]
    molfile_v3000: Annotated[
        str | None,
        Field(
            default=None,
            description="MDL Molfile V3000 block containing 3D coordinates",
        ),
    ]
    energy_kcal_mol: Annotated[
        float | None,
        Field(default=None, description="Force-field potential energy in kcal/mol"),
    ]
    force_field_used: Annotated[
        str | None,
        Field(
            default=None,
            description="Force field applied for minimization (MMFF94 or UFF)",
        ),
    ]
    coordinates_3d: Annotated[
        list[AtomCoordinate3D],
        Field(
            default_factory=list,
            description="Cartesian 3D coordinates of all atoms including hydrogens",
        ),
    ]
    validation: Annotated[
        ValenceValidationResultSchema,
        Field(description="Chemical valence validation result"),
    ]
