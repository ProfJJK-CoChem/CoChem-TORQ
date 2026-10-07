"""Versioned student contracts with atomic-unit coordinates and derivatives."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg "
    "Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(ELEMENTS)}
PRODUCTS = frozenset(
    {
        "geometry",
        "harmonic",
        "equilibrium_constants",
        "rigid_rotor_catalog",
        "anharmonic_force_field",
        "vpt2",
        "ground_state_constants",
        "identification_catalog",
    }
)
PRODUCT_TO_STAGE = {
    product: {"geometry": "equilibrium_geometry", "harmonic": "harmonic_analysis"}.get(
        product, product
    )
    for product in PRODUCTS
}
SPECTROSCOPY_STAGES = (
    "electronic_structure",
    "equilibrium_geometry",
    "equilibrium_constants",
    "harmonic_analysis",
    "anharmonic_force_field",
    "resonance_analysis",
    "vpt2",
    "ground_state_constants",
    "identification_catalog",
)


class PrerequisiteError(RuntimeError):
    """Scientific profile/environment prerequisites block execution (CLI exit 3)."""


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Molecule(Contract):
    symbols: list[str] = Field(min_length=1, max_length=100)
    geometry_bohr: list[list[StrictFloat]]
    charge: StrictInt
    multiplicity: StrictInt = Field(ge=1, le=101)
    atom_ids: list[str] | None = None
    isotopes: list[StrictInt | None] | None = None

    @model_validator(mode="after")
    def molecular_identity(self):
        if any(symbol not in ATOMIC_NUMBERS for symbol in self.symbols):
            raise ValueError(
                "Use canonical element symbols; isotope mass numbers "
                "belong in isotopes."
            )
        coordinates = np.asarray(self.geometry_bohr, dtype=float)
        if (
            coordinates.shape != (len(self.symbols), 3)
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("geometry_bohr must contain finite [N,3] coordinates.")
        if len(coordinates) > 1:
            distances = np.linalg.norm(
                coordinates[:, None] - coordinates[None, :], axis=2
            )
            np.fill_diagonal(distances, np.inf)
            if float(distances.min()) < 1e-6:
                raise ValueError("Coincident nuclei are invalid.")
        electrons = sum(ATOMIC_NUMBERS[s] for s in self.symbols) - self.charge
        spin = self.multiplicity - 1
        if electrons < 1 or spin > electrons or (electrons - spin) % 2:
            raise ValueError(
                "Charge and multiplicity are inconsistent with the electron count."
            )
        if self.atom_ids is not None:
            if len(self.atom_ids) != len(self.symbols) or len(
                set(self.atom_ids)
            ) != len(self.symbols):
                raise ValueError("atom_ids must be unique and match the atom count.")
            if any(not item or len(item) > 128 for item in self.atom_ids):
                raise ValueError("Atom identifiers must be nonempty and bounded.")
        if self.isotopes is not None:
            if len(self.isotopes) != len(self.symbols):
                raise ValueError("isotopes must match the atom count.")
            if any(
                value is not None and (value < 1 or value > 350)
                for value in self.isotopes
            ):
                raise ValueError("Invalid isotope mass number.")
        return self


class Resources(Contract):
    cores: StrictInt = Field(default=1, ge=1, le=4)
    memory_mb: StrictInt = Field(default=2048, ge=256, le=8192)
    wall_seconds: StrictInt = Field(default=600, ge=1, le=1800)


class CatalogSettings(Contract):
    temperature_kelvin: StrictFloat = Field(default=10.0, gt=0, le=1000)
    max_j: StrictInt = Field(default=5, ge=1, le=30)


class CalculationRequest(Contract):
    schema_version: Literal["cochem.torq.request/1"] = "cochem.torq.request/1"
    request_id: UUID = Field(default_factory=uuid4)
    molecule: Molecule
    recipe: str = Field(min_length=1, max_length=128)
    products: list[str] = Field(min_length=1, max_length=8)
    resources: Resources = Field(default_factory=Resources)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    source_provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def known_products(self):
        if set(self.products) - PRODUCTS or len(self.products) != len(
            set(self.products)
        ):
            raise ValueError(
                "Products must be distinct, supported scientific stage names."
            )
        canonical_json(self.source_provenance)
        return self


class StageResult(Contract):
    status: Literal["available", "blocked", "unavailable", "failed"]
    observable: str
    value: dict[str, Any] | None = None
    reason: str | None = None
    parents: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    uncertainty: dict[str, Any] = Field(
        default_factory=lambda: {"status": "uncalibrated"}
    )

    @model_validator(mode="after")
    def truthful_value(self):
        if self.status == "available":
            if self.value is None or self.reason is not None:
                raise ValueError(
                    "An available stage requires a real value and no failure reason."
                )
            canonical_json(self.value)
            from .scientific_values import VALUE_SCHEMAS

            schema = VALUE_SCHEMAS.get(self.observable)
            if schema is not None:
                schema.model_validate(self.value)
        elif self.value is not None or not self.reason:
            raise ValueError(
                "An unavailable stage requires a reason and forbids substitute values."
            )
        return self


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def read_json(path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    from pathlib import Path

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=nonfinite,
    )
