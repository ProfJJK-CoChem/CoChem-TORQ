"""Versioned student contracts with atomic-unit coordinates and derivatives."""

from __future__ import annotations

import json
from hashlib import sha256
from os import PathLike
from typing import Any, Literal, NoReturn
from uuid import UUID, uuid4

import numpy as np
import rfc8785
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)
from typing_extensions import Self

from .scientific_contracts import ScientificGoal

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
        "pes_scan",
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
CANONICALIZATION_PROFILE: Literal["RFC8785"] = "RFC8785"
LEGACY_CANONICALIZATION_PROFILE = "cochem.sorted-json/1"


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
    def molecular_identity(self) -> Self:
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
    serialization_profile: Literal["RFC8785"] = CANONICALIZATION_PROFILE
    request_id: UUID = Field(default_factory=uuid4)
    molecule: Molecule
    recipe: str = Field(min_length=1, max_length=128)
    products: list[str] = Field(min_length=1, max_length=8)
    resources: Resources = Field(default_factory=Resources)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    scientific_goal: ScientificGoal | None = None

    @model_validator(mode="after")
    def known_products(self) -> Self:
        if set(self.products) - PRODUCTS or len(self.products) != len(
            set(self.products)
        ):
            raise ValueError(
                "Products must be distinct, supported scientific product names."
            )
        canonical_json(self.source_provenance)
        return self


class StageResult(Contract):
    status: Literal["available", "blocked", "unavailable", "failed"]
    observable: str
    value: dict[str, Any] | None = None
    reason: str | None = None
    absence_kind: (
        Literal[
            "not_requested", "unsupported", "not_applicable", "failed", "not_computed"
        ]
        | None
    ) = None
    parents: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    uncertainty: dict[str, Any] = Field(
        default_factory=lambda: {"status": "uncalibrated"}
    )

    @model_validator(mode="after")
    def truthful_value(self) -> Self:
        if self.status == "available":
            if (
                self.value is None
                or self.reason is not None
                or self.absence_kind is not None
            ):
                raise ValueError(
                    "An available stage requires a real value and no failure reason."
                )
            canonical_json(self.value)
            from .scientific_values import VALUE_SCHEMAS

            schema_candidate: object = VALUE_SCHEMAS.get(self.observable)
            if schema_candidate is None:
                from .spectroscopy.results import ADVANCED_VALUE_SCHEMAS

                schema_candidate = ADVANCED_VALUE_SCHEMAS.get(self.observable)
            if not isinstance(schema_candidate, type) or not issubclass(
                schema_candidate, BaseModel
            ):
                raise ValueError(
                    "An available scientific observable requires "
                    "a registered typed schema."
                )
            schema_candidate.model_validate(self.value)
        elif self.value is not None or not self.reason:
            raise ValueError(
                "An unavailable stage requires a reason and forbids substitute values."
            )
        return self


def canonical_json(value: Any, *, profile: str = CANONICALIZATION_PROFILE) -> bytes:
    """JCS UTF-8 with IEEE-754, UTF-16 key ordering and numeric-range rejection.

    The legacy profile is only for explicit verification of old version-1
    records. New scientific/cache identities always use RFC 8785. Never cast a
    large integer to float or normalize Unicode to make a hash succeed.
    """
    if profile == CANONICALIZATION_PROFILE:
        return rfc8785.dumps(value)
    if profile == LEGACY_CANONICALIZATION_PROFILE:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    raise ValueError("Unsupported canonical serialization profile.")


def digest(value: Any, *, profile: str = CANONICALIZATION_PROFILE) -> str:
    return sha256(canonical_json(value, profile=profile)).hexdigest()


def _resolved_isotope_evidence(
    request: CalculationRequest,
    retained_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    """Identity-bearing mass observations for requested mass-dependent products.

    Retain authoritative isotope values and source edition/checksum, excluding
    local database paths or process metadata. This key has no accuracy claim.
    """
    if not set(request.products) - {"geometry"}:
        return None
    requested = request.molecule.isotopes or [None] * len(request.molecule.symbols)
    if retained_records is None:
        from Libraries.cochem_isotopes import isotope_record

        retained_records = [
            isotope_record(f"{number}{symbol}" if number is not None else symbol)
            for symbol, number in zip(request.molecule.symbols, requested)
        ]
    if not isinstance(retained_records, list) or len(retained_records) != len(
        requested
    ):
        raise ValueError("Cache identity requires an isotope record for every atom.")
    records = []
    for symbol, number, record in zip(
        request.molecule.symbols, requested, retained_records
    ):
        if not isinstance(record, dict) or not isinstance(record.get("source"), dict):
            raise ValueError("Cache isotope records require an explicit source.")
        source = record["source"]
        mass = record.get("mass_u")
        uncertainty = record.get("mass_uncertainty_u")
        source_digest = source.get("database_sha256")
        if (
            record.get("element") != symbol
            or type(record.get("mass_number")) is not int
            or not 1 <= record["mass_number"] <= 350
            or (number is not None and record["mass_number"] != number)
            or record.get("selection_policy")
            != (
                "explicit_mass_number"
                if number is not None
                else "most_abundant_naturally_occurring_isotope"
            )
            or (type(mass) is not int and type(mass) is not float)
            or not np.isfinite(mass)
            or mass <= 0
            or (
                uncertainty is not None
                and (
                    (type(uncertainty) is not int and type(uncertainty) is not float)
                    or not np.isfinite(uncertainty)
                    or uncertainty < 0
                )
            )
            or source.get("tabulated_mass_is_exact") is not False
            or not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(char not in "0123456789abcdef" for char in source_digest)
            or any(
                not isinstance(source.get(field), str) or not source[field].strip()
                for field in ("database", "distribution_version")
            )
            or (
                "label" in record
                and record["label"] != f"{record['mass_number']}{symbol}"
            )
        ):
            raise ValueError(
                "Cache isotope evidence contradicts atomic identity/source."
            )
        records.append(
            {
                "element": record["element"],
                "mass_number": record["mass_number"],
                "mass_u": record["mass_u"],
                "mass_uncertainty_u": record.get("mass_uncertainty_u"),
                "selection_policy": record["selection_policy"],
                "source": {
                    name: record["source"][name]
                    for name in (
                        "database",
                        "distribution_version",
                        "database_sha256",
                        "tabulated_mass_is_exact",
                    )
                },
            }
        )
    return records


def _constants_evidence(retained: dict[str, Any] | None) -> dict[str, Any]:
    """Validate retained conversion definitions without consulting a local table."""
    if retained is None:
        from .units import constants_provenance

        retained = constants_provenance()
    if not isinstance(retained, dict):
        raise ValueError("Scientific cache requires actual constants provenance.")
    fields = (
        "profile",
        "codata_release",
        "constants",
        "debye_convention",
        "debye_coulomb_metre",
        "debye_exact_reference_coulomb_metre",
    )
    if any(field not in retained for field in fields):
        raise ValueError("Incomplete retained constants definition.")
    definition = {field: retained[field] for field in fields}
    constants = definition["constants"]
    if (
        any(
            not isinstance(definition[field], str) or not definition[field].strip()
            for field in ("profile", "codata_release", "debye_convention")
        )
        or not isinstance(constants, dict)
        or not constants
    ):
        raise ValueError("Invalid retained constants profile/table.")
    for name, record in constants.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(record, dict)
            or set(record) != {"value", "si_unit", "standard_uncertainty", "source"}
            or any(
                not isinstance(record[field], str) or not record[field].strip()
                for field in ("si_unit", "source")
            )
        ):
            raise ValueError("Invalid retained fundamental-constant definition.")
        for field, strictly_positive in (
            ("value", True),
            ("standard_uncertainty", False),
        ):
            value = record[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                or strictly_positive
                and value == 0
            ):
                raise ValueError("Invalid retained constant value/uncertainty.")
    for field in ("debye_coulomb_metre", "debye_exact_reference_coulomb_metre"):
        value = definition[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid retained dipole conversion.")
    encoded = canonical_json(definition)
    definition_sha256 = sha256(encoded).hexdigest()
    if retained.get("definition_sha256") != definition_sha256:
        raise ValueError("Retained constants definition digest mismatch.")
    return {**definition, "definition_sha256": definition_sha256}


def scientific_content(
    request: CalculationRequest,
    recipe: dict[str, Any],
    *,
    resolved_isotope_records: list[dict[str, Any]] | None = None,
    constants_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate reusable scientific identity from record/submission provenance.

    Atom order, exact geometry, state, isotope selection, properties and numerical
    recipe remain identity-bearing. Mass-dependent products additionally bind
    resolved masses, uncertainties and the actual isotope source edition/hash.
    UUIDs, source timestamps, run IDs and resource allocations belong to the
    record digest, not the scientific cache key.
    """
    recipe = dict(recipe)
    recipe.pop("recipe_sha256", None)
    return {
        "schema_version": "cochem.torq.scientific-content/1",
        "serialization_profile": CANONICALIZATION_PROFILE,
        "molecule": request.molecule.model_dump(mode="json"),
        "resolved_isotope_evidence": _resolved_isotope_evidence(
            request, resolved_isotope_records
        ),
        "constants_evidence": _constants_evidence(constants_provenance),
        "geometry_representation": {
            "unit": "bohr",
            "shape": [len(request.molecule.symbols), 3],
            "dtype": "float64",
        },
        "recipe": recipe,
        "products": sorted(request.products),
        "scientific_goal": request.scientific_goal.model_dump(mode="json")
        if request.scientific_goal is not None
        else None,
        "catalog": request.catalog.model_dump(mode="json")
        if "rigid_rotor_catalog" in request.products
        else None,
    }


def scientific_cache_key(
    request: CalculationRequest,
    recipe: dict[str, Any],
    *,
    resolved_isotope_records: list[dict[str, Any]] | None = None,
    constants_provenance: dict[str, Any] | None = None,
) -> str:
    return digest(
        scientific_content(
            request,
            recipe,
            resolved_isotope_records=resolved_isotope_records,
            constants_provenance=constants_provenance,
        )
    )


def read_json(path: str | PathLike[str]) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> NoReturn:
        raise ValueError(f"Nonfinite JSON number: {value}")

    from pathlib import Path

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=nonfinite,
    )
