"""Versioned CODATA constants and explicit physical-unit/frame contracts.

Conversions operate on supplied real values only. Spectroscopic and per-molecule
to per-mole equivalences are opt-in; force/gradient sign and normal-coordinate
conventions are never inferred from unit strings. This profile intentionally
preserves TORQ's historical rounded Debye factor until an explicit recalibration.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import scipy
import scipy.constants as scipy_constants
from numpy.typing import NDArray

CONSTANTS_PROFILE = "cochem.constants.codata-2022-torq-compatible/1"
CODATA_RELEASE = "2022"
CONSTANTS_SOURCE = "https://physics.nist.gov/cuu/Constants/"
DEBYE_CONVENTION = "TORQ compatibility: historical rounded 3.33564e-30 C m per Debye"


class UnitError(ValueError):
    """Invalid numerical data, unit dimensions, or physical-coordinate contract."""


@dataclass(frozen=True)
class ConstantDefinition:
    name: str
    value: float
    si_unit: str
    standard_uncertainty: float
    source: str


def _constant(name: str) -> ConstantDefinition:
    value, unit, uncertainty = scipy_constants.physical_constants[name]
    return ConstantDefinition(
        name, float(value), unit, float(uncertainty), CONSTANTS_SOURCE
    )


_release = re.search(
    r"(\d{4}) CODATA recommended values", scipy_constants.__doc__ or ""
)
if _release is None or _release.group(1) != CODATA_RELEASE:
    raise UnitError(
        "This constants profile requires SciPy's documented CODATA 2022 table"
    )

CONSTANT_DEFINITIONS: Mapping[str, ConstantDefinition] = MappingProxyType(
    {
        name: _constant(name)
        for name in (
            "Bohr radius",
            "Hartree energy",
            "atomic mass constant",
            "electron mass",
            "Planck constant",
            "speed of light in vacuum",
            "Boltzmann constant",
            "Avogadro constant",
            "elementary charge",
            "vacuum electric permittivity",
        )
    }
)

BOHR_METRE = CONSTANT_DEFINITIONS["Bohr radius"].value
HARTREE_JOULE = CONSTANT_DEFINITIONS["Hartree energy"].value
ATOMIC_MASS_KG = CONSTANT_DEFINITIONS["atomic mass constant"].value
ELECTRON_MASS_KG = CONSTANT_DEFINITIONS["electron mass"].value
PLANCK_JOULE_SECOND = CONSTANT_DEFINITIONS["Planck constant"].value
SPEED_OF_LIGHT_METRE_SECOND = CONSTANT_DEFINITIONS["speed of light in vacuum"].value
BOLTZMANN_JOULE_KELVIN = CONSTANT_DEFINITIONS["Boltzmann constant"].value
AVOGADRO_PER_MOL = CONSTANT_DEFINITIONS["Avogadro constant"].value
ELEMENTARY_CHARGE_COULOMB = CONSTANT_DEFINITIONS["elementary charge"].value
EPSILON_0_SI = CONSTANT_DEFINITIONS["vacuum electric permittivity"].value

# Preserve the exact floating-point operation order used by existing TORQ code.
BOHR_ANGSTROM = BOHR_METRE * 1e10
ANGSTROM_BOHR = 1.0 / BOHR_ANGSTROM
ATOMIC_MASS_ELECTRON = ATOMIC_MASS_KG / ELECTRON_MASS_KG
HARTREE_CM1 = HARTREE_JOULE / (PLANCK_JOULE_SECOND * SPEED_OF_LIGHT_METRE_SECOND * 100)
DEBYE_COULOMB_METRE = 3.33564e-30
DEBYE_EXACT_COULOMB_METRE = 1e-21 / SPEED_OF_LIGHT_METRE_SECOND
ATOMIC_DIPOLE_COULOMB_METRE = ELEMENTARY_CHARGE_COULOMB * BOHR_METRE
MHZ_HZ = 1e6
PI = math.pi

# Published-formula compatibility aliases, all supplied by the same profile.
atomic_mass = ATOMIC_MASS_KG
h = PLANCK_JOULE_SECOND
c = SPEED_OF_LIGHT_METRE_SECOND
k = BOLTZMANN_JOULE_KELVIN
epsilon_0 = EPSILON_0_SI
pi = PI


def constants_provenance() -> dict[str, Any]:
    """Return a fresh serializable snapshot; mutations cannot alter definitions."""
    definitions = {
        key: {
            "value": record.value,
            "si_unit": record.si_unit,
            "standard_uncertainty": record.standard_uncertainty,
            "source": record.source,
        }
        for key, record in CONSTANT_DEFINITIONS.items()
    }
    definition = {
        "profile": CONSTANTS_PROFILE,
        "codata_release": CODATA_RELEASE,
        "constants": definitions,
        "debye_convention": DEBYE_CONVENTION,
        "debye_coulomb_metre": DEBYE_COULOMB_METRE,
        "debye_exact_reference_coulomb_metre": DEBYE_EXACT_COULOMB_METRE,
    }
    from .domain import canonical_json

    encoded = canonical_json(definition)
    return {
        **definition,
        "definition_sha256": hashlib.sha256(encoded).hexdigest(),
        "provider": "scipy.constants.physical_constants",
        "scipy_version": scipy.__version__,
        "uncertainty_scope": (
            "fundamental constants only; not molecular-model uncertainty"
        ),
    }


class Dimension(str, Enum):
    LENGTH = "length"
    ENERGY = "energy_per_particle"
    MOLAR_ENERGY = "energy_per_mole"
    FREQUENCY = "frequency_cycles_per_second"
    ANGULAR_FREQUENCY = "angular_frequency_radians_per_second"
    WAVENUMBER = "reciprocal_length"
    DIPOLE = "electric_dipole"
    MASS = "mass"
    TEMPERATURE = "temperature"
    ANGLE = "angle"
    DIMENSIONLESS = "dimensionless"
    GRADIENT = "energy_per_length"
    HESSIAN = "energy_per_length_squared"
    MASS_WEIGHTED_COORDINATE = "length_times_sqrt_mass"


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    dimension: Dimension
    scale_si: float
    convention: str | None = None


UNITS: Mapping[str, UnitDefinition] = MappingProxyType(
    {
        item.symbol: item
        for item in (
            UnitDefinition("m", Dimension.LENGTH, 1.0),
            UnitDefinition("angstrom", Dimension.LENGTH, 1e-10),
            UnitDefinition("bohr", Dimension.LENGTH, BOHR_METRE),
            UnitDefinition("joule", Dimension.ENERGY, 1.0),
            UnitDefinition("hartree", Dimension.ENERGY, HARTREE_JOULE),
            UnitDefinition("eV", Dimension.ENERGY, ELEMENTARY_CHARGE_COULOMB),
            UnitDefinition("J/mol", Dimension.MOLAR_ENERGY, 1.0),
            UnitDefinition("kJ/mol", Dimension.MOLAR_ENERGY, 1000.0),
            UnitDefinition(
                "kcal/mol", Dimension.MOLAR_ENERGY, 4184.0, "thermochemical calorie"
            ),
            UnitDefinition("Hz", Dimension.FREQUENCY, 1.0),
            UnitDefinition("MHz", Dimension.FREQUENCY, MHZ_HZ),
            UnitDefinition("GHz", Dimension.FREQUENCY, 1e9),
            UnitDefinition("rad/s", Dimension.ANGULAR_FREQUENCY, 1.0),
            UnitDefinition(
                "angular_frequency_au",
                Dimension.ANGULAR_FREQUENCY,
                HARTREE_JOULE / (PLANCK_JOULE_SECOND / (2 * PI)),
            ),
            UnitDefinition("m^-1", Dimension.WAVENUMBER, 1.0),
            UnitDefinition("cm^-1", Dimension.WAVENUMBER, 100.0),
            UnitDefinition("C*m", Dimension.DIPOLE, 1.0),
            UnitDefinition(
                "debye", Dimension.DIPOLE, DEBYE_COULOMB_METRE, DEBYE_CONVENTION
            ),
            UnitDefinition(
                "debye_exact",
                Dimension.DIPOLE,
                DEBYE_EXACT_COULOMB_METRE,
                "explicit exact electrostatic-cgs Debye convention",
            ),
            UnitDefinition(
                "atomic_dipole", Dimension.DIPOLE, ATOMIC_DIPOLE_COULOMB_METRE
            ),
            UnitDefinition("kg", Dimension.MASS, 1.0),
            UnitDefinition("u", Dimension.MASS, ATOMIC_MASS_KG),
            UnitDefinition("electron_mass", Dimension.MASS, ELECTRON_MASS_KG),
            UnitDefinition("kelvin", Dimension.TEMPERATURE, 1.0),
            UnitDefinition("radian", Dimension.ANGLE, 1.0),
            UnitDefinition("degree", Dimension.ANGLE, PI / 180),
            UnitDefinition("dimensionless", Dimension.DIMENSIONLESS, 1.0),
            UnitDefinition(
                "hartree/bohr", Dimension.GRADIENT, HARTREE_JOULE / BOHR_METRE
            ),
            UnitDefinition("joule/m", Dimension.GRADIENT, 1.0),
            UnitDefinition(
                "hartree/bohr^2", Dimension.HESSIAN, HARTREE_JOULE / BOHR_METRE**2
            ),
            UnitDefinition("joule/m^2", Dimension.HESSIAN, 1.0),
            UnitDefinition(
                "bohr*sqrt(electron_mass)",
                Dimension.MASS_WEIGHTED_COORDINATE,
                BOHR_METRE * math.sqrt(ELECTRON_MASS_KG),
            ),
            UnitDefinition("m*sqrt(kg)", Dimension.MASS_WEIGHTED_COORDINATE, 1.0),
        )
    }
)


def unit_definition(unit: str) -> UnitDefinition:
    if not isinstance(unit, str) or unit not in UNITS:
        raise UnitError(f"Unsupported explicitly named unit {unit!r}")
    return UNITS[unit]


def real_values(value: Any) -> NDArray[np.float64]:
    """Float64 values backed by immutable bytes; reject silent type coercions."""

    def reject_mixed_types(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                reject_mixed_types(child)
        elif isinstance(
            item, (bool, np.bool_, str, bytes, complex, np.complexfloating)
        ):
            raise UnitError(
                "Physical numbers cannot contain boolean, text or complex entries"
            )

    reject_mixed_types(value)
    try:
        initial = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnitError(
            "Physical values require a rectangular real numeric array"
        ) from exc
    if initial.dtype.kind not in "iuf":
        raise UnitError(
            "Physical values must be real numbers; strings, booleans, objects "
            "and complex values are invalid"
        )
    try:
        array = np.asarray(initial, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnitError("Values must be representable as real float64") from exc
    if not np.isfinite(array).all():
        raise UnitError("Physical values must be finite")
    immutable: NDArray[np.float64] = np.frombuffer(
        array.tobytes(), dtype=np.float64
    ).reshape(array.shape)
    return immutable


Equivalence = Literal["molar", "spectroscopic"] | None


def conversion_factor(
    source: str, target: str, *, equivalence: Equivalence = None
) -> float:
    source_unit, target_unit = unit_definition(source), unit_definition(target)
    if (
        equivalence is not None and not isinstance(equivalence, str)
    ) or equivalence not in (None, "molar", "spectroscopic"):
        raise UnitError(
            "Only explicitly named molar or spectroscopic equivalences are supported"
        )
    if source_unit.dimension == target_unit.dimension:
        # These published factors retain existing TORQ arithmetic exactly.
        preserved = {
            ("bohr", "angstrom"): BOHR_ANGSTROM,
            ("angstrom", "bohr"): ANGSTROM_BOHR,
        }
        return preserved.get(
            (source, target), source_unit.scale_si / target_unit.scale_si
        )
    energy_pair = {source_unit.dimension, target_unit.dimension}
    if equivalence == "molar" and energy_pair == {
        Dimension.ENERGY,
        Dimension.MOLAR_ENERGY,
    }:
        factor = source_unit.scale_si / target_unit.scale_si
        return (
            factor * AVOGADRO_PER_MOL
            if source_unit.dimension == Dimension.ENERGY
            else factor / AVOGADRO_PER_MOL
        )
    spectroscopic = {
        Dimension.ENERGY,
        Dimension.FREQUENCY,
        Dimension.ANGULAR_FREQUENCY,
        Dimension.WAVENUMBER,
    }
    if equivalence == "spectroscopic" and energy_pair <= spectroscopic:
        if source in {"hartree", "angular_frequency_au"} and target == "cm^-1":
            return HARTREE_CM1
        if source == "cm^-1" and target in {"hartree", "angular_frequency_au"}:
            return 1.0 / HARTREE_CM1

        def frequency_scale(unit: UnitDefinition) -> float:
            multipliers = {
                Dimension.ENERGY: 1.0 / PLANCK_JOULE_SECOND,
                Dimension.FREQUENCY: 1.0,
                Dimension.ANGULAR_FREQUENCY: 1.0 / (2 * PI),
                Dimension.WAVENUMBER: SPEED_OF_LIGHT_METRE_SECOND,
            }
            return unit.scale_si * multipliers[unit.dimension]

        return frequency_scale(source_unit) / frequency_scale(target_unit)
    raise UnitError(
        "Unit dimensions differ; the required physical equivalence must be explicit"
    )


def convert(
    value: Any, source: str, target: str, *, equivalence: Equivalence = None
) -> float | np.ndarray:
    array = real_values(value)
    factor = conversion_factor(source, target, equivalence=equivalence)
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = array * factor
    except FloatingPointError as exc:
        raise UnitError("Unit conversion exceeds finite float64 range") from exc
    converted = real_values(result)
    return float(converted) if converted.ndim == 0 else converted


@dataclass(frozen=True)
class CartesianFrame:
    frame_id: str
    reference_frame_id: str
    origin: Any
    origin_unit: str
    axes_columns: Any

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 128
            for item in (self.frame_id, self.reference_frame_id)
        ):
            raise UnitError("Frames require explicit bounded identities")
        if unit_definition(self.origin_unit).dimension != Dimension.LENGTH:
            raise UnitError("A Cartesian frame origin requires an explicit length unit")
        origin, axes = real_values(self.origin), real_values(self.axes_columns)
        if origin.shape != (3,) or axes.shape != (3, 3):
            raise UnitError("Cartesian origin and axes require shapes [3] and [3,3]")
        if not np.allclose(
            axes.T @ axes, np.eye(3), rtol=0, atol=1e-10
        ) or not math.isclose(
            float(np.linalg.det(axes)), 1.0, rel_tol=0, abs_tol=1e-10
        ):
            raise UnitError("Cartesian axes must be orthonormal and right-handed")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axes_columns", axes)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": "cochem.cartesian-frame/1",
            "frame_id": self.frame_id,
            "reference_frame_id": self.reference_frame_id,
            "origin": self.origin.tolist(),
            "origin_unit": self.origin_unit,
            "axes_columns": self.axes_columns.tolist(),
            "convention": (
                "columns are target unit axes expressed in the reference frame"
            ),
        }


_KINDS = MappingProxyType(
    {
        "energy": Dimension.ENERGY,
        "molar_energy": Dimension.MOLAR_ENERGY,
        "frequency": Dimension.FREQUENCY,
        "angular_frequency": Dimension.ANGULAR_FREQUENCY,
        "wavenumber": Dimension.WAVENUMBER,
        "mass": Dimension.MASS,
        "temperature": Dimension.TEMPERATURE,
        "angle": Dimension.ANGLE,
        "dimensionless": Dimension.DIMENSIONLESS,
        "cartesian_coordinates": Dimension.LENGTH,
        "gradient": Dimension.GRADIENT,
        "force": Dimension.GRADIENT,
        "hessian": Dimension.HESSIAN,
        "dipole": Dimension.DIPOLE,
        "mass_weighted_normal_coordinate": Dimension.MASS_WEIGHTED_COORDINATE,
        "dimensionless_normal_coordinate": Dimension.DIMENSIONLESS,
    }
)


@dataclass(frozen=True)
class PhysicalQuantity:
    kind: str
    unit: str
    values: Any
    frame: CartesianFrame | None = None
    atom_ids: tuple[str, ...] | None = None
    normal_basis_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or self.kind not in _KINDS
            or unit_definition(self.unit).dimension != _KINDS[self.kind]
        ):
            raise UnitError("The physical quantity kind and unit dimensions disagree")
        values = real_values(self.values)
        cartesian = {"cartesian_coordinates", "gradient", "force", "hessian", "dipole"}
        if self.kind in cartesian:
            if not isinstance(self.frame, CartesianFrame):
                raise UnitError(
                    "Cartesian vectors/tensors require an explicit validated frame"
                )
            if self.kind == "dipole":
                if values.shape != (3,):
                    raise UnitError("A dipole is a Cartesian vector of shape [3]")
            else:
                ids = self.atom_ids
                if (
                    not isinstance(ids, tuple)
                    or not ids
                    or len(set(ids)) != len(ids)
                    or any(
                        not isinstance(item, str) or not item.strip() for item in ids
                    )
                ):
                    raise UnitError(
                        "Atom-indexed Cartesian quantities require unique "
                        "explicit row IDs"
                    )
                expected = (
                    (3 * len(ids), 3 * len(ids))
                    if self.kind == "hessian"
                    else (len(ids), 3)
                )
                if values.shape != expected:
                    raise UnitError(
                        "The Cartesian array does not match its explicit atom mapping"
                    )
        elif self.frame is not None or self.atom_ids is not None:
            raise UnitError(
                "Cartesian frame/row metadata cannot label a non-Cartesian quantity"
            )
        if "normal_coordinate" in self.kind:
            if (
                not isinstance(self.normal_basis_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.normal_basis_sha256) is None
            ):
                raise UnitError(
                    "Normal coordinates require an explicit mode-basis digest"
                )
        elif self.normal_basis_sha256 is not None:
            raise UnitError(
                "A normal-mode basis cannot silently label another "
                "coordinate convention"
            )
        object.__setattr__(self, "values", values)

    def in_unit(self, target: str) -> PhysicalQuantity:
        return PhysicalQuantity(
            self.kind,
            target,
            convert(self.values, self.unit, target),
            self.frame,
            self.atom_ids,
            self.normal_basis_sha256,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": "cochem.physical-quantity/1",
            "kind": self.kind,
            "unit": self.unit,
            "values": self.values.tolist(),
            "frame": self.frame.to_record() if self.frame is not None else None,
            "atom_ids": list(self.atom_ids) if self.atom_ids is not None else None,
            "normal_basis_sha256": self.normal_basis_sha256,
            "constants_profile": CONSTANTS_PROFILE,
            "coordinate_order": "atom-major xyz" if self.atom_ids is not None else None,
        }


def gradient_to_force(gradient: PhysicalQuantity) -> PhysicalQuantity:
    if not isinstance(gradient, PhysicalQuantity) or gradient.kind != "gradient":
        raise UnitError(
            "Force construction requires an explicitly typed energy gradient"
        )
    return PhysicalQuantity(
        "force", gradient.unit, -gradient.values, gradient.frame, gradient.atom_ids
    )


__all__ = [
    "CONSTANTS_PROFILE",
    "CODATA_RELEASE",
    "CONSTANT_DEFINITIONS",
    "UNITS",
    "BOHR_METRE",
    "BOHR_ANGSTROM",
    "ANGSTROM_BOHR",
    "HARTREE_JOULE",
    "HARTREE_CM1",
    "ATOMIC_MASS_KG",
    "ATOMIC_MASS_ELECTRON",
    "ELECTRON_MASS_KG",
    "DEBYE_COULOMB_METRE",
    "DEBYE_EXACT_COULOMB_METRE",
    "ATOMIC_DIPOLE_COULOMB_METRE",
    "MHZ_HZ",
    "PI",
    "PLANCK_JOULE_SECOND",
    "SPEED_OF_LIGHT_METRE_SECOND",
    "BOLTZMANN_JOULE_KELVIN",
    "AVOGADRO_PER_MOL",
    "ELEMENTARY_CHARGE_COULOMB",
    "EPSILON_0_SI",
    "DEBYE_CONVENTION",
    "ConstantDefinition",
    "UnitDefinition",
    "Dimension",
    "UnitError",
    "constants_provenance",
    "unit_definition",
    "conversion_factor",
    "convert",
    "real_values",
    "CartesianFrame",
    "PhysicalQuantity",
    "gradient_to_force",
    "atomic_mass",
    "h",
    "c",
    "k",
    "epsilon_0",
    "pi",
]
