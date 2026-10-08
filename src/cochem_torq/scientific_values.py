"""Physical payload schemas used by available electronic/geometry/harmonic stages."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)


class PhysicalValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SCFData(PhysicalValue):
    converged: Literal[True]
    energy_hartree: StrictFloat
    cycles: StrictInt = Field(ge=0)
    electron_count: StrictInt = Field(gt=0)


class StabilityData(PhysicalValue):
    status: Literal["stable"]
    internal_stable: Literal[True]
    external_stable: Literal[True]
    scope: str


class ElectronicEnergy(PhysicalValue):
    energy_hartree: StrictFloat
    engine: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    method: dict[str, Any]
    native_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    scf: SCFData
    stability: StabilityData


class OptimizerData(PhysicalValue):
    engine: str
    version: str
    converged: Literal[True]
    final_gradient_verified: Literal[True]
    parameters: dict[str, StrictFloat | StrictInt]
    evaluations: StrictInt = Field(ge=1)
    initial_geometry_bohr: list[list[StrictFloat]]
    final_geometry_source: str
    stationary_point: str


class StationaryGeometry(PhysicalValue):
    symbols: list[str]
    geometry_bohr: list[list[StrictFloat]]
    atom_ids: list[str] | None
    optimization: OptimizerData
    stationary_character: str

    @model_validator(mode="after")
    def dimensions(self):
        if np.asarray(self.geometry_bohr).shape != (len(self.symbols), 3):
            raise ValueError(
                "Optimized geometry dimensions differ from its atomic identity."
            )
        if self.atom_ids is not None and len(self.atom_ids) != len(self.symbols):
            raise ValueError("Optimized geometry lost its atomic mapping.")
        return self


class RotationalConstants(PhysicalValue):
    constants_mhz: tuple[StrictFloat | None, StrictFloat | None, StrictFloat | None]
    principal_moments_u_bohr2: list[StrictFloat] = Field(min_length=3, max_length=3)
    principal_axes_columns: list[list[StrictFloat]]
    rotor_type: Literal["atom", "linear", "nonlinear"]
    isotope_masses_u: list[StrictFloat] = Field(min_length=1)
    geometry_digest: str = Field(pattern="^[0-9a-f]{64}$")
    observable: str

    @model_validator(mode="after")
    def inertia_and_axes(self):
        axes = np.asarray(self.principal_axes_columns)
        if (
            axes.shape != (3, 3)
            or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-10, rtol=0)
            or np.linalg.det(axes) < 0
        ):
            raise ValueError("Principal axes must be a right-handed orthonormal frame.")
        if min(self.isotope_masses_u) <= 0 or min(self.principal_moments_u_bohr2) < 0:
            raise ValueError("Masses and inertia must have physical signs.")
        if any(
            constant is not None and constant <= 0 for constant in self.constants_mhz
        ):
            raise ValueError("Defined rotational constants must be positive.")
        expected_absent = {"atom": 3, "linear": 1, "nonlinear": 0}[self.rotor_type]
        if sum(constant is None for constant in self.constants_mhz) != expected_absent:
            raise ValueError("Undefined rotational axes do not match rotor identity.")
        return self


class HarmonicData(PhysicalValue):
    coordinates_bohr: list[list[StrictFloat]]
    isotope_masses_u: list[StrictFloat]
    frequencies_cm1: list[StrictFloat]
    angular_frequencies_au: list[StrictFloat]
    mass_weighted_modes: list[list[StrictFloat]]
    cartesian_modes: list[list[StrictFloat]]
    dimensionless_to_cartesian: list[list[StrictFloat]] | None
    eigenvalues_au: list[StrictFloat]
    external_rank: StrictInt = Field(ge=3, le=6)
    external_residual_relative: StrictFloat = Field(ge=0)
    symmetry_residual_relative: StrictFloat = Field(ge=0)
    stationary_character: Literal[
        "positive_definite_vibrational_hessian", "nonminimum", "unresolved_zero_modes"
    ]
    harmonic_zpe_hartree: StrictFloat | None
    source_digest: str = Field(pattern="^[0-9a-f]{64}$")
    convention: str

    @model_validator(mode="after")
    def normal_mode_dimensions(self):
        atoms, modes = len(self.coordinates_bohr), len(self.frequencies_cm1)
        if (
            np.asarray(self.coordinates_bohr).shape != (atoms, 3)
            or len(self.isotope_masses_u) != atoms
            or min(self.isotope_masses_u) <= 0
        ):
            raise ValueError(
                "Harmonic coordinates/masses have inconsistent dimensions."
            )
        if modes != 3 * atoms - self.external_rank:
            raise ValueError(
                "The full harmonic mode count must match the projected external rank."
            )
        if (
            len(self.angular_frequencies_au) != modes
            or len(self.eigenvalues_au) != modes
        ):
            raise ValueError(
                "Mode frequencies/eigenvalues have inconsistent dimensions."
            )
        for matrix in (self.mass_weighted_modes, self.cartesian_modes):
            if np.asarray(matrix).shape != (3 * atoms, modes):
                raise ValueError("A normal-mode matrix has inconsistent dimensions.")
        if self.dimensionless_to_cartesian is not None and np.asarray(
            self.dimensionless_to_cartesian
        ).shape != (3 * atoms, modes):
            raise ValueError(
                "The dimensionless coordinate transform has inconsistent dimensions."
            )
        if self.stationary_character == "positive_definite_vibrational_hessian":
            if (
                any(frequency <= 0 for frequency in self.frequencies_cm1)
                or self.harmonic_zpe_hartree is None
            ):
                raise ValueError(
                    "A minimum requires positive signed vibrational modes "
                    "and a defined harmonic ZPE."
                )
        elif self.harmonic_zpe_hartree is not None:
            raise ValueError(
                "A nonminimum/unresolved harmonic result cannot supply a"
                " minimum harmonic ZPE."
            )
        return self


VALUE_SCHEMAS = {
    "electronic_energy": ElectronicEnergy,
    "stationary_geometry": StationaryGeometry,
    "re": StationaryGeometry,
    "Be": RotationalConstants,
    "rotational_constants_at_stationary_geometry": RotationalConstants,
    "harmonic_frequencies": HarmonicData,
}
