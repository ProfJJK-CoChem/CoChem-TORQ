"""Typed spectroscopy stages with explicit availability and scientific provenance.

This module computes isotope-specific rigid-rotor equilibrium constants. Later
stages describe required data contracts; they do not claim a VPT2 or catalog
backend exists. An unavailable correction never erases an earlier valid result.
"""
from __future__ import annotations

from hashlib import sha256
import json
from importlib.metadata import version
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator
from scipy.constants import atomic_mass, h


class ScientificModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Provenance(ScientificModel):
    engine: str
    method: str
    basis_set: str
    parser: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    engine_source_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    digest_scope: str = "source_manifest"
    engine_version: str | None = None
    mass_source: str | None = None

    @field_validator("source_artifacts")
    @classmethod
    def validate_artifact_digests(cls, value: dict[str, str]) -> dict[str, str]:
        import re
        if any(not path or re.fullmatch(r"[a-f0-9]{64}", digest) is None for path, digest in value.items()):
            raise ValueError("Each source artifact requires a path and a SHA-256 digest.")
        return value


class UncertaintyAssessment(ScientificModel):
    status: Literal["uncalibrated", "calibrated"] = "uncalibrated"
    description: str = "No empirical error model or uncertainty calibration is available."
    calibration_reference: str | None = None

    @model_validator(mode="after")
    def require_calibration_evidence(self) -> "UncertaintyAssessment":
        if self.status == "calibrated" and not self.calibration_reference:
            raise ValueError("Calibrated uncertainty requires a calibration reference.")
        return self


T = TypeVar("T", bound=ScientificModel)


class StageResult(ScientificModel, Generic[T]):
    status: Literal["available", "unavailable", "blocked", "failed"]
    value: T | None = None
    reason: str | None = None
    provenance: Provenance
    quality_flags: tuple[str, ...] = ()
    uncertainty: UncertaintyAssessment = Field(default_factory=UncertaintyAssessment)

    @model_validator(mode="after")
    def require_truthful_availability(self) -> "StageResult[T]":
        if self.status == "available":
            if self.value is None or self.reason is not None:
                raise ValueError("Available stages require a value and no missing-data reason.")
        elif self.value is not None or not self.reason:
            raise ValueError("Unavailable stages require a reason and no physical value.")
        return self


class ElectronicStructure(ScientificModel):
    energy_hartree: FiniteFloat
    scf_converged: Literal[True]
    normally_terminated: Literal[True]


class EquilibriumGeometry(ScientificModel):
    symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    optimization_converged: Literal[True]
    stationary_point: Literal["not_characterized", "minimum", "nonminimum"]

    @model_validator(mode="after")
    def validate_atom_mapping(self) -> "EquilibriumGeometry":
        if not self.symbols or len(self.symbols) != len(self.coordinates_angstrom):
            raise ValueError("Geometry symbols and coordinate rows must agree and be nonempty.")
        return self


class EquilibriumConstants(ScientificModel):
    label: Literal["Be"] = "Be"
    isotopologue: tuple[str, ...]
    isotope_masses_u: tuple[FiniteFloat, ...]
    principal_moments_u_angstrom2: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    A_mhz: FiniteFloat | None
    B_mhz: FiniteFloat | None
    C_mhz: FiniteFloat | None
    rotor_type: Literal["atom", "linear", "nonlinear"]
    principal_axes_columns: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    mass_selection: str


class HarmonicAnalysis(ScientificModel):
    frequencies_cm1: tuple[FiniteFloat, ...]
    expected_vibrational_modes: int = Field(ge=0)
    mode_convention: Literal["vibrational_only"] = "vibrational_only"
    stationary_point: Literal["minimum", "nonminimum", "unresolved_zero_modes"]

    @model_validator(mode="after")
    def require_complete_mode_set(self) -> "HarmonicAnalysis":
        if len(self.frequencies_cm1) != self.expected_vibrational_modes:
            raise ValueError("Harmonic characterization requires the complete vibrational mode set.")
        return self


class AnharmonicForceField(ScientificModel):
    cubic_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quartic_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    quartic_scope: Literal["full", "semidiagonal"]
    coordinates_and_units: str
    displacement_convergence_reference: str


class ResonanceAnalysis(ScientificModel):
    treatment: str
    thresholds_and_units: str
    polyads_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    applicability_review: str


class VPT2Result(ScientificModel):
    variant: str
    fundamentals_cm1: tuple[FiniteFloat, ...]
    vibration_rotation_alpha_mhz: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    normal_coordinate_convention: str
    applicability_validation_reference: str


class GroundStateConstants(ScientificModel):
    label: Literal["B0"] = "B0"
    A_mhz: FiniteFloat | None
    B_mhz: FiniteFloat
    C_mhz: FiniteFloat | None
    correction_method: str
    equilibrium_stage_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correction_stage_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class DipoleMoment(ScientificModel):
    components_debye: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    coordinate_frame: Literal["cartesian", "principal_axes"]
    vibrational_state: Literal["equilibrium", "ground_state"]


class CentrifugalDistortion(ScientificModel):
    constants_mhz: dict[str, FiniteFloat]
    hamiltonian: str
    reduction: str
    representation: str
    rotational_state: str


class SpectralCatalog(ScientificModel):
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frequency_units: Literal["MHz"] = "MHz"
    hamiltonian_and_conventions: str
    quantum_number_schema: str
    intensity_convention: str
    partition_function_definition: str
    temperature_kelvin: FiniteFloat = Field(gt=0)
    uncertainty_calibration_reference: str
    applicable_frequency_range_mhz: tuple[FiniteFloat, FiniteFloat]
    independent_validation_reference: str


class SpectroscopyReport(ScientificModel):
    schema_name: Literal["torq.spectroscopy.stages"] = "torq.spectroscopy.stages"
    schema_version: Literal["1.0"] = "1.0"
    electronic_structure: StageResult[ElectronicStructure]
    equilibrium_geometry: StageResult[EquilibriumGeometry]
    equilibrium_constants: StageResult[EquilibriumConstants]
    harmonic_analysis: StageResult[HarmonicAnalysis]
    anharmonic_force_field: StageResult[AnharmonicForceField]
    resonance_analysis: StageResult[ResonanceAnalysis]
    vpt2: StageResult[VPT2Result]
    ground_state_constants: StageResult[GroundStateConstants]
    catalog: StageResult[SpectralCatalog]
    dipole_moment: StageResult[DipoleMoment]
    centrifugal_distortion: StageResult[CentrifugalDistortion]

    @model_validator(mode="after")
    def require_stage_dependencies(self) -> "SpectroscopyReport":
        dependencies = {
            "equilibrium_geometry": ("electronic_structure",),
            "equilibrium_constants": ("equilibrium_geometry",),
            "harmonic_analysis": ("equilibrium_geometry",),
            "anharmonic_force_field": ("harmonic_analysis",),
            "resonance_analysis": ("anharmonic_force_field",),
            "vpt2": ("resonance_analysis",),
            "ground_state_constants": ("equilibrium_constants", "vpt2"),
            "catalog": ("ground_state_constants", "dipole_moment", "centrifugal_distortion"),
        }
        for name, prerequisites in dependencies.items():
            if getattr(self, name).status == "available":
                for prerequisite in prerequisites:
                    if getattr(self, prerequisite).status != "available":
                        raise ValueError(f"Available {name} requires available {prerequisite}.")
        if self.anharmonic_force_field.status == "available" and self.harmonic_analysis.value.stationary_point != "minimum":
            raise ValueError("A semirigid anharmonic force field requires a characterized minimum.")
        return self

    def product_available(self, product: str) -> bool:
        if product not in ("equilibrium_constants", "harmonic_analysis", "ground_state_constants", "catalog"):
            raise ValueError(f"Unsupported requested spectroscopy product: {product}")
        result = getattr(self, product)
        if result.status != "available":
            return False
        if product == "harmonic_analysis":
            return result.value.stationary_point == "minimum"
        return True


def resolve_isotopic_masses(symbols: list[str]) -> tuple[tuple[str, ...], np.ndarray, bool]:
    """Resolve explicit isotopes; plain elements select the most abundant isotope.

    Natural-abundance average atomic weights are inappropriate for a single
    isotopologue. Elements with no tabulated abundance require an explicit mass
    number. The returned flag records whether an implicit isotope was selected.
    """
    import re
    from mendeleev import element, isotope

    labels, masses, selected = [], [], False
    for symbol in symbols:
        symbol = {"D": "2H", "T": "3H"}.get(symbol, symbol)
        match = re.fullmatch(r"(\d+)?([A-Z][a-z]?)", symbol)
        if not match:
            raise ValueError(f"Invalid isotope notation: {symbol}")
        mass_number, base = match.groups()
        if mass_number is not None:
            try:
                record = isotope(base, int(mass_number))
            except Exception as exc:
                raise ValueError(f"No tabulated mass for requested isotope {symbol}.") from exc
        else:
            candidates = [x for x in element(base).isotopes if x.abundance is not None and x.abundance > 0]
            if not candidates:
                raise ValueError(f"An explicit isotope is required for {base}; no natural abundance is tabulated.")
            record = max(candidates, key=lambda x: x.abundance)
            selected = True
        if record is None or record.mass is None or not np.isfinite(record.mass) or record.mass <= 0:
            raise ValueError(f"No finite positive isotope mass is available for {symbol}.")
        labels.append(f"{record.mass_number}{base}")
        masses.append(float(record.mass))
    return tuple(labels), np.asarray(masses), selected


def equilibrium_constants_from_geometry(
    symbols: list[str], coordinates_angstrom: Any
) -> EquilibriumConstants:
    """Compute Be = h/(8 pi^2 I) using resolved isotope masses and SI units.

    An undefined rotation about a linear molecular axis is absent (A=None), not
    an arbitrary large number. These are equilibrium rigid-rotor constants and
    contain no vibrational or centrifugal-distortion correction.
    """
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(symbols), 3) or len(symbols) == 0 or not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates must be a finite nonempty N by 3 array.")
    labels, masses, selected = resolve_isotopic_masses(symbols)
    centered = coords - np.average(coords, axis=0, weights=masses)
    inertia = np.eye(3) * np.sum(masses * np.sum(centered * centered, axis=1))
    inertia -= np.einsum("i,ij,ik->jk", masses, centered, centered)
    moments, axes = np.linalg.eigh(inertia)
    # Eigenvector signs are arbitrary; choose signs deterministically and a
    # right-handed triad. Degenerate subspaces still require explicit care.
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0:
            axes[:, column] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    tolerance = max(float(np.max(np.abs(moments))) * 1e-12, 1e-14)
    if np.any(moments < -tolerance):
        raise ValueError("Inertia tensor has an unphysical negative principal moment.")
    moments[np.abs(moments) <= tolerance] = 0.0  # Explicit floating-point zero criterion.
    if len(symbols) > 1 and moments[-1] == 0:
        raise ValueError("Coincident atoms do not define a molecular rotor.")
    rotor_type = "atom" if len(symbols) == 1 else "linear" if moments[0] == 0 else "nonlinear"
    prefactor = h / (8 * np.pi**2 * atomic_mass * 1e-20 * 1e6)
    constants = [float(prefactor / moment) if moment > 0 else None for moment in moments]
    return EquilibriumConstants(
        isotopologue=labels, isotope_masses_u=tuple(masses),
        principal_moments_u_angstrom2=tuple(moments),
        A_mhz=constants[0], B_mhz=constants[1], C_mhz=constants[2],
        rotor_type=rotor_type, principal_axes_columns=tuple(map(tuple, axes)),
        mass_selection="most_abundant_isotope_for_unspecified_atoms" if selected else "explicit_isotopes",
    )


def build_spectroscopy_report(
    result: Any, symbols: list[str], *, engine: str, method: str, basis_set: str
) -> SpectroscopyReport:
    """Preserve observed engine results and explicitly block unsupported stages."""
    artifact_digests = result.metadata.get("artifact_sha256", {})
    source_manifest = {
        "raw_artifacts": artifact_digests,
        "raw_output_sha256": sha256(result.raw_output.encode("utf-8")).hexdigest(),
        "parsed_energy_hartree": result.energy,
        "parsed_coordinates_angstrom": np.asarray(result.coordinates).tolist(),
        "parsed_gradient_hartree_per_bohr": None if result.gradient is None else np.asarray(result.gradient).tolist(),
        "symbols": symbols, "engine": engine, "method": method, "basis_set": basis_set,
    }
    provenance = Provenance(
        engine=engine, method=method, basis_set=basis_set,
        parser="Libraries.cochem_torq_engine.ORCAStepResult",
        source_sha256=sha256(json.dumps(source_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        source_artifacts=artifact_digests,
        engine_source_manifest_sha256=result.metadata.get("source_manifest_sha256"),
        digest_scope="raw_artifact_manifest_and_consumed_parsed_result",
        engine_version=result.metadata.get("engine_version"),
        mass_source=f"mendeleev {version('mendeleev')}",
    )
    def absent(reason: str, status: str = "blocked") -> dict[str, Any]:
        return dict(status=status, reason=reason, provenance=provenance)

    def available(value: ScientificModel, *flags: str) -> dict[str, Any]:
        return dict(status="available", value=value, provenance=provenance, quality_flags=flags)

    electronic = absent("Electronic convergence and normal termination were not both established.")
    geometry = absent("No converged optimized geometry was established.")
    equilibrium = absent("A converged optimized geometry is required for Be.")
    harmonic = absent("No complete vibrational-mode set with declared convention was parsed.", "unavailable")
    dipole = absent("No dipole with declared units and coordinate frame was parsed.", "unavailable")
    constants = None
    if result.scf_converged is True and result.normally_terminated is True:
        electronic = available(ElectronicStructure(
            energy_hartree=result.energy, scf_converged=True, normally_terminated=True))
        if result.optimization_converged is True and result.metadata.get("geometry_role") == "optimized":
            geometry = available(EquilibriumGeometry(
                symbols=tuple(symbols), coordinates_angstrom=tuple(map(tuple, result.coordinates)),
                optimization_converged=True, stationary_point="not_characterized"),
                "minimum_not_confirmed_without_harmonic_characterization")
            try:
                constants = equilibrium_constants_from_geometry(symbols, result.coordinates)
                flags = ["rigid_rotor_equilibrium_only", "minimum_not_confirmed"]
                if constants.mass_selection != "explicit_isotopes":
                    flags.append("most_abundant_isotopes_selected_explicitly_in_result")
                moments = np.asarray(constants.principal_moments_u_angstrom2)
                if np.any(np.isclose(np.diff(moments), 0, rtol=0, atol=max(float(moments[-1]), 1) * 1e-10)):
                    flags.append("degenerate_principal_axes_require_tensor_frame_review")
                equilibrium = available(constants, *flags)
            except ValueError as exc:
                equilibrium = absent(str(exc), "failed")
    if constants is not None and result.frequencies is not None:
        frequencies = np.asarray(result.frequencies, dtype=float)
        expected = max(0, 3 * len(symbols) - (5 if constants.rotor_type == "linear" else 6))
        convention = result.metadata.get("frequency_mode_convention")
        if convention == "vibrational_only" and frequencies.shape == (expected,) and np.all(np.isfinite(frequencies)):
            stationary = "nonminimum" if np.any(frequencies < 0) else "unresolved_zero_modes" if np.any(frequencies == 0) else "minimum"
            harmonic = available(HarmonicAnalysis(
                frequencies_cm1=tuple(frequencies), expected_vibrational_modes=expected,
                stationary_point=stationary), "harmonic_approximation", "low_frequency_modes_require_applicability_review")
            geometry["value"] = geometry["value"].model_copy(update={"stationary_point": stationary if stationary != "unresolved_zero_modes" else "not_characterized"})
            if stationary == "minimum":
                geometry["quality_flags"] = ()
                equilibrium["quality_flags"] = tuple(flag for flag in equilibrium["quality_flags"] if flag != "minimum_not_confirmed")
            else:
                geometry["quality_flags"] = (stationary,)
                equilibrium["quality_flags"] = tuple(flag for flag in equilibrium["quality_flags"] if flag != "minimum_not_confirmed") + (stationary,)
        else:
            harmonic = absent("Frequency array is incomplete/nonfinite or vibrational-mode convention is undeclared.", "failed")
    if result.dipole_moment is not None and geometry["status"] == "available":
        frame = result.dipole_coordinate_frame
        units = result.dipole_units
        vector = np.asarray(result.dipole_moment, dtype=float)
        if units == "debye" and frame in ("cartesian", "principal_axes") and vector.shape == (3,) and np.all(np.isfinite(vector)):
            dipole = available(DipoleMoment(components_debye=tuple(vector), coordinate_frame=frame,
                vibrational_state="equilibrium"), "equilibrium_dipole_not_vibrationally_averaged")
        else:
            dipole = absent("Dipole dimensions, units or coordinate frame are invalid/undeclared.", "failed")
    return SpectroscopyReport(
        electronic_structure=electronic, equilibrium_geometry=geometry,
        equilibrium_constants=equilibrium, harmonic_analysis=harmonic,
        anharmonic_force_field=absent("No validated cubic/quartic force-field adapter is integrated."),
        resonance_analysis=absent("Requires a validated anharmonic force field and resonance treatment."),
        vpt2=absent("Requires validated force field, resonance treatment and VPT2 applicability."),
        ground_state_constants=absent("No validated vibration-rotation correction is available; Be is not B0."),
        catalog=absent("Requires a validated Hamiltonian, constants, intensities, partition function and uncertainty calibration."),
        dipole_moment=dipole,
        centrifugal_distortion=absent("No convention-qualified centrifugal-distortion result was parsed."),
    )
