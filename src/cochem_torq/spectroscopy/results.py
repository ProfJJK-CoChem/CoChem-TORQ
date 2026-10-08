"""Strict advanced-stage payloads; shape checks do not qualify a physical model.

Native artifact hashes must separately be checked against retained files. These
schemas preserve explicit experimental status and cannot certify identification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from .harmonic import HARTREE_CM1, artifact_digest, finite_array

# Published angular-momentum symbols match the existing catalog payload.
# ruff: noqa: N815

Sha256 = Annotated[str, Field(pattern="^[a-f0-9]{64}$")]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ScientificContext(_Payload):
    """Atom/isotope/mode/frame identity belonging to an actual stage payload."""

    schema_version: Literal["cochem.torq.scientific-context/1"] = (
        "cochem.torq.scientific-context/1"
    )
    evidence_class: Literal["engine_calculation", "mathematical_model"]
    symbols: list[str] = Field(min_length=1)
    atom_ids: list[str] = Field(min_length=1)
    atom_id_policy: Literal["source_identifiers", "input_ordinal"]
    isotope_numbers: list[StrictInt]
    isotope_masses_u: list[StrictFloat]
    isotope_selection_policies: list[
        Literal["explicit_mass_number", "most_abundant_naturally_occurring_isotope"]
    ]
    isotope_reference_sha256: Sha256
    geometry_bohr: list[list[StrictFloat]]
    geometry_sha256: Sha256
    charge: StrictInt
    multiplicity: StrictInt = Field(ge=1)
    frame_type: Literal["cartesian_input", "principal_inertia"]
    frame_axes_columns: list[list[StrictFloat]]
    frame_sha256: Sha256
    mode_count: StrictInt = Field(ge=0)
    mode_order: list[str]
    mode_basis_sha256: Sha256 | None
    recipe_sha256: Sha256
    protocol_sha256: Sha256
    parent_artifact_sha256: list[Sha256] = Field(min_length=1)

    @model_validator(mode="after")
    def identities(self):
        from ..domain import Molecule, digest

        count = len(self.symbols)
        if (
            len(self.isotope_numbers) != count
            or len(self.isotope_masses_u) != count
            or len(self.isotope_selection_policies) != count
            or min(self.isotope_masses_u, default=0) <= 0
        ):
            raise ValueError(
                "Every atom requires its resolved isotope and positive mass."
            )
        Molecule(
            symbols=self.symbols,
            geometry_bohr=self.geometry_bohr,
            atom_ids=self.atom_ids,
            isotopes=self.isotope_numbers,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        if self.geometry_sha256 != digest(self.geometry_bohr):
            raise ValueError(
                "Geometry digest differs from the actual ordered geometry."
            )
        axes = np.asarray(self.frame_axes_columns)
        if (
            axes.shape != (3, 3)
            or not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(axes), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("Frame must be right-handed and orthonormal.")
        if self.frame_type == "cartesian_input" and not np.array_equal(axes, np.eye(3)):
            raise ValueError(
                "Input Cartesian axes must be the explicit identity frame."
            )
        if self.frame_sha256 != digest(
            {"frame_type": self.frame_type, "axes_columns": self.frame_axes_columns}
        ):
            raise ValueError("Frame digest differs from the declared axis matrix.")
        if (
            len(self.mode_order) != self.mode_count
            or len(set(self.mode_order)) != self.mode_count
            or any(not label.strip() for label in self.mode_order)
            or (self.mode_count > 0) != (self.mode_basis_sha256 is not None)
        ):
            raise ValueError("Mode identities and basis digest must match mode count.")
        if len(set(self.parent_artifact_sha256)) != len(self.parent_artifact_sha256):
            raise ValueError("Parent artifact identities must be distinct.")
        return self


def make_scientific_context(
    *,
    molecule: Mapping[str, Any],
    geometry_bohr: Any,
    isotope_provenance: Sequence[Mapping[str, Any]],
    recipe_sha256: str,
    protocol_sha256: str,
    parent_artifact_sha256: Sequence[str],
    evidence_class: Literal["engine_calculation", "mathematical_model"],
    harmonic: Any | None = None,
    principal_axes_columns: Any | None = None,
) -> dict[str, Any]:
    """Bind actual atom/isotope records and arrays to an advanced stage.

    ``isotope_provenance`` is the retained output of ``isotope_record`` or an
    equivalently authenticated resolved-isotope source. Natural isotope ensembles
    cannot be represented by rounding an average atomic mass into a mass number.
    Hashes supplied here identify retained artifacts; the caller verifies bytes.
    """
    from ..domain import digest

    symbols = list(molecule["symbols"])
    records = list(isotope_provenance)
    if len(records) != len(symbols):
        raise ValueError("Actual resolved isotope records are required for every atom.")
    requested = molecule.get("isotopes") or [None] * len(symbols)
    if len(requested) != len(symbols):
        raise ValueError("Requested isotopes must match atom order.")
    for symbol, number, record in zip(symbols, requested, records):
        if (
            record.get("element") != symbol
            or type(record.get("mass_number")) is not int
            or record.get("label") != f"{record['mass_number']}{symbol}"
            or (number is not None and record["mass_number"] != number)
            or record.get("source", {}).get("tabulated_mass_is_exact") is not False
            or record.get("selection_policy")
            != (
                "explicit_mass_number"
                if number is not None
                else "most_abundant_naturally_occurring_isotope"
            )
        ):
            raise ValueError(
                "Isotope evidence differs from the requested atomic identity."
            )
    sources = {r.get("source", {}).get("database_sha256") for r in records}
    if len(sources) != 1 or None in sources:
        raise ValueError("Resolved isotopes need a common authenticated source digest.")
    coords = finite_array(geometry_bohr, (len(symbols), 3))
    masses = [r["mass_u"] for r in records]
    modes = 0
    mode_basis_digest = None
    if harmonic is not None:
        from .harmonic import HarmonicResult

        if (
            not isinstance(harmonic, HarmonicResult)
            or not np.array_equal(coords, harmonic.coordinates_bohr)
            or not np.array_equal(masses, harmonic.isotope_masses_u)
        ):
            raise ValueError(
                "Harmonic basis must match this geometry and isotopologue."
            )
        modes = len(harmonic.frequencies_cm1)
        if modes:
            mode_basis_digest = artifact_digest(
                {
                    "mass_weighted_modes": harmonic.mass_weighted_modes,
                    "isotope_masses_u": harmonic.isotope_masses_u,
                    "geometry_bohr": harmonic.coordinates_bohr,
                    "convention": harmonic.convention,
                }
            )
    axes = (
        np.eye(3)
        if principal_axes_columns is None
        else finite_array(principal_axes_columns, (3, 3))
    )
    frame_type = (
        "cartesian_input" if principal_axes_columns is None else "principal_inertia"
    )
    atom_ids = molecule.get("atom_ids")
    context = ScientificContext(
        evidence_class=evidence_class,
        symbols=symbols,
        atom_ids=list(atom_ids)
        if atom_ids is not None
        else [f"atom-{i}" for i in range(len(symbols))],
        atom_id_policy="source_identifiers"
        if atom_ids is not None
        else "input_ordinal",
        isotope_numbers=[r["mass_number"] for r in records],
        isotope_masses_u=masses,
        isotope_selection_policies=[r["selection_policy"] for r in records],
        isotope_reference_sha256=next(iter(sources)),
        geometry_bohr=coords.tolist(),
        geometry_sha256=digest(coords.tolist()),
        charge=molecule["charge"],
        multiplicity=molecule["multiplicity"],
        frame_type=frame_type,
        frame_axes_columns=axes.tolist(),
        frame_sha256=digest({"frame_type": frame_type, "axes_columns": axes.tolist()}),
        mode_count=modes,
        mode_order=[f"mode-{i}" for i in range(modes)],
        mode_basis_sha256=mode_basis_digest,
        recipe_sha256=recipe_sha256,
        protocol_sha256=protocol_sha256,
        parent_artifact_sha256=list(parent_artifact_sha256),
    )
    return context.model_dump(mode="json")


class DisplacementData(_Payload):
    q_dimensionless: list[StrictFloat]
    energy_hartree: StrictFloat
    geometry_sha256: Sha256
    source_artifact_sha256: Sha256 | None


class ForceFieldData(_Payload):
    scientific_context: ScientificContext
    frequencies_hartree: list[StrictFloat]
    cubic_hartree: list[list[list[StrictFloat]]]
    quartic_hartree: list[list[list[list[StrictFloat]]]]
    cubic_coarse_hartree: list[list[list[StrictFloat]]]
    quartic_coarse_hartree: list[list[list[list[StrictFloat]]]]
    steps_dimensionless: tuple[StrictFloat, StrictFloat]
    reference_energy_hartree: StrictFloat
    derivative_converged: StrictBool
    absolute_tolerance_hartree: StrictFloat = Field(ge=0)
    relative_tolerance: StrictFloat = Field(ge=0)
    evaluation_count: StrictInt = Field(ge=1)
    evaluator_identity: str = Field(min_length=1)
    harmonic_source_digest: Sha256
    displacement_manifest_digest: Sha256
    source_digest: Sha256
    displacement_records: list[DisplacementData] = Field(min_length=1)
    coordinate_convention: Literal[
        "q=sqrt(omega)*Q; dimensionless; Vn=sum(phi_i... q_i...)/n!"
    ]
    quartic_scope: Literal["full"]

    @model_validator(mode="after")
    def force_constants(self):
        count = self.scientific_context.mode_count
        if (
            count < 1
            or len(self.frequencies_hartree) != count
            or min(self.frequencies_hartree) <= 0
        ):
            raise ValueError("Force field requires all positive harmonic modes.")
        fine_and_coarse = []
        for order, fine, coarse in (
            (3, self.cubic_hartree, self.cubic_coarse_hartree),
            (4, self.quartic_hartree, self.quartic_coarse_hartree),
        ):
            arrays = [np.asarray(fine), np.asarray(coarse)]
            for tensor in arrays:
                if tensor.shape != (count,) * order:
                    raise ValueError(
                        "Every force tensor must cover the full mode basis."
                    )
                for axis in range(1, order):
                    if not np.allclose(
                        tensor, tensor.swapaxes(0, axis), rtol=1e-12, atol=1e-15
                    ):
                        raise ValueError("Force tensor lacks permutation symmetry.")
            fine_and_coarse.append(arrays)
        if not self.steps_dimensionless[0] > self.steps_dimensionless[1] > 0:
            raise ValueError(
                "Two positive decreasing displacement scales are required."
            )
        comparison = all(
            np.all(
                np.abs(fine - coarse)
                <= self.absolute_tolerance_hartree
                + self.relative_tolerance * np.abs(fine)
            )
            for fine, coarse in fine_and_coarse
        )
        if bool(comparison) != self.derivative_converged:
            raise ValueError(
                "Derivative convergence flag disagrees with actual tensors."
            )
        if self.evaluation_count != len(self.displacement_records):
            raise ValueError(
                "Evaluation count must equal retained displacement records."
            )
        displacements = [tuple(r.q_dimensionless) for r in self.displacement_records]
        if any(len(q) != count for q in displacements) or len(
            set(displacements)
        ) != len(displacements):
            raise ValueError("Displacement mode vectors must be complete and distinct.")
        reference = [
            r
            for r in self.displacement_records
            if all(q == 0 for q in r.q_dimensionless)
        ]
        if (
            len(reference) != 1
            or reference[0].energy_hartree != self.reference_energy_hartree
        ):
            raise ValueError(
                "Reference energy must match its actual zero-displacement run."
            )
        if self.scientific_context.evidence_class == "engine_calculation" and any(
            r.source_artifact_sha256 is None for r in self.displacement_records
        ):
            raise ValueError(
                "Every engine displacement needs a native artifact digest."
            )
        manifest = [r.model_dump(mode="json") for r in self.displacement_records]
        if self.displacement_manifest_digest != artifact_digest(
            {
                "recipe": self.evaluator_identity,
                "harmonic": self.harmonic_source_digest,
                "displacements": manifest,
            }
        ):
            raise ValueError(
                "Displacement manifest digest differs from retained records."
            )
        if self.source_digest != artifact_digest(
            {
                "harmonic": self.harmonic_source_digest,
                "cubic": self.cubic_hartree,
                "quartic": self.quartic_hartree,
                "steps": self.steps_dimensionless,
                "manifest": self.displacement_manifest_digest,
            }
        ):
            raise ValueError(
                "Force-field digest differs from its tensors and provenance."
            )
        return self


class ResonanceData(_Payload):
    state_a: list[StrictInt]
    state_b: list[StrictInt]
    operator_order: Literal[3, 4]
    harmonic_detuning_cm1: StrictFloat = Field(ge=0)
    coupling_cm1: StrictFloat = Field(ge=0)
    kind: Literal[
        "vibrational", "Fermi", "Darling-Dennison", "strong_anharmonic_coupling"
    ]


class ResonanceAnalysisData(_Payload):
    scientific_context: ScientificContext
    resonances: list[ResonanceData]
    force_field_sha256: Sha256
    coriolis_resonances: Literal["not_implemented"]
    protocol_sha256: Sha256

    @model_validator(mode="after")
    def state_dimensions(self):
        if self.protocol_sha256 != self.scientific_context.protocol_sha256:
            raise ValueError("Resonance protocol differs from its scientific context.")
        for resonance in self.resonances:
            for state in (resonance.state_a, resonance.state_b):
                if len(state) != self.scientific_context.mode_count or min(state) < 0:
                    raise ValueError(
                        "Resonance states require nonnegative mode occupations."
                    )
            if resonance.state_a == resonance.state_b:
                raise ValueError("A resonance must couple distinct states.")
        return self


class VibrationalVPT2Data(_Payload):
    scientific_context: ScientificContext
    ground_energy_hartree: StrictFloat
    fundamental_frequencies_cm1: list[StrictFloat]
    harmonic_frequencies_cm1: list[StrictFloat]
    state_corrections_hartree: list[StrictFloat]
    force_field_digest: Sha256
    variant: Literal["nonresonant rectilinear vibrational Rayleigh–Schrodinger VPT2"]
    rotation_vibration_available: Literal[False]
    independent_scientific_qualification: Literal[False]

    @model_validator(mode="after")
    def corrected_states(self):
        count = self.scientific_context.mode_count
        if (
            count < 1
            or len(self.harmonic_frequencies_cm1) != count
            or len(self.fundamental_frequencies_cm1) != count
            or len(self.state_corrections_hartree) != count + 1
            or min(self.harmonic_frequencies_cm1) <= 0
            or min(self.fundamental_frequencies_cm1) <= 0
        ):
            raise ValueError(
                "VPT2 requires positive fundamentals and ground+mode corrections."
            )
        corrections = np.asarray(self.state_corrections_hartree)
        ground = sum(self.harmonic_frequencies_cm1) / (2 * HARTREE_CM1) + corrections[0]
        fundamental = (
            np.asarray(self.harmonic_frequencies_cm1)
            + (corrections[1:] - corrections[0]) * HARTREE_CM1
        )
        if not np.isclose(
            self.ground_energy_hartree, ground, rtol=1e-12, atol=1e-14
        ) or not np.allclose(
            self.fundamental_frequencies_cm1, fundamental, rtol=1e-12, atol=1e-8
        ):
            raise ValueError(
                "VPT2 energies/fundamentals disagree with retained corrections."
            )
        return self


class RigidRotorLineData(_Payload):
    frequency_mhz: StrictFloat = Field(gt=0)
    upper_J: StrictInt = Field(ge=1, le=30)
    upper_eigenstate_index: StrictInt = Field(ge=0)
    lower_J: StrictInt = Field(ge=0, le=30)
    lower_eigenstate_index: StrictInt = Field(ge=0)
    lower_energy_mhz: StrictFloat = Field(ge=0)
    summed_dipole_strength_debye2: StrictFloat = Field(gt=0)
    relative_absorption_weight_debye2: StrictFloat = Field(ge=0)
    einstein_A_s1: StrictFloat = Field(gt=0)


class RigidRotorCatalogData(_Payload):
    scientific_context: ScientificContext
    lines: list[RigidRotorLineData]
    temperature_kelvin: StrictFloat = Field(gt=0)
    partition_function: StrictFloat = Field(gt=0)
    J_max: StrictInt = Field(ge=1, le=30)
    partition_relative_tail_indicator: StrictFloat = Field(ge=0)
    partition_converged_at_requested_tolerance: StrictBool
    constant_observable: Literal["Be", "B0"]
    model_identity: Literal[
        "TORQ exact finite-J electric-dipole rigid-rotor screening model v1"
    ]
    quantum_number_convention: Literal[
        "J and sorted Hamiltonian eigenstate index; no inferred Ka/Kc assignment"
    ]
    intensity_convention: Literal[
        "sum over M and three lab polarizations; Boltzmann population times "
        "stimulated-emission correction; no instrument model"
    ]
    nuclear_spin_convention: Literal[
        "nuclear-spin weights excluded; every rotational eigenstate weight one; "
        "no permutation-symmetry restrictions applied"
    ]
    identification_qualified: Literal[False]
    dipole_origin: Literal["center_of_mass"]

    @model_validator(mode="after")
    def quantum_numbers(self):
        if self.scientific_context.frame_type != "principal_inertia":
            raise ValueError(
                "Catalog must identify the dipole/rotational principal frame."
            )
        seen = set()
        for line in self.lines:
            if (
                max(line.upper_J, line.lower_J) > self.J_max
                or abs(line.upper_J - line.lower_J) > 1
                or line.upper_eigenstate_index > 2 * line.upper_J
                or line.lower_eigenstate_index > 2 * line.lower_J
            ):
                raise ValueError(
                    "Transition quantum numbers exceed the finite-J basis."
                )
            identity = (
                line.upper_J,
                line.upper_eigenstate_index,
                line.lower_J,
                line.lower_eigenstate_index,
            )
            if identity in seen:
                raise ValueError("Duplicate transition identity.")
            seen.add(identity)
        return self


ADVANCED_VALUE_SCHEMAS = {
    "anharmonic_force_field": ForceFieldData,
    "resonance_analysis": ResonanceAnalysisData,
    "vibrational_only_vpt2": VibrationalVPT2Data,
    "rigid_rotor_transitions": RigidRotorCatalogData,
}
