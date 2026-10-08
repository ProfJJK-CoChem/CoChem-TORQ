"""Canonical, nonresonant leading Watson vibration--rotation corrections.

The nuclear-mass expansion m(epsilon)=m/epsilon**4 keeps dimensionless q fixed.
Rotation is order epsilon**4 and its leading vibrational correction is order
epsilon**6. Exact oscillator intermediate-state sums implement that correction.
This does not implement resonant GVPT2, reduced distortion or identification
accuracy. See docs/development/rovibrational_perturbation.md for the derivation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from ..units import HARTREE_CM1, HARTREE_JOULE, h
from .forcefield import DisplacementEnergy, ForceField
from .harmonic import artifact_digest, finite_array
from .perturbation import _operator_action, analyze_resonances
from .results import ForceFieldData, ScientificContext
from .rovibrational import RovibrationalPrecursors
from .rovibrational_solver import (
    StationaryReference,
    _validated_inputs,
    body_fixed_angular_momentum,
)

Matrix = tuple[tuple[float, ...], ...]
Tensor3 = tuple[Matrix, ...]
Tensor4 = tuple[Tensor3, ...]
State = tuple[int, ...]
RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
_HARTREE_MHZ = HARTREE_JOULE / h / 1e6
_REFERENCE = {
    "repository": "bchangala/nitrogen",
    "commit": "85ab1d0c965b2f613fc810a394de0c2006bd6f6e",
    "path": "nitrogen/vpt/__init__.py",
    "source_sha256": "0a29234b59bc92dcd60c1c8bd16ee4ecff44ad4d9e1e8522bac8f198e7208d30",
    "license": "MIT; Copyright (c) 2020 bchangala",
    "verification": "retrieved independent code; primary fulltext unverified",
}


class _Immutable(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


class WatsonCorrectionProtocol(_Immutable):
    maximum_modes: int = Field(default=12, ge=1, le=12)
    maximum_intermediate_states: int = Field(default=5000, ge=1, le=50000)
    minimum_denominator_cm1: float = Field(default=10.0, gt=0)
    maximum_anharmonic_coupling_ratio: float = Field(default=0.1, gt=0)
    maximum_coriolis_coupling_ratio: float = Field(default=0.1, gt=0)
    maximum_relative_rotational_correction: float = Field(default=0.1, gt=0)
    maximum_relative_vibrational_correction: float = Field(default=0.1, gt=0)
    maximum_J: int = Field(default=1, ge=1, le=5)  # noqa: N815
    coupling_floor_cm1: float = Field(default=1e-6, gt=0)
    independent_scientific_qualification: Literal[False] = False


class VirtualStateContribution(_Immutable):
    operator: Literal["cubic_linear_inertia_response", "coriolis_second_order"]
    virtual_state: tuple[int, ...]
    denominator_hartree: float
    numerator_tensor_hartree2: Matrix
    contribution_tensor_hartree: Matrix | None
    blocking_reason: str | None

    @model_validator(mode="after")
    def denominator_identity(self) -> Self:
        numerator = finite_array(self.numerator_tensor_hartree2, (3, 3))
        if self.contribution_tensor_hartree is None:
            if not self.blocking_reason:
                raise ValueError("Unavailable denominator terms require their reason.")
        else:
            values = finite_array(self.contribution_tensor_hartree, (3, 3))
            if (
                self.denominator_hartree == 0
                or self.blocking_reason is not None
                or not np.allclose(
                    values * self.denominator_hartree,
                    numerator,
                    rtol=1e-12,
                    atol=1e-25,
                )
            ):
                raise ValueError("Virtual-state numerator/denominator binding fails.")
        return self


class WatsonStateCorrection(_Immutable):
    vibrational_state: tuple[int, ...]
    state_identity: Literal[
        "harmonic occupation label; isolated-state applicability gated"
    ] = "harmonic occupation label; isolated-state applicability gated"
    direct_inverse_inertia_tensor_hartree: Matrix
    cubic_response_tensor_hartree: Matrix | None
    coriolis_response_tensor_hartree: Matrix | None
    algebraic_coefficient_tensor_hartree: Matrix | None
    model_constants_status: Literal["available_model_prediction", "blocked"]
    equilibrium_axis_model_constants_mhz: tuple[float, float, float] | None
    blocking_reasons: tuple[str, ...]
    maximum_observed_anharmonic_coupling_ratio: float = Field(ge=0)
    maximum_observed_coriolis_coupling_ratio: float = Field(ge=0)
    relative_rotational_correction_norm: float | None = Field(default=None, ge=0)
    virtual_state_contributions: tuple[VirtualStateContribution, ...]
    harmonic_energy_hartree: float = Field(gt=0)
    quartic_first_order_hartree: float
    cubic_second_order_hartree: float | None
    coriolis_kinetic_first_order_hartree: float = Field(ge=0)
    watson_volume_scalar_hartree: float = Field(lt=0)
    algebraic_semirigid_energy_hartree: float | None
    semirigid_energy_status: Literal["available_model_prediction", "blocked"]
    semirigid_energy_blocking_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def gate_binding(self) -> Self:
        if self.model_constants_status == "available_model_prediction":
            if (
                self.blocking_reasons
                or self.equilibrium_axis_model_constants_mhz is None
                or self.algebraic_coefficient_tensor_hartree is None
            ):
                raise ValueError("An available state must pass all recorded gates.")
        elif (
            not self.blocking_reasons
            or self.equilibrium_axis_model_constants_mhz is not None
        ):
            raise ValueError("Blocked constants must remain unavailable with reasons.")
        for values in (
            self.direct_inverse_inertia_tensor_hartree,
            self.cubic_response_tensor_hartree,
            self.coriolis_response_tensor_hartree,
            self.algebraic_coefficient_tensor_hartree,
        ):
            if values is not None:
                array = finite_array(values, (3, 3))
                if not np.allclose(array, array.T, rtol=1e-11, atol=1e-20):
                    raise ValueError(
                        "Effective rotational tensors must be real symmetric."
                    )
        if self.semirigid_energy_status == "available_model_prediction":
            if (
                self.semirigid_energy_blocking_reasons
                or self.algebraic_semirigid_energy_hartree is None
            ):
                raise ValueError(
                    "Available semirigid energies must pass recorded gates."
                )
        elif not self.semirigid_energy_blocking_reasons:
            raise ValueError("Unavailable semirigid energies require their reasons.")
        if (self.cubic_second_order_hartree is None) != (
            self.algebraic_semirigid_energy_hartree is None
        ):
            raise ValueError("An incomplete cubic sum cannot produce a total energy.")
        if self.algebraic_semirigid_energy_hartree is not None:
            assert self.cubic_second_order_hartree is not None
            expected_energy = (
                self.harmonic_energy_hartree
                + self.quartic_first_order_hartree
                + self.cubic_second_order_hartree
                + self.coriolis_kinetic_first_order_hartree
                + self.watson_volume_scalar_hartree
            )
            if not np.isclose(
                expected_energy,
                self.algebraic_semirigid_energy_hartree,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise ValueError("Semirigid energy disagrees with its separate terms.")
        for operator, expected_response in (
            ("cubic_linear_inertia_response", self.cubic_response_tensor_hartree),
            ("coriolis_second_order", self.coriolis_response_tensor_hartree),
        ):
            records = tuple(
                record
                for record in self.virtual_state_contributions
                if record.operator == operator
            )
            if any(record.contribution_tensor_hartree is None for record in records):
                if expected_response is not None:
                    raise ValueError("Unavailable virtual terms cannot produce a sum.")
            else:
                response = np.zeros((3, 3))
                for record in records:
                    response += np.asarray(record.contribution_tensor_hartree)
                if expected_response is None or not np.allclose(
                    response, expected_response, rtol=1e-12, atol=1e-20
                ):
                    raise ValueError("Virtual-state records disagree with their sum.")
        return self


class SemirigidVPT2Data(_Immutable):
    scientific_context: ScientificContext
    force_field_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    precursor_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    state_energies_hartree: tuple[float, ...]
    ground_energy_hartree: float
    fundamental_frequencies_cm1: tuple[float, ...]
    harmonic_frequencies_cm1: tuple[float, ...]
    quartic_first_order_hartree: tuple[float, ...]
    cubic_second_order_hartree: tuple[float, ...]
    coriolis_kinetic_first_order_hartree: tuple[float, ...]
    watson_volume_scalar_hartree: float
    variant: Literal[
        "nonresonant Watson semirigid VPT2 through physical mass order epsilon^4"
    ] = "nonresonant Watson semirigid VPT2 through physical mass order epsilon^4"
    rotation_vibration_model_available: bool
    full_resonant_gvpt2: Literal[False] = False
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False

    @model_validator(mode="after")
    def energy_binding(self) -> Self:
        count = self.scientific_context.mode_count
        if (
            len(self.state_energies_hartree) != count + 1
            or len(self.fundamental_frequencies_cm1) != count
            or len(self.harmonic_frequencies_cm1) != count
            or self.ground_energy_hartree != self.state_energies_hartree[0]
            or min(self.fundamental_frequencies_cm1) <= 0
            or any(
                len(values) != count + 1
                for values in (
                    self.quartic_first_order_hartree,
                    self.cubic_second_order_hartree,
                    self.coriolis_kinetic_first_order_hartree,
                )
            )
        ):
            raise ValueError(
                "Semirigid energies require complete positive fundamentals."
            )
        harmonic = np.full(
            count + 1, sum(self.harmonic_frequencies_cm1) / (2 * HARTREE_CM1)
        )
        harmonic[1:] += np.asarray(self.harmonic_frequencies_cm1) / HARTREE_CM1
        expected = harmonic + np.asarray(self.quartic_first_order_hartree)
        expected += np.asarray(self.cubic_second_order_hartree)
        expected += np.asarray(self.coriolis_kinetic_first_order_hartree)
        expected += self.watson_volume_scalar_hartree
        if not np.allclose(
            expected, self.state_energies_hartree, rtol=1e-12, atol=1e-14
        ) or not np.allclose(
            (expected[1:] - expected[0]) * HARTREE_CM1,
            self.fundamental_frequencies_cm1,
            rtol=1e-12,
            atol=1e-8,
        ):
            raise ValueError(
                "Semirigid state energies disagree with their contributions."
            )
        return self


class WatsonHarmonicDistortion(_Immutable):
    status: Literal["available_unreduced_harmonic_model"] = (
        "available_unreduced_harmonic_model"
    )
    precursor_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    unreduced_tau_hartree: Tensor4
    linear_inverse_inertia_q_coefficients: Tensor3
    frequencies_hartree: tuple[float, ...]
    hamiltonian_convention: Literal[
        "Hdist=1/4 sum(tau_abcd Ja Jb Jc Jd)=-sum(Tk^2/(2 omega_k))"
    ] = "Hdist=1/4 sum(tau_abcd Ja Jb Jc Jd)=-sum(Tk^2/(2 omega_k))"
    A_reduced_constants: None = None  # noqa: N815
    S_reduced_constants: None = None  # noqa: N815
    effective_constant_reduction_shifts: None = None
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False

    @model_validator(mode="after")
    def response_identity(self) -> Self:
        frequency = finite_array(self.frequencies_hartree)
        first = finite_array(
            self.linear_inverse_inertia_q_coefficients, (3, 3, len(frequency))
        )
        tau = finite_array(self.unreduced_tau_hartree, (3, 3, 3, 3))
        if not len(frequency) or np.any(frequency <= 0):
            raise ValueError("Harmonic response requires all positive frequencies.")
        expected = -0.5 * np.einsum("abk,cdk,k->abcd", first, first, 1 / frequency)
        if not np.allclose(tau, expected, rtol=1e-12, atol=1e-25):
            raise ValueError(
                "Unreduced distortion differs from its actual mode response."
            )
        return self


class WatsonVibrationRotationResult(_Immutable):
    schema_version: Literal["cochem.torq.nonresonant-watson-leading/1"] = (
        "cochem.torq.nonresonant-watson-leading/1"
    )
    scientific_context: ScientificContext
    precursor_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    force_field_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    algorithm_source_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    stationary_reference: StationaryReference
    protocol: WatsonCorrectionProtocol
    equilibrium_coefficient_tensor_hartree: Matrix
    states: tuple[WatsonStateCorrection, ...]
    algebraic_alpha_tensors_hartree: Tensor3 | None
    model_alpha_tensors_mhz: Tensor3 | None
    ground_state_constants_mhz: tuple[float, float, float] | None
    harmonic_distortion: WatsonHarmonicDistortion
    semirigid_vpt2: SemirigidVPT2Data | None
    semirigid_vpt2_blocking_reasons: tuple[str, ...]
    watson_volume_scalar_reference_hartree: float
    independently_derived_volume_scalar_reference_hartree: float
    watson_volume_identity_absolute_residual_hartree: float = Field(ge=0)
    expansion_convention: Literal[
        "m(epsilon)=m/epsilon^4; leading J^2 coefficients through epsilon^6"
    ] = "m(epsilon)=m/epsilon^4; leading J^2 coefficients through epsilon^6"
    alpha_convention: Literal[
        "alpha_i=beta_ground-beta_fundamental_i; beta is coefficient of Ja Jb"
    ] = "alpha_i=beta_ground-beta_fundamental_i; beta is coefficient of Ja Jb"
    axes_convention: Literal[
        "retained equilibrium principal axes; diagonal constants only at leading order"
    ] = "retained equilibrium principal axes; diagonal constants only at leading order"
    reference_implementation: dict[str, str]
    full_resonant_GVPT2: Literal[False] = False  # noqa: N815
    independent_scientific_qualification: Literal[False] = False
    identification_ready: Literal[False] = False
    limitations: tuple[str, ...]
    content_sha256: str = Field(pattern="^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def scientific_readback(self) -> Self:
        count = self.scientific_context.mode_count
        expected_states = (tuple([0] * count),) + tuple(
            tuple(int(index == mode) for index in range(count)) for mode in range(count)
        )
        if tuple(state.vibrational_state for state in self.states) != expected_states:
            raise ValueError("Ground and ordered single-mode states must be retained.")
        equilibrium = finite_array(self.equilibrium_coefficient_tensor_hartree, (3, 3))
        frequency = np.asarray(self.harmonic_distortion.frequencies_hartree)
        if len(frequency) != count:
            raise ValueError("Harmonic response and vibrational mode counts differ.")
        for state in self.states:
            if state.model_constants_status == "available_model_prediction" and (
                state.cubic_second_order_hartree is None
                or state.maximum_observed_anharmonic_coupling_ratio
                >= self.protocol.maximum_anharmonic_coupling_ratio
                or state.maximum_observed_coriolis_coupling_ratio
                >= self.protocol.maximum_coriolis_coupling_ratio
                or state.relative_rotational_correction_norm is None
                or state.relative_rotational_correction_norm
                >= self.protocol.maximum_relative_rotational_correction
            ):
                raise ValueError(
                    "Available rotational states cannot bypass protocol bounds."
                )
            if state.semirigid_energy_status == "available_model_prediction" and (
                state.maximum_observed_anharmonic_coupling_ratio
                >= self.protocol.maximum_anharmonic_coupling_ratio
                or state.algebraic_semirigid_energy_hartree is None
                or abs(
                    state.algebraic_semirigid_energy_hartree
                    - state.harmonic_energy_hartree
                )
                / state.harmonic_energy_hartree
                >= self.protocol.maximum_relative_vibrational_correction
            ):
                raise ValueError(
                    "Available semirigid states cannot bypass protocol bounds."
                )
            for record in state.virtual_state_contributions:
                if len(record.virtual_state) != count or any(
                    occupation < 0 for occupation in record.virtual_state
                ):
                    raise ValueError(
                        "Virtual states require complete valid occupations."
                    )
                denominator = float(
                    np.dot(
                        np.asarray(state.vibrational_state)
                        - np.asarray(record.virtual_state),
                        frequency,
                    )
                )
                if not np.isclose(
                    denominator, record.denominator_hartree, rtol=1e-12, atol=1e-15
                ):
                    raise ValueError(
                        "Virtual denominators differ from their state energies."
                    )
            if state.algebraic_coefficient_tensor_hartree is not None:
                if (
                    state.cubic_response_tensor_hartree is None
                    or state.coriolis_response_tensor_hartree is None
                ):
                    raise ValueError(
                        "Incomplete virtual sums cannot invent total coefficients."
                    )
                expected = equilibrium + np.asarray(
                    state.direct_inverse_inertia_tensor_hartree
                )
                expected += np.asarray(state.cubic_response_tensor_hartree)
                expected += np.asarray(state.coriolis_response_tensor_hartree)
                if not np.allclose(
                    expected,
                    state.algebraic_coefficient_tensor_hartree,
                    rtol=1e-12,
                    atol=1e-20,
                ):
                    raise ValueError(
                        "State tensor disagrees with its separate contributions."
                    )
                if (
                    state.equilibrium_axis_model_constants_mhz is not None
                    and not np.allclose(
                        np.diag(expected) * _HARTREE_MHZ,
                        state.equilibrium_axis_model_constants_mhz,
                        rtol=1e-12,
                        atol=1e-8,
                    )
                ):
                    raise ValueError(
                        "Axis constants disagree with their operator units."
                    )
        available = all(
            state.model_constants_status == "available_model_prediction"
            for state in self.states
        )
        if (
            self.ground_state_constants_mhz
            != self.states[0].equilibrium_axis_model_constants_mhz
            or (self.model_alpha_tensors_mhz is not None) != available
            or self.harmonic_distortion.precursor_sha256 != self.precursor_sha256
            or self.content_sha256
            != artifact_digest(self.model_dump(mode="json", exclude={"content_sha256"}))
        ):
            raise ValueError("Result products, gates or content binding disagree.")
        if self.algebraic_alpha_tensors_hartree is not None:
            alpha = finite_array(self.algebraic_alpha_tensors_hartree, (count, 3, 3))
            if any(
                state.algebraic_coefficient_tensor_hartree is None
                for state in self.states
            ):
                raise ValueError("Missing state tensors cannot produce mode alpha.")
            expected_alpha = np.asarray(
                self.states[0].algebraic_coefficient_tensor_hartree
            ) - np.asarray(
                [
                    state.algebraic_coefficient_tensor_hartree
                    for state in self.states[1:]
                ]
            )
            if not np.allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-20):
                raise ValueError(
                    "Alpha differs from its ground/fundamental definition."
                )
            if self.model_alpha_tensors_mhz is not None and not np.allclose(
                self.model_alpha_tensors_mhz,
                alpha * _HARTREE_MHZ,
                rtol=1e-12,
                atol=1e-8,
            ):
                raise ValueError("Available alpha units disagree with their tensors.")
        if self.semirigid_vpt2 is not None:
            if self.semirigid_vpt2_blocking_reasons or any(
                state.semirigid_energy_status != "available_model_prediction"
                for state in self.states
            ):
                raise ValueError(
                    "Semirigid VPT2 cannot bypass failed state applicability."
                )
            semirigid = self.semirigid_vpt2
            if (
                semirigid.scientific_context != self.scientific_context
                or semirigid.force_field_sha256 != self.force_field_sha256
                or semirigid.precursor_sha256 != self.precursor_sha256
                or semirigid.rotation_vibration_model_available != available
                or semirigid.state_energies_hartree
                != tuple(
                    state.algebraic_semirigid_energy_hartree for state in self.states
                )
                or semirigid.quartic_first_order_hartree
                != tuple(state.quartic_first_order_hartree for state in self.states)
                or semirigid.cubic_second_order_hartree
                != tuple(state.cubic_second_order_hartree for state in self.states)
                or semirigid.coriolis_kinetic_first_order_hartree
                != tuple(
                    state.coriolis_kinetic_first_order_hartree for state in self.states
                )
                or any(
                    state.watson_volume_scalar_hartree
                    != semirigid.watson_volume_scalar_hartree
                    for state in self.states
                )
                or not np.allclose(
                    semirigid.harmonic_frequencies_cm1,
                    frequency * HARTREE_CM1,
                    rtol=1e-12,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    "Semirigid child source, states or terms differ from parent."
                )
        elif not self.semirigid_vpt2_blocking_reasons:
            raise ValueError("Unavailable semirigid VPT2 requires explicit reasons.")
        return self


def _rows(values: RealArray) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in values)


def _tensor3(values: RealArray) -> Tensor3:
    return tuple(_rows(plane) for plane in values)


def _tensor4(values: RealArray) -> Tensor4:
    return tuple(_tensor3(block) for block in values)


def _force_field(data: ForceFieldData) -> ForceField:
    return ForceField(
        finite_array(data.frequencies_hartree),
        finite_array(data.cubic_hartree),
        finite_array(data.quartic_hartree),
        finite_array(data.cubic_coarse_hartree),
        finite_array(data.quartic_coarse_hartree),
        data.steps_dimensionless,
        data.reference_energy_hartree,
        data.derivative_converged,
        data.absolute_tolerance_hartree,
        data.relative_tolerance,
        data.evaluation_count,
        data.evaluator_identity,
        data.harmonic_source_digest,
        data.displacement_manifest_digest,
        data.source_digest,
        tuple(
            DisplacementEnergy(
                tuple(record.q_dimensionless),
                record.energy_hartree,
                record.geometry_sha256,
                record.source_artifact_sha256,
            )
            for record in data.displacement_records
        ),
    )


def _inertia_expansion(
    precursors: RovibrationalPrecursors,
) -> tuple[RealArray, RealArray, RealArray]:
    frequency = np.asarray(precursors.harmonic_frequencies_cm1) / HARTREE_CM1
    inverse = np.linalg.inv(np.asarray(precursors.inertia_electron_mass_bohr2))
    first = np.asarray(precursors.inertia_first_derivative)
    second = np.asarray(precursors.inertia_second_derivative)
    zeta = np.asarray(precursors.coriolis_zeta)
    first_q = first / np.sqrt(frequency)[None, None, :]
    inverse_first = -np.einsum("ab,bck,cd->adk", inverse, first_q, inverse)
    count = len(frequency)
    inverse_second = np.zeros((3, 3, count, count))
    for first_mode in range(count):
        for second_mode in range(count):
            a, b = first_q[:, :, first_mode], first_q[:, :, second_mode]
            ca, cb = zeta[:, first_mode, :], zeta[:, second_mode, :]
            quadratic_metric = (
                0.5 * second[:, :, first_mode, second_mode]
                - 0.5 * (ca @ cb.T + cb @ ca.T)
            ) / sqrt(frequency[first_mode] * frequency[second_mode])
            inverse_second[:, :, first_mode, second_mode] = (
                0.5
                * (
                    inverse @ a @ inverse @ b @ inverse
                    + inverse @ b @ inverse @ a @ inverse
                )
                - inverse @ quadratic_metric @ inverse
            )
    return inverse, inverse_first, inverse_second


def watson_volume_identity(
    precursors: RovibrationalPrecursors,
    Q: tuple[float, ...],  # noqa: N803
) -> tuple[float, float]:
    """Independent analytic fixed-measure scalar and Watson -Tr(mu)/8.

    Q is mass-weighted bohr*sqrt(electron_mass), NOT dimensionless q. Both
    expressions are derived/evaluated; their equality is not assumed here.
    """
    if not isinstance(precursors, RovibrationalPrecursors):
        raise ValueError("Actual validated complete geometric precursors required.")
    precursors = RovibrationalPrecursors.model_validate_json(
        precursors.model_dump_json()
    )
    count = len(precursors.harmonic_frequencies_cm1)
    coordinate = finite_array(Q, (count,))
    first = np.asarray(precursors.inertia_first_derivative)
    second = np.asarray(precursors.inertia_second_derivative)
    zeta = np.asarray(precursors.coriolis_zeta)
    cross = np.einsum("aji,j->ai", zeta, coordinate)
    dcross = np.moveaxis(zeta, 1, 0)
    metric = np.asarray(precursors.inertia_electron_mass_bohr2).copy()
    metric += np.einsum("abk,k->ab", first, coordinate)
    metric += 0.5 * np.einsum("abkl,k,l->ab", second, coordinate, coordinate)
    metric -= cross @ cross.T
    if precursors.zero_inertia_axes or np.min(np.linalg.eigvalsh(metric)) <= 0:
        raise ValueError(
            "Watson volume identity requires a nonsingular nonlinear chart."
        )
    inverse = np.linalg.inv(metric)
    derivative, second_derivative = [], []
    for mode in range(count):
        derivative.append(
            first[:, :, mode]
            + np.einsum("abl,l->ab", second[:, :, mode, :], coordinate)
            - dcross[mode] @ cross.T
            - cross @ dcross[mode].T
        )
        second_derivative.append(
            [
                second[:, :, mode, other]
                - dcross[mode] @ dcross[other].T
                - dcross[other] @ dcross[mode].T
                for other in range(count)
            ]
        )
    volume_gradient = np.asarray(
        [0.25 * np.trace(inverse @ value) for value in derivative]
    )
    volume_hessian = np.asarray(
        [
            [
                0.25
                * np.trace(
                    inverse @ second_derivative[mode][other]
                    - inverse @ derivative[mode] @ inverse @ derivative[other]
                )
                for other in range(count)
            ]
            for mode in range(count)
        ]
    )
    vibration_inverse = np.eye(count) + cross.T @ inverse @ cross
    vibration_inverse_derivative = np.asarray(
        [
            value.T @ inverse @ cross
            + cross.T @ inverse @ value
            - cross.T @ inverse @ derivative[mode] @ inverse @ cross
            for mode, value in enumerate(dcross)
        ]
    )
    derived = 0.5 * (
        np.einsum("iij,j->", vibration_inverse_derivative, volume_gradient)
        + np.einsum("ij,ij->", vibration_inverse, volume_hessian)
        + volume_gradient @ vibration_inverse @ volume_gradient
    )
    return float(derived), float(-np.trace(inverse) / 8)


def _coordinate_action(state: State, mode: int) -> dict[State, float]:
    result = {}
    if state[mode]:
        lowered = list(state)
        lowered[mode] -= 1
        result[tuple(lowered)] = sqrt(state[mode] / 2)
    raised = list(state)
    raised[mode] += 1
    result[tuple(raised)] = sqrt((state[mode] + 1) / 2)
    return result


def _coriolis_action(
    state: State, zeta: RealArray, frequency: RealArray
) -> tuple[dict[State, complex], ...]:
    """Exact pi_a=sum_ji zeta_a,ji sqrt(w_i/w_j) q_j p_i action."""
    axes = []
    for axis in range(3):
        result: dict[State, complex] = defaultdict(complex)
        for coordinate_mode in range(len(state)):
            for momentum_mode in range(len(state)):
                if coordinate_mode == momentum_mode:
                    continue
                coefficient = zeta[axis, coordinate_mode, momentum_mode] * sqrt(
                    frequency[momentum_mode] / frequency[coordinate_mode]
                )
                if coefficient == 0:
                    continue
                for intermediate, coordinate_value in _coordinate_action(
                    state, coordinate_mode
                ).items():
                    for delta in (-1, 1):
                        if delta < 0 and intermediate[momentum_mode] == 0:
                            continue
                        other = list(intermediate)
                        other[momentum_mode] += delta
                        momentum_value = (
                            1j
                            * delta
                            * sqrt((intermediate[momentum_mode] + int(delta > 0)) / 2)
                        )
                        result[tuple(other)] += (
                            coefficient * coordinate_value * momentum_value
                        )
        axes.append({state: value for state, value in result.items() if value != 0})
    return tuple(axes)


def harmonic_distortion_operator(
    distortion: WatsonHarmonicDistortion,
    J: int,  # noqa: N803
) -> ComplexArray:
    """Actual Hermitian unreduced quartic operator, no A/S parameter inference."""
    if not isinstance(distortion, WatsonHarmonicDistortion):
        raise ValueError("A source-bound harmonic response record is required.")
    distortion = WatsonHarmonicDistortion.model_validate_json(
        distortion.model_dump_json()
    )
    operators = body_fixed_angular_momentum(J)
    first = np.asarray(distortion.linear_inverse_inertia_q_coefficients)
    result = np.zeros((2 * J + 1,) * 2, dtype=np.complex128)
    for mode, frequency in enumerate(distortion.frequencies_hartree):
        response = np.zeros_like(result)
        for first_axis in range(3):
            for second_axis in range(3):
                response += (
                    0.5
                    * first[first_axis, second_axis, mode]
                    * (operators[first_axis] @ operators[second_axis])
                )
        result -= response @ response / (2 * frequency)
    if not np.allclose(result, result.conj().T, rtol=1e-12, atol=1e-20):
        raise ValueError("Actual unreduced distortion operator is not Hermitian.")
    result.setflags(write=False)
    return result


def canonical_watson_mass_scaling_probe(
    precursors: RovibrationalPrecursors,
    force_field: ForceFieldData,
    *,
    basis_states: tuple[State, ...],
    J: int,  # noqa: N803
    epsilon: float,
    maximum_matrix_dimension: int = 512,
) -> ComplexArray:
    """Finite Galerkin H2--H6 polynomial for canonical asymptotic validation.

    This mathematical probe scales nuclear masses formally, not isotope
    observations. It retains all oscillator intermediate paths before final
    Galerkin projection. It omits higher mass orders and is not an exact global
    nuclear Hamiltonian or a production spectroscopy profile.
    """
    precursors, force_field = _validated_inputs(precursors, force_field)
    count = len(force_field.frequencies_hartree)
    operators = body_fixed_angular_momentum(J)
    if (
        isinstance(epsilon, bool)
        or not np.isfinite(epsilon)
        or not 0 < epsilon <= 1
        or type(maximum_matrix_dimension) is not int
        or not 1 <= maximum_matrix_dimension <= 2048
        or not basis_states
        or len(set(basis_states)) != len(basis_states)
        or any(
            len(state) != count
            or any(type(n) is not int or not 0 <= n <= 8 for n in state)
            for state in basis_states
        )
        or count > 6
        or len(basis_states) * (2 * J + 1) > maximum_matrix_dimension
    ):
        raise ValueError(
            "A bounded explicit oscillator basis, J and physical epsilon required."
        )
    inverse, first, second = _inertia_expansion(precursors)
    frequency = np.asarray(force_field.frequencies_hartree)
    zeta = np.asarray(precursors.coriolis_zeta)
    cubic = np.asarray(force_field.cubic_hartree)
    quartic = np.asarray(force_field.quartic_hartree)
    states_by_index = {state: index for index, state in enumerate(basis_states)}

    def apply_pi(state: State, axis: int) -> dict[State, complex]:
        return _coriolis_action(state, zeta, frequency)[axis]

    def apply_mu(state: State, a: int, b: int, order: int) -> dict[State, complex]:
        values: dict[State, complex] = defaultdict(complex)
        if order == 0:
            return {state: complex(inverse[a, b])}
        if order == 1:
            for mode in range(count):
                for other, value in _coordinate_action(state, mode).items():
                    values[other] += first[a, b, mode] * value
        else:
            for mode in range(count):
                for other_mode in range(count):
                    for intermediate, value in _coordinate_action(state, mode).items():
                        for other, next_value in _coordinate_action(
                            intermediate, other_mode
                        ).items():
                            values[other] += (
                                second[a, b, mode, other_mode] * value * next_value
                            )
        return dict(values)

    def projected_matrix(action: object) -> ComplexArray:
        # A callable protocol is supplied locally below, never an engine adapter.
        from collections.abc import Callable
        from typing import cast

        evaluate = cast(Callable[[State], dict[State, complex]], action)
        matrix = np.zeros((len(basis_states),) * 2, dtype=np.complex128)
        for column, state in enumerate(basis_states):
            for other, value in evaluate(state).items():
                row = states_by_index.get(other)
                if row is not None:
                    matrix[row, column] += value
        return matrix

    def apply_product(
        state: State, actions: tuple[object, ...]
    ) -> dict[State, complex]:
        from collections.abc import Callable
        from typing import cast

        terms = {state: 1.0 + 0j}
        for action in actions:
            evaluate = cast(Callable[[State], dict[State, complex]], action)
            updated: dict[State, complex] = defaultdict(complex)
            for intermediate, coefficient in terms.items():
                for other, value in evaluate(intermediate).items():
                    updated[other] += coefficient * value
            terms = dict(updated)
        return terms

    identity = np.eye(2 * J + 1)
    energy = np.asarray(
        [np.dot(np.asarray(state) + 0.5, frequency) for state in basis_states]
    )
    result: ComplexArray = np.asarray(
        epsilon**2 * np.kron(np.diag(energy), identity), dtype=np.complex128
    )
    result += epsilon**3 * np.kron(
        projected_matrix(lambda state: _operator_action(state, cubic)), identity
    )
    result += epsilon**4 * np.kron(
        projected_matrix(lambda state: _operator_action(state, quartic)), identity
    )
    for order in range(3):
        rotational = np.zeros_like(result)
        scalar = np.zeros((len(basis_states),) * 2, dtype=np.complex128)
        for axis in range(3):
            scalar -= (
                projected_matrix(lambda state, a=axis: apply_mu(state, a, a, order)) / 8
            )
            for other_axis in range(3):
                mu_action = partial(apply_mu, a=axis, b=other_axis, order=order)
                pi_first = partial(apply_pi, axis=axis)
                pi_second = partial(apply_pi, axis=other_axis)
                rotational += 0.5 * np.kron(
                    projected_matrix(mu_action), operators[axis] @ operators[other_axis]
                )
                mixed = projected_matrix(
                    lambda state: apply_product(state, (pi_second, mu_action))
                )
                mixed += projected_matrix(
                    lambda state: apply_product(state, (mu_action, pi_second))
                )
                rotational -= 0.5 * np.kron(mixed, operators[axis])
                scalar += 0.5 * projected_matrix(
                    lambda state: apply_product(state, (pi_second, mu_action, pi_first))
                )
        result += epsilon ** (4 + order) * (rotational + np.kron(scalar, identity))
    if not np.allclose(result, result.conj().T, rtol=1e-11, atol=1e-15):
        raise ValueError(
            "The retained complete-path Watson polynomial is not Hermitian."
        )
    result.setflags(write=False)
    return result


def calculate_watson_vibration_rotation(
    precursors: RovibrationalPrecursors,
    force_field: ForceFieldData,
    *,
    stationary_reference: StationaryReference,
    protocol: WatsonCorrectionProtocol | None = None,
) -> WatsonVibrationRotationResult:
    """Derive nonresonant canonical leading beta_v/alpha and harmonic response.

    Algebraic coefficients remain observations if an applicability gate fails.
    Their corresponding model constants/alpha are then unavailable. Near-zero
    coupled denominators are never shifted or silently deleted.
    """
    precursors, force_field = _validated_inputs(precursors, force_field)
    if not isinstance(stationary_reference, StationaryReference):
        raise ValueError("An actual stationary-reference gradient record is required.")
    stationary_reference = StationaryReference.model_validate_json(
        stationary_reference.model_dump_json()
    )
    if (
        stationary_reference.source_artifact_sha256
        not in precursors.identity.parent_artifact_sha256
        or stationary_reference.geometry_sha256 != precursors.identity.geometry_sha256
        or stationary_reference.evidence_class
        != precursors.identity.input_evidence_class
        or len(stationary_reference.gradient_hartree_bohr)
        != len(precursors.identity.symbols)
    ):
        raise ValueError(
            "Reference gradient source, frame/geometry or state identity differs."
        )
    if protocol is None:
        protocol = WatsonCorrectionProtocol()
    if not isinstance(protocol, WatsonCorrectionProtocol):
        raise ValueError("A validated bounded Watson correction protocol is required.")
    protocol = WatsonCorrectionProtocol.model_validate_json(protocol.model_dump_json())
    count = len(force_field.frequencies_hartree)
    if count > protocol.maximum_modes:
        raise ValueError(
            "The requested mode count exceeds the explicit operator budget."
        )
    inverse, inverse_first, inverse_second = _inertia_expansion(precursors)
    equilibrium = inverse / 2
    frequency = np.asarray(force_field.frequencies_hartree)
    zeta = np.asarray(precursors.coriolis_zeta)
    cubic_tensor = np.asarray(force_field.cubic_hartree)
    field = _force_field(force_field)
    precursor_sha = artifact_digest(precursors.model_dump(mode="json"))
    distortion = WatsonHarmonicDistortion(
        precursor_sha256=precursor_sha,
        unreduced_tau_hartree=_tensor4(
            -0.5
            * np.einsum("abk,cdk,k->abcd", inverse_first, inverse_first, 1 / frequency)
        ),
        linear_inverse_inertia_q_coefficients=_tensor3(inverse_first),
        frequencies_hartree=tuple(float(value) for value in frequency),
    )
    derived_volume, watson_volume = watson_volume_identity(
        precursors, tuple([0.0] * count)
    )
    volume_residual = abs(derived_volume - watson_volume)
    if volume_residual > 1e-12 * max(abs(watson_volume), 1e-15):
        raise ValueError("Complete-mode fixed-measure Watson volume identity failed.")
    targets = (tuple([0] * count),) + tuple(
        tuple(int(index == mode) for index in range(count)) for mode in range(count)
    )
    states = []
    for target in targets:
        reasons = []
        direct = 0.5 * np.einsum("abkk,k->ab", inverse_second, np.asarray(target) + 0.5)
        cubic = np.zeros((3, 3))
        coriolis = np.zeros((3, 3))
        records = []
        cubic_complete, coriolis_complete = True, True
        cubic_action = _operator_action(target, cubic_tensor)
        pi_action = _coriolis_action(target, zeta, frequency)
        virtual_states = set(cubic_action).union(
            *(action.keys() for action in pi_action)
        )
        if len(virtual_states) > protocol.maximum_intermediate_states:
            raise ValueError("Exact virtual-state support exceeds the declared budget.")
        resonances = analyze_resonances(
            field,
            states=(target,),
            detuning_threshold_cm1=protocol.minimum_denominator_cm1,
            coupling_to_detuning_threshold=protocol.maximum_anharmonic_coupling_ratio,
            coupling_floor_cm1=protocol.coupling_floor_cm1,
        )
        for resonance in resonances:
            reasons.append(
                f"{resonance.kind}: {resonance.state_a}<->{resonance.state_b}; "
                f"coupling={resonance.coupling_cm1:.12g} cm^-1, "
                f"detuning={resonance.harmonic_detuning_cm1:.12g} cm^-1"
            )
        for first_mode in range(count):
            for second_mode in range(first_mode + 1, count):
                if abs(
                    frequency[first_mode] - frequency[second_mode]
                ) * HARTREE_CM1 <= protocol.minimum_denominator_cm1 and (
                    target[first_mode] or target[second_mode]
                ):
                    reasons.append(
                        "occupied near-degenerate harmonic modes "
                        "require an explicit vibrational block"
                    )
        energy_reasons = list(reasons)
        harmonic_energy = float(np.dot(np.asarray(target) + 0.5, frequency))
        quartic_energy = float(
            _operator_action(target, np.asarray(force_field.quartic_hartree)).get(
                target, 0.0
            )
        )
        cubic_energy = 0.0
        cubic_energy_complete = True
        for other, value in cubic_action.items():
            if other == target or value == 0:
                continue
            denominator = float(
                np.dot(np.asarray(target) - np.asarray(other), frequency)
            )
            if abs(denominator) * HARTREE_CM1 <= protocol.minimum_denominator_cm1:
                cubic_energy_complete = False
                cubic_block_reason = (
                    "coupled cubic vibrational denominator requires an explicit block"
                )
                energy_reasons.append(cubic_block_reason)
                reasons.append(cubic_block_reason)
            else:
                cubic_energy += value**2 / denominator
        kinetic_energy = 0.0
        for other in sorted(set().union(*(action.keys() for action in pi_action))):
            pi_values = np.asarray([action.get(other, 0j) for action in pi_action])
            kinetic_energy += 0.5 * float(
                np.real(pi_values.conj() @ inverse @ pi_values)
            )
        semirigid_energy = None
        if cubic_energy_complete:
            semirigid_energy = (
                harmonic_energy
                + quartic_energy
                + cubic_energy
                + kinetic_energy
                + watson_volume
            )
            relative_energy = abs(semirigid_energy - harmonic_energy) / harmonic_energy
            if relative_energy >= protocol.maximum_relative_vibrational_correction:
                energy_reasons.append(
                    "vibrational energy correction exceeds "
                    "the declared relative applicability bound"
                )
        anharmonic_ratio = 0.0
        for other, value in cubic_action.items():
            if other == target or value == 0:
                continue
            denominator = float(
                np.dot(np.asarray(target) - np.asarray(other), frequency)
            )
            if denominator != 0:
                anharmonic_ratio = max(anharmonic_ratio, abs(value / denominator))
        if anharmonic_ratio >= protocol.maximum_anharmonic_coupling_ratio:
            cubic_ratio_reason = (
                f"cubic applicability ratio {anharmonic_ratio:.12g} exceeds "
                f"{protocol.maximum_anharmonic_coupling_ratio}"
            )
            reasons.append(cubic_ratio_reason)
            energy_reasons.append(cubic_ratio_reason)
        for mode in range(count):
            for other, value in _coordinate_action(target, mode).items():
                numerator = (
                    inverse_first[:, :, mode] * value * cubic_action.get(other, 0.0)
                )
                if not np.any(numerator):
                    continue
                denominator = float(
                    np.dot(np.asarray(target) - np.asarray(other), frequency)
                )
                reason = None
                if abs(denominator) * HARTREE_CM1 <= protocol.minimum_denominator_cm1:
                    reason = (
                        "coupled cubic/linear-inertia denominator below "
                        f"{protocol.minimum_denominator_cm1} cm^-1"
                    )
                    cubic_complete = False
                    reasons.append(reason)
                    contribution = None
                else:
                    values = numerator / denominator
                    cubic += values
                    contribution = _rows(values)
                records.append(
                    VirtualStateContribution(
                        operator="cubic_linear_inertia_response",
                        virtual_state=other,
                        denominator_hartree=denominator,
                        numerator_tensor_hartree2=_rows(numerator),
                        contribution_tensor_hartree=contribution,
                        blocking_reason=reason,
                    )
                )
        coriolis_ratio = 0.0
        for other in sorted(set().union(*(action.keys() for action in pi_action))):
            if other == target:
                continue
            pi_values = np.asarray([action.get(other, 0j) for action in pi_action])
            coupling = -inverse @ pi_values
            if not np.any(coupling):
                continue
            complex_numerator = np.outer(coupling.conj(), coupling)
            if not np.allclose(complex_numerator.imag, 0, rtol=0, atol=1e-25):
                raise ValueError("Real isolated-state response became complex.")
            numerator = np.asarray(complex_numerator.real, dtype=float)
            denominator = float(
                np.dot(np.asarray(target) - np.asarray(other), frequency)
            )
            reason = None
            if abs(denominator) * HARTREE_CM1 <= protocol.minimum_denominator_cm1:
                reason = (
                    "coupled Coriolis denominator below "
                    f"{protocol.minimum_denominator_cm1} cm^-1; "
                    "a rovibrational block is required"
                )
                coriolis_complete = False
                reasons.append(reason)
                contribution = None
            else:
                values = numerator / denominator
                coriolis += values
                contribution = _rows(values)
                ratio = (
                    sqrt(protocol.maximum_J * (protocol.maximum_J + 1))
                    * float(np.linalg.norm(coupling))
                    / abs(denominator)
                )
                coriolis_ratio = max(coriolis_ratio, ratio)
                if ratio >= protocol.maximum_coriolis_coupling_ratio:
                    reasons.append(
                        f"Coriolis applicability ratio {ratio:.12g} exceeds "
                        f"{protocol.maximum_coriolis_coupling_ratio}"
                    )
            records.append(
                VirtualStateContribution(
                    operator="coriolis_second_order",
                    virtual_state=other,
                    denominator_hartree=denominator,
                    numerator_tensor_hartree2=_rows(numerator),
                    contribution_tensor_hartree=contribution,
                    blocking_reason=reason,
                )
            )
        total = None
        relative = None
        if cubic_complete and coriolis_complete:
            total = equilibrium + direct + cubic + coriolis
            relative = float(
                np.linalg.norm(direct + cubic + coriolis) / np.linalg.norm(equilibrium)
            )
            if relative >= protocol.maximum_relative_rotational_correction:
                reasons.append(
                    f"rotational correction ratio {relative:.12g} exceeds "
                    f"{protocol.maximum_relative_rotational_correction}"
                )
            if np.any(np.diag(total) <= 0):
                reasons.append("nonpositive corrected equilibrium-axis coefficient")
        if precursors.degenerate_inertia_axis_pairs:
            reasons.append(
                "degenerate equilibrium axes require a validated rotational block"
            )
        reasons = list(dict.fromkeys(reasons))
        available = not reasons and total is not None
        constants = None
        if available and total is not None:
            diagonal = np.diag(total) * _HARTREE_MHZ
            constants = (float(diagonal[0]), float(diagonal[1]), float(diagonal[2]))
        states.append(
            WatsonStateCorrection(
                vibrational_state=target,
                direct_inverse_inertia_tensor_hartree=_rows(direct),
                cubic_response_tensor_hartree=_rows(cubic) if cubic_complete else None,
                coriolis_response_tensor_hartree=_rows(coriolis)
                if coriolis_complete
                else None,
                algebraic_coefficient_tensor_hartree=_rows(total)
                if total is not None
                else None,
                model_constants_status="available_model_prediction"
                if available
                else "blocked",
                equilibrium_axis_model_constants_mhz=constants,
                blocking_reasons=tuple(reasons),
                maximum_observed_anharmonic_coupling_ratio=anharmonic_ratio,
                maximum_observed_coriolis_coupling_ratio=coriolis_ratio,
                relative_rotational_correction_norm=relative,
                virtual_state_contributions=tuple(records),
                harmonic_energy_hartree=harmonic_energy,
                quartic_first_order_hartree=quartic_energy,
                cubic_second_order_hartree=cubic_energy
                if cubic_energy_complete
                else None,
                coriolis_kinetic_first_order_hartree=kinetic_energy,
                watson_volume_scalar_hartree=watson_volume,
                algebraic_semirigid_energy_hartree=semirigid_energy,
                semirigid_energy_status=(
                    "available_model_prediction"
                    if not energy_reasons and semirigid_energy is not None
                    else "blocked"
                ),
                semirigid_energy_blocking_reasons=tuple(dict.fromkeys(energy_reasons)),
            )
        )
    alpha = None
    if all(state.algebraic_coefficient_tensor_hartree is not None for state in states):
        alpha = np.asarray(states[0].algebraic_coefficient_tensor_hartree) - np.asarray(
            [state.algebraic_coefficient_tensor_hartree for state in states[1:]]
        )
    all_available = all(
        state.model_constants_status == "available_model_prediction" for state in states
    )
    semirigid_reasons = list(
        dict.fromkeys(
            reason
            for state in states
            for reason in state.semirigid_energy_blocking_reasons
        )
    )
    semirigid = None
    if not semirigid_reasons and all(
        state.algebraic_semirigid_energy_hartree is not None for state in states
    ):
        energy = np.asarray(
            [state.algebraic_semirigid_energy_hartree for state in states], dtype=float
        )
        fundamentals = (energy[1:] - energy[0]) * HARTREE_CM1
        if np.any(fundamentals <= 0):
            semirigid_reasons.append("nonpositive corrected semirigid fundamental")
        else:
            semirigid = SemirigidVPT2Data(
                scientific_context=force_field.scientific_context,
                force_field_sha256=force_field.source_digest,
                precursor_sha256=precursor_sha,
                state_energies_hartree=tuple(float(value) for value in energy),
                ground_energy_hartree=float(energy[0]),
                fundamental_frequencies_cm1=tuple(
                    float(value) for value in fundamentals
                ),
                harmonic_frequencies_cm1=tuple(
                    float(value) for value in frequency * HARTREE_CM1
                ),
                quartic_first_order_hartree=tuple(
                    state.quartic_first_order_hartree for state in states
                ),
                cubic_second_order_hartree=tuple(
                    float(state.cubic_second_order_hartree)
                    for state in states
                    if state.cubic_second_order_hartree is not None
                ),
                coriolis_kinetic_first_order_hartree=tuple(
                    state.coriolis_kinetic_first_order_hartree for state in states
                ),
                watson_volume_scalar_hartree=watson_volume,
                rotation_vibration_model_available=all_available,
            )
    if semirigid is None and not semirigid_reasons:
        semirigid_reasons.append(
            "incomplete nonresonant vibrational intermediate-state sums"
        )
    payload = {
        "scientific_context": force_field.scientific_context,
        "precursor_sha256": precursor_sha,
        "force_field_sha256": force_field.source_digest,
        "algorithm_source_sha256": _SOURCE_SHA256,
        "stationary_reference": stationary_reference,
        "protocol": protocol,
        "equilibrium_coefficient_tensor_hartree": _rows(equilibrium),
        "states": tuple(states),
        "algebraic_alpha_tensors_hartree": _tensor3(alpha)
        if alpha is not None
        else None,
        "model_alpha_tensors_mhz": _tensor3(alpha * _HARTREE_MHZ)
        if alpha is not None and all_available
        else None,
        "ground_state_constants_mhz": states[0].equilibrium_axis_model_constants_mhz,
        "harmonic_distortion": distortion,
        "semirigid_vpt2": semirigid,
        "semirigid_vpt2_blocking_reasons": tuple(semirigid_reasons),
        "watson_volume_scalar_reference_hartree": watson_volume,
        "independently_derived_volume_scalar_reference_hartree": derived_volume,
        "watson_volume_identity_absolute_residual_hartree": volume_residual,
        "reference_implementation": _REFERENCE,
        "limitations": (
            "leading canonical nonresonant J^2 correction, not resonant GVPT2",
            "isolated ground/single-mode labels; resonances need explicit blocks",
            "experimental constants without independent accuracy calibration",
            "unreduced leading harmonic quartic response; A/S reductions unavailable",
            "no higher distortion, hyperfine, tunneling or nuclear-spin restrictions",
            "input declarations bound; workflow authenticates actual native artifacts",
            "equilibrium-axis diagonal constants; no reduction-induced shifts included",
        ),
    }
    initial = WatsonVibrationRotationResult.model_construct(
        _fields_set=set(payload), **payload, content_sha256="0" * 64
    )
    encoded = initial.model_dump(mode="json", exclude={"content_sha256"})
    return WatsonVibrationRotationResult.model_validate_json(
        json.dumps(
            {
                **encoded,
                "content_sha256": artifact_digest(encoded),
            }
        )
    )
