"""Real numerical/library payload checks; mathematical models are not engine proof."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.domain import StageResult, digest
from cochem_torq.spectroscopy import (
    analyze_hessian,
    analyze_resonances,
    build_force_field,
    equilibrium_rotor,
    rigid_rotor_catalog,
    vibrational_vpt2,
)
from cochem_torq.spectroscopy.harmonic import finite_array
from cochem_torq.spectroscopy.results import (
    ForceFieldData,
    ResonanceAnalysisData,
    RigidRotorCatalogData,
    ScientificContext,
    VibrationalVPT2Data,
    make_scientific_context,
)
from Libraries.cochem_isotopes import isotope_record


def plain(value):
    return json.loads(json.dumps(asdict(value), default=lambda x: x.tolist()))


@pytest.fixture(scope="module")
def mathematical_payloads(tmp_path_factory):
    directory = tmp_path_factory.mktemp("actual-mathematical-records")
    description = directory / "model.txt"
    description.write_text(
        "Analytic one-mode spring plus cubic/quartic polynomial; "
        "not an electronic-structure engine or molecular accuracy reference.\n"
    )
    parent = sha256(description.read_bytes()).hexdigest()
    records = [isotope_record("1H"), isotope_record("1H")]
    masses = [r["mass_u"] for r in records]
    coordinates = np.asarray([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    hessian = np.zeros((6, 6))
    hessian[0, 0] = hessian[3, 3] = 0.4
    hessian[0, 3] = hessian[3, 0] = -0.4
    harmonic = analyze_hessian(coordinates, masses, hessian)
    inverse = np.linalg.pinv(harmonic.dimensionless_to_cartesian)
    omega = harmonic.angular_frequencies_au[0]

    def polynomial(geometry):
        q = (inverse @ (geometry - coordinates).ravel())[0]
        return float(omega * q**2 / 2 + 1e-5 * q**3 / 6 + 1e-5 * q**4 / 24)

    field = build_force_field(
        harmonic,
        polynomial,
        evaluator_identity=description.read_text().strip(),
        absolute_tolerance_hartree=1e-9,
    )
    vpt = vibrational_vpt2(field)
    molecule = {
        "symbols": ["H", "H"],
        "atom_ids": ["left-H", "right-H"],
        "isotopes": [1, 1],
        "charge": 0,
        "multiplicity": 1,
    }
    arguments = {
        "molecule": molecule,
        "geometry_bohr": coordinates,
        "isotope_provenance": records,
        "recipe_sha256": digest({"mathematical_model": description.read_text()}),
        "protocol_sha256": digest({"steps": field.steps_dimensionless}),
        "parent_artifact_sha256": [parent],
        "evidence_class": "mathematical_model",
        "harmonic": harmonic,
    }
    context = make_scientific_context(**arguments)
    field_value = plain(field)
    field_value["scientific_context"] = context
    vpt_value = plain(vpt)
    vpt_value["scientific_context"] = context
    resonance_value = {
        "scientific_context": context,
        "resonances": [plain(r) for r in analyze_resonances(field)],
        "force_field_sha256": field.source_digest,
        "coriolis_resonances": "not_implemented",
        "protocol_sha256": arguments["protocol_sha256"],
    }
    rotor = equilibrium_rotor(coordinates, masses)
    # Explicit mathematical dipole to test rotor formulas; H2 is not claimed polar.
    catalog = rigid_rotor_catalog(
        rotor.constants_mhz,
        [1.0, 0.0, 0.0],
        temperature_kelvin=10.0,
        J_max=3,
        constant_observable="Be",
    )
    catalog_value = plain(catalog)
    catalog_value["dipole_origin"] = "center_of_mass"
    catalog_value["scientific_context"] = make_scientific_context(
        **arguments, principal_axes_columns=rotor.principal_axes_columns
    )
    return {
        "context": context,
        "field": field_value,
        "vpt": vpt_value,
        "resonances": resonance_value,
        "catalog": catalog_value,
        "arguments": arguments,
    }


def test_actual_mathematical_stage_payloads_validate(mathematical_payloads):
    for key, schema in (
        ("field", ForceFieldData),
        ("vpt", VibrationalVPT2Data),
        ("resonances", ResonanceAnalysisData),
        ("catalog", RigidRotorCatalogData),
    ):
        value = schema.model_validate(mathematical_payloads[key])
        assert value.scientific_context.evidence_class == "mathematical_model"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("geometry_sha256", "bad"),
        ("atom_ids", ["duplicate", "duplicate"]),
        ("isotope_numbers", [1]),
        ("isotope_masses_u", [float("nan"), 1.0]),
        ("isotope_selection_policies", ["natural_average", "natural_average"]),
        ("mode_count", 2),
        ("mode_basis_sha256", None),
        ("frame_axes_columns", [[1.0, 0.0, 0.0]]),
        ("frame_axes_columns", [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("recipe_sha256", ""),
        ("parent_artifact_sha256", []),
    ],
)
def test_invalid_identity_and_frame_rejected(mathematical_payloads, field, value):
    payload = deepcopy(mathematical_payloads["context"])
    payload[field] = value
    with pytest.raises(ValidationError):
        ScientificContext.model_validate(payload)


def test_geometry_bytes_digest_and_harmonic_isotopologue_bindings(
    mathematical_payloads,
):
    payload = deepcopy(mathematical_payloads["context"])
    payload["geometry_bohr"][0][0] += 0.1
    with pytest.raises(ValidationError, match="Geometry digest"):
        ScientificContext.model_validate(payload)
    arguments = dict(mathematical_payloads["arguments"])
    arguments["geometry_bohr"] = np.asarray(arguments["geometry_bohr"]) + 1.0
    with pytest.raises(ValueError, match="geometry and isotopologue"):
        make_scientific_context(**arguments)


def test_no_rounding_average_mass_into_isotope_or_losing_source_mapping(
    mathematical_payloads,
):
    arguments = deepcopy(mathematical_payloads["arguments"])
    arguments["isotope_provenance"][0]["selection_policy"] = "natural_average"
    with pytest.raises(ValueError, match="Isotope evidence"):
        make_scientific_context(**arguments)
    arguments = dict(mathematical_payloads["arguments"])
    arguments["molecule"] = dict(arguments["molecule"], atom_ids=None)
    context = make_scientific_context(**arguments)
    assert context["atom_id_policy"] == "input_ordinal"
    assert context["atom_ids"] == ["atom-0", "atom-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cubic_hartree", []),
        ("quartic_hartree", [[[[float("inf")]]]]),
        ("steps_dimensionless", [0.04, 0.08]),
        ("derivative_converged", False),
        ("evaluation_count", 1),
        ("reference_energy_hartree", 1.0),
        ("coordinate_convention", "mass-weighted Q without units"),
        ("quartic_scope", "missing_cross_terms"),
    ],
)
def test_force_field_invalid_or_false_success_rejected(
    mathematical_payloads, field, value
):
    payload = deepcopy(mathematical_payloads["field"])
    payload[field] = value
    with pytest.raises(ValidationError):
        ForceFieldData.model_validate(payload)


def test_engine_label_cannot_hide_missing_displacement_artifacts(mathematical_payloads):
    payload = deepcopy(mathematical_payloads["field"])
    payload["scientific_context"]["evidence_class"] = "engine_calculation"
    with pytest.raises(ValidationError, match="native artifact digest"):
        ForceFieldData.model_validate(payload)


def test_changed_force_tensor_cannot_reuse_old_digest(mathematical_payloads):
    payload = deepcopy(mathematical_payloads["field"])
    payload["cubic_hartree"][0][0][0] += 1e-11
    with pytest.raises(ValidationError, match="Force-field digest"):
        ForceFieldData.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["ground_energy_hartree", "fundamental_frequencies_cm1"]
)
def test_vpt2_values_must_follow_retained_corrections(mathematical_payloads, field):
    payload = deepcopy(mathematical_payloads["vpt"])
    if field == "ground_energy_hartree":
        payload[field] += 0.01
    else:
        payload[field][0] += 10.0
    with pytest.raises(ValidationError, match="retained corrections"):
        VibrationalVPT2Data.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["rotation_vibration_available", "independent_scientific_qualification"]
)
def test_vibrational_only_vpt2_cannot_promote_qualification(
    mathematical_payloads, field
):
    payload = deepcopy(mathematical_payloads["vpt"])
    payload[field] = True
    with pytest.raises(ValidationError):
        VibrationalVPT2Data.model_validate(payload)


def test_resonance_state_dimensions_and_protocol_identity(mathematical_payloads):
    payload = deepcopy(mathematical_payloads["resonances"])
    payload["protocol_sha256"] = digest({"different": "protocol"})
    with pytest.raises(ValidationError, match="protocol differs"):
        ResonanceAnalysisData.model_validate(payload)
    payload = deepcopy(mathematical_payloads["resonances"])
    # Contract rejection input only; no assertion that this is a molecular resonance.
    payload["resonances"] = [
        {
            "state_a": [0, 0],
            "state_b": [1, 0],
            "operator_order": 3,
            "harmonic_detuning_cm1": 1.0,
            "coupling_cm1": 1.0,
            "kind": "vibrational",
        }
    ]
    with pytest.raises(ValidationError, match="mode occupations"):
        ResonanceAnalysisData.model_validate(payload)


def test_catalog_cannot_claim_identification_or_change_axis_basis(
    mathematical_payloads,
):
    payload = deepcopy(mathematical_payloads["catalog"])
    payload["identification_qualified"] = True
    with pytest.raises(ValidationError):
        RigidRotorCatalogData.model_validate(payload)
    payload = deepcopy(mathematical_payloads["catalog"])
    payload["scientific_context"] = mathematical_payloads["context"]
    with pytest.raises(ValidationError, match="principal frame"):
        RigidRotorCatalogData.model_validate(payload)


def test_catalog_rejects_duplicate_and_out_of_range_quantum_numbers(
    mathematical_payloads,
):
    payload = deepcopy(mathematical_payloads["catalog"])
    assert payload["lines"]
    payload["lines"].append(deepcopy(payload["lines"][0]))
    with pytest.raises(ValidationError, match="Duplicate transition"):
        RigidRotorCatalogData.model_validate(payload)
    payload = deepcopy(mathematical_payloads["catalog"])
    payload["lines"][0]["upper_eigenstate_index"] = 99
    with pytest.raises(ValidationError, match="finite-J basis"):
        RigidRotorCatalogData.model_validate(payload)


def test_available_advanced_stage_requires_typed_context(mathematical_payloads):
    value = deepcopy(mathematical_payloads["field"])
    StageResult(status="available", observable="anharmonic_force_field", value=value)
    del value["scientific_context"]
    with pytest.raises(ValidationError, match="scientific_context"):
        StageResult(
            status="available", observable="anharmonic_force_field", value=value
        )


@pytest.mark.parametrize(
    "field",
    [
        "mass_weighted_modes",
        "cartesian_modes",
        "dimensionless_to_cartesian",
        "angular_frequencies_au",
        "frequencies_cm1",
        "eigenvalues_au",
    ],
)
def test_direct_force_field_rejects_inconsistent_mode_normalization_before_work(
    mathematical_payloads, field
):
    harmonic = mathematical_payloads["arguments"]["harmonic"]
    invalid = replace(harmonic, **{field: np.asarray(getattr(harmonic, field)) * 2})
    evaluated = []

    def mathematical_energy(coordinates):
        evaluated.append(coordinates)
        return float(np.sum(coordinates**2))

    with pytest.raises(ValueError, match="coordinate normalization disagree"):
        build_force_field(
            invalid, mathematical_energy, evaluator_identity="explicit analytic model"
        )
    assert evaluated == []


def test_force_field_cannot_discard_complex_energy(mathematical_payloads):
    harmonic = mathematical_payloads["arguments"]["harmonic"]

    def complex_mathematical_quantity(coordinates):
        return np.complex128(np.sum(coordinates**2) + 2j)

    with pytest.raises(ValueError, match="finite converged hartree energy"):
        build_force_field(
            harmonic,
            complex_mathematical_quantity,
            evaluator_identity="deliberately invalid complex mathematical quantity",
        )


@pytest.mark.parametrize("quantity", [np.asarray([1.0 + 1e-16j]), [[1.0 + 2j]]])
def test_real_physical_array_never_discards_imaginary_parts(quantity):
    with pytest.raises(ValueError, match="imaginary parts cannot be discarded"):
        finite_array(quantity)


@pytest.mark.parametrize("field", ["geometry_bohr", "principal_axes_columns"])
def test_context_helper_never_discards_imaginary_parts(mathematical_payloads, field):
    arguments = dict(mathematical_payloads["arguments"])
    original = arguments["geometry_bohr"] if field == "geometry_bohr" else np.eye(3)
    arguments[field] = np.asarray(original, dtype=complex) + 1e-16j
    with pytest.raises(ValueError, match="imaginary parts cannot be discarded"):
        make_scientific_context(**arguments)
