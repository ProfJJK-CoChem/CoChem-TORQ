"""Genuine HF modes and independently evaluated geometric tensor identities.

Displaced inertia is exact geometry mathematics, not another electronic result.
Derived phase/frame/projector cases are labelled mathematical_model and never
substitute for native force fields or independently qualified rovibration.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.engines.pyscf_backend import PySCFBackend
from cochem_torq.spectroscopy.harmonic import (
    analyze_hessian,
    artifact_digest,
    equilibrium_rotor,
)
from cochem_torq.spectroscopy.results import (
    ScientificContext,
    make_scientific_context,
)
from cochem_torq.spectroscopy.rovibrational import (
    RovibrationalPrecursors,
    build_rovibrational_precursors,
)
from cochem_torq.units import ATOMIC_MASS_ELECTRON


@pytest.fixture(scope="module", params=["h2", "water"])
def genuine_modes(request, tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    from Libraries.cochem_isotopes import isotope_record

    symbols = ["H", "H"] if request.param == "h2" else ["O", "H", "H"]
    geometry = (
        [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]
        if request.param == "h2"
        else [[0.0, 0.0, 0.0], [0.0, 0.0, 2.15], [1.9, 0.0, -0.5]]
    )
    backend_request = {
        "molecule": {
            "symbols": symbols,
            "atom_ids": [f"source-atom-{i}" for i in range(len(symbols))],
            "geometry_bohr": geometry,
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {"name": "hf", "basis": "sto-3g", "reference": "restricted"},
        "properties": ["energy", "gradient", "hessian"],
        "settings": {"threads": 1, "scf_energy_tolerance": 1e-12},
    }
    directory = tmp_path_factory.mktemp(f"genuine-{request.param}-precursors")
    native = PySCFBackend().optimize(backend_request, directory / "native")
    assert native["status"] == "complete", native["errors"]
    assert native["stability"]["status"] == "stable"
    assert np.max(np.abs(native["gradient_hartree_bohr"])) < 2e-5
    isotopes = [isotope_record(symbol) for symbol in symbols]
    masses = [record["mass_u"] for record in isotopes]
    harmonic = analyze_hessian(
        native["geometry_bohr"], masses, native["hessian_hartree_bohr2"]
    )
    rotor = equilibrium_rotor(harmonic.coordinates_bohr, masses)
    recipe_sha = artifact_digest(
        {
            "method": native["method"],
            "settings": native["settings"],
            "engine_installation_sha256": native["engine_installation_sha256"],
        }
    )
    protocol_sha = artifact_digest(
        {"purpose": "geometric precursor mathematical verification", "normal_Q": True}
    )
    context = ScientificContext.model_validate(
        make_scientific_context(
            molecule=backend_request["molecule"],
            geometry_bohr=harmonic.coordinates_bohr,
            isotope_provenance=isotopes,
            recipe_sha256=recipe_sha,
            protocol_sha256=protocol_sha,
            parent_artifact_sha256=[native["manifest_sha256"]],
            evidence_class="engine_calculation",
            harmonic=harmonic,
            principal_axes_columns=rotor.principal_axes_columns,
        )
    )
    return request.param, harmonic, rotor, context, native, isotopes, backend_request


def _derived_context(case, harmonic, rotor, protocol):
    _, _, _, old_context, _, isotopes, backend_request = case
    return ScientificContext.model_validate(
        make_scientific_context(
            molecule=backend_request["molecule"],
            geometry_bohr=harmonic.coordinates_bohr,
            isotope_provenance=isotopes,
            recipe_sha256=old_context.recipe_sha256,
            protocol_sha256=artifact_digest(protocol),
            parent_artifact_sha256=old_context.parent_artifact_sha256,
            evidence_class="mathematical_model",
            harmonic=harmonic,
            principal_axes_columns=rotor.principal_axes_columns,
        )
    )


def _inertia_direct(positions, masses):
    """Independent component-wise contraction of the defined geometric tensor."""
    value = np.zeros((3, 3))
    for atom, mass in zip(positions, masses):
        squared = sum(float(x) ** 2 for x in atom)
        for first in range(3):
            for second in range(3):
                value[first, second] += mass * (
                    (squared if first == second else 0) - atom[first] * atom[second]
                )
    return value


@pytest.mark.real_engine
def test_genuine_modes_have_exact_tensor_identities_and_provenance(genuine_modes):
    _, harmonic, rotor, context, native, _, _ = genuine_modes
    value = build_rovibrational_precursors(harmonic, rotor, context)
    zeta = np.asarray(value.coriolis_zeta)
    first = np.asarray(value.inertia_first_derivative)
    second = np.asarray(value.inertia_second_derivative)
    count = len(harmonic.frequencies_cm1)
    assert np.array_equal(zeta, -zeta.swapaxes(1, 2))
    assert np.array_equal(np.diagonal(zeta, axis1=1, axis2=2), np.zeros((3, count)))
    assert np.max(np.abs(zeta)) <= 1 + 1e-12
    assert np.allclose(first, first.swapaxes(0, 1), atol=1e-12)
    assert np.allclose(second, second.swapaxes(0, 1), atol=1e-12)
    assert np.allclose(second, second.swapaxes(2, 3), atol=1e-12)
    assert np.allclose(
        np.trace(second, axis1=0, axis2=1), 4 * np.eye(count), atol=1e-12
    )
    assert value.translation_residual < 1e-12
    assert value.rotational_eckart_residual < 1e-12
    assert value.identity.harmonic_source_sha256 == harmonic.source_digest
    assert value.identity.parent_artifact_sha256 == (native["manifest_sha256"],)
    assert value.identity.isotope_numbers == tuple(context.isotope_numbers)
    assert value.identity.mode_basis_sha256 == context.mode_basis_sha256
    assert value.identity.frame_sha256 == context.frame_sha256
    assert value.identity.algorithm_source_sha256 == artifact_digest_source_file()
    assert value.identity.constants_sha256 != "0" * 64
    assert all(
        item.status == "unavailable" and item.value is None
        for item in value.unavailable_products
    )
    assert not value.identification_ready
    assert not value.independent_scientific_qualification
    assert RovibrationalPrecursors.model_validate_json(value.model_dump_json()) == value


def artifact_digest_source_file():
    from hashlib import sha256

    import cochem_torq.spectroscopy.rovibrational as source

    return sha256(Path(source.__file__).read_bytes()).hexdigest()


@pytest.mark.real_engine
def test_genuine_inertia_derivatives_match_direct_displacements_two_scales(
    genuine_modes,
):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    value = build_rovibrational_precursors(harmonic, rotor, context)
    positions = np.asarray(value.centered_geometry_principal_bohr)
    masses = np.asarray(value.identity.isotope_masses_u) * ATOMIC_MASS_ELECTRON
    displacement = (
        np.asarray(value.mass_weighted_modes_principal) / np.sqrt(masses)[:, None, None]
    )
    first = np.asarray(value.inertia_first_derivative)
    second = np.asarray(value.inertia_second_derivative)
    reference = _inertia_direct(positions, masses)
    count = len(harmonic.frequencies_cm1)
    for step in (0.4, 0.2):
        for mode in range(count):
            plus = _inertia_direct(positions + step * displacement[:, :, mode], masses)
            minus = _inertia_direct(positions - step * displacement[:, :, mode], masses)
            assert np.allclose(
                (plus - minus) / (2 * step), first[:, :, mode], rtol=1e-10, atol=1e-8
            )
            assert np.allclose(
                (plus - 2 * reference + minus) / step**2,
                second[:, :, mode, mode],
                rtol=1e-7,
                atol=1e-7,
            )
            for other in range(mode + 1, count):
                sampled = []
                for sign_first, sign_second in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    sampled.append(
                        _inertia_direct(
                            positions
                            + step
                            * (
                                sign_first * displacement[:, :, mode]
                                + sign_second * displacement[:, :, other]
                            ),
                            masses,
                        )
                    )
                mixed = (sampled[0] - sampled[1] - sampled[2] + sampled[3]) / (
                    4 * step**2
                )
                assert np.allclose(
                    mixed, second[:, :, mode, other], rtol=1e-7, atol=1e-7
                )


@pytest.mark.real_engine
def test_genuine_coriolis_matches_independent_component_contraction(genuine_modes):
    name, harmonic, rotor, context, _, _, _ = genuine_modes
    value = build_rovibrational_precursors(harmonic, rotor, context)
    modes = np.asarray(value.mass_weighted_modes_principal)
    count = len(harmonic.frequencies_cm1)
    expected = np.zeros((3, count, count))
    for axis, (first, second) in enumerate(((1, 2), (2, 0), (0, 1))):
        for mode in range(count):
            for other in range(count):
                expected[axis, mode, other] = sum(
                    atom[first, mode] * atom[second, other]
                    - atom[second, mode] * atom[first, other]
                    for atom in modes
                )
    assert np.allclose(value.coriolis_zeta, expected, rtol=0, atol=1e-15)
    if name == "h2":
        assert not np.any(expected)
        assert value.zero_inertia_axes == (0,)
        assert value.degenerate_inertia_axis_pairs == ((1, 2),)
        mass = np.asarray(harmonic.isotope_masses_u) * ATOMIC_MASS_ELECTRON
        reduced = mass.prod() / mass.sum()
        bond = np.linalg.norm(
            harmonic.coordinates_bohr[1] - harmonic.coordinates_bohr[0]
        )
        assert np.abs(
            np.asarray(value.inertia_first_derivative)[1:, 1:, 0].diagonal()
        ) == pytest.approx(np.full(2, 2 * np.sqrt(reduced) * bond), rel=1e-12)
        assert np.asarray(value.inertia_second_derivative)[
            1:, 1:, 0, 0
        ].diagonal() == pytest.approx([2.0, 2.0], abs=1e-12)
    else:
        assert np.max(np.abs(expected)) > 1e-3


def test_phase_change_is_retained_not_assigned_physical_significance(genuine_modes):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    original = build_rovibrational_precursors(harmonic, rotor, context)
    phase = np.ones(len(harmonic.frequencies_cm1))
    phase[0] = -1
    changed = replace(
        harmonic,
        mass_weighted_modes=harmonic.mass_weighted_modes * phase,
        cartesian_modes=harmonic.cartesian_modes * phase,
        dimensionless_to_cartesian=harmonic.dimensionless_to_cartesian * phase,
    )
    changed_context = _derived_context(
        genuine_modes, changed, rotor, {"mode_phase": phase.tolist()}
    )
    value = build_rovibrational_precursors(changed, rotor, changed_context)
    assert value.identity.input_evidence_class == "mathematical_model"
    assert value.identity.mode_basis_sha256 != original.identity.mode_basis_sha256
    pair_phase = phase[:, None] * phase[None, :]
    assert np.allclose(
        value.coriolis_zeta, np.asarray(original.coriolis_zeta) * pair_phase
    )
    assert np.allclose(
        value.inertia_first_derivative,
        np.asarray(original.inertia_first_derivative) * phase,
    )
    assert np.allclose(
        value.inertia_second_derivative,
        np.asarray(original.inertia_second_derivative) * pair_phase,
    )


def test_translation_and_proper_frame_covariance_preserve_geometric_tensors(
    genuine_modes,
):
    _, harmonic, rotor, context, native, _, _ = genuine_modes
    original = build_rovibrational_precursors(harmonic, rotor, context)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.2, -2.0, 0.4])
    coordinates = harmonic.coordinates_bohr @ rotation.T + translation

    def rotate_modes(array):
        return np.einsum(
            "ij,ajk->aik", rotation, array.reshape(len(coordinates), 3, -1)
        ).reshape(array.shape)

    block = np.kron(np.eye(len(coordinates)), rotation)
    rotated_hessian = block @ np.asarray(native["hessian_hartree_bohr2"]) @ block.T
    changed = replace(
        harmonic,
        coordinates_bohr=coordinates,
        mass_weighted_modes=rotate_modes(harmonic.mass_weighted_modes),
        cartesian_modes=rotate_modes(harmonic.cartesian_modes),
        dimensionless_to_cartesian=rotate_modes(harmonic.dimensionless_to_cartesian),
        source_digest=artifact_digest(
            {
                "geometry_bohr": coordinates,
                "masses_u": harmonic.isotope_masses_u,
                "hessian_hartree_bohr2": rotated_hessian,
            }
        ),
    )
    changed_rotor = replace(
        rotor,
        principal_axes_columns=rotation @ rotor.principal_axes_columns,
        geometry_digest=artifact_digest(
            {"geometry_bohr": coordinates, "masses_u": harmonic.isotope_masses_u}
        ),
    )
    changed_context = _derived_context(
        genuine_modes,
        changed,
        changed_rotor,
        {"proper_rotation": rotation.tolist(), "translation": translation.tolist()},
    )
    value = build_rovibrational_precursors(changed, changed_rotor, changed_context)
    for name in (
        "coriolis_zeta",
        "inertia_first_derivative",
        "inertia_second_derivative",
        "inertia_electron_mass_bohr2",
    ):
        assert np.allclose(
            getattr(value, name), getattr(original, name), rtol=1e-12, atol=1e-10
        )
    assert value.identity.frame_sha256 != original.identity.frame_sha256
    assert value.identity.geometry_sha256 != original.identity.geometry_sha256


def test_precursor_payload_and_parent_copy_are_deeply_immutable(genuine_modes):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    copy = ScientificContext.model_validate(context.model_dump(mode="python"))
    value = build_rovibrational_precursors(harmonic, rotor, copy)
    with pytest.raises(ValidationError):
        value.identification_ready = True
    with pytest.raises(TypeError):
        value.coriolis_zeta[0][0][0] = 1.0
    with pytest.raises(ValidationError):
        value.identity.recipe_sha256 = artifact_digest({"corrupted": True})
    original_mass = value.identity.isotope_masses_u[0]
    copy.isotope_masses_u[0] *= 2
    copy.parent_artifact_sha256.append(artifact_digest({"unrelated": True}))
    assert value.identity.isotope_masses_u[0] == original_mass
    assert len(value.identity.parent_artifact_sha256) == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "mode_norm",
        "cartesian_units",
        "dimensionless_units",
        "frequency",
        "external_rank",
        "external_hessian",
        "complex",
        "translation",
        "identity",
    ],
)
def test_corrupt_input_copy_fails_instead_of_repairing_modes(genuine_modes, corruption):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    if corruption == "mode_norm":
        harmonic = replace(
            harmonic, mass_weighted_modes=harmonic.mass_weighted_modes * 2
        )
    elif corruption == "cartesian_units":
        harmonic = replace(
            harmonic,
            cartesian_modes=harmonic.cartesian_modes * np.sqrt(ATOMIC_MASS_ELECTRON),
        )
    elif corruption == "dimensionless_units":
        harmonic = replace(
            harmonic, dimensionless_to_cartesian=harmonic.cartesian_modes
        )
    elif corruption == "frequency":
        harmonic = replace(harmonic, frequencies_cm1=harmonic.frequencies_cm1 * 2)
    elif corruption == "external_rank":
        harmonic = replace(harmonic, external_rank=3)
    elif corruption == "external_hessian":
        harmonic = replace(harmonic, external_residual_relative=1.0)
    elif corruption == "complex":
        harmonic = replace(
            harmonic,
            mass_weighted_modes=harmonic.mass_weighted_modes.astype(complex) + 1j,
        )
    elif corruption == "translation":
        contaminated = harmonic.mass_weighted_modes.copy()
        contaminated[:, 0] += np.tile([1.0, 0.0, 0.0], len(harmonic.coordinates_bohr))
        harmonic = replace(harmonic, mass_weighted_modes=contaminated)
    else:
        rotor = replace(
            rotor, geometry_digest=artifact_digest({"unrelated_geometry": True})
        )
    with pytest.raises(ValueError):
        build_rovibrational_precursors(harmonic, rotor, context)


def test_reflected_axes_are_rejected_without_guessing_reorientation(genuine_modes):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    reflected = rotor.principal_axes_columns.copy()
    reflected[:, 2] *= -1
    with pytest.raises(ValueError):
        build_rovibrational_precursors(
            harmonic, replace(rotor, principal_axes_columns=reflected), context
        )


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0, True, 0.0])
def test_invalid_orthogonality_policy_is_rejected(genuine_modes, tolerance):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    with pytest.raises(ValueError):
        build_rovibrational_precursors(
            harmonic, rotor, context, orthogonality_tolerance=tolerance
        )


def test_declared_degeneracy_threshold_controls_flags_without_changing_values(
    genuine_modes,
):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    narrow = build_rovibrational_precursors(
        harmonic, rotor, context, mode_degeneracy_relative_tolerance=0.0
    )
    broad = build_rovibrational_precursors(
        harmonic, rotor, context, mode_degeneracy_relative_tolerance=1.0
    )
    count = len(harmonic.frequencies_cm1)
    assert broad.degenerate_mode_pairs == tuple(
        (first, second) for first in range(count) for second in range(first + 1, count)
    )
    assert broad.coriolis_zeta == narrow.coriolis_zeta
    assert broad.inertia_second_derivative == narrow.inertia_second_derivative
    assert not broad.independent_scientific_qualification


@pytest.mark.parametrize(
    "corruption",
    [
        "zeta",
        "first",
        "second",
        "qualified",
        "units",
        "mode_flags",
        "frame",
        "inertia_flags",
        "zero_axes",
        "quality_flags",
        "center",
        "harmonic_residual",
        "atom_ids",
        "input_modes",
        "input_geometry",
        "principal_moments",
        "residuals",
        "parents",
        "isotope_policy",
        "frequency_binding",
        "harmonic_source",
    ],
)
def test_readback_rejects_unrelated_tensors_or_claims(genuine_modes, corruption):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    value = build_rovibrational_precursors(harmonic, rotor, context)
    changed = json.loads(value.model_dump_json())
    if corruption == "zeta":
        changed["coriolis_zeta"][0][0][0] += 1.0
    elif corruption == "first":
        changed["inertia_first_derivative"][0][0][0] += 1.0
    elif corruption == "second":
        changed["inertia_second_derivative"][0][0][0][0] += 1.0
    elif corruption == "qualified":
        changed["independent_scientific_qualification"] = True
    elif corruption == "units":
        changed["normal_coordinate_unit"] = "dimensionless"
    elif corruption == "mode_flags":
        changed["degenerate_mode_pairs"] = [[0, 0]]
    elif corruption == "frame":
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        changed["principal_axes_columns"] = (
            rotation @ np.asarray(changed["principal_axes_columns"])
        ).tolist()
    elif corruption == "inertia_flags":
        changed["degenerate_inertia_axis_pairs"] = [[0, 0]]
    elif corruption == "zero_axes":
        changed["zero_inertia_axes"] = [] if changed["zero_inertia_axes"] else [0]
    elif corruption == "quality_flags":
        changed["quality_flags"] = []
    elif corruption == "center":
        changed["center_of_mass_bohr"] = [1.0, 2.0, 3.0]
    elif corruption == "harmonic_residual":
        changed["harmonic_external_residual"] = 1000.0
    elif corruption == "atom_ids":
        changed["identity"]["atom_ids"][1] = changed["identity"]["atom_ids"][0]
    elif corruption == "input_modes":
        changed["input_mass_weighted_modes"][0][0] += 1e-5
    elif corruption == "input_geometry":
        changed["input_geometry_bohr"][0][0] += 1e-5
    elif corruption == "principal_moments":
        changed["principal_moments_electron_mass_bohr2"][0] += 1.0
    elif corruption == "residuals":
        changed["mode_orthonormality_residual"] = 1.0
    elif corruption == "parents":
        changed["identity"]["parent_artifact_sha256"] = []
    elif corruption == "frequency_binding":
        changed["harmonic_frequencies_cm1"] = [
            2 * x for x in changed["harmonic_frequencies_cm1"]
        ]
    elif corruption == "harmonic_source":
        changed["identity"]["harmonic_source_sha256"] = artifact_digest(
            {"unrelated_native_hessian": True}
        )
    else:
        changed["identity"]["isotope_selection_policies"][0] = "invented_isotope_policy"
    with pytest.raises(ValidationError):
        RovibrationalPrecursors.model_validate_json(json.dumps(changed))


def test_numerically_small_moment_cannot_erase_upstream_linear_zero_axis(genuine_modes):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    value = build_rovibrational_precursors(harmonic, rotor, context)
    if value.zero_inertia_axes:
        changed = json.loads(value.model_dump_json())
        changed["principal_moments_electron_mass_bohr2"][0] = 1e-13
        changed["zero_inertia_axes"] = []
        changed["quality_flags"].remove(
            "zero_inertia_axis_no_inversion_or_finite_rotational_constant"
        )
        with pytest.raises(ValidationError):
            RovibrationalPrecursors.model_validate_json(json.dumps(changed))
    else:
        assert all(moment > 0 for moment in value.principal_moments_electron_mass_bohr2)
        assert (
            RovibrationalPrecursors.model_validate_json(value.model_dump_json())
            == value
        )


def test_actual_isotope_reanalysis_recomputes_com_frame_and_modes(genuine_modes):
    from Libraries.cochem_isotopes import isotope_record

    _, harmonic, rotor, context, native, isotopes, backend_request = genuine_modes
    original = build_rovibrational_precursors(harmonic, rotor, context)
    # An actual native electronic Hessian is reused under the declared BO model.
    # Its unchanged parent is preserved; isotope labels are not engine outputs.
    changed_molecule = dict(backend_request["molecule"])
    changed_molecule["isotopes"] = [None] * (len(isotopes) - 1) + [2]
    changed_records = [*isotopes[:-1], isotope_record("2H")]
    masses = [record["mass_u"] for record in changed_records]
    changed = analyze_hessian(
        harmonic.coordinates_bohr, masses, native["hessian_hartree_bohr2"]
    )
    changed_rotor = equilibrium_rotor(changed.coordinates_bohr, masses)
    changed_context = ScientificContext.model_validate(
        make_scientific_context(
            molecule=changed_molecule,
            geometry_bohr=changed.coordinates_bohr,
            isotope_provenance=changed_records,
            recipe_sha256=context.recipe_sha256,
            protocol_sha256=artifact_digest(
                {
                    "approximation": "Born-Oppenheimer native Cartesian Hessian reuse",
                    "requested_isotopes": changed_molecule["isotopes"],
                }
            ),
            parent_artifact_sha256=context.parent_artifact_sha256,
            evidence_class="mathematical_model",
            harmonic=changed,
            principal_axes_columns=changed_rotor.principal_axes_columns,
        )
    )
    value = build_rovibrational_precursors(changed, changed_rotor, changed_context)
    assert value.identity.input_evidence_class == "mathematical_model"
    assert value.identity.isotope_numbers[-1] == 2
    assert value.identity.isotope_masses_u[-1] == changed_records[-1]["mass_u"]
    assert (
        value.identity.parent_artifact_sha256
        == original.identity.parent_artifact_sha256
    )
    assert value.identity.mode_basis_sha256 != original.identity.mode_basis_sha256
    assert (
        np.linalg.norm(
            np.asarray(value.center_of_mass_bohr) - original.center_of_mass_bohr
        )
        > 1e-4
    )
    assert (
        value.identity.equilibrium_rotor_geometry_sha256
        != original.identity.equilibrium_rotor_geometry_sha256
    )
    assert not value.independent_scientific_qualification


def test_placeholder_source_identity_cannot_become_precursor_provenance(genuine_modes):
    _, harmonic, rotor, context, _, _, _ = genuine_modes
    with pytest.raises(ValueError, match="Missing source"):
        build_rovibrational_precursors(
            replace(harmonic, source_digest="0" * 64), rotor, context
        )
