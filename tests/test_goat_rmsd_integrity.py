"""Declared mathematical point clouds, never fabricated engine observations."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_goat import (
    GAS_CONSTANT_R_CAL_MOL,
    ConformerRecord,
    EnsembleContainer,
    calculate_conformational_entropy,
    compute_rmsd,
    deduplicate_conformers,
    deduplicate_stage_a,
    deduplicate_stage_b_spectroscopic,
    kabsch_align,
    save_ensemble_to_hdf5,
)


def point_cloud():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )


@pytest.mark.parametrize(
    "source,target",
    (
        (np.empty((0, 3)), np.empty((0, 3))),
        (np.zeros((2, 3)), np.zeros((3, 3))),
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.zeros(3), np.zeros(3)),
        (np.array([[np.nan, 0.0, 0.0]]), np.zeros((1, 3))),
        (np.zeros((1, 3)), np.array([[0.0, np.inf, 0.0]])),
        (np.array([[1.0j, 0.0, 0.0]]), np.zeros((1, 3))),
    ),
)
def test_invalid_coordinate_comparison_never_becomes_zero(source, target):
    with pytest.raises(ValueError):
        compute_rmsd(source, target)
    with pytest.raises(ValueError):
        kabsch_align(source, target)


def test_finite_inputs_with_unrepresentable_intermediates_fail_explicitly():
    coordinates = np.full((3, 3), 1e308)
    with pytest.raises(ValueError, match="finite numerical domain"):
        compute_rmsd(coordinates, coordinates)


@pytest.mark.parametrize(
    "labels",
    (
        {"symbols": ["C", "N", "O", "F"]},
        {"symbols2": ["C", "N", "O", "F"]},
        {"symbols": ["C"], "symbols2": ["C"]},
        {"symbols": ["C", "N", "O", "F"], "symbols2": ["C", "N", "N", "F"]},
        {"symbols": ["C", "N", "", "F"], "symbols2": ["C", "N", "", "F"]},
    ),
)
def test_missing_or_conflicting_atom_labels_are_not_unrestricted_matches(labels):
    with pytest.raises(ValueError):
        compute_rmsd(point_cloud(), point_cloud(), **labels)


def test_proper_rotation_translation_and_exact_unique_label_correspondence():
    source = point_cloud()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = source @ rotation + np.array([5.0, -3.0, 2.0])
    assert compute_rmsd(source, target) < 1e-12
    labels = ["C", "N", "O", "F"]
    order = [2, 0, 3, 1]
    reordered_target = target[order]
    target_labels = [labels[index] for index in order]
    assert (
        compute_rmsd(source, reordered_target, symbols=labels, symbols2=target_labels)
        < 1e-12
    )
    assert (
        compute_rmsd(source, reordered_target, symbols=(labels, target_labels)) < 1e-12
    )
    assert compute_rmsd(source, reordered_target) > 0.1


def test_mirror_cannot_use_improper_rotation_or_label_exchange():
    source = point_cloud()
    mirror = source * np.array([-1.0, 1.0, 1.0])
    labels = ["C", "N", "O", "F"]
    assert compute_rmsd(source, mirror) > 0.1
    assert compute_rmsd(source, mirror, symbols=labels, symbols2=labels) > 0.1
    aligned = kabsch_align(source, mirror)
    transformation = np.linalg.lstsq(
        source - source.mean(0), aligned - aligned.mean(0), rcond=None
    )[0]
    assert np.linalg.det(transformation) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "screen",
    (deduplicate_conformers, deduplicate_stage_a, deduplicate_stage_b_spectroscopic),
)
def test_identical_formula_geometry_without_graph_identity_keeps_originals(screen):
    records = [
        ConformerRecord(
            index=index,
            symbols=["C", "N", "O", "F"],
            coordinates=point_cloud().tolist(),
            origin_engine="declared_mathematical_input",
            provenance_tag="input_only_no_engine_observation",
            metadata={"test_scope": "unresolved graph/stereo/state identity"},
        )
        for index in (7, 19)
    ]
    original = [record.model_dump() for record in records]
    retained = screen(records)
    assert len(retained) == 2
    assert all(after is before for after, before in zip(retained, records))
    assert [record.model_dump() for record in retained] == original
    assert all(record.energy_hartree is None for record in retained)


@pytest.mark.parametrize(
    "screen",
    (deduplicate_conformers, deduplicate_stage_a, deduplicate_stage_b_spectroscopic),
)
def test_empty_candidate_input_is_rejected_before_selection(screen):
    record = ConformerRecord(
        index=1,
        symbols=[],
        coordinates=[],
        origin_engine="declared_input",
        provenance_tag="input_only",
    )
    with pytest.raises(ValueError):
        screen([record])


def test_unresolved_candidate_pool_preserves_unavailable_thermodynamics_in_archive(
    tmp_path,
):
    record = ConformerRecord(
        index=4,
        symbols=["C", "N", "O", "F"],
        coordinates=point_cloud().tolist(),
        origin_engine="declared_mathematical_input",
        provenance_tag="input_only",
    )
    ensemble = EnsembleContainer(
        name="Declared candidates, not verified distinct thermodynamic states",
        conformers=[record],
        s_conf_cal_mol_k=None,
        boltzmann_weights=[],
        thermodynamic_state_status="not_established",
        thermodynamic_unavailability_reason="Graph/stereo/state identity unresolved",
        provenance_tag="input_only",
    )
    destination = tmp_path / "unresolved-pool.h5"
    save_ensemble_to_hdf5(ensemble, destination)
    with h5py.File(destination) as archive:
        assert "s_conf_cal_mol_k" not in archive.attrs
        assert "boltzmann_weights" not in archive["conformers"]
        assert archive.attrs["thermodynamic_state_status"] == "not_established"
        assert (
            archive.attrs["thermodynamic_unavailability_reason"]
            == ensemble.thermodynamic_unavailability_reason
        )


@pytest.mark.parametrize(
    "energies,temperature,counts",
    (
        ([], 298.15, None),
        ([np.nan], 298.15, None),
        ([np.inf], 298.15, None),
        ([0.0], 0.0, None),
        ([0.0], -1.0, None),
        ([0.0], np.inf, None),
        ([0.0], 298.15, [0]),
        ([0.0], 298.15, [-1]),
        ([0.0], 298.15, [np.nan]),
        ([0.0], 298.15, [1.5]),
        ([0.0, 1.0], 298.15, [1]),
    ),
)
def test_invalid_declared_states_never_receive_uniform_or_floored_populations(
    energies, temperature, counts
):
    with pytest.raises(ValueError):
        calculate_conformational_entropy(energies, temperature, counts)


def test_declared_microstate_degeneracy_contributes_actual_entropy():
    entropy, probabilities = calculate_conformational_entropy([0.0], degeneracies=[2])
    assert probabilities == [1.0]
    assert entropy == pytest.approx(GAS_CONSTANT_R_CAL_MOL * np.log(2.0))
    entropy, probabilities = calculate_conformational_entropy(
        [0.0, 0.0], degeneracies=[2, 3]
    )
    assert probabilities == pytest.approx([0.4, 0.6])
    assert entropy == pytest.approx(GAS_CONSTANT_R_CAL_MOL * np.log(5.0))


def test_declared_thermal_underflow_remains_zero_not_a_positive_floor():
    entropy, probabilities = calculate_conformational_entropy([0.0, 1e6])
    assert probabilities == [1.0, 0.0]
    assert entropy == 0.0
