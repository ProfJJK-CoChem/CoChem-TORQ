"""Declared geometry/state mathematics and honest legacy identity boundaries.

These examples contain no native engine replies or fabricated molecular energies.
The existing CREST suite separately checks retention with genuine HF observations.
"""

import h5py
import numpy as np
import pytest

from Libraries.cochem_torq_crest import (
    GAS_CONSTANT_R_CAL_MOL,
    ConformerRecord,
    CregenReferee,
    EnsembleContainer,
    UnionAuditReport,
    calculate_conformational_entropy,
    compute_rmsd,
    deduplicate_spectroscopic,
    kabsch_align,
    save_ensemble_to_hdf5,
)


def input_records():
    return [
        ConformerRecord(
            index=index,
            symbols=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]],
            origin_engine=origin,
            metadata={"input_only": True},
        )
        for index, origin in [(41, "DECLARED_INPUT"), (73, "DECLARED_INPUT_REPEAT")]
    ]


def unavailable_report(**changes):
    return UnionAuditReport(
        system_name="declared-input serialization check",
        n_seeds=0,
        n_goat_raw=0,
        n_crest_raw=0,
        n_union_raw=0,
        n_survivors_stage_a=0,
        n_survivors_stage_b=0,
        provenance_tag="serialization_only_no_engine_execution",
        **changes,
    )


def test_missing_identity_or_energy_retains_original_indices_and_origins():
    records = input_records()
    for retained in (
        CregenReferee().referee_ensemble(records),
        CregenReferee().referee_ensemble(records, is_spectroscopic_stage=True),
        deduplicate_spectroscopic(records),
    ):
        assert all(left is right for left, right in zip(retained, records))
        assert len(retained) == len(records)
        assert [item.index for item in retained] == [41, 73]
        assert [item.origin_engine for item in retained] == [
            "DECLARED_INPUT",
            "DECLARED_INPUT_REPEAT",
        ]
        assert all(item.energy_hartree is None for item in retained)


def test_empty_record_set_does_not_invent_thermodynamic_observations():
    ensemble = EnsembleContainer(name="no molecular observations", conformers=[])
    report = unavailable_report()
    assert ensemble.s_conf_cal_mol_k is None
    assert ensemble.boltzmann_weights == []
    assert ensemble.thermodynamic_unavailability_reason
    assert report.n_goat_unique is None
    assert report.n_crest_unique is None
    assert report.n_shared_intersection is None
    assert report.union_coverage_ratio is None
    assert report.identity_unavailability_reason


@pytest.mark.parametrize(
    "name",
    [
        "n_goat_unique",
        "n_crest_unique",
        "n_shared_intersection",
        "union_coverage_ratio",
        "s_conf_goat_cal_mol_k",
        "s_conf_crest_cal_mol_k",
        "s_conf_union_cal_mol_k",
    ],
)
def test_unqualified_identity_metadata_cannot_be_forged_as_zero(name):
    with pytest.raises(ValueError, match="unqualified"):
        unavailable_report(**{name: 0})


@pytest.mark.parametrize(
    "changes", [{"s_conf_cal_mol_k": 0}, {"boltzmann_weights": [1]}]
)
def test_legacy_ensemble_cannot_assert_populations_without_qualified_states(changes):
    with pytest.raises(ValueError, match="verified state identity"):
        EnsembleContainer(name="invalid metadata control", conformers=[], **changes)


def test_hdf5_preserves_unavailability_reasons_and_original_input_identity(tmp_path):
    records = input_records()
    ensemble = EnsembleContainer(name="declared geometry inputs", conformers=records)
    path = save_ensemble_to_hdf5(ensemble, unavailable_report(), tmp_path / "input.h5")
    with h5py.File(path, "r") as archive:
        assert (
            archive.attrs["thermodynamic_state_status"]
            == ensemble.thermodynamic_state_status
        )
        assert archive.attrs["thermodynamic_unavailability_reason"]
        assert archive.attrs["identity_status"].startswith("unavailable")
        assert archive.attrs["identity_unavailability_reason"]
        assert "s_conf_union_cal_mol_k" not in archive.attrs
        assert "union_coverage_ratio" not in archive["union_audit"].attrs
        assert "n_shared_intersection" not in archive["union_audit"].attrs
        assert archive["conformers/conformer_0000"].attrs["index"] == 41
        assert archive["conformers/conformer_0001"].attrs["index"] == 73
        assert (
            archive["conformers/conformer_0001"].attrs["origin_engine"]
            == "DECLARED_INPUT_REPEAT"
        )
        assert "energy_hartree" not in archive["conformers/conformer_0000"].attrs


@pytest.mark.parametrize(
    "target",
    [np.empty((0, 3)), np.zeros((1, 3)), np.array([[0, 0, np.nan], [0, 0, 1]])],
)
def test_invalid_rmsd_geometry_is_unavailable_instead_of_zero(target):
    with pytest.raises(ValueError):
        compute_rmsd(np.array([[0, 0, 0], [0, 0, 1]]), target)


@pytest.mark.parametrize("weights", [[0, 0], [1], [1, -1], [1, np.inf]])
def test_invalid_rmsd_weights_are_rejected(weights):
    coordinates = np.array([[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="positive weight"):
        kabsch_align(coordinates, coordinates, masses=weights)


def test_proper_rotation_and_translation_preserve_weighted_mathematical_rmsd():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = source @ rotation + [2, 3, 4]
    rmsd, aligned = kabsch_align(source, target, masses=np.array([1, 2, 3, 4]))
    assert rmsd < 1e-12
    assert np.allclose(aligned, target, atol=1e-12)
    reflected = source * [-1, 1, 1]
    assert compute_rmsd(source, reflected) > 0.1


def test_declared_state_math_does_not_floor_underflowed_probabilities():
    entropy, weights = calculate_conformational_entropy([0, 1e6], temperature_k=1)
    assert weights == [1.0, 0.0]
    assert entropy == 0.0
    entropy, weights = calculate_conformational_entropy([0, 0])
    assert weights == [0.5, 0.5]
    assert entropy == pytest.approx(GAS_CONSTANT_R_CAL_MOL * np.log(2))


@pytest.mark.parametrize("energies", [[], [np.nan], [np.inf], [[0, 1]]])
def test_invalid_declared_state_math_cannot_invent_uniform_populations(energies):
    with pytest.raises(ValueError):
        calculate_conformational_entropy(energies)
