"""Integrity tests using actual numerical code and missing-resource/input failures.

Analytic model potentials below test calculus, not quantum-engine or experimental
results. No engine output, calculator, dependency, or service is mocked.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from Libraries.cochem_torq_active_learning import (
    ActiveLearningSampler, compute_qbc_energy_variance, compute_max_force_epistemic_std,
)
from Libraries.cochem_torq_inference_errors import ActiveLearningSelectionError, BaselineExecutionError, PhysicsDivergenceError
from Libraries.cochem_torq_delta_ml import DeltaMLEngine, GFN2Result, GFN2xTBEngine, LennardJonesBaselineEngine
from Libraries.cochem_torq_crest import (
    CregenReferee, CrestConfig, CrestRunner, calculate_conformational_entropy, compute_moments_and_constants, parse_xyz_string,
)
from Libraries.cochem_torq_mace import TorqMACETriage, compute_pes_derivatives, evaluate_physical_potential
from Libraries.cochem_torq_masses import get_monoisotopic_mass, resolve_ciaaw_monoisotopic_mass
from Libraries.cochem_torq_quench import ConformalMDQuencher, format_to_qcschema_v1
from Libraries.cochem_torq_vault import (
    MissingDataError, fetch_topos_matrices, get_atomic_mass, get_atomic_number,
    poll_isomer_wavefunctions, standardize_geometry_dataframe,
)
from Libraries.cochem_torq_vibrational import resolve_harmonic_zpe_scale_factor


def test_anchor_missing_does_not_invent_energy():
    with pytest.raises(ActiveLearningSelectionError, match="no anchor evaluator"):
        ActiveLearningSampler().evaluate_configuration([[0, 0, 0]], -1.0, 20.0)


def test_delta_model_missing_does_not_assume_zero_correction():
    with pytest.raises(ActiveLearningSelectionError, match="trained predictor"):
        ActiveLearningSampler().evaluate_configuration([[0, 0, 0]], -1.0, 1.0)


def test_missing_committee_uncertainty_is_not_zero():
    candidate = {"coordinates": [[0, 0, 0]], "energy_hartree": -1.0}
    original = dict(candidate)
    with pytest.raises(ActiveLearningSelectionError, match="uncertainty"):
        ActiveLearningSampler().sample_pes_grid([candidate])
    assert candidate == original


@pytest.mark.parametrize("energies", [[0.0, float("nan")], [0.0, float("inf")]])
def test_nonfinite_committee_predictions_are_rejected(energies):
    with pytest.raises(ActiveLearningSelectionError, match="finite"):
        compute_qbc_energy_variance(energies)


def test_nonfinite_committee_forces_are_rejected():
    with pytest.raises(ActiveLearningSelectionError, match="finite"):
        compute_max_force_epistemic_std(np.full((2, 1, 3), np.nan))


def test_delta_baseline_must_be_explicit():
    with pytest.raises(ValueError, match="explicitly configured"):
        DeltaMLEngine()


def test_delta_forward_requires_predictor_even_with_real_baseline():
    engine = DeltaMLEngine(baseline_engine=LennardJonesBaselineEngine({18: (1.0, 1.0)}))
    with pytest.raises(BaselineExecutionError, match="trained delta predictor"):
        engine.forward(torch.tensor([[0., 0., 0.], [2., 0., 0.]], dtype=torch.float64), [18, 18])


def test_lennard_jones_parameters_are_never_guessed():
    with pytest.raises(BaselineExecutionError, match="parameters unavailable"):
        LennardJonesBaselineEngine().calculate(torch.tensor([[0., 0., 0.], [2., 0., 0.]]), [18, 18])


def test_explicit_analytic_pair_potential_matches_derivative():
    engine = LennardJonesBaselineEngine({18: (1.0, 1.0)})
    coords = torch.tensor([[0., 0., 0.], [1.4, 0., 0.]], dtype=torch.float64)
    energy, forces = engine.calculate(coords, [18, 18])
    assert energy == pytest.approx(4 * ((1/1.4)**12 - (1/1.4)**6))
    h = 1e-5
    plus, minus = coords.clone(), coords.clone()
    plus[1, 0] += h
    minus[1, 0] -= h
    fd_force = -(engine.calculate(plus, [18, 18])[0] - engine.calculate(minus, [18, 18])[0]) / (2*h)
    assert float(forces[1, 0]) == pytest.approx(fd_force, rel=1e-8)
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64))


def test_gfn2_result_cannot_default_energy_or_forces():
    with pytest.raises(TypeError):
        GFN2Result()
    with pytest.raises(ValueError, match="forces"):
        GFN2Result(energy_ev=0.0)


def test_gfn2_runs_actual_engine_or_reports_missing_resource():
    engine = GFN2xTBEngine()
    coords = torch.tensor([[0., 0., 0.], [0., 0., 0.74]], dtype=torch.float64)
    if engine.xtb_available:
        result = engine.calculate(coords, [1, 1])
        assert np.isfinite(result["energy_ev"])
        assert result["forces"].shape == coords.shape
    else:
        with pytest.raises(BaselineExecutionError, match="UNAVAILABLE"):
            engine.calculate(coords, [1, 1])


def test_no_implicit_mace_surrogate_potential():
    with pytest.raises(RuntimeError, match="No potential calculator"):
        evaluate_physical_potential(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])


def test_unsupported_model_does_not_become_emt(tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"symbols": ["H", "H"], "grid_points": []}))
    with pytest.raises(RuntimeError, match="Unsupported calculator"):
        TorqMACETriage(str(grid), model_name="unavailable-model")


def test_emt_only_runs_when_explicitly_selected(tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"symbols": ["H", "H"], "grid_points": []}))
    triage = TorqMACETriage(str(grid), model_name="EMT")
    energy, forces, _ = triage.evaluate_point([[0., 0., 0.], [0., 0., 0.74]])
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()


@pytest.mark.parametrize("angles,energies", [([0], [1]), ([0, 1], [0, 1]), ([0, 0, 1], [0, 0, 1])])
def test_missing_curvature_information_is_not_zero(angles, energies):
    with pytest.raises(ValueError):
        compute_pes_derivatives(angles, energies)


def test_real_finite_difference_code_matches_polynomial_derivatives():
    angles = np.array([0., 1., 3., 6.])
    grad, curvature = compute_pes_derivatives(angles, angles**2)
    np.testing.assert_allclose(grad, 2*angles, atol=1e-12)
    np.testing.assert_allclose(curvature, 2, atol=1e-12)


@pytest.mark.parametrize("symbol", ["Xx", "999C", "C-1"])
def test_unknown_isotope_never_becomes_carbon(symbol):
    with pytest.raises((KeyError, ValueError)):
        get_atomic_mass(symbol)


def test_unknown_atomic_number_never_becomes_carbon():
    with pytest.raises(KeyError):
        get_atomic_number("Xx")


def test_ambiguous_radioisotope_requires_explicit_mass_number():
    with pytest.raises(ValueError, match="Specify an isotope"):
        get_monoisotopic_mass(43)
    with pytest.raises(ValueError, match="Specify an isotope"):
        resolve_ciaaw_monoisotopic_mass(43)
    assert 98 < get_monoisotopic_mass(43, isotope_number=99) < 100


def test_geometry_rejects_mismatched_explicit_masses():
    with pytest.raises(ValueError, match="match atom count"):
        standardize_geometry_dataframe(["H", "H"], np.array([[0., 0., 0.], [0., 0., 0.74]]), masses=[1.0])


def test_geometry_only_hdf5_keeps_missing_energy_and_input(tmp_path):
    archive = tmp_path / "input.h5"
    with h5py.File(archive, "w") as handle:
        conf = handle.create_group("conformers/input_geometry")
        conf.create_dataset("coordinates", data=[[0., 0., 0.], [0., 0., 0.74]])
        conf.create_dataset("symbols", data=np.array(["H", "H"], dtype="S1"))
    before = archive.read_bytes()
    result = fetch_topos_matrices(archive)
    assert result["energy_hartree"] is None
    assert result["energy_status"] == "unavailable"
    assert poll_isomer_wavefunctions(archive)[0]["energy_hartree"] is None
    with pytest.raises(MissingDataError, match="absent"):
        fetch_topos_matrices(archive, "absent")
    with pytest.raises(MissingDataError, match="wavefunction"):
        poll_isomer_wavefunctions(archive, require_gbw=True)
    assert archive.read_bytes() == before


def test_crest_geometry_title_number_is_not_an_energy():
    records = parse_xyz_string("3\ninput geometry sample -99.5\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n")
    assert records[0].energy_hartree is None
    assert records[0].energy_kcal_rel is None
    assert records[0].origin_engine == "IMPORTED"
    original = records[0].model_dump(mode="json")
    retained = CregenReferee().referee_ensemble(records)
    assert len(retained) == len(records)
    assert retained[0] is records[0]
    assert retained[0].model_dump(mode="json") == original
    assert retained[0].energy_hartree is None
    assert retained[0].energy_kcal_rel is None


def test_crest_actual_missing_executable_does_not_create_result(tmp_path):
    seed = tmp_path / "seed.xyz"
    seed.write_text("3\ninput geometry\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n")
    runner = CrestRunner(CrestConfig(crest_bin=str(tmp_path / "absent-crest-executable")))
    with pytest.raises(FileNotFoundError, match="unavailable"):
        runner.run_crest(seed, work_dir=tmp_path / "work")
    assert seed.read_text().startswith("3\ninput geometry")


def test_linear_rotation_axis_is_unavailable_not_zero():
    result = compute_moments_and_constants(["H", "H"], np.array([[0., 0., 0.], [0., 0., 0.74]]))
    assert result["rotational_constants_mhz"][0] is None
    assert result["rotational_constants_mhz"][1] > 0
    assert result["ray_asymmetry_kappa"] is None


@pytest.mark.parametrize("energies", [[], [0., float("nan")]])
def test_missing_conformer_energies_do_not_imply_uniform_populations(energies):
    with pytest.raises(ValueError):
        calculate_conformational_entropy(energies)


def test_missing_physical_results_cannot_be_qcschema_success():
    with pytest.raises(ValueError, match="calculated finite energy"):
        format_to_qcschema_v1(["H", "H"], np.array([[0., 0., 0.], [0., 0., 0.74]]))


def test_quench_does_not_reuse_prediction_at_different_geometry():
    quencher = ConformalMDQuencher(check_interval=1)
    coords = np.array([[0., 0., 0.], [0., 0., 0.1]])
    result = quencher.step(0, ["H", "H"], coords, energy_pred=-1.0)
    assert result["action"] == "QUENCH_PROPOSAL_REQUIRES_EVALUATION"
    assert result["energy_hartree"] is None
    assert result["qcschema"] is None
    np.testing.assert_array_equal(coords, [[0., 0., 0.], [0., 0., 0.1]])


@pytest.mark.parametrize("method", [None, "unknown", "r2scan-3c", "b3lyp", "not-b3lyp"])
def test_method_name_does_not_invent_scaling_factor(method):
    assert resolve_harmonic_zpe_scale_factor(method)[0] == 1.0


def test_nonunit_scaling_requires_complete_calibration_provenance():
    with pytest.raises(PhysicsDivergenceError, match="provenance"):
        resolve_harmonic_zpe_scale_factor("b3lyp", 0.96)


def test_candidate_snapshot_preserves_null_and_rejects_overwrite(tmp_path):
    from Libraries.cochem_torq_active_learning import ActiveLearningHDF5Manager
    path = tmp_path / "pool.h5"
    manager = ActiveLearningHDF5Manager(path)
    record = {"candidate_id": "one", "coordinates": [[0., 0., 0.]], "atomic_numbers": [1], "energy": None, "quality": {"status": "pending"}}
    manager.append_candidate(record)
    before = path.read_bytes()
    stored = manager.read_candidates()[0]
    assert stored["energy"] is None
    assert stored["quality"] == {"status": "pending"}
    with pytest.raises(ValueError, match="already exists"):
        manager.append_candidate(record)
    assert path.read_bytes() == before


def test_candidate_snapshot_real_concurrent_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from Libraries.cochem_torq_active_learning import ActiveLearningHDF5Manager
    path = tmp_path / "pool.h5"
    def append(index):
        manager = ActiveLearningHDF5Manager(path)
        manager.append_candidate({"candidate_id": str(index), "coordinates": [[0., 0., 0.]], "atomic_numbers": [1], "energy": None})
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(append, range(8)))
    assert len(ActiveLearningHDF5Manager(path).read_candidates()) == 8
    assert not list(tmp_path.glob("*.tmp"))


def test_actual_harmonic_hessian_preserves_unscaled_frequency():
    from Libraries.cochem_torq_vibrational import analyze_vibrational_frequencies, CODATA_2022_FREQ_FACTOR
    coords = torch.tensor([[0., 0., 0.], [0., 0., 0.74]], dtype=torch.float64)
    def radial_harmonic_energy(positions):
        return 0.5 * (torch.linalg.norm(positions[1] - positions[0]) - 0.74)**2
    modes = analyze_vibrational_frequencies(coords, [1, 1], radial_harmonic_energy, method="analytic radial harmonic potential")
    assert len(modes.frequencies_cm1) == 1
    assert modes.frequencies_cm1[0] == pytest.approx(CODATA_2022_FREQ_FACTOR * np.sqrt(2/get_monoisotopic_mass(1)))
    assert modes.zpe_scale_factor == 1.0
    assert modes.scaled_frequencies_cm1 is None


def test_qcschema_units_from_real_analytic_potential():
    from scipy.constants import physical_constants
    engine = LennardJonesBaselineEngine({18: (1.0, 1.0)})
    coords = torch.tensor([[0., 0., 0.], [1.4, 0., 0.]], dtype=torch.float64)
    energy_ev, _ = engine.calculate(coords, [18, 18])
    energy_hartree = energy_ev / physical_constants["Hartree energy in eV"][0]
    result = format_to_qcschema_v1(["Ar", "Ar"], coords.numpy(), energy=energy_hartree,
        method="LennardJones(sigma=1 Angstrom, epsilon=1 eV)", molecular_charge=0, molecular_multiplicity=1,
        provenance={"creator": "Libraries.cochem_torq_delta_ml", "version": "source-checkout", "routine": "LennardJonesBaselineEngine.calculate"})
    assert result["molecule"]["geometry"][3] == pytest.approx(1.4/(physical_constants["Bohr radius"][0]*1e10))
    assert result["return_result"] == energy_hartree


@pytest.mark.parametrize("sigma,expected", [(1.0, "SURROGATE_PREDICT"), (20.0, "QUERY_ANCHOR")])
def test_advisory_candidate_gating(sigma, expected):
    from Libraries.cochem_torq_active_learning import ActiveLearner
    result = ActiveLearner.evaluate_candidate_gating(sigma)
    assert result["action"] == expected
    assert result["advisory_only"] is True
    assert result["eligible_for_pruning"] is False
    assert "energy_hartree" not in result
