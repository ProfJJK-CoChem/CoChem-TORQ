"""Exercise actual sampler and pipeline contracts without QC subprocess mocks."""

import numpy as np
import pytest

from Libraries.cochem_torq_active_learning import ActiveLearner, ActiveLearningSampler
from Libraries.cochem_torq_ml_policy import ScreeningBatch
from Libraries.cochem_torq_pipeline import TorqPipeline
from Libraries.torq_config import TorqRunParams
from tests import test_ml_policy as helpers

candidates = helpers.candidates
changed = helpers.changed
model = helpers.model
policy = helpers.policy
prediction = helpers.prediction


def grid(prediction=None, count=8, kind="conformer"):
    # Real-shaped H2 geometries for scheduling; no learned/QC physics is asserted.
    return [
        dict(
            candidate_id=f"grid-{i}",
            atomic_numbers=[1, 1],
            coordinates=np.array([[0, 0, 0], [0, 0, 0.74 + i * 0.01]]),
            scout_energy=-1.0,
            sigma_mev=1.0,
            kind=kind,
            **({"ml_prediction": prediction.model_dump()} if prediction else {}),
        )
        for i in range(count)
    ]


@pytest.mark.parametrize("sigma", [None, 0.0, float("nan"), float("inf"), -1.0])
def test_legacy_uncalibrated_gates_request_qc(sigma):
    assert ActiveLearner.evaluate_candidate_gating(sigma)["query_anchor"]


def test_calibrated_legacy_gate_requires_domain(prediction, policy):
    assert not ActiveLearner.evaluate_candidate_gating(
        1, prediction=prediction, policy=policy
    )["query_anchor"]
    assert ActiveLearner.evaluate_candidate_gating(
        20, prediction=prediction, policy=policy
    )["query_anchor"]
    unknown = changed(prediction, domain=changed(prediction.domain, status="unknown"))
    assert ActiveLearner.evaluate_candidate_gating(
        1, prediction=unknown, policy=policy
    )["query_anchor"]


def test_sampler_cannot_fabricate_qc_energy():
    result = ActiveLearningSampler().evaluate_configuration(
        [[0, 0, 0], [0, 0, 0.74]], -1.0, 100.0
    )
    assert result["action"] == "QC_PENDING"
    assert result["energy_hartree"] == -1.0
    assert result["provenance"] == "[E]"
    assert result["query_anchor"] and not result["qc_evaluated"]


def test_grid_missing_uncertainty_is_not_zero():
    inputs = grid(count=1)
    inputs[0].pop("sigma_mev")
    results = ActiveLearningSampler().sample_pes_grid(inputs)
    assert results[0]["sigma_mev"] is None
    assert results[0]["action"] == "QC_PENDING"
    assert results[0]["ml_decision"]["reason"] == "missing_prediction"


def test_grid_retains_pool_and_dispatches_sentinels(prediction, policy):
    calls = []

    def evaluate_energy(coordinates):
        calls.append(np.asarray(coordinates).copy())
        # Exercise the scalar evaluator boundary, not a QC exclusion claim.
        return -1.1

    inputs = grid(prediction)
    results = ActiveLearningSampler(policy=policy).sample_pes_grid(
        inputs, evaluate_energy
    )
    assert len(results) == len(inputs)
    sentinels = [r for r in results if r["ml_decision"]["sentinel"]]
    assert len(calls) == len(sentinels) == 2
    assert sum(r["action"] == "RETAIN_LOW_PRIORITY" for r in results) == 6
    assert all(
        np.array_equal(r["coordinates"], c["coordinates"])
        for r, c in zip(results, inputs)
    )
    assert all(r["qc_confirmation_required"] and r["qc_evaluated"] for r in sentinels)
    assert all(r["ml_decision"]["candidate"]["qc"] is None for r in results)


def test_legacy_committee_breach_cannot_be_deferred(prediction, policy):
    inputs = grid(prediction)
    for entry in inputs:
        entry["sigma_mev"] = 20.0
    results = ActiveLearningSampler(policy=policy).sample_pes_grid(inputs)
    assert all(r["action"] == "QC_PENDING" for r in results)
    assert all(r["ml_decision"]["action"] == "query_qc" for r in results)


def test_reaction_grid_cannot_use_surrogate(prediction, policy):
    results = ActiveLearningSampler(policy=policy).sample_pes_grid(
        grid(prediction, kind="reaction_path")
    )
    assert all(r["action"] == "QC_PENDING" for r in results)
    assert all(r["ml_decision"]["path_status"] == "unresolved" for r in results)


def test_pipeline_typed_screening_and_audit(prediction, policy, tmp_path):
    pipeline = TorqPipeline(
        TorqRunParams(
            tier="T1",
            wall_time_tier="T1-30min",
            engine="QC",
            method="test",
            basis_set="test",
            keywords=[],
        )
    )
    output = tmp_path / "audit.json"
    batch = pipeline.screen_ml_candidates(candidates(prediction), policy, output)
    assert ScreeningBatch.model_validate_json(output.read_text()) == batch
    with pytest.raises(FileExistsError):
        pipeline.screen_ml_candidates(candidates(prediction), policy, output)
    results = pipeline.run_active_learning_pes_sampling(grid(prediction), policy=policy)
    assert len(results) == 8


def test_nonfinite_scalar_anchor_fails_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        ActiveLearningSampler().evaluate_configuration(
            [], -1, 100, lambda _: float("nan")
        )
