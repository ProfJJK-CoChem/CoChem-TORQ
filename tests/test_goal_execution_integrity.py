"""Genuine HF calculations retain goals without inventing accuracy evidence."""

from copy import deepcopy

import pytest

from cochem_torq.application import execute_request
from cochem_torq.artifacts import seal_shard, verify_shard
from cochem_torq.domain import canonical_json, read_json
from cochem_torq.scientific_contracts import ProductDeclaration, ScientificGoal
from cochem_torq.service import approve_plan, plan_request

pytestmark = pytest.mark.real_engine


def test_actual_goal_calculation_retains_declaration_and_unqualified_status(tmp_path):
    request = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0, 0, 0], [0, 0, 1.4]],
            "charge": 0,
            "multiplicity": 1,
        },
        "recipe": "hf-sto-3g-education",
        "products": ["geometry"],
        "resources": {"cores": 1, "memory_mb": 2048, "wall_seconds": 120},
        "scientific_goal": ScientificGoal(
            products=(
                ProductDeclaration(product_class="A", observables=("geometry",)),
            ),
            chemical_domain="closed-shell hydrogen at a teaching HF/STO-3G level",
            allowed_approximations=("Born-Oppenheimer geometry at HF/STO-3G",),
            target_observables=("geometry",),
            reference_policy="de_novo_no_calibration",
            unknown_achievable_accuracy_reason=(
                "No independent molecular benchmark has established accuracy."
            ),
        ).model_dump(mode="json"),
    }
    review = plan_request(request, execution="local_validation")
    approved = approve_plan(review, actor="explicit genuine HF goal check")
    directory = tmp_path / "actual-goal-hf"
    result = execute_request(
        review["plan"]["request"], directory, approved_plan=approved
    )
    assert result["status"] == "partial", result["errors"]
    assert result["stages"]["equilibrium_geometry"]["status"] == "available"
    assert result["scientific_goal"] == request["scientific_goal"]
    assert result["goal_evaluation"]["accuracy_status"] == "unqualified"
    assert not result["experimental_accuracy_established"]
    manifest = verify_shard(directory)

    # Re-sealing a deliberately changed copy cannot reassign the genuine result.
    import shutil

    changed = tmp_path / "changed-goal"
    shutil.copytree(directory, changed)
    (changed / "manifest.json").unlink()
    payload = deepcopy(read_json(changed / "result.json"))
    payload["scientific_goal"]["chemical_domain"] = "changed declaration rejection"
    (changed / "result.json").write_bytes(canonical_json(payload))
    seal_shard(
        changed,
        request_sha256=manifest["request_sha256"],
        request_id=manifest["request_id"],
        recipe_sha256=manifest["recipe_sha256"],
        source_identity=manifest["source_identity"],
        worker_id=manifest["worker_id"],
    )
    with pytest.raises(ValueError, match="Scientific goal"):
        verify_shard(changed)
