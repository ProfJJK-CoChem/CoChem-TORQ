"""Actual rejected requests, failed records and filesystem identity checks.

No successful engine observations are created here. Failure records come from the
real application rejecting an unqualified profile. Deliberately corrupted copies
test that a valid byte inventory cannot conceal mismatched scientific identities.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cochem_torq.application import validate_request, worker_execute
from cochem_torq.artifacts import seal_shard, verify_shard
from cochem_torq.domain import CalculationRequest, canonical_json


def blocked_request() -> CalculationRequest:
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0, 0, 0], [0, 0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
            },
            "recipe": "revdsd-pbep86-d4-experimental",
            "products": ["geometry"],
        }
    )


def actual_failure_shard(directory: Path, corruption: str | None = None) -> Path:
    directory.mkdir()
    request = blocked_request()
    checked = validate_request(request)
    result = worker_execute(request, directory)
    assert result["status"] == "failed"
    source_identity = deepcopy(result["source_identity"])
    if corruption == "request":
        result["request_sha256"] = "0" * 64
    elif corruption == "source":
        result["source_identity"]["code_sha256"] = "0" * 64
    elif corruption == "recipe":
        result["recipe"]["basis"] = "sto-3g"
    (directory / "request.json").write_bytes(canonical_json(checked["request"]))
    (directory / "result.json").write_bytes(canonical_json(result))
    seal_shard(
        directory,
        request_sha256=checked["request_sha256"],
        request_id=str(request.request_id),
        recipe_sha256=checked["recipe"]["recipe_sha256"],
        source_identity=source_identity,
        worker_id="rejected-profile-contract-check",
    )
    return directory


def test_unqualified_recipe_rejection_has_no_substitute_values_or_engine_files(
    tmp_path: Path,
) -> None:
    result = worker_execute(blocked_request(), tmp_path)
    assert result["status"] == "failed"
    assert not result["identification_ready"]
    assert not result["experimental_accuracy_established"]
    assert result["errors"]
    assert all(
        stage["status"] == "blocked" and stage["value"] is None and stage["reason"]
        for stage in result["stages"].values()
    )
    assert not (tmp_path / "engine").exists()


def test_genuine_rejected_profile_shard_is_integrity_verifiable(tmp_path: Path) -> None:
    directory = actual_failure_shard(tmp_path / "rejected")
    manifest = verify_shard(directory)
    assert {item["path"] for item in manifest["files"]} == {
        "request.json",
        "result.json",
    }


@pytest.mark.parametrize("corruption", ["request", "source", "recipe"])
def test_sealed_but_semantically_mismatched_failed_result_is_rejected(
    tmp_path: Path, corruption: str
) -> None:
    with pytest.raises(ValueError):
        directory = actual_failure_shard(tmp_path / corruption, corruption)
        verify_shard(directory)


def test_invalid_isotope_is_rejected_before_any_engine_work() -> None:
    request = {
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry_bohr": [[0, 0, 0], [1, 0, 1], [-1, 0, 1]],
            "charge": 0,
            "multiplicity": 1,
            "isotopes": [1, None, None],
        },
        "recipe": "hf-sto-3g-education",
        "products": ["geometry"],
    }
    with pytest.raises(ValueError):
        validate_request(request)


def test_requested_equilibrium_constants_require_minimum_characterization() -> None:
    request = blocked_request().model_dump(mode="json")
    request["recipe"] = "hf-sto-3g-education"
    request["products"] = ["equilibrium_constants"]
    report = validate_request(request)
    assert any(
        task["operation"] == "projected_hessian_analysis"
        for task in report["plan"]["tasks"]
    )
