"""Actual unrestricted worker, capability and immutable-shard contracts."""

from copy import deepcopy

import pytest

from cochem_torq.application import execute_request, validate_request, worker_execute
from cochem_torq.artifacts import verify_shard
from cochem_torq.domain import CalculationRequest, StageResult, digest
from cochem_torq.open_shell_service import OpenShellElectronicEnergy
from cochem_torq.registry import profile_capabilities, resolve_exact_capability


def _request(*, dft=False):
    return {
        "molecule": {
            "symbols": ["O", "O"] if dft else ["O", "H"],
            "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]],
            "charge": 0,
            "multiplicity": 3 if dft else 2,
            "atom_ids": ["oxygen-1", "oxygen-2" if dft else "hydrogen-1"],
        },
        "recipe": (
            "uks-pbe-d4-sto-3g-open-shell-validation"
            if dft
            else "uhf-sto-3g-open-shell-validation"
        ),
        "products": ["geometry"],
        "resources": {"cores": 1, "memory_mb": 1500, "wall_seconds": 120},
    }


def test_exact_open_shell_plan_does_not_request_restricted_spectroscopy():
    checked = validate_request(_request(), execution="local_validation")
    assert checked["executable"], checked["blocking_reasons"]
    profile = checked["recipe"]
    assert profile["reference"] == "unrestricted"
    assert profile["local_validation_only"] is True
    assert profile["ui_visible"] is False
    assert profile["derivatives"]["hessian"] == "unsupported"
    plan = checked["plan"]
    assert plan["plan_sha256"] == digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    assert [task["id"] for task in plan["tasks"]] == [
        "unrestricted_initial",
        "unrestricted_optimize",
    ]
    assert plan["spin_projection_applied"] is False
    assert plan["minimum_established"] is False
    assert plan["spectroscopy_requested"] is False
    assert {
        entry["tuple"]["property"] for entry in checked["capabilities"]["resolved"]
    } == {"energy", "gradient", "optimization"}
    assert all(
        entry["tuple"]["electronic_reference"] == "unrestricted_open_shell"
        for entry in checked["capabilities"]["resolved"]
    )


@pytest.mark.parametrize(
    "scope", ["actions", "harmonic", "catalog", "singlet", "charge"]
)
def test_open_shell_preflight_fails_closed_outside_declared_scope(scope):
    request = _request()
    execution = "local_validation"
    if scope == "actions":
        execution = "github_actions"
    elif scope == "harmonic":
        request["products"].append("harmonic")
    elif scope == "catalog":
        request["products"] = ["identification_catalog"]
    elif scope == "singlet":
        request["molecule"]["symbols"] = ["O", "O"]
        request["molecule"]["multiplicity"] = 1
    else:
        request["molecule"]["charge"] = 4
        request["molecule"]["multiplicity"] = 2
    checked = validate_request(request, execution=execution)
    assert checked["executable"] is False
    assert checked["blocking_reasons"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("electronic_reference", "restricted_closed_shell"),
        ("engine_version", "2.13.0"),
        ("property", "hessian"),
        ("recipe_sha256", "0" * 64),
    ],
)
def test_unrestricted_capability_never_aliases_a_different_tuple(field, value):
    record = profile_capabilities("uhf-sto-3g-open-shell-validation")[0]
    exact = record.tuple_definition.model_dump(mode="json")
    assert resolve_exact_capability(exact).availability == "experimental"
    exact[field] = value
    if field == "property":
        exact["derivative"] = "analytic"
    assert resolve_exact_capability(exact).availability == "unknown"


@pytest.fixture(scope="module")
def actual_open_shell_worker(tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    directory = tmp_path_factory.mktemp("actual-unrestricted-worker")
    request = CalculationRequest.model_validate(_request())
    return directory, request, worker_execute(request, directory)


@pytest.mark.real_engine
def test_actual_unrestricted_worker_has_typed_state_without_minimum_claim(
    actual_open_shell_worker,
):
    directory, request, result = actual_open_shell_worker
    assert result["status"] == "complete", result["errors"]
    energy = StageResult.model_validate(result["stages"]["electronic_structure"])
    geometry = StageResult.model_validate(result["stages"]["equilibrium_geometry"])
    assert energy.observable == "open_shell_electronic_energy"
    assert geometry.observable == "open_shell_optimized_geometry"
    observed = OpenShellElectronicEnergy.model_validate(energy.value)
    assert observed.molecule.multiplicity == request.molecule.multiplicity == 2
    assert observed.scf.alpha_electrons == 5 and observed.scf.beta_electrons == 4
    assert observed.spin.spin_squared > 0.75
    assert observed.spin.spin_projection_applied is False
    assert observed.stability.status == "stable"
    assert observed.lowest_electronic_state_certified is False
    assert geometry.value["minimum_established"] is False
    assert geometry.value["molecule"]["atom_ids"] == request.molecule.atom_ids
    assert result["stages"]["harmonic_analysis"]["absence_kind"] == "not_requested"
    assert result["stages"]["equilibrium_constants"]["absence_kind"] == "not_requested"
    assert result["experimental_accuracy_established"] is False
    assert result["identification_ready"] is False
    assert (directory / "unrestricted-initial/wavefunction.chk").is_file()
    assert (directory / "unrestricted-optimization/final/wavefunction.chk").is_file()


@pytest.mark.real_engine
@pytest.mark.parametrize("corruption", ["sector", "projection", "spin"])
def test_typed_actual_open_shell_energy_rejects_false_state_claims(
    actual_open_shell_worker, corruption
):
    _, _, result = actual_open_shell_worker
    value = deepcopy(result["stages"]["electronic_structure"]["value"])
    if corruption == "sector":
        value["scf"]["alpha_electrons"] = 4
    elif corruption == "projection":
        value["spin"]["spin_projection_applied"] = True
    else:
        value["spin"]["spin_squared"] = 0.75
    with pytest.raises(ValueError):
        StageResult.model_validate(
            {
                "status": "available",
                "observable": "open_shell_electronic_energy",
                "value": value,
            }
        )


@pytest.mark.real_engine
def test_actual_unstable_uks_reference_retains_initial_energy_and_stops_geometry(
    tmp_path,
):
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd4")
    result = worker_execute(
        CalculationRequest.model_validate(_request(dft=True)), tmp_path
    )
    assert result["status"] == "partial", result["errors"]
    stage = StageResult.model_validate(result["stages"]["electronic_structure"])
    observation = OpenShellElectronicEnergy.model_validate(stage.value)
    assert observation.stability.status == "unstable"
    assert observation.stability.external_stable is False
    assert observation.stability.candidate_orbitals_applied is False
    assert observation.molecule.multiplicity == 3
    assert result["native_result"]["dispersion"]["implementation"]["version"] == "3.7.0"
    assert result["stages"]["equilibrium_geometry"]["status"] == "unavailable"
    assert not (tmp_path / "unrestricted-optimization").exists()
    assert result["native_result"]["requested_state_preserved"] is True
    assert any(
        error["code"] == "UNRESTRICTED_STATE_NOT_QUALIFIED"
        for error in result["errors"]
    )


@pytest.mark.real_engine
def test_actual_open_shell_approved_worker_and_immutable_shard(tmp_path):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    from cochem_torq.service import approve_plan, plan_request

    request = CalculationRequest.model_validate(_request()).model_dump(mode="json")
    planned = plan_request(request, execution="local_validation")
    approved = approve_plan(planned, actor="actual-open-shell-validation-test")
    directory = tmp_path / "approved-oh-doublet"
    result = execute_request(request, directory, approved_plan=approved)
    assert result["status"] == "complete", result["errors"]
    manifest = verify_shard(directory)
    assert manifest["request_sha256"] == result["request_sha256"]
    assert manifest["recipe_sha256"] == result["recipe_sha256"]
    assert result["plan"]["tasks"][0]["id"] == "unrestricted_initial"
    assert result["native_result"]["method"]["reference"] == "unrestricted"
    StageResult.model_validate(result["stages"]["electronic_structure"])
    StageResult.model_validate(result["stages"]["equilibrium_geometry"])
    for relative in (
        "unrestricted-initial/wavefunction.chk",
        "unrestricted-optimization/final/wavefunction.chk",
    ):
        assert (directory / relative).is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        execute_request(request, directory)
