"""Exact routing and genuine native restart bytes; no manufactured engine outputs."""

import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from cochem_torq.capabilities import (
    CapabilityRecord,
    CapabilityTuple,
    RestartFingerprint,
    ReusePolicy,
    assess_restart_reuse,
    authorize_capability,
    locally_validate_capability,
    prepare_initial_guess,
    register_restart_artifact,
)
from cochem_torq.domain import canonical_json, read_json
from cochem_torq.registry import (
    profile_capabilities,
    resolve_exact_capability,
    route_profile_capabilities,
)


def definition(property_name="energy"):
    return next(
        record.tuple_definition
        for record in profile_capabilities("hf-sto-3g-education")
        if record.tuple_definition.property == property_name
    )


def test_builtin_registry_never_silently_promotes_experimental_profiles():
    records = profile_capabilities("hf-sto-3g-education")
    assert len(records) == 5
    for record in records:
        assert record.availability == "experimental" and record.evidence is None
        assert authorize_capability(record, purpose="controlled_validation")[
            "authorized"
        ]
        with pytest.raises(ValueError, match="do not authorize production"):
            authorize_capability(record, purpose="production")
    assert profile_capabilities("revdsd-pbep86-d4-experimental") == ()


@pytest.mark.parametrize(
    "change",
    [
        {"method": "unregistered-method"},
        {"basis": "different-basis"},
        {"ecp": "unregistered-ecp"},
        {"electronic_reference": "unrestricted"},
        {"engine": "ORCA"},
        {"engine_version": "unqualified-version"},
        {"hardware": "linux-x86_64-cuda"},
        {"property": "efg", "derivative": "analytic"},
    ],
)
def test_similar_tuple_cannot_inherit_exact_capability(change):
    candidate = definition().model_dump(mode="json")
    candidate.update(change)
    record = resolve_exact_capability(candidate)
    assert record.availability == "unknown"
    with pytest.raises(ValueError, match="unknown tuples"):
        authorize_capability(record, purpose="controlled_validation")


def test_known_unsupported_mp2_dipole_is_distinct_from_unknown():
    record = next(
        item
        for item in profile_capabilities("mp2-cc-pvdz-validation")
        if item.tuple_definition.property == "dipole"
    )
    assert record.availability == "unsupported"
    with pytest.raises(ValueError, match="unsupported tuples"):
        authorize_capability(record, purpose="controlled_validation")


def test_local_validation_cannot_be_declared_without_bound_native_evidence():
    with pytest.raises(ValueError, match="requires evidence"):
        CapabilityRecord(
            tuple_definition=definition(),
            availability="locally_validated",
            reason="declared",
        )
    with pytest.raises(ValueError, match="derivative route"):
        CapabilityTuple.model_validate(
            {**definition().model_dump(mode="json"), "derivative": "analytic"}
        )


def test_declared_actions_target_has_exact_native_dependency_closure():
    report = route_profile_capabilities(
        "hf-sto-3g-education", ["equilibrium_constants"], execution="github_actions"
    )
    assert report["executable"] and not report["chemical_accuracy_established"]
    assert {item["tuple"]["property"] for item in report["resolved"]} == {
        "energy",
        "gradient",
        "optimization",
        "dipole",
        "hessian",
    }
    assert report["hardware_evidence"] == "declared_canonical_actions_cpu_target"


@pytest.fixture(scope="module")
def genuine_native(tmp_path_factory):
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")
    from cochem_torq.engines.pyscf_backend import PySCFBackend

    directory = tmp_path_factory.mktemp("cochem_exec_capability_native")
    data = {
        "molecule": {
            "symbols": ["H", "H"],
            "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
            "charge": 0,
            "multiplicity": 1,
        },
        "method": {
            "name": "hf",
            "basis": "sto-3g",
            "reference": "restricted",
            "frozen_core": False,
        },
        "properties": ["energy", "gradient", "hessian", "dipole"],
        "settings": {"threads": 1},
    }
    result = PySCFBackend().evaluate(data, directory)
    assert result["status"] == "complete", result["errors"]
    assert result["scf"]["converged"] and result["stability"]["status"] == "stable"
    return directory


@pytest.fixture(scope="module")
def genuine_optimization(genuine_native, tmp_path_factory):
    from cochem_torq.engines.pyscf_backend import PySCFBackend

    directory = tmp_path_factory.mktemp("cochem_exec_capability_optimization")
    request = read_json(genuine_native / "request.json")
    request["molecule"]["geometry_bohr"] = [
        [0.0, 0.0, -0.8],
        [0.0, 0.0, 0.8],
    ]
    result = PySCFBackend().optimize(request, directory)
    assert result["status"] == "complete", result["errors"]
    assert result["optimization"]["converged"]
    assert result["optimization"]["final_gradient_verified"]
    assert result["geometry_bohr"] != request["molecule"]["geometry_bohr"]
    return directory


@pytest.mark.real_engine
def test_actual_optimization_qualifies_its_exact_outer_and_final_native_case(
    genuine_optimization, tmp_path
):
    exact = definition("optimization")
    record = locally_validate_capability(exact, genuine_optimization)
    receipt = authorize_capability(
        record,
        purpose="production",
        native_request_sha256=record.evidence.native_request_sha256,
    )
    assert (
        receipt["qualification_scope"] == "exact_native_request_numerical_integration"
    )
    assert not receipt["chemical_accuracy_established"]
    artifact = register_restart_artifact(
        genuine_optimization,
        "final/wavefunction.chk",
        tmp_path / "final-sealed.chk",
        artifact_kind="wavefunction",
        recipe_sha256=exact.recipe_sha256,
    )
    actual = read_json(genuine_optimization / "final/result.json")
    assert [list(row) for row in artifact.fingerprint.geometry_bohr] == actual[
        "geometry_bohr"
    ]
    # Both checkpoints are genuine, but the initial one cannot be relabeled
    # with the geometry/state fingerprint of the optimized final calculation.
    with pytest.raises(ValueError, match="checkpoint geometry/state/basis differs"):
        register_restart_artifact(
            genuine_optimization,
            "initial/wavefunction.chk",
            tmp_path / "incorrectly-labeled-initial.chk",
            artifact_kind="wavefunction",
            recipe_sha256=exact.recipe_sha256,
        )
    assert not (tmp_path / "incorrectly-labeled-initial.chk").exists()


def _record_corrupted_native_inventory(directory):
    """Record actual deliberately damaged bytes; never create engine observations."""
    manifest = read_json(directory / "manifest.json")
    manifest["artifacts"] = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and "scratch" not in path.relative_to(directory).parts
    ]
    (directory / "manifest.json").write_bytes(canonical_json(manifest))


@pytest.mark.real_engine
@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("different_genuine_final", "retained final native evidence"),
        ("omitted_child_inventory", "retained final native evidence"),
        ("different_genuine_initial_request", "geometry/state definitions differ"),
        ("truncated_actual_trajectory", "convergence/trajectory is unestablished"),
    ],
)
def test_optimization_cannot_qualify_mismatched_nested_native_provenance(
    genuine_optimization, genuine_native, tmp_path, damage, message
):
    damaged = tmp_path / "deliberately-corrupted-genuine-optimization"
    shutil.copytree(genuine_optimization, damaged)
    if damage == "different_genuine_final":
        shutil.rmtree(damaged / "final")
        shutil.copytree(genuine_native, damaged / "final")
    elif damage == "omitted_child_inventory":
        path = damaged / "final/manifest.json"
        manifest = read_json(path)
        manifest["artifacts"] = [
            entry
            for entry in manifest["artifacts"]
            if entry["path"] != "wavefunction.chk"
        ]
        path.write_bytes(canonical_json(manifest))
    elif damage == "different_genuine_initial_request":
        shutil.copyfile(genuine_native / "request.json", damaged / "request.json")
    else:
        path = damaged / "optimization-trajectory.json"
        path.write_bytes(canonical_json(read_json(path)[:1]))
    _record_corrupted_native_inventory(damaged)
    with pytest.raises(ValueError, match=message):
        locally_validate_capability(definition("optimization"), damaged)


@pytest.mark.real_engine
def test_actual_local_evidence_routes_only_its_source_and_exact_native_request(
    genuine_native,
):
    record = locally_validate_capability(definition(), genuine_native)
    receipt = authorize_capability(
        record,
        purpose="production",
        native_request_sha256=record.evidence.native_request_sha256,
    )
    assert receipt["authorized"] and not receipt["chemical_accuracy_established"]
    with pytest.raises(ValueError, match="exact normalized native request"):
        authorize_capability(
            record, purpose="production", native_request_sha256="0" * 64
        )
    altered = record.model_dump(mode="json")
    altered["evidence"]["source_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different implementation source"):
        authorize_capability(
            CapabilityRecord.model_validate(altered),
            purpose="production",
            native_request_sha256=record.evidence.native_request_sha256,
        )


@pytest.mark.real_engine
def test_actual_native_property_cannot_qualify_another_basis_or_hardware(
    genuine_native,
):
    for field, value in (("basis", "cc-pvdz"), ("hardware", "linux-x86_64-cuda")):
        candidate = definition().model_dump(mode="json")
        candidate[field] = value
        with pytest.raises(ValueError, match="does not match the exact capability"):
            locally_validate_capability(
                CapabilityTuple.model_validate(candidate), genuine_native
            )
    candidate = definition().model_dump(mode="json")
    candidate["recipe_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registered exact experimental recipe"):
        locally_validate_capability(
            CapabilityTuple.model_validate(candidate), genuine_native
        )


def register(genuine_native, destination):
    return register_restart_artifact(
        genuine_native,
        "wavefunction.chk",
        destination,
        artifact_kind="wavefunction",
        recipe_sha256=definition().recipe_sha256,
    )


@pytest.mark.real_engine
def test_authentic_checkpoint_is_preserved_sealed_and_copied_to_distinct_working_file(
    genuine_native, tmp_path
):
    original = genuine_native / "wavefunction.chk"
    original_digest = sha256(original.read_bytes()).hexdigest()
    artifact = register(genuine_native, tmp_path / "sealed" / "checkpoint.chk")
    assert artifact.artifact_sha256 == original_digest
    assert not Path(artifact.sealed_path).stat().st_mode & 0o222
    target_data = artifact.fingerprint.model_dump(mode="json")
    target_data["geometry_bohr"][1][2] += 0.05
    target = RestartFingerprint.model_validate(target_data)
    policy = ReusePolicy(
        purpose="initial_guess", allowed_initial_guess_changes=("geometry_bohr",)
    )
    assessment = prepare_initial_guess(
        artifact, target, tmp_path / "working" / "guess.chk", policy=policy
    )
    assert assessment["compatible"] and assessment["new_physical_calculation_required"]
    assert not assessment["engine_checkpoint_reused"]
    assert (
        Path(assessment["prepared_working_copy"]).read_bytes() == original.read_bytes()
    )
    assert Path(assessment["prepared_working_copy"]).stat().st_mode & 0o200
    assert sha256(original.read_bytes()).hexdigest() == original_digest
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        prepare_initial_guess(
            artifact, target, assessment["prepared_working_copy"], policy=policy
        )
    with pytest.raises(ValueError, match="immutable original"):
        prepare_initial_guess(artifact, target, original, policy=policy)


@pytest.mark.real_engine
@pytest.mark.parametrize(
    "field,value",
    [
        ("atom_ids", ["atom-1", "atom-0"]),
        ("charge", -2),
        ("source_code_sha256", "0" * 64),
        ("basis_definition_sha256", "0" * 64),
        ("engine_version", "different"),
        ("recipe_sha256", "0" * 64),
        ("engine_installation_sha256", "0" * 64),
    ],
)
def test_incompatible_checkpoint_fields_cannot_be_initial_guesses(
    genuine_native, tmp_path, field, value
):
    artifact = register(genuine_native, tmp_path / "checkpoint.chk")
    data = artifact.fingerprint.model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValueError, match="Incompatible restart fingerprint"):
        assess_restart_reuse(
            artifact,
            RestartFingerprint.model_validate(data),
            ReusePolicy(purpose="initial_guess"),
        )


@pytest.mark.real_engine
def test_final_reuse_requires_exact_geometry_and_isotope_identity(
    genuine_native, tmp_path
):
    artifact = register(genuine_native, tmp_path / "checkpoint.chk")
    exact = assess_restart_reuse(
        artifact, artifact.fingerprint, ReusePolicy(purpose="final_result")
    )
    assert exact["compatible"] and not exact["engine_checkpoint_reused"]
    data = artifact.fingerprint.model_dump(mode="json")
    data["geometry_bohr"][1][2] += 0.05
    with pytest.raises(ValueError, match="Incompatible restart fingerprint"):
        assess_restart_reuse(
            artifact,
            RestartFingerprint.model_validate(data),
            ReusePolicy(purpose="final_result"),
        )
    data = artifact.fingerprint.model_dump(mode="json")
    data["isotope_symbols"] = ["2H", "2H"]
    with pytest.raises(ValueError, match="isotope_symbols"):
        assess_restart_reuse(
            artifact,
            RestartFingerprint.model_validate(data),
            ReusePolicy(purpose="final_result"),
        )
    data = artifact.fingerprint.model_dump(mode="json")
    import json

    settings = json.loads(data["numerical_settings_json"])
    settings["scf_energy_tolerance"] *= 10.0
    from cochem_torq.domain import canonical_json

    data["numerical_settings_json"] = canonical_json(settings).decode()
    with pytest.raises(ValueError, match="numerical_settings_json"):
        assess_restart_reuse(
            artifact,
            RestartFingerprint.model_validate(data),
            ReusePolicy(purpose="final_result"),
        )
    with pytest.raises(ValueError, match="exact fingerprint"):
        ReusePolicy(
            purpose="final_result", allowed_initial_guess_changes=("geometry_bohr",)
        )


@pytest.mark.real_engine
def test_actual_hessian_registration_rejects_relabeling_and_non_native_models(
    genuine_native, tmp_path
):
    with pytest.raises(ValueError, match="actual native checkpoint"):
        register_restart_artifact(
            genuine_native,
            "basis-definition.json",
            tmp_path / "wrong.chk",
            artifact_kind="wavefunction",
            recipe_sha256=definition().recipe_sha256,
        )
    with pytest.raises(NotImplementedError, match="no qualified native registration"):
        register_restart_artifact(
            genuine_native,
            "wavefunction.chk",
            tmp_path / "model.chk",
            artifact_kind="model_checkpoint",
            recipe_sha256=definition().recipe_sha256,
        )
    artifact = register_restart_artifact(
        genuine_native,
        "hessian-hartree-bohr2.npy",
        tmp_path / "hessian.npy",
        artifact_kind="hessian",
        recipe_sha256=definition().recipe_sha256,
    )
    assert assess_restart_reuse(
        artifact, artifact.fingerprint, ReusePolicy(purpose="final_result")
    )["compatible"]
    with pytest.raises(ValueError, match="no implemented initial-guess"):
        assess_restart_reuse(
            artifact, artifact.fingerprint, ReusePolicy(purpose="initial_guess")
        )


@pytest.mark.real_engine
def test_changed_authentic_checkpoint_copy_is_rejected_without_repair(
    genuine_native, tmp_path
):
    artifact = register(genuine_native, tmp_path / "checkpoint.chk")
    sealed = Path(artifact.sealed_path)
    sealed.chmod(0o600)
    with sealed.open("ab") as stream:
        stream.write(b"Deliberate changed-byte integrity check")
    sealed.chmod(0o440)
    corrupted_digest = sha256(sealed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="restart bytes changed"):
        assess_restart_reuse(
            artifact, artifact.fingerprint, ReusePolicy(purpose="final_result")
        )
    assert sha256(sealed.read_bytes()).hexdigest() == corrupted_digest
