"""Published canonical vectors and actual failure-record/file compatibility.

No engine success is manufactured. Legacy fixtures encode real application
rejections in the previous serialization contract; they are migration tests,
not a claim to possess an archived scientific calculation from an old release.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
import rfc8785

from cochem_torq.application import validate_request, worker_execute
from cochem_torq.artifacts import inventory, seal_shard, verify_shard
from cochem_torq.domain import (
    CANONICALIZATION_PROFILE,
    LEGACY_CANONICALIZATION_PROFILE,
    CalculationRequest,
    canonical_json,
    digest,
    read_json,
    scientific_cache_key,
    scientific_content,
)
from cochem_torq.registry import get_profile
from tests.test_application_contracts import actual_failure_shard

VECTORS = Path(__file__).parent / "data" / "rfc8785"


@pytest.mark.parametrize(
    "name", ["arrays", "french", "structures", "unicode", "values", "weird"]
)
def test_published_reference_canonical_vectors(name):
    source = json.loads((VECTORS / "SOURCE.json").read_text())
    files = {record["path"]: record for record in source["files"]}
    for kind in ("input", "output"):
        relative = f"testdata/{kind}/{name}.json"
        assert (
            sha256((VECTORS / relative).read_bytes()).hexdigest()
            == files[relative]["sha256"]
        )
    input_bytes = (VECTORS / "testdata" / "input" / f"{name}.json").read_bytes()
    expected_bytes = (VECTORS / "testdata" / "output" / f"{name}.json").read_bytes()
    assert canonical_json(json.loads(input_bytes)) == expected_bytes


# Published in the RFC author's independent reference tests/README, not generated
# from TORQ's serializer. Values are IEEE-754 bit patterns, not unsafe Python ints.
@pytest.mark.parametrize(
    ("ieee_hex", "expected"),
    [
        ("4340000000000001", b"9007199254740994"),
        ("4340000000000002", b"9007199254740996"),
        ("444b1ae4d6e2ef50", b"1e+21"),
        ("3eb0c6f7a0b5ed8d", b"0.000001"),
        ("3eb0c6f7a0b5ed8c", b"9.999999999999997e-7"),
        ("8000000000000000", b"0"),
        ("0000000000000000", b"0"),
    ],
)
def test_published_ieee754_numeric_vectors(ieee_hex, expected):
    number = struct.unpack(">d", bytes.fromhex(ieee_hex))[0]
    assert canonical_json(number) == expected


def test_utf16_key_order_is_distinct_from_codepoint_order():
    payload = {"\ufb33": "Hebrew", "\U0001f600": "emoji"}
    assert canonical_json(payload) == '{"😀":"emoji","דּ":"Hebrew"}'.encode()
    assert canonical_json(
        payload, profile=LEGACY_CANONICALIZATION_PROFILE
    ) != canonical_json(payload)


def test_string_identity_is_preserved_without_unicode_normalization():
    assert canonical_json("A\u030a") != canonical_json("\u00c5")
    assert digest("A\u030a") != digest("\u00c5")


@pytest.mark.parametrize("number", [2**53 - 1, -(2**53 - 1)])
def test_safe_integer_boundary_is_exact(number):
    assert canonical_json(number) == str(number).encode()


@pytest.mark.parametrize(
    "invalid",
    [
        2**53,
        -(2**53),
        float("nan"),
        float("inf"),
        float("-inf"),
        complex(1, 1),
        {1: "non-string key"},
        {"unsupported": {1, 2}},
        "\ud800",
        "\udfff",
        {"\ud800": "unpaired surrogate key"},
    ],
)
def test_non_json_or_out_of_domain_content_is_rejected(invalid):
    with pytest.raises((rfc8785.CanonicalizationError, UnicodeEncodeError)):
        canonical_json(invalid)


def test_unsupported_canonical_profile_rejected():
    with pytest.raises(ValueError, match="Unsupported canonical"):
        canonical_json({"value": 1}, profile="unverified-sorted-format")


def molecular_request():
    return CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0, 0, 0], [0, 0, 1.4]],
                "charge": 0,
                "multiplicity": 1,
                "atom_ids": ["left", "right"],
                "isotopes": [1, 1],
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry", "harmonic"],
            "source_provenance": {
                "imported_at": "2026-10-08T00:00:00Z",
                "workflow_run_id": "observed-record-id",
            },
        }
    )


def test_scientific_key_excludes_submission_uuid_time_and_allocations():
    request = molecular_request()
    recipe = get_profile(request.recipe)
    changed = request.model_dump(mode="json")
    changed["request_id"] = str(uuid4())
    changed["resources"] = {"cores": 2, "memory_mb": 4096, "wall_seconds": 1000}
    changed["source_provenance"] = {
        "imported_at": "2026-10-08T00:01:00Z",
        "workflow_run_id": "another-observed-record-id",
    }
    other = CalculationRequest.model_validate(changed)
    assert digest(request.model_dump(mode="json")) != digest(
        other.model_dump(mode="json")
    )
    assert scientific_cache_key(request, recipe) == scientific_cache_key(other, recipe)
    content = scientific_content(request, recipe)
    assert content["serialization_profile"] == CANONICALIZATION_PROFILE
    assert content["geometry_representation"] == {
        "unit": "bohr",
        "shape": [2, 3],
        "dtype": "float64",
    }


@pytest.mark.parametrize(
    "change", ["geometry", "state", "isotope", "atom_order", "products"]
)
def test_scientific_key_covers_physical_identity_and_requested_products(change):
    request = molecular_request()
    recipe = get_profile(request.recipe)
    changed = request.model_dump(mode="json")
    if change == "geometry":
        changed["molecule"]["geometry_bohr"][1][2] += 0.001
    elif change == "state":
        changed["molecule"]["multiplicity"] = 3
    elif change == "isotope":
        changed["molecule"]["isotopes"] = [2, 2]
    elif change == "atom_order":
        changed["molecule"]["atom_ids"] = ["right", "left"]
        changed["molecule"]["geometry_bohr"].reverse()
    else:
        changed["products"] = ["geometry"]
    other = CalculationRequest.model_validate(changed)
    assert scientific_cache_key(request, recipe) != scientific_cache_key(other, recipe)


@pytest.mark.parametrize(
    "change", ["basis", "convergence", "frozen_core", "dispersion"]
)
def test_scientific_key_covers_complete_numerical_method_recipe(change):
    request = molecular_request()
    recipe = get_profile(request.recipe)
    changed = deepcopy(recipe)
    if change == "basis":
        changed["basis"] = "cc-pvdz"
    elif change == "convergence":
        changed["numerical"]["scf_energy_tolerance"] /= 10
    elif change == "frozen_core":
        changed["frozen_core"] = True
    else:
        changed["dispersion"] = {"method": "D4", "variant": "BJ-EEQ-ATM"}
    assert scientific_cache_key(request, recipe) != scientific_cache_key(
        request, changed
    )


def test_content_order_and_redundant_recipe_digest_do_not_change_cache_identity():
    request = molecular_request()
    recipe = get_profile(request.recipe)
    other_data = request.model_dump(mode="json")
    other_data["products"].reverse()
    other = CalculationRequest.model_validate(other_data)
    changed_recipe = {key: value for key, value in reversed(list(recipe.items()))}
    changed_recipe["recipe_sha256"] = digest({"separate_record": "test"})
    assert scientific_cache_key(request, recipe) == scientific_cache_key(
        other, changed_recipe
    )


def test_catalog_inputs_affect_cache_only_when_catalog_requested():
    request = molecular_request()
    recipe = get_profile(request.recipe)
    other_data = request.model_dump(mode="json")
    other_data["catalog"] = {"temperature_kelvin": 20.0, "max_j": 10}
    other = CalculationRequest.model_validate(other_data)
    assert scientific_cache_key(request, recipe) == scientific_cache_key(other, recipe)
    first_data = request.model_dump(mode="json")
    first_data["products"].append("rigid_rotor_catalog")
    other_data["products"].append("rigid_rotor_catalog")
    first = CalculationRequest.model_validate(first_data)
    second = CalculationRequest.model_validate(other_data)
    assert scientific_cache_key(first, recipe) != scientific_cache_key(second, recipe)


@pytest.mark.parametrize(
    "product",
    [
        "harmonic",
        "equilibrium_constants",
        "rigid_rotor_catalog",
        "anharmonic_force_field",
        "vpt2",
        "ground_state_constants",
        "identification_catalog",
    ],
)
def test_mass_dependent_content_binds_actual_isotope_values_and_source(product):
    import mendeleev

    from Libraries.cochem_isotopes import isotope_record

    data = molecular_request().model_dump(mode="json")
    data["products"] = [product]
    request = CalculationRequest.model_validate(data)
    content = scientific_content(request, get_profile(request.recipe))
    records = content["resolved_isotope_evidence"]
    actual = isotope_record("1H")
    assert len(records) == 2
    assert records[0] == records[1]
    assert records[0] == {
        "element": actual["element"],
        "mass_number": actual["mass_number"],
        "mass_u": actual["mass_u"],
        "mass_uncertainty_u": actual["mass_uncertainty_u"],
        "selection_policy": actual["selection_policy"],
        "source": actual["source"],
    }
    database = Path(mendeleev.__file__).parent / "elements.db"
    assert (
        records[0]["source"]["database_sha256"]
        == sha256(database.read_bytes()).hexdigest()
    )
    assert records[0]["source"]["tabulated_mass_is_exact"] is False
    assert str(database) not in canonical_json(records).decode()


def test_geometry_only_content_has_no_mass_table_dependency():
    data = molecular_request().model_dump(mode="json")
    data["products"] = ["geometry"]
    request = CalculationRequest.model_validate(data)
    assert (
        scientific_content(request, get_profile(request.recipe))[
            "resolved_isotope_evidence"
        ]
        is None
    )


@pytest.mark.parametrize("symbol,numbers", [("H", (1, 2)), ("C", (12, 13))])
def test_actual_isotopologue_mass_observations_change_scientific_cache(symbol, numbers):
    from Libraries.cochem_isotopes import isotope_record

    data = molecular_request().model_dump(mode="json")
    data["molecule"]["symbols"] = [symbol, symbol]
    data["molecule"]["isotopes"] = [numbers[0], numbers[0]]
    first = CalculationRequest.model_validate(data)
    data["molecule"]["isotopes"] = [numbers[1], numbers[1]]
    second = CalculationRequest.model_validate(data)
    recipe = get_profile(first.recipe)
    initial = scientific_content(first, recipe)["resolved_isotope_evidence"]
    changed = scientific_content(second, recipe)["resolved_isotope_evidence"]
    assert initial[0]["mass_u"] == isotope_record(f"{numbers[0]}{symbol}")["mass_u"]
    assert changed[0]["mass_u"] == isotope_record(f"{numbers[1]}{symbol}")["mass_u"]
    assert initial[0]["mass_u"] != changed[0]["mass_u"]
    assert scientific_cache_key(first, recipe) != scientific_cache_key(second, recipe)


def test_default_isotope_selection_is_retained_as_actual_policy():
    from Libraries.cochem_isotopes import isotope_record

    data = molecular_request().model_dump(mode="json")
    data["molecule"]["isotopes"] = None
    request = CalculationRequest.model_validate(data)
    records = scientific_content(request, get_profile(request.recipe))[
        "resolved_isotope_evidence"
    ]
    actual = isotope_record("H")
    assert records[0]["mass_number"] == actual["mass_number"]
    assert records[0]["mass_u"] == actual["mass_u"]
    assert records[0]["selection_policy"] == "most_abundant_naturally_occurring_isotope"


def test_retained_source_projection_ignores_local_paths_without_mutating_records():
    import mendeleev

    from Libraries.cochem_isotopes import isotope_record

    request = molecular_request()
    recipe = get_profile(request.recipe)
    records = [isotope_record("1H"), isotope_record("1H")]
    before = deepcopy(records)
    observed = scientific_content(request, recipe, resolved_isotope_records=records)
    assert records == before
    for record in records:
        record["source"]["local_database_path"] = str(
            Path(mendeleev.__file__).parent / "elements.db"
        )
    assert (
        scientific_content(request, recipe, resolved_isotope_records=records)
        == observed
    )
    assert scientific_cache_key(
        request, recipe, resolved_isotope_records=records
    ) == scientific_cache_key(request, recipe)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_atom",
        "different_isotope",
        "negative_mass",
        "nonfinite_mass",
        "complex_mass",
        "uncertainty",
        "source_digest",
        "selection_policy",
    ],
)
def test_cache_rejects_corrupted_copies_of_actual_isotope_records(corruption):
    from Libraries.cochem_isotopes import isotope_record

    request = molecular_request()
    records = [isotope_record("1H"), isotope_record("1H")]
    if corruption == "missing_atom":
        records.pop()
    elif corruption == "different_isotope":
        records[0] = isotope_record("2H")
    elif corruption == "negative_mass":
        records[0]["mass_u"] *= -1
    elif corruption == "nonfinite_mass":
        records[0]["mass_u"] = float("nan")
    elif corruption == "complex_mass":
        records[0]["mass_u"] = complex(records[0]["mass_u"], 1e-16)
    elif corruption == "uncertainty":
        records[0]["mass_uncertainty_u"] = -1
    elif corruption == "source_digest":
        records[0]["source"]["database_sha256"] = "not-a-digest"
    else:
        records[0]["selection_policy"] = "rounded_average_mass"
    with pytest.raises(ValueError, match="Cache .*isotope"):
        scientific_cache_key(
            request, get_profile(request.recipe), resolved_isotope_records=records
        )


def actual_mass_dependent_failure_shard(directory):
    """Real rejected calculation; isotope resolution is not engine success."""
    directory.mkdir()
    request = molecular_request().model_dump(mode="json")
    request["recipe"] = "revdsd-pbep86-d4-experimental"
    request = CalculationRequest.model_validate(request)
    checked = validate_request(request)
    result = worker_execute(request, directory)
    assert result["status"] == "failed"
    assert all(stage["value"] is None for stage in result["stages"].values())
    (directory / "request.json").write_bytes(canonical_json(checked["request"]))
    (directory / "result.json").write_bytes(canonical_json(result))
    seal_shard(
        directory,
        request_sha256=checked["request_sha256"],
        request_id=str(request.request_id),
        recipe_sha256=checked["recipe"]["recipe_sha256"],
        source_identity=result["source_identity"],
        worker_id="actual-rejected-mass-dependent-profile",
    )
    return directory


def test_actual_failed_mass_shard_retains_portable_isotope_cache_inputs(tmp_path):
    directory = actual_mass_dependent_failure_shard(tmp_path / "actual-failed-harmonic")
    manifest = verify_shard(directory)
    evidence = manifest["scientific_cache_isotope_evidence"]
    assert evidence and evidence[0]["mass_number"] == 1
    assert evidence[0]["source"]["database_sha256"]
    assert not (directory / "engine").exists()
    # Fresh interpreter verifies the genuine sealed failure without importing
    # Mendeleev or resolving/replacing masses from the viewer's installation.
    code = """
import sys
from cochem_torq.artifacts import verify_shard
assert 'mendeleev' not in sys.modules
verified = verify_shard(sys.argv[1])
assert 'mendeleev' not in sys.modules
print(verified['scientific_cache_key'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == manifest["scientific_cache_key"]


def test_removing_sealed_mass_evidence_cannot_enable_local_substitution(tmp_path):
    directory = actual_mass_dependent_failure_shard(tmp_path / "actual-failed-harmonic")
    manifest = read_json(directory / "manifest.json")
    manifest.pop("scientific_cache_isotope_evidence")
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="sealed isotope evidence"):
        verify_shard(directory)


def test_real_failure_seals_conversion_definitions_for_portable_verification(tmp_path):
    directory = actual_failure_shard(tmp_path / "genuine-rejected-profile")
    manifest = verify_shard(directory)
    result = read_json(directory / "result.json")
    evidence = manifest["scientific_cache_constants_evidence"]
    assert evidence["profile"] == result["constants"]["profile"]
    assert evidence["definition_sha256"] == result["constants"]["definition_sha256"]
    definition = {
        key: value for key, value in evidence.items() if key != "definition_sha256"
    }
    assert digest(definition) == evidence["definition_sha256"]
    assert evidence["constants"]["speed of light in vacuum"]["value"] == 299792458
    # A receiver verifies retained definitions without consulting its own
    # SciPy/CODATA provider. This bundle is a real rejection, not engine success.
    program = """
import sys
from cochem_torq.artifacts import verify_shard
assert 'cochem_torq.units' not in sys.modules
record = verify_shard(sys.argv[1])
assert 'cochem_torq.units' not in sys.modules
print(record['scientific_cache_key'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(directory)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == manifest["scientific_cache_key"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_sealed",
        "invalid_digest",
        "changed_sealed",
        "missing_result",
        "changed_result",
    ],
)
def test_corrupted_retained_conversion_evidence_cannot_verify(corruption, tmp_path):
    directory = actual_failure_shard(tmp_path / "genuine-rejected-profile")
    manifest = read_json(directory / "manifest.json")
    result = read_json(directory / "result.json")
    if corruption == "missing_sealed":
        manifest.pop("scientific_cache_constants_evidence")
    elif corruption in ("invalid_digest", "changed_sealed"):
        evidence = manifest["scientific_cache_constants_evidence"]
        evidence["constants"]["speed of light in vacuum"]["value"] += 1
        if corruption == "changed_sealed":
            evidence["definition_sha256"] = digest(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "definition_sha256"
                }
            )
    elif corruption == "missing_result":
        result.pop("constants")
    else:
        constants = result["constants"]
        constants["constants"]["speed of light in vacuum"]["value"] += 1
        retained_fields = set(manifest["scientific_cache_constants_evidence"]) - {
            "definition_sha256"
        }
        constants["definition_sha256"] = digest(
            {key: constants[key] for key in retained_fields}
        )
    if corruption in ("missing_result", "changed_result"):
        (directory / "result.json").write_bytes(canonical_json(result))
        manifest["files"] = inventory(directory)
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="constants|content identity"):
        verify_shard(directory)


def old_encoding_of_actual_failure(tmp_path):
    current = actual_failure_shard(tmp_path / "actual-rejected-profile")
    request = read_json(current / "request.json")
    result = read_json(current / "result.json")
    # RFC output normalizes 0.0 to 0. Reconstruct the actual typed float geometry
    # before encoding the same physical rejection in the older JSON contract.
    request = CalculationRequest.model_validate(request).model_dump(mode="json")
    result["molecule"] = request["molecule"]
    request.pop("serialization_profile", None)
    result.pop("serialization_profile", None)
    result.pop("scientific_cache_key", None)
    recipe = dict(result["recipe"])
    recipe.pop("recipe_sha256")
    old_recipe_digest = digest(recipe, profile=LEGACY_CANONICALIZATION_PROFILE)
    old_request_digest = digest(request, profile=LEGACY_CANONICALIZATION_PROFILE)
    result["recipe"]["recipe_sha256"] = old_recipe_digest
    result["recipe_sha256"] = old_recipe_digest
    result["request_sha256"] = old_request_digest
    legacy = tmp_path / "legacy-encoding-of-actual-failure"
    legacy.mkdir()
    for name, content in (("request.json", request), ("result.json", result)):
        (legacy / name).write_bytes(
            canonical_json(content, profile=LEGACY_CANONICALIZATION_PROFILE) + b"\n"
        )
    manifest = {
        "schema_version": "cochem.torq.shard/1",
        "request_id": result["request_id"],
        "request_sha256": old_request_digest,
        "recipe_sha256": old_recipe_digest,
        "source_identity": result["source_identity"],
        "worker_id": "actual-rejection-legacy-format-test",
        "files": inventory(legacy),
    }
    (legacy / "manifest.json").write_bytes(
        canonical_json(manifest, profile=LEGACY_CANONICALIZATION_PROFILE) + b"\n"
    )
    assert result["status"] == "failed"
    assert all(stage["value"] is None for stage in result["stages"].values())
    return legacy


def test_previous_shard_bytes_verify_without_transcoding_or_new_profile_injection(
    tmp_path,
):
    directory = old_encoding_of_actual_failure(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    request = read_json(directory / "request.json")
    assert "serialization_profile" not in request
    assert (
        CalculationRequest.model_validate(request).serialization_profile
        == CANONICALIZATION_PROFILE
    )
    manifest = verify_shard(directory)
    assert "serialization_profile" not in manifest
    assert "scientific_cache_key" not in manifest
    assert manifest["request_sha256"] == digest(
        request, profile=LEGACY_CANONICALIZATION_PROFILE
    )
    assert manifest["request_sha256"] != digest(request)
    after = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert before == after


def test_previous_recipe_identity_cannot_be_replaced_by_a_different_recipe(tmp_path):
    directory = old_encoding_of_actual_failure(tmp_path)
    result = read_json(directory / "result.json")
    recipe = dict(result["recipe"])
    recipe.pop("recipe_sha256")
    manifest = read_json(directory / "manifest.json")
    assert manifest["recipe_sha256"] == digest(
        recipe, profile=LEGACY_CANONICALIZATION_PROFILE
    )
    # This blocked recipe contains no floats; RFC and legacy happen to agree.
    # A different actual registry recipe must still fail the original identity.
    manifest["recipe_sha256"] = get_profile("hf-sto-3g-education")["recipe_sha256"]
    (directory / "manifest.json").write_bytes(
        canonical_json(manifest, profile=LEGACY_CANONICALIZATION_PROFILE)
    )
    with pytest.raises(ValueError, match="recipe identity mismatch"):
        verify_shard(directory)
