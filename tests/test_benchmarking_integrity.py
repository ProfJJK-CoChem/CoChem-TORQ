"""Mathematical/design checks and genuine native-bundle scalar import.

Constructed design metadata are explicitly mathematical contract exercises,
never measured molecular references or an accepted scientific campaign. No
reference dataset, successful benchmark, engine output or external review is
fabricated. Native observation checks execute an actual HF calculation.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import shutil
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from pydantic import ValidationError

from cochem_torq.artifacts import file_digest
from cochem_torq.benchmarking import (
    AcceptedTarget,
    BenchmarkDesign,
    CurationAttestation,
    ObservableIdentity,
    ReferenceArtifact,
    ReferenceDatum,
    ReferenceManifest,
    SelectedReference,
    VerifiedReferences,
    _metrics,
    _read_frozen,
    _sampling_groups,
    freeze_benchmark,
    grouped_split_conformal,
    import_curated_references,
    import_publication_prediction,
    score_benchmark,
    seal_predictions,
)
from cochem_torq.domain import CalculationRequest, canonical_json, digest, read_json
from cochem_torq.publication import export_publication_bundle


def mathematical_identity(**changes):
    fields = {
        "parent_id": "mathematical-contract-subject-not-reference-evidence",
        "family_id": "mathematical-family",
        "symbols": ("H", "H"),
        "isotope_numbers": (1, 1),
        "charge": 0,
        "multiplicity": 1,
        "conformer": "single-linear-geometry",
        "electronic_state": "declared-ground-singlet",
        "vibrational_state": "equilibrium-not-v0",
        "tunneling_state": "not-applicable",
        "hamiltonian_convention": "Born-Oppenheimer-electronic-model",
        "observable": "electronic_energy",
        "component": "total_electronic_energy",
        "unit": "hartree",
    }
    fields.update(changes)
    return ObservableIdentity(**fields)


def mathematical_design(**changes):
    fields = {
        "benchmark_id": "mathematical-design-only-no-accuracy-claim",
        "proposal_sha256": sha256(b"mathematical proposal metadata").hexdigest(),
        "reference_manifest_sha256": sha256(
            b"deliberately absent reference rejection case"
        ).hexdigest(),
        "curation_sha256": sha256(
            b"deliberately absent curator rejection case"
        ).hexdigest(),
        "recipe_sha256": sha256(b"mathematical recipe metadata").hexdigest(),
        "source_code_sha256": sha256(b"mathematical source metadata").hexdigest(),
        "source_commit": "a" * 40,
        "selected_references": (
            SelectedReference(
                reference_id="mathematical-calibration-design",
                identity=mathematical_identity(),
                source_sha256=sha256(b"mathematical source A").hexdigest(),
                partition="calibration",
            ),
            SelectedReference(
                reference_id="mathematical-held-out-design",
                identity=mathematical_identity(
                    parent_id="mathematical-test-subject",
                    family_id="mathematical-test-family",
                ),
                source_sha256=sha256(b"mathematical source B").hexdigest(),
                partition="held_out",
            ),
        ),
        "targets": (
            AcceptedTarget(
                observable="electronic_energy",
                unit="hartree",
                maximum_absolute_error=0.1,
                rationale="Mathematical scale for design contract test only.",
            ),
        ),
        "accepted_by": "mathematical design contract test; no scientific acceptance",
        "accepted_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "acceptance_record_sha256": sha256(
            b"not-yet-bound mathematical acceptance bytes"
        ).hexdigest(),
        "protocol": "mathematical contract test only",
        "exchangeability_assessment": "not_established",
        "exchangeability_review": "No scientific exchangeability asserted.",
        "nominal_coverage": 0.95,
    }
    fields.update(changes)
    return BenchmarkDesign(**fields)


def actual_mathematical_freeze(tmp_path, **changes):
    design = mathematical_design(**changes)
    acceptance = canonical_json(
        {
            "design_sha256": digest(
                design.model_dump(mode="json", exclude={"acceptance_record_sha256"})
            ),
            "accepted_by": design.accepted_by,
            "purpose": "mathematical contract exercise; no scientific acceptance",
        }
    )
    design = BenchmarkDesign.model_validate(
        {
            **design.model_dump(),
            "acceptance_record_sha256": sha256(acceptance).hexdigest(),
        }
    )
    path = tmp_path / "mathematical-design-freeze.json"
    receipt = freeze_benchmark(design, acceptance, path)
    return path, receipt, design, acceptance


def test_cli_freezes_actual_mathematical_acceptance_bytes_without_scientific_claim(
    tmp_path, capsys
):
    from cochem_torq.cli import main

    _, _, design, acceptance = actual_mathematical_freeze(tmp_path)
    design_path = tmp_path / "mathematical-design-input.json"
    acceptance_path = tmp_path / "mathematical-acceptance-input.json"
    output = tmp_path / "separately-requested-mathematical-freeze.json"
    design_path.write_bytes(canonical_json(design.model_dump(mode="json")))
    acceptance_path.write_bytes(acceptance)
    args = [
        "benchmark-freeze",
        "--design",
        str(design_path),
        "--acceptance-record",
        str(acceptance_path),
        "--output",
        str(output),
        "--json",
    ]
    assert main(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["data"]["freeze_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert read_json(output)["experimental_accuracy_established"] is False
    assert main(args) == 5
    assert "FileExistsError" in capsys.readouterr().out


def test_cli_score_rejects_untrusted_seal_before_opening_absent_reference_sources(
    tmp_path, capsys
):
    from cochem_torq.cli import main

    freeze, receipt, _, _ = actual_mathematical_freeze(tmp_path)
    seal = tmp_path / "deliberately-incomplete-seal.json"
    seal.write_bytes(b"{}")
    output = tmp_path / "must-not-be-created.json"
    args = [
        "benchmark-score",
        "--freeze",
        str(freeze),
        "--expected-freeze-sha256",
        receipt["freeze_sha256"],
        "--prediction-seal",
        str(seal),
        "--expected-prediction-seal-sha256",
        sha256(b"other bytes").hexdigest(),
        "--reference-manifest",
        str(tmp_path / "absent-reference-manifest"),
        "--expected-reference-manifest-sha256",
        sha256(b"absent manifest").hexdigest(),
        "--curation",
        str(tmp_path / "absent-curation"),
        "--expected-curation-sha256",
        sha256(b"absent curation").hexdigest(),
        "--sources",
        str(tmp_path / "absent-source-locations"),
        "--output",
        str(output),
        "--json",
    ]
    assert main(args) == 2
    assert "Prediction-seal bytes differ" in capsys.readouterr().out
    assert not output.exists()


def test_cli_rejects_design_path_symlink_before_freezing(tmp_path, capsys):
    from cochem_torq.cli import main

    _, _, design, acceptance = actual_mathematical_freeze(tmp_path)
    target = tmp_path / "actual-mathematical-design.json"
    target.write_bytes(canonical_json(design.model_dump(mode="json")))
    alias = tmp_path / "design-alias.json"
    alias.symlink_to(target)
    acceptance_path = tmp_path / "mathematical-acceptance.json"
    acceptance_path.write_bytes(acceptance)
    output = tmp_path / "must-not-freeze.json"
    assert (
        main(
            [
                "benchmark-freeze",
                "--design",
                str(alias),
                "--acceptance-record",
                str(acceptance_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "symbolic links" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize(
    "count,rank,status",
    [
        (0, 1, "unavailable_insufficient_independent_calibration_groups"),
        (3, 4, "unavailable_insufficient_independent_calibration_groups"),
        (18, 19, "unavailable_insufficient_independent_calibration_groups"),
        (19, 19, "finite_mathematical_threshold"),
        (39, 38, "finite_mathematical_threshold"),
    ],
)
def test_exact_finite_sample_conformal_rank_without_quantile_interpolation(
    count, rank, status
):
    scores = {f"independent-mathematical-group-{i}": float(i + 1) for i in range(count)}
    result = grouped_split_conformal(scores, nominal_coverage=0.95)
    assert result["order_statistic_rank"] == rank
    assert result["status"] == status
    assert result["threshold"] == (float(rank) if rank <= count else None)
    assert result["coverage_claim"] is None
    assert not result["exchangeability_established"]


def test_identical_scores_are_valid_but_repeated_lines_do_not_increase_units():
    result = grouped_split_conformal(
        {"one-family-with-many-lines": 2.0}, nominal_coverage=0.95
    )
    assert result["calibration_group_count"] == 1
    assert result["threshold"] is None
    tied = grouped_split_conformal(
        {str(i): 2.0 for i in range(19)}, nominal_coverage=0.95
    )
    assert tied["threshold"] == 2.0


def test_parent_weighted_metrics_do_not_weight_many_lines_as_independent_parents():
    rows = [{"parent_id": "mathematical-A", "error": 1.0}] + [
        {"parent_id": "mathematical-B", "error": 3.0} for _ in range(100)
    ]
    metrics = _metrics(rows)
    assert metrics["parent_count"] == 2
    assert metrics["datum_count"] == 101
    assert metrics["parent_weighted_mae"] == 2.0
    assert metrics["parent_weighted_rmse"] == pytest.approx(math.sqrt(5.0))
    assert metrics["maximum_absolute_error"] == 3.0
    empty = _metrics([])
    assert empty["parent_weighted_mae"] is None
    assert empty["parent_weighted_rmse"] is None


def test_design_without_targets_or_matching_units_cannot_freeze():
    with pytest.raises(ValidationError):
        mathematical_design(targets=())
    wrong = AcceptedTarget(
        observable="electronic_energy",
        unit="MHz",
        maximum_absolute_error=0.1,
        rationale="Deliberate unit mismatch rejection case.",
    )
    with pytest.raises(ValidationError, match="units differ"):
        mathematical_design(targets=(wrong,))


@pytest.mark.parametrize("score", [math.nan, math.inf, -1.0, True, "1.0"])
def test_invalid_calibration_observations_are_not_repaired(score):
    with pytest.raises(ValueError, match="scores"):
        grouped_split_conformal({"mathematical-group": score}, nominal_coverage=0.95)


@pytest.mark.parametrize("coverage", [math.nan, math.inf, 0.0, 1.0, True])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValueError, match="coverage"):
        grouped_split_conformal({}, nominal_coverage=coverage)


@pytest.mark.parametrize("kind", ["family", "parent", "source"])
def test_family_parent_and_shared_source_leakage_is_rejected(kind):
    design = mathematical_design()
    calibration, held = design.selected_references
    change = {}
    if kind == "family":
        change["identity"] = mathematical_identity(
            parent_id="separate-parent", family_id=calibration.identity.family_id
        )
    elif kind == "parent":
        change["identity"] = mathematical_identity(
            parent_id=calibration.identity.parent_id, family_id="different-family"
        )
    else:
        change["source_sha256"] = calibration.source_sha256
    held = SelectedReference.model_validate({**held.model_dump(), **change})
    with pytest.raises(ValidationError, match="partitions"):
        mathematical_design(selected_references=(calibration, held))


def test_reference_source_connections_merge_family_sampling_units_transitively():
    selected = []
    for family, source in [
        ("a", "first"),
        ("b", "first"),
        ("b", "second"),
        ("c", "second"),
        ("d", "third"),
    ]:
        selected.append(
            SelectedReference(
                reference_id=f"{family}-{source}",
                identity=mathematical_identity(
                    parent_id=f"parent-{family}", family_id=family
                ),
                source_sha256=sha256(source.encode()).hexdigest(),
                partition="calibration",
            )
        )
    groups = _sampling_groups(selected)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_mathematical_design_freeze_is_atomic_immutable_and_bound_to_acceptance(
    tmp_path,
):
    path, receipt, design, acceptance = actual_mathematical_freeze(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert file_digest(path) == receipt["freeze_sha256"]
    loaded, frozen_at = _read_frozen(path, receipt["freeze_sha256"])
    assert loaded == design and frozen_at >= design.accepted_at
    assert not read_json(path)["experimental_accuracy_established"]
    with pytest.raises(FileExistsError):
        freeze_benchmark(design, acceptance, path)
    with pytest.raises(ValueError, match="digest"):
        _read_frozen(path, sha256(b"wrong independent digest").hexdigest())
    assert not list(tmp_path.glob(".torq-benchmark-*"))


def test_absent_wrong_acceptance_and_symlink_destinations_are_rejected(tmp_path):
    design = mathematical_design()
    with pytest.raises(ValueError, match="actual acceptance-record"):
        freeze_benchmark(design, b"wrong actual bytes", tmp_path / "absent.json")
    path, _, design, acceptance = actual_mathematical_freeze(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        freeze_benchmark(design, acceptance, link / "fresh.json")


def test_duplicate_acceptance_members_are_not_silently_last_wins(tmp_path):
    raw = b'{"accepted_by":"first","accepted_by":"second"}'
    design = mathematical_design(acceptance_record_sha256=sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_benchmark(design, raw, tmp_path / "duplicate.json")


def test_reference_curation_missing_actual_bytes_fails_without_reference_substitution():
    with pytest.raises(ValueError, match="independently retained"):
        import_curated_references(
            b"actual rejection input",
            {},
            b"actual rejection curation input",
            expected_manifest_sha256=sha256(b"absent external artifact").hexdigest(),
            expected_curation_sha256=sha256(b"absent external attestation").hexdigest(),
        )


def test_actual_unaccepted_proposal_bytes_cannot_be_imported_as_curated_references():
    from pathlib import Path

    raw = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/rotational-identification/preregistration.json"
    ).read_bytes()
    curation = b"{}"  # Actual invalid parser input, with no purported review.
    with pytest.raises(ValidationError):
        import_curated_references(
            raw,
            {},
            curation,
            expected_manifest_sha256=sha256(raw).hexdigest(),
            expected_curation_sha256=sha256(curation).hexdigest(),
        )


def test_duplicate_source_manifest_members_fail_before_curation_or_scoring():
    raw = b'{"sources":[],"sources":[]}'
    curation = b"{}"
    with pytest.raises(ValueError, match="Duplicate"):
        import_curated_references(
            raw,
            {},
            curation,
            expected_manifest_sha256=sha256(raw).hexdigest(),
            expected_curation_sha256=sha256(curation).hexdigest(),
        )


def test_scoring_rechecks_actual_reference_bytes_even_for_invalid_direct_objects(
    tmp_path,
):
    # Deliberately malformed direct objects contain no observation, source or
    # review; scoring must parse the supplied bytes instead of trusting them.
    actual_invalid_reference = b"{}"
    actual_invalid_curation = b"{}"
    path, receipt, _, _ = actual_mathematical_freeze(
        tmp_path,
        reference_manifest_sha256=sha256(actual_invalid_reference).hexdigest(),
        curation_sha256=sha256(actual_invalid_curation).hexdigest(),
    )
    invalid = VerifiedReferences(
        manifest=ReferenceManifest.model_construct(),
        manifest_sha256=sha256(actual_invalid_reference).hexdigest(),
        curation=CurationAttestation.model_construct(),
        curation_sha256=sha256(actual_invalid_curation).hexdigest(),
        source_bytes=(),
        manifest_bytes=actual_invalid_reference,
        curation_bytes=actual_invalid_curation,
    )
    with pytest.raises(ValidationError):
        score_benchmark(
            path,
            tmp_path / "no-calculation-evidence.json",
            invalid,
            expected_freeze_sha256=receipt["freeze_sha256"],
            expected_prediction_seal_sha256=sha256(b"absent evidence").hexdigest(),
            destination=tmp_path / "cannot-score.json",
        )
    assert not (tmp_path / "cannot-score.json").exists()


def test_missing_uncertainty_and_predicted_catalogue_origin_cannot_be_promoted():
    identity = mathematical_identity()
    datum_fields = {
        "reference_id": "deliberately-invalid-reference-rejection-case",
        "identity": identity,
        "value": -1.0,
        "standard_uncertainty": None,
        "uncertainty_status": "unavailable",
        "covariance_status": "unavailable",
        "source_sha256": sha256(b"mathematical source metadata only").hexdigest(),
        "datum_locator": "rejection metadata; no published observation",
    }
    with pytest.raises(ValidationError, match="uncertainty"):
        ReferenceDatum(**{**datum_fields, "standard_uncertainty": 0.1})
    source = ReferenceArtifact(
        source_sha256=datum_fields["source_sha256"],
        citation="deliberately predicted reference rejection case",
        persistent_identifier="https://example.org/deliberately-invalid-reference",
        version="rejection metadata",
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reuse_permission="not asserted",
        origin="catalogue_prediction",
    )
    with pytest.raises(ValidationError, match="independent observations"):
        ReferenceManifest(sources=(source,), datums=(ReferenceDatum(**datum_fields),))


def test_boolean_isotope_and_inconsistent_spin_identity_are_rejected():
    with pytest.raises(ValidationError):
        mathematical_identity(isotope_numbers=(True, 1))
    with pytest.raises(ValidationError, match="electron count"):
        mathematical_identity(multiplicity=2)


@pytest.fixture(scope="module")
def genuine_hf_benchmark_bundle(tmp_path_factory):
    from cochem_torq.application import execute_request

    root = tmp_path_factory.mktemp("genuine-hf-benchmark-import")
    identity = mathematical_identity(
        parent_id="hydrogen", family_id="homonuclear-diatomics"
    )
    labels = identity.model_dump(
        mode="json",
        include={
            "parent_id",
            "family_id",
            "conformer",
            "electronic_state",
            "vibrational_state",
            "tunneling_state",
            "hamiltonian_convention",
        },
    )
    request = CalculationRequest.model_validate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
                "charge": 0,
                "multiplicity": 1,
                "isotopes": [1, 1],
            },
            "recipe": "hf-sto-3g-education",
            "products": ["geometry", "harmonic", "equilibrium_constants"],
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
            "source_provenance": {
                "benchmark_identity": labels,
                "purpose": "native prediction import check; no experimental accuracy",
            },
        }
    )
    result = execute_request(request, root / "shard")
    assert result["status"] == "complete", result["errors"]
    receipt = export_publication_bundle(root / "shard", root / "publication")
    return root / "publication", receipt["manifest_sha256"], identity, result


@pytest.mark.real_engine
def test_prediction_import_reads_actual_native_energy_without_accuracy_claim(
    genuine_hf_benchmark_bundle,
):
    bundle, trusted_digest, identity, result = genuine_hf_benchmark_bundle
    imported = import_publication_prediction(
        bundle, expected_manifest_sha256=trusted_digest, identity=identity
    )
    assert (
        imported["value"]
        == result["stages"]["electronic_structure"]["value"]["energy_hartree"]
    )
    assert imported["status"] == "available"
    assert imported["recipe_sha256"] == result["recipe_sha256"]
    assert imported["source_code_sha256"] == result["source_identity"]["code_sha256"]
    assert not read_json(bundle / "publication.json")[
        "experimental_accuracy_established"
    ]


@pytest.mark.real_engine
def test_actual_rotational_axis_and_missing_b0_are_distinguished(
    genuine_hf_benchmark_bundle,
):
    bundle, trusted_digest, identity, result = genuine_hf_benchmark_bundle
    base = identity.model_dump()
    for component in ("A", "B", "C"):
        actual = ObservableIdentity.model_validate(
            {
                **base,
                "observable": "equilibrium_rotational_constant",
                "unit": "MHz",
                "component": component,
            }
        )
        imported = import_publication_prediction(
            bundle, expected_manifest_sha256=trusted_digest, identity=actual
        )
        expected = result["stages"]["equilibrium_constants"]["value"]["constants_mhz"][
            "ABC".index(component)
        ]
        assert imported["value"] == expected
        assert imported["status"] == (
            "unavailable" if expected is None else "available"
        )
    actual = ObservableIdentity.model_validate(
        {
            **base,
            "observable": "ground_state_rotational_constant",
            "unit": "MHz",
            "component": "B",
        }
    )
    imported = import_publication_prediction(
        bundle, expected_manifest_sha256=trusted_digest, identity=actual
    )
    assert imported["value"] is None and imported["reason"]


@pytest.mark.real_engine
def test_actual_bundle_wrong_isotope_and_labels_are_rejected(
    genuine_hf_benchmark_bundle,
):
    bundle, trusted_digest, identity, _ = genuine_hf_benchmark_bundle
    for changes, message in [
        ({"isotope_numbers": (2, 2)}, "isotopes"),
        ({"electronic_state": "uncomputed excited state"}, "labels"),
    ]:
        wrong = ObservableIdentity.model_validate({**identity.model_dump(), **changes})
        with pytest.raises(ValueError, match=message):
            import_publication_prediction(
                bundle, expected_manifest_sha256=trusted_digest, identity=wrong
            )


@pytest.mark.real_engine
def test_extraction_reverification_follows_reads_and_rejects_changed_actual_bytes(
    genuine_hf_benchmark_bundle, tmp_path
):
    # No injected provider or race scheduler. Check the actual guard's position,
    # then change real verified native-bundle bytes. Fail closed without claiming
    # that this test observed a timed concurrent race.
    function = ast.parse(inspect.getsource(import_publication_prediction)).body[0]
    assert isinstance(function, ast.FunctionDef)
    guard = function.body[-2]
    assert isinstance(guard, ast.If)
    calls = [
        node
        for node in ast.walk(guard)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_publication_bundle"
    ]
    assert len(calls) == 1
    assert isinstance(function.body[-1], ast.Return)
    bundle, trusted_digest, identity, _ = genuine_hf_benchmark_bundle
    unchanged = import_publication_prediction(
        bundle, expected_manifest_sha256=trusted_digest, identity=identity
    )
    assert unchanged["status"] == "available"
    changed = tmp_path / "actual-native-bundle-then-damaged"
    shutil.copytree(bundle, changed)
    with (changed / "shard/result.json").open("ab") as stream:
        stream.write(b"\n")  # Same JSON semantics; actual sealed bytes changed.
    with pytest.raises(ValueError, match="inventory/hash mismatch"):
        import_publication_prediction(
            changed, expected_manifest_sha256=trusted_digest, identity=identity
        )


def test_absent_calculation_evidence_cannot_complete_a_frozen_design(tmp_path):
    path, receipt, _, _ = actual_mathematical_freeze(tmp_path)
    with pytest.raises(ValueError, match="actual success/failure evidence"):
        seal_predictions(
            path,
            expected_freeze_sha256=receipt["freeze_sha256"],
            bundles={},
            destination=tmp_path / "missing-predictions.json",
        )
    assert not (tmp_path / "missing-predictions.json").exists()
