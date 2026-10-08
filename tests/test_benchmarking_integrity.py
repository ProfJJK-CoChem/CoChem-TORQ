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
    RigidRotorSelection,
    RotationalLineSelection,
    SelectedReference,
    TheoreticalReferenceProvenance,
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


def mathematical_theoretical_provenance(**changes):
    fields = {
        "method": "mathematical metadata contract; not a published method record",
        "basis": "mathematical metadata contract; not a published basis record",
        "method_details": "No physical or independently validated reference asserted.",
        "geometry_description": "Mathematical context-only geometry declaration.",
        "geometry_source_locator": "mathematical contract; no experimental source",
        "engine_information_status": "not_reported",
        "independence_status": "not_established",
        "independence_review": "Mathematical contract; no independence asserted.",
        "quantity_convention": "Mathematical schema example only.",
    }
    fields.update(changes)
    return TheoreticalReferenceProvenance(**fields)


def mathematical_reference_manifest(identity, **changes):
    """Schema-only metadata: no source acquisition, measurement or review claim."""
    source_digest = sha256(
        b"mathematical metadata; not a source observation"
    ).hexdigest()
    origin = changes.pop("origin", "published_theoretical")
    fields = {
        "reference_id": "mathematical-origin-contract-only",
        "identity": identity,
        "value": 1.0,
        "uncertainty_status": "unavailable",
        "covariance_status": "unavailable",
        "source_sha256": source_digest,
        "datum_locator": "mathematical contract; no extracted physical datum",
        "theoretical_provenance": (
            mathematical_theoretical_provenance()
            if origin == "published_theoretical"
            else None
        ),
    }
    fields.update(changes)
    source = ReferenceArtifact(
        source_sha256=source_digest,
        citation="Mathematical metadata only; not a scientific source citation",
        persistent_identifier="https://example.org/mathematical-metadata-only",
        version="mathematical schema contract",
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reuse_permission="No acquired source or reuse permission asserted",
        origin=origin,
    )
    return ReferenceManifest(sources=(source,), datums=(ReferenceDatum(**fields),))


def test_theoretical_origin_requires_per_datum_computational_context():
    manifest = mathematical_reference_manifest(mathematical_identity())
    assert manifest.sources[0].origin == "published_theoretical"
    assert (
        manifest.datums[0].theoretical_provenance.independence_status
        == "not_established"
    )
    with pytest.raises(ValidationError, match="computational provenance"):
        mathematical_reference_manifest(
            mathematical_identity(), theoretical_provenance=None
        )
    with pytest.raises(ValidationError, match="observed references"):
        mathematical_reference_manifest(
            mathematical_identity(),
            origin="measured",
            theoretical_provenance=mathematical_theoretical_provenance(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"engine_information_status": "reported"},
        {"engine_information_status": "partial"},
        {"engine": "undeclared engine"},
        {"engine_version": "undeclared version"},
    ],
)
def test_missing_engine_facts_are_not_invented(changes):
    with pytest.raises(ValidationError, match="absent engine information"):
        mathematical_theoretical_provenance(**changes)


def test_reported_computational_uncertainty_cannot_be_measurement_uncertainty():
    with pytest.raises(ValidationError, match="actual origin"):
        mathematical_reference_manifest(
            mathematical_identity(),
            standard_uncertainty=0.1,
            uncertainty_status="standard_uncertainty",
        )
    manifest = mathematical_reference_manifest(
        mathematical_identity(),
        standard_uncertainty=0.1,
        uncertainty_status="standard_uncertainty",
        uncertainty_kind="reported_computational_standard_uncertainty",
    )
    assert (
        manifest.datums[0].uncertainty_kind
        == "reported_computational_standard_uncertainty"
    )


def test_harmonic_and_equilibrium_references_cannot_relabel_observed_quantities():
    harmonic = mathematical_identity(
        observable="harmonic_frequency",
        component="mode:0",
        unit="cm^-1",
        harmonic_mode_index=0,
        harmonic_mode_assignment="mathematical one-mode selection; not source evidence",
    )
    equilibrium = mathematical_identity(
        observable="equilibrium_rotational_constant",
        component="B",
        unit="MHz",
    )
    for identity in (harmonic, equilibrium):
        assert mathematical_reference_manifest(identity).datums[0].identity == identity
        with pytest.raises(ValidationError, match="B0/fundamentals"):
            mathematical_reference_manifest(identity, origin="measured")
        assert (
            mathematical_reference_manifest(
                identity,
                origin="semi_experimental",
                correction_sources=("Mathematical correction metadata only",),
            )
            .sources[0]
            .origin
            == "semi_experimental"
        )


def test_harmonic_mode_requires_explicit_mapping_and_exact_index():
    fields = dict(observable="harmonic_frequency", component="mode:0", unit="cm^-1")
    with pytest.raises(ValidationError, match="mode index"):
        mathematical_identity(**fields)
    with pytest.raises(ValidationError, match="mode index"):
        mathematical_identity(
            **fields,
            harmonic_mode_index=1,
            harmonic_mode_assignment="mathematical declared assignment",
        )
    with pytest.raises(ValidationError, match="another observable"):
        mathematical_identity(harmonic_mode_index=0)


def test_screening_transition_and_intensity_conventions_are_explicit():
    line = RotationalLineSelection(
        upper_J=1,
        upper_eigenstate_index=0,
        lower_J=0,
        lower_eigenstate_index=0,
    )
    selection = RigidRotorSelection(
        transition=line,
        constant_observable="Be",
        temperature_kelvin=10.0,
    )
    fields = dict(
        observable="rigid_rotor_transition",
        component="frequency",
        unit="MHz",
        hamiltonian_convention=selection.model_identity,
        rigid_rotor_selection=selection,
    )
    identity = mathematical_identity(**fields)
    with pytest.raises(ValidationError, match="theoretical reference"):
        mathematical_reference_manifest(identity, origin="measured")
    with pytest.raises(ValidationError, match="normalization transition"):
        mathematical_identity(
            **{
                **fields,
                "observable": "rigid_rotor_relative_intensity",
                "component": "relative_absorption_weight_ratio",
                "unit": "dimensionless",
            }
        )
    with pytest.raises(ValidationError, match="identification product"):
        mathematical_identity(**{**fields, "observable": "rotational_transition"})
    with pytest.raises(ValidationError, match="finite-J basis"):
        RotationalLineSelection(
            upper_J=1,
            upper_eigenstate_index=3,
            lower_J=0,
            lower_eigenstate_index=0,
        )


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
    provenance = imported["calculation_provenance"]
    assert provenance["recipe"]["method"] == "hf"
    assert provenance["recipe"]["basis"] == "sto-3g"
    assert provenance["calculated_geometry_sha256"] == digest(
        result["native_result"]["geometry_bohr"]
    )
    assert provenance["engine_version"] == result["native_result"]["engine_version"]


@pytest.mark.real_engine
def test_actual_harmonic_mode_import_retains_native_mode_and_approximation(
    genuine_hf_benchmark_bundle,
):
    bundle, trusted_digest, identity, result = genuine_hf_benchmark_bundle
    harmonic_identity = ObservableIdentity.model_validate(
        {
            **identity.model_dump(),
            "observable": "harmonic_frequency",
            "component": "mode:0",
            "unit": "cm^-1",
            "harmonic_mode_index": 0,
            "harmonic_mode_assignment": "The sole actual H2 stretching eigenmode.",
        }
    )
    imported = import_publication_prediction(
        bundle,
        expected_manifest_sha256=trusted_digest,
        identity=harmonic_identity,
    )
    harmonic = result["stages"]["harmonic_analysis"]["value"]
    assert imported["value"] == harmonic["frequencies_cm1"][0]
    assert imported["approximation_context"]["frequency_kind"] == "harmonic"
    assert (
        imported["approximation_context"]["normal_mode_source_digest"]
        == harmonic["source_digest"]
    )
    absent = ObservableIdentity.model_validate(
        {
            **harmonic_identity.model_dump(),
            "component": "mode:9",
            "harmonic_mode_index": 9,
        }
    )
    with pytest.raises(ValueError, match="eigenmode does not exist"):
        import_publication_prediction(
            bundle,
            expected_manifest_sha256=trusted_digest,
            identity=absent,
        )


@pytest.fixture(scope="module")
def genuine_water_catalog_bundle(tmp_path_factory):
    from cochem_torq.application import execute_request

    root = tmp_path_factory.mktemp("genuine-water-catalog-benchmark-import")
    identity = mathematical_identity(
        parent_id="water",
        family_id="hydrides",
        symbols=("O", "H", "H"),
        isotope_numbers=(16, 1, 1),
        hamiltonian_convention=(
            "TORQ exact finite-J electric-dipole rigid-rotor screening model v1"
        ),
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
                "symbols": ["O", "H", "H"],
                "geometry_bohr": [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.43, 1.11],
                    [0.0, -1.43, 1.11],
                ],
                "charge": 0,
                "multiplicity": 1,
                "isotopes": [16, 1, 1],
            },
            "recipe": "hf-sto-3g-education",
            "products": [
                "geometry",
                "harmonic",
                "equilibrium_constants",
                "rigid_rotor_catalog",
            ],
            "catalog": {"temperature_kelvin": 10.0, "max_j": 5},
            "resources": {"cores": 1, "memory_mb": 1024, "wall_seconds": 120},
            "source_provenance": {
                "benchmark_identity": labels,
                "purpose": (
                    "Native screening extraction check; no reference or accuracy claim."
                ),
            },
        }
    )
    result = execute_request(request, root / "shard")
    assert result["stages"]["rigid_rotor_catalog"]["status"] == "available", result[
        "errors"
    ]
    receipt = export_publication_bundle(root / "shard", root / "publication")
    return root / "publication", receipt["manifest_sha256"], identity, result


@pytest.mark.real_engine
def test_actual_rigid_rotor_frequency_preserves_exact_states_and_model(
    genuine_water_catalog_bundle,
):
    bundle, trusted_digest, identity, result = genuine_water_catalog_bundle
    catalog = result["stages"]["rigid_rotor_catalog"]["value"]
    line = catalog["lines"][0]
    selection = RigidRotorSelection(
        transition=RotationalLineSelection(
            **{
                k: line[k]
                for k in (
                    "upper_J",
                    "upper_eigenstate_index",
                    "lower_J",
                    "lower_eigenstate_index",
                )
            }
        ),
        constant_observable="Be",
        temperature_kelvin=catalog["temperature_kelvin"],
    )
    chosen = ObservableIdentity.model_validate(
        {
            **identity.model_dump(),
            "observable": "rigid_rotor_transition",
            "component": "frequency",
            "unit": "MHz",
            "rigid_rotor_selection": selection,
        }
    )
    imported = import_publication_prediction(
        bundle,
        expected_manifest_sha256=trusted_digest,
        identity=chosen,
    )
    assert imported["value"] == line["frequency_mhz"]
    assert imported["approximation_context"]["constant_observable"] == "Be"
    assert imported["approximation_context"]["identification_qualified"] is False
    wrong_basis = ObservableIdentity.model_validate(
        {
            **chosen.model_dump(),
            "rigid_rotor_selection": {
                **selection.model_dump(),
                "constant_observable": "B0",
            },
        }
    )
    with pytest.raises(ValueError, match="population conventions differ"):
        import_publication_prediction(
            bundle,
            expected_manifest_sha256=trusted_digest,
            identity=wrong_basis,
        )


@pytest.mark.real_engine
def test_actual_rigid_rotor_intensity_is_normalized_with_population_conventions(
    genuine_water_catalog_bundle,
):
    bundle, trusted_digest, identity, result = genuine_water_catalog_bundle
    catalog = result["stages"]["rigid_rotor_catalog"]["value"]
    positive = [
        line
        for line in catalog["lines"]
        if line["relative_absorption_weight_debye2"] > 0
    ]
    numerator, denominator = positive[:2]
    names = ("upper_J", "upper_eigenstate_index", "lower_J", "lower_eigenstate_index")
    selection = RigidRotorSelection(
        transition=RotationalLineSelection(**{k: numerator[k] for k in names}),
        normalization_transition=RotationalLineSelection(
            **{k: denominator[k] for k in names}
        ),
        constant_observable="Be",
        temperature_kelvin=catalog["temperature_kelvin"],
    )
    chosen = ObservableIdentity.model_validate(
        {
            **identity.model_dump(),
            "observable": "rigid_rotor_relative_intensity",
            "component": "relative_absorption_weight_ratio",
            "unit": "dimensionless",
            "rigid_rotor_selection": selection,
        }
    )
    imported = import_publication_prediction(
        bundle,
        expected_manifest_sha256=trusted_digest,
        identity=chosen,
    )
    if catalog["partition_converged_at_requested_tolerance"]:
        assert imported["value"] == (
            numerator["relative_absorption_weight_debye2"]
            / denominator["relative_absorption_weight_debye2"]
        )
    else:
        assert imported["value"] is None and "did not converge" in imported["reason"]
    wrong_temperature = ObservableIdentity.model_validate(
        {
            **chosen.model_dump(),
            "rigid_rotor_selection": {
                **selection.model_dump(),
                "temperature_kelvin": 11.0,
            },
        }
    )
    with pytest.raises(ValueError, match="population conventions differ"):
        import_publication_prediction(
            bundle,
            expected_manifest_sha256=trusted_digest,
            identity=wrong_temperature,
        )


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
