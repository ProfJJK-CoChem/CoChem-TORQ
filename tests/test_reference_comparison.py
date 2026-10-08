"""Descriptive arithmetic and provenance checks, without invented publications.

Mathematical extraction declarations exercise only contracts. Actual publication
comparison checks below use retained source bytes and a real native calculation;
no engine, measured value or independent human review is synthesized.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from cochem_torq.domain import CalculationRequest, canonical_json, digest, read_json
from cochem_torq.reference_comparison import (
    AutomatedDatumExtraction,
    AutomatedReferenceReview,
    compare_published_reference,
    descriptive_residuals,
    import_comparison_references,
    load_comparison_references,
)


def mathematical_extraction(**changes):
    fields = {
        "reference_id": "mathematical-conversion-contract-no-published-reference",
        "source_sha256": sha256(b"mathematical conversion source").hexdigest(),
        "datum_locator": "mathematical scalar conversion check",
        "identity_sha256": sha256(b"mathematical scalar identity").hexdigest(),
        "value": 4000.0,
        "unit": "MHz",
        "standard_uncertainty": None,
        "source_value": 4.0,
        "source_unit": "GHz",
        "unit_conversion_multiplier": 1000.0,
        "source_standard_uncertainty": None,
        "source_value_literal": "4.0",
        "source_value_token_index": 0,
        "extraction_method": "mathematical unit-conversion contract exercise only",
        "source_excerpt_utf8": "4.0 GHz",
    }
    fields.update(changes)
    return AutomatedDatumExtraction(**fields)


def test_signed_descriptive_residuals_preserve_missing_uncertainty():
    result = descriptive_residuals(11.0, 10.0, None)
    assert result["signed_residual"] == 1.0
    assert result["absolute_residual"] == 1.0
    assert result["relative_residual"] == 0.1
    assert result["relative_residual_ppm"] == 100000.0
    assert result["residual_over_reference_standard_uncertainty"] is None
    assert "unavailable" in result["uncertainty_ratio_interpretation"]


def test_negative_electronic_energy_has_explicit_relative_denominator():
    result = descriptive_residuals(-11.0, -10.0, 0.5)
    assert result["signed_residual"] == -1.0
    assert result["relative_residual"] == -0.1
    assert result["residual_over_reference_standard_uncertainty"] == -2.0
    assert (
        "prediction uncertainty and covariance are absent"
        in result["uncertainty_ratio_interpretation"]
    )


def test_zero_reference_keeps_relative_residual_unavailable():
    result = descriptive_residuals(1.0, 0.0, None)
    assert result["signed_residual"] == 1.0
    assert result["relative_residual"] is None
    assert result["relative_residual_ppm"] is None
    assert result["relative_residual_absence_reason"] == "Reference value is zero."


@pytest.mark.parametrize("bad", [True, None, "1", math.nan, math.inf, -math.inf])
def test_nonfinite_or_non_numeric_residuals_are_rejected(bad):
    with pytest.raises(ValueError, match="finite genuine numerical"):
        descriptive_residuals(bad, 1.0, None)
    with pytest.raises(ValueError, match="finite genuine numerical"):
        descriptive_residuals(1.0, bad, None)


@pytest.mark.parametrize("bad", [True, "1", 0.0, -1.0, math.nan, math.inf])
def test_missing_uncertainty_cannot_be_replaced_with_invalid_values(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        descriptive_residuals(2.0, 1.0, bad)


@pytest.mark.parametrize(
    "prediction,reference,uncertainty",
    [(1e308, -1e308, None), (1.0, 1e-308, None), (2.0, 1.0, 1e-320)],
)
def test_residual_overflow_is_rejected_before_a_nonfinite_report(
    prediction, reference, uncertainty
):
    with pytest.raises(ValueError, match="overflow"):
        descriptive_residuals(prediction, reference, uncertainty)


def test_exact_unit_conversion_preserves_unknown_source_uncertainty():
    extraction = mathematical_extraction()
    assert extraction.value == extraction.source_value * 1000.0
    assert extraction.standard_uncertainty is None
    assert extraction.source_standard_uncertainty is None


def test_reported_uncertainty_requires_the_same_source_unit_conversion():
    record = mathematical_extraction(
        source_standard_uncertainty=0.001,
        standard_uncertainty=1.0,
        source_uncertainty_literal="0.001",
        source_uncertainty_token_index=1,
        source_excerpt_utf8="4.0 GHz +/- 0.001 GHz",
    )
    assert record.standard_uncertainty == 1.0
    with pytest.raises(ValidationError, match="Source uncertainty must equal"):
        mathematical_extraction(source_standard_uncertainty=0.001)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"unit_conversion_multiplier": 1.0}, "exact declared unit conversion"),
        ({"value": 4.0}, "differs from source conversion"),
        ({"source_unit": "cm^-1"}, "exact declared unit conversion"),
        ({"standard_uncertainty": 0.1}, "uncertainty differs"),
    ],
)
def test_unimplemented_or_inconsistent_source_conversions_are_rejected(change, reason):
    with pytest.raises(ValidationError, match=reason):
        mathematical_extraction(**change)


def test_automated_review_cannot_claim_human_or_independent_review():
    fields = {
        "reference_manifest_sha256": sha256(b"mathematical manifest").hexdigest(),
        "extractor": "mathematical contract exercise only",
        "reviewed_at": datetime.now(timezone.utc),
        "review_record": "mathematical metadata; no publication or human review",
        "entries": (mathematical_extraction(),),
    }
    review = AutomatedReferenceReview(**fields)
    assert not review.human_reviewed
    assert not review.independent_curation_completed
    for forbidden in ("human_reviewed", "independent_curation_completed"):
        with pytest.raises(ValidationError):
            AutomatedReferenceReview(**fields, **{forbidden: True})
    with pytest.raises(ValidationError, match="future extraction"):
        AutomatedReferenceReview(
            **{**fields, "reviewed_at": datetime.now(timezone.utc) + timedelta(days=1)}
        )
    with pytest.raises(ValidationError, match="aware machine"):
        AutomatedReferenceReview(**{**fields, "reviewed_at": datetime(2026, 1, 1)})


def test_untrusted_reference_bytes_fail_before_parsing_or_source_substitution():
    with pytest.raises(ValueError, match="differ from the supplied digests"):
        import_comparison_references(
            b"{}",
            {},
            b"{}",
            expected_manifest_sha256=sha256(b"different real bytes").hexdigest(),
            expected_review_sha256=sha256(b"{}").hexdigest(),
        )


def test_missing_source_locations_remain_missing(tmp_path):
    with pytest.raises(ValueError, match="actual regular files"):
        load_comparison_references(
            tmp_path / "missing-reference-manifest",
            tmp_path / "missing-review",
            tmp_path / "missing-source-locations",
            expected_manifest_sha256=sha256(b"missing manifest").hexdigest(),
            expected_review_sha256=sha256(b"missing review").hexdigest(),
        )


def test_source_map_symlink_is_refused(tmp_path):
    actual = tmp_path / "actual-location-map"
    actual.write_bytes(b"{}")
    linked = tmp_path / "linked-location-map"
    linked.symlink_to(actual)
    with pytest.raises(ValueError, match="symbolic links"):
        load_comparison_references(
            tmp_path / "missing-reference-manifest",
            tmp_path / "missing-review",
            linked,
            expected_manifest_sha256=sha256(b"missing manifest").hexdigest(),
            expected_review_sha256=sha256(b"missing review").hexdigest(),
        )


def test_actual_numeric_notation_supports_fortran_and_explicit_minus():
    record = mathematical_extraction(
        value=4000.0,
        source_value_literal="4.0D+00",
        source_excerpt_utf8="4.0D+00 GHz",
    )
    assert record.source_value == 4.0
    negative = mathematical_extraction(
        value=-4000.0,
        source_value=-4.0,
        source_value_literal="\N{MINUS SIGN}4.0d+00",
        source_excerpt_utf8="\N{MINUS SIGN}4.0d+00 GHz",
    )
    assert negative.source_value == -4.0


@pytest.fixture(scope="module")
def actual_qm9_water_references():
    packet = (
        Path(__file__).resolve().parents[1] / "benchmarks/published-values/qm9-water"
    )
    pins = read_json(packet / "pins.json")
    references = load_comparison_references(
        packet / "references.json",
        packet / "automated-review.json",
        packet / "source-locations.json",
        expected_manifest_sha256=pins["reference_manifest_sha256"],
        expected_review_sha256=pins["review_sha256"],
    )
    return packet, pins, references


def altered_actual_review(references, update, *, manifest=None, sources=None):
    """Corrupt actual published declarations for rejection; never supply evidence."""
    manifest_data = json.loads(references.manifest_bytes)
    review_data = json.loads(references.review_bytes)
    if manifest is not None:
        manifest(manifest_data)
    manifest_bytes = canonical_json(manifest_data) + b"\n"
    manifest_sha = sha256(manifest_bytes).hexdigest()
    review_data["reference_manifest_sha256"] = manifest_sha
    update(review_data)
    review_bytes = canonical_json(review_data) + b"\n"
    return import_comparison_references(
        manifest_bytes,
        dict(references.source_bytes) if sources is None else sources,
        review_bytes,
        expected_manifest_sha256=manifest_sha,
        expected_review_sha256=sha256(review_bytes).hexdigest(),
    )


def test_actual_publisher_water_values_have_bound_raw_bytes_and_unknown_uncertainty(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references
    assert len(refs.manifest.datums) == 6
    assert len(refs.source_bytes) == 3
    assert refs.review_kind == "automated_extraction"
    assert all(datum.standard_uncertainty is None for datum in refs.manifest.datums)
    assert all(
        source.origin == "published_theoretical" for source in refs.manifest.sources
    )
    review = AutomatedReferenceReview.model_validate_json(refs.review_bytes)
    assert all(
        entry.raw_source_format == "qm9_extended_xyz" for entry in review.entries
    )
    assert not review.human_reviewed
    assert not review.independent_curation_completed


def test_unquoted_arbitrary_value_cannot_be_self_attested_from_real_source(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references

    def changed(review):
        review["entries"][0]["source_value"] += 1.0
        review["entries"][0]["value"] += 1000.0
        review["entries"][0]["source_value_literal"] = str(
            review["entries"][0]["source_value"]
        )

    with pytest.raises(ValueError, match="exact selected numeric token"):
        altered_actual_review(refs, changed)


def test_actual_b_value_cannot_be_mislabeled_as_a_despite_valid_numeric_tokens(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references
    review = json.loads(refs.review_bytes)
    b_record = review["entries"][1]

    def changed(record):
        for key in (
            "source_value",
            "value",
            "source_value_literal",
            "source_value_token_index",
        ):
            record["entries"][0][key] = b_record[key]

    def manifest_changed(manifest):
        manifest["datums"][0]["value"] = b_record["value"]

    with pytest.raises(ValueError, match="exact original property column"):
        altered_actual_review(refs, changed, manifest=manifest_changed)


def test_actual_frequency_cannot_be_reassigned_to_a_different_sorted_rank(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references
    review = json.loads(refs.review_bytes)
    rank_two = review["entries"][5]

    def changed(record):
        for key in (
            "source_value",
            "value",
            "source_value_literal",
            "source_value_token_index",
        ):
            record["entries"][3][key] = rank_two[key]

    def manifest_changed(manifest):
        manifest["datums"][3]["value"] = rank_two["value"]

    with pytest.raises(ValueError, match="exact original row/rank"):
        altered_actual_review(refs, changed, manifest=manifest_changed)


def test_quote_and_context_cannot_be_dropped_from_actual_reference_evidence(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references

    def changed(record):
        record["entries"][0]["source_excerpt_utf8"] += " unretained text"

    with pytest.raises(ValueError, match="excerpt is absent"):
        altered_actual_review(refs, changed)
    actual_sources = dict(refs.source_bytes)
    context_digest = json.loads(refs.review_bytes)["context_sources"][0][
        "source_sha256"
    ]
    del actual_sources[context_digest]
    with pytest.raises(ValueError, match="All and only inventoried original"):
        altered_actual_review(refs, lambda _: None, sources=actual_sources)


def test_actual_source_bytes_cannot_be_replaced_by_altered_source_bytes(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references
    sources = dict(refs.source_bytes)
    digest_key = refs.manifest.sources[0].source_sha256
    sources[digest_key] += b"\n"
    with pytest.raises(ValueError, match="source artifact digest mismatch"):
        altered_actual_review(refs, lambda _: None, sources=sources)


def test_actual_qm9_cannot_bypass_column_validation_by_claiming_generic_text(
    actual_qm9_water_references,
):
    _, _, refs = actual_qm9_water_references

    def changed(record):
        record["entries"][0]["raw_source_format"] = "utf8_numeric_excerpt"

    with pytest.raises(ValueError, match="QM9 source requires its structured locator"):
        altered_actual_review(refs, changed)


@pytest.fixture(scope="module")
def actual_water_publication_bundle(actual_qm9_water_references, tmp_path_factory):
    from cochem_torq.application import execute_request
    from cochem_torq.publication import export_publication_bundle

    packet, pins, _ = actual_qm9_water_references
    request_bytes = (packet / "request.json").read_bytes()
    assert sha256(request_bytes).hexdigest() == pins["request_file_sha256"]
    request = CalculationRequest.model_validate_json(request_bytes)
    assert digest(request.model_dump(mode="json")) == pins["canonical_request_sha256"]
    root = tmp_path_factory.mktemp("actual-water-published-reference-comparison")
    result = execute_request(request, root / "shard")
    assert result["status"] == "complete", result["errors"]
    receipt = export_publication_bundle(root / "shard", root / "publication")
    return root / "publication", receipt["manifest_sha256"], result


@pytest.mark.real_engine
@pytest.mark.parametrize("index", range(6))
def test_genuine_native_water_compares_to_actual_publication_with_explicit_limits(
    actual_qm9_water_references, actual_water_publication_bundle, tmp_path, index
):
    _, _, references = actual_qm9_water_references
    bundle, trusted_manifest, native = actual_water_publication_bundle
    datum = references.manifest.datums[index]
    destination = tmp_path / "actual-descriptive-reference-comparison.json"
    receipt = compare_published_reference(
        references,
        datum.reference_id,
        bundle,
        expected_publication_manifest_sha256=trusted_manifest,
        destination=destination,
    )
    report = read_json(destination)
    assert receipt["status"] == "available"
    assert sha256(destination.read_bytes()).hexdigest() == receipt["comparison_sha256"]
    assert destination.stat().st_mode & 0o777 == 0o400
    if index < 3:
        observed = native["stages"]["equilibrium_constants"]["value"]["constants_mhz"][
            index
        ]
    else:
        observed = native["stages"]["harmonic_analysis"]["value"]["frequencies_cm1"][
            index - 3
        ]
    assert report["prediction"]["value"] == observed
    assert report["reference"]["value"] == datum.value
    assert report["residuals"]["signed_residual"] == observed - datum.value
    assert report["reference_standard_uncertainty"] is None
    assert report["residuals"]["residual_over_reference_standard_uncertainty"] is None
    assert report["reference_review_kind"] == "automated_extraction"
    assert report["reference_source"]["origin"] == "published_theoretical"
    assert (
        report["comparison_context"]["comparison_scope"] == "cross_method_descriptive"
    )
    assert report["comparison_context"]["method_mismatch"]
    assert report["comparison_context"]["basis_mismatch"]
    assert not report["comparison_context"][
        "same_method_numerical_reproduction_established"
    ]
    assert (
        "isotope_identity_not_explicit_in_original_source"
        in report["reference_identity_quality_flags"]
    )
    if index >= 3:
        assert (
            "physical_harmonic_mode_assignment_not_established"
            in report["reference_identity_quality_flags"]
        )
    for qualification in (
        "experimental_accuracy_established",
        "scientific_method_qualified",
        "release_qualification_enabled",
        "prediction_independence_established",
        "held_out_benchmark",
    ):
        assert not report[qualification]
    with pytest.raises(FileExistsError):
        compare_published_reference(
            references,
            datum.reference_id,
            bundle,
            expected_publication_manifest_sha256=trusted_manifest,
            destination=destination,
        )


@pytest.mark.real_engine
def test_changed_genuine_publication_cannot_produce_a_reference_comparison(
    actual_qm9_water_references, actual_water_publication_bundle, tmp_path
):
    _, _, references = actual_qm9_water_references
    bundle, trusted_manifest, _ = actual_water_publication_bundle
    changed = tmp_path / "altered-genuine-publication"
    shutil.copytree(bundle, changed)
    result_path = changed / "shard/result.json"
    result_path.chmod(0o600)
    result_path.write_bytes(result_path.read_bytes() + b"\n")
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        compare_published_reference(
            references,
            references.manifest.datums[0].reference_id,
            changed,
            expected_publication_manifest_sha256=trusted_manifest,
            destination=output,
        )
    assert not output.exists()
