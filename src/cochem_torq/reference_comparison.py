"""Descriptive comparisons with retained publications and genuine native bundles.

This is deliberately separate from a preregistered held-out benchmark. An
automated extraction record can support a transparent worked example without
pretending to be a human curation attestation. Neither path establishes method
accuracy, reference authorship, independence or a release qualification.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from .benchmark_cli import _locations, _regular_file
from .benchmarking import (
    ReferenceDatum,
    ReferenceManifest,
    Sha256,
    _decode,
    _write_once,
    import_curated_references,
    import_publication_prediction,
)
from .domain import digest
from .scientific_contracts import Name, ScientificContract, Unit

_NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])[+\-\N{MINUS SIGN}]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eEdD][+\-\N{MINUS SIGN}]?\d+)?(?![\w.])"
)


def _literal_number(literal: str) -> float:
    return float(literal.replace("D", "E").replace("d", "e").replace("\u2212", "-"))


class AutomatedDatumExtraction(ScientificContract):
    """An actual machine extraction declaration, never independent curation."""

    reference_id: Name
    source_sha256: Sha256
    datum_locator: Name
    identity_sha256: Sha256
    value: StrictFloat
    unit: Unit
    standard_uncertainty: StrictFloat | None = Field(default=None, gt=0)
    source_value: StrictFloat
    source_unit: Unit
    unit_conversion_multiplier: StrictFloat = Field(gt=0)
    source_standard_uncertainty: StrictFloat | None = Field(default=None, gt=0)
    source_value_literal: str = Field(min_length=1, max_length=64)
    source_value_token_index: StrictInt = Field(ge=0)
    source_uncertainty_literal: str | None = Field(
        default=None, min_length=1, max_length=64
    )
    source_uncertainty_token_index: StrictInt | None = Field(default=None, ge=0)
    raw_source_format: Literal["utf8_numeric_excerpt", "qm9_extended_xyz"] = (
        "utf8_numeric_excerpt"
    )
    source_line_number: StrictInt | None = Field(default=None, ge=1)
    extraction_method: Name
    source_excerpt_utf8: str = Field(min_length=1, max_length=8192)
    extraction_text_sha256: Sha256 | None = None
    isotope_identity_evidence: Literal[
        "publisher_explicit", "inferred_engine_default", "not_reported"
    ] = "not_reported"
    isotope_identity_record: Name = "Source isotope convention is not established."
    source_isotope_numbers: tuple[StrictInt, ...] | None = None
    state_identity_evidence: Literal[
        "publisher_explicit", "inferred_chemical_ground_state", "not_reported"
    ] = "not_reported"
    state_identity_record: Name = (
        "Source electronic/state convention is not established."
    )
    harmonic_mode_correspondence: Literal[
        "not_applicable",
        "source_assignment_reviewed",
        "sorted_frequency_rank_only",
        "not_established",
    ] = "not_established"
    harmonic_mode_correspondence_record: Name = (
        "No source mode assignment is established."
    )
    context_source_sha256: tuple[Sha256, ...] = ()

    @model_validator(mode="after")
    def exact_declared_conversion(self) -> Self:
        tokens = _NUMERIC_TOKEN.findall(self.source_excerpt_utf8)
        if (
            self.source_value_token_index >= len(tokens)
            or tokens[self.source_value_token_index] != self.source_value_literal
            or _literal_number(self.source_value_literal) != self.source_value
        ):
            raise ValueError(
                "Source value must equal its exact selected numeric token."
            )
        if self.source_standard_uncertainty is None:
            if (
                self.source_uncertainty_literal is not None
                or self.source_uncertainty_token_index is not None
            ):
                raise ValueError(
                    "Missing source uncertainty cannot have a numeric token."
                )
        elif (
            self.source_uncertainty_literal is None
            or self.source_uncertainty_token_index is None
            or self.source_uncertainty_token_index >= len(tokens)
            or tokens[self.source_uncertainty_token_index]
            != self.source_uncertainty_literal
            or _literal_number(self.source_uncertainty_literal)
            != self.source_standard_uncertainty
        ):
            raise ValueError(
                "Source uncertainty must equal its selected numeric token."
            )
        if self.raw_source_format == "qm9_extended_xyz" and (
            self.source_line_number is None or self.extraction_text_sha256 is not None
        ):
            raise ValueError("A QM9 locator requires the actual raw XYZ line number.")
        conversions = {
            ("GHz", "MHz"): 1000.0,
            ("MHz", "GHz"): 0.001,
        }
        multiplier = (
            1.0
            if self.source_unit == self.unit
            else conversions.get((self.source_unit, self.unit))
        )
        if multiplier is None or self.unit_conversion_multiplier != multiplier:
            raise ValueError("Use an implemented, exact declared unit conversion.")
        if self.value != self.source_value * multiplier:
            raise ValueError(
                "Canonical reference value differs from source conversion."
            )
        converted_uncertainty = (
            self.source_standard_uncertainty * multiplier
            if self.source_standard_uncertainty is not None
            else None
        )
        if self.standard_uncertainty != converted_uncertainty:
            raise ValueError(
                "Reference uncertainty differs from the source conversion."
            )
        if (self.isotope_identity_evidence == "publisher_explicit") != (
            self.source_isotope_numbers is not None
        ):
            raise ValueError(
                "Explicit publisher isotope claims require reported isotopes."
            )
        return self


class TextExtractionProvenance(ScientificContract):
    """A reproducible PDF-to-text transformation with the parent retained."""

    source_sha256: Sha256
    text_sha256: Sha256
    extractor: Literal["pdftotext"]
    extractor_version: Name
    arguments: tuple[Literal["-layout"], Literal["-enc"], Literal["UTF-8"]] = (
        "-layout",
        "-enc",
        "UTF-8",
    )
    extracted_at: datetime

    @model_validator(mode="after")
    def completed_transformation(self) -> Self:
        if self.extracted_at.utcoffset() is None:
            raise ValueError("An aware text-extraction timestamp is required.")
        if self.extracted_at > datetime.now(timezone.utc):
            raise ValueError("A future text extraction is not completed evidence.")
        return self


class ReferenceContextSource(ScientificContract):
    """An original source defining units/methods, without invented numeric datums."""

    source_sha256: Sha256
    persistent_identifier: Name
    description: Name
    retrieved_at: datetime
    reuse_permission: Name

    @model_validator(mode="after")
    def retained_original_source(self) -> Self:
        if self.retrieved_at.utcoffset() is None:
            raise ValueError("Context source retrieval requires an aware timestamp.")
        if not self.persistent_identifier.startswith(("doi:", "https://")):
            raise ValueError(
                "A context source requires an actual DOI or HTTPS locator."
            )
        return self


class AutomatedReferenceReview(ScientificContract):
    schema_version: Literal["cochem.torq.automated-reference-review/1"] = (
        "cochem.torq.automated-reference-review/1"
    )
    reference_manifest_sha256: Sha256
    extractor: Name
    reviewed_at: datetime
    review_record: Name
    human_reviewed: Literal[False] = False
    independent_curation_completed: Literal[False] = False
    entries: tuple[AutomatedDatumExtraction, ...] = Field(min_length=1)
    extraction_artifacts: tuple[TextExtractionProvenance, ...] = ()
    context_sources: tuple[ReferenceContextSource, ...] = ()

    @model_validator(mode="after")
    def completed_automated_record(self) -> Self:
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("An aware machine extraction timestamp is required.")
        if self.reviewed_at > datetime.now(timezone.utc):
            raise ValueError("A future extraction is not completed evidence.")
        if len({entry.reference_id for entry in self.entries}) != len(self.entries):
            raise ValueError("Each datum requires one unique extraction record.")
        if len({item.text_sha256 for item in self.extraction_artifacts}) != len(
            self.extraction_artifacts
        ):
            raise ValueError("Derived text artifact digests must be unique.")
        if any(
            artifact.extracted_at > self.reviewed_at
            for artifact in self.extraction_artifacts
        ):
            raise ValueError("The automated review must follow its text extraction.")
        if len({source.source_sha256 for source in self.context_sources}) != len(
            self.context_sources
        ):
            raise ValueError("Context source digests must be unique.")
        if any(
            source.retrieved_at > self.reviewed_at for source in self.context_sources
        ):
            raise ValueError("Context source retrieval must precede automated review.")
        return self


@dataclass(frozen=True)
class VerifiedComparisonReferences:
    manifest: ReferenceManifest
    manifest_sha256: str
    review_sha256: str
    review_kind: Literal["automated_extraction", "external_curation_attestation"]
    source_bytes: tuple[tuple[str, bytes], ...]
    manifest_bytes: bytes
    review_bytes: bytes
    text_bytes: tuple[tuple[str, bytes], ...] = ()


def _replay_pdf_text_extraction(
    record: TextExtractionProvenance, source: bytes, expected_text: bytes
) -> None:
    """Replay a fixed real converter command; a signed record alone is insufficient."""
    if not source.startswith(b"%PDF-"):
        raise ValueError("A pdftotext proof requires the retained original PDF bytes.")
    if max(len(source), len(expected_text)) > 64 * 1024 * 1024:
        raise ValueError("PDF/text replay exceeds the 64 MiB per-artifact limit.")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise ValueError("Authentic PDF extraction verification requires pdftotext.")
    try:
        version = subprocess.run(
            [executable, "-v"], capture_output=True, check=True, timeout=10
        )
        version_text = (version.stdout + version.stderr).decode("utf-8")
        version_line = version_text.splitlines()[0]
        if version_line != f"pdftotext version {record.extractor_version}":
            raise ValueError("The PDF extraction tool version differs from its record.")
        with tempfile.TemporaryDirectory(prefix="torq-reference-pdf-") as temporary:
            parent = Path(temporary)
            original = parent / "source.pdf"
            converted = parent / "source.txt"
            original.write_bytes(source)
            subprocess.run(
                [executable, *record.arguments, str(original), str(converted)],
                capture_output=True,
                check=True,
                timeout=60,
            )
            if converted.read_bytes() != expected_text:
                raise ValueError("Replayed PDF extraction differs from retained text.")
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError) as error:
        raise ValueError("Authentic PDF text extraction replay failed.") from error


def _verify_qm9_numeric_locator(
    entry: AutomatedDatumExtraction, datum: ReferenceDatum, raw: bytes
) -> None:
    """Bind QM9 property/rank selection to its actual publisher file layout."""
    try:
        lines = raw.decode("utf-8").splitlines()
        atom_count = int(lines[0])
        properties = lines[1].split()
        symbols = tuple(line.split()[0] for line in lines[2 : atom_count + 2])
        if (
            atom_count != len(datum.identity.symbols)
            or symbols != datum.identity.symbols
            or len(properties) != 17
            or properties[0] != "gdb"
            or entry.source_line_number is None
            or entry.source_line_number > len(lines)
            or lines[entry.source_line_number - 1] != entry.source_excerpt_utf8
        ):
            raise ValueError(
                "QM9 identity/line selection differs from original layout."
            )
        if datum.identity.observable == "equilibrium_rotational_constant":
            expected_index = "ABC".index(datum.identity.component) + 1
            if (
                entry.source_line_number != 2
                or entry.source_value_token_index != expected_index
                or entry.source_unit != "GHz"
            ):
                raise ValueError(
                    "QM9 A/B/C requires the exact original property column."
                )
        elif datum.identity.observable == "harmonic_frequency":
            if (
                entry.source_line_number != atom_count + 3
                or entry.source_value_token_index != datum.identity.harmonic_mode_index
                or entry.source_unit != "cm^-1"
            ):
                raise ValueError("QM9 frequency requires its exact original row/rank.")
        else:
            raise ValueError(
                "No direct QM9 extractor exists for this observable; U0 includes ZPVE."
            )
    except (UnicodeError, IndexError, TypeError) as error:
        raise ValueError(
            "Original QM9 numeric locator could not be verified."
        ) from error


def import_comparison_references(
    manifest_bytes: bytes,
    source_artifacts: Mapping[str, bytes],
    review_bytes: bytes,
    *,
    expected_manifest_sha256: str,
    expected_review_sha256: str,
    extraction_texts: Mapping[str, bytes] | None = None,
) -> VerifiedComparisonReferences:
    """Check supplied bytes and declarations; do not assert source authenticity.

    Digests are supplied by the caller and bind retained bytes. A digest does
    not authenticate a publisher or replace an independent scientific review.
    Automated records are accepted only for descriptive comparisons, never by
    the existing curated-reference benchmark import service.
    """
    if (
        sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
        or sha256(review_bytes).hexdigest() != expected_review_sha256
    ):
        raise ValueError("Reference/review bytes differ from the supplied digests.")
    review_data = _decode(review_bytes)
    if not isinstance(review_data, dict):
        raise ValueError("A versioned reference review object is required.")
    if review_data.get("schema_version") == (
        "cochem.torq.reference-curation-attestation/1"
    ):
        if extraction_texts:
            raise ValueError(
                "External curation does not accept an unbound text inventory."
            )
        curated = import_curated_references(
            manifest_bytes,
            source_artifacts,
            review_bytes,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_curation_sha256=expected_review_sha256,
        )
        return VerifiedComparisonReferences(
            curated.manifest,
            curated.manifest_sha256,
            curated.curation_sha256,
            "external_curation_attestation",
            curated.source_bytes,
            curated.manifest_bytes,
            curated.curation_bytes,
        )
    review = AutomatedReferenceReview.model_validate(review_data)
    manifest = ReferenceManifest.model_validate(_decode(manifest_bytes))
    if review.reference_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Automated review belongs to another reference manifest.")
    numeric_sources = {source.source_sha256 for source in manifest.sources}
    context_sources = {source.source_sha256 for source in review.context_sources}
    if numeric_sources & context_sources:
        raise ValueError("Numeric and supporting context inventories must be distinct.")
    if context_sources != {
        source_sha256
        for entry in review.entries
        for source_sha256 in entry.context_source_sha256
    }:
        raise ValueError(
            "Every supporting context source must bind an extracted datum."
        )
    if set(source_artifacts) != numeric_sources | context_sources:
        raise ValueError("All and only inventoried original source bytes are required.")
    for expected, raw in source_artifacts.items():
        if not isinstance(raw, bytes) or sha256(raw).hexdigest() != expected:
            raise ValueError("Original reference source artifact digest mismatch.")
    if any(source.retrieved_at > review.reviewed_at for source in manifest.sources):
        raise ValueError("A source must be retrieved before its extraction review.")
    texts = extraction_texts or {}
    artifacts = {item.text_sha256: item for item in review.extraction_artifacts}
    if set(texts) != set(artifacts):
        raise ValueError("All and only inventoried derived text bytes are required.")
    if set(artifacts) != {
        entry.extraction_text_sha256
        for entry in review.entries
        if entry.extraction_text_sha256 is not None
    }:
        raise ValueError("Every derived text artifact must support a bound datum.")
    retrieved = {
        source.source_sha256: source.retrieved_at for source in manifest.sources
    }
    for text_sha256, artifact in artifacts.items():
        raw_text = texts[text_sha256]
        if (
            not isinstance(raw_text, bytes)
            or sha256(raw_text).hexdigest() != text_sha256
        ):
            raise ValueError("Derived extraction text digest mismatch.")
        if artifact.source_sha256 not in source_artifacts:
            raise ValueError(
                "Derived extraction requires its inventoried original source."
            )
        if retrieved[artifact.source_sha256] > artifact.extracted_at:
            raise ValueError("Original PDF retrieval must precede text extraction.")
        _replay_pdf_text_extraction(
            artifact, source_artifacts[artifact.source_sha256], raw_text
        )
    datums = {datum.reference_id: datum for datum in manifest.datums}
    if set(datums) != {entry.reference_id for entry in review.entries}:
        raise ValueError("Every manifest datum requires its actual extraction record.")
    for entry in review.entries:
        datum = datums[entry.reference_id]
        if (
            entry.source_sha256 != datum.source_sha256
            or entry.datum_locator != datum.datum_locator
            or entry.identity_sha256 != digest(datum.identity.model_dump(mode="json"))
            or entry.value != datum.value
            or entry.unit != datum.identity.unit
            or entry.standard_uncertainty != datum.standard_uncertainty
        ):
            raise ValueError("Automated extraction does not bind the exact datum.")
        if entry.source_isotope_numbers is not None and (
            entry.source_isotope_numbers != datum.identity.isotope_numbers
        ):
            raise ValueError(
                "Explicit source isotopes differ from the declared identity."
            )
        if datum.identity.observable == "harmonic_frequency":
            if entry.harmonic_mode_correspondence == "not_applicable":
                raise ValueError("A harmonic comparison requires explicit mode policy.")
        elif entry.harmonic_mode_correspondence not in {
            "not_applicable",
            "not_established",
        }:
            raise ValueError("A harmonic mode policy cannot label another observable.")
        if entry.extraction_text_sha256 is not None:
            if (
                artifacts[entry.extraction_text_sha256].source_sha256
                != entry.source_sha256
            ):
                raise ValueError("Derived text belongs to another original source.")
            quoted_bytes = texts[entry.extraction_text_sha256]
        else:
            quoted_bytes = source_artifacts[entry.source_sha256]
        if entry.source_excerpt_utf8.encode("utf-8") not in quoted_bytes:
            raise ValueError("Extraction excerpt is absent from retained source bytes.")
        if entry.source_line_number is not None:
            try:
                line = quoted_bytes.decode("utf-8").splitlines()[
                    entry.source_line_number - 1
                ]
            except (UnicodeError, IndexError) as error:
                raise ValueError("The selected source line is unavailable.") from error
            if line != entry.source_excerpt_utf8:
                raise ValueError(
                    "Extraction excerpt differs from the selected source line."
                )
        if entry.raw_source_format == "qm9_extended_xyz":
            _verify_qm9_numeric_locator(entry, datum, quoted_bytes)
        else:
            try:
                actual_lines = quoted_bytes.decode("utf-8").splitlines()
                is_qm9 = (
                    len(actual_lines) >= 2
                    and actual_lines[0].strip().isdigit()
                    and actual_lines[1].split()[0] == "gdb"
                )
            except (UnicodeError, IndexError):
                is_qm9 = False
            if is_qm9:
                raise ValueError(
                    "An actual QM9 source requires its structured locator."
                )
    return VerifiedComparisonReferences(
        manifest,
        expected_manifest_sha256,
        expected_review_sha256,
        "automated_extraction",
        tuple(sorted(source_artifacts.items())),
        manifest_bytes,
        review_bytes,
        tuple(sorted(texts.items())),
    )


def load_comparison_references(
    manifest: Path,
    review: Path,
    source_locations: Path,
    *,
    expected_manifest_sha256: str,
    expected_review_sha256: str,
    extraction_text_locations: Path | None = None,
) -> VerifiedComparisonReferences:
    """Read explicit regular-file inputs without following symbolic links."""
    sources: dict[str, bytes] = {}
    for row in _locations(source_locations, "cochem.torq.reference-source-locations/1"):
        if set(row) != {"source_sha256", "path"} or any(
            not isinstance(row[key], str) or not row[key]
            for key in ("source_sha256", "path")
        ):
            raise ValueError("Each source requires its SHA-256 and actual file path.")
        if row["source_sha256"] in sources:
            raise ValueError("Reference-source digests cannot occur twice.")
        source = Path(row["path"])
        if not source.is_absolute():
            source = source_locations.absolute().parent / source
        sources[row["source_sha256"]] = _regular_file(source).read_bytes()
    texts: dict[str, bytes] = {}
    if extraction_text_locations is not None:
        for row in _locations(
            extraction_text_locations,
            "cochem.torq.reference-extraction-text-locations/1",
        ):
            if set(row) != {"text_sha256", "path"} or any(
                not isinstance(row[key], str) or not row[key]
                for key in ("text_sha256", "path")
            ):
                raise ValueError("Each derived text requires its digest and file path.")
            if row["text_sha256"] in texts:
                raise ValueError("Derived text digests cannot occur twice.")
            text_path = Path(row["path"])
            if not text_path.is_absolute():
                text_path = extraction_text_locations.absolute().parent / text_path
            texts[row["text_sha256"]] = _regular_file(text_path).read_bytes()
    return import_comparison_references(
        _regular_file(manifest).read_bytes(),
        sources,
        _regular_file(review).read_bytes(),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_review_sha256=expected_review_sha256,
        extraction_texts=texts,
    )


def descriptive_residuals(
    prediction: float, reference: float, reference_standard_uncertainty: float | None
) -> dict[str, Any]:
    """Signed arithmetic, without a tolerance, p-value or predictive claim.

    The relative residual uses abs(reference), including for electronic energies.
    Division by a reported reference uncertainty is explicitly not a z-score:
    no prediction uncertainty or covariance has been supplied to this function.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (prediction, reference)
    ) or not all(math.isfinite(value) for value in (prediction, reference)):
        raise ValueError("Residual inputs must be finite genuine numerical values.")
    if reference_standard_uncertainty is not None and (
        isinstance(reference_standard_uncertainty, bool)
        or not isinstance(reference_standard_uncertainty, (int, float))
        or not math.isfinite(reference_standard_uncertainty)
        or reference_standard_uncertainty <= 0
    ):
        raise ValueError("A supplied standard uncertainty must be positive and finite.")
    signed = float(prediction - reference)
    relative = signed / abs(reference) if reference != 0 else None
    scaled = (
        signed / reference_standard_uncertainty
        if reference_standard_uncertainty is not None
        else None
    )
    if not all(
        math.isfinite(value)
        for value in (signed, relative, scaled)
        if value is not None
    ):
        raise ValueError("Residual arithmetic overflowed the finite numerical domain.")
    relative_ppm = relative * 1e6 if relative is not None else None
    if relative_ppm is not None and not math.isfinite(relative_ppm):
        raise ValueError("Relative residual in ppm overflowed.")
    return {
        "signed_residual": signed,
        "absolute_residual": abs(signed),
        "relative_residual": relative,
        "relative_residual_ppm": relative_ppm,
        "relative_residual_definition": "(prediction - reference) / abs(reference)",
        "relative_residual_absence_reason": (
            "Reference value is zero." if relative is None else None
        ),
        "residual_over_reference_standard_uncertainty": scaled,
        "uncertainty_ratio_interpretation": (
            "Descriptive ratio only; prediction uncertainty and covariance are absent."
            if scaled is not None
            else "Reference standard uncertainty is unavailable; no ratio is computed."
        ),
    }


def comparison_provenance_context(
    datum: ReferenceDatum, source_origin: str, prediction: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose cross-method references and unestablished geometry correspondence.

    Labels are compared conservatively; an exact text match is not independent
    validation of the complete method. Geometry digests identify coordinate
    representations, not orientation-invariant physical geometry equivalence.
    """
    calculation = prediction.get("calculation_provenance", {})
    recipe = calculation.get("recipe") or {}
    theoretical = getattr(datum, "theoretical_provenance", None)
    context: dict[str, Any] = {
        "reference_origin": source_origin,
        "native_method": recipe.get("method"),
        "native_basis": recipe.get("basis"),
        "reference_method": None,
        "reference_basis": None,
        "method_labels_match": None,
        "basis_labels_match": None,
        "method_mismatch": None,
        "basis_mismatch": None,
        "geometry_digest_match": None,
        "geometry_correspondence": "not_established",
        "comparison_scope": "descriptive_publication_reference",
        "same_method_numerical_reproduction_established": False,
        "quality_flags": [],
    }
    if theoretical is None:
        return context
    context["reference_method"] = theoretical.method
    context["reference_basis"] = theoretical.basis
    for field in ("method", "basis"):
        native = context[f"native_{field}"]
        reference = context[f"reference_{field}"]
        if isinstance(native, str) and isinstance(reference, str):
            match = native.casefold().strip() == reference.casefold().strip()
            context[f"{field}_labels_match"] = match
            context[f"{field}_mismatch"] = not match
            if not match:
                context["quality_flags"].append(f"different_{field}_labels")
    if context["method_mismatch"] or context["basis_mismatch"]:
        context["comparison_scope"] = "cross_method_descriptive"
    reference_geometry = theoretical.geometry_sha256
    calculated_geometry = calculation.get("calculated_geometry_sha256")
    if reference_geometry is not None and calculated_geometry is not None:
        match = reference_geometry == calculated_geometry
        context["geometry_digest_match"] = match
        context["geometry_correspondence"] = (
            "identical_retained_coordinate_digest"
            if match
            else "different_retained_coordinate_digest"
        )
        if not match:
            context["quality_flags"].append("different_geometry_coordinate_digest")
    else:
        context["quality_flags"].append(
            "reference_geometry_correspondence_not_established"
        )
    context["geometry_digest_interpretation"] = (
        "Digest comparison includes coordinate ordering, orientation and numerical "
        "representation. Different digests do not prove a physical geometry difference."
    )
    context["reference_theoretical_provenance"] = theoretical.model_dump(mode="json")
    return context


def execute_reference_compare(arguments: Any) -> dict[str, Any]:
    """Adapter shared by the installed CLI and the standalone example command."""
    references = load_comparison_references(
        arguments.reference_manifest,
        arguments.review,
        arguments.sources,
        expected_manifest_sha256=arguments.expected_reference_manifest_sha256,
        expected_review_sha256=arguments.expected_review_sha256,
        extraction_text_locations=arguments.derived_texts,
    )
    return compare_published_reference(
        references,
        arguments.reference_id,
        arguments.publication_bundle,
        expected_publication_manifest_sha256=(
            arguments.expected_publication_manifest_sha256
        ),
        destination=arguments.output,
    )


def compare_published_reference(
    references: VerifiedComparisonReferences,
    reference_id: str,
    bundle: str | Path,
    *,
    expected_publication_manifest_sha256: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Write one read-only, no-overwrite descriptive comparison receipt.

    Original source/review bytes are rechecked even for directly constructed
    dataclass objects. The publication importer independently checks native
    evidence, sealed molecular identity, isotope, state and scalar definition.
    Missing native observables remain unavailable rather than becoming values.
    """
    references = import_comparison_references(
        references.manifest_bytes,
        dict(references.source_bytes),
        references.review_bytes,
        expected_manifest_sha256=references.manifest_sha256,
        expected_review_sha256=references.review_sha256,
        extraction_texts=dict(references.text_bytes),
    )
    matching = [
        datum
        for datum in references.manifest.datums
        if datum.reference_id == reference_id
    ]
    if len(matching) != 1:
        raise ValueError("Choose one actual reference ID from the supplied manifest.")
    datum = matching[0]
    prediction = import_publication_prediction(
        bundle,
        expected_manifest_sha256=expected_publication_manifest_sha256,
        identity=datum.identity,
    )
    residuals = (
        descriptive_residuals(
            prediction["value"], datum.value, datum.standard_uncertainty
        )
        if prediction["status"] == "available"
        else None
    )
    source = next(
        source
        for source in references.manifest.sources
        if source.source_sha256 == datum.source_sha256
    )
    automated_entry = None
    identity_flags = []
    if references.review_kind == "automated_extraction":
        automated_review = AutomatedReferenceReview.model_validate(
            _decode(references.review_bytes)
        )
        automated_entry = next(
            entry
            for entry in automated_review.entries
            if entry.reference_id == reference_id
        )
        if automated_entry.isotope_identity_evidence != "publisher_explicit":
            identity_flags.append("isotope_identity_not_explicit_in_original_source")
        if automated_entry.state_identity_evidence != "publisher_explicit":
            identity_flags.append("state_identity_not_explicit_in_original_source")
        if datum.identity.observable == "harmonic_frequency" and (
            automated_entry.harmonic_mode_correspondence != "source_assignment_reviewed"
        ):
            identity_flags.append("physical_harmonic_mode_assignment_not_established")
    payload = {
        "schema_version": "cochem.torq.descriptive-published-reference-comparison/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_kind": "descriptive_single_case",
        "reference": datum.model_dump(mode="json"),
        "reference_source": source.model_dump(mode="json"),
        "reference_manifest_sha256": references.manifest_sha256,
        "reference_review_sha256": references.review_sha256,
        "reference_review_kind": references.review_kind,
        "reference_identity_evidence": (
            automated_entry.model_dump(mode="json")
            if automated_entry is not None
            else None
        ),
        "reference_identity_quality_flags": identity_flags,
        "reference_manifest_utf8": references.manifest_bytes.decode("utf-8"),
        "reference_review_utf8": references.review_bytes.decode("utf-8"),
        "source_artifacts": [
            {"source_sha256": source_sha256, "byte_count": len(raw)}
            for source_sha256, raw in references.source_bytes
        ],
        "derived_text_artifacts": [
            {"text_sha256": text_sha256, "byte_count": len(raw)}
            for text_sha256, raw in references.text_bytes
        ],
        "pdf_text_derivation_replayed": bool(references.text_bytes),
        "prediction": prediction,
        "comparison_context": comparison_provenance_context(
            datum, source.origin, prediction
        ),
        "status": prediction["status"],
        "residual_unit": datum.identity.unit,
        "residuals": residuals,
        "reference_standard_uncertainty": datum.standard_uncertainty,
        "reference_covariance_status": datum.covariance_status,
        "reference_authorship_independently_authenticated": False,
        "external_curation_attestation_present": references.review_kind
        == "external_curation_attestation",
        "human_review_independently_authenticated": False,
        "prediction_independence_established": False,
        "reference_anchors_present": bool(prediction["reference_anchors"]),
        "held_out_benchmark": False,
        "experimental_accuracy_established": False,
        "scientific_method_qualified": False,
        "release_qualification_enabled": False,
        "interpretation": (
            "One descriptive residual does not establish method accuracy, molecular "
            "identification, confidence coverage or exact method reproduction. "
            "Publication metadata and review records are declared provenance; hashes "
            "bind bytes. Retain the original reference source and native bundle."
        ),
    }
    output_sha256 = _write_once(destination, payload)
    output = _regular_file(destination)
    if sha256(output.read_bytes()).hexdigest() != output_sha256:
        raise ValueError("Comparison output changed immediately after publication.")
    output.chmod(0o400)
    return {
        "comparison_path": str(output),
        "comparison_sha256": output_sha256,
        "reference_id": reference_id,
        "status": prediction["status"],
        "reference_review_kind": references.review_kind,
        "experimental_accuracy_established": False,
        "scientific_method_qualified": False,
    }
