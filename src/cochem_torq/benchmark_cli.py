"""Explicit file-based benchmark workflow without supplied reference values.

The CLI requires externally retained pins and actual curator/acceptance records.
It never creates those records, fills missing reference data, or accepts targets.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from .benchmarking import (
    BenchmarkDesign,
    VerifiedReferences,
    freeze_benchmark,
    import_curated_references,
    score_benchmark,
    seal_predictions,
)
from .domain import read_json


def _regular_file(path: str | Path) -> Path:
    file = Path(path).absolute()
    if any(part.is_symlink() for part in (file, *file.parents)):
        raise ValueError("Benchmark input paths cannot traverse symbolic links.")
    if not file.is_file():
        raise ValueError("Benchmark inputs require actual regular files.")
    return file


def _locations(path: Path, schema: str) -> list[dict[str, Any]]:
    file = _regular_file(path)
    data = read_json(file)
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "locations"}
        or data["schema_version"] != schema
        or not isinstance(data["locations"], list)
        or not data["locations"]
        or any(not isinstance(row, dict) for row in data["locations"])
    ):
        raise ValueError("Use the exact nonempty versioned artifact-location map.")
    return data["locations"]


def load_references(
    manifest: Path,
    curation: Path,
    source_locations: Path,
    *,
    expected_manifest_sha256: str,
    expected_curation_sha256: str,
) -> VerifiedReferences:
    """Load original bytes only after checking an explicit source-location map."""
    sources: dict[str, bytes] = {}
    rows = _locations(source_locations, "cochem.torq.reference-source-locations/1")
    for row in rows:
        if set(row) != {"source_sha256", "path"} or any(
            not isinstance(row[key], str) or not row[key]
            for key in ("source_sha256", "path")
        ):
            raise ValueError("Each reference source requires its digest and file path.")
        if row["source_sha256"] in sources:
            raise ValueError("Reference-source digests cannot occur twice.")
        source = Path(row["path"])
        if not source.is_absolute():
            source = source_locations.absolute().parent / source
        sources[row["source_sha256"]] = _regular_file(source).read_bytes()
    return import_curated_references(
        _regular_file(manifest).read_bytes(),
        sources,
        _regular_file(curation).read_bytes(),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_curation_sha256=expected_curation_sha256,
    )


def execute_benchmark_command(arguments: Any) -> dict[str, Any]:
    """Run the requested immutable operation; actual scientific status is separate."""
    if arguments.command == "benchmark-freeze":
        design = BenchmarkDesign.model_validate(
            read_json(_regular_file(arguments.design))
        )
        return freeze_benchmark(
            design,
            _regular_file(arguments.acceptance_record).read_bytes(),
            arguments.output,
        )
    if arguments.command == "benchmark-seal":
        bundles: dict[str, tuple[Path, str]] = {}
        rows = _locations(
            arguments.bundles, "cochem.torq.prediction-bundle-locations/1"
        )
        for row in rows:
            if set(row) != {"reference_id", "path", "manifest_sha256"} or any(
                not isinstance(row[key], str) or not row[key]
                for key in ("reference_id", "path", "manifest_sha256")
            ):
                raise ValueError("A prediction location requires ID, path and digest.")
            if row["reference_id"] in bundles:
                raise ValueError("Prediction reference IDs cannot occur twice.")
            bundle = Path(row["path"])
            if not bundle.is_absolute():
                bundle = arguments.bundles.absolute().parent / bundle
            bundles[row["reference_id"]] = (bundle, row["manifest_sha256"])
        return seal_predictions(
            arguments.freeze,
            expected_freeze_sha256=arguments.expected_freeze_sha256,
            bundles=bundles,
            destination=arguments.output,
        )
    if arguments.command == "benchmark-score":
        # Check the supplied immutable seal before this command opens any actual
        # reference source. The scoring service then verifies the entire seal and
        # every native publication bundle again before numerical comparison.
        seal = _regular_file(arguments.prediction_seal).read_bytes()
        if sha256(seal).hexdigest() != arguments.expected_prediction_seal_sha256:
            raise ValueError("Prediction-seal bytes differ from the trusted digest.")
        sealed = read_json(arguments.prediction_seal)
        if (
            not isinstance(sealed, dict)
            or sealed.get("schema_version")
            != "cochem.torq.sealed-benchmark-predictions/1"
            or sealed.get("freeze_sha256") != arguments.expected_freeze_sha256
        ):
            raise ValueError("The prediction seal does not belong to this freeze.")
    references = load_references(
        arguments.reference_manifest,
        arguments.curation,
        arguments.sources,
        expected_manifest_sha256=arguments.expected_reference_manifest_sha256,
        expected_curation_sha256=arguments.expected_curation_sha256,
    )
    if arguments.command == "benchmark-references":
        return {
            "reference_manifest_sha256": references.manifest_sha256,
            "curation_sha256": references.curation_sha256,
            "source_count": len(references.source_bytes),
            "datum_count": len(references.manifest.datums),
            "external_curation_attestation_verified": True,
            "reference_authorship_independently_established_by_cli": False,
            "experimental_accuracy_established": False,
        }
    if arguments.command != "benchmark-score":
        raise ValueError("Unknown benchmark operation.")
    return score_benchmark(
        arguments.freeze,
        arguments.prediction_seal,
        references,
        expected_freeze_sha256=arguments.expected_freeze_sha256,
        expected_prediction_seal_sha256=arguments.expected_prediction_seal_sha256,
        destination=arguments.output,
    )
