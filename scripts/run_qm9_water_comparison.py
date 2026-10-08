"""Run the pinned, offline QM9 water comparison with a genuine HF calculation.

From the TORQ checkout, run ``python -m scripts.run_qm9_water_comparison
--output-dir /tmp/NEW_DIRECTORY``. Install the genuine calculation dependencies
first. Native failures remain failures; this descriptive example never qualifies
an electronic-structure method, experimental accuracy or a release.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import stat
import sys
import traceback
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO

from cochem_torq.application import execute_request
from cochem_torq.benchmarking import _decode, _safe_parent, _write_once
from cochem_torq.domain import CalculationRequest, digest, read_json
from cochem_torq.publication import (
    export_publication_bundle,
    verify_publication_bundle,
)
from cochem_torq.reference_comparison import (
    execute_reference_compare,
    load_comparison_references,
)

_DEFAULT_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "benchmarks/published-values/qm9-water"
)
_PACKET_FILES = (
    "pins.json",
    "request.json",
    "references.json",
    "automated-review.json",
    "source-locations.json",
    "water.xyz",
    "publisher-readme.txt",
    "publisher-metadata.json",
    "retrieval-proof.json",
)
_MAX_PACKET_FILE_BYTES = 16 * 1024 * 1024


class _Tee(io.TextIOBase):
    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()


def _snapshot_packet(example: Path, destination: Path) -> dict[str, str]:
    """Retain the actual supplied bytes, including the reviewed trust pins."""
    example = example.absolute()
    for component in (example, *example.parents):
        if component.is_symlink():
            raise ValueError("Reference packet paths cannot traverse symlinks.")
    if not example.is_dir():
        raise NotADirectoryError(example)
    destination.mkdir(mode=0o700)
    observed = {}
    for name in _PACKET_FILES:
        source = example / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"The pinned example requires a regular {name} file.")
        before = source.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_PACKET_FILE_BYTES
        ):
            raise ValueError(f"The {name} packet file exceeds its regular-file bound.")
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise ValueError(f"The {name} packet file changed before reading.")
            raw = stream.read(_MAX_PACKET_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (
            len(raw) != before.st_size
            or len(raw) > _MAX_PACKET_FILE_BYTES
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(
                f"The {name} packet file changed or exceeded its read bound."
            )
        target = destination / name
        with target.open("xb") as stream:
            stream.write(raw)
        target.chmod(0o400)
        observed[name] = sha256(raw).hexdigest()
    return observed


def run_example(example: Path, output_dir: Path) -> dict[str, Any]:
    """Calculate once, retain evidence, then compare all six pinned datums."""
    output = output_dir.absolute()
    _safe_parent(output.parent)
    if output.is_symlink():
        raise ValueError("The output directory cannot be a symbolic link.")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": "cochem.torq.qm9-water-comparison-run/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "comparison_kind": "descriptive_cross_method_example",
        "network_reference_retrieval_performed": False,
        "experimental_accuracy_established": False,
        "scientific_method_qualified": False,
        "independent_curation_completed": False,
        "held_out_benchmark": False,
        "release_qualification_enabled": False,
        "comparisons": [],
    }
    log_path = output / "run.log"
    with log_path.open("x", encoding="utf-8") as log:
        log_path.chmod(0o600)
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, log)),
            contextlib.redirect_stderr(_Tee(sys.stderr, log)),
        ):
            try:
                print("Verifying and retaining the pinned QM9 water reference packet.")
                packet = output / "reference-inputs"
                summary["input_file_observed_sha256"] = _snapshot_packet(
                    example, packet
                )
                pins = _decode((packet / "pins.json").read_bytes())
                if pins.get("schema_version") != (
                    "cochem.torq.published-reference-example-pins/1"
                ):
                    raise ValueError("Use the reviewed QM9 water pin schema.")
                request_bytes = (packet / "request.json").read_bytes()
                if sha256(request_bytes).hexdigest() != pins["request_file_sha256"]:
                    raise ValueError("The request bytes differ from the reviewed pin.")
                request = CalculationRequest.model_validate_json(request_bytes)
                if (
                    digest(request.model_dump(mode="json"))
                    != (pins["canonical_request_sha256"])
                    or request.recipe != pins["native_recipe"]
                ):
                    raise ValueError("The calculation differs from the pinned request.")
                references = load_comparison_references(
                    packet / "references.json",
                    packet / "automated-review.json",
                    packet / "source-locations.json",
                    expected_manifest_sha256=pins["reference_manifest_sha256"],
                    expected_review_sha256=pins["review_sha256"],
                )
                reference_ids = pins["reference_ids"]
                if (
                    not isinstance(reference_ids, list)
                    or len(reference_ids) != 6
                    or any(not isinstance(item, str) for item in reference_ids)
                    or len(set(reference_ids)) != 6
                    or set(reference_ids)
                    != {datum.reference_id for datum in references.manifest.datums}
                ):
                    raise ValueError("The example requires exactly six pinned datums.")
                summary["request_resources"] = request.resources.model_dump(mode="json")
                summary["request_sha256"] = pins["canonical_request_sha256"]
                summary["reference_manifest_sha256"] = pins["reference_manifest_sha256"]
                summary["reference_review_sha256"] = pins["review_sha256"]
                print(
                    f"Computing {request.recipe}: {request.resources.cores} core(s), "
                    f"{request.resources.memory_mb} MB, "
                    f"and {request.resources.wall_seconds} seconds maximum wall time."
                )
                native = execute_request(request, output / "native-shard")
                summary["native_status"] = native["status"]
                summary["native_errors"] = native["errors"]
                print(f"Native calculation status: {native['status']}.")
                publication = export_publication_bundle(
                    output / "native-shard", output / "publication"
                )
                verify_publication_bundle(
                    output / "publication",
                    expected_manifest_sha256=publication["manifest_sha256"],
                )
                summary["publication_manifest_sha256"] = publication["manifest_sha256"]
                comparison_dir = output / "comparisons"
                comparison_dir.mkdir(mode=0o700)
                comparisons = []
                identity_flags: set[str] = set()
                for index, reference_id in enumerate(reference_ids):
                    destination = comparison_dir / f"{index + 1:02d}.json"
                    print(
                        f"Comparing {reference_id} against its retained source bytes."
                    )
                    receipt = execute_reference_compare(
                        argparse.Namespace(
                            reference_manifest=packet / "references.json",
                            review=packet / "automated-review.json",
                            sources=packet / "source-locations.json",
                            derived_texts=None,
                            reference_id=reference_id,
                            expected_reference_manifest_sha256=pins[
                                "reference_manifest_sha256"
                            ],
                            expected_review_sha256=pins["review_sha256"],
                            publication_bundle=output / "publication",
                            expected_publication_manifest_sha256=publication[
                                "manifest_sha256"
                            ],
                            output=destination,
                        )
                    )
                    report = read_json(destination)
                    identity_flags.update(report["reference_identity_quality_flags"])
                    comparisons.append(
                        {
                            "reference_id": reference_id,
                            "path": destination.relative_to(output).as_posix(),
                            "sha256": receipt["comparison_sha256"],
                            "status": report["status"],
                            "observable": report["reference"]["identity"]["observable"],
                            "unit": report["residual_unit"],
                            "reference_value": report["reference"]["value"],
                            "prediction_value": report["prediction"]["value"],
                            "residuals": report["residuals"],
                            "comparison_context": report["comparison_context"],
                        }
                    )
                    summary["comparisons"] = comparisons
                summary["reference_identity_quality_flags"] = sorted(identity_flags)
                summary["status"] = (
                    "complete_descriptive_comparison"
                    if native["status"] == "complete"
                    and all(item["status"] == "available" for item in comparisons)
                    else "incomplete_native_or_comparison"
                )
                summary["interpretation"] = (
                    "HF/STO-3G predictions are compared descriptively with published "
                    "QM9 B3LYP/6-31G(2df,p) equilibrium constants and harmonic ranks. "
                    "Method, basis, optimized geometry, isotope/state evidence and "
                    "physical mode-assignment differences remain explicit. This is "
                    "not a same-method reproduction or experimental accuracy gate."
                )
                print(f"Comparison run status: {summary['status']}.")
            except Exception as error:
                summary["error"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc()
                print("Failure retained; no result or reference was substituted.")
    summary_path = output / "summary.json"
    summary_sha256 = _write_once(summary_path, summary)
    summary_path.chmod(0o400)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "status": summary["status"],
        "experimental_accuracy_established": False,
        "release_qualification_enabled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--example-dir", type=Path, default=_DEFAULT_EXAMPLE)
    arguments = parser.parse_args(argv)
    try:
        receipt = run_example(arguments.example_dir, arguments.output_dir)
    except (OSError, ValueError) as error:
        print(f"No existing output was changed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "complete_descriptive_comparison" else 1


if __name__ == "__main__":
    raise SystemExit(main())
