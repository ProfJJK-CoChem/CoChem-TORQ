"""Actual pinned-packet integrity and genuine end-to-end water comparison.

The real-engine check executes the public command and retains its real process
output. No engine, source value, network provider or result is substituted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from cochem_torq.domain import read_json
from cochem_torq.publication import verify_publication_bundle
from scripts.run_qm9_water_comparison import _MAX_PACKET_FILE_BYTES, main

_ROOT = Path(__file__).resolve().parents[1]
_PACKET = _ROOT / "benchmarks/published-values/qm9-water"


def test_existing_directory_is_preserved_without_starting_calculation(tmp_path, capsys):
    output = tmp_path / "existing-owned-data"
    output.mkdir()
    retained = output / "retained.txt"
    retained.write_bytes(b"actual pre-existing user data")
    assert main(["--output-dir", str(output)]) == 2
    assert retained.read_bytes() == b"actual pre-existing user data"
    assert sorted(p.name for p in output.iterdir()) == ["retained.txt"]
    assert "No existing output was changed" in capsys.readouterr().err


def test_changed_actual_request_is_rejected_before_native_execution(tmp_path, capsys):
    damaged = tmp_path / "actual-packet-with-changed-request-bytes"
    shutil.copytree(_PACKET, damaged)
    with (damaged / "request.json").open("ab") as stream:
        stream.write(b"\n")  # Unchanged JSON meaning, different actual pinned bytes.
    output = tmp_path / "failed-packet-verification"
    assert main(["--output-dir", str(output), "--example-dir", str(damaged)]) == 1
    summary = read_json(output / "summary.json")
    assert summary["status"] == "failed"
    assert summary["error"]["type"] == "ValueError"
    assert "reviewed pin" in summary["error"]["message"]
    assert summary["comparisons"] == []
    assert not (output / "native-shard").exists()
    assert summary["experimental_accuracy_established"] is False
    assert "Failure retained" in (output / "run.log").read_text()
    capsys.readouterr()


def test_oversized_actual_packet_file_is_rejected_before_native_execution(tmp_path):
    oversized = tmp_path / "actual-packet-with-oversized-metadata"
    shutil.copytree(_PACKET, oversized)
    metadata = oversized / "publisher-metadata.json"
    metadata.chmod(0o600)
    with metadata.open("r+b") as stream:
        stream.truncate(_MAX_PACKET_FILE_BYTES + 1)
    output = tmp_path / "failed-size-bound"
    assert main(["--output-dir", str(output), "--example-dir", str(oversized)]) == 1
    summary = read_json(output / "summary.json")
    assert "regular-file bound" in summary["error"]["message"]
    assert summary["comparisons"] == []
    assert not (output / "native-shard").exists()


@pytest.mark.real_engine
def test_one_command_computes_and_compares_six_actual_published_water_datums(tmp_path):
    output = tmp_path / "genuine-one-command-water-comparison"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(_ROOT / "src"), str(_ROOT))),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "scripts.run_qm9_water_comparison",
        "--output-dir",
        str(output),
    ]
    process = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (tmp_path / "actual-command-stdout.txt").write_text(process.stdout)
    (tmp_path / "actual-command-stderr.txt").write_text(process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    receipt = json.loads(process.stdout.splitlines()[-1])
    summary_path = output / "summary.json"
    assert sha256(summary_path.read_bytes()).hexdigest() == receipt["summary_sha256"]
    summary = read_json(summary_path)
    assert summary["status"] == "complete_descriptive_comparison"
    assert summary["native_status"] == "complete"
    assert summary["request_resources"] == {
        "cores": 1,
        "memory_mb": 1024,
        "wall_seconds": 120,
    }
    assert summary["experimental_accuracy_established"] is False
    assert summary["scientific_method_qualified"] is False
    assert summary["release_qualification_enabled"] is False
    assert summary["independent_curation_completed"] is False
    assert summary["network_reference_retrieval_performed"] is False
    assert len(summary["comparisons"]) == 6
    verify_publication_bundle(
        output / "publication",
        expected_manifest_sha256=summary["publication_manifest_sha256"],
    )
    pins = read_json(_PACKET / "pins.json")
    assert summary["reference_manifest_sha256"] == pins["reference_manifest_sha256"]
    assert summary["reference_review_sha256"] == pins["review_sha256"]
    assert {row["reference_id"] for row in summary["comparisons"]} == set(
        pins["reference_ids"]
    )
    for row in summary["comparisons"]:
        path = output / row["path"]
        assert sha256(path.read_bytes()).hexdigest() == row["sha256"]
        report = read_json(path)
        assert row["prediction_value"] == report["prediction"]["value"]
        assert row["reference_value"] == report["reference"]["value"]
        assert row["residuals"]["signed_residual"] == (
            row["prediction_value"] - row["reference_value"]
        )
        assert (
            row["comparison_context"]["comparison_scope"] == "cross_method_descriptive"
        )
        assert row["comparison_context"]["method_mismatch"]
        assert row["comparison_context"]["basis_mismatch"]
        assert report["reference_standard_uncertainty"] is None
        assert not report["held_out_benchmark"]
    assert (
        "physical_harmonic_mode_assignment_not_established"
        in (summary["reference_identity_quality_flags"])
    )
    assert (
        "isotope_identity_not_explicit_in_original_source"
        in (summary["reference_identity_quality_flags"])
    )
    assert (output / "run.log").read_text().count("Comparing qm9-water-") == 6
    assert summary_path.stat().st_mode & 0o222 == 0
    before = sha256(summary_path.read_bytes()).hexdigest()
    refused = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert refused.returncode == 2
    assert sha256(summary_path.read_bytes()).hexdigest() == before
