"""Integrity checks on actual pinned public calculation source bytes."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "benchmarks/published-values/qcarchive-nh3"
HELPER = runpy.run_path(str(ROOT / "scripts/run_qcarchive_component_comparison.py"))


def test_original_notebook_native_output_and_license_are_bound() -> None:
    reference = HELPER["load_reference"](PACKET)
    assert reference["source_record_id"] == 1847316
    assert reference["source_molecule_id"] == 2
    assert reference["geometry_bohr_as_printed"][0] == [
        "N",
        "0.000538830000",
        "0.000000000000",
        "0.129095460000",
    ]
    assert (
        reference["source_values_hartree_as_printed"]["double_hybrid_total_energy"]
        == "-56.5380912967599372"
    )
    assert reference["source_engine_version"] is None
    assert reference["source_uncertainty"] is None
    assert reference["benchmark_acceptance"] is False


@pytest.mark.parametrize(
    "name", ["source-notebook.ipynb", "source-stdout.txt", "source-license.txt"]
)
def test_altered_original_source_is_rejected(tmp_path: Path, name: str) -> None:
    copy = tmp_path / "actual-source-copy"
    shutil.copytree(PACKET, copy)
    target = copy / name
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        HELPER["load_reference"](copy)


def test_metadata_matches_actual_source_files() -> None:
    metadata = json.loads((PACKET / "source-metadata.json").read_bytes())
    for item in metadata["source_files"]:
        actual = (PACKET / item["path"]).read_bytes()
        assert len(actual) == item["size_bytes"]
        assert hashlib.sha256(actual).hexdigest() == item["sha256"]
    assert metadata["source_commit"] == "60ee49a3dc881cd64d42840a2a5f3bf31f7b049f"
    assert metadata["license"] == "BSD-3-Clause"
    assert metadata["independent_accuracy_qualification"] is False
    assert metadata["live_api_record_retrieved"] is False


def test_source_only_command_does_not_claim_qualification(capsys) -> None:
    assert HELPER["main"](["--verify-source-only"]) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual["status"] == "original source bytes verified"
    assert actual["benchmark_acceptance"] is False


def test_precommit_receipt_remains_descriptive() -> None:
    actual = json.loads(
        (PACKET / "precommit-native-verification-summary.json").read_bytes()
    )
    assert actual["actual_original_worker_waited"] is True
    assert actual["actual_exit_code"] == 0
    assert actual["actual_peak_owned_tree_rss_bytes"] <= actual["rss_budget_bytes"]
    assert actual["elapsed_seconds"] <= actual["wall_budget_seconds"]
    assert actual["scientific_profile_qualification"] is False
    assert actual["benchmark_acceptance"] is False
    assert actual["comparison_tolerance"] is None
    assert actual["curator_attestation"] is False
    assert actual["exact_revdsd_claimed"] is False
