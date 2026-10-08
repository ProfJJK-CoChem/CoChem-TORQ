"""Audit coverage and genuine software execution evidence; no engine stand-ins."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import pytest

from ci_tools.validate_release_evidence import (
    AuditError,
    inspect_execution_evidence,
    load_json,
    safe_reference,
    source_tree_sha256,
    validate_audit,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/development/full_srs_requirement_audit.json"


def test_complete_document_coverage_remains_incomplete_scientific_release():
    report = validate_audit(ROOT, load_json(AUDIT))
    assert report["coverage_valid"]
    assert report["requirement_count"] == 73
    assert report["acceptance_count"] == 41
    assert not report["full_srs_release_ready"]
    assert report["unresolved_acceptance"]
    assert report["unresolved_requirements"]


def test_full_release_gate_refuses_real_unmet_requirements():
    with pytest.raises(AuditError, match="Full-SRS release gate remains unresolved"):
        validate_audit(ROOT, load_json(AUDIT), require_full_release=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ci_tools/validate_release_evidence.py"),
            "--root",
            str(ROOT),
            "--require-full-release",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert '"full_srs_release_ready": false' in result.stdout


@pytest.mark.parametrize("field", ["requirements", "acceptance"])
def test_missing_or_duplicate_normative_id_is_not_coverage(field):
    audit = deepcopy(load_json(AUDIT))
    audit[field].pop()
    with pytest.raises(AuditError, match="coverage mismatch"):
        validate_audit(ROOT, audit)
    audit = deepcopy(load_json(AUDIT))
    audit[field].append(deepcopy(audit[field][0]))
    with pytest.raises(AuditError, match="coverage mismatch"):
        validate_audit(ROOT, audit)


def test_source_presence_and_declared_pass_cannot_replace_execution_evidence():
    audit = deepcopy(load_json(AUDIT))
    audit["acceptance"][0]["status"] = "passed"
    with pytest.raises(AuditError, match="required actual evidence"):
        validate_audit(ROOT, audit)


def test_stale_specification_and_unknown_gate_reference_are_rejected():
    audit = deepcopy(load_json(AUDIT))
    audit["specification"]["sha256"] = "0" * 64
    with pytest.raises(AuditError, match="canonical SRS changed"):
        validate_audit(ROOT, audit)
    audit = deepcopy(load_json(AUDIT))
    audit["requirements"][0]["acceptance_ids"] = ["V-NONEXISTENT"]
    with pytest.raises(AuditError, match="unknown acceptance"):
        validate_audit(ROOT, audit)


@pytest.mark.parametrize(
    "corruption", ["omitted_dependency", "duplicate_dependency", "weakened_kind"]
)
def test_mandatory_scientific_gates_cannot_be_removed_or_diluted(corruption):
    audit = deepcopy(load_json(AUDIT))
    requirement = next(
        record for record in audit["requirements"] if record["id"] == "TORQ-GEO-003"
    )
    if corruption == "omitted_dependency":
        requirement["acceptance_ids"].remove("V-TS")
        message = "omits canonical mandatory"
    elif corruption == "duplicate_dependency":
        requirement["acceptance_ids"].append("V-TS")
        message = "duplicate acceptance"
    else:
        gate = next(record for record in audit["acceptance"] if record["id"] == "V-DH")
        gate["required_evidence_kinds"].remove("independent_reference")
        message = "weakens mandatory evidence"
    with pytest.raises(AuditError, match=message):
        validate_audit(ROOT, audit)


@pytest.mark.parametrize(
    "relative", ["../outside.xml", "/tmp/report.xml", "tests\\escape.xml"]
)
def test_audit_paths_cannot_escape_the_repository(relative):
    with pytest.raises(AuditError, match="within the repository"):
        safe_reference(ROOT, relative)


def test_source_identity_is_content_based_and_deterministic(tmp_path):
    # Take an actual immutable source snapshot so concurrent implementation work
    # cannot make this mathematical identity check depend on editor timing.
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    directory = tmp_path / "src/cochem_torq"
    directory.mkdir(parents=True)
    (directory / "domain.py").write_bytes(
        (ROOT / "src/cochem_torq/domain.py").read_bytes()
    )
    first = source_tree_sha256(tmp_path)
    assert len(first) == 64 and first == source_tree_sha256(tmp_path)
    with (directory / "domain.py").open("a") as source:
        source.write("\n# Deliberate changed-byte integrity check.\n")
    assert source_tree_sha256(tmp_path) != first


@pytest.fixture(scope="module")
def genuine_mathematical_report(tmp_path_factory):
    # These are the project's real WKB limiting-case software tests. Their genuine
    # execution is not labeled an electronic-engine calculation or spectroscopy
    # qualification by this test or by the audit.
    directory = tmp_path_factory.mktemp("genuine-mathematical-evidence")
    report = directory / "actual-wkb.xml"
    execution_directory = directory / "cochem_exec_actual_mathematical_tests"
    execution_directory.mkdir()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_wkb_integrity.py"),
            "--junitxml",
            str(report),
            "-p",
            "no:cacheprovider",
        ],
        cwd=execution_directory,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    inspect_execution_evidence(report)
    return report


def test_real_mathematical_junit_evidence_and_deliberately_corrupted_copy(
    tmp_path,
    genuine_mathematical_report,
):
    report = genuine_mathematical_report
    tree = ElementTree.parse(report)
    testcase = next(tree.getroot().iter("testcase"))
    ElementTree.SubElement(
        testcase, "failure"
    ).text = "Deliberate corruption of genuine report"
    corrupted = tmp_path / "corrupted.xml"
    tree.write(corrupted)
    with pytest.raises(AuditError, match="actual passing, zero-skip"):
        inspect_execution_evidence(corrupted)
    empty = tmp_path / "empty.xml"
    empty.touch()
    with pytest.raises(AuditError, match="empty"):
        inspect_execution_evidence(empty)


@pytest.mark.parametrize("node_type", ["testsuites", "testsuite"])
@pytest.mark.parametrize(
    "attribute", ["failures", "errors", "skipped", "disabled", "tests"]
)
def test_declared_junit_outcomes_and_counts_cannot_override_actual_cases(
    tmp_path,
    genuine_mathematical_report,
    node_type,
    attribute,
):
    tree = ElementTree.parse(genuine_mathematical_report)
    node = next(node for node in tree.getroot().iter() if node.tag == node_type)
    node.set(
        attribute,
        str(len(list(node.iter("testcase"))) + 1) if attribute == "tests" else "1",
    )
    corrupted = tmp_path / "corrupted-actual-wkb.xml"
    tree.write(corrupted)
    message = (
        "test counts disagree" if attribute == "tests" else "actual passing, zero-skip"
    )
    with pytest.raises(AuditError, match=message):
        inspect_execution_evidence(corrupted)
