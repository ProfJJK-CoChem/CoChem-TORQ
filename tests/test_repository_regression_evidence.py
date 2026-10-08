"""Genuine native inventories and actual pytest report privacy/process checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree

import pytest

from ci_tools.run_repository_regression import curate_engine_evidence, sanitize_junit

ROOT = Path(__file__).resolve().parents[1]
MARKER = "DELIBERATELY_NONCREDENTIAL_PRIVATE_DIAGNOSTIC"


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _replace_inventory_entry(root, relative):
    """Reseal an explicitly corrupted privacy control, never engine observations."""
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    actual = root / relative
    entry = {
        "path": relative,
        "size_bytes": actual.stat().st_size,
        "sha256": _digest(actual),
    }
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != relative
    ] + [entry]
    path.write_text(json.dumps(manifest, sort_keys=True))


@pytest.fixture(scope="module")
def actual_mathematical_junit(tmp_path_factory):
    root = tmp_path_factory.mktemp("cochem_exec_genuine_junit_privacy")
    test = root / "test_real_mathematical_process.py"
    test.write_text(
        "from fractions import Fraction\n"
        "def test_exact_fraction_identity(record_property):\n"
        f"    record_property('private_diagnostic', '{MARKER}')\n"
        f"    print('{MARKER}')\n"
        "    assert Fraction(1, 3) * 3 == 1\n"
        "def test_deliberate_process_rejection():\n"
        f"    raise ValueError('{MARKER}')\n"
    )
    report = root / "actual-report.xml"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test),
            "--junitxml",
            str(report),
            "-o",
            "junit_logging=all",
            "-o",
            "junit_family=xunit1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    assert MARKER in report.read_text()
    return report


def test_curator_cold_import_does_not_load_optional_quantum_engine():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from ci_tools.run_repository_regression import curate_engine_evidence; "
            "assert not any(n == 'pyscf' or n.startswith('pyscf.') "
            "for n in sys.modules)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_actual_junit_preserves_failures_and_removes_private_bodies(
    actual_mathematical_junit,
    tmp_path,
):
    before = _digest(actual_mathematical_junit)
    public = tmp_path / "public.xml"
    report = sanitize_junit(actual_mathematical_junit, public)
    assert report["tests"] == 2 and report["failures"] == 1
    assert report["errors"] == report["skipped"] == 0
    assert report["raw_report_sha256"] == before == _digest(actual_mathematical_junit)
    assert report["public_report_sha256"] == _digest(public)
    assert MARKER not in public.read_text()
    cases = list(ElementTree.parse(public).getroot().iter("testcase"))
    assert len(cases) == 2
    assert sum(case.find("failure") is not None for case in cases) == 1
    assert not list(ElementTree.parse(public).getroot().iter("properties"))
    assert not list(ElementTree.parse(public).getroot().iter("system-out"))
    assert not list(ElementTree.parse(public).getroot().iter("system-err"))
    assert all(
        "file" not in case.attrib and "line" not in case.attrib for case in cases
    )


def test_linked_actual_junit_cannot_be_processed(actual_mathematical_junit, tmp_path):
    linked = tmp_path / "linked.xml"
    linked.symlink_to(actual_mathematical_junit)
    with pytest.raises(ValueError, match="bounded actual JUnit"):
        sanitize_junit(linked, tmp_path / "public.xml")


@pytest.fixture(scope="module")
def actual_h2_native(tmp_path_factory):
    from cochem_torq.engines.pyscf_backend import PySCFBackend

    root = tmp_path_factory.mktemp("genuine-regression-h2") / "native"
    result = PySCFBackend().evaluate(
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry_bohr": [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
                "charge": 0,
                "multiplicity": 1,
            },
            "method": {
                "name": "hf",
                "basis": "sto-3g",
                "reference": "restricted",
                "frozen_core": False,
            },
            "properties": ["energy", "gradient"],
            "settings": {"threads": 1},
        },
        root,
    )
    assert result["status"] == "complete", result["errors"]
    assert result["engine_version"] == "2.14.0"
    assert result["energy_hartree"] == pytest.approx(-1.11671432506255, abs=2e-10)
    return root


@pytest.mark.real_engine
def test_actual_native_bytes_are_copied_and_undeclared_authority_is_excluded(
    actual_h2_native,
    tmp_path,
):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    (source / "authority-token.json").write_text(json.dumps({"token": MARKER}))
    destination = tmp_path / "public"
    report = curate_engine_evidence(source.parent, destination, ROOT)
    assert len(report["bundles"]) == 1
    assert report["rejected_or_unsupported"] == 0
    assert not report["independent_scientific_qualification"]
    identity = _digest(source / "manifest.json")
    bundle = destination / identity
    manifest = json.loads((source / "manifest.json").read_text())
    declared = {item["path"] for item in manifest["artifacts"]} | {"manifest.json"}
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert actual == declared
    assert not list(destination.rglob("*token*"))
    assert report["bundles"][0]["manifest_sha256"] == identity
    assert all(_digest(source / name) == _digest(bundle / name) for name in declared)
    assert report["bytes_copied"] == sum(
        (source / name).stat().st_size for name in declared
    )


@pytest.mark.real_engine
@pytest.mark.parametrize("privacy_case", ["filename", "json-field", "url", "log"])
def test_declared_private_content_cannot_escape_actual_native_inventory(
    actual_h2_native,
    tmp_path,
    privacy_case,
):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    if privacy_case == "filename":
        relative = "student_authority.json"
        (source / relative).write_text(json.dumps({"diagnostic": MARKER}))
    elif privacy_case == "json-field":
        relative = "calculation-note.json"
        (source / relative).write_text(json.dumps({"token": MARKER}))
    elif privacy_case == "url":
        relative = "calculation-note.json"
        (source / relative).write_text(
            json.dumps(
                {
                    "diagnostic": f"https://test-reader:{MARKER}@example.invalid/evidence",
                }
            )
        )
    else:
        relative = "pyscf.log"
        with (source / relative).open("a") as stream:
            stream.write(f"\nAuthorization: Bearer {MARKER}\n")
    _replace_inventory_entry(source, relative)
    destination = tmp_path / "public"
    report = curate_engine_evidence(source.parent, destination, ROOT)
    assert report["bundles"] == [] and report["rejected_or_unsupported"] == 1
    assert report["bytes_copied"] == 0
    assert not list(destination.rglob("*"))


@pytest.mark.real_engine
def test_corrupted_declared_native_bytes_are_rejected(actual_h2_native, tmp_path):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    with (source / "wavefunction.chk").open("ab") as stream:
        stream.write(b"deliberate native-byte corruption rejection case")
    destination = tmp_path / "public"
    report = curate_engine_evidence(source.parent, destination, ROOT)
    assert report["bundles"] == [] and report["rejected_or_unsupported"] == 1
    assert not list(destination.rglob("*"))


@pytest.mark.real_engine
def test_native_source_identity_cannot_be_substituted(actual_h2_native, tmp_path):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    result = json.loads((source / "result.json").read_text())
    result["adapter_source_sha256"] = "0" * 64
    (source / "result.json").write_text(json.dumps(result))
    _replace_inventory_entry(source, "result.json")
    report = curate_engine_evidence(source.parent, tmp_path / "public", ROOT)
    assert report["bundles"] == [] and report["rejected_or_unsupported"] == 1


@pytest.mark.real_engine
def test_declared_symlink_cannot_be_copied(actual_h2_native, tmp_path):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    original = source / "pyscf.log"
    outside = tmp_path / "outside.log"
    shutil.copyfile(original, outside)
    original.unlink()
    original.symlink_to(outside)
    report = curate_engine_evidence(source.parent, tmp_path / "public", ROOT)
    assert report["bundles"] == [] and report["rejected_or_unsupported"] == 1


@pytest.mark.real_engine
def test_native_inventory_cannot_escape_its_owned_directory(actual_h2_native, tmp_path):
    source = tmp_path / "source" / "native"
    shutil.copytree(actual_h2_native, source)
    outside = source.parent / "outside.chk"
    shutil.copyfile(source / "wavefunction.chk", outside)
    before = _digest(outside)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append(
        {
            "path": "../outside.chk",
            "size_bytes": outside.stat().st_size,
            "sha256": before,
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "public"
    report = curate_engine_evidence(source.parent, destination, ROOT)
    assert report["bundles"] == [] and report["rejected_or_unsupported"] == 1
    assert _digest(outside) == before and not list(destination.rglob("*"))


@pytest.mark.real_engine
def test_curated_evidence_never_overwrites_an_existing_target(
    actual_h2_native, tmp_path
):
    destination = tmp_path / "public"
    curate_engine_evidence(actual_h2_native.parent, destination, ROOT)
    before = {
        path.relative_to(destination).as_posix(): _digest(path)
        for path in destination.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        curate_engine_evidence(actual_h2_native.parent, destination, ROOT)
    assert before == {
        path.relative_to(destination).as_posix(): _digest(path)
        for path in destination.rglob("*")
        if path.is_file()
    }
