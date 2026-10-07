"""Check the actual student front door and its executable public contracts.

Historical documentation tests enforced unsupported universal accuracy claims and
obsolete interfaces. These checks resolve the current links, execute the actual
CLI discovery/validation path, and keep research/deployment limits explicit.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "Readme.md"
GUIDE = ROOT / "docs" / "Student_Guide.md"


@pytest.mark.parametrize("document", [README, GUIDE])
def test_student_document_links_resolve_to_actual_files(document):
    raw = document.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        if target.startswith(("https://", "http://")):
            continue
        local = target.split("#", 1)[0]
        assert (document.parent / local).is_file(), f"{document}: unresolved {target}"


def actual_cli(*arguments):
    executed = subprocess.run(
        [sys.executable, "-m", "cochem_torq.cli", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    return json.loads(executed.stdout)


def test_documented_student_recipe_discovery_and_input_validate():
    readme = README.read_text()
    profiles = {profile["id"]: profile for profile in actual_cli("recipes")}
    for identifier in ("hf-sto-3g-education", "hf-cc-pvdz-research"):
        assert identifier in readme
        assert profiles[identifier]["runnable"]
        assert profiles[identifier]["availability"] == "experimental"
        assert "uncalibrated" in profiles[identifier]["accuracy"]
    checked = actual_cli(
        "validate", "--request", str(ROOT / "examples/student/water-hf-teaching.json")
    )
    assert checked["executable"]
    assert checked["execution"] == "github_actions"
    assert checked["request"]["molecule"]["charge"] == 0
    assert checked["request"]["molecule"]["multiplicity"] == 1
    assert checked["blocked_products"] == []
    assert checked["warnings"]


def test_frontdoor_does_not_promote_a_documented_matrix_row_to_method_support():
    matrix = actual_cli("matrix")
    assert len(matrix) == 140
    assert len({row["legacy_row_id"] for row in matrix}) == 140
    assert {row["availability"] for row in matrix} == {"documented"}
    assert {row["dispatch"] for row in matrix} == {"requires_exact_qualified_recipe"}
    readme = README.read_text()
    assert "a documented row does not activate a scientific profile" in readme
    assert "default branch" in readme
    assert "does not establish identification accuracy" in readme


def test_source_license_and_canonical_environment_docs_are_accurate():
    readme = README.read_text()
    assert "Apache License 2.0" in readme
    assert "Apache License" in (ROOT / "LICENSE").read_text()
    assert "GitHub Actions is the canonical" in readme
    assert "GitHub Codespaces is the canonical interface" in readme
    assert "separate\nvirtual environments" in readme
    assert "workflows/calculation.yml" in readme
    assert "12.74%" not in readme
    assert "0.93 F1" not in readme
