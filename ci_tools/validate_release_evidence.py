"""Check full-SRS coverage without turning source presence into execution evidence.

The default mode checks the audit contract and reports unresolved requirements.
``--require-full-release`` additionally requires every acceptance definition to
have actual, source-bound passing evidence of the required kinds. This does not
authenticate a scientific reference or replace its independent scientific review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

SCHEMA = "cochem.torq.full-srs-audit/1"
CLASSIFICATIONS = {
    "missing_implementation",
    "missing_independent_qualification",
    "optional_profile_or_credentials",
    "partial_supported_scope",
    "implemented_and_qualified",
}
STATUSES = {"passed", "failed", "skipped", "blocked", "unrun"}
EVIDENCE_KINDS = {"software", "real_engine", "independent_reference", "platform"}
# These are minimum release evidence classes, not optional audit labels. An
# edited audit cannot weaken a scientific or deployment gate into a software
# gate. Stronger requirements may be declared for a particular enabled profile.
MINIMUM_GATE_EVIDENCE = {
    **dict.fromkeys(
        (
            "V-SCHEMA",
            "V-UNITS",
            "V-GEO",
            "V-ROT",
            "V-ISO",
            "V-SYM",
            "V-METHOD",
            "V-PLAN",
            "V-FAIL",
            "V-STATE",
            "V-RESTART",
            "V-STORAGE",
            "V-EXPORT",
            "V-PACKAGE",
            "V-RELEASE",
        ),
        frozenset({"software"}),
    ),
    **dict.fromkeys(
        (
            "V-STAGES",
            "V-ENGINE",
            "V-OPT",
            "V-ELECTRONIC",
            "V-DERIV",
            "V-VIB",
            "V-PES",
            "V-TS",
        ),
        frozenset({"software", "real_engine"}),
    ),
    **dict.fromkeys(
        (
            "V-VPT2",
            "V-AL",
            "V-CP",
            "V-DH",
            "V-PROP",
            "V-WFN",
            "V-ANALYSIS",
            "V-KINETICS",
            "V-BENCH",
            "V-SPCAT",
        ),
        frozenset({"software", "real_engine", "independent_reference"}),
    ),
    **dict.fromkeys(
        ("V-DVR", "V-UQ", "V-CONSUMER"),
        frozenset({"software", "independent_reference"}),
    ),
    **dict.fromkeys(
        ("V-BUDGET", "V-SECURITY", "V-E2E", "V-HPC", "V-DEPLOY"),
        frozenset({"software", "platform"}),
    ),
}
MAX_REPORT_BYTES = 16 * 1024 * 1024
SOURCE_ROOTS = (
    "Libraries",
    "cochem",
    "src",
    "orchestrator",
    "core",
    "UI",
    "ci_tools",
    "scripts",
    "tests",
    ".github",
    "docker",
    ".devcontainer",
    "HPC_Launchers",
)


class AuditError(ValueError):
    """A coverage declaration or evidence reference is invalid."""


def load_json(path: Path) -> dict:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise AuditError("A bounded, existing JSON audit/report is required.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise AuditError(f"Nonfinite JSON value: {value}")

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=nonfinite,
    )
    if not isinstance(result, dict):
        raise AuditError("The audit/report must be a JSON object.")
    return result


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash reviewable code, tests and runtime configuration, including new files."""
    files = {
        root / "pyproject.toml",
        root / "pytest.ini",
        root / "conftest.py",
        root / ".dockerignore",
    }
    for name in SOURCE_ROOTS:
        directory = root / name
        if directory.is_symlink():
            raise AuditError("Source identity cannot follow symlinks.")
        if directory.is_dir():
            files.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and not any(
                    part in {"__pycache__", ".pytest_cache", ".git", "build", "dist"}
                    for part in path.relative_to(root).parts
                )
                and path.suffix not in {".pyc", ".pyo"}
            )
    records = []
    for path in sorted(files):
        if path.is_symlink():
            raise AuditError("Source identity cannot follow symlinks.")
        if path.is_file():
            records.append(
                {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
            )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_reference(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AuditError("A nonempty repository-relative evidence path is required.")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise AuditError("Evidence/source paths must stay within the repository.")
    target = root / relative
    if any(
        (root / Path(*path.parts[:index])).is_symlink()
        for index in range(1, len(path.parts) + 1)
    ):
        raise AuditError("Evidence/source paths cannot contain symlinks.")
    if not target.is_file():
        raise AuditError(f"Referenced source/evidence file is absent: {relative}")
    return target


def specification_ids(path: Path) -> tuple[set[str], set[str]]:
    contents = path.read_text(encoding="utf-8")
    requirements = re.findall(r"^\*\*(TORQ-[A-Z]+-\d+) —", contents, re.M)
    acceptance = re.findall(r"^\| (V-[A-Z0-9]+) \|", contents, re.M)
    if len(requirements) != len(set(requirements)) or len(acceptance) != len(
        set(acceptance)
    ):
        raise AuditError("Canonical SRS contains duplicate requirement/acceptance IDs.")
    if not requirements or not acceptance:
        raise AuditError(
            "Canonical normative requirements and acceptance IDs are required."
        )
    return set(requirements), set(acceptance)


def specification_dependencies(path: Path) -> dict[str, set[str]]:
    """Read mandatory requirement/gate links from the canonical specification."""
    contents = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^\*\*TORQ-[A-Z]+-\d+ —)", contents, flags=re.M)[1:]
    dependencies = {}
    for section in sections:
        identifier = re.match(r"\*\*(TORQ-[A-Z]+-\d+) —", section).group(1)
        declaration = re.search(r"Acceptance:\s*([^\n]*)", section)
        if declaration is None:
            raise AuditError(f"{identifier} has no canonical acceptance declaration.")
        gates = set(re.findall(r"V-[A-Z0-9]+", declaration.group(1)))
        if not gates:
            raise AuditError(f"{identifier} has no canonical acceptance dependencies.")
        dependencies[identifier] = gates
    return dependencies


def exact_records(records: list, expected: set[str], label: str) -> dict[str, dict]:
    if not isinstance(records, list):
        raise AuditError(f"{label} records must be a list.")
    if any(
        not isinstance(record, dict) or not isinstance(record.get("id"), str)
        for record in records
    ):
        raise AuditError(f"Every {label} record needs its explicit ID.")
    indexed = {record["id"]: record for record in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise AuditError(
            f"{label} coverage mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}, duplicates="
            f"{len(records) - len(indexed)}"
        )
    return indexed


def inspect_execution_evidence(path: Path) -> None:
    if not 0 < path.stat().st_size <= MAX_REPORT_BYTES:
        raise AuditError("Execution evidence is empty or exceeds its size bound.")
    if path.suffix == ".xml":
        root = ElementTree.parse(path).getroot()
        cases = list(root.iter("testcase"))
        if (
            root.tag not in {"testsuite", "testsuites"}
            or not cases
            or any(
                case.find(tag) is not None
                for case in cases
                for tag in ("failure", "error", "skipped")
            )
        ):
            raise AuditError(
                "JUnit evidence must contain actual passing, zero-skip cases."
            )
        for suite in (
            node for node in root.iter() if node.tag in {"testsuite", "testsuites"}
        ):
            for outcome in ("failures", "errors", "skipped", "disabled"):
                declared = suite.get(outcome)
                if declared is not None and declared != "0":
                    raise AuditError(
                        "JUnit evidence must contain actual passing, zero-skip cases."
                    )
            declared_tests = suite.get("tests")
            if declared_tests is not None:
                if not declared_tests.isdecimal() or int(declared_tests) != len(
                    list(suite.iter("testcase"))
                ):
                    raise AuditError(
                        "JUnit test counts disagree with actual case records."
                    )
    elif path.suffix == ".json":
        report = load_json(path)
        if (
            report.get("schema_version") != "cochem.torq.release-execution/1"
            or report.get("status") != "passed"
            or report.get("execution_performed") is not True
            or not isinstance(report.get("executions"), list)
            or not report["executions"]
            or any(
                not isinstance(entry, dict) or entry.get("status") != "passed"
                for entry in report["executions"]
            )
        ):
            raise AuditError(
                "JSON evidence requires the executed release-report contract."
            )
    else:
        raise AuditError(
            "Execution evidence must be an actual JUnit or release-report file."
        )


def validate_audit(
    root: Path, audit: dict, *, require_full_release: bool = False
) -> dict:
    root = root.resolve()
    if audit.get("schema_version") != SCHEMA:
        raise AuditError("Unsupported full-SRS audit schema.")
    if not isinstance(audit.get("specification"), dict):
        raise AuditError("An audit requires its canonical specification identity.")
    specification = safe_reference(root, audit.get("specification", {}).get("path"))
    if file_sha256(specification) != audit["specification"].get("sha256"):
        raise AuditError("The canonical SRS changed; review and update the audit.")
    requirement_ids, acceptance_ids = specification_ids(specification)
    mandatory_dependencies = specification_dependencies(specification)
    if set(MINIMUM_GATE_EVIDENCE) != acceptance_ids:
        raise AuditError(
            "The canonical acceptance catalog needs evidence-policy review."
        )
    requirements = exact_records(
        audit.get("requirements"), requirement_ids, "Requirement"
    )
    gates = exact_records(audit.get("acceptance"), acceptance_ids, "Acceptance")
    for record in requirements.values():
        if record.get("classification") not in CLASSIFICATIONS or not record.get(
            "remaining_work"
        ):
            raise AuditError(
                f"{record['id']} needs an explicit supported-scope/gap classification."
            )
        if (
            not isinstance(record.get("acceptance_ids"), list)
            or not record["acceptance_ids"]
            or set(record["acceptance_ids"]) - acceptance_ids
        ):
            raise AuditError(
                f"{record['id']} has absent/unknown acceptance dependencies."
            )
        if len(set(record["acceptance_ids"])) != len(record["acceptance_ids"]):
            raise AuditError(f"{record['id']} has duplicate acceptance dependencies.")
        if not mandatory_dependencies[record["id"]] <= set(record["acceptance_ids"]):
            raise AuditError(
                f"{record['id']} omits canonical mandatory acceptance dependencies."
            )
        for field in ("source_references", "test_references"):
            if not isinstance(record.get(field), list):
                raise AuditError(f"{record['id']} needs an explicit {field} list.")
        for relative in record.get("source_references", []) + record.get(
            "test_references", []
        ):
            safe_reference(root, relative)
    evidences = audit.get("actual_evidence", [])
    if not isinstance(evidences, list):
        raise AuditError(
            "Actual evidence must be a list, including an explicit "
            "empty list when unrun."
        )
    evidence_by_id = {}
    for evidence in evidences:
        if (
            not isinstance(evidence, dict)
            or not evidence.get("id")
            or evidence["id"] in evidence_by_id
        ):
            raise AuditError("Execution evidence needs unique explicit IDs.")
        if evidence.get("kind") not in EVIDENCE_KINDS:
            raise AuditError(
                "Execution evidence needs its software/scientific/platform kind."
            )
        path = safe_reference(root, evidence.get("path"))
        if file_sha256(path) != evidence.get("sha256"):
            raise AuditError(
                "Execution evidence bytes changed from their recorded identity."
            )
        inspect_execution_evidence(path)
        if evidence.get("source_tree_sha256") != source_tree_sha256(root):
            raise AuditError(
                "Execution evidence belongs to a different "
                "source/test/configuration tree."
            )
        evidence_by_id[evidence["id"]] = evidence
    unresolved = []
    for identifier, gate in gates.items():
        if gate.get("status") not in STATUSES or not gate.get("reason"):
            raise AuditError(f"{identifier} needs actual status and its reason.")
        needed = gate.get("required_evidence_kinds")
        if not isinstance(needed, list) or not needed or set(needed) - EVIDENCE_KINDS:
            raise AuditError(f"{identifier} needs an explicit evidence-kind contract.")
        if not MINIMUM_GATE_EVIDENCE[identifier] <= set(needed):
            raise AuditError(f"{identifier} weakens mandatory evidence kinds.")
        referenced = gate.get("evidence_ids", [])
        if not isinstance(referenced, list) or set(referenced) - set(evidence_by_id):
            raise AuditError(f"{identifier} references unknown execution evidence.")
        supplied = {evidence_by_id[identity]["kind"] for identity in referenced}
        if gate["status"] == "passed" and not set(needed) <= supplied:
            raise AuditError(
                f"{identifier} cannot pass without all required actual evidence kinds."
            )
        if gate["status"] != "passed":
            unresolved.append(
                {"id": identifier, "status": gate["status"], "reason": gate["reason"]}
            )
    for record in requirements.values():
        if record["classification"] == "implemented_and_qualified" and any(
            gates[identifier]["status"] != "passed"
            for identifier in record["acceptance_ids"]
        ):
            raise AuditError(
                f"{record['id']} cannot be qualified while its "
                "required gates are unresolved."
            )
    unresolved_requirements = [
        record["id"]
        for record in requirements.values()
        if record["classification"] != "implemented_and_qualified"
    ]
    result = {
        "schema_version": "cochem.torq.release-audit-result/1",
        "coverage_valid": True,
        "requirement_count": len(requirements),
        "acceptance_count": len(gates),
        "full_srs_release_ready": not unresolved and not unresolved_requirements,
        "unresolved_requirements": unresolved_requirements,
        "unresolved_acceptance": unresolved,
        "classifications": {
            name: sum(
                record["classification"] == name for record in requirements.values()
            )
            for name in sorted(CLASSIFICATIONS)
        },
    }
    if require_full_release and (unresolved or unresolved_requirements):
        raise AuditError(
            "Full-SRS release gate remains unresolved: "
            + ", ".join(f"{item['id']}={item['status']}" for item in unresolved)
            + f"; {len(unresolved_requirements)} requirements "
            "remain incompletely qualified"
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/development/full_srs_requirement_audit.json"),
    )
    parser.add_argument("--require-full-release", action="store_true")
    args = parser.parse_args(argv)
    try:
        audit = load_json(
            args.audit if args.audit.is_absolute() else args.root / args.audit
        )
        report = validate_audit(
            args.root, audit, require_full_release=args.require_full_release
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AuditError, OSError, json.JSONDecodeError, ElementTree.ParseError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "cochem.torq.release-audit-result/1",
                    "coverage_valid": False,
                    "full_srs_release_ready": False,
                    "error": str(error),
                },
                allow_nan=False,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
