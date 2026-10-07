"""Validate SRS review coverage and document contracts; does not test TORQ physics."""

from __future__ import annotations

import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


def main() -> None:
    review = Path(__file__).resolve().parent
    wiki = review.parent
    repo = wiki.parents[1]
    inventory = json.loads((review / "SRS_Point_Inventory.json").read_text())
    revision = inventory["source_revision"]
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def original(name: str) -> bytes:
        return subprocess.check_output(
            ["git", "show", f"{revision}:docs/wiki/{name}"], cwd=repo
        )

    points = inventory["points"]
    point_ids = [point["id"] for point in points]
    require(len(point_ids) == len(set(point_ids)), "Duplicate source-point IDs")
    require(len(points) == 601, "Expected 601 source-point records")
    actual_counts = dict(collections.Counter(p["kind"] for p in points))
    require(actual_counts == inventory["counts"], "Inventory kind counts disagree")
    require(all(p["disposition"] in {"accepted", "corrected", "deferred", "rejected"}
                for p in points), "Unknown source-point disposition")
    require(all(p["reason"] and p["implementation_srs_sections"] for p in points),
            "Source point lacks rationale or SRS target")
    for name, metadata in inventory["sources"].items():
        raw = original(name)
        # The inventory hashes normalized text, retaining original source lines.
        text = raw.decode().replace("\r\n", "\n")
        require(hashlib.sha256(text.encode()).hexdigest() == metadata["sha256"],
                f"Source hash mismatch: {name}")
        lines = text.splitlines()
        require(len(lines) == metadata["lines"], f"Source line-count mismatch: {name}")
        for point in (p for p in points if p["source"] == name):
            require(1 <= point["line"] <= point["end_line"] <= len(lines),
                    f"Invalid source span: {point['id']}")
        if name.endswith(("Comprehensive_SRS.md", "True_SRS_Iterated.md")):
            expected = {i for i, line in enumerate(lines, 1)
                        if re.match(r"\s*- \[ \]", line)}
            covered = {p["line"] for p in points
                       if p["source"] == name and p["kind"] == "proposal"}
            require(expected == covered, f"Checkbox coverage mismatch: {name}")

    historical = [
        "CoChem-TORQ_Comprehensive_SRS.md", "CoChem-TORQ_True_SRS.md",
        "CoChem-TORQ_True_SRS_Iterated.md", "CoChem-TORQ_AutoArchitect_Phase1.md",
        "CoChem-TORQ_Integration_Architecture.md", "Method_Matrix.md",
        "CoChem_User_Manual.md", "CoChem-BASE_Draco_Blueprint.md", "NBO_Analysis.md",
    ]
    for name in historical:
        current = (wiki / name).read_bytes()
        require(current.startswith(b"> **Historical source"), f"No banner: {name}")
        require(current.partition(b"\n\n")[2] == original(name),
                f"Historical content modified: {name}")

    matrix = original("Method_Matrix.md").decode()
    row_ids = set(re.findall(r"\*\*(T\d+[OC]?-(?:10s|1min|30min|1h|3h|12h|1d|3d|1w|1mo))\*\*", matrix))
    matrix_review = (review / "Method_Matrix_Review.md").read_text()
    require(len(row_ids) == 140, f"Expected 140 method rows, got {len(row_ids)}")
    for row in row_ids:
        require(re.search(rf"(?<![\w-]){re.escape(row)}(?![\w-])", matrix_review) is not None,
                f"Unreviewed matrix row: {row}")
    for recipe in range(1, 10):
        require(re.search(rf"\bR{recipe}\b", matrix_review) is not None,
                f"Missing recipe R{recipe}")

    srs = (wiki / "CoChem-TORQ_Implementation_SRS.md").read_text()
    definitions = re.findall(r"\*\*(TORQ-[A-Z]+-\d+) —", srs)
    require(len(definitions) == len(set(definitions)), "Duplicate requirement IDs")
    require(len(definitions) == 73, f"Expected 73 requirements, got {len(definitions)}")
    tests = set(re.findall(r"^\| (V-[A-Z0-9]+) \|", srs, re.M))
    referenced_tests = set(re.findall(r"\bV-[A-Z0-9]+\b", srs))
    require(referenced_tests <= tests, f"Undefined acceptance IDs: {referenced_tests - tests}")
    require(len(tests) == 41, f"Expected 41 acceptance definitions, got {len(tests)}")
    for block in re.split(r"(?=\*\*TORQ-[A-Z]+-\d+ —)", srs)[1:]:
        identifier = re.search(r"TORQ-[A-Z]+-\d+", block).group()
        require(bool(re.search(r"Acceptance: V-", block)),
                f"Requirement lacks acceptance mapping: {identifier}")
    decisions = (review / "Decisions_and_Risks.md").read_text()
    decision_ids = set(re.findall(r"^## (D\d+) —", decisions, re.M))
    require(decision_ids == {f"D{i:02}" for i in range(1, 13)}, "Decision IDs incomplete")
    for block in re.split(r"(?=^## D\d+ —)", decisions, flags=re.M)[1:]:
        for field in ("Benefits", "Cons:", "Risks:", "Recommendation:", "Decision needed:", "Interim:"):
            require(field in block, f"Missing decision field {field}: {block.splitlines()[0]}")

    new_docs = [wiki / "CoChem-TORQ_Implementation_SRS.md",
                wiki / "Method_Matrix_Implementation_Contract.md",
                wiki / "RevDSD_Spectroscopy_Protocol.md",
                wiki / "README.md", wiki / "Handoff_Coding_Manifest.md",
                *review.glob("*.md")]
    checked_links = 0
    for path in new_docs:
        text = path.read_text()
        require(text.count("```") % 2 == 0, f"Unbalanced code fences: {path.name}")
        for content in re.findall(r"```json\s*\n(.*?)\n```", text, re.S):
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON example in {path.name}: {exc}")
        for target in re.findall(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)", text):
            target = target.strip("<>")
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            local = unquote(target.split("#", 1)[0])
            if local:
                require((path.parent / local).exists(),
                        f"Broken relative link: {path.name} -> {target}")
                checked_links += 1

    questions = (review / "Decision_Questions.md").read_text()
    question_ids = re.findall(r"^## ((?:D|ARCH-)\d+) —", questions, re.M)
    require(set(question_ids) == ({f"D{i:02}" for i in range(1, 13)} | {f"ARCH-{i:02}" for i in range(1, 13)}), "Question guide coverage incomplete")
    for block in re.split(r"(?=^## (?:D|ARCH-)\d+ —)", questions, flags=re.M)[1:]:
        rows = [row for row in block.splitlines() if row.startswith("| ")]
        require(len(rows) == 4, f"Question must have exactly three suggestions: {block.splitlines()[0]}")
    claims = (review / "Unverified_Claims_Register.md").read_text()
    claim_ids = re.findall(r"^\| (UC-\d+) \|", claims, re.M)
    require(len(claim_ids) == 29 and len(set(claim_ids)) == 29, "Claim register must have 29 unique entries")

    # Direct arithmetic checks for the corrected contract examples.
    lower, upper = 100 / 1.1, 100 / 0.9
    require(abs(abs(100 - lower) / lower - 0.1) < 1e-12, "Lower score inversion")
    require(abs(abs(100 - upper) / upper - 0.1) < 1e-12, "Upper score inversion")
    require(abs(500 * 15 / 60 / 16 - 7.8125) < 1e-12, "Campaign arithmetic")

    result = {
        "source_revision": revision,
        "status": "passed" if not errors else "failed",
        "source_records": len(points),
        "source_record_counts": actual_counts,
        "matrix_rows": len(row_ids),
        "requirements": len(definitions),
        "acceptance_definitions": len(tests),
        "decisions": len(decision_ids),
        "decision_questions": len(question_ids),
        "claim_register_entries": len(claim_ids),
        "relative_file_links_checked": checked_links,
        "historical_sources_byte_preserved": len(historical),
        "errors": errors,
        "scope": "Document coverage/consistency only; no scientific engine certification",
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
