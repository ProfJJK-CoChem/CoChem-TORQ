"""Require actual execution of every selected release-gate test."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


def validate_report(path: Path) -> int:
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("A bounded, actual pytest JUnit report is required")
    report = ElementTree.parse(path)
    cases = list(report.iter("testcase"))
    if not cases:
        raise ValueError("No selected release-gate tests actually executed")
    rejected = [
        case.attrib.get("name", "unnamed")
        for case in cases
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))
    ]
    if rejected:
        raise ValueError(
            "Mandatory release checks did not pass: " + ", ".join(rejected)
        )
    print(f"All {len(cases)} selected release-gate tests executed without skips.")
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    validate_report(parser.parse_args().report)
