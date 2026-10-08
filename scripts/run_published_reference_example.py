"""Compare one retained published datum with an authentic TORQ native bundle.

No values, review records, engine results or scientific qualifications are
created. All inputs are explicit files and recorded digests. A crawler's actual
automated review may be used for this descriptive operation; a preregistered
benchmark continues to require its separate external curation and acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cochem_torq.reference_comparison import execute_reference_compare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--derived-texts", type=Path)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--expected-reference-manifest-sha256", required=True)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--publication-bundle", required=True, type=Path)
    parser.add_argument("--expected-publication-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = execute_reference_compare(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
