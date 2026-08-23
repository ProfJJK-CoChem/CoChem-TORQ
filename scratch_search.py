import os
import re
import pathlib

lib_dir = pathlib.Path(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries")

patterns = {
    "print_statements": re.compile(r"print\("),
    "subprocess_run": re.compile(r"subprocess\.run"),
    "mock_words": re.compile(r"(?i)(mock|fake|dummy|stub|placeholder)"),
    "hardcoded_paths": re.compile(r"(C:\\|D:\\|/home/|/usr/)"),
    "calc_hess": re.compile(r"(?i)Calc_Hess"),
    "defgrid": re.compile(r"(?i)defgrid[1-5]"),
    "tolmaxg": re.compile(r"(?i)TolMaxG"),
    "pass_stmt": re.compile(r"^\s*pass\s*$"),
}

results = {k: [] for k in patterns}

for p in lib_dir.rglob("*.py"):
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            for key, pat in patterns.items():
                if pat.search(line):
                    results[key].append(f"{p.name}:{i+1} {line.strip()}")

for k, v in results.items():
    print(f"\n--- {k} ({len(v)} matches) ---")
    for match in v[:10]: # Print first 10
        print(match)
    if len(v) > 10:
        print(f"... and {len(v) - 10} more.")
