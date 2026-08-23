import os
import re
import pathlib

lib_dir = pathlib.Path(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries")

patterns = {
    "hardcoded_paths": re.compile(r"([cC]:\\|/home/|/var/|/usr/|os\.path\.abspath\(\".*?\"\)|\b[dD]:\\|os\.path\.join\([^)]*output_dir)"),
    "output_dir": re.compile(r"output_dir"),
    "pathlib": re.compile(r"pathlib"),
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
