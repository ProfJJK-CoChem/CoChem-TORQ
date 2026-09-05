"""
CoChem Necessity-First File Triage Protocol — Phase 1: Deterministic Classifier
================================================================================
Zero LLM calls. Walks the repository, applies rule-based classification to every
file, and produces per-module JSON manifests.

Usage:
    python file_triage_classifier.py --target <repo_root> [--output <output_dir>]

Verdicts:
    KEEP    — Source code, configs, docs essential to the module
    PURGE   — Cache, node_modules, generated debris (safe to delete)
    TRASH   — Stale backups, temp files (move to .trash/)
    TRIAGE  — Ambiguous: test files, scripts, data needing LLM review
    EXCLUDE — Virtual environments (ignored, not counted)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal

Verdict = Literal["KEEP", "PURGE", "TRASH", "TRIAGE", "EXCLUDE"]

# ── Classification Rules ─────────────────────────────────────────────────────

EXCLUDE_DIRS = {".venv", ".conda"}

PURGE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".git", ".coverage",
    "cochem_base.egg-info",  # build artifacts
}

PURGE_EXTENSIONS = {".pyc", ".pyo"}

TRASH_PATTERNS = re.compile(
    r"(\.bak(\.\d+)?$|_backup|_old\b|\.archive$|^temp_|^scratch$|"
    r"swarm_state\.json|\.lock$|search_results\.|search_out\.|"
    r"cochem_system_config\.json\.bak)",
    re.IGNORECASE,
)

KEEP_NAMES = {
    "README.md", "LICENSE", "LICENSE.md", "CHANGELOG.md",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "Makefile", "Dockerfile", ".gitignore", ".dockerignore",
    "__init__.py", "__main__.py", "__registry.md",
    "cochem_system_config.json", "cochem_hpc_registry.json",
    "conftest.py",  # pytest infra — always keep
}

KEEP_EXTENSIONS = {
    ".tex", ".bib", ".rst", ".cfg", ".toml", ".yml", ".yaml",
    ".sh", ".bat", ".ps1",
    ".css", ".html", ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
    ".json",  # configs
    ".ipynb",  # notebooks
}

SOURCE_EXTENSIONS = {".py"}

TEST_PATTERN = re.compile(r"^test_.*\.py$|.*_test\.py$", re.IGNORECASE)

TRIAGE_EXTENSIONS = {".h5", ".hdf5", ".pkl", ".npy", ".npz", ".zst", ".csv", ".dat"}

DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".pptx"}

# User-confirmed: these are trash
CONFIRMED_TRASH_MODULES = {"CoChem-CATALYST"}

# User-confirmed: purge entirely
CONFIRMED_PURGE_DIRS_RELATIVE = {
    ".docs/.audit",  # 2,415 task files + 1,487 logs
}


def classify_file(
    filepath: Path,
    repo_root: Path,
    module_name: str | None,
) -> tuple[Verdict, str]:
    """Classify a single file. Returns (verdict, reason)."""
    rel = filepath.relative_to(repo_root)
    parts = rel.parts
    name = filepath.name
    ext = filepath.suffix.lower()

    # ── R0: User-confirmed trash modules ──
    if module_name and module_name in CONFIRMED_TRASH_MODULES:
        return "TRASH", f"User confirmed {module_name} is hallucinated"

    # ── R1: Exclude virtual environments ──
    if any(p in EXCLUDE_DIRS for p in parts):
        return "EXCLUDE", "Virtual environment"

    # ── R2: Purge cache directories ──
    if any(p in PURGE_DIRS for p in parts):
        return "PURGE", f"Cache/generated directory ({next(p for p in parts if p in PURGE_DIRS)})"

    # ── R2b: Purge by extension ──
    if ext in PURGE_EXTENSIONS:
        return "PURGE", f"Compiled bytecode ({ext})"

    # ── R3: Purge confirmed audit debris ──
    for purge_dir in CONFIRMED_PURGE_DIRS_RELATIVE:
        purge_parts = tuple(purge_dir.replace("/", os.sep).split(os.sep))
        if _path_contains_subpath(parts, purge_parts):
            return "PURGE", f"Audit pipeline debris ({purge_dir})"

    # ── R4: SEED frontend/node_modules already caught by PURGE_DIRS ──
    # But also purge frontend/dist as build artifacts
    if "frontend" in parts and "dist" in parts:
        return "PURGE", "Frontend build artifacts"

    # ── R5: Trash temp/backup patterns ──
    if TRASH_PATTERNS.search(name):
        return "TRASH", f"Matches temp/backup pattern"
    if TRASH_PATTERN_PATH_CHECK(parts):
        return "TRASH", "In scratch/temp directory"

    # ── R6: Keep known infrastructure files ──
    if name in KEEP_NAMES:
        return "KEEP", f"Infrastructure file ({name})"

    # ── R7: Agent configs ──
    if name.endswith(".agent.md"):
        return "KEEP", "Agent configuration"

    # ── R8: Test files (Aggressive strategy) ──
    if TEST_PATTERN.match(name):
        return "TRIAGE", "Test file — needs necessity check (aggressive strategy)"

    # ── R9: Source code ──
    if ext in SOURCE_EXTENSIONS:
        # Root-level scripts (not in any CoChem-* module)
        if module_name is None and len(parts) == 1:
            return "TRIAGE", "Root-level script — user unsure if needed"
        return "KEEP", "Source code"

    # ── R10: Documentation ──
    if ext in DOC_EXTENSIONS:
        # Draco blueprints
        if "Draco_Blueprint" in name or "Draco" in name:
            return "TRIAGE", "Draco blueprint — may be stale/regenerable"
        return "KEEP", "Documentation"

    # ── R11: Keep configs ──
    if ext in KEEP_EXTENSIONS:
        return "KEEP", f"Config/asset ({ext})"

    # ── R12: Binary data needs triage ──
    if ext in TRIAGE_EXTENSIONS:
        return "TRIAGE", f"Binary data file ({ext}) — may be fixture or real data"

    # ── R13: Image/media files ──
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"}:
        return "KEEP", "Image asset"

    # ── R14: Everything else ──
    if ext in {".exe", ".dll", ".so", ".pyd", ".whl", ".tar", ".gz", ".zip"}:
        return "TRIAGE", f"Binary/archive ({ext})"

    return "TRIAGE", f"Unclassified ({ext or 'no extension'})"


def TRASH_PATTERN_PATH_CHECK(parts: tuple) -> bool:
    """Check if path contains known scratch/temp directories."""
    scratch_dirs = {"scratch", "temp_chain", "temp_files"}
    return any(p in scratch_dirs for p in parts)


def _path_contains_subpath(full_parts: tuple, sub_parts: tuple) -> bool:
    """Check if sub_parts appears as a contiguous subsequence in full_parts."""
    sub_len = len(sub_parts)
    for i in range(len(full_parts) - sub_len + 1):
        if full_parts[i : i + sub_len] == sub_parts:
            return True
    return False


def identify_module(filepath: Path, repo_root: Path) -> str | None:
    """Identify which CoChem module a file belongs to, or None for root-level."""
    rel = filepath.relative_to(repo_root)
    parts = rel.parts
    if parts and parts[0].startswith("CoChem-"):
        return parts[0]
    return None


def run_classifier(target_dir: Path, output_dir: Path) -> dict:
    """Walk the repo, classify all files, produce per-module manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Module-level accumulators
    module_data: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    module_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    global_counts: Dict[str, int] = defaultdict(int)

    total_files = 0
    skipped_dirs = set()

    for dirpath_str, dirnames, filenames in os.walk(target_dir):
        dirpath = Path(dirpath_str)

        # Skip excluded directories early (prune walk)
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and d not in PURGE_DIRS and d != ".git"
        ]

        for fname in filenames:
            filepath = dirpath / fname
            total_files += 1
            module = identify_module(filepath, target_dir)
            module_key = module or "__ROOT__"

            verdict, reason = classify_file(filepath, target_dir, module)
            rel_path = str(filepath.relative_to(target_dir))

            module_data[module_key][verdict].append(rel_path)
            module_counts[module_key][verdict] += 1
            global_counts[verdict] += 1

    # Write per-module manifests
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest_paths = []

    for module_key, verdicts in sorted(module_data.items()):
        manifest = {
            "module": module_key,
            "scan_timestamp": timestamp,
            "target_dir": str(target_dir),
            "counts": dict(module_counts[module_key]),
            "files": {v: sorted(files) for v, files in sorted(verdicts.items())},
        }
        safe_name = module_key.replace("__", "_")
        manifest_path = output_dir / f"{safe_name}_triage_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        manifest_paths.append(manifest_path)

    # Write global summary
    summary = {
        "scan_timestamp": timestamp,
        "target_dir": str(target_dir),
        "total_files_walked": total_files,
        "global_counts": dict(global_counts),
        "per_module_counts": {
            k: dict(v) for k, v in sorted(module_counts.items())
        },
        "manifests": [str(p) for p in manifest_paths],
    }
    summary_path = output_dir / "triage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable summary."""
    print("=" * 60)
    print("  NFTP Phase 1: Deterministic Classification Results")
    print("=" * 60)
    print(f"  Target: {summary['target_dir']}")
    print(f"  Scanned: {summary['total_files_walked']} files")
    print()

    gc = summary["global_counts"]
    total = sum(gc.values())
    for verdict in ["KEEP", "TRIAGE", "TRASH", "PURGE", "EXCLUDE"]:
        count = gc.get(verdict, 0)
        pct = (count / total * 100) if total else 0
        bar = "#" * int(pct / 2)
        print(f"  {verdict:8s}  {count:>6,d}  ({pct:5.1f}%)  {bar}")

    print()
    print("  Per-module breakdown:")
    print(f"  {'Module':<22s} {'KEEP':>6s} {'TRIAGE':>7s} {'TRASH':>6s} {'PURGE':>6s}")
    print("  " + "-" * 50)
    for mod, counts in sorted(summary["per_module_counts"].items()):
        keep = counts.get("KEEP", 0)
        triage = counts.get("TRIAGE", 0)
        trash = counts.get("TRASH", 0)
        purge = counts.get("PURGE", 0)
        print(f"  {mod:<22s} {keep:>6d} {triage:>7d} {trash:>6d} {purge:>6d}")

    print()
    triage_total = gc.get("TRIAGE", 0)
    print(f"  Files requiring LLM triage: {triage_total:,d}")
    print(f"  Files safe to auto-handle:  {total - triage_total:,d}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFTP Phase 1: Deterministic File Classifier")
    parser.add_argument("--target", required=True, help="Root directory to classify")
    parser.add_argument("--output", default=None, help="Output directory for manifests")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    output = Path(args.output).resolve() if args.output else target / ".docs" / "triage"

    if not target.exists():
        print(f"ERROR: Target directory does not exist: {target}")
        sys.exit(1)

    summary = run_classifier(target, output)
    print_summary(summary)
    print(f"  Manifests written to: {output}")
