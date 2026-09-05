"""# zero-stub anti-spoof verification
Unit and Integration Tests for NFTP Phase 3: Cleanup Executor (ci_tools/cleanup_executor.py).

Physically validates:
- Manifest loading from directory, single file, and empty directory error handling.
- Action planning across multiple manifests with distinct target_dir roots and absolute/relative paths.
- Directory purge aggregation, threshold qualification, and single file pruning.
- Dry run execution without modifying disk state.
- Physical execution of file and directory purging.
- Physical execution of file trashing with full relative subpath preservation.
- Accurate recording of physical destination path in _undo_manifest.json.
- Exclusion of failed or missing file moves from _undo_manifest.json.
- Human-readable byte formatting and recursive directory sizing.
- Summary printing for dry-run and executed modes.
- CLI argument parsing, mutually exclusive flags, default modes, execute flow, and error exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ci_tools.cleanup_executor import (
    _dir_size,
    _find_purgeable_dirs,
    _fmt_size,
    execute_plan,
    load_manifests,
    plan_actions,
    print_results,
)


def test_fmt_size() -> None:
    """Validate human-readable formatting across byte scales."""
    assert _fmt_size(500) == "500.0 B"
    assert _fmt_size(1024) == "1.0 KB"
    assert _fmt_size(1024 * 1024) == "1.0 MB"
    assert _fmt_size(1024 * 1024 * 1024) == "1.0 GB"
    assert _fmt_size(1024 * 1024 * 1024 * 1024 * 2) == "2.0 TB"


def test_dir_size(tmp_path: Path) -> None:
    """Validate calculation of recursive directory size."""
    test_dir = tmp_path / "size_test"
    test_dir.mkdir()
    f1 = test_dir / "f1.txt"
    f1.write_bytes(b"A" * 100)
    sub = test_dir / "sub"
    sub.mkdir()
    f2 = sub / "f2.txt"
    f2.write_bytes(b"B" * 200)

    assert _dir_size(test_dir) == 300


def test_load_manifests_from_directory_and_single_file(tmp_path: Path) -> None:
    """Validate loading manifests from both a directory and a direct file path."""
    m_dir = tmp_path / "manifests"
    m_dir.mkdir()

    m1_path = m_dir / "module_a_triage_manifest.json"
    m1_data = {
        "module": "module_a",
        "target_dir": str(tmp_path / "repo_a"),
        "files": {"PURGE": ["cache/a.pyc"], "TRASH": ["temp/a.bak"]},
        "counts": {"KEEP": 5, "TRIAGE": 2, "EXCLUDE": 1},
    }
    m1_path.write_text(json.dumps(m1_data), encoding="utf-8")

    m2_path = m_dir / "module_b_triage_manifest.json"
    m2_data = {
        "module": "module_b",
        "target_dir": str(tmp_path / "repo_b"),
        "files": {"PURGE": ["cache/b.pyc"], "TRASH": ["temp/b.bak"]},
        "counts": {"KEEP": 3, "TRIAGE": 1, "EXCLUDE": 0},
    }
    m2_path.write_text(json.dumps(m2_data), encoding="utf-8")

    # Directory load
    loaded_dir = load_manifests(m_dir)
    assert len(loaded_dir) == 2
    assert {m["module"] for m in loaded_dir} == {"module_a", "module_b"}

    # Single file load
    loaded_file = load_manifests(m1_path)
    assert len(loaded_file) == 1
    assert loaded_file[0]["module"] == "module_a"


def test_load_manifests_empty_directory(tmp_path: Path) -> None:
    """Validate loading from an empty directory returns an empty list."""
    empty_dir = tmp_path / "empty_manifests"
    empty_dir.mkdir()
    assert load_manifests(empty_dir) == []


def test_find_purgeable_dirs(tmp_path: Path) -> None:
    """Validate detection and deduplication of purgeable directories."""
    base = tmp_path / "purge_search"
    d1 = base / "qualifying_dir"
    d1.mkdir(parents=True)
    d2 = base / "small_dir"
    d2.mkdir(parents=True)

    files_d1 = []
    for i in range(5):
        f = d1 / f"temp_{i}.tmp"
        f.write_text("temp", encoding="utf-8")
        files_d1.append(str(f))

    files_d2 = []
    for i in range(2):
        f = d2 / f"temp_{i}.tmp"
        f.write_text("temp", encoding="utf-8")
        files_d2.append(str(f))

    all_purge_files = files_d1 + files_d2
    purgeable = _find_purgeable_dirs(all_purge_files, min_files=5)

    assert str(d1) in purgeable
    assert str(d2) not in purgeable


def test_plan_actions_multi_target_dirs_and_purgeable_dirs(tmp_path: Path) -> None:
    """Validate plan building across distinct target roots with directory purge optimization."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    # Create 6 purge files in repo_a/cache so it qualifies for _find_purgeable_dirs
    cache_dir = repo_a / "cache"
    cache_dir.mkdir()
    purge_a_files = []
    for i in range(6):
        f = cache_dir / f"file_{i}.pyc"
        f.write_text("cache", encoding="utf-8")
        purge_a_files.append(f"cache/file_{i}.pyc")

    # Create trash files in repo_a and repo_b
    trash_a = repo_a / "sub_a" / "temp.bak"
    trash_a.parent.mkdir(parents=True)
    trash_a.write_text("trash_a", encoding="utf-8")

    trash_b = repo_b / "sub_b" / "deep" / "temp.bak"
    trash_b.parent.mkdir(parents=True)
    trash_b.write_text("trash_b", encoding="utf-8")

    manifests = [
        {
            "module": "repo_a",
            "target_dir": str(repo_a),
            "files": {"PURGE": purge_a_files, "TRASH": ["sub_a/temp.bak"]},
            "counts": {"KEEP": 10, "TRIAGE": 2, "EXCLUDE": 1},
        },
        {
            "module": "repo_b",
            "target_dir": str(repo_b),
            "files": {"PURGE": [], "TRASH": ["sub_b/deep/temp.bak"]},
            "counts": {"KEEP": 4, "TRIAGE": 0, "EXCLUDE": 0},
        },
    ]

    plan = plan_actions(manifests)

    assert plan["keep_count"] == 14
    assert plan["triage_count"] == 2
    assert plan["exclude_count"] == 1

    # Directory optimization should have caught cache_dir
    assert str(cache_dir) in plan["purge_dirs"]
    # And removed individual files inside cache_dir from purge_files
    assert len(plan["purge_files"]) == 0

    # Trash records must contain both distinct targets
    assert len(plan["trash_files"]) == 2
    records = plan["trash_files"]
    assert any(r["src"] == str(trash_a) and r["target_dir"] == str(repo_a) and r["rel_path"] == "sub_a/temp.bak" for r in records)
    assert any(r["src"] == str(trash_b) and r["target_dir"] == str(repo_b) and r["rel_path"] == "sub_b/deep/temp.bak" for r in records)


def test_plan_actions_with_absolute_paths(tmp_path: Path) -> None:
    """Validate plan building with absolute paths in manifest."""
    repo = tmp_path / "repo_abs"
    repo.mkdir()
    purge_file = repo / "purge.log"
    purge_file.write_text("log", encoding="utf-8")
    trash_file = repo / "trash.tmp"
    trash_file.write_text("tmp", encoding="utf-8")

    manifests = [
        {
            "module": "abs_module",
            "target_dir": str(repo),
            "files": {
                "PURGE": [str(purge_file)],
                "TRASH": [str(trash_file)],
            },
            "counts": {"KEEP": 2, "TRIAGE": 0, "EXCLUDE": 0},
        }
    ]

    plan = plan_actions(manifests)
    assert str(purge_file) in plan["purge_files"]
    assert any(r["src"] == str(trash_file) for r in plan["trash_files"])


def test_execute_plan_dry_run_preserves_disk_state(tmp_path: Path) -> None:
    """Validate that dry_run=True performs zero disk mutations."""
    repo = tmp_path / "repo"
    repo.mkdir()
    trash_root = tmp_path / "trash"

    f_purge = repo / "purge.tmp"
    f_purge.write_text("purge", encoding="utf-8")
    f_trash = repo / "sub" / "trash.bak"
    f_trash.parent.mkdir(parents=True)
    f_trash.write_text("trash", encoding="utf-8")

    plan = {
        "purge_files": [str(f_purge)],
        "purge_dirs": [],
        "trash_files": [{"src": str(f_trash), "target_dir": str(repo), "rel_path": "sub/trash.bak"}],
        "keep_count": 1,
        "triage_count": 0,
    }

    results = execute_plan(plan, trash_root, dry_run=True)

    assert results["dry_run"] is True
    assert results["purged_files"] == 1
    assert results["trashed_files"] == 1
    assert len(results["errors"]) == 0

    # Files must still exist on disk
    assert f_purge.exists()
    assert f_trash.exists()
    assert not trash_root.exists()


def test_execute_plan_physical_execution_and_undo_manifest(tmp_path: Path) -> None:
    """Validate physical file purging, trashing, relative path hierarchy, and exact undo manifest."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    trash_root = tmp_path / "trash_root"

    # Purge file and dir
    p_dir = repo_a / "cache"
    p_dir.mkdir()
    (p_dir / "c1.pyc").write_text("cache", encoding="utf-8")
    p_file = repo_a / "stale.pyc"
    p_file.write_text("stale", encoding="utf-8")

    # Trash files in subdirectories (including same filename across different subdirs)
    t1 = repo_a / "sub_a" / "config.bak"
    t1.parent.mkdir(parents=True)
    t1.write_text("t1_content", encoding="utf-8")

    t2 = repo_a / "sub_b" / "config.bak"
    t2.parent.mkdir(parents=True)
    t2.write_text("t2_content", encoding="utf-8")

    t3 = repo_b / "deep" / "nested" / "archive.bak"
    t3.parent.mkdir(parents=True)
    t3.write_text("t3_content", encoding="utf-8")

    plan = {
        "purge_dirs": [str(p_dir)],
        "purge_files": [str(p_file)],
        "trash_files": [
            {"src": str(t1), "target_dir": str(repo_a), "rel_path": "sub_a/config.bak"},
            {"src": str(t2), "target_dir": str(repo_a), "rel_path": "sub_b/config.bak"},
            {"src": str(t3), "target_dir": str(repo_b), "rel_path": "deep/nested/archive.bak"},
        ],
        "keep_count": 5,
        "triage_count": 0,
    }

    results = execute_plan(plan, trash_root, dry_run=False)

    assert results["dry_run"] is False
    assert results["purged_dirs"] == 1
    assert results["purged_files"] == 1
    assert results["trashed_files"] == 3
    assert len(results["errors"]) == 0

    # Verify purges
    assert not p_dir.exists()
    assert not p_file.exists()

    # Verify source trash files removed
    assert not t1.exists()
    assert not t2.exists()
    assert not t3.exists()

    # Find created timestamped trash folder
    trash_dirs = [d for d in trash_root.iterdir() if d.is_dir()]
    assert len(trash_dirs) == 1
    trash_dest = trash_dirs[0]

    # Verify physical file placement preserves relative directory structure
    dest_t1 = trash_dest / "sub_a" / "config.bak"
    dest_t2 = trash_dest / "sub_b" / "config.bak"
    dest_t3 = trash_dest / "deep" / "nested" / "archive.bak"

    assert dest_t1.is_file()
    assert dest_t1.read_text(encoding="utf-8") == "t1_content"
    assert dest_t2.is_file()
    assert dest_t2.read_text(encoding="utf-8") == "t2_content"
    assert dest_t3.is_file()
    assert dest_t3.read_text(encoding="utf-8") == "t3_content"

    # Verify _undo_manifest.json
    undo_file = trash_dest / "_undo_manifest.json"
    assert undo_file.is_file()
    undo_data = json.loads(undo_file.read_text(encoding="utf-8"))

    assert undo_data["trash_dest"] == str(trash_dest)
    manifest_files = undo_data["files"]
    assert len(manifest_files) == 3

    # Exact destination matching (no flattening to filename)
    file_map = {m["original"]: m["trashed_to"] for m in manifest_files}
    assert file_map[str(t1)] == str(dest_t1)
    assert file_map[str(t2)] == str(dest_t2)
    assert file_map[str(t3)] == str(dest_t3)


def test_execute_plan_raw_string_trash_items(tmp_path: Path) -> None:
    """Validate handling raw string items in trash_files list."""
    repo = tmp_path / "repo_raw"
    repo.mkdir()
    trash_root = tmp_path / "trash_raw"

    f = repo / "raw_item.bak"
    f.write_text("raw item content", encoding="utf-8")

    plan = {
        "purge_dirs": [],
        "purge_files": [],
        "trash_files": [str(f)],
        "target_dir": str(repo),
        "keep_count": 0,
        "triage_count": 0,
    }

    results = execute_plan(plan, trash_root, dry_run=False)
    assert results["trashed_files"] == 1
    assert not f.exists()

    trash_dirs = [d for d in trash_root.iterdir() if d.is_dir()]
    assert len(trash_dirs) == 1
    trashed_dest = trash_dirs[0] / "raw_item.bak"
    assert trashed_dest.is_file()
    assert trashed_dest.read_text(encoding="utf-8") == "raw item content"


def test_execute_plan_failed_and_missing_moves_excluded_from_undo_manifest(tmp_path: Path) -> None:
    """Validate that missing or failed file moves are never recorded into _undo_manifest.json."""
    repo = tmp_path / "repo"
    repo.mkdir()
    trash_root = tmp_path / "trash"

    real_file = repo / "existing.bak"
    real_file.write_text("real content", encoding="utf-8")
    missing_file = repo / "ghost.bak"

    plan = {
        "purge_dirs": [],
        "purge_files": [],
        "trash_files": [
            {"src": str(real_file), "target_dir": str(repo), "rel_path": "existing.bak"},
            {"src": str(missing_file), "target_dir": str(repo), "rel_path": "ghost.bak"},
        ],
        "keep_count": 0,
        "triage_count": 0,
    }

    results = execute_plan(plan, trash_root, dry_run=False)

    assert results["trashed_files"] == 1

    trash_dirs = [d for d in trash_root.iterdir() if d.is_dir()]
    assert len(trash_dirs) == 1
    undo_data = json.loads((trash_dirs[0] / "_undo_manifest.json").read_text(encoding="utf-8"))

    # Only existing.bak should be present
    assert len(undo_data["files"]) == 1
    assert undo_data["files"][0]["original"] == str(real_file)
    assert str(missing_file) not in [f["original"] for f in undo_data["files"]]


def test_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    """Validate output summary printing for dry-run and executed results."""
    results_dry = {
        "dry_run": True,
        "purged_dirs": 1,
        "purged_files": 2,
        "trashed_files": 3,
        "bytes_recovered": 1024,
        "errors": ["Sample non-fatal error"],
    }
    plan = {"keep_count": 10, "triage_count": 4}

    print_results(results_dry, plan)
    out = capsys.readouterr().out
    assert "NFTP Phase 3: Cleanup Results (DRY-RUN)" in out
    assert "Directories purged:  1" in out
    assert "Files purged:        2" in out
    assert "Files trashed:       3" in out
    assert "Space recovered:     1.0 KB" in out
    assert "Sample non-fatal error" in out

    results_exec = {
        "dry_run": False,
        "purged_dirs": 0,
        "purged_files": 0,
        "trashed_files": 1,
        "bytes_recovered": 512,
        "errors": [],
    }
    print_results(results_exec, plan)
    out_exec = capsys.readouterr().out
    assert "NFTP Phase 3: Cleanup Results (EXECUTED)" in out_exec


def test_cli_mutually_exclusive_dry_run_and_execute(tmp_path: Path) -> None:
    """Validate that passing both --dry-run and --execute fails with mutually exclusive error."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "ci_tools.cleanup_executor",
        "--manifests",
        str(manifest_dir),
        "--dry-run",
        "--execute",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode != 0
    assert "not allowed with argument" in res.stderr or "mutually exclusive" in res.stderr.lower()


def test_cli_defaults_to_dry_run(tmp_path: Path) -> None:
    """Validate that omitting --dry-run and --execute defaults to DRY-RUN mode."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    m_file = manifest_dir / "sample_triage_manifest.json"
    m_file.write_text(
        json.dumps({
            "module": "sample",
            "target_dir": str(tmp_path),
            "files": {"PURGE": [], "TRASH": []},
            "counts": {"KEEP": 1},
        }),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "ci_tools.cleanup_executor",
        "--manifests",
        str(manifest_dir),
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0
    assert "Mode: DRY-RUN" in res.stdout


def test_cli_execute_flow(tmp_path: Path) -> None:
    """Validate CLI physical execution with --execute flag."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    trash_dir = tmp_path / "trash"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    target_file = work_dir / "test_artifact.tmp"
    target_file.write_text("hello trash", encoding="utf-8")

    m_file = manifest_dir / "test_triage_manifest.json"
    m_file.write_text(
        json.dumps({
            "module": "test",
            "target_dir": str(work_dir),
            "files": {"PURGE": [], "TRASH": ["test_artifact.tmp"]},
            "counts": {"KEEP": 0},
        }),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "ci_tools.cleanup_executor",
        "--manifests",
        str(manifest_dir),
        "--trash",
        str(trash_dir),
        "--execute",
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert res.returncode == 0
    assert "Mode: EXECUTE" in res.stdout
    assert not target_file.exists()


def test_cli_nonexistent_manifest_path(tmp_path: Path) -> None:
    """Validate CLI error exit when manifest path does not exist."""
    missing = tmp_path / "nonexistent"
    cmd = [
        sys.executable,
        "-m",
        "ci_tools.cleanup_executor",
        "--manifests",
        str(missing),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
    assert "ERROR: Manifest path not found" in res.stdout


def test_cli_empty_manifest_dir(tmp_path: Path) -> None:
    """Validate CLI error exit when manifest directory contains no manifests."""
    empty_manifests = tmp_path / "empty_dir"
    empty_manifests.mkdir()
    cmd = [
        sys.executable,
        "-m",
        "ci_tools.cleanup_executor",
        "--manifests",
        str(empty_manifests),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
    assert "ERROR: No manifests found" in res.stdout
