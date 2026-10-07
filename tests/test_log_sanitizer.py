"""Actual legacy log-chunking API and subprocess checks.

The supplied helper is a chunker, with no redaction or sanitize_log_text/main
API. These tests do not invent those capabilities. Archived repository ORCA
text is used to check preservation; damaged byte variants only test rejection.
This helper is not qualified for publishing secrets or immutable engine data.
"""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from ci_tools import Log_Sanitizer as chunker

SCRIPT = Path(__file__).parents[1] / "ci_tools" / "Log_Sanitizer.py"
ARCHIVED_ORCA = Path(__file__).parents[1] / "test.property.txt"


def test_chunk_size_constant_is_valid_integer():
    assert type(chunker.CHUNK_SIZE_LINES) is int
    assert chunker.CHUNK_SIZE_LINES == 300


def test_sanitize_ansi_escape_sequences():
    """The absent sanitization interface must not be advertised."""
    with pytest.raises(AttributeError):
        getattr(chunker, "sanitize_log_text")


def test_sanitize_line_endings_and_carriage_returns(tmp_path):
    """Check real universal-newline reading and chunk output."""
    path = tmp_path / "newlines.log"
    path.write_bytes(b"line one\r\nline two\rline three\n")
    chunker.sanitize_and_chunk_log(path)
    assert (
        tmp_path / "newlines_chunk_1.log"
    ).read_text() == "line one\nline two\nline three"


def test_sanitize_unprintable_control_characters():
    """Control-character sanitization is unavailable in the actual helper."""
    assert not hasattr(chunker, "sanitize_log_text")


def test_sanitize_credential_and_token_masking():
    """No nonexistent redaction function is substituted for the current API."""
    with pytest.raises(AttributeError):
        getattr(chunker, "redact_credentials")


def test_sanitize_preserves_physical_chemistry_telemetry(tmp_path):
    """Round-trip genuine archived native output without editing observations."""
    raw = ARCHIVED_ORCA.read_bytes()
    path = tmp_path / "native-orca.log"
    path.write_bytes(raw)
    chunker.sanitize_and_chunk_log(path)
    archive = tmp_path / "native-orca_full.archive"
    assert sha256(archive.read_bytes()).hexdigest() == sha256(raw).hexdigest()
    chunks = sorted(
        tmp_path.glob("native-orca_chunk_*.log"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    lines = [line for chunk in chunks for line in chunk.read_text().splitlines()]
    assert lines == raw.decode().splitlines()


def test_sanitize_empty_string_input(tmp_path):
    path = tmp_path / "empty.log"
    path.write_bytes(b"")
    assert chunker.sanitize_and_chunk_log(path) is None
    assert path.exists()
    assert not list(tmp_path.glob("*_chunk_*.log"))


def test_chunking_with_non_utf8_binary_artifacts(tmp_path):
    """An explicitly damaged native copy fails without generating replacement text."""
    raw = ARCHIVED_ORCA.read_bytes() + b"\xff"
    path = tmp_path / "damaged.log"
    path.write_bytes(raw)
    with pytest.raises(SystemExit) as caught:
        chunker.sanitize_and_chunk_log(path)
    assert caught.value.code == 1
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob("*_chunk_*.log"))


def test_empty_zero_byte_log_archiving(tmp_path):
    """Characterize actual empty-input behavior without inventing archive files."""
    path = tmp_path / "zero.log"
    path.write_bytes(b"")
    chunker.sanitize_and_chunk_log(path)
    assert path.read_bytes() == b""
    assert not (tmp_path / "zero_full.archive").exists()


def test_windows_archive_collision_overwrite(tmp_path):
    """No configurable/atomic archive API is implemented by the legacy helper."""
    path = tmp_path / "input.log"
    path.write_text("non-scientific line-index example")
    with pytest.raises(TypeError):
        chunker.sanitize_and_chunk_log(path, chunk_size=1)
    assert path.exists()


def test_multi_chunk_splitting(tmp_path):
    lines = [f"line-index {i}" for i in range(2 * chunker.CHUNK_SIZE_LINES + 7)]
    path = tmp_path / "indexed.log"
    path.write_text("\n".join(lines) + "\n")
    chunker.sanitize_and_chunk_log(path)
    chunks = [tmp_path / f"indexed_chunk_{i}.log" for i in (1, 2, 3)]
    assert [len(p.read_text().splitlines()) for p in chunks] == [300, 300, 7]
    assert [line for p in chunks for line in p.read_text().splitlines()] == lines


def test_nonexistent_log_file_handling(tmp_path):
    with pytest.raises(SystemExit) as caught:
        chunker.sanitize_and_chunk_log(tmp_path / "absent.log")
    assert caught.value.code == 1


def test_cli_execution_success(tmp_path):
    path = tmp_path / "cli.log"
    path.write_bytes(ARCHIVED_ORCA.read_bytes())
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert not path.exists()
    assert (tmp_path / "cli_full.archive").read_bytes() == ARCHIVED_ORCA.read_bytes()


def test_main_cli_function_variants(tmp_path):
    """Exercise the real script entrypoint; no nonexistent main() is called."""
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing.log")],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "does not exist" in missing.stdout


def test_main_cli_default_argv_none():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True
    )
    assert completed.returncode == 1
    assert "Usage:" in completed.stdout


def test_empty_zero_byte_log_overwrites_existing_archive(tmp_path):
    """An empty current log leaves the existing archive untouched."""
    path = tmp_path / "empty-with-prior.log"
    path.write_bytes(b"")
    archive = tmp_path / "empty-with-prior_full.archive"
    archive.write_bytes(ARCHIVED_ORCA.read_bytes())
    before = sha256(archive.read_bytes()).hexdigest()
    chunker.sanitize_and_chunk_log(path)
    assert path.exists()
    assert sha256(archive.read_bytes()).hexdigest() == before
