# zero-stub anti-spoofing engine
"""Zero-Mock Unit and Integration Test Suite for ci_tools/Log_Sanitizer.py.

Module: tests/test_log_sanitizer.py
Target Implementation: ci_tools.Log_Sanitizer

Complies with:
- Method Matrix Standards & Zero-Mock Execution Invariants
- CoChem Anti-Spoofing Protocol v3 (Hardened)
- Telemetry/Log Processing Mandates

Validates:
1. Default configuration constants (CHUNK_SIZE_LINES).
2. Stripping of ANSI color codes and terminal control sequences.
3. Normalization of line endings and carriage return characters.
4. Scrubbing of unprintable control characters and null bytes while preserving tabs and newlines.
5. Masking of sensitive credentials, API keys, Bearer tokens, and private keys.
6. Preservation of legitimate physical chemistry calculation telemetry (energies, coordinates, frequencies).
7. Safe processing of non-UTF-8 / binary corrupted log files without UnicodeDecodeError.
8. Archiving of 0-byte / empty logs without hanging or infinite loop reprocessing.
9. Atomic archive file replacement on Windows without FileExistsError collision.
10. Accurate line chunking and file naming logic across configurable chunk sizes.
11. Handling of non-existent log paths gracefully.
12. CLI execution via subprocess with proper exit codes and argument handling.
13. Entrypoint main() behavior across diverse parameter inputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ci_tools.Log_Sanitizer import (
    CHUNK_SIZE_LINES,
    main,
    sanitize_and_chunk_log,
    sanitize_log_text,
)


def test_chunk_size_constant_is_valid_integer() -> None:
    """Verify CHUNK_SIZE_LINES constant is configured as a positive integer."""
    assert isinstance(CHUNK_SIZE_LINES, int)
    assert CHUNK_SIZE_LINES > 0
    assert CHUNK_SIZE_LINES == 300


def test_sanitize_ansi_escape_sequences() -> None:
    """Verify ANSI escape sequences are completely stripped from log text."""
    raw_text = "\x1b[31;1mERROR:\x1b[0m Calculation diverged at step 42.\x1b[2K\r"
    cleaned = sanitize_log_text(raw_text)
    assert "\x1b" not in cleaned
    assert "ERROR: Calculation diverged at step 42." in cleaned


def test_sanitize_line_endings_and_carriage_returns() -> None:
    """Verify CRLF and standalone carriage returns are normalized to LF."""
    raw_text = "Line 1\r\nLine 2\rProgress: 50%\rLine 3\n"
    cleaned = sanitize_log_text(raw_text)
    assert "\r" not in cleaned
    assert "Line 1\nLine 2\nProgress: 50%\nLine 3\n" == cleaned


def test_sanitize_unprintable_control_characters() -> None:
    """Verify null bytes and corrupt binary control characters are scrubbed while preserving whitespace."""
    raw_text = "Standard log line\x00 with corrupt \x08\x0b\x0c bytes\x1f included.\tTab preserved.\n"
    cleaned = sanitize_log_text(raw_text)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "\x0b" not in cleaned
    assert "\x0c" not in cleaned
    assert "\x1f" not in cleaned
    assert "\tTab preserved.\n" in cleaned
    assert "Standard log line with corrupt  bytes included.\tTab preserved.\n" == cleaned


def test_sanitize_credential_and_token_masking() -> None:
    """Verify API tokens, secrets, Bearer headers, and private keys are masked."""
    raw_text = (
        "Config dump:\n"
        "api_key = AIzaSyD9876543210abcdef1234567890abcdef\n"
        "Authorization: Bearer secret_access_token_123456789\n"
        "ghp_token = ghp_0123456789abcdefghijklmnopqrstuvwxyz01\n"
        "openai_key = sk-proj-1234567890abcdefghijklmnopqrstuvwxyz01\n"
        "aws_key: AKIA1234567890ABCDEF\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Y123...\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    cleaned = sanitize_log_text(raw_text)
    assert "AIzaSyD9876543210abcdef1234567890abcdef" not in cleaned
    assert "secret_access_token_123456789" not in cleaned
    assert "ghp_0123456789abcdefghijklmnopqrstuvwxyz01" not in cleaned
    assert "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz01" not in cleaned
    assert "AKIA1234567890ABCDEF" not in cleaned
    assert "MIIEowIBAAKCAQEA0Y123..." not in cleaned
    assert "[REDACTED]" in cleaned or "[REDACTED_TOKEN]" in cleaned or "[REDACTED_PRIVATE_KEY]" in cleaned


def test_sanitize_preserves_physical_chemistry_telemetry() -> None:
    """Verify scientific calculations, floating points, and molecular coordinates remain intact."""
    raw_chem_log = (
        "--------------------------------------------------------------------\n"
        "          FINAL SINGLE POINT ENERGY      -154.234567890123 Eh\n"
        "          Nuclear Repulsion Energy       34.56789012 Eh\n"
        "          RMS Gradient                   0.00001234 Eh/Bohr\n"
        "          CARTESIAN COORDINATES (ANGSTROEM)\n"
        "          C     0.000000    0.000000    0.000000\n"
        "          O     1.162000    0.000000    0.000000\n"
        "--------------------------------------------------------------------\n"
    )
    cleaned = sanitize_log_text(raw_chem_log)
    assert "-154.234567890123 Eh" in cleaned
    assert "34.56789012 Eh" in cleaned
    assert "0.00001234 Eh/Bohr" in cleaned
    assert "C     0.000000    0.000000    0.000000" in cleaned
    assert "O     1.162000    0.000000    0.000000" in cleaned


def test_sanitize_empty_string_input() -> None:
    """Verify empty input string produces empty output."""
    assert sanitize_log_text("") == ""


def test_chunking_with_non_utf8_binary_artifacts(tmp_path: Path) -> None:
    """Verify log files containing non-UTF-8 bytes do not trigger UnicodeDecodeError."""
    log_file = tmp_path / "quantum_orca.log"
    binary_content = b"FINAL SINGLE POINT ENERGY\n\xe9\xff\xfe ORCA Electronic Energy: -154.2345678\n"
    log_file.write_bytes(binary_content)

    chunks = sanitize_and_chunk_log(log_file, chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0].exists()

    chunk_text = chunks[0].read_text(encoding="utf-8")
    assert "FINAL SINGLE POINT ENERGY" in chunk_text
    assert "ORCA Electronic Energy: -154.2345678" in chunk_text

    archive_file = tmp_path / "quantum_orca_full.archive"
    assert archive_file.exists()
    assert not log_file.exists()


def test_empty_zero_byte_log_archiving(tmp_path: Path) -> None:
    """Verify 0-byte log files are archived without hanging or infinite loop reprocessing."""
    log_file = tmp_path / "empty_run.log"
    log_file.write_text("", encoding="utf-8")

    chunks = sanitize_and_chunk_log(log_file, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].name == "empty_run_chunk_1.log"
    assert chunks[0].exists()
    assert chunks[0].read_text(encoding="utf-8") == ""

    archive_file = tmp_path / "empty_run_full.archive"
    assert archive_file.exists()
    assert not log_file.exists()


def test_windows_archive_collision_overwrite(tmp_path: Path) -> None:
    """Verify existing .archive files are overwritten cleanly on Windows without FileExistsError."""
    log_file = tmp_path / "pipeline.log"
    log_file.write_text("New run log line 1\nNew run log line 2\n", encoding="utf-8")

    archive_file = tmp_path / "pipeline_full.archive"
    archive_file.write_text("Old prior run archive content", encoding="utf-8")

    chunks = sanitize_and_chunk_log(log_file, chunk_size=10)
    assert len(chunks) == 1
    assert archive_file.exists()
    assert not log_file.exists()

    archive_content = archive_file.read_text(encoding="utf-8")
    assert "New run log line 1" in archive_content


def test_multi_chunk_splitting(tmp_path: Path) -> None:
    """Verify chunking splits log content correctly into bounded chunks."""
    log_file = tmp_path / "large_run.log"
    total_lines = 25
    lines = [f"Step {i:03d}: Electronic energy calculation converged" for i in range(total_lines)]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chunk_size = 10
    chunks = sanitize_and_chunk_log(log_file, chunk_size=chunk_size)
    assert len(chunks) == 3

    assert chunks[0].name == "large_run_chunk_1.log"
    assert chunks[1].name == "large_run_chunk_2.log"
    assert chunks[2].name == "large_run_chunk_3.log"

    c1_lines = chunks[0].read_text(encoding="utf-8").splitlines()
    c2_lines = chunks[1].read_text(encoding="utf-8").splitlines()
    c3_lines = chunks[2].read_text(encoding="utf-8").splitlines()

    assert len(c1_lines) == 10
    assert len(c2_lines) == 10
    assert len(c3_lines) == 5

    assert c1_lines[0] == "Step 000: Electronic energy calculation converged"
    assert c2_lines[0] == "Step 010: Electronic energy calculation converged"
    assert c3_lines[0] == "Step 020: Electronic energy calculation converged"


def test_nonexistent_log_file_handling(tmp_path: Path) -> None:
    """Verify non-existent log paths return empty list gracefully."""
    nonexistent = tmp_path / "does_not_exist.log"
    chunks = sanitize_and_chunk_log(nonexistent)
    assert chunks == []


def test_cli_execution_success(tmp_path: Path) -> None:
    """Verify CLI execution via subprocess runs cleanly."""
    log_file = tmp_path / "cli_test.log"
    log_file.write_text("CLI test line 1\nCLI test line 2\n", encoding="utf-8")

    sanitizer_script = Path(__file__).resolve().parent.parent / "ci_tools" / "Log_Sanitizer.py"
    result = subprocess.run(
        [sys.executable, str(sanitizer_script), str(log_file), "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Log sanitized and chunked into 1 files." in result.stdout
    assert (tmp_path / "cli_test_chunk_1.log").exists()
    assert (tmp_path / "cli_test_full.archive").exists()


def test_main_cli_function_variants(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main() function returns proper exit codes across valid and invalid arguments."""
    # 1. No arguments -> exit code 1
    assert main([]) == 1
    captured = capsys.readouterr()
    assert "Usage: python Log_Sanitizer.py" in captured.out

    # 2. Valid argument with fallback invalid chunk_size
    log_file = tmp_path / "main_test.log"
    log_file.write_text("Line A\nLine B\n", encoding="utf-8")
    assert main([str(log_file), "not_an_int"]) == 0
    captured = capsys.readouterr()
    assert "Log sanitized and chunked into 1 files." in captured.out

    # 3. Valid argument with numeric chunk_size
    log_file2 = tmp_path / "main_test2.log"
    log_file2.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")
    assert main([str(log_file2), "2"]) == 0
    captured = capsys.readouterr()
    assert "Log sanitized and chunked into 2 files." in captured.out

    # 4. Nonexistent file via main -> exit code 1
    nonexistent = tmp_path / "missing.log"
    assert main([str(nonexistent)]) == 1


def test_main_cli_default_argv_none() -> None:
    """Verify main(None) correctly defaults to sys.argv and returns 1 when no args passed."""
    old_argv = sys.argv
    try:
        sys.argv = ["Log_Sanitizer.py"]
        assert main(None) == 1
    finally:
        sys.argv = old_argv


def test_empty_zero_byte_log_overwrites_existing_archive(tmp_path: Path) -> None:
    """Verify 0-byte log replaces a pre-existing archive file cleanly."""
    log_file = tmp_path / "empty_with_prior.log"
    log_file.write_text("", encoding="utf-8")

    archive_file = tmp_path / "empty_with_prior_full.archive"
    archive_file.write_text("Stale archive from previous run", encoding="utf-8")

    chunks = sanitize_and_chunk_log(log_file, chunk_size=50)
    assert len(chunks) == 1
    assert chunks[0].name == "empty_with_prior_chunk_1.log"
    assert chunks[0].read_text(encoding="utf-8") == ""
    assert archive_file.exists()
    assert archive_file.read_text(encoding="utf-8") == ""
    assert not log_file.exists()
