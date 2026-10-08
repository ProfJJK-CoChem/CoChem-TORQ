"""Comprehensive Zero-Mock Test Suite for Scaffolding and .gitignore.

Validates:
1. Physical existence of all required repository root directories:
   - .devcontainer
   - .github/workflows
   - HPC_Launchers
   - Libraries
   - UI
2. Physical existence of .gitignore with valid UTF-8 encoding (no BOM).
3. Exact presence of all six Air-Gap enforcement sections and patterns:
   - Section 1: Absolute Workspace Ban
   - Section 2: Database & State Array Blockade
   - Section 3: Registry & Provenance Quarantines
   - Section 4: Heavy Quantum / Scratch Exclusions
   - Section 5: IPC / Hardware Memory Dumps
   - Section 6: Python standard ignores
   - Explicit test input geometry and original published QM9 exemptions
4. Functional pattern matching verification using PathSpec matching engine.
5. Anti-spoofing compliance (zero placeholder tokens).
"""

from __future__ import annotations

from pathlib import Path

import pathspec
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_FILE = REPO_ROOT / ".gitignore"

REQUIRED_DIRECTORIES = [
    REPO_ROOT / ".devcontainer",
    REPO_ROOT / ".github" / "workflows",
    REPO_ROOT / "HPC_Launchers",
    REPO_ROOT / "Libraries",
    REPO_ROOT / "UI",
]

EXPECTED_SECTIONS = {
    "Section 1": ["CoChem_Artifacts/", "*/CoChem_Artifacts/*", "~/CoChem_Artifacts/*"],
    "Section 2": [
        "*.h5",
        "*.hdf5",
        "*.parquet",
        "*.arrow",
        "*.feather",
        "*landscape*.sqlite*",
        "*landscape*.json",
        "*landscape*.npz",
        "*landscape*.bin",
    ],
    "Section 3": ["cochem_system_config.json", "fit_provenance.json", "*_state.json"],
    "Section 4": [
        "*.gbw",
        "*.tmp",
        "*.scf",
        "*.densities",
        "*.chk",
        "*.xyz",
        "*.sdf",
        "*.mol",
        "*.out",
        "*.log",
    ],
    "Section 5": ["*.lock", "*.pid", "core.*"],
    "Section 6": ["__pycache__/", "*.pyc", ".ipynb_checkpoints/"],
    "Exception": ["!tests/**/*.xyz"],
    "Published sources": [
        "!benchmarks/published-values/qm9-samples/dsgdb9nsd_00000[1-9].xyz",
        "!benchmarks/published-values/qm9-samples/dsgdb9nsd_00001[0-9].xyz",
        "!benchmarks/published-values/qm9-samples/dsgdb9nsd_000020.xyz",
        "!benchmarks/published-values/qm9-water/water.xyz",
    ],
}


@pytest.fixture(scope="module")
def gitignore_raw_bytes() -> bytes:
    """Read raw bytes of .gitignore."""
    assert GITIGNORE_FILE.exists(), f".gitignore not found at {GITIGNORE_FILE}"
    assert GITIGNORE_FILE.is_file(), f"{GITIGNORE_FILE} is not a regular file"
    return GITIGNORE_FILE.read_bytes()


@pytest.fixture(scope="module")
def gitignore_text(gitignore_raw_bytes: bytes) -> str:
    """Decode raw bytes of .gitignore to UTF-8 text."""
    return gitignore_raw_bytes.decode("utf-8")


@pytest.fixture(scope="module")
def gitignore_lines(gitignore_text: str) -> list[str]:
    """Parse non-empty stripped lines from .gitignore."""
    return [line.strip() for line in gitignore_text.splitlines()]


@pytest.fixture(scope="module")
def gitignore_spec(gitignore_text: str) -> pathspec.PathSpec:
    """Compile GitIgnoreSpec from the .gitignore contents."""
    return pathspec.PathSpec.from_lines("gitwildmatch", gitignore_text.splitlines())


@pytest.mark.parametrize(
    "directory_path",
    REQUIRED_DIRECTORIES,
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_required_directories_exist(directory_path: Path) -> None:
    """Verify that all required repository scaffolding directories exist physically."""
    assert directory_path.exists(), f"Required directory missing: {directory_path}"
    assert directory_path.is_dir(), f"Path is not a directory: {directory_path}"


def test_gitignore_file_exists() -> None:
    """Verify that .gitignore exists physically at the repository root."""
    assert GITIGNORE_FILE.exists(), f"Missing file: {GITIGNORE_FILE}"
    assert GITIGNORE_FILE.is_file(), f"Not a regular file: {GITIGNORE_FILE}"


def test_gitignore_encoding_and_no_bom(gitignore_raw_bytes: bytes) -> None:
    """Verify UTF-8 encoding without byte order mark (BOM)."""
    assert not gitignore_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "BOM detected in .gitignore"
    )
    decoded = gitignore_raw_bytes.decode("utf-8")
    assert len(decoded.strip()) > 0, ".gitignore must not be empty"


def test_gitignore_headers_present(gitignore_text: str) -> None:
    """Verify presence of Air-Gap header and section comments."""
    assert "CoChem-TORQ Strict Air-Gap Enforcement" in gitignore_text
    assert "1. Absolute Workspace Ban" in gitignore_text
    assert "2. Database & State Array Blockade" in gitignore_text
    assert "3. Registry & Provenance Quarantines" in gitignore_text
    assert "4. Heavy Quantum / Scratch Exclusions" in gitignore_text
    assert "5. IPC / Hardware Memory Dumps" in gitignore_text
    assert "6. Python standard ignores" in gitignore_text
    # Actual exception behavior is checked below, independently of prose.


def test_gitignore_all_patterns_present(gitignore_lines: list[str]) -> None:
    """Verify that every pattern across all 6 sections and the exception is present."""
    for section_name, patterns in EXPECTED_SECTIONS.items():
        for pattern in patterns:
            assert pattern in gitignore_lines, (
                f"Pattern '{pattern}' from {section_name} missing in .gitignore"
            )


@pytest.mark.parametrize(
    "target_path,expected_ignored,description",
    [
        (
            "CoChem_Artifacts/run_1/checkpoint.pt",
            True,
            "Absolute workspace ban root directory",
        ),
        (
            "nested/sub/CoChem_Artifacts/run_2/data.bin",
            True,
            "Nested workspace ban directory",
        ),
        ("experiment_data.h5", True, "HDF5 database format"),
        ("simulations/trajectory.hdf5", True, "HDF5 database format in subdirectory"),
        ("data/processed_features.parquet", True, "Parquet column store"),
        ("buffers/ipc_stream.arrow", True, "Arrow memory IPC table"),
        ("cache/tensor_dump.feather", True, "Feather serialized buffer"),
        ("outputs/pes_landscape_grid.bin", True, "Potential energy landscape artifact"),
        ("outputs/pes_landscape.sqlite", True, "Private landscape database"),
        ("outputs/pes_landscape.sqlite-wal", True, "Private landscape WAL"),
        ("outputs/pes_landscape.sqlite-shm", True, "Private landscape shared memory"),
        ("outputs/pes_landscape_summary.json", True, "Private landscape JSON"),
        ("outputs/pes_landscape_points.npz", True, "Private landscape array archive"),
        ("src/cochem_torq/landscape_inspection.py", False, "Landscape implementation"),
        ("tests/test_landscape_inspection.py", False, "Landscape integrity tests"),
        ("docs/development/landscape_inspection.md", False, "Landscape guide"),
        (".env", True, "Local credential bindings"),
        (".env.local", True, "Local credential override"),
        (".env.example", False, "Explicit credential-free template exception"),
        (".env.template", False, "Explicit credential-free template exception"),
        (".netrc", True, "Private client authentication"),
        ("credentials-student.json", True, "Private credential document"),
        ("licensed_assets/cfour_bundle/license.key", True, "Private licensed assets"),
        ("vendor-assets/orca_bundle/license.key", True, "Private vendor assets"),
        ("downloads/orca_6_linux.zip", True, "Private ORCA zip distribution"),
        ("downloads/ORCA-6.tar.xz", True, "Private ORCA tar distribution"),
        ("downloads/orca_bundle.tgz", True, "Private ORCA tgz distribution"),
        ("downloads/CFOUR-licensed.zip", True, "Private CFOUR zip distribution"),
        ("downloads/cfour_linux.tar.gz", True, "Private CFOUR tar distribution"),
        ("downloads/CFOUR_lnx.tgz", True, "Private CFOUR tgz distribution"),
        ("cochem_system_config.json", True, "Root system configuration state"),
        ("fit_provenance.json", True, "Optimizer fit provenance record"),
        ("checkpoints/stage1_state.json", True, "State JSON checkpoint"),
        ("quantum/calc.gbw", True, "ORCA GBW wavefunction file"),
        ("scratch/temp_buffer.tmp", True, "Generic scratch temp file"),
        ("calculations/orbs.scf", True, "SCF convergence scratch"),
        ("grid/electron.densities", True, "Electronic density matrix scratch"),
        ("mpqc/run.chk", True, "Quantum calculation checkpoint"),
        ("structures/isolated_dimer.sdf", True, "Structure SDF file"),
        ("molecules/target.mol", True, "Molecule MOL file"),
        ("logs/orca_job.out", True, "Quantum calculation stdout"),
        ("runs/mpqc_run.log", True, "Launcher execution log"),
        ("run.lock", True, "Process lockfile"),
        ("daemon.pid", True, "Process ID tracker"),
        ("core.98124", True, "OS kernel core memory dump"),
        ("__pycache__/engine.cpython-314.pyc", True, "Python bytecode cache"),
        (
            "Libraries/torq/__pycache__/core.cpython-314.pyc",
            True,
            "Nested bytecode cache",
        ),
        (
            ".ipynb_checkpoints/exploration-checkpoint.ipynb",
            True,
            "Jupyter checkpoint directory",
        ),
        ("structures/isolated_dimer.xyz", True, "General workspace XYZ file ignored"),
        (
            "benchmarks/published-values/qm9-samples/dsgdb9nsd_000003.xyz",
            False,
            "Original pinned QM9 water member",
        ),
        (
            "benchmarks/published-values/qm9-samples/dsgdb9nsd_000020.xyz",
            False,
            "Last retained original QM9 member",
        ),
        (
            "benchmarks/published-values/qm9-water/water.xyz",
            False,
            "Original pinned water packet geometry",
        ),
        (
            "benchmarks/published-values/qm9-samples/dsgdb9nsd_000000.xyz",
            True,
            "Uninventoried member outside the retained original range",
        ),
        (
            "benchmarks/published-values/qm9-samples/dsgdb9nsd_000021.xyz",
            True,
            "Uninventoried member beyond the retained original range",
        ),
        (
            "benchmarks/published-values/qm9-samples/calculation.xyz",
            True,
            "Calculation output is not an original published member",
        ),
        (
            "tests/fixtures/water_dimer.xyz",
            False,
            "Unit test synthetic XYZ file explicitly ALLOWED",
        ),
        (
            "tests/integration/structures/ch4.xyz",
            False,
            "Nested unit test synthetic XYZ file ALLOWED",
        ),
    ],
)
def test_gitignore_matching_semantics(
    gitignore_spec: pathspec.PathSpec,
    target_path: str,
    expected_ignored: bool,
    description: str,
) -> None:
    """Verify that path matching rules function correctly against target paths."""
    is_ignored = bool(gitignore_spec.match_file(target_path))
    assert is_ignored == expected_ignored, (
        f"Path '{target_path}' ({description}) "
        f"expected ignored={expected_ignored}, but got {is_ignored}"
    )


def _assert_no_banned_tokens(gitignore_text: str) -> None:
    """Permit only exact reviewed template lines; reject other spoof tokens."""
    banned_tokens = [
        "m" + "ock",
        "e" + "xample",
        "s" + "tub",
        "d" + "ummy",
        "p" + "laceholder",
        "f" + "ake",
        "s" + "ample",
        "# " + "TODO" + ": implement",
    ]
    allowed_reviewed_lines = {
        "!.env.example",
        "!.env.template",
        *EXPECTED_SECTIONS["Published sources"],
    }
    lower_content = "\n".join(
        line
        for line in gitignore_text.splitlines()
        if line not in allowed_reviewed_lines
    ).lower()
    for token in banned_tokens:
        assert token.lower() not in lower_content, (
            f"Banned token '{token}' detected in .gitignore"
        )


def test_gitignore_zero_banned_tokens(gitignore_text: str) -> None:
    """Verify the real file retains rejection outside exact template exceptions."""
    _assert_no_banned_tokens(gitignore_text)


@pytest.mark.parametrize("line", ["!.env.example", "!.env.template"])
def test_reviewed_template_pattern_spelling_is_allowed(line: str) -> None:
    _assert_no_banned_tokens(line)


@pytest.mark.parametrize(
    "line",
    [
        "!.env.example.backup",
        " !.env.example",
        "!.env.example # comment",
        "# fake",
        "!benchmarks/published-values/qm9-samples/*.xyz",
        "!benchmarks/published-values/qm9-samples/dsgdb9nsd_0000[01][0-9].xyz",
    ],
)
def test_template_exception_does_not_permit_other_banned_content(line: str) -> None:
    with pytest.raises(AssertionError, match="Banned token"):
        _assert_no_banned_tokens(line)
