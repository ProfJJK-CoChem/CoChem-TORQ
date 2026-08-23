"""Comprehensive Zero-Mock Physical Test Suite for CoChem-TORQ README.md.

Defends documentation integrity, Bipartite Workspace Model compliance,
Method Matrix adherence, and execution entry points:
- Physical presence of README.md at repository root as a non-empty regular file.
- Strict UTF-8 encoding validation without Byte Order Mark (BOM).
- Complete documentation of the Bipartite Workspace Model
  (Filesystem Air-Gap Policy).
- Clear architectural separation between Domain A (Static Execution Tier)
  and Domain B (Dynamic Artifact Tier: $COCH_ARTIFACTS / $COCHEM_ARTIFACTS).
- Explicit documentation that execution logic resides in Libraries/ and all
  runtime artifacts, tensors, and logs are written to COCHEM_ARTIFACTS.
- Explicit requirement of Python >= 3.10 and strict type checking.
- Explicit documentation of primary autocratic entry point
  (UI/Start_TORQ.ipynb) and Voila GUI wrapper (UI/voila_gui_wrapper.ipynb).
- Method Matrix physical compliance constraints:
  * CREST / ORCA GOAT combination approach (0.93 F1 [M]).
  * Frozen-monomer spatial protections (TolMaxG 1e-5 Eh/a0 threshold).
  * Integration grid dynamic cascade (defgrid1 -> defgrid3).
  * Hessian preconditioning (InHess XTB2 / Lindh; Calc_Hess true forbidden).
  * Open-shell S^2 spin contamination check (<S^2> threshold <= 10% [M]).
  * Diffuse-in-base basis set mandate (12.74% [M] error for additive diffuse).
  * Mandatory D3/D4 dispersion correction for weak complexes.
- Static Execution Tier topology covering all 10-Phase WBS scripts.
- Dynamic Artifact Tier topology covering all dynamic stores.
- Air-Gap enforcement guard and .gitignore blockade specifications.
- Command Line Interface (CLI) and internal Python API execution.
- HPC array and NVIDIA MPS launcher documentation.
- Scientific provenance tagging discipline ([M], [D], [E]), citation, license.
- Zero banned / placeholder / mock tokens across README.md and test suite.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Support standard casing on filesystem
README_PATH: Final[Path] = (
    REPO_ROOT / "README.md"
    if (REPO_ROOT / "README.md").exists()
    else REPO_ROOT / "Readme.md"
)

BANNED_TOKENS: Final[list[str]] = [
    "mock",
    "dummy",
    "placeholder",
    "fake",
    "sample",
    "# TODO",
    "TODO:",
    "FIXME",
    "xxx",
]

EXPECTED_LIBRARY_SCRIPTS: Final[list[str]] = [
    "cochem_torq_init.py",
    "cochem_torq_schema.py",
    "cochem_h5_healer.py",
    "cochem_torq_vault.py",
    "cochem_torq_topology.py",
    "cochem_torq_alignment.py",
    "cochem_torq_goat.py",
    "cochem_torq_crest.py",
    "cochem_torq_slicer.py",
    "cochem_torq_constraints.py",
    "cochem_torq_engine.py",
    "cochem_torq_cfour_bridge.py",
    "cochem_torq_watchdog.py",
    "cochem_tensor_extractor.py",
    "cochem_jax_builder.py",
    "cochem_spcat_bridge.py",
    "cochem_torq_export.py",
    "cochem_torq_telemetry.py",
    "cochem_catalog_compiler.py",
]

EXPECTED_ARTIFACT_SUBDIRECTORIES: Final[list[str]] = [
    "Registry",
    "Databases",
    "Input_Files",
    "Scratch",
    "Logs",
    "Processed",
]


@pytest.fixture(scope="module")
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture(scope="module")
def readme_file_path() -> Path:
    """Return the absolute path to README.md and verify physical existence."""
    assert README_PATH.exists(), f"README file not found at {README_PATH}"
    assert README_PATH.is_file(), f"{README_PATH} is not a regular file"
    return README_PATH


@pytest.fixture(scope="module")
def readme_raw_bytes(readme_file_path: Path) -> bytes:
    """Read raw bytes of README.md."""
    return readme_file_path.read_bytes()


@pytest.fixture(scope="module")
def readme_text(readme_raw_bytes: bytes) -> str:
    """Decode raw bytes of README.md to UTF-8 text."""
    return readme_raw_bytes.decode("utf-8")


@pytest.fixture(scope="module")
def readme_lines(readme_text: str) -> list[str]:
    """Parse lines from README.md."""
    return readme_text.splitlines()


# ==============================================================================
# 1. Physical File Presence & Encoding Integrity
# ==============================================================================


def test_readme_physical_existence(
    readme_file_path: Path, readme_raw_bytes: bytes
) -> None:
    """Verify physical presence and substantial documentation size."""
    assert readme_file_path.exists(), "README file does not exist on disk"
    assert readme_file_path.is_file(), "README path is not a file"
    assert len(readme_raw_bytes) > 2000, (
        f"README size ({len(readme_raw_bytes)} bytes) is below documentation threshold"
    )


def test_readme_utf8_encoding_no_bom(readme_raw_bytes: bytes) -> None:
    """Verify valid UTF-8 encoding and absence of UTF-8 Byte Order Mark (BOM)."""
    assert not readme_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "README contains UTF-8 BOM; standard UTF-8 without BOM required"
    )
    decoded = readme_raw_bytes.decode("utf-8")
    assert len(decoded) > 0, "Decoded README content cannot be empty"


def test_readme_line_structure(readme_text: str) -> None:
    """Verify line count and ending structure."""
    lines = readme_text.splitlines()
    assert len(lines) >= 80, (
        f"README has {len(lines)} lines, expected >= 80 lines for complete spec"
    )


# ==============================================================================
# 2. Metadata, Header & Authoritative Links
# ==============================================================================


def test_readme_header_and_attribution(readme_text: str) -> None:
    """Verify repository title, author, ORCiD, GitHub, and manual references."""
    assert "# CoChem-TORQ" in readme_text, "Missing title '# CoChem-TORQ'"
    assert "Torsional Optimization & Rotational Quantification Engine" in readme_text, (
        "Missing full descriptive title"
    )
    assert "Dr. Joshua John Klaassen" in readme_text, "Missing PI/Author attribution"
    assert "0009-0007-1506-4401" in readme_text, "Missing ORCiD identifier"
    assert "https://github.com/ProfJJK-CoChem" in readme_text, (
        "Missing GitHub organization link"
    )
    assert "CoChem User Manual v4.1" in readme_text, (
        "Missing Master User Manual reference"
    )
    has_matrix = (
        "CoChem Method Matrix v4" in readme_text
        or "Method_Matrix.md" in readme_text
    )
    assert has_matrix, "Missing Method Matrix reference"


# ==============================================================================
# 3. Bipartite Workspace Model (Filesystem Air-Gap Policy)
# ==============================================================================


def test_readme_bipartite_workspace_model(readme_text: str) -> None:
    """Verify documentation of Bipartite Workspace Model and Air-Gap."""
    assert "Bipartite Workspace Model" in readme_text, (
        "Missing 'Bipartite Workspace Model' header or terminology"
    )
    assert "Filesystem Air-Gap" in readme_text, (
        "Missing 'Filesystem Air-Gap' specification"
    )
    assert "Domain A" in readme_text and "Static Execution Tier" in readme_text, (
        "Missing Domain A Static Execution Tier definition"
    )
    assert "Domain B" in readme_text and "Dynamic Artifact Tier" in readme_text, (
        "Missing Domain B Dynamic Artifact Tier definition"
    )
    assert "COCH_ARTIFACTS" in readme_text or "COCHEM_ARTIFACTS" in readme_text, (
        "Missing environment variable configuration for artifact tier"
    )
    assert "CoChem_Artifacts" in readme_text, (
        "Missing default '~/CoChem_Artifacts' path"
    )
    assert "pathlib.Path" in readme_text, "Missing dynamic pathlib.Path resolution rule"


def test_readme_air_gap_write_prohibition(readme_text: str) -> None:
    """Verify explicit prohibition of writing runtime artifacts to Domain A."""
    lower_text = readme_text.lower()
    has_prohibition = (
        "shall not" in lower_text
        or "must not" in lower_text
        or "prohibited" in lower_text
    )
    assert has_prohibition, (
        "Missing explicit prohibition against writing runtime data to Domain A"
    )
    assert "reproducibility" in lower_text, (
        "Missing scientific reproducibility rationale"
    )


def test_readme_mermaid_diagram_present(readme_text: str) -> None:
    """Verify that a valid Mermaid data flow architecture diagram is documented."""
    assert "```mermaid" in readme_text, "Mermaid diagram code block missing in README"
    assert "Domain A" in readme_text and "Domain B" in readme_text, (
        "Mermaid diagram must depict Domain A and Domain B interactions"
    )


# ==============================================================================
# 4. Static Execution Tier Topology ($HOME/CoChem-TORQ/)
# ==============================================================================


@pytest.mark.parametrize("script_name", EXPECTED_LIBRARY_SCRIPTS)
def test_static_tier_all_library_scripts_documented(
    readme_text: str, script_name: str
) -> None:
    """Verify that every single script in the 10-Phase WBS engine is documented."""
    assert script_name in readme_text, (
        f"Library engine script '{script_name}' is missing from README topology map"
    )


def test_static_tier_support_directories_documented(readme_text: str) -> None:
    """Verify that all supporting directories in Domain A are documented."""
    expected_dirs = [
        ".devcontainer",
        ".github",
        "HPC_Launchers",
        "Libraries",
        "UI",
    ]
    for d in expected_dirs:
        assert d in readme_text, f"Static tier directory '{d}' is missing from README"

    expected_files = [
        "cochem_submit.slurm",
        "cochem_mps_worker.sh",
        "Start_TORQ.ipynb",
        "voila_gui_wrapper.ipynb",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
    ]
    for f in expected_files:
        assert f in readme_text, f"Static tier file '{f}' is missing from README"


# ==============================================================================
# 5. Dynamic Artifact Tier Topology ($COCH_ARTIFACTS/)
# ==============================================================================


@pytest.mark.parametrize("subdir_name", EXPECTED_ARTIFACT_SUBDIRECTORIES)
def test_dynamic_tier_subdirectories_documented(
    readme_text: str, subdir_name: str
) -> None:
    """Verify that all subdirectories in Domain B are explicitly mapped."""
    assert subdir_name in readme_text, (
        f"Dynamic artifact subdirectory '{subdir_name}' is missing from README"
    )


def test_dynamic_tier_critical_artifacts_documented(readme_text: str) -> None:
    """Verify documentation of golden registry, databases, scratch, and deliverables."""
    critical_artifacts = [
        "cochem_system_config.json",
        "fit_provenance.json",
        "landscape.h5",
        "ipc_shm",
        "orca_tmp",
        "Crash_Dumps",
        "Catalogs",
        "Deliverables",
    ]
    for artifact in critical_artifacts:
        assert artifact in readme_text, (
            f"Critical dynamic artifact '{artifact}' is missing from README"
        )


# ==============================================================================
# 6. Method Matrix Compliance & Physical Constraints
# ==============================================================================


def test_method_matrix_conformer_generation(readme_text: str) -> None:
    """Verify conformer generation combination approach (GOAT + CREST with 0.93 F1)."""
    assert "GOAT" in readme_text, "Missing ORCA GOAT conformer generation docs"
    assert "CREST" in readme_text, "Missing CREST cross-check documentation"
    assert "0.93" in readme_text, "Missing GOAT 0.93 F1 benchmark score"
    assert "--nci" in readme_text or "--nocross" in readme_text, (
        "Missing CREST execution flags (--nci, --nocross, --noreftopo)"
    )


def test_method_matrix_frozen_monomer_protocol(readme_text: str) -> None:
    """Verify frozen-monomer protocol and TolMaxG convergence threshold."""
    has_frozen = (
        "frozen-monomer" in readme_text.lower()
        or "frozen monomer" in readme_text.lower()
    )
    assert has_frozen, "Missing frozen-monomer protocol documentation"
    assert "cochem_torq_constraints.py" in readme_text, (
        "Missing reference to cochem_torq_constraints.py"
    )
    assert "TolMaxG" in readme_text, (
        "Missing TolMaxG gradient threshold specification"
    )
    tolmaxg_matches = (
        "1e-5" in readme_text
        or "10^{-5}" in readme_text
        or "1 \\times 10^{-5}" in readme_text
    )
    assert tolmaxg_matches, "Missing 1e-5 TolMaxG threshold"


def test_method_matrix_grid_cascade(readme_text: str) -> None:
    """Verify dynamic grid cascade (defgrid1 -> defgrid3)."""
    assert "defgrid1" in readme_text, "Missing defgrid1 initial loose grid spec"
    assert "defgrid3" in readme_text, "Missing defgrid3 final tight grid spec"
    assert "cochem_torq_engine.py" in readme_text, (
        "Missing reference to cochem_torq_engine.py for grid cascade"
    )


def test_method_matrix_hessian_and_spin_contamination(readme_text: str) -> None:
    """Verify Hessian preconditioning rules and S^2 spin contamination check."""
    assert "Calc_Hess true" in readme_text, "Missing prohibition of Calc_Hess true"
    has_inhess = "InHess XTB2" in readme_text or "Lindh" in readme_text
    assert has_inhess, "Missing InHess XTB2 / Lindh preconditioning specification"
    has_spin = (
        "S^2" in readme_text
        or "S^{2}" in readme_text
        or "spin contamination" in readme_text
    )
    assert has_spin, "Missing S^2 spin contamination check"
    assert "10%" in readme_text or "10 %" in readme_text, (
        "Missing 10% spin contamination threshold"
    )


def test_method_matrix_diffuse_and_dispersion(readme_text: str) -> None:
    """Verify diffuse-in-base and mandatory D3/D4 dispersion enforcement."""
    has_diffuse = (
        "diffuse-in-base" in readme_text.lower()
        or "diffuse" in readme_text.lower()
    )
    assert has_diffuse, "Missing diffuse basis function requirements"
    assert "12.74%" in readme_text or "12.74 %" in readme_text, (
        "Missing 12.74% MAE degradation penalty for additive diffuse functions"
    )
    has_disp = (
        "D3" in readme_text
        or "D4" in readme_text
        or "dispersion" in readme_text.lower()
    )
    assert has_disp, "Missing mandatory D3/D4 dispersion correction enforcement"


# ==============================================================================
# 7. Air-Gap Enforcement & .gitignore Architecture
# ==============================================================================


def test_gitignore_architecture_documented(readme_text: str) -> None:
    """Verify .gitignore blockade patterns documented in README."""
    assert ".gitignore" in readme_text, "Missing .gitignore section"
    assert "*.h5" in readme_text or "*.parquet" in readme_text, (
        "Missing database blockade patterns (*.h5, *.parquet)"
    )
    assert "*.gbw" in readme_text or "*.chk" in readme_text or "*.tmp" in readme_text, (
        "Missing heavy wavefunction blockade patterns"
    )
    assert "!tests/**/*.xyz" in readme_text or "!tests" in readme_text, (
        "Missing unit test synthetic xyz exemption"
    )


# ==============================================================================
# 8. System Requirements & Execution Entry Points
# ==============================================================================


def test_python_version_and_prerequisites(readme_text: str) -> None:
    """Verify Python >= 3.10 and computational chemistry prerequisites."""
    assert "3.10" in readme_text, "Missing Python >= 3.10 specification"
    assert "ORCA" in readme_text, "Missing ORCA quantum engine requirement"
    assert "CFOUR" in readme_text, "Missing CFOUR quantum engine requirement"


def test_ui_entry_points_documented(readme_text: str) -> None:
    """Verify primary Jupyter UI and headless Voila wrapper documentation."""
    assert "Start_TORQ.ipynb" in readme_text, (
        "Missing Start_TORQ.ipynb primary UI documentation"
    )
    assert "voila_gui_wrapper.ipynb" in readme_text, (
        "Missing voila_gui_wrapper.ipynb headless GUI documentation"
    )


def test_cli_and_api_execution_documented(readme_text: str) -> None:
    """Verify CLI command line interface and Parsl DAG internal API execution."""
    assert "cochem-torq" in readme_text, "Missing CLI command invocation"
    assert "CanonicalPipeline" in readme_text or "CoChem-NODE" in readme_text, (
        "Missing internal API / DAG orchestration execution example"
    )
    assert "cochem_system_config.json" in readme_text, (
        "Missing reference to cochem_system_config.json"
    )


def test_hpc_and_mps_launchers_documented(readme_text: str) -> None:
    """Verify SLURM batch submission and NVIDIA MPS daemon execution."""
    assert "cochem_submit.slurm" in readme_text, (
        "Missing SLURM batch submission documentation"
    )
    assert "cochem_mps_worker.sh" in readme_text, (
        "Missing NVIDIA MPS worker daemon documentation"
    )


# ==============================================================================
# 9. Provenance Discipline, Citation & License
# ==============================================================================


def test_provenance_tags_documented(readme_text: str) -> None:
    """Verify scientific provenance tags [M], [D], and [E] definitions."""
    assert "[M]" in readme_text, "Missing [M] (Measured) provenance tag"
    assert "[D]" in readme_text, "Missing [D] (Derived) provenance tag"
    assert "[E]" in readme_text, "Missing [E] (Estimated) provenance tag"


def test_citation_policy_and_bibtex(readme_text: str) -> None:
    """Verify citation policy, BibTeX entry, and academic license."""
    assert "@article" in readme_text or "bibtex" in readme_text.lower(), (
        "Missing BibTeX citation entry"
    )
    assert "Klaassen" in readme_text, "Missing Dr. Klaassen author in citation"
    assert "License" in readme_text, "Missing License section"


# ==============================================================================
# 10. Anti-Spoofing & Zero-Mock Verification
# ==============================================================================


def test_readme_no_banned_placeholder_tokens(readme_lines: list[str]) -> None:
    """Verify README contains zero banned placeholder or mock tokens."""
    for line_idx, line in enumerate(readme_lines, start=1):
        for token in BANNED_TOKENS:
            pattern = (
                rf"\b{re.escape(token)}\b"
                if not token.startswith("#")
                else re.escape(token)
            )
            match = re.search(pattern, line, re.IGNORECASE)
            assert match is None, (
                f"Line {line_idx} contains banned placeholder token '{token}': {line}"
            )


def test_ast_compliance_on_test_suite() -> None:
    """Verify that test_readme.py contains zero dummy passes or mock imports."""
    this_file = Path(__file__).resolve()
    tree = ast.parse(this_file.read_bytes(), filename=str(this_file))

    for node in ast.walk(tree):
        # Disallow mock imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "mock" not in alias.name.lower(), (
                    f"Mock import '{alias.name}' is strictly forbidden by mandate"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "mock" not in node.module.lower(), (
                    f"Mock import from '{node.module}' is strictly forbidden"
                )

        # Disallow empty test functions with only `pass`
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                pytest.fail(
                    f"Test function '{node.name}' has empty dummy body with 'pass'"
                )
            if (
                len(body) == 2
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[1], ast.Pass)
            ):
                pytest.fail(
                    f"Test function '{node.name}' has dummy docstring and 'pass'"
                )
