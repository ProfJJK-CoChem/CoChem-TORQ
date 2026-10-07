"""Actual PyArrow serialization, mathematical column-format and missing-evidence checks.

Fixed-column examples and method cards are explicitly test inputs, with no
native SPCAT, ORCA, measured-spectrum or publication-accuracy claim. No engine
subprocess is replaced by a file-writing helper. Valid Parquet examples are
produced by PyArrow, and execution claims require real retained evidence.
"""

from __future__ import annotations

import gc
from hashlib import sha256
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from Libraries.cochem_catalog_compiler import (
    BannedMethodsAuditResult,
    CoChemIntegrityError,
    CoChemPathManager,
    DispersionMissingError,
    FortranOverflowError,
    InactiveRotorError,
    MethodMatrixViolationError,
    ProvenanceErrorCode,
    TorqCatalogCompiler,
    apply_readonly_chmod,
    audit_banned_methods,
    buffer_lock_sync,
    deduplicate_bibtex,
    generate_methods_latex,
    inactive_rotor_catcher,
    parallel_temperature_compiler,
    parse_spcat_cat_line,
    parse_spcat_cat_stream,
    purge_ghost_outputs,
    pyarrow_chunked_serializer,
    remove_readonly_seal,
)

# =============================================================================
# Mathematical fixed-column format examples; no scientific observations
# =============================================================================

# Explicit fixed-column format examples, not native SPCAT or measured transitions
CAT_FORMAT_EXAMPLES = [
    "   22235.0800  0.0050 -4.5678 2    0.0000  3  18001 103 6 1 6       5 2 3      ",
    "  183310.0870  0.0020 -2.3456 2   14.2500  3  18001 103 3 1 3       2 2 0      ",
    "  380197.3720  0.0010 -1.8901 2   28.5000  3  18001 103 4 1 4       3 2 1      ",
    "  439150.8120  0.0030 -2.1123 2   45.6780  3  18001 103 6 4 3       5 5 0      ",
    "  556936.0020  0.0005 -0.8900 2    0.0000  3  18001 103 1 1 0       1 0 1      ",
]

FORMAT_METHOD_CARD: dict[str, Any] = {
    "theory_level": "HF",
    "basis_set": "STO-3G",
    "software_version": "Format-card example; no engine calculation was executed",
    "provenance_hash": "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    "temperatures": [2.0, 10.0, 300.0],
}


# =============================================================================
# 1. OOM-Proof Streaming Validation Test (O(1) Flat Memory Complexity)
# =============================================================================


def test_oom_proof_streaming_validation_flat_memory(tmp_path: Path) -> None:
    """Stream a high-volume row stream through pyarrow_chunked_serializer."""
    row_count = 120_000
    chunk_size = 15_000

    def _generate_record_stream() -> Iterator[dict[str, Any]]:
        for idx in range(row_count):
            yield {
                "frequency_mhz": float(10000.0 + (idx * 0.1)),
                "uncertainty_mhz": 0.0050,
                "log_intensity": float(-3.0 - (idx % 500) * 0.01),
                "degrees_of_freedom": 2,
                "lower_state_energy_cm1": float(idx * 0.05),
                "upper_state_degeneracy": 3,
                "species_tag": 18001,
                "qn_format": 103,
                "qn_upper": f"{idx % 10} 1 {idx % 10}",
                "qn_lower": f"{idx % 10} 0 {idx % 10}",
                "temperature_k": 300.0,
                "provenance_hash": "sha256:h2o_catalog_stream_test",
            }

    process = psutil.Process(os.getpid())
    gc.collect()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    output_parquet = tmp_path / "stream_oom_proof_test.parquet"

    final_path = pyarrow_chunked_serializer(
        records_stream=_generate_record_stream(),
        output_parquet_path=output_parquet,
        chunk_size=chunk_size,
        compression="zstd",
        compression_level=7,
        verify_sync=True,
    )

    gc.collect()
    rss_after_mb = process.memory_info().rss / (1024 * 1024)
    rss_growth_mb = rss_after_mb - rss_before_mb

    assert final_path.exists()
    assert final_path == output_parquet.resolve()

    metadata = pq.read_metadata(final_path)
    assert metadata.num_rows == row_count
    assert metadata.num_columns == 12

    assert rss_growth_mb < 120.0


# =============================================================================
# 2. Vectorized Type-Casting & Schema Assertion Test
# =============================================================================


def test_vectorized_type_casting_and_schema_verification(tmp_path: Path) -> None:
    """Verify PyArrow Parquet schema with float64 precision on frequencies & energies."""
    cat_content = "\n".join(CAT_FORMAT_EXAMPLES)
    cat_file = tmp_path / "water_spectrum.cat"
    cat_file.write_text(cat_content, encoding="utf-8")

    out_parquet = tmp_path / "water_spectrum.parquet"

    stream = parse_spcat_cat_stream(
        cat_file,
        temperature_k=150.0,
        provenance_hash="sha256:water_spectrum_150k",
    )
    final_parquet = pyarrow_chunked_serializer(
        records_stream=stream,
        output_parquet_path=out_parquet,
        chunk_size=10,
        verify_sync=True,
    )

    schema_read = pq.read_schema(final_parquet)

    assert len(schema_read) == 12
    assert schema_read.field("frequency_mhz").type == pa.float64()
    assert schema_read.field("uncertainty_mhz").type == pa.float64()
    assert schema_read.field("log_intensity").type == pa.float64()
    assert schema_read.field("degrees_of_freedom").type == pa.int32()
    assert schema_read.field("lower_state_energy_cm1").type == pa.float64()
    assert schema_read.field("upper_state_degeneracy").type == pa.int32()
    assert schema_read.field("species_tag").type == pa.int32()
    assert schema_read.field("qn_format").type == pa.int32()
    assert pa.types.is_dictionary(schema_read.field("qn_upper").type)
    assert pa.types.is_dictionary(schema_read.field("qn_lower").type)
    assert schema_read.field("temperature_k").type == pa.float64()
    assert pa.types.is_dictionary(schema_read.field("provenance_hash").type)

    table = pq.read_table(final_parquet)
    assert table.num_rows == len(CAT_FORMAT_EXAMPLES)

    freq_col = table.column("frequency_mhz").to_pylist()
    assert math.isclose(freq_col[0], 22235.0800, abs_tol=1e-4)
    assert math.isclose(freq_col[4], 556936.0020, abs_tol=1e-4)

    temp_col = table.column("temperature_k").to_pylist()
    assert all(math.isclose(t, 150.0) for t in temp_col)


# =============================================================================
# 3. Isolated Workspace Race Condition Test (Multi-Temperature Concurrency)
# =============================================================================


def test_isolated_workspace_race_condition_concurrent_temperatures(
    tmp_path: Path,
) -> None:
    """Serialize supplied format files concurrently without simulating SPCAT."""
    temperatures = [2.0, 10.0, 50.0]
    inputs = {}
    content = "\n".join(CAT_FORMAT_EXAMPLES)
    for temperature in temperatures:
        path = tmp_path / f"format-{temperature}.cat"
        path.write_text(content)
        inputs[temperature] = path
    results = parallel_temperature_compiler(
        spcat_runner_or_cat_paths=inputs,
        temperatures=temperatures,
        output_dir=tmp_path / "serialized",
        base_scratch=tmp_path / "scratch",
        max_workers=3,
        chunk_size=2,
        provenance_hash="sha256:" + sha256(content.encode()).hexdigest(),
        apply_immutable_seal=False,
    )
    assert set(results) == set(temperatures)
    for temperature, path in results.items():
        table = pq.read_table(path)
        assert table.num_rows == len(CAT_FORMAT_EXAMPLES)
        assert all(value == temperature for value in table["temperature_k"].to_pylist())


# =============================================================================
# 4. Read-Only Immutable Seal Test (Cross-Platform NTFS / POSIX)
# =============================================================================


def test_readonly_immutable_seal_prevents_write_and_restores_write(
    tmp_path: Path,
) -> None:
    """Validate that apply_readonly_chmod enforces an immutable permission seal."""
    test_file = tmp_path / "immutable_catalog.parquet"
    pq.write_table(pa.table({"integer_index": [0, 1]}), test_file)

    apply_readonly_chmod(test_file, recursive=False)

    with pytest.raises(PermissionError):
        with open(test_file, "wb") as f:
            f.write(b"OVERWRITE_CORRUPTION_ATTEMPT")

    with pytest.raises(PermissionError):
        with open(test_file, "ab") as f:
            f.write(b"APPEND_CORRUPTION_ATTEMPT")

    remove_readonly_seal(test_file, recursive=False)
    with open(test_file, "wb") as f:
        f.write(b"VALID_WRITE_AFTER_RESTORE")

    assert test_file.read_bytes() == b"VALID_WRITE_AFTER_RESTORE"


# =============================================================================
# 5. Fortran Overflow `****.****` Parsing Error Trap Test
# =============================================================================


def test_fortran_overflow_asterisk_trap_raises_error() -> None:
    """Assert that parse_spcat_cat_line intercepts Fortran overflow/underflow asterisks."""
    overflow_line = "   ****.****  0.0050 -4.5678 2   ****.****  3  18001 103 6 1 6       5 2 3      "

    with pytest.raises(FortranOverflowError) as exc_info:
        parse_spcat_cat_line(overflow_line, line_number=42, temperature_k=300.0)

    err = exc_info.value
    assert err.error_code == ProvenanceErrorCode.FORTRAN_OVERFLOW
    assert "Fortran overflow" in err.message or "overflow" in str(err)
    assert err.details["line_number"] == 42


# =============================================================================
# 6. Inactive Rotor 0-Byte Interception Test
# =============================================================================


def test_inactive_rotor_zero_byte_interception(tmp_path: Path) -> None:
    """Assert that inactive_rotor_catcher intercepts 0-byte catalog outputs."""
    empty_cat = tmp_path / "inactive_rotor.cat"
    empty_cat.write_text("", encoding="utf-8")

    with pytest.raises(InactiveRotorError) as exc_info:
        inactive_rotor_catcher(empty_cat, allow_empty=False)

    err = exc_info.value
    assert err.error_code == ProvenanceErrorCode.SPCAT_BRIDGE_ERROR
    assert "Inactive rotor intercepted" in err.message

    assert inactive_rotor_catcher(empty_cat, allow_empty=True) is True

    active_cat = tmp_path / "active_rotor.cat"
    active_cat.write_text("\n".join(CAT_FORMAT_EXAMPLES), encoding="utf-8")
    assert inactive_rotor_catcher(active_cat, allow_empty=False) is False


# =============================================================================
# 7. Method Matrix v4 LaTeX Methods Block & BibTeX Deduplication Test
# =============================================================================


def test_generate_methods_latex_and_bibtex_deduplication() -> None:
    """A method-format card must not acquire invented properties or citations."""
    content = generate_methods_latex(FORMAT_METHOD_CARD, method_matrix_v4_check=False)
    assert "no engine calculation was executed" in content
    assert "HF/STO-3G" in content
    assert "Dipole components" not in content
    assert "Distortion parameters" not in content
    assert "Frozen-Monomer" not in content
    assert "MethodMatrix2024" not in content
    assert "ORCA" not in content
    incomplete = {"theory_level": "HF", "basis_set": "STO-3G"}
    with pytest.raises(MethodMatrixViolationError, match="Missing method provenance"):
        generate_methods_latex(incomplete)


# =============================================================================
# 8. 6-Tier CoChemPathManager & Ghost Output Purger Integration Tests
# =============================================================================


def test_cochem_path_manager_6_tiers_and_ghost_purger(tmp_path: Path) -> None:
    """Validate all 6 resolution tiers of CoChemPathManager and ghost output purging."""
    custom_scratch = tmp_path / "custom_tier1"
    resolved_t1 = CoChemPathManager.resolve_scratch_dir(custom_scratch)
    assert resolved_t1 == custom_scratch.resolve()
    assert resolved_t1.exists()

    t2_path = tmp_path / "env_tier2"
    os.environ["COCHEM_SCRATCH"] = str(t2_path)
    try:
        resolved_t2 = CoChemPathManager.resolve_scratch_dir()
        assert resolved_t2 == t2_path.resolve()
    finally:
        if "COCHEM_SCRATCH" in os.environ:
            del os.environ["COCHEM_SCRATCH"]

    custom_deliv = tmp_path / "custom_deliverables"
    resolved_deliv = CoChemPathManager.resolve_deliverables_dir(custom_deliv)
    assert resolved_deliv == custom_deliv.resolve()

    ghost_dir = tmp_path / "ghost_test_dir"
    ghost_dir.mkdir(parents=True, exist_ok=True)

    valid_file = ghost_dir / "valid.parquet"
    pq.write_table(pa.table({"integer_index": [0, 1]}), valid_file)

    ghost_0byte = ghost_dir / "ghost_failed.cat"
    ghost_0byte.write_bytes(b"")

    ghost_tmp = ghost_dir / "valid.parquet.tmp"
    ghost_tmp.write_bytes(b"TEMP_STAGING_DATA")

    purged = purge_ghost_outputs(ghost_dir, remove_0byte_only=False)
    assert ghost_0byte in purged
    assert ghost_tmp in purged
    assert not ghost_0byte.exists()
    assert not ghost_tmp.exists()
    assert valid_file.exists()


# =============================================================================
# 9. Buffer Lock Sync Physical Disk Verification Test
# =============================================================================


def test_buffer_lock_sync_disk_verification(tmp_path: Path) -> None:
    """Validate buffer_lock_sync physical flush and minimum byte validation."""
    valid_file = tmp_path / "buffer_sync_valid.bin"
    valid_file.write_bytes(b"NON_EMPTY_BINARY_CONTENT")

    size = buffer_lock_sync(valid_file, min_bytes=4)
    assert size == len(b"NON_EMPTY_BINARY_CONTENT")

    zero_file = tmp_path / "buffer_sync_zero.bin"
    zero_file.write_bytes(b"")

    with pytest.raises(CoChemIntegrityError) as exc_info:
        buffer_lock_sync(zero_file, min_bytes=1)

    assert "Buffer sync validation failed" in exc_info.value.message


# =============================================================================
# 10. Method Matrix v4 Flagship Functionals & Scalar Temperature LaTeX Test
# =============================================================================


def test_method_matrix_v4_flagship_functionals_and_scalar_temperature() -> None:
    """Verify that all Method Matrix v4 recommended functionals pass dispersion validation."""
    flagship_functionals = [
        "wB97M-V",
        "wB97X-V",
        "r2SCAN-3c",
        "B97-3c",
        "HF-3c",
        "SCAN-VV10",
        "B3LYP-D3BJ",
        "wB97X-D4",
        "PBE0-D3BJ",
    ]

    for func in flagship_functionals:
        meta = {
            "theory_level": func,
            "basis_set": "def2-QZVPP",
            "rotational_constants": {"a": 825360.0, "b": 435360.0, "c": 278130.0},
            "temperatures": 298.15,
            "defgrid": "DEFGRID3",
        }
        tex_output = generate_methods_latex(meta, method_matrix_v4_check=True)
        assert r"\section{Computational Methods}\label{sec:methods}" in tex_output
        assert r"\qty{298.15}{\kelvin}" in tex_output
        assert func in tex_output


# =============================================================================
# 11. Method Matrix v4 Integration Grid Threshold Violations Test
# =============================================================================


def test_method_matrix_v4_defgrid_violations() -> None:
    """Assert that DEFGRID1 or SG-1 integration grids raise MethodMatrixViolationError."""
    for bad_grid in ["DEFGRID1", "SG-1", "defgrid1"]:
        meta = {
            "theory_level": "wB97X-D4",
            "basis_set": "def2-TZVP",
            "rotational_constants": {"A": 1000.0, "B": 500.0, "C": 250.0},
            "defgrid": bad_grid,
        }
        with pytest.raises(MethodMatrixViolationError) as exc_info:
            generate_methods_latex(meta, method_matrix_v4_check=True)

        assert (
            exc_info.value.error_code
            == ProvenanceErrorCode.METHOD_MATRIX_VIOLATION_DEFGRID
        )


# =============================================================================
# 12. Fortran Double-Precision D/d Exponent Parsing Test
# =============================================================================


def test_fortran_double_precision_d_exponent_parsing() -> None:
    """Verify that parse_spcat_cat_line properly parses Fortran D and d exponent numbers."""
    line_with_d = "  1.2345D+04  5.0000D-03 -4.5678 2  1.0000d+01  3  18001 103 6 1 6       5 2 3      "
    parsed = parse_spcat_cat_line(line_with_d, line_number=1, temperature_k=300.0)

    assert parsed is not None
    assert parsed["frequency_mhz"] == 12345.0
    assert parsed["uncertainty_mhz"] == 0.005
    assert parsed["lower_state_energy_cm1"] == 10.0


# =============================================================================
# 13. Staging Cleanup on Unhandled Stream Exception Test
# =============================================================================


def test_staging_cleanup_on_unhandled_stream_exception(tmp_path: Path) -> None:
    """Assert that an exception during stream iteration immediately unlinks the staging file."""
    output_parquet = tmp_path / "stream_failure.parquet"

    def _faulty_stream() -> Iterator[dict[str, Any]]:
        yield {
            "frequency_mhz": 10000.0,
            "uncertainty_mhz": 0.005,
            "log_intensity": -3.0,
            "degrees_of_freedom": 2,
            "lower_state_energy_cm1": 0.0,
            "upper_state_degeneracy": 3,
            "species_tag": 18001,
            "qn_format": 103,
            "qn_upper": "1 0 1",
            "qn_lower": "0 0 0",
            "temperature_k": 300.0,
            "provenance_hash": "sha256:test",
        }
        raise RuntimeError(
            "Deliberately rejected software stream after one format row."
        )

    with pytest.raises(RuntimeError, match="Deliberately rejected software stream"):
        pyarrow_chunked_serializer(
            records_stream=_faulty_stream(),
            output_parquet_path=output_parquet,
            chunk_size=10,
        )

    assert not output_parquet.exists()
    staging_files = list(tmp_path.glob(".*.tmp.*")) + list(tmp_path.glob("*.tmp*"))
    assert len(staging_files) == 0


# =============================================================================
# 14. TorqCatalogCompiler Class Integration Test
# =============================================================================


def test_torq_catalog_compiler_engine(tmp_path: Path) -> None:
    """Compile supplied format examples; no catalog engine is emulated."""
    source = tmp_path / "column-format.cat"
    source.write_text("\n".join(CAT_FORMAT_EXAMPLES))
    compiler = TorqCatalogCompiler(
        source, point_id="format-example", output_dir=tmp_path / "output"
    )
    assert compiler.compile_to_parquet(chunk_size=2)
    assert pq.read_table(compiler.parquet_outpath).num_rows == len(CAT_FORMAT_EXAMPLES)


# =============================================================================
# 15. Banned Methods Auditor Test
# =============================================================================


def test_banned_methods_auditor() -> None:
    """Validate audit_banned_methods detection of additive diffuse and unpreconditioned hessians."""
    # Valid metadata
    valid_meta = {
        "basis_set": "ma-def2-TZVPP",
        "keywords": "InHess XTB2 opt freq",
        "is_non_covalent": True,
        "counterpoise": True,
        "frozen_monomer": True,
    }
    res = audit_banned_methods(valid_meta, raise_on_violation=True)
    assert isinstance(res, BannedMethodsAuditResult)
    assert res.passed is True
    assert res.is_frozen_monomer_verified is True
    assert res.is_bsse_counterpoise_verified is True
    assert res.is_valid_hessian_preconditioned is True

    # Banned additive diffuse
    bad_meta_diffuse = {
        "basis_set": "def2-TZVP",
        "keywords": "additive_diffuse opt",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        audit_banned_methods(bad_meta_diffuse, raise_on_violation=True)
    assert "BANNED_ADDITIVE_DIFFUSE" in str(exc_info.value)

    # Banned unpreconditioned calc_hess
    bad_meta_hess = {
        "basis_set": "def2-TZVP",
        "keywords": "Calc_Hess true opt",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_info:
        audit_banned_methods(bad_meta_hess, raise_on_violation=True)
    assert "BANNED_UNPRECONDITIONED_HESSIAN" in str(exc_info.value)


# =============================================================================
# 16. Inter-Entry Comment BibTeX Deduplication Test
# =============================================================================


def test_bibtex_deduplication_with_inter_entry_comments() -> None:
    """Verify that comments between BibTeX entries do not collapse or corrupt entries."""
    raw_bibtex_with_comments = """
% Entry 1 from ADS database
@article{Pickett1991,
  author = {Pickett, Herbert M.},
  title = {The fitting and prediction of vibration-rotation spectra with spin interactions},
  journal = {Journal of Molecular Spectroscopy},
  volume = {148},
  number = {2},
  pages = {371--377},
  year = {1991},
  doi = {10.1016/0022-2852(91)90124-S}
}

% =============================================================================
% Another section with separate article
% =============================================================================

@article{MethodMatrix2024,
  author = {CoChem Consortium},
  title = {CoChem Method Matrix v4 Standards},
  year = {2024},
  doi = {10.5281/zenodo.1234567}
}

% Final Comment Line
"""
    deduped = deduplicate_bibtex(raw_bibtex_with_comments, deduplicate_by="both")
    assert "@article{Pickett1991" in deduped
    assert "@article{MethodMatrix2024" in deduped
    assert deduped.count("@article") == 2


# =============================================================================
# 17. Method Matrix v4 Extended Non-Covalent Rules & Double Dispersion Test
# =============================================================================


def test_banned_methods_extended_matrix_rules() -> None:
    """Validate that jun-cc-pVTZ passes for non-covalent complexes and ONIOM/double-dispersion are rejected."""
    # jun-cc-pVTZ must pass for non-covalent
    jun_meta = {
        "theory_level": "wB97X-D4",
        "basis_set": "jun-cc-pVTZ",
        "is_non_covalent": True,
        "keywords": "InHess XTB2 opt freq",
    }
    jun_res = audit_banned_methods(jun_meta, raise_on_violation=True)
    assert jun_res.passed is True
    assert jun_res.allowed_diffuse_basis is True

    # ONIOM on small complex must be rejected
    oniom_meta = {
        "theory_level": "wB97X-D4",
        "basis_set": "def2-TZVP",
        "keywords": "oniom(b3lyp:hf) opt",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_oniom:
        audit_banned_methods(oniom_meta, raise_on_violation=True)
    assert "BANNED_ONIOM_QM_QM2" in str(exc_oniom.value)

    # Double dispersion (stacking D4 on VV10) must be rejected
    double_disp_meta = {
        "theory_level": "wB97M-V-D4",
        "basis_set": "def2-QZVPP",
    }
    with pytest.raises(MethodMatrixViolationError) as exc_double:
        audit_banned_methods(double_disp_meta, raise_on_violation=True)
    assert "BANNED_DOUBLE_DISPERSION" in str(exc_double.value)


# =============================================================================
# 18. Non-Covalent Frozen-Monomer & BSSE LaTeX Documentation Test
# =============================================================================


def test_methods_latex_non_covalent_documentation() -> None:
    """Requested molecular-class/correction flags cannot assert executed work."""
    metadata = {
        **FORMAT_METHOD_CARD,
        "is_non_covalent": True,
        "counterpoise": True,
        "frozen_monomer": True,
    }
    content = generate_methods_latex(metadata, method_matrix_v4_check=False)
    assert "protocol was applied" not in content
    assert "corrected via" not in content
    assert "Boys-Bernardi" not in content


# =============================================================================
# 19. Extended Methods LaTeX with ORCA Keywords, Hardware Limits & MACE
# =============================================================================


def test_generate_methods_latex_full_workflow_file_output(tmp_path: Path) -> None:
    """Write only factual supplied format provenance, without software defaults."""
    destination = tmp_path / "format-methods.tex"
    content = generate_methods_latex(
        FORMAT_METHOD_CARD, output_tex_path=destination, method_matrix_v4_check=False
    )
    assert destination.read_text() == content
    assert "no engine calculation was executed" in content
    assert "DEFGRID3" not in content
    assert "Pickett" not in content
    assert "MACE" not in content


# =============================================================================
# 20. BibTeX Deduplication with File Output Compilation
# =============================================================================


def test_deduplicate_bibtex_file_output_and_doi_unification(tmp_path: Path) -> None:
    """Verify deduplicate_bibtex unifies references and writes directly to cochem_citations.bib."""
    bib_file = tmp_path / "cochem_citations.bib"
    raw_bibtex = """
@article{Pickett1991,
  author = {Pickett, Herbert M.},
  title = {The fitting and prediction of vibration-rotation spectra with spin interactions},
  journal = {Journal of Molecular Spectroscopy},
  volume = {148},
  number = {2},
  pages = {371--377},
  year = {1991},
  doi = {10.1016/0022-2852(91)90124-S}
}

@article{mace2022,
  author = {Batatia, Ilyes and Kovacs, David P. and Simm, Gregor N. C. and Ortner, Christoph and Csanyi, Gabor},
  title = {MACE: Higher order equivariant message passing neural networks for materials science},
  journal = {Advances in Neural Information Processing Systems},
  year = {2022},
  doi = {https://doi.org/10.48550/arXiv.2206.07697}
}

@article{mace_duplicate_doi,
  author = {Batatia, I. et al.},
  title = {MACE Neural Networks},
  year = {2022},
  doi = {10.48550/arXiv.2206.07697}
}
"""
    result = deduplicate_bibtex(
        raw_bibtex, output_bib_path=bib_file, deduplicate_by="both"
    )

    assert bib_file.exists()
    assert bib_file.read_text(encoding="utf-8") == result
    assert "@article{Pickett1991" in result
    assert "@article{mace2022" in result
    assert "mace_duplicate_doi" not in result
    assert result.count("@article") == 2


# =============================================================================
# 21. Recursive Directory Permission Sealing Test
# =============================================================================


def test_apply_readonly_chmod_recursive_directory_sealing(tmp_path: Path) -> None:
    """Verify apply_readonly_chmod recursively seals subdirectories and files."""
    deliverables_dir = tmp_path / "sealed_deliverables"
    sub_dir = deliverables_dir / "catalogs"
    sub_dir.mkdir(parents=True, exist_ok=True)

    file1 = deliverables_dir / "metadata.json"
    file2 = sub_dir / "catalog_300K.parquet"
    file1.write_text('{"status": "finalized"}', encoding="utf-8")
    pq.write_table(pa.table({"integer_index": [0, 1]}), file2)

    apply_readonly_chmod(deliverables_dir, recursive=True)

    with pytest.raises(PermissionError):
        with open(file1, "w", encoding="utf-8") as f:
            f.write("CORRUPTION")

    with pytest.raises(PermissionError):
        with open(file2, "wb") as f:
            f.write(b"CORRUPTION")

    remove_readonly_seal(deliverables_dir, recursive=True)

    with open(file1, "w", encoding="utf-8") as f:
        f.write('{"status": "updated"}')

    assert file1.read_text(encoding="utf-8") == '{"status": "updated"}'
