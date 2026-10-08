"""CoChem-TORQ: Method Matrix Parser & Provenance Test Suite.

=============================================================================
Phase 5 (Stage 4.0 / Task 8) Authentic Physical Test Matrix
----------------------------------------------------------
Comprehensive unit and integration test suite for Method Matrix v4 loader,
progressive cascade parser, explicitly selected advisory ML models, rigorous
scientific auxiliary basis validation, deterministic SHA-256 / xxHash-64
cryptographic provenance hashing, and Tripartite Air-Gap isolation.

Zero-Tolerance Anti-Mocking:
Tests exercise declared inputs, actual Mendeleev database lookups, and genuine
cryptographic hashes; model selection is not scientific method qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Libraries.cochem_torq_matrix_loader import (
    _XXHASH_AVAILABLE,
    METHOD_MATRIX_V4_TIERS,
    AirGapReport,
    AirGapViolationError,
    AuxiliaryBasisMismatchError,
    EnvironmentTier,
    ExecutionCascade,
    ExecutionContext,
    InvalidTierError,
    UnsupportedElementError,
    generate_provenance_hash,
    get_atomic_mass,
    get_atomic_number,
    get_covalent_radius,
    get_element_symbol,
    get_isotopic_mass,
    get_repo_root,
    get_vdw_radius,
    parse_execution_cascade,
    resolve_mlff_model,
    validate_auxiliary_basis,
    validate_element_symbols,
)

# =============================================================================
# 1. Mendeleev Dynamic Property Query Tests (Mendeleev Mandate)
# =============================================================================


class TestMendeleevDynamicProperties:
    """Authentic physical tests for dynamic atomic and isotopic retrieval via Mendeleev."""

    def test_dynamic_atomic_numbers(self) -> None:
        """Verify dynamic atomic number retrieval for diverse elements."""
        assert get_atomic_number("H") == 1
        assert get_atomic_number("He") == 2
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("F") == 9
        assert get_atomic_number("Si") == 14
        assert get_atomic_number("P") == 15
        assert get_atomic_number("S") == 16
        assert get_atomic_number("Cl") == 17
        assert get_atomic_number("Fe") == 26
        assert get_atomic_number("Br") == 35
        assert get_atomic_number("I") == 53
        assert get_atomic_number("Pt") == 78
        assert get_atomic_number("U") == 92
        assert get_atomic_number(6) == 6
        assert get_atomic_number(26) == 26

    def test_dynamic_element_symbols(self) -> None:
        """Verify dynamic element symbol canonicalization."""
        assert get_element_symbol("c") == "C"
        assert get_element_symbol("fe") == "Fe"
        assert get_element_symbol(1) == "H"
        assert get_element_symbol(8) == "O"
        assert get_element_symbol(92) == "U"

    def test_dynamic_atomic_masses(self) -> None:
        """Verify standard atomic weights are retrieved dynamically without hardcoding."""
        h_mass = get_atomic_mass("H")
        assert 1.007 < h_mass < 1.009

        c_mass = get_atomic_mass("C")
        assert 12.010 < c_mass < 12.012

        n_mass = get_atomic_mass("N")
        assert 14.006 < n_mass < 14.008

        o_mass = get_atomic_mass("O")
        assert 15.998 < o_mass < 16.000

        fe_mass = get_atomic_mass("Fe")
        assert 55.840 < fe_mass < 55.850

        u_mass = get_atomic_mass("U")
        assert 238.00 < u_mass < 238.05

    def test_dynamic_isotopic_masses(self) -> None:
        """Verify exact isotopic mass queries for physical isotopes."""
        c12_mass = get_isotopic_mass("C", 12)
        assert abs(c12_mass - 12.0000) < 1e-4

        c13_mass = get_isotopic_mass("C", 13)
        assert 13.003 < c13_mass < 13.004

        h1_mass = get_isotopic_mass("H", 1)
        assert 1.007 < h1_mass < 1.009

        h2_mass = get_isotopic_mass("H", 2)
        assert 2.014 < h2_mass < 2.015

        # Query without mass number returns standard mass
        h_std = get_isotopic_mass("H")
        assert abs(h_std - get_atomic_mass("H")) < 1e-6

    def test_dynamic_covalent_and_vdw_radii(self) -> None:
        """Verify Pyykkö covalent and van der Waals radii queries in Angstroms."""
        h_cov = get_covalent_radius("H")
        assert 0.30 < h_cov < 0.35

        c_cov = get_covalent_radius("C")
        assert 0.70 < c_cov < 0.80

        c_vdw = get_vdw_radius("C")
        assert 1.60 < c_vdw < 2.10

    def test_validate_element_symbols(self) -> None:
        """Verify chemical element sequence validation."""
        valid = validate_element_symbols(["c", "H", "O", "N", "Fe:"])
        assert valid == ["C", "H", "O", "N", "Fe"]

        with pytest.raises(UnsupportedElementError):
            validate_element_symbols(["C", "H", "XzUnknownElement"])


# =============================================================================
# 2. MLFF Fallback Hierarchy & Resolution Tests
# =============================================================================


class TestMLFFHierarchyResolution:
    """Only one explicit advisory model is permitted; element lists do not qualify it."""

    @pytest.mark.parametrize(
        "symbols",
        [
            ["C", "H", "O"],
            ["B", "C", "H"],
            ["Si", "H"],
            ["Fe", "C", "H"],
            ["Ar", "H", "O"],
            ["Ne", "C", "H"],
            ["U", "O", "F"],
        ],
    )
    def test_no_model_is_selected_implicitly(self, symbols) -> None:
        with pytest.raises(ValueError, match="requested advisory model"):
            resolve_mlff_model(symbols)

    @pytest.mark.parametrize(
        "symbols", [["B", "H"], ["Si", "H"], ["Fe", "H"], ["U", "O"]]
    )
    def test_incompatible_explicit_model_does_not_trigger_substitution(
        self, symbols
    ) -> None:
        with pytest.raises(UnsupportedElementError, match="no alternative method"):
            resolve_mlff_model(symbols, requested_model="MACE-OFF23")

    def test_multi_model_fallback_chain_is_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="Automatic model substitution is disabled"
        ):
            resolve_mlff_model(
                ["Fe", "C", "H"], fallback_chain=["MACE-OFF23", "GFN2-xTB"]
            )

    @pytest.mark.parametrize("model", ["MACE-OFF23", "AIMNet2"])
    def test_explicit_candidate_selection_is_advisory_and_unqualified(
        self, model
    ) -> None:
        spec, trail = resolve_mlff_model(["C", "H", "O"], requested_model=model)
        assert spec.name == model
        assert len(trail) == 1
        assert "candidate element coverage only" in trail[0]
        assert "unvalidated until checkpoint/domain qualification" in trail[0]

    def test_missing_checkpoint_domain_manifest_is_rejected(self) -> None:
        with pytest.raises(
            UnsupportedElementError, match="checkpoint-specific validated domain"
        ):
            resolve_mlff_model(["C", "H"], requested_model="MACE-POLAR-1")


# =============================================================================
# 3. Auxiliary Basis Scientific Validation Tests
# =============================================================================


class TestAuxiliaryBasisScientificValidation:
    """Authentic physical tests for auxiliary basis pairing validation per Method Matrix v4."""

    def test_valid_standard_dft_pairings(self) -> None:
        """Verify valid standard hybrid and meta-GGA DFT pairings with def2/J."""
        # wB97M-V with def2-TZVP and def2/J
        p1 = validate_auxiliary_basis("wB97M-V", "def2-TZVP", "def2/J")
        assert p1.is_compatible is True
        assert p1.ri_type == "RIJCOSX"

        # wB97X-D4 with def2-QZVPP and def2/J
        p2 = validate_auxiliary_basis("wB97X-D4", "def2-QZVPP", "def2/J")
        assert p2.is_compatible is True

        # r2SCAN-3c with def2-mSVP and def2/J
        p3 = validate_auxiliary_basis("r2SCAN-3c", "def2-mSVP", "def2/J")
        assert p3.is_compatible is True

    def test_valid_double_hybrid_and_correlation_pairings(self) -> None:
        """Verify valid Double-Hybrid DFT pairings with correlation auxiliary basis sets."""
        # revDSD-PBEP86 with def2-TZVPP and def2-TZVPP/C
        p1 = validate_auxiliary_basis(
            "revDSD-PBEP86", "def2-TZVPP", "def2/J def2-TZVPP/C"
        )
        assert p1.is_compatible is True
        assert "RIJCOSX+PT2/C" in p1.ri_type

        # DSD-PBEP86 with def2-QZVPP and def2-QZVPP/C
        p2 = validate_auxiliary_basis("DSD-PBEP86", "def2-QZVPP", "def2/J def2-QZVPP/C")
        assert p2.is_compatible is True

        # MP2 with def2-SVP and def2-SVP/C
        p3 = validate_auxiliary_basis("MP2", "def2-SVP", "def2/J def2-SVP/C")
        assert p3.is_compatible is True

        # DLPNO-CCSD(T) with def2-TZVPP and def2/JK def2-TZVPP/C
        p4 = validate_auxiliary_basis(
            "DLPNO-CCSD(T)", "def2-TZVPP", "def2/JK def2-TZVPP/C"
        )
        assert p4.is_compatible is True

    def test_valid_hf_and_ri_jk_pairings(self) -> None:
        """Verify HF exchange requires def2/JK."""
        p1 = validate_auxiliary_basis(
            "HF", "def2-SVP", "def2/JK", ri_approximation="RI-JK"
        )
        assert p1.is_compatible is True
        assert p1.ri_type == "RI-JK"

    def test_invalid_double_hybrid_missing_correlation_aux_basis(self) -> None:
        """Reject Double-Hybrid calculations where aux_basis only contains def2/J without /C."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("revDSD-PBEP86", "def2-TZVPP", "def2/J")
        assert "lacks correlation fitting basis" in str(exc_info.value)

        with pytest.raises(AuxiliaryBasisMismatchError):
            validate_auxiliary_basis("B2PLYP", "def2-TZVP", "def2/J")

    def test_invalid_ri_jk_with_only_def2_j(self) -> None:
        """Reject RI-JK when only Coulomb def2/J is provided."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis(
                "HF", "def2-TZVP", "def2/J", ri_approximation="RI-JK"
            )
        assert "def2/J' only provides Coulomb fitting" in str(exc_info.value)

    def test_diffuse_and_weak_complex_augmentation_validation(self) -> None:
        """Verify diffuse / weak complex systems require augmented basis sets."""
        # Unaugmented def2-TZVP on weak complex must be rejected
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis(
                "wB97M-V", "def2-TZVP", "def2/J", is_weak_complex=True
            )
        assert "diffuse/weak complex violation" in str(exc_info.value)

        # Minimally augmented ma-def2-TZVP is accepted
        p_ma = validate_auxiliary_basis(
            "wB97M-V", "ma-def2-TZVP", "def2/J", is_weak_complex=True
        )
        assert p_ma.is_compatible is True
        assert p_ma.is_diffuse_compatible is True

        # Dunning aug-cc-pVTZ with AutoAux is accepted
        p_aug = validate_auxiliary_basis(
            "CCSD(T)", "aug-cc-pVTZ", "AutoAux", is_diffuse=True
        )
        assert p_aug.is_compatible is True

    def test_cross_family_basis_mismatch_rejected(self) -> None:
        """Reject Dunning primary basis paired with Ahlrichs auxiliary basis without AutoAux."""
        with pytest.raises(AuxiliaryBasisMismatchError) as exc_info:
            validate_auxiliary_basis("wB97M-V", "cc-pVTZ", "def2/J")
        assert "Basis set family mismatch" in str(exc_info.value)


# =============================================================================
# 4. Method Matrix v4 Progressive Cascade Parser Tests
# =============================================================================


class TestExecutionCascadeParser:
    """Authentic physical tests for parse_execution_cascade across all tiers."""

    def test_parse_all_standard_tiers(self) -> None:
        """Verify parse_execution_cascade succeeds across all Method Matrix v4 tiers."""
        symbols = ["C", "H", "H", "H", "O", "H"]  # Methanol

        for tier_name in METHOD_MATRIX_V4_TIERS:
            cascade = parse_execution_cascade(symbols=symbols, tier=tier_name)
            assert isinstance(cascade, ExecutionCascade)
            assert cascade.tier == tier_name
            assert cascade.selected_mlff is None
            assert cascade.fallback_trail == ["ML advisory stage not requested."]
            assert cascade.provenance_hash is not None
            assert len(cascade.provenance_hash) == 64
            assert len(cascade.stages) == (2 if cascade.is_coupled_cluster else 1)
            assert all(stage["stage_name"] != "ML_Advisory" for stage in cascade.stages)
            assert cascade.element_symbols == ["C", "H", "H", "H", "O", "H"]
            assert set(cascade.atomic_numbers) == {1, 6, 8}

    def test_parse_diffuse_and_weak_complex_escalation(self) -> None:
        """Verify basis set auto-escalation to ma-def2 for weak complex systems."""
        symbols = ["O", "H", "H", "O", "H", "H"]  # Water dimer
        cascade = parse_execution_cascade(
            symbols=symbols,
            tier="T1-1h",
            is_weak_complex=True,
            frozen_monomer=True,
        )
        assert cascade.basis_set == "ma-def2-TZVP"
        assert cascade.is_weak_complex is True
        assert cascade.frozen_monomer is True
        assert "FROZEN_MONOMER" in cascade.extra_keywords
        assert "WEAK_COMPLEX" in cascade.extra_keywords

    def test_parse_diffuse_escalation_across_various_tiers(self) -> None:
        """Verify diffuse basis auto-escalation for def2-SVP, def2-TZVPP, def2-QZVPP."""
        symbols = ["F", "H", "O"]
        # T1-10s with custom def2-SVP
        c_svp = parse_execution_cascade(
            symbols,
            tier="T1-10s",
            is_diffuse=True,
            custom_overrides={"basis_set": "def2-SVP"},
        )
        assert c_svp.basis_set == "ma-def2-SVP"

        # T3-3h with def2-TZVPP
        c_tzvpp = parse_execution_cascade(symbols, tier="T3-3h", is_diffuse=True)
        assert c_tzvpp.basis_set == "ma-def2-TZVPP"

        # T3-12h with def2-QZVPP
        c_qzvpp = parse_execution_cascade(symbols, tier="T3-12h", is_diffuse=True)
        assert c_qzvpp.basis_set == "ma-def2-QZVPP"

    def test_parse_double_hybrid_tier_t3_3h(self) -> None:
        """Verify T3-3h resolves revDSD-PBEP86 with def2-TZVPP/C auxiliary fitting."""
        symbols = ["C", "H", "N", "O"]
        cascade = parse_execution_cascade(symbols=symbols, tier="T3-3h")
        assert cascade.method == "revDSD-PBEP86"
        assert cascade.basis_set == "def2-TZVPP"
        assert "def2-TZVPP/C" in cascade.aux_basis
        assert cascade.tol_max_g == 1e-5
        assert cascade.is_double_hybrid is True

    def test_parse_coupled_cluster_tier_t4_1d(self) -> None:
        """The requested coupled-cluster plan contains two stages without an implicit ML stage."""
        symbols = ["C", "H", "O"]
        cascade = parse_execution_cascade(symbols=symbols, tier="T4-1d")
        assert cascade.method == "DLPNO-CCSD(T)"
        assert cascade.is_coupled_cluster is True
        assert cascade.pno_setting == "TightPNO"
        assert len(cascade.stages) == 2
        assert cascade.stages[1]["stage_name"] == "Coupled_Cluster_Single_Point"

    def test_parse_custom_overrides(self) -> None:
        """Verify custom parameter overrides in cascade resolution."""
        symbols = ["C", "H", "O"]
        overrides = {
            "method": "wB97M-D4",
            "basis_set": "def2-QZVPP",
            "aux_basis": "def2/J",
            "grid_level": "defgrid3",
            "tol_max_g": 5e-5,
            "pno_setting": "TightPNO",
        }
        cascade = parse_execution_cascade(
            symbols, tier="T1-1h", custom_overrides=overrides
        )
        assert cascade.method == "wB97M-D4"
        assert cascade.basis_set == "def2-QZVPP"
        assert cascade.tol_max_g == 5e-5

    def test_invalid_tier_raises_error(self) -> None:
        """Verify invalid tier string raises InvalidTierError."""
        with pytest.raises(InvalidTierError) as exc_info:
            parse_execution_cascade(symbols=["C", "H"], tier="T99-UnknownTier")
        assert "Unrecognized Method Matrix v4 tier" in str(exc_info.value)

    def test_empty_symbols_raises_error(self) -> None:
        """Verify empty element sequence raises UnsupportedElementError."""
        with pytest.raises(UnsupportedElementError):
            parse_execution_cascade(symbols=[], tier="T1-10s")


# =============================================================================
# 5. Cryptographic Provenance Hash & Manifest Persistence Tests
# =============================================================================


class TestProvenanceHashAndManifest:
    """Authentic physical tests for deterministic SHA-256 provenance hashing and manifest generation."""

    def test_deterministic_sha256_hash_repeatability(self) -> None:
        """Verify exact byte-for-byte deterministic reproducibility of SHA-256 hash."""
        symbols = ["C", "H", "O", "N"]
        cascade1 = parse_execution_cascade(symbols=symbols, tier="T1-1h")
        cascade2 = parse_execution_cascade(symbols=symbols, tier="T1-1h")

        hash1 = generate_provenance_hash(cascade=cascade1, persist_manifest=False)
        hash2 = generate_provenance_hash(cascade=cascade2, persist_manifest=False)

        assert hash1 == hash2
        assert len(hash1) == 64
        assert int(hash1, 16) > 0

    def test_provenance_hash_sensitivity_to_parameters(self) -> None:
        """Verify hash diverges when physical or method parameters change."""
        symbols = ["C", "H", "O"]
        c1 = parse_execution_cascade(symbols=symbols, tier="T1-1h")
        c2 = parse_execution_cascade(symbols=symbols, tier="T3-3h")  # Different tier
        c3 = parse_execution_cascade(
            symbols=["C", "H", "S"], tier="T1-1h"
        )  # Different element

        h1 = generate_provenance_hash(cascade=c1, persist_manifest=False)
        h2 = generate_provenance_hash(cascade=c2, persist_manifest=False)
        h3 = generate_provenance_hash(cascade=c3, persist_manifest=False)

        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_manifest_persistence_to_artifacts_dir(self, tmp_path: Path) -> None:
        """Verify manifest writes successfully to Ring 3 persistent artifacts directory."""
        artifacts_dir = tmp_path / "test_artifacts_vault"
        ctx = ExecutionContext(custom_artifacts_dir=artifacts_dir)

        cascade = parse_execution_cascade(
            symbols=["C", "H", "F"], tier="T1-30min", context=ctx
        )
        manifest_file = artifacts_dir / "cochem_deployment_manifest.json"

        h = generate_provenance_hash(
            cascade=cascade, context=ctx, persist_manifest=True
        )
        assert manifest_file.exists()

        with open(manifest_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["manifest_version"] == "4.0.0"
        assert data["provenance_hash_sha256"] == h
        assert data["tier"] == "T1-30min"
        assert data["method"] == "wB97X-D4"
        assert "C" in data["atomic_masses_da"]
        assert "HARTREE_TO_KCAL_MOL" in data["physical_constants_codata"]
        if _XXHASH_AVAILABLE:
            assert data["provenance_hash_xxh64"] is not None


# =============================================================================
# 6. Tripartite Air-Gap Boundary Enforcement Tests
# =============================================================================


class TestAirGapEnforcement:
    """Authentic physical tests for Tripartite Filesystem Air-Gap boundary protection."""

    def test_ring1_repository_write_violation_raised(self) -> None:
        """Verify write attempts into Ring 1 static repository directory raise AirGapViolationError."""
        repo_root = get_repo_root()
        illegal_manifest_path = repo_root / "cochem_deployment_manifest.json"

        ctx = ExecutionContext()
        cascade = parse_execution_cascade(
            symbols=["C", "H"], tier="T1-10s", context=ctx
        )

        with pytest.raises(AirGapViolationError) as exc_info:
            generate_provenance_hash(
                cascade=cascade,
                context=ctx,
                output_manifest_path=illegal_manifest_path,
                persist_manifest=True,
            )
        assert "Tripartite Air-Gap Violation" in str(exc_info.value)

    def test_ring2_scratch_and_ring3_artifacts_paths_disjoint(
        self, tmp_path: Path
    ) -> None:
        """Verify scratch and artifacts directories resolve outside Ring 1 repo."""
        scratch_dir = tmp_path / "scratch_test"
        artifacts_dir = tmp_path / "artifacts_test"

        ctx = ExecutionContext(
            custom_scratch_dir=scratch_dir,
            custom_shm_dir=tmp_path / "shm_test",
            custom_artifacts_dir=artifacts_dir,
        )

        resolved_scratch = ctx.get_scratch_dir("sub_job_1")
        resolved_artifacts = ctx.get_artifacts_dir("sub_vault_1")

        assert resolved_scratch.exists()
        assert resolved_artifacts.exists()
        assert resolved_scratch != resolved_artifacts
        assert not ExecutionContext._is_repo_root_violation(resolved_scratch)
        assert not ExecutionContext._is_repo_root_violation(resolved_artifacts)

    def test_air_gap_report_model(self, tmp_path: Path) -> None:
        """Verify AirGapReport diagnostic model creation."""
        ctx = ExecutionContext()
        report = ctx.verify_air_gap_boundary(tmp_path / "safe_output.json")
        assert isinstance(report, AirGapReport)
        assert report.is_valid is True
        assert report.reason is None

    def test_environment_tier_path_resolutions(self, tmp_path: Path) -> None:
        """Verify path resolutions across different environment tiers."""
        for tier in EnvironmentTier:
            ctx = ExecutionContext(
                tier=tier,
                custom_scratch_dir=tmp_path / f"scratch_{tier.value}",
                custom_artifacts_dir=tmp_path / f"art_{tier.value}",
            )
            s_dir = ctx.get_scratch_dir()
            a_dir = ctx.get_artifacts_dir()
            assert s_dir.exists()
            assert a_dir.exists()


# =============================================================================
# 7. Hardware-Aware Telemetry & Execution Context Tests
# =============================================================================


class TestHardwareTelemetryAndContext:
    """Authentic physical tests for execution context and hardware telemetry."""

    def test_execution_context_detection(self) -> None:
        """Verify execution context detects host environment tier and hardware."""
        ctx = ExecutionContext()
        assert ctx.tier in EnvironmentTier
        assert ctx.num_cores >= 1
        assert ctx.max_memory_mb >= 1024
        assert isinstance(ctx.gpu_available, bool)
        assert isinstance(ctx.session_id, str)
        assert len(ctx.session_id) > 0

    def test_execution_cascade_contains_hardware_affinity(self) -> None:
        """Verify hardware affinity is embedded into ExecutionCascade."""
        cascade = parse_execution_cascade(symbols=["C", "H", "O"], tier="T1-10s")
        assert "tier" in cascade.hardware_affinity
        assert "num_cores" in cascade.hardware_affinity
        assert "max_memory_mb" in cascade.hardware_affinity
        assert cascade.hardware_affinity["num_cores"] >= 1
