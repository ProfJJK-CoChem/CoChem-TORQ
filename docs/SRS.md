---
schema: cochem-srs/5.0
repo: CoChem-TORQ
status: ACTIVE_CORE_STUDENT_ON_HOLD
pipeline_baseline: 5.0.0
source_commit: 3d2a5b24e68f052b70d0a7185c1da753ed328f4f
decision_ref: swarm-1126-8efd
---

# CoChem-TORQ — Canonical Software Requirements Specification (Pipeline 5.0.0)

> **HONEST RELEASE NOTICE (PRESERVED FROM `docs/development/scientific_acceptance_gaps.md`):**
> The canonical student application uses a named, pinned PySCF/geomeTRIC recipe with authentic optimization, property calculations, isotope-specific inertia, and projected normal modes for bounded teaching and numerical validation. It does **not** establish complete SRS qualification or laboratory/astronomical identification accuracy. `experimental_accuracy_established` and `identification_ready` remain `false`. **Student deployment testing is ON HOLD.**

## 1. Purpose, Scope & System Classification

`CoChem-TORQ` is an `ACTIVE_CORE` repository in the CoChem Pipeline 5.0.0 ecosystem providing reaction-coordinate optimization, transition-structure candidate workflows, rovibrational precursor calculations, Pickett/SPCAT external execution bridges, rigid-fragment and NAO/NBO bonding analyses, and canonical `cochem.modules` provider services.

- **Primary installable package:** `src/cochem_torq/` (`cochem_torq`, `cochem_torq.engines`, `cochem_torq.spectroscopy`).
- **CLI entrypoints (`pyproject.toml`):** `cochem-torq` (`src/cochem_torq/cli.py:main`) and `cochem-torq-ui` (`src/cochem_torq/ui.py:main`).
- **Ecosystem module provider:** `cochem.modules` entrypoint `torq = cochem_torq.ecosystem:module_provider` backed by `src/cochem_torq/module_provider.py` and `src/cochem_torq/ecosystem.py`.
- **Legacy compatibility packages retained in wheel under explicit zero-wildcard allowlist (`sys-1126-a3d4`):** `Libraries`, `UI`, `orchestrator`, `scripts`, `core`, and the 15 actively referenced `cochem.*` subpackages required by `.github/workflows/integrity.yml` and `tests/`. Ten unreferenced colliding `cochem.*` subpackages (`cochem.cli`, `cochem.core.diagnostics`, `cochem.core.ipc`, `cochem.gui`, `cochem.gui.widgets`, `cochem.intake`, `cochem.ml`, `cochem.plugins`, `cochem.provenance`, `cochem.telemetry`) were excluded from distribution wheels in Wave 3.

## 2. Pipeline 5.0.0 Architecture Alignment & Resolved Conflicts

`CoChem-TORQ` aligns with the Pipeline 5.0.0 control-plane and execution topology ratified under `swarm-1126-8efd` and `sys-1126-a3d4`:

- **CONTRA-01 (Execution Topology):** Local WSL2 workstation (`~/CoChem_Workstation_Direct/Modules/torq-env`, synced via read-only `git archive HEAD`) and GCP Cloud Run / Colab GPU cloud offload (`cochem-mcp-mailbox-request/2`) replace earlier Codespaces-first assumptions. Student browser UI packaging (`UI/Start_TORQ.ipynb` via `src/cochem_torq/ui.py`) remains a bounded optional teaching interface currently **ON HOLD** for student deployment.
- **CONTRA-02 & CONTRA-03 (Scheduling & Job Harness):** Deferred orchestration routes exclusively through `cochem-scheduled-prompt/1` and `cochem-mcp-mailbox-request/2`; legacy `g01..g22` queue lanes are retired.
- **CONTRA-06 (Packaging & Schema Consolidation):** Open glob wildcards (`Libraries*`, `cochem*`, `scripts*`, `core*`, `orchestrator*`, `UI*`) in `pyproject.toml` are replaced by an explicit zero-wildcard allowlist so `CoChem-TORQ` wheels cannot sweep in sibling `cochem_*` namespaces or unreferenced `cochem.*` collision trees. Typed request/plan/artifact schemas live in `src/cochem_torq/domain.py`, `src/cochem_torq/service.py`, and `src/cochem_torq/artifacts.py`.

## 3. Authoritative Scientific Engines, Methods & Method Matrix Constraints

All computational claims and deck emitters in `CoChem-TORQ` obey the CoChem Method Matrix (`Method_Matrix.md` §4.3–§4.4, §9B.4) and the v4.1 Official Update Specification:

- **Canonical Local/CI QM Engine:** Pinned `PySCF 2.14.0` + `geomeTRIC 1.1.1` + `dftd4 3.7.0` (`src/cochem_torq/engines/`), executing genuine energies, gradients, optimizations, harmonic Hessians, finite-difference anharmonic stencils, Lowdin-AO/NAO Wiberg bond indices, and authentic NBO 7 (`ORCA %nbo` / `GenNBO`) via `src/cochem_torq/module_provider.py`.
- **External Spectroscopy Backend:** Genuine external Pickett `SPCAT`/`SPFIT` execution (`src/cochem_torq/spectroscopy/`) provisioned via `ci_tools/provision_pickett_validation.py` for pre-verified user-supplied decks; automated `.var`/`.int` generation and Watson S-reduction remain explicitly gated by `NotImplementedError` rather than approximated.
- **Method Matrix §4.4 `%geom` & `! TightOpt` Constraint:** Any ORCA QM optimization deck claiming rotational-constant accuracy (`<= 0.5%` in $B$) must pair optimization keywords with the full §4.4 `%geom` block (`TolE <= 1e-7`, `TolRMSG <= 3e-6`, `TolMaxG <= 1e-5`, `TolRMSD <= 5e-5`, `TolMaxD <= 1e-4`), `TightSCF`/`VeryTightSCF`, and `DEFGRID3` (for DFT/Hessian rows). MLFF/ExtOpt conformer pre-screening (`Libraries/cochem_torq_goat.py`, `scripts/oet_client.py`) may use `! GOAT-EXPLORE ExtOpt TightOpt` ONLY when paired with `%scf TolE 1e-5 end` per §4.4 and §9B.4 (pre-optimization only; no spectroscopic accuracy claim). Enforced by `tests/test_tightopt_geom_gate.py`.

## 4. Cross-Repo Interfaces, Contracts & Canonical Schemas

- **`CoChem-BASE` Module Provider (`src/cochem_torq/module_provider.py`, `src/cochem_torq/ecosystem.py`):** Exposes canonical `cochem.module-provider/1` operations including `geometry_analysis`, `research_scan`, `wiberg_lowdin`, `wiberg_nao`, and `nbo_analysis`, preserving explicit isotope masses, charge/multiplicity, and source provenance hashes.
- **`CoChem-TOPOS` Reviewed Conformer Handoff (`src/cochem_torq/reviewed_topos.py`, `docs/TOPOS_HANDOFF.md`):** Consumes reviewed conformer bundles via `cochem-torq import-topos-reviewed` without invoking legacy GOAT/ML discovery silently.
- **`CoChem-SpycFit` & `CoChem-GEOM` Bridge (`src/cochem_torq/chem490_vpt2_bridge.py`):** Exports structured `export_spyfy_geom_handoff` payloads carrying authentic harmonic/anharmonic metadata and explicit qualification flags (`experimental_accuracy_established`, `identification_ready`).
- **Sealed Artifact & Fenced Persistence (`src/cochem_torq/artifacts.py`, `cochem/storage/fenced_pes.py`, `cochem/orchestration/campaign.py`):** Produces SHA-256 manifest-verified calculation shards validated by `verify_shard()` and `ci_tools/validate_release_evidence.py`.

## 5. Verification, Zero-Mock CI Gates & Acceptance Criteria

`CoChem-TORQ` enforces fail-closed verification across local and GitHub Actions CI (`.github/workflows/integrity.yml`):

- **Zero-Mock & Anti-Spoofing Gate (`tests/test_zero_mock_and_no_sim_gate.py`):** Zero banned simulation/mock patterns in production code and zero amnesty debt (`.anti_spoof_amnesty.json`).
- **Strict Air-Gap `.gitignore` Gate (`tests/test_gitignore.py`):** Enforces all 6 Air-Gap exclusion sections including explicit `__pycache__/`, `*.pyc`, and `*.py[cod]` ignores, zero tracked `.pyc` files, and zero banned spoof tokens.
- **Method Matrix `! TightOpt` / `%geom` Gate (`tests/test_tightopt_geom_gate.py`):** Fail-closed regression gate rejecting bare `! TightOpt` decks missing §4.4 `%geom` convergence tolerances (outside the documented MLFF/ExtOpt `%scf TolE 1e-5` pre-screen exception) and guarding `src/cochem_torq/` against new unverified `! TightOpt` emissions.
- **Lint, Format, Type & Native Engine Suites:** `ruff check`, `ruff format --check`, `mypy src/cochem_torq --follow-imports=silent --ignore-missing-imports`, `tests/test_module_provider.py`, `tests/test_packaged_student_ui.py`, `tests/test_pyscf_integration.py`, `tests/test_revdsd_integration.py`, and `ci_tools/run_repository_regression.py` with `ci_tools/require_no_skips.py`.

## 6. Verified Implementation Gaps, Known Holds & Non-Goals

Per `docs/development/scientific_acceptance_gaps.md` and the Pipeline 5.0.0 audit, the following gaps remain explicitly open and un-fabricated:

- **Student Deployment Status:** **ON HOLD.** `experimental_accuracy_established` and `identification_ready` remain `false` across all canonical student outputs.
- **20 Honest `NotImplementedError` Safety Gates:** Retained across `Libraries/cochem_spcat_bridge.py` (5), `Libraries/cochem_torq_pipeline.py` (5), `Libraries/cochem_torq_asymmetric_rotor.py` (3), `Libraries/cochem_torq_engine.py` (2), `Libraries/cochem_torq_spcat.py` (2), `src/cochem_torq/capabilities.py` (2), and `scripts/spcat_runner.py` (1) to block unqualified automated SPCAT `.var`/`.int` generation, Watson S-reduction, CFOUR derivative execution, CP geometry optimization, CABS mappings, and ORCA GPU adapters.
- **Scientific Qualification Remaining:** Exact open-source `revDSD-PBEP86-D4` orbital/LibXC/D4 reconciliation (`D01`), full semirigid rovibrational VPT2 ($\alpha_i$, Coriolis, $B_0 = B_e - \frac{1}{2}\sum\alpha_i$, `D05`), matrix-free Lanczos/Davidson 3D/6D DVR & periodic LAM solver (`D05`), verified IRC/TS endpoint proof (`ts_verified=false`, `D06`), and calibrated identification UQ (`D10`/`D11`).
- **Known Legacy Technical Debt:** Three broken `cochem_base` submodule imports in legacy `Libraries/cochem_torq_compiler.py` and `Libraries/cochem_torq_vibrational.py`; bare `keywords=["Opt", "TightOpt", "InHess Lindh"]` in legacy `Libraries/cochem_torq_pipeline.py`; three open UI/Actions tickets under `.docs/problems/open/`; and no direct `PENDING_WORKSTATION` / `tpo_ledger.sqlite` dispatch wired inside `CoChem-TORQ`.
- **Non-Goals:** `CoChem-TORQ` does not fabricate missing spectroscopic constants, bypass `NotImplementedError` safety gates, or substitute MLFF float32 pre-optimizations for converged §4.4 QM geometries.

## 7. Legacy Specification Disposition (Non-Normative Archive)

All prior multi-version SRS documents in `docs/wiki/` and historical design folders are superseded by this canonical Pipeline 5.0.0 `docs/SRS.md` and indexed as **NON-NORMATIVE ARCHIVE** in `docs/archive/README.md`:

- `docs/wiki/CoChem-TORQ_Implementation_SRS.md` (retained in place for historical traceability and `ci_tools/validate_release_evidence.py` links; superseded by `docs/SRS.md`).
- `docs/wiki/CoChem-TORQ_Comprehensive_SRS.md`, `docs/wiki/CoChem-TORQ_True_SRS.md`, `docs/wiki/CoChem-TORQ_True_SRS_Iterated.md`, `docs/wiki/CoChem-TORQ_AutoArchitect_Phase1.md`, and `docs/wiki/CoChem-TORQ_Integration_Architecture.md` (historical v4.x/draft baselines; non-normative).
- `docs/development/scientific_acceptance_gaps.md` and `docs/development/release_candidate_readiness.md` remain active companion qualification registers synchronized with Section 6 above.
