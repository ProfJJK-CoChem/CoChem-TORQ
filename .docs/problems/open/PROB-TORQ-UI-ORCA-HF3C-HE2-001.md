# PROBLEM REPORT: Student UI Micro-Task Execution Failure for Helium Dimer (He2) in TORQ via ORCA HF-3c

**Problem ID:** `PROB-TORQ-UI-ORCA-HF3C-HE2-001`  
**Repository:** `TORQ` (`D:/__CoChem/GitHub-Repo/CoChem-TORQ`)  
**Interaction Environment:** `GitHub Codespaces`  
**Calculation Environment:** `github-actions`  
**Computational Engine:** `ORCA` (Version: 6.1.1)  
**Hamiltonian/Method:** `HF-3c` (Hartree-Fock with MINIX basis, D3BJ dispersion, and gCP geometric counterpoise)  
**Chemical System:** Helium Dimer van der Waals complex ($\text{He}_2$, intermolecular separation $R = 3.000\,\text{Å}$, $R_e \approx 2.97\,\text{Å}$)  
**Status:** `OPEN`  
**Severity:** `CRITICAL` (Student UI Execution Stoppage / Missing Parameter Controls / Absent CI Dispatch Infrastructure / Nuclear Topology Mismatch)  
**Target Pipeline:** SRS Auto-Heal & Coding Workflow (`trigger_srs_workflow`)  
**Created At:** 2026-09-19T22:15:00-05:00  
**Provenance:** Zero-Mock Physical Triage / Anti-Spoofing Protocol v4 (§1–§14) / Mendeleev Dynamic Mass Mandate  

---

## 1. Executive Summary & Student Impact

An exhaustive physical UI reproduction was conducted simulating an undergraduate science student attempting to execute a quantum chemistry micro-task in the **TORQ** repository inside a standard **GitHub Codespaces** interaction environment targeting remote execution on **github-actions** with computational engine **ORCA** and theoretical method **HF-3c** on the weakly bound Helium dimer ($\text{He}_2$, $R = 3.000\,\text{Å}$).

While physical execution of the local `ORCA 6.1.1` binary (`C:\ORCA_6.1.1\orca.exe`) on the authentic $\text{He}_2$ van der Waals complex confirms that HF-3c terminates normally and converges to an authentic electronic energy of $E = -5.671421147058\,\text{Eh}$ (with dispersion correction $\Delta E_{\text{disp}} = -0.000050296\,\text{Eh}$ and geometric counterpoise correction $\Delta E_{\text{gCP}} = -0.000015187\,\text{Eh}$), an undergraduate student **cannot achieve this calculation through the repository user interface**.

A student operating inside the environment encounters four catastrophic architectural blockers:
1. **Total UI Configuration Deficit in `UI/Start_TORQ.ipynb`**:
   The primary autocratic UI entry point (`UI/Start_TORQ.ipynb`) contains no dropdown widgets or configuration inputs for `Interaction Environment` (`GitHub Codespaces`), `Calculation Environment` (`github-actions`), `Engine` (`ORCA`), or `Method` (`HF-3c`). In Cell 12, it statically hardcodes `engine="ORCA"`, `method="wB97M-V"`, and `basis_set="def2-TZVPP"`. Furthermore, `PRESET_GEOMETRIES` in Cell 4 only provides covalently bound molecules (H2O2, (H2O)2, N2H4, CH3OH) and completely omits the Helium dimer.
2. **Missing Remote Calculation Environment Dispatch Infrastructure (`github-actions`)**:
   `CoChem-TORQ/.github/workflows/` contains zero calculation runner workflow files. There is no GitHub Actions runner dispatch client (REST API / `gh workflow run`) in `Start_TORQ.ipynb` or `Libraries/`. Furthermore, standard ambient tokens in GitHub Codespaces lack write permissions (`actions: write`) to dispatch workflows, and standard GitHub Actions cloud runners lack proprietary licensed ORCA binaries.
3. **Pydantic Validation & Basis Set Conflict for HF-3c in `torq_config.py`**:
   In `Libraries/torq_config.py`, `TorqRunParams` requires a non-augmented triple-zeta basis set when `bsse_correction="counterpoise"` is active, raising `ValueError: Counterpoise correction must be restricted to non-augmented triple zeta basis sets` if HF-3c with MINIX is supplied. Moreover, if an explicit basis set is passed with HF-3c to ORCA, ORCA aborts immediately because HF-3c is a composite method with a predefined MINIX basis set and built-in gCP.
4. **Topological & Nuclear Solver Mismatch for Diatomic Van der Waals Systems ($\text{He}_2$)**:
   TORQ's entire 10-phase pipeline is hardcoded for internal torsional scans ($V(\theta)$) along rotatable dihedral angles. For diatomic noble gas complexes ($\text{He}_2$), `TorqTopology` correctly identifies 0 rotatable dihedrals. The 1D Sinc-DVR solver in Phase 4/7 assumes a periodic angular potential coordinate $\theta \in [0, 2\pi]$ and lacks a 1D radial dissociation coordinate ($V(R)$) for non-covalent dimers.

---

## 2. Reproduction Steps (Physical Student Journey)

1. **Step 1: Open Repository in GitHub Codespaces**:
   - Student launches GitHub Codespaces container on `CoChem-TORQ`.
   - Opens the documented primary UI entry point: `UI/Start_TORQ.ipynb` (`Readme.md` §2 & §7).
2. **Step 2: Molecule Intake Configuration**:
   - Student inspects `preset_dropdown` in Cell 4.
   - Available options: `['Hydrogen Peroxide (H2O2)', 'Water Dimer ((H2O)2)', 'Hydrazine (N2H4)', 'Methanol (CH3OH)']`.
   - **FAILURE 1**: Helium dimer is missing from presets. Student must manually write JSON into the raw text area:
     ```json
     {
       "symbols": ["He", "He"],
       "coordinates": [
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 3.0]
       ],
       "charge": 0,
       "multiplicity": 1
     }
     ```
3. **Step 3: Attempting Environment & Engine Configuration**:
   - Student looks for widgets to select `Interaction Environment: GitHub Codespaces` and `Calculation Environment: github-actions`.
   - **FAILURE 2**: No widgets exist. The notebook has zero interactive controls for interaction or calculation environments.
   - Student looks for widgets to select `Engine: ORCA` and `Method: HF-3c`.
   - **FAILURE 3**: No engine or method widgets exist. Cell 12 hardcodes:
     ```python
     torq_config = TorqRunParams(
         tier="T3-3h",
         wall_time_tier="T3-3h",
         engine="ORCA",
         method="wB97M-V",
         basis_set="def2-TZVPP",
         dispersion="D4",
         bsse_correction="counterpoise",
         keywords=["RIJCOSX", "TightOpt", "TightSCF", "DEFGRID3", "InHess XTB2"]
     )
     ```
4. **Step 4: Attempting to Adapt TorqRunParams to HF-3c**:
   - Student modifies Cell 12 to target HF-3c:
     ```python
     torq_config = TorqRunParams(
         tier="T1",
         wall_time_tier="T1",
         engine="ORCA",
         method="HF-3c",
         basis_set="MINIX",
         bsse_correction="counterpoise",
         keywords=["TightOpt", "TightSCF", "DEFGRID3", "InHess XTB2"]
     )
     ```
   - **FAILURE 4 (Pydantic Validation Error)**:
     ```text
     pydantic_core._pydantic_core.ValidationError: 1 validation error for TorqRunParams
       Value error, Counterpoise correction must be restricted to non-augmented triple zeta basis sets. The basis set does not appear to be triple zeta.
     ```
5. **Step 5: Attempting Remote Dispatch to GitHub Actions**:
   - Student attempts to offload the calculation to GitHub Actions.
   - **FAILURE 5**: `CoChem-TORQ/.github/workflows/` is completely empty (0 YAML workflow files). There is no workflow dispatch client or handler to trigger remote CI runners.
6. **Step 6: Torsional Scan & Nuclear DVR Solver on He2**:
   - In Cell 06, `TorqTopology` returns 0 rotatable dihedrals.
   - In Cell 14, `RelaxedPESTorsionalDVR` executes a periodic 1D angular scan over $\theta \in [0, 2\pi]$ which is physically meaningless for a diatomic noble gas complex whose sole vibrational degree of freedom is the intermolecular distance $R$.

---

## 3. Root Cause Analysis (5 Whys)

1. **Why did the calculation fail to execute in GitHub Actions?**  
   The student UI workflow could not dispatch the ORCA HF-3c calculation payload to the GitHub Actions runner.
2. **Why was the calculation not dispatched?**  
   `UI/Start_TORQ.ipynb` lacks interactive configuration widgets for Calculation Environment (`github-actions`), Engine (`ORCA`), and Method (`HF-3c`), and `CoChem-TORQ/.github/workflows/` contains 0 workflow files.
3. **Why are the required parameters hardcoded in the UI?**  
   The notebook UI was authored as a fixed demonstration script hardcoding `TorqRunParams(engine="ORCA", method="wB97M-V", basis_set="def2-TZVPP")` in Cell 12, without providing reactive `ipywidgets` for runtime student parameter parameterization.
4. **Why does the pipeline reject or mischaracterize the Helium dimer?**  
   TORQ's architecture was engineered specifically for internal torsional rotors in polyatomic molecules. Diatomic van der Waals dimers ($\text{He}_2$) possess 0 dihedrals, causing the topological detector to find no rotatable axes and the DVR solver to lack a 1D radial dissociation coordinate $V(R)$.
5. **Why does TorqRunParams reject HF-3c?**  
   `TorqRunParams` enforces rigid basis set validation designed for high-tier DFT and wave-function methods (requiring non-augmented triple-zeta basis sets for counterpoise), failing to account for composite semi-empirical Hartree-Fock methods (such as HF-3c) which bundle predefined minimal basis sets (MINIX) and intrinsic geometric counterpoise (gCP).

---

## 4. Physical Zero-Mock Execution Evidence

An authentic zero-mock test suite was authored and physically verified on the host system at `tests/test_torq_gui_orca_hf3c_student_journey.py` (6/6 tests passed in 3.04s):

1. **Dynamic Mendeleev Atomic Provenance (Mendeleev Mandate)**:
   - Dynamic query: `mendeleev.element('He')`
   - Atomic Number: `2`
   - Standard Atomic Weight: `4.002602 u` [M]
   - Isotopic Provenance: Genuine CODATA/Mendeleev constants dynamically verified.
2. **Authentic Physical ORCA 6.1.1 Execution on He2**:
   - Host Binary: `C:\ORCA_6.1.1\orca.exe`
   - Input Deck:
     ```text
     ! HF-3c DEFGRID3
     %maxcore 3000
     * xyz 0 1
       He   0.00000000   0.00000000   0.00000000
       He   0.00000000   0.00000000   3.00000000
     *
     ```
   - Physical Observables:
     - Total SCF Energy: `-5.67135566434622 Eh`
     - Dispersion Correction ($\Delta E_{\text{disp}}$): `-0.000050296 Eh`
     - Geometric Counterpoise Correction ($\Delta E_{\text{gCP}}$): `-0.000015187 Eh`
     - Final Single Point Energy: `-5.671421147058 Eh`
     - Termination Marker: `****ORCA TERMINATED NORMALLY****` (Returncode: 0)
3. **Automated Test Results**:
   ```text
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_mendeleev_helium_mass_provenance PASSED
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_torq_ui_environment_and_engine_controls_defect PASSED
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_torq_github_actions_workflow_missing_defect PASSED
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_torq_run_params_hf3c_counterpoise_conflict PASSED
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_physical_he2_orca_hf3c_calculation_execution PASSED
   tests/test_torq_gui_orca_hf3c_student_journey.py::test_torq_torsional_dvr_defect_on_he2 PASSED
   ============================== 6 passed in 3.04s ==============================
   ```

---

## 5. SRS Remediation Plan & Action Items

The Software Requirements Specification (SRS) state machine (`trigger_srs_workflow`) is instructed to implement the following permanent engineering solutions:

1. **Interactive UI Configuration Controls (`UI/Start_TORQ.ipynb`)**:
   - Introduce reactive `ipywidgets.Dropdown` controls in Phase 2 for:
     - `Interaction Environment`: `["GitHub Codespaces", "Local Windows", "Local Linux", "macOS", "HPC Slurm"]`
     - `Calculation Environment`: `["local", "github-actions", "hpc-slurm"]`
     - `Engine`: `["ORCA", "xTB", "PySCF", "CFOUR"]`
     - `Method`: `["HF-3c", "r2SCAN-3c", "wB97M-V", "B3LYP-D4", "GFN2-xTB"]`
   - Wire widget state dynamically into `torq_controller` and Cell 12 execution payloads.
2. **Helium Dimer Preset Integration**:
   - Add `Helium dimer (He2)` ($R = 3.000\,\text{Å}$ and equilibrium $R_e = 2.97\,\text{Å}$) into `PRESET_GEOMETRIES` in Cell 4 with dynamic masses from `mendeleev`.
3. **Composite Method & Basis Set Support in `torq_config.py`**:
   - Update `TorqRunParams` validators to recognize composite 3c methods (`HF-3c`, `r2SCAN-3c`, `B97-3c`):
     - Automatically allow `basis_set=""` or `basis_set="MINIX"`.
     - Automatically suppress Boys-Bernardi counterpoise requirement when built-in gCP is present.
     - Prevent emitting explicit basis set keywords in ORCA decks when running composite methods.
4. **GitHub Actions Dispatch Workflow Authoring**:
   - Author `.github/workflows/cochem_orca_runner.yml` and `.github/workflows/cochem_torq_ci.yml` supporting `workflow_dispatch` payloads.
   - Implement an intelligent fallback in the UI: if running in GitHub Codespaces without write-level Actions dispatch permissions or if cloud ORCA runners are unavailable, provide an informative student diagnostic and allow local or containerized execution.
5. **1D Radial Dissociation Solver Branch in DVR**:
   - In `RelaxedPESTorsionalDVR`, add support for 1D radial distance coordinates $V(R)$ for diatomic and noble-gas van der Waals complexes that lack internal rotatable dihedrals.
