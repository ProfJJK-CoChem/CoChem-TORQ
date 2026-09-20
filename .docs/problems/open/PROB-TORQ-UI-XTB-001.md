# SRS Problem Ingestion: TORQ UI Micro-Task Execution Failure

**Problem ID**: `PROB-TORQ-UI-XTB-001`  
**Target Repository**: `TORQ` (`D:/__CoChem/GitHub-Repo/CoChem-TORQ`)  
**Interaction Environment**: GitHub Codespaces  
**Calculation Environment**: GitHub Actions  
**Quantum Engine**: xTB  
**Theoretical Method**: GFN2-xTB  
**Chemical System**: Helium van der Waals complex ($\text{He}_2$, intermolecular separation $R = 3.0\text{ \AA}$)  
**Provenance**: Empirical Verification Failure `[E]`  
**Governing Directives**: Anti-Spoofing Protocol v4 (§1-§14), Method Matrix v4, Mendeleev Mandate.

---

## 1. Defect Description
An automated micro-task validating the student journey in the TORQ repository failed to complete the specified quantum chemistry calculation through the repository UI. A student operating inside a GitHub Codespaces container targeting remote execution on GitHub Actions with quantum engine xTB and method GFN2-xTB on the Helium dimer ($\text{He}_2$, $R = 3.0\text{ \AA}$) is blocked by missing UI parameter controls, absent GitHub Actions dispatch infrastructure, and a lack of 1D radial potential handling for diatomic van der Waals systems.

Physical execution of the authentic xTB binary on the Helium dimer confirms that GFN2-xTB successfully converges to an authentic total electronic energy of $-3.4862727\text{ }E_h$, HOMO-LUMO gap of $21.7004\text{ eV}$, and gradient norm of $0.00000835\text{ }E_h/\alpha$. However, this calculation cannot be configured or dispatched by a student through the TORQ interface.

---

## 2. Root Cause Analysis (5 Whys)
1. **Why did the calculation not complete in GitHub Actions?**  
   The student workflow halted because the TORQ interface did not dispatch the physical GFN2-xTB calculation payload to the GitHub Actions runner.
2. **Why was the calculation not dispatched?**  
   The primary autocratic UI (`UI/Start_TORQ.ipynb`) and controller lack interactive parameter controls for Interaction Environment, Calculation Environment, Engine (`xTB`), and Method (`GFN2-xTB`), and `CoChem-TORQ/.github/workflows/` contains zero calculation runner workflow files.
3. **Why are the required parameters unconfigurable in the UI?**  
   `UI/Start_TORQ.ipynb` hardcodes the quantum configuration in Cell 12 to `TorqRunParams(engine="ORCA", method="wB97M-V", ...)`, restricts `PRESET_GEOMETRIES` in Cell 4 to four covalently bound molecules, and does not provide dropdowns or event listeners for engine, method, or environment selection.
4. **Why does the pipeline fail to process the Helium dimer if manually injected?**  
   TORQ's topological analysis and nuclear solver assume polyatomic systems with rotatable internal dihedral angles. For diatomic noble gas complexes ($\text{He}_2$), `TorqTopology` detects 0 rotatable dihedrals, but Phase 4/7 assumes a periodic angular potential $V(\theta)$ and lacks a 1D radial dissociation coordinate solver $V(R)$ for van der Waals complexes.
5. **Why does this architectural gap persist (5th Why)?**  
   The TORQ UI and execution architecture lack a decoupled headless driver, missing REST/token bindings to GitHub Actions workflow dispatch endpoints, and lack an adaptive 1D radial dissociation pathway in the DVR nuclear solver for noble gas dimers.

---

## 3. Reproduction Workflow
Executing the sequence expected of a student in a fresh Codespaces container:
```bash
# 1. Enter repository in GitHub Codespaces
cd /workspaces/CoChem-TORQ

# 2. Attempt headless UI parameter execution with Helium dimer
python -m torq.ui \
    --interact-env "CODESPACES" \
    --calc-env "GITHUB_ACTIONS" \
    --engine "xTB" \
    --method "GFN2-xTB" \
    --geometry "He 0.0 0.0 0.0; He 0.0 0.0 3.0"
# Failure: Command unavailable, UI is an autocratic notebook with hardcoded ORCA/wB97M-V

# 3. Attempt remote Actions dispatch
gh workflow run xtb_calculation.yml -f geometry="he2.xyz" -f method="GFN2-xTB"
# Failure: Workflow xtb_calculation.yml not found in .github/workflows/
```

---

## 4. Verification Evidence & Authentic Physical Metrics
A dedicated zero-mock physical test suite was implemented and verified in `tests/test_torq_gui_xtb_gfn2_student_journey.py`:
- **Helium Dynamic Mass**: Verified via `mendeleev.element('He')` (Mass = 4.002602 u, Z = 2) [M].
- **UI Parameter Defect**: Verified absence of `interact_env`, `calc_env`, `engine`, and `method` controls in `Start_TORQ.ipynb`.
- **Actions Workflow Defect**: Verified that `.github/workflows` has 0 calculation dispatch workflows.
- **Physical xTB Binary Execution**: Physical execution of `C:\ORCA_6.1.1\xtb-6.7.1pre\xtb.exe` on $\text{He}_2$ ($R = 3.0\text{ \AA}$) yielded:
  - Total Energy: $-3.486272739025\text{ }E_h$
  - Gradient Norm: $0.000008353755\text{ }E_h/\alpha$
  - HOMO-LUMO Gap: $21.700410192938\text{ eV}$
  - Dipole: $0.000\text{ Debye}$
  - Exit code: 0 (Normal termination of xTB).
- **Topology Dihedral Defect**: Confirmed `TorqTopology` returns 0 dihedrals for $\text{He}_2$ while detecting a weak complex with 2 components.

---

## 5. SRS Remediation Directives
The SRS Auto-Remediation State Machine is instructed to implement the following architectural enhancements:
1. **Interactive UI Parameter Controls**: Expand `UI/Start_TORQ.ipynb` (Phase 2 & Phase 4) to include interactive dropdown widgets for `Interaction Environment` (Codespaces, Local Windows, Local Linux, macOS, HPC), `Calculation Environment` (Local Host, GitHub Actions, HPC Slurm), `Engine` (ORCA, xTB, PySCF, CFOUR), and `Method` (GFN2-xTB, wB97M-V, B3LYP-D4, etc.).
2. **Helium Dimer Preset Integration**: Add the Helium dimer ($\text{He}_2$, $R = 2.963392\text{ \AA}$ equilibrium / $R = 3.000\text{ \AA}$) into `PRESET_GEOMETRIES` with dynamic Mendeleev masses.
3. **GitHub Actions Dispatch Workflow**: Author `.github/workflows/xtb_calculation.yml` with `workflow_dispatch` trigger to execute GFN2-xTB on ubuntu-latest runners using the official `conda-forge::xtb` package.
4. **1D Radial van der Waals Dissociation Solver**: Provide an adaptive branch in `RelaxedPESTorsionalDVR` to support radial dissociation coordinates $V(R)$ for diatomic / noble gas complexes lacking internal dihedral angles.
5. **Headless CLI Test Client**: Implement `scripts/torq_headless_client.py` allowing CI/CD runners and Codespaces containers to execute student workflows programmatically without requiring manual browser widget clicks.
