---
id: PROB-TORQ-UI-CODESPACES-ACTIONS-XTB-GFN2-HE2-001
title: Physical Failure of Student UI Micro-Task in TORQ under GitHub Codespaces with GitHub Actions Engine (xTB / GFN2-xTB on He2)
category: ui_test_failure
severity: CRITICAL
status: OPEN
date_logged: 2026-09-20T00:20:00-05:00
reporter: cochem-audit (Kanban Automated Worker)
target_repository: TORQ (CoChem-TORQ)
target_system: He2 van der Waals complex
target_engine: xTB
target_method: GFN2-xTB
interaction_environment: GitHub Codespaces
calculation_environment: github-actions
provenance: [M]
---

# Problem Report: TORQ Student UI Workflow Failure (GitHub Codespaces + github-actions + xTB / GFN2-xTB)

## 1. Executive Summary & Assigned Micro-Task Specification

Pursuant to the CoChem Swarm Zero-Trust Charter, Anti-Spoofing Protocol v4 (§1 Asymmetric Verification, §3 Zero Mocks or Stub Logic, §7 Counterfeit Compliance, §8 Semantic Spoofing Ban, §13 Data Laundering Ban), and the Mendeleev Dynamic Mass Mandate (`cochem-mendeleev-masses.md`), an authentic, zero-mock physical audit was executed on the **TORQ** repository (`D:/__CoChem/GitHub-Repo/CoChem-TORQ`).

The assigned undergraduate student UI micro-task specification requires:
- **Repository**: `TORQ` (`CoChem-TORQ`, located at `D:/__CoChem/GitHub-Repo/CoChem-TORQ`)
- **Target UI File**: `UI/Start_TORQ.ipynb` (Master Autocratic DOM Orchestrator) and `UI/cochem_torq_controller.py`
- **Interaction Environment**: `GitHub Codespaces`
- **Calculation Environment**: `github-actions`
- **Computational Engine**: `xTB`
- **Quantum Method**: `GFN2-xTB` (Extended Tight-Binding Semiempirical with D4 Dispersion)
- **Chemical Target**: Helium dimer ($\text{He}_2$) van der Waals complex ($R = 3.000\ \text{Å}$)
- **Mass Provenance**: Dynamic query via `mendeleev` Python library (`mendeleev.element('He').mass = 4.002602\ \text{Da}`, $Z=2$) per `cochem-mendeleev-masses.md`.

### Physical Chemistry Benchmark Constants [M]
- **Species**: Helium dimer ($\text{He}_2$) van der Waals complex
- **Interatomic Separation ($R$)**: $3.000\ \text{Å}$ (Equilibrium $R_e \approx 2.97\ \text{Å}$ / $5.61\ \text{Bohr}$)
- **Atomic Mass ($^4\text{He}$)**: $4.002602\ \text{Da}$ dynamically evaluated via `mendeleev`
- **Authentic GFN2-xTB Total Energy**: $-3.486272739025\ \text{E}_{\text{h}}$
- **Authentic GFN2-xTB Gradient Norm**: $0.000008353755\ \text{E}_{\text{h}}/\text{a}_0$
- **Authentic GFN2-xTB HOMO-LUMO Gap**: $21.700410192030\ \text{eV}$
- **Dipole**: $0.000\ \text{Debye}$
- **Equilibrium Rotational Constants**: $A = \infty$, $B = C = 28058.3508\ \text{MHz}$ (Linear Diatomic Rotor, $I_a = 0$, $I_b = I_c = 18.012\ \text{u}\cdot\text{Å}^2$)

---

## 2. Experimental Execution & Forensic Findings

Simulating the exact sequence of actions and commands an undergraduate student executes when launching and driving the interactive repository UI:

```python
import sys, os, json
from pathlib import Path

# 1. Enter repository in GitHub Codespaces
repo_root = Path(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ")
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "Libraries"))
sys.path.insert(0, str(repo_root / "UI"))

# 2. Ingest Helium Dimer van der Waals complex payload
he2_payload = {
    "symbols": ["He", "He"],
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0]
    ],
    "charge": 0,
    "multiplicity": 1
}

# 3. Initialize Controller & Validate Payload
from Libraries.cochem_torq_pipeline import normalize_and_validate_payload
from UI.cochem_torq_controller import TORQPipelineController

controller = TORQPipelineController()
syms, coords = normalize_and_validate_payload(he2_payload)
geom_hash = controller.load_preset("He-He van der Waals complex", syms, coords)
```

### Forensic Defect Ledger

| Parameter | Required Specification | Observed Implementation | Statutory Status |
|---|---|---|---|
| **Target Repository** | `TORQ` | `D:/__CoChem/GitHub-Repo/CoChem-TORQ` | **PASSED** (Boundary maintained) |
| **Interaction Env Widget** | `GitHub Codespaces` | `interact_env_dropdown` completely absent in `Start_TORQ.ipynb` | **FAIL**: No environment selector |
| **Calculation Env Widget** | `github-actions` | `calc_env_dropdown` completely absent in `Start_TORQ.ipynb` | **FAIL**: No execution target selector |
| **Engine Selector** | `xTB` | Hardcoded `engine="ORCA"` in Cell 12; no widget | **FAIL**: Engine hardcoded |
| **Method Selector** | `GFN2-xTB` | Hardcoded `method="wB97M-V"` in Cell 12; no widget | **FAIL**: Method hardcoded |
| **Chemical Presets** | $\text{He}_2$ van der Waals | `PRESET_GEOMETRIES` only contains $\text{H}_2\text{O}_2$, $(\text{H}_2\text{O})_2$, $\text{N}_2\text{H}_4$, $\text{CH}_3\text{OH}$ | **FAIL**: $\text{He}_2$ missing from presets |
| **Remote Actions Dispatch** | `github-actions` trigger | `.github/workflows/` directory contains 0 YAML files | **FAIL**: Zero remote CI dispatch infrastructure |
| **Mass Provenance** | `mendeleev` | Dynamic query `mendeleev.element('He').mass` | **PASSED**: 4.002602 Da verified |
| **Physical Binary Execution** | Genuine xTB executable | `C:\ORCA_6.1.1\xtb-6.7.1pre\xtb.exe he2.xyz --gfn 2` | **PASSED**: Total Energy -3.486273 Eh, Gap 21.700 eV |
| **Topology Analysis** | Dihedral Detection | `TorqTopology` returns `dihedrals = []` | **PASSED**: Linear diatomic 0 dihedrals |
| **Nuclear DVR Solver** | Radial 1D DVR | Cell 14 assumes angular scan $\theta \in [0, 360^\circ]$ for $\text{H}_2\text{O}_2$ | **FAIL**: Lacks radial $V(R)$ solver |
| **Catalog Compilation** | Asymmetric Top | Cell 18 calls `RotationalConstants(a_mhz=...)` | **FAIL**: Raises `TypeError` on unexpected kwarg |

---

## 3. Observed Failure Telemetry & Raw Tracebacks

### Physical xTB Execution Output (Authentic Quantum Chemistry Telemetry):
```text
   -----------------------------------------------------------
   |                   =====================                 |
   |                         x T B                           |
   |                   =====================                 |
   |                          S. Grimme                      |
   |          Mulliken Center for Theoretical Chemistry      |
   |                    University of Bonn                   |
   -----------------------------------------------------------
   * xtb version 6.7.1pre (5071a88) compiled by 'Marcel@Raven' on 2024-07-23
   * GFN2-xTB (w/o D4)
   * Calculation setup:
     charge                :   0
     spin                  :   0
   -------------------------------------------------
   | TOTAL ENERGY               -3.486272739025 Eh   |
   | GRADIENT NORM               0.000008353755 Eh/α |
   | HOMO-LUMO GAP              21.700410192030 eV   |
   -------------------------------------------------
   * finished run on 2026/09/20 at 00:19:44.225
   normal termination of xtb
```

### Student UI Inspection Traceback (Start_TORQ.ipynb):
```text
=== STEP 1: Repository UI Inspection ===
Loaded Start_TORQ.ipynb (21 cells)
Checking for interaction environment widget (GitHub Codespaces): False
Checking for calculation environment widget (github-actions): False
Checking for engine widget (xTB): False
Checking for method widget (GFN2-xTB): False
Checking for He-He preset: False
```

### Cell 18 Spectroscopic Line Catalog TypeError:
```text
Traceback (most recent call last):
  File "Start_TORQ.ipynb", cell 18, line 11
    consts = RotationalConstants(
        a_mhz=A_mhz,
        b_mhz=B_mhz,
        c_mhz=C_mhz,
        dj_khz=2.5,
        djk_khz=-15.0,
        dk_khz=75.0,
        delta_j_khz=0.5,
        delta_k_khz=10.0,
        mu_a_debye=0.0,
        mu_b_debye=0.0,
        mu_c_debye=0.0
    )
TypeError: RotationalConstants.__init__() got an unexpected keyword argument 'a_mhz'
```

### Pytest Verification Log (`tests/test_torq_gui_xtb_gfn2_student_journey.py`):
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: D:\__CoChem\GitHub-Repo\CoChem-TORQ
configfile: pytest.ini
collected 6 items

tests/test_torq_gui_xtb_gfn2_student_journey.py::test_mendeleev_helium_mass_provenance PASSED [ 16%]
tests/test_torq_gui_xtb_gfn2_student_journey.py::test_torq_ui_environment_and_engine_controls_defect PASSED [ 33%]
tests/test_torq_gui_xtb_gfn2_student_journey.py::test_torq_github_actions_workflow_missing_defect PASSED [ 50%]
tests/test_torq_gui_xtb_gfn2_student_journey.py::test_physical_he2_xtb_gfn2_calculation_execution PASSED [ 66%]
tests/test_torq_gui_xtb_gfn2_student_journey.py::test_torq_torsional_dvr_defect_on_he2 PASSED [ 83%]
tests/test_torq_gui_xtb_gfn2_student_journey.py::test_torq_rotational_constants_init_defect PASSED [100%]

============================== 6 passed in 2.61s ==============================
```

---

## 4. Root Cause Analysis (5 Whys)

1. **Why could the student not complete the calculation via GitHub Actions?**  
   The calculation was never dispatched to GitHub Actions because `.github/workflows/` contains 0 workflow dispatch files, and the UI lacks any remote dispatch client.
2. **Why was the calculation not dispatched to xTB / GFN2-xTB in the UI?**  
   `UI/Start_TORQ.ipynb` Cell 12 hardcodes `TorqRunParams(engine="ORCA", method="wB97M-V", basis_set="def2-TZVPP", ...)` without interactive widgets for engine, method, or environment selection.
3. **Why did the student have to manually type the Helium dimer coordinates?**  
   `PRESET_GEOMETRIES` in Cell 4 only provides presets for $\text{H}_2\text{O}_2$, $(\text{H}_2\text{O})_2$, $\text{N}_2\text{H}_4$, and $\text{CH}_3\text{OH}$. The Helium dimer van der Waals benchmark complex is omitted.
4. **Why does downstream nuclear solver and line catalog generation fail for He-He?**  
   TORQ's DVR solver in Cell 14 assumes polyatomic molecules with periodic internal dihedrals $\theta \in [0, 360^\circ]$ and hardcodes $\text{H}_2\text{O}_2$ torsional barrier formulas. It lacks an adaptive 1D radial dissociation coordinate solver $V(R)$ for diatomic van der Waals complexes. Furthermore, Cell 18 invokes `RotationalConstants` with erroneous keyword arguments (`a_mhz` instead of `A`).
5. **Why does this architectural gap persist (5th Why)?**  
   The TORQ UI was developed primarily as an autocratic notebook tailored specifically for polyatomic asymmetric top rotors (Recipe R2) rather than a decoupled, generalized quantum chemistry workbench capable of dispatching calculations across heterogeneous environments (Codespaces -> GitHub Actions) and diverse molecular topologies (diatomic van der Waals dimers to flexible polyatomics).

---

## 5. Directives for SRS Pipeline Auto-Remediation

The SRS Auto-Remediation State Machine is instructed to implement the following architectural repairs:

1. **UI Widget Expansion (`UI/Start_TORQ.ipynb`)**:
   - Add `interact_env_dropdown` with choices: `['GitHub Codespaces', 'Local Workstation', 'HPC Slurm']`.
   - Add `calc_env_dropdown` with choices: `['Local Host', 'github-actions', 'HPC Slurm Array']`.
   - Add `engine_dropdown` with choices: `['xTB', 'ORCA', 'PySCF', 'CFOUR']`.
   - Add `method_dropdown` dynamically populated based on engine (e.g. for `xTB`: `['GFN2-xTB', 'GFN1-xTB', 'GFN-FF']`; for `ORCA`: `['wB97M-V', 'r2SCAN-3c', 'B3LYP-D4']`).
   - Bind `TorqRunParams` in Cell 12 to the selected UI widgets rather than static hardcoding.

2. **Benchmark Preset Integration**:
   - Add `Helium Dimer (He2 van der Waals)` to `PRESET_GEOMETRIES` in Cell 4:
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

3. **GitHub Actions Workflow Dispatch Bridge**:
   - Create `.github/workflows/cochem_xtb_runner.yml` supporting `workflow_dispatch` with inputs: `geometry_xyz`, `charge`, `multiplicity`, `method`.
   - Install `conda-forge::xtb` on `ubuntu-latest` and publish JSON/output artifacts back to the repository / artifact tier.

4. **Adaptive Radial 1D DVR Nuclear Solver**:
   - Enhance `RelaxedPESTorsionalDVR` to detect diatomic van der Waals complexes ($N_{\text{dihedrals}} = 0$, $N_{\text{atoms}} = 2$) and route execution to a 1D radial dissociation coordinate solver $V(R)$ across $R \in [2.0\ \text{Å}, 6.0\ \text{Å}]$.

5. **Fix Cell 18 Spectroscopic Catalog Call**:
   - Correct the instantiation of `RotationalConstants` in Cell 18:
     ```python
     consts = RotationalConstants(
         A=A_mhz,
         B=B_mhz,
         C=C_mhz,
         D_J=2.5e-3,
         D_JK=-15.0e-3,
         D_K=75.0e-3,
         d_1=0.5e-3,
         d_2=10.0e-3,
         mu_a=0.0,
         mu_b=0.0,
         mu_c=0.0
     )
     ```
   - For linear molecules ($A = \infty$), implement a dedicated linear rotor spectroscopic catalog branch ($E(J) = B \cdot J(J+1) - D_J \cdot [J(J+1)]^2$).
