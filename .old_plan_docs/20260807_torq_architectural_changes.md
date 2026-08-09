# CoChem-TORQ Architectural Changes (2026-08-07)

This document outlines the requisite architectural refactoring for the `CoChem-TORQ` module. These changes are designed to strictly implement the advanced quantum mechanics described in *Improving LAM Microwave Predictions* and bridge the module cleanly to the *100 Suggestions Gap Analysis*. Implementing these changes is mandatory before generating the executable TORQ workflow.

## 1. LAM Ingestion & HDF5 Hessian Assimilation
**Target Files:** `Libraries/cochem_torq_grid.py` and `Libraries/cochem_tensor_extractor.py`

**Current State:** 
TORQ currently builds its own classical 1D/2D rigid torsional meshes via `networkx` and prepares to execute its own ORCA geometry optimizations and frequency calculations from scratch.

**Required Architectural Change:**
- **HDF5 Handshake & Hessian Assimilation:** TORQ must hook into `cochem_state.h5` (populated by TOPOS). It must extract the pre-calculated **Force Constant Matrices (Hessians)**. Running redundant `Freq` calculations on the global minimum is computationally wasteful and must be strictly blocked if the Hessian already exists in the HDF5 tensor.
- **The LAM Trigger Response:** The ingestion parser must scan the HDF5 registry for the `LAM_TRIGGER_REQUIRED` boolean flag (set by TOPOS for weak vdW complexes with $E_{int} < 5$ kcal/mol).
- **Decision Gate:** If the LAM flag is `TRUE`, TORQ must bypass the classical harmonic VPT2 subroutine and immediately initialize the Multi-Dimensional Large Amplitude Motion (Sinc-DVR) protocols.

## 2. Colbert-Miller Sinc-DVR Integration
**Target File:** `Libraries/cochem_torq_grid.py`

**Current State:** 
The grid generator creates standard classical PES meshes ($360^\circ$ scans) using simple Cartesian rotation matrices (Rodrigues' formula).

**Required Architectural Change:**
- **From Classical to Quantum Grids:** `cochem_torq_grid.py` must be radically expanded to support the **Colbert-Miller Sinc-Discrete Variable Representation (DVR)**. Instead of just passing classical points to ORCA, the module must use the ORCA single-point energies to build a full $N$-dimensional DVR Hamiltonian matrix.
- **Wavefunction Resolution:** The module must diagonalize this Hamiltonian to solve for the true 3D probability wavefunctions (eigenvectors) and tunneling splitting energy levels (eigenvalues) of the highly fluxional LAM states.
- **Constrained Monomer Relaxation:** When ORCA evaluates the grid points, it must *not* execute rigid static single points. It must perform a constrained optimization at every discrete coordinate, mathematically freezing the intermolecular LAM coordinates while allowing the strong intramolecular covalent bonds to fully relax, capturing the critical structural deformation energy.
- **Adaptive Sparse-Grid Pruning:** Generating full Sinc-DVR grids is computationally staggering. TORQ must first evaluate grid coordinates with MACE. If the repulsive steric energy is >50 kcal/mol above the minimum, TORQ must skip the expensive ORCA constrained optimization and assign a mathematical potential of infinity to that point.
- **Isotopologue Kraitchman Scaffolding:** For downstream SpycFit analysis, the grid generator must calculate Kraitchman $r_s$ substitution coordinates mapped over the theoretical $r_0$ structure for the LAM wavefunctions.

## 3. Advanced Anharmonicity & Resonance Extraction
**Target Files:** `Libraries/cochem_torq_orca.py` and `Libraries/cochem_tensor_extractor.py`

**Current State:** 
The tensor extractor pulls basic energies and converged geometries but fails to capture the intricate rovibrational coupling constants necessary for modern rotational spectroscopy.

**Required Architectural Change:**
- **VPT2 Matrix Extraction:** If the molecule is rigid (LAM flag is `FALSE`), TORQ proceeds with standard VPT2 calculations. `cochem_tensor_extractor.py` must be upgraded to parse the ORCA `%vib` block and extract:
  1.  **Darling-Dennison Resonances**
  2.  **Coriolis Coupling Matrices** ($x, y, z$ axes)
  3.  **Centrifugal Distortion Constants** ($D_J, D_{JK}, D_K, d_1, d_2$)
- **Divergent Distortion Flagging:** If the extractor detects mathematically divergent or unphysical negative centrifugal distortion constants (a common failure of VPT2 in flat potentials), it must halt execution, throw a severe warning, and retroactively switch the molecule into the LAM/DVR protocol.
- **Raman Polarizability:** Ensure extraction of Raman polarizability derivatives from the CP-SCF solver for collision-induced broadening profiles.

## 4. Thermally Averaged NMR Assimilation
**Target File:** `Libraries/cochem_tensor_extractor.py`

**Current State:** 
NMR routines are completely missing.

**Required Architectural Change:**
- **Trajectory Slicing:** If TOPOS ran AIMD conformational sampling, TORQ must slice the trajectory into 100 discrete frames.
- **Tensor Averaging:** TORQ executes GIAO NMR shielding tensor calculations across all 100 frames and dynamically calculates the thermally averaged isotropic shielding constants, providing a massive accuracy boost over single-point gas-phase NMR.

## 5. Background Archiving Sync
**Target File:** `Libraries/cochem_torq_export.py`

**Current State:** 
Grid structures and MACE outputs are dumped into flat `.json` files.

**Required Architectural Change:**
- **Memory-Aware Zstandard HDF5:** Deprecate the flat `.json` export. The massive 3D PES grids and DVR Hamiltonians must be streamed into `cochem_state.h5`. 
- **Daemon Sync:** Ensure that `CoChem-SCRIBE` is notified when the massive DVR matrices are written, prompting SCRIBE to compress the multidimensional grids using `tar.zst` (Zstandard) chunking during idle CPU cycles to prevent I/O bottlenecks.

## 6. Mass Spectrometry (QCxMS) Integration
**Target File:** `Libraries/cochem_torq_qcxms.py` (New Subsystem)

**Required Architectural Change:**
- **Non-Equilibrium Dynamics:** TORQ must integrate the `QCxMS` software to simulate 70 eV Electron Ionization (EI-MS) fragmentation.
- **Trajectory Scaling:** Depending on the selected Time-Tier (from 1 minute to 1 week), TORQ will initialize between 10 and 1,000 non-equilibrium AIMD trajectories on the radical cationic surface, writing the fragment abundance data to the HDF5 tensor for SCRIBE to plot.

## 7. UV-Vis Photochemistry (STEOM-CCSD)
**Target File:** `Libraries/cochem_torq_orca.py`

**Required Architectural Change:**
- **Excited State Execution:** TORQ must orchestrate TD-DFT or `STEOM-CCSD` to calculate exact vertical excitation energies (accurate to 0.03 eV).
- **Surface Hopping:** If photochemical dynamics are triggered, TORQ must execute non-adiabatic surface hopping, tracking the evolution of the Natural Transition Orbitals (NTOs) across the trajectory.

---
**Next Step Readiness:** 
By implementing these structural pillars, `CoChem-TORQ` transforms from a basic rigid-rotor scanner into an advanced, quantum-mechanically rigorous Large Amplitude Motion engine. Once these changes are approved, the exceptionally detailed and complete UI/Execution Workflow for TORQ can be securely generated.
