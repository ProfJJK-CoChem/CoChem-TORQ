# CoChem-TORQ: Software Requirements Specification (True Architecture v2.1)

> [ML safeguards implementation contract](../ML_SAFEGUARDS.md) supersedes the
> ML-only rejection/cutoff and uncalibrated allocation proposals below. ML may
> prioritize and defer reversibly, with a retained pool and QC sentinel audits;
> it cannot establish pathway absence. Unknown/OOD evidence escalates to QC.

**Governing Charters:** Method Matrix v4.2, CoChem Anti-Spoofing Protocol v4.1.1, PMBOK Guide 7th Ed.
**Status:** Deeply contextualized, expanded, and strictly aligned with physical workflows.

---

## 1. Ecosystem Context & Boundary Definitions

### 1.1 Purpose & Role in CoChem
`CoChem-TORQ` acts as the definitive downstream quantum dynamics and precision electronic structure engine. While upstream modules (like `CoChem-TOPOS`) handle initial structural guesses, conformer enumeration (GOAT/CREST), and topology generation, `TORQ` is responsible for rigorously evaluating Potential Energy Surfaces (PES), performing high-level stationary point optimizations, executing real-space/NBO analyses, and ultimately synthesizing microwave spectral observables (rotational constants, dipole moments) with sub-chemical accuracy. 

### 1.2 Why these boundaries matter
Functional requirements in TORQ are not arbitrary code flags; they directly dictate the physical accuracy of the simulated spectra. If TORQ is fed a geometry by TOPOS, TORQ must determine exactly *how much computational budget* to spend on refining it, *which engine* is legally and technically allowed to process it, and *how* to extract the resulting physics into a format a human spectroscopist can use.

---

## 2. The 11-Tier Calculation Engine & The 3 Core Pathways

### 2.1 The 11-Tier Method Matrix Scaling (The 11-Arrow Pipeline)
TORQ must implement dynamic scaling across the 11 theoretical tiers defined by the global Method Matrix—ranging from ultra-fast Machine Learning Force Fields (MLFF) at the lowest tier, up to canonical CCSD(T) extrapolations at the highest tier. 
*   **User Time-Estimation & Selection:** Before executing a heavy job, TORQ MUST calculate and present the user with an estimated Time-to-Completion for each applicable tier. The user explicitly selects which tier they wish to execute.

### 2.2 The 3-Pathway Licensing & Execution Model
To prevent dependency lock-in and ensure accessibility, TORQ MUST implement three distinct execution pathways that the user can seamlessly toggle between based on their local software licenses:

1.  **The Open-Source Pathway (Extending to revDSD):** Uses strictly open-source, non-licensed dependencies. Crucially, this tier does NOT stop at low-level methods. It extends up to advanced **revDSD double-hybrid methods**. Because open-source engines (e.g., PySCF, Psi4) may lack native analytic support for these exact parameterizations, TORQ MUST perform internal mathematical hybridization (mixing exact exchange, MP2 correlation, and dispersion coefficients) natively within its own Python layer.
2.  **The ORCA Pathway (Intermediate):** Utilizes ORCA for high-efficiency DFT, DLPNO-CCSD(T), and intermediate-tier calculations. Requires standard academic licensing.
3.  **The CFOUR Pathway (Restrictive / Highest Tier):** The most restrictively licensed path, but absolutely required for reaching the pinnacle of the 11 tiers (analytic CCSD(T) VPT2, exact anharmonic force fields, spin-rotation tensors).

### 2.3 CPU vs. GPU Hardware Brokering
TORQ MUST poll the host system (via `cochem_system_config.json`) to dynamically allocate hardware resources:
*   **GPU Routing:** MLFF evaluations, tensor contractions, and HDF5 `SpycFit` spectra generation MUST be routed to available CUDA/ROCm graphics cards to bypass CPU bottlenecks.
*   **CPU Routing:** Traditional *ab initio* integrators (HF, MP2, CCSD) MUST be routed to CPU MPI pools. TORQ must inject exact `%maxcore` and `%pal nprocs` memory allocations into the input files, reserving a strict safety buffer for the host OS.

---

## 3. Potential Energy Surfaces (PES) & Cascading Optimizations

### 3.1 Context: Interconversion and Flexible Rotors
For molecules with flexible internal rotors or van der Waals complexes, understanding the energy barriers between conformers/isomers is critical. 

### 3.2 Human-in-the-Loop (HITL) Isomer Interchange Detection
Before launching a massive grid scan, TORQ MUST perform a topological survey to detect possible isomer interchanges (e.g., rotating dihedrals, floppy bonds, ring-puckering angles). 
*   **HITL Selection:** The system MUST present these detected coordinates to the user via the Interaction Environment UI. The user explicitly selects which angles and bonds they wish to actually investigate. Autonomous "blind" scanning of all dihedrals is strictly forbidden to prevent combinatorial explosions.

### 3.3 Dynamic Cascading Calculation Method (The "Expand Out" Protocol)
Once the user selects the coordinates, TORQ implements a smart cascading strategy:
1.  **Machine Learning PES Mapping:** Generate a broad 1D/2D PES map of the selected coordinates using ultra-fast MLFFs.
2.  **Extract Key Points:** Identify the critical minima and transition states from the MLFF map.
3.  **Tier Escalation:** Re-optimize these specific key points at the next higher basis set and method tier.
4.  **Dynamic Expansion:** Align to these higher-data key points and expand calculations outward along the PES grid.
5.  **Convergence Threshold:** This expansion continues upward through the 11 tiers *only* until there is "no significant improvement" observed in the geometry/energy when moving to the next higher basis set/method.

### 3.4 Energy Diagrams
The output of this cascading sequence MUST be synthesized into visual **Energy Diagrams**, explicitly mapping the physical barriers and interconversion kinetics between isomers and conformers for the user.

---

## 4. Symmetry & Advanced Chemical Analysis

### 4.1 Human-in-the-Loop (HITL) Symmetry Assignment
Molecular symmetry dictates computational cost and spectroscopic selection rules. While TORQ may algorithmically suggest a Point Group (e.g., $C_s, C_{2v}$) or a Permutation-Inversion (PI) Group (e.g., $G_6, G_{12}$), it MUST pause execution and present the suggestion to the user. **A human MUST explicitly confirm or manually override the symmetry assignment before high-tier calculations begin.**

### 4.2 Natural Bond Orbital (NBO) Analysis
To understand hyperconjugation and donor-acceptor interactions at stationary points (especially in non-covalent complexes), TORQ MUST include a functional trigger to execute and parse NBO analysis.

### 4.3 Real Space Analysis
The engine MUST support real-space physical electron density mapping (e.g., QTAIM, ELF, or NCI plots) to visually and mathematically classify the nature of the bonding (covalent vs. vdW) at the key points discovered on the PES.

---

## 5. Dual-Bundling Spectroscopic Artifacts

### 5.1 Dual Export Architecture
TORQ MUST produce two parallel artifact bundles for downstream use:
1.  **Legacy SPCAT / SPFIT:** Generation of standard Pickett `.var` and `.int` files. These files MUST be rigorously scrubbed of any `[E]` (Estimated/MLFF) data to ensure that publication-facing legacy formats only contain anchored, *ab initio* values.
2.  **SpycFit HDF5 Integration:** Delivery of high-dimensional tensor data directly into the `CoChem-BASE` HDF5 bus. This feeds the **SpycFit** ecosystem, which utilizes graphics-card-based (GPU) rapid spectra prediction paired with a human-in-the-loop interactive UI for spectral assignment.

---

## 6. Matrix of Interaction & Calculation Environments

### 6.1 Interaction Environments (The UI Layer)
Where the user interfaces with the workflow, views energy diagrams, confirms symmetry, and makes tier selections:
*   **Jupyter Interactive (Local/WSL2):** Rich, step-by-step interactive notebooks.
*   **GitHub Codespaces:** Cloud-based interactive container environments.
*   **Headless/Voila UI:** Code-blind graphical wrappers for non-programmers.

### 6.2 Calculation Environments (The Compute Layer)
Where the TORQ physics engine actually dispatches the 11-Tier jobs:
*   **Local Host:** Execution on the user's local CPU/GPU (ideal for Open-Source/MLFF pathways).
*   **GitHub Actions:** CI/CD runners for automated benchmark evaluation.
*   **HPC / SLURM Clusters:** Remote distributed execution for Tier 11 (CFOUR/ORCA) scaling.
