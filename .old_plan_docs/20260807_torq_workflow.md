# CoChem-TORQ Master Execution Workflow (2026-08-07)

This document maps the complete, step-by-step execution workflow for the `CoChem-TORQ` module. This workflow transforms TORQ from a simple classical scanner into an advanced quantum mechanical Large Amplitude Motion (LAM) and rovibrational engine, strictly adhering to the *Method Matrix* and the *20260807 Architectural Changes*.

## Stage 1: Pipeline Handoff & Memory Assimilation
1. **TOPOS Handoff:** `CoChem-TOPOS` concludes its execution and relinquishes the lock on `cochem_state.h5`. 
2. **Daemon Verification:** TORQ initializes and pings the `CoChem-SCRIBE` background daemon to confirm that memory-aware archiving is active.
3. **Hessian Assimilation:** TORQ maps the HDF5 tensor into memory. Crucially, it extracts the globally optimized geometries and the **pre-calculated Force Constant Matrices (Hessians)**. This completely bypasses the need for TORQ to run redundant initial frequency calculations.
4. **Trajectory Slicing:** If TOPOS ran AIMD, TORQ extracts the trajectory and slices it into 100 discrete geometric frames for downstream thermal averaging.

## Stage 2: The LAM Decision Gate
1. **NCI/BSSE Flag Check:** TORQ queries the HDF5 metadata for each specific isomer or complex.
2. **Logic Branching:**
   - Does the metadata contain the `LAM_TRIGGER_REQUIRED = TRUE` flag? (Typically set for weak vdW complexes with $E_{int} < 5$ kcal/mol).
   - **If FALSE:** The molecule is considered a "Rigid Rotor". TORQ routes execution to **Stage 3A (VPT2)**.
   - **If TRUE:** The harmonic approximation is mathematically invalid. TORQ immediately routes execution to **Stage 3B (Sinc-DVR)**.

## Stage 3A: Rigid Rotor Pathway (VPT2 & Coriolis)
1. **ORCA Escalation:** TORQ utilizes the assimilated Hessian to launch a direct VPT2 calculation in ORCA (appending `ExtremeSCF`).
2. **Tensor Extraction:** The `cochem_tensor_extractor` dynamically parses the ORCA `%vib` output, extracting:
   - Harmonic Frequencies and Infrared Intensities
   - Darling-Dennison Resonances
   - Coriolis Coupling Matrices ($x, y, z$)
   - Quartic Centrifugal Distortion Constants ($D_J, D_{JK}, D_K, d_1, d_2$)
3. **Divergence Guardrail:** 
   - The extractor rigorously checks the Centrifugal Distortion Constants. 
   - **Decision Gate:** Are any constants unphysically negative or mathematically divergent?
   - *Yes:* This proves the potential energy surface is too flat for VPT2. TORQ issues a severe red UI warning, aborts the Rigid Rotor pathway, and forces the molecule into the **Stage 3B (Sinc-DVR)** protocol.
4. **Raman Broadening:** The CP-SCF solver is triggered to extract Raman polarizability derivatives.

## Stage 3B: Large Amplitude Motion Pathway (Sinc-DVR)
1. **Multidimensional Grid Generation:** `cochem_torq_grid.py` abandons classical continuous rotation. Instead, it generates a highly discretized multidimensional mesh mapping the Large Amplitude Motion (e.g., internal rotation or umbrella inversion).
2. **Constrained Monomer Relaxation:** At every discrete coordinate on the mesh, ORCA executes a constrained geometry optimization. The intermolecular LAM coordinates (e.g., center-of-mass distance $R$ and Euler angles) are mathematically frozen, while the strong intramolecular covalent bonds of the constituent monomers are allowed to fully relax. This captures the structural deformation energy missed by rigid approximations.
3. **Colbert-Miller Hamiltonian Construction:** TORQ converts the discrete classical energies into a kinetic and potential $N$-dimensional Hamiltonian matrix using the Colbert-Miller Sinc-DVR methodology.
4. **Wavefunction Diagonalization:** The Hamiltonian is diagonalized to extract:
   - The true 3D probability wavefunctions (eigenvectors).
   - The A/E tunneling splitting energy levels (eigenvalues).
5. **Kraitchman Mapping:** TORQ evaluates the expectation values of the wavefunctions to calculate the theoretical $r_s$ (substitution) coordinates over the rigid $r_0$ structure for downstream isotopic fitting.

## Stage 4A: Photochemical Dynamics & UV-Vis (Optional)
1. **Vertical Excitation:** If triggered, TORQ executes `STEOM-CCSD` (or TD-DFT for lower tiers) to extract precise vertical excitation energies and Natural Transition Orbitals (NTOs).
2. **Surface Hopping:** TORQ launches non-adiabatic surface hopping trajectories to map the conical intersections, tracking the charge-transfer states over time.
3. **Tensor Handoff:** The UV-Vis eigenvalues and oscillator strengths are flushed to the HDF5 tensor for downstream SpycFit convolution.

## Stage 4B: Mass Spectrometry (QCxMS) (Optional)
1. **Ionization Scaling:** If GC-MS/EI-MS simulation is requested, TORQ calculates the vertical ionization potential of the global minimum.
2. **Trajectory Ensemble:** Utilizing the `QCxMS` engine, TORQ launches a swarm of high-kinetic-energy trajectories (10 to 1,000 depending strictly on the Time-Tier) to simulate the 70 eV electron impact.
3. **Fragmentation Serialization:** The mass-to-charge ($m/z$) fragment abundances are tabulated over time and serialized to `cochem_state.h5`.

## Stage 4: Final Serialization & SpycFit Handoff
1. **Data Aggregation:** The final VPT2 resonance matrices, Sinc-DVR tunneling splittings, 3D wavefunctions, Kraitchman structures, and Thermally Averaged NMR shifts are compiled.
2. **HDF5 Commit:** TORQ flushes the massive data blocks to the `cochem_state.h5` tensor.
3. **Memory-Aware Zstandard Compression:** The massive DVR multidimensional grids trigger the `CoChem-SCRIBE` background daemon to engage `tar.zst` (Zstandard) chunking. SCRIBE compresses the grids in memory-aware 500MB batches, safely archiving the data to the SSD without blocking I/O or crashing the Jupyter kernel.
4. **Pipeline Handoff:** TORQ cleanly terminates its subprocesses. A UI prompt informs the user that Phase 3 is complete and directs them to launch the `CoChem-SpycFit` notebook for the final Bayesian spectral fitting and Active Learning assignment loop.
