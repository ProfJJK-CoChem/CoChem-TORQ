# Implementation Gap Analysis and 100 Suggestions for CoChem Methodology Integration

> [!NOTE]
> This document provides a comprehensive evaluation of the `CoChem` repository suite (`CoChem-BASE`, `CoChem-TOPOS`, `CoChem-TORQ`, and `CoChem-SpycFit`) in relation to the state-of-the-art computational guidelines outlined in the **Method Matrix** and **Improving LAM Microwave Predictions** documents.

## Table of Contents
- [Executive Gap Analysis](#executive-gap-analysis)
- [Category 1: CoChem-BASE (Job Routing, Input Generation, & Hardware)](#category-1-cochem-base-job-routing-input-generation--hardware)
- [Category 2: CoChem-TOPOS (Conformer Topology, MD, & NEB)](#category-2-cochem-topos-conformer-topology-md--neb)
- [Category 3: CoChem-TORQ (PES Scans, Wave-Functions, & Anharmonicity)](#category-3-cochem-torq-pes-scans-wave-functions--anharmonicity)
- [Category 4: CoChem-SpycFit (JAX DVR, Spectroscopy, & Active Learning)](#category-4-cochem-spycfit-jax-dvr-spectroscopy--active-learning)
- [Category 5: CoChem-SCRIBE (Academic Publishing, Provenance, & UI Integrations)](#category-5-cochem-scribe-academic-publishing-provenance--ui-integrations)
- [Category 6: Academic Integrity and Scientific Validity](#category-6-academic-integrity-and-scientific-validity)
- [Category 7: User Experience and Didactic Educational Experience](#category-7-user-experience-and-didactic-educational-experience)
- [Category 8: Accuracy Optimization and Comparable Time Costs](#category-8-accuracy-optimization-and-comparable-time-costs)

---

## Executive Gap Analysis

> [!WARNING]
> An analysis of the current codebases reveals significant gaps in adopting the advanced methodologies:
> - **CoChem-TOPOS** uses *mock* JAX-NEB executions instead of true compiled VRAM gradients.
> - **CoChem-TORQ** implements a 1D DVR placeholder, missing the 3D Colbert-Miller Sinc-DVR critical for multi-dimensional Large Amplitude Motion (LAM) quantum smearing.
> - **CoChem-SpycFit** has a high-performance JAX convolver but lacks the advanced Bayesian Parameter Priors and Information-Gain Active Learning algorithms required for autonomous CP-FTMW peak assignment.
> - **CoChem-BASE** provides foundational BSSE and ECP tagging but lacks comprehensive routing for MACE4IR, AIMNet2 fallbacks, and explicit Ab Initio Molecular Dynamics (AIMD) ensemble extraction.

To achieve academic excellence, absolute accuracy, and pedagogical clarity while minimizing computational overhead, the following **100 actionable suggestions** must be implemented. Track your progress using the checkboxes below.

## Category 1: CoChem-BASE (Job Routing, Input Generation, & Hardware)

> [!TIP]
> Focus on establishing robust environments and automated fallbacks across different hardware scales.

- [ ] **Hardware Routing**: Implement a robust detection mechanism for CUDA vs CPU environments in `CoChem-BASE` to dynamically route MACE calculations to CPU/g-xTB on Codespaces as outlined in the Method Matrix.
- [ ] **MLFF Element Fallback**: Add explicit input parsing in `CoChem-BASE` to detect out-of-bounds elements (e.g., transition metals) and trigger the AIMNet2 fallback before MACE-OFF24(m) throws an error.
- [ ] **Dynamic BSSE Overcorrection Flags**: Update the config compiler to automatically suppress the `%geom Counterpoise` block when scaling to the CBS limit or when TightPNO DLPNO-CCSD(T) is detected.
- [ ] **Composite Method CP Awareness**: Ensure that methods inherently parameterized with geometrical counterpoise (e.g., r2SCAN-3c, wB97X-3c) do not redundantly apply external CP flags in the BASE generator.
- [ ] **High-Performance Memory Mappings**: Implement PyArrow IPC shared memory (/dev/shm) integrations for streaming PES grid tensors directly to RAM, bypassing SSD I/O bottlenecks.
- [ ] **Time-Tier Execution Tethers**: Hardcode the 10 temporal wall-clock tiers (10s to 1mo) into the `CoChem-BASE` job manager to forcefully terminate or downscale jobs exceeding their computational budget. These strictly adhere to methods, keywords from the document "D:\GitHub-Repo20360805 Method Matrix"
- [ ] **AutoAux Linear Dependence Handling**: Inject `AutoAux` keywords dynamically when diffuse basis sets (e.g., aug-cc-pVTZ) are employed to prevent basis set linear dependence failures.
- [ ] **RIJCOSX Grid Defaults**: Enforce `DefGrid4` whenever `RIJCOSX` is invoked with double hybrids to prevent grid noise in gradient and Hessian evaluations.
- [ ] **ExtremeSCF for Point Energies**: Guarantee `ExtremeSCF` is appended to all single-point CBS extrapolation scripts in BASE to ensure wave-function convergence.
- [ ] **Automated NTO Extraction**: For TD-DFT and STEOM-CCSD inputs, auto-generate the ORCA blocks required to output Natural Transition Orbitals (NTOs) for active space visualization.
- [ ] **MACE4IR Pipeline Scaffold**: Add input generator templates for MACE4IR foundation model tasks, preparing the backend for dynamic IR generation.
- [ ] **MACE-POLAR-1 Pipeline Scaffold**: Scaffold Raman prediction inputs specific to MACE-POLAR-1 inference.
- [ ] **QCxMS Wrapper Integration**: Build a native wrapper in BASE for QCxMS to automate electron ionization (70 eV) mass spectrometry dynamics.
- [ ] **GPU4PySCF Hook**: Provide an experimental hook to `gpu4pyscf` to accelerate tensor evaluations directly on NVIDIA hardware.
- [ ] **HPC SLURM Script Generator**: Extend BASE to generate dynamic SLURM batch scripts for Multi-Node F12-CCSD(T) and STEOM-CCSD tasks.
## Category 2: CoChem-TOPOS (Conformer Topology, MD, & NEB)

> [!IMPORTANT]
> Focus on transitioning from mocked executions to high-fidelity Machine Learning Force Fields (MLFF) for barrier estimation and Conformational sampling.

- [ ] Detection and selection of monomer, strong complex, and weak complex must be implemented and passed down throughout the rest of the script.
- [ ] **Replace Mock JAX-NEB**: Rewrite the `_execute_jax_neb` function in `cochem_topos_crusher.py` which currently contains a mock calculation, integrating actual MACE-JAX grad and optax minimization.
- [ ] Ensure **Time-Tier Execution Tethers**: Hardcode the 10 temporal wall-clock tiers (10s to 1mo) into the `CoChem-BASE` job manager to forcefully terminate or downscale jobs exceeding their computational budget. These strictly adhere to methods, keywords from the document "D:\GitHub-Repo20360805 Method Matrix" 
- [ ] Ensure that the additional keywords and considerations from "D:\GitHub-Repo20360805 Method Matrix" are applied in the specific sections.
- [ ] isomer detection must include enantiomer classification alongside unique and isomer baskets.
- [ ] Deduplication process must always be automated but with human oversight and a final checkpoint where the human may visually within a GUI group isomers into the duplicate baskets, separate isomers from the duplicate basket. Then script must detect symmetry and allow the user to change the symmetry as an override which is then strictly enforce on future calculations. As monomers are assembled into complexes the user must again go through this process, and again for a final selection.
- [ ] **Global Minimum Stochastic Seeding**: Implement the GOAT (Global Optimizer Algorithm) or comparable stochastic funnel in TOPOS to escape local minima in vdW clusters. This was always meant to be implemented in a looping methodology:
-Each Monomer Fragment of the Complex:
  `EscRm` $\rightarrow$ `Crusher` $\rightarrow$ `GOAT (Diversity)` $\rightarrow$ `Crusher (Interactive)` $\rightarrow$ `GOAT (Normal)` $\rightarrow$ `Crusher (Final)` $\rightarrow$ `HDF5 Serialization`    
  After optimization of the user selected (in the human in the loop deduplication process) monomer fragment isomers. Assemble the monomer isomers into all possible combinations of strong complexes. Then:
  `EscRm` $\rightarrow$ `Crusher` $\rightarrow$ `GOAT (Diversity)` $\rightarrow$ `Crusher (Interactive)` $\rightarrow$ `GOAT (Normal)` $\rightarrow$ `Crusher (Final)` $\rightarrow$ `HDF5 Serialization`
 After optimization of the user selected (in the human in the loop deduplication process)  monomer and/or strong complex fragments. Assemble the monomer and/or strong complex isomers into all possible combinations of weak complexes. Then:
   `EscRm` $\rightarrow$ `Crusher` $\rightarrow$ `GOAT (Diversity)` $\rightarrow$ `Crusher (Interactive)` $\rightarrow$ `GOAT (Normal)` $\rightarrow$ `Crusher (Final)` $\rightarrow$ `HDF5 Serialization`
Maintain the matrices for these isomers for use in accelerating the subsequent complex calculations by passing them off to the next.
- [ ] **Spectroscopic Deduplication**: Replace RMSD-based conformer deduplication in `cochem_topos_crusher.py` with the 'Spectroscopic Override' method (collapsing isomers with identical A, B, C constants within 1.5%).
- [ ] **AIMD Conformational Sampling**: Integrate NVE/NVT ensembles via ASE/LAMMPS in TOPOS for true thermodynamic conformational sampling prior to MLFF triage.
- [ ] **Shake Constraints for AIMD**: Fully map `_apply_shake_constraints` for explicit solvent models ensuring rigid internal water/solvent geometries during vdW solute evolution.
- [ ] **Deperturbed VPT2 (DVPT2)**: Add logic to detect near-degenerate vibrational states and automatically switch from VPT2 to DVPT2 to prevent unphysical vibration-rotation coupling constants.
- [ ] **Constrained Monomer Relaxation**: Implement the 'Rigid Monomer Approximation' module, freezing intramolecular covalent bonds while relaxing intermolecular vectors.
- [ ] **Sparse Interpolation Wave-Function Pinning**: Build the TOPOS logic to take MACE-OFF24 boundaries and query ORCA 6 for sparse DLPNO-CCSD(T) single points.
- [ ] **VRAM Crash Prevention**: Add dynamic batch sizing to MACE-OFF24(m) GPU inference arrays in TOPOS to prevent Out-Of-Memory (OOM) WebGL/CUDA errors.
- [ ] **Solvent Shell Extraction**: Write a TOPOS utility to slice out explicit solvent shells from MD trajectories to compute radial distribution functions.
- [ ] **Non-Adiabatic MD Surface Hopping**: Enable surface hopping MD algorithms in TOPOS for modeling laser-induced photochemistry and excimer formation.
- [ ] **Conical Intersection Locator**: Utilize sTDA / sTD-DFT via g-xTB in TOPOS to rapidly map conical intersections.
- [ ] **Isotope Substitution Loop**: Create a TOPOS automated loop for isotopic substitution (e.g., H -> D) to prepare data for Kraitchman r_s coordinate generation.
- [ ] **Dynamic Trajectory Frame Extraction**: Align extracted MD frames to the Eckart principal axes of inertia to prevent internal motion from aliasing as global rotation.
- [ ] **Topological Extrema Export**: Standardize the JSON output of TOPOS to feed directly into the TORQ grid generator.
- [ ] Currently missing the jupyter notebook which will act as the UI to run TOPOS. Once all python scripts have been written and placed in the correct folders generate the architecture then workflow for the the UI jupyter notebook (in separate responses). Output them as markdown documents within the TOPOS repository.
## Category 3: CoChem-TORQ (PES Scans, Wave-Functions, & Anharmonicity)

> [!TIP]
> Focus on expanding dimensions in Discrete Variable Representation (DVR) and handling resonance issues in perturbation theories.


- [ ] Pass the GOAT search matrices into TORQ for use in the search for IAMs and the calculations to correct them. The vast majority of the calculations needed are done during the isomer/complex searches and optimization work. 
- [ ] Ensure **Time-Tier Execution Tethers**: Hardcode the 10 temporal wall-clock tiers (10s to 1mo) into the `CoChem-BASE` job manager to forcefully terminate or downscale jobs exceeding their computational budget. These strictly adhere to methods, keywords from the document "D:\GitHub-Repo20360805 Method Matrix" 
- [ ] When a weak complex flag exists for the isomer the "D:\Improving LAM Microwave Predictions" protocols must be offered and recommended to the user.
- [ ] **3D Discrete Variable Representation (DVR)**: Expand the TORQ 1D DVR placeholder to a full 3D Sinc-DVR Colbert-Miller implementation in JAX.
- [ ] **DVR Hamiltonian Diagonalization**: Port the kinetic energy operator and 3D potential matrices to the GPU for massive parallel diagonalization via JAX.
- [ ] **Probability Density Integration**: Compute effective rotational constants (r_0) by integrating the classical moment of inertia tensor over the DVR ground-state probability density.
- [ ] **Centrifugal Distortion Divergence Check**: Add a mathematically rigorous check in TORQ to flag perturbative distortion constants (D_J, D_K) that become mathematically divergent.
- [ ] **Multi-Dimensional PES Scans**: Utilize the ORCA `%geom scan Simul_Scan true end` feature for GPU-accelerated simultaneous 2D/3D scans via RIJCOSX.
- [ ] **Zero-Point Vibrational Averaging**: Explicitly convert all theoretical r_e geometries to r_0 structures before outputting microwave prediction lines.
- [ ] **Tamm-Dancoff Approximation (TDA)**: Default to TDA when executing Range-Separated TD-DFT to prevent triplet instability in vdW charge-transfer states.
- [ ] **Explicit Overtones in VPT2**: Extract explicit third and fourth derivatives from VPT2 to map THz shifts and overtones, rather than solely fundamentals.
- [ ] **CP-SCF Solver for Raman**: Enable Coupled-Perturbed SCF calculations (via Polar keyword) for Raman polarizability derivatives, utilizing RIJCOSX acceleration.
- [ ] **Magnetic Shielding Gauge Invariance**: Strictly mandate GIAO (Gauge-Including Atomic Orbital) and Tau Dobson modifiers for meta-GGAs during NMR shielding tensor generation.
- [ ] **NMR Decontracted Basis Sets**: Force the use of `pcSseg-n` and `pcJ-n` basis sets exclusively for NMR J-couplings and shieldings.
- [ ] **Thermally Averaged NMR**: Extract shieldings from 100 AIMD frames and perform a thermal ensemble average to capture fluxional vdW effects.
- [ ] **Mass Spec Trajectory Ensembles**: Scale QCxMS 70 eV electron ionization simulations to 1000 trajectories to achieve statistical experimental agreement.
- [ ] **Rotational Wavefunction Spin Weights**: Enforce strict nuclear spin statistical weights for rotational-vibrational partitions when exporting to SpycFit.
- [ ] **Coriolis Coupling Extraction**: Map Darling-Dennison and Coriolis resonances explicitly to warn the SpycFit fitting engine of localized spectral perturbations.

## Category 4: CoChem-SpycFit (JAX DVR, Spectroscopy, & Active Learning)

> [!IMPORTANT]
> Introduce probabilistic, ML-based Bayesian Parameter Priors and true 3D spatial diagonalization.

- [ ] **Active Learning (Information-Gain)**: Implement Information-Gain Transition Mapping in SpycFit to mathematically propose the next microwave transition that minimizes the covariance matrix.
- [ ] **Bayesian Parameter Priors**: Modify the JAX least-squares fitter to utilize Bayesian priors, restricting centrifugal constants from drifting unphysically far from ab initio DLPNO-CCSD(T) anchors.
- [ ] **Leave-One-Isotopologue-Out (LOIO)**: Develop an automated LOIO Cross-Validation routine in SpycFit to detect statistically over-leveraged structures.
- [ ] **JAX Autodiff Jacobians**: Replace slow finite-difference Jacobians with forward-mode automatic differentiation (`jax.jvp`) for instantaneous rotational level derivatives.
- [ ] **Hardware-Accelerated Diagonalizer**: Send 3D tensors of rotational states (e.g., J=0 to J=80) to the GPU for simultaneous diagonalization.
- [ ] **Dynamic IR Autocorrelation**: Implement FFT on Transition Dipole Moment (TDM) vectors extracted from AIMD trajectories to predict true THz/Far-IR spectra.
- [ ] **Raman Polarizability Autocorrelation**: Implement FFT on polarizability tensors from MACE-POLAR-1 trajectories for collision-induced Raman broadening.
- [ ] **Tunneling Splitting Analysis**: Resolve A/E tunneling splittings directly from the DVR excited state wavefunctions in SpycFit.
- [ ] **GPU VMAP Convolutions**: Fully expand `vmap_gaussian` to handle complex asymmetrical Voigt profiles for dense gas mixtures.
- [ ] **CP-FTMW Spectrometer Ingestion**: Add parsing logic to natively ingest `.csv` and raw broadband CP-FTMW experimental data files.
- [ ] **Automated Line Assignment**: Combine Bayesian priors and theoretical transition strengths to perform preliminary autonomous peak assignment.
- [ ] **Negative Distortion Trap**: Enforce penalty functions to prevent Levenberg-Marquardt or MCMC optimizers from settling into unphysical negative centrifugal distortion constants.
- [ ] **Kraitchman Substitution Visualizer**: Output 3D visual maps of substitution coordinates to visually verify the structural consistency across isotopologues.
- [ ] **Precision Float64 Lockdown**: Add aggressive runtime checks in SpycFit loops to ensure JAX does not silently revert to float32 during massive tensor operations.
- [ ] **Direct .lin/.cat Export**: Generate compliant `.lin` and `.cat` files perfectly formatted for legacy cross-verification (e.g., Pickett's SPFIT).

## Category 5: CoChem-SCRIBE (Academic Publishing, Provenance, & UI Integrations)

> [!IMPORTANT]
> Focus on dynamically extracting the exact physics simulated to generate unassailable, publication-ready artifacts (LaTeX/BibTeX/Figures) while compressing massive tensors.

- [ ] **Zstandard Artifact Compression**: Replace the native .zip functionality in scribe_doc_manager.py with 	ar.zst (Zstandard) to massively reduce the footprint of multidimensional DVR grids without stalling Jupyter execution.
- [ ] **Background Archival Daemons**: Instead of relying on a "Compress Artifacts" button at the end of the notebook, implement a continuous asynchronous background archival process. **This needs to work in memory-aware batches to avoid over-taxing the drive and preventing I/O bottleneck crashes.**
- [ ] **Dynamic Output Parsing**: Refactor scribe_payload_builder.py to parse the raw ORCA .out and MACE .log files to capture adaptive MLFF fallbacks triggered during runtime, rather than just relying on the static registries.
- [ ] **LaTeX Methodology Generation**: Generate a strict LaTeX methodology block containing the exact hardware, MLFF tiers, Basis Sets, DFT functionals, and grid densities (DefGrid4) extracted directly from the workflow.
- [ ] **Dynamic BibTeX API Handoff**: Ensure SCRIBE automatically queries the CrossRef API based on the utilized theories (e.g., pulling the exact citation for wB97X-D4 or Sinc-DVR) to generate an updated .bib file.
- [ ] **Plotly Interactive PES Scans**: Integrate Plotly in SCRIBE to auto-generate .html based interactive 2D/3D Potential Energy Surface (PES) maps of the Large Amplitude Motions calculated by TORQ.
- [ ] **Py3Dmol Structure Generation**: Auto-generate publication-ready .png renders and interactive HTML widgets of the global minimum and TS structures, explicitly highlighting Non-Covalent Interactions (NCI).
- [ ] **Automated Supporting Information (SI) Assembly**: Combine the raw coordinates of the global minima, harmonic frequencies, and anharmonic constants into a journal-ready SI PDF.
- [ ] **Rigorous LLM Constraint Scaffolding**: Expand the strict prompting in scribe_payload_builder.py beyond the "User Guide" to encompass the generation of the "Results & Discussion" section, providing the LLM with the final Bayesian SpycFit parameters.
- [ ] **Offline Fallback Transparency**: If scribe_inference.py fails to hit the Gemini API and triggers its offline fallback, the generated documents must prominently display a watermark stating that the AI interpretation phase was bypassed.
- [ ] **Enantiomer Classification Reporting**: Ensure SCRIBE parses the chiral buckets from TOPOS and explicitly reports the enantiomeric excess or racemic ratios identified during the combinatorial phase.
- [ ] **LAM Physics Reporting**: If the "Weak Complex" flag triggers the LAM protocols in TORQ, SCRIBE must inject a paragraph into the LaTeX file explicitly justifying the use of Sinc-DVR over the rigid-rotor harmonic oscillator approximation.
- [ ] **Wildcard Log Harvesting Refinement**: Stop utilizing a blind *.log harvest in scribe_doc_manager.py. Map and isolate specific execution streams to prevent archiving unrelated system clutter.
- [ ] **Cross-Module Sync Validation**: Integrate a pre-flight check in SCRIBE that verifies the schema versions of BASE, TOPOS, and TORQ match perfectly before it allows the final payload compression to execute.
- [ ] **HDF5 Tensor Extraction**: Implement a SCRIBE utility that unpacks the massive HDF5 registries passed between TOPOS and TORQ, extracting only the human-readable summary tables (e.g., r_0 vs r_e geometries) for publication.


### SCRIBE Integration with CoChem-TOPOS
- [ ] **TOPOS Isomer Carousel Generation**: SCRIBE must auto-generate interactive 3D HTML carousels (py3Dmol) for the top 10 unique conformers identified by the GOAT loop, explicitly coloring regions of conformational divergence.
- [ ] **NCI Surface Mapping Export**: When TOPOS identifies strong/weak complexes, SCRIBE should extract the Non-Covalent Interaction (NCI) grid data and bundle it as .cube files specifically formatted for PyMOL ingestion.
- [ ] **Chirality Distribution Reporting**: SCRIBE must parse the enantiomer classification buckets from the interactive Crusher loop and output a LaTeX table summarizing the enantiomeric excess (ee) or pseudo-racemic distributions.
- [ ] **Thermodynamic Boltzmann Weighting**: SCRIBE must aggregate the final DLPNO-CCSD(T) energies from TOPOS, calculate the Boltzmann populations at 298K, and generate a stacked bar chart (Plotly) representing the ensemble.
- [ ] **AIMD Trajectory Slicing**: If AIMD conformational sampling is utilized, SCRIBE should extract the lowest-energy frame every 10 picoseconds to generate a .gif or .mp4 visual summary of the conformational evolution.
- [ ] **MACE-Screen Rejection Metrics**: SCRIBE must compile the active learning rejection statistics (e.g., '85% discarded due to steric clash') into a pie chart to justify the computational efficiency in the manuscript.
- [ ] **Symmetry Override Documentation**: If a user manually enforces a Point Group (e.g., C2v), SCRIBE must explicitly log this human-in-the-loop intervention in the final LaTeX methodology section to ensure reproducibility.
- [ ] **Interactive Conformer RMSD Matrix**: SCRIBE must generate a clustered heatmap (Plotly) displaying the Root-Mean-Square Deviation (RMSD) matrix of all unique isomers to visually prove structural distinctness.
- [ ] **Solvent Shell Radial Distribution (RDF)**: If explicit solvation was mapped in TOPOS, SCRIBE must calculate and plot the RDF (g(r)) of the solvent molecules relative to the solute center of mass.
- [ ] **Memory-Aware HDF5 Flushing (TOPOS)**: The background archiving daemon must monitor RAM utilization during the massive combinatorial GOAT loop and flush conformer matrices to disk in 500MB chunks to prevent paging.

### SCRIBE Integration with CoChem-TORQ
- [ ] **DVR Probability Density Plots**: SCRIBE must extract the 3D Colbert-Miller Sinc-DVR probability wavefunctions from TORQ and render 2D contour maps overlaying the classical Potential Energy Surface (PES).
- [ ] **LAM Trigger Justification**: When the 'Weak Complex' flag triggers Large Amplitude Motion protocols, SCRIBE must automatically construct a LaTeX paragraph citing the *Improving LAM Microwave Predictions* document to justify the non-rigid rotor approach.
- [ ] **Anharmonic vs Harmonic Overlay**: SCRIBE must generate a Plotly graph directly comparing the harmonic (VPT2) infrared spectrum against the fully anharmonic DVR-derived spectrum to highlight the theoretical shift.
- [ ] **Coriolis Coupling Matrices**: SCRIBE must parse the Darling-Dennison and Coriolis resonance matrices from TORQ and format them into strict APS compliant LaTeX tables for the Supporting Information.
- [ ] **Thermal Shielding Averaging (NMR)**: If thermally averaged NMR is run in TORQ, SCRIBE must generate a histogram showing the distribution of the isotropic shielding constants across the 100 AIMD frames.
- [ ] **PES Simul_Scan Surface Generation**: SCRIBE must take the raw ORCA %geom scan multidimensional outputs and generate a smoothed 3D surface plot (Plotly Surface) illustrating the minimum energy pathways.
- [ ] **Memory-Aware PES Archiving (TORQ)**: Ensure SCRIBE's background daemon utilizes Zstandard (	ar.zst) chunking when archiving massive 3D PES grids, flushing to the SSD only during TORQ's idle CPU cycles to avoid I/O blocking.
- [ ] **Divergent Distortion Flagging**: If TORQ detects mathematically divergent centrifugal distortion constants (D_J, D_K), SCRIBE must highlight these anomalies in a red text box within the finalized PDF report.
- [ ] **Raman Polarizability Tensors**: SCRIBE must extract the Raman polarizability derivatives from the CP-SCF solver and plot the simulated collision-induced Raman broadening profiles.
- [ ] **Isotopologue Kraitchman Substitution Maps**: SCRIBE must auto-generate 3D structural images with semitransparent spheres representing the calculated r_s substitution coordinates mapped over the theoretical r_0 structure.

### SCRIBE Integration with CoChem-SpycFit
- [ ] **Bayesian Prior vs Posterior Plots**: SCRIBE must extract the Levenberg-Marquardt fitting progression from SpycFit and generate a Plotly scatter plot showing the drift of rotational constants from their ab initio priors to the fitted posteriors.
- [ ] **Information-Gain Transition Mapping**: SCRIBE should auto-generate a dynamic table showing the 'Next Best Microwave Transitions' proposed by the active learning algorithm, complete with theoretical intensities.
- [ ] **LOIO Cross-Validation Boxplots**: Upon completion of the Leave-One-Isotopologue-Out (LOIO) routine, SCRIBE must generate a box-and-whisker plot displaying the statistical leverage and variance of each isotopic structure.
- [ ] **Residual Analysis Histograms**: SCRIBE must generate a histogram and Q-Q plot of the final frequency fitting residuals (Observed - Calculated) to prove Gaussian error distribution in the CP-FTMW assignments.
- [ ] **Pickett .lin/.cat Bundle Zipping**: SCRIBE must specifically isolate and package the .lin, .cat, and .fit files generated by SpycFit into a separate, highly compressed Legacy_Verification.zip for Pickett SPFIT cross-validation.
- [ ] **Tunneling Splitting Energy Level Diagrams**: SCRIBE must generate a LaTeX TikZ energy level diagram illustrating the A/E tunneling splittings resolved from the DVR excited state wavefunctions.
- [ ] **Hardware-Accelerated Diagonalization Telemetry**: SCRIBE must parse the JAX VRAM utilization logs and output a benchmark graph proving the millisecond advantage of GPU tensor diagonalization over legacy CPU Fortran codes.
- [ ] **Voigt Convolution Generation**: SCRIBE must output high-resolution .svg vectors of the final JAX map_gaussian Voigt profiles for direct publication without manual replotting.
- [ ] **Negative Distortion Penalty Reporting**: If the fitting optimizer attempts to step into unphysical negative centrifugal distortion space, SCRIBE must log the exact penalty function applied and include it in the SI.
- [ ] **Memory-Aware Tensor Archiving (SpycFit)**: Implement strict memory-mapped file (mmap) reading in SCRIBE when packing the massive 3D tensors of rotational states (J=0 to J=80) to prevent the archiving process from crashing the GPU context.

## Category 6: Academic Integrity and Scientific Validity

> [!CAUTION]
> Maintain strict traceability, reproducibility, and rigorous adherence to methodological boundaries to ensure publications remain defensible.

- [ ] **Methodological Provenance Logging**: Inject an unalterable JSON ledger in all repos tracking the exact method, basis, grid, and hardware used for every tensor.
- [ ] **Algorithmic Transparency**: Ensure all MLFF fallbacks (MACE -> AIMNet2 -> xTB) generate a clear terminal warning indicating the substitution.
- [ ] **Raw Tensor Archival**: Save the raw DVR matrices and MLFF arrays before transformation to allow independent replication of the mathematical 'smear'.
- [ ] **Citation Auto-Generator**: Automatically append BibTeX citations for MACE, ORCA, JAX, and VPT2 based on the specific methodological tier triggered.
- [ ] **Statistical Uncertainty Bounds**: Require SpycFit to report the 95% confidence intervals and standard errors for all Bayesian-fitted rotational constants.
- [ ] **BSSE Validation**: Force a comparative test (Complex vs Fragments) to mathematically prove the Counterpoise correction was applied correctly.
- [ ] **Convergence Verification**: Halt the pipeline and flag an integrity error if any ORCA SCF loop fails to reach the requested `ExtremeSCF` tolerance.
- [ ] **Resonance Warning Flags**: Hard-code visual warnings when Fermi or Coriolis resonance denominators in VPT2 approach zero.
- [ ] **Dispersion Parameter Validation**: Verify that Becke-Johnson D4 parameters are dynamically utilizing fractional coordination numbers, not static defaults.
- [ ] **Basis Set Incompleteness Tracking**: Extrapolate and log the estimated Basis Set Incompleteness Error (BSIE) when failing to reach the CBS limit.
- [ ] **Reproducibility Seeds**: Hardcode PRNG seeds for all stochastic isomer searches and MD trajectories to guarantee bit-for-bit reproducible runs.
- [ ] **Explicit vs Implicit Solvation**: Document explicitly when CPCM is used instead of explicit MACE-MD solvent, citing the loss of specific hydrogen bonding interactions.
- [ ] **Energy Conservation Checks**: Monitor the NVE ensemble drift during AIMD; discard trajectories if total energy fluctuates beyond acceptable thermodynamic thresholds.
- [ ] **Vibrational Averaging Justification**: Produce a localized text output explaining *why* the theoretical r_e structure does not match the predicted r_0 microwave constants.
- [ ] **Blind Fitting Mode**: Introduce a 'blind' mode in SpycFit that prevents the user from altering Bayesian priors manually to force a desired outcome.

## Category 7: User Experience and Didactic Educational Experience

> [!NOTE]
> Abstract complex jargon into actionable, educational insights so students learn *why* the chemistry works, rather than just clicking execute.

- [ ] **Codespaces CPU Graceful Degradation**: Make the transition from GPU to CPU on Codespaces seamless, providing an educational popup explaining *why* MACE was swapped to g-xTB.
- [ ] **'Harmonic Trap' Educational Warnings**: Display a didactic warning when users attempt to use standard harmonic frequencies to fit THz spectra, referencing LAMs.
- [ ] **Visual PES Navigation**: Provide a web-based or local Dash/Streamlit visualization of the 2D MACE potential energy surface for users to physically 'see' the LAM barriers.
- [ ] **Step-by-Step Spectroscopic Deduplication**: Show a live terminal readout of the 'Spectroscopic Override' algorithm, demonstrating how chemically identical structures are merged.
- [ ] **Method Matrix Guided UI**: Build a CLI wizard that asks the user for their available timeframe (10s to 1mo) and property of interest, automatically configuring the inputs.
- [ ] **Real-time JAX Diagonalization Feedback**: Output GPU timing metrics in SpycFit to show students the millisecond advantage of JAX over traditional Fortran diagonalizers.
- [ ] **Jargon-Free Error Handling**: Replace dense ORCA/JAX stack traces with conversational AI explanations (e.g., 'The basis set is too diffuse and caused linear dependence. Dropping to Def2-TZVPP.').
- [ ] **Interactive Molecular 'Smear' View**: Export the DVR probability density maps as `.cube` files so users can view the structural smearing in PyMOL/Avogadro.
- [ ] **Progress Bars for Extrapolations**: Implement detailed TQDM progress bars for massive 1000-trajectory QCxMS or 3D Simul_Scan operations.
- [ ] **Theory Comparison Overlays**: Allow users to overlay rigid-rotor predicted lines vs DVR predicted lines to directly visualize the impact of Large Amplitude Motions.

## Category 8: Accuracy Optimization and Comparable Time Costs

> [!TIP]
> Capitalize on local approximations, resolution of identity (RI), and hardware caching to slash execution time without sacrificing golden-tier accuracy.

- [ ] **Local Pair Natural Orbital (LPNO) Defaults**: Enforce DLPNO-CCSD(T) as the default coupled cluster methodology over canonical CC to drastically reduce time costs.
- [ ] **JIT Compilation Caching**: Implement robust AOT (Ahead-of-Time) or persistent JIT caching in SpycFit to prevent recompiling the DVR Hamiltonian on every run.
- [ ] **Vectorized Batched Inference**: Group multiple MACE structural queries into a single tensor batch to maximize GPU core utilization.
- [ ] **Subsampling AIMD Trajectories**: Optimize time costs by extracting NMR or UV-Vis single points every 500 frames instead of every frame, maintaining statistical accuracy.
- [ ] **STEOM-CCSD over EOM-CCSD**: Strictly route excited state UV-Vis calculations to STEOM-CCSD to achieve ~0.03 eV accuracy at a fraction of the scaling cost.
- [ ] **Resolution of Identity (RI)**: Mandate the use of RI-J and chain-of-spheres (COSX) for all double hybrid calculations to ensure 1-hour wall clocks for 10-atom complexes.
- [ ] **Minimal DFT Pre-Optimization**: Chain r2SCAN-3c optimizations directly into wB97X-D4 to reduce the SCF cycles required by the heavier functional.
- [ ] **MACE-OFF23 for Triage**: Retain the smaller MACE-OFF23 model specifically for hyper-fast initial triage, reserving MACE-OFF24(m) for final topological anchoring.
- [ ] **Automated Hessian Reuse**: Cache and reuse the analytical Hessian matrix during VPT2 transition state evaluations to eliminate redundant derivative calculations.
- [ ] **Sparse Matrix Operations**: Convert all classical rigid-rotor Hamiltonian constructions to sparse matrices in JAX to drop memory footprints by 90%.
- [ ] **GPU Memory Growth Limits**: Configure JAX `XLA_PYTHON_CLIENT_MEM_FRACTION` to prevent out-of-memory cascading failures when chaining TOPOS and SpycFit.
- [ ] **Composite Basis Set Optimization**: Utilize def2-mTZVPP (native to r2SCAN-3c) for 30m tier calculations instead of manual basis definitions to save compilation time.
- [ ] **Fast Multipole Method (FMM)**: Enable FMM automatically when cluster size exceeds 15 atoms to maintain linear scaling of the Coulomb matrix.
- [ ] **Adaptive Integration Grids**: Start geometry optimizations on `DefGrid1` and step up to `DefGrid4` only during the final refinement steps.
- [ ] **Early Stopping Criteria in AIMD**: Implement automated early stopping if the dipole autocorrelation function converges before the 1-nanosecond trajectory finishes.