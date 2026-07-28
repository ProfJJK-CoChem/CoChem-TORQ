# **CoChem-TORQ Architecture: Part 2 — Machine Learning Triage & Ab Initio Escalation (Stages 2.0 \- 3.0)**

## **1\. Architectural Intent**

Evaluating thousands of torsional grid points at the coupled-cluster level is computationally intractable. CoChem-TORQ implements a hierarchical multi-tier funnel, utilizing MACE-OFF23 machine learning force fields for rapid preliminary screening before escalating critical topographic extrema to gold-standard quantum chemistry packages.

## **2\. Module Responsibilities**

* **TorqMACETriage (cochem\_torq\_mace.py)**: Ingests the 1D/2D Cartesian grid and evaluates single-point energies across thousands of conformers.  
* **Memory Governor & VRAM Safeguard**: Actively manages PyTorch tensors by invoking garbage collection (gc.collect()) and CUDA cache purging (torch.cuda.empty\_cache()) at user-defined intervals (vram\_flush\_interval=50).  
* **TorqOrcaExecution (cochem\_torq\_orca.py)**: Interfaces with ORCA 6.1.1 to refine selected topographic extrema. Dynamically manages Out-Of-Memory (OOM) exceptions by intercepting bad\_alloc strings and halving %maxcore allocations on-the-fly.

## **3\. Data Flow & Inter-Process Communication**

\[torq\_grid.json\] ──► \[Stage 2.0: MACE-OFF23 Single-Point Evaluation\]  
                            │  
                            ▼  
                     \[Relative Energy Normalization (0.0 kcal/mol Anchor)\]  
                            │  
                            ▼  
                     \[Extrema Extraction (Basins & Barrier Peaks)\]  
                            │  
                            ▼  
                     \[Stage 3.0: Constrained ORCA 6.1.1 Refinement\]  
                            │  
                            ▼  
                     \[Output: Landscape Database (.h5 / .out)\]

## **4\. Key Dependencies & Failure Points**

* **Dependencies**: torch, ase, mace-torch, subprocess.  
* **Failure Points**:  
  * GPU VRAM fragmentation during large 2D combinatorial grid scans.  
  * ORCA SCF convergence failures on highly distorted torsional transition states.  
* **Validation Checkpoints**: Pre-flight hardware check ensuring CUDA availability, with graceful CPU fallback triggers and automated %maxcore memory backoff loops.