> **Historical source — reviewed and superseded for TORQ implementation.** Use the [implementation SRS](CoChem-TORQ_Implementation_SRS.md), [method implementation contract](Method_Matrix_Implementation_Contract.md) and [decision register](review/Decisions_and_Risks.md). The original content below is preserved for traceability; unsupported claims and conflicting instructions are not normative. Source-line references in the reviews refer to commit `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`, before this banner.

# CoChem-TORQ Integration Architecture

This document defines the formal integration boundaries and data-flow handoffs between **CoChem-TORQ** and the other core modules of the CoChem ecosystem: **CoChem-BASE**, **CoChem-TOPOS**, and **CoChem-SpycFit**. 

## 1. Upstream Integration: CoChem-TOPOS $\rightarrow$ CoChem-TORQ

**CoChem-TOPOS** is the upstream topology generator and conformer enumerator. It relies on extremely fast, low-tier methods (like GOAT and CREST) to explore conformational space broadly and generate initial geometry guesses. 

### Data Handoff Protocol
*   **What TOPOS provides:** A set of candidate coordinates (e.g., `.xyz` or `.sdf` files), initial topology graphs (describing connectivity), and preliminary low-tier energies.
*   **What TORQ expects:** TORQ ingests these unrefined TOPOS candidate structures. 
*   **TORQ's Responsibility:** TORQ is strictly forbidden from "guessing" new conformers blindly. Instead, it takes the TOPOS geometries, performs the **Human-in-the-Loop (HITL) Topological Interconversion** to ask the user which flexible dihedrals to scan, and then executes the **Cascading PES "Expand Out" Protocol**. It refines the TOPOS guesses up the 11-tier Method Matrix until convergence is achieved. 

## 2. Core Infrastructure Integration: CoChem-TORQ $\leftrightarrow$ CoChem-BASE

**CoChem-BASE** provides the foundational infrastructure, orchestrators, UI environments, and data serialization protocols for the entire ecosystem.

### Data Handoff Protocol
*   **Hardware Brokering:** TORQ calls the BASE `cochem_system_config.json` via the interaction matrix to request CPU (MPI pools with `%maxcore`) and GPU resources (for MLFF tensors). 
*   **Task Matrix & Stateful Batching:** TORQ's cascading jobs (which can take hours to weeks) are submitted into the BASE `job_board.db` using the CoChem Kanban state machine. This allows TORQ to be resilient against crashes.
*   **Interaction Environments:** TORQ pushes its outputs (e.g., Energy Diagrams, Symmetry Confirmation Prompts) to BASE's Jupyter, Codespaces, or Headless/Voila UI layers, where the Human-in-the-Loop (HITL) actively makes decisions. 

## 3. Downstream Integration: CoChem-TORQ $\rightarrow$ CoChem-SpycFit

**CoChem-SpycFit** is the downstream microwave spectral assignment and prediction engine. It relies on graphics-card-accelerated HDF5 rendering and human-in-the-loop interactive peak-picking. 

### Data Handoff Protocol
*   **What TORQ provides:** After TORQ has finalized the 11-tier calculations, it extracts the definitive physical observables: Rotational Constants ($A, B, C$), Centrifugal Distortion Constants, Dipole Moments ($\mu_a, \mu_b, \mu_c$), and precise geometry geometries.
*   **Dual-Bundling:**
    1.  **HDF5 Injection:** TORQ bundles the ab initio tensors directly into the `.h5` format expected by SpycFit. This allows SpycFit to rapidly simulate the theoretical spectrum on the GPU.
    2.  **Legacy SPCAT/SPFIT:** For backward compatibility and publication, TORQ also generates standard `.var` and `.int` files. *Crucially, TORQ must filter out any Tier-1 (Estimated/MLFF) data from these legacy files to ensure strict physical rigor.*

---
**Summary of the Flow:**
`TOPOS (Conformer Guesses)` $\rightarrow$ `TORQ (Precision Quantum Scaling + Observables Extraction)` $\rightarrow$ `SpycFit (Interactive Spectral Assignment)`
*All orchestrated by `BASE`.*
