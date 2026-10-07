# CoChem-TORQ Agentic Knowledge Base (Wiki RAG)

Welcome to the CoChem-TORQ local Wiki. This folder contains all the critical architectural documents, functional rules, and upstream/downstream context required for any agent working on the `CoChem-TORQ` module.

**If you are an agent operating in this repository, you MUST read these documents to ensure you do not hallucinate features from other modules.**

## Core Documents

1.  **[CoChem-TORQ True SRS](./CoChem-TORQ_True_SRS.md)**
    *   **Description:** The absolute source of truth for the TORQ module. It outlines the boundaries, the 11-Tier Method Matrix scaling, the 3 Open-Source/ORCA/CFOUR pathways, the PES Cascading protocol, and the strict Human-In-The-Loop (HITL) mandates.

2.  **[Auto-Architect Phase 1 Risk/Reward](./CoChem-TORQ_AutoArchitect_Phase1.md)**
    *   **Description:** An exhaustive breakdown of the architectural pillars of TORQ, featuring pros, cons, and risk/reward metrics deduplicated from the original CoChem discussion history.

3.  **[Integration Architecture](./CoChem-TORQ_Integration_Architecture.md)**
    *   **Description:** Details the strict data handoffs. Explains exactly how TORQ ingests conformer topologies from `CoChem-TOPOS`, how it utilizes `CoChem-BASE` for hardware brokering and task batching, and how it outputs final GPU-ready HDF5 files for `CoChem-SpycFit`.

4.  **[The Method Matrix v4.2](./Method_Matrix.md)**
    *   **Description:** The global CoChem master ledger defining the theoretical calculation tiers (From MLFF to CCSD(T)). TORQ must rigorously follow these tiers.

5.  **[NBO Analysis Details](./NBO_Analysis.md)**
    *   **Description:** The specific Natural Bond Orbital (NBO) analysis instructions and physical theory necessary for TORQ's stationary point validation outputs.

**Important Rule:** Do NOT merge `CoChem-TOPOS` responsibilities (like GOAT/CREST automated conformer discovery) into `CoChem-TORQ`. TORQ is purely for downstream precision refinement, physics extraction, and PES dynamics.
