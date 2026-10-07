# CoChem-TORQ: Handoff Manifest for Coding Agents

## 🚨 MANDATORY ZERO-MOCK DIRECTIVES
All agents stepping into `CoChem-TORQ` to begin physical coding must absolutely read and adhere to the `CoChem-TORQ_Comprehensive_SRS.md` document located in this `docs/wiki/` directory.

### Core Objectives
1. **Physical Engine Integration**: Do not mock shell calls. You are required to physically wire `cochem_torq_engine.py` and the pipeline scripts to properly invoke `PySCF`, `ORCA`, `dftd4`, and `CFOUR` using authentic input blocks and precise output parsing.
2. **The 11-Tier Method Matrix**: All job routing must scale gracefully through the 11 tiers. If the user invokes Tier 5 (revDSD double-hybrid) without an ORCA license, you must implement the **No-License Fallback** mathematically detailed in Chapter 2 of the SRS (combining PySCF exact exchange, MP2, and dftd4).
3. **Upstream Ingestion**: Topos conformer ingestion must strictly use QCElemental/QCIO schemas. Do NOT parse `.xyz` files manually.
4. **Isotopologue Substitution**: As detailed in Chapter 13, you must physically calculate the principal moments of inertia and rotate to the principal axis system when substituting heavy atoms.
5. **No Spoofing**: Ensure `pytest` passes cleanly. Do not skip tests or mock subprocess results. Use the Kan-ban `job_board.db` for orchestrating asynchronous HPC or background tasks.

### Next Steps for Implementation
Review `cochem_torq_pipeline.py` and `cochem_torq_engine.py` against the `CoChem-TORQ_Comprehensive_SRS.md`. 
Determine the missing capabilities (e.g., the missing VPT2 CFOUR integration, or the Isomer PES cascading loop) and submit structured Kan-ban tasks to the orchestrator to build them block-by-block.
