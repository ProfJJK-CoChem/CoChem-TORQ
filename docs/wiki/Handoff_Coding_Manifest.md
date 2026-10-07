# CoChem-TORQ coding handoff

Read the [implementation SRS](CoChem-TORQ_Implementation_SRS.md), [method contract](Method_Matrix_Implementation_Contract.md), [current implementation audit](review/Architecture_Implementation_Audit.md), and [decision register](review/Decisions_and_Risks.md) before coding. They supersede the earlier Comprehensive/True/Iterated narrative requirements. The [point review](review/SRS_Point_Review.md) and [matrix audit](review/Method_Matrix_Review.md) explain each adopted, corrected, deferred or rejected proposal.

1. Begin with WP00: eliminate fabricated frequencies/zero-valued missing properties, false convergence and incorrect optimized-geometry/success reporting. Preserve failed artifacts and make missing data explicit. Do not close these findings by changing tests to accept incorrect behavior.
2. Implement WP01–WP05 as a real local vertical slice: clean package installation, typed contracts and units, exact capability/recipe resolution, bounded durable execution, actual geometry/derivative evaluation and truthful observable/export status.
3. Add PES/TS, correlated/composite, nuclear-motion, analytics and deployment profiles using WP06–WP12 and the decision gates. Do not fabricate an open-source revDSD implementation, JANPA/NBO equivalence, unsupported CC Hessians, experimental calibration or cross-repository compatibility.
4. Implement every selected feature with stable requirement IDs, acceptance evidence, data migration where needed, and explicit limitations. A source file, import, hardcoded result or narrative agent-audit claim is not completion evidence.
5. Run authentic physical engine integration and scientific benchmarks for claimed profiles. Mathematical reference cases and explicitly labeled malformed-input tests verify software/numerics; they do not replace real chemistry evidence. Unavailable required engine tests are blocked/unrun, never passed.
6. Keep TOPOS discovery, BASE task ownership and SpycFit consumer responsibilities explicit. Use fenced task attempts and immutable manifests; HDF5 is not a multiwriter job queue.
7. Mark a release complete only when all enabled-profile requirements and deployment/scientific gates in SRS §15 pass. Material choices in D01–D12 remain explicit decisions; independent core work can proceed meanwhile.

This handoff specifies implementation work. It does not claim the current application satisfies it.
