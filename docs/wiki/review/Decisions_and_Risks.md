# Significant decisions, benefits, costs and risks

Source baseline: `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`. Companion to the [implementation SRS](../CoChem-TORQ_Implementation_SRS.md).

Low-risk specification corrections have been incorporated: consistent quantities/units, truthful result status, exact method identity, typed interfaces, provenance, conservative capability checks, bounded resources, versioned storage, acceptance evidence and correction of mathematical errors. The follow-up implementation repair is recorded separately; neither documentation nor focused software checks certify scientific accuracy or deployment.

The following choices have material scientific, maintenance, operational or scope consequences. D01–D12 and ARCH-01–ARCH-12 are selected as recorded below; the complete spectroscopy scope and no-fabrication policy are approved. Concrete profile versions, domains and benchmark records still require qualification; no unsupported automatic behavior is enabled. Interim behavior permits independent core implementation and makes dependent capabilities explicitly unavailable or experimental. The [601-entry source review](SRS_Point_Review.md) and [full matrix audit](Method_Matrix_Review.md) supply point-specific findings, including proposals not repeated here.

See [all 24 decision questions, each with three options and pros/cons/risks](Decision_Questions.md).

## D01 — Open-source revDSD implementation

**Accepted selection:** Develop exact open-source revDSD with independently gated energy, orbital-response gradients and higher derivatives. The [development and spectroscopy protocol](../RevDSD_Spectroscopy_Protocol.md) specifies a pinned PySCF CPU/DFTD4 implementation basis and independent qualification; existing generic MP2/DFT support is not method certification.

**Problem:** The supplied SRS promises an ORCA-equivalent optimizer by combining HF exchange, MP2 and D4, without defining the complete orbital/OS/SS/response model.

**Benefits of implementing it:** Wider access, transparent equations and less dependence on one engine. **Cons:** A substantial electronic-structure method/derivative implementation and long-term maintenance obligation. **Risks:** Smooth, plausible but wrong energies/forces; double-counted corrections; mismatch between optimized geometry and reported model; publication claims without independent validation.

**Recommendation:** Implement the selected open-source research route; keep each unqualified derivative/property capability experimental. Activate energy first, then gradients and higher derivatives after independent cross-implementation and finite-difference validation. A finite-difference optimizer may be evaluated with explicit cost/convergence limits; it is not an analytic-gradient claim.

**Decision needed:** Resolved in favor of building and independently validating the exact open-source method; benchmark/domain and other engine coverage remain separate decisions. **Interim:** Unsupported exact revDSD requests stop with alternatives; no silent substitution. Gate: V-DH.

## D02 — Method taxonomy and derivative-aware engine strategy

**Accepted selection:** Stage engine coverage: qualify the open-source baseline first, then native/reference and high-order profiles individually. An engine name never qualifies all method/property tuples.

**Problem:** Eleven SRS method families conflict with ten matrix time bins and repeated tier renumbering. The new MPQC analytic-CC-Hessian mandate is unsupported by inspected upstream code; ORCA/CFOUR claims also conflict.

**Benefits of a capability registry:** Correct scientific identity and adaptable hardware/engine routing; no fictitious method at every budget. **Cons:** Migration of existing configs/UI/tests and ongoing version-specific maintenance. **Risks:** Ambiguous historical aliases could run the wrong track; treating energy support as derivative support could produce impossible plans.

**Recommendation:** Adopt the separate family/recipe/row/budget/property axes specified in the SRS. Retain all historical IDs, require explicit resolution of ambiguous aliases, and test exact tuples. This semantic correction is already in the documents; accepted D02 stages production engine coverage behind individual qualification gates.

**Decision needed:** Resolved: open-source baseline first; native/reference and high-order profiles qualify later. Exact engine versions and supported tuples are recorded in each qualification manifest. **Interim:** Core adapter framework plus individually validated available engines; unavailable tuples remain blocked. Gate: V-METHOD/V-ENGINE.

## D03 — Adaptive PES, ML screening and delta learning

**Accepted selection:** Advisory acceleration first. Quantum evaluation remains authoritative; no automatic candidate deletion based only on unvalidated ML predictions.

**Benefits:** Potentially large reductions in physical evaluations, reusable surrogates and focused acquisition of difficult regions. **Cons:** Model training, independent calibration, data management, force consistency and out-of-domain monitoring are substantial. **Risks:** False confidence from low ensemble spread, missed minima/barriers, data leakage, domain shift and biased candidate deletion. A published point-count reduction does not transfer automatically to a new complex.

**Recommendation:** Start with bounded physical scans and optional ML preconditioning. Preserve candidates and independently challenge surrogate coverage. Add active/delta learning only after specifying acquisition, holdouts, missed-feature tests, stopping criteria and uncertainty calibration. Automatic permanent pruning by the draft's fixed energy/variance/rotational cutoffs is rejected as stated.

**Decision needed:** Resolved for the initial profile: advisory acceleration. Future validated pruning needs a new scope/domain decision and independent false-negative evidence. **Interim:** Uncalibrated ML is exploratory; no publication accuracy/candidate rejection solely from it. Gate: V-AL/V-UQ.

## D04 — Counterpoise, frozen monomers and composites

**Accepted selection:** Keep fully relaxed, frozen-monomer and counterpoise variants as separate explicit recipes. No silent recipe substitution or universal CP/freeze default.

**Benefits:** Can reduce BSSE/model error and concentrate high-level effort on important coordinates; frozen monomers may make small-complex campaigns practical. **Cons:** More component calculations and configuration, coupled uncertainties and derivative complexity. **Risks:** Suppressed deformation, inconsistent CP gradients, invalid ordinary VPT2 at a constrained/nonstationary geometry, nonadditive corrections and misleading high-level labels.

**Recommendation:** Keep these as named explicit recipes. Compare relaxed/frozen and CP/raw/half-CP variants against a chosen dimer benchmark before selecting defaults. Record geometry, energy, force-field and property methods separately. No universal CP-at-TZ or frozen-monomer mandate.

**Decision needed:** Resolved: separate explicit recipes. The initial benchmark must still name the actual variants and target observables before any default is justified. **Interim:** No automatic freeze, half-CP or composite substitution; requested supported variants retain labels. Gate: V-CP/V-BENCH.

## D05 — VPT2, large-amplitude motion and PI symmetry

**Accepted selection and scope:** Deliver semirigid VPT2 before separately validated LAM profiles. The complete nine-stage typed chain through high-accuracy rotational-spectroscopy identification remains mandatory program scope.

**Benefits:** Essential improvements beyond Be for semi-rigid and floppy spectra; can resolve isotope shifts and tunneling structure. **Cons:** Higher derivatives, resonance handling, coupled kinetic operators, specialist symmetry and expensive independent benchmarks. **Risks:** Treating a harmonic Hessian as an anharmonic force field; calling a reduced DVR exact molecular physics; double-counting ZPE/modes; wrong nuclear-spin sectors; apparent precise but incorrect A/E splittings.

**Recommendation:** Certify semirigid VPT2 first; qualify each later LAM profile separately against its defined physical domain. General coupled/multidimensional VRT and automated PI assignment should be distinct later profiles. Detection of possible LAM should request a model choice or report B0 unavailable; a 2 kcal/mol rule must not select a solver automatically.

**Decision needed:** Resolved delivery order: semirigid VPT2, then separately validated LAM. Concrete LAM models, domains and benchmark molecules require profile specifications; later scope remains approved. **Interim:** Equilibrium and validated harmonic outputs remain available, with unsupported B0/splittings explicitly absent. Gate: V-VPT2/V-DVR/V-BENCH.

## D06 — Autonomous TS rescue, multireference changes and kinetics

**Accepted selection:** Provide diagnostics and explicit scientific escalation. Automatic recovery is limited to logged, bounded numerical changes within the approved physical model and budget.

**Benefits:** Bounded solver recovery and path branching can reduce manual intervention. Rate models can aid interpretation when their assumptions apply. **Cons:** More state tracking, verification, compute and specialist review; LLM-driven rescue introduces reproducibility and security obligations. **Risks:** Switching electronic states/active spaces, finding unintended endpoints, unbounded branching, false VRI/conical-intersection detection and unjustified low-temperature rates.

**Recommendation:** Allow logged bounded numerical recovery within an approved model; require review for new physics, symmetry constraints or electronic states. Verify TS Hessians and both endpoints. Treat LLM replanning, excited-state searches, automatic active-space selection and kinetics as separately validated extensions. Ground-state scans cannot provide S1−S0 gaps.

**Decision needed:** Resolved automation boundary: diagnostics with scientific escalation and bounded same-model numerical recovery. Quantitative kinetics needs a separately specified and validated profile. **Interim:** No automatic physical-model substitution; unsupported kinetics/nonadiabatic outputs remain absent. Gate: V-TS/V-ELECTRONIC/V-KINETICS.

## D07 — NBO and real-space analytics

**Accepted selection:** Optional separately named analyses: genuine licensed NBO when available, and independently identified open-source alternatives.

**Benefits:** Useful interpretable electronic descriptors and reproducible publication figures. **Cons:** Licensed or additional tools, density interchange complexity, grids and expert interpretation. **Risks:** Mislabeling JANPA as NBO; invented donor–acceptor outputs; unsupported wavefunction conversions; treating a BCP as proof of a bond or extracting ZPE from density topology.

**Recommendation:** Optional requested analyses at selected validated structures. Name each actual algorithm and implementation; NBO requires a legitimate NBO backend. Validate density normalization/integration and keep chemical interpretation distinct from descriptors.

**Decision needed:** Resolved: optional and separately named; no NBO-equivalence claim for other algorithms. **Interim:** Only validated supported descriptors are emitted; no NBO-equivalence claim. Gate: V-WFN/V-ANALYSIS.

## D08 — BASE orchestration, storage and SpycFit contract

**Accepted selection:** Use one versioned BASE service contract plus a standalone local adapter, with one task/commit authority per deployment. Actual BASE/SpycFit compatibility still requires consumer tests.

**Benefits:** One task authority and shared schema reduce duplication; durable lineage and consumer tests improve reproducibility. **Cons:** Cross-repository coordination, migrations and API compatibility work. **Risks:** Two controllers owning state, stale workers committing outputs, SQLite WAL over NFS, HDF5 multiwriter corruption, incompatible consumer tensors and retained duplicated infrastructure.

**Recommendation:** A versioned BASE service contract plus local standalone adapter; fenced leases, one commit authority, local coordinator database, independent worker shards and one HDF5 writer/merger. Confirm the actual SpycFit schema through a consumer test. Parsl/AiiDA/Slurm are implementation options, not interchangeable or simultaneously mandatory brands.

**Decision needed:** Resolved architecture: one BASE service contract plus standalone local adapter. Pin the actual service/interface and SpycFit consumer versions in integration manifests; no framework migration is implied. **Interim:** Core uses defined interfaces and local adapter; cross-repository interoperability remains unverified until actual consumer tests pass. Gate: V-STATE/V-STORAGE/V-CONSUMER.

## D09 — Deployment platforms, licenses and security

**Accepted selection:** Certify Linux CPU first, then named GPU/Slurm profiles independently. Other platforms remain unqualified until their own acceptance evidence exists.

**Benefits:** Linux/WSL2/Codespaces/Slurm profiles broaden access; GPU concurrency can improve throughput; real authenticated services protect shared resources. **Cons:** Platform matrices, native/GPU/MPI dependencies, per-site authorization and operational support. **Risks:** Restricted binary redistribution, unsupported CUDA/ROCm/Metal equivalence, MPS contention/isolation issues, security claims based only on AST/path checks, and needless local complexity from mandatory SSO/enclaves.

**Recommendation:** Certify Linux CPU first, then named optional GPU/WSL2/Slurm profiles. Use legitimate provisioned engines and actual vendor/site terms; no invented license tokens. Optional native/container installation profiles must preserve TLS/integrity. Local approvals use authenticated records plus digests; multi-tenant signing/IAM follows a defined threat model.

**Decision needed:** Resolved platform sequence: Linux CPU first, then independently qualified named GPU/Slurm profiles. Actual site authorization, engine terms and deployment versions remain profile prerequisites. **Interim:** No mandatory MPS, public licensed-engine redistribution, global driver changes or universal cryptographic-service prerequisite. Gate: V-PACKAGE/V-SECURITY/V-HPC/V-DEPLOY.

## D10 — Accuracy targets, calibration data and publication claims

**Accepted selection:** Bounded preregistered benchmark with observable-specific targets; dataset/uncertainty/reference details must be fixed before execution.

**Benefits:** Explicit independent benchmarks make accuracy claims defensible and guide cost allocation. **Cons:** Curating reliable references, executing costly calculations, accounting for uncertain experiments and maintaining held-out datasets. **Risks:** Calibration leakage through isotopologues/parent anchoring, reporting MAE as a confidence interval, transfer of unrelated chemistry benchmarks, selection bias and misleading universal 0.1%/1–2% claims.

**Recommendation:** Specify product A/B/C and preregister per-domain observable targets and reference sets. Start with clearly scoped semi-rigid and weak-complex benchmark profiles, report all errors/failures and keep unsupported claims out of publication bundles. Correct conformal score inversion and coverage assumptions as specified in the method companion.

**Decision needed:** The actual molecules, experimental/reference datasets, target observables/error/coverage and acceptable campaign budget defining scientific success. **Interim:** Numerical validation and sensitivity reporting continue; unknown predictive accuracy stays unknown. Gate: V-UQ/V-BENCH/V-RELEASE.

## D11 — Test policy and completion evidence

**Accepted selection:** Layer mathematical, process and persistence tests with mandatory genuine-engine release gates. Software checks do not establish chemical accuracy.

**Benefits of real engine tests:** Direct evidence of executable scientific workflows and protection against synthetic success. **Cons:** Licensed binaries, expensive runtimes and less control over failure injection. **Risks:** Conflating mathematical/software checks with physical validation could falsely certify a method; missing proprietary engines must never be reported as passing.

**Recommendation:** Keep real-engine integration and scientific benchmarks mandatory for claimed profiles. Permit exact mathematical kernel tests and explicitly identified malformed-input/software-failure tests; these never stand in for physical evidence. Document passed, failed, blocked, skipped and unrun outcomes. The user explicitly requires no fabrication or mocks. New integrity checks import production modules and use exact mathematics, real files, processes and databases; no engine impersonation or AST-extracted substitute code is used.

**Decision needed:** Resolved: layered real checks with mandatory engine qualification. No-mock/no-fabrication remains mandatory. **Interim:** Existing no-mock physical integration policy remains; blocked capability tests cannot establish support. Gate: V-FAIL/V-RELEASE.

## D12 — Non-microwave matrix scope

**Accepted selection:** Rotational core with explicit versioned sibling-module handoffs for adjacent spectral domains.

**Problem:** The global method matrix includes infrared/THz, Raman, NMR, UV–visible, mass spectrometry and excited-state workflows, while the TORQ SRS defines downstream geometry/PES and microwave-observable refinement.

**Benefits of expanding TORQ:** Shared geometry/derivative infrastructure and broader multimodal validation. **Cons:** Additional property surfaces, response/excited-state methods, fragmentation dynamics, references and coordination with other CoChem modules. **Risks:** Conflicting module ownership, a much larger release surface, inappropriate state/spin defaults and claiming complete spectra from incomplete property data.

**Recommendation:** Retain every matrix row in the audit and registry migration map. Certify requested IR/THz outputs only where a validated force field and dipole model support them. Treat Raman/NMR/UV–visible/MS and nonadiabatic extensions as separate opt-in profiles with explicit ownership and acceptance manifests; they are not silently activated by a budget tier.

**Decision needed:** Resolved: the full rotational-spectroscopy route is TORQ scope; adjacent domains require explicit sibling handoffs rather than pretending TORQ supplies all spectra. **Interim:** Core microwave/PES implementation proceeds; unselected adjacent profiles remain visibly deferred. Their rows are preserved, not deleted. Gate: V-METHOD/V-PROP/V-BENCH/V-RELEASE.

## Accepted implementation decisions

These architecture selections define the intended design. They do not establish implementation completion, authorize an external calculation or qualify an engine. All twelve ARCH selections are accepted and recorded in the question guide.

| Decision | Accepted design | Remaining risk and evidence |
|---|---|---|
| ARCH-01 | Strict typed native parsers per qualified engine release; explicit property/convergence validity and artifact lineage. | Format changes require authentic release-specific parser evidence; termination alone cannot establish convergence. V-ENGINE/V-PROP/V-FAIL. |
| ARCH-02 | Plan an identifiable genuine additional calculation for a missing property with explicit extra cost; reuse the approved plan/budget and preserve valid earlier typed stages. | A new decision is needed only for a material change to approved physics/cost scope. No silent substitution; changed geometry/method requires a declared, validated composite. Dependent products remain partial/blocked until valid inputs exist. V-PLAN/V-STAGES/V-FAIL. |
| ARCH-03 | Parse native final geometry with explicit optimization convergence evidence and matching energy/atom order. | Stale coordinates, reordered atoms or a final geometry from another attempt invalidate the result; native artifacts need release-specific qualification. V-OPT/V-GEO/V-ENGINE. |
| ARCH-04 | Distinct exit codes and a typed result manifest identify requested-product status while retaining valid artifacts. | Clients must migrate their handling of partial/blocked results; async submission acceptance is not scientific completion. V-E2E/V-FAIL/V-STAGES. |
| ARCH-05 | Explicit engine adapters dispatch only locally validated exact capability tuples. | Engine/version/property coverage requires genuine execution evidence; undocumented tuples remain unavailable and no adapter may silently route to another engine. V-METHOD/V-ENGINE/V-FAIL. |
| ARCH-06 | Explicit workflow profiles and an immutable recorded task DAG identify refinement, discovery and PES expansion within the approved scope/budget. | Profile changes need recorded lineage and budget handling; finite search is not guaranteed global completeness and a partial path is not a completed cascade. V-PLAN/V-STATE/V-BUDGET. |
| ARCH-07 | Start with a local SQLite coordinator and fenced attempts implementing the BASE contract through the local adapter. | Database files remain local to one coordinating host; host-loss recovery and stale-worker rejection need evidence. No shared SQLite writes over NFS. V-STATE/V-RESTART/V-HPC. |
| ARCH-08 | Fix molecular identity and atom count per dataset; incompatible species use separate datasets. | Validate atom identity/order and all property shapes before append; never crop/pad arrays or represent unknown uncertainty as zero. V-SCHEMA/V-STORAGE. |
| ARCH-09 | Independent immutable worker shards plus one merger; the coordinator fences validation and publication of shard/merged manifests. | Reject stale, duplicate, incomplete or incompatible shards. Single-writer/committed-prefix guards remain necessary during staging; their implementation does not establish the full shard/merger workflow. Crash/restart and deployment evidence remain required. V-STORAGE/V-STATE/V-RESTART/V-HPC. |
| ARCH-10 | Engine workers use separate versioned environments; the coordinator/core remains independently installable. | Pin dependencies and interchange versions; test serialization and version-skew rejection. A working core wheel does not qualify a worker environment. V-PACKAGE/V-SCHEMA/V-ENGINE. |
| ARCH-11 | Maintain independently qualified per-engine images, with license-compliant private/site images where required. | Each image needs real installation and engine execution evidence on its named platform. Existing core devcontainer/wheel checks do not qualify engine images; license, driver and image-maintenance constraints remain. V-DEPLOY/V-ENGINE/V-SECURITY. |
| ARCH-12 | Validated QCSchema AtomicResult plus a versioned TORQ spectroscopy bundle. | Validate each schema and cross-layer provenance; advanced observables must not masquerade as standard fields or acquire invented values. Actual consumer tests remain required. V-SCHEMA/V-EXPORT/V-CONSUMER. |

## Release blockers already established

The baseline [implementation audit](Architecture_Implementation_Audit.md) identified fabricated frequency/zero-property paths, incorrect parsing/final-geometry reporting, routing problems, stale lease completion, persistence defects and package/deployment gaps. Follow-up repair evidence determines which findings are closed; accepted design choices do not close them. Remaining defects and the newly selected shard/merger and per-engine image targets require implementation and verification before their capabilities are advertised.

Specification completeness and software completeness are separate. Core implementation can start from this SRS now. Advanced profiles require the implementation and qualification evidence above before the complete planned program can honestly be called research-validated or production-ready.
