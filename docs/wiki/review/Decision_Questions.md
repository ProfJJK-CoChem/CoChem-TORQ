# Decision questions and three-option tradeoffs

Each question below includes context and exactly three suggestions with pros, cons and risks. D01–D12 cover the significant design choices; ARCH-01–ARCH-12 cover implementation/release concerns. The user has already authorized removing fabrication and stale-result defects. Engineering questions choose a long-term strategy; none asks whether to retain an unsafe behavior. All 24 cards now have accepted selections; implementation and profile qualification remain separate work.

Accepted selections: **D01 exact open-source revDSD development; D02 staged engine coverage; D03 advisory ML acceleration first; D04 explicit separate geometry/CP recipes; D05 semirigid VPT2 before separately validated LAM; D06 diagnostics with scientific escalation and bounded same-model recovery; D07 optional separately named electronic analyses; D08 one BASE service contract with a standalone local adapter; D09 Linux CPU before named GPU/Slurm profiles; D10 bounded preregistered benchmark; D11 layered real tests with mandatory engine release gates; D12 rotational core with sibling handoffs.** The entire typed spectroscopy route through high-accuracy identification is also accepted. No-mock/no-fabrication is mandatory. Unselected alternatives and recommendations outside these selections are not inferred approval.

See the [machine-readable selections](Decision_Selections.json), [risk register](Decisions_and_Risks.md), [revDSD and spectroscopy protocol](../RevDSD_Spectroscopy_Protocol.md), and [original implementation audit](Architecture_Implementation_Audit.md).

## D01 — Which engine policy should govern exact revDSD?

The published functional includes an orbital-generation prescription, semilocal/exact exchange, separate OS/SS perturbative correlation and a particular dispersion variant. Combining ordinary HF, MP2 and D4 does not define it.

**Status:** Accepted: develop exact open-source revDSD. The protocol proposes pinned PySCF CPU + DFTD4 building blocks; they are not an existing certified revDSD implementation.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Qualify an existing native implementation first | Shorter route to a reference calculation. | Licensed binaries and version-specific support. | A matching method name does not prove matching orbitals or derivatives. |
| Offer an accurately named open-source alternative | Immediate transparent baseline using existing supported methods. | Does not provide the requested exact revDSD method. | Users could confuse an alternative with equivalent accuracy unless labels are explicit. |
| Develop exact open-source revDSD — SELECTED | Independent, inspectable implementation and long-term control. | Substantial method derivation, derivative implementation and maintenance. | Plausible but wrong response terms; production remains blocked until separate energy, gradient and higher-derivative gates pass. |

## D02 — Which engine coverage should the first research release certify?

Energy, gradient, Hessian and anharmonic support are separate, version-specific capabilities. Historical budget labels do not select an engine.

**Status:** Accepted: qualify an open-source baseline first, then native/reference and high-order profiles individually.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Stage open-source, native comparison and high-order profiles — SELECTED | Independent core progress with honest capability gates. | Some requested profiles qualify later. | Users must see unqualified profiles clearly; initial coverage is narrower. |
| Center the release on one qualified native engine | Smaller adapter surface and consistent execution. | External dependency and fewer independent checks. | Vendor availability and derivative limits can block the complete workflow. |
| Certify open-source, ORCA, CFOUR and MPQC together | Broad choice and cross-comparison from the outset. | Large installation, licensing and benchmark matrix. | A single unsupported derivative tuple may delay the release; names can mask method differences. |

## D03 — How may learned potentials influence conformer selection?

Low ensemble disagreement does not establish physical accuracy or global search completeness.

**Status:** Accepted: advisory acceleration first. No automatic candidate deletion based only on an unvalidated surrogate.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Advisory acceleration first — SELECTED | Quantum verification remains authoritative; a transparent baseline is retained. | Smaller immediate speed gains. | The underlying search can still miss basins; independent challenges and recall measurements remain necessary. |
| Validated pruning in a defined chemical domain | Larger repeated-workload savings. | Requires held-out family benchmarks, calibration and ongoing audits. | False negatives can remove important conformers; disable outside the calibrated domain. |
| Develop general adaptive learning immediately | Potentially strongest long-term scaling. | Substantial data, model and uncertainty research. | Distribution shift and shared model bias may produce confident but wrong pruning. |

## D04 — Which weak-complex geometry and energy protocol should be primary?

CP energies, CP-optimized structures, frozen monomers and fully relaxed geometries are different models with different derivative requirements.

**Status:** Accepted: keep fully relaxed, frozen-monomer and CP variants as separate explicit recipes; no silent substitution.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Explicit separate recipes — SELECTED | Interpretable comparisons and preserved physical identities. | More configuration and calculations. | Residual basis/deformation error still requires measurement. |
| Frozen-monomer screening followed by full refinement | Cheaper exploration of intermolecular arrangements. | Can miss important coupled deformations. | Constrained screening must not be represented as a fully relaxed spectroscopic structure. |
| Develop CP-corrected optimization as a primary route | Directly addresses geometry-level BSSE. | Expensive derivatives and fragment-state bookkeeping. | Inconsistent CP gradients or basis-dependent overcorrection can distort predictions. |

## D05 — In what order should the approved full spectroscopy scope be delivered?

All nine stages through high-accuracy identification are approved. Delivery starts with semirigid VPT2; each later LAM domain requires its own qualification.

**Status:** Accepted: deliver semirigid VPT2, then separately validated LAM. All nine stages remain approved; earlier valid stages survive later-stage failure.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Semi-rigid VPT2, then separately validated LAM — SELECTED | Clear benchmarkable milestones and earlier useful results. | Floppy species qualify later. | A semi-rigid profile must reject unsuitable large-amplitude cases. |
| VPT2 and one defined 1D rotor together | Early coverage of one important flexible-molecule class. | More coupling, mode-selection and symmetry validation. | Double counting and an inappropriate separable rotor model can corrupt corrections. |
| General coupled rovibrational treatment from the outset | Broadest eventual physical scope. | Largest specialist research and compute burden. | Uncontrolled dimensional reduction or PI-symmetry assignment may yield convincing but wrong splittings. |

## D06 — What automation should TS, multireference diagnostics and kinetics receive?

A diagnostic alone does not select an electronic state or justify a new method; a stationary saddle alone does not establish a reaction pathway or rate.

**Status:** Accepted: diagnostics and explicit scientific escalation; only logged, bounded numerical recovery within the approved physical model is automatic.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Diagnostics with explicit escalation — SELECTED | Defensible stop rules and a focused spectroscopy release. | Difficult cases need researcher input. | Diagnostic thresholds themselves need domain validation. |
| One bounded reaction-class workflow | Useful TS/IRC/kinetics within a defined domain. | Additional endpoint, state and rate-model benchmarks. | Wrong connectivity, electronic state or tunneling model invalidates rates. |
| General automated rescue and kinetics | Broad eventual reach. | Very large algorithmic and maintenance scope. | Ambiguous states and method switching can create false reaction networks. |

## D07 — Which electronic-analysis policy should TORQ adopt?

JANPA, NBO and QTAIM define different quantities. Similar scientific purposes do not make them interchangeable.

**Status:** Accepted: optional analyses named for their actual algorithm and engine; NBO requires genuine NBO execution.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Optional separately named analyses — SELECTED | Supports licensed NBO and transparent open alternatives. | Multiple adapters and interpretation guidance. | Different descriptors must not be compared or labeled as identical quantities. |
| Require licensed NBO | Consistent established NBO-specific outputs. | Licensing and provisioning requirements. | Unavailable installations block that profile; descriptors still do not prove a bonding mechanism. |
| Open-source analyses only | Inspectable, redistributable workflow. | NBO-specific outputs remain unavailable. | Users may incorrectly infer equivalent donor–acceptor energies unless nomenclature is precise. |

## D08 — Who owns execution and result commits across BASE, TORQ and SpycFit?

Two controllers cannot safely own one task state. Consumer compatibility must be tested against an actual versioned schema.

**Status:** Accepted: one versioned BASE service contract plus a standalone local adapter, with one task/commit authority per deployment.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| One BASE contract plus standalone local adapter — SELECTED | Shared authority and independently usable TORQ. | Cross-repository versioning and migration work. | Contract drift until real BASE/SpycFit consumer tests pass. |
| Standalone TORQ first, integrate later | Fewer immediate external dependencies. | Later migration and duplicated infrastructure. | Local schemas may diverge from ecosystem expectations. |
| Adopt a workflow framework now | Established scheduling/provenance facilities. | Major migration and operational setup. | A framework does not automatically solve scientific validity or cross-service commit authority. |

## D09 — Which deployment profile should be certified first?

A platform is supported only after actual package, process, storage and engine checks on that platform.

**Status:** Accepted: certify Linux CPU first, then named GPU/Slurm profiles independently.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Linux CPU first, then named GPU/Slurm profiles — SELECTED | Tractable reproducible baseline. | Accelerators and other platforms qualify later. | CPU research costs can be high; untested platforms stay unqualified. |
| Linux CPU plus one NVIDIA/Slurm site | Early high-throughput operation on a concrete installation. | Hardware/site access and a larger test matrix. | Driver and scheduler differences constrain portability. |
| All desktop platforms and multiple GPU stacks together | Broad initial access. | Largest support and qualification burden. | Engine availability and numerical parity differ; release delay is likely. |

## D10 — How should scientific success and accuracy targets be defined?

A universal percentage error is not justified by a thermochemical benchmark, a few molecules or a mean absolute error. Identification also depends on transitions, uncertainties and competing assignments.

**Status:** Accepted: preregister a bounded benchmark and observable targets. Dataset, scope, tolerances and held-out split must be fixed before qualification.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Preregister a bounded benchmark and observation targets — SELECTED | Defensible independent validation by molecule family and observable. | Requires reference-data curation and campaign budgeting. | A small or biased benchmark limits transfer to new chemistry. |
| Choose target molecules and identification tasks first | Directly measures value for your laboratory/astronomy program. | Narrower generality and more experimental integration. | Tuning to those spectra can leak into validation; reserve independent targets. |
| Run a broad blind multi-family benchmark before release | Stronger generality evidence and failure characterization. | Highest reference-data and compute cost. | Heterogeneous experimental quality and state conventions can obscure method error. |

## D11 — How should real test evidence be organized under the no-mock policy?

No fabricated quantum output, mock physics or simulated success is allowed. Exact mathematical tests and malformed/truncated real-file rejection tests are explicitly different from scientific engine validation.

**Status:** Accepted: layered real mathematical/process checks with mandatory real-engine release gates. No mocks or fabricated engine outputs.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Layer real mathematical, process/storage and engine evidence — SELECTED | Fast software regressions plus genuine scientific acceptance. | Several clearly labeled test lanes to maintain. | Passing lightweight checks must never enable an unrun scientific profile. |
| Run all provisioned real-engine integration tests on every change | Immediate broad regression evidence. | High runtime/licensing/resource demand. | Unprovisioned engines must be reported blocked, not represented as passing. |
| Separate continuous software checks from scheduled engine qualification | Practical expensive-engine campaigns with retained artifacts. | Scientific regressions may be discovered later. | Release gates must require recent matching-version evidence rather than stale reports. |

## D12 — Which adjacent spectral domains should TORQ own?

IR/Raman/NMR/UV–visible/MS need different response surfaces, states and models. Shared geometry does not establish complete spectra.

**Status:** Accepted: TORQ owns the full rotational-spectroscopy core; adjacent domains use explicit versioned sibling-module handoffs.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Rotational core plus explicit sibling handoffs — SELECTED | Clear scope and reusable validated geometry/force-field contracts. | Other modules must implement adjacent observables. | Integration gaps require consumer tests and visible unavailable statuses. |
| Add validated IR and Raman alongside rotational spectroscopy | Shares force fields and selected property derivatives. | Additional intensities, response tensors and resonance validation. | Frequencies without a valid property surface do not supply intensities. |
| Make all matrix spectral domains TORQ deliverables | One broad eventual interface. | Much larger program, including excited states and fragmentation models. | Conflicting ownership and incomplete property data can produce misleading full-spectrum claims. |

## ARCH-01 — How should parsers establish property and convergence validity?

The old parser used zero defaults and normal termination as convergence. Strict missingness and convergence rejection are already being repaired.

**Status:** Accepted: strict typed native parsers for each qualified engine release, with authentic artifact lineage and explicit property/convergence validity.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Strict typed native parser per qualified release — SELECTED | Small auditable surface and clear artifact lineage. | New engine versions require authentic fixtures. | Format changes can block valid jobs until requalified. |
| Use a mature parser library behind the same contract | Reuses broader format expertise. | Adds a versioned dependency and translation layer. | Upstream parser defaults or unsupported fields still require independent checks. |
| Cross-check two independent parsers for critical profiles | Stronger disagreement detection. | Duplicate integration and maintenance. | Shared assumptions can survive both parsers; agreement is not physical validation. |

## ARCH-02 — What should happen when a requested property is missing?

Fabricated frequencies/dipoles have been removed; missing properties cannot produce a successful spectrum.

**Status:** Accepted: plan an identifiable genuine additional calculation for a missing property, with its extra cost explicit. Reuse the existing approved plan/budget; a new decision is needed only for a material change to approved physics or cost scope. Preserve valid earlier typed stages; no silent substitution. A changed geometry or method requires a declared, validated composite. This design choice does not certify a backend.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Keep valid typed stages and block dependent products | Preserves useful science without invented downstream results. | Clients handle partial results. | A client must not promote partial output to a validated catalog. |
| Fail the entire requested product while retaining raw artifacts | Simple strict client contract. | Users may need another request to access valid earlier observables. | Useful results may be overlooked, though they remain recoverable. |
| Run an explicitly requested separate calculation for the missing property — SELECTED | May complete the intended product. | Additional cost and scheduling. | Method/geometry mismatch must be labeled composite; no silent substitution. |

## ARCH-03 — How should final optimized geometry be established?

Copying submitted coordinates into an optimized result is invalid. Actual final geometry and optimization evidence are required.

**Status:** Accepted: parse the native final geometry with explicit optimization convergence evidence, preserving energy and atom-order correspondence.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Parse native final geometry plus convergence evidence — SELECTED | Direct, inexpensive traceability. | Release-specific geometry parsing. | Final coordinates must match the returned energy and atom order. |
| Also reevaluate final gradients independently | Stronger stationary-point evidence. | Extra quantum calculation cost. | Independent settings must match; a small gradient alone does not prove a minimum. |
| Use a separately qualified external optimizer | Uniform optimization orchestration and transparent steps. | New coordinate/constraint/gradient integrations. | Incorrect forces, atom mapping or thresholds invalidate the optimization. |

## ARCH-04 — How should command-line clients expose partial or blocked work?

Failure must not write an input structure labeled optimized or return a complete-success exit code.

**Status:** Accepted: distinct exit codes plus a typed result manifest identify success, partial, blocked and failed outcomes while retaining valid artifacts.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Distinct exit codes and a typed result manifest — SELECTED | Reliable batch automation and retained partial artifacts. | Consumers must handle more than success/failure. | Undocumented scripts may need migration. |
| Single nonzero exit for every incomplete requested product | Simpler compatibility contract. | Less immediate classification by exit code. | Automation must inspect the manifest to distinguish retryable and scientific failures. |
| Submit-and-query CLI with durable task status | Suitable for long workflows and resumability. | Requires stable task service and status commands. | Submission success must never be confused with scientific completion. |

## ARCH-05 — How should engine dispatch be qualified?

The old pipeline could resolve CFOUR but invoke ORCA. Exact requested method/property/engine identity must govern dispatch.

**Status:** Accepted: dispatch through explicit adapters with locally validated exact capability tuples; unavailable tuples remain blocked.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Explicit adapters with locally validated capability tuples — SELECTED | Clear supported surface and early rejection. | Qualification effort per tuple. | Sparse coverage initially; undocumented tuples remain unavailable. |
| Adopt a mature engine-harness interface | Reuses process and schema handling. | Additional dependency and compatibility layer. | Harness support does not certify all derivatives or conventions. |
| Plug-in adapters supplied by sites | Flexible licensed/HPC installations. | Authentication, versioning and plugin conformance work. | A site plugin can misreport capabilities unless real tests are mandatory. |

## ARCH-06 — How should refinement, discovery and PES searches be separated?

Refining supplied candidates must not silently launch GOAT/CREST or imply a complete cascade after one step.

**Status:** Accepted: use explicit workflow profiles and an immutable recorded task DAG; refinement, discovery and PES expansion remain identifiable planned work within the approved scope and budget.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Explicit workflow profiles and an immutable planned DAG — SELECTED | Predictable cost, lineage and resume behavior. | More planner/schema work. | Profile transitions need clear changed-method and budget records. |
| TORQ refinement only; TOPOS owns discovery | Clear ownership and small core. | Requires upstream discovery integration. | Discovery-quality gaps remain upstream and must be visible. |
| Optional TORQ-managed discovery adapter | Convenient single interface. | More engines, search settings and provenance to qualify. | Users may interpret finite heuristic search as guaranteed global completeness. |

## ARCH-07 — Which long-term task authority should enforce fenced leases?

Stale-worker completion is an integrity defect. Opaque tokens and monotonic generations are being enforced now.

**Status:** Accepted: start with a local SQLite coordinator and fenced attempts behind the BASE service contract/local adapter. Keep the database on local storage; no shared multi-host SQLite writes over NFS.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Local SQLite coordinator with fenced attempts — SELECTED | Simple transactional authority and testable recovery. | No shared multi-host SQLite writes over NFS. | Host loss requires durable backup/reconciliation and bounded lease expiry. |
| BASE service with a server database | Central multi-host authority and access control. | Service deployment and migration. | Network partitions and retry semantics still require fencing/idempotence. |
| Workflow platform database as sole authority | Established scheduling integration. | Framework coupling and adapter changes. | External artifact commits can still escape the framework transaction without a commit protocol. |

## ARCH-08 — How should changing molecular sizes be represented in persistent data?

Silent atom cropping/padding and unknown uncertainty represented as zero are prohibited. Strict shape checks and explicit missingness are implemented.

**Status:** Accepted: keep molecular identity and atom count fixed per dataset; reject incompatible species or shapes before append and store them in separate datasets.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Fixed identity and atom count per dataset — SELECTED | Simple validation and efficient arrays. | Many species need separate datasets. | Incorrect grouping must be rejected before append. |
| Versioned ragged arrays with offsets and atom identifiers | Efficient mixed-size ensembles. | More indexing/schema complexity. | Offset or atom-map corruption can silently mix species without validation. |
| One immutable artifact per molecule/result | Strong isolation and lineage. | More files and aggregation work. | Large campaigns need manifest/index lifecycle management. |

## ARCH-09 — What persistence model should support concurrent workers?

SWMR permits one writer. It is not a transaction protocol and cannot justify globally disabling HDF5 locks.

**Status:** Accepted: independent immutable worker shards plus a merger, with coordinator-fenced manifest publication. Single-writer guards remain necessary while writing each shard or merged artifact; those repairs do not establish completion of the full shard/merger design.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| One local writer with committed-prefix publication | Practical current repair and consistent reader snapshots. | Serialized writes and explicit crash recovery. | A killed writer can still leave an HDF5 file requiring offline verified recovery. |
| Independent immutable worker shards plus a merger — SELECTED | Workers cannot corrupt the same file. | Merge/index stage and more artifacts. | Duplicate, incomplete and stale shards require fenced manifest publication. |
| Transactional metadata with immutable object storage | Suitable for multi-node scale and durable lineage. | Larger service and deployment scope. | Cross-store commit/recovery is still nontrivial; object availability is not scientific validity. |

## ARCH-10 — How should installation dependencies and optional engines be packaged?

A checkout import is not proof that a wheel works. The old wheel omitted runtime packages and required undeclared sibling imports.

**Status:** Accepted: run engine workers in separate versioned environments; a small coordinator/core remains independently installable. Pin each environment and validate the worker interchange contract.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Small installable core plus explicit extras | Auditable installs and optional engine independence. | Maintain dependency profiles and clean-wheel checks. | Missing extras must produce clear unavailable states rather than import-time failure. |
| One fully provisioned research environment | Convenient supported lab image. | Large images and dependency conflicts. | Binary license/redistribution terms and platform variants constrain distribution. |
| Engine workers in separate versioned environments — SELECTED | Dependency isolation and reproducible per-engine execution. | More orchestration and interchange work. | Serialization and version skew create additional integration failure modes. |

## ARCH-11 — How should cloud/devcontainer reproducibility be delivered?

The old setup referenced a nonexistent requirements file and user and had no tracked CI. Configuration repairs do not prove that an unrun container build passes.

**Status:** Accepted: maintain independently qualified per-engine images, using license-compliant private/site images where needed. Each image requires real installation and engine tests before qualification; existing core devcontainer/wheel evidence does not qualify engine images.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Maintain a clean core devcontainer and CI lane | Fast reproducible software development and visible results. | Engine qualification remains separate. | A green core lane cannot certify absent quantum programs. |
| Maintain per-engine qualified images — SELECTED | Repeatable native-library combinations. | Many images and ongoing security/version maintenance. | Licenses may prevent public image distribution; host GPU changes still matter. |
| Maintain native lockfile/setup profiles instead of containers | Works on restricted HPC sites. | More host-specific setup instructions. | System libraries and schedulers can vary despite Python locks. |

## ARCH-12 — Which interchange policy should be authoritative?

An arbitrary dictionary named qcschema_output is not schema conformance. Missing energy cannot become zero.

**Status:** Accepted: use validated QCSchema AtomicResult for representable electronic results plus a versioned TORQ spectroscopy bundle for advanced products and lineage. Each payload must satisfy its declared schema; missing values remain absent.

| Suggestion | Pros | Cons | Risks |
|---|---|---|---|
| Validated AtomicResult plus a versioned TORQ spectroscopy bundle — SELECTED | Standard electronic data with honest advanced extensions. | Two documented schema layers. | Consumers must understand which schema governs each quantity. |
| Use QCSchema exclusively where representable | Broad standard tooling compatibility. | Some spectroscopy/lineage data need carefully namespaced extras. | Overloaded extras can become an undocumented second schema. |
| Use a TORQ-native schema with explicit standard adapters | Full domain expressiveness. | More adapter and consumer maintenance. | Calling native results QCSchema without validation must remain impossible. |
