# Release candidate readiness and exact closure gates

Reviewed 2026-10-08. **The complete SRS is not yet qualified for release.** The final complete local regression at `e77e2a4` passed 3034 tests with zero failures/errors/skips, audit and supervising process exit zero, and unchanged tested source. Both final installed-wheel core lanes passed 952 tests each. The preceding complete regression at `57c81eb` and its CLI/notebook and sibling checks remain historical evidence. The deployed student journey and independent scientific qualification remain incomplete. The [aggregate results](release_candidate_check_results.json) distinguish local implementation evidence from release qualification; neither source presence nor a software regression establishes scientific accuracy.

The later main commit
[`5c1da3a`](https://github.com/ProfJJK-CoChem/CoChem-TORQ/commit/5c1da3a15be1dd6fe8a34c418c739391bfaed44e)
passed **all eight hosted jobs** in
[run 37805501782](https://github.com/ProfJJK-CoChem/CoChem-TORQ/actions/runs/37805501782):
both installed-core interpreters, CPU checks, genuine native-engine checks,
ecosystem consumers, interface, offline calculation image and complete repository
regression. The [final main evidence](release_candidate_evidence/main-5c1da3a-base-12f5475/summary.json)
retains the actual API job/step metadata and its original digest. The uploaded
archive download was denied at `productionresultssa7.blob.core.windows.net`;
its bytes and exact hosted case counts remain unverified. The older local
3034-case count is not relabeled as a downloaded hosted count.

The first hosted image built and passed native derivatives, then failed on a
read-only home directory. An intermediate attempt exposed a scratch-filesystem
mismatch. Allocation state now uses writable `/tmp`, and calculation invocations
set `TMPDIR=/results` to retain the strict staging-filesystem check. The native CI
lane installs its existing locked interface prerequisites. Tier tests restore all
actual environment selectors; wall/evidence-loss tests observe and pause genuine
bound processes before exercising production guards. Earlier failed runs remain
preserved. These image/CI/test changes leave application Python bytes unchanged.
The final successful hosted source SHA-256 is
`64dcd2e3dd7c958396c39685b4511deae7f9023dd3be9c92bec4b2fd82ec73bf`;
the local full-suite receipt below remains tied to `e77e2a4`.

The [full-SRS audit](full_srs_requirement_audit.json) covers all **73 normative requirements, 41 acceptance definitions and 140 original historical method-matrix row IDs**. Its current inventory is **56 partial supported scope, one missing full implementation, eight missing independent qualification and eight optional profile/provisioning**. No requirement is classified as implemented and qualified. All 41 complete-scope acceptance gates remain blocked, and the release flag remains false. [Student release scope](student_release_scope.md) provides the complete reconciled requirement-to-code/check map; links are coverage pointers rather than passing gate evidence.

The supplied-candidate workflow already computes genuine bounded restricted electronic structure, optimized geometry, isotope-specific equilibrium constants, harmonic results and finite-J rigid-rotor screening using actual dipoles. Other explicitly named local validation profiles compute actual numerical derivatives, bounded vibrational force fields/VPT2, ghost-basis interaction quantities and saddle/downhill endpoint candidates. New bounded geometry/PES profiles retain supplied-graph mapping, nuclear-position symmetry proposals, nonperiodic fixed valence-coordinate observations, actual prior-density continuation, adaptive physical anchors, constrained-HF residual/KKT/curvature evidence and separately approved target minimum checks. Genuine native points bridge to sole-merger HDF5 storage under current committed coordinator admission. Canonical nonresonant Watson and a separate local Dirichlet variational route retain bounded operator/applicability evidence; default HF/STO-3G water fails the reviewed coupling gate and keeps model B0/alpha/semirigid outputs unavailable. Missing later scientific stages retain earlier results and explicit reasons. Exact revDSD, general B0/rovibrational spectroscopy, calibrated identification accuracy and automated completeness claims remain disabled.

The final tested code is `e77e2a47d4ef23bb4ad6db72519e656336cf0bb2`,
with complete source SHA-256
`a60885f50fe07dab95a4dbbd0ddab4c0113f7fc93edd689c6f73896c025fc95c`
and application SHA-256
`fba071ddb96f091cfe6f94ec5872aa40d59421b5f1ffe7025e94f90a8dc3d4a9`.
Its [current complete local evidence](release_candidate_evidence/e77e2a4/aggregate.json)
records **3034 passed, zero failures/errors/skips**, the actual complete-suite
command without selected-name/marker/path exclusions, and 1743.275 seconds
elapsed. Pytest, the audit and the owned supervising runner each exited zero,
with the full source digest unchanged before and after. The
[evidence index](release_candidate_evidence/e77e2a4/evidence-index.json) binds
sanitized actual JUnit, environment and completion receipts, unchanged originals
by digest, and eight selected verified genuine native inventories. Selected
native bundles are evidence samples, rather than the entire calculation archive.

Both final installed-wheel core lanes passed **952 tests each** on Python 3.10
and 3.12 with zero failures/errors/skips, unchanged final source and 56 deliberate
marker deselections per interpreter. The final wheel SHA-256 is
`7c0a5450c64d7e7c6dc0de568a46794b782ca75ad1e5eb5afb5d8db50711bf13`;
all **249 packaged Python sources** match the final checkout byte for byte.
Core and scoped lanes overlap the complete suite and are not added to its total.
The additional genuine external-Pickett provision and reviewed modern-TOPOS
handoff receipts report 52 and nine passing cases respectively, with zero
failures/errors/skips. Those scoped results do not establish general consumer
conventions, independent molecular accuracy or a live canonical student platform.

BASE main is now
[`12f5475`](https://github.com/ProfJJK-CoChem/CoChem-BASE/commit/12f54759ea5ae64843fa2b2bcce111fc2a5b38e1).
Its complete canonical local suite executed **2345 cases: 2337 passed, eight
skipped, zero failures/errors**. The original Pytest process was waited and
exited zero; source and HEAD stayed unchanged. The eight genuine prerequisites
require isolated MACE/OFF24 models, licensed ORCA/reference/union acceptance,
or physical Slurm. They remain unqualified. A fresh genuine eleven-phase Stage 0
and strict current-schema registry passed with `DEGRADED_OPERATIONAL`; no missing
scientific engine or model was invented. The actual installed wheel matches its
reviewed source and complete payload. Its SHA-256 is
`53ffe13e7ca0b9f4945f308bb532eec4a40df3bf184e3846f75ed8ded4c89118`.

The same published BASE source passed **all nine bounded hosted jobs** in
[run 37812840001](https://github.com/ProfJJK-CoChem/CoChem-BASE/actions/runs/37812840001):
source integrity, installed wheels and CI controls across Linux/macOS/Windows,
and the bounded xTB/PySCF derivative/dashboard route. The new public receipt
records actual job/step metadata; remote artifact bytes and case counts remain
unverified. The full local suite above retains its own source-bound receipt.

Only six BASE jobs consuming private staged drafts receive `contents:write`
and `actions:read`; validation and workflow defaults retain read-only authority.
The source/wheel/catalog, completed current full suite and preceding failed
registry/provider/disk attempts are bound in the
[public evidence index](release_candidate_evidence/main-5c1da3a-base-12f5475/evidence-index.json).
The unchanged disk guard was satisfied by deleting only owned closed build copies,
then executing setup again in a new root. The previous 1905-pass BASE run and
separate real HTTP/widget/kernel lifecycle remain historical scoped observations.

The [student handoff](student_project_handoff.md) identifies a concrete remaining
integration blocker: TOPOS's standalone licensed callers still use the old BASE
asset contract. Their complete owning-project receipt, workflow/source/task,
scientific-request/resource and permission migration requires separate source,
wheel/catalog and native/provider qualification. A source-pin substitution cannot
close it. Students use BASE's dedicated private personal ORCA/CFOUR workflows;
the full coupled licensed route and genuine private student pilot remain incomplete.

The initial expanded full attempt executed **2990 cases: 2987 passed and three
failed**, with zero errors/skips. Pytest and supervisor exited one; the audit
exited zero. The three genuine Git-ignore coverage/semantics/policy failures
and the earlier two failed Watson serialization/missingness checks remain in
[failed-attempt history](release_candidate_evidence/e77e2a4/aggregate.json).
Their original bytes remain bound in the new index. Subsequent fixes and the
separate successful final complete run preserve this history.

The preceding complete-repository candidate was `57c81eb7d699c6717926eeaab25ba0a0a59ca0f9`,
with source SHA-256
`28aff57e17153244db570058e570e47085707337c3ea5ce7b39d8ba4c8de032b`.
Its [complete local repository receipt](release_candidate_evidence/57c81eb/record.json)
records **2486 passed, zero failures/errors/skips**, no marker or name exclusions
and 1122.272 seconds elapsed. Pytest, the inventory audit and the supervising
runner each exited zero, with the source digest unchanged before and after.
Its selected clean installed-wheel core checks passed **880 tests each** on
Python 3.10 and 3.12 with zero failures/errors/skips, 56 deliberate marker
deselections per interpreter and unchanged source. The 232 Python wheel files
matched installed bytes; Ruff/format passed in the 42-file modern
application/coordinator/storage/evidence-tool scope, and configured strict mypy
passed for 36 modern application source files. These are not repository-wide
legacy lint/typing qualification. Legacy GOAT/CREST CLI checks have distinct
syntax/undefined-name and integrity scope, while legacy path/topology strict
typing remains unqualified. Selected lanes overlap the complete repository run;
their counts are not added to its total.

The genuine local CLI exercised 16 command gates, six actual H2 scan calls and a
separately approved target optimization with six optimizer evaluations and an
actual Hessian. It preserved the original candidate and registered a separate
model-minimum result. `criteria_met` is a finite-design numerical stop: its surface
still had one uncomputed design node and status `partial`. The preceding installed
wheel also passed authenticated local notebook start/check/stop, reviewed notebook
retrieval with HTTP 200 and unauthenticated rejection with HTTP 403. Genuine
BASE manifest `fd93301136729913c494ce9b9500c6c96e45f931`, SDK `1a3c633` and
TOPOS `6a01b0f` installed the reviewed TORQ bytes and ran supplied-water geometry
handoffs with unchanged source/environment/dependency receipts. None of these
local checks establishes the live canonical platform or molecular accuracy.

The genuine historical [80767e8 baseline record](release_candidate_evidence/80767e8/record.json)
reports **2085 passed, zero failures/errors/skips**, with unchanged tested source
`56754994707b5107f953ff33bb8327034d895a000bd534d7ed57f3a0b3db9cd9`.
That complete-repository execution predates these new geometry/PES/precursor
implementations. A later `74ffc8b` pytest JUnit reported 2274 passing cases, but the
supervisor ended before the original driver's completion/curation/source-after
receipt; its durable repeat was intentionally interrupted after the isotope
defect was found. Neither is current final qualification. The older 1438-test
collection and 88-file narrow AST observations are
historical policy/collection records, not current execution or scientific evidence.

The historical `7296c37` / source
`a2ee60585f632196a75dbcc7afe21dba7403896ddc46173a02ab72e90557e693`
complete run executed 2486 cases with one failure and zero errors/skips. The old
surrogate test expected `ValueError("energy unavailable")`; the accepted advisory
retention behavior instead preserves the original candidate and unavailable
energy as `None`. The corrected test checks the original object, complete model
JSON and preserved `None`, without changing the 232 packaged Python files or
application digest. The failed receipt remains unchanged and historical. The
separate completed `57c81eb` receipt records the subsequent passing run against
the corrected test/source identity.

## Bounded implementations and remaining full-scope closure

Six formerly missing requirements now have partial supported scope. The complete
`TORQ-VIB-003` scientific/domain implementation and independent qualification remain incomplete. The following closure conditions
preserve the unchanged normative SRS and accepted method matrix; unsupported
constrained or general profiles remain required work, not newly optional scope.

| Requirement | Implemented bounded scope | Remaining work and qualification |
|---|---|---|
| `TORQ-GEO-001` Mapping and alignment — partial | [Supplied-graph comparisons](../../src/cochem_torq/geometry_identity.py) validate explicit full-atom topology, isotope/charge consistency and supported asserted tetrahedral/E–Z stereo through genuine RDKit. Each comparison retains both complete original indexed inputs and digests, explicit atom maps, proper SO(3) alignment and thresholds. Unassigned stereo, truncation and ambiguous mappings retain candidates. | General graph/stereo/conformer identity, enhanced relative/non-tetrahedral/atropisomer stereo, constraint-aware/coupled geometry and independent domain/recall qualification remain incomplete. XYZ cannot supply an invented graph or stereo assignment; fingerprints cannot certify equivalence. |
| `TORQ-GEO-002` Symmetry — partial | [Nuclear-position proposals](../../src/cochem_torq/geometry_identity.py) record proper/improper actions, permutations, residuals, closure and enumeration completeness over declared tolerance sweeps. Separate isotope and authentic mass actions are retained. The ≤24-atom profile assigns only C1/Cs/Ci/C2/C2v/C2h/D2 when rank/completeness/closure permit; other or linear groups remain unassigned. | General classification, actual engine subgroup, dynamically feasible PI groups, spin weights and tunneling rules are absent. Nuclear-position actions do not establish chemical graph automorphisms or feasible nuclear dynamics. A separate bounded HF SLSQP adapter now computes actual constraint residuals, KKT/tangent gradients and optional Lagrangian-nullspace curvature. General engine/constraint-domain qualification and an unconstrained-minimum proof remain required. |
| `TORQ-PES-001` Scan definition — partial | [Approved immutable scans](../../src/cochem_torq/scan.py) declare atom indices, units, domains, finite design/pass order, state, constraints, initial-guess policy and actual-call budget. Bounded closed-shell RHF/STO-3G H2 and nonperiodic fixed bond/angle/dihedral profiles retain actual coordinate residuals, forward/reverse/challenge observations and exact prior-density versus independent-restart policies without unproved symmetry division. | Standalone constrained HF optimization exists with native residual/curvature evidence, but does not imply general relaxed PES execution. Periodic/multistate/relaxed and general coupled-domain applicability, branch completeness and canonical-platform qualification remain outstanding. |
| `TORQ-PES-002` Surface identity — partial | [Immutable point/surface contracts](../../src/cochem_torq/scan.py) bind exact recipe/state/source, native inventories, pass lineage, absolute electronic energies and actual AO density records. Same-geometry independent restarts expose observed energy/density discrepancies. Failed or colliding observations stay explicit. | Actual previous-checkpoint orbital/density consumption in a fresh SCF and a genuine native-point-to-HDF5 bridge now exist under exact committed lineage. Neither establishes branch completeness. General multistate/periodic/relaxed/composite surfaces and independent applicability/deployment campaigns remain required. |
| `TORQ-PES-003` Candidate versus stationary point — partial | [Ledger-linked refinement](../../src/cochem_torq/adaptive.py) preserves sampled extrema as quarantined input-only candidates and verifies original point/native bytes, retained CAS revision and exact separately approved target/state/constraint release. Genuine target optimization, gradient and positive projected Hessian/invariance checks can produce a separate model-minimum result. | General scan/ML/path candidates and stationary/TS domains remain incomplete. Relocated native evidence needs authenticated import. A local model minimum does not establish chemical accuracy, a verified TS/IRC or global completeness; mapped endpoint connectivity needs its own evidence. |
| `TORQ-PES-006` Adaptive algorithm — partial | [Bounded physical acquisition](../../src/cochem_torq/adaptive.py) uses approved seeds, genuine anchors, deterministic maximin diversity, fresh endpoint/frozen interior challenges and actual coordinator budgets. Numerical interpolation is distinctly labeled; `criteria_met`, `budget_exhausted`, `insufficient_coverage` and `failed` remain explicit. No automatic pruning occurs. | No trained surrogate, continuous/between-node error bound, global recall/completeness calibration, grouped molecular-family benchmark or general multidimensional/periodic/relaxed loop exists. `criteria_met` describes finite-design spacing and observed numerical residuals only, not chemical accuracy or complete basin search. |
| `TORQ-VIB-003` VPT2 profile — missing full implementation | Bounded [canonical nonresonant Watson perturbation](../../src/cochem_torq/spectroscopy/rovibrational_perturbation.py), leading alpha/model B0 and unreduced harmonic quartic response now use strict source/isotope/frame/mode/parent identities and applicability gates. Actual default HF/STO-3G water retains B0/model alpha/semirigid as unavailable because its cubic coupling exceeds the reviewed limit. | Full resonant GVPT2/polyads, A/S reduction-induced shifts, higher distortion/hyperfine/tunneling and independent calibrated spectroscopy/identification remain incomplete. A separate local Dirichlet variational solver does not supply periodic large-amplitude dynamics; harmonic/inertia averages cannot substitute. |

Relevant implementation checks are
[geometry identity](../../tests/test_geometry_identity_integrity.py),
[fixed PES scans](../../tests/test_pes_scan_integrity.py),
[adaptive acquisition/refinement](../../tests/test_adaptive_pes_integrity.py) and
[rovibrational precursors](../../tests/test_rovibrational_precursors.py).
The scoped local observations report 35 geometry mathematical/RDKit checks,
nine genuine-engine scan cases within 29 scan checks, and 24 adaptive checks,
including actual target optimization/Hessian verification and budget/failure/
coverage stops. These overlapping observations are included in the independently
recorded complete repository run; they are not added to its total and do not
pass any full-scope acceptance gate.

Use the [geometry/PES guide](pes_and_geometry_validation.md) for exact current
local APIs, declared-input examples and separate target approval/refinement
commands. The [spectroscopy guide](spectroscopy_implementation.md) records the
geometric precursor contract and its typed unavailable higher products.

## Implemented local HDF5 ownership contract

`TORQ-OPS-005` is now partial supported scope. The [fenced HDF5 implementation](../../cochem/storage/fenced_pes.py) and [protocol](../../cochem/storage/FENCED_PES_PROTOCOL.md) seal independent supplied or explicitly dimensionless mathematical samples, verify private source snapshots and current committed source admission, and publish a new version through one coordinator-authorized merger with atomic no-replace installation. Real HDF5/SQLite files, competing Linux processes and an abrupt process exit after rename but before SQLite commit exercise ownership, preserved-byte same-owner recovery and elapsed lease/approval expiry or revocation rejection. Current local test pointers are included in the audit; final source-bound release execution receipts remain separate.

A genuine bounded scan-to-storage bridge now admits exact native point/native-manifest identities through current coordinator receipts, preserving absolute energies, units, atoms and electronic state. Its genuine local observations do not establish general molecular PES accuracy. Source-schema interoperability, deployed filesystem/NFS locking and durability, HDF5 filter availability and cross-host recovery remain outstanding. Original files are retained, expired/revoked owners cannot recover an orphan as success, and bare artifact presence never supplies coordinator authority. SWMR is not multiwriter safety.

## Integrity corrections in the release candidate

Publication now rechecks the current worker lease and campaign approval after the actual publication callback, so elapsed expiry cannot commit stale success. Native capability/restart evidence verifies actual nested optimizer/final-calculation provenance, parent-retained byte inventories and matching source/recipe/engine identities; compatibility alone never asserts engine consumption or transferable scientific accuracy. The separate genuine checkpoint adapter now loads exact saved orbitals, forms the observed initial density and proves consumption at the first callback of a fresh SCF; this is not iteration/DIIS, optimizer or higher-derivative algorithm-state resume. File snapshots and race checks preserve admitted source bytes rather than accepting a concurrently changed input.

Scientific missingness also remains explicit: unavailable isotope masses/radii, harmonic frequencies, physical rotor constants and required molecular inputs cannot acquire invented substitutes. Legacy rotor eigenstates retain their mathematical identity without guessed Ka/Kc assignments; unsupported reduction/intensity catalogs fail rather than using arbitrary intensities. Dimensionless storage/DVR/linear-algebra checks retain their mathematical evidence class and cannot be relabelled native molecular observations. These corrections improve integrity; they do not establish unrun scientific or deployed qualification.

The legacy ingestion boundary now distinguishes requested tabulated isotope
masses, actual standard atomic weights for unspecified selection and explicitly
supplied masses. An absent requested isotope cannot acquire ordinary atomic
weight or an integer mass-number estimate; contradictory supplied masses reject.
Path/IPC and geometry helpers preserve that identity. Radius-based topology is
an unverified geometric proposal using the named database fields. Legacy cascade
documents are planning-only and are rejected before runtime defaults. GOAT/CREST
metrics remain advisory: unknown graph/stereo/state candidates are retained, and
unverified molecular populations/entropy remain unavailable. Actual declared-state
mathematics is separately labeled. These repairs do not qualify general chemical
identity, conformer completeness or the historical method matrix.

## Current local operations and source-bound milestones

The final Python application at `e77e2a4` retains source digest
`fba071ddb96f091cfe6f94ec5872aa40d59421b5f1ffe7025e94f90a8dc3d4a9`.
Seventeen genuine shared-host/native dispatch cases passed with that digest
unchanged before and after: real competing campaigns, owned SIGKILL/waits,
SQLite-blocked exit races in both dispatchers, retained scratch-failure reasons,
zero engine consumption before dispatch, independently observed constrained-child
termination and missing whole-inventory retention. The separate source-activated
storage/process lane passed 100 cases covering host allocation, WAL-consistent
coordinator backup/fenced restore and campaign lifecycle. These bounded lanes
are not a complete-repository or independent scientific qualification; overlapping
counts must not be summed with any final repository run.

[Coordinator recovery](campaign_backup.md) revokes all restored approvals and
active leases, increments generation fences and preserves historical outcomes
and unknown physical costs in a new offline target. It supports exactly known
v0-to-v1 metadata adoption; generic future migrations are not implemented.
[Shared local host allocation](host_allocation.md) is additional campaign
resource authority, with actual host/boot/actor/process binding and atomic
cross-campaign CPU/RAM/scratch/worker/probed-GPU admission. Missing launches,
expired leases and missing/empty pinned constrained inventories remain held.
A campaign restore neither clones the host ledger nor proves original workers
or independently sessioned native children dead.

Both services remain bounded local Linux deployment implementations. Separate
users/namespaces/ledger roots, positive GPU qualification, remote scheduler
coordination, exact lifetime resource peaks and canonical-platform operational
qualification need independent evidence. None of these milestones enables an
unqualified named method or calibrated astronomical/laboratory identification.

## Independent scientific qualification

The eight requirements classified as missing independent qualification are:

- `TORQ-METHOD-004`: exact revDSD primary-source recipe reconciliation, authenticated independent energy components, orbital-response gradients and higher derivatives. This includes unfinished scientific implementation, not merely a missing citation.
- `TORQ-GEO-004`: actual diagnostics and independent domain validation for advertised open-shell, multistate or near-degenerate chemistry; the bounded restricted-state evidence does not transfer automatically. New genuine [HF/GKS internal and restricted-to-unrestricted stability checks](../../tests/test_double_hybrid_stability_integrity.py) preserve the original constrained-reference energy/orbitals and unstable status. They do not establish a correlated double-hybrid minimum or exact-method qualification.
- `TORQ-VIB-001`: method/state/basis-specific derivative-convergence and independent molecular frequency/reference checks for each enabled derivative profile.
- `TORQ-VIB-005`: independent distortion/hyperfine, dipole-surface/vibrational-averaging and identification-band validity for the requested spectral properties.
- `TORQ-SPEC-002`: independent accuracy evidence for every rung of the spectroscopic ladder and held-out observable/transition uncertainty.
- `TORQ-UQ-002`: real reference curation, accepted numerical/predictive targets, immutable preregistration and blind held-out results. The current proposal's incomplete references/targets cannot support calibrated claims.
- `TORQ-VAL-001`: the complete named core campaign, including nonlinear/linear molecules, isotopologue pairs, a real weakly bound dimer, and actual cancellation/restart/installation-failure paths in the supported domain.
- `TORQ-VAL-002`: genuine representative advanced cases and independently preregistered domain benchmarks for each enabled advanced profile, with failures and applicability limits retained.

The 140 historical matrix identities remain documented choices. Exact capability tuples distinguish `documented`, `locally_validated`, `experimental`, `unsupported` and `unknown`; the built-in advanced tuples remain controlled-validation only. Raw native calculation evidence is bound to source, recipe, geometry/state, engine version and actual hardware. That narrow local evidence is not transferable molecular-family accuracy or a universal engine-support guarantee. See [capability and restart contracts](capability_and_restart_contracts.md).

Exact open-source revDSD development follows the [DH0–DH6 protocol](../wiki/RevDSD_Spectroscopy_Protocol.md) and [implementation/source evidence](revdsd_implementation.md). Original orbital-generation, P86 variant, frozen-core and dispersion definitions still require reconciliation with the original primary recipe and independent native components. Custom double-hybrid machinery does not authorize silently selecting secondary coefficients or enabling a named exact recipe.

The eight individually unavailable optional profiles are `TORQ-PES-007` kinetics/bifurcations, `TORQ-VIB-004` coupled LAM/torsion-rotation/VRT, `TORQ-ANA-001` wavefunction interchange, `TORQ-ANA-002` separately named NBO/JANPA/localization analyses, `TORQ-ANA-003` density analyses, `TORQ-EXP-002` genuine SPCAT/SPFIT, `TORQ-EXP-004` actual SpycFit HDF5 consumption, and `TORQ-DEP-003` additional GPU/Slurm/platform profiles. Each needs its own implementation, provisioning where applicable, and genuine scientific/platform gate. Optional status does not make the core rotational route optional.

## Canonical student environment and release evidence

GitHub Actions is the calculation authority, and GitHub Codespaces is the interface environment. CoChem-BASE and CoChem-TOPOS remain separately provisioned repositories with actual pinned versions and genuine producer/consumer tests. A final release needs the following evidence in addition to local numerical/software checks:

1. **Frozen final source:** archive genuine nonempty execution receipts/JUnit and engine artifacts against the complete tested source/configuration identity. Keep mathematical/process, real-engine, independent-reference and platform evidence distinct. A broad test suite is not a substitute for a required scientific gate; unrun/blocked stages stay unrun/blocked.
2. **Default-branch calculation workflow:** make the reviewed `calculation.yml` endpoint available on the repository's default branch and verify an authenticated actual dispatch. Earlier hosted integrity runs on a different commit do not activate that endpoint or qualify later edits.
3. **Live Codespaces journey:** provision an actual supported Codespace and execute import, inspection, observable/recipe selection, exact-plan approval, submit, status/cancel, result download/digest verification, scientific quality review and eligible export. The latest recorded machine-selection request was HTTP 403; no live instance was provisioned. Source/UI presence does not close this gate.
4. **Artifact download:** retrieve original hosted artifact bytes and verify immutable IDs, manifests and digests. Hosted upload and API metadata are genuine earlier evidence, but this session's storage redirect was blocked at `productionresultssa12.blob.core.windows.net`. The saved environment draft allowance is not an applied runtime policy; independent download remains unrun.
5. **Final sibling lifecycle:** verify the final TORQ pin with compatible genuine BASE/TOPOS environments, actual task-service ownership where advertised, cancellation/resume semantics and consumer-side conventions. Prior sibling checks remain tied to their tested revisions.
6. **Deployed operations/security/storage:** qualify current leases/approvals through publication, resources across campaigns, target filesystem/NFS semantics, crash/restart/reconciliation, sealed backup/restore, artifact retention/access controls and legitimate licensed-engine provisioning. Local SQLite/process/HDF5 checks alone do not qualify every deployed platform.

The [current source-bound evidence index](release_candidate_evidence/e77e2a4/evidence-index.json) and [aggregate results](release_candidate_check_results.json) retain the final completed local receipts, unchanged historical/failed attempts and unresolved release gates. The [preceding completed local index](release_candidate_evidence/57c81eb/record.json) remains historical. The [canonical environment record](canonical_environments.md), [historical hosted evidence](hosted-actions-check-results.json), [historical ecosystem evidence](ecosystem-check-results.json) and [historical student release report](student-release-check-results.json) preserve earlier genuine runs and blocked requests. Do not sum overlapping test lanes or use historical image/run identities as current-source acceptance evidence.

## Claims that still need primary/vendor verification

The [unverified claims register](../wiki/review/Unverified_Claims_Register.md) identifies 29 specific historical claims, their evidence gaps, scientific implications and required verification. Examples include transferring thermochemical revDSD benchmarks into a universal rotational-constant accuracy claim; transplanting D3 coefficients into revDSD-D4 or omitting orbital response; asserting unverified ORCA/CFOUR/MPQC derivative/restart/version support; and treating HDF5 SWMR as safe concurrent multiwriter storage. The register also covers incomplete/mismatched citations, universal B0/torsion error bands, unsupported GPU double-hybrid assertions, JANPA/NBO equivalence, undeployed Pickett consumer semantics and unmeasured timing/memory promises.

Those claims remain unverified and cannot enable an advertised method, identification accuracy or deployment capability. Real computed earlier stages, clearly identified mathematical derivations and experimental anchors retain their own evidence labels. The publication exporter preserves raw scientific identities and missing qualifications; an immutable reproducibility bundle is not itself proof of publication-grade accuracy.
