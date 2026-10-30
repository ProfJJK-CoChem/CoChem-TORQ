# Initial PySCF NAO Wiberg validation profile

This is an explicitly experimental, locally controlled pair-descriptor profile,
not an activated Method Matrix production recipe or independent broad NAO
qualification. Governing requirements remain TORQ-ANA-001/002 and
V-WFN/V-ANALYSIS, with optional separately named analyses under accepted D07.
The full 73-requirement/41-gate audit remains unchanged and blocked.

The intended initial operation is `wiberg_nao`, explicitly requesting engine
`pyscf`, method `hf`, basis `sto-3g`, charge 0, multiplicity 1 and complete bounded
CPU/memory options. It uses pinned Linux CPU PySCF 2.14.0, real all-electron
spherical restricted native orbitals and unlabelled H/O nuclei. Missing/other
engines, methods, basis sets, isotope/carbon/open-shell/ECP profiles fail honestly.
H/O support does not establish organic-carbon/student bonding readiness. Fresh
H2/water evidence will be named separately from the mathematical algorithm domain.
NPA, full NBO, E(2), chemical bond classification and empirical accuracy remain
unavailable. Working Löwdin and rigid-scan operations retain their own identity.

For actual AO overlap S, spin-summed RHF density D and genuine PySCF natural-AO
coefficient matrix C, the orthonormal NAO density is `P=C.T S D S C`, with
`C.T S C=I`. Pair descriptors are dimensionless sums of squared real entries of
P over the assigned atom-column blocks. PySCF's pinned algorithm preserves the
original atom/shell column slots through construction and natural-character
restoration; the largest AO coefficient does not define the assigned atom.
`P^2=2P` for the spin-summed closed-shell density. Idempotency, backtransform and
occupied-projector residuals are diagnostics, with no newly invented pass cutoff.
Internal populations/charge balance validate conservation; no NPA charge product
is advertised.

The producer binds actual native checkpoint/basis/energy/occupation/geometry,
AO metric, engine installation, the PySCF nao/orth/elements source files and the
default H/O AOSHELL policy before/after. Outputs retain raw transform/density
matrices, atom-column map, pair descriptors and actual validation residuals.
Existing metric 1e-7, density symmetry 1e-10, electron/charge 1e-7 and native
energy/occupation 1e-8 guards are preserved. Upstream NAO's 1e-9 orthogonality
warning remains recorded; a 1e-7 guard is not proof of 1e-9 precision.

Predeclared validation: the independently derived single 1s-per-H RHF analytic
limit W_HH=1 uses absolute 1e-10, rtol = 0. Fixed water rotation/permutation pairwise
comparisons use absolute 1e-10, rtol = 0 as a newly approved stringent engineering
invariance target, not chemical accuracy. Failure is preserved and investigated;
no posthoc threshold relaxation, favorable orientation selection or profile
activation from failed qualification. A separate genuine split-valence algorithm
case can challenge accidental Löwdin aliasing without enabling that basis in the
public initial profile. Same-function agreement is not an independent NAO
reference campaign, and broad scientific qualification remains outstanding.

The complete native-plus-NAO phase uses TORQ's existing PySCF thread/scratch
context and restores the caller's prior limits on success or failure. The owning
worker process still supplies its BLAS allocation; no new OS hard memory/thread
limit is claimed. Actual requested cores and PySCF threads during analysis are
retained. The final metric's comparison to upstream's warning threshold is
labeled as a comparison, not a fabricated parsed upstream warning; actual engine
messages remain in the owning execution transcript.

The exact STO-3G data file is pinned separately: the engine's pre-existing
Python/shared-library fingerprint does not include basis `.dat` files. Its
qualified wheel-RECORD hash and before/after source identity prevent a coherent
but modified installed basis from being accepted solely through matching labels.
The named STO-3G alias and basis directory must select that exact pinned file;
checkpoint/request comparison parses its absolute path, rather than trusting a
mutable named-loader alias to prove its own correctness. Modified loader policy
is refused before native work.

## Actual controlled evidence and remaining release checks

Local evidence ID `torq-nao-stage2-20261030` binds the **preactivation direct
producer** source SHA256
`c78d50a3f918fcad37bf4e6a9bcdeddccc531d8b1601058344e2c2d114b806b3`.
Its original JUnit SHA256 is
`749d81dbb75514dabb76adb8b37b65d82bcf3d609b1a7f730b9a57d4d4bb2fb4`:
14 controls passed, zero skipped, with six actual RHF evaluations and 48 verified
native artifact entries. Ten intentionally corrupted copied datasets are refusal
controls, not ten additional calculations or scientific reference records.

Actual neutral H2 satisfied its analytic limit. Fixed water rotation/translation
and atom permutation had maximum mapped pair error 6.475939576255918e-15 against
the predeclared absolute target 1e-10 with rtol = 0 engineering target. Actual stretched 3 Å H2 converged as RHF
but had internal stability True/external stability False and native status partial;
the operation refused it and published no NAO pair result. This genuine expected
refusal was not relabelled as a successful scientific calculation.

The separate actual water HF/6-31G algorithm challenge recorded the genuine
upstream NAO function call and retained both NAO and Löwdin matrices from the same
native density. Raw density-element differences are orbital-gauge dependent;
independent artifact recontraction also found gauge-invariant atom-pair
separation: an O–H descriptor was approximately 0.77303685764 in the NAO basis and
0.92760089408 in the Löwdin basis. These are observed model descriptors, not
experimental references or a newly selected difference cutoff. The split-valence
public tuple remains refused. Source/control identity, conservation, scoped
invariance and actual algorithm execution are evidence; same-function agreement
is not an independent NAO benchmark or empirical bond-order accuracy.

The controlled source and its original artifacts remain separately preserved;
changing capability activation changes the owning producer hash. Final release
verification must retain a **new final-source public execute_handoff and fresh
installed owning-wheel/entry-point proof**, with actual wrappers/import origins
and before/after source/RECORD bindings. The controlled preactivation receipt
cannot be relabelled as execution of that later source. These are local evidence
records, not a claim of published downloads or student/hosted qualification.

Core-only installations genuinely lacking PySCF retain unavailable metadata and
an unavailable public result, with reason and no substituted analysis. The
existing boundary checks runtime availability before operation-specific options.
Runtime-independent request controls remain in the core lane; configuration
controls that require the genuine installed PySCF are in the engine-profile lane.
Their lane assignment does not mean each control performs another SCF. Fresh
physical calculations, runtime/configuration controls and copied adversarial
checks remain separate evidence categories; no skipped or mocked absence is used.

## Final source and installed-wheel local verification

The final implementation commit
`54d23c2638a18a001894dd331798d413bb78ceb4` passed the complete 57-case module
selection with zero failures, errors or skips. Its source/public-boundary lane
retained 12 native attempts (11 SCF records plus one invalid-basis pre-SCF
refusal), including the existing genuine Löwdin and rigid-scan controls. A fresh
owning wheel from that clean commit has SHA256
`2d6f81a95e4aeec5240422e6b35d44433c590d13aedf30f9d6b3c53122c070ea`;
all 263 Python payloads match the archived commit. The actual installed core
without PySCF passed 35 cases, with 22 engine cases deselected and zero skips.
Actual installed public H2 and original-water cases both completed. These are
overlapping selections, not additive unique-test or chemical-reference counts.

Both actual installed targets also passed dependency consistency checks without
repair. Their 276 installed hashed payloads and RECORD identities remained
unchanged, as did final producer source and Git identity. Native lanes ran
sequentially with one CPU, 512 MiB engine advisory memory (not an OS RSS limit),
and a 180-second bound plus 10-second owned cleanup; actual process identities
were retained and confirmed terminal. Independent source and artifact review
accepted this exact experimental H/O profile. This adds no NPA/NBO, carbon,
accuracy, hosted/student or full 73-requirement/41-gate qualification.

The [bounded documentary evidence packet](release_candidate_evidence/nao-wiberg-local-20261030/README.md)
binds original receipts, derived XML, actual dependency checks and preserved
earlier failures. Raw free-engine native outputs and the owning wheel remain in
a separately hash-manifested archive outside the source repository. The earlier
preactivation receipts retain their own source identity.
