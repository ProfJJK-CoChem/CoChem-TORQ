# ML automation and pruning implementation contract

Policy version: `ml-safeguards-v1`. This contract refines the existing SRS and
Method Matrix and supersedes their ML-only rejection, aggressive cutoff, and
global-minimum-retention claims. False negatives cost more than false positives.

## Scientific boundary

ML proposes structures, ranks work, and accelerates exploration. It cannot delete
a candidate or establish that a pathway is absent. Rejected ML calculations or
trajectories are invalid *results*; their scientifically relevant geometries
remain eligible for QC. Conformer identity and reaction topology cannot be inferred
from an energy or rotational-constant cutoff alone. A failed QC calculation is
unresolved evidence, never confirmation of exclusion.

`Libraries.cochem_torq_ml_policy` supplies frozen, extra-forbidden, finite-valued
Pydantic contracts. All screening energies are **relative eV**, bound to an
explicit reference identifier and target QC method including basis/protocol.
The reference must encode the reference structure, electronic state and energy
definition (electronic/ZPE/free energy); uncertainty must cover the energy
difference, including the reference. Absolute energies or per-atom variance
cannot be substituted. The legacy scout energy remains Hartree and committee
standard deviation remains meV; neither becomes a calibrated interval implicitly.

## Calibration and domain gate

An ensemble spread is a diagnostic, not validation. Each usable prediction carries
model name/version/checkpoint digest, raw uncertainty, calibration evidence,
and an explicit domain assessment. Calibration evidence binds the same model,
quantity, reference, QC method and domain. Domain IDs must define elements,
charge, multiplicity, bonding/reactivity and configuration regime; element
membership alone is insufficient. The OOD detector version, nonnegative score,
status and reason are mandatory for deferral. Unknown or OOD status, a missing
score/limit, stale checkpoint, unsupported exchangeability, inadequate held-out
data, failed coverage, or an excessively wide interval schedules QC.

`calibrate_energy_interval` computes split-conformal absolute residual order
statistics: rank = ceil((n + 1) * coverage). A rank beyond n raises rather than
clamping to the largest residual and claiming finite coverage. Training,
calibration and validation IDs must be disjoint. Atom counts do not inflate
independent energy sample counts. Independent validation coverage must reach the
declared coverage before deferral is allowed. This is an empirical gate, not a
confidence bound on population coverage. Group correlated frames by independent
chemical system before supplying observations; unique IDs alone do not establish
independence. Calibration is valid only under the stated exchangeability/domain
assumptions and cannot provide an OOD guarantee.

The existing force conformal predictor is not automatically accepted as energy
screening evidence. A production adapter must supply a scoped, validated energy
report. For Delta-ML, the checkpoint and report must cover the combined predictor.

## Conservative scheduling and sentinel audit

Default mode is `prioritize`; there is no inferred energy cutoff. Opt-in `defer`
requires explicit cutoff, maximum calibrated half-width, OOD score limit,
reference and QC method. A conformer may be deferred only if its calibrated
**lower** energy bound is strictly above the cutoff. Equality and overlapping
intervals retain priority. This is reversible scheduling, never deletion.

`ScreeningBatch.decisions` contains every input; `low_priority_pool` retains
every would-be-deferred candidate including sentinels. Original geometry digests
and upstream grid payloads remain available. The calling workflow must preserve
the referenced geometry artifacts and revisit deferred candidates as calibration,
model, scope or scientific priorities change. Budget exhaustion leaves unresolved
work explicit; it cannot be reported as exhaustive chemistry.

Sentinels are chosen from would-be-deferred candidates by SHA-256 ordering of
seed, policy version, candidate ID and geometry digest. Input order and process
random state do not affect selection for a fixed candidate set. The count is
min(pool size, max(min_sentinels, ceil(fraction * pool size))). Defaults are 5%
and at least one, configurable up to a complete audit; a nonempty deferred pool
cannot disable audits. Batch membership changes can change the selected set.
Sentinels enter the QC queue even when ML predicts low uncertainty.

`confirm_qc` attaches candidate/geometry-bound, converged, physically validated
QC evidence in a new snapshot with explicitly validated stationary-point status.
Single points without that validation remain pending. Exclusion from the active window requires the QC
lower bound to exceed the declared cutoff with matching reference and method.
Even QC-excluded records are retained for audit. A sentinel retained by QC
reopens every unresolved deferred candidate and latches `audit_failure` for the
batch. Passing that batch to `screen_candidates(previous_batch=...)` preserves
the latch; the pipeline does so automatically for both typed and grid screening.
Recalibration and a new independently reviewed policy/model are required
before restarting deferral; failed or mismatched QC remains pending.

## Provenance and pathway protection

Each decision includes candidate ID and geometry digest; prediction and model
identity; raw and calibrated uncertainty; calibration/validation dataset digests,
counts and coverage; OOD domain/score/detector/reason; full policy and thresholds;
action, reason, lower bound; sentinel flag/rank; and QC confirmation if available.
QC confirmation records engine, full method, calculation/artifact IDs and digest,
reference, convergence, stationary-point and physical validation. `confirm_qc` preserves prior
decisions in typed history. `persist` writes a new exclusive-create JSON audit
snapshot and refuses to overwrite one. The workflow must persist each batch
before executing downstream decisions; audit write failure must abort dispatch.

Every reaction-path candidate bypasses ML deferral, including a very high,
confident, calibrated predicted barrier. Path existence is confirmed only by
validated QC with exactly one imaginary mode and an IRC connecting the intended
endpoints. Failed searches and unconnected IRCs remain `unresolved` and require
further search; the API deliberately has no global `pathway_absent` result.
A scalar anchor evaluator cannot supply TS/IRC confirmation or exclusion evidence.

## Integration and acceptance

`TorqPipeline.screen_ml_candidates` and `confirm_ml_qc` expose typed screening,
QC confirmation and optional audit snapshot persistence.
`run_active_learning_pes_sampling(policy=..., audit_path=...)` passes the
policy into the existing sampler. Grid entries may provide `ml_prediction` and
`kind`; missing evidence now requests QC. No callback returns `QC_PENDING` and
keeps the scout `[E]`. The old fabricated anchor fallback is removed. Grid
outputs preserve their input payload and include `ml_decision` and an explicit
QC-confirmation requirement. `audit_path` persists before any scalar anchor
callback executes. Existing geometry/QC engines are unchanged.

Policy constants (95% coverage, minimum 20 independent calibration and validation
systems, 5% sentinels) are conservative engineering defaults, **not scientific
guarantees**. No production model/domain is certified by these tests. Domain-specific
cutoffs, OOD detectors, exchangeability and audit size require scientific validation;
the implementation leaves them explicit rather than inventing a universal threshold.

Acceptance tests cover finite-sample and held-out calibration gates, domain and
checkpoint mismatches, missing uncertainty, cutoff overlap/equality, retained
pool, order/seed reproducibility, sentinel rescue, evidence binding/history,
JSON persistence, legacy sampler integration, and QC TS/IRC safeguards.
