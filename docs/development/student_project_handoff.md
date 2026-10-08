# CoChem student project handoff

CoChem-BASE supplies the calculation and interface foundation. CoChem-TOPOS
produces reviewed structure ensembles. CoChem-TORQ consumes those structures
and records the supported electronic-structure and spectroscopy stages with
their original evidence. **GitHub Codespaces is the interface; GitHub Actions
in the student's personal private repository performs the calculations.**

The current code supports bounded calculations and integrity checks. Full-SRS
scientific qualification and the genuine private student deployment pilot are
still incomplete. Consult these maintained guides before enabling a profile:

- [BASE private student engine staging](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/docs/private_student_engine_staging.md)
- [TORQ release candidate readiness and closure gates](https://github.com/ProfJJK-CoChem/CoChem-TORQ/blob/main/docs/development/release_candidate_readiness.md)
- [Reviewed TOPOS ensemble handoff](https://github.com/ProfJJK-CoChem/CoChem-TORQ/blob/main/docs/TOPOS_HANDOFF.md)

## First student project

1. Create or select the student's **personal private** project. The real
   authenticated GitHub account must own it, and GitHub's API must establish
   `private=true` and `owner.type=User`. Open its supported Codespaces interface
   and obtain authorized access to the approved laboratory release. A personal
   Codespace does not automatically receive access to an organization's private
   repository. Keep credentials out of chat, source and receipts.
2. Use the reviewed BASE/TOPOS/TORQ versions and select an enabled calculation
   profile, method, state, basis and observable. Commit the canonical workflow
   and validated calculation request. Record the actual commit, job-file digest,
   cores and memory limits. Review the exact plan before submitting it; a receipt
   binds those values.
3. Stage one approved ORCA or CFOUR distribution from Codespaces using the
   existing reviewed descriptor and installer. Follow the BASE guide, including
   `python -m scripts.private_engine_assets` from the BASE checkout root.
   Scientific CFOUR requests use `--engine cfour` and
   `--workflow-path .github/workflows/cfour_calculation.yml`, with the same four
   real job/resource flags as ORCA. The separate CFOUR provisioning utility uses
   `.github/workflows/cfour_provisioning.yml` without calculation flags and
   establishes provisioning only. Keep the immutable intent, operational
   journal and receipt in a persistent private directory outside Git. Never
   commit or publish licensed archives.
4. Dispatch the owning repository's Actions workflow with the exact receipt,
   receipt SHA-256 and task ID. The first pilot must establish that its
   `contents:read` authority can retrieve the **private draft release and exact
   staged asset**. This provider contract has not yet been observed. A denial
   needs a reviewed private permission or release route before acceptance.
5. Inspect the genuine task status and original native artifacts. Verify the
   actual run attempt, source, request and result digests. Exercise cancellation
   where appropriate. After the real run is terminal, use the exact cleanup or
   repair command from the BASE guide and retain its private history.

No student-owned private target was available for this implementation cycle.
Actual private source authorization, staging upload/readback, owning-token
consumption, dispatch, result retrieval and cleanup therefore remain **UNRUN**.
Local tests do not stand in for this first provider pilot.

## Private assets and recovery

Codespaces uses the student's existing authorized GitHub CLI identity to stage
the approved bytes into a uniquely marked temporary **private draft release**
in that student's repository. Actions uses its own repository authority. There
is no separate asset service and no organization credential shared with the
student repository. Both approved engine descriptors retain their authentic
version, archive, runtime and source pins.

The receipt binds exact live user/repository identities, source and destination
release/asset IDs, descriptor and archive hashes, workflow, branch, committed
request, resources, task and expiry. Changed identities or bytes fail. Expiry
denies new asset consumption; it does not delete stored release assets.

Cleanup closes admission, verifies the genuine terminal owning run, checks for
active consumers and deletes only the exact task-owned asset. The task-owned
private draft release is retained as the ownership ledger when present,
normally empty after that asset's deletion. Unrelated concurrently added assets
are preserved, so the ledger can remain nonempty. Interrupted staging retains
its immutable intent and journal for explicit repair; preserve them until reconciliation
succeeds. Private records and licensed assets do not belong in public Actions
artifacts.

## Producer and spectroscopy evidence

The modern reviewed TOPOS producer contract is pinned at
`ef750eaaf6a990a3b9e1d65eca240ac89b6632ce`. Provision it with
`ci_tools/setup_reviewed_topos.py` and `ci_tools/reviewed-topos-modules.json` in
its separate reviewed-TOPOS environment. The existing legacy geometry SDK
profile remains separate. The importer verifies the review chain and exact
structure/state/protocol bytes and returns a receipt. Its state is
`imported-awaiting-calculation`, with `computation_performed=false`; import alone
does not establish a TORQ calculation or validated chemical identity.

The reviewed scientific code at
[`e77e2a4`](https://github.com/ProfJJK-CoChem/CoChem-TORQ/commit/e77e2a47d4ef23bb4ad6db72519e656336cf0bb2)
includes the modern producer contract. All **nine** focused handoff checks
passed without failures, errors, skips or mocked runtime functions, consuming
retained genuine producer evidence and exercising actual kernel write limits.
The corresponding BASE private project source is
[`44228a4`](https://github.com/ProfJJK-CoChem/CoChem-BASE/commit/44228a4739e400445be97673fa9cf514c8ee9a33). This interface/authentication code is being reconciled with subsequent ecosystem changes on BASE main; its preceding complete regression remains tied to `12dd108`.
Use the readiness report for complete test receipts and later source revisions;
overlapping focused and full-repository test counts must not be summed.

The bounded Pickett consumer passed **52** focused checks with genuine
externally compiled SPCAT/SPFIT and no failures, errors or skips. The native
source's redistribution permission is unresolved; no vendor source or binary
is bundled. Strict native inputs, units, covariance, original outputs and
digests are retained. Mathematical rigid-rotor comparisons and a fit to actual
generated predictions do not establish experimental agreement. An equilibrium
water catalog is a `Be` product; it cannot acquire an invented `B0` correction.

## Scientific limits

The SRS tracks 73 normative requirements, 41 acceptance definitions and all
140 historical method-matrix identities. **All 41 complete-scope gates remain
blocked**; passing software tests does not qualify the complete SRS.

- Exact revDSD-PBEP86-D4 still requires primary-recipe reconciliation,
  independent native component comparisons, analytic orbital-response
  derivatives and validated higher derivatives. Bounded double-hybrid research
  calculations do not enable an exact named method.
- The full spectroscopy ladder remains in scope. General resonant VPT2,
  vibration-corrected constants, independently validated distortion, hyperfine,
  tunneling, dipole surfaces and isotopologue identification require additional
  implementation or qualification. Each failed or unavailable stage retains
  valid earlier results and its reason.
- Publication and laboratory/astronomical accuracy need accepted observable
  targets, genuinely curated cited references, immutable preregistration,
  blind held-out predictions and enough independent calibration units. No
  experimental dataset or uncertainty threshold is fabricated when those are
  missing.
- General conformer completeness, difficult electronic states, optional
  scientific consumers and canonical hosted operations need their own evidence.
  Learned potentials remain advisory until independently verified recall
  supports a stronger role.

For related BASE/TOPOS work, share this document's GitHub URL and preserve the
source pins and evidence links. Private receipts and conversation transcripts
are unnecessary for the scientific or student handoff.
