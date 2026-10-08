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
- [BASE private project creation and reviewed ecosystem](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/.docs/Private_Student_Projects.md)
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
   receipt SHA-256 and task ID. The consuming job requires scoped
   `contents:write` for **private draft-release visibility** and `actions:read`
   for genuine run verification. Keep those permissions on the consuming job.
   Correct permission declarations do not establish a real provider transfer;
   the first student pilot must verify draft visibility and the exact asset.
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

## Current frozen BASE identity and ecosystem profiles

The current frozen BASE candidate is
[`12f5475`](https://github.com/ProfJJK-CoChem/CoChem-BASE/commit/12f54759ea5ae64843fa2b2bcce111fc2a5b38e1).
Its catalog-bound source-content SHA-256 is
`b51bac7f498794a3bada95d32551114f99dd26da4f1a16df0731536152b1d6d6`,
and the actual built wheel SHA-256 is
`53ffe13e7ca0b9f4945f308bb532eec4a40df3bf184e3846f75ed8ded4c89118`.
The authority refresh agrees with the committed source, and the installed
payload and CLI were checked against that wheel. These are integrity and
installation observations, not native scientific or student-provider acceptance.
The candidate includes consuming-job permission corrections. Its genuine
eleven-phase Stage 0 completed with exit zero and a valid strict registry
checksum, status `DEGRADED_OPERATIONAL`; not all methods are qualified. The
registry SHA-256 is
`78e6a6bb7f1a4135bd04ca17b536e3f3bd9b64210a9964a1fe9c1cc004841815`.
An earlier disk refusal and downstream failures remain preserved. Restoring
space by cleaning task-owned temporary files allowed the genuine retry while
retaining the unchanged 1 GB guard.

The completed full BASE run at this exact source recorded **2345 tests:
2337 passed, eight skipped, zero failures/errors**. Pytest completed in 1291.64
seconds; the supervisor waited for the original child and completed in 1302.89
seconds. The source/head stayed identical and the working tree remained clean.
The actual JUnit SHA-256 is
`b8c994545207a462a6c0d468c0ef7f0c3ccc13c0ad302f3258d96b23c35a743d`.

The same published main source also passed **all nine bounded hosted jobs**
in [run 37812840001](https://github.com/ProfJJK-CoChem/CoChem-BASE/actions/runs/37812840001):
source integrity, installed wheels and CI controls across Linux/macOS/Windows,
and the bounded xTB/PySCF derivative/dashboard route. This observation records
actual job/step metadata; remote artifact bytes and case counts were not
verified. The full local counts above remain a separate source-bound result.

The eight explicit deferrals are three native MACE/OFF24/g-xTB checks, three
licensed ORCA/audited-registry/reference/CREST-GOAT checks, and two physical
Slurm-node checks. Their absent prerequisites remain unqualified; a skip is not
a passing native or platform gate. These results do not establish the private
student pilot, complete coupled native acceptance or full scientific accuracy.

The ecosystem has distinct reviewed profiles:

| Profile | Exact source authority | Meaning |
| --- | --- | --- |
| BASE mandatory calculation kit | TOPOS `cf5655158456d763453376b108d26efcd748fe4f` with embedded TORQ `4f323800227dbde00ffb082bd6d9e44d851e1c7a` | A coupled all-three-package kit with independently reviewed wheel/helper identities. |
| Standalone TORQ scientific candidate | TORQ `e77e2a47d4ef23bb4ad6db72519e656336cf0bb2` | Its own bounded scientific implementation and source-bound evidence. |
| TORQ reviewed ensemble importer | TOPOS `ef750eaaf6a990a3b9e1d65eca240ac89b6632ce` in a separate producer environment | Reviewed import and receipt contracts; import alone performs no TORQ calculation. |

The TOPOS project checkout now matches actual main `cf565515`, the verified
mandatory-provider pin. The legacy `6a01b0f2adb7cff02edda6e339facf3d6f93904d`
geometry SDK and modern `ef750ea` reviewed-import profiles stay isolated.

The complete TOPOS provider wheel was rebuilt from the exact `cf565515` source,
matched the reviewed SHA-256
`a8e57f4ed91050f5e45be8121ed390de76524377ff7295286a03b9bf892a9f8f`
on repeated builds and contained no BASE payload collisions. The coupled TORQ
wheel remains pinned at
`d5e7ec3f8e1ae8cef85ca7aefe995219d3523e4e1330e465915a3159816fe617`.
Provider wheel-build/import checks do not establish a complete kit installation
or native calculation; current-source coupled native acceptance is still pending.
Keep this approved coupled kit intact rather than replacing its internal TORQ
with the newer standalone candidate. A change requires a separately reviewed
kit and genuine acceptance. Earlier failed full attempts remain preserved with
their original source and diagnostics.

## Licensed TOPOS caller blocker

**BLOCKED: the approved TOPOS hosted licensed callers do not implement BASE's
current private-staging API.** At TOPOS `cf565515`, the affected paths are
`.github/workflows/orca_hosted.yml`, `.github/workflows/topos_compute.yml` and
`.github/workflows/topos_orca_acceptance.yml`. The wrapper and its consumers
retain the removed `asset-token` provisioning contract and default to BASE
`14202e182e1fa4f99258ed32a8ae9ba8c1c565ad`. They do not supply the complete
receipt/digest/task inputs, private personal-owner guard, receipt-bound run
identity or all required consuming-job permissions. Existing generic secret
selectors do not resolve those contract failures.

Changing only the BASE pin cannot repair this route. The migration must bind
the actual owning private User repository, TOPOS workflow, source/ref, task,
complete scientific request and resources. It must verify the exact live run
and approved staged asset with job-scoped draft/run permissions. A receipt for
a different BASE workflow cannot be reused for a TOPOS caller. The migration
also requires a newly reviewed whole TOPOS wheel and coupled catalog, followed
by genuine native/provider acceptance. Do not modify the approved `cf565515`
kit or substitute a different internal TORQ without recording new authority.

For current supported licensed requests, use BASE's dedicated personal-private
`orca_calculation.yml` or `cfour_calculation.yml` route. Its genuine student
deployment pilot is still **UNRUN**. Free supplied-geometry handoffs, reviewed
ensemble imports and free typed TOPOS operations are separate profiles; their
success cannot qualify this blocked licensed hosted route.

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
The independently checked BASE candidate is
[`12f5475`](https://github.com/ProfJJK-CoChem/CoChem-BASE/commit/12f54759ea5ae64843fa2b2bcce111fc2a5b38e1).
Use the readiness report for complete test receipts and later source revisions;
overlapping focused and full-repository test counts must not be summed.

The actual TORQ main workflow at
[`5c1da3`](https://github.com/ProfJJK-CoChem/CoChem-TORQ/commit/5c1da3a15be1dd6fe8a34c418c739391bfaed44e)
completed [run 37805501782](https://github.com/ProfJJK-CoChem/CoChem-TORQ/actions/runs/37805501782)
on 2026-10-08 with all eight reported jobs successful. Original API metadata
records that outcome. Artifact download was denied by the storage network
policy, so original hosted artifact bytes and test counts remain independently
unverified. This hosted job outcome does not qualify the student private pilot,
the complete SRS or independent scientific accuracy.

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

## Handoff to related BASE and TOPOS work

| Related work | Paths and contract to carry forward | Qualification still required |
| --- | --- | --- |
| BASE student project | `scripts/private_engine_assets.py`, dedicated ORCA/CFOUR calculation workflows and private Actions controller; exact owning repository/source/ref/workflow/task/job/resource receipt binding | Genuine personal-private student staging, draft visibility, calculation, retrieval and terminal cleanup; pilot **UNRUN**. |
| TOPOS licensed callers | `orca_hosted.yml`, `topos_compute.yml`, `topos_orca_acceptance.yml`; migrate the full receipt API and bind the actual TOPOS scientific request/resources | **BLOCKED** until reviewed whole-wheel/catalog migration and genuine native/provider acceptance; a pin-only change or borrowed BASE receipt cannot close it. |
| TORQ producer and spectroscopy consumer | Reviewed ensemble importer, bounded Pickett consumer and source-bound readiness receipts; keep coupled `4f32380`, standalone `e77e2a4` and importer `ef750ea` profiles distinct | All 41 full-SRS gates remain blocked; independent accuracy and private deployment are separate from local/native checks and hosted job metadata. |

Once this reviewed document is published, copy its
[clean GitHub handoff URL](https://github.com/ProfJJK-CoChem/CoChem-TORQ/blob/main/docs/development/student_project_handoff.md)
into the two related BASE/TOPOS discussions. That shares the source pins,
contracts and qualification status without a conversation transcript or private
receipt. Providing the URL does not establish that another discussion received
it; no cross-thread message is sent by this handoff.
