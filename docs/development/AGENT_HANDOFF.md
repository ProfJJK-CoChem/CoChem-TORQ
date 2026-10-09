# CoChem-TORQ agent handoff

Prepared for the user's request to hand this work to another agent. The continuing
objective is **complete CoChem-TORQ's accepted SRS and method matrix, using genuine
calculations and independent published references**. The latest turn prioritizes
a repository handoff. The full research product remains unfinished.
The [machine-readable progress record](AGENT_HANDOFF_PROGRESS.json) records
source-bound baseline evidence, fresh access observations and continuation status.

## Read first

1. [Normative implementation SRS](../wiki/CoChem-TORQ_Implementation_SRS.md).
2. [Method matrix](../wiki/Method_Matrix.md) and its
   [implementation contract](../wiki/Method_Matrix_Implementation_Contract.md).
3. [Complete requirement audit](full_srs_requirement_audit.json): 73 requirements,
   41 acceptance gates; all 41 gates remain blocked and no requirement is fully
   qualified. `actual_evidence` remains empty under the full-release evidence
   contract. Scoped regression receipts are retained separately.
4. [Agent and researcher context](agent_and_researcher_context.md): all 17 remaining
   workstreams, every requirement, dependencies and six conditional external
   prerequisite categories. This is the detailed backlog; do not infer that
   implementation work has become a researcher chore.
5. [Published-reference product](published_reference_product.md),
   [spectroscopy implementation](spectroscopy_implementation.md),
   [geometry/PES implementation](pes_and_geometry_validation.md), and
   [canonical environments](canonical_environments.md).
6. [Bounded benchmark proposal](bounded_preregistration_proposal.md) and its
   [machine-readable proposal](bounded_preregistration_proposal.json). It is a
   proposal, not an accepted/preregistered/executed accuracy campaign.

All 24 D01–D12 and ARCH-01–ARCH-12 design selections are already accepted in the
SRS. Do not ask those questions again. Exact numerical qualification, retained
scientific signoff and site/access prerequisites are separate issues.

## Git baseline and evidence

The prepared baseline on public `main` is documentation commit
[`cbf53f57109c6842ff1ca50025c522058026bfe4`](https://github.com/ProfJJK-CoChem/CoChem-TORQ/commit/cbf53f57109c6842ff1ca50025c522058026bfe4).
It follows the actual corrected scientific code commit
[`d43b8aa3eefc4768f2d52857cd3945677ed3a794`](https://github.com/ProfJJK-CoChem/CoChem-TORQ/commit/d43b8aa3eefc4768f2d52857cd3945677ed3a794).
Check the checkout and remote again; this handoff will itself be committed later.
Do not assume a local branch is `main` or reset existing work to these commits.

The verified code identities are:

- Source/test/configuration tree SHA256:
  `164926d7cb0778a7138398cd85ba8a78906da237f70946ea02cfc15b048e7902`.
- Scientific application SHA256:
  `680f1415ee5e10867709c8d6c67f26f169c2c098d16b81ef0a6ef2331b65e1be`.
- Normative SRS SHA256:
  `a4b76cc928e74f200a73315a7581082778fddb44b518e5a76073e592ff88d023`.

Actual [canonical CI run 37856692326](https://github.com/ProfJJK-CoChem/CoChem-TORQ/actions/runs/37856692326)
passed all eight jobs on `d43b8aa`: full regression **3,277 passes**; installed
Python 3.10 and 3.12 core lanes **1,079 passes each**, with zero failures, errors
or skips. Both original hosted wheels contained **262 byte-verified Python
payloads** matching the frozen code. Sixty-four retained native bundles were
hash-verified; artifact retention was bounded and truncated. Do not say every
native calculation from the full run was retained. Hosted marker-deselection
counts were not recorded and remain unknown.

Separate genuine local selections passed **28 bounded relaxed-scan cases**,
**38 host-allocation cases**, and **50 interface/process cases**. The final
documentation audit tests passed **24 cases**. These selections overlap other
lanes and must not be summed. None qualifies independent identification accuracy.

Portable public evidence lives under:

- [Current W07 source-bound proof](release_candidate_evidence/w07-published-reference-20261008/summary.json)
  and its [file index](release_candidate_evidence/w07-published-reference-20261008/file-index.json).
- [Earlier source4d proof](release_candidate_evidence/final-audited-published-reference-20261008/summary.json)
  and its [file index](release_candidate_evidence/final-audited-published-reference-20261008/file-index.json).

The earlier 3,248-pass/1,078-core outcomes belong to their original source.
The first W07 native attempt remains **24 passes/4 setup errors**, not a successful
scan. RFC 8785 can serialize an integral process timestamp as a JSON integer;
the corrected owner validator accepts finite positive integer/float values while
preserving exact PID/create-time equality. Original failed receipts were retained.
Later genuine runs supply their own source identity and outcomes.

Initial hosted artifact requests also failed, followed by successful fresh
managed requests to the same endpoints. The cause of those initial failures was
not established. Do not assert a proxy denial, missing token or required artifact
network activation from that history. Original ZIPs and private native evidence
remain outside the repository; curated public inventories contain no credentials
or signed download URLs.

## Latest reference-access finding

**The older NIST/QCArchive network blocker is superseded for three actual tested
endpoints.** Normal requests through the inherited proxy, with TLS verification,
returned HTTP 200 and curl exit zero. Original bodies were independently read
back and matched their recorded bytes and SHA256:

| Endpoint | Body bytes | SHA256 |
|---|---:|---|
| [NIST WebBook water](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=800) | 25,844 | `71a6a45e8ba73a635e9e40b3281c33512122d2aff904e4b472d97216a30672ca` |
| [NIST CCCBDB water](https://cccbdb.nist.gov/exp2x.asp?casno=7732185&charge=0) | 50,382 | `b92fab46ef02c8fa83aeab9039a2ff24e110ce3b64a11c3e3929d4348dc3ab43` |
| [QCArchive API information](https://api.qcarchive.molssi.org/api/v1/information) | 630 | `93c228ddf2b78d42b8ba55c7f7cd3a8d6cb65003428b5329e992e193723d3c67` |

Original retrieval receipts use explicit UTC timestamps around
`2026-10-09T00:33:00Z`. The runtime metadata still reports restricted/unknown
network readiness; actual responses establish access only for these requests.
Do not claim all research providers or API records are reachable. An information
endpoint is not a molecular calculation record. Retain the old denied receipts
as history, recheck actual requests and continue source acquisition; do not ask
the user to publish configuration solely to retry these now-accessible endpoints.

Fresh independent [handoff requests and original-byte receipts](handoff/reference-access-20261009/summary.json)
also returned HTTP 200/exit zero at `2026-10-09T20:47:41Z`. They retain their own
body hashes; the earlier source extraction retains its original hashes. The
[source-acquisition note](reference_acquisition_20261009.md) and
[nine-candidate packet](../../benchmarks/published-values/nist-water-source-candidates-20261009/README.md)
preserve selected fundamentals, source-tabulated harmonics and rotational values
with unresolved isotope/state/uncertainty fields. **Zero accepted benchmark
records were added.** Original NIST HTML stays outside the public repository
because the source expressly reserves its SRD compilation rights.

QCSchema is an interchange schema, not a reference database; QCArchive is a
provider. WebBook and CCCBDB pages are actual HTML sources, not invented JSON APIs.
Retrieve original bytes and citations, then type each observation with units,
isotopologue, electronic/vibrational state, experimental/fitted/theoretical status,
uncertainty and reuse rights. Never infer missing confidence conventions.

Existing authentic source packets in
[benchmarks/published-values](../../benchmarks/published-values/README.md) include
CC0 QM9 originals, archived official QCArchive NH3/B2PLYP components, secondary
experimental water/formaldehyde B0, and primary fitted chlorine-isotopologue
constants. NIST-authored CO line candidates previously supplied zero accepted
constants. These categories have different evidentiary scope:

- HF/STO-3G versus QM9 B3LYP/6-31G(2df,p) is a descriptive cross-method comparison.
- Experimental fundamentals are not harmonic frequencies; B0 is not Be.
- Archived B2PLYP component agreement does not establish exact revDSD gradients.
- Fitted constants and fit RMS do not imply an unavailable covariance/confidence
  model or a universal maximum line error.
- Article/software source review is not execution or scientific qualification.

## Scientific work started for continuation

These are bounded next implementation designs. A separate progress record will
identify any completed module or retained draft; do not assume the designs below
are tested code on `main`. The previous detailed context still defines the broad
unfinished scope.

Preservation is now complete: **no new scientific module, native calculation or
scientific test was completed in this continuation**. The GKS integration edit
was saved as an exact unvalidated patch and its production source restored to
the tested baseline. Patch applicability/restoration checks passed; those are
preservation checks, not algorithm validation. The other four topics reached
design-only status. Their portable documents are:

- [GKS response design](handoff/research-designs-20261009/gks-response/DESIGN.md)
  and [exact unfinished patch](handoff/research-designs-20261009/gks-response/revdsd-unfinished-integration.patch).
  The patch references an unwritten module and must not be applied as complete code.
- [CP total-potential optimization design](handoff/research-designs-20261009/counterpoise-optimization/DESIGN_UNIMPLEMENTED.md).
- [IRC implementation design](handoff/research-designs-20261009/irc/UNVALIDATED_IRC_DESIGN.md).
- [Density/EFG design](handoff/research-designs-20261009/density-properties/UNVALIDATED_DESIGN.md),
  with a pinned unmodified Apache-2.0 upstream source and full license retained
  for reference; upstream labels the property implementation “In testing.”
- [Periodic nuclear-motion design](handoff/research-designs-20261009/periodic-nuclear-motion/RESEARCH_DESIGN.md).
- [Selected-file inventory](handoff/research-designs-20261009/file-index.json).

These drafts need derivation review and actual implementation checks before
activation. Their local source snapshots and original metadata remain preserved;
the repository packet intentionally contains only the indexed portable subset.
The copied designs retain their original subtask instructions, including local
paths and restrictions on committing during that subtask. Those are historical
context. This handoff and the user's current instructions govern continuation;
the original subtask restriction does not withdraw the existing authorization
to complete code and push reviewed work.

| Workstream | Concrete implementation direction | Required evidence and limits |
|---|---|---|
| W03 — GKS/custom double-hybrid response | Differentiate the genuine LDA/GGA AO XC potential, atom-attached grid coordinates and Becke weights; solve CPKS and differentiate conventional OS/SS PT2, including Pulay and canonical denominator response. | Compare to actual total-energy finite differences. Reject unsupported frozen-core/open-shell/complex/D4 tuples. Native Hessian `make_h1` alone does not establish complete semilocal moving-grid response. Exact revDSD naming stays blocked until its original recipe and independent evidence are reconciled. |
| W06 — CP geometry optimization | Optimize the CP-corrected **total** potential `E_complex + lambda * sum(E_fragment_own_basis - E_fragment_full_ghost_basis)`, with separately named lambda=1 CP or lambda=0.5 half-CP. Retain all moving-basis-center gradients and explicit fragment states. | Every genuine component is required at each geometry. Optimizing interaction energy alone is not a valid substitute. Fresh final gradients must establish the declared constrained/all-Cartesian stationarity. Curvature/minimum/De/D0 stay unavailable when uncomputed. |
| W08 — Genuine IRC | Use actual restricted-HF TS gradient and projected first-order curvature prerequisites, isotope masses, mapped atom IDs, mass-weighted normalized-gradient continuation and adaptive predictor/corrector control. | Independently evaluate accepted points; retain both directions, owned native lineage and real budgets. A finite path prefix or downhill endpoint candidates do not establish complete reaction connectivity. |
| W05/W10 — Density/EFG properties | Consume an authenticated actual RHF/RKS native density, verify AO basis/state/geometry/installation identity, and compute electronic plus other-nucleus point-charge electrostatic-potential Hessians under explicit signs and units. | Check symmetry, translation/rotation and an independent integral/convention route. Do not claim a correlated MP2 density, unsupported ECP result or quadrupole coupling without an explicit sourced nuclear moment. |
| W04 — Periodic nuclear motion | Build a bounded 1D periodic Laplace–Beltrami/Fourier-Galerkin solver from authentic cyclic PES and positive coordinate metric, with measure `sqrt(g) d(phi)`. | Free-rotor and convergence checks are mathematical evidence. A molecular PES must retain original provenance. Justified phase/permutation sectors are explicit; no general coupled torsion–rotation/J or identification claim follows from a 1D solver. |
| W01 — Reference acquisition | Parse newly accessible source-bound NIST data and fetch actual live QCArchive molecular records with their versions and property identities. | Preserve original bodies, request receipts, exact extraction locations and source licenses. Generic server metadata is access evidence, not a molecular benchmark. |

The current relaxed-scan route is already implemented through API/CLI. It is
water-only, closed-shell RHF/STO-3G, nonperiodic and uses independent MINAO starts.
Its positive genuine integration samples **one unique target through three
scheduled attempts**. This does not establish a full curve, global basin recall,
periodic/multistate coverage or an equilibrium Hessian. A budget stop leaves final
energy, geometry and stationarity unavailable. AO-density comparison is
unavailable when relaxed nuclear centers differ.

The remaining W02, W07, W09, W11–W17 tasks remain in the detailed context:
benchmark/calibration, generalized PES/advisory acquisition, stereochemistry and
state diagnostics, contracts/migrations, immutable application DAG/UI parity,
actual algorithm-state resume, accounting/durability, exact profile activation,
sibling handoffs and authentic release campaigns. The five scientific designs
above do not complete those tasks.

## Environment and verification instructions

In the current cloud workspace, use the existing `/workspace/CoChem-TORQ`,
`/workspace/CoChem-BASE` and `/workspace/CoChem-TOPOS` checkouts. Their project
revisions at preparation were TORQ `cbf53f5…`, BASE
`12f54759ea5ae64843fa2b2bcce111fc2a5b38e1`, and TOPOS
`cf5655158456d763453376b108d26efcd748fe4f`. The isolated installed SDK intentionally
uses different verified BASE/TOPOS revisions: BASE `8346272…`, TOPOS `6a01b0f…`.
Read [student setup/handoff](student_project_handoff.md) before changing it.

The current interpreter is `/workspace/.venvs/cochem-torq/bin/python`: Python
3.12.14, PySCF 2.14.0, SciPy 1.18.1, CPU Torch 2.14.1 and pip 25.2. An actual
dependency check passed with 163 distributions. Use the repository's frozen
requirements for a new supported environment; do not assume a fresh task restores
this installed filesystem or live processes. Saved cloud configuration is not
proof of publication or fresh-task restoration.

Inspect Git status and actual disk/memory first. Recent free scratch was only
about 370 MiB; preserve at least 256 MiB and all original evidence. A complete
local regression needs at least 800 MiB plus reserve; use ordinary canonical
public CI rather than deleting retained native outputs or wheels. Independent
workers require their own exact-version installation proof. Scope new native
studies with real CPU, memory, scratch, evaluation and wall-time limits.

From a supported environment and the repository root:

```sh
git status --short
git rev-parse HEAD
python ci_tools/validate_release_evidence.py
python ci_tools/check_scientific_test_policy.py tests
python -m pytest -q tests/test_release_evidence_validator.py -p no:cacheprovider
```

Use a unique external `--basetemp` and JUnit path for real evidence runs. Set
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `JAX_ENABLE_X64=true` where required. The existing
`COCHEM_DISABLE_SANDBOX_CHECK=1` bypasses only the legacy working-directory-name
check; it does not remove scientific or resource gates.

Lint/type checks are in the pinned canonical workflow. Select meaningful native
and mathematical checks for the changed module, then inspect actual results and
original retained inventories. Do not sum overlapping test counts. Run
`validate_release_evidence.py --require-full-release` only as a release decision:
it currently must reject the incomplete 73/41 scope. Do not weaken it to publish.

## Boundaries and next-agent starting actions

Student private deployment, licensed-asset staging, live student Codespaces/Actions
journeys and student billing are **on hold**. Ordinary public CI and bounded local
research can continue. GitHub Actions remains the canonical calculation authority
and Codespaces the interface. Organization membership does not expose organization
secrets to a personally owned repository; the accepted licensed-engine design
uses the student's own existing authorized lab access to stage private assets.
That held deployment route remains untested. Do not introduce a shared-credential
asset service or publish credentials/configuration/archives.

The user previously authorized pushing completed work to GitHub `main`. Preserve
unrelated changes; fetch and check ancestry before a normal push. Do not force push
or rewrite historical evidence. If a PR is created, attach it to the task through
the app tool. Keep source, tests and dependencies frozen during source-bound runs;
changed code needs new evidence rather than relabeled old passes.

Start by inspecting this handoff's progress record, current branch/status and
canonical CI. Recheck real source endpoints, finish exact source curation, and
resume one bounded scientific implementation with independent checks. Reuse the
accepted designs and existing genuine adapters. Add usable typed/API/CLI paths
only for exact supported domains with declared limitations. Update the requirement
audit without promoting unresolved full-scope gates. End each increment with code,
genuine evidence and a precise remaining-work update.

External human tasks are narrowly conditional: unavailable lawful licenses/access,
undelegated numerical/publication acceptance, explicitly required external human
review, genuinely necessary new physical measurements, account/site/network
activation and the user's deployment hold. Web/article retrieval, algorithm
research, extraction, coding and available CPU validation remain agent tasks.
