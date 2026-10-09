# W06 counterpoise geometry optimization: unimplemented handoff design

Status: **UNIMPLEMENTED / UNVALIDATED DESIGN ONLY**. This packet contains no
production code and no test or calculation result. The assigned agent performed
read-only inspection and design, then stopped when the parent relayed the user's
request to prepare an agent handoff. No native calculation was launched. No
tracked or untracked production file was created or modified by this agent.

The assignment originally reserved these production paths; they have not been
created:

- `src/cochem_torq/engines/counterpoise_optimization.py`
- `tests/test_counterpoise_optimization.py`
- `docs/development/counterpoise_optimization.md`

## Scientific objective

Use a CP-corrected **total** Born–Oppenheimer potential, retaining intrafragment
deformation. At the current full-complex Cartesian geometry R, let E_AB^AB be
the complex energy, E_i^i the energy of physical fragment i in its own basis at
its current internal geometry, and E_i^AB its energy in the complete complex
basis, with all other centers carrying zero nuclear charge:

```text
Delta_BSSE(R) = sum_i [E_i^i(R_i) - E_i^AB(R)]
E_total(CP; R) = E_AB^AB(R) + Delta_BSSE(R)
E_total(half-CP; R) = E_AB^AB(R) + 0.5 * Delta_BSSE(R)
g_total(lambda; R) = g_AB^AB(R)
                    + lambda * sum_i [embed_i(g_i^i(R_i)) - g_i^AB(R)]
lambda(CP) = 1; lambda(half-CP) = 0.5
```

`embed_i` inserts the fragment's own-basis gradient into the matching persistent
full-complex atom rows and zeroes the other rows. The full-basis ghost gradient
must retain **every** center derivative, including the centers carrying no
nuclear charge. This is a derivative of the exact declared CP total potential.

Optimizing `E_AB^AB - sum_i E_i^AB` by itself optimizes a frozen-monomer
interaction-energy definition while removing the native monomer deformation
energy. It is unsuitable as a general CP geometry objective when monomer
internal coordinates can relax. The existing single-point CP interaction
quantity must remain separately named and must not be relabeled a CP total
potential, a CP-optimized geometry or a dissociation energy.

Half-CP is a distinct potential and geometry protocol, not a silent numerical
fallback, a universal error bound, a composite electronic method or an
experimental accuracy claim. A half-CP implementation must carry its explicit
weight in request, result, identity and provenance. Its empirical merits would
need an independent bounded reference campaign.

No De, D0, ZPE, global minimum or local-minimum claim follows from optimization
and a small gradient alone. A qualified CP minimum requires genuine matched
CP curvature evidence; the existing complex-only Hessian cannot stand in for
the Hessian of the CP total potential. A qualified De additionally needs actual
matched relaxed fragment endpoints. A CP Hessian and those endpoint products
are subsequent milestones, not products of the initial optimization profile.

## Existing code to reuse and authenticate

Read the following files before implementing; source hashes in `STATUS.json`
record this agent's actual inspection snapshot, not an implementation receipt:

- `engines/counterpoise.py`: `GhostPySCFBackend.evaluate` takes a full complex
  plus explicit active atom IDs, fragment charge and multiplicity. It retains
  physical/ghost identities, all-center gradients, convergence, actual SCF
  stability, raw checkpoints and manifests.
- `energetics.py`: `Fragment`, `_fragment_molecule`, `_accepted_energy`,
  `_basis_signature` and `electronic_checkpoint_digest`; this code already
  verifies exact Gaussian basis/center identity at retained checkpoints.
  `evaluate_interaction` is a single-point/uncorrected-endpoint service, not a
  CP-total geometry optimizer.
- `engines/diagnostics.py`: `verify_native_artifacts` authenticates retained
  hashes, sizes, paths and stored result fields before checkpoint readback.
- `engines/constrained_optimization.py`: actual Linux parent-death containment,
  original child ownership/waits, immutable observations, physical evaluation
  ceilings and independent final stationarity are useful patterns. Its exact
  constrained profile and result schema must not be impersonated by a CP
  service.
- `engines/pyscf_backend.py`: `_normalize`, `PySCFBackend.evaluate`, `_seal`,
  `_finite`, `_stability` and the real engine installation fingerprint.
- `tests/test_energetics_integration.py`: existing genuine component contractions
  and two-step all-center CP finite differences; these establish current
  single-point behavior, not a new optimization result.

DFT ghost moving-grid response has an explicitly different quadrature issue
already documented in `engines/ENERGETICS_PROTOCOL.md`. Start with a distinct
closed-shell all-electron HF/STO-3G profile and do not expand a passing HF
optimization into DFT, MP2, revDSD or dispersion qualification by analogy.

## Proposed bounded typed contract

The following is a design proposal, not an existing accepted schema:

1. Require a validated `Molecule` with explicit persistent atom IDs and disjoint
   explicit `Fragment` states exactly partitioning the complex. Charges sum;
   complex and fragments are explicit singlets with positive even electron
   counts. Do not infer states, renumber atoms or repartition on recovery.
2. Expose a typed request/result for local programmatic use. The root agent
   owns later registry, application, Actions and release-audit integration.
   A programmatic module alone does not establish a canonical Actions profile.
3. First profile: explicitly HF, STO-3G, restricted, all-electron, no frozen
   core, no density fitting, no dispersion, no ECP/solvent/relativistic change,
   one thread and mandatory real SCF stability at every component. Preserve
   the exact normalized settings and actual engine/optimizer installation
   identities. Confirm pinned installed versions against repository release
   locks before declaring any version-specific profile.
4. Bound the first request to a small complex (suggested maximum six atoms and
   two closed-shell fragments). Expose separately named `counterpoise` and
   `half_counterpoise` corrections. Neither name is a validated profile until
   its actual numerical acceptance is complete.
5. Require an explicit immutable set of movable atom IDs, Cartesian displacement
   bounds, optimizer iteration ceiling, **component-attempt** ceiling, batch
   and overall wall deadlines, memory allowance and artifact byte ceiling.
   Moving only selected rows establishes stationarity only in those rows;
   frozen rows must remain bitwise unchanged in the actual geometry.
6. For F fragments, each complete objective geometry uses exactly `1 + 2F`
   genuine backend component attempts: complex plus every own-basis fragment
   plus every full-basis ghost fragment. The final verification consumes a
   fresh complete batch. Do not confuse an optimizer iteration, an objective
   geometry or a component attempt with a single SCF iteration or a stability
   solver iteration. Preserve both scheduled and actual counts honestly.
7. No stale-disk recovery or cross-geometry result reuse. Let SciPy obtain a
   joint `(energy, gradient)` objective from one freshly completed batch with
   `jac=True`; this avoids double native calculations for separate fun/jac
   callbacks without hiding any component attempt. Always obtain a fresh
   independent final batch even if its coordinates equal a prior batch.

## Owned native execution and evidence

Use a fresh empty, nonsymlinked private workspace. Reject populated workspaces
and authentic corrupted artifacts instead of recovering old scientific values.
Each objective batch should execute in an actual owned Linux subprocess with
`PR_SET_PDEATHSIG`, a checked owner PID/create-time binding and a new process
session, before scientific imports. Set OMP/OpenBLAS/MKL and PySCF to one thread.

Write an exclusive immutable component-start receipt before invoking each
backend, and a separate original completion receipt afterward. Include the
batch index, component identity, atom IDs, requested method/state/geometry,
actual worker identity and actual start/end times. A termination midway through
a component preserves its started attempt without inventing a completed result.
The worker must never exceed its allocated remaining component allowance.

Parent execution should monitor actual wall time, sampled owned-process RSS
and owned artifact bytes; terminate and wait for its actual process group on
an exceeded ceiling. Sampled RSS is not a kernel-enforced memory quota or an
exact peak. Maintain that distinction explicitly. Ownership receipts must
accept finite positive JSON process timestamps whether represented as an
integer or float, while rejecting bool/nonfinite/mismatched values. Reuse the
existing numeric validator rather than reintroducing the earlier timestamp
serialization failure.

Read the fresh worker result only after its actual zero exit and owned wait.
Authenticate every component manifest and inventory; verify exact method,
settings, geometry, fragment state and all-center gradient shape. Require real
SCF convergence and accepted actual stability. Check ghost/complex Gaussian
shell/center identity from authentic checkpoints. Assemble the total potential
and gradient only from all accepted genuine components. If any component is
missing, incomplete or invalid, retain its raw evidence and mark that batch
unavailable; no partial sum is a completed objective.

Aggregate results should preserve every original attempted component and batch,
optimizer outcome, request/source/installation identities, actual used ceilings,
raw native inventories and actual process exits. A final failed optimization
may retain earlier complete observations but must expose no invented final
geometry, energy, gradient, stationarity, uncertainty or minimum. Prefer separate
optional final fields rather than substituting the last successful batch.

## Independent final acceptance

After the optimizer reports success, run the fresh full component batch at its
actual proposed geometry. Independently reconstruct the CP/half-CP total and
selected-row gradient from authenticated native components. Require the actual
selected gradient norm to meet the declared tolerance. L-BFGS-B success can
mean that its projected gradient vanishes at an active displacement bound; that
does **not** establish ordinary stationary geometry. Reject a claimed ordinary
stationary result if its unprojected selected Cartesian gradient fails the
tolerance. If bound-constrained KKT stationarity is ever exposed, give it a
different typed definition with actual multipliers and active-bound evidence.

If all atom rows move and the actual full Cartesian gradient passes, the result
may report an unconstrained **stationary candidate** of this named CP total
potential. If only selected rows move, report a frozen-row constrained
stationary candidate. Neither passes a local-minimum gate until matched genuine
CP curvature is independently computed and characterized. Restricted stability
does not prove a unique SCF root, preserved chemical connectivity, absence of
static correlation or experimental accuracy.

## Meaningful focused tests to implement next

No tests below have been implemented or run for the proposed service.

**Mathematical and contract lane**

- Reject absent/repeated atom IDs, fragment overlap/omission, explicit
  charge/spin inconsistency, unsupported method/state, disabled stability,
  duplicate/unknown movable IDs and inconsistent resource ceilings before
  native launch. These are real contract evaluations, not mocked engines.
- Demonstrate the total-energy identity preserves native monomer deformation;
  CP interaction and CP total identities remain distinct. Pure exact algebra
  tests must be labeled mathematical checks, never method qualification.
- Verify CP versus half-CP request identity and weight, full-row gradient shape,
  truthful unavailable fields and attempted-versus-completed count semantics.
- Reject a real previously populated workspace and genuinely corrupted retained
  native artifact. Do not replace the engine with a stub for these tests.

**Mandatory genuine-engine lane**

- Small two-H2 HF/STO-3G fragments with explicit neutral singlet states provide
  a low-cost initial optimization/derivative fixture. Start away from the
  monomer bond stationary point. Freeze explicitly selected rows only if needed
  for a bounded first fixture, and label its result constrained stationarity.
  This cannot stand in for a full water-dimer CP minimum or binding accuracy.
- Independently calculate every complex/native/ghost component with PySCF and
  verify the assembled CP total energy and all-center analytic gradient. Move
  both an intrafragment nuclear row and a nonphysical center in a ghost monomer
  to ensure ghost response and monomer deformation contribute correctly.
- Use two genuine central-difference steps of independently calculated total
  energies. Check convergence of CP and explicit half-CP derivatives. Retain
  actual raw energies, steps, gradients, engine identity and tolerances; do not
  store predicted expected energies as if measured.
- Independently recompute final selected-row stationarity from the actual native
  complex/native/ghost gradient arrays. Confirm geometry actually changed,
  frozen rows remained unchanged, all accepted component SCFs converged and
  actual stability was checked. Verify the final verification batch is distinct
  from optimizer batches at equal coordinates.
- A genuine request with allowance for only one complete five-component batch
  should retain the original completed observations and then report actual
  budget exhaustion before a new batch, with final fields missing. A small
  displacement bound or insufficient iteration budget should likewise retain
  genuine nonstationarity and no false stationary geometry.
- Check original process binding, actual exit/wait, actual component counters,
  every retained artifact digest and the absence of surviving owned workers.
  Do not satisfy ownership tests using hand-written pretend launch receipts.

**Resource constraints for the next agent**

The parent limited this task to at most 20 MiB of new native artifacts outside
the checkout, one CPU/thread and a preserved 256 MiB `/tmp` reserve. At this
agent's final inspection `/tmp` had about 375 MiB free and `/workspace` about
382 MiB free; those measurements will become stale. Recheck before any run and
notify the coordinating agent before costly native execution. Never delete
original scientific results or protected environments to create room. Do not
run the full regression for this isolated design/implementation milestone.

## Next steps and qualification limits

Implement the three reserved paths, run the focused contract and genuine native
checks, independently review raw components/process evidence, and only then ask
the root agent to integrate a separately named typed stage, profile and mandatory
native Actions gate. Preserve failed first runs and their exact source identity.
Update W06 only to the bounded evidence actually obtained. W06's full De/D0,
CP curvature, thermochemistry and reference-accuracy requirements remain
unfinished. This design packet is not acceptance evidence for any SRS gate.
