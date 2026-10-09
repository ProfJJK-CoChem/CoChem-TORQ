# W08 bounded genuine mass-weighted IRC implementation handoff

Status: **design only; no implementation, native IRC calculation, or new test has been completed**. The user redirected the parent task to an agent handoff before any repository files were written. There are no IRC production changes to preserve or remove. This document is outside the repository and is not scientific qualification evidence.

## Proposed ownership and integration boundary

Implement a new `src/cochem_torq/engines/irc.py`, `tests/test_irc_integration.py`, and `docs/development/irc.md`. The existing `src/cochem_torq/engines/pathway.py` must retain its truthful candidate-only API: its local imaginary-mode displaced minimizations are not IRC trajectories. Prefer a separate callable accepting a supplied target-level TS geometry and independently characterizing it; expose optional integration only after the new route is validated. Root owns application/registry/CI/audit changes.

## Inspected current code

- `engines/pathway.py`: actual geomeTRIC restricted-HF saddle optimization, independent target-level final Hessian/gradient/stability characterization, real displaced endpoint minimizations. Its `PathwayResult.ts_verified` is permanently false and its `irc.status` is unavailable. Atom IDs are mandatory, with tabulated isotope masses resolved by `Libraries.cochem_isotopes`.
- `engines/diagnostics.py`: `verify_native_artifacts` authenticates every retained native artifact, binds scientific values to `result.json`; `stationary_point_diagnostics` projects the actual Hessian and classifies first-order saddle candidates without certifying reaction identity.
- `engines/pyscf_backend.py`: closed-shell all-electron real PySCF energy, gradient, and HF analytic Hessian; sealed manifests; explicit basis/settings and engine installation fingerprints. `_normalize` enforces supported state/method definitions. `_seal` excludes nested files named `manifest.json`, so the new parent receipt must explicitly bind each nested manifest digest.
- `engines/constrained_optimization.py`: existing `_OWNED_WORKER_LAUNCHER` can execute a JSON backend request through its `_native_worker`; it binds the original parent PID/create-time and worker PID/create-time, applies Linux `PR_SET_PDEATHSIG`, and starts a separate process group. `_stop` performs bounded termination/wait. Reuse these original native ownership conventions rather than launching unauthenticated workers.
- `cochem/orchestration/host_allocation.py`: `_valid_process_start_time` accepts exact finite positive JSON int/float timestamps; do not regress the prior RFC8785 integer-versus-float fix by requiring `type(timestamp) is float`.
- `spectroscopy/harmonic.py`: real projected modes are orthonormal in electron-mass mass-weighted coordinates. When choosing isotope-mass-unit path coordinates, explicitly normalize the transformed Cartesian mode; do not silently mix units.
- `tests/test_pathway_diagnostics_integration.py`: current genuine planar NH3 restricted HF/STO-3G saddle fixture, stable one negative mode, two real displaced opposite-umbrella minima. This is a useful seed-generation route, not an already validated IRC.

## Proposed fail-closed initial scope

A first bounded experimental profile can support restricted all-electron HF/STO-3G, closed-shell singlet, 3–10 atoms, explicit unique persistent atom IDs and isotope records, one CPU thread, at most 1024 MiB requested engine memory. Keep unrelated methods, open-shell states, density fitting, ECP, dispersion, solvent, and frozen core unavailable. Public local/native software evidence does not qualify student deployment or a different target-level method.

Normalize the exact method request before acquiring resources. Require actual stability checking. Calculate seed energy, gradient, Hessian, and stability with the same target-level method. Reject before continuation unless native artifacts authenticate, gradient maxima satisfy the declared stationarity threshold, projected Hessian invariance satisfies its threshold, there is exactly one internal negative mode, and no unresolved internal zero modes remain. A supplied geometry or user-labeled TS is never sufficient.

Preserve all actual earlier evaluations on failure. Never manufacture a Hessian, energy, gradient, missing point, endpoint identity, or minimum classification. Endpoint connectivity and chemical-mode relevance remain explicitly unverified until independent evidence is supplied and checked.

## Proposed numerical route

Use isotope-unit coordinates `q = sqrt(M_u) x`, with `x` in bohr and fixed atom row order. The normalized steepest-descent ODE is

`dq/ds = -M_u^(-1/2) grad_x(E) / ||M_u^(-1/2) grad_x(E)||`.

State arc-length units as `bohr * sqrt(u)`; gradient units as `hartree / (bohr * sqrt(u))`. Starting at zero gradient is singular. Obtain the actual negative Cartesian mode from seed characterization, convert to a unit vector in `q`, and evaluate small displacements in both signs. Each seed displacement must be confirmed by a genuine same-method gradient and lower energy. The displacement is a seed construction, not a completed IRC.

For every later step, compute an Euler predictor using the accepted gradient, evaluate its actual gradient, and construct a Heun corrected coordinate from the two normalized directions. Evaluate the corrected point independently. Accept only if the declared embedded Euler/Heun discrepancy is within the local error tolerance, the actual energy decreases within a tight explicit numerical tolerance, the geometry is finite/noncoincident, and atom/state/method identities are unchanged. Record rejected predictors/correctors as actual attempted evaluations. Reduce step on failed error/descent checks; terminate with an explicit finite-prefix reason at the minimum step or exhausted budget.

An embedded Euler/Heun estimate is a local numerical error observation, **not global step-size convergence proof**. A genuine independent smaller-step replay is still needed to substantiate path convergence. Near a minimum, normalized-gradient flow becomes singular; stop at a declared actual gradient threshold, then independently calculate a genuine endpoint gradient/Hessian. Classify a local minimum only if stationary diagnostics accept it. Do not call an exhausted-step endpoint a minimum.

Report each direction independently: accepted finite numerical path prefix; reason for termination; actual cumulative arc length; accepted and rejected evaluations; local error observations; available/unavailable stationary endpoint. A finite numerical path can be useful while full reaction connectivity stays unavailable. `ts_verified` and `reaction_connectivity_verified` must remain false until explicit chemically relevant mode and mapped electronic/structural endpoint acceptance are validated.

## Resource and native evidence requirements

Declare maximum total physical engine calls, steps per direction, rejected attempts, total wall time, per-evaluation wall time, cumulative path length, maximum/minimum step, memory ceiling, and artifact budget. Resource checks precede every launch; one original parent deadline governs all stages, not reset timers per direction. Observe actual parent/worker identity, wait the original worker, preserve authentic original input/owner binding/wait/log/native manifests. Enforce thread environment variables as well as backend `threads=1`.

Sample worker RSS transparently; it is not an exact kernel peak or hard quota. Retain unavailable memory observation when unreadable. Stop worker process groups on deadline or sampled-memory excess; parent death containment remains mandatory. Bind normalized launch molecule/method/settings, actual returned geometry, stable converged SCF, actual engine version, and every scientific point to its native manifest digest. Independently re-read manifest contents and actual bytes rather than trusting supplied native dictionaries.

A top-level manifest alone is insufficient if nested native manifests are omitted by `_seal`; retain and hash original native manifest references in parent records. A later readback should verify actual request/owner/wait/native lineage and point-to-native agreement, while saying that cryptographic integrity alone does not establish scientific algorithm validity.

## Planned focused checks; none run in this agent task

1. Pure mathematical tests of `M^(-1/2)` gradient transformation, mode normalization, coordinate round-trip, constant-direction Euler/Heun agreement, nonconstant-direction local error scaling, invalid masses, finite values, and step ceilings. These are algorithm checks, not quantum engine evidence; no synthetic backend fixtures or scientific-output replacements.
2. Real NH3 inversion: generate the genuine existing saddle with endpoint minimization disabled; independently characterize it in the new owned native route; follow both directions through several genuine normalized-gradient points. Check atom mapping, actual monotone same-method energies, opposite umbrella displacement, sealed native lineage, accepted-error thresholds, retained finite-prefix limits. Do not report full endpoint/IRC connectivity if the bounded run stops early.
3. Actual nonstationary seed rejection using a real displaced geometry and real Hessian/gradient; no surrogate native dictionaries.
4. Genuine exhausted-budget behavior with actual earlier native points retained and unavailable later-stage quantities.
5. Independent smaller-step path replay and endpoint characterization before any stronger numerical convergence claim.

Before expensive checks, coordinate with root. At initial inspection `/tmp` had approximately 375 MiB free and `/workspace` approximately 382 MiB. Preserve at least 256 MiB reserve and cap new evidence in this task below 20 MiB. No dependency installation or Git commit is authorized for this subtask.

## Work actually performed

Read-only repository inspection and design, plus this external handoff document. No production code, tests, native calculations, new scientific values, CI run, dependency installation, or Git commit. No existing files were modified or deleted. Full W08 implementation and scientific qualification remain agent-owned future work.
