# Bounded constrained-relaxation scan experiment

The separately named recipe `hf-sto-3g-relaxed-internal-pes-validation` connects
the existing owned RHF/STO-3G constrained optimizer to TORQ's scan, plan,
approval and `execute` interfaces. The exact inner electronic model and SciPy
SLSQP algorithm retain their existing recipe identity. This is a research
validation route; it does not enable a production method-matrix profile.

The first experiment supports neutral singlet water in stable `O,H,H` order,
one nonperiodic bond or angle coordinate, frozen oxygen and movable hydrogen
rows. It permits one CPU core, at most 1024 MiB RAM, at most 180 seconds total,
and no more than three scheduled optimizer attempts. The existing full-grid
contract requires independent forward, reverse and challenge attempts, so this
initial limit permits **one unique target**, repeated three times. It does not
establish a sampled curve, a complete PES or conformer-search completeness.

Each target first receives a purely mathematical coordinate embedding. An actual
constrained electronic-energy optimization then relaxes the unconstrained
hydrogen coordinates. A fresh final native calculation must independently
provide energy, gradient, checkpoint state, the prescribed coordinate residual
and a sufficiently small tangent gradient. Constrained stationarity does not
establish an unconstrained minimum, an equilibrium geometry, a positive
constrained Hessian, a spectroscopic prediction or experimental accuracy.

## Run the experiment

With the repository's pinned scientific environment active, inspect and approve
the supplied input before execution:

```bash
python -m cochem_torq validate --request examples/relaxed_water_scan_request.json --execution local_validation --json
python -m cochem_torq plan --request examples/relaxed_water_scan_request.json --execution local_validation --output /tmp/relaxed-water-plan.json --json
python -m cochem_torq approve-plan --plan /tmp/relaxed-water-plan.json --actor lab-researcher --scratch-mb 16 --output /tmp/relaxed-water-approval.json --json
python -m cochem_torq execute --request examples/relaxed_water_scan_request.json --approved-plan /tmp/relaxed-water-approval.json --output-dir /tmp/relaxed-water-run --json
```

Use new, empty output paths. Approval binds the actual request, source inventory,
outer scan recipe, inner solver recipe and resource ceilings. This experiment
has no production Actions-submit qualification; the repository's genuine native
CI tests exercise the approved local-validation contract within Actions.

## Counts, failures and original evidence

`point_attempt_count` counts the independent outer optimizations.
`physical_call_count` counts the actual inner native evaluation attempts, which
include failed native calls. They have separate ceilings. Exact-geometry
optimizer cache reuse does not add a native call. A result with incomplete
original launch evidence keeps its exact native count unavailable and reports
only a verified lower bound. Missing stationarity, energy, gradients or state
remain missing; successful earlier inner evaluations stay in the original
inventory and do not become a replacement final result.

The whole-point clock begins before coordinate construction and admission, and
the parent checks that deadline during native execution and final readback.
The shared host ledger admits one outer point worker and verifies its original
nested constrained-engine launch inventory before releasing resources. The
parent samples aggregate resident memory of the actual owned descendants.
Sampled resource maxima do not establish an exact peak or CPU-core-second
measurement. Unknown child death keeps the original host reservation unresolved.

Each original point directory retains `constrained-engine/result.json`, every
evaluation's input, typed observation, owner/parent/wait receipts and native
inventory. The point's native manifest path names the original final evaluation
under `constrained-engine/evaluations/`; it is never copied into a fixed-node
directory. The collector checks the molecular identity, frozen oxygen row,
exact inner model, complete original attempt inventory and final native bytes
before recomputing constraint and tangent-gradient evidence.

Independent restarts compare actual energies and Cartesian coordinates in the
declared frame. Cartesian differences are reproducibility flags; they do not
prove distinct conformers or physical electronic states. AO-density differences
are available only when the final molecular identities and coordinates match
exactly. Densities at different nuclear centers have no interchangeable AO
metric here, so their comparison remains unavailable.

The existing fixed-native HDF5/landscape bridge does not accept this recipe.
Retain the actual relaxed JSON and native artifacts. Periodic coordinates,
multiple unique targets, continuation, adaptive relaxation, curvature/TS checks,
post-HF models and accuracy qualification require further implementation and
genuine validation. The student pilot remains on hold.

## Required validation

`tests/test_relaxed_internal_scans.py` exercises direct invalid declarations,
separate point/native budgets, local-only recipe identity, genuine CLI execution,
free-coordinate motion, an independent analytic single-bond KKT projection,
original native/owned-wait provenance, center-correct density comparisons and a
genuinely exhausted inner budget. Native CI explicitly selects this file. A
successful mathematical test selection cannot qualify an unrun scientific route.
