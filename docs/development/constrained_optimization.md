# Genuine constrained energy optimization

`hf-sto-3g-constrained-pes-validation` supplies a bounded local validation route
using actual restricted HF/STO-3G energies, analytic nuclear gradients and,
when requested, an analytic HF electronic Hessian from PySCF 2.14.0. The
optimizer is the installed SciPy 1.18.1 SLSQP implementation. Its exact Python
and native-library bytes are fingerprinted in the result. This small-basis HF
route establishes execution and derivative consistency; it does not establish
research accuracy or high-accuracy rotational predictions.

`execute_constrained_optimization(molecule, specification, destination,
resources=..., profile=...)` consumes the exact registered profile and a
`ConstrainedOptimizationSpec`. The declaration supplies stable atom IDs,
zero-based immutable atom order, explicit movable Cartesian atom rows, and
typed bond, valence-angle or signed periodic-torsion targets. Input targets
must lie in their declared domains; periodic upper endpoints are excluded,
and valence angles must lie strictly between zero and pi. Bonds are converted
to bohr and angular residuals to radians. Fixed rows retain their initial
coordinates exactly. Redundant or singular constraint Jacobians are rejected.

The SLSQP objective is the actual absolute electronic energy, with actual
analytic Cartesian gradients. Constraint Jacobians use the internal-coordinate
definitions shared with the scan subsystem: analytic bond derivatives and
central differentiation for angles/torsions. Geometry construction or squared
distance to the initial geometry is not an energy objective. No synthetic
energies, gradients, Hessians or convergence substitutes are introduced.

Each distinct objective geometry launches a real PySCF worker and retains the
original request, logs, checkpoint, artifact inventory and typed observation in
`evaluations/evaluation-NNNN/`. Repeated requests for exactly the same immutable
geometry and recipe reuse their actual observation and increment the cache-hit
count. A separate physical call is reserved for fresh final energy/gradient
verification; requested analytic curvature is evaluated in that final call.
If an engine, resource ceiling or mathematical condition fails, the attempt
stops and retains earlier observations. An incomplete optimization never
publishes a qualified final molecule or electronic energy.

Physical-call limits, overall remaining wall budget, per-call wall limits and
observed available host memory are checked before launch. A parent monitors
actual native-worker RSS and elapsed wall time, and terminates the process group
on a ceiling. RSS is sampled; exact CPU core seconds remain explicitly
unavailable. Linux parent-death containment is mandatory: the child activates
`PR_SET_PDEATHSIG(SIGKILL)` before scientific imports, then verifies its expected
parent PID and creation time. Unsupported platforms and owner-change races
fail closed. The evaluation directory and backend request exist before launch;
`owner-binding.json`, `parent-process-observation.json` and
`worker-process.json` retain actual process identity and waited-exit evidence.
An outer process dying is not itself proof of an inner process's death;
allocation recovery must inspect the bound identities and preserve unknown
state until actual death is established.

Final stationarity is independently evaluated on the selected movable rows.
For constraint residual vector `c` and Jacobian `J`, the convention is
`L = E - lambda.c`. SVD supplies an orthonormal basis `Z` for the null space
of `J`. The multiplier solve uses the same declared absolute singular-value
cutoff as the rank check. Qualification requires actual constraint residuals
within the declared bohr/radian tolerance and `Z.T @ gradient` within the
declared hartree/bohr tolerance. Independent multiplier residual and tangent
projection must agree. Reaching a Cartesian displacement bound prevents
qualification. Fixed-atom forces are not silently treated as zero.

Optional constrained curvature uses the actual analytic electronic Hessian
and `H_L = H_E - sum(lambda_i * H_c_i)`. Constraint Hessians are computed by
differentiating the actual coordinate Jacobians at two distinct steps. Both
raw Hessian sets, raw Lagrangian Hessians and numerical convergence/symmetry
checks are retained. Symmetrization occurs only after those checks pass.
Reported curvature is `Z.T @ H_L @ Z` in an unweighted Cartesian tangent
basis, with an explicit zero-mode tolerance. A failed curvature stage remains
unavailable while a valid earlier constrained stationary geometry survives.

The typed `ConstrainedOptimizationResult` separates `initial_molecule` from
`final_molecule`, includes every actual invocation and the independent final
native manifest digest, and exposes `available`, `partial` or `failed` status.
Its successful character is **constrained stationary**, never an unconstrained
minimum or equilibrium geometry. Positive constrained curvature cannot justify
equilibrium rotational spectroscopy. High-accuracy methods, experimental or
independent cross-engine benchmarking, campaign authority fencing and deployed
product qualification remain separate gates.

`tests/test_constrained_optimization.py` performs genuine water bond/angle
energy relaxation, a fresh same-engine final reference and separate displaced
SCF energy checks, analytic constrained-Hessian checks, real call/wall limit
stops, declaration/rank rejection and actual observed native-worker death after
owner SIGKILL. Same-engine consistency and these local tests do not constitute
independent scientific accuracy calibration.
