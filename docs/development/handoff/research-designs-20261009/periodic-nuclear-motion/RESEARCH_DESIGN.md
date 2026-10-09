# W04 periodic nuclear motion: unimplemented research design

Recorded 2026-10-09. This is an outside-checkout handoff, not implementation,
native calculation evidence, a validated scientific profile, or release evidence.
The checked-out HEAD at observation was
`cbf53f57109c6842ff1ca50025c522058026bfe4`.

## Exact work performed

The agent read the existing bounded Dirichlet/Eckart variational solver,
its tests, scientific context/precursor models, units, the PySCF adapter and
native artifact verifier. No production file was written. No test, engine call,
source qualification, or native molecular comparison was performed. No Git
commit was created. No dependent package was installed. No credential or
licensed asset was read. Scope was interrupted for repository handoff before
implementation began.

The requested owned paths remain absent from this agent's changes:

- `src/cochem_torq/spectroscopy/periodic_nuclear_motion.py`
- `tests/test_periodic_nuclear_motion.py`
- `docs/development/periodic_nuclear_motion.md`

Existing `spectroscopy/rovibrational_solver.py` implements a finite local
rectilinear Eckart chart with Dirichlet faces. Its genuine tests are not periodic
LAM evidence. Do not relabel that operator or its source receipts.

## Coherent first implementation boundary

Implement one scalar angular coordinate `phi` on `[0,2*pi)` with a periodic
self-adjoint kinetic operator, imported periodic potential, explicit isotope
and coordinate declarations, retained provenance, convergence comparisons and
immutable typed results. A bounded single-coordinate implementation would
make progress on W04 but would not complete TORQ-VIB-004 or V-DVR.

The safest first operator is the intrinsic one-dimensional constrained-curve
Laplace--Beltrami model, in atomic units (`hbar=1`):

```
g(phi) > 0
mu(phi) = sqrt(g(phi))
H psi = -[1/(2*mu)] d/dphi [mu/g * dpsi/dphi] + V(phi) psi
<u,v> = integral_0^(2*pi) mu(phi) conjugate(u(phi)) v(phi) dphi
```

`g` has units electron-mass*bohr^2; `V` and eigenenergies are hartree;
`phi` is an angle in radians and a dimensionless coordinate. This explicitly
defines constrained motion on a supplied one-dimensional metric. It does not
establish a general molecular torsion--rotation or complete J=0 rovibrational
Hamiltonian. Eliminating other internal/rotational coordinates can introduce
a different volume density and scalar quantum correction. In particular,
`sqrt(g)` must not silently replace the full configuration-space volume density.

An extension accepting a general positive imported measure `mu`, inverse
metric coefficient `a`, and an explicitly derived scalar correction `U` would
instead solve

```
H = -[1/(2*mu)] d/dphi [mu*a * d/dphi] + V + U.
```

Its kinetic reduction and measure must be independently justified by an actual
molecular coordinate derivation. Setting `U=0` or choosing a flat measure is a
model declaration, never a fallback for missing molecular physics. Keep this
extension separate until its derivation and provenance contract are complete.

## Proposed numerical method

Use complex integer Fourier functions `f_m(phi)=exp(i*m*phi)/sqrt(2*pi)`.
For the intrinsic model, assemble the weak form with positive periodic
trapezoidal quadrature weights `w_k`:

```
S_mn = sum_k w_k * mu_k * conjugate(f_mk) * f_nk
T_mn = 0.5 * sum_k w_k * (mu_k/g_k)
                         * conjugate(df_mk) * df_nk
P_mn = sum_k w_k * mu_k * V_k * conjugate(f_mk) * f_nk
H = T + P
H c = E S c; c.conj().T S c = identity
```

This form preserves self-adjointness and avoids a naive unsymmetrized variable
inertia differentiation matrix. Reject excessive Hermiticity error or a
nonpositive/ill-conditioned overlap. Never silently discard an imaginary part.
Report the original residual before any roundoff-only symmetrization. Retain
complex coefficient real/imaginary arrays and integer Fourier labels so a
consumer can replay normalization and eigenpairs.

Require an odd, uniform, endpoint-exclusive source grid so its discrete Fourier
interpolant has no ambiguous even-grid Nyquist convention. Interpolate the PES
as a real periodic trigonometric polynomial. Interpolating `log(g)` and
exponentiating is one possible globally positive metric model; explicitly
record that convention and its difference from interpolating `g`. Interpolated
values are derived model values, not additional engine evaluations. Sampled
metric positivity alone is insufficient to certify a raw Fourier interpolant
positive between samples.

Use two separate comparisons: larger Fourier cutoff at fixed integration grid,
and larger integration grid at fixed Fourier cutoff. Preserve sorted same-sector
low-level differences and the compared state count. A finite comparison is not
a rigorous error bound or external molecular calibration. PES sampling and
coordinate/metric model error require separate native sampling campaigns.

## Proposed typed input and provenance

Follow the existing strict, frozen Pydantic conventions and digest readback.
Revalidate model instances so model_copy/model_construct cannot evade gates.

1. Coordinate declaration: period, orientation, atom IDs, reference frame,
   construction of the complete cyclic path, fixed versus relaxed coordinates,
   isotope numbers/masses and mass-reference bytes, charge/multiplicity,
   coordinate metric definition, units, electronic recipe and zero-energy
   convention. Preserve stable atom ordering and one method/state at all nodes.
2. Cyclic PES: explicit node angles, actual node energies, positive metric
   nodes, source-grid/interpolant declaration and a content digest. Do not reuse
   one geometry's energy at another node. Do not invent endpoint energies,
   incomplete energies, mass metrics, or symmetry-equivalent engine receipts.
3. Molecular import: exact source declaration file hash plus authentic node
   result/manifest identities, actual retained inventory byte/size verification,
   geometry/state/method binding and any independent published source locator.
   Digests are binding identities, not authority to declare authenticity.
   An engine-calculation label alone is insufficient. Check checkpoint/request
   identity as needed rather than trusting an arbitrary self-authored JSON file.
4. Mathematical model input: a separately named construction path with
   `evidence_class=mathematical_model`. Its source digest binds the actual model
   declaration. Mathematical examples cannot supply molecular native evidence.
5. Budget: explicit maximum Fourier dimension, quadrature nodes, workspace
   bytes and retained levels; reject before allocating. A dense estimate is a
   preflight bound, not exact peak RSS or a hard operating-system quota.

Import authentic bytes before extracting scientific values. Preserve originals;
do not repair missing/broken source files or invent results to continue.
`engines/diagnostics.py:verify_native_artifacts` provides relevant existing
byte verification but is not by itself an independent authentication/signature
authority or a complete generalized torsional import contract.

## Symmetry and nuclear-spin boundary

Default to the full periodic integer Fourier space. Do not infer PI symmetry,
spin statistical weights, Berry phases, or antiperiodic boundaries from a
visual rotor pattern or point group.

A bounded explicit spatial sector can use the active translation convention
`(T psi)(phi)=psi(phi+2*pi/q)` with character
`exp(2*pi*i*r/q)`, retaining integers `m == r (mod q)`.
Admission must require:

- An explicit justified molecular permutation associated with that coordinate
  translation, stable atom mapping, its group order and immutable documentary
  provenance.
- Isotope/symbol preservation by the permutation; the declared operation raised
  to `q` must be identity.
- PES and kinetic metric invariance under the translation with a declared
  tolerance and checks of the interpolation as well as source nodes.
- An explicit spatial-character integer `r` and derivation connecting it to any
  claimed total spin/PI restrictions. A numerical phase is not itself proof
  that a spin sector exists or that its statistical weight is correct.

If spin derivation or full molecular permutation-inversion group is unavailable,
preserve that as unavailable. Do not automatically assign spin degeneracy,
forbidden transitions or partition-function weights. Integer Fourier sectors
alone do not implement coupled torsion--rotation symmetry.

## Meaningful first tests (not run)

All of the following are proposed mathematical/process checks, not molecular
accuracy qualification:

- Constant metric and constant PES: free-rotor energies
  `E_m=V0+m^2/(2*g0)`, exact periodic phases, `m`/`-m` degeneracy and weighted
  normalization.
- A mathematical threefold cosine potential: compare the independently derived
  exact finite Fourier Hamiltonian's off-diagonal selection `delta m=+/-3`,
  not a second invocation of the same assembly function.
- A smooth positive variable metric: compare independent numerical weak-form
  integrals and a coordinate reparametrization of a free circle. Test the volume
  measure and zero-mode normalization; do not assume a flat integration measure.
- Basis and quadrature refinements: retain both differences, make a deliberate
  failed refinement stay failed, and refuse mismatched surfaces, sectors or
  energy zeros.
- Reject negative/nonfinite/missing metric, repeated endpoint, nonuniform grid,
  complex physical input, invalid permutations, isotope-changing operations,
  false symmetry, inadmissible phase and over-budget allocation.
- Bind result content and matrix/overlap/coefficient hashes; fail altered or
  missing imported bytes. Preserve the mathematical evidence label and
  `independent_scientific_qualification=False`, `identification_ready=False`.

Do not write fixture JSON masquerading as real engine output. Real import tests
must use actual calculations or genuine retrieved original datasets. Contract
error tests may use explicit mathematical inputs; name them accordingly.

## Native candidate discussed but never calculated

A small rigid H2O2 torsional path might provide genuine open-engine import
coverage. The possible path would use ordered O,O,H,H, fixed O--O/O--H lengths
and valence angles, and actual RHF/STO-3G energies on an exclusive cyclic grid.
Those unoptimized geometry choices are model inputs, not published equilibrium
structures. Derive the tangent metric with the same explicit isotope masses and
frame constraints at every node. Report the exact constrained-curve scope.
This is a candidate only: no coordinates, energies, metric values, receipts,
convergence, isotope choices, reference source or native qualification were
generated. No numerical molecular value is supplied in this design.

Before any native study coordinate resource authority with root: one CPU/thread,
<=20 MiB new native artifacts outside the repository, and preserve at least
256 MiB `/tmp` free space. Do not start a large grid, full regression, student
deployment or licensed-engine staging from this handoff alone.

## Next agent work and release status

1. Independently review the coordinate/measure physics and select the constrained
   1D model boundary before writing the three owned files.
2. Implement mathematical model and authenticated import as separate contracts;
   assemble weak-form matrices, immutable receipts and convergence comparison.
3. Run the mathematical/process tests plus scoped Ruff/mypy/test-policy checks.
4. Acquire or compute a genuine bounded cyclic dataset under actual resource
   authority, preserve native bytes and run import/replay checks. This establishes
   integration, not independent spectroscopic accuracy.
5. For molecular qualification retrieve independent published torsional levels,
   reproduce its kinetic/PES/coordinate and isotope conventions, and calibrate
   observables at preregistered convergence/domain bounds.
6. Generalize to coupled coordinates and J-dependent torsion--rotation only
   after separate kinetic derivations, PI/spin treatment, numerical limits and
   actual independent molecular cases.

W04, TORQ-VIB-004 and V-DVR remain unfinished and blocked. Do not alter shared
advanced_products, application, registry, CI, normative SRS or acceptance audit
to enable an unimplemented profile. No human missing opinion has been identified:
the remaining derivation, coding, open-source calculation and online-source
retrieval are agent research work. Restricted source/hardware authority can be
documented if it actually blocks a later selected campaign.
