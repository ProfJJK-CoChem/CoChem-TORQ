# Rovibrational geometric precursors

[rovibrational.py](../../src/cochem_torq/spectroscopy/rovibrational.py) computes
normal-mode cross products and analytic derivatives of the **full geometric
inertia tensor in fixed equilibrium axes**. It does not implement an effective
rovibrational kinetic operator, rotation-vibration alpha, B0, centrifugal
distortion or a complete VPT2/GVPT2 solver. Every result retains those products
as explicitly unavailable and keeps independent scientific qualification and
identification readiness false. This is a prerequisite implementation for
TORQ-VIB-003, not completion of that requirement.

## Coordinates, units and exact derivation

The isotope-specific geometry is centered at its actual mass-weighted COM.
The supplied equilibrium principal axes are orthonormal column vectors in the
input Cartesian frame with determinant +1. Row geometry vectors therefore
transform as `r_principal = r_centered @ axes_columns`. The same actual axis
matrix transforms each mode vector. Axes and mode phases are retained rather
than guessed or re-diagonalized at displaced geometries.

The existing `HarmonicResult` supplies orthonormal mass-weighted modes L with
`L.T @ L = 1`, expressed in electron-mass coordinates. With isotope masses
converted from u using the recorded constants service, define

\[
\mathbf r_a(Q)=\mathbf r_{e,a}+\sum_k\mathbf u_{ak}Q_k,
\qquad \mathbf u_{ak}=\mathbf L_{ak}/\sqrt{m_a}.
\]

Q has units `bohr*sqrt(electron_mass)`. This is a Cartesian-linear normal-Q
path, not a curvilinear nuclear-motion coordinate system. The COM/Eckart
conditions checked directly are

\[
\sum_a\sqrt{m_a}\mathbf L_{ak}=0,
\qquad
\sum_a\sqrt{m_a}\mathbf r_{e,a}\mathbin\times\mathbf L_{ak}=0.
\]

The geometric Coriolis precursor is defined with its sign/order fixed:

\[
\boldsymbol\zeta_{kl}=\sum_a\mathbf L_{ak}\mathbin\times\mathbf L_{al}.
\]

It is dimensionless and antisymmetric in k,l, with zero diagonal. The
mass-weighted orthonormality bounds every component's magnitude by one. For
the stated fixed-frame linear displacement, the geometric angular momentum
`sum_a m_a r_a(Q) cross dr_a/dt` has the bilinear term
`sum_kl zeta_kl Q_k dQ_l/dt`; the linear term vanishes by the checked Eckart
condition. This does not assemble the full coupled rotation-vibration kinetic
operator. The formula is specifically in normal Q, not dimensionless harmonic
q; substituting dimensionless q without its frequency factors is invalid.

For the geometric inertia tensor

\[
I(Q)=\sum_a m_a[(\mathbf r_a\cdot\mathbf r_a)1-
\mathbf r_a\mathbf r_a^T],
\]

direct differentiation gives

\[
I_{,k}=\sum_a m_a[2(\mathbf r_{e,a}\cdot\mathbf u_{ak})1
-\mathbf r_{e,a}\mathbf u_{ak}^T-\mathbf u_{ak}\mathbf r_{e,a}^T],
\]

\[
I_{,kl}=\sum_a m_a[2(\mathbf u_{ak}\cdot\mathbf u_{al})1
-\mathbf u_{ak}\mathbf u_{al}^T-\mathbf u_{al}\mathbf u_{ak}^T].
\]

The units are `electron_mass*bohr^2` for I,
`bohr*sqrt(electron_mass)` for its first derivative, and dimensionless for
its second derivative. The second derivative is symmetric in both Cartesian
indices and mode indices; its Cartesian trace is `4*delta_kl` for orthonormal
mass-weighted modes. These are tensor derivatives in fixed axes. Derivatives
of re-diagonalized principal moments require eigenvector response and can be
ill-conditioned at degeneracy; the module does not substitute them.

## API and input integrity

`build_rovibrational_precursors(harmonic, rotor, context, ...)` accepts actual
`HarmonicResult`, `EquilibriumRotor` and validated `ScientificContext` objects.
It checks complete positive mode count, actual external-space rank,
mass-weighted normalization, Cartesian and dimensionless transforms,
frequency/eigenvalue consistency, COM and rotational Eckart orthogonality,
declared external-Hessian residual policy, matching isotope/geometry/mode
digests and actual right-handed diagonalizing principal axes. It rejects
missing/placeholder source identities or mismatches rather than repairing them.
Its positive-curvature/residual checks do not independently establish optimized
geometry, the correct electronic state or native engine accuracy.

The output uses immutable tuples and frozen strict models, including copied
atom/isotope/mode/frame/recipe/protocol/parent identities, algorithm source
digest, constants digest and numerical conventions. Mutating an input
context's lists later cannot mutate the result. Typed JSON readback checks its
tensors against the retained actual geometric arrays and rejects inconsistent
values, units, degeneracy flags or qualification claims. Original Cartesian
geometry and mass-weighted modes are retained immutably so their exact digests
can be checked without reversing a floating-point axis transform. Readback also
reconstructs the validated source context, checks the retained COM and frame,
recalculates orthogonality/subspace residuals, and enforces the declared Hessian
residual gate. Copied equilibrium principal moments preserve the upstream
rotor's numerical zero-axis convention; their diagonal tensor is checked against
actual geometric inertia before zero-axis/degeneracy flags are recomputed. The
upstream rotor's canonical zero classification is independently recalculated
from retained original geometry and masses. A harmonic declaration digest binds
the exact supplied frequencies, source digest, mode digest and Hessian residual;
this is an internal integrity binding, not native artifact authentication.
Parent hashes identify
retained artifacts; the calling workflow must verify their original bytes.
This module does not authenticate arbitrary native outputs or promote an
unqualified method.

Every numerical tolerance is retained. Near-degenerate inertia axes and normal
modes are flagged under the explicit declared relative tolerances. Components
and signs depend on the supplied frame/mode gauge inside those subspaces;
they are not separate observable measurements. Linear H2 remains supported
geometrically with its zero-moment axis and degenerate perpendicular axes
marked. No inertia inversion, finite constant on the zero-moment axis or full
linear-molecule rovibrational model is inferred.

## Validation and remaining scientific work

[The tests](../../tests/test_rovibrational_precursors.py) calculate fresh stable
restricted-HF/STO-3G optimized H2 and water Hessians. They compare analytic
inertia derivatives against independently evaluated displaced Cartesian
inertia at two Q step scales, and compare cross products with independent
component contractions. They check tensor symmetries, trace identities,
COM/Eckart residuals, H2's exact geometric limit, proper rotation/translation
and mode-phase transformations, strict missingness and immutable provenance.
An actual isotope remassing check explicitly reuses the native Cartesian
Hessian under the Born-Oppenheimer approximation and reanalyzes masses,
COM, axes and modes; it is not an invented isotope-specific engine result.

Those bounded checks establish geometric software/algebra behavior. Full
VIB003 still requires a defined semirigid rovibrational Hamiltonian including
the required kinetic, rotation-vibration/Coriolis and resonance treatment,
actual alpha/distortion output, and authenticated independent molecular
frequency/alpha/distortion reference comparisons. No averaged inertia,
harmonic frequency or precursor tensor is relabelled as alpha/B0. No textbook
coefficient or bibliography is supplied by assumption; the equations above
follow directly from the recorded geometric convention.
