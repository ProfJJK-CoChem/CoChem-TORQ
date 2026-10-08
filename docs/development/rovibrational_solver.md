# Bounded variational rotation–vibration model

The [solver](../../src/cochem_torq/spectroscopy/rovibrational_solver.py) implements
a finite, local nuclear-motion Hamiltonian derived from the Cartesian kinetic
energy. It couples rotation, normal coordinates, the retained cubic/quartic
electronic potential, and the coordinate volume element. Its eigenenergies are
real computed results for the specified model. They do not constitute complete
semirigid VPT2, fitted Watson constants, calibrated ground-state rotational
constants, or an astronomical identification catalog. `TORQ-VIB-003` remains a
separate requirement.

## Coordinates and metric

The input is a fully validated [geometric precursor
record](rovibrational_precursors.md) plus a typed force field. Their geometry,
isotopologue, atom order, normal modes and phases, principal-axis frame,
electronic state, harmonic source, electronic recipe, and evidence class must
agree. The force field must retain its two-scale derivative comparison and pass
that comparison. Engine evidence requires actual native displacement digests.
Digests identify retained artifacts; callers still authenticate their bytes.

For the retained mass-orthonormal modes (L_{ai}), body-frame coordinates are

\[
 r_a(q)=r_{e,a} + m_a^{-1/2}\sum_iL_{ai}\frac{q_i}{\sqrt{\omega_i}},
 \qquad q_i=\sqrt{\omega_i}Q_i.
\]

Here nuclear masses are in electron masses, distance is in bohr, \(\omega_i\)
is the harmonic frequency in atomic units, and \(q_i\) is dimensionless.
The reference modes satisfy the translational and rotational Eckart conditions.
The body frame is retained at every displacement; it is not re-diagonalized.
For body angular velocity \(\Omega\), direct substitution into
\(T=\frac12\sum_am_a|\Omega\times r_a+\dot r_a|^2\) gives

\[
 T=\tfrac12\begin{pmatrix}\Omega\\\dot q\end{pmatrix}^{T}
 G(q)\begin{pmatrix}\Omega\\\dot q\end{pmatrix},\quad
 G=\begin{pmatrix}I&C\\C^{T}&D\end{pmatrix},\quad
 D_{ij}=\delta_{ij}/\omega_i.
\]

The rotational cross metric is

\[
 C_{ai}(q)=\sum_j\frac{q_j\zeta_{a,ji}}{\sqrt{\omega_j\omega_i}},\qquad
 \zeta_{a,ji}=\sum_b(L_{bj}\times L_{bi})_a.
\]

The Schur complement \(A=I-CD^{-1}C^T\) is the **effective metric**, not an
averaged principal moment. The full Cartesian-linear inertia is evaluated from
the exact retained first and second geometric derivatives.

## Quantum operator and body-fixed convention

Quantization uses the Laplace–Beltrami operator for this coordinate metric.
The implementation uses its weak form and retains the nonconstant volume
element. Relative to equilibrium, that element is

\[
 w(q)=\sqrt{\det A(q)/\det I_e}.
\]

The constant factor \(\sqrt{\det D}\) and orientation normalization cancel
between the Hamiltonian and overlap matrices. On the rotation group SO(3),
body-fixed rotation vector fields are \(X_a=iJ_a\), with
\([J_a,J_b]=-i\epsilon_{abc}J_c\) and \(\hbar=1\). The matrices are the
complex conjugates of the usual space-fixed integer-J matrices in increasing
body projection \(K=-J,\ldots,+J\). The same convention is used in the Coriolis
cross term. Its sign is not independently guessed from a frequency formula.

For a wavefunction in a fixed-J sector, the kinetic quadratic form is

\[
 \langle\psi,T\psi\rangle = \frac12\int w\left[
 (\partial_q\psi)^\dagger D^{-1}(\partial_q\psi)
 + (iJ\psi-CD^{-1}\partial_q\psi)^\dagger
 A^{-1}(iJ\psi-CD^{-1}\partial_q\psi)\right]dq.
\]

This expression follows directly by block inversion of \(G\). It is positive
when the chart is nonsingular. It retains rotational, Coriolis, and additional
vibrational kinetic contributions. Coordinate-volume derivatives are included
by the weighted weak form; no flat-measure transformation is performed and no
Watson pseudopotential is added. Adding such a pseudopotential here would
double count a different representation of the volume effect.

The retained potential relative to the electronic reference energy is

\[
 V(q)-E_e=\tfrac12\sum_i\omega_iq_i^2
 +\frac1{3!}\sum_{ijk}\phi_{ijk}q_iq_jq_k
 +\frac1{4!}\sum_{ijkl}\phi_{ijkl}q_iq_jq_kq_l.
\]

This assumes a stationary electronic reference. Positive projected curvature
alone does not establish stationarity. An actual engine gradient or an explicit
analytic mathematical model must establish the omission of a linear term.
The required `StationaryReference` retains the actual ordered gradient, its
source-artifact identity, the geometry and evidence class, and the declared
gradient tolerance. The source must be an actual retained reference parent.
An absent gradient cannot become a zero gradient. Any accepted finite residual
remains explicitly recorded as an approximation when omitting the linear term.

## Self-adjoint domain and chart certificate

The domain is **SO(3) × a declared finite box**
\(-a_i<q_i<a_i\), with Dirichlet boundary conditions on every box face.
It is not an unannounced truncation of a Gaussian integration over all space.
The Friedrichs extension of the positive kinetic quadratic form, together with
the bounded-on-this-domain polynomial potential, defines the selected
self-adjoint local model. The sine product basis vanishes on all faces:

\[
 \varphi_{n_i}(q_i)=a_i^{-1/2}
 \sin\left[\frac{n_i\pi(q_i+a_i)}{2a_i}\right],\quad
 n_i=1,\ldots,N_i.
\]

Neither a local normal-mode chart nor its Dirichlet boundary is an exact global
molecular coordinate space. Linear molecules, chart transitions, large-amplitude
motion, tunneling paths, spin and nuclear permutation restrictions require
different models. The uniform metric certificate proves local nonsingularity,
not global injectivity or absence of duplicate molecular configurations under
atom permutations or chart overlap.

The solver certifies positive metric over the **entire closed box**, not only
the quadrature samples. Let \(B(q)\) be the Cartesian mass-weighted rotation
map with columns \(\sqrt m(e_a\times r(q))\), and \(P=1-LL^T\). Then

\[
 A(q)=[PB(q)]^T[PB(q)].
\]

Eckart orthogonality gives \(PB(0)=B(0)\). Writing
\(B(q)=B(0)+\sum_i q_i B_i/\sqrt{\omega_i}\), the singular-value perturbation
bound yields

\[
 \lambda_{\min} A(q) \ge
 \left[\sqrt{\lambda_{\min}I_e}
 -\sum_i\frac{a_i}{\sqrt{\omega_i}}\|PB_i\|_2\right]^2.
\]

The bracket must be strictly positive with a numerical margin. A failed
certificate stops execution even if all sampled metrics happen to be positive.
This sufficient condition is deliberately conservative: its failure does not
prove that the box contains a singularity, but the solver does not claim a
uniform certificate it has not established.

## Numerical solution and result meaning

Tensor-product Gauss–Legendre integration gives a Hermitian Hamiltonian \(H\)
and positive overlap \(S\). A generalized Hermitian eigenproblem
\(Hv=ESv\) is solved separately for the requested J sectors. The finite basis
dimension, integration-node budget, residuals, normalization, and actual
Hamiltonian/overlap/coefficient digests are retained. Each eigenenergy is split
into primitive vibrational kinetic, metric-induced vibrational kinetic,
rotational, Coriolis, and potential expectations; their sum must equal the
retained eigenenergy. The laboratory M degeneracy is explicitly \(2J+1\).
The protocol also enforces a conservative dense-array workspace estimate;
approval of that budget is not a statement that the host has that memory.

Providing a fresh `artifact_directory` retains the actual complex Hamiltonian,
overlap, eigenvector columns, eigenenergies, separate operator contributions,
and source/protocol/stationarity metadata in exclusively created named NPZ files. Their
actual file digests are included in the result. An existing directory is
rejected. When no directory is provided, the result explicitly marks matrix
archives unavailable; their hashes do not pretend to be retained files.

All couplings inside a finite basis are diagonalized together. This treats
near-degeneracies inside that basis without perturbative denominators or
arbitrary denominator shifts. It does not establish that an omitted resonant
state is negligible.

`compare_local_variational_results` compares exactly one parameter family at a
time: basis size, quadrature order, or box width. It records the actual energy
difference and whether the requested numerical tolerance passed. A tolerance
pass is a comparison of sorted low eigenenergies, not an independently
calibrated molecular accuracy bound. All three axes need investigation; a
quadrature pass alone cannot establish basis or boundary convergence.

`rovibrational_energy_difference` returns a bound difference between two
actually retained eigenstates. States retain J and sorted eigenstate index.
The routine assigns neither a vibrational band nor an allowed transition.
Selection rules, transition intensity and calibrated identification readiness
remain unavailable. Effective A/B/C, Watson distortion parameters, and
rotation–vibration alpha are explicitly absent; no inverse-inertia expectation
is relabelled as one of these quantities.

## Verification and scientific qualification

The [tests](../../tests/test_rovibrational_solver.py) independently check the
body-fixed angular-momentum algebra, atomic-velocity Gram metric, full-metric
weak form against the Schur-complement implementation, a separable Dirichlet
constant-metric limit, uniform-domain rejection, source-bound state
differences, and separate convergence comparisons.

The genuine integration case optimizes water with restricted HF/STO-3G,
checks its actual gradient, constructs its genuine harmonic Hessian and
**107 actual displaced electronic energy calculations**, retains each native
manifest, and solves J=0 and J=1 models. This qualifies the native input path
and numerical implementation on that bounded case. HF/STO-3G is a validation
recipe, not a spectroscopy-accuracy recipe. Small test boxes/bases verify
implementation; they do not establish boundary/basis convergence or molecular
identification accuracy.

The completed focused verification passed **39 tests** without skips. For the
retained water case, the actual maximum reference-gradient component was
\(3.8743\times10^{-8}\) hartree/bohr. The small test protocol uses box
half-widths `(0.25, 0.25, 0.25)` and sine basis counts `(2, 2, 1)`; it retains
four J=0 and six J=1 eigenstates. Native optimized-reference and displacement
artifacts contribute 108 retained parent identities. Actual Hamiltonian,
overlap and eigenvector archives are replayed against the generalized
eigen-equation in the test.

At that same box/basis, quadrature `(9, 9, 7)` versus `(11, 11, 9)` differed by
**0.1269505373 cm⁻¹** and failed the declared 0.001 cm⁻¹ numerical threshold.
The tightened `(13, 13, 11)` versus `(15, 15, 13)` comparison differed by
**2.4562 × 10⁻⁸ cm⁻¹** and passed that threshold. Both observations are retained;
the first is not overwritten by the second. This is integration convergence
for the specified small finite model, without basis/domain or independent
molecular accuracy qualification.

Useful historical background is Eckart's separation of rotation and vibration,
Podolsky's coordinate-invariant kinetic quantization, and Watson's semirigid
Hamiltonian reduction. Primary full texts and citation metadata could not be
retrieved through the current restricted network (Crossref requests returned
HTTP 403). These references are therefore background leads, not independently
verified publication evidence in this release. The implementation's equations
above are explicitly derived from the Cartesian kinetic metric and checked
against an independent contraction; it does not claim to reproduce a published
Watson parameter implementation.
