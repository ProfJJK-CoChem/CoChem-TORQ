# Canonical nonresonant Watson spectroscopy milestone

The [implementation](../../src/cochem_torq/spectroscopy/rovibrational_perturbation.py)
derives leading vibration–rotation corrections from the complete retained
Eckart mode basis, actual cubic/quartic force field and actual reference
gradient. It also evaluates nonresonant semirigid vibrational energies including
Coriolis and Watson kinetic terms, and the unreduced harmonic quartic
centrifugal response. These are explicitly selected molecular models with
applicability gates, not calibrated identification predictions.

The existing [bounded variational solver](rovibrational_solver.md) and
vibrational-only perturbation result remain separately named calculations.
Neither is silently substituted for this operator.

## Exact local operator and volume transformation

In mass-weighted normal coordinates \(Q\), with \(\hbar=1\), the complete
Cartesian-linear Eckart chart has the geometric metric

\[
 G=\begin{pmatrix}I&C\\C^T&1\end{pmatrix},\quad
 C_{ai}=\sum_j\zeta_{a,ji}Q_j,\quad
 A=I-CC^T,\quad\mu=A^{-1}.
\]

The weighted volume is proportional to \(\sqrt{\det A}\). Transforming a
wavefunction to the fixed volume by \(\phi=(\det A)^{1/4}\psi\) replaces
\(\partial_i\) by \(\partial_i-b_i\), where
\(b_i=\frac14\operatorname{Tr}(\mu\partial_i A)\).
The J=0 vibrational inverse metric is \(M=1+C^T\mu C\). Direct differentiation
of the transformed quadratic form gives the scalar

\[
 U=\tfrac12\left[\partial_i(M_{ij}b_j)+b_iM_{ij}b_j\right].
\]

For a complete orthonormal Eckart basis this equals the Watson scalar
\(U=-\operatorname{Tr}(\mu)/8\). The code independently evaluates the left
side from the retained first/second geometric derivatives and checks that
identity. It does not merely insert the right side and describe an assumed
equality as a test. The complete-mode and nonlinear-chart requirements matter:
the identity is not assumed for an arbitrary incomplete mode subset, linear
rotor or general curvilinear coordinate system.

The corresponding fixed-measure local Watson operator is

\[
 H=\tfrac12p^Tp+V(q)+\tfrac12(J-\pi)_a\mu_{ab}(J-\pi)_b
 -\tfrac18\operatorname{Tr}\mu,
 \qquad \pi_a=\sum_{ji}\zeta_{a,ji}\sqrt{\omega_i/\omega_j}\,q_jp_i.
\]

Here \(q_i=\sqrt{\omega_i}Q_i\), \([q_i,p_j]=i\delta_{ij}\),
and the retained body-fixed generators have
\([J_a,J_b]=-i\epsilon_{abc}J_c\). Cross products and operator sums use that
same convention. Nuclear isotope masses are in electron masses, geometry in
bohr, and all Hamiltonian coefficients in hartree. The coefficient multiplying
\(J_aJ_b\) is \(\beta_{ab}=\mu_{ab}/2\), not \(\mu_{ab}\).

## Physical mass bookkeeping

The formal nuclear-mass family is \(m(\epsilon)=m/\epsilon^4\). With a fixed
Born–Oppenheimer electronic potential and fixed dimensionless normal coordinate,
harmonic frequencies scale as \(\epsilon^2\omega\) and Cartesian vibrational
displacements as \(\epsilon d(q)\). The vibrational angular momentum
\(\pi=\zeta Qp_Q\) is invariant under this scaling. Thus

\[
 \mu(\epsilon)=\epsilon^4\left[
 I_e+\epsilon I_1(q)+\epsilon^2\{I_2(q)-C(q)C(q)^T\}\right]^{-1}.
\]

The inverse in brackets expands as
\(\mu_0+\epsilon\mu_1(q)+\epsilon^2\mu_2(q)+\cdots\), with

\[
 \mu_0=I_e^{-1},\quad
 \mu_1=-\mu_0I_1\mu_0,\quad
 \mu_2=\mu_0I_1\mu_0I_1\mu_0-\mu_0(I_2-CC^T)\mu_0.
\]

The quadratic coefficient \(I_2\) includes **one half** of the second inertia
derivative. Likewise \(\mu_2\) is the quadratic expansion coefficient, not
the second derivative itself. The implementation explicitly symmetrizes its
two normal-coordinate indices when constructing this coefficient.

The physical Hamiltonian orders are

\[
 H=\epsilon^2H_2+\epsilon^3V_3+
 \epsilon^4(V_4+R_0)+\epsilon^5R_1+\epsilon^6R_2+\cdots,
\]

where \(H_2=\sum_i\omega_i(n_i+1/2)\) and
\(R_k=\frac12(J-\pi)\mu_k(J-\pi)-\operatorname{Tr}\mu_k/8\).
This is physical mass bookkeeping. Holding the vibrational metric fixed while
arbitrarily replacing every geometric coordinate by \(\lambda Q\) would be
a different coupling family and is not used to name this model canonical VPT2.
Formal mass-scaled probes are mathematical checks; they are never reported as
observed molecular isotopologues.

## Leading rotation–vibration correction

For an isolated real harmonic occupation state \(v\), the leading rotational
energy through physical order \(\epsilon^6\), relative to its J=0 energy,
is \(\sum_{ab}\beta_{v,ab}J_aJ_b\). Its three correction terms are

\[
 \beta_v=\tfrac12\mu_0
 +\tfrac12\langle v|\mu_2|v\rangle
 +\sum_{m\ne v}\frac{\langle v|\mu_1|m\rangle
                         \langle m|V_3|v\rangle}{E_v-E_m}
 +\sum_{m\ne v}\frac{k_a(v,m)k_b(m,v)}{E_v-E_m},
 \quad k_a=-\sum_b\mu_{0,ab}\langle v|\pi_b|m\rangle.
\]

The cubic/linear-inertia response contains the two cross orders in the
Hermitian second-order effective operator; their factor two cancels the
rotational coefficient's one half. Coriolis matrix elements are imaginary in
the retained real mode basis. Their conjugate products, with the **signed**
harmonic denominator, give the real symmetric rotational response. Ground-state
denominators are negative; their Coriolis response has the corresponding sign.

Other terms at this physical order do not add an isolated real-state J²
correction: the constant \(\mu_0J^2/2\) acts as vibrational identity, so
potential-induced state normalization/energy terms cancel in rotational
differences. Hermitian J-linear expectations involving \(\pi\) vanish for a
real nondegenerate vibrational state. The remaining purely vibrational terms
are shared by J=0 and cancel in that difference. These cancellations do not
authorize treating a resonant or degenerate vibrational block as an isolated
state.

Oscillator actions retain their exact finite intermediate support: no
intermediate Fock state is deleted merely because it lies outside a final
displayed state set. Every virtual denominator and matrix numerator contributing
to the response is retained. A coupled near-zero denominator blocks that
contribution and the affected scientific product; it is never shifted by an
arbitrary epsilon.

Mode tensors are defined as
\(\alpha_i=\beta_{\mathrm{ground}}-\beta_{\mathrm{fundamental},i}\).
For this leading model,
\(\beta_0-\beta_e=-\frac12\sum_i\alpha_i\).
Named constants use the diagonal in the retained nondegenerate equilibrium
principal-axis frame, to the stated leading order. An arbitrary re-diagonalizing
or reduction-induced shift is not silently added. The full real symmetric
operator tensors, including off-diagonal observations, remain retained.

## Nonresonant semirigid vibrational VPT2

At J=0, through physical order \(\epsilon^4\), the isolated-state energy is

\[
 E_v=\sum_i\omega_i(v_i+1/2)
 +\langle V_4\rangle_v
 +\sum_{m\ne v}\frac{|\langle m|V_3|v\rangle|^2}{E_v^{(0)}-E_m^{(0)}}
 +\tfrac12\langle\pi_a\mu_{0,ab}\pi_b\rangle_v
 -\tfrac18\operatorname{Tr}\mu_0.
\]

The last two terms are the actual kinetic contributions missing from a
vibrational-only rectilinear potential calculation. Their oscillator matrix
elements are evaluated directly. The resulting ground and fundamental energies
are independently compared with the retrieved reference implementation's
\(g_0/x_{ij}\) expressions, including their Coriolis and volume terms.

`result.semirigid_vpt2` contains source-bound energies, fundamentals, and each
separate contribution only when all required isolated-state applicability
checks pass. The earlier vibrational-only result remains separately available
under its own name. Resonant GVPT2/polyad dynamics remain a required separate
implementation; no unsupported resonance correction is invented.

## Unreduced harmonic centrifugal response

Let \(\mu_{1,k}\) be the coefficient of dimensionless \(q_k\), and
\(T_k=\frac12\sum_{ab}\mu_{1,ab,k}J_aJ_b\).
Second-order virtual harmonic response gives

\[
 H_{\mathrm{dist}}=-\sum_k\frac{T_k^2}{2\omega_k}
 =\tfrac14\sum_{abcd}\tau_{abcd}J_aJ_bJ_cJ_d,\quad
 \tau_{abcd}=-\tfrac12\sum_k
 \frac{\mu_{1,ab,k}\mu_{1,cd,k}}{\omega_k}.
\]

This is a real, computed, unreduced quartic tensor. Its operator is Hermitian
and negative semidefinite as a sum of negative squares. It is the leading
harmonic response, not full anharmonic distortion. Watson A/S reduction,
representation choices and their associated effective rotational-constant
shifts remain unavailable until separately derived and validated. The code
does not manufacture D_J, D_JK or d_1 values by relabeling tensor elements.

## Provenance, gates and reference checks

Inputs must agree in geometry, ordered atom identity, isotopologue, mode basis
and phase, principal-axis frame, charge, multiplicity, electronic recipe and
harmonic source. The actual source-bound reference gradient must satisfy its
declared stationary-point tolerance. Force-field two-scale convergence is
mandatory.

Default applicability gates retain a 0.1 coupling-ratio bound, an explicit
10 cm⁻¹ denominator threshold, rotational and vibrational correction bounds,
mode/intermediate-state budgets and a bounded J scope. These are explicit
protocol checks, not independent uncertainty calibration. When a gate fails,
algebraic contributions remain diagnostic observations while corresponding
model constants, alpha or semirigid energies remain unavailable. Degenerate
rotational axes block named constants; occupied degenerate vibrational modes
require explicit vibrational blocks. Uncertainty, identification readiness and
full resonant GVPT2 are never asserted from a numerical formula comparison.

A nonzero cubic coupling with a near-zero denominator blocks the isolated
rotational state as well as its semirigid energy, even when that coupling lies
below the resonance analyzer's display floor. Observed cubic coupling ratios
also independently enforce the declared bound. A small coupling does not
justify dividing by a vanishing denominator. Readback cross-checks each
virtual-state denominator against the retained harmonic energies and each
response against its individual recorded contributions. The semirigid child
must match its parent's context, force field, precursors, states, energy terms
and availability; deliberately recomputing a content seal cannot authorize
cross-source substitution or bypass an actual applicability bound.

An independent open implementation was actually retrieved from
[NITROGEN](https://github.com/bchangala/nitrogen/blob/85ab1d0c965b2f613fc810a394de0c2006bd6f6e/nitrogen/vpt/__init__.py),
commit `85ab1d0c965b2f613fc810a394de0c2006bd6f6e`, source SHA-256
`0a29234b59bc92dcd60c1c8bd16ee4ecff44ad4d9e1e8522bac8f198e7208d30`.
The retained source includes `calcAlpha_harm`, `calcAlpha_cor`,
`calcAlpha_anharm`, `calctau`, `calc_g0` and `calc_xij`. Its MIT license identifies
Copyright (c) 2020 bchangala. The production implementation here uses the
derived oscillator sums; tests independently transcribe the retrieved closed
expressions and compare them with those sums.

That source cites J. K. G. Watson, “Simplification of the molecular
vibration-rotation Hamiltonian,” *Molecular Physics* **15**, 479 (1968),
DOI `10.1080/00268976800101381`. The citation is present in the actually
retrieved reference code. The primary article's full text remains unverified
in this environment, so independent code/formula agreement is not presented
as primary-literature or molecular-accuracy qualification.

The [tests](../../tests/test_rovibrational_perturbation.py) include independent
closed-form alpha and g0/xij comparisons, the full volume identity at displaced
coordinates, tensor/operator factor and sign checks, exact intermediate paths,
physical mass-scaling asymptotic diagonalizations, genuine native force-field
integration and failed-gate retention. The mass-scaling probe diagonalizes a
finite Galerkin polynomial H2–H6 with all intermediate operator paths retained
before final projection; omitted higher mass orders remain explicit. It is a
mathematical asymptotic check, not an isotope measurement or a full global
nuclear Hamiltonian.

## Measured verification and retained evidence

The focused suite passed **42 tests, zero skips** in this environment, including
two genuine native-engine tests. The native campaign retained the optimized
HF/STO-3G water reference and **107 actual displaced electronic calculations**.
Its maximum reference-gradient component was
`3.874255405378335e-08 hartree/bohr`.

For that actual force field, the maximum absolute alpha-diagonal discrepancy
from the independently transcribed closed expressions was
`1.4399560103323106e-20 hartree`; the maximum absolute semirigid energy
discrepancy from the independent g0/xij expressions was
`6.938893903907228e-18 hartree`. The independent Watson volume-identity
residuals at the three tested mass-weighted coordinates were
`4.0657581468206416e-20`, `1.3552527156068805e-20`, and
`2.0328790734103208e-20 hartree`.

Those are numerical implementation comparisons. They do not measure
electronic-method accuracy or astronomical/laboratory identification accuracy.
The genuine water ground and three fundamental states had maximum observed
cubic coupling ratios `0.1226343832116868`, `0.10777924407041098`,
`0.26453625919559304`, and `0.25391628970585767`. They therefore failed the
unchanged default **0.1** applicability bound. Ground model constants, model
alpha and semirigid VPT2 products remained unavailable. Actual algebraic
contributions and the unreduced harmonic-distortion model remained retained;
the test did not relax the protocol to obtain an available result.

Mathematical checks also covered physically equivalent normal-mode phase
changes, exact Fermi resonance, degenerate harmonic-mode Coriolis denominators,
strong cubic coupling, source/geometry/evidence binding, result readback
integrity even after deliberate resealing, complete oscillator paths and J=0–5
distortion factors/signs. The
physical mass-scaling comparison retains all six J=0/J=1 Hamiltonians,
eigenvalues, eigenvectors, its declared finite basis, source-bound mathematical
inputs and observed response errors. Those records permit reconstruction of
the energy differences, while explicitly retaining the finite-Galerkin and
H2–H6 truncations.

Run the focused suite with real dependencies and an evidence directory:

```bash
COCHEM_DISABLE_SANDBOX_CHECK=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_rovibrational_perturbation.py \
  --basetemp=/tmp/torq-watson-verification --junitxml=/tmp/torq-watson-junit.xml
```

`COCHEM_DISABLE_SANDBOX_CHECK` disables the legacy launcher sandbox probe for
this mathematical/native test process; it does not disable scientific gates.
The native fixture writes `canonical-watson-leading.json`, actual force field,
geometric precursors and stationary-reference records beside the individual
native manifests. A clean run evaluates real engines; unsupported environments
cannot qualify this scientific method by running only mathematical checks.

Scoped Ruff, strict mypy with imported-module diagnostics suppressed via
`--follow-imports=silent`, and Python 3.10 syntax checks also passed. These
checks apply to this module, rather than asserting repository-wide type-check
success.

## Reference expression attribution

The independent closed reference expressions in the tests were adapted from
the actually retrieved NITROGEN source named above. The corresponding notice
is retained here:

```text
MIT License

Copyright (c) 2020 bchangala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
