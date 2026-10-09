# Unfinished custom GKS first-derivative research milestone

This is an implementation handoff, not an executable or scientifically validated
module. The accompanying patch is incomplete: it references a module that has
not been written. Do not apply it to a release without completing the work below.
No source, finite-difference, native-engine, type, formatting or regression check
was run for this draft. Student deployment remains on hold. No named revDSD,
revDOD, production, spectroscopy or full-SRS qualification is established.

## Baseline and preserved work

The existing `ExperimentalDoubleHybrid` evaluates explicitly named custom
restricted HF/LDA/GGA double hybrids on the specified orbital-generating
functional, with separate energy functional and OS/SS PT2 coefficients.
Its existing HF-only analytic-gradient path has genuine preceding validation.
That validation does not validate any of this proposed GKS work.

`revdsd-unfinished-integration.patch` preserves the only edits attempted here:

1. Rename the private existing `_hf_pt2_analytic_gradient` helper to
   `_canonical_pt2_analytic_gradient`, updating its existing HF call site.
2. Permit an explicitly supplied nuclear derivative of the orbital-generating
   Fock matrix, and an energy-gradient callback receiving the actual occupied
   density response. Preserve the existing HF behavior when neither is supplied.
3. Draft a separate `ExperimentalDoubleHybrid.gks_analytic_gradient` wrapper
   that would import a new `engines.gks_response` module. That module does not
   exist, so this wrapper must not be activated as supplied.

The patch also changes the class description. Those descriptions are statements
of the intended completed implementation, not current availability. The baseline
source is preserved separately, and the checkout is restored to that baseline.

## Source inspection that actually occurred

Read-only inspection used installed PySCF 2.14.0 and the previously preserved
official PySCF-forge source from commit
`d2fcf42f957f9b5f4b28d324670db04f9a03a03e` under
`/workspace/torq-review/primary-method-followup-20261008`.
This inspection did not import or execute the forge implementation.
No package, dependency, runtime or global environment was changed.

Relevant inspected source:

- PySCF `hessian/rks.py`: `make_h1`, `_get_vxc_deriv1`.
- PySCF `grad/rks.py`: `get_vxc_full_response`, `grids_response_cc` and its
  Becke-grid coordinate and partition-weight derivatives.
- PySCF-forge preserved `forge-resp.py` and `forge-grad-dfdh.py`.
- TORQ's existing conventional canonical HF/OS/SS orbital-response implementation.

The installed semilocal RKS Hessian `make_h1` uses `_get_vxc_deriv1`, whose
inspected LDA/GGA branches contain fixed-grid AO/density response, without
atom-attached quadrature coordinate or partition-weight response. Do not simply
substitute this native `make_h1` into the HF helper and call the result a complete
moving-grid GKS derivative. Native RKS energy gradients can request full grid
response, but that alone does not supply the orbital-generating Fock derivative
needed by the nonvariational PT2 energy.

## Proposed algorithm: direct semilocal moving-grid differentiation

Use a new separate module, for example `src/cochem_torq/engines/gks_response.py`.
Initially support real, nondegenerate, all-electron, restricted closed-shell
custom LDA/GGA hybrid recipes, conventional four-center PT2, H through Ne,
one thread, explicit AO tensor and grid-block bounds, and no D4 term.
Reject frozen core, open shell, complex data, MGGA, range-separated or nonlocal
functionals, degeneracy and unsupported D4 definitions before a calculation.
Keep the existing `analytic_gradient` method HF-only. Add the separate GKS
entry point only after the new implementation and its genuine checks pass.

For a grid attached to atom B and nuclear displacement of atom A in Cartesian
direction q, differentiate each AO and its spatial derivatives by

    d_Aq phi_mu(r_g) = (delta_AB - delta_{mu on A}) d_q phi_mu(r_g).

The first term is motion of the quadrature coordinate with its owner atom;
the second is motion of the basis center. GGA requires AO second spatial
derivatives to differentiate the density gradient. Obtain quadrature weight
derivatives from native `pyscf.grad.rks.grids_response_cc`. Process each owner's
coordinates, weights and weight derivatives in bounded blocks, retaining their
owner identity. Verify that these coordinates and weights reproduce the actual
SCF/energy quadrature, including angular pruning, radii adjustment and partition
definition. Reject density-adaptive grid deletion if no complete consistent
derivative of that operation is implemented. The inspected PySCF default
`small_rho_cutoff` is zero, but check its actual value rather than assuming it.

At fixed AO density matrix P, form the LDA/GGA density components

    rho_0 = phi^T P phi
    rho_k = (d_k phi)^T P phi + phi^T P (d_k phi), k=x,y,z.

Differentiate these expressions using the nuclear AO derivatives above.
Use `NumInt.eval_xc_eff` for first and second XC derivatives with respect to
these density components. Independently verify the returned LDA/GGA dimensions
and GGA component convention before implementation; do not substitute raw
LibXC derivatives with respect to sigma for derivatives with respect to the
Cartesian density gradient.

Define the density-component AO bilinears

    T_0(mu,nu) = phi_mu phi_nu
    T_k(mu,nu) = (d_k phi_mu) phi_nu + phi_mu (d_k phi_nu).

Then the semilocal potential and its explicit nuclear derivative at fixed P are

    Vxc(mu,nu) = sum_g w_g sum_t v_t(g) T_t(mu,nu;g)
    d_Aq Vxc = sum_g [(d_Aq w) sum_t v_t T_t
                  + w sum_{t,u} f_tu (d_Aq rho_u) T_t
                  + w sum_t v_t (d_Aq T_t)].

The explicit semilocal energy derivative is

    d_Aq Exc |_P = sum_g [(d_Aq w) rho_0 exc
                         + w sum_t v_t (d_Aq rho_t)].

Build analytic hcore and conventional Coulomb/exchange integral derivatives,
including all four AO-center derivative placements and their signs. Combine
them with the complete semilocal potential derivative to obtain the explicit
orbital-generating Fock derivative at fixed P. A finite difference of the full
SCF energy is an independent validation observation, not an analytic substitute.

Use native restricted CPKS/CPHF with the orbital-generating functional's actual
HF-exchange and XC kernel. The induced-density potential is generated on the
undisplaced reference quadrature; all explicit quadrature motion is already in
the supplied Fock derivative. Include overlap/Pulay derivatives, then reconstruct
all canonical occupied-occupied and virtual-virtual rotations as well as
occupied-virtual response and orbital-energy derivatives. Preserve reference
orbitals and checkpoints byte-for-byte. Fail on response-equation,
orthonormality or native-occupied-response residuals, rather than shifting or
regularizing unqualified denominators.

Differentiate OS and SS conventional PT2 integrals and denominators using the
existing complete canonical formula. No frozen occupied subset is accepted in
this initial GKS milestone. A different energy functional must include its
nonvariational density response; treating it as a stationary RKS gradient is
incorrect. For energy-functional Fock matrix F_E, the reference-energy derivative
can be evaluated directly as

    d E_reference = Tr(H' P) + 1/2 Tr(J' P)
                    - c_HF/4 Tr(K' P) + Exc'|_P + E_nuc'
                    + Tr(F_E P').

Here P' is the actual occupied density response including coefficient/Pulay
normalization, J' and K' are the fixed-P geometric derivatives of conventional
two-electron integrals, and c_HF is the energy functional's exchange coefficient.
This supports different orbital and energy expressions without silently
assuming their equality. The complete custom gradient adds the declared scaled
OS/SS derivatives. No absent component may be filled with invented values.

## Required genuine checks and evidence

1. Pure input checks must reject unsupported frozen core, multiplicity, complex
   coordinates, invalid response/grid/tensor budgets, unsupported functionals and
   D4 before creating an energy calculation. Retain the existing exact-revDSD
   fail-closed gate and the HF behavior.
2. For water in STO-3G, run actual PySCF custom GKS energy/gradient calculations
   at fixed explicit grid controls and tight SCF/CPKS tolerances. At least one
   recipe must have different orbital and energy XC expressions and nonzero OS
   and SS scales, exercising the nonvariational reference-energy response.
3. Compare reference, unscaled OS, unscaled SS and total analytic derivatives
   to genuine complete displaced-energy differences at two decreasing steps.
   Recompute SCF orbitals at every displaced energy; do not reuse frozen density
   or an orbital potential as a purported derivative.
4. Compare the zero-PT2, equal-functional reference gradient to native RKS
   gradient with `grid_response=True`; this is an algorithm consistency check
   within PySCF, not an independent-engine method qualification.
5. Compare explicit Fock derivatives against genuine fixed-P displaced geometry
   Fock evaluations and canonical orbital-energy/density response against actual
   displaced SCF states with explicitly tracked orbital phases/state continuity.
6. Check translation and rotation covariance and demonstrate that moving-grid
   terms are actually present at the chosen quadrature; a fine grid can hide
   their omission numerically. Check all declared response residuals and
   finite values, exact parent/checkpoint hashes, native logs and attempted versus
   complete calculation counts. Preserve failed attempts and their actual exits.
7. Run dedicated formatting/type/software checks and existing HF milestone
   regressions. No full suite or student pilot was requested for this bounded work.
8. Write a separate `docs/development/GKS*` milestone note with exact controls,
   hashes, observed errors, units, scope and unresolved scientific limitations.
   Update misleading HF-specific diagnostic keys and source hashes when the
   shared response helper is generalized. Do not label source review as execution.

## Resource and release boundaries

Use the existing `/workspace/.venvs/cochem-torq` (PySCF 2.14.0/SciPy 1.18.1),
one CPU/thread, no installs and no global dependency mutation. Notify the parent
before costly native work. Keep real artifacts outside the repository and under
20 MiB, preserving at least 256 MiB free on `/tmp`. No native work was started.
Do not commit arbitrary output paths, process identities, signed URLs or tokens.

This is an agent-owned unfinished software/scientific task. Completing this
milestone would still not reconcile the original published revDSD recipe, validate
frozen-core derivatives, establish higher derivatives, independently qualify a
named method, calibrate accuracy or meet all 73 SRS requirements and 41 full-scope
acceptance gates.
