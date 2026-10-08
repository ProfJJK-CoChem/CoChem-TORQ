# Exact revDSD development and executable research machinery

The selected D01 route remains development of exact open-source revDSD-PBEP86-D4. This implementation supplies real restricted double-hybrid research calculations, but **does not activate or certify that named recipe**. `exact_revdsd_pbep86_d4_recipe()` raises `RecipeNotQualifiedError` with the unresolved evidence requirements. The [scientific protocol](../wiki/RevDSD_Spectroscopy_Protocol.md) defines independent DH0–DH6 acceptance milestones.

## Implemented calculations

`cochem_torq.engines.revdsd.ExperimentalDoubleHybrid` performs these operations:

1. Converges the explicitly supplied PySCF restricted orbital-generating HF or generalized Kohn–Sham functional.
2. Evaluates the explicitly supplied energy functional on those same orbitals, separating nuclear repulsion, one-electron, Coulomb, exact exchange and semilocal XC contributions.
3. Computes conventional real-orbital OS and SS second-order correlation from transformed `(ia|jb)` integrals and the original orbital energies. No conversion to HF, density fitting, shifted denominator or altered occupations is performed. Explicit frozen occupied indices are retained in the recipe.
4. When specifically requested, resolves the named D4 BJ-EEQ-ATM entry from the installed DFTD4 parameter table, retains all six effective damping parameters and the table checksum, and computes its actual molecular energy once with the stated charge.
5. Optionally evaluates full-energy centered numerical gradients at two or more user-specified displacement sizes. Every displaced energy reruns the complete orbital, PT2 and dispersion calculation, thereby including their response numerically. The returned step differences are evidence to assess, not an automatic convergence qualification.
6. Optionally evaluates complete-energy Cartesian numerical Hessians at two or more displacement sizes. Diagonal and mixed stencils rerun every displaced energy, retaining all calculation-result hashes. The explicit cost is `1 + 2*(3*N)^2*number_of_scales`; a budget rejects oversized work before execution. Gradient and Hessian invariance diagnostics, orbital-gap ranges and scale differences are retained without automatic scientific qualification. Matrix symmetry follows from the stencil and is explicitly labeled `symmetry_by_construction`.

For restricted real canonical orbitals,

\[
E_{OS}=\sum_{ijab}\frac{(ia|jb)^2}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b},\qquad
E_{SS}=\sum_{ijab}\frac{(ia|jb)[(ia|jb)-(ib|ja)]}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.
\]

The initial domain is closed-shell, integer occupations, first-row elements H through Ne, a nonrelativistic Hamiltonian, real orbitals, conventional PT2, and HF/LDA/GGA hybrid expressions without range separation. The full transformed PT2 tensor has an explicit memory limit. SCF failures, invalid references and resource overruns fail with retained artifacts. The implementation does not provide analytic orbital-response gradients, analytic Hessians, anharmonic revDSD derivatives or a revDSD spectroscopy product. A numerical Hessian does not establish stationarity, a correct electronic state, a qualified vibrational analysis or spectroscopic accuracy.

Results retain `experimental_unqualified` and `independent_method_validation: unavailable`. The research evaluator now runs genuine internal HF/GKS orbital-reference stability and restricted-to-unrestricted diagnostics without applying returned alternative orbitals or changing the physical model. Each diagnostic preserves actual solver controls, implementation hashes, native logs and before/after orbital/checkpoint identities. PySCF's returned external flag covers RHF/RKS to UHF/UKS; real-to-complex checks are logged but are not separately extracted. These diagnostics establish neither stability of the full correlated energy nor continuity between electronic branches. An unstable orbital reference retains its actual constrained-reference energy and an explicit review flag. Numerical-derivative displacement records retain every point's actual diagnostic and quality flags. These narrowed statuses survive downstream handoffs; the evaluator cannot be inserted into a student profile as a qualified named revDSD backend.

## Remaining exact-method evidence

These requirements remain blockers for activating a named revDSD-PBEP86-D4 recipe:

- Reconcile the complete electronic coefficients and orbital-generating definition with the original article and supporting information.
- Resolve the precise P86 correlation implementation, including possible P86/FT/VWN convention differences. LibXC exposes distinct implementations; their coexistence does not establish which defines the target recipe.
- Define the published frozen-core convention and any difference between the original 2019 recipe and implementations named `/2021`.
- Obtain a native independently implemented reference bundle with matched geometries, basis, grid, electronic state, frozen core and decomposed energies. Algebra agreement within PySCF is useful but not independent-engine method validation.

The original citation is Santra, Sylvetsky and Martin, *J. Phys. Chem. A* **123** (2019), 5129–5143, DOI `10.1021/acs.jpca.9b03157`. ACS/Figshare access was denied by the normal environment proxy. No coefficients have been guessed or transplanted from a different dispersion recipe.

## Source observations retained on 2026-10-07

| Immutable source | SHA-256 | What it establishes |
|---|---|---|
| [`ajz34/dh`, `dhutil.py`, commit `7e2d5172e0af1cd835af11a41e3a2b903d2ca487`](https://github.com/ajz34/dh/blob/7e2d5172e0af1cd835af11a41e3a2b903d2ca487/pyscf/dh/dhutil.py) | `c0f655c2b2bd8e8cc4fde19ce76cda5329cb4b9daad0a0923411ed78b571c451` | An upstream **revDSD-PBEP86-D3** entry uses 0.69 HF exchange, 0.4296 P86 correlation, 0.5785 OS and 0.0799 SS. It does not establish the D4 recipe. |
| [`ajz34/pyscf-forge`, `family_martin.json`, commit `4450c1046f9794ca5488a65662f5d21a950f8f8b`](https://github.com/ajz34/pyscf-forge/blob/4450c1046f9794ca5488a65662f5d21a950f8f8b/pyscf/dh/util/xccode/functionals/family_martin.json) | `1d3240e46bda081a599b16034f7f5e2d2d75b250b04cd591e85a54aad1cd5c8f` | Its DSD-PBEP86-D3BJ entry explicitly uses P86_FT and warns about correlation-definition differences. This is not independent proof of the target revDSD-D4 definition. |
| [DFTD4 3.7.0 upstream parameters, commit `7b2ff85a71a3630808fd8a2d972933f75b743f3c`](https://github.com/dftd4/dftd4/blob/7b2ff85a71a3630808fd8a2d972933f75b743f3c/assets/parameters.toml) | `8e44a7fed51e511c9e2ab0938a94f0a21bacc4c70f470a915224239937c1d61f` | The revDSD-PBEP86 D4 dispersion entry resolves to `s6=0.5132, s8=0, s9=1, a1=0.44, a2=3.60, alp=16`. Dispersion alone does not establish electronic coefficients or orbitals. |

The installed DFTD4 3.7.0 wheel parameter file in the current environment has SHA-256 `8254bfc673e763f7589b9be20506438b202a59ce795dc175d6ea9ba623e23e1f`; packaging differs from the upstream asset, and each executed result retains its actual table checksum. The resolved target dispersion values agree with those above. Runtime software is currently PySCF **2.14.0**, LibXC **7.0.0**, DFTD4 **3.7.0**; these are the actual executed versions, rather than a claim to have tested the protocol's proposed PySCF 2.8.0 pin.

## Additional source reconciliation on 2026-10-08

Two immutable secondary sources corroborate a **candidate** electronic coefficient table: HF exchange `0.69`, PBE exchange `0.31`, P86 correlation `0.4210`, OS PT2 `0.5922`, SS PT2 `0.0636`. The Gellrich group's [ORCA input template](https://github.com/baaam24/gellrich_scripts/blob/110278967f64ad6b19be469a9e60804726d35308/custom_2input/rev-DSD-PBEP86_D4_sp.sh) contains those settings, an explicit `FC_ELECTRONS` choice and RI directives. The independent [xcx semilocal code](https://github.com/nmrtist/xcx/blob/db458902604eebd9c75426de8782e3cc01ecadca/crates/xcx/src/functionals/hybrids.rs) cites Table 4 of the 2019 article and agrees on the table; its PT2 weights are metadata for a host calculation.

These observations improve the evidence ledger but **do not open DH0/DH1**. Neither source supplies the original supporting information or matched native molecular component outputs. A template's `C_P86` label is insufficient to establish the LibXC P86/FT/VWN mapping; its explicit approximations also differ from TORQ's conventional CPU research domain. The xcx golden generator compares LibXC semilocal mixtures, rather than independent complete molecular revDSD energies. The original orbital/core definitions and 2019 versus `/2021` distinctions remain unresolved. No candidate recipe is automatically instantiated or silently substituted.

The [machine-readable source ledger](revdsd_source_evidence.json) retains immutable URLs, source hashes, candidate values, source scope and the precise remaining gates. Retrieved source bytes are archived in the review artifacts. No independent native revDSD reference bundle was found in the inspected sources; absence in this search is not proof that such data do not exist elsewhere.

## Running real research checks

The research dependencies are PySCF, DFTD4 3.7.0 and a TOML parser (`tomli`). Run the tests from an actual CoChem calculation sandbox:

```bash
mkdir -p /tmp/cochem_exec_revdsd_integration
cd /tmp/cochem_exec_revdsd_integration
python -m pytest /path/to/CoChem-TORQ/tests/test_revdsd_integration.py -q
```

Eighteen research tests exercise this module. Actual CPU engine checks compare both OS/SS components and total canonical restricted MP2 energies against PySCF's MP2 implementation, verify explicit frozen indices, compare PT2 on actual PBE0 orbitals against the direct MP2 class that preserves those orbitals, compare resolved dispersion against native DFTD4, and check complete numerical MP2 gradients against actual analytic MP2 gradients. The new MP2+D4 Hessian check performs 145 genuine full-energy calculations and compares a different stencil built from native MP2 and D4 gradients. Rigid translation/rotation and identical-atom permutation are checked on real molecular energies/components. Strict SCF limits verify genuine nonconvergence retains raw and derivative failure records; budget rejection performs no engine calculation. Complex coordinates and orbital energies are rejected before casting rather than discarding imaginary parts. No engine outputs are mocked.

These checks validate reusable software machinery and numerical differentiation on the tested cases. They do **not** constitute independent validation of exact revDSD, validate a double-hybrid analytic gradient, or establish rotational-spectroscopy accuracy.

Every energy execution saves `input.json`, the actual PySCF log, an orbital checkpoint, resolved basis contents and their checksum, a component/provenance result, and artifact hashes. Failed runs retain a failure record. Numerical gradients retain all displacement identities and parent-result checksums. No missing physical quantity is inserted to complete a later spectroscopy stage.
