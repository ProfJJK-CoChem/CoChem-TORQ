# Exact revDSD development and executable research machinery

The selected D01 route remains development of exact open-source revDSD-PBEP86-D4. This implementation supplies real restricted double-hybrid research calculations, but **does not activate or certify that named recipe**. `exact_revdsd_pbep86_d4_recipe()` raises `RecipeNotQualifiedError` with the unresolved evidence requirements. The [scientific protocol](../wiki/RevDSD_Spectroscopy_Protocol.md) defines independent DH0–DH6 acceptance milestones.

## Implemented calculations

`cochem_torq.engines.revdsd.ExperimentalDoubleHybrid` performs these operations:

1. Converges the explicitly supplied PySCF restricted orbital-generating HF or generalized Kohn–Sham functional.
2. Evaluates the explicitly supplied energy functional on those same orbitals, separating nuclear repulsion, one-electron, Coulomb, exact exchange and semilocal XC contributions.
3. Computes conventional real-orbital OS and SS second-order correlation from transformed `(ia|jb)` integrals and the original orbital energies. No conversion to HF, density fitting, shifted denominator or altered occupations is performed. Explicit frozen occupied indices are retained in the recipe.
4. When specifically requested, resolves the named D4 BJ-EEQ-ATM entry from the installed DFTD4 parameter table, retains all six effective damping parameters and the table checksum, and computes its actual molecular energy once with the stated charge.
5. Optionally evaluates full-energy centered numerical gradients at two or more user-specified displacement sizes. Every displaced energy reruns the complete orbital, PT2 and dispersion calculation, thereby including their response numerically. The returned step differences are evidence to assess, not an automatic convergence qualification.

For restricted real canonical orbitals,

\[
E_{OS}=\sum_{ijab}\frac{(ia|jb)^2}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b},\qquad
E_{SS}=\sum_{ijab}\frac{(ia|jb)[(ia|jb)-(ib|ja)]}{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.
\]

The initial domain is closed-shell, integer occupations, first-row elements H through Ne, a nonrelativistic Hamiltonian, real orbitals, conventional PT2, and HF/LDA/GGA hybrid expressions without range separation. The full transformed PT2 tensor has an explicit memory limit. SCF failures, invalid references and resource overruns fail with retained artifacts. The implementation does not provide analytic orbital-response gradients, Hessians, anharmonic derivatives or a revDSD spectroscopy product.

Results state `experimental_unqualified`, `reference_stability: not_evaluated`, and `independent_method_validation: unavailable`. These statuses must survive downstream handoffs. This research evaluator must not be inserted into a student release profile as a qualified named revDSD backend.

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

## Running real research checks

The research dependencies are PySCF, DFTD4 3.7.0 and a TOML parser (`tomli`). Run the tests from an actual CoChem calculation sandbox:

```bash
mkdir -p /tmp/cochem_exec_revdsd_integration
cd /tmp/cochem_exec_revdsd_integration
python -m pytest /path/to/CoChem-TORQ/tests/test_revdsd_integration.py -q
```

The ten tests use genuine CPU engine calculations. They compare both OS/SS components and total canonical restricted MP2 energies against PySCF's MP2 implementation, verify explicit frozen indices, compare PT2 on actual PBE0 orbitals against the direct MP2 class that preserves those orbitals, compare the resolved D4 correction with the native DFTD4 method-name API, and check complete numerical HF-MP2 gradients against actual analytic MP2 gradients. A second derivative check includes an explicitly requested D4 contribution and compares the complete numerical derivative with actual MP2 and native D4 derivatives. A deliberately stringent SCF iteration limit confirms genuine nonconvergence retains the raw failure evidence. No engine outputs are mocked.

These checks validate reusable software machinery and numerical differentiation on the tested cases. They do **not** constitute independent validation of exact revDSD, validate a double-hybrid analytic gradient, or establish rotational-spectroscopy accuracy.

Every energy execution saves `input.json`, the actual PySCF log, an orbital checkpoint, resolved basis contents and their checksum, a component/provenance result, and artifact hashes. Failed runs retain a failure record. Numerical gradients retain all displacement identities and parent-result checksums. No missing physical quantity is inserted to complete a later spectroscopy stage.
