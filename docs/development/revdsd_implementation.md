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

The initial energy domain is closed-shell, integer occupations, first-row elements H through Ne, a nonrelativistic Hamiltonian, real orbitals, conventional PT2, and HF/LDA/GGA hybrid expressions without range separation. The full transformed PT2 tensor has an explicit memory limit. SCF failures, invalid references and resource overruns fail with retained artifacts. The implementation does not provide analytic **GKS double-hybrid** orbital-response gradients, analytic Hessians, anharmonic revDSD derivatives or a revDSD spectroscopy product. The narrower analytic HF/PT2 milestone below is implemented. A numerical Hessian does not establish stationarity, a correct electronic state, a qualified vibrational analysis or spectroscopic accuracy.

Results retain `experimental_unqualified` and `independent_method_validation: unavailable`. The research evaluator now runs genuine internal HF/GKS orbital-reference stability and restricted-to-unrestricted diagnostics without applying returned alternative orbitals or changing the physical model. Each diagnostic preserves actual solver controls, implementation hashes, native logs and before/after orbital/checkpoint identities. PySCF's returned external flag covers RHF/RKS to UHF/UKS; real-to-complex checks are logged but are not separately extracted. These diagnostics establish neither stability of the full correlated energy nor continuity between electronic branches. An unstable orbital reference retains its actual constrained-reference energy and an explicit review flag. Numerical-derivative displacement records retain every point's actual diagnostic and quality flags. These narrowed statuses survive downstream handoffs; the evaluator cannot be inserted into a student profile as a qualified named revDSD backend.

## Remaining exact-method evidence

These requirements remain blockers for activating a named revDSD-PBEP86-D4 recipe:

- Reconcile the complete electronic coefficients and orbital-generating definition with the original article and supporting information.
- Resolve the precise P86 correlation implementation, including possible P86/FT/VWN convention differences. LibXC exposes distinct implementations; their coexistence does not establish which defines the target recipe.
- Define the published frozen-core convention and any difference between the original 2019 recipe and implementations named `/2021`.
- Obtain a native independently implemented reference bundle with matched geometries, basis, grid, electronic state, frozen core and decomposed energies. Algebra agreement within PySCF is useful but not independent-engine method validation.

The original citation is Santra, Sylvetsky and Martin, “Minimally Empirical Double-Hybrid Functionals Trained against the GMTKN55 Database: revDSD-PBEP86-D4, revDOD-PBE-D4, and DOD-SCAN-D4,” *J. Phys. Chem. A* **123** (2019), 5129–5143, DOI `10.1021/acs.jpca.9b03157`. Publisher HTML and PDF access were denied by the normal environment proxy; the additional metadata retrieval below succeeded. No coefficients have been guessed or transplanted from a different dispersion recipe.

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

## Authentic primary metadata and P86 reconciliation

The official [Figshare API DOI lookup](https://api.figshare.com/v2/articles?resource_doi=10.1021/acs.jpca.9b03157), [article record](https://api.figshare.com/v2/articles/8258924) and [original version record](https://api.figshare.com/v2/articles/8258924/versions/1) were retrieved through the normal proxy on 2026-10-08. These are primary publisher metadata, with the correct publication title, authors, DOI, original 2019 timeline and SI DOI `10.1021/acs.jpca.9b03157.s001`. They identify the authentic SI file as `jp9b03157_si_001.pdf`, file ID `15437975`, 787,293 bytes and publisher-supplied/computed MD5 `efe4fd1654c9eab5a5f4e9536d433c7f`. The official chain is API → `ndownloader.figshare.com` → `s3-eu-west-1.amazonaws.com`; only redirect hosts were retained, with signed queries and destination paths discarded. The actual PDF download redirect remained blocked with HTTP 403 at the final host. The stated size and MD5 are **metadata observations**, not a claim that those PDF bytes were downloaded or checksum-verified locally.

The primary abstract explicitly distinguishes **xrevDSD**, whose initial orbital evaluation uses full DFT correlation, from the target **revDSD**. That distinction is now retained as an authentic source observation. The abstract does not supply the target's complete orbital-generating equation, technical coefficient table, frozen-core convention or P86 implementation. It cannot complete DH0.

A further bounded read-only search found the matching author/title/publication-DOI
entry in an immutable [public ChemRxiv metadata archive](https://github.com/MatthiasGolomb/SciPubCrawl/blob/5a00bb4de7a74b76710ff795b9bb00b655b7be89/examples/lithium_metal_anode/search/chemrxiv_dumps/chemrxiv_2019-05-01_2019-06-01.jsonl),
with preprint DOI `10.26434/chemrxiv.7903388.v4`, dated 22 May 2019. This is a
**secondary metadata locator**, not the original technical manuscript or SI.
Historical preprint Figshare endpoints returned HTTP 404; DOI/Crossref access
returned HTTP 403 through the normal proxy. No authentic author manuscript or
publisher-MD5-matching SI mirror was retrieved in the bounded GitHub search.
The ledger retains the actual metadata and retrieval-observation hashes; the
original technical recipe and derivative/independent-engine gates remain closed.

The executed PySCF 2.14.0 / LibXC 7.0.0 installation separately exposes `GGA_C_P86` (132), `GGA_C_P86_FT` (217), `GGA_C_P86VWN` (252) and `GGA_C_P86VWN_FT` (253). Actual evaluations at explicitly declared mathematical density points yield different correlation values for all four variants. Consequently these labels cannot be treated as interchangeable. This narrows the ambiguity to an explicit implementation mapping; it does not establish which variant the original method prescribes. The machine-readable ledger retains the actual API response hashes, installed identities, numerical observation and blocking status.

## Analytic orbital-response derivative milestone

`ExperimentalDoubleHybrid.analytic_gradient()` now computes genuine analytic first derivatives for explicitly specified **HF-generated spin-component-scaled PT2 models**, optionally with native DFTD4 dispersion. Both `orbital_xc` and `energy_xc` must be the explicit expression `HF`. This narrow domain has no numerical quadrature grid. A GKS or semilocal expression fails before any energy calculation; no HF approximation is substituted for it.

The calculation performs one full genuine reference-energy evaluation, then consumes its saved orbital checkpoint without changing any original input, basis, log, stability record, energy result or checkpoint. It solves native restricted CPHF to obtain the induced density, constructs the complete nondegenerate canonical orbital response—including occupied/occupied and virtual/virtual rotations—and differentiates orbital energies, all four AO two-electron integral legs, OS/SS numerators and PT2 denominators. Frozen occupied orbitals remain excluded from correlation while their canonical orbital response is included. A declared D4 term contributes the actually evaluated native DFTD4 gradient. No displaced-energy calculation or fitted derivative is used by this analytic method.

For canonical orbital coefficients $C$ and $C'=CU$, the response obeys

\[
U_{pq}=\frac{F'_{pq}-\epsilon_q S'_{pq}}{\epsilon_q-\epsilon_p}\quad(p\ne q),\qquad
U_{pp}=-\tfrac12 S'_{pp},\qquad
\epsilon'_p=F'_{pp}-\epsilon_p S'_{pp}.
\]

Here $F'$ includes the self-consistent induced-density potential from CPHF, and all quantities in the equation are transformed to the original MO basis. The derivative includes $C'$ and the explicit AO-integral derivatives. Actual occupied/virtual CPHF equation residuals, agreement with the native occupied response, and overlap-derivative orthogonality are checked against retained controls. The canonical equation residual is also recorded, but its algebraic construction is not independent validation. The workspace estimate is checked before allocating full AO derivative tensors. Degenerate canonical subspaces are rejected rather than shifted, regularized or assigned arbitrary rotations. Failures preserve the valid original energy and a separate derivative-failure record without a gradient.

The new real-engine tests compare canonical all-electron and frozen-core MP2 gradients against PySCF's separate native MP2-gradient implementation; independently differentiate actual OS, SS and complete energy components at two displacement sizes for explicitly spin-scaled models; compare D4 gradients against native DFTD4; and test Cartesian translation/rotation covariance. These are software and derivative-milestone checks within the tested HF domain. Both analytical routes still use PySCF integral/reference machinery: **they are not independent-engine revDSD qualification**. All returned derivatives retain `experimental_unqualified`, `exact_revdsd_qualification: false`, `GKS_analytic_response_available: false` and `independent_method_validation: unavailable`.

Completing the revDSD analytic-gradient milestone still requires authentic recipe reconciliation, complete moving-grid GKS response (including the distinction between orbital and final-energy functionals), independently matched native energy/gradient components, and validation on the accepted chemistry domain. Analytic Hessians and third/fourth revDSD derivatives remain separate unimplemented milestones.

`hessian_from_analytic_gradients()` now supplies a separate numerical second-derivative milestone for the supported HF/PT2/D4 domain. It center-differences actual analytic gradients at two or more explicit displacement sizes, retains raw **unsymmetrized** total and component matrices, and selects the smallest requested step without extrapolating or constructing missing values. One reference plus two gradients per Cartesian coordinate per step gives the exact planned cost `1 + 2*(3*N)*number_of_steps`: 25 genuine reference/gradient calculations for H₂ at two steps, compared with 145 full-energy calculations in the existing energy-stencil route.

The caller must declare symmetry, cross-step agreement, translation and rotation-covariance thresholds. Translation residuals have units hartree/bohr²; nonstationary rotational covariance compares $H(\boldsymbol{\omega}\times\mathbf r)$ with $\boldsymbol{\omega}\times\mathbf g$ and has units hartree/bohr. These are different quantities and use separate thresholds. All scales must pass these numerical consistency gates. A failed gate retains the actually calculated raw matrices and all genuine parents in a failure artifact, and raises rather than returning an accepted Hessian. Failed gradient evaluations separately record attempted gradients, completed gradients, attempted energies and completed energies; a valid energy is retained when its response derivative fails.

Every displaced node preserves the real analytic-gradient artifact, energy result, orbital checkpoint, native response log, stability record and response controls with hashes. The native CPHF residuals remain independently enforced at every node. No matrix symmetrization or analytic-Hessian label is applied. Two-step agreement does not supply a formal discretization error bound or establish stationarity, continuity of the electronic state, harmonic validity, exact revDSD identity or spectroscopy accuracy; all of those qualifications remain explicitly false or unavailable. Genuine tests compare the new Hessian with a separately evaluated native MP2/D4 gradient stencil, check retained component reconstruction, exercise a real numerical-consistency rejection and verify separate failed-gradient/completed-energy accounting.

## Retrieved independent native output and the `/2021` distinction

An immutable public [H₂ ORCA calculation](https://github.com/avcopan/project-cyclopentene-oh-addition/blob/edef53de01a5b634ebbf637d73ab8679661b270c/calc/basis/H2/opt_orca.out) was retrieved on 2026-10-08, SHA-256 `744d9b92ac7327ba7d3c7d8a0a4b92ac66ab87ea6b5825a4f99b3fe4db0607db`. Its native ORCA 6.1.1 output is explicitly **REVDSD-PBEP86-D4/2021**, with def2-TZVPP, RIJ-COSX, RI-MP2 and DefGrid3. It reports HF exchange `0.69`, PBE exchange `0.31`, P86 correlation `0.4224`, OS scaling `0.5935`, SS scaling `0.0566` and a P86 LDA component named VWN-5. These differ from the secondary 2019 candidate table `0.4210/0.5922/0.0636`; transplanting the 2021 coefficients into the original target would change the method. The native input/output and supplied coordinates are retained with immutable URLs and hashes in the source ledger.

At its first actual H₂ geometry (bond `0.740000000000` Å; native internal value `1.398397339102` bohr), the output reports SCF energy `−1.14951162347332` hartree, scaled RI-MP2 correlation `−0.020919796` hartree and dispersion `−0.000066089` hartree. Four genuine TORQ conventional-integral/LibXC calculations using the explicitly reported **2021** electronic coefficients produced SCF differences of approximately `6.17×10⁻⁵` to `1.31×10⁻⁴` hartree and scaled-PT2 differences of approximately `3.72×10⁻⁶` to `5.01×10⁻⁶` hartree. The conventional/reference algorithms and grids are not matched to the native RI/COSX calculation. This experiment records the disagreement; it does not select a variant by proximity or qualify an exact recipe.

This native observation narrows the version distinction and supplies a real independent engine artifact, but does not complete DH0/DH1. H₂ has no frozen core and its same-spin PT2 contribution vanishes, so it cannot establish the general frozen-core convention or validate same-spin components. The output does not print the 2021 D4 damping parameters, and those parameters were not guessed from the 2019 table. The original primary article/SI, exact P86/FT mapping, fully matched settings and independent molecular component/derivative bundle for the accepted original target remain required.

## Running real research checks

The research dependencies are PySCF, DFTD4 3.7.0 and a TOML parser (`tomli`). Run the tests from an actual CoChem calculation sandbox:

```bash
mkdir -p /tmp/cochem_exec_revdsd_integration
cd /tmp/cochem_exec_revdsd_integration
python -m pytest /path/to/CoChem-TORQ/tests/test_revdsd_integration.py -q
```

The existing [energy/numerical derivative checks](../../tests/test_revdsd_integration.py) and new [analytic response milestone checks](../../tests/test_revdsd_exact_milestones.py) exercise this module. Actual CPU engine checks compare both OS/SS components and total canonical restricted MP2 energies against PySCF's MP2 implementation, verify explicit frozen indices, compare PT2 on actual PBE0 orbitals against the direct MP2 class that preserves those orbitals, compare resolved dispersion against native DFTD4, and check complete numerical MP2 gradients against actual analytic MP2 gradients. The MP2+D4 Hessian check performs 145 genuine full-energy calculations and compares a different stencil built from native MP2 and D4 gradients. Rigid translation/rotation and identical-atom permutation are checked on real molecular energies/components. Strict SCF limits verify genuine nonconvergence retains raw and derivative failure records; budget rejection performs no engine calculation. Complex coordinates and orbital energies are rejected before casting rather than discarding imaginary parts. No engine outputs are mocked.

These checks validate reusable software machinery and numerical differentiation on the tested cases. They do **not** constitute independent validation of exact revDSD, validate a double-hybrid analytic gradient, or establish rotational-spectroscopy accuracy.

Every energy execution saves `input.json`, the actual PySCF log, an orbital checkpoint, resolved basis contents and their checksum, a component/provenance result, and artifact hashes. Failed runs retain a failure record. Numerical gradients retain all displacement identities and parent-result checksums. No missing physical quantity is inserted to complete a later spectroscopy stage.
