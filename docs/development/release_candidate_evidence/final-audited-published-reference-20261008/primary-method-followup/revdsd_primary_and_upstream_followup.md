# revDSD primary-source and derivative implementation follow-up

Recorded 2026-10-08. This is source acquisition and code review. It changes no TORQ code, recipe activation, release gate or scientific qualification. No engine calculation, dependency installation, deployment or human curator attestation occurred in this review. The original 2019 revDSD-PBEP86-D4 definition remains unresolved.

## Newly acquired primary publication evidence

Santra, Bursch and Wittmann, *Modified Opposite-Spin-Scaled Double-Hybrid Functionals*, DOI [10.1021/acs.jpca.5c01035](https://doi.org/10.1021/acs.jpca.5c01035), has a genuine publisher SI packet, DOI `10.1021/acs.jpca.5c01035.s001`. Publisher metadata reports release on 2025-07-24. The original 1,057,521-byte PDF was downloaded from [publisher file56530359](https://ndownloader.figshare.com/files/56530359); its computed MD5 `4ed69adec03b79ddc50e05b8e70ff6f2` matches the publisher record. SHA256 is `684d08a1e9e4fd7bf3897a1f7dbe34d50ac66b77c3dfa275d3a0fa053cce11e1`. Metadata and original bytes are retained in the external review archive.

Table S5, PDF page12, reports the following row, with column labels checked against the original table. The exact excerpt and its byte/character locations are retained; the derived full text was extracted with `pdftotext25.03.0 -layout`.

| Quantity | Reported 2025 SI value |
| --- | --- |
| HF exchange `aX` | 0.69 |
| DFA exchange `aX,DFA` | 0.31 |
| DFA correlation `aC,DFA` | 0.4224 |
| Opposite-spin PT2 `aOS` | 0.5935 |
| Same-spin PT2 `aSS` | 0.0566 |
| D4 `s6` | 0.5917 |
| D4 `a1` | 0.3710 |
| D4 `a2` | 4.2014 |
| D4 `s8` | [0], fixed during optimization |
| D4 `s9` | [1.0], fixed during optimization |

The row cites both the original 2019 paper and the 2021 update. Its electronic and dispersion parameters differ from the original-2019 candidates in the current ledger. It therefore establishes a later reported tuple and does **not** resolve or authorize the original 2019 recipe. Its GMTKN55 performance statistics do not establish geometry, derivatives, rotational constants or identification accuracy for TORQ.

The SI is **CC BY-NC4.0**, attributed to Golokesh Santra, Markus Bursch and Lukas Wittmann: [license](https://creativecommons.org/licenses/by-nc/4.0/). The accompanying exact excerpt retains this attribution and license; it is not project MIT material.

The self-identified author GitHub account `santra-compchem` publishes [MOS-DH](https://github.com/santra-compchem/MOS-DH/tree/fea882b2db1ba4635be2b6f5d2839ee3a506fcc6). Its README links the author preprint `10.48550/arXiv.2408.13813`; its native benchmark summary starts with literal numbers `0.42244 0.59354 0.05662 0.59166 0.00000 1.00000`. Their unlabeled order in that file is not independently treated as a complete recipe. No license file was present in the inspected repository tree, so its original files remain outside the public candidate packet and redistribution permission is not inferred.

## Genuine official PySCF-forge implementation source

The official [PySCF-forge source](https://github.com/pyscf/pyscf-forge/tree/d2fcf42f957f9b5f4b28d324670db04f9a03a03e) contains a substantial double-hybrid implementation, rather than only a method alias. All acquired code bytes are bound to commit `d2fcf42f957f9b5f4b28d324670db04f9a03a03e`, verified against their Git blob SHA1 and SHA256, and covered by the acquired Apache-2.0 license. The source calls PySCF-forge a staging ground. `setup.py` reports version1.1.1 and depends on unversioned `pyscf`, `numpy!=2.4.*` and `sympy`; this does not establish compatibility with the protocol's PySCF2.8 pin or the current PySCF2.14 installation. No compatibility execution was performed here.

The exact registered entry in `pyscf/dh/xccode/functionals/family_bdh.json`, lines212–214, is:

```text
revDSD_PBEP86_D4
0.69*HF + 0.31*PBE, 0.4210*P86_FT + 0.5922*MP2_OS + 0.0636*MP2_SS
    + DFTD4(xc=revdsdpbep86)
ref: 10.1021/acs.jpca.9b03157
```

This matches the previous **secondary candidate** electronic tuple. It supplies an explicit `P86_FT` software choice and no separate `code_scf`. The parser excludes PT2 and dispersion from the SCF functional; the restricted constructor builds density-fitted GKS orbitals from the remaining exchange/correlation terms. These are inspectable upstream software semantics, not verification of the original publication's orbital equation, core convention or P86 flavor.

| Inspected implementation | Established source facts and limits |
| --- | --- |
| `rdfdh.py`, `dh.py`, `mp2_ajz.py` | RI-JK/DF-PT2 energy implementation. Restricted OS/SS contraction implements `E_OS=D`, `E_SS=D-X`, so `(cOS+cSS)D-cSS X` equals `cOS E_OS+cSS E_SS`. This algebra check does not validate numerical integrals or method identity. |
| `resp.py` | Actual GGA XC kernel, hybrid CPKS operator, PT2 density, Lagrangian and relaxed Z-vector density code. GGA derivative paths reject meta-GGA and range-separated references. |
| `grad/dfdh.py` | Actual overlap, one-/two-/three-center integral and auxiliary-fitting derivative terms, nonvariational response terms and delegated D4 gradient are present. This is useful implementation evidence, not an executed gradient acceptance result. |
| Frozen core | The AJZ energy backend explicitly rejects frozen core; alternative energy backends pass a frozen setting to PySCF. Response tensors use full occupied slices, and no frozen-orbital response qualification was found. Frozen-core analytic gradients must remain unavailable until their separate derivative contract is established. |
| Moving grid | The gradient directly calls `grad.rks.get_vxc` and `hessian.rks._get_vxc_deriv1`. The inspected, hash-bound PySCF2.14.0 helper paths use fixed quadrature; the full grid-weight/coordinate response path is separate. Complete atom-centered moving-grid response remains to be implemented and validated. |
| D4 | Energy and gradient delegate to `pyscf.dispersion.dftd4.DFTD4Dispersion` with an `xc` and version argument. Adapter/kernel versions, EEQ/ATM conventions, charge handling and resolved parameters still require pinned source review and native tests. |
| Hessian | No dedicated double-hybrid Hessian implementation was present in the inspected `pyscf/dh` tree. Calling an RKS Hessian helper inside a gradient does not supply a double-hybrid Hessian. |

The source includes genuine MP2 and XYG3 gradient checks against central energy differences, plus stored B2PLYP/XYGJ-OS/explicit-tuple gradient checks. It includes energy comparisons for other double hybrids and alternative MP2 backends. The inspected tests contain no original2019 revDSD-PBEP86-D4 gradient case. Presence of test source is distinguished from successful execution; none of those tests were run in this follow-up.

The original implementation author's [README](https://github.com/ajz34/dh/blob/7e2d5172e0af1cd835af11a41e3a2b903d2ca487/README.md) supplies a peer-reviewed algorithm locator: Su, Zhang and Xu, *Analytic Derivatives for the XYG3 Type of Doubly Hybrid Density Functionals: Theory, Implementation, and Assessment*, *J. Comput. Chem.*34 (2013),1759–1774, DOI [10.1002/jcc.23312](https://doi.org/10.1002/jcc.23312). This is a genuine source citation; the complete primary algorithm paper was not acquired here. That README states that its Hessian is not implemented. The paper's existence and source code's availability do not establish exact revDSD-D4 derivative qualification.

## Minimum defensible next implementation milestone

These are agent-owned coding and numerical-validation tasks. They should run after the current product regression finishes, in a separately pinned comparison environment; do not add an unqualified staging package to a production release merely because its named functional exists.

1. **Freeze a custom restricted GKS/PT2 model and its evidence.** Start with a literal, explicitly unqualified all-electron, closed-shell GGA recipe already expressible in TORQ and PySCF-forge, without dispersion, frozen core, range separation or meta-GGA. A B2PLYP-like literal mixture can isolate the generic GKS response problem without depending on unresolved P86_FT/original2019 identity. Archive orbital and final-energy functionals, OS/SS normalization, basis, grids, convergence, integrals and every input. A named physical method requires its own primary identity review.
2. **Qualify the comparison implementation first.** Pin the acquired forge commit, a compatible PySCF build, LibXC and basis bytes. Execute the genuine upstream energy and numerical-gradient checks, retaining actual failures and all native components. Forge's density fitting differs from TORQ's selected conventional integral baseline: demonstrate fitting convergence or compare explicitly declared matching DF models before interpreting differences as a gradient defect.
3. **Implement the complete TORQ stationary Lagrangian and GKS response.** Differentiate OS/SS PT2 amplitudes and denominators, nonstationary semilocal energy, hybrid XC kernel, overlap/Pulay and AO integral terms. Solve and record CPKS/Z-vector residuals. Validate individual terms rather than accepting a fortuitously small total error at an optimized geometry.
4. **Separate quadrature obligations.** A narrowly labeled fixed-quadrature algebra experiment may compare the same declared fixed nodes and weights in total-energy differences and analytic response. It cannot authorize production atom-centered moving-grid derivatives. Implement weight and grid-coordinate response for production grids and compare against complete energy differences that rebuild those grids at each displaced geometry.
5. **Retain real convergence evidence.** Use genuine source-bound water/ammonia and additional first-row closed-shell molecular geometries, including nonstationary geometries. Compare several symmetric finite-difference steps, tightened SCF/CPKS thresholds, multiple grids and bases. Retain state/stability checks, component errors, units and translation/rotation/permutation checks. Determine tolerances from observed convergence before accepting a bounded capability.
6. **Use the right independence claim.** TORQ-versus-forge can supply separately coded double-hybrid algebra/response checks, but both share PySCF integrals and LibXC. It does not replace matched independent-engine evidence for the target named method. Resolve the original2019 primary equation, P86/core/D4 identity and authenticated native reference components before enabling that recipe.
7. **Accept subsequent products separately.** Add the versioned D4 energy/response only after its own tests. Frozen core, numerical Hessians, cubic/quartic force fields, resonance treatment, vibration-corrected constants and catalog accuracy remain separate milestones. A first derivative pass cannot authorize VPT2, B0 or identification claims.

Original publication acquisition and source reconciliation remain agent research work, subject to lawful access and the actual network policy. A person is not required to hunt ordinary papers or manually transcribe coefficients. Any genuinely external prerequisites should be limited to environment-owner activation of the saved network configuration, lawful authorization if needed, unavailable independent numerical evidence, or a formally required external review. No universal human-signature requirement is introduced here.

## Retained evidence

The companion JSON binds acquired source URLs, commits, original byte counts, SHA256/Git blob checks, the primary SI quote location and licenses. Original PDFs, complete extracted text, author benchmark files, official code, API receipts and mathematical/literal checks remain in the external review archive. The public candidate packet contains this review, its JSON, a licensed exact TableS5 excerpt and a hash index. It contains no credential values, private native logs, engine binaries or invented calculation results.
