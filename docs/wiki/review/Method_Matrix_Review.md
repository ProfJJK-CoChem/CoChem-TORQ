# Complete Method Matrix scientific audit

This consolidated audit covers the historical Method_Matrix at baseline `d7a4739`, including all 20 revision claims, every numbered section, all 140 distinct method row IDs, recipes R1–R9, quick starts, Appendix A and the reference list. It also checks relevant scientific claims in Comprehensive_SRS chapters 2, 4, 5, 7, 9 and 13.

Coverage check: a parser extracted 140 distinct bold table-row IDs from the source; all 140 occur explicitly in the row-disposition annex, with no omissions. The source has 11 method-family categories in the SRS but ten recurring walltime bins in the Matrix; they are independent axes.

This review modifies no application files and makes no claim of live electronic-structure engine validation. Verified upstream evidence is distinguished from literature citations that could not be independently checked. The proposed canonical SRS and method contract should supersede historical normative claims; this audit retains line references to the original documents.

# Method matrix and scientific SRS review

Source baseline: `d7a4739`. Source line numbers below refer to the unchanged `docs/wiki/Method_Matrix.md` (MM) and `docs/wiki/CoChem-TORQ_Comprehensive_SRS.md` (SRS) at that commit. Review date: 2026-10-06 (America/Chicago). This is a scientific and implementation audit, not a claim that a runnable engine has been implemented or that the bibliography has passed peer review.

## Disposition and evidence convention

- **Adopt:** preserve the scientific objective in the implementation SRS with explicit data, validity and acceptance contracts.
- **Correct:** implement the documentation correction; it removes a demonstrable error or unsupported unconditional assertion.
- **Validate:** preserve as an optional recipe/hypothesis, unavailable for automatic production dispatch until an engine-version-specific fixture and in-domain benchmark pass.
- **Decision:** material change in scope, scientific method, computational expenditure, or external dependency; present pros, cons and risks to the PI.

The reference audit and `official-sources/` distinguish freshly retrieved upstream code/docs from merely cited literature. Journal pages and several vendor manuals are outside the configured network policy. A citation in MM alone is not independent verification. All legacy `[M]` measurements mean “reported in the named external source,” not “reproduced for TORQ.” Ratios derived from a particular benchmark are not universal bounds. Historical conference/agent-audit statements are provenance of drafting, not scientific validation.

## Immediate blockers for a coding-ready contract

1. **Two unrelated tier axes are conflated.** SRS 221–403 defines 11 method families. MM 2926–3297 defines ten recurring walltime bins across observable families. Neither is an accuracy ladder. Maintain separate identifiers for method family, exact recipe, requested property/derivative, product class, budget and engine capability.
2. **Unsupported derivative routing.** MM 2111–2129 requires MPQC then ORCA analytic CCSD(T) Hessians, while MM 2177–2202 says ORCA analytic Hessians are SCF-only. Fresh official MPQC source at commit `98fa03e8db0481c63554dd2720164ee463e1402b`, `src/mpqc/chemistry/qc/lcao/cc/ccsd.h:196–198`, accepts energy order zero only. CCSD(T)F12 energy exists, but that does not establish analytic gradients/Hessians. Require per-engine/version/model/derivative capability fixtures. A numerical route needs an explicit stencil, convergence check and budget.
3. **Ground-state constants are mislabeled.** A geometry and isotope masses yield equilibrium or trial rigid-rotor constants, not measured-effective `A0/B0/C0`. Harmonic frequencies do not generally supply vibration–rotation alpha constants. SRS 261–262, 352–353, 741, 983–989 and MM 1208/13.4 require correction.
4. **Unsafe scientific rejection rules.** ML ensemble spread, 1.5% mismatch of cheap rotational constants, an inertial-defect sign, a QTAIM bond path, and T1/D1 thresholds cannot prove a geometry is wrong or a higher method is correct. Keep diagnostics with qualified validity and preserve candidate lineage. Only malformed inputs, declared state/fragment contradictions and failed numerical checks justify unambiguous rejection.
5. **Accuracy and throughput guarantees lack evidence.** Neither 0.1% constants nor publication readiness follow from a tier name, a walltime bin, a basis cardinal number, or a neural committee standard deviation. Define a held-out benchmark, errors and calibration populations, numerical convergence and model discrepancy separately.
6. **No-license revDSD fallback is a new method implementation.** SRS 400–403 does not specify orbital generation, exact exchange/correlation coefficients, same/opposite-spin perturbative correlation, dispersion parameters, frozen core, orbital response or gradient validation. A sum of standard HF, MP2 and D4 energies is not a validated revDSD-PBEP86-D4 optimizer. Fail unsupported rather than silently substitute.
7. **NBO/JANPA equivalence is false for the documented workflow.** SRS 1622–1671 invents a guaranteed JANPA “E2 Module” and interchangeable NBO/NRT behavior. The official JANPA wiki freshly retrieved by the reference audit documents NAO/NPA and LPO/CLPO; its [E2_pert page](https://sourceforge.net/p/janpa/wiki/E2_pert/) explicitly says E(2) is not implemented at least in v2.02, though ingredients can be exported for a related calculation. Distinguish population analysis, localized orbitals, donor–acceptor perturbation analysis, and proprietary NBO program outputs. Do not label an alternative as NBO-equivalent without direct evidence. See [Scientific references](Scientific_References.md) for retrieved sources and limitations.
8. **Several examples are not executable or scientifically valid.** Isotope loop only copies a Hessian; literal `(paired with CABS)` appears in ORCA keyword lines; wrong half-line DVR kinetic operator; wrong encoded negative SPCAT quantum number; incorrect split-conformal inversion; `available_accelerators=3` denotes three GPU IDs, not three MPS slots on one GPU. See detailed annexes.

## Independent axes and exact recipe identity

The SRS method families are retained only as compatibility categories:

| Legacy family | Meaning | Required clarification |
|---|---|---|
| 1 | ML potential | checkpoint/version/hash, training domain, elements, charge/spin, precision, derivative and uncertainty calibration |
| 2 | SQM/tight binding | exact Hamiltonian/version, parameters and supported charge/spin |
| 3 | low-cost/composite DFT | full composite definition; r2SCAN alone plus arbitrary basis is not r2SCAN-3c |
| 4 | hybrid DFT | functional exact name, basis, grid, dispersion/nonlocal correlation, fitting approximations |
| 5 | double hybrid | exact recipe and spin-component coefficients, orbitals, perturbative correlation, dispersion, validated derivative |
| 6 | MP2 | RHF/ROHF/UHF reference, canonical/DF/local, frozen core, orbital/auxiliary basis |
| 7 | local CC | implementation, local thresholds, triples definition, approximation/convergence settings |
| 8 | CCSD | implementation and reference; not generally superior to a validated double hybrid for every property |
| 9 | CCSD(T) | precise triples convention, reference, frozen core, basis, derivative support |
| 10 | F12 | method approximation (a/b/c or code-specific), CABS/RI auxiliaries, geminal/exponents, complementary singles and triples conventions |
| 11 | composite | machine-readable list of component calculations, combination equation and fitted/extrapolated parameters |

MM's recurring budget bins are **10s, 1min, 30min, 1h, 3h, 12h, 1d, 3d, 1w, 1mo** (ten). They remain historical workload labels; R2's 5h estimate is not an eleventh recurring row bin. A budget is a ceiling, not a method promise. A single exact recipe may fit several bins depending on system, requested derivative, hardware and parallelism. A method family may appear in many observable tables. MPQC CCSD(T)F12 and ORCA DLPNO-CCSD(T1) must never share an indistinguishable method identifier. The decision card and quick starts use legacy unqualified `T3-*`/`T4-*` IDs, while actual tables use `T3O-*`/`T3C-*`, `T4O-*`/`T4C-*` (and dual tracks for6 and8). Resolve aliases explicitly; do not silently choose a track or invent extra rows.

The input recipe must distinguish `geometry_method`, `energy_method`, `force_field_method`, `property_method`, `nuclear_motion_model`, isotope masses and calibration source. Reporting a high-level single-point energy does not upgrade the geometry or its rotational constants.

## Review of all twenty MM revision claims

These are MM 19–42, in original order.

| # | Disposition | Coding-ready correction / retained value |
|---|---|---|
| 1 GPU role | Correct / Adopt | Application benchmarking rather than peak-FLOPS reasoning is correct. Report tested GPU/model/basis/precision/derivative; remove blanket no-accuracy-penalty claim outside validated configurations. |
| 2 GPU crossover | Validate | Measure latency and throughput on actual host. Dividing a 32-core crossover to infer 8-core crossover is not a portable gate. Free-code exclusivity/capability negatives require release-scoped evidence. |
| 3 concurrency | Adopt / Validate | Resource tokens and CPU/GPU overlap are useful. 85% real overlap and 3.1x are workload estimates; reserve measured host CPU, memory, scratch and GPU space. |
| 4 mandatory MPS | Correct | MPS is optional host-dependent configuration. Probe support and authorization; no global power or compute-mode mutation in routine jobs; one GPU's client slots are not multiple CUDA device IDs. |
| 5 state reuse | Adopt | Key reusable artifacts by geometry, state, method/basis, code/version and isotope-dependent transformation. Do not hard-code 6N numerical gradients for every CC implementation. |
| 6 initial Hessians forbidden | Correct | A model/reused Hessian is an economical default; exact initial Hessians may be justified for TS or ill-conditioned cases within budget. An absolute ban is scientifically unwarranted. |
| 7 restartability | Adopt / Validate | Capture version-specific restart manifests. Differentiate valid checkpoint restart, safe task decomposition and recomputation; no generic promise from a filename. |
| 8 HDF5 | Adopt / Correct | HDF5 is a storage format, QCSchema a data model. Single-writer transactions, checksums, stable IDs, migration and dimension checks required; a PySCF chkfile is not a universal wavefunction interchange guarantee. |
| 9 isotopologues | Correct / Adopt | Reuse the Born–Oppenheimer electronic PES/Cartesian derivatives, change masses and redo mode/PAS transformations. A Hessian alone does not contain anharmonic corrections; no fixed 6–15x saving guarantee. |
| 10 frozen monomers | Decision | Useful validated approximation for chosen observables; not universal default. It omits deformation and coupling and does not license unconstrained VPT2 at a nonstationary composite geometry. |
| 11 ChS/junChS | Adopt / Validate | Preserve exact published recipe and independent method-level metadata. Equilibrium rotational benchmarks and binding-energy benchmarks are different endpoints. License and derivative support remain prerequisites. |
| 12 additive diffuse/ONIOM prohibition | Correct | A benchmark failure is not a universal mathematical prohibition; assess additivity per recipe. ONIOM has little obvious value for 5–10 atoms but can remain out-of-scope, not declared invalid physics. D3/D4 are not exclusively pairwise when ATM terms are enabled. |
| 13 Be/B0 | Adopt / Correct | Explicit quantities mandatory. Retain observed benchmark statistics with domain; never turn MAE into individual guaranteed error. |
| 14 ORCA/CFOUR tracks | Adopt / Correct | Derivative-aware routing correct; 49 vs176400 is a particular unsymmetrized N=10 calculation, not a universal cost ratio. MPQC additions conflict with these tracks. |
| 15 restored ORCA capabilities | Validate | Use installed-release official docs and fixtures for canonical CC gradients, VPT2 output, spin reference, and property export. Do not infer analytic CC Hessian from analytic gradient. |
| 16 GOAT/CREST union | Adopt / Validate | Union and per-source coverage metadata useful. One F1 benchmark does not guarantee best search in weak dimers. Manual topology enumeration is also not a general completeness proof. |
| 17 MLFF GOAT | Adopt / Validate | Use ML as optional enumerator with out-of-domain/error handling. MACE interaction errors cannot be relabeled AIMNet2 errors. Exact checkpoint and shared reference required. |
| 18 PES active learning | Adopt / Decision | Active/Delta learning is optional accelerated research path. Published 200/472-point success does not prove this complex converges in those counts or justifies fixed one-month to3d re-tiering. |
| 19 provenance | Adopt / Correct | Replace M/D/E-only certainty with externally reported, locally reproduced, derived, estimated, not verified. Any routing policy needs scope, evidence and calibration, including numerical thresholds. |
| 20 presentation/optimism | Adopt / Correct | Stable IDs, failure cases and route diagrams useful. Remove claims that every inconsistency was fixed and every number tagged; document actual audit coverage. A literature-assisted assignment is not evidence of universal numeric accuracy. |

## MM front matter through section 7: complete section dispositions

| Section and source lines | Disposition | Evaluation |
|---|---|---|
| title/revision/front claims 1–60 | Correct | Preserve as historical provenance; no binding science from “conference” or “all implemented.” Internal contradictions remain. |
| decision card 62–100 | Correct | Products useful; fixed accuracy, fixed crossover and mandatory spend order are unvalidated. A near-zero dipole component can still be observed. Quartic distortion is not free unless a suitable force field and implemented property routine exist. |
| QS1 104–144 | Validate / Decision | Search/refinement/report sequence useful. Frozen monomers plus VPT2 requires explicit hybrid force-field validity; placeholder constraints are not deployable syntax. Simultaneous jobs do not automatically cost zero extra walltime. |
| QS2 146–159 | Correct | Three parent constants cannot uniquely determine all internal coordinates. Specify fitted parameter subset/regularization/covariance. Copying `.hess` without mass replacement produces no isotope prediction; force-constant reuse is not automatic vibrational-correction reuse. |
| QS3 161–176 | Adopt / Validate | Campaign store, batched high-level points and held-out checks useful. No universal required dataset size. Validate approximation across the wavefunction-support domain; precondition matrix-free eigensolver and converge basis/grid. |
| 1.1–1.3 180–225 | Correct | Separate absolute, experimental-assisted and differential products. Product B can leak test information; declare fitted measurements. Error cancellation of differences must be measured, and relative error near a zero difference is ill-defined. |
| 1.4 227–235 | Adopt | Separate assignment and method-benchmark purposes. Energies can influence populations and structural plausibility, so “contributes nothing” to assignment is too strong. |
| 1.5 236–242 | Adopt / Correct | Cost is not accuracy. Pareto dominance requires comparable observable, data/domain, uncertainty and resource measure. Observed max error is useful but not population bound. |
| 2.1 246–250 | Adopt / Decision | Small weak-complex microwave scope is coherent. State supported elements, electronic states, fragments and isotope range. SRS open-shell/macrocomplex additions need scope decision. |
| 2.2 252–257 | Adopt / Decision | Exclude photodynamics, periodic solids and resonance Raman from core release. Open-shell extension is material scope; do not claim AIMNet2-NSE alone solves open-shell spectroscopy. |
| 2.3 259–274 | Correct | Keep equilibrium/effective/substitution/mass-dependent distinctions. Sign of semi-experimental correction contradicts5.2; define it explicitly. Effective structures may be descriptive for fluxional species but need a model; “physically meaningless” is overbroad. |
| 2.4 276–280 | Correct | Structural-model uncertainty matters; one Ar–oxazole difference is not universal experimental floor or general prohibition on sub-pm structure inference. |
| 3.0 284–295 | Adopt / Correct | Distinguish Be and B0 and frozen coordinate status. Better Be can improve B0 even if vibration error dominates; “does not move B0 at all” is false. |
| 3.1 297–333 | Correct | Convert accuracy table to benchmark hypotheses and user targets with observables/domains. Means are not intervals. Binding topology, quadrupole and tunneling ceilings cannot be inferred universally from single examples. |
| 3.2 335–345 | Adopt / Correct | Approximate theory can guide assignment; declare search/fit evidence and multiple-testing controls. Manual topology enumeration is not by itself completeness proof. |
| 3.3 347–362 | Decision | Adaptive acquisition by expected effect on target observables is sensible. Hardcoded geometry/vibration/property expenditure order is not universally optimal. D0 matters in stability/population models. |
| 4.1 368–377 | Correct / Adopt | B=h/(8pi²I) in Hz; pseudo-diatomic I=mu R² yields dB/B=-2dR/R locally. Do not apply this derivative to all polyatomic A/B/C without full inertia Jacobian. Use pinned CODATA masses/constants and unit schema. |
| 4.2 379–390 | Validate | Arithmetic is a pseudo-diatomic illustration, not measured molecule constants or a general full-geometry uncertainty budget. Include approximation label and inputs. |
| 4.3 392–419 | Correct | In multiple coordinates residual displacement is approximately -H_internal^(-1)g, not universal g/k. Gradient tolerance bounds depend on curvature/noise and coordinate units. A stopping threshold does not imply every job incurs its worst-case error. |
| 4.4 421–437 | Adopt / Correct | Tight thresholds and final residual diagnostics useful. SCF/grid noise must be checked empirically. Comparing float32 energy resolution in Eh directly with gradient tolerance in Eh/bohr does not prove an absolute impossibility. |
| 4.5 441–470 | Validate / Correct | T-shaped example demonstrates one sensitivity pattern; generic A versus B/C assertions do not hold for every orientation/rotor. Claimed local script must be archived or values marked nonreproduced. No universal “no method errs by83mÅ.” |
| 4.6 472–487 | Correct | Rotational constants recomputed after a bond perturbation are derived calculations even if reported in a paper; label them externally reported computational sensitivities, not experiments. |
| 4.7 489–506 | Correct | CP can affect geometry. No universal 3% cap for nonaugmented TZ or blanket CP-at-TZ switch. Suspicious B3LYP/cc-pVDZ-F12+CABS label requires source check; a CABS is not generally part of ordinary B3LYP. This claimed basis must not be guessed. |
| 4.8 508–525 | Correct | Compare ae/fc at the same basis to isolate core correlation. Difference between fc/cc-pVQZ and ae/cc-pCVQZ includes basis effects; -0.81% is not isolated frozen-core bias. No global fc accuracy cap. Isotopologues in benchmark are clustered, not74 independent molecules. |
| 5.1 531–561 | Correct | Matrix inversion before averaging, frame/gauge and effective kinetic operator explicit. “Element-wise inverse” is wrong. Frequency prefactor required. Averaged kinetic coefficients need not equal effective Hamiltonian fit constants; no universal C-axis reliability ordering. |
| 5.2 563–571 | Adopt / Correct | Define DeltaBvib=B0-Be, then BeSE=B0exp-DeltaBvibcalc. Include degeneracy in alpha sum for linear/degenerate modes; propagate experimental/correction uncertainty and fitted-data provenance. |
| 5.3 573–594 | Adopt / Correct | Counts useful for declared stencil. 2(3N-6)+1 analytic Hessians is a specific normal-coordinate scheme; symmetry/reuse and linear degeneracies alter practical counts. Energy-only Hessian can need 2d²+1 distinct energies, not necessarily d² nested gradients. |
| 5.4 596–614 | Adopt / Correct | beta*hbar*omega gives bead-count scale for a chosen primitive factorization. 2.2 is not guaranteed convergence; require bead/time-step/temperature/statistical tests. Distinguish angular frequency from wavenumber, and finite-T estimator from zero-T spectroscopy. |
| 5.5 616–626 | Adopt / Validate | Consecutive threshold exponents and all other fixed local settings are part of this fitted CPS recipe. PNO convergence is separate from basis and method accuracy; do not transfer thresholds across codes. |
| 5.6 628–630 | Validate | Version-specific supported extrapolation driver; record equations/exponents/basis pair. Text simultaneously gives fixed and optimized beta; exact chosen recipe must resolve it. |
| 5.7 632–643 | Adopt / Correct | Fragment/ghost calculation metadata essential. Raw/CP/half-CP spread is diagnostic, not rigorous confidence bracket. F12 does not mathematically remove all BSSE; geometry correction sign and performance are not universal. Do not double-add gCP/dispersion. |
| 5.8 645–651 | Correct | “MPQC CCSD(T)-F12/TightPNO” cited to Guo/Neese2018 appears source/engine misattribution and must be verified before use. Remove fixed basis-size/time claims and contradictory GPU peak-FLOPS claim. ExtOpt may return analytic gradients; it is not inherently6N numerical. |
| 6.1 658–668 | Adopt / Correct | Signed PAS components require fixed right-handed axis convention; ordinary intensities depend on squared transition moments. 0.1D is a user display threshold, not a physics darkness criterion. Known camphor example itself disproves that exclusion. |
| 6.2 670–674 | Adopt / Correct | EFG, isotope-specific nuclear Q, conversion convention and PAS rotation required. Single chlorinated-complex benchmark cannot settle methods for all N/Cl/Br/I. EFG principal axes generally differ from inertia axes. |
| 6.3 676–680 | Adopt / Correct | Deuterium hyperfine optional if spin>=1 and observable supported. Basis/vibrational convergence needed; a DZ estimate may be exploratory rather than “worthless.” |
| 6.4 682–695 | Correct | Geometric Delta=-2 sum m c²<=0; effective Delta0 can be positive from rovibrational effects. Sign mismatch is not invalid structure proof. Planar moments satisfy non-strict >= ordering and effective moments are not always literal geometric mass moments. |
| 6.5 697–701 | Adopt / Validate | Alpha permits approximate excited-state constants under VPT validity; compute all requested states with units/degeneracies. Jet populations are not generally Boltzmann at rotational temperature; satellites are not universally second strongest. |
| 6.6 703–707 | Adopt / Correct | Direct nuclear dipolar tensor needs gyromagnetic ratios, isotope spins and vibrational orientational averaging. It applies beyond spin1/2 and is not an exact model-free bond ruler. Separate direct and electron-mediated spin-spin terms. No universal50kHz resolvability. |
| 6.7 709–711 | Validate | Spin-rotation is an engine property with gauge, method and vibrational corrections; one D2O agreement is not universal attainable precision. |
| 6.8 713–731 | Adopt / Validate | Fit periodic potential and kinetic coefficients; reduced barrier requires common energy units. F for a methyl rotor is molecule-dependent through coupling; V3 is an effective fit parameter when dynamics model differs. |
| 6.9 733–739 | Correct | 279650/6 spans4.67 orders, not9. Barrier height/reduced mass alone insufficient tunneling prediction. Standard equilibrium PIMD does not directly deliver spectral splitting; specialized path-integral estimators/instanton/variational methods can. “No MD protocol in principle” overstates. |
| 6.10 741–747 | Adopt / Correct | BO mass substitutions and parent-assisted predictions useful. Define relative error denominator for tiny shifts. No guaranteed cancellation, especially H/D ZPE, DBOC and floppy motion; distinguish parent calibration from validation. |
| 7 749–766 | Adopt / Decision | Point group is not molecular permutation-inversion group. Feasibility depends on dynamics and spectroscopic resolution; above-barrier classical motion is not necessary for tunneling. Use explicit operations, isotopes, state symmetry and validated group/spin weights. Single-basin alignment labels required; automatic general group discovery is research scope. |

## Equations and implementable scientific acceptance rules

1. **Inertia and isotope consistency.** For COM-centered coordinates, `I=sum_i m_i[(r_i.r_i)1-r_i r_i^T]`. Use isotope masses with source/version, not atomic numbers or rounded natural-abundance masses. Diagonalize a real symmetric tensor, order `Ia<=Ib<=Ic`, establish a right-handed PAS, and record degeneracy handling. `A,B,C=h/(8pi² I)` in Hz. Linear species have a zero axial moment and require a linear-rotor representation, not an infinite exported A. Confirm isotope substitution changes masses, COM, inertia/PAS and transformed properties while the electronic BO artifact remains immutable.
2. **Vibration and effective constants.** Define `DeltaBvib=B0-Be`; then `Bv=Be-sum_r alpha_r(v_r+d_r/2)`, `DeltaBvib=-sum_r d_r alpha_r/2`, and `BeSE=B0exp-DeltaBvibcalc`. A complete suitable force-field treatment supplies alpha; a Cartesian harmonic Hessian by itself does not. Qualify perturbative model, resonances, rotational Hamiltonian/reduction and omitted large-amplitude coordinates. Numerical convergence does not establish physical model validity.
3. **Inverse-inertia illustration.** For centered R fluctuations with variance sigma² and I=muR², `<1/I>=(1/(mu R0²))[1+3sigma²/R0²+...]`; `1/<I>=(1/(mu R0²))[1-sigma²/R0²+...]`. Their leading difference is **4sigma²/R0²**, not3. The MM table agrees with4. Tensor averaging uses the matrix inverse in a declared common frame and the correct effective kinetic metric; geometry-only averaging is an approximate observable, not automatically a fitted B0.
4. **Convergence versus accuracy.** Adjacent recipe differences are deterministic sensitivity estimates. They are not a statistical test or proof of convergence to truth. Record numerical error (SCF/grid/geometry/FD/nuclear solver), electronic-structure model discrepancy, nuclear-motion approximation, and empirical calibration separately. A stopping rule may accept numerical convergence while returning `accuracy_unverified` or `budget_exhausted` for the scientific target.
5. **CP and dissociation.** For a dimer at one geometry, `Eint_raw=E_AB^AB-E_A^A(R_A)-E_B^B(R_B)` and `Eint_CP=E_AB^AB-E_A^AB(R_A;ghost_B)-E_B^AB(R_B;ghost_A)`. Every leg shares reference conventions, basis and charge/spin definitions. Binding relative to optimized isolated monomers additionally includes deformation. Define positive `De=-Ebind` and `D0=De-[ZPEcomplex-sum ZPEmonomers]` under the declared nuclear model. Report unavailable legs explicitly. Half-CP is an approximation, not statistical coverage.
6. **Method evidence.** CPU/GPU equivalence tests compare exactly matching method, basis, auxiliaries, grid, thresholds, geometry, state and output units. A supported gradient can be checked against converged finite differences; Hessian against gradient differences and symmetry/translation/rotation residuals. No global relative-only tolerance around zero; use `atol+rtol*scale` with documented reason.
7. **Spectral error propagation.** Frequency band alone cannot select an electronic-structure method. From a specified Hamiltonian and parameter covariance, propagate with `Sigma_nu=J Sigma_theta J^T` (or sampled nonlinear propagation), add calibrated model discrepancy, and compare with chosen line-search tolerance/resolution. Unknown covariance remains unknown; one MAE is not a covariance. Transition-intensity model must specify populations, partition function, degeneracy, line strength, polarization and instrumental response. Universal `I proportional mu² nu²` cannot predict empirical SNR.
8. **Analytics.** A density interchange adapter declares basis order, shell representations/normalization, occupations, spin blocks, ECP/core treatment and density kind (SCF, relaxed/unrelaxed correlated). Validate electron count `Tr(DS)`, orthonormality, and selected density/property evaluations. Molden is not universally lossless; changing file container does not recover absent information. QTAIM descriptors are continuous model-dependent evidence, not an unambiguous covalent/noncovalent classifier or VPT2 validity gate.
9. **Uncertainty.** Ensemble standard deviation has energy units and is not variance; it detects disagreement, not guaranteed error. Calibrate on independent molecule/complex groups, preserve non-random adaptive sampling status, define out-of-domain checks and abstention. Product B fitting and Product C isotope correlations must not leak across train/calibration/test partitions. Full split-conformal correction and small-sample limitations are in the tiers annex.
10. **Publication package.** Include exact raw inputs/outputs, complete engine/method metadata, executable versioned analysis, all excluded/failed jobs with reasons, convergence plots/tables, held-out benchmark definitions, data licenses, experimental source and uncertainty, code/environment locks, seeds and artifact checksums. Exported LaTeX or high-resolution figures is formatting, not scientific validation.

## Relevant SRS chapter-specific findings

| Chapter and line range | Required correction and implementation disposition |
|---|---|
| 2, 221–407 | Replace smooth accuracy ladder with independent method/budget/property axes. Resolve QCSchema/QCIO type names; graph hashes are not graph-isomorphism proofs. Define exact double-hybrid recipes. Replace invented license tokens and “signed SHA-256.” T1/D1 threshold exceedance is a warning, not confirmed breakdown. Only quadrupolar isotopes need quadrupole output; unsupported property status explicit. |
| 4, 557–851 | Distinguish preflight geometry hygiene from zero-point energy. Do not discard candidates from unsupported ML/cheap rotational thresholds. Fingerprints may propose torsions but do not prove pathway completeness or graph equivalence. Use circular torsion features, deterministic index mapping, finite budgets and declared coordinate constraints. Group-theory rules classify states, not arbitrary TS optimization constraints. |
| 5, 852–993 | Numerical stabilization of Be across two methods does not validate global PES, barriers or B0. High-level critical points alone do not guarantee PES quality between them. An active-learning campaign needs coverage of bound-region/tunneling paths and held-out reference points. Automatic RO/AP-UHF/DLPNO fallback changes science and needs an allowed-policy decision with explicit recipe identity. |
| 7, 1182–1389 | Ground and excited vibrational effective Hamiltonian parameters need correct force field/nuclear-motion model; QTAIM/dispersion scalar thresholds do not establish VPT2 failure. Correct intensity model, EFG projection and full symmetric traceless tensor conventions; do not diagonalize EFG in inertia axes and discard physical off-diagonals. |
| 9, 1509–1722 | Replace assumed Multiwfn/JANPA=NBO engine with declared capability adapters; verify licenses for exact releases. Separate labels (NPA, WBI, localized orbitals, NBO-program E2/NRT) and never fabricate unsupported results. Density model/interchange checks matter more than a nominal file extension. QTAIM bond paths need interpretation, not automatic hard classifications. |
| 13, 2241–2247 | Current chapter is only one paragraph and an empty improvement placeholder. Specify isotope enumeration cap, abundance/source, isotopic graph automorphisms, masses, COM/PAS remapping, Hessian mass weighting, force-field mode transformation, nuclear spin statistics, dipole/EFG frame transformation, BO limitations and separate calibration/validation identity. Minor isotopologues do not generally dominate dense spectral regions. |

Chapter4 proposed additions require particular caution: an LLM TS optimizer has no verified99.9% guarantee; geodesic interpolation/NEB are options, not guaranteed clash-free MEPs; mass-weighted projected transverse Hessian can retain negative eigenvalues; VRI detection requires projected Hessian/gradient conditions, not merely the second-lowest eigenvalue; a universal2kcal/mol switch to1D DVR cannot certify decoupling; Wigner correction is a high-temperature rate approximation, not a tunneling level splitting or “Wigner–Eckart” method; ground-state AIMNet2 does not return an S1−S0 gap; an absent BCP need not mean physical dissociation. Isotope force-field reuse and derivative-aware explicit fallback are low-risk requirements when their validity limitations are retained.

Chapter9's20 suggestions: density-residual anomaly model and learned BCP locator are optional research; GEBF does not preserve exact local orbitals; NRT capability is unverified; arbitrary E2>15 and Rydberg population>2 traps do not establish rovibrational invalidity; AIMD-NCI is a separate thermal model; SWEBOK defines no IQA scaling law; DORI does not universally replace ELF/NCI; a promolecular density alone lacks the kinetic-energy information for standard ELF; QCSchema is not inherently HDF5; adaptive-grid convergence and shell-convention fidelity are valuable; finite-domain Laplacian tests need units, domain and numerical convergence, not universal1e-4; visual topology overlays are useful but not a “true bond” adjudicator; LOL and Hirshfeld-I have no universal superiority/basis-independence guarantee; open-shell hyperfine is separate scope; EFG-derived quadrupole prediction requires nuclear Q and converged approximation, not the word “exact.”

## Material choices for the PI

| Choice | Pros | Cons | Scientific/implementation risks | Conservative coding baseline |
|---|---|---|---|---|
| Small closed-shell microwave core versus general open-shell/macromolecule/photochemistry platform | Core gives tractable acceptance set and engines; expansion broadens science | Broad scope multiplies Hamiltonians, basis/property support and benchmarks | Plausible-looking unsupported open-shell spectra; delivery scope unbounded | Core small weak-complex workflow; explicitly unsupported cases remain representable and preserved |
| MPQC as mandatory derivative backend versus optional high-level energy backend | Open-source CCSD(T)F12 energy pathway | Analytic CC derivative claim disproved by checked upstream source; numeric costs can be large | Incorrect Hessian routing, hidden approximate method substitution | MPQC energy adapter only until derivative-specific capability evidence exists |
| Frozen-monomer composite as default | Reduces search dimension and can improve selected inertia components | Omits deformation/coupling, may not be stationary for chosen force field | Biased constants/energies and invalid VPT2 assumptions | Optional named approximation with relaxed-reference checks and residuals |
| Custom open-source revDSD implementation | Avoids proprietary dependency, potential reproducibility | Requires substantial derivative/method development | Wrong gradients can silently produce wrong geometry | Capability unavailable unless published/pinned implementation with regression fixtures; expose other exact recipes honestly |
| Active/Delta learning PES in primary release | Can reduce costly points and enable dynamics | Adds training, acquisition, independent validation and model lifecycle | Missed wells or barriers, underestimated committee error, data leakage | Optional research campaign; high-level truth and held-out tests authoritative |
| General LAM/DVR/tunneling and PI automation | Handles scientifically central fluxional effects | Coordinate-dependent kinetic operator, coupling, symmetry and convergence are substantial | 1D reduction or wrong boundary conditions produces precise wrong spectra | Validated named reduced models only; generic routine reports unsupported/needs model |
| NBO replacement by JANPA/Multiwfn | Freely accessible analyses may cover useful populations/localization | Does not establish NBO/NRT algorithm identity or licensing equivalence | Mislabelled publication results and invented E2/NRT output | Explicit analysis-family labels and capability tests; licensed NBO optional separate adapter |
| Empirical parent/template calibration | Often narrows spectral windows efficiently | Needs reference and applies only within validated structural domain | Circular validation, underdetermined structure fits, poor H/D transfer | Product B distinctly labeled with fitted data and independent holdout |
| Strict candidate culling from ML/cheap constants | Saves compute | Can delete correct minima irreversibly from downstream campaign | False negative assignment, missed topology and biased uncertainty | Rank/flag and preserve; cull only explicit user policy with auditable recall validation |
| Mandatory advanced analyses, GPU MPS or HPC infrastructure | May improve throughput/property completeness in equipped labs | Adds license, admin, hardware and runtime dependencies | Blocks simple laptop workflows or mutates shared host globally | Optional adapters/profiles with capability probes and headless single-host baseline |

## Annex integration

The companion `matrix-operational-review.md` audits every section from8 through12, recipesR1–R9, state/HDF5/engine/licensing examples and upstream capability evidence. `matrix-tiers-review.md` audits every table row identifier/family from13 through14, §§15–22, AppendixA and bibliography coverage. These annexes complete coverage of the3990-line Method_Matrix. Any retained source citations in the new SRS should link the claim to this review status and exact upstream evidence rather than copying the historic `[M]` label as certification.


---

# Method Matrix operational/scientific review (§§8–12)

Source: `docs/wiki/Method_Matrix.md`, baseline `d7a4739`; line references refer to that baseline, before any review banner. All lines 768–2925 were read. This review changes no repository files. Dispositions below are proposals for the new implementation contract, not claims that the existing software implements them.

## Evidence and priorities

Direct official-source checks, retrieved through the environment's inherited proxy on 2026-10-07:

- MPQC public repository `ValeevGroup/mpqc` commit `98fa03e8db0481c63554dd2720164ee463e1402b`, copied under `/workspace/torq-review/official-sources/mpqc`. `src/mpqc/chemistry/qc/lcao/cc/ccsd.h:196–198` explicitly implements `can_evaluate(Energy*)` as `energy->order()==0`; `CCSD_F12` inherits this implementation. `f12/ccsd_t_f12.h:18` names KeyVal type `CCSD(T)F12`, and lines 47–67 combine CCSD(F12) with triples energy. `f12/ccsd_f12.h:43–45` exposes approximation C/D, CABS singles, and CABS V-intermediate switches. Thus the public code supports an F12 energy implementation, but neither its existence nor the name certifies equivalence with Molpro F12b, another engine's F12 variant, or the complete-basis limit. No MPQC build or numerical run was performed.
- `https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/README.md`, saved as `official-sources/gpu4pyscf-README.md`: lines 83–89 list g/i angular-momentum ceilings, meta-GGA **with density Laplacian as a limitation**, double hybrids unsupported, and TDDFT Hessian unsupported. Matrix line 847 reverses the Laplacian restriction. README support is not validation of every reference, ECP, solvent and functional combination.
- `https://raw.githubusercontent.com/Parsl/parsl/master/parsl/executors/high_throughput/executor.py`, saved as `official-sources/parsl-executor.py`: lines 312–315 transform an integer `available_accelerators=n` into device IDs `0..n-1`. Matrix `available_accelerators=3` does not create three slots on its single GPU.
- Other manual and paper citations below were assessed as statements in the supplied matrix, not independently fetched or verified. The current network configuration lists package-manager/GitHub destinations; journal and chemistry-manual hosts were not added or bypassed. All release-dependent syntax and capabilities require pinned upstream sources and executable acceptance fixtures before becoming supported production routes.

## Blocking corrections for the coding specification

1. **False analytic-Hessian fallback:** §8D line 2117 routes ORCA to analytic CCSD(T) Hessians; §9.1 line 2137 and §9.3 line 2182 explicitly say ORCA has SCF-only analytic Hessians. Replace engine-name branching with a versioned capability tuple `(method, reference, derivative order, numerical/analytic, basis/ECP, relativistic/solvation/dispersion options)`. Unknown capability fails closed. CFOUR is a candidate route only for the specific supported build/reference combination, not a universal CCSD(T) promise.
2. **False universal cost:** line 2118's ~176,000 single points is the matrix's N=10 nested DLPNO finite-difference illustration, not the cost of every MPQC/Psi4 CCSD(T) Hessian or VPT2 calculation. Derive job counts from N, linearity, independent displacements, derivative provider, stencil, symmetry and reuse; expose time uncertainty and budgets before launch. A numerical Hessian from analytic gradients is different from one differentiated twice from energies.
3. **F12 is not exact CBS:** line 2126's finite cc-pVTZ-F12 'hits CBS' guarantee is invalid. Pin F12 ansatz/approximation, triples convention/scaling, geminal parameters, orbital/JKFIT/MP2FIT/CABS bases, thresholds, frozen-core space and CP treatment. Separate source support, local implementation validation, basis convergence and observable accuracy.
4. **Broken distributed-resource example:** lines 1378–1387 enumerate three distinct accelerators on a one-GPU machine. Seven anchor P-cores plus three dedicated P-core feeders exceed eight P-cores. Independent executors with `block`/`block-reverse` do not establish disjoint cpusets or identify physical P-cores. Require discovered topology, explicit resource leases, GPU UUID allocation and aggregate CPU/RAM/VRAM admission control. MPS is an optional benchmarked backend mode, not a mandatory prerequisite.
5. **Isotope output is physically unchanged:** lines 1646–1649 merely copy the same Hessian and rerun a tool, with no mass substitution, then suppress failures. Isotopic identity must be an atom-index→nuclide/mass mapping; verify substituted masses and changed inertia/mass-weighted frequencies. Reuse of a Born–Oppenheimer Cartesian force field is valid; transformed normal-coordinate anharmonic constants/resonances and mass-dependent corrections require separate treatment.
6. **Worked input is not executable:** lines 1653/1817 place literal `(paired with CABS)` prose in ORCA keywords; line 2192 contains `MPQC CCSD(T)-F12-F12`. Recipe titles and engine strings were corrupted by substitutions. A basis name containing F12 does not turn ordinary CCSD(T) into an F12 method. Historical snippets must not be copied into generators.
7. **Unsafe scientific storage:** lines 1998–2014 append columns independently; missing gradients yield different row lengths; convergence defaults to true; method definitions and Hessians can be overwritten; no checksum-based immutable calculation identity; no commit marker or crash recovery. `delta_pairs` joins identifiers without checking identical geometry/charge/spin/basis identity; `dvr_grid` accepts duplicate/out-of-range/unvalidated IDs and holes. Require validated immutable records, atomic success commit, finite arrays, explicit units, masks, duplicate conflict handling, restart/retry identity and a single-writer/sharded ingest contract.
8. **Unsupported culling guarantee:** G4 line 1312's Spearman rho≥0.9 on top10+random10 cannot prove no discarded low-energy basin. A 10/12 kcal/mol screen window and Tukey/IQR committee threshold are heuristics, not calibrated safety guarantees. Publication workflows preserve candidates or use a validated recall procedure with an explicit residual omission risk; unvalidated culling is opt-in exploratory only.
9. **Harmonic Hessian is not B0:** the diagram at 1208 suggests an analytic Hessian directly gives B0. It validates local curvature and provides harmonic information; B0 requires defined vibration–rotation treatment. Likewise CP/SP reranking can select different conformers but does not improve the fixed conformer's geometry-derived Be.
10. **Unit arithmetic error:** line 1829: 0.05% of 3.5 Å = 0.00175 Å = **0.175 pm**, not 1.8 pm. Also ΔB/B≈−2ΔR/R is a simplified dominated-inertia sensitivity, not a universal tensor identity.
11. **Do not destroy distinct species using a prediction window:** line 2575 merges structures if their predicted rotational constants overlap. Distinct isomers may have similar A/B/C yet different dipoles, hyperfine constants, isotope signatures and transitions. Preserve structural IDs and represent spectroscopic ambiguity as a group/overlap relation; derive resolution from transition uncertainty and instrument settings.
12. **Composite corrections must isolate the intended effect:** line 2393 defines an ae/core-valence versus fc/valence-basis difference, which mixes core correlation and basis change. A CV increment must use ae−fc at the **same** core-valence basis, same geometry/coordinate convention, same Hamiltonian/reference and other settings. Parameter-wise geometry corrections require documented matching coordinates; do not transplant correlation-energy n^-3 extrapolation rules to geometry parameters without the published geometry protocol.

## Complete section disposition

`Retain` means retain the principle with ordinary typed contracts and tests. `Revise` means materially correct before implementation. `Decision` means an optional scientific/deployment policy with significant tradeoffs, disabled or non-authoritative until approved/validated.

| Section; source lines | Disposition | Required treatment |
|---|---|---|
| 8.0; 770–780 | Retain | Preserve named teaching/workstation/HPC profiles, but actual resources come from machine/job discovery. Setup labels are not capability facts. |
| 8.1; 782–812 | Revise | P-core-only MPI is a benchmarkable default, not a theorem; never assume CPU IDs 0–15 identify physical P-cores. KMP settings govern Intel OpenMP and do not bind MPI ranks. Aggregate all process memory/scratch and quantify headroom; `%maxcore` is a budget, not allocated bytes or a hard ceiling. Avoid fixed 16-way independent-job launch without memory admission. The CPU FLOP/bandwidth roofline is a hypothesis, not proof each named kernel is bandwidth-limited. |
| 8.2; 814–855 | Revise | Keep GPU DFT as an optional validated backend. Distinguish FP64 representation from physical accuracy, cross-engine equivalence, grids and deterministic tolerance. Correct Laplacian restriction. Do not turn one benchmark's agreement into a universal bound. CPU reoptimization can itself change method/grid/approximation; if used, report it as a distinct method stage. |
| 8.3; 857–900 | Revise | Local measured latency/throughput surfaces replace basis-count cutoffs and peak-scaled speedups. 'DF always' is not universally valid; validate DF error/auxiliaries for the requested derivative. CPU/GPU memory spilling/fallback are backend/version properties. Global claims 'no GPU CCSD(T) in any free code' require current source review and must not be specification gates. |
| 8.4; 902–980 | Revise | Good matched/default comparison concept. Explicit functional definition, radial/angular/pruning grids, DF auxiliary exponents, SCF residual and state are needed. DF exchange is approximate, not 'exact DF'. `mf.grids.prune=True` is not a documented prune callable. An energy difference <1 mHa neither certifies derivative correctness nor proves agreement sufficient for microwave accuracy. Require convergence flags, replicate timings, uncertainty, cold/warm/end-to-end times and gradient/Hessian comparisons. Iteration savings legitimately affect end-to-end speed; report separately rather than declaring them not speedup. `nvidia-smi dmon` does not measure SM occupancy. |
| 8.4a; 982–1000 | Revise | Queue wall caps/site charging are configuration, not universal 48h law. Distinguish optimizer warm-start from exact checkpoint restart and absence of documented restart from proof of impossibility. 6N displaced energies form a central numerical gradient; displaced gradients form a Hessian. Scratch estimates require engine/storage mode/occupied and virtual sizes and measurements; SATA/network/OS-disk blanket prohibitions are unjustified. |
| 8.4b; 1002–1018 | Revise | Pin current runner/account policies and discover actual quotas. Published public minutes are not unconditional unlimited compute entitlement. Engine license entitlement and redistribution differ; exclude proprietary binaries from shared images by default without claiming all private use forbidden. Store essential restart artifacts durably; retention is policy. |
| 8.5; 1020–1152 | Replace | Overlapping branches wrongly force SCF Hessians/TDDFT/dipole/solvation to ORCA before GPU branch; engine presence does not establish capability. Product A/B/C must derive from requested provenance, not mere existence of an experimental analogue. Unknown calibration produces uncalibrated uncertainty, not a fabricated window. Route by capability/accuracy/domain/resource/license/budget evidence; never silently reduce scientific method. |
| 8A.1; 1160–1185 | Revise | Retain contention accounting. No fixed 85% free parallelism, 1.20 slowdown, negligible PCIe or mandatory P-core feeder can be generalized from one MACE profile. Measure aggregate job/memory/thermal performance. No automatic global board-power changes. |
| 8A.2; 1187–1219 | Revise/Decision | Advisory geometry/Hessian proposals are useful with provenance. Candidate culls influence authoritative output and cannot be called harmless. Co-running streams is not automatically free; declared asynchronous novelty requires literature review. Snapshot live state only at supported immutable checkpoints; no mutation of running anchor inputs. High-level final energies alone do not establish exhaustive discovery. |
| 8A.3; 1221–1249 | Revise/Decision | Require model element/charge/spin/domain checks. Import Hessian with atom ordering, units, symmetry and regularization checks; translation/rotation modes are not necessarily the lowest six (five linear; rotations only null at stationary point). A poor preconditioner can change reached basin or fail convergence; no unconditional 'does not bias minimum'. Fixed fmax/compile restrictions are version/model-specific. |
| 8A.4; 1251–1300 | Revise/Decision | MPS optional, version/device/admin capability probed; serial GPU mode must work. Private pipe/log dirs, lifecycle cleanup, device UUID and explicit memory margin; never change compute mode or power cap without deployment-level authorization. Worker count chosen by measured throughput and host demand. A five-second VRAM snapshot is not peak-memory profiling. |
| 8A.5; 1302–1342 | Revise | Retain authoritative/advisory provenance, Hessian stationarity checks, abort guide and model hashes. Replace sampled-rank cull guarantee; calibrate UQ on independent target-domain errors. RMSD must handle atom permutations, fragment alignment and linear-angle singularities. Separate electronic success, geometry convergence and stationary-point classification. |
| 8A.6; 1344–1410 | Replace examples | Parsl is viable but not itself scientific provenance or a resource correctness guarantee. Correct accelerator IDs, disjoint cpusets, process trees, model caching, timeout/cancellation, scratch and safe argv. The shell f-string at 1401 can interpret user paths as commands; use allowlisted argv/quoting. Local→Slurm is more than replacing a provider: allocations/launchers/modules/storage/credentials differ. |
| 8A.7; 1412–1457 | Revise | Cost example is hypothetical, not expected performance or robust benefit. Includes inconsistent 3 MPS workers/1 P-core and omits proper high-level G4 audit scheduling; timeline ends 3.9h but claims 3.6h. Record critical path, resource graph and alternatives. |
| 8B.1; 1463–1467 | Revise | Geometry/force-field reuse is valuable. 'One isotopologue cost' excludes mass-dependent corrections, new coordinate transformations, changed resonances and rovibrational solutions. |
| 8B.2; 1469–1510 | Revise | Keep adapter-specific state inventory with exact engine-version/ABI/atom/basis compatibility and checksums. `.opt` trajectory claim conflicts with earlier 'trajectory does not survive'. GPU arrays are not necessarily plain NumPy arrays and need supported host conversion. Charge and multiplicity belong to immutable species records, not only sidecars. Never infer unknown restart support as universal absence. |
| 8B.3; 1512–1545 | Revise/Decision | Model Hessian is a reasonable default; blanket ban on calculated optimization Hessians is unwarranted for difficult curvature. Hybrid force field is a defined approximation; cost is not zero and mode matching/rotation must be implemented. Different low frequencies do not alone establish matching normal vectors. |
| 8B.4; 1547–1823 | Replace examples | Snippets have invalid tokens, relative seed path after chdir, PATH-based ORCA launch contrary to own requirement, no timeout/cancellation/resource validation, reused directories, no converged-optimization parser, no seed copy in Python, incomplete parser validation, overwritten HDF5 history, and false isotope processing. Add isolated attempt directories, absolute executable paths, immutable inputs, finite/shape-checked parsed outputs and convergence checks. `FCMINT` handoff alone does not implement a hybrid anharmonic transformation. |
| 8B.5; 1826–1845 | Revise | D1 arithmetic correction above. A small imaginary mode can be numerical noise, not proof wrong basin. D2 'always safe preconditioner' too strong; check/regulate. A Hessian off a minimum is meaningful but harmonic stationary-point interpretation is limited. One 20% frequency change cannot detect mode mixing. D3 stability tests may be unavailable and high iterations are not proof of basin change. D4 CP electronic energies on non-CP optimized geometry are valid if reported as composite; species occupancy/basis checks are sound. D5 unique attempts and no stale autostart are sound. |
| 8B.6; 1847–1874 | Revise | Decompose only mathematically separable work. Analytic Hessian/individual CC point cannot always be subdivided by an external driver. Restart artifacts need committed durable storage, hashes, compatibility and retention; Actions cache is evictable and must not be sole durable checkpoint source. Validate exact CFOUR parallel derivative workflow. |
| 8C.1; 1880–1895 | Revise | HDF5 bulk arrays + catalogue is reasonable; chunk sizes workload-specific. Attribute names borrowed from QCSchema do not make a conformant schema readable by QCArchive; QCSchema geometry uses Bohr while example uses Å. Publish a versioned explicit mapping/export and validate with actual schema. |
| 8C.2; 1897–2093 | Replace code contract | All blocking storage corrections above. Validate metadata upon reopening; immutable method IDs; row and gradient masks; finite values; explicit pending/failed/converged/validated status; shape consistency; unique attempts; input/artifact hashes; transaction/commit markers; no silently overwritten Hessians. Specify missing-grid behavior before solver launch. |
| 8C.3; 2095–2107 | Revise | `todo()` does not ensure idempotence under concurrent submissions/crashes. Matching point IDs alone does not ensure identical geometries. Store exact CP legs/definition, not a mutable string. MPI HDF5 is collective parallel I/O, not arbitrary multiwriter transactions. Single collector/shards plus validated atomic merge is the default. |
| 8D; 2111–2118 | Replace | Remove false ORCA Hessian route and universal nested-SP count. Hybrid fallback changes the method and requires informed selection plus explicit result labeling, not silent 'exact physics'. |
| 9.0; 2125–2128 | Revise/Decision | MPQC F12 energy path supported in inspected source; version/build-specific acceptance gate remains. Remove finite-basis=CBS assertion. A policy restricting MPQC to validated SP energetics is conservative; restricting source geometry to one DFT engine has no general scientific justification. Any accepted geometry must retain provenance/validity. |
| 9.1 legacy; 2130–2131 | Revise/Decision | An open default is sensible; 'legacy' is not a scientific-quality classification. Optional licensed backends remain useful; current entitlement/availability, not presumed wet-signature turnaround, controls deployment. |
| 9.1 multicode; 2133–2150 | Revise | Keep derivative-cost distinctions; do not confuse force-field stencil count with universal VPT2 implementation. 49 Hessians at N=10 is a stated strategy/linearity assumption. Restrict CCSD(T) analytic second derivatives by reference/build/property. |
| 9.2; 2152–2175 | Revise | Treat all license terms, public release dates, build hazards and turnaround as versioned procurement information, not a scientific routing theorem. Protected institutional installs may serve entitled users; private shared images are not automatically equivalent to public redistribution. No unattended bundled proprietary binaries. |
| 9.3; 2177–2202 | Revise | Normalize capability tuples and evidence statuses; remove malformed F12 method. Separate canonical CC, DLPNO unrelaxed density, relaxed response, reference types and derivative options. 'Not on page' means unverified, not unavailable everywhere. Reject unsupported requests before compute. |
| 9.4; 2204–2219 | Revise | Benchmark thread/rank and memory semantics by pinned engine/module; a table's claimed CFOUR global memory must be verified for actual MPI/OpenMP build. No automatic unbinding override that violates scheduler cpusets. |
| 9.5; 2221–2314 | Revise | Ar–HCN has four physical atoms, not five; X is dummy, not isotope-bearing atom. Provided coordinates are illustrative, not optimized; isotope block needs actual official syntax/physical atom mapping validation. Straight angles require coordinate-safe treatment; XYZ2INT alternatives mean a blanket 'hand-write ZMAT' mandate is too restrictive. Runtime template tests must prove derivative method, linearity and output interpretation. |
| 9.6; 2316–2326 | Revise/Decision | Keep modular search→refine→energy→rovibrational→spectral stages. Electronic energy Boltzmann weights require temperature/statistical/kinetic assumptions and free-energy/ZPE/degeneracy treatment; jet populations are not equilibrium by default. Single 'best' conformer may omit observed metastable states. Gate licensed/flexible/LAM paths explicitly. |
| 9.7; 2328–2342 | Revise | Remove exclusivity claims 'only route' unless scoped to implemented adapters. Distinguish free of charge from open-source/redistributable. Educational licenses do not authorize research use automatically. PySCF/DFT can feed an external finite-difference anharmonic driver even without native VPT2, at defined cost/validation. |
| 9A.1; 2348–2363 | Revise/Decision | Frozen monomers are an optional approximation, not universally safe or universally default. Sensitivity of A/B/C to internal and external motion depends on tensor axes and molecule. Quantify deformation with paired relaxed/control calculations and report constrained stationarity. |
| 9A.2; 2365–2375 | Retain with revision | Keep relaxed/frozen-iso/frozen-inc provenance, remove universal A<0.2% guarantees. Record source, uncertainty, constraints and constrained gradients. |
| 9A.3; 2377–2385 | Revise | Focal-point combinations valid only with clearly defined legs and convergence; additivity is approximated, not generally independent. Exact complete recipe/coordinate and method identity needed for gradient combination. Literature performance is not out-of-domain certification. |
| 9A.4; 2387–2431 | Revise/Decision | Correct CV same-basis difference and published geometry extrapolation protocol. Specify jun/aug/F12 family per leg. Do not generalize benchmark accuracy or claim F12 is always 10x cheaper. Different geometry versus force/energy additive schemes are distinct. Template regression needs training provenance, domain and propagated covariance. Thermochemical recipes may be out of initial scope without false universal 'none produces better rotational constant'. |
| 9A.5; 2433–2439 | Revise | Conservative diffuse-basis default reasonable; universal 'no additive diffuse correction ever' exceeds specific evidence. ONIOM does not require covalent cuts/link atoms; exclude based on scope/benefit instead. D3/D4 ATM approximations do not include full many-body induction, but self-consistent supermolecular DFT does include induction; do not imply the whole DFT model lacks it. |
| 9A.6; 2441–2457 | See R1–R9 table | Make each recipe opt-in/applicability-checked; no literature mean becomes a per-system prediction interval. |
| 9A.7; 2459–2474 | Revise | Preserve no double-counting dispersion; CP leg geometry and basis identity. Remove universal DZ/TZ/half-CP/F12-free rule: select validated convention and report CP/no-CP sensitivity. Binding versus interaction energy needs both monomer deformation terms (not an unspecified universal 'fourth leg'). Frozen gradient size includes inter-level inconsistency and is not alone proof of physical deformation. Core/diffuse/BSSE and extrapolation provenance remain explicit. |
| 9B.1; 2480–2488 | Revise/Decision | GOAT/CREST multi-engine search useful, but TS-conformer benchmark and 70–120-atom failure study cannot prove superiority for 5–10-atom weak complexes. Open deployment defaults to available validated open search; optional GOAT entitlement. |
| 9B.2; 2490–2512 | Revise | Pin algorithms/version/flags; do not generalize every CREST workflow as MD or all constraints as inexact. Resolve internally contradictory cost '125x' versus '4x'. More found structures not proof greater coverage; sociological claims not needed. |
| 9B.3; 2514–2579 | Revise | Hand-enumerated sites are a documented search space, not a mathematical completeness proof. Preserve independent seeds, source identities, fragments/charge/spin and random seeds. Common refinement + structural deduplication sound, but never merge merely for overlapping A/B/C. Check bthr units and version rather than mixing percentage/fraction literals. Per-run directories prevent overwriting ensembles. |
| 9B.4; 2581–2608 | Revise | Element/domain/charge/multiplicity model support mandatory (organic model cannot serve Ar just because target is vdW). Pin/check wrapper capabilities and dependencies. `%scf TolE` may not control external optimizer energy criterion; verify wrapper/optimizer interface. Numerical versus analytic g-xTB gradients determined by pinned binary, not stale tutorial; measure derivative correctness/cost. |
| 9B.5; 2610–2616 | Revise | Search budget not universally ≤1h/nonbottleneck. Keep calibration and coverage diagnostics; spend based on measured scientific uncertainty and missing candidates. |
| 10.1; 2624–2644 | Retain with tests | Version-pin ORCA/OPI exact invocation/quoting; serialize argv safely; electronic-structure settings ignored by ExtOpt must be explicitly declared as external-model identity. |
| 10.2; 2646–2680 | Retain with tests | Parser must validate record count/types, paths, atom order, charge/spin, ncores, gradient flag and point-charge support. Atomic output creation, request IDs, finite values, permissions, no stale .engrad accepted; differential fixtures for supported versions. |
| 10.3; 2682–2693 | Retain | Force→gradient sign and eV/Å→Eh/bohr factors are correct. Test by finite differences, not just constant algebra. Wrong gradient scaling can affect convergence without necessarily changing exact stationary geometry, so avoid asserted guaranteed symptom. |
| 10.4; 2695–2699 | Revise | Validate each optimizer/NEB/MD/frequency mode as capability; external energy/gradient interface alone does not prove all workflows supported. Property and solvent information must come explicitly from external provider; no silent native ORCA property assumptions. |
| 10.5; 2701–2711 | Revise | Treat wrapper-suite support as pinned/versioned; hard-coded all-version lack of MACE-OFF can age. Resolve g-xTB derivative route via pinned source and probes. Server mode optimization remains optional supported mode; account for startup, health, idempotency and timeouts. |
| 10.6; 2713–2725 | Revise | Illustrative Opt/Freq/NumFreq keywords must be tested for valid interaction; Linux-only and macOS workaround claims conflict. Explicit validated parameter-file lookup in controlled config, no blind files in home. |
| 10.7; 2727–2787 | Revise | Correct unit code, but default aliases mutable, checkpoint unpinned, no element check, path scoping, finite/shape checks, atomic output, timeout, actual thread-limit use, output reuse protection or persistent serving. float64 arithmetic does not establish accuracy beyond learned potential. |
| 10.8; 2789–2804 | Revise/Decision | Committee spread normalized by sqrt(N) presumes size/error behavior; define estimator and units explicitly. Training-error IQR does not calibrate OOD error or coverage; use independent target-domain calibration, report failures, and never average incompatible Hamiltonians/checkpoints. |
| 11.1; 2810–2822 | Revise | Exclude unlicensed redistribution by default; validate exact ORCA release/license/entitlements. Old EULA is not authoritative proof of all current institutional/cloud restrictions; do not diagnose commercial infrastructure as commercial research automatically. |
| 11.2; 2824–2826 | Revise/Decision | Separate code, weights, dataset and dependencies licenses with hashes/versions; evaluate deployment entitlement. Orb/SevenNet license alone does not justify substituting materials-trained models into molecular work. |
| 11.3; 2828–2830 | Revise | Free software labels still carry obligations; free of charge does not establish redistribution. Maintain SBOM/license inventory, exact package and model artifacts, notice/license delivery. Unknown relevant license blocks packaging, not all unrelated computation. |
| 12.1; 2836–2840 | Revise | Calibration includes workload, versions, numerical settings, concurrency, resources, repeated timing and dispersion. One SP cannot predict all gradients/Hessians/optimization wall times. |
| 12.2; 2842–2849 | Revise | Matrix's two-track taxonomy conflicts with new open MPQC/PySCF policy. Represent each validated recipe as an adapter graph rather than inherited ORCA/CFOUR row labels. No requirement to invent a method at every budget tier. |
| 12.3; 2851–2875 | Retain with revision | Good quantity/domain/provenance/resource/state/license/limitation columns. Include explicit uncalibrated uncertainty state, normative IDs, exact method tuple, applicability, evidence version, validation fixtures, and software support status. Core-hours must be summed by stage and separate allocated from consumed; GPU-hours separate. |
| 12.4; 2877–2886 | Revise | CPU/GPU device, parallel/decomposable, restartable and critical-path seriality are separate axes. Nonrestartable analytic Hessian need not be serial or a CPU-only operation. MPS assumption not universal. |
| 12.5; 2888–2918 | Retain principles; revise guarantees | Preserve weakest-link qualification, tails/domain, convergence sensitivity, no assumed monotonicity, Be/B0 labeling, measurement provenance. Independent vibrational-correction levels reduce common-method dependence but do not by themselves remove circularity; independence derives from data lineage. Numerical thresholds/basis/core corrections cannot guarantee a universal % error. Literature `[M]` differs from locally reproduced `[M]`. Percentage max errors unstable near zero; report absolute errors and denominator rule too. |
| 12.6; 2920–2922 | Replace | Confirm verifier actually exists before claiming enforcement. Validate substantive capabilities, references and method/quantity/schema linkage; ten tiers and cosmetic table shape cannot establish science. Permit unavailable/uncalibrated cells and scientifically justified missing tiers. |

## R1–R9 recipe-by-recipe ruling

| Recipe; line | Retained purpose | Correction / implementation contract | Risk and default |
|---|---|---|---|
| R1; 2447 | Cheap constrained geometry and Be screening | Explicit monomer source/re/SE status and uncertainty, fragment masks, actual rigid-body DOF (not universally six), isotope masses, constrained gradients and basis/dispersion treatment. No guaranteed A<0.2%. | Frozen-fragment deformation bias; default optional screening recipe, no publication accuracy promise. |
| R2; 2448 | High-quality monomer + DFT intermolecular optimization + high-level energy + vibrational correction | Title MPQC versus steps DLPNO versus 'ORCA throughout' must be resolved into actual separate engine stages. Energy refinement cannot improve Be. Geometry/force-field stationary-point and mode-map consistency essential; constrained/frozen geometry cannot be fed to unrestricted full VPT2 without an explicit projected/relaxed protocol. | Major cost/approximation choice. Separate optional constrained model from default fully relaxed model; withhold B0 if anharmonic correction validity fails. |
| R3; 2449 | Published junChS-F12 geometry composite | Implement exact cited F12b/CBS/CV protocol, same-basis CV increments, coordinate mapping, versioned license-gated analytic/numerical gradient capability. MPQC F12 energy is not a drop-in F12b gradient implementation. | License, cost, basis/domain and additivity risk; optional validated reference recipe. |
| R4; 2450 | Conventional CBS+CV composite | Distinguish ChS, junChS and energy-gradient versus parameter-addition variants; ensure diffuse functions and same-basis CV increment. ORCA is not license-free: academic no-charge is still license-gated. Accuracy a calibration result. | Multiple expensive geometry legs and additivity; optional high-accuracy recipe with budget and in-domain validation. |
| R5; 2451 | Transfer monomer structural corrections | Pin coefficients, units, bond typing, training data, covariance and applicable base method; no regression for intermolecular coordinate without validated data. Label empirical calibration and prevent leakage. | Distribution shift and bias; opt-in anchored/empirical product, not de novo. |
| R6; 2452 | Semi-experimental isotopic prediction | Three constants generally cannot identify a full geometry uniquely. Define constrained parameterization, weighted fit/Jacobian, degeneracy, prior constraints and covariance; parent B0 is not directly equilibrium inertia. Recompute masses and isotopologue vibrational corrections. | Circularity/nonidentifiability/model error; opt-in measured-parent product with independent validation. |
| R7; 2453 | Focal-point composite gradient | Exact legs, dimensions, same geometry/reference, CP convention, weighting/extrapolation, derivative consistency and convergence checks. An implementation may use a vetted external driver, not exclusively Psi4. | Additivity plus finite-difference/noisy gradients, substantial validation burden; optional advanced recipe. |
| R8; 2454 | SP energetic refinement/reranking on known geometries | Preserve valid energy-only recipe. It gives no fixed-geometry Be change but can affect which conformer's spectrum is considered and populations under declared models; therefore not scientifically 'worthless'. | Modest extra cost and geometry error remain; permissible labeled energetic refinement. |
| R9; 2455 | Layered QM treatment | Out of initial 5–10 atom scope by cost/benefit and validation, not because ONIOM necessarily requires covalent cuts. | Partition/link/embedding error and unnecessary complexity; defer unless expanded system scope justified. |

## Decisions with material pros/cons (do not silently enable)

- **Frozen monomers as production default:** lower optimization cost and potentially better intramonomer structure; cons include deformation suppression, nonstationarity, ambiguous full force fields and axis-dependent bias. Recommend fully relaxed default; explicit constrained recipe with paired validation if desired.
- **Surrogate culling and asynchronous steering:** saves expensive evaluations and can explore additional basins; can irreversibly omit relevant minima or steer basin selection while all final numbers look high-level. Recommend proposal-only scouts and structural candidate retention until recall/coverage validated; any destructive culling is an explicit exploratory mode.
- **Substituted hybrid VPT2:** major cost reduction; assumes adequate normal-coordinate compatibility and perturbative regime, can fail for low modes/resonances/LAM. Require signed method change, coordinate transformation/overlap tests and separate validation; withhold the requested CCSD(T)-VPT2 label.
- **Licensed CFOUR/Molpro/ORCA versus all-open deployment:** licensed engines provide useful derivative/property capability but add entitlement, redistribution and reproducibility burden. An open core with optional licensed adapters is practical; no unsupported open analytic CCSD(T) route should be fabricated to fill a feature gap.
- **MPS and GPU/CPU co-scheduling:** may increase throughput on small jobs; increases host contention, device accounting ambiguity and failure interactions. Default serial GPU jobs plus disjoint CPU leases; enable MPS only after supported-device/driver probes and measured throughput gains.
- **Unconditional CPU reoptimization after GPU:** can offer independent numerical check, but doubles work and may change approximations/grid/backend. Prefer validated numerical-equivalence tolerance and reproducibility checks; a CPU authoritative finish is an explicitly costed optional recipe, not proof all GPU geometry is unfit.
- **External benchmark-based advertised accuracy:** informative for planning; not a guarantee on weak complexes, isotope differences or a new engine build. Keep literature evidence separate from held-out local scientific validation and suppress calibrated-coverage language until coverage conditions are met.

## Minimal acceptance fixtures implied by this review

Capability negative tests (ORCA CC analytic Hessian, gpu4pyscf double hybrid, MPQC analytic derivative) must fail before scheduling; pinned supported positive controls must identify exact methods and convergence. Resource tests verify one physical GPU cannot be allocated as devices 1/2, no cpuset oversubscription, total memory admission and teardown. Scientific tests include sign/units finite differences, translated/rotated atom-permuted Hessian round trips, linear 5 versus nonlinear 6 external modes, actual isotope mass substitution with analytic inertia examples, no duplicate dispersion, same-geometry CP legs, same-basis CV difference, and no claim Be→B0 from harmonic Hessian alone. Storage tests include crash between columns, duplicate retry, missing gradients, changed geometry under same ID, NaN energy, incorrect units, unknown method ID, conflicting merges and incomplete DVR grids. Search tests preserve distinct structures with equal/overlapping rotational constants and cannot promote sample Spearman correlation into a completeness proof. These are scientifically meaningful gates; none requires executing the illustrative broken snippets.


---

# Method Matrix §§13–23 and Appendix A: scientific and implementation review

Review target: `docs/wiki/Method_Matrix.md`, baseline commit `d7a4739`, lines 2926–3990. Review scope includes every method-row family in Tables 1–10, every wall-time tier, expansion blocks, §§15–23, and Appendix A. This report does not modify the repository.

## Evidence limits and disposition vocabulary

The supplied text was inspected in full. Internal mathematical, dimensional, arithmetic, and consistency checks are directly checked in this review. Published benchmark numbers and version-specific executable capabilities have **not** been independently verified against their primary journal/manual sources: the cloud proxy policy lists package/GitHub hosts, not those journal/manual hosts. A citation's presence is not verification. No source was fetched by bypassing that policy. Readable public code was inspected at `https://raw.githubusercontent.com/Ltotheois/Pyckett/master/pyckett/__init__.py` (69,568 bytes retrieved during this review) to verify Pickett integer conventions. That is corroborating implementation evidence, not a frozen-release certification of SPCAT.

- **Retain with contract**: useful method or requirement; implement only with the qualifications below.
- **Correct**: definite mathematical, physical, or internal inconsistency; correction is safe and does not require choosing a new scientific scope.
- **Qualify**: a claim may be valid for a named source's test system; it cannot serve as a universal guarantee or scheduling gate.
- **Experimental/defer**: substantial solver, calibration, capability, or validation work; keep it in the roadmap behind an explicit feature/capability gate.
- **Decision**: material scientific/computational tradeoff for the user, with safe default stated below.

The canonical implementation SRS and method contract should supersede this historical matrix's normative language. A source table's time, percent error, `[M]` marker, or named program does not establish a validated TORQ implementation.

## Highest-priority corrections

### M-T01 — Conformal interval inversion and coverage claim (3401–3416)

The calibration score is `s = |B_calc − B_exp| / B_exp`, but the emitted interval is `B_pred*(1 ± q)`. Those are not the same prediction set. For positive rotational constants and `0 ≤ q < 1`, solving the score inequality gives

`B_exp ∈ [B_pred/(1+q), B_pred/(1−q)]`.

For example, prediction 100 and q=0.1 give [90.9091,111.1111], not [90,110]. If q≥1, there is no finite upper bound using that score. Alternatively define scores on log ratios and emit multiplicative `B_pred*exp(±q)`, or divide scores by the *prediction* and retain the symmetric formula. The choice must be fixed before calibration.

The quantile rank `ceil((n+1)*(1−alpha))` is correct; n=22 and alpha=.1 give rank 21. If rank>n, the standard conservative finite-sample construction gives an unbounded prediction set, not a clipped maximum score. The stated n=22 cannot be assumed: the six diagnostic entries include isotopologues, a paired-species entry, missing constants, and overlapping/nonindependent observables. Calibration/test splits must be by independent molecular family, separated from method selection, hyperparameter tuning, parent anchoring, and training. Curated benchmark mixtures are not automatically exchangeable with deployment targets. Conformal coverage is marginal under the stated exchangeability assumptions; neither distribution-free conditional coverage per species nor extrapolation coverage follows. ABC and multiple-transition simultaneous coverage require a family-level score or multiplicity policy.

**Safe contract**: implement correct score inversion, provenance-separated calibration, explicit sample count/quantile rank, `uncalibrated`/`out_of_domain` status, and no numerical coverage claim until an independent evaluation exists. Choosing/calculating the reference corpus is a material work package.

### M-T02 — Incorrect radial DVR operator (3610–3616)

The half-line matrix printed in A.2 omits a diagonal term and a factor of two off diagonal. For the standard reduced radial sine/sinc DVR, `r_i=i*Δr`, i=1,2,..., Dirichlet boundary at r=0, with prefactor `ħ²/(2μ Δr²)`, the dimensionless matrix is

```
T_ii       = π²/3 − 1/(2 i²)
T_ij,i≠j   = 2 (−1)^(i−j) [1/(i−j)² − 1/(i+j)²].
```

This follows directly by odd reflection of the infinite-line sinc kinetic operator about r=0. At i=1 the correct diagonal is 2.789868..., not 3.289868.... The radial wavefunction/measure, angular basis and boundary conditions must be part of the solver contract; simply substituting a Cartesian sinc matrix is invalid. In the Jacobi Hamiltonian, define whether angular momentum operators are dimensionless to avoid adding ħ² twice. Angular coupling and monomer rotation cannot be omitted merely because there is a radial scan.

**Safe contract**: correct the formula; require Hermiticity, free-rotor angular spectra, an analytic radial potential, grid/box convergence, and a reference coupled-coordinate calculation before a new solver can publish observables. General 6-D VRT remains a substantial feature decision.

### M-T03 — Harmonic Hessians do not determine general vibration–rotation α (3073, 3088–3091)

A harmonic Hessian supplies harmonic frequencies, normal modes, and harmonic-force-field centrifugal-distortion contributions under the appropriate model. General vibration–rotation interaction constants include anharmonic contributions that require cubic force information (or a separately justified approximation). A bare `.hess` file does not, by itself, establish a full VPT2 isotopologue B0 calculation. Force-field reuse across isotopes is valuable under a Born–Oppenheimer PES approximation, but must carry all required Cartesian derivatives, transformations, resonances and isotope-dependent kinetic terms.

**Safe contract**: label harmonic-only output as such; never infer VPT2 availability from a harmonic Hessian; require full derivative artifact availability and an actual isotope reanalysis capability test. Also reject applying ordinary free-molecule VPT2 to a constrained nonstationary geometry without a documented constrained/reduced model.

### M-T04 — Rotational averaging, finite temperature, and conformer mixtures (3063–3065, 3075–3079, 3092–3094, 3177, 3638–3644)

1. Classical finite-temperature MD does not supply a rigorous upper bound on the quantum zero-point correction. The proposed “bounded upper estimate” has no bound derivation and must become an uncalibrated classical diagnostic.
2. `μ=I⁻¹` has inverse-inertia units. Averaging μ and then inverting it (3093) produces an inertia, not a rotational coefficient. A body-frame inverse-inertia approximation has frequency-valued coefficients `h/(8π²) * <I⁻¹>` (Hz after SI conversion), while a full rovibrational kinetic-energy tensor can differ from simple geometric inertia. Direct averaging of those frequency-valued tensor elements is algebraically equivalent; averaging independently rediagonalized instantaneous A/B/C is generally not.
3. Eckart/body-frame embedding, tensor alignment, degeneracy handling, Coriolis couplings, and the Hamiltonian from which effective constants are extracted must be explicit. A vibrational expectation value is not automatically an experimentally fitted Watson B0 for a floppy/tunnelling system.
4. PIMD at 50 K samples a finite-temperature quantum distribution. It does not automatically produce a 0 K ground-state observable or the 2 K jet population. Bead convergence alone cannot establish temperature/ground-state convergence. Ordinary PIMD also does not directly supply real-time quantum spectra or tunnelling splittings.
5. A mixture of conformers produces a weighted **sum of conformer spectra**, each with its own Hamiltonian. Boltzmann averaging the constants into one spectrum is invalid unless a separately justified fast-exchange model applies. Jet populations may be kinetically trapped; equilibrium Boltzmann weights need a declared assumption and sensitivity range.
6. At 5 K, a classical harmonic mode has mean **total** energy kBT ≈ 3.48 cm⁻¹, with half in each of the kinetic and potential terms. The text's 1.74 cm⁻¹ is the potential contribution, not total mode energy. Likewise exp(−ΔE/kBT) is a dimensionless Boltzmann/Arrhenius factor, not a barrier-crossing rate without a prefactor and dynamical model. These corrections preserve the valid conclusion that classical low-temperature sampling misses zero-point motion.

**Safe contract**: keep per-conformer spectra and population provenance; quarantine unsupported absolute B0/upper-bound output; distinguish equilibrium, expectation, and effective-Hamiltonian parameters. Choice of rigid-monomer PIMD, DMC, or VRT is material.

### M-T05 — Deduplication can discard real species and “dark” branches (2929, 3584–3598)

The lead table says Stage-A `ΔE=.100 kcal/mol`; Appendix A says `.05`, and search windows differ (12 versus 6 kcal/mol). Pin a version/configuration and one authoritative value set. Loosening a merge threshold merges *more* pairs, so the claim that it must be looser than method error “or genuine conformers are merged” is reversed. Method bias versus experimental structure is also different from repeatability of optimized replicas of one basin.

Retain an AND-based geometrical/energy/constants duplicate check, but use atom mapping that preserves elements, isotopes, fragments and chemically relevant connectivity; hydrogen-only rearrangements cannot be safely ignored by a heavy-atom-only RMSD test. Fixed tolerances are configurable numerical duplicate criteria, not proof of experimental indistinguishability. At 12 GHz, .1%=12 MHz, far above a 2 kHz line precision.

An absolute .1 D “dark” cut is a detection assumption, not a selection rule. It contradicts the document's own .08 D branch example. Nonzero weak transitions must not disappear from archival output: annotate or apply an explicit user/instrument intensity filter to a view, retaining unfiltered transitions. Enantiomer/rotamer degeneracy, nuclear-spin weights and partition-function degeneracy must not be counted twice; handle each symmetry label/isotopologue consistently.

### M-T06 — SPCAT interchange defects (3422–3448)

The `a0…z9` convention is wrong: lowercase encodes negative values, e.g. `a0=−10`, `a9=−19`, `z9=−269`. Uppercase `A0=100` through `Z9=359`; ordinary 10–99 remain decimal. Checked by executing `pickett_int` and `format_` extracted from Pyckett public source: 10, 99, a0, a9, z9, A0 and Z9 round-trip correctly. Retrieved source SHA-256: `310d6f1aec9a00a7e3705e290599f294e84acdbc5bf1a23804220d56616255a1`. Six two-character upper-state quantum numbers occupy columns 56–67; lower-state quantum numbers occupy 68–79. Relative lower start is 13, not 14 as source line 3440 states. Header widths verified in Pyckett sum to 55 before quantum numbers.

Do not hard-code generic parameter-sign/coupling transformations from narrative examples. Versioned export profiles must define Watson reduction, representation, spin coupling, parameter IDs, units, uncertainty/covariance interpretation and round-trip golden files. The body alternates `.par+.int` and `.var+.int`; test the pinned binary's actual supported input contract and use one documented interface. SPCAT ERR does not become a calibrated prediction interval merely by filling it with a planning error. Parameter covariance must be propagated or its absence stated. Two-K and 2–20 GHz are configurable defaults, not universal output restrictions.

### M-T07 — Table-derived cost and error claims are not transferable (2964–2994, 3004–3057, 3295, 3305–3328)

- 500 points × 15 min=125 serial core-hours **only if each point uses one core**; /16 workers = 7.8125 h ideal wall time, not 10.4 h (2992). If points are parallel multi-core jobs, core-hours and worker count change.
- 2500 trajectories × 1000 steps × 5–15 s=3472.22–10416.67 single-core hours, not 21504 (3295). 21504 is a 128-core × 168 h *budget*, not the calculated workload.
- A 32000/64000-size matrix or card TFLOP rate cannot establish an application's crossover/performance; retain local measured throughput/capacity tests.
- Active-learning point-count savings in H2O–He or a different Δ-learning molecule do not prove an order-of-magnitude wall-time reduction on a new 6D system. Split sampling/acquisition, training, independent validation, electronic-structure, and solver costs.
- MAE/RMSD/max error on named systems are different statistics; none is a universal symmetric±prediction interval. F12/basis/core method strings and target observable must match the cited benchmark.
- Apparent Pareto dominance across unmatched benchmarks, differing observables, hardware, licensing, or deformation constraints is a heuristic recommendation, not demonstrated dominance. “No extra wall time” still consumes extra resources and can contend.

**Safe contract**: preserve published estimates only as provenance-tagged planning data; derive runtime from measured tasks/resources and leave accuracy uncalibrated. Never route solely from a universal time-tier label.

### M-T08 — Internal-rotation/tunnelling statements exceed evidence (3204–3219, 3389, 3525–3526, 3650–3662)

One ammonia–formic-acid span is neither a universal ±14% error bound nor a universal performance ceiling. The largest quoted discrepancy is `(195.18−168.3)/195.18=13.77%`, while the upper is 9.03%; these are observed method errors for that case. A 1D constrained continuous path generally provides an *upper* bound to the minimum-energy-path barrier when comparable endpoints are used; a discontinuous envelope from relaxed scans may not give any useful path bound. The source's general “lower bound” claim is incorrect.

The water-dimer range 6–279650 MHz spans about 4.67 orders of magnitude, not nine. This range alone proves no universal factor-of-three prediction ceiling. Methyl rotation does not guarantee two distinct resolved lines for every transition: degeneracies, forbidden/weak transitions, unresolved splittings and multiple-top couplings matter. The formula `s=4V3/(9F)` requires V3 and F in the **same energy or frequency units**; source line 3656 mixes a cm−1 barrier and GHz F without specifying conversion. Correlated barrier and kinetic-parameter uncertainty must be propagated through the torsion/VRT solver.

### M-T09 — Guard policy must retain evidence and avoid unsupported physical guarantees (3334–3376)

“Discarded” should mean excluded from scientific publication/promotion, not deletion of the run and evidence. Preserve raw artifacts with rejected/quarantined status, reason codes and retries.

Some triggers are sound diagnostics, but their quoted numbers are unvalidated defaults, not physical theorems: 10% spin deviation, 15i cm−1, 25% CP/raw energy, 10% force-relative uncertainty, 5% bad frames, “more than a few” overlap combinations. Relative metrics become singular near zero spin, energy or force. Use absolute+relative floors, isotope/electronic-state-aware diagnostics, a versioned policy, and explicit thresholds/unit definitions. If output lacks a required diagnostic, status is unknown, not passed. T1/D1 alone cannot certify single-reference adequacy; orbital stability also needs method-specific support. Low imaginary modes require recomputation/convergence evidence, not automatic reclassification as numerical artifacts.

Do not force all MS ions/fragments to charge +1 and multiplicity 2 or all open-shell references to unrestricted form. Electron count, ionic channel, fragments and supported restricted-open-shell/unrestricted choices determine allowed states. S66 energies cannot calibrate rotational constants; geometry benchmarks cannot calibrate interaction-energy errors. PNO extrapolation spread is a convergence diagnostic, not total error. Correlated and numerical uncertainties must remain separate.

### M-T10 — Benchmark and publication contract incomplete (3380–3397, 3488–3514)

The six entries do not exercise every claimed observable (Raman, UV/visible, NMR, MS are absent). Rows 5–6 omit actual rotational constants; row 6 compares *different compositions*, not two conformers of one composition. Dataset records need exact isotopologue, state/tunnelling symmetry, Hamiltonian/reduction, units, uncertainty and covariance, fitted/fixed status, source table/page, measurement conditions and machine-readable values. The water-dimer effective/tunnelling-state constants cannot be compared to a generic rigid-rotor output as if identical observables. Any computed prior can induce correlations in remaining fitted constants: excluding only a fixed A does not establish independence of B/C; retain the dependency annotation and, where possible, reference sensitivity/refits.

Reporting all observables “in MHz” is dimensionally inappropriate for dipoles, lengths, NMR shielding, etc.; use natural units, plus MHz only for energy/frequency quantities with a valid conversion. Signed dipole components need a reproducible, right-handed axis convention; signs are coordinate-dependent and near-degenerate inertia eigenvectors can rotate/swap. For observational intensity, squared components and interference/Hamiltonian conventions matter.

Provenance requirements are strong and should be retained, adding schema versions, immutable artifact hashes, units, model/engine feature probe results, source/reuse dependency DAG, retry/resource history, calibration/train/test membership, covariance, population assumptions, rejected results and data/software/model licences. Full raw proprietary binaries need not be archived/distributed; capture hashes and acquisition instructions. Reproducibility means declared tolerances and rerunnable inputs, not guaranteed bitwise identity across hardware.

## Complete row-family disposition

The universal tier set is `U={10s,1min,30min,1h,3h,12h,1d,3d,1w,1mo}`. Each occurrence below names all ten rows in that family; ORCA/CFOUR track gaps remain explicit unavailable capabilities. Tables 1, 2, 5, 7, 9, 10 contain 10 rows each and dual-track Tables 3, 4, 6, 8 contain 20 each: **140 distinct row IDs reviewed**. All expansion blocks inherit the preceding corrections and require engine/version-specific executable recipes rather than copying illustrative snippets literally.

### §13.1 — T1-{U}, conformer search (2928–2958)

| Rows | Disposition and coding contract |
|---|---|
| T1-10s | Retain as hand-seeded local optimization; no coverage/completeness guarantee. |
| T1-1min | Retain GOAT capability; local search budget, explicit seeds/degeneracy, no assumed 1 min completion. |
| T1-30min | Retain MLFF enumeration with domain/precision metadata and QM reoptimization; Stage A follows M-T05. |
| T1-1h | Retain independent CREST ensemble; constrain/report dissociation, versions and seed sensitivity. Individual published trajectory events are examples, not universal failure times. |
| T1-3h | Retain union, common-level reoptimization and conservative deduplication; no ensemble-energy interval without calibration. |
| T1-12h | Experimental resource-heavy QM search; choose measured budget/coverage gate rather than fixed worker advice. |
| T1-1d | Retain convergence diagnostics; matching degeneracy and thermodynamic conventions does not prove global coverage. |
| T1-3d | Retain higher-level reoptimization/ranking when separately validated; .1% output is not guaranteed by a method label. |
| T1-1w | Experimental system-specific training; independent heldout basins, training/licensing provenance, data-size learning curve required. |
| T1-1mo | Optional search-completeness evidence; cannot be declared dominated by same-start dual-device searches without coverage/cost comparison. |

### §13.2 — T2-{U}, PES and DVR (2960–2994)

| Rows | Disposition and coding contract |
|---|---|
| T2-10s | Retain qualitative 1D topology scan; specify relaxed versus rigid coordinates consistently with expansion. |
| T2-1min | Retain MLFF presampling; low-level errors can also move boundaries, so even “well shape only” needs domain checks. FP32 legitimate use is conditional, not unique to this row. |
| T2-30min | Retain DFT prescreen PES; benchmark long-range dissociation/asymptotics and reference energy zero; do not use ROT34 geometry error as energy evidence. |
| T2-1h | Retain grid with bidirectional continuity/hysteresis, but 960 relaxed points/hour unmeasured. Dependency-aware neighbor starts cannot simply race with independent parallel jobs. |
| T2-3h | Retain 1D solver only for a specified effective Hamiltonian with convergence; M-T02. |
| T2-12h | Experimental Δ+active learning: paired same-geometry low/high data, independent test domain, symmetry and asymptotic constraints; no transferred 3–10 cm−1 guarantee. |
| T2-1d | Experimental committee acquisition; calibrate uncertainty and reserve independent validation. Two disagreeing models do not detect common bias. |
| T2-3d | Experimental 3D variational model; fix cost arithmetic, coordinate measure/KEO and active-coordinate validity. |
| T2-1w | Experimental 6D VRT; resource forecast and effective rotational extraction reference required; no universal 1–5 cm−1 interval. |
| T2-1mo | Separate flexible-monomer PES scope; modern supported adapters/licences and cost validation before adoption. Legacy program version list is not a deployable recipe. |

### §13.3 — T3O-{U} and T3C-{U}, equilibrium geometry (2996–3057)

The Be/B0 distinction is correct and essential. “Be is not an observable” should mean it is an equilibrium model parameter not directly equal to fitted ground-state constants; do not use that wording to imply it cannot be inferred. No universal ΔBvib=.1–.7% bound, all-frozen-core bias, DFT basis ceiling, or floppy 1–2% floor follows from cited examples.

| Rows | Disposition and coding contract |
|---|---|
| T3O-10s | Qualitative preoptimization and diagnostic Be only; replace borrowed benchmark accuracy with uncalibrated status. |
| T3O-1min | Retain frozen-monomer approximation with monomer provenance, residual force/deformation assessment and constraint map; not universal dominance. |
| T3O-30min | Retain diffuse/basis/CP sensitivity protocol; reconcile blanket “do not CP below TZ” with T5 policy instead of hard-coding contradiction. |
| T3O-1h | Retain diffuse basis option; deleted functions can silently change model, so report overlap truncation and convergence. |
| T3O-3h | Split geometry, CP energy and VPT2 tasks: single-point CP energy does not CP-correct an optimized geometry; a substituted Hessian at a nonstationary geometry needs justification. |
| T3O-12h | Retain named composite protocol only if all legs, coordinate maps, basis/extrapolation and geometry increments are exact and executable; error benchmark remains scoped. |
| T3O-1d | Experimental numerical-gradient F12 route; verify actual engine/API and parallel task counts. It is not proven dominated from unmatched benchmarks. |
| T3O-3d | Correct method specification: an F12-optimized orbital basis does not turn canonical AUTOCI CCSD(T) into an F12 method, and CABS is not required merely by choosing that orbital basis. Capability tests required. |
| T3O-1w | Optional expensive composite; each correction has same target structure/coordinate convention and convergence evidence. |
| T3O-1mo | Optional external-engine F12 reference; licence/capability-gated, no universal dominance or Molpro-only-in-principle claim. |
| T3C-10s; T3C-1min | Retain unsupported/gap state; do not fabricate a replacement inside CFOUR. |
| T3C-30min | Retain MP2 structural check; parallel support and memory are version/build-specific. |
| T3C-1h | Reclassify as property task, not geometry result. |
| T3C-3h | Retain analytic-gradient optimization after coordinate/derivative feature probe. |
| T3C-12h | Retain basis-convergence step; correct inconsistent .43%, .812% and −.81% values by source/statistic/domain, not averaging them. |
| T3C-1d | Retain all-electron/core-valence protocol with explicit electron mask. |
| T3C-3d | Retain composite with same complete leg contract as T3O-12h. |
| T3C-1w | Optional post-CCSD(T) geometry increments with measured restart/cost. |
| T3C-1mo | Optional relativistic/DBOC corrections; distinguish an energy correction from its gradient/optimized-geometry contribution; public-version capabilities must be probed. |

### §13.4 — T4O-{U} and T4C-{U}, vibrational averaging (3059–3124)

| Rows | Disposition and coding contract |
|---|---|
| T4O-10s | Retain inertial defect/planar moments with isotope masses and physical units. Inertial defect is not “three subtractions on A,B,C”; convert to moments first. A static planar geometry has zero defect within numerical tolerance and does not directly test vibrational physics. |
| T4O-1min | Retain measured-parent calibration as distinct Product B/C; three parent constants do not uniquely determine all coordinates of a scaled geometry. Parameterize/fix structure, propagate uncertainty, reject unidentifiable fits; do not call simple scaling an r_e^SE determination. |
| T4O-30min | Correct harmonic/α distinction, M-T03. |
| T4O-1h | Retain VPT2 for supported stable nonresonant/semi-rigid cases, full derivative provenance; 49 Hessians assumes an unconstrained nonlinear 10-atom case and central differences. |
| T4O-3h | Correct conformer spectrum mixture, M-T04. |
| T4O-12h | Retain isotope force-field reuse with full required derivatives and reanalysis feature test; `.hess` alone is insufficient for general anharmonic B0. |
| T4O-1d | Experimental rigid-rotor PIMD via actual supported driver; `%md` alone proves no PIMD implementation. Distinguish classical diagnostic and quantum finite-T result. |
| T4O-3d | Correct inverse-inertia units/embedding; publish model-effective constants or levels with convergence, not cm−1 “equivalent” uncertainty without mapping. |
| T4O-1w | Experimental ground-state DMC; finite population/timestep/descendant-weight biases, independent replicates and embedding convergence required. |
| T4O-1mo | Experimental VRT/tunnelling on an explicitly specified rigid versus flexible surface; current table and expansion disagree about which. |
| T4C-10s; T4C-1min | Preserve explicit gaps. |
| T4C-30min | Retain harmonic sanity check; exact program restart/output capability test. |
| T4C-1h | Retain supported CCSD(T) analytic Hessian, with reference/spin restrictions recorded. |
| T4C-3h | Retain VIBROT for its actual derivatives; isotope symmetry and lower-symmetry requirements explicit. |
| T4C-12h | Retain VPT2/centrifugal outputs only when parser+units+Hamiltonian profile exists; oxirane errors remain scoped. |
| T4C-1d | Property averaging additionally needs dipole/EFG property derivatives; force field alone does not generate them. |
| T4C-3d | Optional high-level force field; displacement manifest, symmetry and resume verification required. |
| T4C-1w | Retain full-derivative isotope reuse; budget marginal cost rather than assigning a week to a reuse operation. |
| T4C-1mo | Optional small corrections: not all keyword combinations automatically produce every desired corrected spectroscopic quantity. |

### §13.5 — T5-{U}, energetics (3126–3156)

| Rows | Disposition and coding contract |
|---|---|
| T5-10s | Screening interaction energies only; distinguish frozen-fragment interaction, relaxed binding, electronic De and zero-point D0 with sign convention. |
| T5-1min | Screening gate may not irreversibly cull on uncalibrated MLFF energy; assess false-negative risk beyond rank correlation. |
| T5-30min | Retain composite without double-counting embedded dispersion/gCP. Three-leg raw/CP results are not automatically meaningful for a composite with built-in gCP; document validated combination. |
| T5-1h | Retain raw/CP/half-CP provenance; half-CP/CP-free F12 is not universal prescription or proof of negligible BSSE. |
| T5-3h | Retain composite F12 energy only with exact supported method definitions/bases/auxiliaries/leg formulas; A14 statistics are not general ± bounds. |
| T5-12h | Retain DLPNO energy/LED; using an F12 orbital basis+CABS alone does not request F12. The illustrative input needs executable keyword correction/probe. |
| T5-1d | Retain PNO extrapolation only for supported threshold pair/convention; spread is convergence uncertainty, not total physical error. |
| T5-3d | Retain canonical F12 reference capability; cannot prove dominance by composite on different benchmark cases. |
| T5-1w | Optional post-CC corrections; correct “D3/D4 is pairwise-additive”: implemented variants can include an Axilrod–Teller–Muto three-body term. Specify model/options; no universal 15–20% three-body correction. |
| T5-1mo | Optional SAPT interpretation; computational tier depends on system/method and should not be one month by definition. Decomposition components are model/convention-dependent. |

### §14.1 — T6O-{U} and T6C-{U}, secondary observables (3162–3198)

| Rows | Disposition and coding contract |
|---|---|
| T6O-10s | Retain moments/defect; r−3 dipolar interaction needs nuclei, gyromagnetic ratios, angular tensor and vibration averaging before a numerical coupling claim. |
| T6O-1min | Retain low-level dipole diagnostic; no hard branch exclusion from threshold or assumed ±.5 D error. |
| T6O-30min | Retain hybrid dipole/EFG with axes, units, nuclear Q source and uncertainty. |
| T6O-1h | Retain property-specific basis test; one basis/method benchmark does not supply universal 5%. |
| T6O-3h | Retain harmonic quartic constants for applicable Hamiltonian; uncertainty uncalibrated until benchmarked. |
| T6O-12h | Retain supported density-dependent properties; explicitly distinguish unrelaxed density, relaxed response and energy derivative. “CC quality” alone is not an error model. |
| T6O-1d | Requires independent property derivative surfaces plus vibrational model, not force field only. |
| T6O-3d | Correct conformer constants averaging and validate export using M-T06. |
| T6O-1w | Retain excited vibrational-state constants only with supported mode/resonance treatment; not universally second strongest jet features. |
| T6O-1mo | Keep unavailable capability until engine/version evidence; future adapters can implement equivalent scientific outputs. |
| T6C-10s; T6C-1min | Preserve gaps. |
| T6C-30min | Retain supported multipoles/EFG with tensor conventions and numerical checks. |
| T6C-1h | Retain property method; EFG×Q(mbarn)×234.96474 produces kHz under the stated atomic-unit/nuclear Q convention; state isotope, sign and Q uncertainty. |
| T6C-3h | Retain quartic derivation with profile; oxirane errors not universal. |
| T6C-12h | Retain sextic capability after golden fixture; degree/order/convention and perturbative validity explicit. |
| T6C-1d | Retain supported spin-rotation; isotope and magnetic convention; D2O 3% example not general experimental accuracy. |
| T6C-3d | Retain property derivative+force-field averaging contract. |
| T6C-1w | Optional DBOC derivative/geometry effects, not scalar energy substitution into constants. |
| T6C-1mo | Optional relativistic closure; automated validated export is preferable to manual transcription; engine licensing not assumed impossible for classes without checking current terms. |

### §14.2 — T7-{U}, LAM (3200–3219)

| Rows | Disposition and coding contract |
|---|---|
| T7-10s; T7-1min | Retain qualitative prescan; MLFF shape may also be wrong out of domain. |
| T7-30min; T7-1h | Retain declared 1D scan with continuity/hysteresis; correct bound claim and remove universal ±14%. |
| T7-3h | Retain 2D coupling model; active-coordinate convergence and kinetic coupling contract. |
| T7-12h | Experimental splitting solver; units of V3/F, rotor symmetry/statistical weights, validation and uncertainty propagation mandatory. |
| T7-1d | Retain per-conformer barriers as separate outputs. |
| T7-3d | Optional high-level path energies; path relaxation/kinetic errors remain separate from electronic accuracy. |
| T7-1w | Experimental instanton/WKB under validity criteria; no universal factor-of-three promise. |
| T7-1mo | Experimental full VRT; line assignments require symmetry/selection rules beyond band energies. |

### §14.3 — T8O-{U} and T8C-{U}, IR/THz (3221–3255)

| Rows | Disposition and coding contract |
|---|---|
| T8O-10s; T8O-1min | Retain diagnostic harmonic modes only; MLFF soft modes require precision/force convergence. |
| T8O-30min; T8O-1h | Retain harmonic frequencies/intensities with stationary structure, grid and mode classification; separate harmonic from experimental fundamental errors. |
| T8O-3h | Retain validated VPT2 with resonance/low-mode domain gates. |
| T8O-12h | Hybrid force fields are experimental unless mode mapping and equilibrium compatibility are validated; 100 cm−1 cutoff is heuristic, not sufficient or necessary validity criterion. |
| T8O-1d | Experimental classical/approximate-quantum correlation spectrum; require actual dipole surface, window/resolution, temperature and quantum-correction assumptions. A full spectrum is not exact quantum anharmonic spectroscopy. |
| T8O-3d; T8O-1w | Experimental variational frequencies require transitions and dipole matrix elements for intensities; fit RMSE is not a guaranteed frequency error floor. |
| T8O-1mo | Preserve current adapter gap; not scientific impossibility of numerical high-level force fields in general. |
| T8C-10s; T8C-1min | Preserve gaps. |
| T8C-30min; T8C-1h | Retain supported harmonic methods and reference restrictions. |
| T8C-3h | α-only VIBROT does not supply full fundamentals/spectrum; correct classification. |
| T8C-12h | Retain supported VPT2 with resonance and derivatives fixtures. |
| T8C-1d | Resolve inconsistency with nonpublic GUINEA/anharmonic-intensity limits before offering resonance/intensity feature. |
| T8C-3d | Optional expensive force field with measured convergence/resume. |
| T8C-1w | Retain isotope reuse with all derivative artifacts and proper mode reassignment. |
| T8C-1mo | Optional small corrections; not an automatic complete spectrum from keyword stacking. |

### §14.4 — T9-{U}, Raman (3257–3274)

| Rows | Disposition and coding contract |
|---|---|
| T9-10s; T9-1min | A static polarizability alone does not establish Raman activity; require derivatives/surface and selection rules. Diagnostic-only unless supported property model exists. |
| T9-30min; T9-1h; T9-3h | Optional supported polarizability derivative workflow; declare laser frequency, basis convergence, units and scattering geometry. |
| T9-12h | Experimental anharmonic Raman unless pinned engine genuinely supplies property derivatives and resonance treatment. |
| T9-1d | Experimental polarizability-correlation spectrum; surface calibration and dynamical assumptions required. |
| T9-3d | Placzek depolarization calculation is a property-processing step, not inherently 3 days; approximation validity documented, not universally disproven for floppy complexes. |
| T9-1w | PES band origins alone give no Raman activity: require polarizability-surface transition matrix elements. |
| T9-1mo | Separate resonance-Raman/full property surface scope; neither is a more expensive route to microwave constants. |

### §14.5 — T10-{U}, NMR/UV/MS (3276–3295)

| Rows | Disposition and coding contract |
|---|---|
| T10-10s | Electronic orbital gap is not an optical excitation prediction; diagnostic only. |
| T10-1min; T10-30min | Optional excitation modules with actual state characterization, oscillator strengths and target-domain benchmarks; no universal error bands. |
| T10-1h; T10-3h | Optional shielding/shift module with explicit referencing, solvent, populations and property derivative corrections. |
| T10-12h | Optional supported STEOM state/root/reference profile; large-cost extension. |
| T10-1d; T10-3d | Experimental QCxMS ensembles; fixed 500/2000 trajectories do not prove convergence, require branching ratio uncertainty and channel coverage. Charge/spin channels not fixed blindly. |
| T10-1w | Correct cost arithmetic M-T07; publish actual engine level/timestep/steps/resources and statistical stopping rule. |
| T10-1mo | Separate multi-reference/excited-state scope with concrete state-specific capabilities, not a generic final tier. |

## Non-table section dispositions

| Section | Disposition |
|---|---|
| §15.1 | Keep multidimensional cost/accuracy choice concept; replace absolute dominance with measured, matched-domain recommendation evidence. Rows with different observables cannot be called dominated by each other. |
| §15.2 | Retain assignment versus characterization profiles; avoid a universal CFOUR requirement or absolute measured-parent superiority without parameter-identifiability and domain tests. |
| §15.3 | Retain warning that cost is not monotonic in accuracy; examples illustrate limited cases. |
| §16.1 | Retain complete SCF, grid/orientation, optimization, basis, core and precision diagnostics with fixed units. Comparing energy noise (Eh) directly to gradient tolerance (Eh/bohr) is dimensionally invalid without a displacement scale. Scalar Δr=g/k is only a mode-local harmonic estimate; a full Hessian pseudoinverse and treatment of constrained modes are needed for vector residuals. |
| §16.2 | Retain validation and convergence per observable/domain; implement M-T09 policy and do not require unrelated excited-state/MS benchmarks for the initial microwave-only release. |
| §16.3 | Replace delete/discard wording with quarantine/promotion gate and evidence-retaining, method-specific policies. |
| §17 | Retain six cases as diagnostic seeds, not a complete calibrated corpus; see M-T10. |
| §17.5 | Correct formula and assumptions per M-T01; no nominal coverage assertion until operational calibration. |
| §18 | Retain SPCAT artifact objective plus validated profile and golden fixture; see M-T06. Strongly coupled/tunnelling cases may require a different export Hamiltonian/profile rather than fabricated Watson files. |
| §19.1 | Retain open-source teaching option; cluster size does not remove allocations, queues or concurrency limits. Current resource, pricing and licence terms are external facts requiring verification. No 50-seat guarantee or guaranteed $50/day price. |
| §19.2 | Retain evidence/error-budget lab; repeat timing locally and use prerecorded heavy examples if needed. Signed and absolute errors both carry information. CP geometry change is example-specific, not an expected constant 4 pm. |
| §20.1 | Retain local timing method, but a single point is insufficient to certify optimizations, regions with difficult SCF convergence, full surface campaigns or runtime tails. Separate median, p95 and maximum resource use; record warm/cold caches and failures. |
| §20.2 | Retain provenance requirements with M-T10 additions; model names and capabilities are version-specific and must be verified via artifacts, not prose. |
| §20.3 | Retain prior disclosure and theory/measurement provenance; remove blanket ban on Bayesian experimental inference. Such inference can be legitimate if disclosed and dependencies are handled; it cannot be labeled independent validation. |
| §21.1 | Recast seven universal “hard limits” as present implementation/evidence limits. Published-error examples do not rule out future or model-specific accuracy, all ML ranking, or completeness arguments beyond hand enumeration. |
| §21.2 | Retain open-measurement register; add uncertainty model, corpus independence, orbital-basis versus F12-method distinctions, derivative availability and exported Hamiltonian acceptance. |
| §21.3 | Retain staged benchmark/provenance roadmap; licence purchase and advanced solver/corpus construction are user decisions. No universal Molpro-only geometry ceiling. |
| §22 | Keep historical editorial provenance separate from scientific evidence. Agent conference agreement does not validate findings; linked reports must exist or be marked unavailable. |
| Appendix A.1 | Correct duplicate criteria, thresholds and dark-branch policy per M-T05. |
| Appendix A.2 | Correct half-line DVR matrix, avoid hardware-derived runtime promises, and replace “DMC dimension-agnostic” with a statement that accounts for cost and variance. Ground-state potential calls and walker cost grow with system size. |
| Appendix A.3 | Retain quantum zero-point distinction and rigid-model option. PIGLET/contraction accuracy depends on the observable and system. DMC has finite timestep, population, descendant-weight and fixed-node bias where applicable. A Cartesian implementation may avoid a bespoke kinetic-energy operator, but rigid constraints and curvilinear coordinates do not. |
| Appendix A.4 | Retain explicit rotor Hamiltonian/software profiles; see M-T08. Current appendix ±10% conflicts with table ±14%; neither is a general interval. |
| §23 | 319 numbered reference entries parsed; 319 unique URLs; each URL occurs in the pre-reference body. This verifies internal mechanical consistency only. Metadata, primary-source quality, exact supporting pages/tables, access dates, versions, DOI and quote verification remain needed before publication. |

## Material decisions to present to the user

1. **Release scope**: initially support conventional microwave equilibrium/semirigid spectroscopy and trustworthy export; separate LAM/PIMD/DMC/VRT/Raman/NMR/UV/MS into gated research extensions. Pros: executable, testable initial scope and fewer false scientific guarantees. Cons: the 140-row conceptual matrix is not fully automated at first release. Risks of immediately accepting all scope: multiple unvalidated solvers, incompatible Hamiltonians, unsupported engine features and major compute/licence cost. Safe default: retain all requirements as staged feature contracts, never mark unimplemented rows operational.
2. **Calibration and publication**: build an independent experiment-backed corpus with expert transcription/refits and molecular-family splits. Pros: defensible uncertainty and error claims. Cons: data acquisition/licensing and specialist scientific work. Risks: sparse floppy-molecule data, leaked semi-experimental priors, dependence, mismatched states. Safe default: tag numbers as uncalibrated without coverage claims.
3. **Quantum motion model**: semirigid VPT2, rigid-monomer quantum averaging, or full VRT for LAM. Pros/cons vary in accuracy, cost and interpretation. Risks: a finite-temperature result at 50 K is not automatically B0; omitted coordinates/couplings, extrapolated surface tails and embedding dependence matter. Safe default: reject unsupported B0 and preserve Be plus explicit model limitations.
4. **Licensed engine acquisitions**: Molpro/CFOUR/high-level external adapters may improve derivative availability and composite workflows. Pros: validated analytic paths. Cons: acquisition/distribution/CI constraints and duplicated maintenance. Risks: legal restrictions, version-specific capabilities and assumed rather than measured speedups. Safe default: a capability registry with an unavailable state and supported existing engine paths; no licence purchase implied.
5. **Culling and “dark” filters**: preserving more candidates/weak lines increases cost and output volume, but reduces missed assignments. Irreversible MLFF energy culling, hard dipole cuts or approximate-constant-based species merging can miss real species. Safe default: conservative deduplication, archive all candidates, user-configured view filters and audited pruning.

The definite corrections above can be adopted immediately in the canonical SRS without claiming any new experimental accuracy or computational capability.
