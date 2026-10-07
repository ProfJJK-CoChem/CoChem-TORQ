# Method matrix implementation contract

Version 1.0; source baseline `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`.

This is the normative companion to the [implementation SRS](CoChem-TORQ_Implementation_SRS.md). The [historical Method Matrix](Method_Matrix.md) is a research/planning source, not executable policy. Its full content and original stable row IDs are preserved. The [complete scientific audit](review/Method_Matrix_Review.md) evaluates every section, all twenty revision claims, recipes R1–R9, and all 140 table rows. This contract incorporates defensible corrections; it does not silently activate unvalidated methods or turn planning estimates into measured performance.

## 1. Identifier and migration rules

Keep five independent identifiers:

1. `legacy_row_id`: exact table/track/budget identity, such as `T3O-12h`.
2. `method_family`: F01–F11 from SRS §5, a classification only.
3. `recipe_id` and `recipe_version`: complete executable scientific model.
4. `product`: A, B, C or BENCHMARK, with observable identity.
5. `budget`: explicit numeric resource ceilings, independent of all the above.

The source contains ten recurring budget labels: `10s`, `1min`, `30min`, `1h`, `3h`, `12h`, `1d`, `3d`, `1w`, `1mo`. The SRS's eleven labels refer to method families, not those budgets. Treat month/week/day labels as legacy text until a plan supplies exact wall seconds and resource allocations. Never invent a one-to-one mapping.

Tables 3, 4, 6 and 8 have separate O/C tracks; tables 1, 2, 5, 7, 9 and 10 do not. Each table/track has ten rows, giving 140 distinct row IDs. Historical aliases such as `T3-12h` or `T4-1d` require explicit track resolution, or fail as ambiguous. They must not silently default to ORCA. Preserve aliases in migration metadata rather than rewriting past provenance.

| Table family | Purpose | Owner and minimum additional contract |
|---|---|---|
| T1 | Conformer/isomer search | TOPOS ownership; TORQ imports candidates with source-specific energy/topology provenance. |
| T2 | Intermolecular PES and nuclear-motion preparation | Coordinates, state-consistent energies, coverage, surrogate validity, kinetic operator and solver convergence. |
| T3O/T3C | Equilibrium structure and Be | Full optimization method, derivative route, constraints/frozen monomers, stationarity and isotope masses. |
| T4O/T4C | Vibrational corrections and B0 | Force-field order, nuclear-motion solver, resonances, BO/isotope assumptions and coupled-mode validity. |
| T5 | Energetics | Energy reference, CP/deformation, composite components, De/D0/free-energy distinction. |
| T6O/T6C | Dipoles, quadrupole coupling, distortion and related observables | Isotope spin/quadrupole moment, response/force-field method, frame/units and tensor/Hamiltonian conventions. |
| T7 | Large-amplitude/internal rotation | Explicit torsion-rotation model, periodicity, symmetry sectors, coupling and level/spectrum convergence. |
| T8O/T8C | Infrared, terahertz and far-infrared | Exact vibrational model, dipole derivatives/surface, resonance treatment, intensities and supported solver profile. |
| T9 | Raman | Polarizability derivatives/surface, laser/scattering conventions, vibrational states and explicit non-microwave ownership decision. |
| T10 | NMR, ultraviolet–visible and mass spectrometry | Explicit property/state/reference/fragmentation model and non-microwave ownership decision; no blanket +1/doublet rule. Dynamics used within these rows needs its own integrator/ensemble/convergence contract. |

For exact source table titles and each row's disposition, use the audit. Names above group contracts; they do not rename source IDs or promise implementation of adjacent CoChem modules.

## 2. Recipe schema and activation gate

Every executable recipe must specify:

| Field group | Required content |
|---|---|
| Identity | Recipe version/digest, original row/recipe references, full scientific citation, method family, intended product/domain. |
| Electronic model | Exact method and orbital-generation model; charge/spin/reference; dispersion/nonlocal correlation; frozen core; relativistic/ECP treatment. |
| Basis | Orbital, auxiliary, fitting and CABS definitions/checksums; spherical/cartesian conventions; F12 particulars. |
| Calculation roles | Distinct geometry, electronic-energy, force-field and property methods; component equations for composites. |
| Numerical profile | SCF/optimization/integration/derivative settings; analytic versus finite-difference route; constraints; restart compatibility. |
| Nuclear motion | Harmonic, VPT2, rotor, DVR/VRT or other explicitly validated model; mass/frame conventions and applicability. |
| Engine | Exact release/build, adapter version, platform libraries, license/provisioning requirements and capability tuple. |
| Planning | Estimated job counts, per-task latency/resource ranges and total CPU/GPU/scratch use, provenance and pilot measurements. |
| Evidence | Unit/kernel tests, authentic integration runs, scientific benchmark domain/metrics, limitations and independent references. |

Activation states are `draft`, `documented`, `experimental`, `locally_validated` and `retired`. Unknown is distinct from unsupported. A documented recipe may be run as a labeled validation experiment; automated production dispatch requires locally validated scope. A source `[M]` flag never satisfies this gate by itself.

### Engine evidence available in this review

| Engine/feature | What is established | Implementation consequence |
|---|---|---|
| GPU4PySCF | Upstream README distinguishes HF/DFT derivatives, experimental correlated methods and unsupported double hybrids; docs drift across versions. | Certify a pinned method/derivative/profile. Do not use GPU presence as a revDSD or all-methods enable switch. |
| MPQC CCSD/CCSD(T)-F12 | Inspected upstream CCSD energy capability accepts derivative order zero; F12 energy machinery does not establish analytic CC Hessians. | Remove mandatory MPQC→ORCA analytic-CC-Hessian routing. Use verified derivatives or an explicitly budgeted numerical route. |
| ORCA | Source documents contain conflicting derivative assertions; retrieved external-tool contracts are not certification of all methods. | Validate canonical/local triples, references and requested derivatives separately against selected release documentation and real runs. |
| CFOUR | QCEngine maps Hessian requests to `VIBRATION=EXACT`; full vendor manual was not independently accessible during this review. | Do not extrapolate wrapper method recognition to every Hessian/VPT2/F12/property combination. Document exact supported tuples before activation. |
| JANPA | Official documentation supports NAO/NPA and localized-orbital analyses; its E(2) page says the proposed feature was not implemented at least in v2.02. | Retain actual analysis identities. NBO and donor–acceptor calculations need their own validated implementation. |

See [source evidence](review/Scientific_References.md) for immutable upstream revisions and limits. None of these observations is a live TORQ engine integration test.

## 3. Corrections to scientific calculations and claims

### 3.1 Rotational averaging and semi-experimental quantities

Use matrix inversion, not elementwise inversion, in an inverse-inertia model. Transform to a consistent body frame before averaging:

\[
\mathbf B^{Hz}_{geom}=\frac{h}{8\pi^2}\langle\mathbf I^{-1}\rangle.
\]

This is an explicitly specified geometric/kinetic approximation. It is not generally the complete effective rovibrational Hamiltonian of a floppy molecule. Do not invert the average inverse inertia again and report the result as a frequency. For a scalar toy model \(I=\mu R^2\) with mean displacement zero and variance \(\sigma_R^2\), the second-order relative difference between \(\langle1/I\rangle\) and \(1/\langle I\rangle\) is approximately \(4\sigma_R^2/R_0^2\); the 3 coefficient applies to \(\langle1/I\rangle\) relative to \(1/(\mu R_0^2)\), a different comparison.

Use one sign convention throughout: \(\Delta B_{vib}=B_0-B_e\), so \(B_e^{SE}=B_0^{exp}-\Delta B_{vib}^{calc}\). A harmonic Hessian cannot supply general VPT2 alpha constants. Reusing Cartesian force constants across isotopes requires actual mass substitution and mode/axis reanalysis; copying a file under a new name changes nothing. Degenerate modes require the appropriate degeneracy factors or explicit component enumeration.

A planar equilibrium structure has geometric inertial-defect constraints, but experimental ground-state inertial defects include vibrational/rovibrational contributions and can have different signs. A sign discrepancy is a diagnostic, not sufficient evidence to reject a conformer. A conformer mixture normally produces a sum of spectra, not a spectrum formed by averaging its constants. An exchange-averaged spectrum requires a separate validated dynamical model.

### 3.2 Uncertainty calibration

For positive constants and the source's relative residual score

\[
s_i=|B_i^{pred}-B_i^{ref}|/B_i^{ref},
\]

a calibrated threshold \(0\le q<1\) implies

\[
B^{ref}\in[B^{pred}/(1+q),\ B^{pred}/(1-q)].
\]

It does not imply \(B^{pred}(1\pm q)\). At \(q\ge1\), this score does not provide a finite upper bound. A log-ratio score or prediction-normalized residual is a different valid design and must be fixed before calibration.

Split-conformal order statistic rank is \(k=\lceil(n+1)(1-\alpha)\rceil\); if \(k>n\), do not clip it to manufacture coverage. State the resulting unbounded/unsupported interval. Exchangeability, calibration independence, molecule-family grouping and chemical-domain assumptions are mandatory. Marginal coverage is not conditional per-molecule accuracy or simultaneous coverage of A/B/C and every transition; use an explicitly defined family score/multiplicity policy where that is required.

Energy benchmarks cannot calibrate rotational constants; equilibrium-geometry statistics cannot validate interaction-energy errors; related isotopologues are not automatically independent calibration samples. MAE, RMSE and observed maxima are separate statistics, not interchangeable prediction-interval half-widths. Uncalibrated output must say so.

### 3.3 Nuclear-motion models

The historical half-line radial DVR formula is corrected for a reduced radial sine/sinc basis with \(r_i=i\Delta r\), positive integers \(i\), Dirichlet origin and prefactor \(\hbar^2/(2\mu\Delta r^2)\):

\[
T_{ii}=\pi^2/3-1/(2i^2),\qquad
T_{ij}=2(-1)^{i-j}\left[\frac1{(i-j)^2}-\frac1{(i+j)^2}\right]\ (i\ne j).
\]

This is one specified basis/operator, not a replacement for a general coupled Jacobi or torsion-rotation Hamiltonian. Coordinate measure, boundary conditions, angular momenta/ħ convention and coupling terms must be explicit. Require Hermiticity, analytic limiting cases, grid/box convergence and a reference coupled-coordinate test before enabling scientific claims.

Classical low-temperature MD is a diagnostic and does not bound the quantum zero-point correction. Classical harmonic mean total energy is \(k_BT\), with half kinetic and half potential, not \(k_BT/2\) total. PIMD at a finite temperature is not automatically a zero-temperature observable, a jet population, a tunneling splitting or real-time quantum spectrum. Converge beads, timestep, equilibration, statistical error and temperature regime separately. An exponential Boltzmann/Arrhenius factor without a prefactor has no rate units.

A reduced internal-rotation parameter such as \(s=4V_3/(9F)\) requires V3 and F in the same energy/frequency units. A continuous constrained path can overestimate the minimum-energy-path barrier; a discontinuous relaxed envelope need not define a physical path or a useful bound. No universal factor-of-three splitting error or exact DVR accuracy follows from a literature example.

### 3.4 Electronic methods and physical guards

- A double hybrid includes exact method-dependent orbital/semilocal/OS/SS/dispersion definitions and response terms; a sum of ordinary HF, MP2 and D4 is insufficient.
- Named `-3c` recipes and CC/F12/PNO variants cannot be reconstructed from an approximate label. Do not transfer ORCA/DLPNO literature timings or thresholds to MPQC under the same name.
- Core-correlation error must be isolated at a controlled basis/reference; comparing ae/cc-pCVQZ to fc/cc-pVQZ changes both basis and correlation treatment.
- CP and frozen-monomer models are explicit recipe choices. There is no universal nonaugmented-TZ accuracy ceiling or universal optimal CP rule. D3/D4 may include many-body ATM terms; describing them as exclusively pairwise is inaccurate.
- Exact initial Hessians are allowed when useful and budgeted; an economic preference for model Hessians is not a physical prohibition.
- T1/D1, spin contamination, soft modes, ML variance, dipole magnitude and QTAIM descriptors are scoped diagnostics. Near-zero denominators require absolute-plus-relative policies. Missing evidence cannot pass a guard.
- A 0.1 D display/detection threshold cannot delete nonzero archival transitions; the source itself discusses weaker observed components.

## 4. Numerical, resource and software corrections

Counts depend on actual stencil, symmetry and derivative support. Do not multiply a specific N=10 cost ratio into all systems. A pilot single point is insufficient to price an entire optimization/anharmonic workflow without derivative and convergence estimates. Include startup, training, validation, file I/O and contention in campaign cost. Sum allocated and consumed CPU core-hours separately; GPU-hours are not CPU core-hours.

For the source's simple illustration, 500 single-core points ×15 min =125 core-hours; 16 ideal concurrent workers yield 7.8125 hours before overhead, not 10.4 hours. A 128-core ×168-hour budget is 21,504 allocated core-hours, not evidence that a specified trajectory workload consumes exactly that amount. Published point-count reductions and crossover estimates are hypotheses for local measurement.

One GPU with several MPS clients is still one physical CUDA device. Executor device-ID arrays must match actual devices, with concurrency represented separately. MPS, MIG, power-mode changes and driver configuration are deployment-specific capabilities; worker jobs cannot mutate them globally as a default. CPU/GPU placement, parallel decomposability, critical-path seriality and restartability are independent axes.

HDF5 uses one writer or separately merged shards; SWMR is not multiwriter access. Include scientific attributes such as units and model identity in manifest verification. Restart artifacts must be checked for geometry/state/method/engine compatibility. Preserve raw scientific failures even when excluded from downstream promotion.

SPCAT quantum-number fields must follow the pinned format: lowercase `a0` represents −10, `a9` −19 and `z9` −269; uppercase `A0` represents 100 and `Z9` 359. A nominal two-character field is not a general decimal integer. The six upper-state fields occupy columns 56–67 and lower-state fields 68–79 in the reviewed 79-column catalog convention. Use authentic round-trip cases, not a loose whitespace parser or guessed lowercase encoding. The exact file/Hamiltonian contract remains versioned.

## 5. Preserved recipes and activation decisions

R1–R9 and all T-row recipes remain discoverable in the historical source and audit. They are not automatically enabled or deleted. For each one, the implementation team must register exact inputs, output quantity, model/engine capability, numerical tolerances, licensed dependencies and a validation manifest. The audit's row-specific findings take precedence over contradictory narrative examples.

Frozen-monomer refinement, half-CP/composites, open-source double hybrids, active/delta learning, high-dimensional VRT/PIMD, automatic recovery, hardware concurrency and adjacent-spectrum scope have significant tradeoffs. See [D01–D12](review/Decisions_and_Risks.md). Until selected and validated, the planner must expose their unavailability or experimental status and offer supported alternatives with honest labels.

## 6. Matrix acceptance checklist

Before a row is presented as executable:

1. Its unambiguous historical ID and complete modern recipe are recorded.
2. Requested quantity distinguishes Be, B0, energy, correction, tensor or diagnostic.
3. Every method/property/derivative is supported by the pinned engine profile.
4. Source capabilities and literature statistics have evidence status and domain.
5. Actual resource estimates fit an approved numeric budget.
6. Missingness, failure, cancellation, restart and artifact compatibility are tested.
7. Real physical output, independent numerical checks and applicable benchmark gates pass.
8. The release capability ledger and documentation reflect the measured result.

This checklist activates only the tested scope; it does not validate a neighboring row by analogy.
