# Scientific acceptance boundaries and remaining SRS work

The canonical student application uses a named, pinned PySCF/geomeTRIC recipe,
real optimization and property calculations, isotope-specific inertia and
projected normal modes. It retains all nine spectroscopy stage names and
preserves valid earlier results when a requested advanced product is blocked.
This supports bounded teaching and numerical validation. It does not establish
complete implementation of the SRS or high-accuracy laboratory/astronomical
identification. `experimental_accuracy_established` and `identification_ready`
remain false.

The normative sources are the [implementation SRS](../wiki/CoChem-TORQ_Implementation_SRS.md),
[method-matrix contract](../wiki/Method_Matrix_Implementation_Contract.md),
[accepted decisions](../wiki/review/Decision_Questions.md) and
[revDSD protocol](../wiki/RevDSD_Spectroscopy_Protocol.md). A passing mathematical
or software test cannot activate an unqualified scientific matrix row.

## What the current numerical evidence establishes

- Cartesian geometry and derivatives use bohr, hartree, hartree/bohr and
  hartree/bohr². Explicit isotope masses determine the COM, principal axes,
  moments and MHz equilibrium constants. Mass weighting uses electron masses;
  complete normal modes retain imaginary and unresolved frequencies.
- The charged-molecule dipole-origin transformation is
  `mu_COM=mu_origin−charge*COM*e`, followed by conversion to debye and projection
  onto the same right-handed principal axes. Charge is in elementary-charge
  units and COM is converted from bohr. A Cartesian component is not silently
  relabeled a principal-axis component.
- The separately named rigid-rotor screening solver calculates actual finite-J
  Hamiltonian eigenvalues and electric-dipole matrix elements, with MHz
  frequencies, debye² strengths, temperature-dependent populations and Einstein
  A in s⁻¹. It excludes centrifugal distortion, hyperfine, torsion and
  nuclear-spin/permutation-symmetry restrictions. Partition truncation and
  absence of identification qualification remain explicit.
- Actual energy displacements produce cubic/quartic normal-coordinate tensors
  and recorded two-scale convergence. Vibrational-only VPT2 and explicit
  second-order resonance polyads have independent mathematical limit checks.
  A genuine HF/STO-3G H2 calculation provides optimization/Hessian/displacement
  wiring evidence, including a correctly blocked conservative perturbative
  applicability gate. This evidence does not supply Coriolis, alpha, distortion
  or a high-accuracy molecular benchmark.
- The explicit local `hf-sto-3g-anharmonic-validation` runner now executes the
  bounded full derivative stencil, independently verifies its reference
  gradient, preserves authentic failed SCFs and source hashes, and returns
  `experimental_unqualified`. Its numerical force-field/resonance stages may
  be available; rotational VPT2/B0/catalog qualification remains blocked.
- BASE/TOPOS handoffs preserve supplied state, isotope and atom identities,
  source method and actual source observations. Canonical student dispatch does
  not invoke the legacy discovery/ML/GOAT paths as an implicit refinement step.
  An exhaustive scientific qualification of every legacy module is a separate
  claim and is not established by import-path inspection.

## Dependencies before high-accuracy identification

| Gap | Required next implementation and evidence | Accepted decision / gate |
| --- | --- | --- |
| Exact open-source revDSD | Reconcile the complete primary-source orbital-generation and energy definitions, P86/LibXC variants, OS/SS scaling, core treatment and D4 variant. Implement the exact identity, independently compare energies to an authenticated native/reference bundle, then separately qualify numerical/orbital-response gradients and higher derivatives. Custom double-hybrid research calculations cannot be renamed revDSD. | D01; TORQ-METHOD-004; V-METHOD, V-DERIV. |
| Full matrix dispatch | Convert each enabled historical row into an unambiguous versioned recipe with exact derivative/property/engine/platform scope and measured evidence. Preserve all 140 documentary IDs, O/C tracks, R1–R9 and independent budgets. HF teaching recipes are additional reference calculations rather than proxies for matrix DFT/double-hybrid/correlated rows. | D02; method-matrix activation checklist; V-METHOD. |
| Complete typed result/provenance service | The implemented geometry, equilibrium-inertia and harmonic payloads have strict scientific schemas and immutable dependency checks. Extend equivalent validation to all advanced force-field/rovibrational payloads, canonical units, isotope/frame identities and dependency digests; separate dipole, distortion and hyperfine properties. Availability envelopes alone cannot qualify an advanced scientific contract. | ARCH-01, ARCH-05; TORQ-SPEC-001; V-STAGES, V-SCHEMA. |
| Semirigid rovibrational VPT2 | Add the selected qualified rotation-vibration/Coriolis and kinetic-coordinate convention, complete required force constants, resonance closure and independently authenticated molecular comparisons. Validate fundamentals, alpha and relevant distortion together; vibrational-only oscillator perturbation is an explicitly narrower implementation. | D05; TORQ-VIB-003; V-VPT2. |
| B0 and composites | Obtain authentic axis-/isotope-matched alpha or independently solved rovibrational corrections. Apply `B0=Be−½Σalpha` with the correct component/degeneracy convention. Mixed-level geometry/force-field recipes must identify both levels and demonstrate transfer applicability. A formula accepting supplied alpha is not an alpha-producing engine. | D04/D05; TORQ-SCI-004; V-VPT2, V-ISO. |
| Effective rotational catalogs | Implement/provision and pin an actual qualified catalog backend or full effective Hamiltonian, reduction/representation and state assignment. Qualify centrifugal distortion, hyperfine where applicable, dipole/intensity definitions, nuclear-spin weights, isotopologues, truncation and frequency range. The teaching rigid rotor remains a separate product. | TORQ-SPEC-002; V-PROP, V-SPCAT, V-CONSUMER. |
| Calibrated identification uncertainty | Freeze a bounded preregistered molecular-family benchmark with authenticated reference values/uncertainties, product A/B/C identity, calibration/training/held-out separation, numerical tolerances and predictive coverage targets. Run independent molecular and transition comparisons; method differences and numerical residuals alone are not confidence intervals. | D10/D11; V-BENCH, V-UQ. |
| LAM and fluxional species | Implement and independently validate the selected reduced/coupled kinetic operator, periodicity, boundary conditions, symmetry sectors, mode accounting and numerical convergence. Preserve semirigid successes and branch incompatible systems rather than force VPT2. | D05; TORQ-VIB-004; V-DVR. |

## Remaining full-SRS program scope

These accepted deliverables still need explicit scientific activation and
consumer evidence beyond the bounded student refinement path:

- Advisory ML/PES acceleration with actual training artifacts, state/domain
  identity, independent quantum checks and search-recall validation before
  pruning (D03). Include documented TOPOS candidate retention and exclusions.
- Separate fully relaxed, frozen-monomer, counterpoise and composite energy
  workflows, with fragment charge/spin definitions, real component calculations
  and explicit `De/D0`/ZPE/free-energy identities (D04).
- Transition-state/IRC/kinetics profiles, multireference diagnostics and bounded
  same-model numerical recovery; changes in state/model remain review events
  (D06), with actual representative failure cases.
- Optional licensed NBO and separately named population/localization/real-space
  analyses, with authentic wavefunction/density interfaces and their own
  convergence/interpretation contracts (D07).
- A complete BASE task-service integration beside the standalone adapter,
  cancellation/restart and coordinator publication recovery, per-deployment
  authority, real end-to-end sibling consumer tests and owned shard/merger
  semantics (D08 and ARCH-07–09). File contracts alone do not certify the whole
  distributed service.
- Per-engine/platform image qualification, numerical/library fingerprints,
  installation/license availability and genuine GitHub-hosted calculation plus
  Codespaces interface journeys. Linux CPU qualification precedes named GPU or
  Slurm releases (D09 and ARCH-10–11).
- Explicit versioned IR/Raman/NMR/UV–visible/MS sibling ownership and consumer
  contracts; geometry or vibrational frequencies alone do not supply their
  required property surfaces, response, excited states or fragmentation (D12).

Every unavailable capability remains visible. Publication-ready or complete-SRS
release labels require the applicable preregistered acceptance evidence, not
the mere presence of files, stage names, historical recipes or passing tests.
