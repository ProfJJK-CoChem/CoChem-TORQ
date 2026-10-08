# Explicit unrestricted electronic-state validation

`OpenShellPySCFBackend` executes genuine PySCF 2.14.0 UHF or UKS calculations,
using LibXC 7.0.0 for the named PBE/B3LYP functionals and genuine DFTD4 3.7.0
for explicitly requested D4(BJ)-EEQ-ATM dispersion. There is no spin projection,
synthetic singlet, substituted method, fabricated derivative or automatic
application of orbitals returned by a stability diagnostic.

The requested molecular multiplicity fixes the native alpha/beta electron-count
sector, `2 M_S = multiplicity - 1`. An unrestricted determinant generally is
not an eigenfunction of `S²`. Therefore preservation of that sector establishes
neither an exact total-spin state nor an electronic term, orbital branch or
lowest-energy state. Every converged native observation retains the actual
`<S²>/ħ²`, its difference from the requested `S(S+1)`, and the effective
`sqrt(1 + 4<S²>/ħ²)` multiplicity. The effective value is explicitly not a state
label. No Yamaguchi correction or other projected energy is supplied.

## Bounded implemented scope

- Canonical element symbols H through Ne, 1–12 atoms, finite real coordinates,
  explicit integer charge from −2 through +2 and multiplicity from 2 through 7.
  Electron-count positivity, spin magnitude and parity are independently checked.
- Explicit unrestricted reference with UHF, UKS-PBE or UKS-B3LYP and an actual
  named all-electron PySCF basis. The initial application profiles fix STO-3G;
  they are validation experiments, not method-matrix high-accuracy substitutes.
- Genuine total energy and analytic nuclear gradients. UKS gradients include
  moving-grid response. Requested D4 energy and gradients are included exactly
  once using the native wrapper and separately retained genuine D4 observations.
- Genuine geomeTRIC 1.1.1 optimization followed by a fresh unrestricted final
  calculation. Native optimizer convergence, an independent final gradient and
  requested stability gates must all pass before an optimized geometry is
  available. Such a geometry remains unclassified and its accuracy uncalibrated.

Density fitting, ECPs, frozen-core treatments, solvation, relativistic models,
broken-symmetry singlet preparation, complex orbitals, spin projection and
unrestricted response spectroscopy are outside this adapter's declared scope.
Unsupported requests fail before an engine workspace or physical value exists.
Numerical controls, threads, memory, SCF cycles, grid level and optimizer steps
have explicit finite bounds. The approved application owns the process wall
limit and OS resource/campaign fencing.

## Spin and reference stability are separate observations

The adapter invokes the actual native internal and external stability routines
separately. The returned internal flag refers to real unrestricted alpha/beta
orbital variations; the returned external flag refers to the native UHF/UKS to
GHF/GKS route. Native real-to-complex diagnostics are retained in the log but are
not independently extracted or certified. This limited scope is explicit in
the result and typed schema.

An unstable or unavailable requested stability diagnostic makes the native
result partial. A genuine converged unprojected energy and gradient remain
available as observations of the retained determinant. The application stops
before geometry optimization when its requested reference stability gate fails.
It never silently selects the candidate orbitals or a different spin sector.
Nonconverged SCF results retain authentic logs/checkpoints and rejected raw SCF
metadata; accepted electronic energies and gradients remain absent.

Stability does not establish branch continuity along an optimization, a global
minimum, chemical accuracy or spin purity. A fresh final SCF calculation can find
another determinant in the same sector; branch continuity remains explicitly
unqualified even after numerical stationarity passes.

## Typed application and immutable artifact contracts

Three separately named profiles are enabled only for explicit local validation:

| Profile | Exact reference | Declared product |
| --- | --- | --- |
| `uhf-sto-3g-open-shell-validation` | UHF | Unclassified optimized geometry |
| `uks-pbe-d4-sto-3g-open-shell-validation` | UKS-PBE-D4 | Unclassified optimized geometry |
| `uks-b3lyp-d4-sto-3g-open-shell-validation` | UKS-B3LYP-D4 | Unclassified optimized geometry |

They declare `ui_visible=False`, do not run in the public classroom Actions
workflow and do not request restricted dipoles, Hessians, rotational constants
or catalogs. Exact capability resolution includes the unrestricted reference,
method, basis, engine/version, property/derivative, recipe digest and hardware.
An unrestricted tuple cannot resolve to a nearby restricted implementation.

`open_shell_electronic_energy` is a separately typed observable containing the
actual determinant energy, molecular coordinates/state, electron counts, spin
expectation, stability flags and native/source/installation hashes. Restricted
`ElectronicEnergy` and `StabilityData` acceptance rules remain unchanged.
`open_shell_optimized_geometry` requires the actual final gradient and optimizer
gates, preserves atomic identities and explicitly states `minimum_established=False`.

The plan first requests a genuine initial state observation, then conditionally
optimizes that exact unrestricted model/sector. Genuine initial observations
remain available when a later stage fails; they retain their own original
geometry and manifest. They are never relabeled as optimized observations.
Every calculation preserves the named basis definition, input, native SCF and
stability logs, actual wavefunction checkpoint, LibXC components/references and,
when applicable, D4 parameter-table and implementation fingerprints. Every
optimizer evaluation retains its native checkpoint and actual energy, gradient,
coordinates, occupancies and spin diagnostics. Hash checks ensure the initial
artifacts are unchanged by optimization.

## Verification and remaining scope

`tests/test_open_shell_engine.py` uses actual OH-doublet UHF calculations,
actual O₂-triplet PBE-D4/B3LYP-D4 calculations, native checkpoint readback,
complete Cartesian energy differences at two displacement scales for the UHF
gradient, two-scale DFT bond-gradient comparisons, genuine external instability,
genuine SCF nonconvergence and genuine optimization checkpoints. The service
tests validate exact preflight/plan routing, typed state integrity, retention of
an unstable UKS observation and approved immutable-shard execution. They do not
use mocked engines or manufactured physical results.

Unrestricted Hessians remain typed unsupported in this adapter. A separate
genuine OH UHF feasibility experiment compared the native analytic Hessian with
full gradient-difference Hessians at 0.002 and 0.001 bohr, yielding maximum
differences approximately 3.01×10⁻⁶ and 7.52×10⁻⁷ hartree/bohr². This bounded
feasibility observation is not a registered, retained adapter derivative gate
and does not establish unrestricted harmonic spectroscopy. UKS Hessians and
subsequent vibration-rotation, VPT2, ground-state constants and identification
require their own real derivative, state-continuity and independent accuracy
validation milestones.

These small-basis observations establish tested software/native derivative
behavior for the stated examples. They supply no calibrated experimental error,
publication-qualified chemical accuracy or astronomical/laboratory identification
claim. Exact published revDSD development remains a separate blocked scientific
milestone; UHF/UKS observations do not qualify that method.
