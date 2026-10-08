# Fragment energetics and saddle candidate protocol

These are explicit numerical research/teaching profiles. Software qualification
on the molecules below establishes reproducible component calculations and
derivative conventions. It does not establish transferable experimental binding,
thermochemical or spectroscopic accuracy.

## Required state and geometry contracts

`energetics.InteractionRequest` requires a molecule with persistent atom IDs,
an exact method/basis definition, and disjoint fragments that partition every
atom. Each fragment has an explicit integer charge and multiplicity. Fragment
charges sum to the complex charge. The implemented restricted CPU profiles
require a singlet complex and singlet fragments with valid positive even
electron counts. An unpaired atom is not silently assigned a singlet.

Coordinates use bohr, energies use hartree, and all Cartesian/basis-center
gradients use hartree/bohr. Nuclear isotope masses are resolved from the actual
Mendeleev database, with database and distribution digests retained when
harmonic/thermochemical products are calculated.

| Geometry protocol | Actual operations | Scientific quantity |
| --- | --- | --- |
| `supplied_frozen_monomers` | Complex single point and each isolated monomer at the supplied complex geometry | Uncorrected interaction energy; no De, D0 or geometry-relaxation claim |
| `counterpoise_single_point` | Those single points plus each physical monomer in the full complex ghost basis | Boys–Bernardi CP interaction energy and BSSE increment at the supplied geometry; no CP-optimized geometry claim |
| `fully_relaxed` | Independent complex/monomer geomeTRIC optimizations, actual final gradients, SCF stability and projected Hessians | Local-minimum uncorrected binding/De, deformation and matched-isotopologue harmonic D0; optional explicit RRHO corrections |

Fully relaxed De/D0/RRHO currently requires restricted HF with an analytic
Hessian. DFT/MP2 numerical endpoint-Hessian profiles require separate
qualification. A supplied "relaxed" geometry never certifies an endpoint.
Local optimization does not prove the global minimum or preserved chemical
fragment connectivity. The result retains those limitations explicitly.

## Energy definitions

Let the fragments be indexed by `i`, `R_i` their internal geometry in the
complex, `R_i*` their independently optimized reference geometry, and `AB` the
full complex Gaussian basis.

```text
Eint(raw) = Ecomplex(AB) - sum_i Ei(own basis; Ri)
Eint(CP)  = Ecomplex(AB) - sum_i Ei(full AB ghost basis; Ri)
BSSE     = sum_i [Ei(own basis; Ri) - Ei(full AB ghost basis; Ri)]
Eint(CP) = Eint(raw) + BSSE
Edeform  = sum_i [Ei(own basis; Ri) - Ei(own basis; Ri*)]
Ebind    = Eint(raw) + Edeform = Ecomplex* - sum_i Ei*
De       = -Ebind
D0(harmonic) = De + sum_i ZPEi* - ZPEcomplex*
```

`De` is positive for binding relative to the selected relaxed fragment states.
It is distinct from a raw interaction energy and from a free energy. D0 uses
actual matched endpoint Hessians and explicit isotope masses. Harmonic D0 is
not mislabeled an anharmonic or experimentally calibrated dissociation energy.
No ZPE, imaginary mode, missing frequency or missing component is fabricated.

## Ghost centers and DFT grids

Every ghost center retains the complex Gaussian shells, exponents,
contractions, angular convention and coordinates while carrying zero nuclear
charge. Actual retained checkpoints establish identical full-complex versus
ghost-basis shell signatures. Analytic gradients include derivatives with
respect to every physical and ghost basis center; omitting a ghost derivative
would produce an inconsistent CP force.

PySCF 2.14's default ghost DFT energy-radius adjustment and moving-grid response
use inconsistent ghost radius descriptors. This adapter explicitly uses the
same full-complex atom-centered Becke/Treutler partition for both monomer energy
and its moving-grid response, without density pruning. The partition uses the
complex element descriptors solely for quadrature. The physical monomer
Hamiltonian and AO evaluations retain the actual zero-charge ghost molecule.
The actual quadrature arrays and grid definition are retained in raw artifacts.
Independent finite differences qualify this derivative definition.

D4(BJ)-EEQ-ATM is calculated separately on physical fragment atoms with the
explicit fragment charge and actual library-resolved damping parameters.
Ghosts are never dispersion atoms. Intrafragment D4 is counted once and cancels
from the native-minus-ghost BSSE increment; the full interaction contains the
actual complex-minus-fragment D4 contribution. Functional alias, resolved
LibXC components, versions and dispersion parameters remain explicit.

## Thermochemistry bounds

Optional RRHO results use the ideal-gas standard pressure, explicit temperature
and rotational symmetry number for every species. Translational, classical
rigid-rotor and actual harmonic-oscillator terms are separately defined, with
ZPE included exactly once. Low positive modes are retained and flagged for
uncalibrated entropy; they are not replaced by invented cutoff frequencies.
Electronic excited states and nuclear-spin statistical weights are omitted
and reported as model limitations. Classical rotation and semirigid RRHO need
domain validation for low-temperature or floppy-species use.

## Saddle candidate and diagnostic bounds

`engines.pathway.locate_saddle_candidate(PathwayRequest, workspace)` executes a
bounded actual restricted-HF/geomeTRIC transition optimization, retains the
actual initial analytic Hessian, and independently recalculates the final
gradient/Hessian/SCF stability. A first-order saddle candidate must have exactly
one resolved projected negative mode and pass stationarity/invariance gates.
Optional positive/negative imaginary-mode displacements are each relaxed with
the same real method to lower-energy, Hessian-characterized minima.

Those displaced minimizations are explicitly **not IRC trajectories**. The
result cannot set `ts_verified=true`; chemical mode relevance, mapped
structural/electronic endpoint identity and a qualified both-direction IRC or
an explicitly accepted connectivity alternative remain required. No arbitrary
spin, active-space or method change is an automatic rescue.

Checkpoint diagnostics authenticate raw manifests and actual checkpoint
geometry/energy before reporting occupied/virtual energies, occupations, the
orbital gap and the exact restricted closed-shell spin identity. Restricted
integer occupations and singlet spin do not establish single-reference
sufficiency. T1/D1 and correlated natural occupations remain explicitly missing
without actual correlated-amplitude/density calculations and conventions.

## Actual qualification tests

- HF/cc-pVDZ He2: native independent CP energy contractions, genuine nonzero
  ghost-center response and two-scale CP energy finite differences.
- MP2/cc-pVDZ He2: independent native MP2 ghost energy and two-scale CP
  gradient/energy consistency.
- PBE-D4 and B3LYP-D4/def2-SVP two-H2 fragments: physical-only fragment D4,
  exact ghost-shell identity and two-scale CP gradient/energy consistency.
- HF/STO-3G water dimer: genuine complex and monomer optimization/minimum gates,
  deformation/De/harmonic-D0 identities, isotope provenance, RRHO pressure
  scaling and actual `G=H−TS` consistency.
- HF/STO-3G NH3: an actual planar umbrella saddle candidate, independently
  characterized imaginary mode and two lower-energy pyramidal minima, with
  truthful IRC/connectivity limits.
- Retained artifact and checkpoint corruption causes explicit diagnostic
  rejection. Legacy ORCA CP optimization cannot silently remove ghost centers.

The mandatory files are `tests/test_energetics_integration.py`,
`tests/test_pathway_diagnostics_integration.py` and
`tests/torq/test_counterpoise_frozen_monomer.py`. Numerical/software acceptance
is distinct from a preregistered independent experimental accuracy campaign.
