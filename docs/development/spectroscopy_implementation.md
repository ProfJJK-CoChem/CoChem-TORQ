# Numerical spectroscopy implementation

The canonical numerical package is `src/cochem_torq/spectroscopy/`. It performs
actual calculations from supplied, validated quantities. Its advanced
vibrational algorithms are research implementations: independent molecular
reference qualification and complete rovibrational response remain required
before claiming high-accuracy identification. A computed screening line is
never relabeled an SPCAT prediction or a calibrated identification catalog.

## Implemented numerical stages

| Operation | Numerical implementation | Evidence and boundary |
| --- | --- | --- |
| Equilibrium constants | Isotope-specific COM/inertia tensor, right-handed principal axes, `h/(8π²I)` in MHz. Undefined linear-axis rotation is absent. | Independent inertia limits, translation/rotation invariance, isotope mass scaling. Explicit positive masses are supplied; upstream isotope selection records its database identity. |
| Harmonic analysis | Cartesian Hessian in hartree/bohr²; mass weighting in electron masses; SVD translation/rotation projection; symmetric eigensolution; complete signed frequencies. | Diatomic reduced-mass reference, nonlinear 3N−6/linear 3N−5 limits, mode mass normalization and rotated-Hessian covariance. Symmetry and external residuals remain visible. |
| Isotopologues | Recompute COM, external projection, modes and frequencies using the same Born–Oppenheimer Cartesian Hessian and explicit changed isotope masses. | H2/D2 reduced-mass ratio. This approximation excludes isotope-specific electronic corrections. |
| Harmonic ZPE | Half the sum of positive atomic-unit frequencies for a completely stable internal Hessian. | Independent oscillator limit. An imaginary/zero mode blocks the semirigid ZPE product instead of being deleted or floored. |
| Cubic/quartic field | Actual evaluator energies at dimensionless normal-coordinate displacements, centered derivatives through fourth order, complete permutation-symmetric tensors, two displacement scales, fixed convergence gate and task budget. | One- and three-mode polynomial derivatives, full mixed-index factorial multiplicities, genuine HF/STO-3G displaced SCFs. Numerical derivatives are explicitly numerical. |
| Resonance diagnostics | Exact finite harmonic-oscillator polynomial matrix elements; detuning and coupling/detuning criteria; explicit Fermi/Darling–Dennison/general vibrational labels. | A mathematically specified Fermi pair; real H2 field exceeds the chosen conservative coupling gate and is retained while the next stage is blocked. Coriolis resonances are unavailable. |
| Vibrational VPT2 | Quartic first order plus cubic second order, using exact finite sums over oscillator states connected by the cubic operator. | Independent closed-form one-dimensional oscillator shifts and independently diagonalized oscillator Hamiltonian. Rotation-vibration interactions, Coriolis and curvilinear kinetic terms are excluded; `rotation_vibration_available=False`. |
| Explicit polyads | Hermitian second-order Van Vleck effective Hamiltonian with cubic/quartic internal matrix elements and symmetrized external cubic denominators. | Explicit Fermi pair, Hermiticity and orthogonality checks. Omitted resonant external states are rejected; no denominator shift is introduced. |
| Supplied rotation-vibration corrections | `B0(axis)=Be(axis)−Σ degeneracy_i*alpha_i(axis)/2`, with exact geometry/isotope correspondence and external correction provenance required. | Formula/lineage checks only. The package does not invent alpha from its vibrational-only VPT2 result. A real qualified source of alpha remains necessary. |
| Rigid-rotor screening lines | Exact finite-J asymmetric/symmetric/linear Hamiltonian eigensolutions, integer Wigner 3j dipole matrix elements, thermal absorption weights and Einstein A. | J=1 asymmetric eigenvalues, symmetric top limit, linear Hönl–London factors, independent partition sum and dipole-axis sum rule. Excludes distortion, hyperfine, internal rotation and nuclear-spin symmetry restrictions. Identification qualification is explicitly false. |

## Units and mode conventions

Given isotope masses `m` in atomic mass units, the analysis converts to electron
masses and diagonalizes the internal block of `D=M⁻¹/² H M⁻¹/²`. The returned
mass-weighted modes `L` satisfy `L.T @ L=I`; Cartesian modes `M⁻¹/² L` satisfy
`cartesian_modes.T @ M @ cartesian_modes=I`. Coordinates are bohr. Atomic-unit
mode angular frequencies are square roots of the internal eigenvalues. A
negative eigenvalue produces a negative wavenumber, not a frequency substitute.

The dimensionless normal coordinate is `q_i=√ω_i Q_i` and Cartesian displacement
is `M⁻¹/² L diag(1/√ω) q`. Its harmonic Hamiltonian is
`Σ_i ω_i(n_i+1/2)` and its potential is
`V3=Σ_ijk phi_ijk q_i q_j q_k/3!`,
`V4=Σ_ijkl phi_ijkl q_i q_j q_k q_l/4!`, with `phi` in hartree.
These factorials are part of the data convention and may not be omitted when
importing a force field from another program.

## Evaluator and artifact API

```python
from copy import deepcopy
from importlib.metadata import version
from itertools import count
from pathlib import Path

from cochem_torq.engines.pyscf_backend import PySCFBackend
from cochem_torq.spectroscopy import (
    EnergyEvaluation, analyze_hessian, build_force_field,
    equilibrium_rotor, vibrational_vpt2,
)

backend = PySCFBackend()
request = {
    "molecule": {
        "symbols": ["H", "H"], "geometry_bohr": [[-.8, 0, 0], [.8, 0, 0]],
        "charge": 0, "multiplicity": 1,
    },
    "method": {"name": "hf", "basis": "sto-3g", "reference": "restricted"},
    "properties": ["energy", "gradient", "hessian"],
    "settings": {"threads": 1, "scf_energy_tolerance": 1e-13},
}
workspace = Path("h2-spectroscopy-run")
workspace.mkdir()  # Use a new directory; prior results are never overwritten.
optimized = backend.optimize(request, workspace / "optimization")
if optimized["status"] != "complete":
    raise RuntimeError(optimized.get("errors"))
masses_u = [1.00782503223, 1.00782503223]  # explicit 1H isotope masses
harmonic = analyze_hessian(
    optimized["geometry_bohr"], masses_u, optimized["hessian_hartree_bohr2"],
)
rotor = equilibrium_rotor(optimized["geometry_bohr"], masses_u)
displacement_ids = count()

def evaluate_displacement(coordinates_bohr):
    displaced = deepcopy(request)
    displaced["molecule"]["geometry_bohr"] = coordinates_bohr.tolist()
    displaced["properties"] = ["energy"]
    record = backend.evaluate(
        displaced, workspace / f"displacement-{next(displacement_ids):03d}",
    )
    if record["status"] != "complete":
        raise RuntimeError(record.get("errors"))
    return EnergyEvaluation(record["energy_hartree"], record["manifest_sha256"])

field = build_force_field(
    harmonic, evaluate_displacement,
    evaluator_identity=f"PySCF {version('pyscf')}; RHF/STO-3G; experimental",
    steps=(0.08, 0.04),
)
# The conservative applicability gate can block this minimal-basis model.
try:
    vibrational_result = vibrational_vpt2(field)
except ValueError as error:
    print(f"Vibrational correction blocked: {error}; force field retained.")
```

The evaluator is a real computational callback, not an engine substitution.
An ordinary finite scalar is also accepted for explicitly identified
mathematical model potentials; those evaluations have no raw-engine artifact
digest and cannot count as live-engine qualification. Force-field results retain
all displaced dimensionless coordinates, actual energies, geometry digests and
supplied source-result manifest digests. Raw backend inputs/outputs remain in
their immutable calculation workspaces. The harmonic source digest includes
geometry, masses and original Cartesian Hessian; both coarse and fine derivative
tensors and the comparison tolerances are retained.

The default Hessian symmetry gate is a **relative** Frobenius residual of
`1e−8`. An approved numerical-gradient Hessian profile must declare its own
consistent relative tolerance before evaluation; an upstream absolute tolerance
is not the same quantity. Averaging `H` and `H.T` is allowed only below the
declared gate and its measured residual is reported. The external-motion
residual is reported separately, since projection alone cannot certify a
stationary geometry or invariance of an inaccurate Hessian.

## Rotational screening API

```python
from cochem_torq.spectroscopy import rigid_rotor_catalog

# Correct a charged molecule's dipole origin to its COM upstream, then rotate.
dipole_axes_debye = dipole_at_com_cartesian_debye @ rotor.principal_axes_columns
screening = rigid_rotor_catalog(
    rotor.constants_mhz, dipole_axes_debye,
    temperature_kelvin=5.0, J_max=20, constant_observable="Be",
)
```

The identity is `TORQ exact finite-J electric-dipole rigid-rotor screening model
v1`. Quantum numbers are `J` plus sorted Hamiltonian eigenstate index; `Ka/Kc`
assignments are not inferred. Dipole strengths sum over magnetic states and
three laboratory polarizations. Relative absorption weights are the summed
dipole strength times `exp(−E_lower/kT)*(1−exp(−hν/kT))/Q`. They do not include
instrument response, species concentration or a line shape. Einstein A divides
the summed strength by the upper magnetic degeneracy and uses the SI
electric-dipole spontaneous-emission formula. Nuclear-spin weights and
permutation-symmetry restrictions are explicitly excluded.

The partition sum includes the supplied finite-J states. The fraction in the
last two J shells is an explicit convergence indicator, not a rigorous bound on
the omitted infinite tail. Failure at the requested tolerance remains visible.
`J_max` is bounded to 30 for the direct integer Racah implementation. More
extensive angular momentum requires a separately validated stable recurrence.

## Explicit local anharmonic-validation runner

`cochem_torq.research_pipeline.execute_anharmonic_validation(request, harmonic,
backend_request, workspace)` executes the named
`hf-sto-3g-anharmonic-validation` protocol using actual PySCF 2.14.0
RHF/STO-3G calculations. It requires an explicitly requested anharmonic product,
the correct electronic/isotope state and a positive complete internal Hessian.
The scope is one to three modes; a larger mode set fails before displaced work.
Both the CPU/memory budget and a complete derivative stencil count are checked.
The two steps 0.08/0.04 require seven distinct reference/displacement
calculations for one mode, and 107 for three modes, below the declared cap of
200. The first actual reference calculation also verifies its independent
gradient against recorded stationarity criteria.

Each converged SCF result is matched to its actual geometry/method and raw
manifest bytes before being used. Unconverged SCFs retain their authentic logs,
absent accepted energies and manifest hashes. A failed reference gradient,
derivative convergence comparison or perturbative applicability check retains
earlier actual observations and blocks the dependent correction. There is no
calculator substitution and no generated alpha.

The result identity is `experimental_unqualified`, with a separate numerical
`outcome` and force-field/resonance/vibrational-only-VPT2 stages. Raw reference
and displaced workspaces, protocol/input snapshots, progressive checkpoints and
a final research-artifact inventory are retained. Full rotational VPT2, B0 and
identification readiness remain unqualified even if a vibrational-only
calculation succeeds. This local research recipe is distinct from the bounded
canonical GitHub Actions teaching recipe.

`tests/test_anharmonic_research.py` executes the real runner, verifies every
inventory/hash, actual failed-SCF preservation and independent rejection of a
nonstationary reference, and confirms invalid scope rejection before creating
workspaces. These tests provide numerical/process evidence, not independent
high-accuracy molecular spectroscopy qualification.

## Verification and scientific gates

Run in the canonical calculation environment with the real optional PySCF and
geomeTRIC dependencies installed:

```bash
python -m pytest tests/test_spectroscopy_physics.py
```

The observed cloud run passed 18 tests, including genuine RHF/STO-3G H2
optimization, analytic Cartesian Hessian, seven displaced converged SCFs, a
two-scale cubic/quartic field and Hessian/energy-curvature consistency. This is
software/numerical wiring evidence for that bounded recipe. HF/STO-3G is not a
high-accuracy spectroscopy method. Its conservative coupling gate blocks
nonresonant VPT2; the test verifies retention of the valid force field and the
truthful blocked correction rather than widening the criterion to manufacture
success. Engine absence explicitly skips the optional live evidence test and
does not establish release qualification.

The full SRS V-VPT2 gate still requires authenticated independent molecular
force-field comparisons, Coriolis/rotation-vibration interactions, alpha and
distortion validation. V-SPCAT/V-BENCH require a genuine qualified effective
Hamiltonian/backend, statistical conventions, independent measured transitions
and calibrated transition uncertainties. None of these gates follows from
passing the mathematical tests. The exact open-source revDSD energy/derivative
qualification is also independent of this spectroscopy package.
