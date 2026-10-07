# Real CPU calculation adapter

`PySCFBackend` uses an installed PySCF engine; `optimize` additionally uses
geomeTRIC. DFT dispersion profiles use the separately installed DFTD4 package.
Imports do not launch calculations. The application's registry selects the
qualified method/basis/operation tuple before execution.

| Explicit recipe | Energy | Gradient | Cartesian Hessian | Dipole |
| --- | --- | --- | --- | --- |
| Restricted all-electron HF | SCF | Analytic | Analytic | Total density expectation |
| PBE-D4 or B3LYP-D4 | RKS plus D4(BJ)-EEQ-ATM | Analytic, full moving-grid response plus D4 | Numerical derivative of the complete gradient | Total density expectation |
| Canonical restricted all-electron MP2 | RHF plus unscaled OS/SS MP2 | Analytic | Numerical derivative of the complete gradient | Unavailable |

Closed-shell singlets and explicitly named orbital bases are required.
Frozen-core correlation, density fitting, ECPs, solvent models, open-shell
references, D3, arbitrary custom functionals, and unsupported properties are
rejected. Bare PBE/B3LYP may be run as explicitly flagged validation
experiments; they do not satisfy a dispersion-bearing production recipe.
The adapter does not infer high accuracy from a method name.

```python
from pathlib import Path
from cochem_torq.engines import PySCFBackend

request = {
    "molecule": {
        "symbols": ["O", "H", "H"],
        "geometry_bohr": [[0, 0, 0], [0, 0, 2.15], [1.9, 0, -0.5]],
        "charge": 0,
        "multiplicity": 1,
    },
    "method": {
        "name": "hf", "basis": "sto-3g", "reference": "restricted",
        "frozen_core": False,
    },
    "properties": ["energy", "gradient", "hessian", "dipole"],
    "settings": {"threads": 1},
}
result = PySCFBackend().optimize(request, Path("water-attempt"))
```

An attempt directory must be empty. The call writes native logs and
wavefunction checkpoints, canonical `.npy` tensors, effective input settings,
optimization trajectory, parsed results, and a SHA-256 artifact manifest.
All Hessian displacements and failed attempts remain available. The optimizer
must report actual convergence and pass an independent final gradient check;
the spectroscopy analyzer then separately classifies the stationary point.
SCF internal and external restricted-reference stability is checked by
default. An unstable reference is retained as a partial result, with no
promotion to an accepted optimized structure.

Coordinates use bohr; energy uses hartree; gradients use hartree/bohr;
unweighted Cartesian Hessians use hartree/bohr² in atom-major xyz order.
The gradient is the energy derivative; a force is its negative. Dipoles use
debye and the input Cartesian frame, with origin at zero. For charged species,
translation changes the total dipole by the net-charge origin term. Isotope
labels are retained; the Born–Oppenheimer electronic potential does not depend
on nuclear masses. Isotope masses belong to downstream nuclear-motion
analysis.

The numerical Hessian uses centered differences at 0.002 and 0.001 bohr by
default. It returns the finer-step tensor without symmetrizing it, records the
step difference and both absolute and relative symmetry residuals, and
rejects values outside the declared gates. The spectroscopy analyzer must
use a separately declared numerical symmetry tolerance. Current PySCF GGA
analytic Hessians did not pass the complete finite-grid derivative checks
in this qualification, so this adapter advertises a numerical DFT route.

`tests/test_pyscf_integration.py` performs genuine calculations, independent
energy/gradient/Hessian checks, non-equilibrium water optimization, isotope
remassing, charged-origin checks, actual SCF and optimization failures, and
restricted-reference instability. Tested tuples include HF/STO-3G,
HF/cc-pVDZ, PBE-D4/def2-SVP, B3LYP-D4/def2-SVP and MP2/cc-pVDZ. Their
numerical evidence is bounded to the actual small systems tested; publication
accuracy and a transferable spectroscopy error model require additional
independent benchmark evidence.

Concurrent calls require separate worker processes: PySCF thread/scratch
settings and Python log redirection are process-local global state.
