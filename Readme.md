# CoChem-TORQ

**CoChem-TORQ** is the Quantum Resonance and Anharmonicity Engine of the CoChem suite.

It resolves whether a molecular complex should be treated classically or quantum-mechanically:
- **Rigid Rotor Pathway:** If the complex is strongly bound, TORQ executes VPT2 analysis to extract Darling-Dennison resonances, Coriolis coupling matrices, and quartic centrifugal distortion constants.
- **Large Amplitude Motion (LAM) Pathway:** If the `LAM_TRIGGER_REQUIRED` flag is detected, TORQ generates a highly discretized multidimensional mesh mapping the fluxional coordinates. It utilizes the **Colbert-Miller Sinc-DVR** Hamiltonian to solve for the true 3D probability wavefunctions and tunneling splittings.
- **Constrained Monomer Relaxation:** To avoid rigid monomer approximations during the 3D DVR mesh scans, TORQ executes constrained ORCA relaxations at every point, allowing covalent frameworks to deform dynamically.
- **Adaptive Pruning:** TORQ intercepts repulsive grid points (>50 kcal/mol) using MACE single points, skipping the expensive DFT relaxations to exponentially save compute time.

## Usage
Please refer to the authoritative [CoChem Master User Manual](../CoChem-BASE/CoChem_Master_User_Manual.md) for full execution instructions across the entire 5-module pipeline.