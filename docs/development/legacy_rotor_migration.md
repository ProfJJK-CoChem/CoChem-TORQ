# Legacy asymmetric-rotor correction

`Libraries.cochem_torq_asymmetric_rotor` retains a numerical rotor Hamiltonian
kernel. It does not establish molecular spectroscopic accuracy or identification
eligibility. Several previous outputs lacked the scientific definitions needed
to support those claims:

- The default dipole was one Debye on the a axis, even when no dipole calculation
  or measurement had been supplied.
- Ascending eigenvalues were assigned Ka/Kc labels through a heuristic rather
  than a verified quantum-state assignment.
- The constructor accepted Watson S reduction while building an A-reduced matrix.
- Transition intensities used an arbitrary frequency normalization and numerical
  floors; small nonzero dipoles were dropped.

The corrected constants contract preserves absent dipoles as `None`. All three
components must be supplied explicitly before a catalog request can be assessed.
The implemented numerical matrix accepts only Watson A reduction. All zero
distortion parameters explicitly select a rigid approximation; they do not
establish that physical centrifugal distortion vanishes.

`solve_energy_levels()` now returns `(J, sorted_eigenstate_index)` keys and energies
in MHz. This replaces the previous `(J, Ka, Kc)` keys. The index identifies the
numerical eigenstate within that J block; Ka/Kc remains unassigned. Independent
Hamiltonian-convention validation and the applicable J range are still required
for a molecular quartic-distortion profile.

The legacy transition and Parquet catalog methods raise explicit errors before
creating output directories or files. Supplying genuine dipoles does not qualify
the missing transition matrix elements, state assignments or intensity convention.
For the separately named rigid approximation, call
`cochem_torq.spectroscopy.rotational.rigid_rotor_catalog` with actual constants and
principal-axis dipoles. That API labels its finite-J screening model, line-strength
and population conventions, omitted distortion/hyperfine/spin-statistics terms,
partition-convergence indicator and unqualified identification status.

`tests/test_asymmetric_rotor.py` checks independent analytic J=1 asymmetric-top,
prolate symmetric-top and spherical-top limits, real finite-valued input contracts,
explicit missingness and rejection without output mutation. These mathematical
and process checks do not establish agreement with experimental transitions or
constitute electronic-engine integration evidence.
