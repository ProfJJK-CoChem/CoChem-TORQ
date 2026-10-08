# Authenticated native checkpoint consumption

TORQ can now load the actual PySCF orbitals and occupations from a completed,
converged, stable native single-point or optimization-final checkpoint and use
the resulting density as the initial guess for a **new SCF calculation**. This
closes the specific gap between restart compatibility/preparation and actual
engine consumption. It does not restore an interrupted SCF iteration, DIIS
history, optimizer trajectory, or a previously computed final physical result.

The implementation is in
[checkpoint_restart.py](../../src/cochem_torq/engines/checkpoint_restart.py).
The existing compatibility and sealing policy remains in
[capabilities.py](../../src/cochem_torq/capabilities.py).

## Interface and execution

`prepare_checkpoint_density(request, workspace, *, source_native_directory,
recipe_sha256, allow_geometry_change=False,
expected_source_manifest_sha256=None, project_orbitals=True)` returns a
`PreparedCheckpointDensity` with the actual PySCF molecule, mean-field object,
normalized request and loaded initial density. Preparation does not execute SCF
and reports `engine_checkpoint_reused=false`.

`prepared.run_scf()` calls the real `mf.kernel(dm0=prepared.initial_density)`.
Its observation hook checks the first actual iteration's `dm_last` against the
loaded density's SHA-256. A successful consumed claim requires this observed
readback. The original and independent input files are reauthenticated before
and after the kernel. The helper starts exactly one new SCF and must be closed
with `prepared.close()` to release its real engine log.

`evaluate_with_checkpoint(...)` performs preparation, genuine SCF and the
requested energy, analytic gradient, HF analytic Hessian or restricted HF/DFT
dipole calculation, then publishes a complete native artifact manifest. MP2
correlation is freshly recalculated from the new SCF orbitals. D4 is evaluated
using the exact supported adapter and registered parameter/version definition.
Numerical DFT/MP2 Hessians and optimizations are not exposed through this
initial-density route. They require their existing separate calculation routes.

The complete wrapper sets and restores the requested PySCF thread/scratch
environment. Low-level callers own an isolated process and must set the bounded
thread/scratch environment for the entire calculation and derivative sequence;
PySCF's thread and scratch configuration are process-global.

## Identity and preserved evidence

Both source and target must match the same **registered method-profile hash**,
including the pinned engine version, named method and basis, dispersion version,
restricted state, unfrozen/all-electron treatment, numerical recipe and element
domain. An arbitrary supplied hash cannot establish a recipe. A PES plan's
coordinate/control identity remains separate from this electronic recipe.

The source manifest, all inventoried files, requested and observed molecule,
actual basis definitions, installed engine fingerprint and current adapter are
verified. Actual checkpoint orbitals must have the expected dimensions, finite
real values, valid restricted occupations, correct electron count and source AO
orthonormality. The checkpoint SCF energy must agree with its retained result.
Optimization sources must authenticate their final child calculation, optimizer
trajectory and parent inventory before the final checkpoint is selected.

The atom ordering, atom identifiers, isotope symbols and explicit isotope
declarations, charge, multiplicity, basis, method and numerical settings remain
exact. Geometry can change **only** when the caller expressly permits
`allow_geometry_change=True` for initial-guess use. Final-result reuse remains
false. No chemical atom permutation or state change is inferred.

Every new workspace is empty and separate from the source. It retains a copied
read-only native snapshot, a read-only sealed checkpoint, a distinct working
input checkpoint, the actual supplied AO density in NPY and canonical raw form,
and a separate new output wavefunction. Inventoried source files are copied with
independent inodes and their actual recorded digests. Sources, sealed inputs and
snapshots are not overwritten or repaired when corrupt.

`checkpoint-preparation.json` records the unconsumed state.
`checkpoint-consumption.json` records the first-cycle input-density readback,
startup and first-cycle energies, projected electron count, source/target
fingerprints, recipe, installed implementation and actual consumed status. The
new native manifest seals these files together with new scientific outputs.

## Scientific limits and validation

PySCF's native checkpoint loader constructs the initial density from actual saved
orbitals and occupations. With `project_orbitals=True`, its explicit
`init_guess_by_chkfile(..., project=True)` projects and normalizes those orbitals
in the target AO basis. A large geometry change can produce a poor initial guess.
Projection is not an electronic-root continuity proof. If projection is disabled,
the unchanged AO density must still have the correct electron count in the target
overlap metric; TORQ rejects an inconsistent density instead of renormalizing it
silently or substituting a fresh guess.

The genuine H2/water tests in
[test_native_checkpoint_consumption.py](../../tests/test_native_checkpoint_consumption.py)
check preparation-only status, first-iteration readback, new wavefunction/result
authentication, preservation of source bytes, explicit geometry continuation
against an independently initialized SCF, and analytic HF Hessians. Deliberately
damaged actual checkpoints and inventories, altered occupations/orbital energies,
state/basis/method/atom-order/isotope mismatches, changed density/input files,
arbitrary recipe hashes and stale workspaces must reject execution. Genuine
optimization-final consumption is also checked.

These numerical integration cases establish authentic checkpoint consumption and
bounded result agreement. They do not qualify chemical accuracy, speedup, general
SCF-root following, open-shell/multireference continuation, distributed recovery,
or the full SRS restart acceptance gate. Later PES execution must retain the
consumption evidence with its coordinate, parent-point and storage identities.
