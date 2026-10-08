# Exact capability and restart contracts

The release-candidate capability registry resolves method, named basis/ECP,
electronic reference, requested property, derivative route, engine version,
optimizer version, hardware class and recipe digest as separate identity fields.
The immutable `CapabilityTuple` and `CapabilityRecord` contracts distinguish
`documented`, `experimental`, `locally_validated`, `unsupported` and `unknown`.
An unlisted tuple cannot inherit another tuple's status by changing a method label,
basis, reference, engine release, property or device class.

Built-in restricted HF, PBE-D4, B3LYP-D4 and MP2 native declarations retain
`experimental` status. They authorize explicit controlled validation on the named
Linux x86_64 CPU target. MP2 dipoles remain separately unsupported because a
relaxed-response implementation has not been supplied. The exact revDSD profile
has no runnable tuple: unresolved primary definitions and independent derivative
qualification cannot be filled with invented capabilities.

`application.validate_request` resolves the native dependency closure before
dispatch. A public Actions plan declares the canonical worker target; the actual
local worker checks its observed OS/CPU architecture again. This declaration is
not a measurement of an unlaunched runner or a timing benchmark. The plan binds
the immutable capability records and controlled-validation purpose.

`locally_validate_capability` accepts one retained genuine completed native
PySCF case with verified request/result/basis/installation inventories, SCF
convergence/stability, property presence, matching method/derivative/version and
actual platform. It rejects a stale adapter source. Its evidence identifies the
actual result/manifest bytes, exact normalized native request, scientific source,
basis definitions and installed engine. Production authorization requires these
same bytes, current source and exact native request. This numerical-integration
scope cannot qualify another geometry or imply chemical/experimental accuracy.
No built-in experimental profile is automatically promoted by report existence.

## Native restart registration and use

`register_restart_artifact` accepts an authentic native wavefunction checkpoint or
Cartesian Hessian array from a completed, converged, stable bundle. Registration
validates native checkpoint structure or finite float64 Hessian shape, preserves
the original, and writes a distinct new read-only sealed copy. Existing paths and
symlink paths cannot be used to overwrite originals. Other artifact families have
typed identities but require separately implemented authentic registration adapters.

The immutable fingerprint contains exact Bohr coordinates, atom/isotope row
identities, charge/multiplicity/reference, basis-definition digest, full method
definition, recipe, engine/version/installation, adapter/source and canonical
numerical settings. Numerical settings and method objects are retained as exact
canonical strings to prevent nested mutation of an ostensibly frozen record.

Final-result compatibility requires every fingerprint field to match. A requested
initial guess can explicitly allow geometry or numerical-setting changes only;
state, isotope/atom order, basis, physical method/recipe and implementation changes
remain incompatible. Initial-guess preparation creates a separate mutable working
file from the verified read-only sealed input and requires a new physical calculation.

Compatibility checks and working-copy preparation always report
`engine_checkpoint_reused: false`. An engine adapter has not consumed the prepared
file merely because compatibility passed. The student resume workflow still creates
a newly reviewed same-model attempt and does not claim native checkpoint continuation.
No scientific values are repaired, filled or substituted after a mismatch.

## Evidence limits

`tests/test_capability_restart_integrity.py` separates software-only exact-routing
checks from mandatory `real_engine` cases. The latter calculate genuine H2 with the
current pinned PySCF adapter, inspect actual native checkpoint/Hessian bytes,
exercise compatible preparation and deliberately changed-byte rejection, and verify
that originals remain unchanged. These tests qualify bounded software behavior;
they do not satisfy full crash/restart deployment, independent chemical benchmarks,
revDSD response, or high-accuracy rotational identification gates.
