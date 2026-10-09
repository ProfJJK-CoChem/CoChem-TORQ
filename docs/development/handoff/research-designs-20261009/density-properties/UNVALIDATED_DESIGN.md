# Electronic-field-gradient milestone: unvalidated handoff design

Status: **NOT IMPLEMENTED / NOT TESTED / NOT QUALIFIED**. No EFG, hyperfine,
nuclear-quadrupole coupling, ELF or NCI calculation was executed in this task.
No production, test or documentation file in the checkout was created or edited
by the density-properties agent. Work stopped at the user's handoff request.

The authentic, unmodified upstream file `official-rhf-efg.py` in this directory
was retrieved from:

https://raw.githubusercontent.com/pyscf/properties/4eee5a430fb47eca5962f36fdcaf75c2b87e7ede/pyscf/prop/efg/rhf.py

Repository/commit: `pyscf/properties`,
`4eee5a430fb47eca5962f36fdcaf75c2b87e7ede`. Source SHA-256:
`cb79741703b4174dcdaa1b661953c38eb51dde6eeb82da0d5203b9647180ffd5`.
The actual managed-proxy/TLS-verified curl request completed with exit 0.
The original source retains its Apache-2.0 license notice and explicitly marks
the upstream EFG implementation "In testing". Retrieval and source inspection
do not establish numerical validation, present-version compatibility, or a
production-qualified scientific capability. No source execution occurred.

## Proposed bounded implementation

Own new files only: `src/cochem_torq/engines/density_properties.py`,
`tests/test_density_properties.py`, and
`docs/development/density_properties.md`. The primary agent owns application,
registry, CI, capability gates, common documentation and audit integration.

Expose frozen typed request/result contracts. The request should bind an
existing stable, complete, converged sealed PySCF native bundle by the exact
manifest, result and normalized-request SHA-256 identities, and select explicit
stable atom IDs. Retain the source identity, checkpoint, parsed basis,
installation fingerprint, geometry, orbital occupations, density and computed
integrals/verification arrays in a new empty workspace. Reject symlinks,
out-of-bundle paths, stale output, unsupported schema, incomplete inventory,
changed source bytes and unauthenticated input. Recheck copied/consumed bytes
against original identities. Never accept a user-supplied density array as
proof of a real engine calculation.

Use existing `_native_bundle`, `_actual_file` and `_file_sha256` contracts in
`src/cochem_torq/capabilities.py`; inspect their optimized-source handling and
bind the actual final native bundle for optimized inputs. The established
native format is `cochem-torq.engine-artifacts.v1`; accepted source result format
is `cochem-torq.pyscf-result.v1` with status complete, SCF converged and stability
stable. `engines/checkpoint_restart.py::_validate_checkpoint_orbitals` supplies
an existing real-orbital/AO-metric/occupation/energy check to inspect and reuse
where appropriate. Do not infer that inventory hashes confer publication
authority or prevent deliberate forgery by an authorized local caller.

Initial supported scope: all-electron real restricted closed-shell singlet
RHF, and only separately tested named RKS densities if genuine tests succeed.
Start RHF first; do not advertise RKS before testing it. Reject ECP, ghosts,
finite/gaussian nuclei, relativity, fractional charge/occupation, complex or
open-shell orbitals, correlated MP2/CC densities, density-fitting substitutions
and unsupported dispersion-decorated/source recipes. Check loaded checkpoint
geometry, nuclear charges/spin, basis and orbital dimensions/occupations against
normalized source request/result, including current engine-installation
fingerprint and actual AO metric. Point nuclei use PySCF
`mol._atm[:, gto.mole.NUC_MOD_OF] == gto.mole.NUC_POINT` (inspected installed
source has NUC_MOD_OF=2 and NUC_POINT=1; import constants rather than hard-code).

Postprocessing must use one PySCF/BLAS CPU thread. Bound atom/AO counts,
numerical verification calls and artifact bytes before allocating resources;
agree exact limits with the primary agent. This task had a <=20 MiB artifact
allowance and required at least 256 MiB free disk reserve. The primary agent
must run any native operation through the established owned-process/budget
route; an in-process Python API does not itself impose hard wall/RSS limits.

## Physical convention to implement and test

Define V_ab as the **symmetric traceless Hessian of electrostatic potential at
the probed nucleus**, in the input Cartesian frame. Atomic units are
Eh/(e*a0^2) = e/(4*pi*epsilon0*a0^3), not an energy-only Hessian. Coordinate
derivatives here move the field-observation point at fixed molecular density;
they are not derivatives with respect to nuclear geometry and are not an
orbital-response property.

For each other physical point nucleus B, the contribution at A is
Z_B [3 d_a d_b - |d|^2 delta_ab]/|d|^5, where d=R_B-R_A. Exclude A's own
singular self-field. Retain nuclear and electronic contributions separately.

For the spin-summed AO density D, compute the analytic Hessian integrals of
1/|r-R_A| using installed libcint/PySCF `int1e_ipiprinv` and
`int1e_iprinvip`. The inspected official source combines ipiprinv + iprinvip
then adds its AO-index transpose. Its quadrupolar operator adds the explicit
4*pi/3 times AO-value-product contact correction to each diagonal. Derive and
document this relation independently before final implementation. For the
unmodified fixed-density Coulomb Hessian C, check trace(C)=-4*pi*rho(R_A),
retain raw values, and form the electronic traceless contribution
-[C - trace(C)*I/3]. Symmetry/contact checks must fail closed when outside the
declared tolerances. Do not silently symmetrize a materially asymmetric raw
tensor or discard numerical failures. Raw values, trace, rho and residuals
must remain in artifacts. Preserve the distinction between full potential
Hessian (nonzero contact trace) and nuclear quadrupolar traceless EFG.

Return Cartesian tensor and eigenvalues/eigenvectors with a documented
principal-axis convention. Handle zero/degenerate tensors explicitly: a PAS
orientation is not unique, eta is undefined when Vzz=0, and eigensolver signs
or axis choices are not new physical information. Do not convert EFG into a
hyperfine Hamiltonian, transition list, isotope moment or nuclear quadrupole
constant from an element label. The initial milestone should keep coupling
unavailable. A later coupling implementation needs a separately authenticated,
explicitly sourced signed isotope-specific Q, nuclear spin, uncertainty and
convention, plus audited conversion constants. Local file hashes alone do not
validate a published Q extraction.

## Required genuine checks: all remain UNRUN

1. Real bounded PySCF RHF H2/He/water native bundles with explicit atom IDs,
   stable wavefunction evidence, original checkpoints and retained outcomes.
2. Compare analytic integral contraction with a separately implemented direct
   finite difference of the **fixed-density** electronic potential
   -Tr[D * int1e_rinv(R)] plus physical-other-nuclear potential. Use two
   independently declared probe steps and retain the raw potential values,
   both Hessians and convergence residual. Subtract the measured isotropic
   trace before comparing traceless tensors; do not refit a tolerance after
   seeing a failed result.
3. Independently compare the contact-corrected operator with the genuine
   pinned upstream source convention. Do not execute its automatic isotope
   table analysis or present upstream defaults as authenticated nuclear Q.
   Any extracted/reimplemented source needs attribution and license handling.
4. Translation covariance and proper rotation covariance using actual fresh
   native densities for transformed molecular geometries. Tensor expectation:
   V_rot = R*V*R.T. Preserve signed eigenvalues, symmetry and tracelessness.
5. Spherical closed-shell atom should have EFG zero within a prospective
   numerical tolerance; actual residual is not known yet. Symmetric H2 and
   anisotropic water exercise additional tensor structure.
6. Verify all output inventory sizes/hashes and native input identities;
   reject modified source/checkpoint, unsupported state/nuclear model,
   incomplete bundles, missing atom IDs, nonfinite arrays and stale output.
   Use mathematical/process checks for input policy; no fabricated engine
   output, monkeypatched native calculation or fake checkpoint fixtures.
7. Run focused ruff/mypy and tests in the pinned existing environment, retain
   actual JUnit/logs and tool exit status outside checkout. Only add the new
   real-engine tests to canonical CI after a genuine bounded run succeeds.

Passing these checks establishes only the stated exact density/operator
integration cases. It does not establish basis convergence, molecular-family
accuracy, correlated response, finite-nucleus corrections, spectroscopy
identification accuracy, or completion of W05/W10/full TORQ SRS. ELF/NCI were
not pursued; they require separate definitions and genuine convergence tests.

## Checks actually performed before handoff

Read runtime/onboarding skills and managed environment status; inspected
existing native bundle/checkpoint/engine APIs and installed point-nucleus
constants; retrieved and read the 7,841-byte pinned upstream source. No
electronic-structure call, EFG evaluation, finite difference, test run,
production patch, package installation or capability qualification occurred.
The checkout's other agents' modified files were preserved untouched.
