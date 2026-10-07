# Scientific-result integrity audit

Reviewed 2026-10-07 in the implementation worktree based on source commit
`d7a4739a5f7d6f22ed659b32eeb4706bef16225e`. This report describes inspected paths
and specific repairs. It is not an exhaustive proof that every historical module
is correct, a publication benchmark, or a remote deployment certification.

## Scope and inspection method

The audit searched the canonical application under `src/cochem_torq`, historical
scientific libraries under `Libraries`, shared/mobile code under `cochem`, and
scripts under `scripts`. At the inspection checkpoint those trees contained
216 files. The search
`rg -n -i 'fake|mock|fabricat|fallback|np\.zeros|torch\.zeros|return 0\.0' src Libraries cochem scripts`
returned 352 candidate lines. Candidate counts depend on the changing worktree;
they are search evidence, not counts of scientific defects. The result paths were
manually triaged for whether absent observations were replaced by invented values,
whether a different model was selected under the requested label, and whether
convergence/qualification metadata exceeded the actual evidence.

An array initialized to zero and filled by actual calculations is not invented
scientific data. An exact selection-rule zero, an empty pair sum in an explicitly
named pair potential, and a declared equivalent CPU implementation are also
different from replacing an absent observation with zero. A filesystem path
fallback does not establish a physical model. Conversely, comments saying
"authentic physics" do not qualify an unverified formula or parameter table.

## Concrete repaired paths

| Path | Inspected defect | Current behavior and evidence boundary |
|---|---|---|
| [Student notebook](../../UI/Start_TORQ.ipynb) and [student application](../../src/cochem_torq/student_app.py) | Historical notebook execution contained hardcoded substituted frequencies. | The notebook is an actual interactive launcher; the tracked executed notebook containing those values was removed. It imports genuine input coordinates, previews their real mass frame, and uses shared validation/submission/result verification. It never generates a spectrum to bridge a failed calculation. |
| [Application](../../src/cochem_torq/application.py) and [artifacts](../../src/cochem_torq/artifacts.py) | A failed late stage or timeout could discard valid prior observations; inventory verification alone did not bind all scientific identities. | Typed stage records preserve availability and checkpoint evidence. Semantic verification binds request, recipe and source identities to actual sealed artifacts; mismatches are rejected. Partial scientific status stays distinct from workflow completion. |
| [D3 adapter](../../Libraries/cochem_torq_dispersion_d3.py) | Unverified atomic C6/C8/radius power laws and mixing formulas were presented as D3/D4. | The implementation delegates D3(BJ) and optional ATM to actual `simple-dftd3`, requires an explicit functional, and retains actual parameter-table/version/hash evidence. D4 is a separately named model. Legacy guessed configs and unknown recipes are rejected. Actual first derivatives are exposed; the Torch bridge rejects unqualified higher derivatives. This is a dispersion correction, not a complete electronic method. |
| [OET bridge](../../scripts/oet_client.py) and [GOAT](../../Libraries/cochem_torq_goat.py) | Missing OET service activated invented atomic reference energies, charges and pair parameters; the PHYSICAL seed search perturbed coordinates and assigned unsupported results. | Both historical scientific routes now raise explicit unavailable errors. The selected daemon or ORCA executable is used without a replacement engine. Missing energy units, derivatives, uncertainty or success evidence are rejected or explicitly unavailable. Requested point charges cannot disappear silently. Input state/resources and XYZ frame validity are checked. |
| [RDKit bridge](../../cochem/mobile/rdkit_bridge.py) | Missing force fields/single atoms yielded successful zero energies; minimizer status was ignored; radical count implied a ground-state spin; explicit isotope labels were lost. | Energy is `None` when no actual valid force-field value is available; minimizer convergence is actual, and radical multiplicity must be explicit. The selected electronic state is not claimed to be a verified ground state. Actual tabulated isotope identities/masses survive input, XYZ and result metadata. RDKit force-field geometry is distinct from an ab initio equilibrium structure. |
| [Mobile conformer engine](../../cochem/mobile/conformer_engine.py) | A missing force field or unsuccessful minimizer could be returned as successful minimization. | The current source retains actual partial coordinates, returns unavailable energy and unsuccessful optimization, and identifies the actual MMFF94/UFF model and minimizer status. Separate execution evidence is required for the mobile route; it is not activated by the student notebook. |
| [Formula weights](../../cochem/mobile/inorganic/models.py) | Invalid/unknown formulas could yield a partial or zero mass while the result was called exact. | The current parser rejects invalid/unknown formula data. Its actual elemental-weight quantity is explicitly conventional formula-unit molar mass, distinct from an isotope-resolved exact mass. |
| [Distance telemetry](../../Libraries/cochem_torq_telemetry.py) | A single atom could be assigned an invented 999 Å pair distance; nonfinite coordinates could become an invented zero-distance pair. | No pair returns `(None, None)`; invalid/overflowing geometry raises. Actual finite pair distances are calculated without dropping atoms or adding replacement coordinates. |
| [Mobile payload serializer](../../cochem/mobile/payload_serializer.py) and [receiver](../../cochem/mobile/airgap_receiver.py) | A public fallback key authenticated unconfigured jobs; shared handler configuration could replace another receiver's key/storage; job identifiers affected paths; staged manifest identities were not bound to signed file identities; malformed XYZ atoms could disappear. | Authentication requires an explicit/configured nonblank key, preserving its exact bytes. Each receiver owns its configuration. Safe job identifiers cannot create nested paths; fresh receiver filenames contain no supplied identifier. Staged job/timestamp identities must match the signed file. Stored receiver files retain the exact authenticated bytes and WAL records. XYZ retains blank comments, enforces declared counts and all atom records, and rejects nonfinite coordinates/unknown elements. JSON rejects nonfinite numbers and duplicate keys. Actual conventional Mendeleev weights remain distinct from isotope-resolved exact masses. |
| [Historical SPCAT script](../../scripts/spcat_runner.py) | Invented deck uncertainties/partition values and a quiet replacement solver could yield unsupported catalog outputs. | The facade is disabled until scientifically qualified; unavailable catalog generation writes no deck, scratch directory or result. Its tested Ray parameter is an algebraic software check; no native SPCAT catalog or spectroscopic accuracy is certified. |

The root application and earlier library repairs also remove missing-frequency,
missing-dipole, unavailable-engine, energy-parser and unsupported-wavefunction
substitutions. Their individual acceptance evidence is tracked in
[the release scope](student_release_scope.md) and
[scientific acceptance gaps](scientific_acceptance_gaps.md). The table above does
not imply every method-matrix row, scientific property or historical adapter is
now implemented or qualified.

## Executed checks for this audit subset

A local run of the following actual test files completed with **53 passed**:

- [D3 reference integrity](../../tests/test_dispersion_reference_integrity.py):
  actual DFTD3 1.6.0 results compared to the direct reference-library interface,
  independently displaced energies, derivative signs/units, parameter provenance
  and rejected unsupported higher derivatives/configurations.
- [OET result integrity](../../tests/test_oet_result_integrity.py) and
  [OET input/CLI contracts](../../tests/test_oet_client.py): actual loopback
  connection failure, no generated result file, strict file/state contracts and
  a repository-native ORCA derivative file used only for parsing/serialization
  and unit/sign checks. No artificial live electronic-structure server is used.
- [RDKit result integrity](../../tests/test_rdkit_result_integrity.py): genuine
  RDKit 2026.3.6 and Mendeleev calculations, independent MMFF energy re-evaluation,
  real UFF execution, nonconvergence/missing parameters, explicit spin/state,
  isotope retention and async/batch propagation.

These checks validate their stated software/numerical contracts. They do not
qualify a complete DFT-D3 recipe, the chemistry of an archived ORCA file, a live
OET model server, exact revDSD, or high-accuracy spectroscopic identification.
Other repaired paths have their separately maintained tests and actual outcomes;
the presence of a linked test file is not evidence that it ran.

Additional focused local checks executed in `/tmp/cochem_exec_student_release`
with `/workspace/.venvs/cochem-torq/bin/python` (Python 3.12),
`PYTHONDONTWRITEBYTECODE=1` and
`PYTHONPATH=/workspace/CoChem-TORQ/src:/workspace/CoChem-TORQ`:

- [Mobile authentication integrity](../../tests/test_mobile_auth_integrity.py):
  **57 passed**, no skips, at the initial prepared-environment checkpoint.
  Following the optional package import repair, **58 passed**, no skips, with
  `/tmp/torq-wheel-check/bin/python` (Python 3.12.14) using actual core dependencies
  and RDKit absent. The additional subprocess check confirms core imports do
  not load molecular implementations, while requesting the genuine conformer
  API without RDKit raises `ImportError`. The mobile facade preserves its 106
  public exports through lazy imports; its inorganic facade preserves 29 exports
  and loads notebook UI dependencies only when a UI export is requested.
  The extended import-boundary regression also passed in
  `/tmp/torq-hosted-cpu-reproduction/bin/python` with actual RDKit installed and
  `anywidget`, `ipywidgets` and `traitlets` absent: **1 passed**, no skips.
  Genuine scientific models/conformer APIs remain importable while requesting
  the actual inorganic widget API raises `ImportError`.
  JUnit: `/tmp/cochem_exec_student_release/mobile-inorganic-clean-cpu.xml`.
  This is an import contract check, not notebook frontend qualification.
  Actual standard-library HMAC checks
  canonical bytes, explicit/fresh random keys and tamper rejection. Actual files
  exercise staged size/signature/identity checks; actual loopback HTTP receivers
  exercise authentication, separate keys/storage, protected source/symlink
  rejection, exact signed-byte persistence and actual SQLite WAL records.
  Strict JSON and XYZ failures are checked without replacing application code.
  JUnit: `/tmp/cochem_exec_student_release/mobile-auth-integrity.xml`.
  Clean-core JUnit: `/tmp/cochem_exec_student_release/mobile-auth-clean-core.xml`.
  The test-source API inventory found zero explicit runtime-replacement APIs;
  that narrow AST result is not a proof of universal no-fabrication safety.
- [SPCAT script integrity](../../tests/test_spcat_script_integrity.py):
  **3 passed**, no skips: actual Ray-parameter arithmetic, required-property
  validation and fail-closed process/file contracts. No scientific engine output
  is substituted. JUnit:
  `/tmp/cochem_exec_student_release/spcat-script-integrity.xml`.

These local checks do not certify remote deployment, adversarial same-user
filesystem mutation, replay prevention, TLS transport, or native SPCAT physics.
The mobile checks concern transport/input integrity, not molecular or electronic
structure qualification. No Docker/GPU/HPC execution is established by them.

## Remaining evidence and review boundaries

The historical Delta-ML constructor's guessed D3 configuration now raises rather
than silently selecting an arbitrary reference functional. Activating that model
requires an explicit dispersion/low-level reference policy, training target and
double-counting review. Installing DFTD3 is not sufficient to qualify Delta-ML.

Some geometry/radius helpers select among differently defined actual database
radii. Those are not invented numbers, but their model definitions and provenance
must be reviewed before claiming a specified scientific radius convention.
Historical model, licensed engine, GPU and HPC paths need genuine provisioning
and independently scoped evidence; failed imports or unsupported tuples remain
blocked. The focused transport/authentication checks above cover their stated
local contracts and remain separate from scientific qualification.

The user-approved no-fabrication policy applies to public APIs as well as the
student route. No repaired public function should be re-enabled by weakening a
test or changing its label to accommodate unsupported outputs. The full scientific
SRS additionally needs independently sourced experimental/engine references,
preregistered observable targets and actual canonical deployment evidence before
publication or identification claims can be made.
