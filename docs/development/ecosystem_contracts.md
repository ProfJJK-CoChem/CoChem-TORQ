# BASE, TOPOS and TORQ integration contracts

## Canonical environments and ownership

GitHub Actions is the canonical calculation environment. GitHub Codespaces is
the canonical interface environment. The Codespaces UI imports structures,
previews validated plans, submits an approved calculation artifact, and reads
the genuine Actions result. Installing an engine in a Codespace does not change
that deployment policy. Local execution is an explicitly selected development
profile and is not evidence that the hosted deployment has passed.

BASE owns identity, installation and shared orchestration policy. TOPOS owns
discovery. TORQ consumes candidates, refines selected candidates, and computes
its supported rotational-spectroscopy products. Neither accepting an XYZ nor
reading a TOPOS tier proves that the candidate was optimized or that its
producer method is suitable for the next scientific stage.

The sibling repositories were retrieved and inspected at these immutable
revisions. No sibling application files were changed:

| Module | Repository revision | Observed distribution | Observed boundary |
| --- | --- | --- | --- |
| BASE | `1a3c633f6cb0c6256223298ed95d71196ffc3689` | `CoChem-BASE==1.0.0` | `cochem_base.interfaces.artifact_handoff`; isolated module installation/execution adapters |
| TOPOS | `6a01b0f2adb7cff02edda6e339facf3d6f93904d` | `cochem-topos==0.1.0` | `cascade_engine.cochem_cascade_hdf5.CascadeHDF5Serializer` |

## Separate installations

Use the reviewed BASE installer and a pinned module manifest to provision each
sibling into its own environment. The platform bootstrap uses
`COCHEM_MODULE_ROOT` for the external module-installation root and
`COCHEM_MODULE_MANIFEST` for its reviewed manifest. These are outside the TORQ
checkout. This matters because the current BASE distribution packages namespaces
named `cochem`, `cochem_torq`, and `Libraries` as well as `cochem_base`. Installing
BASE and TORQ into one interpreter can silently select the wrong implementation.

The actual BASE `scripts/module-distribution.json` already pins TOPOS to the
revision above. It pins TORQ to historical commit
`d7a4739a5f7d6f22ed659b32eeb4706bef16225e`. A published student TORQ revision
therefore requires a reviewed BASE lock update and new isolated-adapter checks.
The current local TORQ working tree must not be described as already deployed by
that upstream lock.

Distribution installation, metadata discovery and numerical method qualification
are distinct observations. TORQ's `module_provider()` supplies metadata for the
`cochem.modules` entry-point group; it does not advertise unrun operations as
scientifically qualified.

Existing BASE and TORQ recipe/molecule digests retain their explicit legacy
serialization profile `cochem.sorted-json/1`. This profile uses sorted Python
JSON serialization and is not RFC 8785 canonical JSON. New TORQ readers accept
legacy conformer records without the additive `serialization_profile` field;
older strict TORQ readers can reject new records carrying it. This is not a claim
of bidirectional schema compatibility. BASE-owned manifest shape and method
identity bytes are preserved. An RFC 8785 migration requires a separately
versioned interchange contract rather than recomputing these historical digests.

## TORQ conformer contract

The implementation is `src/cochem_torq/ecosystem.py`. Its strict Pydantic records
forbid unknown fields and validate existing instances again at boundaries.

`MoleculeHandoff` requires:

- A molecular identifier and one unique stable `atom_id` per source row.
- Canonical element symbols, with an optional explicit isotope mass number.
- Exactly one finite three-component Cartesian row per atom and explicitly
  declared `angstrom` or `bohr` units.
- Explicit integer charge and positive integer multiplicity. Electron count,
  spin bound and electron/multiplicity parity must be compatible.
- Optional topology digest, without inferring connectivity from coordinates.

An absent isotope remains unspecified. No average mass or arbitrary isotope is
inserted. The handoff validates isotope identifiers structurally; the downstream
mass resolver must verify measured isotope masses before a mass-dependent
spectroscopic result. Electron parity rejects impossible combinations and does
not prove that a supplied spin is the ground state.

`MethodProvenance` records the actual producer's engine/version, method, explicitly
nullable basis, recipe identifier and complete settings. Its SHA-256 must match
canonical JSON for the full record. The producer recipe is separate from the
requested TORQ target recipe. Tier names do not identify a scientific method.

`EnergyObservation` requires a finite Hartree value, an explicit total or relative
electronic-energy designation, the molecular geometry/state digest and the exact
source-artifact digest. A relative energy also requires a reference ID. Moving an
energy to a different geometry, atom mapping or electronic state fails validation.
No missing energy is replaced by zero.

`ConformerHandoff` records source metadata and quality flags. Convergence remains
`unknown` unless supplied with a preserved independent evidence-artifact digest.
A digest proves byte identity; interpreting that evidence still belongs to the
producer parser and scientific validator. A file serializer is not a convergence
validator.

`to_application_molecule()` returns the actual application fields
`symbols`, `geometry_bohr`, `charge`, `multiplicity`, `atom_ids`, and `isotopes`.
Angstrom inputs are converted explicitly using the shared versioned CODATA
[constants service](units_and_constants.md), preserving the original Bohr-radius
factor and operation order. `to_application_provenance()` is retained in the application request's
`source_provenance` field, including the complete handoff and source atom map.

`write_conformer_handoff()` revalidates nested records, enforces the existing
source-tree write boundary, flushes a temporary file, and publishes an exclusive
complete JSON artifact. It cannot replace an existing user artifact.
`load_conformer_handoff()` rejects duplicate JSON keys and nonfinite JSON values.

## Consume the actual BASE artifact contract

The observed BASE schema is `cochem.module-handoff/1`. Its real producer is
`cochem_base.interfaces.artifact_handoff.prepare_module_handoff()`, and its real
validator is `load_module_handoff()`. BASE supports several artifact kinds; this
TORQ receiver currently implements the `geometry_xyz` boundary.

```python
from cochem_torq.ecosystem import read_base_handoff

candidate = read_base_handoff(
    "/path/to/external/base-package/handoff.json",
    molecule_id="student-assigned-molecule-id",
    charge=0,
    multiplicity=1,
    atom_ids=["oxygen", "hydrogen-1", "hydrogen-2"],
)
molecule = candidate.to_application_molecule()
source_provenance = candidate.to_application_provenance()
```

The charge, spin and row IDs above are an explicit illustrative user declaration;
the receiver does not infer them from XYZ. BASE's actual manifest omits them.
The receiver independently checks schema, recipient, pending status, validation
scope, regular-file location, byte count, SHA-256, source frame size, symbols and
declared Angstrom units. An escaping path, symlink escaping the package, altered
metadata, duplicate JSON key or changed file fails. Original BASE operation,
options, capability snapshot and manifest digest remain in provenance.

The received value remains an input geometry with no claimed energy,
optimization, harmonic characterization, or scientific execution. Importing the
manifest does not submit a calculation. The selected approved TORQ plan determines
the subsequent operation.

BASE's current execution service
`cochem_base.interfaces.module_execution.execute_module_handoff()` accepts only
reviewed `geometry_analysis` operations through `module_adapter_torq.py` and
`module_adapter_topos.py`. Its TORQ adapter performs rigid geometry/inertia
analysis and checks independent units and frame preservation. Its TOPOS adapter
performs connectivity hashing. Neither adapter implements general TORQ refinement,
VPT2, discovery, or Actions submission. Those abilities must not be inferred from
their successful geometry-operation reports.

## Consume the actual TOPOS HDF5 contract

The real TOPOS writer has this public signature:

```python
CascadeHDF5Serializer.write_tier_data(
    geom_id, tier_id, energy, gradient=None, hessian=None, geometry=""
)
```

It creates an HDF5 root `format_version="1.0"`; the selected
`<geom_id>/<tier_id>` group holds `geometry_xyz` as a scalar byte string and
`electronic_energy_hartree` as an attribute. Optional gradient/Hessian datasets
exist, but the current writer does not record their units or derivative provenance.
TORQ therefore imports the geometry and explicitly observed energy; it does not
promote those optional arrays to a qualified force field.

`read_topos_handoff()` acquires the writer's real adjacent `<database>.lock`, reads
the exact selected group, and verifies that the source digest stays unchanged
during the locked snapshot. All of these externally omitted declarations are
mandatory: molecular ID, charge, multiplicity, stable source-row IDs, geometry
units, complete producer recipe, energy kind, and explicitly nullable relative
reference ID. Missing declarations fail before use. The default convergence
status is unknown; an affirmative/negative convergence assertion requires a
preserved independent evidence artifact.

The real writer overwrites a tier group when rerunning and has no commit marker.
Only locked snapshots from the pinned writer are supported. A live producer using
another locking convention is an incompatible profile. Historical source bytes
should be copied into immutable campaign artifacts before archival claims; no
reference to a mutable landscape file constitutes permanent evidence.

## Task-service boundary and current BASE gap

The accepted D08 policy is one BASE service contract plus a standalone adapter.
There must be one task/commit authority per deployment. The safe implemented
TORQ coordinator is `cochem.orchestration.sqlite_queue.SQLiteTaskQueue`:

| Operation | Required attempt ownership |
| --- | --- |
| `enqueue_task(...)` | Coordinator-authorized submission |
| `lease_task(worker_pid, worker_host)` | Returns immutable `TaskRecord`, opaque `lease_token`, monotonic `lease_generation` |
| `heartbeat(task_id, ..., lease_token=..., lease_generation=...)` | Original attempt credentials |
| `complete_task(task_id, result, lease_token=..., lease_generation=...)` | Original attempt credentials; rejects stale worker completion |
| `fail_task(task_id, error_message, ..., lease_token=..., lease_generation=...)` | Original attempt credentials |

Workers retain their original credentials. They may not fetch a new token by task
ID to revive a stale attempt. SQLite stays on the coordinating host's local
filesystem; workers call a coordinator transport rather than writing it over NFS.

The pinned BASE queue has `complete_task(task_id, result)` and
`fail_task(task_id, error_message)` without tokens/generations, and its task schema
has no fencing fields. That concrete API does not satisfy the TORQ ownership
contract. TORQ uses its standalone fenced coordinator until BASE publishes and
qualifies a compatible service. Combining both databases as competing authorities
or adapting away the credentials is unsupported. The sibling interoperability
test explicitly checks this known mismatch so a changed upstream revision prompts
a reviewed contract update.

## Return a genuine TORQ electronic result to BASE

`export_base_calculation_result(run_directory, destination)` exports the actual
published TORQ worker shard into BASE's observed `calculation_result` artifact
boundary. It verifies the complete shard before and after preparation, checks the
preserved native engine manifest and native `result.json`, and compares the parsed
electronic and final-geometry stages against those native bytes. Missing energies,
unavailable stages, unchecked/unstable wavefunctions, mismatched state/row mapping,
and missing optimizer/final-gradient evidence prevent publication.

The resulting JSON explicitly reports `quantity="total_electronic_energy"`,
Hartree units, actual engine/version/method, complete recipe, observed final Bohr
coordinates, charge/spin, source row mapping, quality flags and uncertainty. Its
scope is `observed_electronic_energy_at_verified_final_geometry`. The explicit
`converged`, `energy_hartree` and `engine` fields match the real BASE inspector.
It retains both original source handoffs and request/native/shard digests. It
never upgrades that quantity to a vibrationally corrected constant or an
experimentally calibrated spectroscopic identification claim.

The reverse consumer test runs an actual H2 PySCF calculation through TORQ's
application, exports its authentic immutable bundle, and passes the exported JSON
to real BASE `prepare_module_handoff()` and `load_module_handoff()`. A corrupted
copy of that actual run fails checksum verification and cannot publish a result.
The first isolated installation test also exposed TOPOS's undeclared `filelock`
import. The canonical module manifest supplies a reviewed pinned adapter
dependency; adding it requires a new verified installation receipt, rather than
silently mutating an already qualified environment.

## Genuine verification and qualification limits

`tests/test_ecosystem_integration.py` invokes the actual BASE producer/validator
and actual TOPOS serializer in subprocesses with only the corresponding sibling
checkout on the import path. It never installs conflicting namespaces into the
TORQ environment, injects substitute modules, patches an engine, or constructs
purported calculation results. The TOPOS serializer is supplied observed geometry,
energy and gradient from tracked `test.engrad`; `test.inp` and `test.property.txt`
record the source input and ORCA version. This archived input is explicitly marked
unqualified as a scientific recipe. The tests establish file interchange and
state-integrity behavior, not new electronic-structure or publication accuracy.

Damaged copies test rejection of missing/nonfinite energies, invalid source rows,
changed metadata, checksum mismatches, directory escape, and unsupported schema.
Additional checks cover spin parity, duplicate mappings, unit conversion,
geometry-bound energy, full recipe hashes and exclusive publication.

The optional real sibling profiles accept `COCHEM_BASE_SOURCE` and
`COCHEM_TOPOS_SOURCE` plus their respective `COCHEM_BASE_PYTHON` and
`COCHEM_TOPOS_PYTHON` interpreter paths. Without a real pinned checkout, the profile
is skipped with an explicit unverified message. `COCHEM_REQUIRE_ECOSYSTEM=1`
turns an absent sibling into a failed mandatory gate. Imports use `-B` and
`PYTHONDONTWRITEBYTECODE=1` to preserve installation-receipt inventories.
A skip is never a scientific qualification pass. Canonical Actions jobs must provision these pins and require
the interoperability profile, alongside the applicable actual-engine gates.

Local verification on 2026-10-07 passed all 33 checks against the separately
installed BASE/TOPOS environments with the corrected pinned adapter requirements:
31 forward/schema checks and 2 actual TORQ-to-BASE checks. The reverse calculation
used real PySCF 2.14.0 and geomeTRIC 1.1.1, including a separate final-geometry
Hessian. Ruff and targeted strict mypy checks passed. This verifies the reviewed
interchange paths and the bounded H2 teaching calculation; it does not qualify
unimplemented sibling services or high-accuracy spectroscopy methods.
