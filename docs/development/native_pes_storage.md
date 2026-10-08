# Native scan evidence and coordinator-fenced PES storage

The native bridge in [`pes_storage.py`](../../src/cochem_torq/pes_storage.py)
connects approved, actual fixed-coordinate HF/STO-3G scan observations to the
immutable HDF5 shard and merge protocol. It preserves the actual original
engine files, verifies the stored RHF state, and admits the exact original
coordinator commits inside the storage publication transaction.

This completes a local native-observation storage path. It does not establish
experimental accuracy, complete a molecular search, validate every method-matrix
entry, or qualify a deployed Actions calculation and Codespaces interface.

## Supported observations

The bridge accepts the exact versioned `hf-sto-3g-pes-validation` and
`hf-sto-3g-internal-pes-validation` profiles when their scan helper and native
adapter support the declared request. Each admitted frame must be an available
`ScanPointResult` with an actual converged, stable restricted state, native
checkpoint, gradient, and the exact approved atom order, geometry, charge,
multiplicity, recipe, scan, request and implementation-source identities.

Coordinates are stored in **bohr**, absolute electronic energies in **hartree**.
Grid coordinates retain their own units in the approved scan declaration. A
fixed scan frame remains a nonstationary sample. A missing or failed observation
cannot supply an HDF5 energy, and repeated physical observations retain their
distinct UUIDs rather than becoming additional molecular states.

The internal-coordinate profile also supports its explicitly approved
`previous_point_density` policy. The first point of each forward/reverse pass
and every challenge point use an independent guess. A continued point requires
the exact preceding available point of its own pass. Its retained native record
must prove consumption of that checkpoint density by a fresh SCF calculation;
iteration-state or final-result reuse is not implied. A failed parent cannot be
replaced silently with an independent guess.

## Data and authority path

1. Execute an approved scan through `ApprovedScanExecutor`. The scan's SQLite
   coordinator, point directories, reviewed request and approval are retained.
2. Open that same coordinator database and call
   `prepare_native_scan_collection(..., workspace=..., points=...,
   destination=...)`. Supply distinct actual points in increasing physical-call
   order. The destination is a new immutable native bundle.
   Collections containing continued children must include all actual consumed
   ancestors. A child-only shard is allowed after that complete bundle is sealed;
   its publication still admits every consumed ancestor's commit.
3. Explicitly authorize a finite storage campaign through `CampaignCoordinator`.
   Its engine and recipe scope must match `collection.identity`. Register the
   exact `native_shard_task_payload(collection, frame_ids)` before allocating,
   dispatching/leasing, collecting and validating the shard writer.
4. Call `publish_native_scan_shard` with that writer's current revision, lease
   token, generation and actual measured usage. The function derives coordinates
   and energies exclusively from the verified native observations.
5. Register the exact `merge_task_payload(collection.identity, sources)` and
   publish with `merge_pes_shards`. Shard order must preserve original
   physical-call order. Both the committed shards and original engine attempts
   are admitted under the sole coordinator transaction.
6. Read authoritative results with `verify_committed_native_pes`. A bare HDF5
   path, a valid checksum, or a native-evidence label alone cannot establish a
   coordinator commit.

Storage approval is separate from calculation approval: creating additional
writers must not silently enlarge the original calculation task or resource
budget. Operator authority and worker lease tokens remain private. Remote
adapters must authenticate callers before invoking the local coordinator.

## Preserved native bundle

The bundle contains the original approved-plan bytes and, for each selected
point, original `point-result.json`, `point-definition.json`,
`backend-request.json`, native `manifest.json`, and every actual native artifact
listed by that manifest. This includes the real request/result/log, molecular
basis, installed-engine fingerprint, geometry array and MO checkpoint. Scratch
files and unrelated runtime secrets are not required evidence.

The bundle manifest binds each file's bytes and size, original point directory,
point/native manifest hashes, complete source coordinator commit and exact
profile. The PES source identity binds the bundle manifest hash, request, scan,
approved plan and original implementation-source identity. Native files are
copied and verified against their source bytes before a Linux atomic no-replace
rename publishes the bundle. Existing destinations cannot be overwritten.

The copied bundle remains an independent readback of the original bytes. The
original native directories and source coordinator must also remain available
for new publication; their current ownership and evidence are rechecked at
admission. A moved archive requires an explicit new identity and authorization,
because absolute source/bundle locations are part of the current protocol.

## Scientific and process checks

The bridge revalidates the native engine inventory and typed electronic,
gradient and AO-density evidence. It additionally reads the actual PySCF
checkpoint molecule, basis, orbitals, occupations and energy, reconstructs RHF
energy and analytic nuclear gradient from that stored state, and compares them
with the committed observation. The reconstruction tolerance is an explicit
`1e-10` hartree and `1e-10` hartree/bohr absolute check; it is a numerical
consistency check, not an experimental uncertainty estimate or another SCF
state search.

For continued points, the bridge additionally validates the exact five-field
backend reference (`policy`, `source_native_directory`,
`source_manifest_sha256`, `recipe_sha256`, `parent_point_id`) against the original
same-pass predecessor. It checks the original parent checkpoint, separately
retained sealed/restart inputs, complete native source snapshot, and actual
checkpoint-consumption record. The stored initial AO density, raw density bytes
and observed first SCF cycle must share their genuine hash. Reprojection of the
actual parent orbitals into the child geometry and source/target fingerprint
checks provide numerical and provenance readback. These checks still do not
prove SCF root continuity across a surface.

Checkpoint-consumption flags require actual booleans; numeric `1`/`0` cannot
substitute for `true`/`false`. Startup energy and electron-count observations
must have their declared finite numerical types. Recipe, approval, committed
result and provenance comparisons use canonical JSON bytes to retain the same
distinction throughout source identity checking.

Every native shard, merge and orphan recovery invokes this admission path
automatically in `fenced_pes.py`. Callers cannot bypass it with a no-op callback
or supply arbitrary arrays under a native label. Original completed worker
leases may expire legitimately; their source admission instead requires the
latest settled succeeded attempt, unchanged generation, revision and exact
commit. Stale, superseded, altered, unavailable, absent-authority or malformed
evidence fails closed.

The existing supplied-data and dimensionless mathematical storage APIs retain
their separate evidence classes. Neither those classes nor genuine native
observations set `scientific_accuracy_established` to true.

## Validation

[`test_native_pes_storage.py`](../../tests/test_native_pes_storage.py) executes
genuine approved PySCF H2 workers, preserves native bytes, publishes independent
shards and merges them under the actual coordinator. It also checks altered
point quantities, unavailable observations, method/unit mismatch, invalid or
reversed frame identities, arbitrary arrays, missing native bundles,
corrupted snapshot/original files and absent source authority. Engine and
coordinator APIs are not patched.

The existing fenced-store tests continue to cover immutable publication,
stale/revoked writers, committed-source generation admission, budgets and actual
process-crash recovery. Any release report must bind its actual test counts to
the precise source and wheel being qualified; this guide does not treat an
unrun test or an earlier source's pass as current evidence.

Additional genuine worker tests cover forward and reverse H2 density
continuation, independent challenge calls, complete ancestor admission, exact
parent-reference rejection and the general-molecule water storage path.
