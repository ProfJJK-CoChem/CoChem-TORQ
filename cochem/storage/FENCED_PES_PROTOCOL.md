# Coordinator-owned PES storage

`fenced_pes.py` implements the local ownership contract for TORQ-OPS-005.
It uses real HDF5 files and the same `CampaignCoordinator` SQLite database
as its worker and merger attempts. These storage checks do not qualify an
electronic-structure method, a molecular PES, a learned potential or a search.

## Declared samples and exact identity

`PESIdentity` records the ordered atom identifiers, molecular composition,
electronic state, isotopes when supplied, recipe, source identity, engine,
coordinate units and energy units. Each supplied frame requires a unique
physical UUID, finite coordinates and its own finite scalar energy. Missing
energies, incomplete arrays and complex-to-real casts are rejected.

The current evidence classes are `supplied_data` and
`mathematical_validation`. Caller-supplied samples remain scientifically
unqualified. Mathematical validation requires dimensionless values and is
never labelled as native electronic-structure evidence. No synthetic sample
is generated to repair unavailable data. A future native-engine adapter must
bind the real execution request, immutable native result and qualified method
to this contract before claiming molecular qualification.

## Publication and admission

1. The operator-approved task payload is produced by `shard_task_payload`
   or `merge_task_payload`. It contains exact sample identities or sealed
   source references. The task's allowed engine and recipe must agree.
2. The writer authenticates its current validating attempt revision, lease
   token, generation and campaign approval before creating staging files.
3. A private attempt-specific staging directory receives `samples.h5` and
   `manifest.json`. Arrays are complete before publication, the HDF5 file
   closes, and files and directory are flushed with `fsync`. SHA-256, size,
   schema, typed datasets and the exact committed frame count are checked.
4. A merger reads private copies verified against each sealed source
   manifest. Every source must have the same complete identity. Repeated
   sample UUIDs and duplicate source attempts are rejected.
5. Inside the coordinator publication callback, source admission checks the
   latest successful attempt, current task generation, settled reservation,
   committed revision and exact stored artifact receipt. The original
   source bytes are reverified. Historical completed source leases may
   expire; they are not authority for a new merger.
6. A target-specific advisory lock serializes cooperating publishers.
   Linux `renameat2(RENAME_NOREPLACE)` atomically installs a new directory
   without replacing any existing target, and the parent is flushed.
   The coordinator checks approval and the merger lease before and after
   the callback and commits its durable receipt under revision CAS.

`verify_pes_artifact` checks bytes and schema. It does **not** establish
authority. Consumers use `verify_committed_pes`, and subsequent mergers also
use transaction-bound current-source admission. Copying a valid artifact to
another path does not create a committed receipt for the copy. Partial,
failed, uncommitted, incompatible and stale sources are excluded; there is
no implicit partial-source policy.

## Interrupted publication

Filesystem rename and SQLite commit are separate operations. A crash or
post-publication lease expiry can leave authentic sealed bytes without a
successful database receipt. Readers must reject such an orphan as
authoritative output.

`recover_pes_publication` can commit an intact orphan only under its **same**
current attempt, revision, generation, approved payload and live lease.
It verifies the parents and exact bytes again and never modifies the orphan.
Expired, revoked or superseded ownership cannot recover it. A newly
authorized attempt must build a new version from admitted immutable sources.
Incomplete staging is not repaired into a success or treated as usable PES
data; rerun from the original supplied inputs or admitted sources.

## Runtime contract and validation limits

The canonical runtime for this protocol is Linux with mandatory HDF5 file
locking, working `flock`, SQLite WAL/FULL durability, `fsync` and atomic
no-replace rename on the output filesystem. Disabled or best-effort
`HDF5_USE_FILE_LOCKING` policies are rejected. Unsupported atomic rename or
filesystem operations fail; there is no silent unlocked publication fallback.
Closed immutable artifacts require no SWMR readers. Existing standalone SWMR
stores remain separate numerical APIs and must not be advertised as
coordinator-authoritative merely because they can be opened.

Local tests use actual dimensionless function evaluations, real HDF5/SQLite
files, real competing processes and abrupt `os._exit` between rename and
database commit. They verify integrity and process ownership. They do not
validate NFS locking/durability, deployed HDF5 filter availability, cross-host
recovery, native engine-to-PES sampling integration or molecular accuracy.
Those remain explicit deployment and scientific qualification gates.

The caller must supply genuine measured resource usage for the complete
attempt; unavailable telemetry stays null and the coordinator charges
reserved ceilings separately. The storage API does not invent CPU, GPU,
memory, scratch or accuracy measurements. It is not an OS resource supervisor.
