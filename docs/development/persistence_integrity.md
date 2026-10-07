# Persistence integrity and migration

The queue and two storage APIs enforce attempt ownership and truthful presence of
stored results. These are infrastructure guarantees, not validation of the
physical correctness of numbers supplied by a caller.

## SQLite attempt leases

`lease_task` returns a `TaskRecord` with a cryptographically random `lease_token`
and a monotonic `lease_generation`. Preserve both from the original attempt and
pass them as keyword arguments to `heartbeat`, `complete_task` and `fail_task`.
A task ID, host or PID alone is insufficient. Never retrieve a replacement
attempt's token to complete an earlier attempt.

```python
attempt = queue.lease_task(worker_pid, worker_host)
if attempt is not None:
    credentials = {
        "lease_token": attempt.lease_token,
        "lease_generation": attempt.lease_generation,
    }
    result = execute_and_validate(attempt.payload)
    queue.complete_task(attempt.task_id, result, **credentials)
```

`execute_and_validate` above denotes the application's actual execution and
scientific validation boundary; the queue does not implement that calculation.
Missing credentials raise `LeaseRequiredError`. A stale heartbeat returns `False`;
stale completion/failure raises `LeaseLostError`. Repeating an identical successful
completion with the same credentials is idempotent. A different payload is a
conflict, even from the same attempt. Nonfinite JSON numbers are rejected.

Use one SQLite connection per thread/process, local storage, and a coordinator
service for remote workers. SQLite WAL is not approved for arbitrary NFS mounts.

**Upgrade:** stop all old workers/coordinators and back up the database before
opening it with the new queue. The transactional migration adds the token and
generation columns. Legacy `RUNNING` rows have no verifiable ownership; migration
invalidates them, consumes one attempt, and returns them to `PENDING` or marks them
`FAILED` when the attempt budget is exhausted. Their previous workers must not
continue writing through old code. Completed records are retained. Downgrading to
unfenced code would restore the integrity defect and is unsupported.

`ProcessTreeManager.register_process` accepts both original lease credentials.
The reaper passes only these recorded credentials. If they are missing, it defers
task state changes to heartbeat expiration; if they are stale, it preserves the
replacement attempt.

## HDF5 writes and reads

`SWMRPESStore` holds an exclusive OS-backed writer lock for the entire lifetime of
a persistent writer. A writer generation advances on every new owner, and a
process cannot use an inherited writer handle. Native HDF5 file locking remains
enabled. `HDF5StorageManager` holds the same style of lock for its entire batch
transaction. Concurrent writer attempts wait or time out; a second writer cannot
silently open a persistent owner’s file.

Both APIs use schema version `2.0.0` and publish a `committed_frames` marker only
after all corresponding arrays have been written and flushed. Readers capture
one marker snapshot, refresh arrays, and return only that committed prefix.
Allocated but unpublished tails are not results. Successful ordinary reopening
truncates unpublished tails before appending. Required coordinate, energy and
force payloads must be finite.

`HDF5StorageManager` uses an exact immutable atom count; its historical parameter
name `max_atoms` no longer means that input atoms may be cropped or padded. With
explicit `atom_ids`, each append must supply the same ordered identifiers. If
identity was not supplied, the file records `molecular_identity_status` as
`unavailable`; consumers must not infer chemical identity from array position.
`SWMRPESStore` likewise requires an independently validated external molecule
record for chemical identity. These numerical APIs do not establish element,
isotope, charge, state or unit provenance on their own.

Unknown uncertainty is `NaN` with `uncertainty_available=False`. A supplied zero is
retained only as an explicitly supplied value, never a missing-value default.
Consumers must interpret the mask, and the application must independently specify
the uncertainty quantity, unit, model and calibration domain. Initial allocation
of zero-filled samples is rejected; a new store begins with zero committed rows.

**HDF5 migration and crash recovery:** existing unmarked schemas are rejected.
Do not infer historical completeness or unknown uncertainty from zero values.
Retain the original file, independently verify its arrays and provenance, and
write only validated data to a new schema-2 file. Files left by an abruptly killed
HDF5 writer may retain HDF5 consistency flags that prevent another writer from
opening them. A SWMR reader can still retrieve the committed prefix, as verified
by the kill-boundary test. Recover to a separate file from that verified prefix,
or use an independently reviewed offline HDF5 recovery procedure; never bypass
file locking or silently clear consistency flags.

## Validation scope

`tests/test_persistence_integrity.py` runs real SQLite and HDF5 operations:
concurrent process leasing; stale completion/failure/heartbeat rejection; schema
migration; idempotent completion; process-reaper behavior; concurrent writer
rejection; live SWMR reads during appends; and a real process kill after flushing
an incomplete batch. Arrays are arithmetic storage inputs with no physical claim.
There are no mocked engines in these checks.

The tests qualify the filesystem where they run. They do not establish arbitrary
network filesystem support, atomicity under power loss, all possible HDF5 crash
boundaries, or scientific accuracy. Other legacy HDF5 writers in TORQ are separate
APIs and require individual migration to this contract before equivalent guarantees
can be claimed for them.

## Training groups and MD trajectories

`HDF5DatasetManager` no longer sets `HDF5_USE_FILE_LOCKING=FALSE`. It rejects that
unsafe process setting, validates ordered atomic numbers and exact finite batch
arrays, and publishes a per-group committed length. Its new
`read_trajectory_batch` uses the same ownership lock and returns only committed
rows. Because it creates groups dynamically, it uses serialized locked I/O and
does not claim active SWMR group creation. The separately implemented Lightning
DataModule has a different legacy input schema and is not implicitly a reader for
these trajectory groups.

`HDF5TrajectoryWriter` creates new files only, so reopening an existing path cannot
truncate past research data. It keeps ownership until close and validates complete
frames before allocation. The original integration `step` is stored explicitly;
readers no longer invent a step from the saved-row index. Readers return only the
committed prefix. Existing trajectories without this marker require migration.

Temperature is derived only if the writer receives an explicit positive
`temperature_dof` consistent with the caller's dynamics constraints. With no
specified degree count it is unavailable (`NaN` and
`temperature_available=False`), avoiding an unsupported assumption that center of
mass constraints were applied. Potential/kinetic energies and vector quantities
remain independently available. These changes also have direct storage tests.
