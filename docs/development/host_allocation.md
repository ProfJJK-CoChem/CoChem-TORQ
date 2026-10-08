# Shared local host resource allocation

[host_allocation.py](../../cochem/orchestration/host_allocation.py) adds one private
SQLite allocation authority across TORQ campaigns in a local deployment. It
reserves CPU cores, RAM, scratch capacity, worker slots and explicitly enabled
physical NVIDIA GPU UUIDs using atomic transactions and revision/lease checks.
The campaign's reviewed scientific plan, method/engine permission, approved budget
and attempt lifecycle remain separate required authorities.

All participating calculations must open the **same private ledger** using the
same deployment operator capability and actor. The bootstrap helper uses the
user's state directory rather than a checkout/campaign directory. Running callers
with separate configured ledger roots creates separate allocation domains and
cannot establish host-wide coordination between them. This service cannot govern
unrelated software that does not participate in the deployment ledger.

## Bootstrap and a pinned policy

`bootstrap_host_allocation(actor=...)` creates or opens the common private ledger
for the actual local OS user, hostname, Linux boot and PID/mount namespace. An
optional absolute `directory` configures the shared deployment root. The default
is `$XDG_STATE_HOME/cochem-torq/host-allocation`, or
`~/.local/state/cochem-torq/host-allocation` if that variable is absent.

The initial default policy is conservative: CPU cores are bounded by the actual
process affinity and cgroup-v2 quota, RAM by observed available memory/cgroup
limits, scratch by observed free filesystem capacity, and workers by the CPU
limit and an upper bound of 16. Default GPU capacity is disabled. RAM and scratch
fields named `*_mb` use **MiB, exactly 1024² bytes**.

The database and opaque operator capability are created together in a private
staging bundle and atomically published without replacement. A real file lock
serializes competing bootstrap processes. The root/bundle are `0700`, database,
capability and lock files are `0600`, and symlinks are rejected. Neither object
representations nor allocation receipts include plaintext capabilities.

An existing policy is preserved. Subsequent lower free-memory/free-scratch probes
affect admission, while the reviewed policy remains unchanged. Supplying a
conflicting policy rejects rather than silently rewriting the deployment's
capacity. A policy change requires a separate reviewed operational transition;
there is no unrestricted live capacity-edit API.

```python
import os
from pathlib import Path

from cochem.orchestration.host_allocation import (
    HostAllocationPolicy,
    HostAllocationRequest,
    bootstrap_host_allocation,
)

actor = f"local-os-user:{os.getuid()}"
ledger = bootstrap_host_allocation(actor=actor)
```

A configured policy must fit a genuine host probe when initialized:

```python
policy = HostAllocationPolicy(
    max_cpu_cores=1,
    max_memory_mb=512,
    max_scratch_mb=256,
    max_workers=1,
    gpu_ids=[],
    scratch_probe_root="/tmp",
)
ledger = bootstrap_host_allocation(
    actor=actor,
    directory=Path("private-shared-deployment-root"),
    policy=policy,
)
```

Choose one deployment root for every BASE/TOPOS/TORQ caller. The scratch probe
root binds the policy to one actual filesystem device; requested workspaces may
be elsewhere on that same device but must be real owned directories with no other
writers. A separate filesystem is rejected rather than borrowing free capacity
from the wrong disk. Additional independent filesystem pools are not implemented.

## Actual probes and admission

Each acquisition uses a `BEGIN IMMEDIATE` transaction to recheck the pinned
host/boot/namespace, read current genuine CPU/RAM/scratch probes, and sum **all
active allocations across campaigns**, including expired leases and unknown
unbound dispatches. Admission rejects if the combined request exceeds any limit.
An active unique campaign/attempt constraint prevents duplicate concurrent
reservations for the same attempt.

RAM/scratch checks conservatively include all outstanding reservations against
currently observed availability. Because an existing worker's actual consumption
may already reduce that availability, the check can double count that consumption
and reject a request before a less conservative scheduler would. It never
silently drops reservations or converts unknown resource use to zero.

`probe_host_resources` reads actual Linux affinity, cgroup-v2 quota/memory files,
`psutil` memory observations and `statvfs` free space. cgroup-v1 recovery/limit
interpretation and non-Linux execution are unqualified and rejected. A probe is an
observation, not a promise that external system workloads cannot change capacity
immediately afterward. This ledger supplies managed admission; OS-level cgroup
resource enforcement is an additional deployment responsibility.

Positive GPU inventory requires a successful actual `nvidia-smi --query-gpu=uuid`
probe. Absent/failed tools produce explicit unavailable evidence and zero enabled
GPU capacity. A CPU-only request remains supported. Fake UUIDs, duplicate slots,
UUIDs absent from the current probe, and slots already reserved by another
allocation are rejected. The implementation does not infer GPU availability from
a configuration flag, benchmark, mock probe or external job's presumed completion.
MIG partitions and non-NVIDIA inventory are not qualified here.

## Same-process execution

For a trusted worker that executes its calculation in the current process,
acquire and bind the real process identity together:

```python
workspace = Path("private-existing-workspace")
# Create/review this owned workspace before asking for allocation.
acquired = ledger.acquire(
    campaign_id="reviewed-campaign-id",
    attempt_id="current-attempt-id",
    workspace=workspace,
    request=HostAllocationRequest(
        cores=1, memory_mb=128, scratch_mb=32, worker_slots=1, gpu_ids=[]
    ),
    pid=os.getpid(),
    lease_seconds=60.0,
)
allocation_id = acquired.receipt["allocation_id"]
revision = acquired.receipt["revision"]

# During genuine work, heartbeat uses the current CAS revision and private token.
current = ledger.heartbeat(allocation_id, revision, acquired.lease_token)
revision = current["revision"]

# Call only after this owned worker's computational block actually finishes.
finished = ledger.finish(
    allocation_id, revision, acquired.lease_token,
    reason="The actual computational block and cleanup completed.",
)
```

A same-process finish is an authenticated operational worker report, not evidence
that the OS process stopped or that the scientific method passed validation. The
receipt states that distinction. It releases the reservation only under the
current lease and exact bound process identity. The scientific result still needs
its independent typed validation, publication fencing and campaign accounting.

Heartbeat and finish use an exact durable revision and private lease token; stale
revisions, expired/wrong tokens, changed workspace identity and terminal records
reject. Lease durations must be finite, positive and no longer than 3600 seconds.
Every heartbeat increments the CAS revision. Terminal allocation records and
events are immutable/append-only through SQLite triggers.

## Child dispatch and conservative recovery

For child processes, reserve first without a PID, then mark dispatch as started
**before calling `Popen`**. Bind only the genuine live child returned by actual
process creation. Binding observes ownership and PID/create time inside the
transaction; a child already gone before binding is rejected.

```python
reserved = ledger.acquire(
    campaign_id="reviewed-campaign-id", attempt_id="new-current-attempt-id",
    workspace=workspace,
    request=HostAllocationRequest(cores=1, memory_mb=128, scratch_mb=32),
)
pending = ledger.begin_dispatch(
    reserved.receipt["allocation_id"], reserved.receipt["revision"],
    reserved.lease_token,
)
# child = the actual Popen returned by the already approved launch adapter
# bound = ledger.bind_dispatch(
#     pending["allocation_id"], pending["revision"], reserved.lease_token,
#     pid=child.pid,
# )
```

The operator can reconcile a bound worker after genuine `Popen.wait()` or equivalent
observed termination. Reconciliation does not require a surviving lease, so a dead
worker with an expired lease can be settled. It verifies the exact host realm and
actual recorded PID/create time. Accepted observations are an absent PID, a real
PID with a different creation time proving the original process ended, or the
original process's actual zombie state. A live process or inaccessible liveness
remains reserved. Timeout/heartbeat expiry alone is never death evidence.

If the actual child exits between the monitor's `poll()` and a host heartbeat,
`observe_worker_exit` checks the same unexpired lease, revision, actor, workspace,
host and real bound PID/create time. It records the actual exit and keeps the
allocation active. The dispatchers first perform the actual `Popen.wait()` and
then collect/reconcile. A wrong/expired lease or still-live process remains a
failure; the exit path never restores authority or marks science successful.

The constrained optimization adapter starts each native evaluation in a separate
session. Consequently, outer-worker death, a process-group signal and Linux
parent-death-signal activation are insufficient to release resources. Before
child reconciliation, the ledger examines the adapter's durable
`constrained-engine/evaluations` launch inventory. It validates exact outer
owner/child PID and creation-time bindings, parent observations and actual waited
return receipts. Each identified child must independently be observed absent,
reused or a zombie. An actual parent wait may establish a quick child's exit when
its creation time was never observed; that missing time stays missing. A partial
launch without sufficient binding/wait evidence, a live child or inconsistent
receipt retains the host allocation. No guessed descendants are inserted.

Constrained requests pin `child_inventory_scope="constrained-evaluation-v1"`
in the original durable allocation request. That scope requires a nonempty valid
observed evaluation inventory; deleting the entire inventory or leaving it empty
fails closed. Changing a workspace file cannot downgrade this pinned scope.
An early constrained-worker failure before any inventory was created can therefore
conservatively leave resources/costs held for reviewed recovery. Single-worker
requests use the separately named default scope; the known nested directory is
still inspected whenever it is present.

Terminal receipts retain the evaluation identities, genuine receipt hashes and
actual child-death proofs. The ordinary dispatcher also keeps scientific cost
accounting unknown when nested launch/death evidence is insufficient. Explicit
operator recovery can reconcile once exact evidence and actual termination are
available. This rule covers the known constrained adapter's inventory contract;
an independently introduced subprocess adapter requires its own reviewed
inventory/ownership contract before use.

```python
# child.wait() must have actually completed before this call.
# reconciled = ledger.reconcile(
#     bound["allocation_id"], bound["revision"],
#     reason="The actual owned child wait completed; original process is dead.",
# )
```

If the reserving process dies before binding, the ledger cannot prove whether an
unbound child launched. Those resources remain held for reviewed recovery; it
does not infer no dispatch from missing metadata. `release_undispatched` permits
only the exact live reserving owner under its current lease to release a
`reserved` record **before any dispatch starts**. `dispatching` records cannot use
that shortcut, including children that exited before their creation identity was
bound. There is deliberately no force-release API based only on a claimed reason.

A changed host/boot/namespace identity rejects opening the old ledger. The default
helper selects a separate current-boot realm, while historical reservations remain
in their original ledger without being relabeled as observed completions. Restored
or moved host ledgers must not be used to revive old process capabilities.

## Resource observations and evidence limits

`ObservedHostUsage` accepts actual observed RAM/scratch peaks and an explicit
measurement source. Either field may remain `None`. The ledger does not fill
missing measurements, infer a peak from a requested budget, or substitute zeros.
Measurement accuracy and sampling scope belong to the real monitoring adapter;
for example, the maximum of sampled RSS values is an observed sampled peak, not a
claim about an unobserved complete lifetime. Scientific campaign CPU/wall/GPU
consumption settlement remains separate.

[The genuine host tests](../../tests/test_host_allocation.py) cover every CPU/RAM/
scratch/worker limit, separate campaigns, private authority/actor/boot bindings,
strict leases and missingness, workspace replacement/permissions/symlinks, actual
child wait/reconciliation, expired-live and unknown-dispatch retention, genuine
SIGKILL, simultaneous bootstrap, simultaneous cross-campaign acquisition and
simultaneous heartbeat CAS. CPU tests do not qualify a positive GPU production
profile. No physical observations or capacity probes are mocked.

```bash
PYTHONPATH=src:. COCHEM_DISABLE_SANDBOX_CHECK=1 python -m pytest -q \
  tests/test_host_allocation.py tests/test_campaign_backup.py \
  tests/test_campaign_coordinator.py
```

## Native dispatcher integration

The approved local PES scan executor and ordinary spectroscopy/constrained
dispatcher both bootstrap this common ledger with actor
`local-os-user:<actual uid>`. They acquire the full reviewed CPU/RAM/scratch/worker
reservation in addition to campaign-budget authority, mark dispatch before actual
process creation, bind the real child identity, heartbeat both authorities and
perform actual owned waits before reconciliation. Host capacity rejection occurs
before native launch. Failed/unknown launches do not release reservations from
missing metadata or heartbeat expiry.

A rejected pre-dispatch scan records zero **engine** wall consumption because no
native worker was launched. Its separately observed parent admission interval is
reported as operational setup and never charged as physical engine execution.
After genuine launch, scientific wall settlement uses the actual owned execution
interval. Early scratch/wall/memory failures retain their observed budget reason
alongside any unavailable native-manifest diagnostic; absent quantities remain
absent.

New point publications and ordinary sealed shards include typed terminal host
provenance with host/policy/request/receipt hashes, actual PID/create time and
death proofs. Publication compares this provenance with the immutable durable
ledger record. Historic point records lacking the optional field remain readable;
new commits cannot silently omit or alter their actual host proof. Resource
admission is operational evidence and does not establish method accuracy.

[The real native integration tests](../../tests/test_scan_host_allocation.py)
exercise H2 PES calculations, water optimization/Hessian calculations, two actual
competing campaigns, genuine SIGKILL, scratch exhaustion, blocked dispatch,
deliberately lost nested launch receipts, constrained native-child termination and
actual native exits during a heartbeat blocked by a real SQLite writer. No engine
or process behavior is mocked. These are bounded local validation checks; the
full SRS scientific qualification still requires independent benchmark evidence.

```bash
PYTHONPATH=src:. COCHEM_DISABLE_SANDBOX_CHECK=1 python -m pytest -q \
  tests/test_scan_host_allocation.py
```

Remote schedulers, multiple OS users,
separate namespace realms and independently configured ledgers need an additional
shared deployment authority; this local milestone does not silently claim it.
