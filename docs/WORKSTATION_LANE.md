# Workstation lane (Phase 4 of the workstation job runner plan)

GitHub Actions remains TORQ's canonical calculation environment. Requests the
public classroom worker cannot run (more than 2 cores, 2048 MB or 600 s, or
research recipes) can be queued to the lab workstation through a Google Drive
folder the user assigns. The workstation's
[job runner](https://github.com/ProfJJK-CoChem/cochem_workstation_job_runner)
runs the ordinary explicit local validation with its `torq_execute` template:

```
cochem-torq execute --request request.json --approved-plan approved_plan.json --output-dir result
```

Workstation results form a **separate, labelled evidence lane**. They are
accepted only after `verify_shard` (request identity and complete byte
inventory) and a matching implementation identity (`code_sha256`); a
workstation "COMPLETED" status is never scientific success by itself.

## Commands

```
cochem-torq workstation recommend --request request.json        # Actions or workstation, with reasons
cochem-torq workstation assign  --folder "G:/My Drive/CoChem workstation" --student alice
cochem-torq plan --request request.json --execution local_validation --output plan.json
cochem-torq approve-plan --plan plan.json --actor alice --expires-at 2026-10-20T00:00:00+00:00 --output approved.json
cochem-torq workstation submit  --request request.json --approved-plan approved.json --idempotency-key run-7
cochem-torq workstation status  cochem-torq:<request-id>
cochem-torq workstation poll    --destination results/            # ingest every finished job
cochem-torq workstation cancel  cochem-torq:<request-id> --reason "no longer needed"
```

`--folder`/`--student` default to `COCHEM_TORQ_WORKSTATION_FOLDER` /
`COCHEM_WORKSTATION_FOLDER` and `COCHEM_TORQ_WORKSTATION_STUDENT` /
`COCHEM_WORKSTATION_STUDENT`. The approval must remain valid for at least 24 h
because the workstation may hold a job while its owner uses the machine.

## TPO ledger

`~/.local/state/cochem-torq/tpo_ledger.sqlite` (or `--ledger`) records every
submission: `PENDING_WORKSTATION` -> `PAUSED_WORKSTATION` /
`RUNNING_WORKSTATION` -> `INGESTED`, or `FAILED_WORKSTATION`,
`CANCELLED_WORKSTATION`, `REJECTED_RESULT` (returned results failed
verification). Its event table is append-only (SQLite triggers reject UPDATE
and DELETE). Idempotency keys prevent duplicate submissions; ingested shards
are published once to an immutable `workstation-<request-id>/` folder.

## Workstation requirements

TORQ's local execution qualifies host capacity from cgroup-v2 CPU/RAM limits.
On the Windows workstation, run the `torq_execute` template inside WSL2
(`launcher = "wsl"` in the runner config) with cgroup v2 available (systemd
enabled in the distribution, or `cgroup_no_v1=all` in `.wslconfig`
`kernelCommandLine`). The workstation must run the same TORQ source as the
submitter: the approved plan binds `code_sha256`.

## Code and tests

* `src/cochem_torq/workstation.py` - routing, ledger, folder transport,
  submission, polling, verified ingestion.
* `tests/test_workstation_dispatch.py` - the returned shard is TORQ's own real
  `worker_execute` (PySCF) result for the exact submitted request, sealed with
  `seal_shard`; only the runner's protocol files are written by the test.
