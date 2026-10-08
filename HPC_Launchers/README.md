# Optional native launchers

GitHub Actions is the canonical student calculation environment. Codespaces runs
the student interface. These launchers are optional local or cluster adapters;
successful direct Bash tests do not qualify a Slurm scheduler or GPU profile.

`cochem_submit.slurm` accepts an explicit shell command as its arguments and
propagates that command's real exit status. For a TORQ calculation with no command
arguments, it requires all three environment variables:

| Variable | Required value |
| --- | --- |
| `PYTHON_EXEC` | Absolute interpreter path in the installed calculation environment. |
| `COCHEM_REQUEST_FILE` | Existing normalized request file produced during plan review. |
| `COCHEM_APPROVED_PLAN_FILE` | Existing approval artifact for that exact request, current source, scientific recipe, and resource budget. |

The adapter passes both files to `cochem_torq.cli execute`. It supplies no default
geometry, method, or approval. Expired approvals, source changes, unsupported
methods, and inconsistent budgets remain calculation blockers. The CLI retains
and verifies actual engine artifacts; an unsuccessful calculation does not become
a successful initialization message. Explicit `COCHEM_ARTIFACTS`, `SCRATCH`, and
`TMPDIR` determine output placement. Scheduler allocation, cancellation, requeue,
and staging still require qualification against an actual named cluster.

`cochem_mps_worker.sh` requires an explicit worker command, an observed NVIDIA
device, the native MPS controller, `setsid`, and `timeout`. It rejects missing
prerequisites before launching workers or creating scratch directories. Default
pipe/log directories are private unique children of `COCHEM_SCRATCH` or `TMPDIR`.
Explicit `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` must be distinct,
absolute paths that do not already exist. The launcher cleans up only the worker
process group and control directories it created; it never enumerates or kills
other MPS sessions. Native control operations have bounded waits. CPU-host tests
establish prerequisite rejection; they do not establish GPU concurrency, memory
accounting, or scientific-engine qualification.
