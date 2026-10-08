#!/usr/bin/env bash
# ==============================================================================
# Optional CoChem-TORQ SLURM adapter for an explicitly approved calculation.
# GitHub Actions remains the canonical student calculation environment.
# Direct Bash execution tests this adapter, not scheduler allocation/qualification.
# ==============================================================================
#SBATCH --job-name=CoChem-TORQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Dynamic repository root resolution (Air-Gap compliant)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Dynamic COCHEM_ARTIFACTS resolution
if [[ -z "${COCHEM_ARTIFACTS:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        export COCHEM_ARTIFACTS="${SCRATCH}/cochem_torq_${SLURM_JOB_ID:-$$}"
    elif [[ -n "${TMPDIR:-}" ]]; then
        export COCHEM_ARTIFACTS="${TMPDIR}/cochem_torq_${SLURM_JOB_ID:-$$}"
    else
        export COCHEM_ARTIFACTS="/tmp/cochem_torq_${UID:-$USER}_${SLURM_JOB_ID:-$$}"
    fi
else
    export COCHEM_ARTIFACTS
fi

# Secure directory hierarchy creation
mkdir -p "${COCHEM_ARTIFACTS}/Scratch" "${COCHEM_ARTIFACTS}/Logs" "${COCHEM_ARTIFACTS}/Outputs"

# Export environment variables for compute processes
export TMPDIR="${COCHEM_ARTIFACTS}/Scratch"
export PYTHONUNBUFFERED=1

# Dynamic PYTHONPATH resolution
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${REPO_ROOT}"
fi

# Explicit commands are user-authorized shell passthrough. They do not establish
# an electronic-structure profile or scientific qualification.
if [[ $# -gt 0 ]]; then
    "$@"
    exit $?
fi

# Batch science requires an explicit reviewed request, approval, and interpreter.
# No geometry, method, budget, model, or success result is inferred here.
[[ -n "${PYTHON_EXEC:-}" ]] || {
    echo "[CoChem-TORQ] PYTHON_EXEC must name the installed calculation interpreter." >&2
    exit 64
}
[[ "${PYTHON_EXEC}" == /* && -x "${PYTHON_EXEC}" ]] || {
    echo "[CoChem-TORQ] PYTHON_EXEC must be an absolute executable interpreter path." >&2
    exit 64
}
[[ -n "${COCHEM_REQUEST_FILE:-}" && -f "${COCHEM_REQUEST_FILE}" ]] || {
    echo "[CoChem-TORQ] COCHEM_REQUEST_FILE must name an explicit request file." >&2
    exit 64
}
[[ -n "${COCHEM_APPROVED_PLAN_FILE:-}" && -f "${COCHEM_APPROVED_PLAN_FILE}" ]] || {
    echo "[CoChem-TORQ] COCHEM_APPROVED_PLAN_FILE must name the reviewed approval." >&2
    exit 64
}
JOB_ID="${SLURM_JOB_ID:-$$}_task_${SLURM_ARRAY_TASK_ID:-1}"
"${PYTHON_EXEC}" -m cochem_torq.cli execute \
    --request "${COCHEM_REQUEST_FILE}" \
    --approved-plan "${COCHEM_APPROVED_PLAN_FILE}" \
    --output-dir "${COCHEM_ARTIFACTS}/Outputs/${JOB_ID}"
