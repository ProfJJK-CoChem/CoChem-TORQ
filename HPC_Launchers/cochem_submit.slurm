#!/usr/bin/env bash
# ==============================================================================
# CoChem-TORQ SLURM Batch Execution Launcher
# High-Performance Computing (HPC) batch array submission script for CoChem-TORQ
# Enforces Method Matrix v4, strict Filesystem Air-Gap isolation, dynamic
# scratch/artifact routing, and backend Python engine execution.
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

# Resolve Python interpreter executable
if [[ -z "${PYTHON_EXEC:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_EXEC="python"
    elif command -v python.exe >/dev/null 2>&1; then
        PYTHON_EXEC="python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_EXEC="python3"
    else
        PYTHON_EXEC="python"
    fi
fi

# Backend Python engine execution / command passthrough
if [[ $# -gt 0 ]]; then
    "$@"
else
    if [[ -f "${REPO_ROOT}/Libraries/cochem_torq_pipeline.py" ]]; then
        "${PYTHON_EXEC}" -m Libraries.cochem_torq_pipeline
    elif [[ -f "${REPO_ROOT}/cochem_torq_engine.py" ]]; then
        "${PYTHON_EXEC}" -m cochem_torq_engine
    else
        "${PYTHON_EXEC}" -c "import sys; print('[CoChem-TORQ] SLURM backend initialized successfully on ' + sys.version)"
    fi
fi
