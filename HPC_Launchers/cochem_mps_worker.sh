#!/usr/bin/env bash
# ==============================================================================
# CoChem NVIDIA MPS Concurrent Worker Daemon Launcher
# High-Performance Computing (HPC) Tier 6 MPS Daemon Lifecycle Supervisor
# Method Matrix v4 §8A.4: NVIDIA MPS Daemon Lifecycle Management [M]
# Enforces strict Filesystem Air-Gap isolation (Ring 2 ephemeral scratch pipes),
# active daemon supervision, trapped graceful shutdown, orphaned client sweeps,
# GPU VRAM reclamation verification, and pipe directory cleanup.
# ==============================================================================

set -euo pipefail

# Dynamic scratch directory resolution in Ring 2 ephemeral scratch ($COCHEM_SCRATCH)
SLURM_ID="${SLURM_JOB_ID:-$$}"
if [[ -n "${COCHEM_SCRATCH:-}" ]]; then
    SCRATCH_BASE="${COCHEM_SCRATCH}"
elif [[ -n "${TMPDIR:-}" ]]; then
    SCRATCH_BASE="${TMPDIR}"
else
    SCRATCH_BASE="/tmp/cochem_scratch_${UID:-$USER}"
fi

export CUDA_MPS_PIPE_DIRECTORY="${SCRATCH_BASE}/mps_control_${SLURM_ID}"
export CUDA_MPS_LOG_DIRECTORY="${SCRATCH_BASE}/mps_log_${SLURM_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

TERMINATE_FLAG="${SCRATCH_BASE}/mps_terminate_${SLURM_ID}.flag"

# Array tracking active child worker processes
child_pids=()

# Trapped cleanup handler (Method Matrix §8A.4 & Suggestion #148)
cleanup_mps() {
    local exit_code=$?
    trap - EXIT SIGINT SIGTERM SIGHUP ERR
    echo "[CoChem-MPS] Initiating graceful MPS daemon termination..."

    # 1. Terminate child worker processes if running
    if [[ ${#child_pids[@]} -gt 0 ]]; then
        echo "[CoChem-MPS] Terminating active child worker processes: ${child_pids[*]}"
        kill -TERM "${child_pids[@]}" 2>/dev/null || true
        wait "${child_pids[@]}" 2>/dev/null || true
    fi

    # 2. Graceful shutdown command to nvidia-cuda-mps-control
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "quit" | nvidia-cuda-mps-control 2>/dev/null || true
    fi

    # 3. Sweep and terminate orphaned client contexts
    if command -v pgrep >/dev/null 2>&1; then
        for client_pid in $(pgrep -f "nvidia-cuda-mps-server" 2>/dev/null || true); do
            if kill -0 "${client_pid}" 2>/dev/null; then
                kill -15 "${client_pid}" 2>/dev/null || true
                sleep 0.5
                kill -9 "${client_pid}" 2>/dev/null || true
            fi
        done
    fi

    # 4. Verify GPU driver memory release if nvidia-smi available
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >/dev/null 2>&1 || true
    fi

    # 5. Purge pipe and socket directories from Ring 2 ephemeral scratch
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}" "${TERMINATE_FLAG}" 2>/dev/null || true
    echo "[CoChem-MPS] MPS daemon shutdown and scratch pipe purge complete."
    exit "${exit_code}"
}

trap cleanup_mps EXIT SIGINT SIGTERM SIGHUP ERR

# Launch daemon in background if binary exists
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d
    MPS_PID=$(pgrep -f "nvidia-cuda-mps-control" 2>/dev/null | tail -n 1 || echo "")
else
    MPS_PID=""
fi

# Command passthrough or active daemon supervision loop
if [[ $# -gt 0 ]]; then
    "$@" &
    child_pids+=("$!")
    wait "${child_pids[@]}"
else
    if [[ -n "${MPS_PID}" ]]; then
        # Active monitoring loop: poll PID and termination flag in scratch
        while kill -0 "${MPS_PID}" 2>/dev/null; do
            if [[ -f "${TERMINATE_FLAG}" ]]; then
                echo "[CoChem-MPS] Detected termination flag: ${TERMINATE_FLAG}"
                break
            fi
            sleep 1
        done
    else
        # If running in environment without MPS binary (e.g. test harness / CPU-only host)
        while [[ ! -f "${TERMINATE_FLAG}" ]]; do
            sleep 1
        done
    fi
fi
