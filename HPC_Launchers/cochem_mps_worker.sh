#!/usr/bin/env bash
# ==============================================================================
# CoChem NVIDIA MPS Concurrent Worker Daemon Launcher
# Enables NVIDIA Multi-Process Service (MPS) control daemon with strict Air-Gap
# isolation, dynamic scratch/log routing, and graceful trap termination.
# ==============================================================================

set -euo pipefail

# Dynamic directory resolution for MPS pipe and log directories (Air-Gap compliant)
if [[ -z "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
    if [[ -n "${COCHEM_ARTIFACTS:-}" ]]; then
        export CUDA_MPS_PIPE_DIRECTORY="${COCHEM_ARTIFACTS}/Scratch/mps_pipe"
    else
        export CUDA_MPS_PIPE_DIRECTORY="/tmp/cochem_mps_${UID:-$USER}_$$/pipe"
    fi
else
    export CUDA_MPS_PIPE_DIRECTORY
fi

if [[ -z "${CUDA_MPS_LOG_DIRECTORY:-}" ]]; then
    if [[ -n "${COCHEM_ARTIFACTS:-}" ]]; then
        export CUDA_MPS_LOG_DIRECTORY="${COCHEM_ARTIFACTS}/Logs/mps_log"
    else
        export CUDA_MPS_LOG_DIRECTORY="/tmp/cochem_mps_${UID:-$USER}_$$/log"
    fi
else
    export CUDA_MPS_LOG_DIRECTORY
fi

# Ensure pipe and log directories exist physically
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

# Graceful cleanup handler for MPS daemon
cleanup() {
    local exit_code=$?
    trap - EXIT SIGINT SIGTERM SIGHUP
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "quit" | nvidia-cuda-mps-control 2>/dev/null || true
    fi
    exit "${exit_code}"
}

trap cleanup EXIT SIGINT SIGTERM SIGHUP

# Start NVIDIA MPS control daemon in background
nvidia-cuda-mps-control -d

# Command passthrough or monitor/wait mode
if [[ $# -gt 0 ]]; then
    "$@"
else
    wait
fi
