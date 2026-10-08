#!/usr/bin/env bash
# Optional NVIDIA MPS launcher. Canonical student calculations use GitHub Actions.
# No CPU substitute is available: an observed CUDA device and native MPS control
# are prerequisites. Only newly created, private control directories are owned.
set -euo pipefail
umask 077

fail() {
    echo "[CoChem-MPS] $*" >&2
    exit 69
}

[[ $# -gt 0 ]] || fail "An explicit worker command is required."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable."
command -v nvidia-cuda-mps-control >/dev/null 2>&1 || fail "Native MPS control is unavailable."
command -v setsid >/dev/null 2>&1 || fail "setsid is required for owned worker cleanup."
command -v timeout >/dev/null 2>&1 || fail "timeout is required for bounded native control."
command -v realpath >/dev/null 2>&1 || fail "realpath is required for directory ownership."
GPU_UUIDS="$(timeout 10s nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)" \
    || fail "The NVIDIA driver cannot query a CUDA device."
[[ -n "${GPU_UUIDS//[[:space:]]/}" ]] || fail "No CUDA devices were observed."

SCRATCH_BASE="${COCHEM_SCRATCH:-${TMPDIR:-/tmp}}"
[[ "${SCRATCH_BASE}" == /* ]] || fail "Scratch must be an absolute path."
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${SCRATCH_BASE}/cochem_mps_pipe_$$}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${SCRATCH_BASE}/cochem_mps_log_$$}"
for owned_directory in "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"; do
    [[ "${owned_directory}" == /* && "${owned_directory}" != / ]] \
        || fail "MPS directories must be absolute private paths."
    [[ ! -e "${owned_directory}" && ! -L "${owned_directory}" ]] \
        || fail "Refusing an existing MPS directory: ${owned_directory}"
done
[[ "${CUDA_MPS_PIPE_DIRECTORY}" != "${CUDA_MPS_LOG_DIRECTORY}" ]] \
    || fail "MPS pipe and log directories must differ."
CUDA_MPS_PIPE_DIRECTORY="$(realpath -m -- "${CUDA_MPS_PIPE_DIRECTORY}")"
CUDA_MPS_LOG_DIRECTORY="$(realpath -m -- "${CUDA_MPS_LOG_DIRECTORY}")"
[[ "${CUDA_MPS_PIPE_DIRECTORY}" != "${CUDA_MPS_LOG_DIRECTORY}" \
    && "${CUDA_MPS_PIPE_DIRECTORY}" != "${CUDA_MPS_LOG_DIRECTORY}/"* \
    && "${CUDA_MPS_LOG_DIRECTORY}" != "${CUDA_MPS_PIPE_DIRECTORY}/"* ]] \
    || fail "MPS pipe and log directories must not overlap."

owned_directories=()
worker_pid=""
daemon_start_attempted=0
cleanup_mps() {
    local result=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${worker_pid}" ]] && kill -0 "${worker_pid}" 2>/dev/null; then
        kill -TERM -- "-${worker_pid}" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "${worker_pid}" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "${worker_pid}" 2>/dev/null; then
            kill -KILL -- "-${worker_pid}" 2>/dev/null || true
        fi
        wait "${worker_pid}" 2>/dev/null || true
    fi
    if [[ ${daemon_start_attempted} -eq 1 ]]; then
        # Control is routed through this launcher's fresh pipe directory. Never
        # enumerate or signal MPS servers belonging to other sessions/users.
        printf 'quit\n' | timeout 5s nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    for owned_directory in "${owned_directories[@]}"; do
        rm -rf -- "${owned_directory}"
    done
    exit "${result}"
}
trap cleanup_mps EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for owned_directory in "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"; do
    mkdir -p -- "$(dirname -- "${owned_directory}")"
    mkdir -- "${owned_directory}"
    owned_directories+=("${owned_directory}")
done
daemon_start_attempted=1
timeout 10s nvidia-cuda-mps-control -d || fail "Native MPS daemon startup failed."
printf 'get_server_list\n' | timeout 5s nvidia-cuda-mps-control >/dev/null 2>&1 \
    || fail "The private MPS control endpoint is unavailable."
setsid -- "$@" &
worker_pid=$!
wait "${worker_pid}"
