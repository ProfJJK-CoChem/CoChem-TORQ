"""Hybrid Architecture Aware Thread & Core Tuner.
Dynamically budgets CPU and GPU resources according to Method Matrix v4 §8A contention budgeting.
Strictly adheres to Zero-Mock mandate and physical OS topology discovery.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger("cochem.core.hardware.topology")


@dataclasses.dataclass(slots=True, frozen=True)
class HardwareTopology:
    """Immutable hardware topology and core budgeting specification."""

    total_logical_cpus: int
    total_physical_cores: int
    p_cores: int
    e_cores: int
    resource_ceiling: int
    scout_cores: int
    anchor_cores: int
    gpu_mps_workers: int
    environment_variables: Dict[str, str]


class TopologyDiscoveryEngine:
    """Hardware discovery and Scout-and-Anchor budgeting engine."""

    def __init__(self) -> None:
        self._cached_topology: Optional[HardwareTopology] = None

    def discover_p_e_cores(self) -> Tuple[int, int]:
        """Distinguish Intel Performance (P) cores from Efficient (E) cores."""
        logical_total = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        physical_total = psutil.cpu_count(logical=False) or max(1, logical_total // 2)

        # 1. Windows NT: Query GetLogicalProcessorInformationEx via Win32 API
        if sys.platform == "win32":
            try:
                # RelationProcessorCore = 0
                class PROCESSOR_RELATIONSHIP(ctypes.Structure):
                    _fields_ = [
                        ("Flags", ctypes.c_ubyte),
                        ("EfficiencyClass", ctypes.c_ubyte),
                        ("Reserved", ctypes.c_ubyte * 20),
                        ("GroupCount", ctypes.c_ushort),
                        ("GroupMask", ctypes.c_size_t * 1),
                    ]

                class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
                    _fields_ = [
                        ("Relationship", ctypes.c_uint),
                        ("Size", ctypes.c_ulong),
                        ("Processor", PROCESSOR_RELATIONSHIP),
                    ]

                buffer_size = ctypes.c_ulong(0)
                # First call to query required buffer size
                ctypes.windll.kernel32.GetLogicalProcessorInformationEx(
                    0,  # RelationProcessorCore
                    None,
                    ctypes.byref(buffer_size),
                )

                if buffer_size.value > 0:
                    buffer = (ctypes.c_byte * buffer_size.value)()
                    res = ctypes.windll.kernel32.GetLogicalProcessorInformationEx(
                        0,
                        ctypes.byref(buffer),
                        ctypes.byref(buffer_size),
                    )
                    if res != 0:
                        offset = 0
                        p_count = 0
                        e_count = 0
                        while offset < buffer_size.value:
                            info = ctypes.cast(
                                ctypes.byref(buffer, offset),
                                ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
                            ).contents
                            if info.Relationship == 0:  # RelationProcessorCore
                                eff_class = info.Processor.EfficiencyClass
                                if eff_class > 0:
                                    p_count += 1
                                else:
                                    e_count += 1
                            if info.Size == 0:
                                break
                            offset += info.Size

                        if (p_count + e_count) > 0:
                            return max(1, p_count), e_count
            except Exception as win_err:
                logger.debug("Windows processor information query bypassed: %s", win_err)

        # 2. Linux: Parse sysfs cpufreq
        if sys.platform.startswith("linux"):
            try:
                cpu_dir = pathlib.Path("/sys/devices/system/cpu")
                freqs: List[int] = []
                for p in cpu_dir.glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
                    try:
                        f_val = int(p.read_text().strip())
                        freqs.append(f_val)
                    except (OSError, ValueError):
                        continue
                if freqs and len(set(freqs)) > 1:
                    max_freq = max(freqs)
                    p_count = sum(1 for f in freqs if f == max_freq)
                    e_count = len(freqs) - p_count
                    return max(1, p_count), e_count
            except Exception as linux_err:
                logger.debug("Linux cpufreq query bypassed: %s", linux_err)

        # 3. Darwin (Apple Silicon): Query sysctl for perflevel0 (P-cores) and perflevel1 (E-cores)
        if sys.platform == "darwin":
            try:
                res0 = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                res1 = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                p_val = int(res0.stdout.strip()) if res0.returncode == 0 and res0.stdout.strip().isdigit() else 0
                e_val = int(res1.stdout.strip()) if res1.returncode == 0 and res1.stdout.strip().isdigit() else 0
                if p_val > 0 or e_val > 0:
                    return max(1, p_val), e_val
            except Exception as darwin_err:
                logger.debug("Darwin sysctl query bypassed: %s", darwin_err)

        # Uniform fallback
        return physical_total, 0

    def resolve_resource_ceiling(self) -> int:
        """Resolve effective CPU ceiling from Slurm, Cgroups v1/v2, affinity, or physical cores."""
        # 1. Slurm environment variable
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                val = int(slurm_cpus)
                if val >= 1:
                    return val
            except ValueError as _e:
                logger.debug(f"Ignored exception: {_e}")

        # 2. Linux Cgroups v2: /sys/fs/cgroup/cpu.max (quota period)
        cgroup_v2 = pathlib.Path("/sys/fs/cgroup/cpu.max")
        if cgroup_v2.exists():
            try:
                parts = cgroup_v2.read_text().strip().split()
                if len(parts) >= 2 and parts[0] != "max":
                    quota = float(parts[0])
                    period = float(parts[1])
                    if period > 0:
                        cores = int(quota / period)
                        if cores >= 1:
                            return cores
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")

        # 3. Linux Cgroups v1: cpu.cfs_quota_us / cpu.cfs_period_us
        cgroup_v1_quota = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        cgroup_v1_period = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if cgroup_v1_quota.exists() and cgroup_v1_period.exists():
            try:
                quota_val = float(cgroup_v1_quota.read_text().strip())
                period_val = float(cgroup_v1_period.read_text().strip())
                if quota_val > 0 and period_val > 0:
                    cores = int(quota_val / period_val)
                    if cores >= 1:
                        return cores
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")

        # 4. POSIX process affinity
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity_cores = len(os.sched_getaffinity(0))
                if affinity_cores >= 1:
                    return affinity_cores
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")

        # 5. Physical cores fallback
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        return max(1, physical)

    def discover_topology(self, concurrent_workers: int = 1) -> HardwareTopology:
        """Calculate Scout-and-Anchor core budget and thread environment variables."""
        logical_total = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        physical_total = psutil.cpu_count(logical=False) or max(1, logical_total // 2)
        ceiling = self.resolve_resource_ceiling()
        p_cores, e_cores = self.discover_p_e_cores()

        # Clamp P-cores to available ceiling
        effective_p_cores = max(1, min(p_cores, ceiling))

        # Scout-and-Anchor Concurrency Budget (Method Matrix v4 §8A):
        # Scout Core: 1 physical P-core dedicated to host orchestration and MACE ML
        # Anchor Cores: Remaining (N_P-cores - 1) cores across QC subprocesses (ORCA/CFOUR)
        if effective_p_cores > 2:
            scout_cores = 1
            anchor_cores = effective_p_cores - 1
        elif effective_p_cores == 2:
            scout_cores = 1
            anchor_cores = 1
        else:
            scout_cores = 1
            anchor_cores = 0

        # Worker thread injection values with dynamic contention budgeting
        budget_pool = anchor_cores if anchor_cores > 0 else scout_cores
        budgeted_threads = max(1, budget_pool // max(1, concurrent_workers))
        env_vars = {
            "OMP_NUM_THREADS": str(budgeted_threads),
            "MKL_NUM_THREADS": str(budgeted_threads),
            "OPENBLAS_NUM_THREADS": str(budgeted_threads),
            "VECLIB_MAXIMUM_THREADS": str(budgeted_threads),
            "NUMEXPR_NUM_THREADS": str(budgeted_threads),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(max(1, 100 // max(1, concurrent_workers))),
        }

        # GPU MPS Worker Ceiling: 2 to 4 concurrent processes
        gpu_mps_workers = max(2, min(4, ceiling))

        topology = HardwareTopology(
            total_logical_cpus=logical_total,
            total_physical_cores=physical_total,
            p_cores=effective_p_cores,
            e_cores=e_cores,
            resource_ceiling=ceiling,
            scout_cores=scout_cores,
            anchor_cores=anchor_cores,
            gpu_mps_workers=gpu_mps_workers,
            environment_variables=env_vars,
        )
        self._cached_topology = topology
        return topology

    def get_worker_env(
        self,
        concurrent_workers: int = 1,
        worker_index: int = 0,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Construct subprocess execution environment with dynamically budgeted thread variables."""
        topo = self.discover_topology(concurrent_workers=concurrent_workers)
        base_env = dict(os.environ)
        base_env.update(topo.environment_variables)

        # Dynamic host thread budgeting per worker (Method Matrix §8A)
        budget_pool = topo.anchor_cores if topo.anchor_cores > 0 else topo.scout_cores
        budgeted_threads = max(1, budget_pool // max(1, concurrent_workers))
        base_env["OMP_NUM_THREADS"] = str(budgeted_threads)
        base_env["MKL_NUM_THREADS"] = str(budgeted_threads)
        base_env["OPENBLAS_NUM_THREADS"] = str(budgeted_threads)
        base_env["VECLIB_MAXIMUM_THREADS"] = str(budgeted_threads)
        base_env["NUMEXPR_NUM_THREADS"] = str(budgeted_threads)

        # Zero-CUDA-Locking Directive: Non-locking MPS GPU apportionment
        base_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, 100 // max(1, concurrent_workers)))

        # Zero-CUDA-Locking Directive (§8A, §19): Non-initializing GPU discovery [M]
        available_gpus = self.get_available_gpus()
        num_gpus = len(available_gpus)
        if num_gpus > 0:
            assigned_gpu = available_gpus[worker_index % num_gpus]
            base_env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        else:
            base_env["CUDA_VISIBLE_DEVICES"] = ""

        if extra_env is not None:
            base_env.update(extra_env)
        return base_env

    def get_available_gpus(self) -> List[int]:
        """Return list of zero-indexed GPU identifiers available without initializing CUDA runtime."""
        count = self._discover_gpu_count()
        return list(range(count))

    @staticmethod
    def _discover_gpu_count() -> int:
        """Non-initializing GPU count query guaranteeing zero CUDA runtime lock in parent (§8A, §19) [M]."""

        # Tier 1: NVML query
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return int(count)
        except Exception as exc:
            logger.debug(f"NVML GPU query unavailable: {exc}")

        # Tier 2: nvidia-smi CLI subprocess query
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
                if lines and lines[0].isdigit():
                    return int(lines[0])
        except Exception as exc:
            logger.debug(f"nvidia-smi GPU query unavailable: {exc}")

        return 0

    def pin_scout_affinity(self, core_index: int = 0) -> bool:
        """Bind host orchestration process to specific core index to avoid thread migration.

        Supports Windows multi-group affinity for >64 logical cores via SetThreadGroupAffinity (§19) [M].
        """
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {core_index})
                return True
            elif sys.platform == "win32":
                total_cpus = os.cpu_count() or 1
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.GetCurrentThread.restype = ctypes.c_void_p
                k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
                k32.SetProcessAffinityMask.restype = ctypes.c_int

                if core_index >= 64 or total_cpus > 64:
                    class GROUP_AFFINITY(ctypes.Structure):
                        _fields_ = [
                            ("Mask", ctypes.c_size_t),
                            ("Group", ctypes.c_ushort),
                            ("Reserved", ctypes.c_ushort * 3),
                        ]

                    group = int(core_index) // 64
                    core_in_group = int(core_index) % 64
                    mask = 1 << core_in_group

                    ga = GROUP_AFFINITY()
                    ga.Group = group
                    ga.Mask = mask
                    prev_ga = GROUP_AFFINITY()

                    k32.SetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                    k32.SetThreadGroupAffinity.restype = ctypes.c_int
                    thread_handle = k32.GetCurrentThread()
                    res = k32.SetThreadGroupAffinity(
                        thread_handle,
                        ctypes.byref(ga),
                        ctypes.byref(prev_ga),
                    )
                    return bool(res != 0)
                else:
                    mask = 1 << max(0, int(core_index))
                    handle = k32.GetCurrentProcess()
                    res = k32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
                    return bool(res != 0)
        except Exception as pin_err:
            logger.debug("Affinity pinning error on core %d: %s", core_index, pin_err)
            return False
        return False


# Architectural alias
HardwareTopologyEngine = TopologyDiscoveryEngine

