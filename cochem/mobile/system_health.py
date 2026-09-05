"""Real-Time Mobile System Health Hardware Monitor.

Module: cochem.mobile.system_health
Authoritative Reference: SRS Chunk 02 BASE UI & Web (Part 2), Prompt 4.

Provides asynchronous, non-blocking hardware saturation telemetry for local workstations
and HPC compute nodes:
1. Real-time GPU monitoring (VRAM allocation, compute utilization, core temperature) via
   pynvml or torch.cuda with headless CPU fallback.
2. Real-time CPU monitoring (per-core load percentage, total physical/logical counts) via psutil.
3. System memory & swap allocation tracking (total, used, free, percentage).
4. Process handle / file descriptor tracking.
5. Thermal throttling and saturation alert engine:
   - GPU Temperature: Warning at 80°C, Critical at 85°C.
   - VRAM Allocation: Warning at 90%, Critical at 95%.
   - RAM Utilization: Warning at 90%, Critical at 95%.
   - CPU Utilization: Warning at 95%.
6. Asynchronous streaming via stream_telemetry() generator and synchronous poll_telemetry().
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, List, Optional, Tuple

import psutil
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# GPU thermal and memory alert thresholds
GPU_TEMP_WARNING_C: float = 80.0
GPU_TEMP_CRITICAL_C: float = 85.0
VRAM_PERCENT_WARNING: float = 90.0
VRAM_PERCENT_CRITICAL: float = 95.0
RAM_PERCENT_WARNING: float = 90.0
RAM_PERCENT_CRITICAL: float = 95.0
CPU_PERCENT_WARNING: float = 95.0


class AlertSeverity(str, Enum):
    """Severity classification for hardware telemetry alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class HealthAlert(BaseModel):
    """Pydantic model representing an active hardware threshold warning or critical alarm."""

    model_config = ConfigDict(frozen=True)

    severity: AlertSeverity
    component: str
    message: str
    metric_value: float
    threshold_value: float


class GPUMetrics(BaseModel):
    """Hardware saturation metrics for a physical GPU device."""

    model_config = ConfigDict(frozen=True)

    device_index: int
    device_name: str
    temperature_celsius: Optional[float] = None
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    vram_percent: float
    utilization_percent: Optional[float] = None


class CPUMetrics(BaseModel):
    """Multi-core processor workload and thread utilization metrics."""

    model_config = ConfigDict(frozen=True)

    physical_cores: int
    logical_cores: int
    overall_percent: float
    per_core_percent: List[float]


class MemoryMetrics(BaseModel):
    """Volatile RAM and swap memory allocation metrics."""

    model_config = ConfigDict(frozen=True)

    ram_total_bytes: int
    ram_used_bytes: int
    ram_free_bytes: int
    ram_percent: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_free_bytes: int
    swap_percent: float


class ProcessMetrics(BaseModel):
    """Current Python process footprint and OS handle consumption."""

    model_config = ConfigDict(frozen=True)

    pid: int
    rss_bytes: int
    rss_mb: float
    open_handles_or_fds: int


class HardwareTelemetryFrame(BaseModel):
    """Single complete frame of multi-sensor hardware saturation telemetry."""

    model_config = ConfigDict(frozen=True)

    frame_id: str
    timestamp_utc: str
    cpu: CPUMetrics
    memory: MemoryMetrics
    process: ProcessMetrics
    gpu_available: bool
    gpus: List[GPUMetrics]
    alerts: List[HealthAlert]
    status: str


class SystemHealthMonitor:
    """Non-blocking hardware health and resource saturation monitor."""

    def __init__(
        self,
        gpu_temp_warning_c: float = GPU_TEMP_WARNING_C,
        gpu_temp_critical_c: float = GPU_TEMP_CRITICAL_C,
        vram_percent_warning: float = VRAM_PERCENT_WARNING,
        vram_percent_critical: float = VRAM_PERCENT_CRITICAL,
        ram_percent_warning: float = RAM_PERCENT_WARNING,
        ram_percent_critical: float = RAM_PERCENT_CRITICAL,
        cpu_percent_warning: float = CPU_PERCENT_WARNING,
    ) -> None:
        self.gpu_temp_warning_c = gpu_temp_warning_c
        self.gpu_temp_critical_c = gpu_temp_critical_c
        self.vram_percent_warning = vram_percent_warning
        self.vram_percent_critical = vram_percent_critical
        self.ram_percent_warning = ram_percent_warning
        self.ram_percent_critical = ram_percent_critical
        self.cpu_percent_warning = cpu_percent_warning

        self._pynvml_available: bool = False
        self._init_pynvml()

    def _init_pynvml(self) -> None:
        """Attempt to initialize NVIDIA Management Library (NVML)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
        except Exception:
            self._pynvml_available = False

    def poll_gpu_metrics(self) -> Tuple[bool, List[GPUMetrics]]:
        """Query GPU VRAM, temperature, and utilization via NVML or torch.cuda fallback."""
        gpu_list: List[GPUMetrics] = []

        if self._pynvml_available:
            try:
                import pynvml
                device_count = pynvml.nvmlDeviceGetCount()
                for idx in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name_bytes = pynvml.nvmlDeviceGetName(handle)
                    dev_name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_total = int(mem_info.total)
                    vram_used = int(mem_info.used)
                    vram_free = int(mem_info.free)
                    vram_pct = round((vram_used / vram_total) * 100.0, 2) if vram_total > 0 else 0.0

                    try:
                        temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp_c = None

                    try:
                        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        util_pct = float(util_rates.gpu)
                    except Exception:
                        util_pct = None

                    gpu_list.append(
                        GPUMetrics(
                            device_index=idx,
                            device_name=dev_name,
                            temperature_celsius=temp_c,
                            vram_total_bytes=vram_total,
                            vram_used_bytes=vram_used,
                            vram_free_bytes=vram_free,
                            vram_percent=vram_pct,
                            utilization_percent=util_pct,
                        )
                    )
                return True, gpu_list
            except Exception as exc:
                logger.debug(f"NVML polling error: {exc}. Falling back to torch.cuda.")

        # Fallback to torch.cuda if available
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                for idx in range(count):
                    dev_name = torch.cuda.get_device_name(idx)
                    props = torch.cuda.get_device_properties(idx)
                    vram_total = int(props.total_memory)
                    vram_used = int(torch.cuda.memory_allocated(idx))
                    vram_free = max(0, vram_total - vram_used)
                    vram_pct = round((vram_used / vram_total) * 100.0, 2) if vram_total > 0 else 0.0

                    gpu_list.append(
                        GPUMetrics(
                            device_index=idx,
                            device_name=dev_name,
                            temperature_celsius=None,
                            vram_total_bytes=vram_total,
                            vram_used_bytes=vram_used,
                            vram_free_bytes=vram_free,
                            vram_percent=vram_pct,
                            utilization_percent=None,
                        )
                    )
                return True, gpu_list
        except Exception as exc:
            logger.debug(f"torch.cuda polling fallback failed: {exc}")

        return False, []

    def poll_cpu_metrics(self) -> CPUMetrics:
        """Query CPU utilization, per-core percent, and physical/logical core counts."""
        phys_cores = psutil.cpu_count(logical=False) or 1
        logic_cores = psutil.cpu_count(logical=True) or 1
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        overall = psutil.cpu_percent(interval=None, percpu=False)

        return CPUMetrics(
            physical_cores=phys_cores,
            logical_cores=logic_cores,
            overall_percent=float(overall),
            per_core_percent=[float(p) for p in per_core],
        )

    def poll_memory_metrics(self) -> MemoryMetrics:
        """Query physical RAM and swap memory metrics."""
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        return MemoryMetrics(
            ram_total_bytes=int(vm.total),
            ram_used_bytes=int(vm.used),
            ram_free_bytes=int(vm.free),
            ram_percent=float(vm.percent),
            swap_total_bytes=int(sw.total),
            swap_used_bytes=int(sw.used),
            swap_free_bytes=int(sw.free),
            swap_percent=float(sw.percent),
        )

    def poll_process_metrics(self) -> ProcessMetrics:
        """Query memory and handle count for current process."""
        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss = int(mem_info.rss)
        rss_mb = round(rss / (1024 * 1024), 2)

        # Cross-platform handle / file descriptor retrieval
        handles_count = 0
        if hasattr(proc, "num_handles"):
            try:
                handles_count = int(proc.num_handles())
            except Exception:
                handles_count = 0
        elif hasattr(proc, "num_fds"):
            try:
                handles_count = int(proc.num_fds())
            except Exception:
                handles_count = 0

        return ProcessMetrics(
            pid=os.getpid(),
            rss_bytes=rss,
            rss_mb=rss_mb,
            open_handles_or_fds=handles_count,
        )

    def evaluate_alerts(
        self,
        cpu: CPUMetrics,
        memory: MemoryMetrics,
        gpus: List[GPUMetrics],
    ) -> List[HealthAlert]:
        """Evaluate saturation and thermal thresholds across all components."""
        alerts: List[HealthAlert] = []

        # GPU Alerts
        for g in gpus:
            # Temperature checks
            if g.temperature_celsius is not None:
                if g.temperature_celsius >= self.gpu_temp_critical_c:
                    alerts.append(
                        HealthAlert(
                            severity=AlertSeverity.CRITICAL,
                            component=f"GPU_{g.device_index}_TEMPERATURE",
                            message=f"GPU {g.device_index} core temperature is {g.temperature_celsius:.1f}°C (CRITICAL >= {self.gpu_temp_critical_c}°C). Throttling imminent.",
                            metric_value=g.temperature_celsius,
                            threshold_value=self.gpu_temp_critical_c,
                        )
                    )
                elif g.temperature_celsius >= self.gpu_temp_warning_c:
                    alerts.append(
                        HealthAlert(
                            severity=AlertSeverity.WARNING,
                            component=f"GPU_{g.device_index}_TEMPERATURE",
                            message=f"GPU {g.device_index} core temperature is {g.temperature_celsius:.1f}°C (WARNING >= {self.gpu_temp_warning_c}°C).",
                            metric_value=g.temperature_celsius,
                            threshold_value=self.gpu_temp_warning_c,
                        )
                    )

            # VRAM allocation checks
            if g.vram_percent >= self.vram_percent_critical:
                alerts.append(
                    HealthAlert(
                        severity=AlertSeverity.CRITICAL,
                        component=f"GPU_{g.device_index}_VRAM",
                        message=f"GPU {g.device_index} VRAM allocation is {g.vram_percent:.1f}% (CRITICAL >= {self.vram_percent_critical}%). Risk of CUDA OOM.",
                        metric_value=g.vram_percent,
                        threshold_value=self.vram_percent_critical,
                    )
                )
            elif g.vram_percent >= self.vram_percent_warning:
                alerts.append(
                    HealthAlert(
                        severity=AlertSeverity.WARNING,
                        component=f"GPU_{g.device_index}_VRAM",
                        message=f"GPU {g.device_index} VRAM allocation is {g.vram_percent:.1f}% (WARNING >= {self.vram_percent_warning}%).",
                        metric_value=g.vram_percent,
                        threshold_value=self.vram_percent_warning,
                    )
                )

        # RAM Alerts
        if memory.ram_percent >= self.ram_percent_critical:
            alerts.append(
                HealthAlert(
                    severity=AlertSeverity.CRITICAL,
                    component="HOST_RAM",
                    message=f"System RAM utilization is {memory.ram_percent:.1f}% (CRITICAL >= {self.ram_percent_critical}%). OOM killer risk.",
                    metric_value=memory.ram_percent,
                    threshold_value=self.ram_percent_critical,
                )
            )
        elif memory.ram_percent >= self.ram_percent_warning:
            alerts.append(
                HealthAlert(
                    severity=AlertSeverity.WARNING,
                    component="HOST_RAM",
                    message=f"System RAM utilization is {memory.ram_percent:.1f}% (WARNING >= {self.ram_percent_warning}%).",
                    metric_value=memory.ram_percent,
                    threshold_value=self.ram_percent_warning,
                )
            )

        # CPU Alerts
        if cpu.overall_percent >= self.cpu_percent_warning:
            alerts.append(
                HealthAlert(
                    severity=AlertSeverity.WARNING,
                    component="HOST_CPU",
                    message=f"Total CPU utilization is {cpu.overall_percent:.1f}% (WARNING >= {self.cpu_percent_warning}%). Host is compute-saturated.",
                    metric_value=cpu.overall_percent,
                    threshold_value=self.cpu_percent_warning,
                )
            )

        return alerts

    def poll_telemetry(self) -> HardwareTelemetryFrame:
        """Collect a synchronous, complete snapshot frame of hardware health telemetry."""
        now_iso = datetime.now(timezone.utc).isoformat()
        frame_uuid = str(uuid.uuid4())

        cpu_data = self.poll_cpu_metrics()
        mem_data = self.poll_memory_metrics()
        proc_data = self.poll_process_metrics()
        gpu_avail, gpu_data = self.poll_gpu_metrics()

        active_alerts = self.evaluate_alerts(cpu_data, mem_data, gpu_data)

        status_str = "HEALTHY"
        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            status_str = "CRITICAL"
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            status_str = "WARNING"

        return HardwareTelemetryFrame(
            frame_id=frame_uuid,
            timestamp_utc=now_iso,
            cpu=cpu_data,
            memory=mem_data,
            process=proc_data,
            gpu_available=gpu_avail,
            gpus=gpu_data,
            alerts=active_alerts,
            status=status_str,
        )

    async def stream_telemetry(
        self,
        interval_seconds: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> AsyncGenerator[HardwareTelemetryFrame, None]:
        """Asynchronously yield telemetry frames at specified interval without blocking loop."""
        frames_emitted = 0
        while max_frames is None or frames_emitted < max_frames:
            frame = self.poll_telemetry()
            yield frame
            frames_emitted += 1
            if max_frames is not None and frames_emitted >= max_frames:
                break
            await asyncio.sleep(interval_seconds)
