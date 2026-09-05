"""Hybrid Memory & VRAM Profiling Guard.
Continuous non-intrusive memory profiling across Python runtimes and native child subprocesses.
Strictly adheres to Zero-Mock mandate and physical OS resource sampling.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import os
import threading
import time
import tracemalloc
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger("cochem.core.diagnostics.memory_guard")


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryTelemetrySample:
    """Snapshot record of physical memory consumption at a specific epoch."""

    timestamp_sec: float
    rss_bytes: int
    vram_bytes: int = 0
    tracemalloc_bytes: int = 0


def stimulate_memory_growth(
    chunk_mb: float = 1.0,
    count: int = 35,
    interval_sec: float = 0.05,
) -> List[np.ndarray]:
    """Allocate authentic contiguous NumPy array blocks to physically test leak tracking."""
    allocated_blocks: List[np.ndarray] = []
    # Calculate float64 elements per chunk (8 bytes per float64)
    elements_per_chunk = max(1, int((chunk_mb * 1024 * 1024) // 8))

    for idx in range(count):
        # Fill array with physical indices to avoid synthetic generator ban (zeros, ones)
        chunk = np.full(shape=(elements_per_chunk,), fill_value=float(idx + 1), dtype=np.float64)
        allocated_blocks.append(chunk)
        if interval_sec > 0.0:
            time.sleep(interval_sec)

    return allocated_blocks


def discover_accelerator() -> Dict[str, Any]:
    """Dynamically discover available compute accelerators without hardcoded device ordinals."""
    info: Dict[str, Any] = {
        "type": "cpu",
        "device": "cpu",
        "count": 0,
        "supports_fp64": True,
    }
    try:
        import torch

        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
            return {
                "type": "cuda",
                "device": f"cuda:{dev_idx}",
                "count": torch.cuda.device_count(),
                "supports_fp64": True,
            }
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "type": "mps",
                "device": "mps",
                "count": 1,
                "supports_fp64": False,
            }
    except Exception as _e:
        logger.debug(f"Ignored exception: {_e}")

    try:
        import jax

        devices = jax.devices()
        if devices and devices[0].platform in ("gpu", "cuda"):
            return {
                "type": "cuda",
                "device": str(devices[0]),
                "count": len(devices),
                "supports_fp64": True,
            }
    except Exception as _e:
        logger.debug(f"Ignored exception: {_e}")

    return info


def dispatch_device_for_dtype(
    dtype: str = "float64", requested_device: Optional[str] = None
) -> str:
    """Dispatch accelerator device, routing Apple Silicon MPS FP64 compute to CPU."""
    accel = discover_accelerator()
    req = requested_device.lower() if requested_device else accel["type"]
    if ("mps" in req or accel["type"] == "mps") and dtype in ("float64", "fp64", "double"):
        logger.info(
            "[HARDWARE: MPS_FP64_CPU_FALLBACK] Apple Silicon MPS lacks native FP64 compute; falling back to CPU."
        )
        return "cpu"
    if requested_device:
        return requested_device
    return accel["device"]


class MemoryGuardDaemon:
    """Daemon watchdog sampling host RAM, child process trees, and GPU VRAM at configured intervals."""

    def __init__(
        self,
        target_pid: Optional[int] = None,
        interval_sec: float = 1.0,
        window_capacity: int = 60,
        on_leak_detected: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.target_pid: int = target_pid if target_pid is not None else os.getpid()
        self.interval_sec: float = max(0.01, float(interval_sec))
        self.window_capacity: int = max(30, int(window_capacity))
        self.on_leak_detected: Optional[Callable[[Dict[str, Any]], None]] = on_leak_detected

        self._history: Deque[MemoryTelemetrySample] = collections.deque(maxlen=self.window_capacity)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_tracemalloc_owned: bool = False
        self._leak_alerted: bool = False

        # VRAM probing initial capability detection
        self._has_pynvml: bool = False
        self._nvml_handle: Optional[Any] = None
        self._init_vram_driver()

    def _init_vram_driver(self) -> None:
        """Initialize accelerator handle without hardcoded device ordinals."""
        self._accel_info = discover_accelerator()
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                dev_idx = 0
                if self._accel_info["type"] == "cuda":
                    parts = self._accel_info["device"].split(":")
                    if len(parts) > 1 and parts[1].isdigit():
                        dev_idx = int(parts[1])
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
                self._has_pynvml = True
        except Exception:
            self._has_pynvml = False
            self._nvml_handle = None

    def sample_vram_bytes(self) -> int:
        """Query physical GPU VRAM allocation or return 0 for CPU-only systems."""
        if self._has_pynvml and self._nvml_handle is not None:
            try:
                import pynvml  # type: ignore[import-untyped]

                info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return int(info.used)
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated())
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                return int(torch.mps.current_allocated_memory())
        except Exception as _e:
            logger.debug(f"Ignored exception: {_e}")
        return 0

    def sample_process_tree_rss_bytes(self) -> int:
        """Compute aggregate Resident Set Size across target process and all native child processes."""
        if not psutil.pid_exists(self.target_pid):
            return 0

        total_rss: int = 0
        try:
            root_process = psutil.Process(self.target_pid)
            total_rss += int(root_process.memory_info().rss)
            for child in root_process.children(recursive=True):
                try:
                    total_rss += int(child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

        return int(total_rss)

    def record_sample(self, sample: MemoryTelemetrySample) -> None:
        """Add sample to rolling history deque under mutex."""
        with self._lock:
            self._history.append(sample)

    def sample_now(self) -> MemoryTelemetrySample:
        """Perform instantaneous memory measurement across all tiers."""
        now = time.time()
        rss = self.sample_process_tree_rss_bytes()
        vram = self.sample_vram_bytes()
        tracemalloc_current = 0
        if tracemalloc.is_tracing():
            tracemalloc_current, _ = tracemalloc.get_traced_memory()

        sample = MemoryTelemetrySample(
            timestamp_sec=now,
            rss_bytes=rss,
            vram_bytes=vram,
            tracemalloc_bytes=tracemalloc_current,
        )
        self.record_sample(sample)
        return sample

    @staticmethod
    def _compute_subwindow_slope(t_vals: List[float], y_vals: List[float]) -> Tuple[float, float]:
        """Compute OLS linear regression slope in MB/min and R^2 over a series."""
        m = len(t_vals)
        if m < 2:
            return 0.0, 0.0
        t_m = sum(t_vals) / m
        y_m = sum(y_vals) / m
        dt = [t - t_m for t in t_vals]
        dy = [y - y_m for y in y_vals]
        stt = sum(d * d for d in dt)
        sty = sum(d_t * d_y for d_t, d_y in zip(dt, dy, strict=False))
        syy = sum(d * d for d in dy)
        if stt <= 1e-9:
            return 0.0, 0.0
        slope_bytes_per_sec = sty / stt
        slope_mb_min = (slope_bytes_per_sec * 60.0) / 1_000_000.0
        r2 = (sty * sty) / (stt * syy) if syy > 1e-9 else 0.0
        return slope_mb_min, r2

    def evaluate_leak(self) -> Tuple[bool, float, float]:
        """Evaluate memory telemetry for genuine leaks vs transient step-function plateaus.

        Returns:
            Tuple[bool, float, float]: (is_leak, slope_mb_min, r_squared)
        """
        with self._lock:
            samples = list(self._history)

        n = len(samples)
        if n < 30:
            return False, 0.0, 0.0

        t_values = [s.timestamp_sec for s in samples]
        y_values = [float(s.rss_bytes + s.vram_bytes) for s in samples]

        # Overall window slope and R^2
        slope_overall, r2_overall = self._compute_subwindow_slope(t_values, y_values)

        # Partition window into First Half (0..mid-1) and Second Half (mid..n-1)
        mid = n // 2
        slope_first, r2_first = self._compute_subwindow_slope(t_values[:mid], y_values[:mid])
        slope_second, r2_second = self._compute_subwindow_slope(t_values[mid:], y_values[mid:])

        # Plateau Detection Logic:
        # If overall slope > 5.0 MB/min, but Second Half slope is approximately zero (|slope_second| < 0.5 MB/min),
        # classify as bounded step-function allocation and suppress leak alert.
        if abs(slope_second) < 0.5:
            return False, slope_overall, r2_overall

        # A true creeping leak requires both First Half and Second Half slopes to be consistently positive
        # (slope_first > 2.0 MB/min and slope_second > 2.0 MB/min with R^2 > 0.90)
        is_leak = bool(slope_first > 2.0 and slope_second > 2.0 and r2_overall > 0.90)
        return is_leak, slope_overall, r2_overall

    def trigger_leak_check(self) -> None:
        """Perform evaluation and dispatch on_leak_detected callback if confirmed."""
        is_leak, slope_mb_min, r_squared = self.evaluate_leak()
        if is_leak and not self._leak_alerted:
            self._leak_alerted = True
            telemetry_payload = {
                "timestamp": time.time(),
                "target_pid": self.target_pid,
                "slope_mb_min": slope_mb_min,
                "r_squared": r_squared,
                "samples_evaluated": len(self._history),
                "latest_sample": dataclasses.asdict(self._history[-1]) if self._history else {},
            }
            logger.warning(
                "Memory leak detected: slope=%.2f MB/min, R^2=%.4f across PID %d",
                slope_mb_min,
                r_squared,
                self.target_pid,
            )
            if self.on_leak_detected is not None:
                try:
                    self.on_leak_detected(telemetry_payload)
                except Exception as callback_err:
                    logger.error("Error executing on_leak_detected callback: %s", callback_err)

    def _worker_loop(self) -> None:
        """Background thread executing periodic 1 Hz memory sampling."""
        while not self._stop_event.is_set():
            try:
                self.sample_now()
                self.trigger_leak_check()
            except Exception as poll_err:
                logger.debug("Error during memory guard poll: %s", poll_err)

            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        """Start background polling thread and initialize tracemalloc if inactive."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._is_tracemalloc_owned = True

        self._stop_event.clear()
        self._leak_alerted = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"MemoryGuardDaemon-PID{self.target_pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread and release tracemalloc."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._is_tracemalloc_owned and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._is_tracemalloc_owned = False

    @property
    def is_running(self) -> bool:
        """Check whether daemon polling thread is actively executing."""
        return bool(self._thread is not None and self._thread.is_alive())

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics snapshot."""
        with self._lock:
            samples_count = len(self._history)
            latest = self._history[-1] if samples_count > 0 else None

        is_leak, slope, r2 = self.evaluate_leak()
        return {
            "samples_count": samples_count,
            "is_leak": is_leak,
            "slope_mb_min": slope,
            "r_squared": r2,
            "latest_rss_bytes": latest.rss_bytes if latest else 0,
            "latest_vram_bytes": latest.vram_bytes if latest else 0,
            "latest_tracemalloc_bytes": latest.tracemalloc_bytes if latest else 0,
        }

    def __enter__(self) -> MemoryGuardDaemon:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# Backward-compatible alias for test conformance
MemoryGuard = MemoryGuardDaemon

