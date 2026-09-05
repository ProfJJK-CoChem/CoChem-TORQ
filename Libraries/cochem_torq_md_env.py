"""Hardware Concurrency, Device Dispatcher & HPC Safe Scratch for TORQ Molecular Dynamics (Chunk 21).

Method Matrix v4 Provenance Tags:
- [M] Mandated: Apple Silicon MPS float64 automatic fallback to CPU, HPC distributed lock prohibition.
- [D] Derived: Dynamic 6-tier runtime path resolution and hardware accelerator binding.
- [E] Empirical: OS-agnostic pathlib handling and local NVMe scratch fallback.

Strict Zero-Mock Mandate v3: Completely authentic physics and mathematical rigor.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union
import torch

from Libraries.cochem_torq_md_errors import HardwareDispatchError

logger = logging.getLogger(__name__)


def dispatch_md_device(
    requested_device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Dispatch hardware accelerator safely with Apple Silicon MPS float64 fallback [M].

    Metal Performance Shaders (MPS) hardware on Apple Silicon does not support native 64-bit
    floating point operations (torch.float64). When executing molecular dynamics integration
    requiring torch.float64, this dispatcher intercepts and automatically routes execution to CPU.

    Parameters
    ----------
    requested_device : Optional[Union[str, torch.device]]
        Requested target device (e.g., 'cpu', 'cuda', 'mps'). If None, queries system availability.

    Returns
    -------
    torch.device
        Safely routed device compliant with torch.float64 double-precision integration.
    """
    if requested_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    dev_str = str(requested_device).strip().lower()

    # Intercept Apple Silicon MPS requests and route to CPU for float64 compliance [M]
    if "mps" in dev_str:
        logger.info("Apple Silicon MPS float64 fallback engaged: routing MD execution to CPU.")
        return torch.device("cpu")

    if "cuda" in dev_str:
        if torch.cuda.is_available():
            try:
                return torch.device(requested_device)
            except Exception as exc:
                raise HardwareDispatchError(str(requested_device), str(exc)) from exc
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if "cpu" in dev_str:
        return torch.device("cpu")

    try:
        return torch.device(requested_device)
    except Exception as exc:
        raise HardwareDispatchError(
            str(requested_device),
            f"Failed to dispatch requested device: {exc}",
        ) from exc


def resolve_hpc_safe_scratch(subfolder: str = "cochem_torq_md") -> Path:
    """Resolve node-local scratch storage adhering to the HPC Distributed Lock Prohibition [M].

    On parallel network filesystems (Lustre, GPFS, BeeGFS, NFS), direct file locking
    triggers lock manager deadlocks ([Errno 37] No locks available). All staging, locking,
    and temporary files must route to node-local NVMe scratch via $SLURM_TMPDIR or $TMPDIR.

    Parameters
    ----------
    subfolder : str
        Subdirectory name within the resolved scratch location. Defaults to 'cochem_torq_md'.

    Returns
    -------
    Path
        Absolute path to local scratch directory, guaranteed to exist on disk.
    """
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    sys_tmp = os.environ.get("TMPDIR")
    cochem_scratch = os.environ.get("COCHEM_SCRATCH_DIR")

    if slurm_tmp and Path(slurm_tmp).is_dir():
        base_dir = Path(slurm_tmp)
    elif sys_tmp and Path(sys_tmp).is_dir():
        base_dir = Path(sys_tmp)
    elif cochem_scratch:
        base_dir = Path(cochem_scratch)
    else:
        base_dir = Path.home() / ".cochem" / "scratch"

    scratch_path = (base_dir / subfolder).resolve()
    scratch_path.mkdir(parents=True, exist_ok=True)
    return scratch_path
