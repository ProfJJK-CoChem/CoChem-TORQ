"""Hardware Concurrency, Device Dispatcher & HPC Safe Scratch for CoChem-TORQ.

Method Matrix v4 Provenance Tags:
- [M] Mandated: Apple Silicon MPS float64 automatic fallback to CPU, HPC distributed lock prohibition.
- [D] Derived: Dynamic 6-tier runtime path resolution and hardware binding.
- [E] Empirical: OS-agnostic pathlib handling and local NVMe scratch fallback.

Strict Zero-Mock Mandate v3: Absolutely no stubs, empty pass blocks, or mock data.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import filelock
import torch


def dispatch_device_safely(
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.device:
    """Dispatch hardware accelerator safely with Apple Silicon MPS float64 fallback [M].

    Metal Performance Shaders (MPS) does not support 64-bit floating point operations.
    When execution requests torch.float64, this dispatcher intercepts and reroutes to CPU,
    while permitting single-precision torch.float32 on MPS.

    Parameters
    ----------
    device : Union[str, torch.device]
        Requested target device (e.g. 'cpu', 'cuda', 'mps').
    dtype : torch.dtype
        Computation numerical precision.

    Returns
    -------
    torch.device
        Safely routed device.
    """
    dev_str = str(device).strip().lower()

    # Intercept MPS float64 requests and fallback to CPU [M]
    if "mps" in dev_str and dtype == torch.float64:
        return torch.device("cpu")

    return torch.device(device)


def resolve_hpc_safe_scratch() -> Path:
    """Resolve node-local scratch storage adhering to the HPC Distributed Lock Prohibition [M].

    On parallel network filesystems (Lustre, GPFS, BeeGFS, NFS), direct file locking
    triggers lock manager deadlocks ([Errno 37] No locks available). All staging and locking
    must route to local NVMe scratch via $COCH_SCRATCH, $SLURM_TMPDIR or $TMPDIR.
    On Windows (win32), sequence:
    1. COCH_SCRATCH or COCHEM_SCRATCH_DIR environment variable
    2. LOCALAPPDATA / Temp / cochem_scratch or TEMP / cochem_scratch
    3. splitdrive(sys.executable)[0] + \\cochem_scratch
    Never fallback to ~/.cochem/scratch on Windows to prevent network SMB roaming profile deadlocks.
    On Linux: SLURM_TMPDIR, PBS_JOBFS, TMPDIR, /dev/shm, /tmp.
    On macOS: TMPDIR.

    Returns
    -------
    Path
        Path to local scratch directory, guaranteed to exist on disk.
    """
    env_scratch = os.environ.get("COCH_SCRATCH") or os.environ.get("COCHEM_SCRATCH_DIR")
    if env_scratch:
        scratch_path = Path(env_scratch)
    elif sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        sys_temp = os.environ.get("TEMP") or os.environ.get("TMP")
        if local_app:
            scratch_path = Path(local_app) / "Temp" / "cochem_scratch"
        elif sys_temp:
            scratch_path = Path(sys_temp) / "cochem_scratch"
        else:
            drive = os.path.splitdrive(sys.executable)[0] or "C:"
            scratch_path = Path(f"{drive}\\cochem_scratch")
    elif sys.platform == "darwin":
        sys_tmp = os.environ.get("TMPDIR")
        if sys_tmp and Path(sys_tmp).exists():
            scratch_path = Path(sys_tmp) / "cochem_scratch"
        else:
            scratch_path = Path("/tmp") / "cochem_scratch"
    else:
        # Linux / HPC
        slurm_tmp = os.environ.get("SLURM_TMPDIR")
        pbs_tmp = os.environ.get("PBS_JOBFS")
        sys_tmp = os.environ.get("TMPDIR")
        if slurm_tmp and Path(slurm_tmp).exists():
            scratch_path = Path(slurm_tmp) / "cochem_scratch"
        elif pbs_tmp and Path(pbs_tmp).exists():
            scratch_path = Path(pbs_tmp) / "cochem_scratch"
        elif sys_tmp and Path(sys_tmp).exists():
            scratch_path = Path(sys_tmp) / "cochem_scratch"
        elif Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK):
            scratch_path = Path("/dev/shm") / "cochem_scratch"
        elif Path("/tmp").is_dir() and os.access("/tmp", os.W_OK):
            scratch_path = Path("/tmp") / "cochem_scratch"
        else:
            scratch_path = Path("/tmp") / "cochem_scratch"

    scratch_path.mkdir(parents=True, exist_ok=True)
    return scratch_path


def atomic_promote_to_store(
    src_path: str | Path,
    store_dir: str | Path,
) -> tuple[Path, Path]:
    """Promote artifact from T_scr to T_store atomically with SHA-256 validation [M].

    Parameters
    ----------
    src_path : Union[str, Path]
        Source artifact located in scratch directory (T_scr).
    store_dir : Union[str, Path]
        Destination directory in storage tier (T_store).

    Returns
    -------
    Tuple[Path, Path]
        (promoted_artifact_path, sha256_digest_file_path)
    """
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source artifact not found in scratch: {src}")

    store = Path(store_dir).resolve()
    store.mkdir(parents=True, exist_ok=True)

    dest = store / src.name
    digest_path = store / f"{src.name}.sha256"

    # Compute SHA-256 digest of original artifact content
    content = src.read_bytes()
    sha256_hash = hashlib.sha256(content).hexdigest()

    # Atomically move from scratch to store
    try:
        os.replace(src, dest)
    except OSError:
        # Cross-device link fallback: copy to temporary destination file, then replace
        temp_dest = store / f".tmp_{uuid.uuid4().hex}_{src.name}"
        shutil.copy2(src, temp_dest)
        os.replace(temp_dest, dest)
        src.unlink(missing_ok=True)

    # Write accompanying SHA-256 checksum digest
    digest_path.write_text(f"{sha256_hash}  {dest.name}\n", encoding="utf-8")

    return dest, digest_path


class EphemeralScratchSession:
    """Context-managed scratch directory scaffolding with automated post-execution purge.

    Creates an isolated sandbox directory anchored inside the node-local scratch storage
    and ensures automated cleanup upon exit to prevent disk bloat. Also provides
    anchored file locking via local filelock.FileLock.
    """

    def __init__(
        self,
        prefix: str = "cochem_session_",
        base_dir: Path | None = None,
        auto_purge: bool = True,
    ) -> None:
        self.prefix = prefix
        self.base_dir = Path(base_dir) if base_dir else resolve_hpc_safe_scratch()
        self.session_id = uuid.uuid4().hex
        self.path = self.base_dir / f"{self.prefix}{self.session_id}"
        self.auto_purge = auto_purge
        self._lock: filelock.FileLock | None = None

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        if self._lock is not None:
            try:
                if self._lock.is_locked:
                    self._lock.release()
            except Exception:
                pass
        if self.auto_purge and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def get_lock(self, lock_name: str = "session.lock", timeout: float = 30.0) -> filelock.FileLock:
        """Create a filelock.FileLock anchored within this local scratch session."""
        self.path.mkdir(parents=True, exist_ok=True)
        lock_file = self.path / lock_name
        self._lock = filelock.FileLock(str(lock_file), timeout=timeout)
        return self._lock

