"""Cross-platform Reader-Writer and atomic file locking architecture.

Implements Suggestion #28:
- Authentic OS-level kernel locking: LockFileEx/UnlockFileEx on Windows, fcntl.flock on POSIX.
- RWFileLock supporting concurrent shared readers and exclusive writer with writer-priority.
- Thread-safe and cross-process safe with re-entrancy depth tracking.
- AtomicFileLock maintaining backward compatibility.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger("cochem.concurrency.atomic_file_lock")

# Win32 Kernel primitives via ctypes
if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.windll.kernel32
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002

    def _os_lock_acquire(fd: int, exclusive: bool, timeout: float) -> bool:
        """Acquire OS-level lock on Windows using LockFileEx."""
        handle = msvcrt.get_osfhandle(fd)
        flags = _LOCKFILE_FAIL_IMMEDIATELY | (_LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0)
        ov = _OVERLAPPED()
        t0 = time.time()
        while True:
            if _kernel32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(ov)):
                return True
            if time.time() - t0 >= timeout:
                return False
            time.sleep(0.002)

    def _os_lock_release(fd: int) -> None:
        """Release OS-level lock on Windows using UnlockFileEx."""
        handle = msvcrt.get_osfhandle(fd)
        ov = _OVERLAPPED()
        _kernel32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(ov))

else:
    import fcntl

    def _os_lock_acquire(fd: int, exclusive: bool, timeout: float) -> bool:
        """Acquire OS-level lock on POSIX using fcntl.flock."""
        flags = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fd, flags)
                return True
            except (BlockingIOError, OSError):
                if time.time() - t0 >= timeout:
                    return False
                time.sleep(0.002)

    def _os_lock_release(fd: int) -> None:
        """Release OS-level lock on POSIX using fcntl.flock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as _e:
            logger.debug(f"Ignored exception: {_e}")


class RWFileLockTimeoutError(TimeoutError):
    """Raised when RWFileLock acquisition times out."""


class RWFileLock:
    """Kernel-level Reader-Writer file lock with writer-priority protocol.

    Uses two OS kernel lock files:
    - writer_intent: acquired shared by readers, acquired exclusive by writer.
      When a writer arrives, it acquires writer_intent exclusive, immediately blocking
      all subsequent readers.
    - data_lock: acquired shared by readers for read duration, acquired exclusive
      by writer for write duration.
    """

    def __init__(self, lock_path: Path | str, timeout: float = 10.0) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)

        self._intent_path = str(self.lock_path.with_name(f"{self.lock_path.name}.writer_intent.lock"))
        self._data_path = str(self.lock_path.with_name(f"{self.lock_path.name}.data.lock"))

        self._local = threading.local()

    def _get_read_depth(self) -> int:
        return getattr(self._local, "read_depth", 0)

    def _set_read_depth(self, val: int) -> None:
        self._local.read_depth = val

    def _get_write_depth(self) -> int:
        return getattr(self._local, "write_depth", 0)

    def _set_write_depth(self, val: int) -> None:
        self._local.write_depth = val

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None) -> Generator[None, None, None]:
        """Shared read lock allowing unbounded concurrent readers with writer priority."""
        t_limit = self.timeout if timeout is None else float(timeout)
        t0 = time.time()

        # Thread-level reentrancy for active writer
        if self._get_write_depth() > 0:
            yield
            return

        # Thread-level reentrancy for active reader
        depth = self._get_read_depth()
        if depth > 0:
            self._set_read_depth(depth + 1)
            try:
                yield
            finally:
                self._set_read_depth(self._get_read_depth() - 1)
            return

        # Open dedicated file descriptors for this thread/operation
        fd_intent = os.open(self._intent_path, os.O_RDWR | os.O_CREAT)
        fd_data = os.open(self._data_path, os.O_RDWR | os.O_CREAT)
        try:
            # Step 1: Check writer intent (acquire shared on intent lock)
            remaining = max(0.001, t_limit - (time.time() - t0))
            if not _os_lock_acquire(fd_intent, exclusive=False, timeout=remaining):
                raise RWFileLockTimeoutError(
                    f"Timed out waiting for writer intent on {self.lock_path} after {t_limit:.2f}s"
                )

            try:
                # Step 2: Acquire shared read lock on data file
                remaining = max(0.001, t_limit - (time.time() - t0))
                if not _os_lock_acquire(fd_data, exclusive=False, timeout=remaining):
                    raise RWFileLockTimeoutError(
                        f"Timed out acquiring shared read lock on {self.lock_path} after {t_limit:.2f}s"
                    )
            finally:
                # Release writer intent so another writer can request intent
                _os_lock_release(fd_intent)

            self._set_read_depth(1)
            try:
                yield
            finally:
                self._set_read_depth(0)
                _os_lock_release(fd_data)
        finally:
            os.close(fd_intent)
            os.close(fd_data)

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None) -> Generator[None, None, None]:
        """Exclusive write lock with writer-priority blocking new incoming readers."""
        t_limit = self.timeout if timeout is None else float(timeout)
        t0 = time.time()

        # Thread-level reentrancy for active writer
        depth = self._get_write_depth()
        if depth > 0:
            self._set_write_depth(depth + 1)
            try:
                yield
            finally:
                self._set_write_depth(self._get_write_depth() - 1)
            return

        fd_intent = os.open(self._intent_path, os.O_RDWR | os.O_CREAT)
        fd_data = os.open(self._data_path, os.O_RDWR | os.O_CREAT)
        try:
            # Step 1: Acquire exclusive writer intent to block new readers immediately
            remaining = max(0.001, t_limit - (time.time() - t0))
            if not _os_lock_acquire(fd_intent, exclusive=True, timeout=remaining):
                raise RWFileLockTimeoutError(
                    f"Timed out acquiring writer intent on {self.lock_path} after {t_limit:.2f}s"
                )

            try:
                # Step 2: Acquire exclusive lock on data file (waits for active readers to clear)
                remaining = max(0.001, t_limit - (time.time() - t0))
                if not _os_lock_acquire(fd_data, exclusive=True, timeout=remaining):
                    raise RWFileLockTimeoutError(
                        f"Timed out acquiring exclusive write lock on {self.lock_path} after {t_limit:.2f}s"
                    )

                self._set_write_depth(1)
                try:
                    yield
                finally:
                    self._set_write_depth(0)
                    _os_lock_release(fd_data)
            finally:
                _os_lock_release(fd_intent)
        finally:
            os.close(fd_intent)
            os.close(fd_data)

    def acquire(self, shared: bool = False, timeout: Optional[float] = None) -> bool:
        """Acquire lock (shared read if shared=True, else exclusive write)."""
        t = self.timeout if timeout is None else timeout
        cm = self.read_lock(timeout=t) if shared else self.write_lock(timeout=t)
        cm.__enter__()
        self._local.cm = cm
        return True

    def release(self) -> None:
        """Release currently held lock."""
        cm = getattr(self._local, "cm", None)
        if cm is not None:
            self._local.cm = None
            cm.__exit__(None, None, None)

    def __enter__(self) -> RWFileLock:
        self.acquire(shared=False)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class AtomicFileLock(RWFileLock):
    """Drop-in compatible wrapper around RWFileLock defaulting to exclusive locking."""

    def __init__(self, lock_path: Path | str, timeout: float = 10.0, **kwargs: Any) -> None:
        super().__init__(lock_path=lock_path, timeout=timeout)


__all__ = ["RWFileLock", "AtomicFileLock", "RWFileLockTimeoutError"]
