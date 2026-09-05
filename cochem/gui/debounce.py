"""
CoChem GUI Trailing Debounce Utilities (REQ-MOB-007).

Strict Zero-Mock Mandate:
- Thread-safe synchronization via threading.Lock and threading.Timer.
- Full cancellation, immediate flushing, and pending status verification.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any


class TrailingDebounce:
    """
    Thread-safe trailing-edge debounce coordinator.

    Delays executing the target callback until after `wait_seconds` have elapsed
    since the last time it was invoked. Rapid consecutive calls reset the timer
    so only the latest arguments are processed upon timer expiry.
    """

    def __init__(self, callback: Callable[..., Any], wait_seconds: float = 0.25) -> None:
        """
        Initialize the trailing debouncer.

        Args:
            callback: Target callable to execute when debouncing window closes.
            wait_seconds: Trailing window duration in seconds (default: 0.25 s / 250 ms).
        """
        self._callback: Callable[..., Any] = callback
        self._wait_seconds: float = max(0.0, float(wait_seconds))
        self._lock: threading.Lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._last_args: tuple[Any, ...] = ()
        self._last_kwargs: dict[str, Any] = {}
        self._has_pending: bool = False
        self._call_count: int = 0
        self._execution_count: int = 0

    @property
    def wait_seconds(self) -> float:
        """Return configured wait seconds."""
        return self._wait_seconds

    @property
    def call_count(self) -> int:
        """Return total number of invocations submitted."""
        with self._lock:
            return self._call_count

    @property
    def execution_count(self) -> int:
        """Return total number of actual executions dispatched."""
        with self._lock:
            return self._execution_count

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Submit a new call. Resets the trailing timer and stores latest arguments.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            self._last_args = args
            self._last_kwargs = kwargs
            self._has_pending = True
            self._call_count += 1

            self._timer = threading.Timer(self._wait_seconds, self._on_timer_expired)
            self._timer.daemon = True
            self._timer.start()

    def _on_timer_expired(self) -> None:
        """Private callback triggered upon threading.Timer expiration."""
        args: tuple[Any, ...] = ()
        kwargs: dict[str, Any] = {}
        should_run = False

        with self._lock:
            if self._has_pending:
                args = self._last_args
                kwargs = self._last_kwargs
                self._has_pending = False
                self._timer = None
                self._execution_count += 1
                should_run = True

        if should_run:
            self._callback(*args, **kwargs)

    def cancel(self) -> None:
        """
        Cancel any pending debounced execution without running the callback.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._has_pending = False
            self._last_args = ()
            self._last_kwargs = {}

    def flush(self) -> Any:
        """
        Immediately execute any pending debounced call and return its result.
        If no execution is pending, returns None.
        """
        args: tuple[Any, ...] = ()
        kwargs: dict[str, Any] = {}
        should_run = False

        with self._lock:
            if self._has_pending:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                args = self._last_args
                kwargs = self._last_kwargs
                self._has_pending = False
                self._last_args = ()
                self._last_kwargs = {}
                self._execution_count += 1
                should_run = True

        if should_run:
            return self._callback(*args, **kwargs)
        return None

    def is_pending(self) -> bool:
        """
        Check whether an execution is currently pending in the debounce window.
        """
        with self._lock:
            return self._has_pending


def debounce(
    target: float | Callable[..., Any] | None = None,
    wait_seconds: float = 0.25,
) -> Any:
    """
    Decorator for wrapping functions with trailing debounce semantics.

    Supports usage as `@debounce`, `@debounce(0.25)`, or `@debounce(wait_seconds=0.25)`.
    """
    if callable(target):
        func = target
        debouncer = TrailingDebounce(func, wait_seconds=0.25)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            debouncer(*args, **kwargs)

        wrapper.cancel = debouncer.cancel  # type: ignore[attr-defined]
        wrapper.flush = debouncer.flush  # type: ignore[attr-defined]
        wrapper.is_pending = debouncer.is_pending  # type: ignore[attr-defined]
        wrapper.debouncer = debouncer  # type: ignore[attr-defined]
        return wrapper

    def decorator(func: Callable[..., Any]) -> Any:
        delay = wait_seconds if target is None else float(target)
        debouncer = TrailingDebounce(func, wait_seconds=delay)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            debouncer(*args, **kwargs)

        wrapper.cancel = debouncer.cancel  # type: ignore[attr-defined]
        wrapper.flush = debouncer.flush  # type: ignore[attr-defined]
        wrapper.is_pending = debouncer.is_pending  # type: ignore[attr-defined]
        wrapper.debouncer = debouncer  # type: ignore[attr-defined]
        return wrapper

    return decorator
