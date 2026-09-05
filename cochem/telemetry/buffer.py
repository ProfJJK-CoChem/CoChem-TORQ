"""
Bounded Ring-Buffer Telemetry & OOM Mitigation.

Invariants:
- FIFO Ring Buffer capped at <= 64 KB total byte size and <= 500 records.
- Thread-safe circular buffering via collections.deque and threading.Lock.
- Monotonic sequence tracking, eviction accounting, and cursor delta queries.
- DOM Viewport snapshot helpers and SSE stream formatting.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from cochem.telemetry.exceptions import RingBufferOverflowError
from cochem.telemetry.schemas import TelemetryLogLevel, TelemetryRecord

MAX_BUFFER_BYTES: int = 65536  # 64 KB limit
MAX_BUFFER_RECORDS: int = 500  # 500 records limit


class BoundedRingBuffer:
    """
    Thread-safe circular FIFO telemetry ring buffer bounded by record count and byte size.
    """

    def __init__(
        self,
        max_bytes: int = MAX_BUFFER_BYTES,
        max_records: int = MAX_BUFFER_RECORDS,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_records = max_records
        self._lock = threading.Lock()
        self._buffer: deque[Tuple[TelemetryRecord, int]] = deque()  # (record, byte_size)
        self._current_bytes: int = 0
        self._start_seq: int = 0
        self._next_seq: int = 0
        self._total_bytes_evicted: int = 0
        self._total_records_evicted: int = 0

    @property
    def start_seq(self) -> int:
        with self._lock:
            return self._start_seq

    @property
    def next_seq(self) -> int:
        with self._lock:
            return self._next_seq

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def total_bytes_evicted(self) -> int:
        with self._lock:
            return self._total_bytes_evicted

    @property
    def total_records_evicted(self) -> int:
        with self._lock:
            return self._total_records_evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _append_under_lock(self, record: TelemetryRecord) -> None:
        """Internal helper to insert a record and evict oldest entries under lock."""
        record_bytes = len(record.model_dump_json().encode("utf-8"))

        if record_bytes > self.max_bytes:
            raise RingBufferOverflowError(
                f"Record size ({record_bytes} bytes) exceeds total ring buffer capacity ({self.max_bytes} bytes)"
            )

        # If buffer is empty, start_seq aligns with incoming record
        if len(self._buffer) == 0:
            self._start_seq = record.seq

        # Evict oldest entries until within record count and byte capacity
        while len(self._buffer) >= self.max_records or (
            self._current_bytes + record_bytes > self.max_bytes and len(self._buffer) > 0
        ):
            evicted_rec, evicted_bytes = self._buffer.popleft()
            self._current_bytes -= evicted_bytes
            self._total_bytes_evicted += evicted_bytes
            self._total_records_evicted += 1
            if self._buffer:
                self._start_seq = self._buffer[0][0].seq
            else:
                self._start_seq = record.seq

        self._buffer.append((record, record_bytes))
        self._current_bytes += record_bytes
        self._next_seq = max(self._next_seq, record.seq + 1)

    def append(self, record: TelemetryRecord) -> None:
        """Append a TelemetryRecord to the ring buffer, evicting oldest records if limits exceeded."""
        with self._lock:
            self._append_under_lock(record)

    def append_record(
        self,
        level: Union[TelemetryLogLevel, str],
        message: str,
        module: str = "cochem",
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryRecord:
        """Create and append a new TelemetryRecord atomically using current next_seq."""
        if isinstance(level, str):
            level_enum = TelemetryLogLevel(level.upper())
        else:
            level_enum = level

        with self._lock:
            seq = self._next_seq
            timestamp = datetime.now(timezone.utc).isoformat()
            record = TelemetryRecord(
                seq=seq,
                timestamp=timestamp,
                level=level_enum,
                module=module,
                job_id=job_id,
                message=message,
                metadata=metadata or {},
            )
            self._append_under_lock(record)
            return record

    def get_delta(self, cursor_seq: int) -> Tuple[List[TelemetryRecord], int, bool]:
        """
        Query incremental telemetry delta since cursor_seq.

        Returns:
            (records, next_seq, resync_required)
            resync_required is True if cursor_seq < start_seq (records were evicted).
        """
        with self._lock:
            if not self._buffer:
                resync = cursor_seq < self._start_seq and self._total_records_evicted > 0
                return [], self._next_seq, resync

            # Case 1: Client cursor is older than oldest record in buffer (eviction gap)
            if cursor_seq < self._start_seq:
                all_records = [rec for rec, _ in self._buffer]
                return all_records, self._next_seq, True

            # Case 2: Client cursor is up to date or ahead
            if cursor_seq >= self._next_seq:
                return [], self._next_seq, False

            # Case 3: Incremental slice
            records = [rec for rec, _ in self._buffer if rec.seq >= cursor_seq]
            return records, self._next_seq, False

    def get_viewport_snapshot(self, max_lines: int = 50) -> List[TelemetryRecord]:
        """Return the most recent max_lines records for DOM viewport rendering."""
        with self._lock:
            if not self._buffer:
                return []
            records = [rec for rec, _ in self._buffer]
            return records[-max_lines:]

    def format_sse_event(
        self,
        records: List[TelemetryRecord],
        next_seq: int,
        resync_required: bool,
    ) -> str:
        """Format telemetry records as a Server-Sent Events (SSE) data chunk."""
        payload = {
            "records": [rec.model_dump() for rec in records],
            "next_seq": next_seq,
            "resync_required": resync_required,
        }
        return f"event: telemetry\ndata: {json.dumps(payload)}\n\n"

    def clear(self) -> None:
        """Clear all records in the ring buffer."""
        with self._lock:
            self._buffer.clear()
            self._current_bytes = 0
            self._start_seq = self._next_seq
