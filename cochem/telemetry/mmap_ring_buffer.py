"""High-throughput OS-agnostic memory-mapped circular ring buffer for event emission.

Provenance & Specifications:
- Method Matrix [M]: High-throughput non-blocking telemetry stream.
- Binary Packing [D]: Little-endian 64-byte cache-aligned header with 20-byte slot layout.
- Concurrency [E]: FileLock coordination on node-local storage with crash recovery.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from filelock import FileLock

# Header layout constants
HEADER_FORMAT: str = "<4sIIIQQQ24s"
HEADER_SIZE: int = 64  # struct.calcsize("<4sIIIQQQ24s") == 64

SLOT_HEADER_FORMAT: str = "<IQII"
SLOT_HEADER_SIZE: int = 20  # struct.calcsize("<IQII") == 20

MAGIC_BYTES: bytes = b"CCHM"
FORMAT_VERSION: int = 1
DEFAULT_BUFFER_CAPACITY: int = 65536
DEFAULT_SLOT_SIZE: int = 512

# Slot state flags
SLOT_FREE: int = 0x00
SLOT_WRITING: int = 0x01
SLOT_COMMITTED: int = 0x02
SLOT_CORRUPT: int = 0x03

HEARTBEAT_TIMEOUT_SEC: float = 2.0


class MmapRingBuffer:
    """Memory-mapped circular ring buffer for multi-process event emission."""

    def __init__(
        self,
        file_path: Union[str, Path],
        capacity: int = DEFAULT_BUFFER_CAPACITY,
        slot_size: int = DEFAULT_SLOT_SIZE,
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.file_path.with_name(f"{self.file_path.name}.lock")
        self._lock = FileLock(str(self.lock_path), timeout=10.0)

        self.capacity = capacity
        self.slot_size = slot_size
        self.max_payload_len = self.slot_size - SLOT_HEADER_SIZE
        self.total_file_size = HEADER_SIZE + (self.capacity * self.slot_size)

        self._file_handle: Optional[Any] = None
        self._mm: Optional[mmap.mmap] = None
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Create or validate memory-mapped sparse file with 64-byte binary header."""
        with self._lock:
            exists = self.file_path.exists()
            file_size = self.file_path.stat().st_size if exists else 0

            if not exists or file_size < self.total_file_size:
                with open(self.file_path, "a+b") as initial_f:
                    initial_f.truncate(self.total_file_size)
                    initial_f.flush()

                # Initialize binary header
                reserved = b"\x00" * 24
                initial_header = struct.pack(
                    HEADER_FORMAT,
                    MAGIC_BYTES,
                    FORMAT_VERSION,
                    self.capacity,
                    self.slot_size,
                    0,  # head_seq
                    0,  # tail_seq
                    0,  # dropped_records
                    reserved,
                )
                with open(self.file_path, "r+b") as initial_f:
                    initial_f.seek(0)
                    initial_f.write(initial_header)
                    initial_f.flush()

            self._file_handle = open(self.file_path, "r+b")
            self._mm = mmap.mmap(self._file_handle.fileno(), length=self.total_file_size)

            # Read existing header to synchronize capacity and slot size
            header_bytes = self._mm[0:HEADER_SIZE]
            (
                magic,
                ver,
                cap,
                s_size,
                head,
                tail,
                dropped,
                _,
            ) = struct.unpack(HEADER_FORMAT, header_bytes)

            if magic != MAGIC_BYTES:
                raise ValueError(f"Invalid magic bytes in telemetry buffer: {magic!r}")

            self.capacity = cap
            self.slot_size = s_size
            self.max_payload_len = self.slot_size - SLOT_HEADER_SIZE
            self.total_file_size = HEADER_SIZE + (self.capacity * self.slot_size)

    @property
    def mm(self) -> mmap.mmap:
        if self._mm is None or self._mm.closed:
            raise RuntimeError("MmapRingBuffer is closed or uninitialized.")
        return self._mm

    def write_record(self, payload: Union[bytes, str]) -> int:
        """Atomically claim a sequence slot and write payload into circular buffer."""
        payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(payload_bytes) > self.max_payload_len:
            raise ValueError(
                f"Payload size ({len(payload_bytes)} bytes) exceeds slot payload capacity ({self.max_payload_len} bytes)"
            )

        with self._lock:
            header_bytes = self.mm[0:HEADER_SIZE]
            (
                magic,
                ver,
                cap,
                s_size,
                head_seq,
                tail_seq,
                dropped_records,
                reserved,
            ) = struct.unpack(HEADER_FORMAT, header_bytes)

            if head_seq >= tail_seq + cap:
                dropped_records += 1
                tail_seq = head_seq - cap + 1

            claimed_seq = head_seq
            new_head_seq = head_seq + 1

            slot_idx = claimed_seq % cap
            slot_offset = HEADER_SIZE + (slot_idx * s_size)

            # Mark slot WRITING
            now_ns = time.time_ns()
            pid = os.getpid()
            slot_header = struct.pack(
                SLOT_HEADER_FORMAT,
                SLOT_WRITING,
                now_ns,
                pid,
                len(payload_bytes),
            )
            self.mm[slot_offset : slot_offset + SLOT_HEADER_SIZE] = slot_header
            self.mm[
                slot_offset + SLOT_HEADER_SIZE : slot_offset + SLOT_HEADER_SIZE + len(payload_bytes)
            ] = payload_bytes

            # Transition slot to COMMITTED
            self.mm[slot_offset : slot_offset + 4] = struct.pack("<I", SLOT_COMMITTED)

            # Commit updated header
            updated_header = struct.pack(
                HEADER_FORMAT,
                magic,
                ver,
                cap,
                s_size,
                new_head_seq,
                tail_seq,
                dropped_records,
                reserved,
            )
            self.mm[0:HEADER_SIZE] = updated_header
            self.mm.flush()

            return int(claimed_seq)

    def read_records(
        self,
        cursor_seq: Optional[int] = None,
        max_records: int = 100,
    ) -> List[Tuple[int, int, int, bytes]]:
        """Read committed records starting from cursor_seq or current tail_seq."""
        records: List[Tuple[int, int, int, bytes]] = []

        with self._lock:
            header_bytes = self.mm[0:HEADER_SIZE]
            (
                magic,
                ver,
                cap,
                s_size,
                head_seq,
                tail_seq,
                dropped_records,
                reserved,
            ) = struct.unpack(HEADER_FORMAT, header_bytes)

            start_seq = tail_seq if cursor_seq is None else cursor_seq
            if start_seq < head_seq - cap:
                start_seq = head_seq - cap

            current_seq = start_seq
            count = 0

            while current_seq < head_seq and count < max_records:
                slot_idx = current_seq % cap
                slot_offset = HEADER_SIZE + (slot_idx * s_size)

                slot_hdr_bytes = self.mm[slot_offset : slot_offset + SLOT_HEADER_SIZE]
                (
                    status,
                    timestamp_ns,
                    pid,
                    payload_len,
                ) = struct.unpack(SLOT_HEADER_FORMAT, slot_hdr_bytes)

                if status == SLOT_WRITING:
                    elapsed_sec = (time.time_ns() - timestamp_ns) / 1e9
                    if elapsed_sec > HEARTBEAT_TIMEOUT_SEC:
                        # Mark corrupt and skip
                        self.mm[slot_offset : slot_offset + 4] = struct.pack("<I", SLOT_CORRUPT)
                        self.mm.flush()
                        current_seq += 1
                        continue
                    # Slot still being actively written, halt consumption
                    break
                elif status == SLOT_COMMITTED:
                    payload_data = bytes(
                        self.mm[
                            slot_offset + SLOT_HEADER_SIZE : slot_offset + SLOT_HEADER_SIZE + payload_len
                        ]
                    )
                    records.append((current_seq, timestamp_ns, pid, payload_data))
                    count += 1
                    current_seq += 1
                else:
                    # FREE or CORRUPT slot
                    current_seq += 1

            if cursor_seq is None and records:
                # Update consumer tail_seq
                new_tail = records[-1][0] + 1
                updated_header = struct.pack(
                    HEADER_FORMAT,
                    magic,
                    ver,
                    cap,
                    s_size,
                    head_seq,
                    new_tail,
                    dropped_records,
                    reserved,
                )
                self.mm[0:HEADER_SIZE] = updated_header
                self.mm.flush()

        return records

    def get_header_stats(self) -> Dict[str, Any]:
        """Inspect current ring buffer operational telemetry."""
        with self._lock:
            header_bytes = self.mm[0:HEADER_SIZE]
            (
                magic,
                ver,
                cap,
                s_size,
                head_seq,
                tail_seq,
                dropped_records,
                _,
            ) = struct.unpack(HEADER_FORMAT, header_bytes)

            return {
                "magic": magic.decode("ascii", errors="replace"),
                "version": ver,
                "capacity": cap,
                "slot_size": s_size,
                "head_seq": head_seq,
                "tail_seq": tail_seq,
                "dropped_records": dropped_records,
            }

    def close(self) -> None:
        """Safely flush and release memory map and file resources."""
        if self._mm is not None and not self._mm.closed:
            self._mm.flush()
            self._mm.close()
            self._mm = None

        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> MmapRingBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
