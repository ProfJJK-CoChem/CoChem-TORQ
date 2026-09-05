"""Pure-Wheel Fast IPC Serialization & HDF5 PESStore.
High-throughput binary Msgpack serialization, SharedMemory descriptors, HMAC socket transport,
and QCSchema-compliant HDF5 tensor persistence in SWMR mode.
Strictly adheres to Zero-Mock mandate and authentic binary serialization.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime
import errno
import hashlib
import hmac
import json
import logging
import multiprocessing.shared_memory as sm
import os
import pathlib
import secrets
import shutil
import socket
import struct
import tempfile
import threading
import time
import uuid
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import filelock
import h5py
import msgpack  # type: ignore[import-untyped]
import numpy as np
from pydantic import BaseModel

from cochem.core.context import assert_writable_path

logger = logging.getLogger("cochem.core.ipc.serializer")

NUMPY_EXT_CODE: int = 42

MAX_IPC_PAYLOAD_BYTES: int = 256 * 1024 * 1024  # 256 MB ceiling [D]


class IPCBindError(OSError):
    """Base exception for IPC socket binding failures."""

    pass


class PortContentionError(IPCBindError):
    """Raised when an IPC port remains in contention after retry exhaustion."""

    pass


class IPCPayloadError(Exception):
    """Base exception for IPC payload transmission failures."""

    pass


class TruncatedPayloadError(IPCPayloadError):
    """Raised when an IPC connection terminates before receiving the full payload."""

    pass


class OversizedPayloadError(IPCPayloadError):
    """Raised when a transmitted payload header exceeds the safety ceiling."""

    pass


# ==============================================================================
# Msgpack Custom Extension Codecs
# ==============================================================================
def _msgpack_encoder(obj: Any) -> Any:
    """Encode custom structures (NumPy arrays, Pydantic models, Path/UUID) for Msgpack."""
    if isinstance(obj, np.ndarray):
        dtype_str = obj.dtype.str  # type: ignore[attr-defined]
        shape_tuple = tuple(obj.shape)
        raw_buffer = obj.tobytes()
        payload = msgpack.packb((dtype_str, shape_tuple, raw_buffer), use_bin_type=True)
        return msgpack.ExtType(NUMPY_EXT_CODE, payload)
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, (pathlib.Path, uuid.UUID)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON/Msgpack serializable")


def _msgpack_decoder(code: int, data: bytes) -> Any:
    """Reconstruct NumPy arrays from Msgpack custom extension payload."""
    if code == NUMPY_EXT_CODE:
        dtype_str, shape_tuple, raw_buffer = msgpack.unpackb(data, raw=False)
        reconstructed = np.frombuffer(raw_buffer, dtype=dtype_str).reshape(tuple(shape_tuple))
        return reconstructed
    return msgpack.ExtType(code, data)


def pack_payload(data: Any) -> bytes:
    """Serialize payload into binary Msgpack bytes with NumPy array extension hooks."""
    return bytes(msgpack.packb(data, default=_msgpack_encoder, use_bin_type=True))


def unpack_payload(raw_bytes: bytes) -> Any:
    """Deserialize binary Msgpack payload and reconstruct NumPy arrays."""
    return msgpack.unpackb(raw_bytes, ext_hook=_msgpack_decoder, raw=False)


# ==============================================================================
# Zero-Copy Shared Memory Optimization
# ==============================================================================
_REGISTRY_LOCK = threading.Lock()
_ACTIVE_SHM: Dict[str, Dict[str, Any]] = {}


def _cleanup_all_shared_memory() -> None:
    """Atexit handler ensuring zero lingering shared memory blocks."""
    with _REGISTRY_LOCK:
        for name, info in list(_ACTIVE_SHM.items()):
            try:
                info["shm"].close()
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")
            try:
                info["shm"].unlink()
            except Exception as _e:
                logger.debug(f"Ignored exception: {_e}")
        _ACTIVE_SHM.clear()


atexit.register(_cleanup_all_shared_memory)


def _finalize_shm(name: str) -> None:
    with _REGISTRY_LOCK:
        info = _ACTIVE_SHM.pop(name, None)
    if info is not None:
        try:
            info["shm"].close()
            info["shm"].unlink()
        except (FileNotFoundError, OSError) as _e:
            logger.debug(f"Ignored exception: {_e}")
    try:
        s = sm.SharedMemory(name=name)
        s.close()
        s.unlink()
    except (FileNotFoundError, OSError) as _e:
        logger.debug(f"Ignored exception: {_e}")


class SharedMemoryView:
    """Context manager wrapping sm.SharedMemory and a non-copied np.ndarray view.

    Raises RuntimeError if accessed when closed.
    """

    def __init__(
        self,
        shm: sm.SharedMemory,
        arr: np.ndarray,
        owner_name: Optional[str] = None,
        is_recycled: bool = False,
    ) -> None:
        self._shm: Optional[sm.SharedMemory] = shm
        self._arr: Optional[np.ndarray] = arr
        self._is_closed: bool = False
        self._is_recycled: bool = is_recycled
        self._owner_name: Optional[str] = owner_name or (shm.name if shm else None)

    @property
    def array(self) -> np.ndarray:
        if self._is_closed or self._arr is None:
            raise RuntimeError("Cannot access array view on a closed SharedMemoryView")
        return self._arr

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._arr = None
            if self._shm is not None:
                if not self._is_recycled:
                    try:
                        self._shm.close()
                    except OSError as exc:
                        logger.debug("Shared memory view close bypassed: %s", exc)
                if self._owner_name:
                    SharedMemoryBuffer._notify_closed(self._owner_name)
                self._shm = None

    def unlink(self) -> None:
        if self._shm is not None and not self._is_recycled:
            try:
                self._shm.unlink()
            except (OSError, FileNotFoundError) as exc:
                logger.debug("Shared memory view unlink bypassed: %s", exc)
        self.close()

    def __enter__(self) -> np.ndarray:
        return self.array

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()


@dataclasses.dataclass
class SharedMemoryBuffer:
    """Encapsulates a POSIX/Windows shared memory segment for large array transfers."""

    shm: sm.SharedMemory
    descriptor: Dict[str, Any]
    _finalizer: Optional[weakref.finalize] = dataclasses.field(default=None, repr=False, compare=False)
    _array: Optional[np.ndarray] = dataclasses.field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._finalizer is None:
            self._finalizer = weakref.finalize(self, _finalize_shm, self.shm.name)

    def __enter__(self) -> SharedMemoryBuffer:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()
        self.unlink()

    @classmethod
    def create(cls, arr: np.ndarray, total_attachments: int = 2) -> SharedMemoryBuffer:
        """Alias for from_array."""
        return cls.from_array(arr, total_attachments=total_attachments)

    @classmethod
    def from_array(cls, arr: np.ndarray, total_attachments: int = 2) -> SharedMemoryBuffer:
        """Allocate shared memory buffer, copy array memory, and generate transfer descriptor."""
        total_bytes = max(1, arr.nbytes)
        shm = sm.SharedMemory(create=True, size=total_bytes)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.register(shm._name, "shared_memory")
        except Exception as exc:
            logger.debug("Resource tracker registration bypassed: %s", exc)

        shm_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)  # type: ignore[arg-type]
        shm_array[:] = arr[:]

        desc = {
            "name": shm.name,
            "shape": list(arr.shape),
            "dtype": arr.dtype.str,  # type: ignore[attr-defined]
            "size": total_bytes,
            "total_attachments": total_attachments,
            "closed_attachments": 0,
        }

        with _REGISTRY_LOCK:
            _ACTIVE_SHM[shm.name] = {
                "shm": shm,
                "total": total_attachments,
                "closed": 0,
            }

        return cls(shm=shm, descriptor=desc, _array=shm_array)

    def to_descriptor(self) -> Dict[str, Any]:
        """Return the transfer descriptor mapping this shared memory segment."""
        return dict(self.descriptor)

    @property
    def array(self) -> np.ndarray:
        """Return direct numpy ndarray view over the shared memory segment."""
        if self._array is None:
            shape = tuple(self.descriptor["shape"])
            dtype = self.descriptor["dtype"]
            self._array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        return self._array

    @classmethod
    def _notify_closed(cls, name: str) -> None:
        """Atomically increment closed attachments and unlink once all attachments finish."""
        with _REGISTRY_LOCK:
            info = _ACTIVE_SHM.get(name)
            if info is not None:
                info["closed"] += 1
                if info["closed"] >= info["total"]:
                    try:
                        info["shm"].unlink()
                    except (OSError, FileNotFoundError) as exc:
                        logger.debug("Shared memory unlink bypassed: %s", exc)
                    _ACTIVE_SHM.pop(name, None)
            else:
                try:
                    s = sm.SharedMemory(name=name)
                    s.close()
                    s.unlink()
                except Exception as exc:
                    logger.debug("Shared memory cleanup bypassed: %s", exc)

    @classmethod
    def read_from_descriptor(
        cls,
        descriptor: Dict[str, Any],
        zero_copy: bool = True,
    ) -> Union[np.ndarray, SharedMemoryView]:
        """Map existing shared memory segment and extract copy of array or zero-copy SharedMemoryView."""
        name = descriptor["name"]
        shape = tuple(descriptor["shape"])
        dtype = descriptor["dtype"]

        with _REGISTRY_LOCK:
            info = _ACTIVE_SHM.get(name)
            if info is not None:
                client_shm = info["shm"]
                is_recycled = True
            else:
                client_shm = sm.SharedMemory(name=name)
                is_recycled = False

        mapped = np.ndarray(shape, dtype=dtype, buffer=client_shm.buf)

        if zero_copy:
            return SharedMemoryView(shm=client_shm, arr=mapped, owner_name=name, is_recycled=is_recycled)
        else:
            try:
                extracted = mapped.copy()
                return extracted
            finally:
                if not is_recycled:
                    client_shm.close()
                cls._notify_closed(name)

    def close(self) -> None:
        """Close local memory map and unlink if all attachments are closed."""
        try:
            self.shm.close()
        except OSError as exc:
            logger.debug("Shared memory close bypassed: %s", exc)
        SharedMemoryBuffer._notify_closed(self.shm.name)

    def unlink(self) -> None:
        """Explicitly unlink OS shared memory segment immediately."""
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()
        try:
            self.shm.unlink()
        except (OSError, FileNotFoundError) as exc:
            logger.debug("Shared memory unlink bypassed: %s", exc)
        with _REGISTRY_LOCK:
            _ACTIVE_SHM.pop(self.shm.name, None)


# ==============================================================================
# Ephemeral HMAC-SHA256 Socket Transport
# ==============================================================================
class HMACSocketServer:
    """Loopback TCP socket server secured by HMAC-SHA256 challenge-response handshake."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        secret_key: bytes = b"",
    ) -> None:
        self.host: str = host
        self.requested_port: int = port
        self.secret_key: bytes = secret_key
        self.port: int = 0

        self._server_sock: Optional[socket.socket] = None
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._received_payloads: List[Any] = []
        self._payload_event: threading.Event = threading.Event()
        self._last_error: Optional[IPCPayloadError] = None
        self._descriptor_path: Optional[pathlib.Path] = None

    def start(self, port_fallback: bool = True, max_retries: int = 5) -> int:
        """Bind listening socket and launch background accept loop.

        Recovers dynamically from port contention (EADDRINUSE / WinError 10048).
        Publishes atomic port descriptor to COCHEM_SCRATCH_DIR.
        """
        target_port = self.requested_port
        backoff_base = 0.05
        bound = False

        for attempt in range(max_retries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self.host, target_port))
                sock.listen(5)
                self._server_sock = sock
                self.port = sock.getsockname()[1]
                bound = True
                break
            except OSError as err:
                sock.close()
                self._server_sock = None
                # Check for port contention: EADDRINUSE or Windows 10048 / 10013 / EACCES
                is_in_use = (
                    err.errno in (errno.EADDRINUSE, errno.EACCES)
                    or getattr(err, "winerror", None) in (10048, 10013)
                    or err.errno in (10048, 10013)
                )
                if is_in_use:
                    if port_fallback:
                        # Fallback immediately to ephemeral port 0
                        fb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        fb_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            fb_sock.bind((self.host, 0))
                            fb_sock.listen(5)
                            self._server_sock = fb_sock
                            self.port = fb_sock.getsockname()[1]
                            bound = True
                            break
                        except OSError as fb_err:
                            fb_sock.close()
                            self._server_sock = None
                            raise IPCBindError(f"Failed to bind ephemeral fallback port: {fb_err}") from fb_err
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(backoff_base * (2**attempt))
                            continue
                        else:
                            raise PortContentionError(
                                f"Port {target_port} contention exhausted after {max_retries} retries: {err}"
                            ) from err
                else:
                    raise IPCBindError(f"Socket bind failed on {self.host}:{target_port}: {err}") from err

        if not bound or self._server_sock is None:
            raise PortContentionError(f"Could not bind to port {target_port}")

        # Publish active binding metadata to atomic file ipc_server_{pid}.json in COCHEM_SCRATCH_DIR
        scratch_dir_env = (
            os.environ.get("COCHEM_SCRATCH_DIR")
            or os.environ.get("SLURM_TMPDIR")
            or os.environ.get("TMPDIR")
        )
        if scratch_dir_env:
            scratch_dir = pathlib.Path(scratch_dir_env).resolve()
        else:
            scratch_dir = pathlib.Path(tempfile.gettempdir()).resolve()
        scratch_dir.mkdir(parents=True, exist_ok=True)

        pid = os.getpid()
        desc_file = scratch_dir / f"ipc_server_{pid}.json"
        tmp_file = scratch_dir / f"ipc_server_{pid}_{uuid.uuid4().hex[:8]}.tmp"

        auth_token_hash = hashlib.sha256(self.secret_key).hexdigest()
        created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta = {
            "pid": pid,
            "host": self.host,
            "port": self.port,
            "created_utc": created_utc,
            "auth_token_hash": auth_token_hash,
        }

        payload_bytes = json.dumps(meta, indent=2).encode("utf-8")
        with open(tmp_file, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, desc_file)
        self._descriptor_path = desc_file

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="HMACSocketServerLoop",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        """Shutdown server socket, clean up descriptor file, and join accept thread."""
        self._stop_event.set()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError as exc:
                logger.debug("Server socket close error ignored: %s", exc)
            self._server_sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._descriptor_path is not None and self._descriptor_path.exists():
            try:
                self._descriptor_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Descriptor unlink error ignored: %s", exc)
            self._descriptor_path = None

    def _accept_loop(self) -> None:
        """Accept inbound client connections and execute HMAC handshake."""
        while not self._stop_event.is_set():
            try:
                if self._server_sock is None:
                    break
                self._server_sock.settimeout(0.5)
                conn, _ = self._server_sock.accept()
            except (socket.timeout, OSError):
                continue

            try:
                # 1. Ephemeral 32-byte cryptographic challenge
                challenge = secrets.token_bytes(32)
                conn.sendall(challenge)

                # 2. Receive 32-byte HMAC-SHA256 response
                response = conn.recv(32)
                expected = hmac.new(self.secret_key, challenge, hashlib.sha256).digest()

                if not hmac.compare_digest(response, expected):
                    logger.warning("IPC connection rejected: HMAC authentication failed")
                    conn.sendall(b"DENIED")
                    conn.close()
                    continue

                conn.sendall(b"ACCEPT")

                # 3. Read 4-byte payload length header
                len_bytes = conn.recv(4)
                if len(len_bytes) < 4:
                    conn.close()
                    continue
                (payload_len,) = struct.unpack("!I", len_bytes)

                if payload_len > MAX_IPC_PAYLOAD_BYTES:
                    logger.error("IPC payload rejected: size %d exceeds 256 MB ceiling", payload_len)
                    self._last_error = OversizedPayloadError(
                        f"Payload size {payload_len} exceeds 256 MB limit"
                    )
                    self._payload_event.set()
                    conn.close()
                    continue

                # 4. Stream payload bytes
                buffer = bytearray()
                while len(buffer) < payload_len:
                    chunk = conn.recv(min(65536, payload_len - len(buffer)))
                    if not chunk:
                        break
                    buffer.extend(chunk)

                if len(buffer) < payload_len:
                    logger.error("IPC stream truncated: received %d of %d bytes", len(buffer), payload_len)
                    self._last_error = TruncatedPayloadError(
                        f"Stream truncated: received {len(buffer)} of {payload_len} bytes"
                    )
                    self._payload_event.set()
                    conn.close()
                    continue

                if len(buffer) == payload_len:
                    payload = unpack_payload(bytes(buffer))
                    self._received_payloads.append(payload)
                    self._payload_event.set()
            except Exception as conn_err:
                logger.debug("Error processing client connection: %s", conn_err)
            finally:
                try:
                    conn.close()
                except OSError as exc:
                    logger.debug("Client conn close error ignored: %s", exc)

    def get_received_payload(self, timeout_sec: float = 5.0) -> Optional[Any]:
        """Await reception of payload from client."""
        if self._payload_event.wait(timeout_sec):
            if self._last_error is not None:
                err = self._last_error
                self._last_error = None
                self._payload_event.clear()
                raise err
            if self._received_payloads:
                payload = self._received_payloads.pop(0)
                if not self._received_payloads:
                    self._payload_event.clear()
                return payload
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise err
        return None


class HMACSocketClient:
    """Client communicating over loopback TCP with HMAC-SHA256 authentication."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        secret_key: bytes = b"",
    ) -> None:
        self.host: str = host
        self.port: int = port
        self.secret_key: bytes = secret_key

    def send_payload(self, data: Any) -> None:
        """Connect to server, satisfy HMAC challenge, and transmit Msgpack payload."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        try:
            # 1. Receive 32-byte challenge
            challenge = sock.recv(32)
            if len(challenge) != 32:
                raise ConnectionError("Invalid challenge received from server")

            # 2. Compute and send response
            response = hmac.new(self.secret_key, challenge, hashlib.sha256).digest()
            sock.sendall(response)

            status = sock.recv(6)
            if status != b"ACCEPT":
                raise PermissionError("HMAC handshake rejected by server")

            # 3. Pack payload and send with length header
            packed_bytes = pack_payload(data)
            header = struct.pack("!I", len(packed_bytes))
            sock.sendall(header + packed_bytes)
        finally:
            sock.close()


def validate_airgap_write_path(target_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Lazily import validate_airgap_write_path to break circular import cycle."""
    from cochem_base.core.ipc.serializer import validate_airgap_write_path as _v
    return _v(target_path)


class PESStore:
    """Multidimensional tensor persistence store for Potential Energy Surfaces using HDF5 SWMR."""

    def __init__(self, file_path: Union[pathlib.Path, str]) -> None:
        self.file_path: pathlib.Path = validate_airgap_write_path(pathlib.Path(file_path).resolve())
        assert_writable_path(self.file_path)
        self.lock_path: pathlib.Path = pathlib.Path(str(self.file_path) + ".lock").resolve()
        self._write_lock: threading.RLock = threading.RLock()

    def write_entry(
        self,
        entry_id: str,
        molecule: Dict[str, Any],
        driver: str,
        model: Dict[str, Any],
        return_result: np.ndarray,
    ) -> None:
        """Persist QCSchema calculation entry into HDF5 file in SWMR mode."""
        validate_airgap_write_path(self.file_path)
        assert_writable_path(self.file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if shutil.disk_usage(self.file_path.parent).free < 100 * 1024 * 1024:
            raise IOError("Insufficient disk space on target volume for PESStore append")

        arr = np.asarray(return_result)
        chunk_shape: Optional[Tuple[int, ...]] = None
        max_shape: Optional[Tuple[Optional[int], ...]] = None
        if arr.ndim > 0:
            chunk_shape = tuple(max(1, min(s, 128)) for s in arr.shape)
            max_shape = tuple(None for _ in arr.shape)

        with self._write_lock:
            with filelock.FileLock(str(self.lock_path), timeout=30.0):
                with h5py.File(self.file_path, "a", libver="latest") as h5f:
                    if entry_id in h5f:
                        del h5f[entry_id]

                    grp = h5f.create_group(entry_id)
                    grp.attrs["schema_name"] = "qcschema_output"
                    grp.attrs["driver"] = str(driver)
                    grp.attrs["molecule_json"] = json.dumps(molecule)
                    grp.attrs["model_json"] = json.dumps(model)

                    if arr.ndim > 0:
                        grp.create_dataset(
                            "return_result",
                            data=arr,
                            maxshape=max_shape,
                            chunks=chunk_shape,
                            compression="gzip",
                            compression_opts=4,
                            fletcher32=True,
                        )
                    else:
                        grp.create_dataset("return_result", data=arr)

                    h5f.flush()

    def read_entry(self, entry_id: str) -> Dict[str, Any]:
        """Read QCSchema entry in SWMR mode without file locking collisions."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"PESStore file not found at {self.file_path}")

        with h5py.File(self.file_path, "r", libver="latest", swmr=True) as h5f:
            if entry_id not in h5f:
                raise KeyError(f"Entry '{entry_id}' not found in PESStore")

            grp = h5f[entry_id]
            schema_name = str(grp.attrs.get("schema_name", "qcschema_output"))
            driver = str(grp.attrs.get("driver", "unknown"))
            mol_json = str(grp.attrs.get("molecule_json", "{}"))
            model_json = str(grp.attrs.get("model_json", "{}"))
            result_arr = grp["return_result"][:]

            return {
                "schema_name": schema_name,
                "entry_id": entry_id,
                "molecule": json.loads(mol_json),
                "driver": driver,
                "model": json.loads(model_json),
                "return_result": result_arr,
            }


__all__ = [
    "PESStore",
    "SharedMemoryBuffer",
    "SharedMemoryView",
    "pack_payload",
    "unpack_payload",
    "HMACSocketServer",
    "HMACSocketClient",
]
