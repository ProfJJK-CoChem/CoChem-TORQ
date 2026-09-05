"""Offline-First Progressive Web App (PWA) Cache Controller.

Module: cochem.mobile.pwa_cache
Implements local SQLite Write-Ahead Logging (WAL) cache controller for offline molecular calculation queues.
- ACID durability: PRAGMA journal_mode=WAL, PRAGMA synchronous=NORMAL, PRAGMA busy_timeout=10000.
- Non-zero atom count verification and dynamic chemical symbol lookup via Mendeleev.
- Pairwise Euclidean distance verification enforcing r_ij >= 0.5 Angstroms against atomic overlap catastrophe.
- Backend idempotency via deterministic SHA-256 canonical JSON digest.
- Non-blocking FIFO synchronization with exponential backoff and replay deduplication.
- Configurable offline storage quota enforcement raising OfflineStorageExceededError.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from mendeleev import element as _mendeleev_element
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_QUOTA_BYTES: int = 500 * 1024 * 1024  # 500 MB
DEFAULT_BUSY_TIMEOUT_MS: int = 10000
DEFAULT_BASE_BACKOFF_SECONDS: float = 1.0
DEFAULT_MAX_BACKOFF_SECONDS: float = 300.0
DEFAULT_MAX_RETRIES: int = 5
MIN_INTERATOMIC_SEPARATION_ANGSTROMS: float = 0.5


class OfflineStorageExceededError(Exception):
    """Raised when offline cache storage exceeds the configured maximum quota."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PWACacheValidationError(Exception):
    """Raised when molecular payload fails physical constraints or schema validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class QueueStatus(str, Enum):
    """Status lifecycle for queued offline molecular calculations."""

    PENDING = "PENDING"
    SYNCED = "SYNCED"
    FAILED = "FAILED"


class SyncReport(BaseModel):
    """Pydantic model representing queue synchronization execution metrics."""

    model_config = ConfigDict(frozen=True)

    synced_count: int = 0
    failed_count: int = 0
    deduplicated_count: int = 0
    status: str = "COMPLETED"
    synced_ids: List[str] = Field(default_factory=list)
    deduplicated_ids: List[str] = Field(default_factory=list)
    failed_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    total_processed: int = 0


class QueuedJobRecord(BaseModel):
    """Pydantic representation of an offline calculation job in the SQLite queue."""

    model_config = ConfigDict(frozen=True)

    payload_id: str
    payload_json: str
    payload_type: str
    status: QueueStatus
    created_at: str
    sync_attempts: int
    last_attempt: Optional[str] = None
    size_bytes: int


def verify_coordinate_distances(
    atoms: List[Tuple[str, float, float, float]],
    min_separation_angstroms: float = MIN_INTERATOMIC_SEPARATION_ANGSTROMS,
) -> None:
    """Verify positive pairwise Euclidean coordinate separations between all atoms.

    Calculates r_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2) for all pairs i < j.
    Enforces r_ij >= min_separation_angstroms (0.5 Angstroms) to prevent atomic overlap catastrophe.
    """
    n_atoms = len(atoms)
    if n_atoms < 2:
        return

    for i in range(n_atoms):
        sym_i, x_i, y_i, z_i = atoms[i]
        for j in range(i + 1, n_atoms):
            sym_j, x_j, y_j, z_j = atoms[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dz = z_i - z_j
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_separation_angstroms:
                raise PWACacheValidationError(
                    f"Atomic overlap catastrophe detected: distance between atom {i} ({sym_i}) "
                    f"and atom {j} ({sym_j}) is {dist:.4f} Angstroms (< {min_separation_angstroms} Angstroms threshold)."
                )


def validate_smiles_payload(smiles_str: str) -> int:
    """Validate molecular SMILES representation using RDKit and dynamic Mendeleev verification."""
    clean_smiles = smiles_str.strip()
    if not clean_smiles:
        raise PWACacheValidationError("SMILES payload string is empty.")

    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        raise PWACacheValidationError(
            f"Invalid SMILES specification: '{clean_smiles}' could not be parsed."
        )

    atom_count = mol.GetNumAtoms()
    if atom_count <= 0:
        raise PWACacheValidationError("Molecular SMILES specification contains zero atoms.")

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        try:
            elem = _mendeleev_element(sym)
            if not elem or not elem.symbol:
                raise ValueError(f"No element returned for symbol '{sym}'")
        except Exception as exc:
            raise PWACacheValidationError(
                f"Invalid chemical symbol '{sym}' in SMILES: {exc}"
            ) from exc

    return atom_count


def validate_xyz_payload(
    xyz_str: str,
    min_separation_angstroms: float = MIN_INTERATOMIC_SEPARATION_ANGSTROMS,
) -> List[Tuple[str, float, float, float]]:
    """Validate 3D .xyz molecular payload, dynamically resolving symbols via Mendeleev."""
    lines = [ln.strip() for ln in xyz_str.strip().splitlines() if ln.strip()]
    if not lines:
        raise PWACacheValidationError("XYZ payload block contains zero lines.")

    coord_lines = lines
    if len(lines) >= 2:
        first_token = lines[0].split()[0]
        if first_token.isdigit() and len(lines[0].split()) == 1:
            coord_lines = lines[2:]

    parsed_atoms: List[Tuple[str, float, float, float]] = []
    for idx, line in enumerate(coord_lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        raw_sym = parts[0].strip()
        clean_sym = "".join(c for c in raw_sym if c.isalpha()).capitalize()
        if not clean_sym:
            raise PWACacheValidationError(
                f"XYZ line {idx + 1} lacks a valid chemical symbol: '{line}'"
            )

        try:
            elem = _mendeleev_element(clean_sym)
            if not elem or not elem.symbol:
                raise ValueError(f"No element returned for '{clean_sym}'")
        except Exception as exc:
            raise PWACacheValidationError(
                f"Invalid chemical symbol '{clean_sym}' in XYZ line {idx + 1}: {exc}"
            ) from exc

        try:
            x_f = float(parts[1])
            y_f = float(parts[2])
            z_f = float(parts[3])
        except ValueError as exc:
            raise PWACacheValidationError(
                f"XYZ line {idx + 1} contains non-numeric coordinates: '{line}'"
            ) from exc

        parsed_atoms.append((elem.symbol, x_f, y_f, z_f))

    if len(parsed_atoms) == 0:
        raise PWACacheValidationError("XYZ payload contains zero valid coordinate atom lines.")

    verify_coordinate_distances(parsed_atoms, min_separation_angstroms=min_separation_angstroms)
    return parsed_atoms


def validate_atoms_list_payload(
    atoms_list: List[Dict[str, Any]],
    min_separation_angstroms: float = MIN_INTERATOMIC_SEPARATION_ANGSTROMS,
) -> List[Tuple[str, float, float, float]]:
    """Validate list of atomic dictionaries with dynamic Mendeleev lookup and distance enforcement."""
    if not isinstance(atoms_list, list) or len(atoms_list) == 0:
        raise PWACacheValidationError("Atoms payload must be a non-empty list of atomic records.")

    parsed: List[Tuple[str, float, float, float]] = []
    for idx, item in enumerate(atoms_list):
        if not isinstance(item, dict):
            raise PWACacheValidationError(f"Atom at index {idx} must be a dictionary.")
        raw_sym = str(item.get("element", item.get("symbol", ""))).strip()
        clean_sym = "".join(c for c in raw_sym if c.isalpha()).capitalize()
        if not clean_sym:
            raise PWACacheValidationError(
                f"Atom at index {idx} has missing or empty chemical symbol."
            )

        try:
            elem = _mendeleev_element(clean_sym)
            if not elem or not elem.symbol:
                raise ValueError(f"No element returned for '{clean_sym}'")
        except Exception as exc:
            raise PWACacheValidationError(
                f"Invalid chemical symbol '{clean_sym}' at atom index {idx}: {exc}"
            ) from exc

        try:
            x_f = float(item["x"])
            y_f = float(item["y"])
            z_f = float(item["z"])
        except (KeyError, ValueError, TypeError) as exc:
            raise PWACacheValidationError(
                f"Atom at index {idx} contains invalid coordinates: {exc}"
            ) from exc

        parsed.append((elem.symbol, x_f, y_f, z_f))

    verify_coordinate_distances(parsed, min_separation_angstroms=min_separation_angstroms)
    return parsed


def validate_elements_and_coordinates_payload(
    elements: List[str],
    coordinates: List[List[float]],
    min_separation_angstroms: float = MIN_INTERATOMIC_SEPARATION_ANGSTROMS,
) -> List[Tuple[str, float, float, float]]:
    """Validate paired elements and coordinates lists with dynamic Mendeleev lookup."""
    if not isinstance(elements, list) or not isinstance(coordinates, list):
        raise PWACacheValidationError("Elements and coordinates must both be lists.")
    if len(elements) == 0:
        raise PWACacheValidationError("Elements list contains zero atoms.")
    if len(elements) != len(coordinates):
        raise PWACacheValidationError(
            f"Mismatched elements count ({len(elements)}) and coordinates count ({len(coordinates)})."
        )

    parsed: List[Tuple[str, float, float, float]] = []
    for idx, (sym_raw, coord_raw) in enumerate(zip(elements, coordinates, strict=True)):
        clean_sym = "".join(c for c in str(sym_raw) if c.isalpha()).capitalize()
        if not clean_sym:
            raise PWACacheValidationError(f"Element at index {idx} is invalid.")
        try:
            elem = _mendeleev_element(clean_sym)
            if not elem or not elem.symbol:
                raise ValueError(f"No element returned for '{clean_sym}'")
        except Exception as exc:
            raise PWACacheValidationError(
                f"Invalid chemical symbol '{clean_sym}' at index {idx}: {exc}"
            ) from exc

        if not isinstance(coord_raw, (list, tuple)) or len(coord_raw) != 3:
            raise PWACacheValidationError(
                f"Coordinate at index {idx} must be a 3D coordinate [x, y, z]."
            )
        try:
            x_f = float(coord_raw[0])
            y_f = float(coord_raw[1])
            z_f = float(coord_raw[2])
        except (ValueError, TypeError) as exc:
            raise PWACacheValidationError(
                f"Coordinate at index {idx} contains non-numeric values: {exc}"
            ) from exc

        parsed.append((elem.symbol, x_f, y_f, z_f))

    verify_coordinate_distances(parsed, min_separation_angstroms=min_separation_angstroms)
    return parsed


def validate_physical_payload(payload: Dict[str, Any]) -> str:
    """Validate molecular payload according to physical and chemical constraints.

    Returns the canonical payload type string ('xyz', 'smiles', or '3d_atoms').
    """
    if not isinstance(payload, dict) or not payload:
        raise PWACacheValidationError("Payload must be a non-empty dictionary.")

    has_validated = False
    detected_type = str(payload.get("type", "")).strip().lower()

    if "xyz" in payload:
        xyz_content = payload["xyz"]
        if not isinstance(xyz_content, str):
            raise PWACacheValidationError("Field 'xyz' must be a string.")
        validate_xyz_payload(xyz_content)
        has_validated = True
        if not detected_type:
            detected_type = "xyz"

    if "smiles" in payload:
        smiles_content = payload["smiles"]
        if not isinstance(smiles_content, str):
            raise PWACacheValidationError("Field 'smiles' must be a string.")
        validate_smiles_payload(smiles_content)
        has_validated = True
        if not detected_type:
            detected_type = "smiles"

    if "atoms" in payload:
        validate_atoms_list_payload(payload["atoms"])
        has_validated = True
        if not detected_type:
            detected_type = "3d_atoms"

    if "elements" in payload and "coordinates" in payload:
        validate_elements_and_coordinates_payload(payload["elements"], payload["coordinates"])
        has_validated = True
        if not detected_type:
            detected_type = "3d_atoms"

    if not has_validated:
        raise PWACacheValidationError(
            "Payload must contain valid molecular SMILES, 3D XYZ, or atomic coordinate specifications."
        )

    return detected_type or "molecular_calculation"


class PWACacheManager:
    """Offline-First Progressive Web App (PWA) Cache Controller.

    Provides high-integrity SQLite Write-Ahead Logging (WAL) storage for queueing
    molecular calculation jobs when network connectivity is lost, enforcing
    dynamic Mendeleev physical validations, positive coordinate separations,
    backend idempotency, and storage quotas.
    """

    def __init__(
        self,
        db_path: Union[Path, str] = "pwa_cache.db",
        max_storage_bytes: int = DEFAULT_STORAGE_QUOTA_BYTES,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        network_timeout_s: float = 10.0,
        transport_handler: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.max_storage_bytes = max_storage_bytes
        self.busy_timeout_ms = busy_timeout_ms
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.max_retries = max_retries
        self.network_timeout_s = network_timeout_s
        self.transport_handler = transport_handler

        self._init_sqlite_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure SQLite connection with WAL journal mode and busy timeout."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_ms / 1000.0)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_sqlite_schema(self) -> None:
        """Initialize the queued_calculations table schema and indexing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queued_calculations (
                    payload_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'SYNCED', 'FAILED')),
                    created_at TEXT NOT NULL,
                    sync_attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt TEXT,
                    size_bytes INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queued_fifo
                ON queued_calculations(status, created_at);
                """
            )

    def verify_wal_mode(self) -> str:
        """Query and return active SQLite journal mode."""
        with self._get_connection() as conn:
            cur = conn.execute("PRAGMA journal_mode;")
            row = cur.fetchone()
            return str(row[0]).lower() if row else "unknown"

    def verify_synchronous_mode(self) -> int:
        """Query and return active SQLite synchronous pragma value (1 = NORMAL)."""
        with self._get_connection() as conn:
            cur = conn.execute("PRAGMA synchronous;")
            row = cur.fetchone()
            return int(row[0]) if row else -1

    def verify_busy_timeout(self) -> int:
        """Query and return active SQLite busy timeout in milliseconds."""
        with self._get_connection() as conn:
            cur = conn.execute("PRAGMA busy_timeout;")
            row = cur.fetchone()
            return int(row[0]) if row else -1

    def get_storage_usage_bytes(self) -> int:
        """Calculate total payload bytes stored in the database."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM queued_calculations;")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_disk_usage_bytes(self) -> int:
        """Calculate physical disk storage occupied by the database and WAL files."""
        total_bytes = 0
        file_candidates = [
            self.db_path,
            self.db_path.with_name(self.db_path.name + "-wal"),
            self.db_path.with_name(self.db_path.name + "-shm"),
        ]
        for f in file_candidates:
            if f.exists():
                total_bytes += f.stat().st_size
        return total_bytes

    def queue_molecule(self, payload: Dict[str, Any]) -> str:
        """Validate molecular payload and commit it to the SQLite WAL queue.

        Returns deterministic SHA-256 payload digest. If the payload already exists,
        returns the existing digest without duplicate insertion.
        """
        payload_type = validate_physical_payload(payload)

        # Deterministic canonical serialization
        canonical_str = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        canonical_bytes = canonical_str.encode("utf-8")
        payload_id = hashlib.sha256(canonical_bytes).hexdigest()
        size_bytes = len(canonical_bytes)

        # Idempotency check: Return existing digest without duplicate insert
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT payload_id FROM queued_calculations WHERE payload_id = ?;",
                (payload_id,),
            )
            existing = cur.fetchone()
            if existing is not None:
                return payload_id

        # Quota enforcement: Prevent queue corruption by rejecting before insertion
        current_payload_bytes = self.get_storage_usage_bytes()
        if current_payload_bytes + size_bytes > self.max_storage_bytes:
            raise OfflineStorageExceededError(
                f"Offline storage quota exceeded: current queue ({current_payload_bytes} bytes) + "
                f"incoming payload ({size_bytes} bytes) exceeds limit ({self.max_storage_bytes} bytes)."
            )

        if self.max_storage_bytes >= 1024 * 1024:
            current_disk_bytes = self.get_disk_usage_bytes()
            if current_disk_bytes + size_bytes > self.max_storage_bytes:
                raise OfflineStorageExceededError(
                    f"Physical storage quota exceeded: database disk files ({current_disk_bytes} bytes) "
                    f"exceeds limit ({self.max_storage_bytes} bytes)."
                )

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO queued_calculations (
                    payload_id, payload_json, payload_type, status, created_at, sync_attempts, last_attempt, size_bytes
                ) VALUES (?, ?, ?, 'PENDING', ?, 0, NULL, ?);
                """,
                (payload_id, canonical_str, payload_type, now_iso, size_bytes),
            )

        return payload_id

    def synchronize_pending_queue(
        self,
        network_endpoint: str,
        transport_handler: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ) -> SyncReport:
        """Perform non-blocking FIFO synchronization with exponential backoff and replay deduplication.

        Dispatches pending items to network_endpoint via transport_handler or standard HTTP.
        """
        handler = transport_handler or self.transport_handler

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT payload_id, payload_json, payload_type, status, created_at, sync_attempts, last_attempt, size_bytes
                FROM queued_calculations
                WHERE status = 'PENDING'
                ORDER BY created_at ASC, rowid ASC;
                """
            )
            rows = cur.fetchall()

        if not rows:
            return SyncReport(
                synced_count=0,
                failed_count=0,
                deduplicated_count=0,
                status="NO_PENDING",
                synced_ids=[],
                deduplicated_ids=[],
                failed_ids=[],
                errors=[],
                total_processed=0,
            )

        synced_ids: List[str] = []
        deduplicated_ids: List[str] = []
        failed_ids: List[str] = []
        errors: List[str] = []

        now_utc = datetime.now(timezone.utc)

        for row in rows:
            payload_id = str(row["payload_id"])
            payload_json = str(row["payload_json"])
            sync_attempts = int(row["sync_attempts"])
            last_attempt_str = row["last_attempt"]

            # Exponential backoff verification
            if sync_attempts > 0 and last_attempt_str is not None:
                try:
                    last_attempt_dt = datetime.fromisoformat(str(last_attempt_str))
                    elapsed = (now_utc - last_attempt_dt).total_seconds()
                    backoff_window = min(
                        self.max_backoff_seconds,
                        self.base_backoff_seconds * (2 ** (sync_attempts - 1)),
                    )
                    if elapsed < backoff_window:
                        continue
                except ValueError as parse_err:
                    logger.warning(
                        "Malformed last_attempt timestamp '%s' for payload %s: %s",
                        last_attempt_str,
                        payload_id,
                        parse_err,
                    )

            payload_dict = json.loads(payload_json)
            now_iso = datetime.now(timezone.utc).isoformat()

            try:
                if handler is not None:
                    response_obj = handler(payload_dict, network_endpoint)
                else:
                    req = urllib.request.Request(
                        network_endpoint,
                        data=payload_json.encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "X-Payload-ID": payload_id,
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self.network_timeout_s) as response:
                        body_bytes = response.read()
                        response_obj = (
                            json.loads(body_bytes.decode("utf-8"))
                            if body_bytes
                            else {"status": "ok"}
                        )

                # Evaluate replay deduplication vs synced
                is_deduplicated = False
                if isinstance(response_obj, dict):
                    status_str = str(response_obj.get("status", "")).lower()
                    code_val = response_obj.get("code")
                    if (
                        response_obj.get("deduplicated") is True
                        or status_str in ("deduplicated", "already_exists", "duplicate")
                        or code_val in (409, 208)
                    ):
                        is_deduplicated = True
                    elif status_str in ("error", "failed") or "error" in response_obj:
                        err_msg = str(response_obj.get("error", "Remote transport failure"))
                        raise RuntimeError(err_msg)

                with self._get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE queued_calculations
                        SET status = 'SYNCED', last_attempt = ?, sync_attempts = sync_attempts + 1
                        WHERE payload_id = ?;
                        """,
                        (now_iso, payload_id),
                    )

                if is_deduplicated:
                    deduplicated_ids.append(payload_id)
                else:
                    synced_ids.append(payload_id)

            except urllib.error.HTTPError as http_err:
                # HTTP 409 Conflict represents replay deduplication
                if http_err.code in (409, 208):
                    with self._get_connection() as conn:
                        conn.execute(
                            """
                            UPDATE queued_calculations
                            SET status = 'SYNCED', last_attempt = ?, sync_attempts = sync_attempts + 1
                            WHERE payload_id = ?;
                            """,
                            (now_iso, payload_id),
                        )
                    deduplicated_ids.append(payload_id)
                else:
                    new_attempts = sync_attempts + 1
                    new_status = "FAILED" if new_attempts >= self.max_retries else "PENDING"
                    with self._get_connection() as conn:
                        conn.execute(
                            """
                            UPDATE queued_calculations
                            SET status = ?, last_attempt = ?, sync_attempts = ?
                            WHERE payload_id = ?;
                            """,
                            (new_status, now_iso, new_attempts, payload_id),
                        )
                    failed_ids.append(payload_id)
                    errors.append(
                        f"HTTP {http_err.code} error on payload {payload_id}: {http_err.reason}"
                    )

            except Exception as exc:
                new_attempts = sync_attempts + 1
                new_status = "FAILED" if new_attempts >= self.max_retries else "PENDING"
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE queued_calculations
                        SET status = ?, last_attempt = ?, sync_attempts = ?
                        WHERE payload_id = ?;
                        """,
                        (new_status, now_iso, new_attempts, payload_id),
                    )
                failed_ids.append(payload_id)
                errors.append(f"Transport error on payload {payload_id}: {str(exc)}")

        total_processed = len(synced_ids) + len(deduplicated_ids) + len(failed_ids)
        if total_processed == 0:
            final_status = "NO_PENDING"
        elif not failed_ids:
            final_status = "SUCCESS"
        elif synced_ids or deduplicated_ids:
            final_status = "PARTIAL"
        else:
            final_status = "FAILED"

        return SyncReport(
            synced_count=len(synced_ids),
            failed_count=len(failed_ids),
            deduplicated_count=len(deduplicated_ids),
            status=final_status,
            synced_ids=synced_ids,
            deduplicated_ids=deduplicated_ids,
            failed_ids=failed_ids,
            errors=errors,
            total_processed=total_processed,
        )

    def get_pending_count(self) -> int:
        """Return total number of items currently pending synchronization."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM queued_calculations WHERE status = 'PENDING';")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_total_count(self) -> int:
        """Return total number of items in the queue across all statuses."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM queued_calculations;")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_queue_status(self, payload_id: str) -> Optional[QueueStatus]:
        """Retrieve queue status for a specific payload ID."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT status FROM queued_calculations WHERE payload_id = ?;",
                (payload_id,),
            )
            row = cur.fetchone()
            return QueueStatus(row["status"]) if row else None

    def get_payload_record(self, payload_id: str) -> Optional[QueuedJobRecord]:
        """Retrieve the complete queued record for a specific payload ID."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT payload_id, payload_json, payload_type, status, created_at, sync_attempts, last_attempt, size_bytes
                FROM queued_calculations
                WHERE payload_id = ?;
                """,
                (payload_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return QueuedJobRecord(
                payload_id=row["payload_id"],
                payload_json=row["payload_json"],
                payload_type=row["payload_type"],
                status=QueueStatus(row["status"]),
                created_at=row["created_at"],
                sync_attempts=row["sync_attempts"],
                last_attempt=row["last_attempt"],
                size_bytes=row["size_bytes"],
            )

    def get_payload(self, payload_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the deserialized payload dictionary for a specific payload ID."""
        record = self.get_payload_record(payload_id)
        if not record:
            return None
        data = json.loads(record.payload_json)
        if isinstance(data, dict):
            return dict(data)
        return None

    def clear_synced_payloads(self) -> int:
        """Remove all SYNCED records from the queue, returning the count of deleted rows."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM queued_calculations WHERE status = 'SYNCED';")
            return cur.rowcount

    def purge_all(self) -> None:
        """Purge all records from the queue."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM queued_calculations;")
