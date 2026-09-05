"""Deterministic Canonical JSON Serializer, HMAC-SHA256 Signer, and Tripartite Stager.

Strictly adhering to SRS Chunk 08 (REQ-MOB-071) and the Zero-Mock Mandate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mendeleev import element as _mendeleev_element
from pydantic import BaseModel

from cochem.mobile.job_state import ExecutionPayload, ManifestReference

STAGE_THRESHOLD_BYTES: int = 65536  # 64 KB exact threshold (65,536 bytes)
DEFAULT_HMAC_ENV_VAR: str = "COCHEM_HMAC_SECRET"
DEFAULT_HMAC_SECRET_FALLBACK: str = "cochem_airgap_secret_key_v1_secure_default"


def get_hmac_secret(explicit_secret: Optional[str] = None) -> str:
    """Resolve HMAC secret key from explicit argument or environment with fallback."""
    if explicit_secret is not None and explicit_secret.strip():
        return explicit_secret.strip()
    return os.environ.get(DEFAULT_HMAC_ENV_VAR, DEFAULT_HMAC_SECRET_FALLBACK)


def canonical_json_dumps(obj: Any) -> str:
    """Serialize object to deterministic canonical JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_serialize(data: Union[Dict[str, Any], BaseModel, ExecutionPayload]) -> bytes:
    """Serialize dictionary or Pydantic model into deterministic canonical UTF-8 bytes.

    Automatically excludes the 'hmac_sha256' signature field from the canonical hash representation.
    """
    if isinstance(data, ExecutionPayload):
        raw_dict = data.to_canonical_dict(exclude_signature=True)
    elif isinstance(data, BaseModel):
        raw_dict = data.model_dump()
        raw_dict.pop("hmac_sha256", None)
    elif isinstance(data, dict):
        raw_dict = dict(data)
        raw_dict.pop("hmac_sha256", None)
    else:
        raise TypeError(f"Unsupported payload type for canonical serialization: {type(data)}")

    return canonical_json_dumps(raw_dict).encode("utf-8")


def sign_payload(canonical_bytes: bytes, secret_key: Optional[str] = None) -> str:
    """Compute HMAC-SHA256 signature string over canonical UTF-8 bytes."""
    secret = get_hmac_secret(secret_key)
    return hmac.new(secret.encode("utf-8"), canonical_bytes, hashlib.sha256).hexdigest()


def verify_payload_signature(
    canonical_bytes: bytes, signature: str, secret_key: Optional[str] = None
) -> bool:
    """Verify HMAC-SHA256 signature using constant-time digest comparison."""
    if not signature or not signature.strip():
        return False
    expected_signature = sign_payload(canonical_bytes, secret_key)
    return hmac.compare_digest(expected_signature, signature.strip())


def get_coch_src() -> Path:
    """Get $COCH_SRC directory with robust environment resolution."""
    raw = os.environ.get("COCH_SRC")
    if raw and raw.strip():
        return Path(raw.strip()).resolve()
    # Default to parent directory / repo root / src
    cwd = Path.cwd()
    candidate = cwd / "src"
    if candidate.exists():
        return candidate.resolve()
    return cwd.resolve()


def get_coch_artifacts() -> Path:
    """Get $COCH_ARTIFACTS directory with robust environment resolution."""
    raw = os.environ.get("COCH_ARTIFACTS")
    if raw and raw.strip():
        return Path(raw.strip()).resolve()
    return (Path.cwd() / "artifacts").resolve()


def get_cochem_state_dir() -> Path:
    """Get $COCHEM_STATE_DIR directory with robust environment resolution."""
    raw = os.environ.get("COCHEM_STATE_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).resolve()
    return (Path.cwd() / "state").resolve()


def get_job_artifact_dir(job_id: str, base_artifacts_dir: Optional[Path] = None) -> Path:
    """Construct POSIX-normalized job artifact directory path ($COCH_ARTIFACTS/jobs/{job_id}/)."""
    base = base_artifacts_dir or get_coch_artifacts()
    return base / "jobs" / job_id


def get_job_status_path(job_id: str, state_dir: Optional[Path] = None) -> Path:
    """Construct job status JSON path ($COCHEM_STATE_DIR/{job_id}.status.json)."""
    base = state_dir or get_cochem_state_dir()
    return base / f"{job_id}.status.json"


def get_job_lock_path(job_id: str, state_dir: Optional[Path] = None) -> Path:
    """Construct job lock file path ($COCHEM_STATE_DIR/{job_id}.lock)."""
    base = state_dir or get_cochem_state_dir()
    return base / f"{job_id}.lock"


def ensure_tripartite_dirs(
    job_id: str,
    src_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    state_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """Ensure Tripartite Air-Gap Isolation directories exist and return (src, job_artifacts, state)."""
    resolved_src = (src_dir or get_coch_src()).resolve()
    resolved_artifacts = (artifacts_dir or get_coch_artifacts()).resolve()
    resolved_state = (state_dir or get_cochem_state_dir()).resolve()

    resolved_job_artifacts = resolved_artifacts / "jobs" / job_id

    resolved_src.mkdir(parents=True, exist_ok=True)
    resolved_job_artifacts.mkdir(parents=True, exist_ok=True)
    resolved_state.mkdir(parents=True, exist_ok=True)

    return resolved_src, resolved_job_artifacts, resolved_state


def validate_xyz_structure_dynamic(xyz_block: str) -> List[Tuple[str, float, float, float, float]]:
    """Dynamically validate XYZ coordinate elements and look up atomic masses via Mendeleev.

    Returns:
        List of tuples: (element_symbol, x, y, z, dynamic_atomic_weight)
    """
    lines = [ln.strip() for ln in xyz_block.strip().splitlines() if ln.strip()]
    if not lines:
        return []

    # Check for standard XYZ header (atom count on line 0, comment on line 1)
    coord_lines = lines
    if len(lines) >= 2:
        first_token = lines[0].split()[0]
        if first_token.isdigit() and len(lines[0].split()) == 1:
            coord_lines = lines[2:]

    parsed_atoms: List[Tuple[str, float, float, float, float]] = []
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0].strip()
        # Clean symbol of non-alpha characters if any
        clean_sym = "".join(c for c in sym if c.isalpha())
        if not clean_sym:
            continue

        # Dynamic retrieval from mendeleev library
        elem = _mendeleev_element(clean_sym.capitalize())
        atomic_mass = float(elem.atomic_weight or elem.mass or 0.0)

        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        parsed_atoms.append((elem.symbol, x, y, z, atomic_mass))

    return parsed_atoms


def calculate_xyz_molecular_mass_dynamic(xyz_block: str) -> float:
    """Calculate total molecular mass dynamically using Mendeleev IUPAC atomic weights."""
    atoms = validate_xyz_structure_dynamic(xyz_block)
    return sum(atom[4] for atom in atoms)


def stage_or_inline_payload(
    payload: ExecutionPayload,
    secret_key: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[bool, Union[ExecutionPayload, ManifestReference]]:
    """Evaluate payload size threshold (64 KB / 65,536 bytes) and route inline vs staged manifest.

    Returns:
        (is_staged, ExecutionPayload_or_ManifestReference)
        - If size <= 65,536 bytes: (False, ExecutionPayload with attached hmac_sha256)
        - If size > 65,536 bytes: (True, ManifestReference pointing to staged payload.json)
    """
    canonical_bytes = canonical_serialize(payload)
    signature = sign_payload(canonical_bytes, secret_key)
    payload_size = len(canonical_bytes)

    if payload_size <= STAGE_THRESHOLD_BYTES:
        # Inline route
        inline_payload = payload.model_copy(deep=True)
        inline_payload.hmac_sha256 = signature
        return False, inline_payload

    # Staged manifest route (> 64 KB)
    job_art_dir = (artifacts_dir or get_job_artifact_dir(payload.job_id)).resolve()
    job_art_dir.mkdir(parents=True, exist_ok=True)

    target_file = job_art_dir / "payload.json"
    tmp_file = job_art_dir / "payload.json.tmp"

    # Atomic write to temporary file then replace
    with open(tmp_file, "wb") as f:
        f.write(canonical_bytes)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, target_file)

    manifest = ManifestReference(
        job_id=payload.job_id,
        manifest_uri=target_file.as_posix(),
        file_size_bytes=payload_size,
        hmac_sha256=signature,
        created_at_utc=payload.created_at_utc,
    )
    return True, manifest


def load_staged_payload(
    manifest_ref: ManifestReference, secret_key: Optional[str] = None
) -> ExecutionPayload:
    """Load, verify HMAC-SHA256 signature, and deserialize staged payload.json from manifest reference."""
    file_path = Path(manifest_ref.manifest_uri)
    if not file_path.exists():
        raise FileNotFoundError(f"Staged payload file does not exist: {manifest_ref.manifest_uri}")

    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) != manifest_ref.file_size_bytes:
        raise ValueError(
            f"Staged payload size mismatch: expected {manifest_ref.file_size_bytes} bytes, "
            f"got {len(raw_bytes)} bytes."
        )

    if not verify_payload_signature(raw_bytes, manifest_ref.hmac_sha256, secret_key):
        raise ValueError(
            f"Cryptographic HMAC-SHA256 integrity verification failed for payload {manifest_ref.job_id}"
        )

    parsed_json = json.loads(raw_bytes.decode("utf-8"))
    payload = ExecutionPayload.model_validate(parsed_json)
    payload.hmac_sha256 = manifest_ref.hmac_sha256
    return payload
