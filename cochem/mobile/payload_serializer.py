"""Deterministic Canonical JSON Serializer, HMAC-SHA256 Signer, and Tripartite Stager.

Strictly adhering to SRS Chunk 08 (REQ-MOB-071) and the Zero-Mock Mandate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from cochem.mobile.job_state import ExecutionPayload, ManifestReference

STAGE_THRESHOLD_BYTES: int = 65536  # 64 KB exact threshold (65,536 bytes)
DEFAULT_HMAC_ENV_VAR: str = "COCHEM_HMAC_SECRET"


def get_hmac_secret(explicit_secret: Optional[str] = None) -> str:
    """Require an explicit or configured nonblank key, preserving its exact bytes.

    An explicitly empty key is rejected even when the environment contains a key.
    Keys are not trimmed, guessed, generated, logged or replaced by a public value.
    """
    secret = (
        os.environ.get(DEFAULT_HMAC_ENV_VAR)
        if explicit_secret is None
        else explicit_secret
    )
    if not isinstance(secret, str) or not secret.strip():
        raise ValueError(
            "Configure a nonempty authentication key explicitly or through "
            "COCHEM_HMAC_SECRET before signing or receiving payloads."
        )
    return secret


def canonical_json_dumps(obj: Any) -> str:
    """Serialize object to deterministic canonical JSON string."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def strict_json_loads(raw: Union[str, bytes]) -> Any:
    """Read JSON without nonfinite numbers or silently overwritten object keys."""

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON numbers must be finite.")
        return parsed

    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON number: {value}")

    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_float=finite_float,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def validate_job_id(job_id: str) -> str:
    """Require an opaque identifier that is safe as a single filename component."""
    if (
        not isinstance(job_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job_id) is None
    ):
        raise ValueError(
            "job_id must be 1-128 ASCII letters, digits, underscores or hyphens."
        )
    return job_id


def canonical_serialize(
    data: Union[Dict[str, Any], BaseModel, ExecutionPayload],
) -> bytes:
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
        raise TypeError(
            f"Unsupported payload type for canonical serialization: {type(data)}"
        )

    return canonical_json_dumps(raw_dict).encode("utf-8")


def sign_payload(canonical_bytes: bytes, secret_key: Optional[str] = None) -> str:
    """Compute HMAC-SHA256 signature string over canonical UTF-8 bytes."""
    secret = get_hmac_secret(secret_key)
    return hmac.new(secret.encode("utf-8"), canonical_bytes, hashlib.sha256).hexdigest()


def verify_payload_signature(
    canonical_bytes: bytes, signature: str, secret_key: Optional[str] = None
) -> bool:
    """Verify HMAC-SHA256 signature using constant-time digest comparison."""
    secret = get_hmac_secret(secret_key)
    if (
        not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", signature.strip()) is None
    ):
        return False
    expected_signature = sign_payload(canonical_bytes, secret)
    return hmac.compare_digest(expected_signature, signature.strip().lower())


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


def get_job_artifact_dir(
    job_id: str, base_artifacts_dir: Optional[Path] = None
) -> Path:
    """Construct POSIX-normalized job artifact directory path ($COCH_ARTIFACTS/jobs/{job_id}/)."""
    base = base_artifacts_dir or get_coch_artifacts()
    return base / "jobs" / validate_job_id(job_id)


def get_job_status_path(job_id: str, state_dir: Optional[Path] = None) -> Path:
    """Construct job status JSON path ($COCHEM_STATE_DIR/{job_id}.status.json)."""
    base = state_dir or get_cochem_state_dir()
    return base / f"{validate_job_id(job_id)}.status.json"


def get_job_lock_path(job_id: str, state_dir: Optional[Path] = None) -> Path:
    """Construct job lock file path ($COCHEM_STATE_DIR/{job_id}.lock)."""
    base = state_dir or get_cochem_state_dir()
    return base / f"{validate_job_id(job_id)}.lock"


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

    resolved_job_artifacts = resolved_artifacts / "jobs" / validate_job_id(job_id)

    resolved_src.mkdir(parents=True, exist_ok=True)
    resolved_job_artifacts.mkdir(parents=True, exist_ok=True)
    resolved_state.mkdir(parents=True, exist_ok=True)

    return resolved_src, resolved_job_artifacts, resolved_state


def validate_xyz_structure_dynamic(
    xyz_block: str,
) -> List[Tuple[str, float, float, float, float]]:
    """Validate one standard XYZ frame without dropping atoms or isotope identity.

    Returns ``(isotope_label, x, y, z, tabulated_mass_u)`` per atom. Explicit
    isotopes use their actual database mass. Bare elements select the most
    abundant natural isotope under the shared, explicitly documented mass policy.
    These tabulated masses are not exact SI constants or IUPAC average weights.
    """
    from Libraries.cochem_isotopes import isotope_record

    if not isinstance(xyz_block, str) or not xyz_block.strip():
        raise ValueError("XYZ coordinates must be nonempty.")
    # Keep physical line positions: the standard XYZ comment may be blank.
    lines = xyz_block.splitlines()
    if re.fullmatch(r"[+-]?[0-9]+", lines[0].strip()) is None:
        raise ValueError(
            "Standard XYZ requires an integer atom count on its first line."
        )
    declared_count = int(lines[0].strip())
    if declared_count <= 0 or len(lines) < 2:
        raise ValueError("XYZ requires a positive atom count and a comment line.")
    coord_lines = lines[2:]
    if len(coord_lines) != declared_count:
        raise ValueError(
            f"XYZ atom count mismatch: declared {declared_count}, got {len(coord_lines)}."
        )

    parsed_atoms: List[Tuple[str, float, float, float, float]] = []
    for index, line in enumerate(coord_lines, start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Invalid XYZ atom record {index}: expected element and x, y, z."
            )
        try:
            x, y, z = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"Invalid XYZ coordinates in atom record {index}."
            ) from exc
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError(f"XYZ coordinates in atom record {index} must be finite.")
        resolved_isotope = isotope_record(parts[0])
        parsed_atoms.append(
            (resolved_isotope["label"], x, y, z, resolved_isotope["mass_u"])
        )

    return parsed_atoms


def calculate_xyz_molecular_mass_dynamic(xyz_block: str) -> float:
    """Sum actual isotope masses using the shared explicit/default isotope policy."""
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
    validate_job_id(payload.job_id)
    validate_xyz_structure_dynamic(payload.molecule_xyz)
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
    # Missing authentication is an error before any file contents are read.
    resolved_secret = get_hmac_secret(secret_key)
    validate_job_id(manifest_ref.job_id)
    file_path = Path(manifest_ref.manifest_uri)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Staged payload file does not exist: {manifest_ref.manifest_uri}"
        )

    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) != manifest_ref.file_size_bytes:
        raise ValueError(
            f"Staged payload size mismatch: expected {manifest_ref.file_size_bytes} bytes, "
            f"got {len(raw_bytes)} bytes."
        )

    if not verify_payload_signature(
        raw_bytes, manifest_ref.hmac_sha256, resolved_secret
    ):
        raise ValueError(
            f"Cryptographic HMAC-SHA256 integrity verification failed for payload {manifest_ref.job_id}"
        )

    parsed_json = strict_json_loads(raw_bytes.decode("utf-8"))
    if not isinstance(parsed_json, dict):
        raise ValueError("Staged JSON payload must be a root object.")
    unexpected = set(parsed_json) - (
        set(ExecutionPayload.model_fields) - {"hmac_sha256"}
    )
    if unexpected:
        raise ValueError("Staged payload contains unsupported fields.")
    payload = ExecutionPayload.model_validate(parsed_json)
    if payload.job_id != manifest_ref.job_id:
        raise ValueError("Staged payload job_id does not match the manifest identity.")
    if payload.created_at_utc != manifest_ref.created_at_utc:
        raise ValueError(
            "Staged payload created_at_utc does not match the manifest identity."
        )
    validate_xyz_structure_dynamic(payload.molecule_xyz)
    payload.hmac_sha256 = manifest_ref.hmac_sha256
    return payload
