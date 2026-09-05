"""Cryptographic Merkle Provenance Stamper & Ed25519 Signer.

Provides chunk-level Merkle SHA-256 verification and Ed25519 asymmetric signature
generation and validation for tamper detection on HDF5 trajectory files.
"""

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import ConfigDict, Field
from pydantic.main import BaseModel


class ProvenanceMetadata(BaseModel):
    """Execution provenance metadata model for stamped HDF5 files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    git_commit_hash: str = Field(..., pattern=r"^[a-f0-9]{40,64}$")
    git_dirty_flag: bool
    python_version: str
    cochem_version: str
    environment_lock_hash: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    codata_version: str = Field(default="CODATA 2018")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cli_command: str
    thermodynamic_state: Dict[str, float] = Field(
        default_factory=lambda: {"T_K": 298.15, "P_atm": 1.0}
    )


class ProvenanceStamper:
    """Computes Merkle chunk digests and manages Ed25519 cryptographic provenance stamps."""

    @classmethod
    def compute_chunks_sha256(cls, h5_file: h5py.File) -> str:
        """Compute Merkle SHA-256 digest across all dataset chunks and metadata.

        Synchronizes file buffers to disk and deterministically visits all datasets
        in lexicographical order, reading raw chunk payloads directly from HDF5 filter
        pipelines without array memory allocations.

        Args:
            h5_file: Open h5py.File instance.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        h5_file.flush()
        hasher = hashlib.sha256()

        dset_paths: List[str] = []

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                dset_paths.append(name)

        h5_file.visititems(_visitor)
        dset_paths.sort()

        for path in dset_paths:
            dset = h5_file[path]
            hasher.update(path.encode("utf-8"))
            hasher.update(str(dset.dtype).encode("utf-8"))
            hasher.update(str(dset.shape).encode("utf-8"))

            if dset.chunks is not None:
                for chunk_info in dset.iter_chunks():
                    chunk_offset = tuple(
                        chunk_info[i].start or 0 for i in range(len(chunk_info))
                    )
                    try:
                        raw_chunk = dset.id.read_direct_chunk(chunk_offset)
                        chunk_bytes = (
                            raw_chunk[1]
                            if isinstance(raw_chunk, tuple)
                            else raw_chunk
                        )
                        hasher.update(chunk_bytes)
                    except RuntimeError as exc:
                        err_msg = str(exc).lower()
                        if "not allocated" in err_msg or "chunk storage" in err_msg:
                            hasher.update(b"UNALLOCATED_CHUNK")
                        else:
                            raise
            else:
                contiguous_bytes = np.ascontiguousarray(dset[()]).tobytes()
                hasher.update(contiguous_bytes)

        return hasher.hexdigest()

    @classmethod
    def stamp(
        cls,
        file_path: Path,
        metadata: ProvenanceMetadata,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> str:
        """Stamp HDF5 file with Merkle chunk root digest, metadata, and optional Ed25519 signature.

        Args:
            file_path: Path to existing HDF5 file.
            metadata: Execution provenance metadata instance.
            private_key: Optional Ed25519 private key for signing.

        Returns:
            Base64-encoded signature string (empty string if no private key provided).
        """
        resolved_path = Path(file_path).resolve()
        with h5py.File(resolved_path, "r+") as f:
            chunk_digest = cls.compute_chunks_sha256(f)

            meta_dict = metadata.model_dump(mode="json")
            meta_dict["chunk_sha256_root"] = chunk_digest

            canonical_json = json.dumps(
                meta_dict, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")

            sig_b64 = ""
            if private_key is not None:
                sig_bytes = private_key.sign(canonical_json)
                sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

            f.attrs["git_commit_hash"] = metadata.git_commit_hash
            f.attrs["git_dirty_flag"] = metadata.git_dirty_flag
            f.attrs["python_version"] = metadata.python_version
            f.attrs["cochem_version"] = metadata.cochem_version
            f.attrs["environment_lock_hash"] = metadata.environment_lock_hash
            f.attrs["codata_version"] = metadata.codata_version
            f.attrs["timestamp_utc"] = metadata.timestamp_utc.isoformat()
            f.attrs["cli_command"] = metadata.cli_command
            f.attrs["thermodynamic_state"] = json.dumps(
                metadata.thermodynamic_state, sort_keys=True
            )

            f.attrs["canonical_manifest_json"] = canonical_json.decode("utf-8")
            f.attrs["chunk_sha256_root"] = chunk_digest
            f.attrs["provenance_signature"] = sig_b64

            f.flush()
            return sig_b64

    @classmethod
    def verify(
        cls,
        file_path: Path,
        public_key: Optional[ed25519.Ed25519PublicKey] = None,
    ) -> bool:
        """Verify Merkle chunk integrity and optional cryptographic signature of stamped HDF5 file.

        Args:
            file_path: Path to HDF5 file.
            public_key: Optional Ed25519 public key to verify signature.

        Returns:
            True if all chunk digests and signatures are valid; False otherwise.
        """
        resolved_path = Path(file_path).resolve()
        try:
            with h5py.File(resolved_path, "r") as f:
                stored_manifest_raw = f.attrs.get("canonical_manifest_json", None)
                stored_sig = f.attrs.get("provenance_signature", "")

                if stored_manifest_raw is None:
                    return False

                if isinstance(stored_manifest_raw, bytes):
                    stored_manifest_str = stored_manifest_raw.decode("utf-8")
                    stored_manifest_bytes = stored_manifest_raw
                else:
                    stored_manifest_str = str(stored_manifest_raw)
                    stored_manifest_bytes = stored_manifest_str.encode("utf-8")

                try:
                    meta_dict = json.loads(stored_manifest_str)
                except Exception:
                    return False

                expected_chunk_hash = meta_dict.get("chunk_sha256_root", "")
                if not expected_chunk_hash:
                    return False

                try:
                    actual_chunk_hash = cls.compute_chunks_sha256(f)
                except (RuntimeError, OSError):
                    return False

                if expected_chunk_hash != actual_chunk_hash:
                    return False

                if public_key is not None:
                    if not stored_sig:
                        return False
                    try:
                        sig_bytes = base64.b64decode(stored_sig)
                        public_key.verify(sig_bytes, stored_manifest_bytes)
                    except (InvalidSignature, ValueError, TypeError):
                        return False

                return True
        except (OSError, RuntimeError):
            return False
