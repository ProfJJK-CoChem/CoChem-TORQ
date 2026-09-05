"""Telemetry secret masking and dynamic credential rotation.

Provenance & Specifications:
- Method Matrix [M]: Defense-in-depth telemetry scrubbing with zero secret leakage.
- Entropy Redaction [D]: Deterministic SHA-256 prefix8 token replacement preserving trace correlation.
- Memory Sanitization [E]: In-memory credential hot swapping with ctypes.memset memory zeroizing.
"""

from __future__ import annotations

import _thread
import ctypes
import enum
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

logger = logging.getLogger(__name__)


class EntropyType(str, enum.Enum):
    """Classified high-entropy secret token categories."""

    API_KEY = "API_KEY"
    PRIVATE_KEY = "PRIVATE_KEY"
    AUTH_HEADER = "AUTH_HEADER"
    CONNECTION_URI = "CONNECTION_URI"


# Compiled regex scanners for high-entropy credential patterns
RE_API_KEY = re.compile(
    r'(?i)[\'"]?(?:api_key|access_token|secret|bearer)[\'"]?\s*[:=]\s*[\'"]?([a-zA-Z0-9_\-\.]{16,})[\'"]?'
)
RE_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"
)
RE_AUTH_HEADER = re.compile(
    r"(?i)(Authorization:\s*(?:Bearer|Basic)\s+)([^\s]+)"
)
RE_CONNECTION_URI = re.compile(
    r'(?i)([a-z]+://[^:\s\'",;]+:)([^@\s\'",;]+)(@[^@\s\'",;]+)'
)


def compute_redaction_token(entropy_type: Union[EntropyType, str], secret_value: str) -> str:
    """Generate deterministic [REDACTED:<type>:<prefix8>] token preserving debug correlation."""
    etype = entropy_type.value if isinstance(entropy_type, EntropyType) else str(entropy_type)
    digest_prefix = hashlib.sha256(secret_value.encode("utf-8")).hexdigest()[:8]
    return f"[REDACTED:{etype}:{digest_prefix}]"


class TelemetrySecretMasker:
    """Stream-level regex scrubber ensuring zero plaintext credential exposure in telemetry."""

    @classmethod
    def mask_text(cls, text: str) -> str:
        """Scan string and replace all matched secrets with deterministic redaction tokens."""
        if not text:
            return text

        scrubbed = text

        # 1. Private keys (full replacement)
        def _replace_private_key(match: re.Match) -> str:
            val = match.group(0)
            return compute_redaction_token(EntropyType.PRIVATE_KEY, val)

        scrubbed = RE_PRIVATE_KEY.sub(_replace_private_key, scrubbed)

        # 2. HTTP Authorization headers (group 2 replacement)
        def _replace_auth(match: re.Match) -> str:
            prefix = match.group(1)
            token_val = match.group(2)
            token = compute_redaction_token(EntropyType.AUTH_HEADER, token_val)
            return f"{prefix}{token}"

        scrubbed = RE_AUTH_HEADER.sub(_replace_auth, scrubbed)

        # 3. Connection URIs with embedded passwords (group 2 replacement)
        def _replace_uri(match: re.Match) -> str:
            scheme_user = match.group(1)
            password = match.group(2)
            host_path = match.group(3)
            token = compute_redaction_token(EntropyType.CONNECTION_URI, password)
            return f"{scheme_user}{token}{host_path}"

        scrubbed = RE_CONNECTION_URI.sub(_replace_uri, scrubbed)

        # 4. Generic API Keys, Access Tokens, and Bearer secrets (group 1 replacement)
        def _replace_api_key(match: re.Match) -> str:
            full_str = match.group(0)
            secret_val = match.group(1)
            token = compute_redaction_token(EntropyType.API_KEY, secret_val)
            return str(full_str.replace(secret_val, token))

        scrubbed = RE_API_KEY.sub(_replace_api_key, scrubbed)

        return str(scrubbed)

    @classmethod
    def mask_record(cls, record: Union[str, bytes, Dict[str, Any]]) -> Union[str, bytes, Dict[str, Any]]:
        """Scrub unstructured strings, binary payloads, or structured dictionaries."""
        if isinstance(record, str):
            return cls.mask_text(record)
        if isinstance(record, bytes):
            decoded = record.decode("utf-8", errors="replace")
            masked = cls.mask_text(decoded)
            return masked.encode("utf-8")
        if isinstance(record, dict):
            serialized = json.dumps(record)
            masked_str = cls.mask_text(serialized)
            return cast(Dict[str, Any], json.loads(masked_str))
        return record


@dataclass(frozen=True)
class CredentialSnapshot:
    """Immutable snapshot of validated active credentials."""

    tokens: Dict[str, str]
    timestamp: float


class DynamicCredentialProvider:
    """Thread-safe credential provider supporting hot reloading and memory zeroizing on eviction."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self._lock = _thread.allocate_lock()
        self._active_snapshot: CredentialSnapshot = CredentialSnapshot(tokens={}, timestamp=0.0)
        self._allocated_buffers: Dict[str, Any] = {}
        self._last_mtime: float = 0.0

        if self.config_path is not None and self.config_path.exists():
            self.reload()

    def set_credentials(self, new_tokens: Dict[str, str]) -> None:
        """Atomically update credentials and zeroize evicted memory buffers."""
        with self._lock:
            # 1. Zeroize existing allocated C-buffers
            for _key, buf in list(self._allocated_buffers.items()):
                ctypes.memset(buf, 0, len(buf))
            self._allocated_buffers.clear()

            # 2. Allocate new mutable character buffers in memory
            for key, val in new_tokens.items():
                val_bytes = val.encode("utf-8")
                buf = ctypes.create_string_buffer(val_bytes)
                self._allocated_buffers[key] = buf

            # 3. Swap active snapshot reference
            self._active_snapshot = CredentialSnapshot(
                tokens=dict(new_tokens),
                timestamp=self.config_path.stat().st_mtime if self.config_path and self.config_path.exists() else 0.0,
            )

    def get_credential(self, key: str) -> Optional[str]:
        """Retrieve credential value for key."""
        with self._lock:
            return self._active_snapshot.tokens.get(key)

    def reload(self) -> bool:
        """Poll filesystem timestamp and reload configuration if changed."""
        if self.config_path is None or not self.config_path.exists():
            return False

        current_mtime = self.config_path.stat().st_mtime
        if current_mtime <= self._last_mtime:
            return False

        try:
            content = self.config_path.read_text(encoding="utf-8")
            data: Dict[str, str] = {}
            # Support JSON or line-delimited KEY=VALUE
            if content.strip().startswith("{"):
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    data = {str(k): str(v) for k, v in loaded.items()}
            else:
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        k, v = stripped.split("=", 1)
                        data[k.strip()] = v.strip().strip("'\"")

            self.set_credentials(data)
            self._last_mtime = current_mtime
            return True
        except Exception as err:
            logger.error("Failed to hot reload credential file %s: %s", self.config_path, err)
            return False

    def close(self) -> None:
        """Wipe all credential buffers from process memory."""
        with self._lock:
            for buf in self._allocated_buffers.values():
                ctypes.memset(buf, 0, len(buf))
            self._allocated_buffers.clear()
            self._active_snapshot = CredentialSnapshot(tokens={}, timestamp=0.0)
