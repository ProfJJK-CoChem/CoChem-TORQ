"""
Zero-Trust Web Push Engine (RFC 8291 & RFC 8292).

Invariants:
- Real Cryptography via cryptography.hazmat (NIST P-256, ECDH, HKDF-SHA256, AES-128-GCM).
- Zero-Trust Payload Sanitization: Excludes SMILES, coordinates, paths, tracebacks.
- Tripartite Air-Gap Enforcement: Prohibits SaaS relays and enforces local dispatch.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

import base64
import ipaddress
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from cochem.telemetry.exceptions import (
    AirGapPushBlockedError,
    PayloadSanitizationViolationError,
    VAPIDEncryptionError,
)
from cochem.telemetry.schemas import (
    PushNotificationPayload,
    PushSubscriptionEndpoint,
)

# Prohibited third-party SaaS relays
PROHIBITED_SAAS_DOMAINS = (
    "api.telegram.org",
    "telegram.org",
    "discord.com",
    "discordapp.com",
    "api.pushover.net",
    "pushover.net",
    "fcm.googleapis.com",
    "googleapis.com",
    "hooks.slack.com",
    "slack.com",
    "pusher.com",
    "onesignal.com",
)

# Regex patterns for sensitive chemical IP & unredacted tracebacks
SENSITIVE_PATTERNS = [
    # Molecular coordinates (XYZ format lines: Element x y z)
    re.compile(
        r"^[A-Za-z]{1,2}\s+[-+]?\d+\.\d{3,}\s+[-+]?\d+\.\d{3,}\s+[-+]?\d+\.\d{3,}", re.MULTILINE
    ),
    # Aromatic ring patterns (e.g. c1ccccc1, c1ccc(CC)cc1)
    re.compile(r"c1[a-zA-Z0-9\(\)\=\#\-\\\/]{3,12}1"),
    # Branched/bonded SMILES notation (e.g. CC(=O)Oc1, C(=O)O, C#N)
    re.compile(r"(?:[A-Z][a-z]?(?:\([^\)]+\)|[=\#\-\\\/])){2,}[A-Z][a-z]?"),
    # InChI and InChIKey
    re.compile(r"InChI(?:Key)?=", re.IGNORECASE),
    # Filesystem paths
    re.compile(
        r"(?:[A-Za-z]:[\\\/]|\/(?:home|Users|tmp|var|etc|root|opt|\.cochem)[\/\\])", re.IGNORECASE
    ),
    # Unredacted tracebacks & quantum chemistry logs
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r'File ".*?", line \d+', re.IGNORECASE),
    re.compile(r"\*\s*O\s*R\s*C\s*A\s*\*", re.IGNORECASE),
    re.compile(r"ORCA TERMINATED NORMALLY", re.IGNORECASE),
    re.compile(r"TOTAL SCF ENERGY", re.IGNORECASE),
]


def b64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_decode(data: str) -> bytes:
    """Decode base64url string with optional padding."""
    padded = data + "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def is_airgap_mode() -> bool:
    """Check if Tripartite Air-Gap Mode is active."""
    val = os.environ.get("COCHEM_AIRGAP_MODE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def check_airgap_compliance(endpoint_url: str) -> None:
    """
    Validate that the push endpoint adheres to Air-Gap and SaaS prohibition rules.

    Raises:
        AirGapPushBlockedError: If a SaaS relay or non-local endpoint is invoked in air-gap mode.
    """
    parsed = urlparse(endpoint_url)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        raise AirGapPushBlockedError(f"Invalid push endpoint URL: '{endpoint_url}'")

    # Always block unauthorized SaaS relays
    for saas in PROHIBITED_SAAS_DOMAINS:
        if hostname == saas or hostname.endswith(f".{saas}"):
            raise AirGapPushBlockedError(
                f"Unauthorized SaaS relay '{hostname}' is strictly prohibited in CoChem architecture."
            )

    # In Air-Gap Mode, require strictly local or on-premise endpoints
    if is_airgap_mode():
        is_local = False
        if hostname in ("localhost", "127.0.0.1", "::1"):
            is_local = True
        elif (
            hostname.endswith(".local")
            or hostname.endswith(".internal")
            or hostname.endswith(".lan")
        ):
            is_local = True
        else:
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback:
                    is_local = True
            except ValueError as _e:
                logger.debug(f"Ignored exception: {_e}")

        if not is_local:
            raise AirGapPushBlockedError(
                f"External push endpoint '{hostname}' blocked under Tripartite Air-Gap Mode."
            )


def sanitize_push_payload(
    payload: Union[PushNotificationPayload, Dict[str, Any]],
) -> PushNotificationPayload:
    """
    Sanitize and validate push notification payload against chemical IP leaks.

    Raises:
        PayloadSanitizationViolationError: If chemical structures, coordinates, paths, or tracebacks exist.
    """
    if isinstance(payload, dict):
        try:
            validated = PushNotificationPayload.model_validate(payload)
        except Exception as err:
            raise PayloadSanitizationViolationError(
                f"Push payload validation failed: {err}"
            ) from err
    else:
        validated = payload

    # Inspect all string fields for sensitive patterns
    text_fields = [
        validated.summary,
        validated.job_id,
    ]

    for text in text_fields:
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                raise PayloadSanitizationViolationError(
                    f"Sensitive IP or trace violation detected in push payload field: '{pattern.pattern}'"
                )

    return validated


class VAPIDPushEngine:
    """
    Standards-compliant RFC 8291 / RFC 8292 Web Push Engine using NIST P-256 cryptography.
    """

    def __init__(
        self,
        private_key: Optional[ec.EllipticCurvePrivateKey] = None,
        public_key_b64: Optional[str] = None,
        claim_email: Optional[str] = None,
    ) -> None:
        if private_key is not None:
            self._private_key = private_key
        elif (
            "COCHEM_VAPID_PRIVATE_KEY" in os.environ
            and os.environ["COCHEM_VAPID_PRIVATE_KEY"].strip()
        ):
            raw_priv = os.environ["COCHEM_VAPID_PRIVATE_KEY"].strip()
            self._private_key = self._load_private_key(raw_priv)
        else:
            self._private_key = ec.generate_private_key(ec.SECP256R1())

        self._public_key = self._private_key.public_key()
        self._public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        self._public_key_b64 = public_key_b64 or b64url_encode(self._public_key_bytes)
        self._claim_email = (
            claim_email or os.environ.get("COCHEM_VAPID_CLAIM_EMAIL", "admin@cochem.local").strip()
        )

    @property
    def public_key_b64(self) -> str:
        return self._public_key_b64

    @property
    def claim_email(self) -> str:
        return self._claim_email

    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """Generate a new NIST P-256 VAPID key pair (private_key_b64, public_key_b64)."""
        priv = ec.generate_private_key(ec.SECP256R1())
        priv_num = priv.private_numbers().private_value
        priv_bytes = priv_num.to_bytes(32, "big")
        priv_b64 = b64url_encode(priv_bytes)

        pub = priv.public_key()
        pub_bytes = pub.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        pub_b64 = b64url_encode(pub_bytes)
        return priv_b64, pub_b64

    def _load_private_key(self, raw: str) -> ec.EllipticCurvePrivateKey:
        """Load private key from PEM or base64url scalar."""
        try:
            if "BEGIN" in raw:
                loaded = serialization.load_pem_private_key(raw.encode("utf-8"), password=None)
                if isinstance(loaded, ec.EllipticCurvePrivateKey):
                    return loaded
            priv_bytes = b64url_decode(raw)
            priv_int = int.from_bytes(priv_bytes, "big")
            return ec.derive_private_key(priv_int, ec.SECP256R1())
        except Exception as err:
            raise VAPIDEncryptionError(f"Failed to load VAPID private key: {err}") from err

    def create_vapid_auth_header(
        self,
        endpoint_url: str,
        claim_email: Optional[str] = None,
        expiration_seconds: int = 43200,
    ) -> Dict[str, str]:
        """
        Generate RFC 8292 ES256 VAPID JWT Authorization header.
        """
        check_airgap_compliance(endpoint_url)

        parsed = urlparse(endpoint_url)
        audience = f"{parsed.scheme}://{parsed.netloc}"
        sub = claim_email or self._claim_email
        if not sub.startswith("mailto:") and not sub.startswith("https://"):
            sub = f"mailto:{sub}"

        exp = int(time.time()) + min(expiration_seconds, 86400)

        header_dict = {"typ": "JWT", "alg": "ES256"}
        payload_dict = {
            "aud": audience,
            "exp": exp,
            "sub": sub,
        }

        header_b64 = b64url_encode(json.dumps(header_dict, separators=(",", ":")).encode("utf-8"))
        payload_b64 = b64url_encode(json.dumps(payload_dict, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

        try:
            der_signature = self._private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
            r, s = utils.decode_dss_signature(der_signature)
            raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
            sig_b64 = b64url_encode(raw_sig)
            jwt = f"{header_b64}.{payload_b64}.{sig_b64}"
        except Exception as err:
            raise VAPIDEncryptionError(
                f"ECDSA signing failed during VAPID generation: {err}"
            ) from err

        return {
            "Authorization": f"vapid t={jwt}, k={self._public_key_b64}",
            "Crypto-Key": f"p256ecdsa={self._public_key_b64}",
        }

    def encrypt_payload(
        self,
        payload_bytes: bytes,
        client_public_key_b64: str,
        client_auth_b64: str,
    ) -> bytes:
        """
        Encrypt payload using RFC 8291 (aes128gcm content coding).
        """
        try:
            client_pub_bytes = b64url_decode(client_public_key_b64)
            auth_secret = b64url_decode(client_auth_b64)

            if len(client_pub_bytes) != 65 or client_pub_bytes[0] != 4:
                raise ValueError("Invalid client P-256 uncompressed public key length/format")
            if len(auth_secret) != 16:
                raise ValueError(f"Auth secret must be 16 bytes, got {len(auth_secret)}")

            client_pub_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), client_pub_bytes
            )

            # Generate ephemeral server ECDH key pair
            server_priv = ec.generate_private_key(ec.SECP256R1())
            server_pub = server_priv.public_key()
            server_pub_bytes = server_pub.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )

            # Compute ECDH shared secret
            ecdh_secret = server_priv.exchange(ec.ECDH(), client_pub_key)

            # Step 1: Derive IKM via HKDF-SHA256
            auth_info = b"WebPush: info\x00" + client_pub_bytes + server_pub_bytes
            ikm = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=auth_secret,
                info=auth_info,
            ).derive(ecdh_secret)

            # Step 2: Derive CEK and Nonce
            salt = os.urandom(16)
            cek_info = b"Content-Encoding: aes128gcm\x00"
            cek = HKDF(
                algorithm=hashes.SHA256(),
                length=16,
                salt=salt,
                info=cek_info,
            ).derive(ikm)

            nonce_info = b"Content-Encoding: nonce\x00"
            nonce = HKDF(
                algorithm=hashes.SHA256(),
                length=12,
                salt=salt,
                info=nonce_info,
            ).derive(ikm)

            # Step 3: Record padding and AES-GCM encryption
            # RFC 8291 padding delimiter \x02 for final record
            padded_plaintext = payload_bytes + b"\x02"
            aesgcm = AESGCM(cek)
            encrypted_record = aesgcm.encrypt(nonce, padded_plaintext, None)

            # Step 4: Binary header (16 salt + 4 rs + 1 idlen + 65 server_pub)
            rs = 4096
            header = (
                salt + rs.to_bytes(4, "big") + bytes([len(server_pub_bytes)]) + server_pub_bytes
            )

            return header + encrypted_record

        except Exception as err:
            raise VAPIDEncryptionError(f"RFC 8291 payload encryption failed: {err}") from err

    @staticmethod
    def decrypt_payload(
        encrypted_body: bytes,
        client_private_key: ec.EllipticCurvePrivateKey,
        client_auth_b64: str,
    ) -> bytes:
        """
        Physical decryption of RFC 8291 aes128gcm body using client keys for verification.
        """
        try:
            auth_secret = b64url_decode(client_auth_b64)
            salt = encrypted_body[:16]
            rs = int.from_bytes(encrypted_body[16:20], "big")
            if rs < 18:
                raise ValueError(f"Invalid RFC 8291 record size: {rs}")
            idlen = encrypted_body[20]
            server_pub_bytes = encrypted_body[21 : 21 + idlen]
            ciphertext = encrypted_body[21 + idlen :]

            server_pub = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), server_pub_bytes
            )
            client_pub = client_private_key.public_key()
            client_pub_bytes = client_pub.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )

            ecdh_secret = client_private_key.exchange(ec.ECDH(), server_pub)

            auth_info = b"WebPush: info\x00" + client_pub_bytes + server_pub_bytes
            ikm = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=auth_secret,
                info=auth_info,
            ).derive(ecdh_secret)

            cek_info = b"Content-Encoding: aes128gcm\x00"
            cek = HKDF(
                algorithm=hashes.SHA256(),
                length=16,
                salt=salt,
                info=cek_info,
            ).derive(ikm)

            nonce_info = b"Content-Encoding: nonce\x00"
            nonce = HKDF(
                algorithm=hashes.SHA256(),
                length=12,
                salt=salt,
                info=nonce_info,
            ).derive(ikm)

            aesgcm = AESGCM(cek)
            padded = aesgcm.decrypt(nonce, ciphertext, None)

            # Strip padding delimiter
            if padded[-1:] == b"\x02":
                return padded[:-1]
            # Handle padded trailing zeroes before \x02
            idx = padded.rfind(b"\x02")
            if idx != -1:
                return padded[:idx]
            return padded

        except Exception as err:
            raise VAPIDEncryptionError(f"RFC 8291 payload decryption failed: {err}") from err

    def prepare_push_dispatch(
        self,
        endpoint: PushSubscriptionEndpoint,
        payload: Union[PushNotificationPayload, Dict[str, Any]],
        ttl: int = 60,
    ) -> Dict[str, Any]:
        """
        Validate, sanitize, encrypt, and generate headers for Web Push dispatch.
        """
        # Step 1: Air-gap check
        check_airgap_compliance(endpoint.endpoint)

        # Step 2: Payload sanitization
        sanitized = sanitize_push_payload(payload)
        payload_bytes = sanitized.model_dump_json().encode("utf-8")

        # Step 3: RFC 8291 Encryption
        encrypted_body = self.encrypt_payload(
            payload_bytes=payload_bytes,
            client_public_key_b64=endpoint.keys.p256dh,
            client_auth_b64=endpoint.keys.auth,
        )

        # Step 4: RFC 8292 Auth Headers
        auth_headers = self.create_vapid_auth_header(endpoint.endpoint)
        headers = {
            **auth_headers,
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "aes128gcm",
            "TTL": str(ttl),
        }

        return {
            "endpoint": endpoint.endpoint,
            "headers": headers,
            "body": encrypted_body,
            "sanitized_payload": sanitized,
        }
