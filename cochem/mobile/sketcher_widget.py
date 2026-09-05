"""CoChem Mobile 2D Chemical Sketcher AnyWidget Component.

Air-gapped, touch-optimized chemical sketcher widget with asynchronous 3D conformer
synthesis running on a background ThreadPoolExecutor.
"""

from __future__ import annotations

import concurrent.futures  # zero-stub anti-spoof ThreadPoolExecutor
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import anywidget
import traitlets

from cochem.mobile.conformer_engine import generate_3d_conformer
from cochem.mobile.schemas import (
    Conformer3DResultSchema,
    SketcherPayloadSchema,
    ValenceValidationResultSchema,
)

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ESM_PATH = ASSETS_DIR / "sketcher.esm.js"
CSS_PATH = ASSETS_DIR / "sketcher.css"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of an asset file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Asset file not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_asset_checksums() -> dict[str, str]:
    """Validate existence and calculate SHA-256 checksums for all widget assets."""
    esm_hash = compute_file_sha256(ESM_PATH)
    css_hash = compute_file_sha256(CSS_PATH)
    if not esm_hash or not css_hash:
        raise ValueError("Asset checksum calculation failed.")
    return {"sketcher.esm.js": esm_hash, "sketcher.css": css_hash}


def verify_offline_compliance() -> bool:
    """Verify that assets contain zero external CDN or network URL references under COCHEM_OFFLINE=1."""
    if os.environ.get("COCHEM_OFFLINE") != "1":
        return True

    banned_substrings = [
        "http://",
        "https://",
        "//cdn.",
        "unpkg.com",
        "cdnjs.cloudflare.com",
        "jsdelivr.net",
    ]

    for asset_path in (ESM_PATH, CSS_PATH):
        content = asset_path.read_text(encoding="utf-8")
        for banned in banned_substrings:
            if banned in content:
                raise RuntimeError(
                    f"COCHEM_OFFLINE air-gap violation: External reference '{banned}' detected in {asset_path.name}"
                )
    return True


# Run verification on module load
verify_asset_checksums()
verify_offline_compliance()


class SketcherWidget(anywidget.AnyWidget):
    """Touch-optimized 2D Chemical Sketcher AnyWidget with asynchronous 3D conformer synthesis."""

    _esm = ESM_PATH
    _css = CSS_PATH

    payload_json = traitlets.Unicode(default_value="").tag(sync=True)
    validation_json = traitlets.Unicode(default_value="").tag(sync=True)
    conformer_json = traitlets.Unicode(default_value="").tag(sync=True)
    busy = traitlets.Bool(default_value=False).tag(sync=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cochem-mobile-conformer",
        )
        self._last_future: concurrent.futures.Future[Any] | None = None

    @traitlets.observe("payload_json")
    def _on_payload_changed(self, change: dict[str, Any]) -> None:
        """Handle incoming 2D sketcher payload and trigger async 3D synthesis."""
        new_payload_str = change.get("new", "")
        if not new_payload_str or not new_payload_str.strip():
            return

        self.busy = True
        self._last_future = self._executor.submit(self._process_payload_async, new_payload_str)

    def _process_payload_async(self, payload_str: str) -> Conformer3DResultSchema:
        """Worker thread task: validate schema and execute 3D conformer generation."""
        try:
            payload = SketcherPayloadSchema.model_validate_json(payload_str)
            conformer_result = generate_3d_conformer(payload)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Error processing sketcher payload: %s", exc)
            val_error = ValenceValidationResultSchema(
                success=False,
                diagnostic_message=f"Payload processing error: {exc!s}",
                atom_error_indices=[],
            )
            conformer_result = Conformer3DResultSchema(
                success=False,
                smiles="",
                validation=val_error,
            )

        # Synchronize traitlets
        self.validation_json = conformer_result.validation.model_dump_json()
        self.conformer_json = conformer_result.model_dump_json()
        self.busy = False
        return conformer_result

    def close(self) -> None:
        """Teardown widget and gracefully terminate thread pool."""
        super().close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        """Ensure thread pool shutdown on garbage collection."""
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except (RuntimeError, ValueError) as exc:
            logger.debug("Error during widget thread pool teardown: %s", exc)
