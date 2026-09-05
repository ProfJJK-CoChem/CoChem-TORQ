"""
CoChem Mobile PWA Subsystem.

Invariants:
- Standalone PWA shell integration for iOS/Android.
- Service Worker static asset caching and dynamic route bypass.
- Tornado HTTP route handlers and HTML header injection.
"""

from __future__ import annotations

from cochem.mobile.pwa.shell import (
    ManifestHandler,
    ServiceWorkerHandler,
    get_manifest_headers,
    get_manifest_path,
    get_pwa_dir,
    get_pwa_meta_tags,
    get_pwa_tornado_handlers,
    get_service_worker_headers,
    get_sw_path,
    inject_pwa_headers,
)

__all__ = [
    "ManifestHandler",
    "ServiceWorkerHandler",
    "get_manifest_headers",
    "get_manifest_path",
    "get_pwa_dir",
    "get_pwa_meta_tags",
    "get_pwa_tornado_handlers",
    "get_service_worker_headers",
    "get_sw_path",
    "inject_pwa_headers",
]
