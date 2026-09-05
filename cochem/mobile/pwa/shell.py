"""
Voila & Tornado PWA Shell Integration.

Invariants:
- Inject PWA headers, viewport, manifest link, and service worker registration into HTML templates.
- Tornado HTTP handlers serving sw.js with Service-Worker-Allowed: / and no-cache.
- Tornado HTTP handlers serving manifest.json with application/manifest+json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

try:
    import tornado.web

    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False


def get_pwa_dir() -> Path:
    """Return the filesystem directory containing PWA assets."""
    return Path(__file__).parent.resolve()


def get_manifest_path() -> Path:
    """Return the absolute path to manifest.json."""
    return get_pwa_dir() / "manifest.json"


def get_sw_path() -> Path:
    """Return the absolute path to sw.js."""
    return get_pwa_dir() / "sw.js"


def get_pwa_meta_tags() -> str:
    """Return HTML snippet of PWA meta tags and manifest link."""
    return (
        '    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">\n'
        '    <meta name="apple-mobile-web-app-capable" content="yes">\n'
        '    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">\n'
        '    <meta name="theme-color" content="#1a1a2e">\n'
        '    <link rel="manifest" href="/manifest.json">\n'
        '    <link rel="apple-touch-icon" href="/static/icons/cochem-icon-192.png">\n'
        "    <script>\n"
        '      if ("serviceWorker" in navigator) {\n'
        '        window.addEventListener("load", () => {\n'
        '          navigator.serviceWorker.register("/sw.js", { scope: "/" })\n'
        '            .then((reg) => console.log("[CoChem PWA] SW registered:", reg.scope))\n'
        '            .catch((err) => console.warn("[CoChem PWA] SW registration failed:", err));\n'
        "        });\n"
        "      }\n"
        "    </script>"
    )


def inject_pwa_headers(html_content: str) -> str:
    """Inject PWA meta tags and SW registration into an HTML string before </head>."""
    tags = get_pwa_meta_tags()
    if "</head>" in html_content:
        return html_content.replace("</head>", f"{tags}\n</head>", 1)
    elif "<head>" in html_content:
        return html_content.replace("<head>", f"<head>\n{tags}", 1)
    else:
        return f"<head>\n{tags}\n</head>\n{html_content}"


def get_service_worker_headers() -> Dict[str, str]:
    """Return standard HTTP headers for serving sw.js."""
    return {
        "Content-Type": "application/javascript; charset=utf-8",
        "Service-Worker-Allowed": "/",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def get_manifest_headers() -> Dict[str, str]:
    """Return standard HTTP headers for serving manifest.json."""
    return {
        "Content-Type": "application/manifest+json; charset=utf-8",
        "Cache-Control": "public, max-age=3600",
    }


if TORNADO_AVAILABLE:

    class ServiceWorkerHandler(tornado.web.RequestHandler):
        """Tornado handler serving the PWA Service Worker with required headers."""

        def get(self) -> None:
            sw_file = get_sw_path()
            if not sw_file.exists():
                self.set_status(404)
                self.write("Service worker script not found")
                return

            for header, value in get_service_worker_headers().items():
                self.set_header(header, value)

            with open(sw_file, "r", encoding="utf-8") as f:
                self.write(f.read())

    class ManifestHandler(tornado.web.RequestHandler):
        """Tornado handler serving manifest.json."""

        def get(self) -> None:
            manifest_file = get_manifest_path()
            if not manifest_file.exists():
                self.set_status(404)
                self.write("Manifest file not found")
                return

            for header, value in get_manifest_headers().items():
                self.set_header(header, value)

            with open(manifest_file, "r", encoding="utf-8") as f:
                self.write(f.read())

    def get_pwa_tornado_handlers() -> List[Tuple[str, Type[tornado.web.RequestHandler]]]:
        """Return Tornado route mappings for PWA endpoints."""
        return [
            (r"/sw\.js", ServiceWorkerHandler),
            (r"/manifest\.json", ManifestHandler),
        ]

else:  # pragma: no cover
    ServiceWorkerHandler = None  # type: ignore
    ManifestHandler = None  # type: ignore

    def get_pwa_tornado_handlers() -> List[Any]:  # type: ignore
        return []
