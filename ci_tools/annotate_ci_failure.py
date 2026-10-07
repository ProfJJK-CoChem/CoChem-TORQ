"""Expose bounded diagnostics from actual failed CI files to check annotations."""

from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit


def annotate(message: str) -> None:
    # Only diagnostic values are emitted. Redact any present injected credential
    # if an upstream tool unexpectedly includes it in its error text.
    secret_values = set()
    for name, value in os.environ.items():
        if value and not name.endswith("_CONFIGURED") and any(
            marker in name.upper()
            for marker in (
                "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                "AUTHORIZATION", "API_KEY", "PRIVATE_KEY",
            )
        ):
            secret_values.add(value)
        # Package index and proxy credentials often reside in URL variables,
        # whose names do not contain TOKEN/SECRET. Remove encoded and decoded
        # components even when diagnostics print only a credential fragment.
        try:
            url = urlsplit(value)
            if url.scheme and "@" in url.netloc:
                for component in (url.username, url.password):
                    if component:
                        secret_values.update((component, unquote(component)))
        except ValueError:
            pass
    message = re.sub(
        r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/?#@]+@",
        r"\1[redacted]@",
        message,
    )
    for value in sorted(secret_values, key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    message = message[-6000:]
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error::{escaped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-log", type=Path)
    group.add_argument("--junit", type=Path)
    arguments = parser.parse_args()
    if arguments.build_log:
        with arguments.build_log.open("rb") as source:
            source.seek(0, 2)
            offset = max(0, source.tell() - 24000)
            source.seek(offset)
            data = source.read()
            if offset:
                # A bounded read can begin in the middle of a credential. Drop
                # that incomplete first line before any diagnostic is emitted.
                boundary = data.find(b"\n")
                data = data[boundary + 1:] if boundary >= 0 else b""
            tail = data.decode("utf-8", errors="replace")
        annotate("The actual calculation image build failed. Build log tail:\n" + tail)
        return
    if not arguments.junit.is_file():
        annotate("The actual pytest process failed before producing its JUnit report.")
        return
    root = ET.parse(arguments.junit).getroot()
    reported = 0
    for case in root.iter("testcase"):
        detail = case.find("failure")
        if detail is None:
            detail = case.find("error")
        if detail is None:
            continue
        identity = f"{case.get('classname', '')}.{case.get('name', '')}"
        annotate(
            f"Actual failed test: {identity}\n"
            f"{detail.text or detail.get('message', '')}"
        )
        reported += 1
        if reported == 8:
            break
    if not reported:
        annotate("The actual pytest process failed; no failed test case was recorded.")


if __name__ == "__main__":
    main()
