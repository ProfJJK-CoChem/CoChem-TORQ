"""Launch the packaged student notebook from an installed TORQ environment."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from hashlib import sha256
from importlib.resources import files
from pathlib import Path


def prepare_notebook(workspace: str | Path) -> Path:
    """Create an independent notebook copy; preserve existing student files."""
    root = Path(workspace).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("The student interface workspace cannot be a symlink.")
    payload = files("UI").joinpath("Start_TORQ.ipynb").read_bytes()
    directory = root / "UI"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    notebook = directory / f"Start_TORQ-{sha256(payload).hexdigest()[:12]}.ipynb"
    if notebook.is_symlink():
        raise ValueError("The student notebook cannot be a symlink.")
    if notebook.exists():
        if not notebook.is_file():
            raise ValueError("The student notebook must be a regular file.")
    else:
        with notebook.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        notebook.chmod(0o600)
    return notebook


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cochem-torq-ui",
        description=(
            "Open the packaged TORQ student interface; calculations require review."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "CoChem_Artifacts" / "torq-interface",
    )
    arguments = parser.parse_args(argv)
    if not 1024 <= arguments.port <= 65535:
        parser.error("Choose a nonprivileged port from 1024 through 65535.")
    notebook = prepare_notebook(arguments.workspace)
    from .cli import main as cli_main

    return cli_main(
        [
            "interface",
            "--host",
            arguments.host,
            "--port",
            str(arguments.port),
            "--notebook",
            str(notebook),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
