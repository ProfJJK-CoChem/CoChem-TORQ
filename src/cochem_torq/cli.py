"""Student application CLI. Nonzero exits distinguish invalid, failed and partial runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

from .domain import CalculationRequest, PrerequisiteError, canonical_json, read_json


def _emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _parser():
    parser = argparse.ArgumentParser(
        prog="cochem-torq",
        description="Real calculations in GitHub Actions; the student interface in Codespaces.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "recipes", help="List exact recipe availability and scientific limitations"
    )
    commands.add_parser(
        "matrix", help="List all 140 preserved method-matrix identities"
    )
    for name in ("validate", "execute", "submit", "_worker"):
        command = commands.add_parser(
            name,
            help=argparse.SUPPRESS
            if name == "_worker"
            else f"{name.capitalize()} an explicit calculation request",
        )
        command.add_argument("--request", required=True, type=Path)
        if name in {"execute", "_worker"}:
            command.add_argument("--output-dir", required=True, type=Path)
        if name == "validate":
            command.add_argument(
                "--execution",
                choices=["github_actions", "local_validation"],
                default="github_actions",
            )
        if name == "execute":
            command.add_argument("--request-id")
            command.add_argument("--expected-sha256")
    status = commands.add_parser("status", help="Read a real hosted Actions run status")
    status.add_argument("run_id")
    download = commands.add_parser(
        "download", help="Download and verify genuine hosted result bundles"
    )
    download.add_argument("run_id")
    download.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="Verify the complete inventory of an immutable worker shard"
    )
    verify.add_argument("directory", type=Path)
    merge = commands.add_parser(
        "merge", help="Merge independently sealed, matching worker shards"
    )
    merge.add_argument("shards", nargs="+", type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    export_base = commands.add_parser(
        "export-base", help="Export a verified actual electronic result for BASE"
    )
    export_base.add_argument("directory", type=Path)
    export_base.add_argument("--destination", required=True, type=Path)
    interface = commands.add_parser(
        "interface", help="Start the actual JupyterLab student notebook"
    )
    interface.add_argument("--host", default="127.0.0.1")
    interface.add_argument("--port", type=int, default=8888)
    interface.add_argument("--notebook", type=Path, default=Path("UI/Start_TORQ.ipynb"))
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "recipes":
            from .registry import list_method_profiles

            _emit(list_method_profiles())
        elif arguments.command == "matrix":
            from .registry import matrix_index

            _emit(matrix_index())
        elif arguments.command in {"validate", "execute", "submit", "_worker"}:
            request = read_json(arguments.request)
            from .application import (
                execute_request,
                submit_request,
                validate_request,
                worker_execute,
            )

            if arguments.command == "validate":
                checked = validate_request(request, execution=arguments.execution)
                _emit(checked)
                return 0 if checked["executable"] else 3
            if arguments.command == "submit":
                _emit(submit_request(request))
            elif arguments.command == "_worker":
                model = CalculationRequest.model_validate(request)
                result = worker_execute(model, arguments.output_dir)
                (arguments.output_dir / "result.json").write_bytes(
                    canonical_json(result) + b"\n"
                )
                return (
                    0
                    if result["status"] == "complete"
                    else 4
                    if result["status"] == "partial"
                    else 5
                )
            elif arguments.command == "execute":
                if (
                    arguments.expected_sha256
                    and sha256(arguments.request.read_bytes()).hexdigest()
                    != arguments.expected_sha256
                ):
                    raise ValueError("Request transport digest mismatch.")
                if (
                    arguments.request_id
                    and str(request.get("request_id")) != arguments.request_id
                ):
                    raise ValueError("Request UUID mismatch.")
                result = execute_request(request, arguments.output_dir)
                _emit(
                    {
                        "status": result["status"],
                        "request_id": result["request_id"],
                        "output_directory": str(arguments.output_dir.resolve()),
                        "identification_ready": result["identification_ready"],
                        "experimental_accuracy_established": result[
                            "experimental_accuracy_established"
                        ],
                    }
                )
                return (
                    0
                    if result["status"] == "complete"
                    else 4
                    if result["status"] == "partial"
                    else 5
                )
        elif arguments.command == "status":
            from .application import get_run_status

            _emit(get_run_status(arguments.run_id))
        elif arguments.command == "download":
            from .application import download_run_results

            _emit(
                [
                    str(path)
                    for path in download_run_results(
                        arguments.run_id, arguments.destination
                    )
                ]
            )
        elif arguments.command == "verify":
            from .artifacts import verify_shard

            manifest = verify_shard(arguments.directory)
            _emit(
                {
                    "status": "verified",
                    "request_id": manifest["request_id"],
                    "files": len(manifest["files"]),
                    "scope": "byte integrity and matching identities; scientific accuracy is separate",
                }
            )
        elif arguments.command == "merge":
            from .artifacts import merge_shards

            _emit(
                {
                    "output_directory": str(
                        merge_shards(arguments.shards, arguments.output_dir)
                    )
                }
            )
        elif arguments.command == "export-base":
            from .ecosystem import export_base_calculation_result

            _emit(
                {
                    "output_file": str(
                        export_base_calculation_result(
                            arguments.directory, arguments.destination
                        )
                    )
                }
            )
        elif arguments.command == "interface":
            if not 1 <= arguments.port <= 65535:
                raise ValueError("Port must be in 1..65535.")
            notebook = arguments.notebook.resolve()
            if not notebook.is_file():
                raise ValueError(
                    "The student notebook is missing; run from the repository or supply --notebook."
                )
            # exec preserves the owned PID/start-time tracked by Codespaces lifecycle.
            command = [
                sys.executable,
                "-m",
                "jupyterlab",
                "--no-browser",
                f"--ServerApp.ip={arguments.host}",
                f"--ServerApp.port={arguments.port}",
                "--ServerApp.port_retries=0",
                f"--ServerApp.root_dir={notebook.parent.parent}",
                f"--ServerApp.default_url=/lab/tree/{notebook.relative_to(notebook.parent.parent)}",
            ]
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                command.append("--allow-root")
            os.execv(sys.executable, command)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        _emit(
            {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
        )
        return (
            3
            if isinstance(exc, PrerequisiteError)
            else 2
            if isinstance(exc, ValueError)
            else 5
        )


if __name__ == "__main__":
    raise SystemExit(main())
