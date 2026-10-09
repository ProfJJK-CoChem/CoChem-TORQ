"""Student CLI with explicit invalid, failed and partial exit codes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, NoReturn

from .domain import CalculationRequest, PrerequisiteError, canonical_json, read_json


def _emit(
    value: object,
    *,
    structured: bool = False,
    status: str | None = None,
    request_id: str | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> None:
    if structured:
        value = {
            "schema_version": "cochem.torq.cli-response/1",
            "request_id": request_id
            or (value.get("request_id") if isinstance(value, dict) else None),
            "status": status
            or (
                value.get("status", "available")
                if isinstance(value, dict)
                else "available"
            ),
            "data": value,
            "errors": errors or [],
        }
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="cochem-torq",
        description=(
            "Real calculations in GitHub Actions; the student interface in Codespaces."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the versioned service response contract",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "recipes", help="List exact recipe availability and scientific limitations"
    )
    commands.add_parser(
        "matrix", help="List all 140 preserved method-matrix identities"
    )
    for name in ("validate", "execute", "_worker"):
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
            command.add_argument("--approved-plan", type=Path)
            command.add_argument("--expected-approval-sha256")
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
    reviewed = commands.add_parser(
        "import-topos-reviewed",
        help="Preserve an exact reviewed TOPOS member as a new TORQ request",
    )
    reviewed.add_argument("--handoff", required=True, type=Path)
    reviewed.add_argument("--producer-python", required=True, type=Path)
    reviewed.add_argument("--member", required=True)
    reviewed.add_argument("--recipe", required=True)
    reviewed.add_argument("--product", action="append", required=True)
    reviewed.add_argument("--output-dir", required=True, type=Path)
    reviewed.add_argument("--expected-handoff-sha256")
    reviewed.add_argument("--cores", type=int, default=1)
    reviewed.add_argument("--memory-mb", type=int, default=2048)
    reviewed.add_argument("--wall-seconds", type=int, default=600)
    merge = commands.add_parser(
        "merge", help="Merge independently sealed, matching worker shards"
    )
    merge.add_argument("shards", nargs="+", type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    export_base = commands.add_parser(
        "export-base",
        aliases=["export"],
        help="Export a verified actual electronic result for BASE",
    )
    export_base.add_argument("directory", type=Path)
    export_base.add_argument("--destination", required=True, type=Path)
    export_base.add_argument(
        "--format", choices=["base", "publication"], default="base"
    )
    candidates = commands.add_parser(
        "candidates", help="Inspect and reversibly select actual retained candidates"
    )
    candidates.add_argument(
        "action",
        choices=["list", "inspect", "register", "retain", "exclude", "restore"],
    )
    candidates.add_argument("--ledger", required=True, type=Path)
    candidates.add_argument("--candidate-id")
    candidates.add_argument("--revision", type=int)
    candidates.add_argument("--request", type=Path)
    candidates.add_argument("--actor")
    candidates.add_argument("--reason")
    compare = commands.add_parser(
        "geometry-compare",
        help="Compare supplied explicit graphs with proper rotations",
    )
    compare.add_argument("--source", required=True, type=Path)
    compare.add_argument("--target", required=True, type=Path)
    compare.add_argument("--policy", required=True, type=Path)
    compare.add_argument("--output", type=Path)
    symmetry = commands.add_parser(
        "symmetry-proposal",
        help="Verify finite supplied-graph geometric symmetry actions",
    )
    symmetry.add_argument("--geometry", required=True, type=Path)
    symmetry.add_argument("--policy", required=True, type=Path)
    symmetry.add_argument("--output", type=Path)
    candidate_request = commands.add_parser(
        "candidate-request",
        help="Export a scan candidate's separate refinement request",
    )
    candidate_request.add_argument("--candidate", required=True, type=Path)
    candidate_request.add_argument("--output", required=True, type=Path)
    refine = commands.add_parser(
        "refine-candidate",
        help="Execute an approved retained candidate and verify its minimum",
    )
    refine.add_argument("--candidate", required=True, type=Path)
    refine.add_argument("--ledger", required=True, type=Path)
    refine.add_argument("--revision", required=True, type=int)
    refine.add_argument("--actor", required=True)
    refine.add_argument("--approved-plan", required=True, type=Path)
    refine.add_argument("--output-dir", required=True, type=Path)
    refine.add_argument("--verification-output", required=True, type=Path)
    doctor = commands.add_parser(
        "doctor", help="Inspect actual dependency and canonical execution prerequisites"
    )
    doctor.add_argument(
        "--execution",
        choices=["github_actions", "local_validation"],
        default="github_actions",
    )
    doctor.add_argument("--check-github", action="store_true")
    plan = commands.add_parser(
        "plan", help="Review a request-bound immutable plan and explicit cost ceiling"
    )
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument(
        "--execution",
        choices=["github_actions", "local_validation"],
        default="github_actions",
    )
    plan.add_argument("--output", type=Path)
    approve = commands.add_parser(
        "approve-plan", help="Record the caller's explicit approval of a reviewed plan"
    )
    approve.add_argument("--plan", type=Path, required=True)
    approve.add_argument("--actor", required=True)
    approve.add_argument("--expires-at", type=datetime.fromisoformat)
    approve.add_argument("--scratch-mb", type=int, default=64)
    approve.add_argument("--output", type=Path, required=True)
    for name in ("submit", "run"):
        command = commands.add_parser(
            name,
            help="Submit an approved plan to GitHub Actions"
            if name == "submit"
            else "Submit, wait and verify actual hosted results",
        )
        command.add_argument("--approved-plan", type=Path, required=True)
        command.add_argument("--request", type=Path)
        command.add_argument("--idempotency-key", required=True)
        if name == "run":
            command.add_argument("--wait-seconds", type=int, default=600)
            command.add_argument("--poll-seconds", type=int, default=5)
            command.add_argument("--destination", type=Path, required=True)
    cancel = commands.add_parser(
        "cancel", help="Request cancellation of an owned, identity-matched Actions run"
    )
    cancel.add_argument("run_id")
    cancel.add_argument("--reason", required=True)
    resume = commands.add_parser(
        "resume",
        help="Create a new compatible same-model attempt with explicit lineage",
    )
    resume.add_argument("directory", type=Path)
    resume.add_argument("--output", type=Path, required=True)
    interaction = commands.add_parser(
        "interaction",
        help="Execute a genuine explicitly named local energetics protocol",
    )
    interaction.add_argument("--request", type=Path, required=True)
    interaction.add_argument("--output-dir", type=Path, required=True)
    backup = commands.add_parser(
        "backup", help="Copy and verify an immutable actual result shard"
    )
    backup.add_argument("--source", type=Path, required=True)
    backup.add_argument("--destination", type=Path, required=True)
    for name in ("restore", "rollback"):
        command = commands.add_parser(
            name, help="Restore verified bytes into a fresh compatible destination"
        )
        command.add_argument("--backup", type=Path, required=True)
        command.add_argument("--destination", type=Path, required=True)
        command.add_argument("--expected-manifest-sha256", required=True)
    interface = commands.add_parser(
        "interface", help="Start the actual JupyterLab student notebook"
    )
    interface.add_argument("--host", default="127.0.0.1")
    interface.add_argument("--port", type=int, default=8888)
    interface.add_argument("--notebook", type=Path, default=Path("UI/Start_TORQ.ipynb"))
    reference_fetch = commands.add_parser(
        "reference-fetch", help="Retrieve bounded actual HTTPS publication bytes"
    )
    reference_fetch.add_argument("--url", required=True)
    reference_fetch.add_argument("--citation", required=True)
    reference_fetch.add_argument("--reuse-permission", required=True)
    reference_fetch.add_argument("--output-dir", required=True, type=Path)
    reference_fetch.add_argument("--expected-sha256")
    reference_fetch.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    reference_fetch.add_argument("--timeout-seconds", type=float, default=30.0)
    reference_compare = commands.add_parser(
        "reference-compare", help="Compare genuine predictions with published values"
    )
    for name in (
        "reference-manifest",
        "review",
        "sources",
        "publication-bundle",
        "output",
    ):
        reference_compare.add_argument("--" + name, type=Path, required=True)
    reference_compare.add_argument("--derived-texts", type=Path)
    reference_compare.add_argument("--reference-id", required=True)
    for name in (
        "reference-manifest",
        "review",
        "publication-manifest",
    ):
        reference_compare.add_argument("--expected-" + name + "-sha256", required=True)
    for name in (
        "benchmark-references",
        "benchmark-freeze",
        "benchmark-seal",
        "benchmark-score",
    ):
        command = commands.add_parser(
            name, help="Run a pinned, externally reviewed benchmark artifact operation"
        )
        if name == "benchmark-freeze":
            command.add_argument("--design", required=True, type=Path)
            command.add_argument("--acceptance-record", required=True, type=Path)
        if name in {"benchmark-seal", "benchmark-score"}:
            command.add_argument("--freeze", required=True, type=Path)
            command.add_argument("--expected-freeze-sha256", required=True)
        if name == "benchmark-seal":
            command.add_argument("--bundles", required=True, type=Path)
        if name == "benchmark-score":
            command.add_argument("--prediction-seal", required=True, type=Path)
            command.add_argument("--expected-prediction-seal-sha256", required=True)
        if name in {"benchmark-references", "benchmark-score"}:
            command.add_argument("--reference-manifest", required=True, type=Path)
            command.add_argument("--expected-reference-manifest-sha256", required=True)
            command.add_argument("--curation", required=True, type=Path)
            command.add_argument("--expected-curation-sha256", required=True)
            command.add_argument("--sources", required=True, type=Path)
        if name != "benchmark-references":
            command.add_argument("--output", required=True, type=Path)
    workstation = commands.add_parser(
        "workstation",
        help="Queue heavy approved requests to the lab workstation (Drive folder)",
    )
    station = workstation.add_subparsers(dest="workstation_command", required=True)
    for name, text in (
        ("assign", "Prepare the assigned Drive folder (inbox/, jobs/, identity)"),
        ("recommend", "Report whether Actions or the workstation can run a request"),
        ("submit", "Queue an approved request (recorded PENDING_WORKSTATION)"),
        ("status", "Refresh one queued request from the folder"),
        ("poll", "Refresh all queued requests and ingest verified results"),
        ("cancel", "Withdraw or cancel a queued request"),
    ):
        command = station.add_parser(name, help=text)
        command.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name != "recommend":
            command.add_argument(
                "--folder", type=Path, help="Assigned workstation Drive folder"
            )
            command.add_argument("--student", help="Student ID recorded in the folder")
            command.add_argument(
                "--ledger", type=Path, help="TPO ledger (tpo_ledger.sqlite)"
            )
        if name in {"recommend", "submit"}:
            command.add_argument("--request", type=Path, required=True)
        if name == "submit":
            command.add_argument("--approved-plan", type=Path, required=True)
            command.add_argument("--idempotency-key", required=True)
        if name in {"status", "cancel"}:
            command.add_argument("client_job_id")
        if name == "cancel":
            command.add_argument("--reason", required=True)
        if name == "poll":
            command.add_argument("--destination", type=Path, required=True)
    for command in commands.choices.values():
        if not any(action.dest == "json" for action in command._actions):
            command.add_argument(
                "--json", action="store_true", default=argparse.SUPPRESS
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    structured = "--json" in (argv if argv is not None else sys.argv[1:])
    request_id: str | None = None

    def emit(
        value: object,
        *,
        status: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        _emit(
            value,
            structured=structured,
            request_id=request_id,
            status=status,
            errors=errors,
        )

    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "recipes":
            from .registry import list_method_profiles

            emit(list_method_profiles())
        elif arguments.command == "matrix":
            from .registry import matrix_index

            emit(matrix_index())
        elif arguments.command == "reference-fetch":
            from .reference_sources import execute_reference_fetch

            fetched = execute_reference_fetch(arguments)
            emit(fetched, status=fetched["status"])
            return 0 if fetched["status"] == "retrieved" else 3
        elif arguments.command == "reference-compare":
            from .reference_comparison import execute_reference_compare

            emit(execute_reference_compare(arguments), status="descriptive_comparison")
        elif arguments.command.startswith("benchmark-"):
            from .benchmark_cli import execute_benchmark_command

            emit(execute_benchmark_command(arguments), status="verified")
        elif arguments.command in {"validate", "execute", "_worker"}:
            request = read_json(arguments.request)
            request_id = request.get("request_id")
            from .application import (
                execute_request,
                validate_request,
                worker_execute,
            )

            if arguments.command == "validate":
                checked = validate_request(request, execution=arguments.execution)
                request_id = checked["request"]["request_id"]
                emit(
                    checked, status="validated" if checked["executable"] else "blocked"
                )
                return 0 if checked["executable"] else 3
            if arguments.command == "_worker":
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
                approval = (
                    read_json(arguments.approved_plan)
                    if arguments.approved_plan
                    else None
                )
                if arguments.expected_approval_sha256 and (
                    arguments.approved_plan is None
                    or sha256(arguments.approved_plan.read_bytes()).hexdigest()
                    != arguments.expected_approval_sha256
                ):
                    raise ValueError("Approved plan transport digest mismatch.")
                if os.environ.get("GITHUB_ACTIONS") == "true" and approval is None:
                    raise PrerequisiteError(
                        "Canonical hosted execution requires the submitted "
                        "reviewed approval."
                    )
                result = execute_request(
                    request, arguments.output_dir, approved_plan=approval
                )
                emit(
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

            emit(get_run_status(arguments.run_id))
        elif arguments.command == "download":
            from .application import download_run_results

            emit(
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
            emit(
                {
                    "status": "verified",
                    "request_id": manifest["request_id"],
                    "files": len(manifest["files"]),
                    "scope": (
                        "byte integrity and matching identities; scientific "
                        "accuracy is separate"
                    ),
                }
            )
        elif arguments.command == "merge":
            from .artifacts import merge_shards

            emit(
                {
                    "output_directory": str(
                        merge_shards(arguments.shards, arguments.output_dir)
                    )
                }
            )
        elif arguments.command in {"export-base", "export"}:
            if arguments.format == "publication":
                from .publication import export_publication_bundle

                output = export_publication_bundle(
                    arguments.directory, arguments.destination
                )
                emit(output)
            else:
                from .ecosystem import export_base_calculation_result

                emit(
                    {
                        "output_file": str(
                            export_base_calculation_result(
                                arguments.directory, arguments.destination
                            )
                        )
                    }
                )
        elif arguments.command == "candidates":
            from .candidate_ledger import CandidateLedger

            with CandidateLedger(arguments.ledger) as ledger:
                if arguments.action == "list":
                    emit({"candidates": ledger.candidates(include_excluded=True)})
                elif arguments.action == "inspect":
                    if not arguments.candidate_id:
                        raise ValueError("Inspection requires --candidate-id.")
                    emit(
                        {
                            "candidate": ledger.inspect(arguments.candidate_id),
                            "history": ledger.history(arguments.candidate_id),
                            "selection": ledger.selection_snapshot(),
                        }
                    )
                elif arguments.action == "register":
                    if (
                        arguments.request is None
                        or not arguments.actor
                        or not arguments.reason
                    ):
                        raise ValueError(
                            "Registration requires request, actor and reason."
                        )
                    emit(
                        ledger.register_request(
                            read_json(arguments.request),
                            actor=arguments.actor,
                            reason=arguments.reason,
                        )
                    )
                else:
                    if (
                        not arguments.candidate_id
                        or arguments.revision is None
                        or not arguments.actor
                        or not arguments.reason
                    ):
                        raise ValueError(
                            "Selection requires candidate ID, revision, "
                            "actor and reason."
                        )
                    emit(
                        getattr(ledger, arguments.action)(
                            arguments.candidate_id,
                            expected_revision=arguments.revision,
                            actor=arguments.actor,
                            reason=arguments.reason,
                        )
                    )
        elif arguments.command == "geometry-compare":
            from .geometry_identity import (
                ComparisonPolicy,
                IndexedGeometry,
                compare_indexed_geometries,
            )

            comparison = compare_indexed_geometries(
                IndexedGeometry.model_validate(read_json(arguments.source)),
                IndexedGeometry.model_validate(read_json(arguments.target)),
                ComparisonPolicy.model_validate(read_json(arguments.policy)),
            ).model_dump(mode="json")
            if arguments.output:
                _write_new(arguments.output, comparison)
            emit(comparison)
        elif arguments.command == "symmetry-proposal":
            from .geometry_identity import (
                IndexedGeometry,
                SymmetryPolicy,
                propose_nuclear_symmetry,
            )

            proposal = propose_nuclear_symmetry(
                IndexedGeometry.model_validate(read_json(arguments.geometry)),
                SymmetryPolicy.model_validate(read_json(arguments.policy)),
            ).model_dump(mode="json")
            if arguments.output:
                _write_new(arguments.output, proposal)
            emit(proposal)
        elif arguments.command == "candidate-request":
            from .adaptive import StationaryCandidate

            candidate = StationaryCandidate.model_validate(
                read_json(arguments.candidate)
            )
            candidate_model = candidate.candidate_request
            request_id = str(candidate_model.request_id)
            _write_new(arguments.output, candidate_model.model_dump(mode="json"))
            emit(
                {
                    "request": candidate_model.model_dump(mode="json"),
                    "requires_plan_review": True,
                },
                status="needs_review",
            )
        elif arguments.command == "refine-candidate":
            from .adaptive import (
                StationaryCandidate,
                validate_candidate_refinement,
                verify_candidate_minimum,
            )
            from .application import execute_request
            from .candidate_ledger import CandidateLedger

            destination = arguments.output_dir.resolve()
            verification_path = arguments.verification_output.resolve()
            if (
                verification_path == destination
                or destination in verification_path.parents
            ):
                raise ValueError(
                    "Keep verification output outside the immutable result shard."
                )
            if verification_path.exists() or verification_path.is_symlink():
                raise FileExistsError("Select a new verification output path.")
            candidate = StationaryCandidate.model_validate(
                read_json(arguments.candidate)
            )
            approval = read_json(arguments.approved_plan)
            with CandidateLedger(arguments.ledger) as ledger:
                candidate_model = validate_candidate_refinement(
                    candidate,
                    ledger=ledger,
                    expected_revision=arguments.revision,
                    approved_plan=approval,
                )
                request_id = str(candidate_model.request_id)
                execute_request(
                    candidate_model, arguments.output_dir, approved_plan=approval
                )
                verification = verify_candidate_minimum(
                    candidate,
                    ledger=ledger,
                    expected_revision=arguments.revision,
                    approved_plan=approval,
                    result_shard=arguments.output_dir,
                    actor=arguments.actor,
                ).model_dump(mode="json")
            _write_new(verification_path, verification)
            emit(verification, status="verified_minimum")
        elif arguments.command == "doctor":
            from .service import doctor

            report = doctor(
                execution=arguments.execution, check_github=arguments.check_github
            )
            emit(report)
            return 3 if report["status"] == "blocked" else 0
        elif arguments.command == "plan":
            from .service import plan_request

            review = plan_request(
                read_json(arguments.request), execution=arguments.execution
            )
            request_id = review["request_id"]
            if arguments.output:
                _write_new(arguments.output, review)
            emit(review, status="needs_review" if review["executable"] else "blocked")
            return 0 if review["executable"] else 3
        elif arguments.command == "approve-plan":
            from .service import approve_plan

            approved = approve_plan(
                read_json(arguments.plan),
                actor=arguments.actor,
                expires_at=arguments.expires_at,
                max_scratch_bytes=arguments.scratch_mb * 1024**2,
            )
            _write_new(arguments.output, approved)
            request_id = approved["plan"]["request"]["request_id"]
            emit(approved, status="approved")
        elif arguments.command in {"submit", "run"}:
            from .application import submit_request
            from .service import validate_approved_plan

            approved = read_json(arguments.approved_plan)
            request = (
                read_json(arguments.request)
                if arguments.request
                else approved["plan"]["request"]
            )
            validate_approved_plan(approved, request=request)
            request_id = request["request_id"]
            if arguments.command == "run" and not (
                1 <= arguments.wait_seconds <= 1800
                and 1 <= arguments.poll_seconds <= 30
            ):
                raise ValueError(
                    "Wait must be 1..1800 seconds; poll interval 1..30 seconds."
                )
            receipt = submit_request(
                request,
                approved_plan=approved,
                idempotency_key=arguments.idempotency_key,
            )
            if arguments.command == "submit":
                accepted = receipt["status"] in {
                    "dispatched",
                    "dispatched_run_lookup_unavailable",
                }
                emit(
                    receipt,
                    status="accepted" if accepted else "dispatch_outcome_unknown",
                )
                return 0 if accepted else 4
            if not receipt.get("run_id"):
                emit(receipt, status="accepted_pending_run_correlation")
                return 4
            from .application import download_run_results, get_run_status

            deadline = time.monotonic() + arguments.wait_seconds
            while True:
                hosted = get_run_status(receipt["run_id"])
                if hosted["status"] == "completed":
                    break
                if time.monotonic() >= deadline:
                    emit(
                        {"submission": receipt, "hosted": hosted},
                        status="accepted_wait_timeout",
                    )
                    return 4
                time.sleep(
                    min(arguments.poll_seconds, max(0, deadline - time.monotonic()))
                )
            paths = download_run_results(receipt["run_id"], arguments.destination)
            from .student_app import inspect_downloaded_results

            results = inspect_downloaded_results(paths)
            complete = all(item["scientific_status"] == "complete" for item in results)
            emit(
                {"submission": receipt, "hosted": hosted, "results": results},
                status="complete" if complete else "partial",
            )
            return 0 if complete else 4
        elif arguments.command == "workstation":
            from .workstation import WorkstationQueue, recommend_execution

            action = arguments.workstation_command
            if action == "recommend":
                emit(recommend_execution(read_json(arguments.request)))
                return 0
            queue = WorkstationQueue.from_environment(
                arguments.folder,
                student_id=arguments.student,
                ledger_path=arguments.ledger,
            )
            try:
                if action == "assign":
                    emit(queue.assign())
                elif action == "submit":
                    emit(
                        queue.submit(
                            read_json(arguments.request),
                            approved_plan=read_json(arguments.approved_plan),
                            idempotency_key=arguments.idempotency_key,
                        ),
                        status="submitted",
                    )
                elif action == "status":
                    emit(queue.status(arguments.client_job_id))
                elif action == "poll":
                    emit(queue.poll(arguments.destination))
                else:
                    emit(queue.cancel(arguments.client_job_id, reason=arguments.reason))
            finally:
                queue.close()
        elif arguments.command == "cancel":
            from .github import GitHubActions

            emit(
                GitHubActions.from_environment().cancel(
                    arguments.run_id, reason=arguments.reason
                )
            )
        elif arguments.command == "resume":
            from .service import resume_request

            request = resume_request(arguments.directory)
            _write_new(arguments.output, request)
            request_id = request["request_id"]
            emit(
                {
                    "request": request,
                    "output_file": str(arguments.output),
                    "requires_plan_review": True,
                },
                status="needs_review",
            )
        elif arguments.command == "import-topos-reviewed":
            from .reviewed_topos import import_reviewed_topos_member

            request = import_reviewed_topos_member(
                arguments.handoff,
                arguments.output_dir,
                producer_python=arguments.producer_python,
                member_id=arguments.member,
                recipe=arguments.recipe,
                products=arguments.product,
                resources={
                    "cores": arguments.cores,
                    "memory_mb": arguments.memory_mb,
                    "wall_seconds": arguments.wall_seconds,
                },
                expected_handoff_sha256=arguments.expected_handoff_sha256,
            )
            emit(
                {
                    "request": request,
                    "import_directory": str(arguments.output_dir.absolute()),
                    "computation_performed": False,
                    "requires_plan_review": True,
                },
                status="needs_review",
            )
        elif arguments.command == "interaction":
            from .energetics import evaluate_interaction

            interaction_result = evaluate_interaction(
                read_json(arguments.request), arguments.output_dir
            )
            emit(
                interaction_result.model_dump(mode="json")
                if hasattr(interaction_result, "model_dump")
                else interaction_result
            )
        elif arguments.command == "backup":
            from .operations import backup_shard

            emit(
                backup_shard(arguments.source, arguments.destination), status="verified"
            )
        elif arguments.command in {"restore", "rollback"}:
            from .operations import restore_shard, rollback_shard

            operation = (
                restore_shard if arguments.command == "restore" else rollback_shard
            )
            value = operation(
                arguments.backup,
                arguments.destination,
                expected_manifest_sha256=arguments.expected_manifest_sha256,
            )
            emit({"output_directory": str(value)}, status="verified")
        elif arguments.command == "interface":
            if not 1 <= arguments.port <= 65535:
                raise ValueError("Port must be in 1..65535.")
            notebook = arguments.notebook.resolve()
            if not notebook.is_file():
                raise ValueError(
                    "The student notebook is missing; run from the "
                    "repository or supply --notebook."
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
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        code = (
            "PREREQUISITE_UNAVAILABLE"
            if isinstance(exc, PrerequisiteError)
            else "INVALID_REQUEST"
            if isinstance(exc, (ValueError, KeyError, TypeError))
            else "EXECUTION_ERROR"
        )
        emit(
            {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
            errors=[
                {
                    "code": code,
                    "message": str(exc),
                    "details": {"exception_type": type(exc).__name__},
                }
            ],
        )
        return (
            3
            if isinstance(exc, PrerequisiteError)
            else 2
            if isinstance(exc, (ValueError, KeyError, TypeError))
            else 5
        )
    except KeyboardInterrupt:
        emit(
            {
                "message": (
                    "The interface operation was interrupted; inspect the "
                    "owned hosted run before retrying."
                )
            },
            status="cancelled",
            errors=[
                {
                    "code": "INTERRUPTED",
                    "message": "Interface operation interrupted.",
                    "details": {},
                }
            ],
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
