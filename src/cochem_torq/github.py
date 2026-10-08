"""Authenticated GitHub Actions transport through the official GitHub CLI.

GitHub failures remain visible. Calculation submission never silently runs in
Codespaces, and hosted workflow completion is distinct from scientific success.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from .artifacts import verify_shard
from .domain import PrerequisiteError, canonical_json, digest, read_json
from .github_auth import private_gh_environment
from .github_provenance import scientific_source_binding, verify_controller_artifact


class GitHubAccessError(PrerequisiteError):
    pass


class GitHubActions:
    def __init__(
        self,
        repository: str,
        ref: str = "main",
        state_directory: str | Path | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Specify a GitHub owner/repository.")
        if (
            not re.fullmatch(r"[A-Za-z0-9_./-]+", ref)
            or ".." in ref
            or ref.startswith("-")
        ):
            raise ValueError("Specify an explicit valid calculation branch or tag.")
        self.repository, self.ref = repository, ref
        self.state_directory = Path(
            state_directory or Path.home() / ".local/state/cochem-torq/submissions"
        )

    @classmethod
    def from_environment(cls) -> GitHubActions:
        return cls(
            os.environ.get(
                "COCHEM_TORQ_GITHUB_REPOSITORY",
                os.environ.get("GITHUB_REPOSITORY", "ProfJJK-CoChem/CoChem-TORQ"),
            ),
            os.environ.get("COCHEM_TORQ_CALCULATION_REF", "main"),
        )

    def _api(self, endpoint: str, *, data: dict[str, Any] | None = None) -> Any:
        if shutil.which("gh") is None:
            raise GitHubAccessError(
                "GitHub CLI is unavailable. Install gh in the interface "
                "environment and authenticate with access to the "
                "calculation repository."
            )
        command = ["gh", "api", "--hostname", "github.com", endpoint]
        if data is not None:
            command += ["--method", "POST", "--input", "-"]
        try:
            process = subprocess.run(
                command,
                input=canonical_json(data) if data is not None else None,
                capture_output=True,
                timeout=45,
                env=private_gh_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise GitHubAccessError(
                "GitHub API timed out; check the request/run ID before "
                "retrying a dispatch."
            ) from exc
        if process.returncode:
            # gh API error messages can embed arguments; preserve a concise
            # action/status only, and never print authenticated command output.
            raise GitHubAccessError(
                f"GitHub API access failed (gh exit {process.returncode}). "
                "Check gh auth status, repository Actions permissions "
                "and the network policy."
            )
        return json.loads(process.stdout) if process.stdout.strip() else None

    def submit(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        approved_plan_sha256: str | None = None,
        approved_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            if (
                not isinstance(idempotency_key, str)
                or not idempotency_key.strip()
                or len(idempotency_key) > 256
            ):
                raise ValueError("An idempotency key must contain 1..256 characters.")
            self.state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock = self.state_directory / ".submission.lock"
            if lock.is_symlink():
                raise ValueError("The submission lock cannot be a symlink.")
            descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "a+b") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                return self._submit(
                    request,
                    idempotency_key=idempotency_key,
                    approved_plan_sha256=approved_plan_sha256,
                    approved_plan=approved_plan,
                )
        return self._submit(request, approved_plan=approved_plan)

    def _submit(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        approved_plan_sha256: str | None = None,
        approved_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = canonical_json(request)
        if len(payload) > 16384:
            raise ValueError(
                "Actions request JSON exceeds the 16 KiB classroom "
                "limit. Use a verified artifact handoff for larger "
                "inputs."
            )
        from .service import validate_approved_plan

        if approved_plan is None:
            raise ValueError(
                "Hosted dispatch requires the complete reviewed plan approval."
            )
        approved = validate_approved_plan(approved_plan, request=request)
        if approved.approval.plan_sha256 != approved_plan_sha256:
            raise ValueError("Approved plan transport digest mismatch.")
        approval_bytes = canonical_json(approved.model_dump(mode="json"))
        if len(approval_bytes) > 32768:
            raise ValueError("Actions plan approval exceeds the 32 KiB limit.")
        approval_digest = digest(approved.model_dump(mode="json"))
        request_id = str(UUID(str(request["request_id"])))
        self.state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        primary = self.state_directory / f"{request_id}.json"
        key_digest = (
            digest({"repository": self.repository, "idempotency_key": idempotency_key})
            if idempotency_key is not None
            else None
        )
        key_path = (
            self.state_directory / f"key-{key_digest}.json" if key_digest else None
        )
        existing_path = key_path if key_path and key_path.exists() else primary
        if existing_path.exists():
            if existing_path.is_symlink():
                raise ValueError("Submission receipts cannot be symlinks.")
            existing = read_json(existing_path)
            if not isinstance(existing, dict):
                raise GitHubAccessError("The retained submission receipt is malformed.")
            if (
                key_digest
                and existing.get("idempotency_key_sha256") == key_digest
                and existing.get("request_sha256") == digest(request)
                and existing.get("approved_plan_sha256") == approved_plan_sha256
                and existing.get("repository") == self.repository
                and existing.get("ref") == self.ref
                and existing.get("approval_sha256") == approval_digest
            ):
                # Retain ambiguity; a replay cannot send a second request.
                return existing
            raise GitHubAccessError(
                "Request UUID or idempotency key already belongs to "
                "another receipt; inspect it before creating a new "
                "attempt."
            )
        try:
            workflow = self._api(
                f"repos/{self.repository}/actions/workflows/calculation.yml"
            )
        except GitHubAccessError as exc:
            raise GitHubAccessError(
                "The calculation workflow is unavailable. Deploy the "
                "reviewed workflow on the default branch and verify "
                "repository Actions access before student submission."
            ) from exc
        if workflow.get("state") != "active":
            raise GitHubAccessError(
                "The calculation workflow is disabled; repository "
                "Actions settings must enable it before submission."
            )
        source = self._api(f"repos/{self.repository}/commits/{self.ref}")
        source_sha = source.get("sha", "")
        if not re.fullmatch("[0-9a-f]{40}", source_sha):
            raise GitHubAccessError(
                "GitHub did not provide a verifiable source commit."
            )
        binding = scientific_source_binding(self._api, self.repository, source_sha)
        reviewed_commit = approved.source_identity.get("git_commit")
        if (
            reviewed_commit is not None
            and reviewed_commit != binding["scientific_commit"]
        ):
            raise GitHubAccessError(
                "The selected Actions ref differs from the reviewed "
                "implementation commit."
            )
        inputs = {
            "request_b64": base64.b64encode(payload).decode("ascii"),
            "request_sha256": digest(request),
            "request_id": str(request_id),
            "expected_source_sha": source_sha,
            "engine": "pyscf",
            "approved_plan_b64": base64.b64encode(approval_bytes).decode("ascii"),
            "approved_plan_sha256": approval_digest,
        }
        if len(canonical_json(inputs)) > 65535:
            raise ValueError(
                "The exact request and approval exceed GitHub's dispatch payload limit."
            )
        receipt = {
            "schema_version": "cochem.torq.submission/1",
            "request_id": str(request_id),
            "request_sha256": digest(request),
            "repository": self.repository,
            "ref": self.ref,
            "source_commit": source_sha,
            "scientific_source_binding": binding,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": "dispatch_requested",
            "run_id": None,
            "idempotency_key_sha256": key_digest,
            "approved_plan_sha256": approved_plan_sha256,
            "approval_sha256": approval_digest,
            "expected_recipe_sha256": approved.plan["scientific_recipe"][
                "recipe_sha256"
            ],
            "expected_code_sha256": approved.source_identity["code_sha256"],
            "workflow_id": workflow["id"],
            "workflow_url": f"https://github.com/{self.repository}/actions/workflows/calculation.yml",
        }
        with primary.open("xb") as stream:
            stream.write(canonical_json(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if key_path:
            self._save_receipt(key_path, receipt)
        try:
            self._api(
                f"repos/{self.repository}/actions/workflows/calculation.yml/dispatches",
                data={"ref": self.ref, "inputs": inputs},
            )
        except GitHubAccessError:
            receipt["status"] = "dispatch_outcome_unknown_check_github_before_retry"
            self._save_receipt(primary, receipt)
            if key_path:
                self._save_receipt(key_path, receipt)
            raise
        receipt["status"] = "dispatched"
        self._save_receipt(primary, receipt)
        if key_path:
            self._save_receipt(key_path, receipt)
        # Correlate by UUID-bearing workflow title, never by "most recent run".
        for attempt in range(5):
            try:
                runs = self._api(
                    f"repos/{self.repository}/actions/workflows/calculation.yml/runs?event=workflow_dispatch&per_page=30"
                )
            except GitHubAccessError:
                receipt["status"] = "dispatched_run_lookup_unavailable"
                break
            matches = [
                run
                for run in runs.get("workflow_runs", [])
                if str(request_id) in run.get("display_title", "")
                and run.get("head_sha") == source_sha
            ]
            if len(matches) > 1:
                raise GitHubAccessError(
                    "Multiple runs match this request UUID; inspect the "
                    "workflow and select the intended run explicitly."
                )
            if matches:
                receipt.update(
                    run_id=str(matches[0]["id"]), run_url=matches[0]["html_url"]
                )
                break
            if attempt < 4:
                time.sleep(1)
        self._save_receipt(primary, receipt)
        if key_path:
            self._save_receipt(key_path, receipt)
        if receipt["run_id"]:
            self._save_receipt(
                self.state_directory / f"{receipt['run_id']}.json", receipt
            )
        return receipt

    def cancel(self, run_id: str, *, reason: str) -> dict[str, Any]:
        if not run_id.isdigit() or not reason.strip() or len(reason) > 512:
            raise ValueError(
                "Cancellation requires a numeric run ID and a bounded reason."
            )
        path = self.state_directory / f"{run_id}.json"
        if path.is_symlink() or not path.is_file():
            raise GitHubAccessError(
                "Cancellation requires this interface's owned submission receipt."
            )
        receipt = read_json(path)
        if not isinstance(receipt, dict):
            raise GitHubAccessError("The owned submission receipt is malformed.")
        try:
            request_id = str(UUID(receipt["request_id"]))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GitHubAccessError(
                "The owned submission receipt has no valid request UUID."
            ) from exc
        run = self._api(f"repos/{self.repository}/actions/runs/{run_id}")
        if (
            receipt.get("repository") != self.repository
            or str(receipt.get("run_id")) != run_id
            or receipt.get("source_commit") != run.get("head_sha")
            or request_id not in (run.get("display_title") or "")
            or receipt.get("workflow_id") != run.get("workflow_id")
        ):
            raise GitHubAccessError(
                "Owned receipt does not match the actual "
                "run/source/workflow identities."
            )
        if run["status"] != "completed":
            self._api(f"repos/{self.repository}/actions/runs/{run_id}/cancel", data={})
        record = {
            "schema_version": "cochem.torq.cancellation/1",
            "run_id": run_id,
            "request_id": receipt["request_id"],
            "repository": self.repository,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "status": "already_completed"
            if run["status"] == "completed"
            else "cancellation_requested",
            "scientific_status": "inspect_verified_result_bundle",
        }
        self._save_receipt(self.state_directory / f"cancel-{run_id}.json", record)
        return record

    @staticmethod
    def _save_receipt(path: Path, receipt: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.pending")
        with temporary.open("wb") as stream:
            stream.write(canonical_json(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def status(self, run_id: str) -> dict[str, Any]:
        if not run_id.isdigit():
            raise ValueError("A GitHub Actions run ID must be numeric.")
        run = self._api(f"repos/{self.repository}/actions/runs/{run_id}")
        return {
            "execution": "github_actions",
            "repository": self.repository,
            "run_id": run_id,
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "source_commit": run["head_sha"],
            "workflow_id": run.get("workflow_id"),
            "event": run.get("event"),
            "run_url": run["html_url"],
            "scientific_status": "inspect_verified_result_bundle",
            "display_title": run.get("display_title"),
        }

    def download(self, run_id: str, destination: Path) -> list[Path]:
        run = self.status(run_id)
        if run["status"] != "completed":
            raise GitHubAccessError(
                "The calculation workflow has not finished; results are not yet final."
            )
        receipt_path = self.state_directory / f"{run_id}.json"
        if receipt_path.is_symlink():
            raise GitHubAccessError("Submission receipts cannot be symlinks.")
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        if receipt is not None and (
            not isinstance(receipt, dict)
            or receipt.get("repository") != self.repository
            or receipt.get("source_commit") != run["source_commit"]
            or str(receipt.get("run_id")) != run_id
            or receipt.get("workflow_id") != run["workflow_id"]
            or str(receipt.get("request_id")) not in (run.get("display_title") or "")
            or run["event"] != "workflow_dispatch"
        ):
            raise GitHubAccessError(
                "Submission receipt and hosted run have different "
                "repository/source/workflow/request identities."
            )
        binding = scientific_source_binding(
            self._api, self.repository, run["source_commit"]
        )
        if (
            receipt
            and "scientific_source_binding" in receipt
            and receipt["scientific_source_binding"] != binding
        ):
            raise GitHubAccessError(
                "The retained scientific source binding differs from GitHub's "
                "immutable controller/catalog objects."
            )
        if binding["route"] == "native-base-controller" and not receipt:
            raise GitHubAccessError(
                "A separate native controller result requires this interface's "
                "owned request and approval receipt."
            )
        target = destination.absolute() / f"run-{run_id}"
        if target.exists():
            raise FileExistsError(
                "Downloaded runs are immutable; choose a new destination."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".torq-download-", dir=target.parent))
        try:
            process = subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run_id,
                    "--repo",
                    self.repository,
                    "--pattern",
                    "torq-result-*",
                    "--dir",
                    str(staging),
                ],
                capture_output=True,
                timeout=120,
                env=private_gh_environment(),
            )
            if process.returncode:
                raise GitHubAccessError(
                    "GitHub result artifact download failed; it may be "
                    "unavailable or expired."
                )
            roots = [
                path.parent
                for path in staging.rglob("manifest.json")
                if read_json(path).get("schema_version") == "cochem.torq.shard/1"
            ]
            if not roots:
                raise GitHubAccessError(
                    "The workflow produced no sealed TORQ result shard."
                )
            for root in roots:
                manifest = verify_shard(
                    root,
                    expected_request_sha256=receipt["request_sha256"]
                    if receipt
                    else None,
                    expected_recipe_sha256=receipt.get("expected_recipe_sha256")
                    if receipt
                    else None,
                )
                if receipt and manifest["source_identity"].get(
                    "code_sha256"
                ) != receipt.get("expected_code_sha256"):
                    raise GitHubAccessError(
                        "Downloaded worker implementation differs from the "
                        "reviewed source."
                    )
                declared = manifest["source_identity"].get("declared_workflow_commit")
                if declared != binding["scientific_commit"] or manifest[
                    "source_identity"
                ].get("git_commit") not in {None, binding["scientific_commit"]}:
                    raise GitHubAccessError(
                        "Artifact provenance does not match the pinned scientific "
                        "source commit."
                    )
                if binding["route"] == "native-base-controller":
                    assert receipt is not None
                    artifact = staging / root.relative_to(staging).parts[0]
                    verify_controller_artifact(artifact, binding, receipt, manifest)
                if manifest["request_id"] not in (run.get("display_title") or ""):
                    raise GitHubAccessError(
                        "Artifact request UUID does not match the hosted "
                        "workflow title."
                    )
            staging.rename(target)
            return [target / root.relative_to(staging) for root in roots]
        finally:
            if staging.exists():
                shutil.rmtree(staging)
