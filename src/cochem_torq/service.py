"""Reviewed plans and CLI lifecycle contracts, separate from scientific results.

An approval record expresses the caller's explicit budget decision. It does not
provide GitHub permissions, a cryptographic signature or scientific qualification.
"""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, StrictInt, model_validator

from .domain import CalculationRequest, Contract, PrerequisiteError, digest, read_json


class Approval(Contract):
    actor: str = Field(min_length=1, max_length=128)
    approved_at: datetime
    expires_at: datetime
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_wall_seconds: StrictInt = Field(ge=1, le=1800)
    max_cpu_core_seconds: StrictInt = Field(ge=1, le=7200)
    max_memory_mb: StrictInt = Field(ge=256, le=8192)
    max_scratch_bytes: StrictInt = Field(ge=1024 * 1024, le=1024**3)
    max_tasks: StrictInt = Field(ge=1, le=1000)
    permitted_retries: StrictInt = Field(default=0, ge=0, le=2)
    authorization_kind: Literal["explicit_caller_budget_approval"] = (
        "explicit_caller_budget_approval"
    )

    @model_validator(mode="after")
    def timestamps(self) -> Approval:
        if (
            not self.actor.strip()
            or self.approved_at.utcoffset() is None
            or self.expires_at.utcoffset() is None
            or self.expires_at <= self.approved_at
        ):
            raise ValueError(
                "An approval requires an actor and increasing aware times."
            )
        return self


class ApprovedPlan(Contract):
    schema_version: Literal["cochem.torq.approved-plan/1"] = (
        "cochem.torq.approved-plan/1"
    )
    plan: dict[str, Any]
    approval: Approval
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: dict[str, Any]

    @model_validator(mode="after")
    def binding(self) -> ApprovedPlan:
        plan = dict(self.plan)
        if (
            not isinstance(plan.get("request"), dict)
            or not isinstance(plan["request"].get("resources"), dict)
            or not isinstance(plan.get("resources"), dict)
            or not isinstance(plan.get("tasks"), list)
            or not plan["tasks"]
            or any(not isinstance(task, dict) for task in plan["tasks"])
        ):
            raise ValueError(
                "An approved plan requires a complete request/resources object "
                "and a nonempty task-object list."
            )
        declared = plan.pop("plan_sha256", None)
        if declared != digest(plan) or declared != self.approval.plan_sha256:
            raise ValueError("Approval and the immutable plan digest do not match.")
        if self.approval_sha256 != digest(self.approval.model_dump(mode="json")):
            raise ValueError("Approval record digest mismatch.")
        request = CalculationRequest.model_validate(plan["request"])
        if plan["resources"] != request.resources.model_dump(mode="json"):
            raise ValueError("Plan resources differ from the bound request resources.")
        if plan.get("request_sha256") != digest(request.model_dump(mode="json")):
            raise ValueError("The plan must bind the complete original request.")
        resources = request.resources
        if (
            self.approval.max_wall_seconds < resources.wall_seconds
            or self.approval.max_cpu_core_seconds
            < resources.cores * resources.wall_seconds
            or self.approval.max_memory_mb < resources.memory_mb
            or self.approval.max_tasks < len(plan["tasks"])
        ):
            raise ValueError(
                "The immutable plan exceeds its explicitly approved budget."
            )
        return self


def plan_request(
    request: dict[str, Any] | CalculationRequest, *, execution: str = "github_actions"
) -> dict[str, Any]:
    from .application import validate_request

    if not isinstance(execution, str) or execution not in {
        "github_actions",
        "local_validation",
    }:
        raise ValueError("Unknown execution environment.")
    if isinstance(request, CalculationRequest):
        model = request
    elif isinstance(request, dict):
        model = CalculationRequest.model_validate(request)
    else:
        raise ValueError("A plan requires a request object or CalculationRequest.")
    checked = validate_request(model, execution=execution)
    return {
        "schema_version": "cochem.torq.plan-review/1",
        "request_id": checked["request"]["request_id"],
        "executable": checked["executable"],
        "blocking_reasons": checked["blocking_reasons"],
        "warnings": checked["warnings"],
        "execution": execution,
        "plan": checked["plan"],
        "approval": None,
    }


def approve_plan(
    review: dict[str, Any],
    *,
    actor: str,
    expires_at: datetime | None = None,
    max_scratch_bytes: int = 64 * 1024 * 1024,
    permitted_retries: int = 0,
) -> dict[str, Any]:
    if (
        not isinstance(review, dict)
        or review.get("schema_version") != "cochem.torq.plan-review/1"
    ):
        raise ValueError("Approve a versioned TORQ plan-review artifact.")
    if (
        not isinstance(review.get("plan"), dict)
        or not isinstance(review["plan"].get("request"), dict)
        or not isinstance(review.get("execution"), str)
        or review["execution"] not in {"github_actions", "local_validation"}
    ):
        raise ValueError(
            "A plan review requires a plan/request object "
            "and a supported execution environment."
        )
    request = CalculationRequest.model_validate(review["plan"]["request"])
    # Recompute from current code/registry instead of trusting a mutable review flag.
    current = plan_request(
        request.model_dump(mode="json"), execution=review["execution"]
    )
    if current["plan"] != review["plan"]:
        raise ValueError("Plan or scientific recipe changed; review the current plan.")
    if not current["executable"]:
        raise PrerequisiteError("; ".join(current["blocking_reasons"]))
    now = datetime.now(timezone.utc)
    approval = Approval(
        actor=actor,
        approved_at=now,
        expires_at=expires_at or now + timedelta(hours=1),
        plan_sha256=current["plan"]["plan_sha256"],
        max_wall_seconds=request.resources.wall_seconds,
        max_cpu_core_seconds=request.resources.wall_seconds * request.resources.cores,
        max_memory_mb=request.resources.memory_mb,
        max_scratch_bytes=max_scratch_bytes,
        max_tasks=len(current["plan"]["tasks"]),
        permitted_retries=permitted_retries,
    )
    from .application import source_identity

    return ApprovedPlan(
        plan=current["plan"],
        approval=approval,
        approval_sha256=digest(approval.model_dump(mode="json")),
        source_identity=source_identity(),
    ).model_dump(mode="json")


def validate_approved_plan(
    value: dict[str, Any], *, request: dict[str, Any] | None = None
) -> ApprovedPlan:
    from .application import create_plan, source_identity
    from .registry import get_profile

    approved = ApprovedPlan.model_validate(value)
    now = datetime.now(timezone.utc)
    if approved.approval.approved_at > now or approved.approval.expires_at <= now:
        raise PrerequisiteError(
            "The plan approval is future-dated or expired; review it again."
        )
    model = CalculationRequest.model_validate(approved.plan["request"])
    if request is not None and model.model_dump(
        mode="json"
    ) != CalculationRequest.model_validate(request).model_dump(mode="json"):
        raise ValueError(
            "Submission request differs from the explicitly approved plan."
        )
    if create_plan(model, get_profile(model.recipe)) != approved.plan:
        raise PrerequisiteError(
            "Current scientific recipe/plan differs from the reviewed plan."
        )
    if (
        not approved.source_identity.get("code_sha256")
        or approved.source_identity["code_sha256"] != source_identity()["code_sha256"]
    ):
        raise PrerequisiteError(
            "The reviewed implementation changed; "
            "review and approve the current source."
        )
    return approved


def resume_request(directory: str | Path) -> dict[str, Any]:
    """Create a new same-model attempt with explicit checkpoint reuse status."""
    from .application import source_identity
    from .artifacts import verify_shard

    root = Path(directory)
    manifest = verify_shard(root)
    original = CalculationRequest.model_validate(read_json(root / "request.json"))
    result = read_json(root / "result.json")
    if result["status"] == "complete":
        raise ValueError("This attempt is complete; resume is for incomplete attempts.")
    current = source_identity()
    old = manifest["source_identity"]
    if not old.get("code_sha256") or old["code_sha256"] != current.get("code_sha256"):
        raise PrerequisiteError(
            "Source compatibility is not established; create and review a new request."
        )
    payload = original.model_dump(mode="json")
    payload["request_id"] = str(uuid4())
    payload["source_provenance"] = {
        **payload["source_provenance"],
        "attempt_lineage": {
            "previous_request_id": str(original.request_id),
            "previous_manifest_sha256": digest(manifest),
            "scientific_cache_key": manifest.get("scientific_cache_key"),
            "recovery": "new_same_model_attempt",
            "engine_checkpoint_reused": False,
        },
    }
    return CalculationRequest.model_validate(payload).model_dump(mode="json")


def doctor(
    *, execution: str = "github_actions", check_github: bool = False
) -> dict[str, Any]:
    from .registry import list_method_profiles

    if not isinstance(execution, str) or execution not in {
        "github_actions",
        "local_validation",
    }:
        raise ValueError("Unknown execution environment.")
    versions: dict[str, str | None] = {}
    for name in (
        "cochem-torq",
        "pydantic",
        "rfc8785",
        "qcelemental",
        "pyscf",
        "geometric",
        "dftd4",
        "jupyterlab",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    blocked = []
    if execution == "local_validation" and (
        versions["pyscf"] != "2.14.0" or versions["geometric"] != "1.1.1"
    ):
        blocked.append(
            "The pinned local PySCF/optimizer calculation environment is unavailable."
        )
    if execution == "github_actions" and shutil.which("gh") is None:
        blocked.append(
            "The interface needs the GitHub CLI for authenticated Actions transport."
        )
    hosted = {"status": "not_checked", "calculation_dispatch_verified": False}
    if check_github:
        from .github import GitHubAccessError, GitHubActions

        client = GitHubActions.from_environment()
        try:
            repository = client._api(f"repos/{client.repository}")
            workflow = client._api(
                f"repos/{client.repository}/actions/workflows/calculation.yml"
            )
            hosted = {
                "status": "available"
                if workflow.get("state") == "active"
                else "disabled",
                "repository": client.repository,
                "default_branch": repository["default_branch"],
                "workflow_state": workflow.get("state"),
                "calculation_dispatch_verified": False,
            }
            if workflow.get("state") != "active":
                blocked.append("The canonical calculation workflow is disabled.")
        except GitHubAccessError as exc:
            hosted = {
                "status": "unavailable",
                "message": str(exc),
                "calculation_dispatch_verified": False,
            }
            blocked.append(str(exc))
    return {
        "schema_version": "cochem.torq.doctor/1",
        "python": sys.version.split()[0],
        "execution": execution,
        "dependencies": versions,
        "github": hosted,
        "status": "blocked" if blocked else "available",
        "blocking_reasons": blocked,
        "method_profiles": list_method_profiles(),
        "full_srs_release_qualified": False,
    }
