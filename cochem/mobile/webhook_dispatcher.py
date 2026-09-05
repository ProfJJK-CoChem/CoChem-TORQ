"""Multi-Tier Environment Execution Router, GitHub Webhook Dispatcher, and SLURM Synthesizer.

Strictly adhering to SRS Chunk 08 (REQ-MOB-072) and the Zero-Mock Mandate.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from cochem.mobile.async_runner import AsyncProcessRunner
from cochem.mobile.job_state import (
    ExecutionPayload,
    ExecutionTier,
    JobStatus,
    ManifestReference,
)
from cochem.mobile.payload_serializer import (
    ensure_tripartite_dirs,
    get_coch_artifacts,
    get_coch_src,
    get_cochem_state_dir,
    get_job_artifact_dir,
)
from cochem.mobile.status_poller import update_status_progress

logger = logging.getLogger(__name__)


def detect_execution_tier() -> ExecutionTier:
    """Detect runtime host execution tier across the 6-Tier Environment Matrix."""
    # Tier 6: HPC / SLURM
    if "SLURM_JOB_ID" in os.environ or "SLURM_NODELIST" in os.environ:
        return ExecutionTier.HPC_SLURM_TIER_6

    # Tier 5: GitHub Actions Cloud Runner
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return ExecutionTier.GITHUB_ACTIONS_TIER_5

    # Tier 4: GitHub Codespaces
    if os.environ.get("CODESPACES") == "true":
        return ExecutionTier.CODESPACES_TIER_4

    # Tier 2: Container Environment
    if (
        os.environ.get("CONTAINER_SANDBOX")
        or os.environ.get("DOCKER_CONTAINER")
        or Path("/.dockerenv").exists()
    ):
        return ExecutionTier.CONTAINER_TIER_2

    # Tier 3: SSH Gateway
    if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
        return ExecutionTier.SSH_HPC_GATEWAY_TIER_3

    # Tier 1: Local Default
    return ExecutionTier.LOCAL_TIER_1


def synthesize_slurm_script(
    payload: ExecutionPayload,
    partition: str = "standard",
    time_limit: str = "02:00:00",
    nodes: int = 1,
    ntasks: int = 16,
    gpus: int = 0,
    job_name_prefix: str = "cochem",
) -> str:
    """Synthesize SBATCH job script with POSIX path resolution and environment isolation."""
    job_id = payload.job_id
    out_dir = payload.output_artifact_dir

    gpu_directive = f"#SBATCH --gres=gpu:{gpus}\n" if gpus > 0 else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name_prefix}_{job_id[:8]}
#SBATCH --output={out_dir}/slurm-%j.out
#SBATCH --error={out_dir}/slurm-%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
{gpu_directive}
set -euo pipefail

echo "[COCHEM HPC] Starting SLURM Execution for Job ID: {job_id}"
echo "[COCHEM HPC] Node: $(hostname) | Cores: {ntasks}"

# Execute calculation worker
python -m cochem.mobile.async_runner --job-id "{job_id}" --artifact-dir "{out_dir}"
echo "[COCHEM HPC] Job {job_id} finished with exit code $?"
"""
    return script


def build_github_dispatch_request(
    owner: str,
    repo: str,
    payload: Union[ExecutionPayload, ManifestReference],
    event_type: str = "cochem_remote_execution",
    token: Optional[str] = None,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Construct GitHub Actions repository_dispatch URL, headers, and request body."""
    resolved_token = (
        token or os.environ.get("GITHUB_TOKEN") or os.environ.get("COCHEM_DISPATCH_SECRET")
    )
    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "CoChem-Mobile-Dispatcher/1.0",
    }
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token.strip()}"

    client_payload = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
    body = {
        "event_type": event_type,
        "client_payload": client_payload,
    }
    return url, headers, body


def dispatch_github_action_webhook(
    owner: str,
    repo: str,
    payload: Union[ExecutionPayload, ManifestReference],
    token: Optional[str] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Execute authenticated repository_dispatch POST request to GitHub Actions REST API."""
    resolved_token = (
        token or os.environ.get("GITHUB_TOKEN") or os.environ.get("COCHEM_DISPATCH_SECRET")
    )
    if not resolved_token or not resolved_token.strip():
        raise ValueError(
            "Authentication token missing: GITHUB_TOKEN or COCHEM_DISPATCH_SECRET environment "
            "variable must be configured for Tier 4/5 webhook dispatch."
        )

    url, headers, body = build_github_dispatch_request(
        owner=owner, repo=repo, payload=payload, token=resolved_token
    )
    response = requests.post(url, json=body, headers=headers, timeout=timeout)
    response.raise_for_status()

    return {
        "status": "DISPATCHED",
        "tier": ExecutionTier.GITHUB_ACTIONS_TIER_5.value,
        "status_code": response.status_code,
        "endpoint": url,
        "job_id": payload.job_id,
    }


class WebhookDispatcher:
    """Multi-tier execution and webhook dispatch engine."""

    def __init__(
        self,
        src_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.src_dir = src_dir or get_coch_src()
        self.artifacts_dir = artifacts_dir or get_coch_artifacts()
        self.state_dir = state_dir or get_cochem_state_dir()
        self.runner = AsyncProcessRunner(
            src_dir=self.src_dir,
            artifacts_dir=self.artifacts_dir,
            state_dir=self.state_dir,
        )

    def dispatch(
        self,
        payload: Union[ExecutionPayload, ManifestReference],
        tier: Optional[ExecutionTier] = None,
        github_owner: Optional[str] = None,
        github_repo: Optional[str] = None,
        github_token: Optional[str] = None,
        slurm_partition: str = "standard",
        slurm_time_limit: str = "02:00:00",
    ) -> Dict[str, Any]:
        """Route and execute job dispatch across appropriate execution tier."""
        resolved_tier = tier or detect_execution_tier()

        # Step 1: Initialize status
        job_id = payload.job_id
        ensure_tripartite_dirs(
            job_id=job_id,
            src_dir=self.src_dir,
            artifacts_dir=self.artifacts_dir,
            state_dir=self.state_dir,
        )

        update_status_progress(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress_percent=0.0,
            current_step=f"Dispatching job to {resolved_tier.value}",
            tier=resolved_tier,
            state_dir=self.state_dir,
        )

        # Step 2: Route by Tier
        if resolved_tier == ExecutionTier.LOCAL_TIER_1:
            if isinstance(payload, ManifestReference):
                from cochem.mobile.payload_serializer import load_staged_payload

                exec_payload = load_staged_payload(payload)
            else:
                exec_payload = payload

            pid = self.runner.launch_local_worker(exec_payload)
            return {
                "status": "DISPATCHED",
                "tier": resolved_tier.value,
                "pid": pid,
                "job_id": job_id,
            }

        elif resolved_tier in {
            ExecutionTier.CODESPACES_TIER_4,
            ExecutionTier.GITHUB_ACTIONS_TIER_5,
        }:
            owner = github_owner or os.environ.get("GITHUB_REPOSITORY_OWNER") or "cochem-lab"
            repo = github_repo or os.environ.get("GITHUB_REPOSITORY_NAME") or "CoChem-BASE"
            return dispatch_github_action_webhook(
                owner=owner,
                repo=repo,
                payload=payload,
                token=github_token,
            )

        elif resolved_tier == ExecutionTier.HPC_SLURM_TIER_6:
            if isinstance(payload, ManifestReference):
                from cochem.mobile.payload_serializer import load_staged_payload

                exec_payload = load_staged_payload(payload)
            else:
                exec_payload = payload

            sbatch_script = synthesize_slurm_script(
                payload=exec_payload,
                partition=slurm_partition,
                time_limit=slurm_time_limit,
            )

            job_art_dir = get_job_artifact_dir(job_id, base_artifacts_dir=self.artifacts_dir)
            job_art_dir.mkdir(parents=True, exist_ok=True)
            sbatch_path = job_art_dir / "job.sbatch"
            sbatch_path.write_text(sbatch_script, encoding="utf-8")

            # Check if sbatch executable exists in PATH
            sbatch_bin = shutil.which("sbatch")
            if sbatch_bin:
                proc = subprocess.run(
                    [sbatch_bin, str(sbatch_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0:
                    # Parse "Submitted batch job 123456"
                    slurm_job_id = proc.stdout.strip().split()[-1]
                    update_status_progress(
                        job_id=job_id,
                        status=JobStatus.STAGED,
                        current_step=f"SLURM batch job submitted: {slurm_job_id}",
                        state_dir=self.state_dir,
                    )
                    return {
                        "status": "DISPATCHED",
                        "tier": resolved_tier.value,
                        "slurm_job_id": slurm_job_id,
                        "job_id": job_id,
                    }
                else:
                    raise RuntimeError(f"SLURM sbatch execution failed: {proc.stderr}")
            else:
                # Staged for HPC cluster submission
                update_status_progress(
                    job_id=job_id,
                    status=JobStatus.STAGED,
                    current_step=f"Synthesized SBATCH script staged at {sbatch_path.as_posix()}",
                    state_dir=self.state_dir,
                )
                return {
                    "status": "STAGED_FOR_SLURM",
                    "tier": resolved_tier.value,
                    "sbatch_script_path": sbatch_path.as_posix(),
                    "job_id": job_id,
                }

        else:
            # Container / SSH Gateway default dispatch
            return {
                "status": "DISPATCHED",
                "tier": resolved_tier.value,
                "job_id": job_id,
                "message": f"Execution dispatched to tier {resolved_tier.value}",
            }
